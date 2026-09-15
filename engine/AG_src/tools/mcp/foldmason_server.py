"""
FoldMason MCP Server — Step07: Structure-Based Multiple Sequence Alignment
==========================================================================
Exposes FoldMason CLI commands as MCP tool endpoints for structural alignment
of candidate peptide binders.

FoldMason performs structure-based multiple sequence alignment (MSA) using
3Di alphabet representation of protein backbones, enabling alignment even
when sequence identity is too low for traditional MSA tools.

IMPORTANT: This is an *interface definition*. Actual execution requires:
    - FoldMason binary on PATH (install from: https://github.com/steineggerlab/foldmason)
    - Input PDB files on disk

Tools exposed:
    1. easy_msa         — compute structural MSA from a list of PDB paths
    2. compute_lddt     — compute per-column lDDT scores for an existing MSA
    3. refine_msa       — iterative refinement of an existing MSA
    4. cluster_structures — cluster structures by 3Di similarity

Usage:
    server = FoldMasonMCPServer()
    result = server.dispatch(
        "easy_msa",
        pdb_paths=["/data/binder1.pdb", "/data/binder2.pdb"],
        output_prefix="/tmp/foldmason_run",
    )
    fasta = result["aa_fasta"]
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base_server import MCPServer, MCPTool

logger = logging.getLogger(__name__)

_FOLDMASON_BIN = shutil.which("foldmason") or "foldmason"


def _run_foldmason(args: list[str], timeout: int = 600) -> subprocess.CompletedProcess[str]:
    """Run a foldmason command and return the CompletedProcess result.

    Args:
        args:    Command-line arguments (the binary name is prepended automatically).
        timeout: Maximum execution time in seconds.

    Raises:
        RuntimeError: If foldmason exits with a non-zero return code.
    """
    cmd = [_FOLDMASON_BIN, *args]
    logger.debug("Running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"foldmason failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    return proc


class FoldMasonMCPServer(MCPServer):
    """MCP server wrapping FoldMason CLI for structural MSA.

    All tools use subprocess calls to the foldmason binary.
    """

    def __init__(self) -> None:
        super().__init__("foldmason")
        self._register_all()

    def _register_all(self) -> None:
        tools = [
            MCPTool(
                name="easy_msa",
                description=(
                    "Compute a structure-based multiple sequence alignment from "
                    "a list of PDB files using FoldMason. "
                    "Returns aligned sequences in FASTA, 3Di alphabet FASTA, "
                    "a Newick tree, and an HTML report."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_paths", "output_prefix"],
                    "properties": {
                        "pdb_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Absolute paths to input PDB files.",
                        },
                        "output_prefix": {
                            "type": "string",
                            "description": "Output file prefix (directory must exist).",
                        },
                        "report_mode": {
                            "type": "integer",
                            "default": 1,
                            "enum": [0, 1, 2],
                            "description": (
                                "0 = no report, 1 = HTML report (default), "
                                "2 = HTML + tree."
                            ),
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "aa_fasta": {
                            "type": "string",
                            "description": "Path to amino-acid aligned FASTA file.",
                        },
                        "3di_fasta": {
                            "type": "string",
                            "description": "Path to 3Di-alphabet aligned FASTA file.",
                        },
                        "newick_tree": {
                            "type": "string",
                            "description": "Path to guide tree in Newick format.",
                        },
                        "html_report": {
                            "type": "string",
                            "description": "Path to HTML alignment report (or null).",
                        },
                    },
                },
                handler=self._easy_msa,
            ),
            MCPTool(
                name="compute_lddt",
                description=(
                    "Compute per-column lDDT scores for a structural MSA, "
                    "indicating how well-conserved each alignment column is."
                ),
                input_schema={
                    "type": "object",
                    "required": ["db_path", "msa_path"],
                    "properties": {
                        "db_path": {
                            "type": "string",
                            "description": "Path to the FoldMason database directory.",
                        },
                        "msa_path": {
                            "type": "string",
                            "description": "Path to the alignment file (.alnfasta or similar).",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "lddt_scores": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Per-column lDDT scores (0–100).",
                        },
                        "mean_lddt": {
                            "type": "number",
                            "description": "Mean lDDT across all alignment columns.",
                        },
                    },
                },
                handler=self._compute_lddt,
            ),
            MCPTool(
                name="refine_msa",
                description=(
                    "Iteratively refine an existing structural MSA to improve "
                    "alignment quality."
                ),
                input_schema={
                    "type": "object",
                    "required": ["db_path", "msa_path"],
                    "properties": {
                        "db_path": {
                            "type": "string",
                            "description": "Path to the FoldMason database directory.",
                        },
                        "msa_path": {
                            "type": "string",
                            "description": "Path to the input alignment file.",
                        },
                        "iterations": {
                            "type": "integer",
                            "default": 100,
                            "description": "Number of refinement iterations.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "refined_msa": {
                            "type": "string",
                            "description": "Path to the refined alignment FASTA.",
                        },
                        "score_before": {"type": "number"},
                        "score_after": {"type": "number"},
                    },
                },
                handler=self._refine_msa,
            ),
            MCPTool(
                name="cluster_structures",
                description=(
                    "Cluster a set of PDB structures by 3Di structural similarity. "
                    "Uses FoldMason's structural clustering module."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_paths"],
                    "properties": {
                        "pdb_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Absolute paths to input PDB files.",
                        },
                        "precluster": {
                            "type": "boolean",
                            "default": True,
                            "description": "Run fast pre-clustering for large datasets.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "clusters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "cluster_id": {"type": "integer"},
                                    "representative": {"type": "string"},
                                    "members": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                        "n_clusters": {"type": "integer"},
                    },
                },
                handler=self._cluster_structures,
            ),
        ]
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------
    # Tool handler implementations
    # ------------------------------------------------------------------

    def _easy_msa(
        self,
        pdb_paths: list[str],
        output_prefix: str,
        report_mode: int = 1,
    ) -> dict[str, Any]:
        """Run foldmason easy-msa on a list of PDB files."""
        output_prefix_path = Path(output_prefix)
        output_prefix_path.parent.mkdir(parents=True, exist_ok=True)

        # Write a temporary text file listing PDB paths (foldmason accepts a list file)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as fh:
            fh.write("\n".join(pdb_paths))
            list_file = fh.name

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                args = [
                    "easy-msa",
                    list_file,
                    str(output_prefix_path),
                    tmp_dir,
                    "--report-mode", str(report_mode),
                ]
                _run_foldmason(args)
        finally:
            Path(list_file).unlink(missing_ok=True)

        # Resolve output file paths (foldmason naming convention)
        aa_fasta = str(output_prefix_path) + "_aa.fasta"
        tdi_fasta = str(output_prefix_path) + "_3di.fasta"
        newick = str(output_prefix_path) + ".nw"
        html = str(output_prefix_path) + ".html" if report_mode >= 1 else None

        return {
            "aa_fasta": aa_fasta if Path(aa_fasta).exists() else None,
            "3di_fasta": tdi_fasta if Path(tdi_fasta).exists() else None,
            "newick_tree": newick if Path(newick).exists() else None,
            "html_report": html if html and Path(html).exists() else None,
        }

    def _compute_lddt(
        self,
        db_path: str,
        msa_path: str,
    ) -> dict[str, Any]:
        """Compute per-column lDDT for an existing structural MSA."""
        # IMPLEMENTATION SKELETON
        # foldmason msa2lddt <db> <msa> <output>
        output_path = Path(msa_path).with_suffix(".lddt.tsv")
        _run_foldmason(
            ["msa2lddt", db_path, msa_path, str(output_path)]
        )

        # Parse TSV: column_idx \t lddt_score
        scores: list[float] = []
        if output_path.exists():
            for line in output_path.read_text().splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        scores.append(float(parts[1]))
                    except ValueError:
                        continue

        mean_lddt = sum(scores) / len(scores) if scores else 0.0
        return {"lddt_scores": scores, "mean_lddt": mean_lddt}

    def _refine_msa(
        self,
        db_path: str,
        msa_path: str,
        iterations: int = 100,
    ) -> dict[str, Any]:
        """Iterative MSA refinement."""
        # IMPLEMENTATION SKELETON
        output_path = Path(msa_path).with_suffix(".refined.fasta")
        _run_foldmason(
            [
                "msa-refine",
                db_path,
                msa_path,
                str(output_path),
                "--num-iterations", str(iterations),
            ]
        )
        return {
            "refined_msa": str(output_path) if output_path.exists() else None,
            "score_before": None,  # foldmason does not expose per-iteration scores
            "score_after": None,
        }

    def _cluster_structures(
        self,
        pdb_paths: list[str],
        precluster: bool = True,
    ) -> dict[str, Any]:
        """Cluster structures by 3Di similarity."""
        # IMPLEMENTATION SKELETON
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            list_file = tmp / "input.txt"
            list_file.write_text("\n".join(pdb_paths))
            output_prefix = tmp / "clusters"

            extra_args = ["--cluster-steps", "1"] if precluster else []
            _run_foldmason(
                [
                    "easy-cluster",
                    str(list_file),
                    str(output_prefix),
                    tmp_dir,
                    *extra_args,
                ]
            )

            # Parse cluster TSV: representative \t member
            tsv = Path(str(output_prefix) + "_cluster.tsv")
            clusters_raw: dict[str, list[str]] = {}
            if tsv.exists():
                for line in tsv.read_text().splitlines():
                    parts = line.strip().split("\t")
                    if len(parts) == 2:
                        rep, member = parts
                        clusters_raw.setdefault(rep, []).append(member)

        clusters = [
            {
                "cluster_id": i,
                "representative": rep,
                "members": members,
            }
            for i, (rep, members) in enumerate(clusters_raw.items())
        ]
        return {"clusters": clusters, "n_clusters": len(clusters)}


def get_server() -> FoldMasonMCPServer:
    """Convenience factory used by the pipeline."""
    return FoldMasonMCPServer()
