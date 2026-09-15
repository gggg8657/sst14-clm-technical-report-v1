"""
PyMOL MCP Server — Step07: Structural Visualisation
====================================================
Exposes PyMOL headless rendering as MCP tool endpoints so the agent can
generate publication-quality structural images as part of the agentic flow.

All tools generate a PyMOL script (.pml) and run PyMOL in command-line mode:
    pymol -c script.pml

IMPORTANT: This is an *interface definition* with working subprocess skeletons.
Actual rendering requires:
    - PyMOL installed and on PATH (open-source or commercial)
    - For electrostatics: APBS installed and on PATH

Tools exposed:
    1. render_overview          — full complex overview at low resolution
    2. render_closeup           — binding pocket close-up
    3. render_interface_contacts — interface hydrogen bonds / contacts
    4. render_electrostatics    — surface electrostatic potential
    5. render_plddt_spectrum    — pLDDT B-factor coloured ribbon
    6. batch_render             — apply a template PML script to many PDBs
    7. create_comparison_panel  — side-by-side comparison PNG

Usage:
    server = PyMOLMCPServer()
    result = server.dispatch(
        "render_overview",
        receptor_pdb="/data/sstr2.pdb",
        peptide_pdb="/data/binder1.pdb",
        output_png="/output/overview.png",
    )
    png_path = result["png_path"]
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base_server import MCPServer, MCPTool

logger = logging.getLogger(__name__)

_PYMOL_BIN = shutil.which("pymol") or "pymol"

# Default rendering resolution
_DEFAULT_WIDTH = 2400
_DEFAULT_HEIGHT = 1800

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _sanitize_pml_path(path: str) -> str:
    """Reject paths containing characters that are special in PML context."""
    if re.search(r'[\n\r]', path):
        raise ValueError(f"Illegal characters in path for PML script: {path!r}")
    resolved = Path(path).resolve()
    return str(resolved)


_ALLOWED_PML_COMMANDS = re.compile(
    r'^(load|hide|show|color|orient|zoom|bg_color|set|ray|png|quit|distance|spectrum|ramp_new|select|deselect|cmd|create|delete|disable|enable|rebuild|util|cartoon|stick|sphere|surface|mesh|label|unlabel)\b',
    re.MULTILINE,
)


def _validate_pml_template(template: str) -> None:
    """Raise ValueError if template contains disallowed PML commands."""
    for line in template.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if not _ALLOWED_PML_COMMANDS.match(stripped):
            raise ValueError(f"Disallowed PML command in template: {stripped!r}")


def _run_pymol(pml_script: str, timeout: int = 120) -> None:
    """Write a PML script to a temp file and execute PyMOL headlessly.

    Args:
        pml_script: Contents of the PyMOL script to execute.
        timeout:    Maximum execution time in seconds.

    Raises:
        RuntimeError: If PyMOL exits with a non-zero return code.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pml", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(pml_script)
        script_path = fh.name

    try:
        cmd = [_PYMOL_BIN, "-c", script_path]
        logger.debug("Running PyMOL: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"PyMOL failed (exit {proc.returncode}):\n{proc.stderr}"
            )
    finally:
        Path(script_path).unlink(missing_ok=True)


class PyMOLMCPServer(MCPServer):
    """MCP server for PyMOL headless structural visualisation."""

    def __init__(self) -> None:
        super().__init__("pymol")
        self._register_all()

    def _register_all(self) -> None:
        tools = [
            MCPTool(
                name="render_overview",
                description=(
                    "Render a full overview image of the receptor–peptide complex "
                    "as a high-resolution PNG."
                ),
                input_schema={
                    "type": "object",
                    "required": ["receptor_pdb", "peptide_pdb", "output_png"],
                    "properties": {
                        "receptor_pdb": {
                            "type": "string",
                            "description": "Path to the receptor PDB file.",
                        },
                        "peptide_pdb": {
                            "type": "string",
                            "description": "Path to the peptide/binder PDB file.",
                        },
                        "output_png": {
                            "type": "string",
                            "description": "Output PNG file path.",
                        },
                        "resolution": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "default": [2400, 1800],
                            "description": "[width, height] in pixels.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "png_path": {"type": "string"},
                    },
                },
                handler=self._render_overview,
            ),
            MCPTool(
                name="render_closeup",
                description=(
                    "Render a close-up view of the binding pocket, centred on "
                    "specified hotspot residues."
                ),
                input_schema={
                    "type": "object",
                    "required": [
                        "receptor_pdb",
                        "peptide_pdb",
                        "pocket_residues",
                        "output_png",
                    ],
                    "properties": {
                        "receptor_pdb": {"type": "string"},
                        "peptide_pdb": {"type": "string"},
                        "pocket_residues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Residue identifiers to centre on, "
                                "e.g. ['B122', 'B127']."
                            ),
                        },
                        "output_png": {"type": "string"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"png_path": {"type": "string"}},
                },
                handler=self._render_closeup,
            ),
            MCPTool(
                name="render_interface_contacts",
                description=(
                    "Render interface hydrogen bonds and non-bonded contacts "
                    "between receptor and peptide."
                ),
                input_schema={
                    "type": "object",
                    "required": ["receptor_pdb", "peptide_pdb", "output_png"],
                    "properties": {
                        "receptor_pdb": {"type": "string"},
                        "peptide_pdb": {"type": "string"},
                        "output_png": {"type": "string"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"png_path": {"type": "string"}},
                },
                handler=self._render_interface_contacts,
            ),
            MCPTool(
                name="render_electrostatics",
                description=(
                    "Render the molecular surface coloured by electrostatic "
                    "potential computed from a pre-calculated dx map."
                ),
                input_schema={
                    "type": "object",
                    "required": [
                        "receptor_pdb",
                        "peptide_pdb",
                        "dx_map",
                        "output_png",
                    ],
                    "properties": {
                        "receptor_pdb": {"type": "string"},
                        "peptide_pdb": {"type": "string"},
                        "dx_map": {
                            "type": "string",
                            "description": "Path to the electrostatic potential dx file.",
                        },
                        "output_png": {"type": "string"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"png_path": {"type": "string"}},
                },
                handler=self._render_electrostatics,
            ),
            MCPTool(
                name="render_plddt_spectrum",
                description=(
                    "Render a ribbon diagram coloured by pLDDT confidence "
                    "(stored in the B-factor column)."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_path", "output_png"],
                    "properties": {
                        "pdb_path": {
                            "type": "string",
                            "description": "Path to the PDB file with pLDDT in B-factor.",
                        },
                        "output_png": {"type": "string"},
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"png_path": {"type": "string"}},
                },
                handler=self._render_plddt_spectrum,
            ),
            MCPTool(
                name="batch_render",
                description=(
                    "Apply a template PML script to multiple PDB files, "
                    "substituting the INPUT_PDB and OUTPUT_PNG placeholders."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_paths", "template_script", "output_dir"],
                    "properties": {
                        "pdb_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "PDB files to render.",
                        },
                        "template_script": {
                            "type": "string",
                            "description": (
                                "PML script template. Use {input_pdb} and "
                                "{output_png} as placeholders."
                            ),
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Directory where output PNGs are written.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "png_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "n_rendered": {"type": "integer"},
                        "n_failed": {"type": "integer"},
                    },
                },
                handler=self._batch_render,
            ),
            MCPTool(
                name="create_comparison_panel",
                description=(
                    "Render multiple binder PDB files side-by-side with the same "
                    "receptor and produce a combined comparison PNG."
                ),
                input_schema={
                    "type": "object",
                    "required": ["pdb_paths", "receptor_pdb", "output_png"],
                    "properties": {
                        "pdb_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Binder PDB files to compare.",
                        },
                        "receptor_pdb": {
                            "type": "string",
                            "description": "Receptor PDB file to show in each panel.",
                        },
                        "output_png": {
                            "type": "string",
                            "description": "Output comparison panel PNG path.",
                        },
                    },
                },
                output_schema={
                    "type": "object",
                    "properties": {"png_path": {"type": "string"}},
                },
                handler=self._create_comparison_panel,
            ),
        ]
        for tool in tools:
            self.register(tool)

    # ------------------------------------------------------------------
    # PML script builders
    # ------------------------------------------------------------------

    @staticmethod
    def _pml_load_complex(
        receptor_pdb: str,
        peptide_pdb: str,
        receptor_name: str = "receptor",
        peptide_name: str = "peptide",
    ) -> str:
        """Return PML lines to load receptor and peptide."""
        return (
            f"load {_sanitize_pml_path(receptor_pdb)}, {receptor_name}\n"
            f"load {_sanitize_pml_path(peptide_pdb)}, {peptide_name}\n"
        )

    @staticmethod
    def _pml_setup_render(width: int, height: int, output_png: str) -> str:
        """Return PML lines to configure ray-tracing and save the image."""
        return (
            f"set ray_trace_mode, 1\n"
            f"set antialias, 2\n"
            f"ray {width}, {height}\n"
            f"png {_sanitize_pml_path(output_png)}, dpi=300\n"
            f"quit\n"
        )

    # ------------------------------------------------------------------
    # Tool handler implementations
    # ------------------------------------------------------------------

    def _render_overview(
        self,
        receptor_pdb: str,
        peptide_pdb: str,
        output_png: str,
        resolution: list[int] | None = None,
    ) -> dict[str, Any]:
        """Render full complex overview."""
        w, h = (resolution or [_DEFAULT_WIDTH, _DEFAULT_HEIGHT])
        pml = (
            self._pml_load_complex(receptor_pdb, peptide_pdb)
            + "hide everything\n"
            + "show cartoon, receptor\n"
            + "show cartoon, peptide\n"
            + "color marine, receptor\n"
            + "color orange, peptide\n"
            + "orient\n"
            + "zoom all, 5\n"
            + "bg_color white\n"
            + self._pml_setup_render(w, h, output_png)
        )
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        _run_pymol(pml)
        return {"png_path": output_png}

    def _render_closeup(
        self,
        receptor_pdb: str,
        peptide_pdb: str,
        pocket_residues: list[str],
        output_png: str,
    ) -> dict[str, Any]:
        """Render close-up view centred on pocket residues."""
        # Build residue selection from e.g. ["B122", "B127"] -> "resi 122+127 and chain B"
        # Group by chain
        by_chain: dict[str, list[str]] = {}
        for resid in pocket_residues:
            chain = resid[0]
            rnum = resid[1:]
            by_chain.setdefault(chain, []).append(rnum)

        sel_parts = [
            f"(resi {'+'.join(rnums)} and chain {chain})"
            for chain, rnums in by_chain.items()
        ]
        pocket_sel = " or ".join(sel_parts) if sel_parts else "all"

        pml = (
            self._pml_load_complex(receptor_pdb, peptide_pdb)
            + "hide everything\n"
            + "show cartoon, receptor\n"
            + "show cartoon, peptide\n"
            + "show sticks, receptor within 5 of peptide\n"
            + "show sticks, peptide\n"
            + "color marine, receptor\n"
            + "color orange, peptide\n"
            + "color yellow, receptor within 5 of peptide\n"
            + f"zoom {pocket_sel}, 8\n"
            + "bg_color white\n"
            + self._pml_setup_render(_DEFAULT_WIDTH, _DEFAULT_HEIGHT, output_png)
        )
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        _run_pymol(pml)
        return {"png_path": output_png}

    def _render_interface_contacts(
        self,
        receptor_pdb: str,
        peptide_pdb: str,
        output_png: str,
    ) -> dict[str, Any]:
        """Render interface H-bonds and contacts."""
        pml = (
            self._pml_load_complex(receptor_pdb, peptide_pdb)
            + "hide everything\n"
            + "show cartoon, receptor\n"
            + "show cartoon, peptide\n"
            + "show sticks, receptor within 4 of peptide\n"
            + "show sticks, peptide within 4 of receptor\n"
            + "color marine, receptor\n"
            + "color orange, peptide\n"
            # H-bonds between chains
            + "distance hbonds, receptor, peptide, 3.5, mode=2\n"
            + "show dashes, hbonds\n"
            + "color yellow, hbonds\n"
            + "orient\n"
            + "zoom receptor within 6 of peptide, 5\n"
            + "bg_color white\n"
            + self._pml_setup_render(_DEFAULT_WIDTH, _DEFAULT_HEIGHT, output_png)
        )
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        _run_pymol(pml)
        return {"png_path": output_png}

    def _render_electrostatics(
        self,
        receptor_pdb: str,
        peptide_pdb: str,
        dx_map: str,
        output_png: str,
    ) -> dict[str, Any]:
        """Render electrostatic surface potential."""
        pml = (
            self._pml_load_complex(receptor_pdb, peptide_pdb)
            + f"load {_sanitize_pml_path(dx_map)}, esp\n"
            + "hide everything\n"
            + "show surface, receptor\n"
            + "show cartoon, peptide\n"
            + "ramp_new cmap, esp, [-5, 0, 5], [red, white, blue]\n"
            + "set surface_color, cmap, receptor\n"
            + "color orange, peptide\n"
            + "orient\n"
            + "zoom all, 5\n"
            + "bg_color white\n"
            + self._pml_setup_render(_DEFAULT_WIDTH, _DEFAULT_HEIGHT, output_png)
        )
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        _run_pymol(pml)
        return {"png_path": output_png}

    def _render_plddt_spectrum(
        self,
        pdb_path: str,
        output_png: str,
    ) -> dict[str, Any]:
        """Render pLDDT B-factor coloured ribbon."""
        pml = (
            f"load {_sanitize_pml_path(pdb_path)}, model\n"
            "hide everything\n"
            "show cartoon, model\n"
            # Spectrum from blue (pLDDT=50) to red (pLDDT=100) via B-factor
            "spectrum b, blue_white_red, model, minimum=50, maximum=100\n"
            "orient\n"
            "zoom all, 5\n"
            "bg_color white\n"
            + self._pml_setup_render(_DEFAULT_WIDTH, _DEFAULT_HEIGHT, output_png)
        )
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        _run_pymol(pml)
        return {"png_path": output_png}

    def _batch_render(
        self,
        pdb_paths: list[str],
        template_script: str,
        output_dir: str,
    ) -> dict[str, Any]:
        """Apply a template PML script to multiple PDB files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        _validate_pml_template(template_script)

        png_paths: list[str] = []
        n_failed = 0

        for pdb_path in pdb_paths:
            stem = Path(pdb_path).stem
            output_png = str(out_dir / f"{stem}.png")
            safe_pdb = _sanitize_pml_path(pdb_path)
            safe_png = _sanitize_pml_path(output_png)
            pml = template_script.replace("{input_pdb}", safe_pdb).replace(
                "{output_png}", safe_png
            )
            try:
                _run_pymol(pml)
                png_paths.append(output_png)
            except Exception as exc:
                logger.warning("batch_render failed for %s: %s", pdb_path, exc)
                n_failed += 1

        return {
            "png_paths": png_paths,
            "n_rendered": len(png_paths),
            "n_failed": n_failed,
        }

    def _create_comparison_panel(
        self,
        pdb_paths: list[str],
        receptor_pdb: str,
        output_png: str,
    ) -> dict[str, Any]:
        """Render each binder with the receptor and compose a panel.

        Individual images are rendered to a temp directory, then composed
        into a grid using ImageMagick (montage). Falls back to returning
        individual paths if ImageMagick is unavailable.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            individual_pngs: list[str] = []
            for i, binder_pdb in enumerate(pdb_paths):
                out_png = str(Path(tmp_dir) / f"panel_{i:03d}.png")
                try:
                    self._render_overview(
                        receptor_pdb=receptor_pdb,
                        peptide_pdb=binder_pdb,
                        output_png=out_png,
                        resolution=[800, 600],
                    )
                    individual_pngs.append(out_png)
                except Exception as exc:
                    logger.warning("Panel %d render failed: %s", i, exc)

            # Compose with ImageMagick montage if available
            montage_bin = shutil.which("montage")
            if montage_bin and individual_pngs:
                cols = min(4, len(individual_pngs))
                Path(output_png).parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    montage_bin,
                    "-tile", f"{cols}x",
                    "-geometry", "+4+4",
                    *individual_pngs,
                    output_png,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    logger.warning("montage failed: %s", proc.stderr)
                    # Fall back: just copy the first image
                    if individual_pngs:
                        shutil.copy(individual_pngs[0], output_png)
            elif individual_pngs:
                logger.warning(
                    "ImageMagick montage not found. Copying first panel as output."
                )
                Path(output_png).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(individual_pngs[0], output_png)

        return {"png_path": output_png}


def get_server() -> PyMOLMCPServer:
    """Convenience factory used by the pipeline."""
    return PyMOLMCPServer()
