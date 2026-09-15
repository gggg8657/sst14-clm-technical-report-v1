"""
DiffDock Tool — Step05: Protein–Ligand Docking
===============================================
Wraps the NVIDIA NIM DiffDock endpoint for blind protein–ligand docking.
Accepts PDB + SDF (or SMILES via RDKit conversion) and returns ranked poses.

Endpoint:
    https://health.api.nvidia.com/v1/biology/mit/diffdock

Reference client: PRST_N_FM/bionemo/diffdock_client.py
This module defines the *agentic tool interface* on top of the same API.

Typical use in the SSTR2 pipeline:
    - dock small molecules or peptide-derived fragments into SSTR2 pocket
    - obtain confidence-ranked binding poses for downstream MM-PBSA scoring
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
_TIMEOUT_S = 300

# RDKit is optional — only required for dock_smiles()
try:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import AllChem  # type: ignore[import-untyped]

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


class DiffDockTool(BaseTool):
    """Tool wrapper for the DiffDock NIM API.

    Interface definition:
        dock(protein_pdb, ligand_sdf, num_poses)          -> ToolResult
        dock_smiles(protein_pdb, smiles, num_poses)       -> ToolResult (needs RDKit)

    execute() dispatcher accepts action="dock" | "dock_smiles".
    """

    name = "diffdock"
    description = (
        "Dock a small molecule into a protein binding site using DiffDock "
        "(diffusion-based blind docking). Returns ranked binding poses with "
        "confidence scores."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary actions
    # ------------------------------------------------------------------

    def dock(
        self,
        protein_pdb: str,
        ligand_sdf: str,
        num_poses: int = 10,
        time_divisions: int = 20,
        steps: int = 18,
    ) -> ToolResult:
        """Dock a ligand (SDF format) into a protein structure.

        Args:
            protein_pdb:    PDB content of the receptor protein (str).
            ligand_sdf:     Ligand structure in SDF format (str).
            num_poses:      Number of docking poses to generate (default 10).
            time_divisions: Diffusion time discretisation (default 20).
            steps:          Diffusion reverse-process steps (default 18).

        Returns:
            ToolResult where data contains the raw API response plus a
            convenience field:
                - "poses": list of dicts sorted by confidence (descending)
                    each dict: {"pose_pdb": str, "confidence": float, "rank": int}
                - "best_pose": The top-ranked pose dict.

        Example:
            tool = DiffDockTool()
            result = tool.dock(
                protein_pdb=Path("sstr2.pdb").read_text(),
                ligand_sdf=Path("octreotide.sdf").read_text(),
                num_poses=10,
            )
            top_pose = result.data["best_pose"]["pose_pdb"]
        """
        if not protein_pdb:
            return ToolResult.fail("protein_pdb must not be empty")
        if not ligand_sdf:
            return ToolResult.fail("ligand_sdf must not be empty")

        payload: dict[str, Any] = {
            "ligand": ligand_sdf,
            "ligand_file_type": "sdf",
            "protein": protein_pdb,
            "num_poses": num_poses,
            "time_divisions": time_divisions,
            "steps": steps,
            "save_trajectory": False,
            "is_staged": False,
        }

        try:
            data, elapsed_ms = self._post_timed("", payload)
            poses = self._parse_poses(data)
            data["poses"] = poses
            data["best_pose"] = poses[0] if poses else None

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                action="dock",
                n_poses=len(poses),
            )
        except Exception as exc:
            logger.error("DiffDock dock failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name, action="dock")

    def dock_smiles(
        self,
        protein_pdb: str,
        smiles: str,
        num_poses: int = 10,
    ) -> ToolResult:
        """Dock a ligand specified as a SMILES string.

        Converts SMILES to 3-D SDF using RDKit (MMFF94 geometry optimisation)
        before calling the DiffDock API.

        Requires: RDKit (conda install -c conda-forge rdkit)

        Args:
            protein_pdb: PDB content of the receptor protein (str).
            smiles:      SMILES string of the ligand.
                         Example: "CC(=O)Nc1ccc(O)cc1"  (paracetamol)
            num_poses:   Number of docking poses to generate.

        Returns:
            ToolResult — same structure as dock(), plus:
                - data["smiles"]: The input SMILES string echoed back.
        """
        if not _HAS_RDKIT:
            return ToolResult.fail(
                "RDKit is required for dock_smiles(). "
                "Install with: conda install -c conda-forge rdkit"
            )

        try:
            ligand_sdf = self._smiles_to_sdf(smiles)
        except ValueError as exc:
            return ToolResult.fail(f"SMILES to SDF conversion failed: {exc}")

        result = self.dock(protein_pdb, ligand_sdf, num_poses=num_poses)
        if result.success:
            result.data["smiles"] = smiles
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _smiles_to_sdf(smiles: str) -> str:
        """Convert a SMILES string to a 3-D SDF string using RDKit.

        Args:
            smiles: Valid SMILES string.

        Returns:
            SDF-format string with MMFF94-optimised 3-D coordinates.

        Raises:
            ValueError: If the SMILES is invalid or embedding fails.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            raise ValueError(f"3-D embedding failed for SMILES: {smiles!r}")
        AllChem.MMFFOptimizeMolecule(mol)

        # Write to a temp file and read back (SDWriter requires a path)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".sdf")
        os.close(tmp_fd)
        try:
            writer = Chem.SDWriter(tmp_path)
            writer.write(mol)
            writer.close()
            return Path(tmp_path).read_text(encoding="utf-8")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _parse_poses(data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and rank docking poses from the raw API response.

        The DiffDock API returns poses under the "output" key as a list of dicts
        with "pose_pdb" and "confidence" fields.

        Args:
            data: Raw API response dict.

        Returns:
            List of pose dicts sorted by confidence descending, with rank added.
        """
        raw_poses: list[dict[str, Any]] = data.get("output", [])
        if not isinstance(raw_poses, list):
            return []

        # Sort by confidence descending (higher = better)
        sorted_poses = sorted(
            raw_poses,
            key=lambda p: float(p.get("confidence", 0.0)),
            reverse=True,
        )
        for rank, pose in enumerate(sorted_poses, start=1):
            pose["rank"] = rank

        return sorted_poses

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "dock"        (default) — dock with SDF ligand
            "dock_smiles"           — dock with SMILES ligand (needs RDKit)
        """
        action = kwargs.pop("action", "dock")
        if action == "dock":
            return self.dock(**kwargs)
        if action == "dock_smiles":
            return self.dock_smiles(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> DiffDockTool:
    """Convenience factory used by the pipeline."""
    return DiffDockTool(**kwargs)
