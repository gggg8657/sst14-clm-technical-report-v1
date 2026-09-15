"""
MolMIM Tool — Auxiliary: Small Molecule Generation & Optimisation
=================================================================
Wraps the NVIDIA NIM MolMIM endpoint for latent-space molecular generation
and property-guided optimisation using CMA-ES.

Endpoint:
    https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate

Reference client: PRST_N_FM/bionemo/molmim_client.py
This module defines the *agentic tool interface* on top of the same API.

Typical use in the SSTR2 pipeline:
    - Generate analogues of seed somatostatin-like small molecules
    - Optimise QED / logP / docking-score-guided properties
    - Complement the peptide binder arm with a small-molecule arm
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/nvidia/molmim/generate"
_TIMEOUT_S = 120


class MolMIMTool(BaseTool):
    """Tool wrapper for the MolMIM NIM API.

    Interface definition:
        generate(smi, num_molecules, algorithm, property_name) -> ToolResult

    execute() dispatcher accepts action="generate".
    """

    name = "molmim"
    description = (
        "Generate and optimise small molecules in MolMIM latent space. "
        "Supports CMA-ES property-guided optimisation (QED, logP, …) "
        "and random neighbourhood sampling."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary action
    # ------------------------------------------------------------------

    def generate(
        self,
        smi: str,
        num_molecules: int = 10,
        algorithm: str = "CMA-ES",
        property_name: str = "QED",
        minimize: bool = False,
        min_similarity: float = 0.3,
        particles: int = 20,
        iterations: int = 3,
    ) -> ToolResult:
        """Generate optimised molecules from a seed SMILES.

        Uses the CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
        algorithm to explore the MolMIM latent space while optimising the
        specified molecular property.

        Args:
            smi:           Seed SMILES string (starting point for optimisation).
            num_molecules: Number of molecules to return (default 10).
            algorithm:     Optimisation algorithm. "CMA-ES" is the primary option.
                           Use "none" for random neighbourhood sampling.
            property_name: Molecular property to optimise.
                           Common options: "QED", "logP".
                           The API also supports custom scoring via plugin.
            minimize:      If True, minimise the property (e.g. minimise logP).
                           If False (default), maximise it.
            min_similarity: Minimum Tanimoto similarity to the seed molecule
                            (range 0–1, default 0.3). Ensures chemical relevance.
            particles:     CMA-ES population size (default 20).
            iterations:    Number of CMA-ES generations (default 3).

        Returns:
            ToolResult where data contains:
                - "molecules": list of molecule dicts, each containing:
                    - "smiles" or "sample": SMILES string of the generated molecule
                    - "score":              Property score (float)
                    - "similarity":         Tanimoto similarity to seed (float)
                - "n_molecules": Number of molecules returned.

        Example:
            tool = MolMIMTool()
            result = tool.generate(
                smi="CC(=O)Nc1ccc(O)cc1",
                num_molecules=20,
                property_name="QED",
                min_similarity=0.4,
            )
            for mol in result.data["molecules"]:
                print(mol.get("smiles") or mol.get("sample"), mol["score"])
        """
        if not smi:
            return ToolResult.fail("smi (seed SMILES) must not be empty")

        payload: dict[str, Any] = {
            "smi": smi,
            "num_molecules": num_molecules,
            "algorithm": algorithm,
            "property_name": property_name,
            "minimize": minimize,
            "min_similarity": min_similarity,
            "particles": particles,
            "iterations": iterations,
        }

        try:
            data, elapsed_ms = self._post_timed("", payload)

            # The hosted API sometimes returns molecules as a JSON string
            molecules = data.get("molecules", [])
            if isinstance(molecules, str):
                try:
                    molecules = json.loads(molecules)
                except json.JSONDecodeError:
                    logger.warning("Could not parse molecules JSON string")
                    molecules = []
            data["molecules"] = molecules
            data["n_molecules"] = len(molecules)

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                algorithm=algorithm,
                property_name=property_name,
            )
        except Exception as exc:
            logger.error("MolMIM generate failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "generate" (default)
        """
        action = kwargs.pop("action", "generate")
        if action == "generate":
            return self.generate(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> MolMIMTool:
    """Convenience factory used by the pipeline."""
    return MolMIMTool(**kwargs)
