"""
OpenFold3 Tool — Step01: Target Structure Prediction
======================================================
Wraps the NVIDIA NIM OpenFold3 endpoint for AlphaFold3-style complex prediction.
This endpoint is NEW relative to PRST_N_FM/bionemo/ (no reference client exists).

Endpoint:
    https://health.api.nvidia.com/v1/biology/openfold/openfold3

Typical use in the SSTR2 pipeline:
    - Predict the apo or holo structure of SSTR2 with a peptide partner
    - Generate reference complex structures for downstream hotspot analysis
"""

from __future__ import annotations

import logging
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/openfold/openfold3"
_TIMEOUT_S = 600  # structure prediction can be slow


class OpenFold3Tool(BaseTool):
    """Tool wrapper for the OpenFold3 NIM API.

    Supports single-chain and multi-chain complex prediction including
    protein–protein and protein–ligand inputs.

    Interface definition:
        predict_complex(sequences, msa, templates) -> ToolResult

    execute() dispatcher accepts action="predict_complex" with matching kwargs.
    """

    name = "openfold3"
    description = (
        "Predict 3-D structure of a protein complex (protein–protein or "
        "protein–ligand) using OpenFold3 / AlphaFold3-style deep learning."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary action
    # ------------------------------------------------------------------

    def predict_complex(
        self,
        sequences: list[dict[str, Any]],
        msa: str | None = None,
        templates: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        """Predict a multi-chain complex structure.

        Args:
            sequences: List of chain descriptors. Each element is one of:
                - Protein chain:
                    {"protein": {"id": "A", "sequence": "MKTAYIA..."}}
                - Ligand (SMILES):
                    {"ligand": {"id": "X", "smiles": "CCO"}}
                - RNA/DNA chains follow the same pattern with key "rna"/"dna".
            msa:       Optional pre-computed MSA string in A3M format for the
                       first protein chain. When None the API runs its own MSA.
            templates: Optional list of template dicts. Each dict should
                       contain {"pdb_id": "...", "chain_id": "..."} or a
                       raw mmCIF block under "mmcif".

        Returns:
            ToolResult where data contains:
                - "mmcif":           Full predicted complex in mmCIF format (str)
                - "confidence":      Per-chain confidence metrics (dict)
                - "pae_matrix":      Predicted aligned error matrix (list[list[float]])
                - "plddt_per_residue": Per-residue pLDDT scores (list[float])

        Example:
            tool = OpenFold3Tool()
            result = tool.predict_complex(
                sequences=[
                    {"protein": {"id": "A", "sequence": "MGNLS..."}},
                    {"protein": {"id": "B", "sequence": "ACKNFF..."}},
                ]
            )
            mmcif_text = result.data["mmcif"]
        """
        if not sequences:
            return ToolResult.fail("sequences list must not be empty")

        payload: dict[str, Any] = {"sequences": sequences}
        if msa is not None:
            payload["msa"] = msa
        if templates is not None:
            payload["templates"] = templates

        try:
            data, elapsed_ms = self._post_timed("", payload)
            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                n_chains=len(sequences),
            )
        except Exception as exc:
            logger.error("OpenFold3 predict_complex failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions (passed as action="..." kwarg):
            "predict_complex" (default) — calls predict_complex()

        All remaining kwargs are forwarded to the target method.
        """
        action = kwargs.pop("action", "predict_complex")
        if action == "predict_complex":
            return self.predict_complex(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> OpenFold3Tool:
    """Convenience factory used by the pipeline."""
    return OpenFold3Tool(**kwargs)
