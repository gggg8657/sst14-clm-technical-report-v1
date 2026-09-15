"""
Boltz-2 Tool — Step05: Complex Structure & Binding Affinity Prediction
=======================================================================
Wraps the NVIDIA NIM Boltz-2 endpoint for protein complex structure prediction
with integrated binding affinity scoring.

This is a NEW tool (no reference client in PRST_N_FM/bionemo/).
Boltz-2 complements DiffDock by providing:
  - Full complex structure prediction (backbone + side-chains)
  - Predicted binding affinity (ΔG, Kd estimate)
  - Confidence scores equivalent to AlphaFold3 metrics

Endpoint:
    https://health.api.nvidia.com/v1/biology/mit/boltz2/predict

Typical use in the SSTR2 pipeline:
    - Rank candidate peptide binders by predicted binding affinity
    - Alternative to DiffDock when a full complex structure is needed
    - Cross-validate top candidates before experimental synthesis
"""

from __future__ import annotations

import logging
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
_TIMEOUT_S = 600  # complex prediction with affinity can be slow


class Boltz2Tool(BaseTool):
    """Tool wrapper for the Boltz-2 NIM API.

    Boltz-2 accepts a list of chain descriptors (same schema as OpenFold3)
    and returns the predicted complex structure together with a binding
    affinity score.

    Interface definition:
        predict_complex(sequences) -> ToolResult

    execute() dispatcher accepts action="predict_complex".
    """

    name = "boltz2"
    description = (
        "Predict protein complex structure AND binding affinity using Boltz-2. "
        "Returns complex coordinates (mmCIF) and a predicted ΔG binding score. "
        "Complements DiffDock for peptide binder ranking."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary action
    # ------------------------------------------------------------------

    def predict_complex(
        self,
        sequences: list[dict[str, Any]],
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        diffusion_samples: int = 1,
    ) -> ToolResult:
        """Predict a protein complex structure with binding affinity.

        Args:
            sequences:        List of chain descriptors. Each element is a dict:
                              - Protein chain:
                                {"protein": {"id": "A", "sequence": "MKTAY..."}}
                              - Ligand (SMILES):
                                {"ligand": {"id": "X", "smiles": "CCO"}}
                              The first protein chain is treated as the receptor
                              and the second as the binder (for affinity scoring).
            recycling_steps:  Number of Evoformer recycling iterations (default 3).
                              More cycles -> higher accuracy, slower.
            sampling_steps:   Diffusion sampling steps for the structure module
                              (default 200).
            diffusion_samples: Number of structure samples to generate (default 1).
                               Use >1 for ensemble predictions.

        Returns:
            ToolResult where data contains:
                - "mmcif":              Predicted complex in mmCIF format (str)
                - "binding_affinity":   Predicted binding free energy ΔG in kcal/mol
                                        (float | None — may be None if model cannot
                                        produce an estimate)
                - "affinity_confidence": Model confidence in the affinity prediction
                                         (float in [0, 1])
                - "confidence_score":   Overall complex confidence (float in [0, 100])
                - "plddt_per_chain":    Per-chain mean pLDDT scores (dict[str, float])
                - "pae_matrix":         Predicted aligned error (list[list[float]])

        Example:
            tool = Boltz2Tool()
            result = tool.predict_complex(
                sequences=[
                    {"protein": {"id": "A", "sequence": "MGNLS..."}},  # SSTR2
                    {"protein": {"id": "B", "sequence": "ACKNFF..."}}, # binder
                ]
            )
            ddg   = result.data["binding_affinity"]
            mmcif = result.data["mmcif"]
        """
        if not sequences:
            return ToolResult.fail("sequences list must not be empty")
        if len(sequences) < 2:
            return ToolResult.fail(
                "At least 2 chains are required for complex/affinity prediction "
                "(receptor + binder)"
            )

        payload: dict[str, Any] = {
            "sequences": sequences,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
        }

        try:
            data, elapsed_ms = self._post_timed("", payload)

            # Normalise affinity field — different API versions may use
            # different key names
            affinity = (
                data.get("binding_affinity")
                or data.get("affinity_kcal_mol")
                or data.get("predicted_affinity")
            )
            data["binding_affinity"] = affinity

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                n_chains=len(sequences),
                has_affinity=affinity is not None,
            )
        except Exception as exc:
            logger.error("Boltz2 predict_complex failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "predict_complex" (default)
        """
        action = kwargs.pop("action", "predict_complex")
        if action == "predict_complex":
            return self.predict_complex(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> Boltz2Tool:
    """Convenience factory used by the pipeline."""
    return Boltz2Tool(**kwargs)
