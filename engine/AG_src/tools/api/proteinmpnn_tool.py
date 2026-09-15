"""
ProteinMPNN Tool — Step03: Sequence Design (Inverse Folding)
=============================================================
Wraps the NVIDIA NIM ProteinMPNN endpoint.
Given a backbone PDB, samples optimised amino-acid sequences that would
fold into that backbone.

Endpoint:
    https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict

Reference client: PRST_N_FM/bionemo/proteinmpnn_client.py
This module defines the *agentic tool interface* on top of the same API.

Typical use in the SSTR2 pipeline:
    - Take RFdiffusion backbone -> predict sequences -> ESMFold for validation
"""

from __future__ import annotations

import logging
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
_TIMEOUT_S = 120


class ProteinMPNNTool(BaseTool):
    """Tool wrapper for the ProteinMPNN NIM API.

    Interface definition:
        predict_sequences(backbone_pdb, num_seq, sampling_temp) -> ToolResult
        parse_fasta(fasta_text) -> list[dict]

    execute() dispatcher accepts action="predict_sequences".
    """

    name = "proteinmpnn"
    description = (
        "Design amino-acid sequences for a given protein backbone using "
        "ProteinMPNN inverse-folding model."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary action
    # ------------------------------------------------------------------

    def predict_sequences(
        self,
        backbone_pdb: str,
        num_seq: int = 8,
        sampling_temp: float = 0.2,
        ca_only: bool = False,
        use_soluble_model: bool = True,
    ) -> ToolResult:
        """Sample amino-acid sequences for the given backbone.

        Args:
            backbone_pdb:      PDB content of the backbone structure (str).
                               Typically the output of RFdiffusion.
            num_seq:           Number of sequences to generate per target
                               (default 8; increase for diversity screening).
            sampling_temp:     Sampling temperature in [0, 1].
                               Lower = more conservative (closer to consensus).
                               Higher = more diverse / exploratory.
                               Recommended range: 0.1 – 0.5.
            ca_only:           If True, use only Cα atoms (faster, less accurate).
            use_soluble_model: Use the soluble-protein variant of the model
                               (recommended for peptide binder design).

        Returns:
            ToolResult where data contains:
                - "sequences":  Multi-FASTA text string with all designed sequences.
                - "parsed":     list[dict] — each dict has "header" and "sequence".
                                Convenience field pre-populated by parse_fasta().

        Example:
            tool = ProteinMPNNTool()
            result = tool.predict_sequences(backbone_pdb_text, num_seq=16)
            for entry in result.data["parsed"]:
                print(entry["header"], entry["sequence"])
        """
        if not backbone_pdb:
            return ToolResult.fail("backbone_pdb must not be empty")

        # The API requires sampling_temp as a list
        temp_list = [sampling_temp] if isinstance(sampling_temp, (int, float)) else sampling_temp

        payload: dict[str, Any] = {
            "input_pdb": backbone_pdb,
            "num_seq_per_target": num_seq,
            "sampling_temp": temp_list,
            "ca_only": ca_only,
            "use_soluble_model": use_soluble_model,
        }

        try:
            data, elapsed_ms = self._post_timed("", payload)

            # Pre-parse the FASTA for downstream convenience
            fasta_text: str = data.get("sequences", "")
            data["parsed"] = self.parse_fasta(fasta_text) if fasta_text else []

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                n_sequences=len(data["parsed"]),
            )
        except Exception as exc:
            logger.error("ProteinMPNN predict_sequences failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    # ------------------------------------------------------------------
    # Utility: FASTA parser
    # ------------------------------------------------------------------

    @staticmethod
    def parse_fasta(fasta_text: str) -> list[dict[str, str]]:
        """Parse a multi-FASTA string into a list of header/sequence dicts.

        Args:
            fasta_text: Raw multi-FASTA string as returned by the API.

        Returns:
            List of dicts: [{"header": "...", "sequence": "..."}, ...]

        Example:
            entries = ProteinMPNNTool.parse_fasta(">seq1 score=0.42\\nMKITA...")
            # -> [{"header": "seq1 score=0.42", "sequence": "MKITA..."}]
        """
        entries: list[dict[str, str]] = []
        current_header: str | None = None
        seq_parts: list[str] = []

        for line in fasta_text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    entries.append(
                        {"header": current_header, "sequence": "".join(seq_parts)}
                    )
                current_header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)

        if current_header is not None:
            entries.append(
                {"header": current_header, "sequence": "".join(seq_parts)}
            )

        return entries

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "predict_sequences" (default)
        """
        action = kwargs.pop("action", "predict_sequences")
        if action == "predict_sequences":
            return self.predict_sequences(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> ProteinMPNNTool:
    """Convenience factory used by the pipeline."""
    return ProteinMPNNTool(**kwargs)
