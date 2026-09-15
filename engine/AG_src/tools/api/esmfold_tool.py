"""
ESMFold Tool — Step04: Sequence-to-Structure Validation
========================================================
Wraps the NVIDIA NIM ESMFold endpoint for fast, MSA-free structure prediction.
Used to validate designed sequences from ProteinMPNN by predicting their
3-D structure and checking self-consistency (pLDDT, TM-score vs. backbone).

Endpoint:
    https://health.api.nvidia.com/v1/biology/nvidia/esmfold

Reference client: PRST_N_FM/bionemo/esmfold_client.py
This module defines the *agentic tool interface* on top of the same API.

Typical use in the SSTR2 pipeline:
    - Fold each ProteinMPNN sequence -> filter by pLDDT >= threshold
    - Compare predicted fold against RFdiffusion backbone (TM-score)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
_TIMEOUT_S = 120

# PDB column positions (0-indexed): B-factor occupies columns 60-65 inclusive.
_BFACTOR_COL_START = 60
_BFACTOR_COL_END = 66


class ESMFoldTool(BaseTool):
    """Tool wrapper for the ESMFold NIM API.

    Interface definition:
        predict_structure(sequence)        -> ToolResult
        batch_predict(sequences)           -> list[ToolResult]
        extract_plddt(result_data)         -> float
        extract_pdb(result_data)           -> str

    execute() dispatcher accepts action="predict_structure" | "batch_predict".
    """

    name = "esmfold"
    description = (
        "Predict protein 3-D structure from sequence alone (no MSA required) "
        "using ESMFold. Returns PDB coordinates and per-residue pLDDT confidence."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary actions
    # ------------------------------------------------------------------

    def predict_structure(self, sequence: str) -> ToolResult:
        """Predict the 3-D structure of a single amino-acid sequence.

        Args:
            sequence: Single-letter amino-acid sequence (standard 20 AA).
                      Example: "AGCKNFFWKTFTSC"

        Returns:
            ToolResult where data contains the raw API response plus:
                - "pdb":         Extracted PDB content string (str | None)
                - "mean_plddt":  Mean per-residue confidence score (float | None)
                - "input_sequence": The input sequence echoed back (str)

        Example:
            tool = ESMFoldTool()
            result = tool.predict_structure("ACKNFFWK")
            pdb_text = result.data["pdb"]
            plddt    = result.data["mean_plddt"]
        """
        if not sequence:
            return ToolResult.fail("sequence must not be empty")

        payload: dict[str, Any] = {"sequence": sequence}

        try:
            data, elapsed_ms = self._post_timed("", payload)
            pdb_text = self.extract_pdb(data)
            mean_plddt = self.extract_plddt(data, pdb_content=pdb_text)

            data["pdb"] = pdb_text
            data["mean_plddt"] = mean_plddt
            data["input_sequence"] = sequence

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                sequence_length=len(sequence),
            )
        except Exception as exc:
            logger.error("ESMFold predict_structure failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    def batch_predict(self, sequences: list[str]) -> list[ToolResult]:
        """Predict structures for a list of sequences sequentially.

        Individual failures are captured as failed ToolResults; they do not
        abort processing of the remaining sequences.

        Args:
            sequences: List of amino-acid sequences to fold.

        Returns:
            List of ToolResult objects, one per input sequence (same order).
            Each result's data["input_sequence"] echoes the source sequence.

        Example:
            tool = ESMFoldTool()
            results = tool.batch_predict(["ACKNFF", "MKTAY"])
            for r in results:
                print(r.data["input_sequence"], r.data.get("mean_plddt"))
        """
        results: list[ToolResult] = []
        for i, seq in enumerate(sequences):
            logger.info(
                "ESMFold batch %d/%d: %s...", i + 1, len(sequences), seq[:20]
            )
            result = self.predict_structure(seq)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Static extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_pdb(result: dict[str, Any]) -> str | None:
        """Extract the PDB content string from a raw API response dict.

        Tries multiple known response key patterns in order:
            pdbs[0]  ->  pdb  ->  output  (if it contains ATOM/HETATM records)

        Args:
            result: Raw dict returned by the ESMFold API.

        Returns:
            PDB content string, or None if not found.
        """
        if not isinstance(result, dict):
            return None

        # 1. pdbs list (some NIM versions)
        pdbs = result.get("pdbs")
        if isinstance(pdbs, list) and pdbs:
            return pdbs[0]

        # 2. pdb single string
        pdb = result.get("pdb")
        if isinstance(pdb, str) and pdb.strip():
            return pdb

        # 3. output fallback
        output = result.get("output", "")
        if isinstance(output, str) and ("ATOM" in output or "HETATM" in output):
            return output

        logger.warning(
            "Could not extract PDB from ESMFold response. Keys: %s",
            list(result.keys()),
        )
        return None

    @staticmethod
    def extract_plddt(
        result: dict[str, Any],
        pdb_content: str | None = None,
    ) -> float | None:
        """Extract the mean pLDDT confidence score from a raw API response.

        Tries in order:
            1. result["mean_plddt"]         (scalar float)
            2. result["plddt"]              (scalar or list -> mean)
            3. result["plddt_scores"]       (list or nested list -> mean)
            4. B-factor column in PDB text  (ESMFold stores pLDDT as B-factor)

        Args:
            result:      Raw dict returned by the ESMFold API.
            pdb_content: Optional pre-extracted PDB text. When None, it is
                         extracted from result if needed for B-factor fallback.

        Returns:
            Mean pLDDT as a float in [0, 100], or None if not recoverable.
        """
        if not isinstance(result, dict):
            return None

        # 1. Direct scalar key
        val = result.get("mean_plddt")
        if isinstance(val, (int, float)):
            return float(val)

        # 2. plddt key (scalar or list)
        val = result.get("plddt")
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, list) and val:
            return sum(val) / len(val)

        # 3. plddt_scores (possibly nested list)
        val = result.get("plddt_scores")
        if isinstance(val, list) and val:
            flat = val[0] if isinstance(val[0], list) else val
            return sum(flat) / len(flat) if flat else None

        # 4. Fallback: parse B-factor column from PDB text
        pdb_text = pdb_content or ESMFoldTool.extract_pdb(result)
        if pdb_text:
            return ESMFoldTool._plddt_from_bfactor(pdb_text)

        logger.warning(
            "Could not extract pLDDT from ESMFold response. Keys: %s",
            list(result.keys()),
        )
        return None

    @staticmethod
    def _plddt_from_bfactor(pdb_text: str) -> float | None:
        """Compute mean pLDDT by averaging B-factor values in ATOM/HETATM records.

        ESMFold (and AlphaFold) encode per-residue pLDDT in the B-factor
        column (characters 60-65, 1-indexed columns 61-66).

        Args:
            pdb_text: Raw PDB content string.

        Returns:
            Mean pLDDT float, or None if no valid records found.
        """
        bfactors: list[float] = []
        for line in pdb_text.splitlines():
            if line.startswith(("ATOM", "HETATM")) and len(line) >= _BFACTOR_COL_END:
                try:
                    bfactors.append(float(line[_BFACTOR_COL_START:_BFACTOR_COL_END]))
                except ValueError:
                    continue
        if bfactors:
            return sum(bfactors) / len(bfactors)
        return None

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "predict_structure" (default) — single sequence
            "batch_predict"              — list of sequences (returns list of ToolResult
                                           wrapped as data["results"])
        """
        action = kwargs.pop("action", "predict_structure")
        if action == "predict_structure":
            return self.predict_structure(**kwargs)
        if action == "batch_predict":
            sequences = kwargs.pop("sequences", [])
            results = self.batch_predict(sequences)
            return ToolResult.ok(
                data={"results": [r.__dict__ for r in results]},
                tool=self.name,
                action="batch_predict",
            )
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> ESMFoldTool:
    """Convenience factory used by the pipeline."""
    return ESMFoldTool(**kwargs)
