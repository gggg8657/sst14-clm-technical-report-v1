"""
ESM2 Tool — Auxiliary: Sequence Embeddings & Zero-Shot Mutation Analysis
=========================================================================
Wraps the NVIDIA NIM ESM2-650M endpoint for protein language model embeddings.

Endpoint:
    https://health.api.nvidia.com/v1/biology/meta/esm2-650m

There is no reference client in PRST_N_FM/bionemo/ for this tool.
This is a new addition for the agentic layer.

Typical use in the SSTR2 pipeline:
    - Compute sequence embeddings for clustering / similarity scoring
    - Zero-shot mutation effect analysis (compare wild-type vs. mutant embeddings)
    - Sequence similarity screening before expensive structure prediction
    - Embedding-guided filtering of ProteinMPNN output sequences
"""

from __future__ import annotations

import logging
from typing import Any

from .base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_ENDPOINT = "https://health.api.nvidia.com/v1/biology/meta/esm2-650m"
_TIMEOUT_S = 60


class ESM2Tool(BaseTool):
    """Tool wrapper for the ESM2-650M NIM API.

    ESM2 is a protein language model trained on UniRef90. Its per-residue
    and mean embeddings encode structural and functional information without
    requiring any structure prediction.

    Interface definition:
        get_embeddings(sequences) -> ToolResult

    execute() dispatcher accepts action="get_embeddings".
    """

    name = "esm2"
    description = (
        "Extract protein sequence embeddings using ESM2-650M language model. "
        "Embeddings capture structural and functional information and support "
        "zero-shot mutation scoring and sequence similarity analysis."
    )
    endpoint = _ENDPOINT
    timeout = _TIMEOUT_S

    # ------------------------------------------------------------------
    # Primary action
    # ------------------------------------------------------------------

    def get_embeddings(self, sequences: list[str]) -> ToolResult:
        """Get ESM2 embeddings for a list of protein sequences.

        Args:
            sequences: List of amino-acid sequences to embed.
                       Each sequence should use standard single-letter codes.
                       Maximum sequence length and batch size depend on the
                       API tier; split large batches if needed.

        Returns:
            ToolResult where data contains:
                - "embeddings":       list of per-sequence mean embeddings.
                                      Shape: [n_sequences, embedding_dim].
                                      embedding_dim is 1280 for ESM2-650M.
                - "per_residue":      list of per-residue embedding arrays.
                                      Shape: [n_sequences, seq_len, 1280].
                                      May be None if the API returns mean-only.
                - "n_sequences":      Number of sequences embedded (int).
                - "embedding_dim":    Embedding dimensionality (int).

        Example:
            tool = ESM2Tool()
            result = tool.get_embeddings(["ACKNFFWK", "MKTAYIA"])
            emb_0 = result.data["embeddings"][0]  # 1280-d vector for first seq
        """
        if not sequences:
            return ToolResult.fail("sequences list must not be empty")

        payload: dict[str, Any] = {"sequences": sequences}

        try:
            data, elapsed_ms = self._post_timed("", payload)

            embeddings = data.get("embeddings", [])
            per_residue = data.get("per_residue_embeddings", None)

            embedding_dim = 0
            if embeddings and isinstance(embeddings[0], list):
                embedding_dim = len(embeddings[0])

            data["embeddings"] = embeddings
            data["per_residue"] = per_residue
            data["n_sequences"] = len(embeddings)
            data["embedding_dim"] = embedding_dim

            return ToolResult.ok(
                data=data,
                elapsed_ms=elapsed_ms,
                tool=self.name,
                n_sequences=len(sequences),
                embedding_dim=embedding_dim,
            )
        except Exception as exc:
            logger.error("ESM2 get_embeddings failed: %s", exc)
            return ToolResult.fail(str(exc), tool=self.name)

    # ------------------------------------------------------------------
    # execute() dispatcher
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> ToolResult:
        """Dispatch to the appropriate action.

        Supported actions:
            "get_embeddings" (default)
        """
        action = kwargs.pop("action", "get_embeddings")
        if action == "get_embeddings":
            return self.get_embeddings(**kwargs)
        return ToolResult.fail(f"Unknown action '{action}' for {self.name}")


def get_tool(**kwargs: Any) -> ESM2Tool:
    """Convenience factory used by the pipeline."""
    return ESM2Tool(**kwargs)
