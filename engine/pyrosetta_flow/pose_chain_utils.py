"""Utilities for assigning peptide chains in PyRosetta poses."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def chain_lengths(pose) -> dict[str, int]:
    """Return PDB chain IDs and their residue counts, preserving pose order."""
    lengths: dict[str, int] = {}
    pdb_info = pose.pdb_info()
    for index in range(1, pose.total_residue() + 1):
        chain = pdb_info.chain(index)
        lengths[chain] = lengths.get(chain, 0) + 1
    return lengths


def detect_peptide_chain(pose, target_len: int) -> str:
    """Select the chain matching ``target_len``, or the closest-length chain.

    A mismatch is kept recoverable for curated structures with differing chain
    assignments, but is always logged so unexpected inputs remain visible.
    """
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")

    lengths = chain_lengths(pose)
    if not lengths:
        raise ValueError("pose contains no PDB chains")

    exact = [chain for chain, length in lengths.items() if length == target_len]
    if exact:
        if len(exact) > 1:
            logger.warning(
                "Multiple chains match peptide target_len=%d: %s; selecting chain %s",
                target_len, exact, exact[0],
            )
        return exact[0]

    selected = min(
        lengths,
        key=lambda chain: (abs(lengths[chain] - target_len), lengths[chain]),
    )
    logger.warning(
        "No chain matches peptide target_len=%d (chain lengths: %s); "
        "auto-selected chain %s (length=%d)",
        target_len, lengths, selected, lengths[selected],
    )
    return selected
