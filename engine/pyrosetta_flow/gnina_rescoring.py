"""GNINA CNN Rescoring wrapper for FlexPepDock output PDBs.

Calls GNINA in --score_only mode via subprocess to obtain CNN-based
binding scores, then integrates multiple score terms using Exponential
Rank Consensus (ECR).

When the ``gnina`` binary is not found on PATH the module falls back to
*dry-run* mode: a warning is logged and deterministic mock scores are
returned so that downstream pipeline stages can still execute.

Python 3.9 compatible (no ``X | Y`` union syntax).
"""
from __future__ import annotations

import logging
import math
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GNINA_BIN: Optional[str] = shutil.which("gnina")

_DRY_RUN_SCORES: Dict[str, float] = {
    "gnina_cnn_score": 0.0,
    "gnina_cnn_affinity": 0.0,
    "gnina_vina_score": 0.0,
    "gnina_dry_run": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_gnina_available() -> bool:
    """Return True if the gnina binary is on PATH."""
    return shutil.which("gnina") is not None


def _parse_gnina_output(stdout: str) -> Dict[str, float]:
    """Parse GNINA --score_only stdout into a score dict.

    GNINA score-only mode emits a table like::

        CNNscore  CNNaffinity  Vina
        0.85      6.12         -7.3

    We look for the header line containing ``CNNscore`` and parse the
    subsequent data line.
    """
    lines = stdout.strip().splitlines()
    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if "CNNscore" in line:
            header_idx = i
            break

    if header_idx is None or header_idx + 1 >= len(lines):
        logger.warning("Could not parse GNINA output:\n%s", stdout)
        return {
            "gnina_cnn_score": float("nan"),
            "gnina_cnn_affinity": float("nan"),
            "gnina_vina_score": float("nan"),
        }

    tokens = lines[header_idx + 1].split()
    try:
        return {
            "gnina_cnn_score": float(tokens[0]),
            "gnina_cnn_affinity": float(tokens[1]),
            "gnina_vina_score": float(tokens[2]),
        }
    except (IndexError, ValueError) as exc:
        logger.warning("GNINA output parse error (%s): %s", exc, stdout)
        return {
            "gnina_cnn_score": float("nan"),
            "gnina_cnn_affinity": float("nan"),
            "gnina_vina_score": float("nan"),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def split_receptor_peptide(
    complex_pdb: str,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
) -> Tuple[str, str]:
    """Split a complex PDB file into receptor and peptide temporary files.

    Parameters
    ----------
    complex_pdb:
        Path to the input complex PDB.
    receptor_chain:
        Chain identifier for the receptor (default ``"A"``).
    peptide_chain:
        Chain identifier for the peptide (default ``"B"``).

    Returns
    -------
    tuple[str, str]
        ``(receptor_path, peptide_path)`` — paths to temporary PDB files.
        The caller is responsible for cleanup (or use with a
        ``tempfile.TemporaryDirectory``).

    Raises
    ------
    FileNotFoundError
        If *complex_pdb* does not exist.
    ValueError
        If no ATOM/HETATM lines match the requested chain IDs.
    """
    pdb_path = Path(complex_pdb)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {complex_pdb}")

    receptor_lines: List[str] = []
    peptide_lines: List[str] = []

    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            # PDB format: chain ID is column 21 (0-indexed)
            if len(line) < 22:
                continue
            chain = line[21]
            if chain == receptor_chain:
                receptor_lines.append(line)
            elif chain == peptide_chain:
                peptide_lines.append(line)

    if not receptor_lines:
        raise ValueError(
            f"No ATOM/HETATM records for receptor chain '{receptor_chain}' "
            f"in {complex_pdb}"
        )
    if not peptide_lines:
        raise ValueError(
            f"No ATOM/HETATM records for peptide chain '{peptide_chain}' "
            f"in {complex_pdb}"
        )

    # Write temporary files (caller must clean up)
    rec_fd, rec_path = tempfile.mkstemp(suffix="_receptor.pdb")
    pep_fd, pep_path = tempfile.mkstemp(suffix="_peptide.pdb")

    try:
        with open(rec_fd, "w", encoding="utf-8") as f:
            f.writelines(receptor_lines)
            f.write("END\n")
        with open(pep_fd, "w", encoding="utf-8") as f:
            f.writelines(peptide_lines)
            f.write("END\n")
    except Exception:
        # Clean up on failure
        Path(rec_path).unlink(missing_ok=True)
        Path(pep_path).unlink(missing_ok=True)
        raise

    return rec_path, pep_path


def gnina_rescore(
    complex_pdb: str,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
    timeout: int = 120,
) -> Dict[str, float]:
    """Rescore a single complex PDB with GNINA in score-only mode.

    Parameters
    ----------
    complex_pdb:
        Path to the FlexPepDock output PDB.
    receptor_chain:
        Chain ID for the receptor.
    peptide_chain:
        Chain ID for the peptide/ligand.
    timeout:
        Subprocess timeout in seconds.

    Returns
    -------
    dict[str, float]
        Keys: ``gnina_cnn_score``, ``gnina_cnn_affinity``,
        ``gnina_vina_score``.  In dry-run mode an extra
        ``gnina_dry_run: 1.0`` flag is included.
    """
    if not _is_gnina_available():
        logger.warning(
            "gnina binary not found — returning dry-run mock scores for %s",
            complex_pdb,
        )
        return dict(_DRY_RUN_SCORES)

    rec_path: Optional[str] = None
    pep_path: Optional[str] = None
    try:
        rec_path, pep_path = split_receptor_peptide(
            complex_pdb,
            receptor_chain=receptor_chain,
            peptide_chain=peptide_chain,
        )

        cmd = [
            "gnina",
            "--receptor", rec_path,
            "--ligand", pep_path,
            "--score_only",
        ]
        logger.info("Running GNINA: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            logger.error(
                "GNINA failed (rc=%d) for %s:\n%s",
                result.returncode,
                complex_pdb,
                result.stderr,
            )
            return {
                "gnina_cnn_score": float("nan"),
                "gnina_cnn_affinity": float("nan"),
                "gnina_vina_score": float("nan"),
                "gnina_error": 1.0,
            }

        return _parse_gnina_output(result.stdout)

    except subprocess.TimeoutExpired:
        logger.error("GNINA timed out (%ds) for %s", timeout, complex_pdb)
        return {
            "gnina_cnn_score": float("nan"),
            "gnina_cnn_affinity": float("nan"),
            "gnina_vina_score": float("nan"),
            "gnina_timeout": 1.0,
        }
    finally:
        if rec_path is not None:
            Path(rec_path).unlink(missing_ok=True)
        if pep_path is not None:
            Path(pep_path).unlink(missing_ok=True)


def batch_gnina_rescore(
    pdb_paths: List[str],
    max_workers: int = 4,
    receptor_chain: str = "A",
    peptide_chain: str = "B",
) -> List[Dict[str, float]]:
    """Rescore a batch of PDB files with GNINA in parallel.

    Parameters
    ----------
    pdb_paths:
        List of complex PDB file paths.
    max_workers:
        Maximum concurrent GNINA processes.
    receptor_chain:
        Chain ID for the receptor.
    peptide_chain:
        Chain ID for the peptide.

    Returns
    -------
    list[dict[str, float]]
        Score dicts in the same order as *pdb_paths*.
    """
    if not pdb_paths:
        return []

    results: List[Optional[Dict[str, float]]] = [None] * len(pdb_paths)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                gnina_rescore,
                pdb,
                receptor_chain=receptor_chain,
                peptide_chain=peptide_chain,
            ): idx
            for idx, pdb in enumerate(pdb_paths)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error(
                    "GNINA rescoring failed for %s: %s",
                    pdb_paths[idx],
                    exc,
                )
                results[idx] = {
                    "gnina_cnn_score": float("nan"),
                    "gnina_cnn_affinity": float("nan"),
                    "gnina_vina_score": float("nan"),
                    "gnina_error": 1.0,
                }

    return [r for r in results if r is not None]


def exponential_rank_consensus(
    candidates: List[Dict],
    score_keys: Optional[List[str]] = None,
) -> List[Dict]:
    """Compute Exponential Rank Consensus (ECR) across multiple score terms.

    ECR assigns each candidate a rank per score key, then computes::

        ECR_i = sum_k  exp(-rank_{i,k} / N)

    where *N* is the number of candidates and the sum is over score keys.
    Higher ECR values indicate better consensus ranking.

    For score keys where *lower* is better (all GNINA/Vina scores), ranks
    are assigned in ascending order (lowest value = rank 1).

    Parameters
    ----------
    candidates:
        List of candidate dicts.  Each must contain the fields listed in
        *score_keys*.  The dicts are **not** modified in-place; new dicts
        with an ``ecr_score`` field are returned.
    score_keys:
        Score fields to include.  Defaults to
        ``["gnina_cnn_score", "gnina_cnn_affinity", "gnina_vina_score"]``.

    Returns
    -------
    list[dict]
        Copies of input dicts augmented with ``ecr_score`` and
        ``ecr_ranks`` (per-key ranks), sorted descending by ``ecr_score``.
    """
    if score_keys is None:
        score_keys = ["gnina_cnn_score", "gnina_cnn_affinity", "gnina_vina_score"]

    if not candidates:
        return []

    n = len(candidates)

    # Build rank tables.  For all supported GNINA scores, lower is better.
    # NaN values are pushed to the worst rank.
    rank_table: Dict[str, List[int]] = {}
    for key in score_keys:
        values = []
        for i, cand in enumerate(candidates):
            val = cand.get(key, float("nan"))
            if not isinstance(val, (int, float)) or math.isnan(val):
                val = float("inf")
            values.append((val, i))
        # Sort ascending (lower is better → rank 1)
        values.sort(key=lambda t: t[0])
        ranks = [0] * n
        for rank_pos, (_, orig_idx) in enumerate(values):
            ranks[orig_idx] = rank_pos + 1  # 1-based rank
        rank_table[key] = ranks

    # Compute ECR scores
    out: List[Dict] = []
    for i, cand in enumerate(candidates):
        per_key_ranks = {key: rank_table[key][i] for key in score_keys}
        ecr = sum(math.exp(-per_key_ranks[key] / n) for key in score_keys)
        new_cand = dict(cand)
        new_cand["ecr_score"] = round(ecr, 6)
        new_cand["ecr_ranks"] = per_key_ranks
        out.append(new_cand)

    out.sort(key=lambda c: c["ecr_score"], reverse=True)
    return out
