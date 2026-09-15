"""
step03_sequence.py
==================
Step 03: 백본당 서열 K개 생성 (Sequence Design via ProteinMPNN)

ProteinMPNN NIM API를 사용하여 각 RFdiffusion 백본 구조에 대해
K개의 아미노산 서열을 설계(역폴딩)한다.

Uses the ProteinMPNN NIM API (inverse folding) to design K amino acid
sequences for each backbone PDB produced by Step02.

Input:
    - backbone PDB paths  (Step02Output.backbone_pdbs)
    - k_seq_per_backbone (K), sampling_temp from pipeline_config.yaml

Output:
    - 03_sequence/bb{i:02d}_sequences.fasta  (one file per backbone)

Public API:
    design_sequences(backbones, config)             -> Step03Output
    design_for_backbone(backbone_pdb, num_seq,
                        sampling_temp)              -> List[str]
    save_sequences(sequences, output_dir)           -> List[Path]
    compute_sequence_diversity(sequences)           -> Dict[str, float]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import Step03Output, SequenceEntry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_NUM_SEQ: int = 8
_DEFAULT_SAMPLING_TEMP: float = 0.1
_DEFAULT_CHAIN_DESIGN: str = "A"   # ProteinMPNN designs the binder chain


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def design_sequences(
    backbones: List[str],
    config: Dict[str, Any],
) -> Step03Output:
    """모든 백본에 대해 K개의 서열을 설계하고 FASTA 파일로 저장한다.

    Orchestration entry for Step 03.  For each backbone PDB in *backbones*,
    calls ProteinMPNN to produce ``k_seq_per_backbone`` amino acid sequences,
    then writes one FASTA file per backbone.

    Args:
        backbones: List of file paths to backbone PDB files (Step02 output).
        config:    Full pipeline configuration dict.

    Returns:
        Step03Output containing all SequenceEntry records.
    """
    run_id: str = config.get("run_id", "default_run")
    output_base: Path = Path(config.get("output_base_dir", "runs")) / run_id
    out_dir: Path = output_base / "03_sequence"
    out_dir.mkdir(parents=True, exist_ok=True)

    iteration_cfg: Dict[str, Any] = config.get("iteration", {})
    num_seq: int = int(iteration_cfg.get("k_seq_per_backbone", _DEFAULT_NUM_SEQ))
    sampling_temp: float = float(
        iteration_cfg.get("sampling_temp", _DEFAULT_SAMPLING_TEMP)
    )

    # Sequence constraints (fixed positions)
    seq_constraints: Dict[str, Any] = config.get("sequence_constraints", {})
    constraints_enabled: bool = seq_constraints.get("enabled", False)
    fixed_positions: Dict[int, str] = {}
    if constraints_enabled:
        raw_fixed = seq_constraints.get("fixed_positions", {})
        fixed_positions = {int(k): str(v) for k, v in raw_fixed.items()}
        logger.info(
            "[Step03] Sequence constraints enabled: %s",
            {k: v for k, v in sorted(fixed_positions.items())},
        )

    all_entries: List[SequenceEntry] = []

    for bb_idx, backbone_path in enumerate(backbones):
        logger.info(
            "[Step03] Designing %d sequences for backbone %02d: %s",
            num_seq,
            bb_idx,
            backbone_path,
        )
        try:
            sequences = design_for_backbone(backbone_path, num_seq, sampling_temp)
        except Exception as exc:
            logger.error("[Step03] design_for_backbone failed for bb%02d: %s", bb_idx, exc)
            sequences = []

        # Save to FASTA and build SequenceEntry records
        fasta_path = out_dir / f"bb{bb_idx:02d}_sequences.fasta"
        fasta_lines: List[str] = []
        for seq_idx, seq in enumerate(sequences):
            # Enforce sequence constraints if enabled
            if constraints_enabled and fixed_positions:
                enforced = enforce_sequence_constraints(seq, fixed_positions)
                if enforced is None:
                    logger.warning(
                        "[Step03] bb%02d seq%02d: too short for constraints, skipped.",
                        bb_idx, seq_idx,
                    )
                    continue
                if enforced != seq.upper():
                    logger.info(
                        "[Step03] bb%02d seq%02d: constraints enforced.",
                        bb_idx, seq_idx,
                    )
                seq = enforced

            seq_id = f"bb{bb_idx:02d}_seq{seq_idx:02d}"
            fasta_lines.append(f">{seq_id}")
            fasta_lines.append(seq)
            all_entries.append(
                SequenceEntry(
                    backbone_idx=bb_idx,
                    seq_idx=seq_idx,
                    sequence=seq,
                    fasta_path=str(fasta_path),
                    seq_id=seq_id,
                )
            )
        fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
        logger.info(
            "[Step03] Backbone %02d: %d sequences written -> %s",
            bb_idx,
            len(sequences),
            fasta_path,
        )

    logger.info(
        "[Step03] Total sequences designed: %d across %d backbones.",
        len(all_entries),
        len(backbones),
    )
    return Step03Output(sequences=all_entries)


def design_for_backbone(
    backbone_pdb: str,
    num_seq: int = _DEFAULT_NUM_SEQ,
    sampling_temp: float = _DEFAULT_SAMPLING_TEMP,
) -> List[str]:
    """ProteinMPNN API를 호출하여 단일 백본에 대한 서열 목록을 반환한다.

    Args:
        backbone_pdb:  Path to the backbone PDB file (binder + receptor complex).
        num_seq:       Number of sequences to design.
        sampling_temp: Sampling temperature (lower = more conservative).
                       Typical range: 0.05 – 0.3.

    Returns:
        List of amino acid sequences (one-letter code), length == num_seq.
        May be shorter on partial API failure.

    Raises:
        RuntimeError: On total API failure.
    """
    import requests

    pdb_content = Path(backbone_pdb).read_text(encoding="utf-8")
    api_key = _resolve_api_key()
    endpoint = "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
    payload: Dict[str, Any] = {
        "pdb": pdb_content,
        "chains_to_design": [_DEFAULT_CHAIN_DESIGN],
        "num_seq_per_target": num_seq,
        "sampling_temp": sampling_temp,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Step03] ProteinMPNN API HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    # API may return a list of dicts with 'sequence' key, or a 'sequences' list
    raw_sequences = (
        data.get("sequences")
        or data.get("result", {}).get("sequences")
        or []
    )
    parsed: List[str] = []
    for item in raw_sequences:
        if isinstance(item, str):
            parsed.append(item.strip().upper())
        elif isinstance(item, dict):
            seq = item.get("sequence") or item.get("seq") or ""
            if seq:
                parsed.append(seq.strip().upper())

    if not parsed:
        raise RuntimeError(
            f"[Step03] ProteinMPNN returned no sequences. Response keys: {list(data.keys())}"
        )
    return parsed[:num_seq]


def save_sequences(
    sequences: List[SequenceEntry],
    output_dir: Path,
) -> List[Path]:
    """SequenceEntry 목록을 FASTA 파일로 저장하고 경로 목록을 반환한다.

    Groups entries by backbone index and writes one FASTA file per backbone.

    Args:
        sequences:  List of SequenceEntry records.
        output_dir: Directory to write FASTA files into.

    Returns:
        List of Path objects for written FASTA files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_backbone: Dict[int, List[SequenceEntry]] = {}
    for entry in sequences:
        by_backbone.setdefault(entry.backbone_idx, []).append(entry)

    written: List[Path] = []
    for bb_idx, entries in sorted(by_backbone.items()):
        fasta_path = output_dir / f"bb{bb_idx:02d}_sequences.fasta"
        lines: List[str] = []
        for entry in sorted(entries, key=lambda e: e.seq_idx):
            lines.append(f">{entry.seq_id}")
            lines.append(entry.sequence)
        fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(fasta_path)
    return written


def compute_sequence_diversity(sequences: List[str]) -> Dict[str, float]:
    """서열 목록의 다양성 지표(평균 해밍 거리, 고유 서열 비율)를 계산한다.

    Computes simple diversity metrics for a list of amino acid sequences:
    * ``mean_hamming``: Average pairwise Hamming distance (normalised 0-1).
    * ``unique_ratio``: Fraction of sequences that are unique.
    * ``mean_length``: Mean sequence length.

    Args:
        sequences: List of amino acid sequences (one-letter code).

    Returns:
        Dict with keys ``mean_hamming``, ``unique_ratio``, ``mean_length``.
    """
    if not sequences:
        return {"mean_hamming": 0.0, "unique_ratio": 0.0, "mean_length": 0.0}

    unique_ratio = len(set(sequences)) / len(sequences)
    mean_length = sum(len(s) for s in sequences) / len(sequences)

    # Pairwise hamming over same-length pairs only
    hamming_scores: List[float] = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            s1, s2 = sequences[i], sequences[j]
            min_len = min(len(s1), len(s2))
            if min_len == 0:
                continue
            mismatches = sum(a != b for a, b in zip(s1[:min_len], s2[:min_len]))
            # Penalise length difference
            length_penalty = abs(len(s1) - len(s2)) / max(len(s1), len(s2))
            hamming_scores.append(mismatches / min_len + length_penalty)

    mean_hamming = sum(hamming_scores) / len(hamming_scores) if hamming_scores else 0.0

    return {
        "mean_hamming": round(mean_hamming, 4),
        "unique_ratio": round(unique_ratio, 4),
        "mean_length": round(mean_length, 2),
    }


def validate_sequence_constraints(
    sequence: str,
    fixed_positions: Dict[int, str],
) -> bool:
    """서열이 고정 잔기 제약조건을 만족하는지 검증한다.

    Args:
        sequence:        Amino acid sequence (one-letter code).
        fixed_positions: Dict mapping 1-indexed position -> required amino acid.
                         e.g. {3: "C", 7: "F", 8: "W", 9: "K", 10: "T", 13: "C"}

    Returns:
        True if all constraints are satisfied.
    """
    for pos, expected_aa in fixed_positions.items():
        idx = pos - 1  # Convert to 0-indexed
        if idx < 0 or idx >= len(sequence):
            return False
        if sequence[idx].upper() != expected_aa.upper():
            return False
    return True


def enforce_sequence_constraints(
    sequence: str,
    fixed_positions: Dict[int, str],
) -> Optional[str]:
    """서열의 고정 위치를 강제 적용하여 제약조건을 충족시킨다.

    Replaces amino acids at fixed positions with the required residues.
    Returns None if the sequence is too short for any constraint position.

    Args:
        sequence:        Amino acid sequence (one-letter code).
        fixed_positions: Dict mapping 1-indexed position -> required amino acid.

    Returns:
        Corrected sequence string, or None if sequence is too short.
    """
    max_pos = max(fixed_positions.keys()) if fixed_positions else 0
    if len(sequence) < max_pos:
        logger.warning(
            "[Step03] Sequence too short (%d) for constraint at position %d. Skipping.",
            len(sequence), max_pos,
        )
        return None

    seq_list = list(sequence.upper())
    changed = []
    for pos, expected_aa in fixed_positions.items():
        idx = pos - 1
        if seq_list[idx] != expected_aa.upper():
            changed.append(f"pos{pos}: {seq_list[idx]}->{expected_aa}")
            seq_list[idx] = expected_aa.upper()

    if changed:
        logger.info("[Step03] Enforced constraints: %s", ", ".join(changed))

    return "".join(seq_list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_api_key() -> str:
    """환경변수 / 키 파일에서 NVIDIA API 키를 탐색한다."""
    for var in ("NGC_CLI_API_KEY", "NVIDIA_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    for name in ("ngc.key", "molmim.key"):
        for directory in (Path(__file__).parent.parent.parent, Path.cwd()):
            key_file = directory / name
            if key_file.exists():
                val = key_file.read_text(encoding="utf-8").strip()
                if val:
                    return val
    raise ValueError("NVIDIA API key not found. Set NGC_CLI_API_KEY or create ngc.key.")


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step03: Sequence design standalone test")
    parser.add_argument("--backbone-pdbs", nargs="+", required=True,
                        help="Paths to backbone PDB files")
    parser.add_argument("--num-seq", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/test_run")
    args = parser.parse_args()

    cfg: Dict[str, Any] = {
        "run_id": "test_run",
        "output_base_dir": args.output_dir,
        "iteration": {"k_seq_per_backbone": args.num_seq},
    }
    result = design_sequences(args.backbone_pdbs, cfg)
    print(f"Step03: {len(result.sequences)} sequences designed.")
    seqs = [e.sequence for e in result.sequences]
    diversity = compute_sequence_diversity(seqs)
    print(f"Diversity metrics: {diversity}")
