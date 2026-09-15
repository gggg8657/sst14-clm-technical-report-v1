"""
step03b_blosum_mutation.py
==========================
Step 03b: BLOSUM62 기반 텍스트 레벨 시퀀스 변이 생성 (Approach B)
         BLOSUM62 Text-Level Sequence Mutation Generator

Approach B 전략:
  1) Seed 시퀀스(DOTATATE)에서 고정 위치를 제외한 가변 위치 식별
  2) BLOSUM62 치환 행렬로 진화적으로 타당한 변이만 생성
  3) 단일 변이체 + 조합 변이체 생성
  4) Kyte-Doolittle 소수성 제약으로 극단적 변이 차단
  5) 안정성 사전 스크리닝(Step 08 predict_half_life)으로 빠른 필터링

Seed: AGCKNFFWKTFTSC (14-aa DOTATATE)
Fixed: 3,13=C (disulfide), 7-10=FWKT (hotspot)
Mutable: 1,2,4,5,6,11,12,14 (8 positions)

Public API:
    load_blosum62()                        -> Dict[Tuple[str,str], int]
    get_plausible_substitutions(aa, min_score) -> List[Tuple[str, int]]
    generate_single_mutants(seed, fixed, ...)  -> List[VariantEntry]
    generate_combinatorial_variants(...)       -> List[VariantEntry]
    validate_constraints(seq, fixed)           -> bool
    compute_blosum_distance(seq1, seq2)        -> int
    hydrophobicity_check(seq, max_delta)       -> bool
    run_approach_b(config)                     -> Step03bOutput
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.io_schemas import Step03bOutput, VariantEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard 20 amino acids
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# ---------------------------------------------------------------------------
# BLOSUM62 substitution matrix (upper triangle, symmetric)
# Source: Henikoff & Henikoff (1992) PNAS 89:10915-10919
# Score = log-odds ratio of observed substitution frequency vs expected
# ---------------------------------------------------------------------------
_BLOSUM62_RAW: Dict[str, Dict[str, int]] = {
    "A": {"A":4, "C":0, "D":-2,"E":-1,"F":-2,"G":0, "H":-2,"I":-1,"K":-1,"L":-1,"M":-1,"N":-2,"P":-1,"Q":-1,"R":-1,"S":1, "T":0, "V":0, "W":-3,"Y":-2},
    "C": {"C":9, "D":-3,"E":-4,"F":-2,"G":-3,"H":-3,"I":-1,"K":-3,"L":-1,"M":-1,"N":-3,"P":-3,"Q":-3,"R":-3,"S":-1,"T":-1,"V":-1,"W":-2,"Y":-2},
    "D": {"D":6, "E":2, "F":-3,"G":-1,"H":-1,"I":-3,"K":-1,"L":-4,"M":-3,"N":1, "P":-1,"Q":0, "R":-2,"S":0, "T":-1,"V":-3,"W":-4,"Y":-3},
    "E": {"E":5, "F":-3,"G":-2,"H":0, "I":-3,"K":1, "L":-3,"M":-2,"N":0, "P":-1,"Q":2, "R":0, "S":0, "T":-1,"V":-2,"W":-3,"Y":-2},
    "F": {"F":6, "G":-3,"H":-1,"I":0, "K":-3,"L":0, "M":0, "N":-3,"P":-4,"Q":-3,"R":-3,"S":-2,"T":-2,"V":-1,"W":1, "Y":3},
    "G": {"G":6, "H":-2,"I":-4,"K":-2,"L":-4,"M":-3,"N":0, "P":-2,"Q":-2,"R":-2,"S":0, "T":-2,"V":-3,"W":-2,"Y":-3},
    "H": {"H":8, "I":-3,"K":-1,"L":-3,"M":-2,"N":1, "P":-2,"Q":0, "R":0, "S":-1,"T":-2,"V":-3,"W":-2,"Y":2},
    "I": {"I":4, "K":-3,"L":2, "M":1, "N":-3,"P":-3,"Q":-3,"R":-3,"S":-2,"T":-1,"V":3, "W":-3,"Y":-1},
    "K": {"K":5, "L":-2,"M":-1,"N":0, "P":-1,"Q":1, "R":2, "S":0, "T":-1,"V":-2,"W":-3,"Y":-2},
    "L": {"L":4, "M":2, "N":-3,"P":-3,"Q":-2,"R":-2,"S":-2,"T":-1,"V":1, "W":-2,"Y":-1},
    "M": {"M":5, "N":-2,"P":-2,"Q":0, "R":-1,"S":-1,"T":-1,"V":1, "W":-1,"Y":-1},
    "N": {"N":6, "P":-2,"Q":0, "R":0, "S":1, "T":0, "V":-3,"W":-4,"Y":-2},
    "P": {"P":7, "Q":-1,"R":-2,"S":-1,"T":-1,"V":-2,"W":-4,"Y":-3},
    "Q": {"Q":5, "R":1, "S":0, "T":-1,"V":-2,"W":-2,"Y":-1},
    "R": {"R":5, "S":-1,"T":-1,"V":-3,"W":-3,"Y":-2},
    "S": {"S":4, "T":1, "V":-2,"W":-3,"Y":-2},
    "T": {"T":5, "V":0, "W":-2,"Y":-2},
    "V": {"V":4, "W":-3,"Y":-1},
    "W": {"W":11,"Y":2},
    "Y": {"Y":7},
}

# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale
# ---------------------------------------------------------------------------
KYTE_DOOLITTLE: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}


# ---------------------------------------------------------------------------
# BLOSUM62 accessor
# ---------------------------------------------------------------------------

def load_blosum62() -> Dict[Tuple[str, str], int]:
    """내장 BLOSUM62 행렬을 대칭 딕셔너리로 반환한다.

    Returns:
        Dict mapping (aa1, aa2) tuple to BLOSUM62 score.
    """
    matrix: Dict[Tuple[str, str], int] = {}
    for aa1, row in _BLOSUM62_RAW.items():
        for aa2, score in row.items():
            matrix[(aa1, aa2)] = score
            matrix[(aa2, aa1)] = score
    return matrix


# Module-level cached matrix
_BLOSUM62 = load_blosum62()


def get_blosum_score(aa1: str, aa2: str) -> int:
    """두 아미노산 간 BLOSUM62 점수를 반환한다."""
    return _BLOSUM62.get((aa1.upper(), aa2.upper()), -4)


# ---------------------------------------------------------------------------
# Substitution candidates
# ---------------------------------------------------------------------------

def get_plausible_substitutions(
    aa: str,
    min_score: int = 0,
) -> List[Tuple[str, int]]:
    """특정 아미노산의 BLOSUM62 점수 >= min_score인 치환 후보를 반환한다.

    자기 자신(identity)은 제외한다.

    Args:
        aa:        원본 아미노산 (1-letter code)
        min_score: 최소 BLOSUM62 점수 (기본값: 0)

    Returns:
        (치환 아미노산, BLOSUM62 점수) 튜플 리스트, 점수 내림차순.

    Example:
        get_plausible_substitutions("A", 0) → [("S", 1), ("G", 0)]
    """
    aa = aa.upper()
    subs: List[Tuple[str, int]] = []
    for other in AMINO_ACIDS:
        if other == aa:
            continue
        score = get_blosum_score(aa, other)
        if score >= min_score:
            subs.append((other, score))
    return sorted(subs, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

def validate_constraints(
    sequence: str,
    fixed_positions: Dict[int, str],
) -> bool:
    """고정 위치 제약 조건을 검증한다.

    Args:
        sequence:        아미노산 시퀀스 (1-letter code)
        fixed_positions: 고정 위치 → 아미노산 매핑 (1-indexed)

    Returns:
        모든 고정 위치가 올바르면 True, 아니면 False.
    """
    for pos, expected_aa in fixed_positions.items():
        idx = pos - 1  # 1-indexed → 0-indexed
        if idx < 0 or idx >= len(sequence):
            return False
        if sequence[idx].upper() != expected_aa.upper():
            return False
    return True


# ---------------------------------------------------------------------------
# Hydrophobicity check
# ---------------------------------------------------------------------------

def _compute_hydrophobicity(sequence: str) -> float:
    """시퀀스의 평균 Kyte-Doolittle 소수성 점수를 계산한다."""
    if not sequence:
        return 0.0
    total = sum(KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence.upper())
    return total / len(sequence)


def hydrophobicity_check(
    sequence: str,
    reference: str = "AGCKNFFWKTFTSC",
    max_hydro_delta: float = 2.0,
) -> bool:
    """변이 시퀀스가 원본 대비 소수성 변화 제한 내에 있는지 확인한다.

    Args:
        sequence:        변이 시퀀스
        reference:       원본 시퀀스 (기본값: DOTATATE)
        max_hydro_delta: 허용 최대 소수성 변화량

    Returns:
        변화량 <= max_hydro_delta이면 True.
    """
    ref_hydro = _compute_hydrophobicity(reference)
    mut_hydro = _compute_hydrophobicity(sequence)
    return abs(mut_hydro - ref_hydro) <= max_hydro_delta


# ---------------------------------------------------------------------------
# BLOSUM distance
# ---------------------------------------------------------------------------

def compute_blosum_distance(seq1: str, seq2: str) -> int:
    """두 시퀀스 간 BLOSUM62 총점을 계산한다 (유사도 지표).

    같은 길이여야 한다. 길이 다르면 짧은 쪽 기준.

    Returns:
        BLOSUM62 pair score 합계 (높을수록 유사).
    """
    min_len = min(len(seq1), len(seq2))
    return sum(
        get_blosum_score(seq1[i], seq2[i])
        for i in range(min_len)
    )


# ---------------------------------------------------------------------------
# Single mutant generation
# ---------------------------------------------------------------------------

def generate_single_mutants(
    seed: str,
    fixed_positions: Dict[int, str],
    min_blosum: int = 0,
    reference: str = "",
    max_hydro_delta: float = 2.0,
) -> List[VariantEntry]:
    """각 가변 위치에서 단일 치환 변이체를 생성한다.

    가변 위치 8개 x 평균 ~5개 치환 = ~40개 변이체.

    Args:
        seed:            원본 시퀀스
        fixed_positions: 고정 위치 매핑 (1-indexed)
        min_blosum:      최소 BLOSUM62 치환 점수
        reference:       소수성 검사 기준 시퀀스 (빈 문자열이면 seed 사용)
        max_hydro_delta: 소수성 변화 제한

    Returns:
        VariantEntry 리스트 (단일 변이체).
    """
    ref = reference or seed
    variants: List[VariantEntry] = []
    var_counter = 0

    for pos_1indexed in range(1, len(seed) + 1):
        if pos_1indexed in fixed_positions:
            continue

        idx = pos_1indexed - 1
        original_aa = seed[idx]
        subs = get_plausible_substitutions(original_aa, min_blosum)

        for sub_aa, blosum_score in subs:
            # Build mutated sequence
            mutated = list(seed)
            mutated[idx] = sub_aa
            mutated_seq = "".join(mutated)

            # Validate constraints (safety check)
            if not validate_constraints(mutated_seq, fixed_positions):
                continue

            # Hydrophobicity check
            if not hydrophobicity_check(mutated_seq, ref, max_hydro_delta):
                continue

            var_counter += 1
            mutation_label = f"{original_aa}{pos_1indexed}{sub_aa}"
            variants.append(VariantEntry(
                variant_id=f"var_{var_counter:03d}",
                sequence=mutated_seq,
                parent_sequence=seed,
                mutations=[mutation_label],
                n_mutations=1,
                blosum_total_score=blosum_score,
                source="single_mutant",
            ))

    logger.info(
        "[Step03b] Generated %d single mutants from seed %s...",
        len(variants), seed[:8],
    )
    return variants


# ---------------------------------------------------------------------------
# Combinatorial variant generation
# ---------------------------------------------------------------------------

def generate_combinatorial_variants(
    seed: str,
    fixed_positions: Dict[int, str],
    max_mutations: int = 3,
    max_variants: int = 200,
    min_blosum: int = 0,
    strategy: str = "random",
    reference: str = "",
    max_hydro_delta: float = 2.0,
    rng_seed: int = 42,
) -> List[VariantEntry]:
    """다중 위치 조합 변이체를 생성한다.

    Args:
        seed:            원본 시퀀스
        fixed_positions: 고정 위치 매핑 (1-indexed)
        max_mutations:   동시 변이 최대 수 (2~max_mutations)
        max_variants:    최대 생성 변이체 수
        min_blosum:      최소 BLOSUM62 치환 점수
        strategy:        "random" (무작위 샘플링) | "greedy" (상위 조합)
        reference:       소수성 검사 기준 시퀀스
        max_hydro_delta: 소수성 변화 제한
        rng_seed:        랜덤 시드

    Returns:
        VariantEntry 리스트 (조합 변이체).
    """
    ref = reference or seed
    rng = random.Random(rng_seed)

    # Build per-position substitution map
    mutable_positions: List[int] = [
        pos for pos in range(1, len(seed) + 1)
        if pos not in fixed_positions
    ]
    pos_subs: Dict[int, List[Tuple[str, int]]] = {}
    for pos in mutable_positions:
        subs = get_plausible_substitutions(seed[pos - 1], min_blosum)
        if subs:
            pos_subs[pos] = subs

    variants: List[VariantEntry] = []
    seen_sequences: set = set()
    var_counter = 0

    for n_mut in range(2, max_mutations + 1):
        if len(variants) >= max_variants:
            break

        # Get all position combinations of size n_mut
        available_positions = [p for p in mutable_positions if p in pos_subs]
        if len(available_positions) < n_mut:
            continue

        pos_combos = list(itertools.combinations(available_positions, n_mut))

        if strategy == "random":
            rng.shuffle(pos_combos)
            # Limit iterations to avoid excessive computation
            pos_combos = pos_combos[:max_variants * 2]

        for combo in pos_combos:
            if len(variants) >= max_variants:
                break

            # For each position combo, pick substitutions
            if strategy == "greedy":
                # Use top-scoring substitution at each position
                sub_choices = [
                    (pos, pos_subs[pos][0][0], pos_subs[pos][0][1])
                    for pos in combo
                ]
            else:
                # Random substitution at each position
                sub_choices = [
                    (pos, *rng.choice(pos_subs[pos]))
                    for pos in combo
                ]

            # Build mutated sequence
            mutated = list(seed)
            mutations: List[str] = []
            total_blosum = 0
            for pos, sub_aa, score in sub_choices:
                original_aa = seed[pos - 1]
                mutated[pos - 1] = sub_aa
                mutations.append(f"{original_aa}{pos}{sub_aa}")
                total_blosum += score

            mutated_seq = "".join(mutated)

            # Dedup
            if mutated_seq in seen_sequences:
                continue
            seen_sequences.add(mutated_seq)

            # Validate
            if not validate_constraints(mutated_seq, fixed_positions):
                continue
            if not hydrophobicity_check(mutated_seq, ref, max_hydro_delta):
                continue

            var_counter += 1
            variants.append(VariantEntry(
                variant_id=f"var_c{var_counter:03d}",
                sequence=mutated_seq,
                parent_sequence=seed,
                mutations=mutations,
                n_mutations=n_mut,
                blosum_total_score=total_blosum,
                source="combinatorial",
            ))

    logger.info(
        "[Step03b] Generated %d combinatorial variants (max_mut=%d, strategy=%s)",
        len(variants), max_mutations, strategy,
    )
    return variants


# ---------------------------------------------------------------------------
# Pipeline integration entry point
# ---------------------------------------------------------------------------

def run_approach_b(
    config: Dict[str, Any],
) -> Step03bOutput:
    """Approach B 파이프라인 진입점: 설정에서 변이체를 생성한다.

    Args:
        config: pipeline_config.yaml의 approach_b 섹션 또는 전체 config.

    Returns:
        Step03bOutput with all generated variants.
    """
    ab_cfg = config.get("approach_b", config)
    seed = ab_cfg.get("seed_sequence", "AGCKNFFWKTFTSC")  # 원본 DOTATATE: Cys3-Cys14 이황화결합
    fixed_raw = ab_cfg.get("fixed_positions", {3: "C", 7: "F", 8: "W", 9: "K", 10: "T", 14: "C"})
    fixed_positions = {int(k): v for k, v in fixed_raw.items()}
    min_blosum = int(ab_cfg.get("min_blosum_score", 0))
    max_mutations = int(ab_cfg.get("max_mutations_per_variant", 3))
    max_variants = int(ab_cfg.get("max_variants", 200))
    strategy = str(ab_cfg.get("strategy", "random"))
    max_hydro_delta = float(ab_cfg.get("hydrophobicity_max_delta", 2.0))

    # Generate single mutants
    singles = generate_single_mutants(
        seed=seed,
        fixed_positions=fixed_positions,
        min_blosum=min_blosum,
        max_hydro_delta=max_hydro_delta,
    )

    # Generate combinatorial variants
    remaining = max(0, max_variants - len(singles))
    combos = generate_combinatorial_variants(
        seed=seed,
        fixed_positions=fixed_positions,
        max_mutations=max_mutations,
        max_variants=remaining,
        min_blosum=min_blosum,
        strategy=strategy,
        max_hydro_delta=max_hydro_delta,
    )

    all_variants = singles + combos

    # Re-number variant IDs sequentially
    for i, v in enumerate(all_variants):
        v.variant_id = f"var_{i + 1:03d}"

    logger.info(
        "[Step03b] Approach B total: %d variants (%d single + %d combo) from seed %s",
        len(all_variants), len(singles), len(combos), seed,
    )

    return Step03bOutput(
        variants=all_variants,
        seed_sequence=seed,
        fixed_positions=fixed_positions,
        total_generated=len(all_variants),
        strategy="approach_b",
    )
