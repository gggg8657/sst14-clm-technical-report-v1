from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

from .schema import CandidateResult, FlowConfig

AA_NO_CYS = list("ADEFGHIKLMNPQRSTVWY")


def notebook_mapping() -> List[Dict[str, str]]:
    """Notebook 단계와 파이프라인 모듈 단계의 대응표."""
    return [
        {"notebook": "SSTR2-SST14 구조 입력 준비", "pipeline": "validate_template_pose"},
        {"notebook": "SST14 변이 생성", "pipeline": "generate_random_mutant"},
        {"notebook": "mutate -> dock -> ddG 평가", "pipeline": "run_mutate_then_dock_iteration"},
        {"notebook": "후보 선별 + 다음 실험 가설", "pipeline": "run_planner_critic_loop"},
        {"notebook": "실험 요약 기록", "pipeline": "run_reporter_and_emit_artifacts"},
    ]


def validate_config(config: FlowConfig) -> None:
    if not Path(config.template_pdb).exists():
        raise FileNotFoundError(f"Template PDB not found: {config.template_pdb}")
    if config.n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")
    if not config.design_positions:
        raise ValueError("design_positions must not be empty")


MAX_RANDOM_MUTATIONS = 3


def generate_random_mutant(
    original_seq: str,
    design_positions: List[int],
    rng: random.Random,
    n_mutations: int | None = None,
    max_random_mutations: int = MAX_RANDOM_MUTATIONS,
) -> str:
    seq = list(original_seq)
    valid_positions = [p for p in design_positions if 1 <= p <= len(seq)]
    if not valid_positions:
        return original_seq
    if n_mutations is None:
        n_mutations = rng.randint(1, max(1, min(max_random_mutations, len(valid_positions))))
    mutate_positions = rng.sample(valid_positions, k=min(n_mutations, len(valid_positions)))
    for pos in mutate_positions:
        idx = pos - 1
        current = seq[idx]
        candidates = [aa for aa in AA_NO_CYS if aa != current]
        seq[idx] = rng.choice(candidates)
    return "".join(seq)


def generate_guided_mutant(
    original_seq: str,
    design_positions: List[int],
    guidance: Dict[str, Any],
    rng: random.Random,
) -> str:
    """Planner guidance를 반영한 mutation 생성.

    guidance = {
        "focus_positions": [5, 6],
        "suggested_mutations": {"5": ["W", "F"], "6": ["E", "D"]},
    }
    """
    seq = list(original_seq)
    focus = guidance.get("focus_positions", [])
    suggestions = guidance.get("suggested_mutations", {})

    # focus_positions 중 design_positions에 있는 것만 사용
    valid_focus = [p for p in focus if p in design_positions and 1 <= p <= len(seq)]

    if not valid_focus:
        # guidance가 비어있으면 random fallback
        return generate_random_mutant(original_seq, design_positions, rng)

    # 2~MAX_RANDOM_MUTATIONS개 focus position 선택, 단 focus가 1개면 1개 허용
    n_mut = max(1, rng.randint(2, max(2, min(MAX_RANDOM_MUTATIONS, len(valid_focus)))))
    chosen = rng.sample(valid_focus, k=min(n_mut, len(valid_focus)))

    for pos in chosen:
        idx = pos - 1
        pos_key = str(pos)
        if pos_key in suggestions and suggestions[pos_key]:
            # Planner가 추천한 아미노산 중 현재와 다른 것 선택
            candidates = [aa for aa in suggestions[pos_key] if aa != seq[idx] and aa != "C"]
            if candidates:
                seq[idx] = rng.choice(candidates)
                continue
        # 추천 없으면 랜덤
        candidates = [aa for aa in AA_NO_CYS if aa != seq[idx]]
        seq[idx] = rng.choice(candidates)

    return "".join(seq)


def generate_combo_mutant(
    original_seq: str,
    focus_positions: List[int],
    suggested_residues: Dict[str, List[str]],
    design_positions: List[int],
    rng: random.Random,
    combo_size: int = 2,
) -> str:
    """focus 위치 쌍에 bandit/BO 추천 잔기를 할당하는 조합 변이 생성.

    generate_guided_mutant의 단일치환 편향을 보완하기 위해 combo_size개 위치에
    동시 변이를 강제한다. runner.py에서 다음과 같이 호출한다:

        seq = generate_combo_mutant(
            seq, focus_positions, suggested_residues, design_positions, rng, combo_size=2
        )

    Args:
        original_seq: 변이 전 서열
        focus_positions: bandit/BO가 추천한 위치 목록 (1-indexed)
        suggested_residues: 위치별 추천 아미노산 dict {5: [W,F], 6: [E,D]}
        design_positions: MUTABLE_POSITIONS_1IDX와 동일한 변이 허용 위치 목록
        rng: 재현성용 Random 인스턴스
        combo_size: 동시 변이 위치 수 (기본 2)

    Returns:
        변이된 서열 문자열
    """
    seq = list(original_seq)
    valid_focus = [
        p
        for p in focus_positions
        if p in design_positions and 1 <= p <= len(seq) and seq[p - 1] != "C"
    ]

    chosen = rng.sample(valid_focus, k=min(combo_size, len(valid_focus)))
    for pos in chosen:
        idx = pos - 1
        pos_key = str(pos)
        candidates = [
            aa for aa in suggested_residues.get(pos_key, []) if aa != seq[idx] and aa != "C"
        ]
        if not candidates:
            candidates = [aa for aa in AA_NO_CYS if aa != seq[idx]]
        seq[idx] = rng.choice(candidates)

    return "".join(seq)


def candidate_to_dict(candidate: CandidateResult) -> Dict[str, object]:
    return asdict(candidate)


def choose_objective_mode(requested: str, iteration: int) -> str:
    if requested in {"ddg_only", "ddg_plus_constraints"}:
        return requested
    # auto mode: 초반 탐색은 ddg_only, 이후는 제약 포함
    return "ddg_only" if iteration == 1 else "ddg_plus_constraints"


def get_bandit_guidance(records: List[Dict[str, Any]], n_focus: int = 3) -> Dict[str, Any]:
    """Create a PositionBandit, initialize from history, and return guidance dict.

    The returned dict is compatible with Planner's mutation_guidance format:
        {"focus_positions": [5, 6, 11], "source": "bandit_thompson"}
    """
    from .bandit import PositionBandit

    bandit = PositionBandit()
    bandit.initialize_from_history(records)
    focus = bandit.sample_focus_positions(n=n_focus)
    return {
        "focus_positions": focus,
        "source": "bandit_thompson",
        "arm_stats": {str(k): v for k, v in bandit.get_arm_stats().items()},
    }


def generate_intensification_mutant(
    original_seq: str,
    intensify_guidance: Dict[str, Any],
    design_positions: List[int],
    rng: random.Random,
    partial_combination: bool = True,
) -> str:
    """high_confidence seed guidance를 반영한 local intensification 변이 생성.

    intensify_guidance는 bandit.get_intensification_guidance() 반환값.
    partial_combination=True이면 focus_positions 중 일부(2~3개)만 선택하여
    부분조합 탐색을 수행한다 (전체를 한꺼번에 변이하지 않음).

    Args:
        original_seq: 변이 시작 서열 (native reference)
        intensify_guidance: get_intensification_guidance() 반환 dict
        design_positions: 변이 허용 위치 목록 (scaffold 제약 적용 후)
        rng: 재현성용 Random 인스턴스
        partial_combination: True이면 focus_positions 중 2~3개 부분조합 선택

    Returns:
        변이된 서열 문자열 (실패 시 original_seq 반환)
    """
    focus = intensify_guidance.get("focus_positions", [])
    suggestions = intensify_guidance.get("suggested_mutations", {})

    if not focus:
        return generate_random_mutant(original_seq, design_positions, rng)

    # partial_combination: focus 중 2~3개 부분조합 선택
    valid_focus = [p for p in focus if p in design_positions and 1 <= p <= len(original_seq)]
    if not valid_focus:
        return generate_random_mutant(original_seq, design_positions, rng)

    if partial_combination and len(valid_focus) > 2:
        # 2~min(3, len) 개 선택
        n_pick = rng.randint(2, min(3, len(valid_focus)))
        chosen = rng.sample(valid_focus, k=n_pick)
    else:
        chosen = valid_focus[:min(3, len(valid_focus))]

    seq = list(original_seq)
    for pos in chosen:
        idx = pos - 1
        pos_key = str(pos)
        pos_suggestions = [
            aa for aa in suggestions.get(pos_key, [])
            if aa != seq[idx] and aa != "C"
        ]
        if pos_suggestions:
            seq[idx] = rng.choice(pos_suggestions)
        else:
            # fallback: random non-Cys
            pool = [aa for aa in AA_NO_CYS if aa != seq[idx]]
            if pool:
                seq[idx] = rng.choice(pool)

    return "".join(seq)


def generate_dedup_fallback(
    original_seq: str,
    design_positions: List[int],
    seen_sequences: "set[str]",
    rng: random.Random,
    max_attempts: int = 200,
    min_mutations: int = 2,
    max_mutations: int = 5,
) -> Optional[str]:
    """seen_sequences에 없는 유효한 fallback 서열을 확정적으로 생성한다.

    2단계 fall-back의 2단계(룰베이스 랜덤 재생성) 전담 함수.
    - FWKT pharmacophore + Cys 이황화결합 보존 (generate_random_mutant의 AA_NO_CYS 기반)
    - seen_sequences에 없는 서열 보장 (시도 상한 max_attempts)
    - 변이 수를 min_mutations ~ max_mutations 범위에서 단계적 확장 (탐색 공간 단계적 개방)

    Args:
        original_seq: native/reference 서열
        design_positions: 변이 허용 위치 목록 (scaffold 제약 적용 후, Cys/FWKT 제외)
        seen_sequences: 이미 평가한 서열 집합 (원본 수정 안 함)
        rng: 재현성용 Random 인스턴스
        max_attempts: 최대 시도 횟수 (기본 200)
        min_mutations: 첫 시도 최소 변이 수 (기본 2)
        max_mutations: 최대 변이 수 (기본 5, design_positions 길이로 clip)

    Returns:
        새 서열 문자열 (None: max_attempts 소진 = 탐색 공간 포화).

    Note:
        None 반환 시 호출자는 last_proposal을 dedup_exhausted provenance로 제출하거나
        로그를 남겨야 한다.
    """
    n_mutable = len(design_positions)
    max_n = min(max_mutations, n_mutable)

    for attempt in range(max_attempts):
        # 시도가 쌓일수록 변이 수를 단계적으로 늘려 탐색 공간 확장
        stage = attempt // max(1, max_attempts // 4)
        n_mut = min(min_mutations + stage, max_n)
        n_mut = max(n_mut, min_mutations)

        candidate = generate_random_mutant(
            original_seq=original_seq,
            design_positions=design_positions,
            rng=rng,
            n_mutations=n_mut,
        )
        if (
            candidate != original_seq
            and candidate not in seen_sequences
        ):
            return candidate

    return None


def get_high_confidence_seeds(
    mmgbsa_path: "Any",
) -> List[Dict[str, Any]]:
    """mmgbsa_consensus.json에서 high_confidence seed 목록을 반환한다.

    adapter 레이어 래퍼 — bandit.load_high_confidence_seeds 위임.

    Args:
        mmgbsa_path: mmgbsa_consensus.json 경로 (Path 또는 str)

    Returns:
        [{"sequence": str, "dg_bind": float, "consensus_flag": str}, ...]
        dg_bind 오름차순(강결합 우선).
    """
    from pathlib import Path
    from .bandit import load_high_confidence_seeds
    return load_high_confidence_seeds(Path(mmgbsa_path))
