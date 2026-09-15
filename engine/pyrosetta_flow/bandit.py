"""
Multi-Armed Bandit Position Optimizer
======================================
Thompson Sampling bandit that learns which mutable positions are most
promising for producing favorable ddG values.  Each mutable position
maintains a Beta(alpha, beta) distribution; positions that historically
yield improvements get higher alpha, and positions that yield worsening
get higher beta.

Usage:
    bandit = PositionBandit()
    bandit.initialize_from_history(records)
    focus = bandit.sample_focus_positions(n=3)

2026-06-25 개선:
- sample_focus_with_exploration(): tried-focus 블랙리스트를 받아 미탐색 위치 강제 탐색
- update_from_mmgbsa(): MM-GBSA 데몬 consensus 신호로 개선 가중 강화(직교 확인)
  caveat: MM-GBSA는 45분 주기 단일 스냅샷 비동기 신호 — 보조용, 도킹 대체 아님.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

REFERENCE_SEQUENCE = "AGCKNFFWKTFTSC"
# Mutable positions (1-indexed, matching adapter.py convention)
MUTABLE_POSITIONS_1IDX = [1, 2, 4, 5, 6, 11, 12, 13]

DDG_PLAUSIBLE_MIN = -60.0
DDG_PLAUSIBLE_MAX = 200.0


class PositionBandit:
    """Thompson Sampling bandit over mutable peptide positions.

    Each position has a Beta(alpha, beta) prior.  Observing an improvement
    increments alpha; observing a worsening increments beta.
    """

    def __init__(
        self,
        positions: list[int] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ) -> None:
        self.positions = positions or list(MUTABLE_POSITIONS_1IDX)
        self.arms: dict[int, dict[str, float]] = {
            pos: {"alpha": prior_alpha, "beta": prior_beta}
            for pos in self.positions
        }

    def initialize_from_history(self, records: list[dict]) -> None:
        """Bootstrap arm parameters from historical experiment records.

        For each candidate, identify which mutable positions were mutated
        relative to WT.  If the candidate's ddG is better (lower) than the
        WT mean, increment alpha for those positions; otherwise increment beta.
        """
        candidates = [
            r for r in records
            if r.get("record_type") == "candidate"
            and r.get("status") == "success"
            and DDG_PLAUSIBLE_MIN <= float(r.get("ddg", 999)) <= DDG_PLAUSIBLE_MAX
            and float(r.get("ddg", 999)) < 900
        ]
        if not candidates:
            return

        # Compute WT baseline mean ddG
        wt_ddgs: list[float] = []
        for r in candidates:
            seq = r.get("sequence", "")
            if seq == REFERENCE_SEQUENCE:
                wt_ddgs.append(float(r["ddg"]))

        # If no WT observations, use the median of all ddG values as baseline
        if wt_ddgs:
            baseline = sum(wt_ddgs) / len(wt_ddgs)
        else:
            all_ddgs = sorted(float(r["ddg"]) for r in candidates)
            baseline = all_ddgs[len(all_ddgs) // 2]

        for r in candidates:
            seq = r.get("sequence", "")
            if len(seq) != len(REFERENCE_SEQUENCE):
                continue
            ddg = float(r["ddg"])
            # Identify which mutable positions were changed
            mutated_positions = []
            for pos in self.positions:
                idx = pos - 1  # convert to 0-indexed
                if idx < len(seq) and seq[idx] != REFERENCE_SEQUENCE[idx]:
                    mutated_positions.append(pos)

            if not mutated_positions:
                continue

            improved = ddg < baseline
            for pos in mutated_positions:
                if improved:
                    self.arms[pos]["alpha"] += 1.0
                else:
                    self.arms[pos]["beta"] += 1.0

    def sample_focus_positions(self, n: int = 3, rng: random.Random | None = None) -> list[int]:
        """Thompson-sample from each arm's Beta distribution and return top-n positions."""
        if rng is None:
            rng = random.Random()

        samples: list[tuple[float, int]] = []
        for pos, params in self.arms.items():
            # Beta distribution sample via random.betavariate
            theta = rng.betavariate(params["alpha"], params["beta"])
            samples.append((theta, pos))

        # Return top-n positions with highest sampled values
        samples.sort(reverse=True)
        return [pos for _, pos in samples[:n]]

    def sample_focus_with_exploration(
        self,
        n: int = 3,
        tried_focus: Optional[List[FrozenSet[int]]] = None,
        max_retries: int = 10,
        rng: Optional[random.Random] = None,
    ) -> List[int]:
        """tried-focus 블랙리스트를 피해 미탐색 위치를 강제 탐색하는 샘플링.

        tried_focus 에 있는 frozenset 조합과 동일한 결과는 최대 max_retries 번
        재샘플한다. 재샘플 시 미탐색/저빈도 위치에 exploration boost(beta 가중
        임시 하향)를 적용해 고착 탈피. 모든 시도가 블랙리스트와 겹치면 가장
        안 나온 위치를 강제 포함한다.

        Args:
            n: 반환할 위치 수
            tried_focus: 최근 K iteration 의 focus frozenset 목록 (블랙리스트)
            max_retries: 재샘플 최대 횟수
            rng: 재현성용 Random 인스턴스

        Returns:
            focus position 리스트 (len ≤ n)

        caveat: exploration boost 는 해당 샘플링 호출 동안만 임시 적용 — arm
        파라미터를 영구 변경하지 않는다.
        """
        if rng is None:
            rng = random.Random()
        if not tried_focus:
            return self.sample_focus_positions(n=n, rng=rng)

        blacklist: Set[FrozenSet[int]] = set(tried_focus)

        # 블랙리스트에 없는 위치 집합 추적 (빈도 기반)
        position_counts: Dict[int, int] = defaultdict(int)
        for fs in tried_focus:
            for pos in fs:
                position_counts[pos] += 1

        def _sample_once(arms_override: Optional[Dict[int, Dict[str, float]]] = None) -> List[int]:
            arms = arms_override or self.arms
            samples: List[tuple] = []
            for pos, params in arms.items():
                theta = rng.betavariate(params["alpha"], params["beta"])
                samples.append((theta, pos))
            samples.sort(reverse=True)
            return [pos for _, pos in samples[:n]]

        # 1차: 단순 재샘플 시도
        for _ in range(max_retries):
            candidate = _sample_once()
            candidate_set = frozenset(candidate)
            if candidate_set not in blacklist:
                return candidate

        # 2차: exploration boost — 최근 많이 쓰인 위치의 beta 를 임시 상향
        # (theta 샘플이 낮아져 덜 선택됨)
        boosted_arms: Dict[int, Dict[str, float]] = {}
        max_count = max(position_counts.values()) if position_counts else 1
        for pos, params in self.arms.items():
            cnt = position_counts.get(pos, 0)
            # 빈도 비례 beta 가중 (+2 per 최대빈도 대비 비율)
            extra_beta = 2.0 * (cnt / max(max_count, 1))
            boosted_arms[pos] = {
                "alpha": params["alpha"],
                "beta": params["beta"] + extra_beta,
            }
        for _ in range(max_retries):
            candidate = _sample_once(boosted_arms)
            candidate_set = frozenset(candidate)
            if candidate_set not in blacklist:
                return candidate

        # 3차: 블랙리스트에 없는 미탐색 위치를 강제 포함 (least-tried 우선)
        sorted_positions = sorted(
            self.arms.keys(),
            key=lambda p: position_counts.get(p, 0),
        )
        # 최소 빈도 위치를 1개 이상 포함하도록 강제 + 나머지는 bandit 샘플
        forced = sorted_positions[:max(1, n // 2)]
        remaining_arms = {p: self.arms[p] for p in self.arms if p not in forced}
        if remaining_arms and len(forced) < n:
            extra_samples: List[tuple] = []
            for pos, params in remaining_arms.items():
                theta = rng.betavariate(params["alpha"], params["beta"])
                extra_samples.append((theta, pos))
            extra_samples.sort(reverse=True)
            need = n - len(forced)
            extra = [pos for _, pos in extra_samples[:need]]
        else:
            extra = []
        result = (forced + extra)[:n]
        return result

    def update(
        self,
        position: int,
        improved: bool,
        ddg: Optional[float] = None,
        baseline: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> None:
        """Update a single arm after observing a result.

        If ddg, baseline, and sigma are available, use a continuous Z-score
        reward: r = (baseline - ddg) / sigma, so lower ddG gives positive
        reward.  Small absolute scores below 0.5 are treated as noise and do
        not update the arm.  If sigma is not passed, DDG_SIGMA is tried from
        the environment before falling back to the legacy binary update.
        """
        if position not in self.arms:
            return

        if ddg is not None and baseline is not None:
            if sigma is None:
                env_sigma = os.getenv("DDG_SIGMA")
                if env_sigma is not None:
                    try:
                        sigma = float(env_sigma)
                    except ValueError:
                        sigma = None

            if sigma is not None and sigma > 0.0:
                r = (baseline - ddg) / sigma
                if abs(r) < 0.5:
                    return
                if r > 0.5:
                    self.arms[position]["alpha"] += r
                elif r < -0.5:
                    self.arms[position]["beta"] += abs(r)
                return

        if improved:
            self.arms[position]["alpha"] += 1.0
        else:
            self.arms[position]["beta"] += 1.0

    def update_from_mmgbsa(
        self,
        mmgbsa_results: Dict[str, Any],
        native_dg: float,
        boost_weight: float = 1.5,
    ) -> int:
        """MM-GBSA consensus 결과로 변이 위치에 추가 보상을 부여한다.

        MM-GBSA로 native 보다 낮은 dg_bind 가 확인된(beats_native=True) 서열의
        변이 위치를 특정하고, consensus_flag='high_confidence' 이면 boost_weight
        를 곱한 추가 alpha 를 적립한다.

        caveat:
        - MM-GBSA 는 45분 주기 단일 스냅샷 비동기 신호이며 MD/엔트로피 미포함.
        - 도킹 신호를 대체하지 않고 보조하는 용도로만 사용한다.
        - boost_weight 는 MMGBSA_BANDIT_BOOST 환경변수로 조정 가능.

        Args:
            mmgbsa_results: mmgbsa_consensus.json 의 ``results`` dict
                            {sequence: {dg_bind, beats_native, consensus_flag, ...}}
            native_dg: mmgbsa_consensus.json 의 ``native_dg`` 값
            boost_weight: high_confidence 에 적용할 추가 alpha 배수 (기본 1.5)

        Returns:
            업데이트된 arm 수
        """
        updated = 0
        for seq, entry in mmgbsa_results.items():
            if seq == REFERENCE_SEQUENCE:
                continue
            if not entry.get("beats_native", False):
                continue
            flag = entry.get("consensus_flag", "")
            multiplier = boost_weight if flag == "high_confidence" else 1.0
            # 변이 위치 특정 (REFERENCE_SEQUENCE 대비)
            if len(seq) != len(REFERENCE_SEQUENCE):
                continue
            mutated = [
                pos for pos in self.positions
                if (pos - 1) < len(seq) and seq[pos - 1] != REFERENCE_SEQUENCE[pos - 1]
            ]
            if not mutated:
                continue
            for pos in mutated:
                self.arms[pos]["alpha"] += multiplier
            updated += len(mutated)
        return updated

    def get_arm_stats(self) -> dict[int, dict[str, float]]:
        """Return current arm parameters and expected value for diagnostics."""
        stats = {}
        for pos, params in self.arms.items():
            a, b = params["alpha"], params["beta"]
            stats[pos] = {
                "alpha": a,
                "beta": b,
                "expected_value": round(a / (a + b), 4),
                "n_observations": int(a + b - 2),  # subtract priors
            }
        return stats


# ---------------------------------------------------------------------------
# Intensification helpers — high_confidence seed 주변 local 탐색
# ---------------------------------------------------------------------------

# 보존적 치환 그룹 (물리화학적 유사성 기반)
CONSERVATIVE_SUBSTITUTIONS: Dict[str, List[str]] = {
    # 지방족 비극성
    "I": ["V", "L"],
    "V": ["I", "L"],
    "L": ["I", "V"],
    # 방향족
    "W": ["F", "Y"],
    "F": ["W", "Y"],
    "Y": ["F", "W"],
    # 작은 극성/무극성
    "A": ["G", "S"],
    "G": ["A"],
    "S": ["T", "A"],
    "T": ["S", "V"],
    # 산성
    "D": ["E", "N"],
    "E": ["D", "Q"],
    # 염기성
    "K": ["R", "H"],
    "R": ["K", "H"],
    "H": ["K", "R"],
    # 아미드
    "N": ["Q", "D"],
    "Q": ["N", "E"],
    # 함황
    "M": ["L", "I"],
    "P": ["A", "G"],
}


def get_intensification_guidance(
    seed_sequence: str,
    reference_sequence: str,
    mutable_positions: List[int],
    rng: Optional[random.Random] = None,
    max_suggestions_per_pos: int = 4,
) -> Dict[str, Any]:
    """high_confidence seed 서열에서 local intensification guidance를 생성한다.

    seed_sequence와 reference_sequence를 비교하여 변이된 위치(변이 포지션)를 찾고,
    그 주변(pos±1) + 보존적 치환 + 부분조합 후보를 제안한다.

    Rules:
    - FWKT pharmacophore(pos7-10) 및 Cys(pos3, pos14)는 절대 변이 금지.
    - mutable_positions에 없는 위치는 skip.
    - 반환값은 generate_guided_mutant() 호환 guidance dict.

    Args:
        seed_sequence: high_confidence 서열 (예: "AICLNWFWKTVISC")
        reference_sequence: native 서열 (예: "AGCKNFFWKTFTSC")
        mutable_positions: 변이 가능 위치 목록 (1-indexed, scaffold 제약 이미 적용됨)
        rng: 재현성용 Random 인스턴스
        max_suggestions_per_pos: 위치당 최대 제안 아미노산 수

    Returns:
        {
            "focus_positions": [int, ...],      # 탐색 우선 위치 (변이 위치 + 인접)
            "suggested_mutations": {str: [str]}, # 위치별 추천 아미노산
            "intensify_seed": str,               # 사용된 seed 서열
            "source": "intensification"
        }
    """
    if rng is None:
        rng = random.Random()

    ref_len = len(reference_sequence)
    seed_len = len(seed_sequence)
    if seed_len != ref_len:
        # 길이 불일치: 빈 guidance 반환
        return {"focus_positions": [], "suggested_mutations": {}, "source": "intensification"}

    mutable_set = set(mutable_positions)

    # 1. seed vs native 변이 위치 파악 (1-indexed)
    mutated_in_seed: List[int] = [
        pos for pos in mutable_positions
        if seed_sequence[pos - 1] != reference_sequence[pos - 1]
    ]

    # 2. 인접 위치 확장 (pos±1 이면서 mutable_set에 있는 것)
    adjacent_positions: List[int] = []
    for pos in mutated_in_seed:
        for neighbor in (pos - 1, pos + 1):
            if neighbor in mutable_set and neighbor not in mutated_in_seed:
                adjacent_positions.append(neighbor)

    # 3. focus 후보 = 변이 위치 + 인접 위치 (중복 제거, 순서 보존)
    focus_candidates: List[int] = []
    seen_focus: Set[int] = set()
    for pos in mutated_in_seed + adjacent_positions:
        if pos not in seen_focus:
            focus_candidates.append(pos)
            seen_focus.add(pos)

    # 4. 위치별 suggested_mutations 구성
    #    - 변이 위치: seed에 있는 아미노산 + 그 보존적 치환
    #    - 인접 위치: 보존적 치환 + 랜덤 샘플
    suggested: Dict[str, List[str]] = {}
    for pos in focus_candidates:
        idx = pos - 1
        seed_aa = seed_sequence[idx] if idx < seed_len else reference_sequence[idx]
        ref_aa = reference_sequence[idx]
        candidates: List[str] = []

        # seed에서 변이된 아미노산 자체를 우선 포함
        if seed_aa != ref_aa and seed_aa != "C":
            candidates.append(seed_aa)

        # seed_aa의 보존적 치환 추가
        for conserv in CONSERVATIVE_SUBSTITUTIONS.get(seed_aa, []):
            if conserv not in candidates and conserv != ref_aa and conserv != "C":
                candidates.append(conserv)

        # ref_aa의 보존적 치환도 포함 (seed와는 다른 방향)
        for conserv in CONSERVATIVE_SUBSTITUTIONS.get(ref_aa, []):
            if conserv not in candidates and conserv != ref_aa and conserv != "C":
                candidates.append(conserv)

        # 부족하면 랜덤 보충 (AA_NO_CYS에서)
        _aa_pool = list("ADEFGHIKLMNPQRSTVWY")
        if len(candidates) < max_suggestions_per_pos:
            rng.shuffle(_aa_pool)
            for aa in _aa_pool:
                if aa not in candidates and aa != ref_aa and aa != "C":
                    candidates.append(aa)
                    if len(candidates) >= max_suggestions_per_pos:
                        break

        if candidates:
            suggested[str(pos)] = candidates[:max_suggestions_per_pos]

    return {
        "focus_positions": focus_candidates,
        "suggested_mutations": suggested,
        "intensify_seed": seed_sequence,
        "source": "intensification",
    }


def load_high_confidence_seeds(
    mmgbsa_path: "Path",
    beats_native: bool = True,
) -> List[Dict[str, Any]]:
    """mmgbsa_consensus.json에서 high_confidence 서열을 seed로 로드한다.

    beats_native=True이면 native보다 낮은 dg_bind를 가진 항목만 포함.

    Args:
        mmgbsa_path: mmgbsa_consensus.json 경로
        beats_native: True이면 beats_native=True인 항목만 (기본값)

    Returns:
        [{"sequence": str, "dg_bind": float, "consensus_flag": str}, ...]
        dg_bind 오름차순(강결합 우선) 정렬.
    """
    import json
    try:
        raw = json.loads(mmgbsa_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    results = raw.get("results", {})
    seeds: List[Dict[str, Any]] = []
    for seq, entry in results.items():
        if entry.get("consensus_flag") != "high_confidence":
            continue
        if beats_native and not entry.get("beats_native", False):
            continue
        seeds.append({
            "sequence": seq,
            "dg_bind": float(entry.get("dg_bind", 0.0)),
            "consensus_flag": entry.get("consensus_flag", ""),
        })

    seeds.sort(key=lambda x: x["dg_bind"])
    return seeds
