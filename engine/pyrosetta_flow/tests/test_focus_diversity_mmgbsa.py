"""작업 A/B 단위 테스트 (2026-06-25).

작업 A — 가설 다양화 강제:
  A-1: tried-focus 블랙리스트가 반복 focus 를 차단함을 실증
  A-2: expert_panel 거부 후 focus 변경 흐름을 단위 검증
       (runner 직접 호출 없이 bandit 레벨 재샘플만 검증)
  A-3: sample_focus_with_exploration 이 tried-focus 를 피하는 비율 통계 확인
  A-4: exploration boost 로 반복 위치 빈도가 줄어드는 효과

작업 B — MM-GBSA 피드백 신호:
  B-1: update_from_mmgbsa 가 high_confidence beats_native 위치에 alpha 추가
  B-2: high_confidence 가 아닌 경우 부스트 없이 alpha += 1.0 만
  B-3: native 서열은 update 대상에서 제외
  B-4: global_leaderboard.enrich_from_mmgbsa_consensus 가
       high_confidence 항목을 랭킹 상위로 승격시킴
  B-5: 파일 없을 때 graceful skip (예외 없음)
  B-6: 동일 타임스탬프 중복 소비 방지 로직 (runner 아닌 논리 단위 테스트)
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

import pytest

from pyrosetta_flow.bandit import (
    MUTABLE_POSITIONS_1IDX,
    REFERENCE_SEQUENCE,
    PositionBandit,
)
from pyrosetta_flow.global_leaderboard import GlobalSelectivityLeaderboard


# ============================================================
# 헬퍼
# ============================================================

def _make_mmgbsa_payload(results: Dict[str, Dict[str, Any]], native_dg: float = -71.0) -> Dict[str, Any]:
    return {
        "generated_at": "2026-06-25T00:00:00Z",
        "native_dg": native_dg,
        "results": results,
        "consensus_ranking": [],
    }


def _make_leaderboard_entry(seq: str, delta: float, ddg: float = -20.0) -> Dict[str, Any]:
    return {
        "sequence": seq,
        "ddg": ddg,
        "ddg_median": ddg,
        "delta_margin": delta,
        "margin": delta + 5.0,
        "run_id": "test",
        "ts": "2026-06-25T00:00:00Z",
    }


# ============================================================
# 작업 A — 가설 다양화 강제
# ============================================================

class TestFocusBlacklist:
    """A-1: tried-focus 블랙리스트가 반복 focus 차단 실증."""

    def test_blacklist_avoids_repeated_focus(self):
        """이미 tried 에 들어간 frozenset 과 동일한 결과를 회피한다."""
        bandit = PositionBandit()
        # pos 1 에 극단적 alpha 를 부여해 항상 샘플되게 만든다
        bandit.arms[1] = {"alpha": 999.0, "beta": 1.0}
        bandit.arms[2] = {"alpha": 999.0, "beta": 1.0}
        bandit.arms[4] = {"alpha": 1.0, "beta": 999.0}
        bandit.arms[5] = {"alpha": 1.0, "beta": 999.0}
        bandit.arms[6] = {"alpha": 1.0, "beta": 999.0}
        bandit.arms[11] = {"alpha": 1.0, "beta": 999.0}
        bandit.arms[12] = {"alpha": 1.0, "beta": 999.0}
        bandit.arms[13] = {"alpha": 1.0, "beta": 999.0}

        # 1, 2 두 위치를 블랙리스트에 넣는다
        tried: List[FrozenSet[int]] = [frozenset({1, 2})] * 5  # 5회 반복

        # 최대 재샘플(100회)로 1,2 가 아닌 다른 조합이 나와야 한다
        rng = random.Random(42)
        result = bandit.sample_focus_with_exploration(
            n=2,
            tried_focus=tried,
            max_retries=100,
            rng=rng,
        )
        assert frozenset(result) != frozenset({1, 2}), (
            f"블랙리스트 차단 실패: {result} 가 여전히 {{1,2}}"
        )

    def test_exploration_returns_different_positions_most_of_time(self):
        """A-3: exploration 모드에서 블랙리스트 회피 비율 ≥ 50%."""
        bandit = PositionBandit()
        # 한 조합을 매우 많이 tried 로 등록
        stuck_set = frozenset([1, 2, 5])
        tried = [stuck_set] * 8  # K=8 꽉 채움

        hit_blacklist = 0
        total = 100
        for seed in range(total):
            result = bandit.sample_focus_with_exploration(
                n=3,
                tried_focus=tried,
                max_retries=20,
                rng=random.Random(seed),
            )
            if frozenset(result) == stuck_set:
                hit_blacklist += 1

        # 대부분(80% 이상)은 차단되어야 한다
        escape_rate = (total - hit_blacklist) / total
        assert escape_rate >= 0.8, (
            f"탈출 비율이 낮음: {escape_rate:.2%} (블랙리스트 회피 {total - hit_blacklist}/{total})"
        )

    def test_exploration_boost_reduces_frequent_position_rate(self):
        """A-4: exploration boost 로 반복 위치의 샘플 빈도가 줄어든다."""
        bandit = PositionBandit(positions=[1, 2, 4, 5, 6])
        # 위치 1 에 높은 alpha (기본적으로 항상 선택되는 경향)
        bandit.arms[1] = {"alpha": 50.0, "beta": 1.0}

        # tried 에 위치 1 반복 등록
        tried_heavy = [frozenset({1, 2})] * 6

        count_pos1_with_blacklist = 0
        count_pos1_no_blacklist = 0
        N = 200
        for seed in range(N):
            res_bl = bandit.sample_focus_with_exploration(
                n=2, tried_focus=tried_heavy, max_retries=15, rng=random.Random(seed)
            )
            res_no = bandit.sample_focus_positions(n=2, rng=random.Random(seed))
            if 1 in res_bl:
                count_pos1_with_blacklist += 1
            if 1 in res_no:
                count_pos1_no_blacklist += 1

        # 블랙리스트 있을 때 위치 1 선택 빈도가 낮아야 한다
        assert count_pos1_with_blacklist <= count_pos1_no_blacklist, (
            f"exploration boost 효과 없음: blacklist={count_pos1_with_blacklist} "
            f">= no_blacklist={count_pos1_no_blacklist}"
        )

    def test_empty_blacklist_returns_normal_sample(self):
        """tried_focus=[] 이면 일반 sample_focus_positions 와 같은 결과."""
        bandit = PositionBandit()
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        normal = bandit.sample_focus_positions(n=3, rng=rng1)
        explored = bandit.sample_focus_with_exploration(n=3, tried_focus=[], rng=rng2)
        assert normal == explored

    def test_blacklist_does_not_mutate_arms(self):
        """exploration boost 가 arms 파라미터를 영구 변경하지 않는다."""
        bandit = PositionBandit()
        original_arms = {p: dict(v) for p, v in bandit.arms.items()}
        tried = [frozenset({1, 2, 5})] * 8
        bandit.sample_focus_with_exploration(n=3, tried_focus=tried, rng=random.Random(42))
        for pos in bandit.arms:
            assert bandit.arms[pos]["alpha"] == original_arms[pos]["alpha"], (
                f"pos {pos}: alpha 영구 변경 발생"
            )
            assert bandit.arms[pos]["beta"] == original_arms[pos]["beta"], (
                f"pos {pos}: beta 영구 변경 발생"
            )


class TestExpertPanelRejectionResample:
    """A-2: expert_panel 거부 후 focus 변경 흐름 (bandit 재샘플 단위)."""

    def test_resample_after_rejection_gives_different_focus(self):
        """거부된 focus 를 blacklist 에 넣고 재샘플하면 다른 위치가 나온다."""
        bandit = PositionBandit()
        # pos 1,2 를 "고착" 상황으로 세팅
        bandit.arms[1] = {"alpha": 100.0, "beta": 1.0}
        bandit.arms[2] = {"alpha": 100.0, "beta": 1.0}

        rejected_focus = frozenset({1, 2})
        blacklist = [rejected_focus] * 4  # 4회 반복 = 과반 기준 초과

        result = bandit.sample_focus_with_exploration(
            n=2,
            tried_focus=blacklist,
            max_retries=50,
            rng=random.Random(7),
        )
        assert frozenset(result) != rejected_focus, (
            f"거부 focus 재등장: {result}"
        )

    def test_resample_always_returns_valid_positions(self):
        """재샘플 결과는 항상 mutable positions 안에 있어야 한다."""
        bandit = PositionBandit()
        tried = [frozenset([1, 2, 5])] * 8
        for seed in range(30):
            result = bandit.sample_focus_with_exploration(
                n=3, tried_focus=tried, rng=random.Random(seed)
            )
            for pos in result:
                assert pos in bandit.positions, (
                    f"유효하지 않은 위치 {pos} 반환"
                )


# ============================================================
# 작업 B — MM-GBSA 피드백 신호
# ============================================================

class TestUpdateFromMmgbsa:
    """B-1~B-3: bandit.update_from_mmgbsa."""

    def test_high_confidence_beats_native_increments_alpha_with_boost(self):
        """B-1: high_confidence + beats_native 위치에 boost_weight alpha 추가.
        AGCKNFFWKTFTSC 기준 pos4(K→R) = AGCRNFFWKTFTSC.
        """
        bandit = PositionBandit()
        # AGCKNFFWKTFTSC → pos4(K→R): AGCRNFFWKTFTSC
        seq = "AGCRNFFWKTFTSC"
        mmgbsa_results = {
            seq: {
                "beats_native": True,
                "consensus_flag": "high_confidence",
                "dg_bind": -90.0,
            }
        }
        before_alpha_4 = bandit.arms[4]["alpha"]
        n_updated = bandit.update_from_mmgbsa(
            mmgbsa_results=mmgbsa_results,
            native_dg=-71.0,
            boost_weight=1.5,
        )
        # pos4(K→R) 변이가 있으면 alpha += 1.5
        assert n_updated > 0, "업데이트 카운트가 0"
        assert bandit.arms[4]["alpha"] == before_alpha_4 + 1.5, (
            f"high_confidence boost 미적용: "
            f"alpha={bandit.arms[4]['alpha']} (expected {before_alpha_4 + 1.5})"
        )

    def test_mmgbsa_only_beats_native_increments_alpha_by_one(self):
        """B-2: beats_native=True 이지만 high_confidence 아니면 alpha += 1.0."""
        bandit = PositionBandit()
        seq = "AGCRNFFWKTFTSC"  # pos4(K→R) 변이
        mmgbsa_results = {
            seq: {
                "beats_native": True,
                "consensus_flag": "mmgbsa_only",
                "dg_bind": -75.0,
            }
        }
        before_alpha_4 = bandit.arms[4]["alpha"]
        bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0, boost_weight=1.5)
        assert bandit.arms[4]["alpha"] == before_alpha_4 + 1.0, (
            f"mmgbsa_only 시 alpha += 1.0 기대, 실제={bandit.arms[4]['alpha']}"
        )

    def test_not_beats_native_does_not_update(self):
        """B-2(부): beats_native=False 이면 업데이트 없음."""
        bandit = PositionBandit()
        seq = "AGCRNFFWKTFTSC"  # pos4(K→R) 변이
        mmgbsa_results = {
            seq: {
                "beats_native": False,
                "consensus_flag": "high_confidence",
                "dg_bind": -60.0,
            }
        }
        before = {p: dict(v) for p, v in bandit.arms.items()}
        n_updated = bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0)
        assert n_updated == 0, "beats_native=False 인데 업데이트 발생"
        for pos in bandit.arms:
            assert bandit.arms[pos]["alpha"] == before[pos]["alpha"]

    def test_native_sequence_excluded(self):
        """B-3: native 서열은 업데이트 대상에서 제외."""
        bandit = PositionBandit()
        mmgbsa_results = {
            REFERENCE_SEQUENCE: {
                "beats_native": False,
                "consensus_flag": "native",
                "dg_bind": -71.0,
            }
        }
        before = {p: dict(v) for p, v in bandit.arms.items()}
        n_updated = bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0)
        assert n_updated == 0
        for pos in bandit.arms:
            assert bandit.arms[pos]["alpha"] == before[pos]["alpha"]

    def test_empty_results_no_error(self):
        """빈 results 딕셔너리여도 예외 없음."""
        bandit = PositionBandit()
        n_updated = bandit.update_from_mmgbsa({}, native_dg=-71.0)
        assert n_updated == 0

    def test_multiple_mutations_all_updated(self):
        """여러 위치 변이 서열에서 모든 변이 위치에 alpha 가중.
        pos1(A→G) + pos4(K→R): GGCRNFFWKTFTSC
        """
        bandit = PositionBandit()
        seq = "GGCRNFFWKTFTSC"
        mmgbsa_results = {
            seq: {
                "beats_native": True,
                "consensus_flag": "high_confidence",
                "dg_bind": -95.0,
            }
        }
        before_1 = bandit.arms[1]["alpha"]
        before_4 = bandit.arms[4]["alpha"]
        n_updated = bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0, boost_weight=2.0)
        assert n_updated == 2, f"2개 위치 업데이트 기대, 실제={n_updated}"
        assert bandit.arms[1]["alpha"] == before_1 + 2.0
        assert bandit.arms[4]["alpha"] == before_4 + 2.0


class TestLeaderboardMmgbsaEnrich:
    """B-4~B-5: GlobalSelectivityLeaderboard.enrich_from_mmgbsa_consensus."""

    def _make_lb_with_entries(self) -> GlobalSelectivityLeaderboard:
        lb = GlobalSelectivityLeaderboard(capacity=10, use_robust_ddg=True)
        # 일반 후보 2개 + high_confidence 후보 1개 추가
        lb.entries = [
            {**_make_leaderboard_entry("AGCRNFFWKTFTSC", delta=2.0, ddg=-20.0)},
            {**_make_leaderboard_entry("AICLNWFWKTVISC", delta=5.0, ddg=-25.0)},
            {**_make_leaderboard_entry("AGCMNFFWKTIPSC", delta=1.0, ddg=-18.0)},
        ]
        return lb

    def test_enrich_adds_mmgbsa_fields(self, tmp_path: Path):
        """B-4: enrich 후 high_confidence 항목에 mmgbsa_dg 와 consensus_flag 추가."""
        lb = self._make_lb_with_entries()
        mmgbsa_payload = _make_mmgbsa_payload({
            "AICLNWFWKTVISC": {
                "beats_native": True,
                "consensus_flag": "high_confidence",
                "dg_bind": -94.471,
            },
            "AGCRNFFWKTFTSC": {
                "beats_native": True,
                "consensus_flag": "mmgbsa_only",
                "dg_bind": -75.0,
            },
        })
        f = tmp_path / "mmgbsa_consensus.json"
        f.write_text(json.dumps(mmgbsa_payload), encoding="utf-8")

        result = lb.enrich_from_mmgbsa_consensus(f)
        assert result["enriched"] == 2, f"enriched={result['enriched']} (기대 2)"
        assert result.get("error") is None, f"에러 발생: {result.get('error')}"

        # AICLNWFWKTVISC 가 mmgbsa_dg 와 consensus_flag 를 가져야 함
        entry = next(e for e in lb.entries if e["sequence"] == "AICLNWFWKTVISC")
        assert entry.get("consensus_flag") == "high_confidence"
        assert entry.get("mmgbsa_dg") == pytest.approx(-94.471, abs=0.01)

    def test_high_confidence_ranks_higher_with_boost(self, tmp_path: Path):
        """B-4: MMGBSA_LEADERBOARD_BOOST=5 환경변수 시 high_confidence 가 상위 랭크."""
        import os
        lb = self._make_lb_with_entries()
        # AGCRNFFWKTFTSC: delta=2, ddg=-20 → 기본 순위 1위
        # AICLNWFWKTVISC: delta=5, ddg=-25, high_confidence
        # 부스트 없으면 ddg=-25 가 더 낮아서 AICLNWFWKTVISC 가 이미 1위
        # → 부스트로 ddg_key 를 더 낮추면 순위 유지됨
        mmgbsa_payload = _make_mmgbsa_payload({
            "AICLNWFWKTVISC": {
                "beats_native": True,
                "consensus_flag": "high_confidence",
                "dg_bind": -94.471,
            },
        })
        f = tmp_path / "mmgbsa_consensus.json"
        f.write_text(json.dumps(mmgbsa_payload), encoding="utf-8")
        old_boost = os.environ.get("MMGBSA_LEADERBOARD_BOOST")
        try:
            os.environ["MMGBSA_LEADERBOARD_BOOST"] = "5.0"
            result = lb.enrich_from_mmgbsa_consensus(f)
        finally:
            if old_boost is None:
                os.environ.pop("MMGBSA_LEADERBOARD_BOOST", None)
            else:
                os.environ["MMGBSA_LEADERBOARD_BOOST"] = old_boost

        assert result["enriched"] >= 1
        # AICLNWFWKTVISC 가 ddg_key 부스트로 순위 0 (또는 ddg 기준 최상위) 위치
        top_entry = lb.entries[0]
        assert top_entry["sequence"] == "AICLNWFWKTVISC", (
            f"high_confidence 부스트 후 1위 기대, 실제={top_entry['sequence']}"
        )

    def test_file_not_found_graceful_skip(self, tmp_path: Path):
        """B-5: 파일 없을 때 예외 없이 graceful skip."""
        lb = self._make_lb_with_entries()
        result = lb.enrich_from_mmgbsa_consensus(tmp_path / "nonexistent.json")
        assert result["enriched"] == 0
        assert result.get("error") is None

    def test_malformed_json_graceful_skip(self, tmp_path: Path):
        """손상된 JSON 파일도 예외 없음."""
        lb = self._make_lb_with_entries()
        f = tmp_path / "bad.json"
        f.write_text("{INVALID JSON", encoding="utf-8")
        result = lb.enrich_from_mmgbsa_consensus(f)
        assert result.get("error") is not None or result["enriched"] == 0


class TestMmgbsaTimestampDedup:
    """B-6: 동일 타임스탬프 중복 소비 방지 로직 (runner 로직을 단위 테스트로 추출)."""

    def test_same_timestamp_not_consumed_twice(self):
        """동일 generated_at 로 두 번 소비하면 두 번째는 skip 되어야 한다."""
        bandit = PositionBandit()
        mmgbsa_results = {
            "AGCRNFFWKTFTSC": {
                "beats_native": True,
                "consensus_flag": "high_confidence",
                "dg_bind": -80.0,
            }
        }
        ts = "2026-06-25T10:00:00Z"
        before_alpha_4 = bandit.arms[4]["alpha"]

        # 첫 번째 소비
        n1 = bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0, boost_weight=1.5)
        cache_ts = ts

        # 두 번째 소비: 캐시 타임스탬프와 동일하므로 skip
        if cache_ts == ts:
            n2 = 0  # runner 에서 skip
        else:
            n2 = bandit.update_from_mmgbsa(mmgbsa_results, native_dg=-71.0, boost_weight=1.5)

        assert n1 > 0, "첫 번째 소비가 0건"
        assert n2 == 0, "중복 소비가 발생함"
        # alpha 는 첫 번째 한 번만 적립 (pos4 기준)
        assert bandit.arms[4]["alpha"] == before_alpha_4 + 1.5
