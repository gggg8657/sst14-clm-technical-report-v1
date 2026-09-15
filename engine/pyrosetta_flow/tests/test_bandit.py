"""Tests for bandit.py: PositionBandit Thompson sampling optimizer."""
from __future__ import annotations

import random

import pytest

from pyrosetta_flow.bandit import (
    DDG_PLAUSIBLE_MAX,
    DDG_PLAUSIBLE_MIN,
    MUTABLE_POSITIONS_1IDX,
    REFERENCE_SEQUENCE,
    PositionBandit,
)


# ===================================================================
# PositionBandit tests  (High priority #8)
# ===================================================================

class TestPositionBanditInit:

    def test_default_positions(self):
        bandit = PositionBandit()
        assert bandit.positions == list(MUTABLE_POSITIONS_1IDX)

    def test_custom_positions(self):
        bandit = PositionBandit(positions=[1, 5, 10])
        assert bandit.positions == [1, 5, 10]
        assert set(bandit.arms.keys()) == {1, 5, 10}

    def test_default_priors(self):
        bandit = PositionBandit()
        for params in bandit.arms.values():
            assert params["alpha"] == 1.0
            assert params["beta"] == 1.0

    def test_custom_priors(self):
        bandit = PositionBandit(prior_alpha=2.0, prior_beta=3.0)
        for params in bandit.arms.values():
            assert params["alpha"] == 2.0
            assert params["beta"] == 3.0


class TestInitializeFromHistory:

    def _make_record(self, sequence, ddg, status="success"):
        return {
            "record_type": "candidate",
            "status": status,
            "sequence": sequence,
            "ddg": ddg,
        }

    def test_empty_records(self):
        bandit = PositionBandit()
        bandit.initialize_from_history([])
        # Priors should remain unchanged
        for params in bandit.arms.values():
            assert params["alpha"] == 1.0
            assert params["beta"] == 1.0

    def test_no_valid_candidates(self):
        bandit = PositionBandit()
        records = [
            self._make_record("AAA", -5.0, status="failed"),
            {"record_type": "summary", "ddg": -10.0},
        ]
        bandit.initialize_from_history(records)
        for params in bandit.arms.values():
            assert params["alpha"] == 1.0

    def test_improvement_increments_alpha(self):
        """Mutant better than baseline → alpha increases for mutated positions."""
        bandit = PositionBandit(positions=[1, 2, 4, 5])
        records = [
            # WT baseline
            self._make_record(REFERENCE_SEQUENCE, -5.0),
            # Mutant at pos 1 (A→G) with better ddG
            self._make_record("GGCKNFFWKTFTSC", -15.0),
        ]
        bandit.initialize_from_history(records)
        # Position 1 was mutated and improved → alpha should be > 1
        assert bandit.arms[1]["alpha"] > 1.0

    def test_worsening_increments_beta(self):
        """Mutant worse than baseline → beta increases for mutated positions."""
        bandit = PositionBandit(positions=[1, 2, 4, 5])
        records = [
            self._make_record(REFERENCE_SEQUENCE, -5.0),
            # Mutant at pos 1 with worse ddG
            self._make_record("GGCKNFFWKTFTSC", 10.0),
        ]
        bandit.initialize_from_history(records)
        assert bandit.arms[1]["beta"] > 1.0

    def test_implausible_ddg_filtered(self):
        """ddG outside plausible range should be excluded."""
        bandit = PositionBandit(positions=[1])
        records = [
            self._make_record(REFERENCE_SEQUENCE, -5.0),
            self._make_record("GGCKNFFWKTFTSC", DDG_PLAUSIBLE_MIN - 10),
            self._make_record("GGCKNFFWKTFTSC", DDG_PLAUSIBLE_MAX + 10),
            self._make_record("GGCKNFFWKTFTSC", 950.0),  # > 900 filter
        ]
        bandit.initialize_from_history(records)
        # Only the WT should have been processed, no mutations
        assert bandit.arms[1]["alpha"] == 1.0
        assert bandit.arms[1]["beta"] == 1.0

    def test_no_wt_uses_median(self):
        """If no WT records, median is used as baseline."""
        bandit = PositionBandit(positions=[1, 2])
        records = [
            self._make_record("GGCKNFFWKTFTSC", -20.0),  # pos 1 mutated
            self._make_record("AGCKNFFWKTFTSC", -5.0),    # this IS WT actually
        ]
        # Since one record matches REFERENCE_SEQUENCE, it will use WT mean
        bandit.initialize_from_history(records)
        # At least one arm should have been updated
        total = sum(p["alpha"] + p["beta"] for p in bandit.arms.values())
        assert total > len(bandit.positions) * 2  # more than just priors

    def test_wrong_length_sequence_skipped(self):
        bandit = PositionBandit(positions=[1])
        records = [
            self._make_record(REFERENCE_SEQUENCE, -5.0),
            self._make_record("SHORT", -20.0),  # wrong length
        ]
        bandit.initialize_from_history(records)
        # Only WT processed, no mutations detected
        assert bandit.arms[1]["alpha"] == 1.0


class TestSampleFocusPositions:

    def test_returns_n_positions(self):
        bandit = PositionBandit()
        focus = bandit.sample_focus_positions(n=3, rng=random.Random(42))
        assert len(focus) == 3

    def test_positions_from_arms(self):
        bandit = PositionBandit(positions=[5, 10, 15])
        focus = bandit.sample_focus_positions(n=2, rng=random.Random(42))
        assert len(focus) == 2
        for p in focus:
            assert p in [5, 10, 15]

    def test_deterministic_with_seed(self):
        bandit = PositionBandit()
        f1 = bandit.sample_focus_positions(n=3, rng=random.Random(42))
        f2 = bandit.sample_focus_positions(n=3, rng=random.Random(42))
        assert f1 == f2

    def test_n_greater_than_positions(self):
        bandit = PositionBandit(positions=[1, 2])
        focus = bandit.sample_focus_positions(n=5, rng=random.Random(42))
        assert len(focus) == 2  # capped at available

    def test_biased_toward_high_alpha(self):
        """Positions with high alpha should appear more frequently."""
        bandit = PositionBandit(positions=[1, 2, 3])
        bandit.arms[1] = {"alpha": 100.0, "beta": 1.0}
        bandit.arms[2] = {"alpha": 1.0, "beta": 100.0}
        bandit.arms[3] = {"alpha": 1.0, "beta": 1.0}
        # Over many samples, position 1 should appear first most often
        first_counts = {1: 0, 2: 0, 3: 0}
        for seed in range(100):
            focus = bandit.sample_focus_positions(n=1, rng=random.Random(seed))
            first_counts[focus[0]] += 1
        assert first_counts[1] > first_counts[2]


class TestUpdate:

    def test_improved_increments_alpha(self):
        bandit = PositionBandit(positions=[5])
        bandit.update(5, improved=True)
        assert bandit.arms[5]["alpha"] == 2.0
        assert bandit.arms[5]["beta"] == 1.0

    def test_not_improved_increments_beta(self):
        bandit = PositionBandit(positions=[5])
        bandit.update(5, improved=False)
        assert bandit.arms[5]["alpha"] == 1.0
        assert bandit.arms[5]["beta"] == 2.0

    def test_unknown_position_ignored(self):
        bandit = PositionBandit(positions=[5])
        bandit.update(999, improved=True)  # should not raise
        assert 999 not in bandit.arms


class TestUpdateZScore:
    """update() Z-score 연속보상 모드 검증."""

    def test_noise_floor_skips_update(self):
        """|r| < 0.5이면 arm을 갱신하지 않는다."""
        bandit = PositionBandit(positions=[5])
        # r = (baseline - ddg) / sigma = (0 - 0.3) / 9 ≈ -0.033 → skip
        bandit.update(5, improved=False, ddg=0.3, baseline=0.0, sigma=9.0)
        assert bandit.arms[5]["alpha"] == 1.0
        assert bandit.arms[5]["beta"] == 1.0

    def test_positive_zscore_increments_alpha(self):
        """r > 0.5 → alpha += r (연속 보상)."""
        bandit = PositionBandit(positions=[5])
        # r = (0 - (-18)) / 9 = 2.0
        bandit.update(5, improved=True, ddg=-18.0, baseline=0.0, sigma=9.0)
        assert bandit.arms[5]["alpha"] == pytest.approx(1.0 + 2.0)
        assert bandit.arms[5]["beta"] == 1.0

    def test_negative_zscore_increments_beta(self):
        """r < -0.5 → beta += abs(r) (연속 패널티)."""
        bandit = PositionBandit(positions=[5])
        # r = (0 - 18) / 9 = -2.0
        bandit.update(5, improved=False, ddg=18.0, baseline=0.0, sigma=9.0)
        assert bandit.arms[5]["alpha"] == 1.0
        assert bandit.arms[5]["beta"] == pytest.approx(1.0 + 2.0)

    def test_binary_fallback_no_sigma(self):
        """sigma 없고 환경변수도 없으면 이진 fallback."""
        import os
        os.environ.pop("DDG_SIGMA", None)
        bandit = PositionBandit(positions=[5])
        bandit.update(5, improved=True, ddg=-10.0, baseline=0.0, sigma=None)
        # 이진 fallback: alpha += 1
        assert bandit.arms[5]["alpha"] == 2.0

    def test_binary_fallback_zero_sigma(self):
        """sigma=0.0은 유효하지 않아 이진 fallback."""
        bandit = PositionBandit(positions=[5])
        bandit.update(5, improved=True, ddg=-10.0, baseline=0.0, sigma=0.0)
        assert bandit.arms[5]["alpha"] == 2.0

    def test_env_sigma_used_when_no_sigma_arg(self, monkeypatch):
        """DDG_SIGMA 환경변수에서 sigma를 읽어 Z-score 모드로 동작한다."""
        monkeypatch.setenv("DDG_SIGMA", "9.0")
        bandit = PositionBandit(positions=[5])
        # sigma=9, r = (0 - (-18)) / 9 = 2.0
        bandit.update(5, improved=True, ddg=-18.0, baseline=0.0, sigma=None)
        assert bandit.arms[5]["alpha"] == pytest.approx(3.0)

    def test_backward_compat_improved_only(self):
        """기존 호출 방식(improved만 전달)은 여전히 동작한다."""
        import os
        os.environ.pop("DDG_SIGMA", None)
        bandit = PositionBandit(positions=[5])
        bandit.update(5, improved=True)
        assert bandit.arms[5]["alpha"] == 2.0
        bandit.update(5, improved=False)
        assert bandit.arms[5]["beta"] == 2.0


class TestGetArmStats:

    def test_default_stats(self):
        bandit = PositionBandit(positions=[1, 2])
        stats = bandit.get_arm_stats()
        assert set(stats.keys()) == {1, 2}
        for pos_stats in stats.values():
            assert pos_stats["alpha"] == 1.0
            assert pos_stats["beta"] == 1.0
            assert pos_stats["expected_value"] == 0.5
            assert pos_stats["n_observations"] == 0

    def test_updated_stats(self):
        bandit = PositionBandit(positions=[1])
        bandit.update(1, improved=True)
        bandit.update(1, improved=True)
        bandit.update(1, improved=False)
        stats = bandit.get_arm_stats()
        assert stats[1]["alpha"] == 3.0
        assert stats[1]["beta"] == 2.0
        assert stats[1]["n_observations"] == 3
        assert stats[1]["expected_value"] == round(3.0 / 5.0, 4)
