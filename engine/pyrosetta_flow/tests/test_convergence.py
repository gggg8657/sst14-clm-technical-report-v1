"""Tests for convergence.py: Mann-Whitney U, ConvergenceDetector."""
from __future__ import annotations

import math

import pytest

from pyrosetta_flow.convergence import (
    ConvergenceDetector,
    _mann_whitney_u,
    _norm_cdf,
    _rank_data,
)


# ===================================================================
# _mann_whitney_u tests  (Critical priority #4)
# ===================================================================

class TestMannWhitneyU:

    def test_identical_distributions(self):
        """Same data → high p-value (no significant difference)."""
        sample = [-10.0, -9.0, -8.0, -7.0, -6.0]
        p = _mann_whitney_u(sample, list(sample))
        assert p > 0.05

    def test_clearly_different_distributions(self):
        """Very different distributions → low p-value."""
        sample1 = [-50.0, -45.0, -40.0, -35.0, -30.0]
        sample2 = [10.0, 15.0, 20.0, 25.0, 30.0]
        p = _mann_whitney_u(sample1, sample2)
        assert p < 0.05

    def test_small_samples_return_1(self):
        """n_total < 8 → p=1.0 (insufficient data)."""
        p = _mann_whitney_u([1.0, 2.0], [3.0])
        assert p == 1.0

    def test_empty_sample(self):
        assert _mann_whitney_u([], [1.0, 2.0]) == 1.0
        assert _mann_whitney_u([1.0], []) == 1.0

    def test_ties_handled(self):
        """Tied values should not cause errors."""
        sample1 = [1.0, 1.0, 1.0, 1.0, 2.0]
        sample2 = [1.0, 1.0, 2.0, 2.0, 3.0]
        p = _mann_whitney_u(sample1, sample2)
        assert 0.0 <= p <= 1.0

    def test_single_distinct_value(self):
        """All same values → sigma=0 → p=1.0."""
        sample = [5.0] * 5
        p = _mann_whitney_u(sample, list(sample))
        assert p == 1.0

    def test_p_value_range(self):
        """p-value should always be in [0, 1]."""
        import random
        rng = random.Random(42)
        for _ in range(20):
            s1 = [rng.gauss(0, 1) for _ in range(10)]
            s2 = [rng.gauss(0, 1) for _ in range(10)]
            p = _mann_whitney_u(s1, s2)
            assert 0.0 <= p <= 1.0


# ===================================================================
# _rank_data tests
# ===================================================================

class TestRankData:

    def test_no_ties(self):
        ranks = _rank_data([10.0, 20.0, 30.0])
        assert ranks == [1.0, 2.0, 3.0]

    def test_all_tied(self):
        ranks = _rank_data([5.0, 5.0, 5.0])
        assert ranks == [2.0, 2.0, 2.0]  # average of ranks 1,2,3

    def test_partial_ties(self):
        ranks = _rank_data([10.0, 20.0, 20.0, 30.0])
        assert ranks == [1.0, 2.5, 2.5, 4.0]


# ===================================================================
# _norm_cdf tests
# ===================================================================

class TestNormCDF:

    def test_zero(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-10

    def test_large_positive(self):
        assert _norm_cdf(5.0) > 0.999

    def test_large_negative(self):
        assert _norm_cdf(-5.0) < 0.001

    def test_known_value(self):
        # CDF(1.96) ≈ 0.975
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001


# ===================================================================
# ConvergenceDetector tests  (High priority #7)
# ===================================================================

class TestConvergenceDetector:

    def test_insufficient_history(self):
        """Need at least 2*window_size iterations."""
        det = ConvergenceDetector(window_size=3)
        det.add_iteration(1, [-10.0, -9.0, -8.0])
        converged, details = det.is_converged()
        assert converged is False
        assert "Need at least" in details["recommendation"]

    def test_convergence_detection_plateau(self):
        """Identical data across windows → convergence."""
        det = ConvergenceDetector(window_size=2, significance_level=0.05)
        # 4 iterations of very similar data
        for i in range(1, 5):
            det.add_iteration(i, [-10.0, -10.1, -9.9])
        converged, details = det.is_converged()
        assert converged is True
        assert details["p_value"] is not None
        assert "Converged" in details["recommendation"]

    def test_not_converged_improving(self):
        """Clearly improving trend → not converged."""
        det = ConvergenceDetector(window_size=2, significance_level=0.05)
        det.add_iteration(1, [0.0, 1.0, 2.0])
        det.add_iteration(2, [0.0, 1.0, 2.0])
        det.add_iteration(3, [-50.0, -45.0, -40.0])
        det.add_iteration(4, [-50.0, -45.0, -40.0])
        converged, details = det.is_converged()
        assert converged is False

    def test_high_variance_not_converged(self):
        """Even with no statistical difference, high CV prevents convergence."""
        det = ConvergenceDetector(window_size=2, significance_level=0.05)
        # Same mean but high variance
        det.add_iteration(1, [-100.0, 100.0])
        det.add_iteration(2, [-100.0, 100.0])
        det.add_iteration(3, [-100.0, 100.0])
        det.add_iteration(4, [-100.0, 100.0])
        converged, details = det.is_converged()
        # CV should be very high → not converged (or p might be high but CV > 0.15)
        if details["cv"] is not None and details["cv"] >= 0.15:
            assert converged is False

    def test_custom_window_size(self):
        det = ConvergenceDetector(window_size=1, significance_level=0.05)
        det.add_iteration(1, [-10.0, -10.0])
        det.add_iteration(2, [-10.0, -10.0])
        converged, details = det.is_converged()
        # With window_size=1, 2 iterations is enough
        assert details["p_value"] is not None

    def test_add_iteration_accumulates(self):
        det = ConvergenceDetector(window_size=3)
        for i in range(1, 8):
            det.add_iteration(i, [-float(i)])
        assert len(det._history) == 7
