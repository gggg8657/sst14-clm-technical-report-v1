"""
Statistical Convergence Detection
==================================
Detects when iterative optimization has plateaued using Mann-Whitney U test
and coefficient of variation. No scipy dependency -- rank-sum computed manually.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _rank_data(values: List[float]) -> List[float]:
    """Assign ranks to values, handling ties with average rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average rank for tied values
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def _mann_whitney_u(sample1: List[float], sample2: List[float]) -> float:
    """Compute two-sided Mann-Whitney U p-value.

    For n >= 8 uses normal approximation with continuity correction.
    For smaller n returns 1.0 (not enough data to reject).
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 1.0

    combined = sample1 + sample2
    ranks = _rank_data(combined)

    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0

    n_total = n1 + n2
    if n_total < 8:
        return 1.0

    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n_total + 1) / 12.0)
    if sigma == 0:
        return 1.0

    z = abs(u1 - mu) / sigma

    # Two-sided p-value from standard normal (approximation via error function)
    p = 2.0 * (1.0 - _norm_cdf(z))
    return p


def _norm_cdf(z: float) -> float:
    """Standard normal CDF approximation using the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class ConvergenceDetector:
    """Track top-k ddG values per iteration and detect convergence plateau."""

    def __init__(self, window_size: int = 3, significance_level: float = 0.05) -> None:
        self.window_size = window_size
        self.significance_level = significance_level
        self._history: List[Tuple[int, List[float]]] = []  # (iteration, ddgs)

    def add_iteration(self, iteration: int, top_k_ddgs: List[float]) -> None:
        """Record top-k ddG values for a completed iteration."""
        self._history.append((iteration, list(top_k_ddgs)))

    def is_converged(self) -> Tuple[bool, Dict]:
        """Check whether optimisation has converged.

        Returns:
            (converged, details) where details contains p_value, cv,
            n_windows, and a human-readable recommendation.
        """
        n = len(self._history)
        details: Dict = {
            "p_value": None,
            "cv": None,
            "n_windows": n,
            "recommendation": "",
        }

        if n < 2 * self.window_size:
            details["recommendation"] = (
                f"Need at least {2 * self.window_size} iterations "
                f"(have {n}); keep running."
            )
            return False, details

        # Current window and previous window of ddG values (flattened)
        current_window = []
        for _, ddgs in self._history[-self.window_size:]:
            current_window.extend(ddgs)

        previous_window = []
        for _, ddgs in self._history[-2 * self.window_size: -self.window_size]:
            previous_window.extend(ddgs)

        # Mann-Whitney U test: p > significance_level means NO significant difference
        p_value = _mann_whitney_u(previous_window, current_window)
        details["p_value"] = round(p_value, 6)

        # Coefficient of variation of current window ddG values
        if current_window:
            mean_val = sum(current_window) / len(current_window)
            if mean_val != 0:
                variance = sum((x - mean_val) ** 2 for x in current_window) / len(current_window)
                cv = math.sqrt(variance) / abs(mean_val)
            else:
                cv = 0.0
        else:
            cv = 0.0
        details["cv"] = round(cv, 6)

        no_significant_improvement = p_value > self.significance_level
        low_variation = cv < 0.15

        converged = no_significant_improvement and low_variation
        if converged:
            details["recommendation"] = (
                f"Converged: p={p_value:.4f} (>{self.significance_level}), "
                f"CV={cv:.4f} (<0.15). Consider stopping."
            )
        else:
            reasons = []
            if not no_significant_improvement:
                reasons.append(f"p={p_value:.4f} (still improving)")
            if not low_variation:
                reasons.append(f"CV={cv:.4f} (high variance)")
            details["recommendation"] = f"Not converged: {'; '.join(reasons)}. Keep running."

        return converged, details
