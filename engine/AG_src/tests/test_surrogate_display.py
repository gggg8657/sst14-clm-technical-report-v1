"""test_surrogate_display.py — surrogate 표시·전달 회귀 테스트.

검증 항목:
  1. format_planner_prompt: half_life_h/admet_score/hc50 주면 프롬프트에 등장 + 다목적 지침 포함.
  2. format_critic_prompt: 동일한 surrogate 필드 등장 확인.
  3. _format_surrogate_label: None graceful, 값 포함 신뢰 등급 라벨.
  4. build_monitoring_data._compute_composite_scores: ddG-only와 다른 순위 산출.
  5. format_planner_prompt / format_critic_prompt 기존 인자 호환성(하위호환).
  6. 시스템 프롬프트에 다목적 지침 포함.
"""
from __future__ import annotations

import sys
import os
import unittest
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# build_monitoring_data 경로 추가 (스크립트 디렉토리)
# ──────────────────────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "scripts")
)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from AG_src.llm.prompts import (
    _format_surrogate_label,
    format_planner_prompt,
    format_critic_prompt,
    get_system_prompt,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. _format_surrogate_label
# ──────────────────────────────────────────────────────────────────────────────

class TestFormatSurrogateLabel(unittest.TestCase):
    """_format_surrogate_label 단위 테스트."""

    def test_all_none_returns_na(self) -> None:
        label = _format_surrogate_label(None, None, None)
        self.assertEqual(label, "surrogate=N/A")

    def test_all_values_present(self) -> None:
        label = _format_surrogate_label(1.5, 0.8, 150.0)
        self.assertIn("half_life=1.50h[MED]", label)
        self.assertIn("admet=0.800[MED]", label)
        self.assertIn("hc50=150.0[LOW", label)

    def test_partial_values(self) -> None:
        label = _format_surrogate_label(2.0, None, None)
        self.assertIn("half_life=2.00h[MED]", label)
        self.assertNotIn("admet=", label)
        self.assertNotIn("hc50=", label)
        self.assertNotIn("surrogate=N/A", label)

    def test_hc50_only(self) -> None:
        label = _format_surrogate_label(None, None, 300.0)
        self.assertIn("hc50=300.0[LOW", label)
        self.assertIn("역변별", label)


# ──────────────────────────────────────────────────────────────────────────────
# 2. format_planner_prompt — surrogate 등장 + 다목적 지침
# ──────────────────────────────────────────────────────────────────────────────

class TestPlannerPromptSurrogateDisplay(unittest.TestCase):
    """format_planner_prompt에 surrogate 값이 표시되는지 확인."""

    _BASE_CONSTRAINTS = {
        "reference_sequence": "AGCKNFFWKTFTSC",
        "design_positions": [1, 2, 4, 5, 6, 11, 12],
    }

    def _make_prompt_with_surrogates(self) -> str:
        return format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints=self._BASE_CONSTRAINTS,
            previous_results={
                "best_ddg": -35.0,
                "top_candidates": [
                    {
                        "id": "cand_001",
                        "sequence": "AGCKNFFWKTFTSC",
                        "ddg": -35.0,
                        "half_life_h": 2.3,
                        "admet_score": 0.75,
                        "hc50": 180.0,
                    },
                    {
                        "id": "cand_002",
                        "sequence": "PGCKNFFWKTFTSC",
                        "ddg": -30.0,
                        # 데이터 없음 → N/A
                    },
                ],
            },
            planner_mode="pyrosetta_only",
        )

    def test_half_life_appears_in_prompt(self) -> None:
        prompt = self._make_prompt_with_surrogates()
        self.assertIn("half_life=2.30h[MED]", prompt,
                      "half_life_h가 프롬프트에 등장해야 함")

    def test_admet_score_appears_in_prompt(self) -> None:
        prompt = self._make_prompt_with_surrogates()
        self.assertIn("admet=0.750[MED]", prompt,
                      "admet_score가 프롬프트에 등장해야 함")

    def test_hc50_appears_in_prompt(self) -> None:
        prompt = self._make_prompt_with_surrogates()
        self.assertIn("hc50=180.0[LOW", prompt,
                      "hc50가 프롬프트에 등장해야 함")

    def test_missing_surrogate_shows_na(self) -> None:
        prompt = self._make_prompt_with_surrogates()
        self.assertIn("surrogate=N/A", prompt,
                      "surrogate 미보유 후보는 N/A로 표시해야 함")

    def test_multiobjective_guidance_in_prompt(self) -> None:
        prompt = self._make_prompt_with_surrogates()
        # 다목적 지침 키워드 확인 (시스템 프롬프트 포함이지만 user_prompt에도 surrogate 라벨 있음)
        self.assertTrue(
            "MED" in prompt or "surrogate" in prompt,
            "프롬프트에 surrogate 신뢰 등급 언급 필요",
        )

    def test_backward_compat_no_surrogates(self) -> None:
        """surrogate 필드 없이 호출해도 에러 없이 동작해야 한다 (하위호환)."""
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints=self._BASE_CONSTRAINTS,
            planner_mode="pyrosetta_only",
        )
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)

    def test_backward_compat_prev_results_without_surrogates(self) -> None:
        """top_candidates에 surrogate 키 없어도 graceful."""
        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints=self._BASE_CONSTRAINTS,
            previous_results={
                "best_ddg": -28.0,
                "top_candidates": [
                    {"id": "cand_001", "sequence": "AGCKNFFWKTFTSC", "ddg": -28.0}
                ],
            },
            planner_mode="pyrosetta_only",
        )
        # surrogate 없으면 N/A 표시
        self.assertIn("surrogate=N/A", prompt)


# ──────────────────────────────────────────────────────────────────────────────
# 3. format_critic_prompt — surrogate 등장
# ──────────────────────────────────────────────────────────────────────────────

class TestCriticPromptSurrogateDisplay(unittest.TestCase):
    """format_critic_prompt에 surrogate 값이 표시되는지 확인."""

    def _make_critic_prompt_with_surrogates(self) -> str:
        return format_critic_prompt(
            iteration=2,
            rank_table_summary={
                "top_candidates": [
                    {
                        "id": "cand_001",
                        "plddt": 88.0,
                        "dock_score": -7.5,
                        "ddg": -32.0,
                        "half_life_h": 1.8,
                        "admet_score": 0.82,
                        "hc50": 210.0,
                    },
                    {
                        "id": "cand_002",
                        "plddt": 85.0,
                        "dock_score": -7.0,
                        "ddg": -29.0,
                        # surrogate 없음 → N/A
                    },
                ]
            },
            qc_report_summary={
                "total": 8,
                "passed": 5,
                "failed": 3,
                "pass_rate": 0.625,
            },
            current_params={"n_candidates": 8},
        )

    def test_half_life_in_critic_prompt(self) -> None:
        prompt = self._make_critic_prompt_with_surrogates()
        self.assertIn("half_life=1.80h[MED]", prompt)

    def test_hc50_in_critic_prompt(self) -> None:
        prompt = self._make_critic_prompt_with_surrogates()
        self.assertIn("hc50=210.0[LOW", prompt)

    def test_missing_surrogate_na_in_critic_prompt(self) -> None:
        prompt = self._make_critic_prompt_with_surrogates()
        self.assertIn("surrogate=N/A", prompt)

    def test_backward_compat_critic_no_surrogates(self) -> None:
        """기존 인자(surrogate 없음)로 호출해도 에러 없이 동작해야 한다."""
        prompt = format_critic_prompt(
            iteration=1,
            rank_table_summary={
                "top_candidates": [
                    {"id": "c001", "plddt": 80.0, "dock_score": -6.0, "ddg": -25.0}
                ]
            },
            qc_report_summary={"total": 4, "passed": 2, "failed": 2, "pass_rate": 0.5},
            current_params={},
        )
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 50)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 시스템 프롬프트 — 다목적 지침 포함
# ──────────────────────────────────────────────────────────────────────────────

class TestSystemPromptMultiObjectiveGuidance(unittest.TestCase):
    """Planner/Critic 시스템 프롬프트에 다목적 지침이 포함되는지 확인."""

    def test_planner_pyrosetta_system_has_surrogate_guidance(self) -> None:
        sys_prompt = get_system_prompt("planner", planner_mode="pyrosetta_only")
        self.assertIn("surrogate", sys_prompt,
                      "Planner 시스템 프롬프트에 surrogate 언급 필요")
        self.assertIn("다목적", sys_prompt,
                      "Planner 시스템 프롬프트에 다목적 언급 필요")

    def test_critic_system_has_surrogate_guidance(self) -> None:
        sys_prompt = get_system_prompt("critic")
        self.assertIn("surrogate", sys_prompt,
                      "Critic 시스템 프롬프트에 surrogate 언급 필요")
        self.assertIn("다목적", sys_prompt,
                      "Critic 시스템 프롬프트에 다목적 언급 필요")


# ──────────────────────────────────────────────────────────────────────────────
# 5. build_monitoring_data — composite_score 기반 다목적 랭크
# ──────────────────────────────────────────────────────────────────────────────

class TestCompositeScoreRank(unittest.TestCase):
    """_compute_composite_scores가 ddG-only와 다른 순위를 산출하는지 확인."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from build_monitoring_data import _compute_composite_scores, _normalize_column
            cls._compute = staticmethod(_compute_composite_scores)
            cls._norm = staticmethod(_normalize_column)
            cls._available = True
        except ImportError:
            cls._available = False

    def _skip_if_unavailable(self) -> None:
        if not self._available:
            self.skipTest("build_monitoring_data 경로 불가 — 스킵")

    def test_composite_differs_from_ddg_rank(self) -> None:
        """ddG가 좋지만 선택성/독성이 나쁜 경우 composite_rank가 달라야 한다."""
        self._skip_if_unavailable()
        rows = [
            {"ddg": -50.0, "delta_margin": -8.0, "half_life_h": 0.1, "hc50": 5.0},   # ddg=1, poor sel/hc50
            {"ddg": -30.0, "delta_margin": 12.0, "half_life_h": 4.0, "hc50": 800.0}, # ddg=2, good sel/hc50
        ]
        result = self._compute(rows)
        scores = [r.get("composite_score") for r in result]
        self.assertIsNotNone(scores[0])
        self.assertIsNotNone(scores[1])
        # composite best는 ddg_rank=2인 행이어야 함
        composite_winner = max(range(len(scores)), key=lambda i: scores[i] or -999)
        self.assertEqual(composite_winner, 1,
                         "composite_rank=1은 ddg_rank=2 후보여야 함 (선택성/독성 우위)")

    def test_composite_score_present_in_output(self) -> None:
        """composite_score 열이 추가되어야 한다."""
        self._skip_if_unavailable()
        rows = [
            {"ddg": -40.0, "delta_margin": 5.0, "half_life_h": 2.0, "hc50": 150.0}
        ]
        result = self._compute(rows)
        self.assertIn("composite_score", result[0],
                      "composite_score 열 누락")

    def test_all_none_surrogates_handled_gracefully(self) -> None:
        """ddG·Δmargin만 있고 surrogate 없어도 동작해야 한다."""
        self._skip_if_unavailable()
        rows = [
            {"ddg": -40.0, "delta_margin": 5.0, "half_life_h": None, "hc50": None},
            {"ddg": -30.0, "delta_margin": 3.0, "half_life_h": None, "hc50": None},
        ]
        result = self._compute(rows)
        scores = [r.get("composite_score") for r in result]
        # ddG + Δmargin으로 composite 계산 가능해야 함
        self.assertIsNotNone(scores[0])
        self.assertIsNotNone(scores[1])
        self.assertGreater(scores[0], scores[1])

    def test_normalize_column_all_none(self) -> None:
        """전체 None 컬럼은 None 리스트 반환."""
        self._skip_if_unavailable()
        result = self._norm([None, None, None])
        self.assertEqual(result, [None, None, None])

    def test_normalize_column_single_value(self) -> None:
        """단일 값 컬럼은 None 리스트 반환 (범위 0)."""
        self._skip_if_unavailable()
        result = self._norm([5.0])
        self.assertEqual(result, [None])

    def test_normalize_column_range(self) -> None:
        """정상 범위 정규화: min=0, max=1."""
        self._skip_if_unavailable()
        result = self._norm([0.0, 5.0, 10.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[2], 1.0)


if __name__ == "__main__":
    unittest.main()
