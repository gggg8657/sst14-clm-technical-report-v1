"""test_planner_diversity.py — L1+L2 Planner 다양화 회귀 테스트

검증 항목:
  L1-a  _PLANNER_TEMPERATURE 상수가 0.3 초과임을 확인
  L1-a  _create_plan_via_llm이 temperature=_PLANNER_TEMPERATURE로 llm_generate_json 호출
  L1-b  plan_history가 있을 때 format_planner_prompt에 반복회피 섹션 등장
  L1-b  plan_history가 없을 때(첫 iteration) 반복회피 섹션 미등장
  L1-b  _build_plan_history_for_prompt가 mutation_guidance에서 올바르게 추출
  L2    _PLANNER_SYSTEM_PYROSETTA_ONLY에 pos11 탈피 지침이 포함됨
  L2    format_planner_prompt pyrosetta_only 모드에서도 반복회피 섹션 삽입
  기존  기존 테스트 무영향(import py_compile)
"""

from __future__ import annotations

import py_compile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from AG_src.agents.planner import (
    ExperimentPlan,
    PlannerAgent,
    _PLANNER_TEMPERATURE,
    _normalize_planner_mode,
)
from AG_src.llm.prompts import (
    _build_repetition_avoidance_section,
    format_planner_prompt,
    get_system_prompt,
)

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------

SST14_REF = "AGCKNFFWKTFTSC"
MUTABLE_POS: List[int] = [1, 2, 4, 5, 6, 11, 12, 13]


def _make_plan_history(n: int = 3) -> List[Dict[str, Any]]:
    """테스트용 plan_history 목록 생성 (pyrosetta_only 형식)."""
    entries = [
        {
            "iteration": 1,
            "focus_positions": [1, 2, 12],
            "strategy": "charge_optimization",
            "hypothesis": "pos1/2/12 charge 변경으로 SSTR3/5 회피",
        },
        {
            "iteration": 2,
            "focus_positions": [11],
            "strategy": "aromatic_enrichment",
            "hypothesis": "pos11 F→D 로 ECL2 상보 강화",
        },
        {
            "iteration": 3,
            "focus_positions": [1, 2, 12],
            "strategy": "charge_optimization",
            "hypothesis": "pos1/2/12 재시도 charge 변경",
        },
    ]
    return entries[:n]


# ---------------------------------------------------------------------------
# L1-a: temperature 상수 및 전달 검증
# ---------------------------------------------------------------------------

class TestPlannerTemperatureConstant(unittest.TestCase):
    """_PLANNER_TEMPERATURE 상수가 올바른 범위에 있는지 검사."""

    def test_planner_temperature_above_default(self) -> None:
        """_PLANNER_TEMPERATURE > 0.3 (DEFAULT_TEMPERATURE) 이어야 한다."""
        self.assertGreater(
            _PLANNER_TEMPERATURE,
            0.3,
            f"_PLANNER_TEMPERATURE={_PLANNER_TEMPERATURE} 이 DEFAULT_TEMPERATURE(0.3) 이하임",
        )

    def test_planner_temperature_in_valid_range(self) -> None:
        """_PLANNER_TEMPERATURE 는 0.0~1.0 범위여야 한다."""
        self.assertGreaterEqual(_PLANNER_TEMPERATURE, 0.0)
        self.assertLessEqual(_PLANNER_TEMPERATURE, 1.0)

    def test_planner_temperature_at_least_0_7(self) -> None:
        """다양성 확보를 위해 0.7 이상이어야 한다."""
        self.assertGreaterEqual(
            _PLANNER_TEMPERATURE,
            0.7,
            f"_PLANNER_TEMPERATURE={_PLANNER_TEMPERATURE} < 0.7",
        )


class TestPlannerLLMCallUsesTemperature(unittest.TestCase):
    """_create_plan_via_llm이 _PLANNER_TEMPERATURE를 llm_generate_json에 전달하는지 검사."""

    def _make_agent_with_mock_provider(self) -> PlannerAgent:
        agent = PlannerAgent(llm_provider="none", planner_mode="pyrosetta_only")
        # llm_provider를 mock LLMProvider 인스턴스로 교체
        mock_provider = MagicMock()
        mock_provider.provider_name = "MockProvider"
        mock_provider.generate_json.return_value = None  # LLM이 None 반환 → 규칙 폴백
        agent.llm_provider = mock_provider
        return agent

    def test_create_plan_calls_generate_json_with_planner_temperature(self) -> None:
        """_create_plan_via_llm이 temperature=_PLANNER_TEMPERATURE로 generate_json 호출."""
        agent = self._make_agent_with_mock_provider()
        # _create_plan_via_llm 직접 호출
        agent._create_plan_via_llm(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": SST14_REF},
        )
        # generate_json이 호출되었는지 확인
        self.assertTrue(
            agent.llm_provider.generate_json.called,
            "generate_json 이 호출되지 않음",
        )
        call_kwargs = agent.llm_provider.generate_json.call_args
        # keyword argument 'temperature'가 전달되었는지
        passed_temp = call_kwargs.kwargs.get("temperature")
        self.assertIsNotNone(
            passed_temp,
            f"temperature kwarg 가 generate_json 에 전달되지 않음. call_args={call_kwargs}",
        )
        self.assertAlmostEqual(
            passed_temp,
            _PLANNER_TEMPERATURE,
            places=4,
            msg=f"전달된 temperature={passed_temp} != _PLANNER_TEMPERATURE={_PLANNER_TEMPERATURE}",
        )

    def test_llm_generate_json_passes_temperature_to_provider(self) -> None:
        """base_agent.llm_generate_json이 temperature 인자를 provider에 전달."""
        agent = self._make_agent_with_mock_provider()
        agent.llm_generate_json("test prompt", temperature=0.75)
        self.assertTrue(agent.llm_provider.generate_json.called)
        call_kwargs = agent.llm_provider.generate_json.call_args
        self.assertEqual(call_kwargs.kwargs.get("temperature"), 0.75)

    def test_llm_generate_json_without_temperature_uses_default(self) -> None:
        """temperature=None이면 provider의 기본값을 사용 (kwargs에 temperature 없음)."""
        agent = self._make_agent_with_mock_provider()
        agent.llm_generate_json("test prompt")  # temperature 미전달
        call_kwargs = agent.llm_provider.generate_json.call_args
        # temperature가 없거나 None
        temp_in_call = call_kwargs.kwargs.get("temperature")
        self.assertIsNone(
            temp_in_call,
            f"temperature=None 시 provider 호출에 temperature 포함되지 않아야 함, got {temp_in_call}",
        )


# ---------------------------------------------------------------------------
# L1-b: plan_history 직렬화 및 반복회피 섹션 검증
# ---------------------------------------------------------------------------

class TestBuildRepetitionAvoidanceSection(unittest.TestCase):
    """_build_repetition_avoidance_section 함수 단위 테스트."""

    def test_returns_empty_when_no_history(self) -> None:
        """plan_history 가 None/빈 리스트이면 빈 문자열 반환."""
        self.assertEqual(_build_repetition_avoidance_section(None), "")
        self.assertEqual(_build_repetition_avoidance_section([]), "")

    def test_section_header_present(self) -> None:
        """plan_history 있을 때 섹션 헤더가 포함되어야 한다."""
        history = _make_plan_history(2)
        section = _build_repetition_avoidance_section(history)
        self.assertIn("이미 시도한 전략", section)
        self.assertIn("반복 금지", section)

    def test_positions_in_section(self) -> None:
        """각 이력의 focus_positions 가 섹션에 포함되어야 한다."""
        history = _make_plan_history(2)
        section = _build_repetition_avoidance_section(history)
        # iter 1: [1, 2, 12]
        self.assertIn("[1, 2, 12]", section)
        # iter 2: [11]
        self.assertIn("[11]", section)

    def test_strategy_in_section(self) -> None:
        """strategy 문자열이 섹션에 포함되어야 한다."""
        history = _make_plan_history(1)
        section = _build_repetition_avoidance_section(history)
        self.assertIn("charge_optimization", section)

    def test_n_recent_limits_output(self) -> None:
        """n_recent=2 이면 최근 2개 iter만 표시."""
        history = _make_plan_history(3)
        section = _build_repetition_avoidance_section(history, n_recent=2)
        # iter 1은 안 보여야 함 (최근 2개 = iter 2, 3)
        self.assertNotIn("iter 1:", section)
        self.assertIn("iter 2:", section)
        self.assertIn("iter 3:", section)

    def test_failure_instruction_present(self) -> None:
        """같은 위치 반복이 FAILURE임을 섹션에 명시해야 한다."""
        history = _make_plan_history(1)
        section = _build_repetition_avoidance_section(history)
        self.assertIn("FAILURE", section)


class TestFormatPlannerPromptWithHistory(unittest.TestCase):
    """format_planner_prompt에 plan_history 주입 시 반복회피 섹션 검증."""

    def _make_prompt(
        self,
        iteration: int = 3,
        plan_history: Optional[List[Dict[str, Any]]] = None,
        planner_mode: str = "pyrosetta_only",
    ) -> str:
        return format_planner_prompt(
            iteration=iteration,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": SST14_REF, "design_positions": MUTABLE_POS},
            planner_mode=planner_mode,
            plan_history=plan_history,
        )

    def test_repetition_section_present_when_history_given(self) -> None:
        """plan_history 있을 때 반복회피 섹션이 프롬프트에 등장해야 한다."""
        prompt = self._make_prompt(plan_history=_make_plan_history(2))
        self.assertIn("이미 시도한 전략", prompt)
        self.assertIn("반복 금지", prompt)

    def test_repetition_section_absent_when_no_history(self) -> None:
        """plan_history=None 이면 반복회피 섹션이 없어야 한다."""
        prompt = self._make_prompt(plan_history=None)
        self.assertNotIn("이미 시도한 전략", prompt)

    def test_history_positions_appear_in_prompt(self) -> None:
        """이미 시도한 focus_positions가 프롬프트에 포함되어야 한다."""
        history = _make_plan_history(2)
        prompt = self._make_prompt(plan_history=history)
        self.assertIn("[1, 2, 12]", prompt)

    def test_iter1_no_repetition_section_even_with_history(self) -> None:
        """iteration=1(첫 플랜)이고 plan_history 없으면 반복회피 섹션 없음."""
        prompt = self._make_prompt(iteration=1, plan_history=None)
        self.assertNotIn("이미 시도한 전략", prompt)

    def test_repetition_section_with_default_planner_mode(self) -> None:
        """default planner_mode 에서도 plan_history 있으면 섹션 삽입."""
        prompt = self._make_prompt(planner_mode="default", plan_history=_make_plan_history(1))
        self.assertIn("이미 시도한 전략", prompt)


class TestBuildPlanHistoryForPrompt(unittest.TestCase):
    """PlannerAgent._build_plan_history_for_prompt 메서드 검증."""

    def _make_agent_with_plans(self) -> PlannerAgent:
        agent = PlannerAgent(llm_provider="none", planner_mode="pyrosetta_only")
        # 가짜 ExperimentPlan 추가
        from AG_src.agents.planner import ExperimentPlan, StepConfig
        import copy
        for i, (fp, strategy) in enumerate([
            ([1, 2, 12], "charge_optimization"),
            ([11], "aromatic_enrichment"),
        ], start=1):
            params = {
                "n_backbone": 50, "k_seq": 8,
                "mutation_guidance": {"focus_positions": fp, "strategy": strategy, "suggested_mutations": {}, "n_guided": 0},
            }
            plan = ExperimentPlan(
                run_id=f"test_iter{i:02d}",
                iteration=i,
                parameters=params,
                steps_config=[],
                gates={"esmfold_plddt_min": 75, "docking_top_pct": 20, "rosetta_ddg_max": -5.0},
                hypothesis=f"iter {i} hypothesis",
            )
            agent._plans.append(plan)
        return agent

    def test_extracts_focus_positions(self) -> None:
        """mutation_guidance.focus_positions 가 올바르게 추출되어야 한다."""
        agent = self._make_agent_with_plans()
        history = agent._build_plan_history_for_prompt()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["focus_positions"], [1, 2, 12])
        self.assertEqual(history[1]["focus_positions"], [11])

    def test_extracts_strategy(self) -> None:
        """strategy 가 올바르게 추출되어야 한다."""
        agent = self._make_agent_with_plans()
        history = agent._build_plan_history_for_prompt()
        self.assertEqual(history[0]["strategy"], "charge_optimization")
        self.assertEqual(history[1]["strategy"], "aromatic_enrichment")

    def test_returns_empty_when_no_plans(self) -> None:
        """_plans 가 비어 있으면 빈 리스트 반환."""
        agent = PlannerAgent(llm_provider="none", planner_mode="pyrosetta_only")
        history = agent._build_plan_history_for_prompt()
        self.assertEqual(history, [])

    def test_iteration_numbers_correct(self) -> None:
        """iteration 번호가 올바르게 추출되어야 한다."""
        agent = self._make_agent_with_plans()
        history = agent._build_plan_history_for_prompt()
        self.assertEqual(history[0]["iteration"], 1)
        self.assertEqual(history[1]["iteration"], 2)


# ---------------------------------------------------------------------------
# L2: pos11 탈피 지침 검증
# ---------------------------------------------------------------------------

class TestPos11TangentGuidanceInSystemPrompt(unittest.TestCase):
    """_PLANNER_SYSTEM_PYROSETTA_ONLY에 pos11 탈피 + ECL2/TM5 지침이 포함됨을 검사."""

    def setUp(self) -> None:
        self.system_prompt = get_system_prompt("planner", planner_mode="pyrosetta_only")

    def test_pos11_bias_warning_present(self) -> None:
        """pos11 편중 탈피 경고가 시스템 프롬프트에 있어야 한다."""
        self.assertIn("pos11", self.system_prompt)

    def test_ecl2_positions_mentioned(self) -> None:
        """ECL2 위치(192/193/195/197)가 시스템 프롬프트에 있어야 한다."""
        self.assertIn("ECL2", self.system_prompt)
        # 적어도 하나의 ECL2 residue 번호
        ecl2_residues = ["192", "193", "195", "197"]
        mentioned = any(r in self.system_prompt for r in ecl2_residues)
        self.assertTrue(mentioned, f"ECL2 residue 번호({ecl2_residues}) 중 하나도 없음")

    def test_tm5_positions_mentioned(self) -> None:
        """TM5 위치가 시스템 프롬프트에 있어야 한다."""
        self.assertIn("TM5", self.system_prompt)

    def test_multi_position_mutation_guidance(self) -> None:
        """2~3개 위치 동시 변이 탐색 지침이 포함되어야 한다."""
        self.assertIn("2~3", self.system_prompt)

    def test_diversity_requirement_section_present(self) -> None:
        """DIVERSITY REQUIREMENT 섹션이 시스템 프롬프트에 있어야 한다."""
        self.assertIn("DIVERSITY REQUIREMENT", self.system_prompt)

    def test_repetition_avoidance_failure_label(self) -> None:
        """같은 positions 반복이 FAILURE임을 시스템 프롬프트가 명시해야 한다."""
        self.assertIn("FAILURE", self.system_prompt)

    def test_default_planner_mode_not_affected(self) -> None:
        """default 모드 시스템 프롬프트는 변경되지 않아야 한다(pos11 탈피 없음)."""
        default_prompt = get_system_prompt("planner", planner_mode="default")
        # default 모드는 pos11 편중 탈피 지침이 없어야 함
        self.assertNotIn("pos11 편중 탈피", default_prompt)


# ---------------------------------------------------------------------------
# 기존 테스트 무영향 — import + py_compile 검증
# ---------------------------------------------------------------------------

class TestImportSafety(unittest.TestCase):
    """수정된 파일들이 import/compile 에러 없이 로드되는지 확인."""

    def test_planner_py_compiles(self) -> None:
        """AG_src/agents/planner.py py_compile 통과."""
        path = str(
            Path(__file__).parent.parent / "agents" / "planner.py"
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"planner.py 컴파일 실패: {e}")

    def test_prompts_py_compiles(self) -> None:
        """AG_src/llm/prompts.py py_compile 통과."""
        path = str(
            Path(__file__).parent.parent / "llm" / "prompts.py"
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"prompts.py 컴파일 실패: {e}")

    def test_base_agent_py_compiles(self) -> None:
        """AG_src/agents/base_agent.py py_compile 통과."""
        path = str(
            Path(__file__).parent.parent / "agents" / "base_agent.py"
        )
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            self.fail(f"base_agent.py 컴파일 실패: {e}")

    def test_planner_imports(self) -> None:
        """planner 모듈이 ImportError 없이 import 되어야 한다."""
        try:
            import importlib
            import AG_src.agents.planner as m
            importlib.reload(m)
        except ImportError as e:
            self.fail(f"planner import 실패: {e}")

    def test_prompts_imports(self) -> None:
        """prompts 모듈이 ImportError 없이 import 되어야 한다."""
        try:
            import importlib
            import AG_src.llm.prompts as m
            importlib.reload(m)
        except ImportError as e:
            self.fail(f"prompts import 실패: {e}")


if __name__ == "__main__":
    unittest.main()
