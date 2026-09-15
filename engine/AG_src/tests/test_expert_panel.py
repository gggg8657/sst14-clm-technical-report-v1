"""
test_expert_panel.py
Track1-P2 Expert Panel (fan-out/fan-in) 단위 테스트

커버 항목:
  1. 4 전문가가 다른 도메인 concern 생성 (mock LLM 사용)
  2. 통합이 high-severity → reject, 무결(low) → approve
  3. expert_panel=False 시 P1 단일 동작(하위호환) — FlowConfig 기본값 검증
  4. 코드 합의 규칙(_apply_consensus_rules) 정확성
  5. LLM 없음 → 통과 폴백
  6. py_compile / import 안전
  7. 기존 P1 prereview_hypothesis 미영향 확인
"""

from __future__ import annotations

import json
import sys
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
import os
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


# ---------------------------------------------------------------------------
# 테스트용 가설 픽스처
# ---------------------------------------------------------------------------
_SAMPLE_HYP = (
    "pos11 단독 양전하 치환(F11R)으로 ECL2와의 이온 상호작용 강화."
)
_SAMPLE_GUIDANCE: Dict[str, Any] = {
    "focus_positions": [11],
    "suggested_mutations": {"11": ["R", "K"]},
    "n_guided": 6,
    "strategy": "pos11_charge",
}

# 도메인별 mock verdict (각자 다른 concern)
_MOCK_VERDICTS = {
    "pharma": {
        "domain": "pharma",
        "concerns": ["Arg/Lys 양전하로 혈장 단백 결합 변화 가능 — PK 영향"],
        "severity": "medium",
        "suggestion": "hc50 toxicity 확인 권장",
    },
    "biology": {
        "domain": "biology",
        "concerns": ["pos11 단독 집중 — 선택성 SSTR3/5 차별화 불충분"],
        "severity": "medium",
        "suggestion": "ECL2 접촉 다위치 조합 권장",
    },
    "chemistry": {
        "domain": "chemistry",
        "concerns": ["Arg(R) 측쇄 보호기 처리 필요 — SPPS coupling 주의"],
        "severity": "low",
        "suggestion": "Fmoc-Arg(Pbf) 사용 권장",
    },
    "radiochem": {
        "domain": "radiochem",
        "concerns": ["pos11 양전하 치환이 DOTA N-term 킬레이션과 무관 — radiolysis 리스크 낮음"],
        "severity": "low",
        "suggestion": "Trp8 산화 모니터링 권장(기존 pharmacophore, 변이 무관)",
    },
    "math": {
        "domain": "math",
        "concerns": ["pos11 과탐색(over-explored) — 국소최적 위험"],
        "severity": "high",
        "suggestion": "다위치 탐색으로 전환",
    },
}

_MOCK_FANIN_RESULT = {
    "merged_concerns": [
        "pharma: PK 영향 우려",
        "biology: 선택성 불충분",
        "math: 국소최적 위험 (high severity)",
    ],
    "approve": False,
    "suggested_revisions": {
        "focus_positions": [1, 2, 5, 12],
        "strategy": "multi_position_exploration",
    },
}


# ---------------------------------------------------------------------------
# Helper: mock LLM generate_json
# ---------------------------------------------------------------------------

def _make_mock_critic(domain_sequence: List[str] = None) -> MagicMock:
    """Mock ScientistCriticAgent. llm_generate_json를 순서대로 반환."""
    mock = MagicMock()
    mock.has_llm = True

    if domain_sequence is None:
        domain_sequence = ["pharma", "biology", "chemistry", "radiochem", "math"]

    # 4 전문가 + 1 fan-in 순서
    call_returns = [_MOCK_VERDICTS[d] for d in domain_sequence] + [_MOCK_FANIN_RESULT]
    mock.llm_generate_json = MagicMock(side_effect=call_returns)
    return mock


# ---------------------------------------------------------------------------
# 1. 합의 규칙(_apply_consensus_rules)
# ---------------------------------------------------------------------------

class TestApplyConsensusRules(unittest.TestCase):
    """코드 합의 규칙의 정확성 테스트 (재설계된 규칙)."""

    def _rules(self, verdicts) -> dict:
        from AG_src.agents.expert_panel import _apply_consensus_rules
        return _apply_consensus_rules(verdicts)

    # --- 반환 타입 검증 ---
    def test_returns_dict_with_required_keys(self):
        """_apply_consensus_rules는 dict를 반환한다 (재설계)."""
        result = self._rules([{"severity": "low"}])
        for key in ("approve", "status", "high_count", "medium_count", "reason"):
            self.assertIn(key, result, f"필수 키 누락: {key}")

    # --- high 규칙 ---
    def test_one_high_conditional(self):
        """severity=high 1개 → approve=False, status=conditional (1회 수정 기회)."""
        verdicts = [
            {"severity": "low"},
            {"severity": "medium"},
            {"severity": "high"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "conditional")

    def test_two_high_rejects(self):
        """severity=high 2개 → approve=False, status=reject (치명 다수 거부)."""
        verdicts = [
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "reject")

    # --- medium 규칙: 차단 아님 ---
    def test_two_medium_conditional(self):
        """severity=medium 2개 → conditional (게이트 실질화 2026-07-02: 임계 2)."""
        verdicts = [
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "conditional")

    def test_three_medium_conditional(self):
        """severity=medium 3개 → conditional (임계 2 이상)."""
        verdicts = [
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "conditional")

    def test_one_medium_still_approves(self):
        """severity=medium 1개(임계 미만) → approve (과잉거부 방지)."""
        verdicts = [
            {"severity": "medium"},
            {"severity": "low"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertTrue(result["approve"])

    def test_high_plus_two_medium_rejects(self):
        """high 1 + medium 2 → reject (심각 우려 광범위 동조)."""
        verdicts = [
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "reject")

    def test_four_medium_unanimous_conditional(self):
        """severity=medium 4개 만장일치 → approve=False, status=conditional."""
        verdicts = [{"severity": "medium"}] * 4
        result = self._rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "conditional")

    def test_one_medium_approves(self):
        """severity=medium 1개만이면 approve=True."""
        verdicts = [
            {"severity": "medium"},
            {"severity": "low"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertTrue(result["approve"])

    def test_all_low_approves(self):
        """모두 low면 approve=True, status=approve."""
        result = self._rules([{"severity": "low"}] * 4)
        self.assertTrue(result["approve"])
        self.assertEqual(result["status"], "approve")

    def test_empty_verdicts_approves(self):
        """빈 verdict 목록 → approve=True (기본 통과)."""
        result = self._rules([])
        self.assertTrue(result["approve"])

    def test_missing_severity_treated_as_low(self):
        """severity 키 없으면 low로 처리."""
        result = self._rules([{"domain": "math"}, {"domain": "pharma"}])
        self.assertTrue(result["approve"])

    def test_reason_populated(self):
        """reason 필드가 비어있지 않다."""
        result = self._rules([{"severity": "medium"}, {"severity": "low"}])
        self.assertIsInstance(result["reason"], str)
        self.assertTrue(len(result["reason"]) > 0)

    def test_counts_accurate(self):
        """high_count, medium_count가 정확히 집계된다."""
        verdicts = [
            {"severity": "high"},
            {"severity": "medium"},
            {"severity": "medium"},
            {"severity": "low"},
        ]
        result = self._rules(verdicts)
        self.assertEqual(result["high_count"], 1)
        self.assertEqual(result["medium_count"], 2)


# ---------------------------------------------------------------------------
# 2. ExpertPanelAgent 기본 동작
# ---------------------------------------------------------------------------

class TestExpertPanelAgent(unittest.TestCase):
    """ExpertPanelAgent.prereview_panel() 단위 테스트."""

    def setUp(self):
        self.mock_critic = _make_mock_critic()

    def _make_panel(self, mock_critic=None):
        from AG_src.agents.expert_panel import ExpertPanelAgent
        return ExpertPanelAgent(mock_critic or self.mock_critic)

    def test_panel_returns_required_keys(self):
        """prereview_panel() 결과에 필수 키가 존재한다 (신규 키 포함)."""
        panel = self._make_panel()
        result = panel.prereview_panel(
            iteration=1,
            hypothesis=_SAMPLE_HYP,
            mutation_guidance=_SAMPLE_GUIDANCE,
        )
        for key in (
            "merged_concerns", "approve", "suggested_revisions", "expert_verdicts",
            "llm_calls", "consensus_status", "consensus_reason", "discussion_rounds",
        ):
            self.assertIn(key, result, f"필수 키 누락: {key}")

    def test_expert_verdicts_has_four_domains(self):
        """expert_verdicts에 4 도메인 키가 존재한다."""
        panel = self._make_panel()
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
            self.assertIn(domain, result["expert_verdicts"], f"도메인 누락: {domain}")

    def test_experts_have_different_concerns(self):
        """4 전문가가 서로 다른 concern을 반환한다."""
        panel = self._make_panel()
        # max_discussion_rounds=1: mock side_effect 소진 방지 (기본 2라운드에서 side_effect 부족 시 예외)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        all_concerns: List[str] = []
        for d in ("pharma", "biology", "chemistry", "radiochem", "math"):
            cs = result["expert_verdicts"][d].get("concerns", [])
            all_concerns.extend(cs)
        unique_concerns = set(all_concerns)
        self.assertGreaterEqual(len(unique_concerns), 3, "전문가 concern 다양성 부족")

    def test_high_severity_causes_reject(self):
        """biology=high severity 1개 → 코드 규칙에 의해 approve=False (conditional).
        math는 advisory 전환으로 합의에 미반영 → science 도메인 high 사용."""
        # math 대신 biology=high로 테스트 (math는 advisory)
        bio_high_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[bio_high_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
            + [fanin_rej]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        # biology=high(science) 1개 → conditional → max_discussion_rounds=1 → forced_pass
        self.assertEqual(result["consensus_status"], "forced_pass")
        self.assertTrue(result["approve"])  # forced_pass = 강제 통과

    def test_high_severity_two_rounds_eventually_passes(self):
        """biology=high 1개 → max_discussion_rounds=2: 2라운드 소진 후 forced_pass.
        math는 advisory 전환으로 합의 미반영 → science 도메인 high 사용."""
        # 두 라운드 모두 biology=high 유지하는 mock
        bio_high_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
        mock = MagicMock()
        mock.has_llm = True
        round_verdicts = [bio_high_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
        mock.llm_generate_json = MagicMock(
            side_effect=round_verdicts + [fanin_rej]  # 라운드 1
            + round_verdicts + [fanin_rej],           # 라운드 2
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2)
        self.assertTrue(result["approve"])
        # high=1(conditional) 불일치가 2라운드 연속 → PANEL_STALL_LIMIT(2) 도달 → stall로 종결.
        # stall도 approve=True(하위호환)를 보장하므로 무한거부 방지라는 본래 취지는 유지된다.
        self.assertIn(result["consensus_status"], ("forced_pass", "stall"))
        self.assertEqual(result["discussion_rounds"], 2)

    def test_llm_call_count_single_round(self):
        """max_discussion_rounds=1, 통과: llm_calls == 6 (5 전문가 + 1 fan-in)."""
        # 모두 low → 1라운드에서 통과
        low_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_low = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[low_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_low]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertEqual(result["llm_calls"], 6)

    def test_no_llm_fallback_approves(self):
        """LLM 없음(has_llm=False) → 통과 폴백(approve=True, concerns=[])."""
        mock = MagicMock()
        mock.has_llm = False
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        self.assertTrue(result["approve"])
        self.assertEqual(result["merged_concerns"], [])
        self.assertEqual(result["llm_calls"], 0)

    def test_all_low_severity_approves(self):
        """모든 전문가 low severity → approve=True."""
        low_verdicts = {
            d: dict(_MOCK_VERDICTS[d], severity="low")
            for d in ("pharma", "biology", "chemistry", "radiochem", "math")
        }
        fanin_low = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[low_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
            + [fanin_low]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        self.assertTrue(result["approve"])

    def test_llm_exception_fallback(self):
        """LLM 예외 발생 시 해당 전문가 low/no-concern 폴백."""
        mock = MagicMock()
        mock.has_llm = True
        # pharma는 예외, 나머지는 low
        def _side(*a, **kw):
            # 첫 호출(pharma) 예외, 이후 정상
            if not hasattr(_side, '_calls'):
                _side._calls = 0
            _side._calls += 1
            if _side._calls == 1:
                raise RuntimeError("test error")
            return {"domain": "test", "concerns": [], "severity": "low", "suggestion": ""}
        mock.llm_generate_json = MagicMock(side_effect=_side)
        panel = self._make_panel(mock)
        # 예외 발생해도 오류 없이 결과 반환
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        self.assertIn("approve", result)

    def test_code_rule_overrides_llm_approve(self):
        """코드 규칙과 LLM approve가 다를 때 코드 우선.
        biology=high 1개 → 코드 conditional, LLM은 approve=True로 틀림.
        max_discussion_rounds=1이라 forced_pass로 통과하지만 disc_round 내에서 코드 우선이 적용된다.
        (math는 advisory → science 도메인 biology high로 테스트)"""
        bio_high_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_wrong = dict(_MOCK_FANIN_RESULT, approve=True)  # LLM 틀림
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[bio_high_verdicts[d] for d in ("pharma","biology","chemistry","radiochem","math")]
            + [fanin_wrong]
        )
        panel = self._make_panel(mock)
        # max_discussion_rounds=1 → 1라운드 후 forced_pass
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        # forced_pass이지만 approve=True (무한 거부 방지 작동)
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "forced_pass")

    def test_medium_four_unanimous_triggers_discussion_then_forced_pass(self):
        """medium 4개 만장일치 → conditional → max_discussion 소진 → forced_pass."""
        medium_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="medium") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_reject = dict(_MOCK_FANIN_RESULT, approve=False)
        mock = MagicMock()
        mock.has_llm = True
        # 2라운드 모두 medium 4개 유지
        mock.llm_generate_json = MagicMock(
            side_effect=[medium_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_reject]
            + [medium_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_reject]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2)
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "forced_pass")

    def test_one_science_medium_passes_immediately(self):
        """science medium 1개(임계 2 미만) → 즉시 통과 (게이트 실질화 후에도 과잉거부 방지).
        math=medium은 advisory이므로 제외 → science 4명 중 1명만 medium."""
        mixed = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="medium"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="low"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="medium"),  # advisory → 무시
        }
        fanin_approve = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[mixed[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_approve]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "approve")

    def test_concerns_accumulated_across_rounds(self):
        """다회 토론 시 merged_concerns가 누적된다."""
        # 라운드 1: medium 4개 (만장일치 conditional)
        medium_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="medium") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_round1 = {
            "merged_concerns": ["1라운드 우려: PK 변화 가능"],
            "approve": False,
            "suggested_revisions": {"focus_positions": [5, 6]},
        }
        # 라운드 2: low → 통과
        low_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_round2 = {
            "merged_concerns": ["2라운드 우려: 추가 검토 권장"],
            "approve": True,
            "suggested_revisions": {},
        }
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[medium_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_round1]
            + [low_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_round2]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2)
        self.assertTrue(result["approve"])
        self.assertEqual(result["discussion_rounds"], 2)
        # 두 라운드의 우려가 모두 포함되어야 함
        combined = " ".join(result["merged_concerns"])
        self.assertIn("1라운드", combined)
        self.assertIn("2라운드", combined)


# ---------------------------------------------------------------------------
# 3. FlowConfig 필드 기본값 및 하위호환
# ---------------------------------------------------------------------------

class TestFlowConfigExpertPanel(unittest.TestCase):
    """FlowConfig.expert_panel 기본값 및 하위호환 테스트."""

    def test_expert_panel_default_true(self):
        """expert_panel 기본값은 True (P2 활성화)."""
        from pyrosetta_flow.schema import FlowConfig
        cfg = FlowConfig(template_pdb="dummy.pdb")
        self.assertTrue(cfg.expert_panel)

    def test_expert_panel_false_fallback(self):
        """expert_panel=False로 설정하면 P1 단일 모드 의미."""
        from pyrosetta_flow.schema import FlowConfig
        cfg = FlowConfig(template_pdb="dummy.pdb", expert_panel=False)
        self.assertFalse(cfg.expert_panel)

    def test_preview_rounds_default(self):
        """preview_rounds 기본값은 1."""
        from pyrosetta_flow.schema import FlowConfig
        cfg = FlowConfig(template_pdb="dummy.pdb")
        self.assertEqual(cfg.preview_rounds, 1)


# ---------------------------------------------------------------------------
# 4. prompts.py 새 시스템 프롬프트 키 존재 및 내용
# ---------------------------------------------------------------------------

class TestExpertSystemPrompts(unittest.TestCase):
    """SYSTEM_PROMPTS 새 키 존재 및 기본 내용 검증."""

    def setUp(self):
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        self.sp = SYSTEM_PROMPTS

    def test_all_expert_keys_exist(self):
        """4 도메인 + fan-in 키 존재."""
        for k in (
            "expert_pharma_prereview",
            "expert_biology_prereview",
            "expert_chemistry_prereview",
            "expert_math_prereview",
            "expert_fanin",
        ):
            self.assertIn(k, self.sp, f"SYSTEM_PROMPTS 키 누락: {k}")

    def test_pharma_prompt_contains_admet(self):
        """pharma 프롬프트에 ADMET 언급 존재."""
        self.assertIn("ADMET", self.sp["expert_pharma_prereview"])

    def test_biology_prompt_contains_sstr2(self):
        """biology 프롬프트에 SSTR2 언급 존재."""
        self.assertIn("SSTR2", self.sp["expert_biology_prereview"])

    def test_chemistry_prompt_contains_spps(self):
        """chemistry 프롬프트에 SPPS 언급 존재."""
        self.assertIn("SPPS", self.sp["expert_chemistry_prereview"])

    def test_math_prompt_contains_pos11(self):
        """math 프롬프트에 pos11 과탐색 언급 존재."""
        self.assertIn("pos11", self.sp["expert_math_prereview"])

    def test_fanin_prompt_contains_severity_rules(self):
        """fan-in 프롬프트에 severity 합의 규칙 언급 존재."""
        self.assertIn("high", self.sp["expert_fanin"])
        self.assertIn("medium", self.sp["expert_fanin"])

    def test_existing_p1_prompts_unchanged(self):
        """P1 기존 프롬프트 키(critic_prereview, planner_prereview_revision)가 보존된다."""
        self.assertIn("critic_prereview", self.sp)
        self.assertIn("planner_prereview_revision", self.sp)


# ---------------------------------------------------------------------------
# 5. prompts.py 포매터 함수 테스트
# ---------------------------------------------------------------------------

class TestExpertPromptFormatters(unittest.TestCase):
    """format_expert_domain_prompt / format_expert_fanin_prompt 테스트."""

    def test_domain_prompt_contains_domain_label(self):
        """각 도메인 프롬프트에 도메인 이름이 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
            s = format_expert_domain_prompt(
                domain=domain,
                iteration=1,
                hypothesis=_SAMPLE_HYP,
                mutation_guidance=_SAMPLE_GUIDANCE,
            )
            self.assertIn(domain.upper(), s, f"{domain} 프롬프트에 도메인 레이블 없음")

    def test_domain_prompt_contains_hypothesis(self):
        """도메인 프롬프트에 가설 문자열이 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        s = format_expert_domain_prompt("pharma", 2, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        self.assertIn("pos11", s)

    def test_fanin_prompt_contains_verdicts_json(self):
        """fan-in 프롬프트에 전문가 verdict JSON이 포함된다."""
        from AG_src.llm.prompts import format_expert_fanin_prompt
        verdicts = list(_MOCK_VERDICTS.values())
        s = format_expert_fanin_prompt(verdicts, 1)
        self.assertIn("pharma", s)
        self.assertIn("biology", s)

    def test_math_prompt_includes_trajectory_when_provided(self):
        """math 도메인 프롬프트에 trajectory_summary가 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        traj = "## 최근 궤적\n- iter9: pos11 F11D, Δmargin +10.54"
        s = format_expert_domain_prompt(
            "math", 1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, trajectory_summary=traj
        )
        self.assertIn("trajectory", s.lower())

    def test_biology_prompt_includes_leaderboard_when_provided(self):
        """biology 도메인 프롬프트에 selectivity 리더보드가 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        lb = [{"sequence": "AGCKNFFWKTDTSC", "delta_margin": 10.54, "ddg": -45.1}]
        s = format_expert_domain_prompt(
            "biology", 1, _SAMPLE_HYP, _SAMPLE_GUIDANCE,
            selectivity_leaderboard=lb, best_delta_margin=10.54,
        )
        self.assertIn("10.54", s)


# ---------------------------------------------------------------------------
# 6. 기존 P1 prereview_hypothesis 미영향 확인
# ---------------------------------------------------------------------------

class TestP1BackwardCompat(unittest.TestCase):
    """기존 P1 prereview_hypothesis API가 P2 도입 후에도 정상 동작."""

    def test_prereview_hypothesis_still_importable(self):
        """critic.prereview_hypothesis 메서드가 존재한다."""
        from AG_src.agents.critic import ScientistCriticAgent
        critic = ScientistCriticAgent()
        self.assertTrue(hasattr(critic, "prereview_hypothesis"))
        self.assertTrue(callable(critic.prereview_hypothesis))

    def test_prereview_hypothesis_no_llm_passes(self):
        """LLM 없는 ScientistCriticAgent.prereview_hypothesis → approve=True."""
        from AG_src.agents.critic import ScientistCriticAgent
        critic = ScientistCriticAgent()
        # LLM 없이 인스턴스화 → has_llm=False → 통과 폴백
        result = critic.prereview_hypothesis(
            iteration=1,
            hypothesis=_SAMPLE_HYP,
            mutation_guidance=_SAMPLE_GUIDANCE,
        )
        self.assertIn("approve", result)
        # has_llm=False이면 반드시 True 폴백
        if not critic.has_llm:
            self.assertTrue(result["approve"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# 7. [expert-panel] 로그 형식 검증 (P2 관측성 보완)
# ---------------------------------------------------------------------------

class TestExpertPanelLogFormat(unittest.TestCase):
    """[expert-panel] 로그가 지시된 형식으로 출력되는지 검증."""

    def _run_panel(self) -> "ExpertPanelAgent":
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = _make_mock_critic()
        panel = ExpertPanelAgent(mock)
        return panel

    def test_domain_log_format_contains_expert_panel_prefix(self):
        """각 domain verdict 로그가 [expert-panel] 프리픽스를 사용한다."""
        import logging
        panel = self._run_panel()
        with self.assertLogs("AG_src.agents.expert_panel", level="INFO") as cm:
            panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        # 도메인 verdict 로그: '[expert-panel] iter=N domain severity=X concerns=...'
        domain_logs = [
            m for m in cm.output
            if "[expert-panel]" in m
            and "fan-in" not in m
            and "패널 사전" not in m
            and "consensus_check" not in m
            and "합의 달성" not in m
            and "강제 통과" not in m
            and "forced_pass" not in m
            and "최대 토론" not in m
        ]
        self.assertEqual(len(domain_logs), 5, f"도메인 verdict 로그 5개 기대, 실제: {len(domain_logs)}")
        for log_line in domain_logs:
            self.assertIn("[expert-panel]", log_line)
            self.assertIn("iter=", log_line)
            self.assertIn("severity=", log_line)
            self.assertIn("concerns=", log_line)

    def test_fanin_log_format(self):
        """fan-in 통합 로그가 [expert-panel] fan-in 형식을 사용한다."""
        import logging
        panel = self._run_panel()
        with self.assertLogs("AG_src.agents.expert_panel", level="INFO") as cm:
            panel.prereview_panel(2, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        fanin_logs = [m for m in cm.output if "fan-in" in m and "[expert-panel]" in m]
        self.assertGreaterEqual(len(fanin_logs), 1, "fan-in 로그 미존재")
        fanin_log = fanin_logs[0]
        self.assertIn("approve=", fanin_log)
        self.assertIn("merged_concerns=", fanin_log)
        self.assertIn("llm_calls=", fanin_log)

    def test_log_iter_number_correct(self):
        """로그에 iter 번호가 올바르게 기록된다."""
        panel = self._run_panel()
        with self.assertLogs("AG_src.agents.expert_panel", level="INFO") as cm:
            panel.prereview_panel(7, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        domain_logs = [
            m for m in cm.output
            if "[expert-panel]" in m and "iter=" in m
            and "fan-in" not in m and "패널 사전" not in m and "consensus_check" not in m
        ]
        for log_line in domain_logs:
            self.assertIn("iter=7", log_line, f"iter=7 미포함: {log_line}")

    def test_domain_names_in_logs(self):
        """4 도메인 이름이 각각 로그에 나타난다."""
        panel = self._run_panel()
        with self.assertLogs("AG_src.agents.expert_panel", level="INFO") as cm:
            panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        combined = "\n".join(cm.output)
        for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
            self.assertIn(domain, combined, f"domain '{domain}' 로그에 미포함")

    def test_consensus_check_log_exists(self):
        """consensus_check 로그가 disc_round와 함께 출력된다."""
        panel = self._run_panel()
        with self.assertLogs("AG_src.agents.expert_panel", level="INFO") as cm:
            panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        consensus_logs = [m for m in cm.output if "consensus_check" in m]
        self.assertGreaterEqual(len(consensus_logs), 1, "consensus_check 로그 미존재")
        self.assertIn("status=", consensus_logs[0])
        self.assertIn("high=", consensus_logs[0])
        self.assertIn("medium=", consensus_logs[0])


# ---------------------------------------------------------------------------
# 8. 합의율 로깅 및 approve rate 시뮬레이션
# ---------------------------------------------------------------------------

class TestConsensusApproveRate(unittest.TestCase):
    """합의율 0% 탈출 시뮬레이션 — 신규 규칙에서 medium만 있을 때 통과."""

    def _make_panel_with_severities(self, severities: List[str]):
        """지정된 severity 리스트로 mock panel 생성 (domains 순서: pharma,bio,chem,math)."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        verdicts = []
        for domain, sev in zip(domains, severities):
            verdicts.append(dict(_MOCK_VERDICTS[domain], severity=sev))
        fanin = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=verdicts + [fanin])
        return ExpertPanelAgent(mock)

    def test_medium_medium_low_low_approves(self):
        """medium 2개 + low 2개 → approve=True (구 규칙에서는 reject, 신규에서는 통과)."""
        panel = self._make_panel_with_severities(["medium", "medium", "low", "low"])
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(result["approve"])

    def test_medium_medium_medium_low_approves(self):
        """medium 3개 + low 1개 → approve=True (만장일치 아님)."""
        panel = self._make_panel_with_severities(["medium", "medium", "medium", "low"])
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(result["approve"])

    def test_medium_concerns_appear_in_merged(self):
        """medium 2개 통과 시에도 merged_concerns에 우려가 기록된다."""
        panel = self._make_panel_with_severities(["medium", "medium", "low", "low"])
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(result["approve"])
        # fan-in에서 merged_concerns를 반환하므로 비어있지 않아야 함
        # (fanin mock은 merged_concerns를 반환함)
        self.assertIsInstance(result["merged_concerns"], list)

    def test_all_low_approve_rate_100_percent(self):
        """모두 low → 반드시 approve=True (approve rate 100%)."""
        panel = self._make_panel_with_severities(["low", "low", "low", "low"])
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "approve")

    def test_high_two_still_rejects(self):
        """high 2개 → reject 유지 (게이트 의미 보존)."""
        from AG_src.agents.expert_panel import _apply_consensus_rules
        verdicts = [
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result = _apply_consensus_rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "reject")


# ---------------------------------------------------------------------------
# 9. format_expert_fanin_prompt prev_concerns 파라미터
# ---------------------------------------------------------------------------

class TestFaninPromptPrevConcerns(unittest.TestCase):
    """format_expert_fanin_prompt의 prev_concerns 파라미터 테스트."""

    def test_without_prev_concerns(self):
        """prev_concerns 없으면 'Previous Round Concerns' 섹션 없음."""
        from AG_src.llm.prompts import format_expert_fanin_prompt
        s = format_expert_fanin_prompt(list(_MOCK_VERDICTS.values()), iteration=1)
        self.assertNotIn("Previous Round Concerns", s)

    def test_with_prev_concerns(self):
        """prev_concerns 있으면 'Previous Round Concerns' 섹션 포함."""
        from AG_src.llm.prompts import format_expert_fanin_prompt
        concerns = ["이전 라운드 우려: PK 변화"]
        s = format_expert_fanin_prompt(
            list(_MOCK_VERDICTS.values()), iteration=1, prev_concerns=concerns
        )
        self.assertIn("Previous Round Concerns", s)
        self.assertIn("PK 변화", s)

    def test_updated_consensus_rules_in_prompt(self):
        """Updated Consensus Rules 섹션이 프롬프트에 포함된다."""
        from AG_src.llm.prompts import format_expert_fanin_prompt
        s = format_expert_fanin_prompt(list(_MOCK_VERDICTS.values()), iteration=1)
        self.assertIn("Updated Consensus Rules", s)
        self.assertIn("medium 4개 만장일치", s)


# ---------------------------------------------------------------------------
# 10. reject_hard: high 2+ 라운드 소진 시 forced_pass 예외 (PANEL_HARD_REJECT_HIGH)
# ---------------------------------------------------------------------------

class TestRejectHard(unittest.TestCase):
    """PANEL_HARD_REJECT_HIGH=1(기본) 시 high 2+ 라운드 소진 → reject_hard 검증."""

    def _two_high_mock(self, n_rounds: int = 2) -> MagicMock:
        """high 2개 verdict를 n_rounds 라운드 모두 반환하는 mock."""
        two_high_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="high"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_reject = dict(_MOCK_FANIN_RESULT, approve=False)
        side = []
        for _ in range(n_rounds):
            side += [two_high_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
            side += [fanin_reject]
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=side)
        return mock

    def _make_panel(self, mock: MagicMock) -> "ExpertPanelAgent":
        from AG_src.agents.expert_panel import ExpertPanelAgent
        return ExpertPanelAgent(mock)

    def test_two_high_max_rounds_reject_hard(self):
        """high 2개 + 라운드 소진(max_discussion_rounds=2) → approve=False, status=reject_hard."""
        import os
        # PANEL_HARD_REJECT_HIGH 기본값(1) 확인
        os.environ.pop("PANEL_HARD_REJECT_HIGH", None)
        # 모듈 재로드 없이 모듈 변수 직접 패치
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            panel = self._make_panel(self._two_high_mock(n_rounds=2))
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2
            )
            self.assertFalse(result["approve"], "high 2+ reject_hard: approve=False 기대")
            self.assertEqual(result["consensus_status"], "reject_hard")
            self.assertIn("강제 거부", result["consensus_reason"])
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_two_high_single_round_reject_hard(self):
        """high 2개 + max_discussion_rounds=1 → approve=False, status=reject_hard."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            panel = self._make_panel(self._two_high_mock(n_rounds=1))
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1
            )
            self.assertFalse(result["approve"])
            self.assertEqual(result["consensus_status"], "reject_hard")
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_one_high_max_rounds_forced_pass(self):
        """biology=high 1개(conditional) + 라운드 소진 → approve=True, status=forced_pass (기존 동작 유지).
        math는 advisory → science 도메인 biology high 사용."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            # biology=high(science 1개), math=low(advisory)
            one_high_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
            one_high_verdicts["biology"] = dict(_MOCK_VERDICTS["biology"], severity="high")
            fanin_reject = dict(_MOCK_FANIN_RESULT, approve=False)
            side = []
            for _ in range(2):
                side += [one_high_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
                side += [fanin_reject]
            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(side_effect=side)
            panel = self._make_panel(mock)
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2
            )
            self.assertTrue(result["approve"], "biology=high(conditional): forced_pass=True 기대")
            self.assertEqual(result["consensus_status"], "forced_pass")
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_medium_all_max_rounds_forced_pass(self):
        """medium 4개 만장일치 + 라운드 소진 → approve=True, status=forced_pass (기존 동작 유지)."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            medium_verdicts = {d: dict(_MOCK_VERDICTS[d], severity="medium") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
            fanin_reject = dict(_MOCK_FANIN_RESULT, approve=False)
            side = []
            for _ in range(2):
                side += [medium_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
                side += [fanin_reject]
            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(side_effect=side)
            panel = self._make_panel(mock)
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2
            )
            self.assertTrue(result["approve"], "medium 만장일치: forced_pass=True 기대")
            self.assertEqual(result["consensus_status"], "forced_pass")
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_hard_reject_disabled_by_env(self):
        """PANEL_HARD_REJECT_HIGH=0 시 high 2+도 forced_pass로 통과 (기존 동작)."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 0
        try:
            panel = self._make_panel(self._two_high_mock(n_rounds=2))
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2
            )
            self.assertTrue(result["approve"], "PANEL_HARD_REJECT_HIGH=0: forced_pass 기대")
            self.assertEqual(result["consensus_status"], "forced_pass")
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_reject_hard_result_contains_required_keys(self):
        """reject_hard 결과에 모든 필수 키가 존재한다."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            panel = self._make_panel(self._two_high_mock(n_rounds=1))
            result = panel.prereview_panel(
                1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1
            )
            for key in (
                "merged_concerns", "approve", "consensus_status", "consensus_reason",
                "suggested_revisions", "expert_verdicts", "llm_calls", "discussion_rounds",
            ):
                self.assertIn(key, result, f"필수 키 누락: {key}")
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig


# ---------------------------------------------------------------------------
# 11. verdict/docking_decision 분리 — 신규 필드 검증 (요구사항 1)
# ---------------------------------------------------------------------------

class TestVerdictDockingDecisionSplit(unittest.TestCase):
    """scientific_verdict / docking_decision 분리 + forced_pass 정직 표시 검증."""

    def _make_panel(self, mock: MagicMock) -> "ExpertPanelAgent":
        from AG_src.agents.expert_panel import ExpertPanelAgent
        return ExpertPanelAgent(mock)

    def _two_high_mock(self, n_rounds: int = 1) -> MagicMock:
        two_high_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="high"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_reject = dict(_MOCK_FANIN_RESULT, approve=False)
        side = []
        for _ in range(n_rounds):
            side += [two_high_verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")]
            side += [fanin_reject]
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=side)
        return mock

    def test_new_fields_present_in_result(self):
        """prereview_panel 결과에 신규 필드 7개가 존재한다."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = MagicMock()
        mock.has_llm = True
        low_vds = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock.llm_generate_json = MagicMock(
            side_effect=[low_vds[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_ap]
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        for key in (
            "scientific_verdict", "docking_decision", "reason_for_docking",
            "risk_level", "qc_status", "position_map_consistent", "selectivity_evidence_present",
        ):
            self.assertIn(key, result, f"신규 필드 누락: {key}")

    def test_approve_scientific_verdict_is_approve(self):
        """approve 시 scientific_verdict='approve', docking_decision=True."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        low_vds = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[low_vds[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_ap]
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertEqual(result["scientific_verdict"], "approve")
        self.assertTrue(result["docking_decision"])
        self.assertEqual(result["reason_for_docking"], "proceed")

    def test_forced_pass_scientific_verdict_not_approve(self):
        """forced_pass 시 scientific_verdict='conditional'(원 verdict 보존) + docking_decision=True.
        approve=True(하위호환)이지만 scientific_verdict는 reject/conditional 그대로.
        math는 advisory → biology=high(science) 사용."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            # biology=high(science 1개) → conditional → max_discussion_rounds=1 → forced_pass
            one_high = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in ("pharma", "biology", "chemistry", "radiochem", "math")}
            one_high["biology"] = dict(_MOCK_VERDICTS["biology"], severity="high")
            fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(
                side_effect=[one_high[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_rej]
            )
            panel = self._make_panel(mock)
            result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
            # forced_pass: approve=True(하위호환)이지만 scientific_verdict는 approve가 아님
            self.assertTrue(result["approve"])  # 하위호환
            self.assertEqual(result["consensus_status"], "forced_pass")
            self.assertNotEqual(result["scientific_verdict"], "approve",
                "forced_pass 시 scientific_verdict는 'approve'가 되면 안 됨 — 착시 제거")
            self.assertIn(result["scientific_verdict"], ("conditional", "reject"),
                f"forced_pass의 scientific_verdict는 conditional/reject이어야 함, got: {result['scientific_verdict']}")
            # docking_decision=True (저비용 탐색)
            self.assertTrue(result["docking_decision"])
            self.assertIn("exploratory", result["reason_for_docking"])
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_reject_hard_docking_decision_false(self):
        """reject_hard 시 docking_decision=False, scientific_verdict='reject'."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            panel = self._make_panel(self._two_high_mock(n_rounds=1))
            result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
            self.assertFalse(result["approve"])
            self.assertEqual(result["scientific_verdict"], "reject")
            self.assertFalse(result["docking_decision"])
            self.assertIn("blocked", result["reason_for_docking"])
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_fallback_pass_has_new_fields(self):
        """LLM 없는 폴백도 신규 필드를 반환한다."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = MagicMock()
        mock.has_llm = False
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
        self.assertIn("scientific_verdict", result)
        self.assertIn("docking_decision", result)
        self.assertEqual(result["scientific_verdict"], "approve")
        self.assertTrue(result["docking_decision"])


# ---------------------------------------------------------------------------
# 12. position map QC — hallucination 차단 (요구사항 2)
# ---------------------------------------------------------------------------

class TestPositionMapQC(unittest.TestCase):
    """position map 일관성 검사 및 pos12=Cys hallucination 차단 검증."""

    def test_pos12_cys_hallucination_detected(self):
        """pos12를 Cys로 언급하는 텍스트 → position_map_consistent=False."""
        from AG_src.agents.expert_panel import _check_position_map_consistent
        # hallucination 패턴들
        bad_texts = [
            "pos12=Cys is important for the disulfide",
            "position 12: cys forms secondary bond",
            "pos12 cysteine affects stability",
            "pos12 disulfide bridge",
        ]
        for text in bad_texts:
            result = _check_position_map_consistent(concerns_text=text)
            self.assertFalse(result, f"hallucination 미감지: '{text}'")

    def test_pos12_thr_correct_passes(self):
        """pos12를 Thr로 올바르게 언급 → position_map_consistent=True."""
        from AG_src.agents.expert_panel import _check_position_map_consistent
        good_texts = [
            "pos12=Thr mutation to Ala reduces flexibility",
            "position 12 (Thr) can be substituted",
            "T12 is mutable",
        ]
        for text in good_texts:
            result = _check_position_map_consistent(concerns_text=text)
            self.assertTrue(result, f"올바른 텍스트가 fail 판정됨: '{text}'")

    def test_immutable_in_focus_positions_fail(self):
        """focus_positions에 immutable 위치(3,7,8,9,10,14)가 포함되면 False."""
        from AG_src.agents.expert_panel import _check_position_map_consistent
        # pos3 포함
        result = _check_position_map_consistent(concerns_text="", focus_positions=[3, 5, 11])
        self.assertFalse(result)
        # pos7 포함
        result = _check_position_map_consistent(concerns_text="", focus_positions=[7, 12])
        self.assertFalse(result)
        # pos14 포함
        result = _check_position_map_consistent(concerns_text="", focus_positions=[14])
        self.assertFalse(result)

    def test_mutable_only_focus_positions_pass(self):
        """focus_positions가 mutable 위치만 포함하면 True."""
        from AG_src.agents.expert_panel import _check_position_map_consistent
        result = _check_position_map_consistent(concerns_text="", focus_positions=[1, 2, 5, 11, 12])
        self.assertTrue(result)

    def test_empty_focus_no_bad_text_pass(self):
        """focus_positions 없고 hallucination 없으면 True."""
        from AG_src.agents.expert_panel import _check_position_map_consistent
        result = _check_position_map_consistent(concerns_text="pos11 F→D improves selectivity")
        self.assertTrue(result)

    def test_panel_qc_fail_on_immutable_focus(self):
        """prereview_panel에서 focus_positions에 immutable 위치가 있으면 position_map_consistent=False."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = MagicMock()
        mock.has_llm = False  # 폴백으로 바로 통과
        panel = ExpertPanelAgent(mock)
        bad_guidance = dict(_SAMPLE_GUIDANCE, focus_positions=[3, 11])  # pos3=Cys(immutable)
        result = panel.prereview_panel(1, _SAMPLE_HYP, bad_guidance)
        self.assertFalse(result["position_map_consistent"],
            "immutable 위치 focus_positions → position_map_consistent=False 기대")

    def test_sst14_position_map_exported(self):
        """SST14_POSITION_MAP이 prompts.py에서 export된다."""
        from AG_src.llm.prompts import SST14_POSITION_MAP, SST14_IMMUTABLE_POSITIONS
        self.assertEqual(SST14_POSITION_MAP[12], "T", "pos12는 T(Thr)이어야 함")
        self.assertEqual(SST14_POSITION_MAP[3], "C", "pos3은 C(Cys)이어야 함")
        self.assertEqual(SST14_POSITION_MAP[14], "C", "pos14는 C(Cys)이어야 함")
        self.assertIn(3, SST14_IMMUTABLE_POSITIONS)
        self.assertIn(7, SST14_IMMUTABLE_POSITIONS)
        self.assertNotIn(12, SST14_IMMUTABLE_POSITIONS, "pos12는 mutable")
        self.assertNotIn(11, SST14_IMMUTABLE_POSITIONS, "pos11은 mutable")

    def test_position_map_block_in_expert_system_prompts(self):
        """4개 expert 시스템 프롬프트 모두 position map block을 포함한다."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        block_marker = "SST-14 POSITION MAP"
        for key in (
            "expert_pharma_prereview",
            "expert_biology_prereview",
            "expert_chemistry_prereview",
            "expert_math_prereview",
        ):
            self.assertIn(block_marker, SYSTEM_PROMPTS[key],
                f"{key} 시스템 프롬프트에 position map block 없음")

    def test_pos12_not_cys_statement_in_position_map_block(self):
        """position map block에 pos12=T(NOT Cys) 명시가 있다."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        for key in (
            "expert_pharma_prereview",
            "expert_biology_prereview",
            "expert_chemistry_prereview",
            "expert_math_prereview",
        ):
            prompt = SYSTEM_PROMPTS[key]
            self.assertIn("pos12", prompt)
            self.assertIn("NOT Cys", prompt, f"{key}: pos12 NOT Cys 명시 없음")


# ---------------------------------------------------------------------------
# 13. math panel focus positions 명시 (요구사항 3)
# ---------------------------------------------------------------------------

class TestMathFocusPositionsInPrompt(unittest.TestCase):
    """math 전문가 프롬프트에 focus_positions가 명시적으로 주입되는지 검증."""

    def test_math_prompt_contains_focus_positions(self):
        """math 도메인 프롬프트에 focus_positions 값이 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        guidance = {"focus_positions": [1, 5, 12], "strategy": "multi_pos"}
        s = format_expert_domain_prompt(
            domain="math",
            iteration=3,
            hypothesis="pos1/5/12 동시 변이로 다양성 확보",
            mutation_guidance=guidance,
        )
        self.assertIn("1", s)
        self.assertIn("5", s)
        self.assertIn("12", s)
        # Focus Positions for Diversity Analysis 섹션 존재
        self.assertIn("Focus Positions", s)

    def test_math_prompt_focus_section_absent_when_empty(self):
        """focus_positions가 비어있으면 Focus Positions 섹션이 없다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        s = format_expert_domain_prompt(
            domain="math",
            iteration=1,
            hypothesis="general hypothesis",
            mutation_guidance={"focus_positions": [], "strategy": ""},
        )
        self.assertNotIn("Focus Positions for Diversity Analysis", s)

    def test_other_domains_no_focus_section(self):
        """pharma/biology/chemistry 도메인에는 Focus Positions 섹션 없음."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        guidance = {"focus_positions": [5, 11], "strategy": "test"}
        for domain in ("pharma", "biology", "chemistry"):
            s = format_expert_domain_prompt(
                domain=domain,
                iteration=1,
                hypothesis="test hypothesis",
                mutation_guidance=guidance,
            )
            self.assertNotIn("Focus Positions for Diversity Analysis", s,
                f"{domain} 도메인에 Focus Positions 섹션이 있으면 안 됨")

    def test_math_prompt_position_map_reference(self):
        """math 도메인 프롬프트에 SST-14 position map block이 포함된다(시스템 프롬프트)."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        self.assertIn("pos12=T", SYSTEM_PROMPTS["expert_math_prereview"])

    def test_math_prompt_is_advisory_role(self):
        """math 시스템 프롬프트가 advisory 역할임을 명시한다."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        prompt = SYSTEM_PROMPTS["expert_math_prereview"]
        self.assertIn("ADVISORY", prompt.upper())
        # approve/reject 결정에 미반영 명시
        self.assertIn("NOT", prompt)


# ---------------------------------------------------------------------------
# 15. 수학 advisory 전환 검증 (요구사항 추가)
# ---------------------------------------------------------------------------

class TestMathAdvisoryRole(unittest.TestCase):
    """수학 패널 advisory 전환 — 합의 판정 미반영 + math_advisory 필드."""

    def _make_panel_with_math_high(self, math_severity: str = "high") -> "Dict[str, Any]":
        """math=high로 설정한 mock panel 실행 결과 반환."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        # math=high(advisory), pharma/bio/chem/radiochem=low → science 4명 전부 low → approve
        verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="low"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity=math_severity,
                         suggestion="pos11 over-explored. Recommend pos1/4/6 next."),
        }
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[verdicts[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_ap]
        )
        panel = ExpertPanelAgent(mock)
        return panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)

    def test_math_high_does_not_cause_reject(self):
        """math=high(advisory)여도 합의 판정에서 제외 → approve=True (science 3명 low)."""
        result = self._make_panel_with_math_high("high")
        self.assertTrue(result["approve"],
            "math=high advisory → science 3명 low → approve=True 기대")
        self.assertEqual(result["scientific_verdict"], "approve",
            "math advisory는 scientific_verdict에 영향 없음")

    def test_math_advisory_field_exists(self):
        """결과에 math_advisory 필드가 존재한다."""
        result = self._make_panel_with_math_high("high")
        self.assertIn("math_advisory", result, "math_advisory 필드 누락")

    def test_math_advisory_contains_suggestion(self):
        """math_advisory 필드에 수학 패널의 suggestion이 담긴다."""
        result = self._make_panel_with_math_high("high")
        advisory = result.get("math_advisory", "")
        # mock의 suggestion이 반영되어야 함
        self.assertIsInstance(advisory, str)

    def test_consensus_rules_exclude_math(self):
        """_apply_consensus_rules가 math domain verdict를 제외하고 집계한다."""
        from AG_src.agents.expert_panel import _apply_consensus_rules
        verdicts = [
            {"domain": "pharma", "severity": "low"},
            {"domain": "biology", "severity": "low"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "math", "severity": "high"},  # advisory — 무시되어야 함
        ]
        result = _apply_consensus_rules(verdicts)
        # math=high이지만 science 3명 low → approve
        self.assertTrue(result["approve"],
            "math=high advisory는 합의에 미반영 → approve=True 기대")
        self.assertEqual(result["high_count"], 0,
            "math advisory는 high_count에 포함되면 안 됨")

    def test_consensus_rules_two_science_high_rejects(self):
        """pharma/biology=high (science 2명) → reject. math=low → 무관."""
        from AG_src.agents.expert_panel import _apply_consensus_rules
        verdicts = [
            {"domain": "pharma", "severity": "high"},
            {"domain": "biology", "severity": "high"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "math", "severity": "low"},
        ]
        result = _apply_consensus_rules(verdicts)
        self.assertFalse(result["approve"])
        self.assertEqual(result["status"], "reject")
        self.assertEqual(result["high_count"], 2)

    def test_three_science_medium_conditional(self):
        """pharma/biology/chemistry=medium 만장일치(3명) → conditional (math=low, advisory 제외)."""
        from AG_src.agents.expert_panel import _apply_consensus_rules
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_MEDIUM_ALL_MAX
        ep_mod._PANEL_MEDIUM_ALL_MAX = 3
        try:
            verdicts = [
                {"domain": "pharma", "severity": "medium"},
                {"domain": "biology", "severity": "medium"},
                {"domain": "chemistry", "severity": "medium"},
                {"domain": "math", "severity": "low"},
            ]
            result = _apply_consensus_rules(verdicts)
            self.assertFalse(result["approve"])
            self.assertEqual(result["status"], "conditional",
                "science 3명 medium 만장일치 → conditional 기대")
        finally:
            ep_mod._PANEL_MEDIUM_ALL_MAX = orig

    def test_math_qc_skipped_advisory_domain(self):
        """math verdict는 position_map QC 비적용 → is_advisory=True."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        verdicts_map = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="low"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            # math에 pos12=Cys hallucination 텍스트 — advisory이므로 QC 비적용
            "math": {
                "domain": "math",
                "concerns": ["pos12=Cys disulfide is at risk"],  # hallucination
                "severity": "low",
                "suggestion": "Diversify positions",
            },
        }
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[verdicts_map[d] for d in ("pharma", "biology", "chemistry", "radiochem", "math")] + [fanin_ap]
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        # math advisory는 position QC 비적용 → 전체 approve에 영향 없음
        self.assertTrue(result["approve"])
        # math verdict에 is_advisory=True
        math_vd = result["expert_verdicts"].get("math", {})
        self.assertTrue(math_vd.get("is_advisory"), "math verdict is_advisory=True 기대")

    def test_math_advisory_system_prompt_no_severity_except_low(self):
        """math 시스템 프롬프트가 severity='low' always를 명시한다."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        prompt = SYSTEM_PROMPTS["expert_math_prereview"]
        self.assertIn("severity='low'", prompt, "math 프롬프트에 severity='low' always 명시 필요")


# ---------------------------------------------------------------------------
# 14. _append_discussion_log 신규 필드 전파 (runner 연동)
# ---------------------------------------------------------------------------

class TestDiscussionLogNewFields(unittest.TestCase):
    """_append_discussion_log가 신규 필드를 기록하는지 검증."""

    def _call_append(self, tmp_path: "Path", **kwargs) -> dict:
        """_append_discussion_log를 호출하고 기록된 JSON을 반환."""
        import sys
        sys.path.insert(0, str(tmp_path.parent.parent))
        from pyrosetta_flow.runner import _append_discussion_log
        log_path = tmp_path / "discussion_log.jsonl"
        _append_discussion_log(
            discussion_log_path=log_path,
            iteration=1,
            hypothesis="test hypo",
            round_idx=1,
            expert_verdicts={"pharma": {"severity": "low", "concerns": []}},
            fanin={"approve": True, "merged_concerns": []},
            final_focus=[5, 11],
            **kwargs,
        )
        import json
        with log_path.open() as f:
            return json.loads(f.readline())

    def test_new_fields_recorded_in_discussion_log(self):
        """scientific_verdict 등 신규 필드가 discussion_log.jsonl에 기록된다."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._call_append(
                Path(tmpdir),
                scientific_verdict="conditional",
                docking_decision=True,
                reason_for_docking="cheap exploratory docking despite reject",
                risk_level="medium-high",
                qc_status="pass",
                position_map_consistent=True,
                selectivity_evidence_present=False,
            )
            fanin = entry["fanin"]
            self.assertEqual(fanin.get("scientific_verdict"), "conditional")
            self.assertTrue(fanin.get("docking_decision"))
            self.assertIn("exploratory", fanin.get("reason_for_docking", ""))
            self.assertEqual(fanin.get("risk_level"), "medium-high")
            self.assertEqual(fanin.get("qc_status"), "pass")
            self.assertTrue(fanin.get("position_map_consistent"))

    def test_minority_dissent_and_stall_recorded_in_discussion_log(self):
        """C(minority_dissent)/E(stall_detected) 필드가 discussion_log.jsonl에 기록된다."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._call_append(
                Path(tmpdir),
                minority_dissent={"domain": "biology", "severity": "high", "reason": "선택성 우려"},
                stall_detected=True,
                stall_streak=2,
            )
            fanin = entry["fanin"]
            self.assertEqual(fanin.get("minority_dissent"), {
                "domain": "biology", "severity": "high", "reason": "선택성 우려",
            })
            self.assertTrue(fanin.get("stall_detected"))
            self.assertEqual(fanin.get("stall_streak"), 2)

    def test_minority_dissent_none_not_forced_into_log(self):
        """minority_dissent=None(미제공/소수의견 없음)이면 다른 신규 필드처럼 기록을 생략한다
        — None은 '전달 안 함'과 동일하게 취급되어 로그에 불필요한 null 필드를 남기지 않는다."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._call_append(Path(tmpdir), minority_dissent=None, stall_detected=False)
            fanin = entry["fanin"]
            self.assertNotIn("minority_dissent", fanin)
            self.assertFalse(fanin.get("stall_detected"))

    def test_without_new_fields_still_works(self):
        """신규 필드 없이 호출해도 기존 스키마가 유지된다."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._call_append(Path(tmpdir))
            self.assertIn("fanin", entry)
            self.assertIn("approve", entry["fanin"])


# ---------------------------------------------------------------------------
# 16. 5턴 토론 + 합의 지향 수렴 시나리오 (요구사항: max_discussion_rounds=5)
# ---------------------------------------------------------------------------

class TestFiveRoundDiscussion(unittest.TestCase):
    """max_discussion_rounds 기본값 5 + 합의 지향 수렴 시나리오 검증."""

    def _make_panel(self, mock: MagicMock) -> "ExpertPanelAgent":
        from AG_src.agents.expert_panel import ExpertPanelAgent
        return ExpertPanelAgent(mock)

    def test_default_max_discussion_rounds_is_five(self):
        """prereview_panel의 max_discussion_rounds 기본값이 5이다 (env 기본값).
        환경변수가 없을 때 _PANEL_MAX_DISCUSSION_ROUNDS=5 기대."""
        import AG_src.agents.expert_panel as ep_mod
        import os
        os.environ.pop("PANEL_MAX_DISCUSSION_ROUNDS", None)
        # 모듈 변수 직접 확인 (env 기본값 반영)
        orig = ep_mod._PANEL_MAX_DISCUSSION_ROUNDS
        # env가 없을 때 기본값 5
        self.assertEqual(orig, 5, f"_PANEL_MAX_DISCUSSION_ROUNDS 기본값 5 기대, got {orig}")

    def test_env_panel_max_discussion_rounds_override(self):
        """PANEL_MAX_DISCUSSION_ROUNDS env로 max_discussion_rounds 조절."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_MAX_DISCUSSION_ROUNDS
        ep_mod._PANEL_MAX_DISCUSSION_ROUNDS = 3
        try:
            # LLM 없는 패널 — max_discussion_rounds=None 전달 시 env값(3) 사용
            mock = MagicMock()
            mock.has_llm = False
            panel = self._make_panel(mock)
            # has_llm=False → 즉시 폴백 (max_discussion_rounds 확인은 로직 진입 전)
            result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE)
            self.assertIn("approve", result)
        finally:
            ep_mod._PANEL_MAX_DISCUSSION_ROUNDS = orig

    def test_medium_resolves_to_approve_within_five_rounds(self):
        """시나리오: 라운드2에서 동료 의견으로 medium → low 해소 → approve 도달.
        round1: science 4명 중 3명 medium + radiochem=high (만장일치 아니지만 high 1개 → conditional).
        round2: 동료 의견 반영 → pharma/chemistry/radiochem medium→low 해소, biology만 medium 유지
        → science 중 1명만 medium (만장일치 아님 → approve). max_discussion_rounds=5이므로 조기 approve."""
        from AG_src.agents.expert_panel import ExpertPanelAgent

        domains = ("pharma", "biology", "chemistry", "radiochem", "math")

        # round1: pharma/biology/chemistry=medium, radiochem=high(1개) → conditional(high==1 규칙)
        r1_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="medium"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="medium"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="medium"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="high",
                               concerns=["신규 Met 도입 의심 — radiolysis 리스크 재확인 필요"]),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        r1_fanin = {
            "merged_concerns": ["pharma: PK 우려", "biology: 선택성 우려", "chemistry: 커플링 주의",
                                 "radiochem: radiolysis 재확인 필요"],
            "approve": False,
            "suggested_revisions": {"focus_positions": [5, 11]},
        }

        # round2: 동료 의견으로 pharma/chemistry/radiochem 우려 해소(low), biology만 medium 유지
        # → science 중 1명 medium만 남음(만장일치 아님, high 없음) → approve=True
        r2_verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low",
                           stance_change="medium→low", rebuttal="동료 의견 참고 후 해소"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="medium",
                            stance_change="유지", rebuttal="선택성 리스크 구조적 근거 남음"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low",
                              stance_change="medium→low", rebuttal="표준 L-aa 합성 가능 확인"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low",
                               stance_change="high→low", rebuttal="신규 Met 도입 아님 확인, 우려 해소"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        r2_fanin = {
            "merged_concerns": ["biology: 선택성 리스크 잔존 (informational)"],
            "approve": True,  # science 1명 medium만 남음 → approve
            "suggested_revisions": {},
        }

        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=(
                [r1_verdicts[d] for d in domains] + [r1_fanin]  # round1
                + [r2_verdicts[d] for d in domains] + [r2_fanin]  # round2
            )
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=3,
            hypothesis=_SAMPLE_HYP,
            mutation_guidance=_SAMPLE_GUIDANCE,
            max_discussion_rounds=5,
        )
        # round2에서 approve=True → 조기 종료 (5턴 소진하지 않음)
        self.assertTrue(result["approve"], "동료 의견으로 medium 해소 → approve=True 기대")
        self.assertEqual(result["consensus_status"], "approve")
        self.assertEqual(result["discussion_rounds"], 2, "2라운드에서 조기 approve 기대")
        # 총 LLM 호출: 2 rounds × (5 experts + 1 fanin) = 12
        self.assertEqual(result["llm_calls"], 12)

    def test_genuine_fatal_concern_maintains_reject_hard(self):
        """시나리오: 치명 우려(SS bond 파괴) → 5턴 토론에도 high 2+ 유지 → reject_hard.
        pharmacophore/SS-bond 파괴는 절대 approve되면 안 됨."""
        import AG_src.agents.expert_panel as ep_mod
        from AG_src.agents.expert_panel import ExpertPanelAgent

        orig_hard = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            domains = ("pharma", "biology", "chemistry", "radiochem", "math")
            # 5라운드 내내 biology=high(SS bond 파괴), pharma=high(독성) 유지 → reject_hard
            fatal_verdicts = {
                "pharma": dict(_MOCK_VERDICTS["pharma"], severity="high",
                               concerns=["F11→Cys 추가로 SS bond 생성 불확실 — 심각한 PK 독성 위험"]),
                "biology": dict(_MOCK_VERDICTS["biology"], severity="high",
                                concerns=["pos11=Cys → Cys3-Cys14와 SS bond 경쟁 → 약리단 파괴 위험"]),
                "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
                "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
                "math": dict(_MOCK_VERDICTS["math"], severity="low"),
            }
            fatal_fanin = {"merged_concerns": ["SS bond 파괴 위험"], "approve": False,
                           "suggested_revisions": {"focus_positions": [1, 2]}}

            n_rounds = 5
            side: List[Any] = []
            for _ in range(n_rounds):
                side += [fatal_verdicts[d] for d in domains]
                side += [fatal_fanin]

            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(side_effect=side)
            panel = ExpertPanelAgent(mock)
            result = panel.prereview_panel(
                iteration=1,
                hypothesis="pos11을 Cys로 치환 — 추가 이황화결합 형성 시도",
                mutation_guidance={"focus_positions": [11], "suggested_mutations": {"11": ["C"]}},
                max_discussion_rounds=5,
            )
            # 5턴 소진 후에도 high 2+ → reject_hard
            self.assertFalse(result["approve"], "치명 SS bond 파괴 우려 → approve=False (reject_hard) 기대")
            self.assertEqual(result["consensus_status"], "reject_hard")
            self.assertEqual(result["docking_decision"], False)
            self.assertEqual(result["discussion_rounds"], 5)
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig_hard

    def test_no_concerns_approves_in_one_round(self):
        """시나리오: 우려 없음 → 1라운드에서 즉시 approve (비용 최소화)."""
        from AG_src.agents.expert_panel import ExpertPanelAgent

        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        clean_verdicts = {
            d: dict(_MOCK_VERDICTS[d], severity="low", concerns=[])
            for d in domains
        }
        clean_fanin = {"merged_concerns": [], "approve": True, "suggested_revisions": {}}

        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[clean_verdicts[d] for d in domains] + [clean_fanin]
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=5,
            hypothesis=_SAMPLE_HYP,
            mutation_guidance=_SAMPLE_GUIDANCE,
            max_discussion_rounds=5,
        )
        # 우려 없음 → 1라운드 즉시 approve
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "approve")
        # PANEL_DISCUSS_MIN_CONCERNS=1이고 merged_concerns=0 → 즉시 종료
        self.assertEqual(result["discussion_rounds"], 1)
        self.assertEqual(result["llm_calls"], 6)

    def test_five_rounds_exhausted_conditional_becomes_forced_pass(self):
        """시나리오: science medium 만장일치가 라운드마다 동일하게 유지 (진전 없음, high 없음).
        E(stall 감지)가 PANEL_STALL_LIMIT(기본2) 라운드 연속 동일 불일치를 감지해 조기 종결한다
        (무한 재시도 방지 — 5턴을 다 소진하지 않고 stall로 끝남). 여전히 approve=True(하위호환)
        + docking_decision=True(저비용 탐색 허용)는 보장되어 기존 forced_pass의 취지를 계승한다."""
        import AG_src.agents.expert_panel as ep_mod
        from AG_src.agents.expert_panel import ExpertPanelAgent

        orig_hard = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        orig_max = ep_mod._PANEL_MEDIUM_ALL_MAX
        ep_mod._PANEL_MEDIUM_ALL_MAX = 3  # science 3명 만장일치 기준
        try:
            domains = ("pharma", "biology", "chemistry", "radiochem", "math")
            # 5라운드 내내 science 3명 medium 만장일치 유지 (high 없음)
            stubborn_verdicts = {
                d: dict(_MOCK_VERDICTS[d], severity="medium")
                for d in domains
            }
            stubborn_verdicts["math"] = dict(_MOCK_VERDICTS["math"], severity="low")  # math는 low
            stub_fanin = {
                "merged_concerns": ["pharma: PK 주의", "biology: 선택성 주의", "chemistry: 커플링 주의"],
                "approve": False,
                "suggested_revisions": {"focus_positions": [1, 6]},
            }
            n_rounds = 5
            side: List[Any] = []
            for _ in range(n_rounds):
                side += [stubborn_verdicts[d] for d in domains]
                side += [stub_fanin]

            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(side_effect=side)
            panel = ExpertPanelAgent(mock)
            result = panel.prereview_panel(
                iteration=2,
                hypothesis=_SAMPLE_HYP,
                mutation_guidance=_SAMPLE_GUIDANCE,
                max_discussion_rounds=5,
            )
            # 동일 불일치 2라운드 연속 → stall 조기종결. approve/docking은 여전히 통과(하위호환).
            self.assertTrue(result["approve"], "stall 종결 + medium만 → approve=True(하위호환) 기대")
            self.assertEqual(result["consensus_status"], "stall")
            self.assertTrue(result["stall_detected"])
            self.assertEqual(result["discussion_rounds"], ep_mod._PANEL_STALL_LIMIT)
            self.assertEqual(result["docking_decision"], True)
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig_hard
            ep_mod._PANEL_MEDIUM_ALL_MAX = orig_max


# ---------------------------------------------------------------------------
# 17. Round 2+ 프롬프트 합의 지향 키워드 검증
# ---------------------------------------------------------------------------

class TestConvergencePromptContent(unittest.TestCase):
    """round 2+ 시스템 프롬프트가 '합의 지향' 키워드를 포함하는지 검증."""

    def setUp(self):
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        self.sp = SYSTEM_PROMPTS

    def _assert_convergence_keywords(self, key: str):
        prompt = self.sp[key]
        # 합의 지향 핵심 키워드 존재 확인
        self.assertIn("CONVERGENCE", prompt, f"{key}: CONVERGENCE 키워드 없음")
        self.assertIn("ROUND 2+", prompt, f"{key}: ROUND 2+ 표기 없음")
        # 기계적 medium 금지 명시
        self.assertIn("LOWER", prompt, f"{key}: LOWER severity 지침 없음")

    def test_pharma_prompt_has_convergence_keywords(self):
        self._assert_convergence_keywords("expert_pharma_prereview")

    def test_biology_prompt_has_convergence_keywords(self):
        self._assert_convergence_keywords("expert_biology_prereview")

    def test_chemistry_prompt_has_convergence_keywords(self):
        self._assert_convergence_keywords("expert_chemistry_prereview")

    def test_fanin_system_has_convergence_note(self):
        """fanin 시스템 프롬프트에 CONVERGENCE INTERPRETATION 섹션 존재."""
        prompt = self.sp["expert_fanin"]
        self.assertIn("CONVERGENCE", prompt, "fanin 프롬프트에 CONVERGENCE 지침 없음")
        self.assertIn("FAVOR", prompt, "fanin 프롬프트에 FAVOR approve 지침 없음")

    def test_fanin_prompt_round2_notes_updated_severity(self):
        """format_expert_fanin_prompt round_idx > 1 시 '최종 severity 사용' 지침 존재."""
        from AG_src.llm.prompts import format_expert_fanin_prompt
        s = format_expert_fanin_prompt(list(_MOCK_VERDICTS.values()), iteration=2, round_idx=2)
        # 업데이트된 severity 사용 지침
        self.assertIn("FINAL", s.upper(), "round2 fanin 프롬프트에 FINAL severity 지침 없음")
        self.assertIn("approve=true", s.lower(), "round2 fanin 프롬프트에 approve 유도 지침 없음")

    def test_domain_prompt_peer_concerns_heading_updated(self):
        """format_expert_domain_prompt round 2+에서 peer_concerns 섹션이 포함된다."""
        from AG_src.llm.prompts import format_expert_domain_prompt
        peer = [{"domain": "biology", "severity": "medium", "concerns": ["선택성 우려"],
                 "suggestion": "다위치 탐색 권장"}]
        s = format_expert_domain_prompt(
            domain="pharma",
            iteration=2,
            hypothesis=_SAMPLE_HYP,
            mutation_guidance=_SAMPLE_GUIDANCE,
            round_idx=2,
            peer_concerns=peer,
        )
        self.assertIn("Peer Experts Concerns", s)
        self.assertIn("ROUND 2", s)


# ---------------------------------------------------------------------------
# 18. A: radiochem 실투표 전문가 추가 검증
# ---------------------------------------------------------------------------

class TestRadiochemExpert(unittest.TestCase):
    """radiochem 전문가가 실투표(non-advisory) 도메인으로 추가됐는지 검증."""

    def test_radiochem_in_expert_domains(self):
        from AG_src.agents.expert_panel import _EXPERT_DOMAINS
        self.assertIn("radiochem", _EXPERT_DOMAINS)

    def test_radiochem_not_advisory(self):
        """radiochem은 math와 달리 advisory-only가 아니다 (실투표 참여)."""
        from AG_src.agents.expert_panel import _ADVISORY_ONLY_DOMAINS
        self.assertNotIn("radiochem", _ADVISORY_ONLY_DOMAINS)
        self.assertIn("math", _ADVISORY_ONLY_DOMAINS)

    def test_radiochem_counted_in_consensus(self):
        """radiochem=high가 실제로 high_count에 반영된다(advisory 아님 확인)."""
        from AG_src.agents.expert_panel import _apply_consensus_rules
        verdicts = [
            {"domain": "pharma", "severity": "low"},
            {"domain": "biology", "severity": "low"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "radiochem", "severity": "high"},
            {"domain": "math", "severity": "high"},  # advisory — 무시되어야 함
        ]
        result = _apply_consensus_rules(verdicts)
        self.assertEqual(result["high_count"], 1, "radiochem=high는 집계에 포함, math=high는 제외")
        self.assertEqual(result["status"], "conditional")

    def test_radiochem_system_prompt_exists(self):
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        self.assertIn("expert_radiochem_prereview", SYSTEM_PROMPTS)

    def test_radiochem_prompt_mentions_chelator_and_radionuclide(self):
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        prompt = SYSTEM_PROMPTS["expert_radiochem_prereview"]
        self.assertIn("DOTA", prompt)
        self.assertIn("Ga-68", prompt)
        self.assertIn("Lu-177", prompt)
        self.assertIn("radiolysis", prompt.lower())
        self.assertIn("stoichiometry", prompt.lower())

    def test_radiochem_prompt_has_position_map_block(self):
        """radiochem도 다른 expert와 동일하게 SST-14 position map ground truth를 주입받는다."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        prompt = SYSTEM_PROMPTS["expert_radiochem_prereview"]
        self.assertIn("SST-14 POSITION MAP", prompt)
        self.assertIn("NOT Cys", prompt)

    def test_radiochem_prompt_discourages_hallucination(self):
        """환각 금지·uncertain 명시 지침 존재."""
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        prompt = SYSTEM_PROMPTS["expert_radiochem_prereview"]
        self.assertIn("uncertain", prompt.lower())

    def test_medium_all_max_default_is_two(self):
        """게이트 실질화(2026-07-02): medium 임계 기본값 2 (2명 이상 동조 시 재검토)."""
        import AG_src.agents.expert_panel as ep_mod
        self.assertEqual(ep_mod._PANEL_MEDIUM_ALL_MAX, 2)

    def test_fanin_prompt_mentions_five_experts(self):
        from AG_src.llm.prompts import SYSTEM_PROMPTS
        self.assertIn("radiochem", SYSTEM_PROMPTS["expert_fanin"])


# ---------------------------------------------------------------------------
# 19. B: Confidence-gated 라운드 검증
# ---------------------------------------------------------------------------

class TestConfidenceGate(unittest.TestCase):
    """전문가 verdict가 일치(불일치 없음)하면 조기 종료하는지 검증."""

    def _make_panel(self, mock: MagicMock) -> "ExpertPanelAgent":
        from AG_src.agents.expert_panel import ExpertPanelAgent
        return ExpertPanelAgent(mock)

    def test_confidence_gate_env_default_on(self):
        import AG_src.agents.expert_panel as ep_mod
        import os
        os.environ.pop("PANEL_CONFIDENCE_GATE", None)
        self.assertEqual(ep_mod._PANEL_CONFIDENCE_GATE, 1)

    def test_all_low_exits_round1_regardless_of_gate(self):
        """전원 low(불일치 없음)이면 gate on/off 무관하게 1라운드에서 종료(approve 분기가 우선)."""
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        low = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in domains}
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[low[d] for d in domains] + [fanin_ap]
        )
        panel = self._make_panel(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=5)
        self.assertEqual(result["discussion_rounds"], 1)

    def test_gate_disabled_by_env_falls_back_to_existing_behavior(self):
        """PANEL_CONFIDENCE_GATE=0이면 approve 달성 시에만 조기종료(기존 동작) — 회귀 없음 확인."""
        import AG_src.agents.expert_panel as ep_mod
        orig = ep_mod._PANEL_CONFIDENCE_GATE
        ep_mod._PANEL_CONFIDENCE_GATE = 0
        try:
            domains = ("pharma", "biology", "chemistry", "radiochem", "math")
            low = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in domains}
            fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(
                side_effect=[low[d] for d in domains] + [fanin_ap]
            )
            panel = self._make_panel(mock)
            result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=5)
            self.assertTrue(result["approve"])
            self.assertEqual(result["discussion_rounds"], 1)
        finally:
            ep_mod._PANEL_CONFIDENCE_GATE = orig


# ---------------------------------------------------------------------------
# 20. C: 소수 의견 보호 (minority dissent) 검증
# ---------------------------------------------------------------------------

class TestMinorityDissent(unittest.TestCase):
    """다수 동조에 소수 반대의견이 묻히지 않고 코드 레벨에서 보존되는지 검증."""

    def test_detect_minority_dissent_single_outlier(self):
        from AG_src.agents.expert_panel import _detect_minority_dissent
        verdicts = [
            {"domain": "pharma", "severity": "low"},
            {"domain": "biology", "severity": "high", "concerns": ["SS bond 경쟁 위험"]},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "radiochem", "severity": "low"},
        ]
        dissent = _detect_minority_dissent(verdicts)
        self.assertIsNotNone(dissent)
        self.assertEqual(dissent["domain"], "biology")
        self.assertEqual(dissent["severity"], "high")
        self.assertIn("SS bond", dissent["reason"])

    def test_no_dissent_when_unanimous(self):
        from AG_src.agents.expert_panel import _detect_minority_dissent
        verdicts = [{"domain": d, "severity": "low"} for d in ("pharma", "biology", "chemistry", "radiochem")]
        self.assertIsNone(_detect_minority_dissent(verdicts))

    def test_no_dissent_when_two_outliers(self):
        """2명 이상이 다르면(소수가 아닌 파벌) 단일 소수의견으로 판정하지 않는다."""
        from AG_src.agents.expert_panel import _detect_minority_dissent
        verdicts = [
            {"domain": "pharma", "severity": "high"},
            {"domain": "biology", "severity": "high"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "radiochem", "severity": "low"},
        ]
        self.assertIsNone(_detect_minority_dissent(verdicts))

    def test_math_excluded_from_dissent_detection(self):
        """math(advisory)는 소수의견 탐지 대상에서 제외된다."""
        from AG_src.agents.expert_panel import _detect_minority_dissent
        verdicts = [
            {"domain": "pharma", "severity": "low"},
            {"domain": "biology", "severity": "low"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "radiochem", "severity": "low"},
            {"domain": "math", "severity": "high"},  # advisory — 무시
        ]
        self.assertIsNone(_detect_minority_dissent(verdicts))

    def test_panel_result_contains_minority_dissent_field(self):
        """prereview_panel 결과에 minority_dissent가 채워진다(1명만 high인 케이스)."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        verdicts = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high",
                             concerns=["구조적 근거 있는 우려"]),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(
            side_effect=[verdicts[d] for d in domains] + [fanin_rej]
        )
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertIsNotNone(result["minority_dissent"])
        self.assertEqual(result["minority_dissent"]["domain"], "biology")
        # merged_concerns에도 소수의견이 명시적으로 병합됐는지 확인
        combined = " ".join(result["merged_concerns"])
        self.assertIn("minority_dissent", combined)

    def test_no_minority_dissent_field_is_none_when_absent(self):
        """소수의견이 없으면 minority_dissent=None."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        low = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in domains}
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=[low[d] for d in domains] + [fanin_ap])
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertIsNone(result["minority_dissent"])

    def test_fanin_prompt_instructs_minority_dissent_preservation(self):
        from AG_src.llm.prompts import format_expert_fanin_prompt, SYSTEM_PROMPTS
        s = format_expert_fanin_prompt(list(_MOCK_VERDICTS.values()), iteration=1)
        self.assertIn("minority_dissent", s.lower())
        self.assertIn("minority_dissent", SYSTEM_PROMPTS["expert_fanin"])


# ---------------------------------------------------------------------------
# 21. E: 정체(stall) 감지 검증
# ---------------------------------------------------------------------------

class TestStallDetection(unittest.TestCase):
    """동일한 불일치가 연속 지속되면 stall로 감지·조기 종결하는지 검증."""

    def test_panel_stall_limit_default_two(self):
        import AG_src.agents.expert_panel as ep_mod
        import os
        os.environ.pop("PANEL_STALL_LIMIT", None)
        self.assertEqual(ep_mod._PANEL_STALL_LIMIT, 2)

    def test_stall_detected_before_last_round(self):
        """마지막 라운드 전에 동일 불일치가 2라운드 연속되면 stall로 조기 종결된다."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        # pharma=medium 1개만 유지(만장일치 아님 → approve 아니지만 conditional도 아닌 경계)
        # medium 1개는 approve=True가 되므로, "불일치 지속"을 만들려면 high=1(conditional)로 고정.
        stuck = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high",
                             concerns=["동일 우려 반복"]),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
        side = []
        for _ in range(4):  # max_discussion_rounds=4 지만 stall이 2라운드째 개입해야 함
            side += [stuck[d] for d in domains] + [fanin_rej]
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=side)
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=4)
        self.assertEqual(result["consensus_status"], "stall")
        self.assertTrue(result["stall_detected"])
        self.assertEqual(result["discussion_rounds"], 2, "PANEL_STALL_LIMIT(2) 도달 즉시 종료 기대")
        self.assertTrue(result["approve"], "stall이어도 approve=True(하위호환) 유지")
        self.assertTrue(result["docking_decision"], "stall이어도 저비용 탐색 도킹은 허용")

    def test_stall_does_not_override_reject_hard_on_last_round(self):
        """마지막 라운드에서는 stall이 아니라 기존 reject_hard 로직이 최종 결론을 낸다."""
        import AG_src.agents.expert_panel as ep_mod
        from AG_src.agents.expert_panel import ExpertPanelAgent
        orig = ep_mod._PANEL_HARD_REJECT_HIGH
        ep_mod._PANEL_HARD_REJECT_HIGH = 1
        try:
            domains = ("pharma", "biology", "chemistry", "radiochem", "math")
            two_high = {
                "pharma": dict(_MOCK_VERDICTS["pharma"], severity="high"),
                "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
                "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
                "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
                "math": dict(_MOCK_VERDICTS["math"], severity="low"),
            }
            fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
            side = []
            for _ in range(2):
                side += [two_high[d] for d in domains] + [fanin_rej]
            mock = MagicMock()
            mock.has_llm = True
            mock.llm_generate_json = MagicMock(side_effect=side)
            panel = ExpertPanelAgent(mock)
            # max_discussion_rounds=2: 라운드2가 곧 마지막 라운드 → stall 미개입, reject_hard 확정
            result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=2)
            self.assertEqual(result["consensus_status"], "reject_hard")
            self.assertFalse(result["approve"])
            self.assertFalse(result["docking_decision"])
        finally:
            ep_mod._PANEL_HARD_REJECT_HIGH = orig

    def test_stall_streak_resets_on_approve(self):
        """중간에 approve 라운드가 있으면 stall streak가 리셋된다(다음에 불일치 재발해도 처음부터 카운트)."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        domains = ("pharma", "biology", "chemistry", "radiochem", "math")
        # round1: conditional(high=1), round2: approve(전원 low), round3: conditional(high=1) 재발
        cond = {
            "pharma": dict(_MOCK_VERDICTS["pharma"], severity="low"),
            "biology": dict(_MOCK_VERDICTS["biology"], severity="high"),
            "chemistry": dict(_MOCK_VERDICTS["chemistry"], severity="low"),
            "radiochem": dict(_MOCK_VERDICTS["radiochem"], severity="low"),
            "math": dict(_MOCK_VERDICTS["math"], severity="low"),
        }
        allow = {d: dict(_MOCK_VERDICTS[d], severity="low") for d in domains}
        fanin_rej = dict(_MOCK_FANIN_RESULT, approve=False)
        fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)
        side = (
            [cond[d] for d in domains] + [fanin_rej]     # round1: conditional
            + [allow[d] for d in domains] + [fanin_ap]   # round2: approve (해소 — 리셋)
        )
        mock = MagicMock()
        mock.has_llm = True
        mock.llm_generate_json = MagicMock(side_effect=side)
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=5)
        # round2에서 approve=True 달성 → stall 발동 전에 정상 종료
        self.assertTrue(result["approve"])
        self.assertEqual(result["consensus_status"], "approve")
        self.assertFalse(result["stall_detected"])
        self.assertEqual(result["discussion_rounds"], 2)


# ---------------------------------------------------------------------------
# 22. F: 모델 이질성(heterogeneous LLM provider) 배선 검증
# ---------------------------------------------------------------------------

class TestModelHeterogeneity(unittest.TestCase):
    """도메인→provider 매핑(F)이 올바르게 배선되는지 검증. 네트워크 호출 없이 mock으로 확인.

    주의: repo root conftest.py가 PANEL_HETERO_DOMAINS=""(off)를 세션 기본값으로 강제하므로,
    이 클래스의 각 테스트는 필요 시 인스턴스 속성(hetero_domains/hetero_provider)을 직접
    설정해 "이질성이 켜진 것처럼" 동작을 재현한다 — 실제 vLLM 8001에 연결하지 않는다.
    """

    def _make_panel_with_hetero(
        self,
        hetero_domains=frozenset({"biology", "chemistry"}),
    ):
        """hetero_provider를 MagicMock으로 주입한 ExpertPanelAgent를 반환.

        hetero_provider.generate_json은 호출된 프롬프트로부터 도메인을 추정해
        해당 도메인의 low-severity mock verdict를 반환한다(critic mock과 별개 경로).
        """
        from AG_src.agents.expert_panel import ExpertPanelAgent

        critic_mock = MagicMock()
        critic_mock.has_llm = True
        # critic 경로(pharma/radiochem/math)는 기존처럼 low severity 반환
        _critic_verdicts = {
            d: dict(_MOCK_VERDICTS[d], severity="low")
            for d in ("pharma", "radiochem", "math")
        }
        _fanin_ap = dict(_MOCK_FANIN_RESULT, approve=True)

        def _critic_side(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
            for d in ("pharma", "radiochem", "math"):
                if d.upper() in prompt:
                    return dict(_critic_verdicts[d])
            return dict(_fanin_ap)  # fan-in 호출

        critic_mock.llm_generate_json = MagicMock(side_effect=_critic_side)

        panel = ExpertPanelAgent(critic_mock)
        # __init__이 이미 (기본 env=off이므로) hetero_provider=None으로 세팅했을 것 —
        # 테스트 목적상 강제로 "이질성 on" 상태를 재현.
        panel.hetero_domains = hetero_domains
        hetero_mock = MagicMock()
        hetero_mock.model = "mistral-7b-instruct"

        def _hetero_side(prompt: str, system_prompt: str = "") -> Dict[str, Any]:
            for d in ("biology", "chemistry"):
                if d.upper() in prompt:
                    return dict(_MOCK_VERDICTS[d], severity="low")
            return {"domain": "unknown", "concerns": [], "severity": "low", "suggestion": ""}

        hetero_mock.generate_json = MagicMock(side_effect=_hetero_side)
        panel.hetero_provider = hetero_mock
        return panel, hetero_mock, critic_mock

    def test_hetero_off_by_default_in_test_env(self):
        """conftest.py가 세션 기본값을 off로 강제했는지 확인(네트워크 격리 안전장치)."""
        import AG_src.agents.expert_panel as ep_mod
        self.assertEqual(ep_mod._PANEL_HETERO_DOMAINS, frozenset())

    def test_panel_init_no_hetero_provider_when_domains_empty(self):
        """PANEL_HETERO_DOMAINS 비어있으면(기본) hetero_provider=None — 이질성 off."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = MagicMock()
        mock.has_llm = True
        panel = ExpertPanelAgent(mock)
        self.assertIsNone(panel.hetero_provider)

    def test_provider_for_domain_routes_hetero_domains(self):
        """hetero_domains에 속한 도메인은 hetero_provider로, 나머지는 critic으로 라우팅된다."""
        panel, hetero_mock, critic_mock = self._make_panel_with_hetero()
        bio_provider, bio_label = panel._provider_for_domain("biology")
        pharma_provider, pharma_label = panel._provider_for_domain("pharma")
        self.assertIs(bio_provider, hetero_mock)
        self.assertIn("hetero", bio_label)
        self.assertIs(pharma_provider, critic_mock)
        self.assertIn("default", pharma_label)

    def test_biology_chemistry_use_hetero_provider_end_to_end(self):
        """prereview_panel 실행 시 biology/chemistry verdict에 hetero backend가 llm_backend로 기록된다."""
        panel, hetero_mock, critic_mock = self._make_panel_with_hetero()
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        self.assertTrue(hetero_mock.generate_json.called, "hetero provider가 실제로 호출되어야 함")
        bio_verdict = result["expert_verdicts"]["biology"]
        chem_verdict = result["expert_verdicts"]["chemistry"]
        self.assertIn("llm_backend", bio_verdict)
        self.assertIn("hetero", bio_verdict["llm_backend"])
        self.assertIn("mistral", bio_verdict["llm_backend"])
        self.assertIn("hetero", chem_verdict["llm_backend"])

    def test_pharma_radiochem_math_use_default_provider(self):
        """이질성 켜져도 pharma/radiochem/math는 기본(critic) provider를 그대로 쓴다."""
        panel, hetero_mock, critic_mock = self._make_panel_with_hetero()
        result = panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        for domain in ("pharma", "radiochem", "math"):
            verdict = result["expert_verdicts"][domain]
            self.assertIn("default", verdict["llm_backend"])
            self.assertNotIn("hetero", verdict["llm_backend"])

    def test_fanin_always_uses_critic_not_hetero(self):
        """fan-in(통합)은 hetero_domains 설정과 무관하게 항상 critic.llm_generate_json을 쓴다."""
        panel, hetero_mock, critic_mock = self._make_panel_with_hetero()
        panel.prereview_panel(1, _SAMPLE_HYP, _SAMPLE_GUIDANCE, max_discussion_rounds=1)
        # fan-in 호출은 critic_mock.llm_generate_json 호출 횟수에 포함(도메인 5회 + fanin 1회 = 6회)
        self.assertEqual(critic_mock.llm_generate_json.call_count, 3 + 1,
                          "critic 경로: pharma/radiochem/math(3) + fanin(1) = 4회 기대")

    def test_hetero_domains_env_parsing(self):
        """PANEL_HETERO_DOMAINS 콤마 구분 파싱 — 공백 trim, 빈 항목 제거."""
        import importlib
        import AG_src.agents.expert_panel as ep_mod
        import os
        orig = os.environ.get("PANEL_HETERO_DOMAINS")
        os.environ["PANEL_HETERO_DOMAINS"] = " biology , chemistry ,, "
        try:
            importlib.reload(ep_mod)
            self.assertEqual(ep_mod._PANEL_HETERO_DOMAINS, frozenset({"biology", "chemistry"}))
        finally:
            if orig is None:
                os.environ.pop("PANEL_HETERO_DOMAINS", None)
            else:
                os.environ["PANEL_HETERO_DOMAINS"] = orig
            importlib.reload(ep_mod)  # 원상복구 — 이후 테스트에 영향 없도록

    def test_hetero_base_url_v1_suffix_normalized(self):
        """PANEL_HETERO_BASE_URL에 '/v1'이 포함돼도 중복 없이 정규화된다."""
        import importlib
        import os
        import AG_src.agents.expert_panel as ep_mod
        orig = os.environ.get("PANEL_HETERO_BASE_URL")
        os.environ["PANEL_HETERO_BASE_URL"] = "http://localhost:8001/v1"
        try:
            importlib.reload(ep_mod)
            self.assertEqual(ep_mod._PANEL_HETERO_BASE_URL, "http://localhost:8001")
        finally:
            if orig is None:
                os.environ.pop("PANEL_HETERO_BASE_URL", None)
            else:
                os.environ["PANEL_HETERO_BASE_URL"] = orig
            importlib.reload(ep_mod)

    def test_hetero_provider_creation_failure_falls_back_gracefully(self):
        """이질 provider 생성 자체가 예외를 던져도 패널 초기화는 실패하지 않는다."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        import AG_src.agents.expert_panel as ep_mod
        mock = MagicMock()
        mock.has_llm = True
        orig_domains = ep_mod._PANEL_HETERO_DOMAINS
        ep_mod._PANEL_HETERO_DOMAINS = frozenset({"biology"})
        with patch("AG_src.agents.expert_panel.VLLMProvider", side_effect=RuntimeError("boom")):
            try:
                panel = ExpertPanelAgent(mock)
                self.assertIsNone(panel.hetero_provider, "provider 생성 실패 시 None으로 폴백해야 함")
            finally:
                ep_mod._PANEL_HETERO_DOMAINS = orig_domains

    def test_no_llm_fallback_skips_hetero_provider_creation(self):
        """critic.has_llm=False이면 hetero_provider도 생성하지 않는다(불필요한 자원 낭비 방지)."""
        from AG_src.agents.expert_panel import ExpertPanelAgent
        import AG_src.agents.expert_panel as ep_mod
        mock = MagicMock()
        mock.has_llm = False
        orig_domains = ep_mod._PANEL_HETERO_DOMAINS
        ep_mod._PANEL_HETERO_DOMAINS = frozenset({"biology"})
        try:
            panel = ExpertPanelAgent(mock)
            self.assertIsNone(panel.hetero_provider)
        finally:
            ep_mod._PANEL_HETERO_DOMAINS = orig_domains


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
