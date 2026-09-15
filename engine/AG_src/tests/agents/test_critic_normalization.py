"""
test_critic_normalization.py
_normalize_failure_breakdown 및 ScientistCriticAgent.analyze_results 정규화 경로
테스트.

수정 배경:
    qc_ranker._build_qc_report()는 fail_reason 첫 어절(예: "pLDDT_mean", "ddG")을
    failure_breakdown 키로 사용한다. critic.py의 FAILURE_ACTION_MAP은 FailureType.*
    상수(예: "low_plddt", "good_dock_bad_ddg")만 인식하므로, 정규화 없이 그대로
    사용하면 propose_changes에서 모든 키가 스킵되어 changes=[] → 하드코드 fallback
    가설이 무한 반복된다.
    _normalize_failure_breakdown()는 critic 측에서 이 불일치를 흡수한다.
"""

import sys
import os

import pytest

# AG_src 패키지 루트를 sys.path에 추가
_AG_SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if _AG_SRC_ROOT not in sys.path:
    sys.path.insert(0, _AG_SRC_ROOT)

from AG_src.agents.critic import (
    FailureType,
    ScientistCriticAgent,
    _normalize_failure_breakdown,
)
from AG_src.agents.qc_ranker import Candidate, QCReport, RankTable


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_qc_report(
    failure_breakdown: dict[str, int],
    pass_rate: float = 0.0,
    total: int = 10,
) -> QCReport:
    passed = int(total * pass_rate)
    return QCReport(
        run_id="test-run",
        total_input=total,
        passed_count=passed,
        failed_count=total - passed,
        failure_breakdown=failure_breakdown,
        gates_applied={},
        pass_rate=pass_rate,
    )


def _make_rank_table(candidates: list[Candidate] | None = None) -> RankTable:
    return RankTable(
        run_id="test-run",
        iteration=1,
        ranked_candidates=candidates or [],
        weights={},
    )


def _make_candidate(cid: str = "bb01_seq01") -> Candidate:
    return Candidate(
        candidate_id=cid,
        backbone_id=1,
        seq_id=1,
        sequence="ACDEF",
        plddt_mean=80.0,
        plddt_interface=78.0,
        dock_score=-5.0,
        ddg=-8.0,
    )


# ---------------------------------------------------------------------------
# 1. 단순 직접 매핑 테스트
# ---------------------------------------------------------------------------

class TestNormalizeFailureBreakdownDirect:
    def test_plddt_mean_maps_to_low_plddt(self):
        result = _normalize_failure_breakdown({"pLDDT_mean": 5})
        assert result == {FailureType.LOW_PLDDT: 5}

    def test_ddg_maps_to_good_dock_bad_ddg(self):
        result = _normalize_failure_breakdown({"ddG": 3})
        assert result == {FailureType.GOOD_DOCK_BAD_DDG: 3}

    def test_clash_maps_to_high_clash(self):
        result = _normalize_failure_breakdown({"clash": 4})
        assert result == {FailureType.HIGH_CLASH: 4}

    def test_constraint_violations_maps_to_high_clash(self):
        result = _normalize_failure_breakdown({"constraint_violations": 2})
        assert result == {FailureType.HIGH_CLASH: 2}

    def test_selectivity_margin_maps_to_poor_selectivity(self):
        result = _normalize_failure_breakdown({"selectivity_margin": 6})
        assert result == {FailureType.POOR_SELECTIVITY: 6}

    def test_offtarget_max_score_maps_to_poor_selectivity(self):
        result = _normalize_failure_breakdown({"offtarget_max_score": 1})
        assert result == {FailureType.POOR_SELECTIVITY: 1}


# ---------------------------------------------------------------------------
# 2. 합산 테스트 — 같은 FailureType으로 여러 키가 누적되어야 함
# ---------------------------------------------------------------------------

class TestNormalizeAccumulation:
    def test_plddt_mean_and_interface_accumulate(self):
        """pLDDT_mean=3, pLDDT_interface=2 → LOW_PLDDT=5"""
        result = _normalize_failure_breakdown({"pLDDT_mean": 3, "pLDDT_interface": 2})
        assert result == {FailureType.LOW_PLDDT: 5}

    def test_clash_and_constraint_accumulate(self):
        """clash=4, constraint_violations=2 → HIGH_CLASH=6"""
        result = _normalize_failure_breakdown({"clash": 4, "constraint_violations": 2})
        assert result == {FailureType.HIGH_CLASH: 6}

    def test_selectivity_and_offtarget_accumulate(self):
        """selectivity_margin=3, offtarget_max_score=2 → POOR_SELECTIVITY=5"""
        result = _normalize_failure_breakdown(
            {"selectivity_margin": 3, "offtarget_max_score": 2}
        )
        assert result == {FailureType.POOR_SELECTIVITY: 5}


# ---------------------------------------------------------------------------
# 3. unknown 키 스킵 테스트
# ---------------------------------------------------------------------------

class TestNormalizeUnknownSkip:
    def test_unknown_key_is_skipped(self):
        """매핑 불가 키는 조용히 스킵, 나머지는 정상 처리."""
        result = _normalize_failure_breakdown({"unknown_key": 1, "ddG": 2})
        assert result == {FailureType.GOOD_DOCK_BAD_DDG: 2}

    def test_all_unknown_returns_empty(self):
        result = _normalize_failure_breakdown({"mystery": 5, "foo_bar": 3})
        assert result == {}


# ---------------------------------------------------------------------------
# 4. fuzzy fallback 테스트
# ---------------------------------------------------------------------------

class TestNormalizeFuzzyFallback:
    def test_fuzzy_plddt_mixed_case(self):
        result = _normalize_failure_breakdown({"SomePLDDT_value": 2})
        assert result == {FailureType.LOW_PLDDT: 2}

    def test_fuzzy_ddg(self):
        result = _normalize_failure_breakdown({"bad_ddg_score": 1})
        assert result == {FailureType.GOOD_DOCK_BAD_DDG: 1}

    def test_fuzzy_clash(self):
        result = _normalize_failure_breakdown({"heavy_clash_count": 3})
        assert result == {FailureType.HIGH_CLASH: 3}

    def test_fuzzy_selectivity(self):
        result = _normalize_failure_breakdown({"poor_selectivity_score": 4})
        assert result == {FailureType.POOR_SELECTIVITY: 4}

    def test_fuzzy_off_target(self):
        result = _normalize_failure_breakdown({"off-target_binding": 2})
        assert result == {FailureType.POOR_SELECTIVITY: 2}

    def test_fuzzy_duplicate_sequence(self):
        result = _normalize_failure_breakdown({"duplicate_count": 5})
        assert result == {FailureType.LOW_SEQUENCE_DIVERSITY: 5}


# ---------------------------------------------------------------------------
# 5. FailureType 상수 키 그대로 통과 테스트 (이미 정규화된 경우)
# ---------------------------------------------------------------------------

class TestNormalizePassthrough:
    def test_already_normalized_low_plddt(self):
        result = _normalize_failure_breakdown({FailureType.LOW_PLDDT: 7})
        assert result == {FailureType.LOW_PLDDT: 7}

    def test_already_normalized_mixed(self):
        raw = {
            FailureType.HIGH_CLASH: 3,
            FailureType.GOOD_DOCK_BAD_DDG: 2,
        }
        assert _normalize_failure_breakdown(raw) == raw


# ---------------------------------------------------------------------------
# 6. 통합: analyze_results → proposed_changes 비어있지 않아야 함
# ---------------------------------------------------------------------------

class TestAnalyzeResultsIntegration:
    def test_plddt_breakdown_produces_nonempty_changes(self):
        """qc_ranker 스타일 키(pLDDT_mean=10)로 QCReport를 만들면
        analyze_results는 LOW_PLDDT 대응 변경을 제안해야 한다."""
        agent = ScientistCriticAgent()
        rank_table = _make_rank_table([_make_candidate()])
        qc_report = _make_qc_report({"pLDDT_mean": 10}, pass_rate=0.1, total=11)

        analysis = agent.analyze_results(
            rank_table=rank_table,
            qc_report=qc_report,
            iteration=1,
            current_params={"mpnn_temperature": 0.2, "peptide_length_max": 30},
        )

        assert len(analysis.proposed_changes) > 0, (
            "pLDDT_mean=10 breakdown이 있을 때 proposed_changes가 비어있으면 안 된다."
        )
        assert FailureType.LOW_PLDDT in {ch.failure_type for ch in analysis.proposed_changes}

    def test_hypothesis_mentions_low_plddt_action(self):
        """LOW_PLDDT 실패가 있을 때 hypothesis가 pLDDT 관련 내용을 포함해야 한다."""
        agent = ScientistCriticAgent()
        rank_table = _make_rank_table([_make_candidate()])
        qc_report = _make_qc_report({"pLDDT_mean": 10}, pass_rate=0.1, total=11)

        analysis = agent.analyze_results(
            rank_table=rank_table,
            qc_report=qc_report,
            iteration=1,
            current_params={"mpnn_temperature": 0.2, "peptide_length_max": 30},
        )

        assert "low_plddt" in analysis.hypothesis.lower() or "plddt" in analysis.hypothesis.lower(), (
            f"hypothesis에 pLDDT 관련 내용이 없음: {analysis.hypothesis}"
        )


# ---------------------------------------------------------------------------
# 7. 회귀: failure_breakdown={}, pass_rate=0.0 → 새 동적 메시지
# ---------------------------------------------------------------------------

class TestGenerateHypothesisRegression:
    def setup_method(self):
        self.agent = ScientistCriticAgent()

    def test_empty_breakdown_zero_passrate_returns_dynamic_message(self):
        """빈 breakdown + pass_rate=0.0 → 전 후보 게이트 실패 동적 메시지 반환."""
        hypothesis = self.agent.generate_hypothesis([], qc_pass_rate=0.0)
        assert "게이트 실패" in hypothesis or "pass_rate" in hypothesis, (
            f"동적 메시지가 아님: {hypothesis}"
        )
        # 기존 하드코드 fallback과 달라야 함
        assert hypothesis != "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."

    def test_empty_breakdown_zero_passrate_shows_passrate_value(self):
        """동적 메시지가 실제 pass_rate 수치를 포함해야 한다."""
        hypothesis = self.agent.generate_hypothesis([], qc_pass_rate=0.0)
        assert "0.0%" in hypothesis

    def test_empty_breakdown_high_passrate_returns_hardcode_fallback(self):
        """빈 breakdown + pass_rate=0.8 → 기존 하드코드 그대로 반환."""
        hypothesis = self.agent.generate_hypothesis([], qc_pass_rate=0.8)
        assert hypothesis == "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."

    def test_no_passrate_arg_returns_hardcode_fallback(self):
        """qc_pass_rate 인수 없이 호출 시 하드코드 그대로 (하위 호환)."""
        hypothesis = self.agent.generate_hypothesis([])
        assert hypothesis == "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."

    def test_boundary_passrate_009_is_dynamic(self):
        """pass_rate=0.09 (< 0.1) → 동적 메시지."""
        hypothesis = self.agent.generate_hypothesis([], qc_pass_rate=0.09)
        assert hypothesis != "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."

    def test_boundary_passrate_010_is_hardcode(self):
        """pass_rate=0.10 (= 0.1) → 기존 하드코드 (경계값은 동적 미해당)."""
        hypothesis = self.agent.generate_hypothesis([], qc_pass_rate=0.10)
        assert hypothesis == "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."
