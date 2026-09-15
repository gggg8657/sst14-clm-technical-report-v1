"""
critic.py
SSTR2 펩타이드 바인더 Co-Scientist - Scientist Critic 에이전트
Role: 비판적 검토 / 원인 분석 (Critical Review & Root Cause Analysis)

Scientist Critic은 QC&Ranker의 결과를 구조적 근거에 기반하여 해석하고,
실패 유형을 분류한 뒤 다음 iteration에서 변경할 파라미터(최대 2개)를 제안한다.
매 iteration마다 변경점과 가설을 명시적으로 기록하여 원인-결과 추적을 가능하게 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base_agent import BaseAgent
from .qc_ranker import Candidate, QCReport, RankTable
from ..llm.prompts import (
    get_system_prompt,
    format_critic_prompt,
    format_prereview_critic_prompt,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ParameterChange:
    """Critic이 제안하는 단일 파라미터 변경 사항.

    Attributes:
        parameter_name: 변경할 파라미터 이름 (planner.ExperimentPlan.parameters 키)
        old_value: 현재(이전) 값
        new_value: 제안하는 새 값
        rationale: 변경 근거 (구조적 증거 포함)
        expected_effect: 예상 효과 (구체적으로 기술)
        failure_type: 대응하는 실패 유형
    """
    parameter_name: str
    old_value: Any
    new_value: Any
    rationale: str
    expected_effect: str
    failure_type: str = ""


@dataclass
class CriticAnalysis:
    """Critic 분석 결과 - Planner에게 전달되는 핵심 출력물.

    Attributes:
        iteration: 분석 대상 iteration 번호
        failure_summary: 실패 유형별 카운트 딕셔너리
        structural_insights: 구조적 분석 내용 (자유 문자열)
        proposed_changes: 제안 파라미터 변경 목록 (최대 2개)
        hypothesis: 다음 iteration을 위한 과학적 가설
        top_candidate_ids: 이번 iteration에서 가장 우수한 후보 ID 목록
        created_at: 분석 생성 시각
    """
    iteration: int
    failure_summary: dict[str, int]
    structural_insights: str
    proposed_changes: list[ParameterChange]  # 최대 2개
    hypothesis: str
    top_candidate_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Failure type constants and action mapping
# ---------------------------------------------------------------------------

class FailureType:
    """실패 유형 상수 정의."""
    LOW_PLDDT = "low_plddt"
    GOOD_DOCK_BAD_DDG = "good_dock_bad_ddg"
    POCKET_SPECIFIC_FAILURE = "pocket_specific_failure"
    LOW_SEQUENCE_DIVERSITY = "low_sequence_diversity"
    HIGH_CLASH = "high_clash"
    POOR_SELECTIVITY = "poor_selectivity"


# 실패 유형 -> 수정 액션 매핑 (prompt/001_ag_set 기반)
# qc_ranker.py fail_reason 첫 어절 -> FailureType 상수 정규화 테이블
# (qc_ranker.py 측 변경 없이 critic 측에서 흡수)
_FAIL_REASON_TO_TYPE: dict[str, str] = {
    "pLDDT_mean":            FailureType.LOW_PLDDT,
    "pLDDT_interface":       FailureType.LOW_PLDDT,
    "ddG":                   FailureType.GOOD_DOCK_BAD_DDG,
    "clash":                 FailureType.HIGH_CLASH,
    "constraint_violations": FailureType.HIGH_CLASH,
    "selectivity_margin":    FailureType.POOR_SELECTIVITY,
    "offtarget_max_score":   FailureType.POOR_SELECTIVITY,
    "dock_score":            FailureType.POCKET_SPECIFIC_FAILURE,
    "sequence":              FailureType.LOW_SEQUENCE_DIVERSITY,
    "duplicate":             FailureType.LOW_SEQUENCE_DIVERSITY,
}


def _normalize_failure_breakdown(raw: dict[str, int]) -> dict[str, int]:
    """qc_ranker 키를 FailureType 상수로 정규화한다.

    qc_ranker._build_qc_report()는 fail_reason의 첫 어절(예: "pLDDT_mean",
    "ddG", "clash")을 키로 사용하는 반면, FAILURE_ACTION_MAP은
    FailureType.* 상수(예: "low_plddt", "good_dock_bad_ddg")를 키로 기대한다.
    이 함수는 critic 측에서 두 컨벤션의 불일치를 흡수한다.

    정규화 우선순위:
        1. _FAIL_REASON_TO_TYPE 직접 매핑
        2. 소문자 부분 문자열 fuzzy 매핑
        3. 매핑 불가 키는 스킵 (unknown 무음 처리)

    같은 FailureType으로 합산되는 복수 키의 카운트는 누적된다.
    """
    normalized: dict[str, int] = {}
    for raw_key, cnt in raw.items():
        # 이미 FailureType 상수라면 그대로 통과
        if raw_key in (
            FailureType.LOW_PLDDT,
            FailureType.GOOD_DOCK_BAD_DDG,
            FailureType.POCKET_SPECIFIC_FAILURE,
            FailureType.LOW_SEQUENCE_DIVERSITY,
            FailureType.HIGH_CLASH,
            FailureType.POOR_SELECTIVITY,
        ):
            normalized[raw_key] = normalized.get(raw_key, 0) + cnt
            continue

        mapped: Optional[str] = _FAIL_REASON_TO_TYPE.get(raw_key)
        if mapped is None:
            # fuzzy fallback: 소문자 부분 문자열 검색
            lk = raw_key.lower()
            if "plddt" in lk:
                mapped = FailureType.LOW_PLDDT
            elif "ddg" in lk:
                mapped = FailureType.GOOD_DOCK_BAD_DDG
            elif "clash" in lk or "constraint" in lk or "violation" in lk:
                mapped = FailureType.HIGH_CLASH
            elif "selectivity" in lk or "offtarget" in lk or "off-target" in lk:
                mapped = FailureType.POOR_SELECTIVITY
            elif "duplicate" in lk or "sequence" in lk:
                mapped = FailureType.LOW_SEQUENCE_DIVERSITY
            elif "dock" in lk or "pocket" in lk:
                mapped = FailureType.POCKET_SPECIFIC_FAILURE
            else:
                continue  # 매핑 불가 → 스킵
        normalized[mapped] = normalized.get(mapped, 0) + cnt
    return normalized


FAILURE_ACTION_MAP: dict[str, dict[str, Any]] = {
    FailureType.LOW_PLDDT: {
        "description": "ESMFold 구조 신뢰도(pLDDT) 낮음",
        "actions": [
            {
                "parameter": "mpnn_temperature",
                "delta_multiplier": 0.5,   # 현재값의 50% 감소
                "direction": "decrease",
                "rationale": "MPNN temperature를 낮춰 더 보수적인 서열 샘플링으로 안정 구조 유도",
                "expected_effect": "구조 신뢰도(pLDDT) 향상, 다양성 일부 희생",
            },
            {
                "parameter": "peptide_length_max",
                "fixed_delta": -5,
                "direction": "decrease",
                "rationale": "과도하게 긴 펩타이드는 구조 예측 신뢰도 저하 유발",
                "expected_effect": "pLDDT 상승, 탐색 공간 축소",
            },
        ],
    },
    FailureType.GOOD_DOCK_BAD_DDG: {
        "description": "도킹 점수는 우수하나 Rosetta ddG 불량",
        "actions": [
            {
                "parameter": "hotspot_res",
                "direction": "update",
                "rationale": "hotspot residue를 명시하여 FlexPepDock constraint 강화",
                "expected_effect": "ddG 개선, 포켓 내 결합 자세 안정화",
            },
            {
                "parameter": "rosetta_relax_cycles",
                "fixed_delta": 3,
                "direction": "increase",
                "rationale": "Rosetta relax cycle 증가로 sidechain 재배치 최적화",
                "expected_effect": "ddG 개선, clash 감소",
            },
        ],
    },
    FailureType.POCKET_SPECIFIC_FAILURE: {
        "description": "특정 포켓에서만 반복적 실패",
        "actions": [
            {
                "parameter": "contigs",
                "direction": "update",
                "rationale": "포켓 정의(contig) 재설정 - residue 범위 조정",
                "expected_effect": "새 포켓 정의로 바인딩 모드 다각화",
            },
            {
                "parameter": "docking_engine",
                "direction": "toggle",  # diffdock <-> boltz2
                "rationale": "도킹 엔진 전환으로 포켓 샘플링 다각화",
                "expected_effect": "다른 도킹 엔진 특성으로 새로운 결합 포즈 발견",
            },
        ],
    },
    FailureType.LOW_SEQUENCE_DIVERSITY: {
        "description": "서열 다양성 부족 (중복 서열 다수)",
        "actions": [
            {
                "parameter": "mpnn_temperature",
                "fixed_delta": 0.1,
                "direction": "increase",
                "rationale": "MPNN temperature 증가로 서열 다양성 확보",
                "expected_effect": "서열 공간 탐색 확대, pLDDT 소폭 감소 가능",
            },
            {
                "parameter": "n_backbone",
                "delta_multiplier": 1.5,
                "direction": "increase",
                "rationale": "백본 수 증가로 구조 공간 다각화",
                "expected_effect": "더 다양한 구조 토폴로지 탐색",
            },
        ],
    },
    FailureType.HIGH_CLASH: {
        "description": "Rosetta clash score 과다",
        "actions": [
            {
                "parameter": "rosetta_relax_cycles",
                "fixed_delta": 5,
                "direction": "increase",
                "rationale": "relax cycle 추가로 구조적 충돌 해소",
                "expected_effect": "clash_count 감소, 더 안정적인 복합체 구조",
            },
        ],
    },
    FailureType.POOR_SELECTIVITY: {
        "description": "SSTR2 선택성 부족 - off-target(SSTR1/3/4/5)에도 강하게 결합",
        "actions": [
            {
                "parameter": "hotspot_residues",
                "direction": "add",
                "rationale": "SSTR2 고유 포켓 잔기에 대한 hotspot constraint 강화",
                "expected_effect": "SSTR2 특이적 결합 강화로 선택성 마진 개선",
            },
            {
                "parameter": "peptide_length_bias",
                "direction": "increase",
                "rationale": "펩타이드 길이를 늘려 SSTR2 특이적 접촉면 확보",
                "expected_effect": "off-target 접촉면 감소, selectivity_margin 개선",
            },
            {
                "parameter": "rfdiffusion_secondary_bias",
                "direction": "adjust",
                "rationale": "2차 구조 bias를 SSTR2 포켓 형태에 맞게 조정",
                "expected_effect": "SSTR2 포켓에 맞는 헬릭스/루프 비율 최적화",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Scientist Critic agent
# ---------------------------------------------------------------------------

class ScientistCriticAgent(BaseAgent):
    """비판적 검토 및 원인 분석 에이전트.

    역할:
        1. 각 후보가 좋거나 나쁜 이유를 구조적 근거로 설명
        2. 다음 iteration에서 변경할 파라미터/전략 제안 (최대 2개)
        3. 실패 유형을 규칙 기반으로 분류하고 대응 액션에 매핑
        4. 변경점·가설·예상 효과를 명시적으로 기록

    핵심 원칙:
        - 매 iteration 변경은 최대 2개 파라미터로 제한 (원인-결과 추적)
        - 실패 유형 분류 -> 액션 매핑 테이블 기반 객관적 제안
        - 가설은 검증 가능한 형태로 명시
    """

    def __init__(self, llm_provider: str = "claude") -> None:
        super().__init__(
            name="ScientistCritic",
            role="비판적 검토/원인 분석",
            description=(
                "QC&Ranker 결과를 구조적 근거로 해석하고, 실패 유형을 분류하여 "
                "다음 iteration 파라미터 변경(최대 2개)을 제안하며 가설을 기록한다."
            ),
            llm_provider=llm_provider,
        )
        self._analysis_history: list[CriticAnalysis] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_results(
        self,
        rank_table: RankTable,
        qc_report: QCReport,
        iteration: int,
        current_params: Optional[dict[str, Any]] = None,
        selectivity_info: Optional[dict[str, Any]] = None,
        silo_b_log_path: Optional[Path] = None,
        surrogate_map: Optional[dict[str, dict[str, Any]]] = None,
    ) -> CriticAnalysis:
        """QC 결과를 분석하여 CriticAnalysis를 생성한다.

        분석 흐름:
            1. 실패 후보에서 실패 유형 분류
            2. 상위 후보의 구조적 특징 파악
            3. 가장 빈번한 실패 유형 기반으로 파라미터 변경 제안 (최대 2개)
            4. 가설 생성

        Args:
            rank_table: QCRanker가 생성한 랭킹 테이블
            qc_report: QCRanker가 생성한 QC 보고서
            iteration: 현재 반복 번호
            current_params: 현재 ExperimentPlan.parameters (변경 계산용)
            silo_b_log_path: Silo B experiment_log.jsonl 경로 (궤적 주입용).
                **pyrosetta_flow 경로만 허용.**
            surrogate_map: {candidate_id: {half_life_h, admet_score, hc50}} — runner가 전달.
                없으면 빈 dict(graceful). 계산 X, 전달만.

        Returns:
            CriticAnalysis: Planner에게 전달할 분석 결과
        """
        self.log(f"Iteration {iteration} 결과 분석 시작")

        params = current_params or {}
        _surrogate_map = surrogate_map or {}
        all_candidates = rank_table.ranked_candidates

        # --- LLM 분기: LLM이 있으면 프롬프트 기반 분석 시도 ---
        if self.has_llm:
            llm_analysis = self._analyze_via_llm(
                rank_table, qc_report, iteration, params,
                selectivity_info=selectivity_info,
                silo_b_log_path=silo_b_log_path,
                surrogate_map=_surrogate_map,
            )
            if llm_analysis is not None:
                self._analysis_history.append(llm_analysis)
                self.log(f"LLM 분석 완료: {len(llm_analysis.proposed_changes)}개 변경 제안")
                return llm_analysis
            self.log("LLM 분석 실패, 규칙 기반 폴백", level="warning")

        # --- 규칙 기반 폴백 (기존 로직) ---
        # 실패 유형 분류
        # (rank_table에는 통과 후보만 있으므로 qc_report.failure_breakdown 활용)
        failure_summary = dict(qc_report.failure_breakdown)
        # qc_ranker 키(첫 어절) → FailureType 상수 정규화 (키 컨벤션 불일치 흡수)
        failure_summary = _normalize_failure_breakdown(failure_summary)

        # 실패 유형 -> 횟수 기반으로 주요 실패 유형 결정
        dominant_failures = self._get_dominant_failures(failure_summary, qc_report)

        # 구조적 인사이트 생성
        insights = self._generate_structural_insights(all_candidates, qc_report)

        # 파라미터 변경 제안 (최대 2개)
        proposed = self.propose_changes(
            failure_analysis={"dominant": dominant_failures, "summary": failure_summary},
            current_params=params,
        )[:2]  # 엄격하게 2개 제한

        # 가설 생성 (pass_rate 전달 → 전 후보 실패 케이스 동적 메시지)
        hypothesis = self.generate_hypothesis(proposed, qc_pass_rate=qc_report.pass_rate)

        top_ids = [c.candidate_id for c in all_candidates[:5]]

        analysis = CriticAnalysis(
            iteration=iteration,
            failure_summary=failure_summary,
            structural_insights=insights,
            proposed_changes=proposed,
            hypothesis=hypothesis,
            top_candidate_ids=top_ids,
        )

        self._analysis_history.append(analysis)
        self.log(f"분석 완료: {len(proposed)}개 변경 제안, 가설: {hypothesis[:60]}...")
        return analysis

    def classify_failures(
        self, failed_candidates: list[Candidate]
    ) -> dict[str, int]:
        """실패 후보들을 실패 유형별로 분류하고 카운트를 반환한다.

        분류 기준:
            - 'low_plddt': fail_reasons에 'pLDDT' 키워드 포함
            - 'good_dock_bad_ddg': dock_score 상위권이나 ddG 기준 실패
            - 'high_clash': fail_reasons에 'clash' 키워드 포함
            - 'pocket_specific': 특정 backbone_id에서만 반복 실패
            - 'low_sequence_diversity': 중복 서열 (sequence 기준)

        Args:
            failed_candidates: 게이트 실패 Candidate 목록

        Returns:
            {failure_type: count} 딕셔너리
        """
        counts: dict[str, int] = {
            FailureType.LOW_PLDDT: 0,
            FailureType.GOOD_DOCK_BAD_DDG: 0,
            FailureType.POCKET_SPECIFIC_FAILURE: 0,
            FailureType.LOW_SEQUENCE_DIVERSITY: 0,
            FailureType.HIGH_CLASH: 0,
            FailureType.POOR_SELECTIVITY: 0,
        }

        # 서열 중복 확인
        seq_set: set[str] = set()
        duplicate_seq_count = 0
        for c in failed_candidates:
            if c.sequence in seq_set:
                duplicate_seq_count += 1
            seq_set.add(c.sequence)
        counts[FailureType.LOW_SEQUENCE_DIVERSITY] = duplicate_seq_count

        # 백본별 실패 집중도 확인 (특정 포켓 실패 판단)
        backbone_fail: dict[int, int] = {}
        for c in failed_candidates:
            backbone_fail[c.backbone_id] = backbone_fail.get(c.backbone_id, 0) + 1
        if backbone_fail:
            max_fail_backbone = max(backbone_fail.values())
            if max_fail_backbone >= 3:
                counts[FailureType.POCKET_SPECIFIC_FAILURE] += max_fail_backbone

        # fail_reasons 파싱
        for c in failed_candidates:
            for reason in c.fail_reasons:
                reason_lower = reason.lower()
                if "plddt" in reason_lower:
                    counts[FailureType.LOW_PLDDT] += 1
                if "clash" in reason_lower:
                    counts[FailureType.HIGH_CLASH] += 1
                if "ddg" in reason_lower and c.dock_score < -1.0:
                    # 도킹은 좋은데 ddG 실패
                    counts[FailureType.GOOD_DOCK_BAD_DDG] += 1
                if "selectivity" in reason_lower or "off-target" in reason_lower or "offtarget" in reason_lower:
                    counts[FailureType.POOR_SELECTIVITY] += 1

        return {k: v for k, v in counts.items() if v > 0}

    def propose_changes(
        self,
        failure_analysis: dict[str, Any],
        current_params: dict[str, Any],
    ) -> list[ParameterChange]:
        """실패 분석을 바탕으로 파라미터 변경을 제안한다.

        FAILURE_ACTION_MAP을 참조하여 규칙 기반으로 변경을 제안하며,
        Planner에서 최대 2개만 적용한다.

        Args:
            failure_analysis: classify_failures() 결과 및 dominant 실패 유형
            current_params: 현재 ExperimentPlan.parameters

        Returns:
            ParameterChange 목록 (우선순위 순)
        """
        dominant: list[str] = failure_analysis.get("dominant", [])
        changes: list[ParameterChange] = []

        for failure_type in dominant[:3]:  # 상위 3개 실패 유형만 처리
            if failure_type not in FAILURE_ACTION_MAP:
                continue
            actions = FAILURE_ACTION_MAP[failure_type]["actions"]
            for action in actions[:1]:  # 유형당 1개만 선택
                param_name = action["parameter"]
                old_val = current_params.get(param_name)
                new_val = self._compute_new_value(action, old_val)
                if new_val is None:
                    continue
                changes.append(
                    ParameterChange(
                        parameter_name=param_name,
                        old_value=old_val,
                        new_value=new_val,
                        rationale=action["rationale"],
                        expected_effect=action["expected_effect"],
                        failure_type=failure_type,
                    )
                )
                if len(changes) >= 2:
                    break
            if len(changes) >= 2:
                break

        self.log(f"파라미터 변경 제안: {len(changes)}개")
        for ch in changes:
            self.log(f"  {ch.parameter_name}: {ch.old_value} -> {ch.new_value}")

        return changes

    def generate_hypothesis(
        self,
        changes: list[ParameterChange],
        qc_pass_rate: Optional[float] = None,
    ) -> str:
        """제안된 변경사항으로부터 검증 가능한 과학적 가설을 생성한다.

        Args:
            changes: propose_changes() 반환값
            qc_pass_rate: QCReport.pass_rate (0~1). 제공 시 전 후보 실패
                케이스(pass_rate < 0.1)를 별도 동적 메시지로 구분한다.

        Returns:
            가설 문자열 (다음 iteration의 ExperimentPlan.hypothesis에 저장됨)
        """
        if not changes:
            # 전 후보 게이트 실패: 동적 메시지로 원인 명시
            if qc_pass_rate is not None and qc_pass_rate < 0.1:
                return (
                    f"전 후보 게이트 실패(pass_rate={qc_pass_rate * 100:.1f}%). "
                    "게이트 임계치 완화 또는 백본 다양성 확대 검토 필요."
                )
            # 정상 통과율인데 변경사항 없음: 기존 하드코드 그대로
            return "이전 iteration과 동일한 파라미터로 재현성 확인 및 후보 다양성 탐색."

        parts: list[str] = []
        for ch in changes:
            parts.append(
                f"[{ch.failure_type}] {ch.parameter_name}을(를) "
                f"{ch.old_value} -> {ch.new_value}로 변경하면 "
                f"{ch.expected_effect}이(가) 예상된다."
            )

        hypothesis = " ".join(parts)
        hypothesis += (
            " 이 가설은 다음 iteration 결과의 pLDDT/ddG/도킹 점수 분포 변화로 검증한다."
        )
        return hypothesis

    # ------------------------------------------------------------------
    # Pre-Review API (Track1-P1: Planner↔Critic 사전검토 루프)
    # ------------------------------------------------------------------

    def prereview_hypothesis(
        self,
        iteration: int,
        hypothesis: str,
        mutation_guidance: dict[str, Any],
        trajectory_summary: Optional[str] = None,
        selectivity_leaderboard: Optional[list[dict[str, Any]]] = None,
        best_delta_margin: Optional[float] = None,
        round_idx: int = 1,
    ) -> dict[str, Any]:
        """도킹 전 Planner 가설을 사전검토한다 (경량 모드 — QC 결과 불필요).

        입력=Planner의 hypothesis+mutation_guidance(+선택성 리더보드·궤적).
        출력=비판/개선제안 JSON:
            {
              "concerns": [str, ...],
              "suggested_revisions": {
                  "focus_positions": [int, ...],
                  "strategy": str,
                  "additional_positions": [int, ...]
              },
              "approve": bool
            }

        LLM 미사용 또는 실패 시 approve=True(통과) 폴백 — 기존 흐름 보존.

        Args:
            iteration: 현재 반복 번호
            hypothesis: Planner 가설 문자열
            mutation_guidance: Planner mutation_guidance dict
            trajectory_summary: 최근 궤적 요약 문자열 (선택)
            selectivity_leaderboard: 선택성 리더보드 상위 항목 (선택)
            best_delta_margin: 현재까지 최고 Δmargin (선택)
            round_idx: 사전검토 라운드 번호 (1-indexed)

        Returns:
            사전검토 결과 dict (approve 키 포함)
        """
        self.log(
            f"[pre-review] Iteration {iteration} Round {round_idx} 사전검토 시작 "
            f"(focus={mutation_guidance.get('focus_positions', [])})"
        )

        if not self.has_llm:
            self.log("[pre-review] LLM 없음 — 통과 폴백", level="warning")
            return {"concerns": [], "suggested_revisions": {}, "approve": True}

        prompt = format_prereview_critic_prompt(
            iteration=iteration,
            hypothesis=hypothesis,
            mutation_guidance=mutation_guidance,
            trajectory_summary=trajectory_summary,
            selectivity_leaderboard=selectivity_leaderboard,
            best_delta_margin=best_delta_margin,
            round_idx=round_idx,
        )
        system = get_system_prompt("critic_prereview")

        try:
            result = self.llm_generate_json(prompt, system_prompt=system)
        except Exception as exc:
            self.log(f"[pre-review] LLM 예외: {exc} — 통과 폴백", level="error")
            return {"concerns": [], "suggested_revisions": {}, "approve": True}

        if result is None:
            self.log("[pre-review] LLM 응답 None — 통과 폴백", level="warning")
            return {"concerns": [], "suggested_revisions": {}, "approve": True}

        # 필수 키 검증 및 정규화
        approved: bool = bool(result.get("approve", True))
        concerns: list[str] = result.get("concerns") or []
        revisions: dict[str, Any] = result.get("suggested_revisions") or {}

        self.log(
            f"[pre-review] approve={approved}, concerns={len(concerns)}, "
            f"suggested_positions={revisions.get('focus_positions', [])}"
        )
        return {
            "concerns": concerns,
            "suggested_revisions": revisions,
            "approve": approved,
        }

    def get_analysis_history(self) -> list[CriticAnalysis]:
        """지금까지 수행된 모든 CriticAnalysis 목록을 반환한다."""
        return list(self._analysis_history)

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """컨텍스트에서 랭킹 테이블과 QC 보고서를 받아 분석을 수행한다.

        context 키:
            - rank_table (RankTable): QCRanker 랭킹 결과
            - qc_report (QCReport): QCRanker QC 보고서
            - iteration (int): 현재 반복 번호
            - current_params (dict): 현재 실험 파라미터
            - silo_b_log_path (Path | str, optional): Silo B experiment_log.jsonl 경로.
                None이면 format_critic_prompt 내 기본 경로 사용.
                **pyrosetta_flow 경로만 허용.**

        Returns:
            {'status': str, 'critic_analysis': CriticAnalysis}
        """
        rank_table: RankTable = context["rank_table"]
        qc_report: QCReport = context["qc_report"]
        iteration: int = context.get("iteration", 1)
        current_params: dict[str, Any] = context.get("current_params", {})
        # Silo B experiment_log 경로: runner가 넘긴 Path 또는 문자열 → Path로 정규화
        _raw_log = context.get("silo_b_log_path")
        silo_b_log_path: Optional[Path] = Path(_raw_log) if _raw_log is not None else None
        # 2026-06-10: in-loop 선택성 피드백 (Δmargin>0 = native 초과 선택성). 없으면 None.
        selectivity_info: Optional[dict[str, Any]] = None
        if context.get("selectivity_leaderboard") or context.get("best_delta_margin") is not None:
            selectivity_info = {
                "leaderboard": context.get("selectivity_leaderboard", []),
                "best_delta_margin": context.get("best_delta_margin"),
            }

        # surrogate_map: {candidate_id: {half_life_h, admet_score, hc50}} — runner가 전달
        # (없으면 빈 dict, 데이터 없는 후보는 None으로 graceful)
        surrogate_map: dict[str, dict[str, Any]] = context.get("surrogate_map") or {}

        analysis = self.analyze_results(
            rank_table, qc_report, iteration, current_params,
            selectivity_info=selectivity_info,
            silo_b_log_path=silo_b_log_path,
            surrogate_map=surrogate_map,
        )
        return {"status": "ok", "critic_analysis": analysis}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_via_llm(
        self,
        rank_table: RankTable,
        qc_report: QCReport,
        iteration: int,
        current_params: dict[str, Any],
        selectivity_info: Optional[dict[str, Any]] = None,
        silo_b_log_path: Optional[Path] = None,
        surrogate_map: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Optional[CriticAnalysis]:
        """LLM을 통해 CriticAnalysis를 생성한다. 실패 시 None 반환."""
        _smap = surrogate_map or {}
        try:
            top_cands = []
            for c in rank_table.ranked_candidates[:5]:
                sur = _smap.get(c.candidate_id, {})
                top_cands.append({
                    "id": c.candidate_id,
                    "plddt": round(c.plddt_mean, 1),
                    "dock_score": round(c.dock_score, 2),
                    "ddg": round(c.ddg, 1),
                    # surrogate 값 (계산 X, 전달만) — None 허용(graceful)
                    "half_life_h": sur.get("half_life_h"),
                    "admet_score": sur.get("admet_score"),
                    "hc50": sur.get("hc50"),
                })
            rank_summary = {"top_candidates": top_cands, "total": len(rank_table.ranked_candidates)}
            qc_summary = {
                "total": qc_report.total_input,
                "passed": qc_report.passed_count,
                "failed": qc_report.total_input - qc_report.passed_count,
                "pass_rate": qc_report.pass_rate,
                "gate_results": dict(qc_report.failure_breakdown),
            }

            prompt = format_critic_prompt(
                iteration=iteration,
                rank_table_summary=rank_summary,
                qc_report_summary=qc_summary,
                current_params=current_params,
                selectivity_info=selectivity_info,
                silo_b_log_path=silo_b_log_path,
            )
            system = get_system_prompt("critic")
            result = self.llm_generate_json(prompt, system_prompt=system)
            if result is None:
                return None

            proposed = []
            for ch in result.get("parameter_changes", [])[:2]:
                proposed.append(ParameterChange(
                    parameter_name=ch.get("parameter_name", ""),
                    old_value=ch.get("old_value"),
                    new_value=ch.get("new_value"),
                    rationale=ch.get("rationale", ""),
                    expected_effect=ch.get("expected_effect", ""),
                    failure_type=result.get("failure_analysis", {}).get("primary_failure_type", ""),
                ))

            fa = result.get("failure_analysis", {})
            failure_summary = {
                k: v for k, v in fa.items()
                if k != "primary_failure_type" and isinstance(v, int)
            }

            return CriticAnalysis(
                iteration=iteration,
                failure_summary=failure_summary,
                structural_insights=result.get("overall_assessment", ""),
                proposed_changes=proposed,
                hypothesis=result.get("hypothesis_update", ""),
                top_candidate_ids=[c.candidate_id for c in rank_table.ranked_candidates[:5]],
            )
        except Exception as exc:
            self.log(f"LLM 분석 예외: {exc}", level="error")
            return None

    def _get_dominant_failures(
        self, failure_summary: dict[str, int], qc_report: QCReport
    ) -> list[str]:
        """빈도 기준으로 주요 실패 유형을 정렬하여 반환한다."""
        sorted_failures = sorted(failure_summary.items(), key=lambda x: x[1], reverse=True)

        # qc_report의 pass_rate가 매우 낮으면 low_plddt를 우선
        if qc_report.pass_rate < 0.1 and FailureType.LOW_PLDDT not in failure_summary:
            return [FailureType.LOW_PLDDT] + [k for k, _ in sorted_failures]

        return [k for k, _ in sorted_failures]

    def _generate_structural_insights(
        self, candidates: list[Candidate], qc_report: QCReport
    ) -> str:
        """랭킹 상위 후보와 QC 통계로 구조적 인사이트를 생성한다."""
        if not candidates:
            return "QC 통과 후보 없음. 파라미터 전면 재검토 필요."

        top = candidates[0]
        avg_plddt = sum(c.plddt_mean for c in candidates) / len(candidates)
        avg_ddg = sum(c.ddg for c in candidates) / len(candidates)

        insights = (
            f"통과 후보 {len(candidates)}개 중 최상위: {top.candidate_id} "
            f"(pLDDT={top.plddt_mean:.1f}, ddG={top.ddg:.2f}, "
            f"dock={top.dock_score:.2f}). "
            f"전체 평균: pLDDT={avg_plddt:.1f}, ddG={avg_ddg:.2f}. "
            f"통과율: {qc_report.pass_rate*100:.1f}%."
        )
        return insights

    @staticmethod
    def _compute_new_value(action: dict[str, Any], old_val: Any) -> Any:
        """액션 딕셔너리와 현재 값으로 새 값을 계산한다."""
        if old_val is None:
            return None

        direction = action.get("direction", "")

        if "fixed_delta" in action:
            delta = action["fixed_delta"]
            try:
                return type(old_val)(old_val + delta)
            except (TypeError, ValueError):
                return None

        if "delta_multiplier" in action:
            mult = action["delta_multiplier"]
            try:
                return type(old_val)(old_val * mult)
            except (TypeError, ValueError):
                return None

        if direction == "toggle":
            # docking_engine: diffdock <-> boltz2
            if old_val == "diffdock":
                return "boltz2"
            if old_val == "boltz2":
                return "diffdock"

        if direction == "update":
            # hotspot_res 등: 실제 LLM이 채워야 할 부분; 여기서는 마커 반환
            return f"<LLM_SUGGEST:{old_val}>"

        return None
