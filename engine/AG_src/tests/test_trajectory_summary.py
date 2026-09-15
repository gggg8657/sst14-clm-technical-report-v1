"""test_trajectory_summary.py — format_trajectory_summary 단위 테스트.

검증 항목:
- 개선/정체/악화 패턴이 요약에 반영되는지
- 빈 로그·소량 로그에서도 안 깨지는지
- 출력이 토큰 바운드(8 iteration 제한) 내인지
- Silo A/B 분리 — Silo B 경로만 사용
- 환각 없음 — 실 데이터만 반영
- format_planner_prompt / format_critic_prompt 시그니처 하위 호환
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from AG_src.llm.prompts import (
    format_critic_prompt,
    format_planner_prompt,
    format_trajectory_summary,
)


# ---------------------------------------------------------------------------
# 헬퍼: 모의 실험 레코드 생성
# ---------------------------------------------------------------------------

def _make_record(
    iteration: int,
    ddg: Optional[float],
    status: str = "success",
    sequence: str = "AGCKNFFWKTFTSC",
    sel_margin: Optional[float] = None,
    delta_margin: Optional[float] = None,
    mutation_source: str = "llm_guided",
) -> Dict[str, Any]:
    return {
        "record_type": "candidate",
        "iteration": iteration,
        "status": status,
        "sequence": sequence,
        "ddg": ddg,
        "selectivity_margin": sel_margin,
        "delta_margin": delta_margin,
        "mutation_source": mutation_source,
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 기본 동작 테스트
# ---------------------------------------------------------------------------

class TestFormatTrajectorySummary:

    def test_empty_list_returns_no_data(self, tmp_path: Path) -> None:
        """빈 입력 + 존재하지 않는 로그 경로 → '데이터 없음' 반환 (환각 금지)."""
        nonexistent = tmp_path / "no_log.jsonl"
        result = format_trajectory_summary(None, silo_b_log_path=nonexistent)
        assert "데이터 없음" in result or "없음" in result

    def test_empty_log_file(self, tmp_path: Path) -> None:
        """빈 JSONL 파일 → '데이터 없음'."""
        log = tmp_path / "experiment_log.jsonl"
        log.write_text("", encoding="utf-8")
        result = format_trajectory_summary(None, silo_b_log_path=log)
        assert "없음" in result

    def test_nonexistent_log(self, tmp_path: Path) -> None:
        """존재하지 않는 파일 → '없음' 반환, 예외 없음."""
        log = tmp_path / "no_such_file.jsonl"
        result = format_trajectory_summary(None, silo_b_log_path=log)
        assert "없음" in result

    def test_single_iteration(self, tmp_path: Path) -> None:
        """단일 iteration → 에러 없이 요약 반환."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -30.0) for _ in range(5)]
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "iter 1" in result
        assert "ddG=-30" in result

    def test_improvement_marked(self, tmp_path: Path) -> None:
        """ddG 개선 시 ↑개선 표시 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = (
            [_make_record(1, -20.0)] * 3 +
            [_make_record(2, -25.0)] * 3  # 개선
        )
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "↑개선" in result

    def test_stagnation_marked(self, tmp_path: Path) -> None:
        """정체 시 →정체 표시."""
        log = tmp_path / "experiment_log.jsonl"
        records = (
            [_make_record(1, -20.0)] * 3 +
            [_make_record(2, -20.05)] * 3  # 정체 (0.1 미만)
        )
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "→정체" in result

    def test_stagnation_pattern_detected(self, tmp_path: Path) -> None:
        """3+ iteration 연속 정체 시 [패턴] 경고."""
        log = tmp_path / "experiment_log.jsonl"
        records = []
        for it in range(1, 6):
            records += [_make_record(it, -20.0)] * 3  # 모두 동일 ddG
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        # [패턴] 또는 정체 관련 메시지
        assert "[패턴]" in result or "정체" in result

    def test_selectivity_included(self, tmp_path: Path) -> None:
        """selectivity_margin이 있으면 요약에 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = [
            _make_record(1, -30.0, sel_margin=5.0, delta_margin=2.5)
        ] * 3
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "margin" in result or "Δmargin" in result

    def test_k_limits_iterations(self, tmp_path: Path) -> None:
        """k=3이면 최근 3 iteration만 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = []
        for it in range(1, 11):  # 10 iterations
            records += [_make_record(it, -float(it) * 5)] * 3
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=3, silo_b_log_path=log)
        # iter 1~7은 없어야 함
        for it in range(1, 8):
            assert f"iter {it}:" not in result
        # iter 8, 9, 10은 있어야 함
        for it in range(8, 11):
            assert f"iter {it}:" in result

    def test_k_capped_at_8(self, tmp_path: Path) -> None:
        """k=100을 줘도 최대 8 iteration만."""
        log = tmp_path / "experiment_log.jsonl"
        records = []
        for it in range(1, 15):
            records += [_make_record(it, -float(it) * 3)] * 2
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=100, silo_b_log_path=log)
        # 최대 8 iteration 포함 — "iter N:" 패턴 카운트
        import re
        found = re.findall(r"iter \d+:", result)
        assert len(found) <= 8

    def test_output_bounded_size(self, tmp_path: Path) -> None:
        """출력 크기가 과도하게 크지 않음 (토큰 바운드)."""
        log = tmp_path / "experiment_log.jsonl"
        records = []
        for it in range(1, 20):
            records += [_make_record(it, -float(it) * 2, sequence="AGCKNFFWKTFTSC")] * 5
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        # 대략 2000자 이하 (프롬프트 컨텍스트 바운드)
        assert len(result) < 3000

    def test_failed_records_excluded(self, tmp_path: Path) -> None:
        """status=failed 레코드는 ddG 통계에서 제외."""
        log = tmp_path / "experiment_log.jsonl"
        records = [
            _make_record(1, 999.0, status="failed"),
            _make_record(1, 999.0, status="failed"),
            _make_record(1, -20.0, status="success"),
        ]
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        # best ddG는 -20.0이어야 함 (999 제외)
        assert "ddG=-20" in result

    def test_list_records_input(self) -> None:
        """records 직접 전달(List 형식) 동작."""
        records = [
            _make_record(1, -25.0),
            _make_record(1, -28.0),
            _make_record(2, -32.0),
        ]
        result = format_trajectory_summary(records, k=6)
        assert "iter 1" in result or "iter 2" in result

    def test_grouped_records_input(self) -> None:
        """이미 그룹화된 레코드 리스트 입력."""
        grouped = [
            {"iteration": 1, "records": [_make_record(1, -20.0), _make_record(1, -22.0)]},
            {"iteration": 2, "records": [_make_record(2, -25.0)]},
        ]
        result = format_trajectory_summary(grouped, k=6)
        assert "iter 1" in result
        assert "iter 2" in result

    def test_no_hallucination_no_ddg(self, tmp_path: Path) -> None:
        """ddg=None 레코드만 있으면 N/A 표시 (수치 지어내기 금지)."""
        log = tmp_path / "experiment_log.jsonl"
        records = [{"iteration": 1, "status": "success", "ddg": None, "sequence": "AAA"}]
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "N/A" in result or "없음" in result

    def test_header_present(self, tmp_path: Path) -> None:
        """출력에 섹션 헤더가 있어야 함."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -20.0)]
        _write_jsonl(log, records)
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        assert "최근 궤적" in result or "trajectory" in result.lower()

    def test_corrupt_lines_tolerated(self, tmp_path: Path) -> None:
        """손상된 라인이 섞여도 정상 파싱 라인은 처리."""
        log = tmp_path / "experiment_log.jsonl"
        with log.open("w", encoding="utf-8") as fh:
            fh.write("{invalid json}\n")
            fh.write(json.dumps(_make_record(1, -18.0)) + "\n")
            fh.write("{also bad}\n")
        result = format_trajectory_summary(None, k=6, silo_b_log_path=log)
        # 정상 레코드가 처리됐어야 함
        assert "iter 1" in result


# ---------------------------------------------------------------------------
# format_planner_prompt 하위 호환 테스트
# ---------------------------------------------------------------------------

class TestFormatPlannerPromptWithTrajectory:

    def test_backward_compat_no_trajectory_args(self) -> None:
        """기존 시그니처(trajectory 인수 없음)도 동작해야 함."""
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
        )
        assert "Iteration 1" in prompt

    def test_trajectory_injected_when_iteration_gt_1(self, tmp_path: Path) -> None:
        """iteration > 1이면 궤적 섹션이 프롬프트에 포함됨."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -20.0)] * 3
        _write_jsonl(log, records)
        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            silo_b_log_path=log,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()

    def test_no_trajectory_at_iteration_1(self, tmp_path: Path) -> None:
        """iteration=1이면 궤적 섹션 없음 (데이터 없는 첫 iteration)."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -20.0)] * 3
        _write_jsonl(log, records)
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            silo_b_log_path=log,
        )
        # iteration=1이면 궤적 미주입
        assert "최근 궤적" not in prompt

    def test_trajectory_records_param_used(self) -> None:
        """trajectory_records 파라미터 직접 전달 시 반영."""
        records = [_make_record(3, -30.0), _make_record(3, -32.0)]
        prompt = format_planner_prompt(
            iteration=4,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            trajectory_records=records,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()

    def test_previous_results_and_trajectory_coexist(self, tmp_path: Path) -> None:
        """previous_results 섹션과 궤적 섹션이 함께 포함됨."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -20.0)] * 3
        _write_jsonl(log, records)
        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            previous_results={"best_ddg": -20.0, "n_passed": 3},
            silo_b_log_path=log,
        )
        assert "Previous Iteration Results" in prompt
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()

    def test_pyrosetta_mode_with_trajectory(self, tmp_path: Path) -> None:
        """pyrosetta_only 모드에서도 궤적 섹션 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(2, -35.0)] * 5
        _write_jsonl(log, records)
        prompt = format_planner_prompt(
            iteration=3,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            planner_mode="pyrosetta_only",
            silo_b_log_path=log,
        )
        assert "MUST" in prompt
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()


# ---------------------------------------------------------------------------
# format_critic_prompt 하위 호환 테스트
# ---------------------------------------------------------------------------

class TestFormatCriticPromptWithTrajectory:

    def test_backward_compat_no_trajectory_args(self) -> None:
        """기존 시그니처(trajectory 인수 없음)도 동작."""
        prompt = format_critic_prompt(
            iteration=1,
            rank_table_summary={"top_candidates": []},
            qc_report_summary={"total": 5, "passed": 3, "failed": 2, "pass_rate": 0.6},
            current_params={"n_candidates": 8},
        )
        assert "Iteration 1" in prompt

    def test_trajectory_in_critic_iteration_gt_1(self, tmp_path: Path) -> None:
        """Critic 프롬프트도 iteration > 1이면 궤적 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -22.0)] * 4
        _write_jsonl(log, records)
        prompt = format_critic_prompt(
            iteration=2,
            rank_table_summary={"top_candidates": []},
            qc_report_summary={"total": 5, "passed": 3, "failed": 2, "pass_rate": 0.6},
            current_params={"n_candidates": 8},
            silo_b_log_path=log,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()

    def test_selectivity_and_trajectory_coexist(self, tmp_path: Path) -> None:
        """선택성 정보와 궤적이 함께 포함됨."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -25.0, sel_margin=3.0)] * 3
        _write_jsonl(log, records)
        prompt = format_critic_prompt(
            iteration=2,
            rank_table_summary={"top_candidates": []},
            qc_report_summary={"total": 5, "passed": 3, "failed": 2, "pass_rate": 0.6},
            current_params={},
            selectivity_info={"leaderboard": [], "best_delta_margin": 1.5},
            silo_b_log_path=log,
        )
        assert "SELECTIVITY" in prompt
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()
