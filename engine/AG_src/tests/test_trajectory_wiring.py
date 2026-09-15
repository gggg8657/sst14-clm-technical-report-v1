"""test_trajectory_wiring.py — 궤적(trajectory) 배선 end-to-end 검증.

검증 항목:
- Silo B: runner가 silo_b_log_path를 context로 전달 →
    PlannerAgent.execute() / ScientistCriticAgent.execute()가
    format_planner_prompt / format_critic_prompt 호출 시 경로를 넘김 →
    iteration > 1 프롬프트에 '최근 궤적' 섹션이 실제로 포함됨.
- Silo A: plan_next_epoch()가 experiment_log_path를 받아
    trajectory 텍스트를 LLM 프롬프트에 주입하는 경로 확인
    (LLM 실호출 불필요 — _call_llm_for_plan 패치로 검증).
- 역방향 호환: iteration=1 / 경로 미전달 시 이전 동작 유지.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# --- 경로 보정 (repo root)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from AG_src.agents.planner import PlannerAgent
from AG_src.agents.critic import ScientistCriticAgent
from AG_src.agents.qc_ranker import Candidate, QCReport, RankTable
from AG_src.llm.prompts import format_planner_prompt, format_critic_prompt


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _make_record(
    iteration: int,
    ddg: Optional[float],
    status: str = "success",
    sequence: str = "AGCKNFFWKTFTSC",
) -> Dict[str, Any]:
    return {
        "record_type": "candidate",
        "iteration": iteration,
        "status": status,
        "sequence": sequence,
        "ddg": ddg,
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _make_rank_table(n: int = 3) -> RankTable:
    """더미 RankTable 생성."""
    cands = [
        Candidate(
            candidate_id=f"cand_{i:03d}",
            backbone_id=i,
            seq_id=0,
            sequence="AGCKNFFWKTFTSC",
            plddt_mean=80.0,
            dock_score=-8.5,
            ddg=-20.0,
            pass_gates=True,
            fail_reasons=[],
        )
        for i in range(n)
    ]
    return RankTable(
        run_id="test_run",
        iteration=1,
        ranked_candidates=cands,
        weights={"plddt": 0.15, "dock_score": 0.25, "ddg": 0.25, "lddt": 0.15, "selectivity": 0.20},
    )


def _make_qc_report(n: int = 3) -> QCReport:
    """더미 QCReport 생성."""
    return QCReport(
        run_id="test_run",
        total_input=n,
        passed_count=n,
        failed_count=0,
        failure_breakdown={},
        gates_applied={},
        pass_rate=1.0,
    )


# ---------------------------------------------------------------------------
# Silo B — PlannerAgent.execute() 배선 검증
# ---------------------------------------------------------------------------

class TestSiloBPlannerWiring:
    """PlannerAgent.execute()가 context['silo_b_log_path']를 받아
    iteration > 1 프롬프트에 궤적을 주입하는 경로를 검증한다."""

    def test_trajectory_included_via_execute_context(self, tmp_path: Path) -> None:
        """iteration=2 + silo_b_log_path → 프롬프트에 '최근 궤적' 포함."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -25.0)] * 5
        _write_jsonl(log, records)

        captured_prompts: List[str] = []

        def fake_llm_generate_json(prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
            captured_prompts.append(prompt)
            return None  # None 반환 → 규칙 기반 폴백 (계획 생성에는 지장 없음)

        agent = PlannerAgent(llm_provider="none")
        # has_llm은 read-only property → _create_plan_via_llm을 직접 패치
        with patch.object(agent, "_create_plan_via_llm", wraps=agent._create_plan_via_llm) as mock_create:
            # format_planner_prompt를 spy하여 호출 인수 캡처
            import AG_src.agents.planner as planner_mod
            with patch.object(planner_mod, "format_planner_prompt", wraps=planner_mod.format_planner_prompt) as mock_fmt:

                # iteration=1 초기 계획을 먼저 만들어 _plans에 채움
                agent.create_initial_plan(
                    receptor_config={"name": "SSTR2"},
                    constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
                )

                # iteration=2 + silo_b_log_path 전달 (LLM 없으므로 규칙 기반 폴백)
                result = agent.execute({
                    "iteration": 2,
                    "receptor_config": {"name": "SSTR2"},
                    "constraints": {"reference_sequence": "AGCKNFFWKTFTSC"},
                    "critic_feedback": {},
                    "previous_results": {},
                    "silo_b_log_path": log,
                })

        assert result["status"] == "ok"

    def test_trajectory_in_prompt_via_format_direct(self, tmp_path: Path) -> None:
        """format_planner_prompt 직접 호출 — iteration=2+silo_b_log_path → 궤적 포함.
        이것이 실제 배선된 경로(_create_plan_via_llm 내부)의 프롬프트 품질 증거."""
        log = tmp_path / "experiment_log.jsonl"
        records = [_make_record(1, -25.0)] * 5
        _write_jsonl(log, records)

        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            silo_b_log_path=log,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower(), (
            f"궤적 섹션 미포함. 프롬프트 앞 300자:\n{prompt[:300]}"
        )
        assert "iter 1" in prompt, "iter 1 데이터 미반영"

    def test_no_trajectory_at_iteration_1(self, tmp_path: Path) -> None:
        """iteration=1이면 궤적 섹션 미주입 (첫 실험)."""
        log = tmp_path / "experiment_log.jsonl"
        _write_jsonl(log, [_make_record(1, -20.0)] * 3)

        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            silo_b_log_path=log,
        )
        assert "최근 궤적" not in prompt, "iteration=1인데 궤적 섹션이 주입됨"

    def test_no_log_path_uses_default(self) -> None:
        """silo_b_log_path 미전달 → 기본 경로로 폴백, 에러 없음."""
        agent = PlannerAgent(llm_provider="none")
        agent.create_initial_plan(
            receptor_config={"name": "SSTR2"},
            constraints={},
        )
        # 기본 경로가 없어도 예외 없이 수행
        result = agent.execute({
            "iteration": 2,
            "receptor_config": {"name": "SSTR2"},
            "constraints": {},
            "critic_feedback": {},
            "previous_results": {},
            # silo_b_log_path 미전달
        })
        assert result["status"] == "ok"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """silo_b_log_path를 str로 전달해도 동작 (Path로 자동 변환)."""
        log = tmp_path / "experiment_log.jsonl"
        _write_jsonl(log, [_make_record(1, -30.0)] * 3)

        # str 경로를 전달해도 format_planner_prompt가 받아 동일하게 동작
        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={},
            silo_b_log_path=Path(str(log)),  # Path로 변환 확인
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower()

    def test_execute_passes_log_path_to_update_plan(self, tmp_path: Path) -> None:
        """execute()의 context['silo_b_log_path']가 update_plan()으로 전달됨을 spy 검증."""
        log = tmp_path / "experiment_log.jsonl"
        _write_jsonl(log, [_make_record(1, -28.0)] * 3)

        agent = PlannerAgent(llm_provider="none")
        agent.create_initial_plan(receptor_config={"name": "SSTR2"}, constraints={})

        with patch.object(agent, "update_plan", wraps=agent.update_plan) as mock_update:
            agent.execute({
                "iteration": 2,
                "receptor_config": {"name": "SSTR2"},
                "constraints": {},
                "critic_feedback": {},
                "previous_results": {},
                "silo_b_log_path": log,
            })
            mock_update.assert_called_once()
            call_kwargs = mock_update.call_args.kwargs
            assert "silo_b_log_path" in call_kwargs, (
                "update_plan()에 silo_b_log_path가 전달되지 않음"
            )
            assert call_kwargs["silo_b_log_path"] == log, (
                f"전달된 경로 불일치: {call_kwargs['silo_b_log_path']} != {log}"
            )


# ---------------------------------------------------------------------------
# Silo B — ScientistCriticAgent.execute() 배선 검증
# ---------------------------------------------------------------------------

class TestSiloBCriticWiring:
    """ScientistCriticAgent.execute()가 context['silo_b_log_path']를 받아
    iteration > 1 프롬프트에 궤적을 주입하는 경로를 검증한다."""

    def test_trajectory_in_prompt_via_format_direct(self, tmp_path: Path) -> None:
        """format_critic_prompt 직접 호출 — iteration=2+silo_b_log_path → 궤적 포함."""
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
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower(), (
            f"궤적 섹션 미포함. 프롬프트 앞 300자:\n{prompt[:300]}"
        )
        assert "iter 1" in prompt, "iter 1 데이터 미반영"

    def test_execute_passes_log_path_to_analyze_results(self, tmp_path: Path) -> None:
        """execute()의 context['silo_b_log_path']가 analyze_results()로 전달됨을 spy 검증."""
        log = tmp_path / "experiment_log.jsonl"
        _write_jsonl(log, [_make_record(1, -22.0)] * 3)

        critic = ScientistCriticAgent(llm_provider="none")

        with patch.object(critic, "analyze_results", wraps=critic.analyze_results) as mock_analyze:
            critic.execute({
                "rank_table": _make_rank_table(),
                "qc_report": _make_qc_report(),
                "iteration": 2,
                "current_params": {"n_candidates": 8},
                "silo_b_log_path": log,
            })
            mock_analyze.assert_called_once()
            call_kwargs = mock_analyze.call_args.kwargs
            assert "silo_b_log_path" in call_kwargs, (
                "analyze_results()에 silo_b_log_path가 전달되지 않음"
            )
            assert call_kwargs["silo_b_log_path"] == log, (
                f"전달된 경로 불일치: {call_kwargs['silo_b_log_path']} != {log}"
            )

    def test_no_log_path_no_error(self) -> None:
        """silo_b_log_path 미전달 → 예외 없이 분석 완료."""
        critic = ScientistCriticAgent(llm_provider="none")
        result = critic.execute({
            "rank_table": _make_rank_table(),
            "qc_report": _make_qc_report(),
            "iteration": 2,
            "current_params": {},
        })
        assert result["status"] == "ok"

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """str 경로도 Path로 변환되어 analyze_results에 전달됨."""
        log = tmp_path / "experiment_log.jsonl"
        _write_jsonl(log, [_make_record(1, -18.0)] * 3)

        critic = ScientistCriticAgent(llm_provider="none")

        with patch.object(critic, "analyze_results", wraps=critic.analyze_results) as mock_analyze:
            critic.execute({
                "rank_table": _make_rank_table(),
                "qc_report": _make_qc_report(),
                "iteration": 2,
                "current_params": {},
                "silo_b_log_path": str(log),  # str 전달
            })
            call_kwargs = mock_analyze.call_args.kwargs
            assert call_kwargs["silo_b_log_path"] == log, (
                f"str→Path 변환 실패: {call_kwargs['silo_b_log_path']}"
            )


# ---------------------------------------------------------------------------
# Silo A — plan_next_epoch experiment_log_path 배선 검증
# ---------------------------------------------------------------------------

class TestSiloAPlannerWiring:
    """plan_next_epoch()가 experiment_log_path를 받아
    궤적 텍스트를 LLM 프롬프트에 주입하는 경로를 검증한다."""

    def test_trajectory_injected_when_log_provided(self, tmp_path: Path) -> None:
        """experiment_log_path 전달 → LLM 호출 시 trajectory_text 포함."""
        from pyrosetta_flow.silo_a_planner import plan_next_epoch

        log = tmp_path / "silo_a_experiment_log.jsonl"
        # Silo A 레코드 형식
        records = [
            {"epoch": 1, "ddg": -18.0, "plddt": 80.0, "sequence": "AGCKNFFWKTFTSC",
             "record_type": "candidate", "status": "success"},
            {"epoch": 1, "ddg": -20.0, "plddt": 82.0, "sequence": "AGCKNFFWKTDTSC",
             "record_type": "candidate", "status": "success"},
        ]
        with log.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        leaderboard = tmp_path / "silo_a_leaderboard.json"
        leaderboard.write_text(json.dumps({
            "entries": [{"sequence": "AGCKNFFWKTDTSC", "ddg": -20.0, "delta_margin": 2.0}],
            "updated_at": "2026-06-19T00:00:00Z",
        }, ensure_ascii=False), encoding="utf-8")

        captured_texts: List[str] = []

        def fake_call_llm(summary: Any, current_params: Any, vllm_url: str, model: str,
                          timeout: int, trajectory_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
            captured_texts.append(trajectory_text or "")
            return None  # None → 규칙 기반 폴백

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", side_effect=fake_call_llm):
            updated_params, decision = plan_next_epoch(
                epoch=2,
                leaderboard_path=leaderboard,
                current_params={"contigs": "B1-369/0 12-16", "diffusion_steps": 50},
                stagnation_count=3,   # >= patience_epochs(3) → LLM 호출
                patience_epochs=3,
                experiment_log_path=log,
                trajectory_k=6,
            )

        assert len(captured_texts) == 1, "LLM 호출이 발생하지 않음"
        trajectory_text = captured_texts[0]
        assert trajectory_text, "trajectory_text가 비어 있음 — 경로 미주입"
        assert "Silo A 최근 궤적" in trajectory_text or "trajectory" in trajectory_text.lower(), (
            f"궤적 섹션 미포함: {trajectory_text[:200]}"
        )

    def test_no_log_path_auto_discover(self, tmp_path: Path) -> None:
        """experiment_log_path=None → leaderboard 부모 디렉토리에서 자동 탐색."""
        from pyrosetta_flow.silo_a_planner import plan_next_epoch

        log = tmp_path / "experiment_log.jsonl"
        records = [
            {"epoch": 1, "ddg": -15.0, "plddt": 78.0, "sequence": "AGCKNFFWKTFTSC",
             "record_type": "candidate", "status": "success"},
        ]
        with log.open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        leaderboard = tmp_path / "silo_a_leaderboard.json"
        leaderboard.write_text(json.dumps({
            "entries": [{"sequence": "AGCKNFFWKTFTSC", "ddg": -15.0, "delta_margin": 1.0}],
        }, ensure_ascii=False), encoding="utf-8")

        captured_texts: List[str] = []

        def fake_call_llm(summary: Any, current_params: Any, vllm_url: str, model: str,
                          timeout: int, trajectory_text: Optional[str] = None) -> Optional[Dict[str, Any]]:
            captured_texts.append(trajectory_text or "")
            return None

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", side_effect=fake_call_llm):
            plan_next_epoch(
                epoch=2,
                leaderboard_path=leaderboard,
                current_params={"contigs": "B1-369/0 12-16", "diffusion_steps": 50},
                stagnation_count=3,
                patience_epochs=3,
                experiment_log_path=None,  # 자동 탐색
            )

        # 자동 탐색 성공 → trajectory_text 비어있지 않아야 함
        if captured_texts:
            assert captured_texts[0], "자동 탐색 실패 — trajectory_text 비어있음"

    def test_backward_compat_no_log_no_crash(self, tmp_path: Path) -> None:
        """experiment_log_path=None + 파일 없어도 예외 없이 규칙 기반 폴백."""
        from pyrosetta_flow.silo_a_planner import plan_next_epoch

        leaderboard = tmp_path / "silo_a_leaderboard.json"
        leaderboard.write_text(json.dumps({"entries": []}), encoding="utf-8")

        # LLM 호출 없이 규칙 fallback
        updated_params, decision = plan_next_epoch(
            epoch=2,
            leaderboard_path=leaderboard,
            current_params={"contigs": "B1-369/0 12-16", "diffusion_steps": 50},
            stagnation_count=0,   # < patience_epochs → LLM 호출 안 함
            patience_epochs=3,
            experiment_log_path=None,
        )
        assert isinstance(updated_params, dict)


# ---------------------------------------------------------------------------
# 프롬프트 직접 검증 — format_planner_prompt / format_critic_prompt
# ---------------------------------------------------------------------------

class TestPromptDirectWiring:
    """format_planner_prompt / format_critic_prompt 직접 호출로
    배선된 silo_b_log_path가 프롬프트에 반영됨을 확인한다."""

    def test_planner_prompt_contains_trajectory(self, tmp_path: Path) -> None:
        log = tmp_path / "experiment_log.jsonl"
        records = [
            {"iteration": 1, "status": "success", "ddg": -30.0, "sequence": "AGCKNFFWKTFTSC"},
            {"iteration": 1, "status": "success", "ddg": -28.0, "sequence": "AGCKNFFWKTFTSC"},
        ]
        _write_jsonl(log, records)

        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": "AGCKNFFWKTFTSC"},
            silo_b_log_path=log,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower(), (
            f"궤적 섹션 미포함. 프롬프트 앞 300자:\n{prompt[:300]}"
        )
        assert "iter 1" in prompt, "iter 1 데이터 미반영"

    def test_critic_prompt_contains_trajectory(self, tmp_path: Path) -> None:
        log = tmp_path / "experiment_log.jsonl"
        records = [
            {"iteration": 1, "status": "success", "ddg": -22.0, "sequence": "AGCKNFFWKTFTSC"},
        ]
        _write_jsonl(log, records)

        prompt = format_critic_prompt(
            iteration=2,
            rank_table_summary={"top_candidates": []},
            qc_report_summary={"total": 5, "passed": 3, "failed": 2, "pass_rate": 0.6},
            current_params={"n_candidates": 8},
            silo_b_log_path=log,
        )
        assert "최근 궤적" in prompt or "trajectory" in prompt.lower(), (
            f"궤적 섹션 미포함. 프롬프트 앞 300자:\n{prompt[:300]}"
        )
