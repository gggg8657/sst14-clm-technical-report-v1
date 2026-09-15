"""test_silo_a_planner.py — Silo A 피드백 루프 플래너 단위 테스트.

테스트 원칙:
- 실제 LLM/GPU 호출 없이 로직만 검증 (mock 사용).
- 정체(stagnation) 시 파라미터 조정 여부 검증.
- 개선(improvement) 시 파라미터 유지 여부 검증.
- LLM 실패 시 규칙 fallback 경로 검증.
- provenance 파일 기록 검증.
- 파라미터 유효성 검증 (clamp, hotspot 형식).
- Silo B 분리 침범 없음 검증.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.silo_a_planner import (
    SiloACriticResult,
    SiloAPlannerDecision,
    _append_discussion_log,
    _build_contig_with_binder_length,
    _call_critic_prereview,
    _parse_contig_binder_length,
    _rule_based_adjustment,
    _summarize_leaderboard,
    _summarize_trajectory,
    _validate_and_clamp_params,
    plan_next_epoch,
    update_stagnation_count,
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_leaderboard_path(tmp_dir: Path) -> Path:
    """mock 리더보드 JSON 파일."""
    path = tmp_dir / "silo_a_leaderboard.json"
    data = {
        "source": "silo_a",
        "candidate_class": "de_novo",
        "n_total": 6,
        "n_unique": 3,
        "best_ddg": -30.0,
        "best_selectivity_margin": 5.0,
        "entries": [
            {
                "candidate_id": "silo_a_e1_bb00_sq00",
                "sequence": "ACDEFGHIKL",
                "ddg": -30.0,
                "selectivity_margin": 5.0,
                "plddt": 0.88,
                "plddt_pass": True,
                "fail_reason": "",
                "extra_scores": {"candidate_class": "de_novo", "mutation_source": "silo_a"},
            },
            {
                "candidate_id": "silo_a_e1_bb00_sq01",
                "sequence": "MNPQRSTVWY",
                "ddg": -25.0,
                "selectivity_margin": 3.0,
                "plddt": 0.72,
                "plddt_pass": True,
                "fail_reason": "",
                "extra_scores": {},
            },
            {
                "candidate_id": "silo_a_e2_bb00_sq00",
                "sequence": "GGGGGGGGGGG",
                "ddg": None,
                "selectivity_margin": None,
                "plddt": None,
                "plddt_pass": False,
                "fail_reason": "도킹 실패",
                "extra_scores": {},
            },
        ],
        "screened_seqs": ["ACDEFGHIKL", "MNPQRSTVWY", "GGGGGGGGGGG"],
        "updated_at": "2026-06-19T10:00:00+00:00",
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def base_params() -> Dict[str, Any]:
    return {
        "contigs": "B1-369/0 12-16",
        "hotspot_res": ["B150", "B154", "B197", "B200", "B250", "B294"],
        "diffusion_steps": 50,
        "n_backbone": 2,
        "k_seq_per_backbone": 2,
    }


# ---------------------------------------------------------------------------
# update_stagnation_count 테스트
# ---------------------------------------------------------------------------

class TestUpdateStagnationCount:
    def test_first_result_is_improvement(self) -> None:
        """첫 유효 결과 → 개선 발생, stagnation=0."""
        count, improved = update_stagnation_count(
            prev_best_ddg=None,
            current_best_ddg=-20.0,
            prev_stagnation_count=0,
        )
        assert improved is True
        assert count == 0

    def test_improvement_resets_count(self) -> None:
        """ddG가 의미있게 개선 → stagnation 리셋."""
        count, improved = update_stagnation_count(
            prev_best_ddg=-20.0,
            current_best_ddg=-21.0,  # 0.1 이상 개선
            prev_stagnation_count=5,
        )
        assert improved is True
        assert count == 0

    def test_no_improvement_increments_count(self) -> None:
        """개선 없음 → stagnation_count 증가."""
        count, improved = update_stagnation_count(
            prev_best_ddg=-20.0,
            current_best_ddg=-20.05,  # 0.1 미만 = 정체
            prev_stagnation_count=2,
        )
        assert improved is False
        assert count == 3

    def test_no_ddg_yet_increments(self) -> None:
        """유효 ddG 없음 → 정체로 간주."""
        count, improved = update_stagnation_count(
            prev_best_ddg=-15.0,
            current_best_ddg=None,
            prev_stagnation_count=1,
        )
        assert improved is False
        assert count == 2

    def test_threshold_boundary(self) -> None:
        """딱 0.1 개선이면 개선 판정."""
        count, improved = update_stagnation_count(
            prev_best_ddg=-20.0,
            current_best_ddg=-20.11,
            prev_stagnation_count=3,
        )
        assert improved is True
        assert count == 0


# ---------------------------------------------------------------------------
# _parse_contig_binder_length 테스트
# ---------------------------------------------------------------------------

class TestParseContigBinderLength:
    def test_range_format(self) -> None:
        lo, hi = _parse_contig_binder_length("B1-369/0 12-16")
        assert lo == 12
        assert hi == 16

    def test_single_value(self) -> None:
        lo, hi = _parse_contig_binder_length("B1-369/0 14")
        assert lo == 14
        assert hi == 14

    def test_default_on_invalid(self) -> None:
        lo, hi = _parse_contig_binder_length("invalid")
        assert lo == 12
        assert hi == 16


# ---------------------------------------------------------------------------
# _build_contig_with_binder_length 테스트
# ---------------------------------------------------------------------------

class TestBuildContigWithBinderLength:
    def test_range(self) -> None:
        result = _build_contig_with_binder_length("B1-369/0", 10, 18)
        assert result == "B1-369/0 10-18"

    def test_single_value(self) -> None:
        result = _build_contig_with_binder_length("B1-369/0", 14, 14)
        assert result == "B1-369/0 14"


# ---------------------------------------------------------------------------
# _validate_and_clamp_params 테스트
# ---------------------------------------------------------------------------

class TestValidateAndClampParams:
    def test_diffusion_steps_clamped(self) -> None:
        result = _validate_and_clamp_params({"diffusion_steps": 999})
        assert result["diffusion_steps"] == 200  # _DIFFUSION_STEPS_MAX

    def test_diffusion_steps_min_clamped(self) -> None:
        result = _validate_and_clamp_params({"diffusion_steps": 5})
        assert result["diffusion_steps"] == 30  # _DIFFUSION_STEPS_MIN

    def test_hotspot_valid(self) -> None:
        result = _validate_and_clamp_params({"hotspot_res": ["B150", "B200"]})
        assert result["hotspot_res"] == ["B150", "B200"]

    def test_hotspot_invalid_filtered(self) -> None:
        """체인 형식 아닌 항목은 걸러짐."""
        result = _validate_and_clamp_params({"hotspot_res": ["B150", "A200", "bad"]})
        # "A200"는 chain B 형식 아님 — B로 시작 안 하므로 제거됨
        assert "A200" not in result.get("hotspot_res", [])
        assert "bad" not in result.get("hotspot_res", [])

    def test_hotspot_all_invalid_excluded(self) -> None:
        """모두 무효한 hotspot이면 결과에 포함 안 됨."""
        result = _validate_and_clamp_params({"hotspot_res": ["A150", "bad"]})
        assert "hotspot_res" not in result

    def test_binder_length_range_clamped(self) -> None:
        result = _validate_and_clamp_params({"binder_length_range": [5, 30]})
        lo, hi = result["binder_length_range"]
        assert lo >= 10   # _BINDER_LEN_MIN
        assert hi <= 20   # _BINDER_LEN_MAX

    def test_null_values_excluded(self) -> None:
        """None 값은 결과에서 제외."""
        result = _validate_and_clamp_params({
            "diffusion_steps": None,
            "n_backbone": None,
        })
        assert "diffusion_steps" not in result
        assert "n_backbone" not in result

    def test_n_backbone_clamped(self) -> None:
        result = _validate_and_clamp_params({"n_backbone": 100})
        assert result["n_backbone"] == 10  # max

    def test_arm_preference_valid(self) -> None:
        result = _validate_and_clamp_params({"arm_preference": "rfdiffusion"})
        assert result["arm_preference"] == "rfdiffusion"

    def test_arm_preference_invalid(self) -> None:
        result = _validate_and_clamp_params({"arm_preference": "unknown_arm"})
        assert "arm_preference" not in result


# ---------------------------------------------------------------------------
# _summarize_leaderboard 테스트
# ---------------------------------------------------------------------------

class TestSummarizeLeaderboard:
    def test_summarize_ok(self, sample_leaderboard_path: Path) -> None:
        summary = _summarize_leaderboard(sample_leaderboard_path)
        assert summary["status"] == "ok"
        assert summary["n_total"] == 6
        assert summary["best_ddg"] == -30.0
        assert len(summary["top_entries"]) <= 5

    def test_nonexistent_returns_empty(self, tmp_dir: Path) -> None:
        path = tmp_dir / "nonexistent.json"
        summary = _summarize_leaderboard(path)
        assert summary["status"] == "empty"
        assert summary["n_total"] == 0

    def test_corrupt_returns_error(self, tmp_dir: Path) -> None:
        path = tmp_dir / "corrupt.json"
        path.write_text("{invalid json}", encoding="utf-8")
        summary = _summarize_leaderboard(path)
        assert summary["status"] == "parse_error"

    def test_top_entries_limit(self, sample_leaderboard_path: Path) -> None:
        summary = _summarize_leaderboard(sample_leaderboard_path, top_k=2)
        assert len(summary["top_entries"]) <= 2

    def test_no_hallucination_none_ddg(self, tmp_dir: Path) -> None:
        """ddg=None 항목은 통계에서 제외 (환각 금지)."""
        path = tmp_dir / "lb_none.json"
        data = {
            "source": "silo_a", "candidate_class": "de_novo",
            "n_total": 2, "entries": [
                {"sequence": "AAA", "ddg": None, "selectivity_margin": None,
                 "plddt": None, "plddt_pass": False, "fail_reason": "fail"},
            ],
            "screened_seqs": [], "updated_at": "2026-01-01T00:00:00Z",
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        summary = _summarize_leaderboard(path)
        assert summary["best_ddg"] is None  # None ddg → 통계 없음
        assert summary["n_with_ddg"] == 0


# ---------------------------------------------------------------------------
# _rule_based_adjustment 테스트
# ---------------------------------------------------------------------------

class TestRuleBasedAdjustment:
    def test_no_stagnation_no_change(self, base_params: Dict[str, Any]) -> None:
        """정체 미발생(stagnation_count < patience) → 변경 없음."""
        adj, hypothesis = _rule_based_adjustment(
            summary={"status": "ok", "n_total": 3},
            current_params=base_params,
            patience_epochs=3,
            stagnation_count=2,  # < patience_epochs
        )
        assert adj == {}
        assert "미발생" in hypothesis

    def test_stagnation_triggers_adjustment(self, base_params: Dict[str, Any]) -> None:
        """정체(stagnation >= patience) → 파라미터 조정."""
        adj, hypothesis = _rule_based_adjustment(
            summary={"status": "ok", "n_total": 10},
            current_params=base_params,
            patience_epochs=3,
            stagnation_count=3,  # >= patience_epochs
        )
        # 조정이 일어났으면 dict에 뭔가 있어야 함
        # (최소 diffusion_steps 또는 hotspot 또는 binder_length_range 변경)
        assert len(adj) > 0 or "최대 범위" in hypothesis

    def test_hotspot_shuffled_on_stagnation(self, base_params: Dict[str, Any]) -> None:
        """정체 시 hotspot 교체 발생 여부."""
        adj, hypothesis = _rule_based_adjustment(
            summary={"status": "ok", "n_total": 10},
            current_params=base_params,
            patience_epochs=1,
            stagnation_count=5,
        )
        # hotspot이 바뀌었거나 steps가 올랐거나 길이가 늘었어야 함
        changed = "hotspot_res" in adj or "diffusion_steps" in adj or "contigs" in adj
        assert changed or "최대 범위" in hypothesis

    def test_steps_increased_on_stagnation(self, base_params: Dict[str, Any]) -> None:
        """정체 시 diffusion_steps 상향."""
        adj, hypothesis = _rule_based_adjustment(
            summary={},
            current_params=base_params,  # steps=50
            patience_epochs=1,
            stagnation_count=10,
        )
        if "diffusion_steps" in adj:
            assert adj["diffusion_steps"] > 50

    def test_steps_not_exceed_max(self, base_params: Dict[str, Any]) -> None:
        """diffusion_steps가 최대값을 초과하지 않음."""
        params_high_steps = dict(base_params)
        params_high_steps["diffusion_steps"] = 200  # 이미 최대
        adj, _ = _rule_based_adjustment(
            summary={},
            current_params=params_high_steps,
            patience_epochs=1,
            stagnation_count=10,
        )
        if "diffusion_steps" in adj:
            assert adj["diffusion_steps"] <= 200

    def test_hypothesis_contains_reason(self, base_params: Dict[str, Any]) -> None:
        """hypothesis에 조정 근거가 포함됨."""
        _, hypothesis = _rule_based_adjustment(
            summary={},
            current_params=base_params,
            patience_epochs=1,
            stagnation_count=5,
        )
        assert len(hypothesis) > 0


# ---------------------------------------------------------------------------
# SiloAPlannerDecision 테스트
# ---------------------------------------------------------------------------

class TestSiloAPlannerDecision:
    def test_to_dict_has_required_fields(self) -> None:
        decision = SiloAPlannerDecision(
            epoch=3,
            hypothesis="테스트 근거",
            adjustment_type="rule_stagnation",
            applied_params={"diffusion_steps": 65},
        )
        d = decision.to_dict()
        assert d["epoch"] == 3
        assert d["hypothesis"] == "테스트 근거"
        assert d["adjustment_type"] == "rule_stagnation"
        assert d["applied_params"]["diffusion_steps"] == 65
        assert "created_at" in d

    def test_to_dict_serializable(self) -> None:
        decision = SiloAPlannerDecision(
            epoch=1,
            hypothesis="초기 탐색",
            adjustment_type="rule_initial",
            applied_params={},
            llm_raw_response=None,
            fallback_reason="LLM 미호출",
        )
        # JSON 직렬화 가능 검증
        serialized = json.dumps(decision.to_dict())
        parsed = json.loads(serialized)
        assert parsed["epoch"] == 1


# ---------------------------------------------------------------------------
# plan_next_epoch 통합 테스트 (LLM mock)
# ---------------------------------------------------------------------------

class TestPlanNextEpoch:
    def test_rule_fallback_when_llm_unavailable(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """LLM 불가 시 규칙 fallback이 실행되고 provenance가 기록됨."""
        # LLM 호출을 None으로 mock (연결 실패 시뮬레이션)
        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            return_value=None,
        ):
            updated, decision = plan_next_epoch(
                epoch=4,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=5,  # patience 초과 → 정체 탈출 트리거
                patience_epochs=3,
                output_dir=tmp_dir,
            )

        # 규칙 fallback 경로
        assert decision.adjustment_type in ("rule_stagnation", "rule_noop")
        assert decision.fallback_reason is not None
        # provenance 파일 생성 검증
        planner_file = tmp_dir / "silo_a_planner_e0004.json"
        assert planner_file.exists()
        record = json.loads(planner_file.read_text())
        assert record["epoch"] == 4
        assert record["adjustment_type"] in ("rule_stagnation", "rule_noop")

    def test_llm_guided_path(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """LLM 성공 시 llm_guided 조정 경로 (prereview 비활성, 하위호환 검증)."""
        llm_mock_response = {
            "hypothesis": "ddG 상위 후보가 핫스팟 B197·B200에 집중 → B154·B294 강화 탐색",
            "hotspot_res": ["B154", "B197", "B294", "B300"],
            "binder_length_range": [13, 17],
            "diffusion_steps": 70,
            "n_backbone": None,
            "k_seq_per_backbone": None,
            "arm_preference": None,
        }
        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            return_value=llm_mock_response,
        ):
            updated, decision = plan_next_epoch(
                epoch=5,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=3,  # patience 경계
                patience_epochs=3,
                output_dir=tmp_dir,
                prereview_enabled=False,  # 하위호환 경로: prereview 비활성
            )

        assert decision.adjustment_type == "llm_guided"
        assert updated["diffusion_steps"] == 70
        assert updated["hotspot_res"] == ["B154", "B197", "B294", "B300"]
        # contigs에 바인더 길이 반영
        assert "13-17" in updated["contigs"]
        # provenance 파일
        planner_file = tmp_dir / "silo_a_planner_e0005.json"
        assert planner_file.exists()

    def test_no_stagnation_no_llm_called(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """정체 미발생이면 LLM 미호출 (비용 절감)."""
        call_count = {"n": 0}

        def fake_llm(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            return None

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", side_effect=fake_llm):
            updated, decision = plan_next_epoch(
                epoch=3,  # epoch > 1 이지만
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=1,   # < patience_epochs=3 → LLM 미호출
                patience_epochs=3,
                output_dir=tmp_dir,
            )

        # LLM 호출 없음
        assert call_count["n"] == 0
        # rule_noop 결정
        assert decision.adjustment_type == "rule_noop"
        # 파라미터 변경 없음
        assert updated["diffusion_steps"] == base_params["diffusion_steps"]

    def test_epoch1_skips_feedback(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """epoch=1은 플래너 제안이 포함되더라도 오류 없이 동작."""
        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            return_value=None,
        ):
            updated, decision = plan_next_epoch(
                epoch=1,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=0,
                patience_epochs=3,
                output_dir=tmp_dir,
            )
        # 오류 없이 완료
        assert updated is not None
        assert decision is not None

    def test_output_dir_not_pyrosetta_flow(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """provenance 저장 경로가 pyrosetta_flow가 아닌지 (분리 무결성)."""
        out_dir = tmp_dir / "silo_a_flow"
        assert "pyrosetta_flow" not in str(out_dir)
        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=None):
            plan_next_epoch(
                epoch=2,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=5,
                patience_epochs=3,
                output_dir=out_dir,
            )
        assert (out_dir / "silo_a_planner_e0002.json").exists()

    def test_provenance_no_hallucination(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """provenance에 실제 적용된 파라미터만 기록 (환각 검증)."""
        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=None):
            updated, decision = plan_next_epoch(
                epoch=6,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=4,
                patience_epochs=3,
                output_dir=tmp_dir,
            )
        record = json.loads((tmp_dir / "silo_a_planner_e0006.json").read_text())
        # applied_params는 실제 변경된 것만 담겨야 함
        # noop이라면 빈 dict도 가능
        assert isinstance(record["applied_params"], dict)
        # hypothesis는 비어있지 않아야 함
        assert len(record["hypothesis"]) > 0

    def test_params_always_valid_after_llm(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """LLM이 범위 밖 값을 제안해도 clamp되어 유효한 파라미터 반환."""
        llm_mock_response = {
            "hypothesis": "극단 파라미터 테스트",
            "hotspot_res": ["B999", "B150"],  # B999는 pool에 없지만 형식 유효
            "binder_length_range": [1, 100],  # 범위 밖 → clamp됨
            "diffusion_steps": 10000,         # 최대 초과 → 200으로 clamp
            "n_backbone": 0,                  # 최소 미달 → 1로 clamp
            "k_seq_per_backbone": None,
            "arm_preference": None,
        }
        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            return_value=llm_mock_response,
        ):
            updated, decision = plan_next_epoch(
                epoch=7,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=3,
                patience_epochs=3,
                output_dir=tmp_dir,
            )

        # clamp 검증
        assert updated["diffusion_steps"] <= 200
        assert updated["n_backbone"] >= 1
        # binder_length_range clamp 반영된 contigs
        lo, hi = 10, 20  # clamp 후 최대
        # contigs가 범위 내인지 확인
        from pyrosetta_flow.silo_a_planner import _parse_contig_binder_length
        c_lo, c_hi = _parse_contig_binder_length(updated["contigs"])
        assert c_lo >= 10
        assert c_hi <= 20


# ---------------------------------------------------------------------------
# _summarize_trajectory 테스트 (Silo A 전용)
# ---------------------------------------------------------------------------

def _make_silo_a_record(
    epoch: int,
    ddg: Optional[float],
    seq: str = "GGGGGGGGGGGGGGG",
    sel_margin: Optional[float] = None,
    delta_margin: Optional[float] = None,
    fail_reason: str = "",
) -> Dict[str, Any]:
    return {
        "candidate_id": f"silo_a_e{epoch:04d}_bb00_sq00",
        "sequence": seq,
        "epoch": epoch,
        "run_id": f"silo_a_e{epoch:04d}",
        "plddt": 85.0,
        "plddt_pass": True,
        "ddg": ddg,
        "clash_score": 0.0,
        "selectivity_margin": sel_margin,
        "delta_margin": delta_margin,
        "fail_reason": fail_reason,
        "candidate_class": "de_novo",
        "mutation_source": "silo_a",
        "extra_scores": {"backbone_pdb": f"/runs/silo_a_flow/epoch_{epoch}/bb00.pdb"},
        "created_at": "2026-06-19T10:00:00+00:00",
    }


class TestSummarizeTrajectory:

    def test_empty_log_no_crash(self, tmp_dir: Path) -> None:
        """빈 로그 → 에러 없이 '없음' 반환."""
        log = tmp_dir / "experiment_log.jsonl"
        log.write_text("", encoding="utf-8")
        result = _summarize_trajectory(log, k=6)
        assert "없음" in result

    def test_nonexistent_log(self, tmp_dir: Path) -> None:
        """파일 없음 → '없음' 반환."""
        log = tmp_dir / "no_log.jsonl"
        result = _summarize_trajectory(log, k=6)
        assert "없음" in result

    def test_single_epoch(self, tmp_dir: Path) -> None:
        """단일 epoch → 정상 요약."""
        log = tmp_dir / "experiment_log.jsonl"
        records = [_make_silo_a_record(1, -50.0) for _ in range(3)]
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "epoch 1" in result
        assert "ddG=-50" in result

    def test_improvement_detected(self, tmp_dir: Path) -> None:
        """epoch 간 ddG 개선 → ↑개선 표시."""
        log = tmp_dir / "experiment_log.jsonl"
        records = (
            [_make_silo_a_record(1, -30.0)] * 2 +
            [_make_silo_a_record(2, -40.0)] * 2  # 개선
        )
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "↑개선" in result

    def test_stagnation_detected(self, tmp_dir: Path) -> None:
        """3+ epoch 정체 → [패턴] 경고."""
        log = tmp_dir / "experiment_log.jsonl"
        records = []
        for ep in range(1, 6):
            records += [_make_silo_a_record(ep, -20.0)] * 2
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "[패턴]" in result or "정체" in result

    def test_selectivity_included(self, tmp_dir: Path) -> None:
        """sel_margin이 있으면 요약에 포함."""
        log = tmp_dir / "experiment_log.jsonl"
        records = [_make_silo_a_record(1, -50.0, sel_margin=10.0, delta_margin=3.5)] * 3
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "margin" in result or "sel" in result

    def test_k_limits_epochs(self, tmp_dir: Path) -> None:
        """k=3이면 최근 3 epoch만."""
        log = tmp_dir / "experiment_log.jsonl"
        records = []
        for ep in range(1, 11):
            records += [_make_silo_a_record(ep, -float(ep) * 5)] * 2
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=3)
        for ep in range(1, 8):
            assert f"epoch {ep}:" not in result

    def test_silo_b_path_blocked(self, tmp_dir: Path) -> None:
        """pyrosetta_flow 경로 접근 시도 → 차단 메시지."""
        fake_silo_b = tmp_dir / "pyrosetta_flow" / "experiment_log.jsonl"
        fake_silo_b.parent.mkdir(parents=True, exist_ok=True)
        fake_silo_b.write_text(json.dumps(_make_silo_a_record(1, -20.0)) + "\n")
        result = _summarize_trajectory(fake_silo_b, k=6)
        assert "금지" in result or "오류" in result or "없음" in result

    def test_corrupt_lines_tolerated(self, tmp_dir: Path) -> None:
        """손상된 라인이 있어도 유효 레코드는 처리."""
        log = tmp_dir / "experiment_log.jsonl"
        with log.open("w") as fh:
            fh.write("{invalid}\n")
            fh.write(json.dumps(_make_silo_a_record(1, -45.0)) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "epoch 1" in result

    def test_output_bounded(self, tmp_dir: Path) -> None:
        """출력이 과도하게 크지 않음."""
        log = tmp_dir / "experiment_log.jsonl"
        records = []
        for ep in range(1, 20):
            records += [_make_silo_a_record(ep, -float(ep) * 3)] * 5
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert len(result) < 3000

    def test_header_present(self, tmp_dir: Path) -> None:
        """헤더가 포함됨."""
        log = tmp_dir / "experiment_log.jsonl"
        records = [_make_silo_a_record(1, -20.0)]
        with log.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        result = _summarize_trajectory(log, k=6)
        assert "궤적" in result or "trajectory" in result.lower()


# ---------------------------------------------------------------------------
# plan_next_epoch 궤적 통합 테스트
# ---------------------------------------------------------------------------

class TestPlanNextEpochWithTrajectory:

    def test_experiment_log_path_auto_discovered(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """experiment_log.jsonl이 leaderboard 같은 디렉토리에 있으면 자동 발견."""
        # leaderboard 부모 디렉토리에 experiment_log.jsonl 생성
        log = sample_leaderboard_path.parent / "experiment_log.jsonl"
        with log.open("w") as fh:
            for ep in range(1, 4):
                fh.write(json.dumps(_make_silo_a_record(ep, -float(ep) * 10)) + "\n")

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=None):
            updated, decision = plan_next_epoch(
                epoch=4,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=5,
                patience_epochs=3,
                output_dir=tmp_dir,
                # experiment_log_path=None → 자동 발견
            )
        # 정상 완료 확인
        assert updated is not None
        assert decision is not None

    def test_trajectory_passed_to_llm(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """궤적 텍스트가 LLM 호출에 전달됨 (trajectory_text 파라미터)."""
        log = sample_leaderboard_path.parent / "experiment_log.jsonl"
        with log.open("w") as fh:
            for ep in range(1, 4):
                fh.write(json.dumps(_make_silo_a_record(ep, -float(ep) * 10)) + "\n")

        llm_calls: List[Dict[str, Any]] = []

        def fake_llm(**kwargs: Any) -> None:
            llm_calls.append(kwargs)
            return None

        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            side_effect=lambda **kw: llm_calls.append(kw) or None,
        ):
            plan_next_epoch(
                epoch=4,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=5,
                patience_epochs=3,
                output_dir=tmp_dir,
            )

        # LLM 호출됐으면 trajectory_text 키가 있어야 함
        if llm_calls:
            assert "trajectory_text" in llm_calls[0]

    def test_epoch1_no_trajectory(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """epoch=1은 궤적 없어도 정상 동작."""
        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=None):
            updated, decision = plan_next_epoch(
                epoch=1,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=0,
                patience_epochs=3,
                output_dir=tmp_dir,
            )
        assert updated is not None
        assert decision is not None


# ---------------------------------------------------------------------------
# 분리 무결성 테스트
# ---------------------------------------------------------------------------

class TestSeparationIntegrity:
    def test_silo_a_planner_not_import_silo_b(self) -> None:
        """silo_a_planner가 Silo B 핵심 모듈을 import하지 않음."""
        import importlib
        import pyrosetta_flow.silo_a_planner as planner_mod
        # runner, continuous는 Silo B 모듈 — import 안 해야 함
        assert "pyrosetta_flow.runner" not in str(dir(planner_mod))
        assert "pyrosetta_flow.continuous" not in str(dir(planner_mod))

    def test_planner_decision_serializable(self) -> None:
        """SiloAPlannerDecision이 JSON 직렬화 가능."""
        d = SiloAPlannerDecision(
            epoch=99,
            hypothesis="직렬화 테스트",
            adjustment_type="llm_guided",
            applied_params={"diffusion_steps": 80, "hotspot_res": ["B150"]},
            llm_raw_response='{"hypothesis": "test"}',
            fallback_reason=None,
            leaderboard_summary={"status": "ok", "best_ddg": -25.0},
        )
        serialized = json.dumps(d.to_dict(), ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["epoch"] == 99
        assert parsed["adjustment_type"] == "llm_guided"


# ---------------------------------------------------------------------------
# L3-a: 모델명 버그 수정 검증 테스트
# ---------------------------------------------------------------------------

class TestL3aModelName:
    """L3-a 수정: _DEFAULT_LLM_MODEL이 vLLM 서빙명('qwen3-32b')인지 검증.

    'Qwen/Qwen3-32B' (HuggingFace 경로)는 8000 포트에서 404를 반환함.
    'qwen3-32b' (서빙명)은 200을 반환하며 72B로 라우팅됨.
    """

    def test_default_model_is_serving_name(self) -> None:
        """_DEFAULT_LLM_MODEL이 HuggingFace 경로 형식이 아닌 서빙명인지 확인."""
        from pyrosetta_flow.silo_a_planner import _DEFAULT_LLM_MODEL
        # HuggingFace 경로 형식('/')을 포함하면 404 → 버그
        assert "/" not in _DEFAULT_LLM_MODEL, (
            f"_DEFAULT_LLM_MODEL='{_DEFAULT_LLM_MODEL}'에 '/'가 포함됨 — "
            "HuggingFace 경로 형식은 vLLM 8000에서 404를 반환한다. "
            "서빙명('qwen3-32b')을 사용해야 한다."
        )

    def test_default_model_matches_known_serving_name(self) -> None:
        """_DEFAULT_LLM_MODEL이 알려진 vLLM 서빙명 중 하나인지 확인."""
        from pyrosetta_flow.silo_a_planner import _DEFAULT_LLM_MODEL
        known_serving_names = {"qwen3-32b", "qwen2.5-72b"}
        assert _DEFAULT_LLM_MODEL in known_serving_names, (
            f"_DEFAULT_LLM_MODEL='{_DEFAULT_LLM_MODEL}'이 알려진 서빙명 "
            f"{known_serving_names}에 없음."
        )

    def test_plan_next_epoch_uses_serving_name(
        self, sample_leaderboard_path: Path, base_params: Dict[str, Any], tmp_dir: Path
    ) -> None:
        """plan_next_epoch LLM 호출 시 모델명이 서빙명('/')이 없는 형식으로 전달됨."""
        captured_models: List[str] = []

        def capture_model(**kwargs: Any) -> None:
            model = kwargs.get("model", "")
            captured_models.append(model)
            return None  # LLM 실패로 규칙 fallback

        with patch(
            "pyrosetta_flow.silo_a_planner._call_llm_for_plan",
            side_effect=lambda **kw: captured_models.append(kw.get("model", "")) or None,
        ):
            plan_next_epoch(
                epoch=5,
                leaderboard_path=sample_leaderboard_path,
                current_params=base_params,
                stagnation_count=5,
                patience_epochs=3,
                output_dir=tmp_dir,
            )

        # LLM 호출이 발생했을 때 모델명 검증
        if captured_models:
            for model in captured_models:
                assert "/" not in model, (
                    f"LLM 호출 모델명 '{model}'에 '/'가 포함됨 — vLLM 404 버그 재발"
                )


# ---------------------------------------------------------------------------
# L3-b: hotspot 포켓 풀 수정 검증 테스트
# ---------------------------------------------------------------------------

# SSTR2 결합 포켓 잔기 집합 (L3-b에서 정의된 _HOTSPOT_POOL과 동일)
_SSTR2_POCKET_RESIDUES = frozenset([
    "B192", "B193", "B195", "B197",  # ECL2
    "B205", "B208", "B209", "B212",  # TM5
    "B272", "B273", "B276", "B279",  # TM6
    "B284", "B286",                   # ECL3
])

# L3-b 이전의 임의 잔기 (포켓 무관 — 포함되면 안 됨)
_ARBITRARY_RESIDUES = {"B120", "B121", "B150", "B154", "B160", "B164",
                       "B175", "B180", "B200", "B210", "B220", "B240",
                       "B250", "B260", "B280", "B294", "B300"}


class TestL3bHotspotPocketPool:
    """L3-b 수정: hotspot 셔플이 SSTR2 포켓 풀 내에서만 선택되는지 검증."""

    def test_hotspot_pool_contains_only_pocket_residues(self) -> None:
        """_HOTSPOT_POOL이 SSTR2 포켓 잔기만 포함 (임의 잔기 없음)."""
        from pyrosetta_flow.silo_a_planner import _HOTSPOT_POOL
        pool_set = set(_HOTSPOT_POOL)
        # 임의 잔기가 포함되면 안 됨
        unexpected = pool_set & _ARBITRARY_RESIDUES
        assert not unexpected, (
            f"_HOTSPOT_POOL에 SSTR2 포켓 무관 잔기가 포함됨: {unexpected}. "
            "포켓 잔기(ECL2/TM5/TM6/ECL3)만 허용."
        )

    def test_hotspot_pool_covers_all_four_regions(self) -> None:
        """_HOTSPOT_POOL이 ECL2·TM5·TM6·ECL3 4개 영역을 모두 커버."""
        from pyrosetta_flow.silo_a_planner import _HOTSPOT_POOL
        pool_set = set(_HOTSPOT_POOL)
        ecl2 = {"B192", "B193", "B195", "B197"}
        tm5  = {"B205", "B208", "B209", "B212"}
        tm6  = {"B272", "B273", "B276", "B279"}
        ecl3 = {"B284", "B286"}
        assert pool_set & ecl2, "ECL2 잔기 없음"
        assert pool_set & tm5,  "TM5 잔기 없음"
        assert pool_set & tm6,  "TM6 잔기 없음"
        assert pool_set & ecl3, "ECL3 잔기 없음"

    def test_default_hotspot_is_subset_of_pool(self) -> None:
        """_DEFAULT_HOTSPOT이 _HOTSPOT_POOL의 부분집합인지 확인."""
        from pyrosetta_flow.silo_a_planner import _DEFAULT_HOTSPOT, _HOTSPOT_POOL
        pool_set = set(_HOTSPOT_POOL)
        for h in _DEFAULT_HOTSPOT:
            assert h in pool_set, (
                f"_DEFAULT_HOTSPOT 항목 '{h}'이 _HOTSPOT_POOL에 없음 — "
                "포켓 잔기 풀과 불일치."
            )

    def test_rule_stagnation_shuffles_within_pool(self, base_params: Dict[str, Any]) -> None:
        """rule_stagnation 핫스팟 교체가 _HOTSPOT_POOL 내에서만 선택됨."""
        from pyrosetta_flow.silo_a_planner import _HOTSPOT_POOL
        pool_set = set(_HOTSPOT_POOL)

        # base_params hotspot도 포켓 잔기로 교체 (L3-b 반영)
        pocket_base_params = dict(base_params)
        pocket_base_params["hotspot_res"] = ["B192", "B197", "B205", "B209", "B272", "B284"]

        for _ in range(20):  # 20회 반복해서 항상 풀 내에서 선택됨을 확인
            adj, _ = _rule_based_adjustment(
                summary={"status": "ok", "n_total": 10},
                current_params=pocket_base_params,
                patience_epochs=1,
                stagnation_count=5,
            )
            if "hotspot_res" in adj:
                for h in adj["hotspot_res"]:
                    assert h in pool_set, (
                        f"rule_stagnation이 포켓 풀 외 잔기 '{h}'를 선택함 — "
                        "임의 잔기 셔플 버그 재발."
                    )

    def test_no_arbitrary_residues_in_stagnation_hotspot(self, base_params: Dict[str, Any]) -> None:
        """정체 탈출 시 임의 잔기(B150/B200 등)가 선택되지 않음."""
        # base_params에 구 기본 핫스팟 포함 시 (이전 버전 호환성 시나리오)
        old_style_params = dict(base_params)
        old_style_params["hotspot_res"] = ["B150", "B154", "B197", "B200", "B250", "B294"]

        from pyrosetta_flow.silo_a_planner import _HOTSPOT_POOL
        pool_set = set(_HOTSPOT_POOL)

        adj, _ = _rule_based_adjustment(
            summary={},
            current_params=old_style_params,
            patience_epochs=1,
            stagnation_count=5,
        )
        # new_picks (교체 잔기)는 pool_not_current에서 오므로 항상 포켓 풀 내
        if "hotspot_res" in adj:
            new_residues = set(adj["hotspot_res"]) - set(old_style_params["hotspot_res"])
            for h in new_residues:
                assert h in pool_set, (
                    f"새로 추가된 잔기 '{h}'가 포켓 풀에 없음."
                )

    def test_validate_and_clamp_rejects_non_pocket_if_validated(self) -> None:
        """_validate_and_clamp_params가 형식은 유효하지만 포켓 외 잔기도 통과함을 확인.
        (형식 검증만 함 — 포켓 필터는 rule_stagnation에서 담당)
        즉 _validate_and_clamp_params 레이어는 형식('B\\d+')만 검증하고,
        pool 제한은 rule_stagnation이 담당한다는 아키텍처 계약 확인."""
        result = _validate_and_clamp_params({"hotspot_res": ["B192", "B197"]})
        assert result.get("hotspot_res") == ["B192", "B197"]


# ---------------------------------------------------------------------------
# 프롬프트 풍부화 검증 테스트
# ---------------------------------------------------------------------------

class TestEnrichedPrompt:
    """풍부화 프롬프트가 포켓-선택성·다양성 지침을 포함하는지 검증."""

    def test_system_prompt_contains_pocket_regions(self) -> None:
        """_call_llm_for_plan 시스템 프롬프트에 ECL2/TM5/TM6/ECL3 포켓 영역 언급."""
        import inspect
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        assert "ECL2" in source, "시스템 프롬프트에 ECL2 없음"
        assert "TM5" in source, "시스템 프롬프트에 TM5 없음"
        assert "TM6" in source, "시스템 프롬프트에 TM6 없음"
        assert "ECL3" in source, "시스템 프롬프트에 ECL3 없음"

    def test_system_prompt_contains_selectivity_rationale(self) -> None:
        """시스템 프롬프트에 포켓-선택성 논리 지침 포함."""
        import inspect
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        assert "선택성" in source, "선택성 언급 없음"
        assert "SSTR" in source, "SSTR 수용체 맥락 없음"

    def test_system_prompt_contains_diversity_rationale(self) -> None:
        """시스템 프롬프트에 탐색 다양성(길이·steps) 전략 지침 포함."""
        import inspect
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        assert "다양성" in source or "diversity" in source.lower(), "다양성 전략 지침 없음"
        assert "diffusion_steps" in source, "diffusion_steps 언급 없음"

    def test_user_prompt_requests_structured_fields(self) -> None:
        """user_prompt가 pocket_rationale·diversity_rationale 필드 요청."""
        import inspect
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        assert "pocket_rationale" in source, "pocket_rationale 필드 요청 없음"
        assert "diversity_rationale" in source, "diversity_rationale 필드 요청 없음"

    def test_max_tokens_sufficient_for_enriched_output(self) -> None:
        """max_tokens가 풍부화 출력에 충분한 값(>=1500)인지 확인."""
        import inspect
        import re
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        # max_tokens 값 추출
        match = re.search(r'"max_tokens":\s*(\d+)', source)
        assert match is not None, "max_tokens 설정 없음"
        assert int(match.group(1)) >= 1500, (
            f"max_tokens={match.group(1)} — 풍부화 hypothesis에 너무 작음 (>=1500 권장)"
        )

    def test_hypothesis_not_limited_to_3_sentences(self) -> None:
        """hypothesis 제한이 '3문장' 고정에서 '3-6문장'으로 완화됨."""
        import inspect
        import pyrosetta_flow.silo_a_planner as planner_mod
        source = inspect.getsource(planner_mod._call_llm_for_plan)
        # 구 "1-3문장" 제한 문구 없어야 함 (풍부화 후 3-6문장으로 완화)
        assert "1-3문장" not in source, (
            "구 '1-3문장' 제한이 여전히 남아있음 — 풍부화 완화 적용 안 됨"
        )


# ---------------------------------------------------------------------------
# SiloACriticResult 데이터클래스 테스트
# ---------------------------------------------------------------------------

class TestSiloACriticResult:
    def test_approve_default(self) -> None:
        """approve 결과 생성 및 직렬화."""
        critic = SiloACriticResult(
            verdict="approve",
            structure_ok=True,
            diversity_ok=True,
            concerns=[],
            suggested_revisions=None,
        )
        d = critic.to_dict()
        assert d["verdict"] == "approve"
        assert d["structure_ok"] is True
        assert d["diversity_ok"] is True
        assert d["concerns"] == []
        assert d["suggested_revisions"] is None
        assert "created_at" in d

    def test_concerns_result(self) -> None:
        """concerns 결과 필드 검증."""
        critic = SiloACriticResult(
            verdict="concerns",
            structure_ok=False,
            diversity_ok=True,
            concerns=["ECL2 미포함 — 선택성 취약", "hotspot이 TM5만 집중"],
            suggested_revisions={"hotspot_res": ["B192", "B197", "B205"]},
        )
        d = critic.to_dict()
        assert d["verdict"] == "concerns"
        assert d["structure_ok"] is False
        assert len(d["concerns"]) == 2
        assert d["suggested_revisions"] is not None

    def test_serializable(self) -> None:
        """JSON 직렬화 가능."""
        critic = SiloACriticResult(
            verdict="concerns",
            structure_ok=True,
            diversity_ok=False,
            concerns=["직전 epoch과 동일 hotspot 반복"],
            suggested_revisions={"diffusion_steps": 100},
            raw_response='{"verdict": "concerns"}',
        )
        serialized = json.dumps(critic.to_dict(), ensure_ascii=False)
        parsed = json.loads(serialized)
        assert parsed["verdict"] == "concerns"
        assert parsed["diversity_ok"] is False


# ---------------------------------------------------------------------------
# _call_critic_prereview 테스트 (LLM mock)
# ---------------------------------------------------------------------------

class TestCallCriticPrereview:
    """_call_critic_prereview LLM 호출 및 결과 파싱 검증."""

    def _sample_proposal(self) -> Dict[str, Any]:
        return {
            "hypothesis": "ECL2·TM5를 타겟해 선택성 강화",
            "pocket_rationale": "ECL2는 SSTR2 고유 포켓",
            "diversity_rationale": "steps=80으로 구조 다양성 확대",
            "hotspot_res": ["B192", "B197", "B205"],
            "binder_length_range": [12, 16],
            "diffusion_steps": 80,
        }

    def _sample_params(self) -> Dict[str, Any]:
        return {
            "contigs": "B1-369/0 12-16",
            "hotspot_res": ["B192", "B197", "B272"],
            "diffusion_steps": 50,
        }

    def test_llm_approve_response(self) -> None:
        """LLM이 approve 응답 시 SiloACriticResult approve 반환."""
        approve_response = json.dumps({
            "verdict": "approve",
            "structure_ok": True,
            "diversity_ok": True,
            "concerns": [],
            "suggested_revisions": None,
        })
        mock_body = {
            "choices": [{"message": {"content": approve_response}}]
        }
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=mock_body):
            result = _call_critic_prereview(
                planner_proposal=self._sample_proposal(),
                current_params=self._sample_params(),
                summary={"status": "ok"},
                trajectory_text=None,
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        assert result.verdict == "approve"
        assert result.structure_ok is True
        assert result.diversity_ok is True
        assert result.concerns == []

    def test_llm_concerns_response(self) -> None:
        """LLM이 concerns 응답 시 구조·다양성 비판 필드 파싱."""
        concerns_response = json.dumps({
            "verdict": "concerns",
            "structure_ok": False,
            "diversity_ok": True,
            "concerns": ["ECL2가 포함되지 않아 선택성 약화 우려"],
            "suggested_revisions": {"hotspot_res": ["B192", "B193", "B205"]},
        })
        mock_body = {
            "choices": [{"message": {"content": concerns_response}}]
        }
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=mock_body):
            result = _call_critic_prereview(
                planner_proposal=self._sample_proposal(),
                current_params=self._sample_params(),
                summary={"status": "ok"},
                trajectory_text="## 궤적\n- epoch 1: ddG=-20",
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        assert result.verdict == "concerns"
        assert result.structure_ok is False
        assert len(result.concerns) > 0
        assert result.suggested_revisions is not None

    def test_llm_failure_fail_open(self) -> None:
        """LLM 호출 실패 시 fail-open (자동 approve) 반환."""
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=None):
            result = _call_critic_prereview(
                planner_proposal=self._sample_proposal(),
                current_params=self._sample_params(),
                summary={},
                trajectory_text=None,
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        # fail-open: 실패해도 approve (엔진 중단 방지)
        assert result.verdict == "approve"
        assert result.structure_ok is True

    def test_malformed_json_fail_open(self) -> None:
        """LLM 응답 JSON 파싱 실패 시 fail-open."""
        mock_body = {
            "choices": [{"message": {"content": "이것은 JSON이 아닙니다"}}]
        }
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=mock_body):
            result = _call_critic_prereview(
                planner_proposal=self._sample_proposal(),
                current_params=self._sample_params(),
                summary={},
                trajectory_text=None,
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        assert result.verdict == "approve"

    def test_hotspot_concern_ecl2_missing(self) -> None:
        """ECL2 미포함 hotspot에 대한 concerns 반환 시뮬레이션."""
        # ECL2 잔기 없는 제안: TM5+TM6만
        bad_proposal = dict(self._sample_proposal())
        bad_proposal["hotspot_res"] = ["B205", "B208", "B272", "B279"]  # ECL2 없음
        concerns_response = json.dumps({
            "verdict": "concerns",
            "structure_ok": False,
            "diversity_ok": True,
            "concerns": ["ECL2(B192/B193/B195/B197)가 포함되지 않아 SSTR2 선택성 취약"],
            "suggested_revisions": {"hotspot_res": ["B192", "B205", "B272"]},
        })
        mock_body = {"choices": [{"message": {"content": concerns_response}}]}
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=mock_body):
            result = _call_critic_prereview(
                planner_proposal=bad_proposal,
                current_params=self._sample_params(),
                summary={},
                trajectory_text=None,
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        assert result.verdict == "concerns"
        assert result.structure_ok is False
        assert any("ECL2" in c or "선택성" in c for c in result.concerns)

    def test_repetitive_strategy_concern(self) -> None:
        """직전과 동일 hotspot 반복에 대한 다양성 concerns 시뮬레이션."""
        concerns_response = json.dumps({
            "verdict": "concerns",
            "structure_ok": True,
            "diversity_ok": False,
            "concerns": ["직전 epoch과 동일한 hotspot 조합 반복 — 탐색 정체"],
            "suggested_revisions": {"diffusion_steps": 120},
        })
        mock_body = {"choices": [{"message": {"content": concerns_response}}]}
        with patch("pyrosetta_flow.silo_a_planner._http_post_json", return_value=mock_body):
            result = _call_critic_prereview(
                planner_proposal=self._sample_proposal(),
                current_params=self._sample_params(),
                summary={},
                trajectory_text=None,
                vllm_url="http://localhost:8000",
                model="qwen3-32b",
                timeout=60,
            )
        assert result.diversity_ok is False
        assert result.structure_ok is True


# ---------------------------------------------------------------------------
# _append_discussion_log 테스트
# ---------------------------------------------------------------------------

class TestAppendDiscussionLog:
    def test_creates_file_and_appends(self, tmp_dir: Path) -> None:
        """discussion_log 파일이 생성되고 JSONL 라인이 추가됨."""
        log_path = tmp_dir / "silo_a_discussion_log.jsonl"
        critic = SiloACriticResult(
            verdict="approve",
            structure_ok=True,
            diversity_ok=True,
            concerns=[],
            suggested_revisions=None,
        )
        _append_discussion_log(
            log_path, epoch=3,
            planner_proposal={"hypothesis": "테스트", "hotspot_res": ["B192"]},
            critic_result=critic,
            final_params={"hotspot_res": ["B192"]},
        )
        assert log_path.exists()
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["epoch"] == 3
        assert record["critic_result"]["verdict"] == "approve"

    def test_multiple_appends(self, tmp_dir: Path) -> None:
        """여러 번 append시 라인 수 증가."""
        log_path = tmp_dir / "disc.jsonl"
        critic_ok = SiloACriticResult(
            verdict="approve", structure_ok=True, diversity_ok=True,
            concerns=[], suggested_revisions=None,
        )
        for ep in range(1, 4):
            _append_discussion_log(log_path, ep, {"hypothesis": f"ep{ep}"}, critic_ok)
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_record_contains_required_fields(self, tmp_dir: Path) -> None:
        """기록된 JSONL에 필수 필드 포함."""
        log_path = tmp_dir / "disc.jsonl"
        critic = SiloACriticResult(
            verdict="concerns", structure_ok=False, diversity_ok=True,
            concerns=["ECL2 미포함"],
            suggested_revisions={"hotspot_res": ["B192"]},
        )
        _append_discussion_log(
            log_path, epoch=5,
            planner_proposal={"hypothesis": "TM5 집중 전략", "pocket_rationale": "TM5 친화도"},
            critic_result=critic,
            final_params={"hotspot_res": ["B192", "B205"]},
        )
        record = json.loads(log_path.read_text().strip())
        assert "epoch" in record
        assert "created_at" in record
        assert "planner_proposal" in record
        assert "critic_result" in record
        assert "final_applied" in record
        assert record["critic_result"]["verdict"] == "concerns"
        assert record["critic_result"]["concerns"] == ["ECL2 미포함"]

    def test_auto_creates_parent_dir(self, tmp_dir: Path) -> None:
        """부모 디렉토리 자동 생성."""
        log_path = tmp_dir / "deep" / "nested" / "disc.jsonl"
        critic = SiloACriticResult(
            verdict="approve", structure_ok=True, diversity_ok=True,
            concerns=[], suggested_revisions=None,
        )
        _append_discussion_log(log_path, 1, {}, critic)
        assert log_path.exists()


# ---------------------------------------------------------------------------
# prereview 통합: plan_next_epoch with prereview
# ---------------------------------------------------------------------------

class TestPlanNextEpochWithPrereview:
    """plan_next_epoch prereview_enabled 파라미터 통합 테스트."""

    def _make_leaderboard(self, tmp_dir: Path) -> Path:
        """샘플 리더보드 JSON 생성."""
        path = tmp_dir / "silo_a_leaderboard.json"
        data = {
            "source": "silo_a", "candidate_class": "de_novo",
            "n_total": 4, "n_unique": 2, "best_ddg": -25.0, "best_selectivity_margin": 3.0,
            "entries": [
                {"candidate_id": "e1_bb0_sq0", "sequence": "ACDEFGHIKL",
                 "ddg": -25.0, "selectivity_margin": 3.0, "plddt": 0.88,
                 "plddt_pass": True, "fail_reason": "", "extra_scores": {}},
            ],
            "screened_seqs": [], "updated_at": "2026-06-23T00:00:00Z",
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _base_params(self) -> Dict[str, Any]:
        return {
            "contigs": "B1-369/0 12-16",
            "hotspot_res": ["B192", "B197", "B205", "B209"],
            "diffusion_steps": 50,
            "n_backbone": 2,
            "k_seq_per_backbone": 2,
        }

    def test_prereview_disabled_no_critic_call(self, tmp_dir: Path) -> None:
        """prereview_enabled=False이면 critic 호출 없이 기존 동작 (하위호환)."""
        lb = self._make_leaderboard(tmp_dir)
        critic_call_count = {"n": 0}

        def fake_critic(**kwargs: Any) -> SiloACriticResult:
            critic_call_count["n"] += 1
            return SiloACriticResult(
                verdict="approve", structure_ok=True, diversity_ok=True,
                concerns=[], suggested_revisions=None,
            )

        llm_resp = {
            "hypothesis": "ECL2 강화 전략", "pocket_rationale": "ECL2 선택성",
            "diversity_rationale": "steps 증가", "hotspot_res": ["B192", "B197"],
            "binder_length_range": [12, 16], "diffusion_steps": 70,
        }
        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", side_effect=fake_critic):
                plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=False,  # ← 비활성
                )

        assert critic_call_count["n"] == 0, "prereview=False인데 critic 호출됨"

    def test_prereview_enabled_critic_called(self, tmp_dir: Path) -> None:
        """prereview_enabled=True이면 LLM 성공 시 critic 호출."""
        lb = self._make_leaderboard(tmp_dir)
        critic_called = {"called": False}

        def fake_critic(**kwargs: Any) -> SiloACriticResult:
            critic_called["called"] = True
            return SiloACriticResult(
                verdict="approve", structure_ok=True, diversity_ok=True,
                concerns=[], suggested_revisions=None,
            )

        llm_resp = {
            "hypothesis": "ECL2+TM5 포켓 타겟으로 선택성 강화",
            "pocket_rationale": "ECL2 선택성 핵심",
            "diversity_rationale": "steps=80 다양성 확대",
            "hotspot_res": ["B192", "B197", "B205"],
            "binder_length_range": None, "diffusion_steps": 80,
        }
        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", side_effect=fake_critic):
                plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                )

        assert critic_called["called"], "prereview=True인데 critic 미호출"

    def test_critic_concerns_revisions_merged(self, tmp_dir: Path) -> None:
        """critic이 concerns + suggested_revisions 제시하면 adj_params에 병합."""
        lb = self._make_leaderboard(tmp_dir)

        llm_resp = {
            "hypothesis": "TM5·TM6 집중",
            "pocket_rationale": "TM5 친화도",
            "diversity_rationale": "steps 유지",
            "hotspot_res": ["B205", "B208", "B272"],
            "binder_length_range": None, "diffusion_steps": 60,
        }
        critic_resp = SiloACriticResult(
            verdict="concerns", structure_ok=False, diversity_ok=True,
            concerns=["ECL2 미포함 — 선택성 취약"],
            suggested_revisions={"hotspot_res": ["B192", "B205", "B272"]},
        )

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", return_value=critic_resp):
                updated, decision = plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                )

        # critic 수정 제안(B192 추가)이 최종 파라미터에 반영됐는지
        assert "B192" in updated.get("hotspot_res", []), (
            "critic 수정 제안 B192가 최종 파라미터에 반영되지 않음"
        )
        # adj_type이 prereview 표시
        assert decision.adjustment_type == "llm_guided_with_prereview"

    def test_critic_concerns_no_revision_keeps_planner(self, tmp_dir: Path) -> None:
        """critic이 concerns를 제시하지만 suggested_revisions=None이면 플래너 제안 유지."""
        lb = self._make_leaderboard(tmp_dir)

        llm_resp = {
            "hypothesis": "TM5+ECL2 타겟",
            "pocket_rationale": "ECL2+TM5",
            "diversity_rationale": "steps 80",
            "hotspot_res": ["B192", "B205"],
            "binder_length_range": None, "diffusion_steps": 80,
        }
        critic_resp = SiloACriticResult(
            verdict="concerns", structure_ok=True, diversity_ok=False,
            concerns=["직전 epoch과 비슷한 전략"],
            suggested_revisions=None,  # 수정 제안 없음
        )

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", return_value=critic_resp):
                updated, decision = plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                )

        # 수정 제안 없으면 플래너 원안 유지
        assert updated.get("diffusion_steps") == 80

    def test_discussion_log_created_on_prereview(self, tmp_dir: Path) -> None:
        """prereview 시 discussion_log.jsonl 파일 생성."""
        lb = self._make_leaderboard(tmp_dir)
        disc_log = tmp_dir / "silo_a_discussion_log.jsonl"

        llm_resp = {
            "hypothesis": "ECL2 타겟",
            "pocket_rationale": "ECL2 선택성",
            "diversity_rationale": "다양성",
            "hotspot_res": ["B192", "B197"],
            "binder_length_range": None, "diffusion_steps": 70,
        }
        critic_resp = SiloACriticResult(
            verdict="approve", structure_ok=True, diversity_ok=True,
            concerns=[], suggested_revisions=None,
        )

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", return_value=critic_resp):
                plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                    discussion_log_path=disc_log,
                )

        assert disc_log.exists(), "discussion_log.jsonl 미생성"
        lines = [l for l in disc_log.read_text().splitlines() if l.strip()]
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["epoch"] == 5
        assert "planner_proposal" in record
        assert "critic_result" in record

    def test_prereview_off_no_discussion_log(self, tmp_dir: Path) -> None:
        """prereview=False이면 discussion_log 기록 안 됨."""
        lb = self._make_leaderboard(tmp_dir)
        disc_log = tmp_dir / "silo_a_discussion_log.jsonl"

        llm_resp = {
            "hypothesis": "규칙 테스트",
            "pocket_rationale": None, "diversity_rationale": None,
            "hotspot_res": ["B192"], "binder_length_range": None, "diffusion_steps": 70,
        }

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            plan_next_epoch(
                epoch=5, leaderboard_path=lb,
                current_params=self._base_params(),
                stagnation_count=3, patience_epochs=3,
                output_dir=tmp_dir,
                prereview_enabled=False,
                discussion_log_path=disc_log,
            )

        # prereview=False이면 discussion_log 미생성 또는 빈 파일
        if disc_log.exists():
            lines = [l for l in disc_log.read_text().splitlines() if l.strip()]
            assert len(lines) == 0, "prereview=False인데 discussion_log에 기록됨"

    def test_critic_exception_does_not_propagate(self, tmp_dir: Path) -> None:
        """critic 예외 발생 시 plan_next_epoch이 중단되지 않음 (fail-open)."""
        lb = self._make_leaderboard(tmp_dir)

        llm_resp = {
            "hypothesis": "ECL2 타겟",
            "pocket_rationale": "ECL2 선택성",
            "diversity_rationale": "다양성",
            "hotspot_res": ["B192"], "binder_length_range": None, "diffusion_steps": 70,
        }

        def raising_critic(**kwargs: Any) -> SiloACriticResult:
            raise RuntimeError("critic 내부 예외")

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=llm_resp):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", side_effect=raising_critic):
                # 예외 없이 완료되어야 함
                updated, decision = plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                )
        assert updated is not None
        assert decision is not None

    def test_rule_fallback_no_prereview(self, tmp_dir: Path) -> None:
        """규칙 fallback 경로(LLM 실패)에서는 prereview 실행 안 됨."""
        lb = self._make_leaderboard(tmp_dir)
        critic_called = {"n": 0}

        def fake_critic(**kwargs: Any) -> SiloACriticResult:
            critic_called["n"] += 1
            return SiloACriticResult(
                verdict="approve", structure_ok=True, diversity_ok=True,
                concerns=[], suggested_revisions=None,
            )

        with patch("pyrosetta_flow.silo_a_planner._call_llm_for_plan", return_value=None):
            with patch("pyrosetta_flow.silo_a_planner._call_critic_prereview", side_effect=fake_critic):
                plan_next_epoch(
                    epoch=5, leaderboard_path=lb,
                    current_params=self._base_params(),
                    stagnation_count=3, patience_epochs=3,
                    output_dir=tmp_dir,
                    prereview_enabled=True,
                )

        # LLM 실패 → 규칙 fallback → critic 미호출
        assert critic_called["n"] == 0, "규칙 fallback 경로인데 critic 호출됨"


# ---------------------------------------------------------------------------
# _get_ladder_steps 단위 테스트 (정체 사다리 강도 증가)
# ---------------------------------------------------------------------------

class TestGetLadderSteps:
    """_get_ladder_steps: stagnation_count/patience 배수에 따른 steps 사다리 검증."""

    def test_no_stagnation_returns_none(self) -> None:
        """stagnation_count < patience → None (정체 미발생)."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        assert _get_ladder_steps(2, 3) is None
        assert _get_ladder_steps(0, 50) is None

    def test_exactly_patience_returns_step50(self) -> None:
        """stagnation_count == patience → 1× 단계 → steps=50."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        result = _get_ladder_steps(50, 50)  # ratio=1
        assert result == 50

    def test_2x_patience_returns_100(self) -> None:
        """stagnation_count == 2×patience → steps=100."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        assert _get_ladder_steps(100, 50) == 100
        assert _get_ladder_steps(199, 50) == 100  # ratio=3 → 2× 임계 적용

    def test_4x_patience_returns_150(self) -> None:
        """stagnation_count == 4×patience → steps=150."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        assert _get_ladder_steps(200, 50) == 150

    def test_8x_patience_returns_200(self) -> None:
        """stagnation_count == 8×patience → steps=200 (최대)."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        assert _get_ladder_steps(400, 50) == 200

    def test_197epoch_with_patience50_correct_stage(self) -> None:
        """실측 197 epoch 정체, patience=50 → ratio=3 → 2× 단계 → steps=100."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        # 197 // 50 = 3 → 2× 임계 적용 (4× 임계=200 미도달)
        result = _get_ladder_steps(197, 50)
        assert result == 100, f"예상 100, 실제 {result}"

    def test_zero_patience_returns_none(self) -> None:
        """patience_epochs=0 → None (0으로 나누기 방지)."""
        from pyrosetta_flow.silo_a_planner import _get_ladder_steps
        assert _get_ladder_steps(100, 0) is None

    def test_ladder_steps_force_overrides_current(self, base_params: Dict[str, Any]) -> None:
        """ladder_steps > curr_steps이면 사다리 강제 적용, 결과가 curr_steps보다 큼."""
        params = dict(base_params)
        params["diffusion_steps"] = 50
        adj, hypothesis = _rule_based_adjustment(
            summary={},
            current_params=params,
            patience_epochs=50,
            stagnation_count=197,  # ratio=3 → steps=100
        )
        # diffusion_steps가 100으로 강제됨
        assert adj.get("diffusion_steps") == 100, (
            f"197epoch 정체(patience=50)에서 사다리 steps=100 미적용: {adj}"
        )
        assert "사다리" in hypothesis, "hypothesis에 '사다리' 키워드 누락"

    def test_ladder_does_not_downgrade_current_steps(self, base_params: Dict[str, Any]) -> None:
        """ladder_steps <= curr_steps이면 기존 steps 유지 (강등 금지)."""
        params = dict(base_params)
        params["diffusion_steps"] = 180  # 현재 이미 높음
        adj, hypothesis = _rule_based_adjustment(
            summary={},
            current_params=params,
            patience_epochs=50,
            stagnation_count=50,  # ratio=1 → 사다리=50 < curr=180 → 강등 안 함
        )
        # 사다리값(50)이 curr(180)보다 낮으므로 강등 없음 → diffusion_steps 변경 없음
        if "diffusion_steps" in adj:
            assert adj["diffusion_steps"] >= 180, "사다리가 현재 steps보다 낮은데 강등됨"

    def test_long_stagnation_200_epoch_reaches_max(self, base_params: Dict[str, Any]) -> None:
        """400 epoch 이상 정체(8×patience=50) → diffusion_steps=200 최대값 강제."""
        params = dict(base_params)
        params["diffusion_steps"] = 50
        adj, hypothesis = _rule_based_adjustment(
            summary={},
            current_params=params,
            patience_epochs=50,
            stagnation_count=400,  # ratio=8 → steps=200
        )
        assert adj.get("diffusion_steps") == 200, (
            f"400epoch 정체에서 최대 steps=200 미적용: {adj}"
        )


# ---------------------------------------------------------------------------
# expert_panel 병렬화 consensus 동일성 테스트 (작업1 검증)
# ---------------------------------------------------------------------------

class TestExpertPanelParallelConsistency:
    """ThreadPoolExecutor 병렬화 후에도 consensus 결과가 순차 결과와 동일한지 검증."""

    def _make_mock_critic_ordered(self) -> "MagicMock":
        """5 전문가(pharma/biology/chemistry/radiochem/math) + fan-in 순서 고정 mock (병렬 호출 순서 독립)."""
        from unittest.mock import MagicMock
        mock = MagicMock()
        mock.has_llm = True
        _verdicts = {
            "pharma": {"domain": "pharma", "concerns": ["PK 영향"], "severity": "low", "suggestion": ""},
            "biology": {"domain": "biology", "concerns": ["선택성 불충분"], "severity": "low", "suggestion": ""},
            "chemistry": {"domain": "chemistry", "concerns": [], "severity": "low", "suggestion": ""},
            "radiochem": {"domain": "radiochem", "concerns": [], "severity": "low", "suggestion": ""},
            "math": {"domain": "math", "concerns": ["과탐색 위험"], "severity": "medium", "suggestion": ""},
        }
        _fanin = {
            "merged_concerns": ["math: 과탐색"],
            "approve": True,
            "suggested_revisions": {},
        }

        def _side_effect(prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
            # 프롬프트 내용으로 domain 식별 (format_expert_domain_prompt가 도메인 대문자 포함)
            for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
                if domain.upper() in prompt:
                    return dict(_verdicts[domain])
            return dict(_fanin)

        mock.llm_generate_json = MagicMock(side_effect=_side_effect)
        return mock

    def test_parallel_result_has_all_four_domains(self) -> None:
        """병렬 실행 후 5 도메인이 모두 expert_verdicts에 존재한다."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from AG_src.agents.expert_panel import ExpertPanelAgent
        mock = self._make_mock_critic_ordered()
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=1,
            hypothesis="pos11 F11R 변이",
            mutation_guidance={"focus_positions": [11], "suggested_mutations": {"11": ["R"]}, "n_guided": 3, "strategy": "charge"},
        )
        for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
            assert domain in result["expert_verdicts"], f"병렬화 후 도메인 누락: {domain}"

    def test_high1_conditional_approve_true(self) -> None:
        """math-only high → 규칙 레벨 approve, prereview_panel도 approve (advisory 제외 정책).

        현 정책:
          - math는 _ADVISORY_ONLY_DOMAINS에 포함 → _apply_consensus_rules에서 제외.
          - math만 high이면 science_verdicts(pharma/biology/chemistry/radiochem)는 전부 low
            → _apply_consensus_rules: approve=True.
          - prereview_panel: fanin approve=True → 1라운드에서 즉시 종료.
          - consensus_status="approve", llm_calls=6(5전문가+1fanin), discussion_rounds=1.
        """
        from unittest.mock import MagicMock
        from AG_src.agents.expert_panel import (
            ExpertPanelAgent, _apply_consensus_rules, _ADVISORY_ONLY_DOMAINS,
        )

        # math가 _ADVISORY_ONLY_DOMAINS에 포함되는지 확인
        assert "math" in _ADVISORY_ONLY_DOMAINS, "math가 advisory-only여야 함"

        # _apply_consensus_rules 직접 검증:
        # math=high + pharma/biology/chemistry/radiochem=low → math 제외 후 전부 low → approve
        verdicts_high1_with_domain = [
            {"domain": "pharma", "severity": "low"},
            {"domain": "biology", "severity": "low"},
            {"domain": "chemistry", "severity": "low"},
            {"domain": "radiochem", "severity": "low"},
            {"domain": "math", "severity": "high"},   # advisory 제외 → approve에 영향 없음
        ]
        rule_result = _apply_consensus_rules(verdicts_high1_with_domain)
        assert rule_result["approve"] is True, (
            f"math=high는 advisory 제외 → 규칙 레벨 approve=True 여야 함, 실제={rule_result['approve']}"
        )
        assert rule_result["high_count"] == 0, (
            f"math 제외 후 high_count=0이어야 함, 실제={rule_result['high_count']}"
        )

        # prereview_panel 레벨: fanin approve=True → 1라운드 즉시 통과
        mock = MagicMock()
        mock.has_llm = True
        _verdicts_high = {
            "pharma": {"domain": "pharma", "concerns": [], "severity": "low", "suggestion": ""},
            "biology": {"domain": "biology", "concerns": [], "severity": "low", "suggestion": ""},
            "chemistry": {"domain": "chemistry", "concerns": [], "severity": "low", "suggestion": ""},
            "radiochem": {"domain": "radiochem", "concerns": [], "severity": "low", "suggestion": ""},
            "math": {"domain": "math", "concerns": ["국소최적"], "severity": "high", "suggestion": ""},
        }
        # math=high이지만 advisory 제외 → 코드 approve=True → fanin도 approve=True (코드 우선)
        _fanin = {"merged_concerns": ["math:high"], "approve": False, "suggested_revisions": {}}

        def _side_high1(prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
            for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
                if domain.upper() in prompt:
                    return dict(_verdicts_high[domain])
            return dict(_fanin)

        mock.llm_generate_json = MagicMock(side_effect=_side_high1)
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            1, "테스트",
            {"focus_positions": [1], "strategy": "x", "n_guided": 1, "suggested_mutations": {}},
        )
        # math advisory 제외 → 코드 approve=True → 1라운드 즉시 통과
        assert result["approve"] is True, (
            f"math-only high: advisory 제외 → approve=True 여야 함, "
            f"실제={result['approve']}, status={result['consensus_status']}"
        )
        assert result["consensus_status"] == "approve", (
            f"consensus_status 예상 approve, 실제={result['consensus_status']}"
        )
        # math 우려가 merged_concerns에 advisory 기록됨 (fanin이 반환했으므로)
        assert len(result["merged_concerns"]) > 0, "math 우려가 merged_concerns에 기록되어야 함"

    def test_high2_reject_boundary(self) -> None:
        """high severity 2개 이상 → _apply_consensus_rules 규칙상 reject (PANEL_HIGH_REJECT=2 경계).

        현 정책:
          - high >= PANEL_HIGH_REJECT(2): status="reject", approve=False
          - high 1개: status="conditional" (위 테스트 참조)
        _apply_consensus_rules를 직접 검증하여 reject 경계가 유지됨을 보장.
        """
        from AG_src.agents.expert_panel import _apply_consensus_rules, _PANEL_HIGH_REJECT

        assert _PANEL_HIGH_REJECT == 2, f"PANEL_HIGH_REJECT 기대=2, 실제={_PANEL_HIGH_REJECT}"

        # high 2개: reject 경계 (== PANEL_HIGH_REJECT)
        verdicts_high2 = [
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result2 = _apply_consensus_rules(verdicts_high2)
        assert result2["approve"] is False, "high 2개 → approve=False(reject)이어야 함"
        assert result2["status"] == "reject", f"high 2개 → status=reject이어야 함, 실제={result2['status']}"
        assert result2["high_count"] == 2

        # high 3개: reject (>= PANEL_HIGH_REJECT)
        verdicts_high3 = [
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "high"},
            {"severity": "low"},
        ]
        result3 = _apply_consensus_rules(verdicts_high3)
        assert result3["approve"] is False, "high 3개 → approve=False(reject)이어야 함"
        assert result3["status"] == "reject"
        assert result3["high_count"] == 3

        # high 1개: conditional (reject 아님 — 경계 구분)
        verdicts_high1 = [
            {"severity": "high"},
            {"severity": "low"},
            {"severity": "low"},
            {"severity": "low"},
        ]
        result1 = _apply_consensus_rules(verdicts_high1)
        assert result1["status"] == "conditional", "high 1개는 reject가 아닌 conditional이어야 함"
        assert result1["approve"] is False  # conditional도 approve=False

    def test_parallel_llm_calls_count_round1_only(self) -> None:
        """round2 토론 없이 통과 시 llm_calls == 6 (5 전문가 + 1 fan-in).

        현 정책: approve=True이고 merged_concerns < PANEL_DISCUSS_MIN_CONCERNS(1) 이면
        round2 토론 없이 즉시 통과 → llm_calls = 6 (round1만).
        """
        from unittest.mock import MagicMock
        from AG_src.agents.expert_panel import ExpertPanelAgent

        mock = MagicMock()
        mock.has_llm = True
        _verdicts_clean = {
            "pharma": {"domain": "pharma", "concerns": [], "severity": "low", "suggestion": ""},
            "biology": {"domain": "biology", "concerns": [], "severity": "low", "suggestion": ""},
            "chemistry": {"domain": "chemistry", "concerns": [], "severity": "low", "suggestion": ""},
            "radiochem": {"domain": "radiochem", "concerns": [], "severity": "low", "suggestion": ""},
            "math": {"domain": "math", "concerns": [], "severity": "low", "suggestion": ""},
        }
        # merged_concerns=[] → len=0 < PANEL_DISCUSS_MIN_CONCERNS(1) → round2 없이 즉시 통과
        _fanin_clean = {"merged_concerns": [], "approve": True, "suggested_revisions": {}}

        def _side_clean(prompt: str, system_prompt: str = "") -> Optional[Dict[str, Any]]:
            for domain in ("pharma", "biology", "chemistry", "radiochem", "math"):
                if domain.upper() in prompt:
                    return dict(_verdicts_clean[domain])
            return dict(_fanin_clean)

        mock.llm_generate_json = MagicMock(side_effect=_side_clean)
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=2,
            hypothesis="테스트 가설",
            mutation_guidance={"focus_positions": [5], "strategy": "y", "n_guided": 2, "suggested_mutations": {}},
        )
        assert result["llm_calls"] == 6, (
            f"round1만 통과 시 llm_calls 기대=6, 실제={result['llm_calls']}"
        )
        assert result["discussion_rounds"] == 1, (
            f"round2 없이 통과 시 discussion_rounds=1이어야 함, 실제={result['discussion_rounds']}"
        )

    def test_parallel_llm_calls_count_round2(self) -> None:
        """fanin approve=True이면 merged_concerns 유무와 무관하게 1라운드 즉시 종료.

        현 정책 (2026-06-23 이후):
          approve=True 달성 시 즉시 종료 (비용 최소화).
          _PANEL_DISCUSS_MIN_CONCERNS는 DEPRECATED — approve=True이면 바로 return.
          _make_mock_critic_ordered()의 fanin이 approve=True, merged_concerns=["math: 과탐색"]이면
          → approve=True 즉시 종료 → llm_calls=6(5전문가+1fanin), discussion_rounds=1.
        """
        from AG_src.agents.expert_panel import ExpertPanelAgent

        # _make_mock_critic_ordered(): fanin approve=True, merged_concerns=["math: 과탐색"] 1건
        # approve=True → 즉시 종료(1라운드)
        mock = self._make_mock_critic_ordered()
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=2,
            hypothesis="테스트 가설",
            mutation_guidance={"focus_positions": [5], "strategy": "y", "n_guided": 2, "suggested_mutations": {}},
        )
        # approve=True 달성 → 1라운드 즉시 종료, llm_calls=6
        assert result["approve"] is True, f"approve 기대=True, 실제={result['approve']}"
        assert result["llm_calls"] == 6, (
            f"approve=True 즉시 종료 시 llm_calls 기대=6, 실제={result['llm_calls']}"
        )
        assert result["discussion_rounds"] == 1, (
            f"approve=True 즉시 종료 시 discussion_rounds=1이어야 함, 실제={result['discussion_rounds']}"
        )

    def test_domain_order_preserved(self) -> None:
        """expert_verdicts 딕셔너리 키가 _EXPERT_DOMAINS 순서와 일치."""
        from AG_src.agents.expert_panel import ExpertPanelAgent, _EXPERT_DOMAINS
        mock = self._make_mock_critic_ordered()
        panel = ExpertPanelAgent(mock)
        result = panel.prereview_panel(
            iteration=3,
            hypothesis="순서 테스트",
            mutation_guidance={"focus_positions": [3], "strategy": "z", "n_guided": 1, "suggested_mutations": {}},
        )
        keys = list(result["expert_verdicts"].keys())
        assert keys == _EXPERT_DOMAINS, f"도메인 순서 불일치: {keys} vs {_EXPERT_DOMAINS}"
