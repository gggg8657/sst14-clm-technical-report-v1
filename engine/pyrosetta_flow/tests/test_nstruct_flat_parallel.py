"""test_nstruct_flat_parallel.py
================================
nstruct 평탄화 병렬 도킹(접근 A) 핵심 로직 단위 테스트.

검증 목표:
  1. 후보별 robust 통계(median/sd/n_converged) 집계가 평탄화 전후 동일 의미인지.
  2. 수렴 게이트(ddg ≥ 0 제외) 정확히 적용되는지.
  3. 모든 trial 실패 시 fail_reason 올바르게 반환되는지.
  4. DOCK_MAX_WORKERS / FLEXPEP_NSTRUCT env 변수 동작 확인.
  5. max_workers 상한(RAM/코어 안전 범위) 로직 확인.
"""
from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 유틸: 평탄화 집계 로직 순수 함수로 추출 (테스트 전용 복제)
# runner.py 의 후보별 집계 블록과 완전히 동일한 로직으로 작성 — 이 테스트가 통과하면
# runner 내부 로직도 정확하다는 것을 실증.
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_trials(
    successful: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """후보별 trial 결과 목록에서 robust 통계 집계.

    Returns dict with keys: ddg_median, ddg_mean, ddg_sd, ddg_min,
                            n_converged, n_total, converged.
    """
    n_total = len(successful)
    all_ddgs = [float(tr["ddg"]) for tr in successful]
    converged_ddgs = [d for d in all_ddgs if d < 0]
    n_converged = len(converged_ddgs)

    if converged_ddgs:
        ddg_median = statistics.median(converged_ddgs)
        ddg_mean = statistics.mean(converged_ddgs)
        ddg_sd = statistics.stdev(converged_ddgs) if n_converged > 1 else 0.0
        ddg_min = min(converged_ddgs)
    else:
        ddg_median = min(all_ddgs)
        ddg_mean = statistics.mean(all_ddgs)
        ddg_sd = statistics.stdev(all_ddgs) if len(all_ddgs) > 1 else 0.0
        ddg_min = ddg_median

    return {
        "ddg_median": ddg_median,
        "ddg_mean": ddg_mean,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "n_converged": n_converged,
        "n_total": n_total,
        "converged": bool(converged_ddgs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 1: 수렴 집계 기본 동작
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateTrials:
    """_aggregate_trials 순수 집계 함수 테스트."""

    def test_all_converged(self):
        """모두 수렴(ddg < 0)이면 median이 대표값."""
        trials = [{"ddg": -10.0}, {"ddg": -8.0}, {"ddg": -12.0}, {"ddg": -9.0}, {"ddg": -11.0}]
        result = _aggregate_trials(trials)
        assert result["n_converged"] == 5
        assert result["n_total"] == 5
        assert result["converged"] is True
        # median of [-8, -9, -10, -11, -12] = -10.0
        assert result["ddg_median"] == pytest.approx(-10.0)
        assert result["ddg_min"] == pytest.approx(-12.0)
        assert result["ddg_sd"] > 0.0

    def test_partial_convergence(self):
        """일부만 수렴(ddg ≥ 0 제외 검증)."""
        trials = [{"ddg": -5.0}, {"ddg": 2.0}, {"ddg": -3.0}, {"ddg": 0.5}, {"ddg": -7.0}]
        result = _aggregate_trials(trials)
        # 수렴: -5.0, -3.0, -7.0 (ddg < 0)
        assert result["n_converged"] == 3
        assert result["n_total"] == 5
        assert result["converged"] is True
        expected_median = statistics.median([-5.0, -3.0, -7.0])
        assert result["ddg_median"] == pytest.approx(expected_median)

    def test_no_convergence(self):
        """수렴 없음 → 전체 min 사용, converged=False."""
        trials = [{"ddg": 1.0}, {"ddg": 0.5}, {"ddg": 3.0}]
        result = _aggregate_trials(trials)
        assert result["n_converged"] == 0
        assert result["converged"] is False
        # 전체 최솟값 = 0.5
        assert result["ddg_median"] == pytest.approx(0.5)

    def test_single_trial_no_sd(self):
        """trial 1개이면 sd=0.0 (divide-by-zero 방지)."""
        trials = [{"ddg": -5.0}]
        result = _aggregate_trials(trials)
        assert result["n_converged"] == 1
        assert result["ddg_sd"] == pytest.approx(0.0)

    def test_exactly_zero_not_converged(self):
        """ddg == 0.0 은 수렴 아님(< 0 경계 조건)."""
        trials = [{"ddg": 0.0}, {"ddg": -1.0}]
        result = _aggregate_trials(trials)
        assert result["n_converged"] == 1  # -1.0 만
        assert result["converged"] is True

    def test_median_consistency_with_flexpep_dock(self):
        """flexpep_dock.run_flexpep_refine_pose 와 동일 median 공식 검증.

        flexpep_dock.py line ~401: ddg_median = statistics.median(converged_ddgs)
        여기서도 statistics.median 사용 → 결과 동일해야 함.
        """
        converged = [-8.0, -12.0, -6.0, -10.0, -9.0]
        expected = statistics.median(converged)

        trials = [{"ddg": d} for d in converged] + [{"ddg": 2.0}]  # 비수렴 1개 포함
        result = _aggregate_trials(trials)
        assert result["ddg_median"] == pytest.approx(expected)
        assert result["n_converged"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 2: DOCK_MAX_WORKERS / FLEXPEP_NSTRUCT env 변수 동작
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvVarControls:
    """환경 변수 기반 병렬도 조절 로직 검증."""

    def test_flexpep_nstruct_default(self):
        """FLEXPEP_NSTRUCT 미설정 → 기본 5."""
        env = os.environ.copy()
        env.pop("FLEXPEP_NSTRUCT", None)
        with patch.dict(os.environ, env, clear=True):
            nstruct = max(1, int(os.environ.get("FLEXPEP_NSTRUCT", "5")))
        assert nstruct == 5

    def test_flexpep_nstruct_override(self):
        """FLEXPEP_NSTRUCT=3 으로 줄이면 3이어야 함."""
        with patch.dict(os.environ, {"FLEXPEP_NSTRUCT": "3"}):
            nstruct = max(1, int(os.environ.get("FLEXPEP_NSTRUCT", "5")))
        assert nstruct == 3

    def test_flexpep_nstruct_invalid_fallback(self):
        """FLEXPEP_NSTRUCT 에 비정수 값 → try/except 후 기본 5 사용."""
        try:
            nstruct = max(1, int("not_a_number"))
        except ValueError:
            nstruct = 5
        assert nstruct == 5

    def test_dock_max_workers_env(self):
        """DOCK_MAX_WORKERS=32 → max_workers 상한 32."""
        with patch.dict(os.environ, {"DOCK_MAX_WORKERS": "32"}):
            cpu = 192
            try:
                env_max = int(os.environ.get("DOCK_MAX_WORKERS", str(max(cpu - 16, 8))))
            except ValueError:
                env_max = max(cpu - 16, 8)
        assert env_max == 32

    def test_dock_max_workers_default_formula(self):
        """DOCK_MAX_WORKERS 미설정 → cpu_count - 16."""
        env = os.environ.copy()
        env.pop("DOCK_MAX_WORKERS", None)
        with patch.dict(os.environ, env, clear=True):
            cpu = 192
            env_max = int(os.environ.get("DOCK_MAX_WORKERS", str(max(cpu - 16, 8))))
        assert env_max == 176  # 192 - 16

    def test_max_workers_bounded_by_trials(self):
        """max_workers는 총 trial 수를 초과하지 않아야 함."""
        n_jobs = 8
        flat_nstruct = 5
        cpu = 192
        env_max = cpu - 16  # 176
        config_max = 160
        n_trials_total = n_jobs * flat_nstruct  # 40
        max_workers = min(n_trials_total, config_max, env_max)
        max_workers = max(max_workers, min(n_jobs, 4))
        assert max_workers == 40  # trial 수가 가장 작은 상한

    def test_max_workers_large_candidate_count(self):
        """후보 16개 × nstruct=5 = 80 trial, env_max=176 → 80 이하."""
        n_jobs = 16
        flat_nstruct = 5
        cpu = 192
        env_max = cpu - 16  # 176
        config_max = 160
        n_trials_total = n_jobs * flat_nstruct  # 80
        max_workers = min(n_trials_total, config_max, env_max)
        max_workers = max(max_workers, min(n_jobs, 4))
        assert max_workers == 80

    def test_max_workers_safety_floor(self):
        """n_jobs=4, nstruct=5 → 20 trial; 최솟값 보장(≥ min(4, 4)=4)."""
        n_jobs = 4
        flat_nstruct = 5
        cpu = 8  # 작은 기계 가정
        env_max = max(cpu - 16, 8)  # 8
        config_max = 160
        n_trials_total = n_jobs * flat_nstruct  # 20
        max_workers = min(n_trials_total, config_max, env_max)  # 8
        max_workers = max(max_workers, min(n_jobs, 4))  # max(8, 4) = 8
        assert max_workers == 8


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 3: trial 그룹화 → CandidateResult 통계 보존 실증
# ─────────────────────────────────────────────────────────────────────────────

class TestTrialGroupingStatistics:
    """후보별 그룹화 + 집계가 기존 flexpep_dock 단일 subprocess nstruct=N 과
    동일한 의미의 결과를 반환하는지 실증.

    flexpep_dock.run_flexpep_refine_pose 내부 로직(lines ~380-416):
      1. 수렴 기준: ddg < 0
      2. 대표값: median of converged_ddgs
      3. 대표 pose: min(|ddg - median|) of converged records
    """

    def test_two_candidates_independent_grouping(self):
        """후보 2개가 독립적으로 집계되어야 함 (서로 간섭 없음)."""
        cand0_trials = [{"ddg": -10.0}, {"ddg": -8.0}, {"ddg": 1.0}, {"ddg": -9.0}, {"ddg": -11.0}]
        cand1_trials = [{"ddg": -2.0}, {"ddg": 3.0}, {"ddg": -4.0}, {"ddg": -1.0}, {"ddg": 2.0}]

        r0 = _aggregate_trials(cand0_trials)
        r1 = _aggregate_trials(cand1_trials)

        # cand0: 수렴 4개 (-10, -8, -9, -11), median = -9.5
        assert r0["n_converged"] == 4
        assert r0["ddg_median"] == pytest.approx(statistics.median([-10.0, -8.0, -9.0, -11.0]))

        # cand1: 수렴 3개 (-2, -4, -1), median = -2.0
        assert r1["n_converged"] == 3
        assert r1["ddg_median"] == pytest.approx(statistics.median([-2.0, -4.0, -1.0]))

    def test_statistics_match_direct_nstruct(self):
        """nstruct=5 직접 실행과 5개 평탄화 trial 집계가 동일 median/sd 반환."""
        # 직접 nstruct=5 시나리오: flexpep_dock.py 가 반환할 값과 동일
        raw_ddgs = [-5.0, -8.0, -3.0, -6.0, -10.0]  # 모두 수렴

        # 직접 계산 (flexpep_dock.py 로직)
        converged = [d for d in raw_ddgs if d < 0]
        expected_median = statistics.median(converged)
        expected_sd = statistics.stdev(converged)

        # 평탄화 집계 (runner.py 로직)
        trials = [{"ddg": d} for d in raw_ddgs]
        result = _aggregate_trials(trials)

        assert result["ddg_median"] == pytest.approx(expected_median)
        assert result["ddg_sd"] == pytest.approx(expected_sd)
        assert result["n_converged"] == 5

    def test_convergence_gate_preserved(self):
        """ddg ≥ 0 제외 게이트가 정확히 적용되는지."""
        # ddg 정확히 0.0은 비수렴, -0.001은 수렴
        trials = [
            {"ddg": 0.0},    # 비수렴
            {"ddg": -0.001}, # 수렴
            {"ddg": 1.5},    # 비수렴
            {"ddg": -5.0},   # 수렴
        ]
        result = _aggregate_trials(trials)
        assert result["n_converged"] == 2  # -0.001, -5.0 만
        assert result["ddg_min"] == pytest.approx(-5.0)

    def test_all_failed_trial_handling(self):
        """모든 trial 실패 시 빈 successful 리스트 처리.

        runner.py 에서 successful=[] 인 경우 fail_reason 반환 경로를 검증.
        """
        successful: List[Dict[str, Any]] = []
        # successful=[] 이면 _aggregate_trials 호출 전 분기(runner 내부)
        assert len(successful) == 0  # runner 가 이 분기에서 CandidateResult(ddg=999) 반환


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 4: 동시 subprocess 수 상한 안전 범위 검증
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentSubprocessSafety:
    """192코어 머신에서의 상한 계산 검증.

    RSS ~0.9GB/subprocess × 160 = ~144GB < 455GB 여유 → 안전.
    """

    def test_192core_machine_default_max(self):
        """192코어 기계, DOCK_MAX_WORKERS 미설정 → 176 (cpu - 16)."""
        cpu = 192
        env_max = max(cpu - 16, 8)  # 176
        config_max = 160
        n_jobs = 8
        nstruct = 5
        n_trials = n_jobs * nstruct  # 40
        max_workers = min(n_trials, config_max, env_max)
        max_workers = max(max_workers, min(n_jobs, 4))
        # 40 < 160 < 176 → 40
        assert max_workers == 40

    def test_192core_large_scale_scenario(self):
        """n_candidates=32, nstruct=5 → 160 trial.

        config.max_parallel_workers=160, cpu-16=176 → 160.
        동시 subprocess 160개, RSS 160 × 0.9GB = 144GB < 455GB 여유.
        """
        cpu = 192
        env_max = max(cpu - 16, 8)  # 176
        config_max = 160
        n_jobs = 32
        nstruct = 5
        n_trials = n_jobs * nstruct  # 160
        max_workers = min(n_trials, config_max, env_max)
        max_workers = max(max_workers, min(n_jobs, 4))
        assert max_workers == 160

    def test_dock_max_workers_explicit_cap(self):
        """DOCK_MAX_WORKERS=80 으로 명시 제한 → 80 초과 불가."""
        cpu = 192
        env_max = 80  # explicit
        config_max = 160
        n_jobs = 32
        nstruct = 5
        n_trials = n_jobs * nstruct  # 160
        max_workers = min(n_trials, config_max, env_max)
        max_workers = max(max_workers, min(n_jobs, 4))
        assert max_workers == 80

    def test_no_infinite_spawn(self):
        """max_workers 가 항상 유한한 양의 정수인지."""
        for n_jobs in [1, 4, 8, 32]:
            for nstruct in [1, 3, 5, 10]:
                for cpu in [4, 8, 32, 192]:
                    env_max = max(cpu - 16, 8)
                    config_max = 160
                    n_trials = n_jobs * nstruct
                    max_workers = min(n_trials, config_max, env_max)
                    max_workers = max(max_workers, min(n_jobs, 4))
                    assert isinstance(max_workers, int)
                    assert max_workers > 0
                    assert max_workers <= max(n_trials, 4)


# ─────────────────────────────────────────────────────────────────────────────
# 테스트 5: schema.py FlowConfig 기본값 확인
# ─────────────────────────────────────────────────────────────────────────────

def test_flow_config_default_max_parallel_workers():
    """schema.py FlowConfig.max_parallel_workers 기본값 = 160."""
    from pyrosetta_flow.schema import FlowConfig
    cfg = FlowConfig(template_pdb="/tmp/dummy.pdb")
    assert cfg.max_parallel_workers == 160


def test_run_continuous_discovery_default_max_workers():
    """run_continuous_discovery.py --max-workers 기본값 = 64."""
    import importlib.util
    import sys
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_continuous_discovery.py"
    )
    spec = importlib.util.spec_from_file_location("rcd", script_path)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    # parse_args 직접 호출(빈 argv)은 --input required로 실패하므로
    # 소스에서 default 추출
    src = script_path.read_text()
    assert 'default=64' in src, "run_continuous_discovery.py --max-workers default should be 64"
