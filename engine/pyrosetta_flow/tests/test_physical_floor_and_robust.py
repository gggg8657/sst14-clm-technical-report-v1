"""test_physical_floor_and_robust.py — physical floor + robust nstruct 단위 테스트 (2026-06-29).

테스트 항목:
  F-1: _ddg_physical_floor() 기본값 -150 반환
  F-2: DDG_PHYSICAL_FLOOR 환경변수로 floor 값 변경
  F-3: load_ddg_robust_stats — floor 이하 outlier(-510 등)가 수집에서 제외됨
  F-4: load_ddg_robust_stats — floor 이상 정상값은 수집됨
  F-5: load_ddg_robust_stats — floor 이하 값만 있는 서열은 통계 없음
  F-6: add_measurement — floor 이하 ddg 는 unphysical=True 마킹
  F-7: add_measurement — 정상 ddg 는 unphysical=False
  R-1: _is_non_robust_single — ddg_n_converged=1, sd=0 → True
  R-2: _is_non_robust_single — ddg_n_converged=5, sd>0 → False
  R-3: _is_non_robust_single — ddg_median=None → True (미산출)
  R-4: add_measurement — 단일값은 robust_pending=True
  R-5: add_measurement — n_converged≥5 있는 extra 는 robust_pending=False
  R-6: _resort — robust_pending 후보는 non-robust 보다 정렬 후순위
  R-7: apply_robust_ddg_stats — 통계 적용 후 robust_pending 재계산됨
  M-1: mmgbsa_daemon TOP_N == 100
  S-1: flexpep_dock _ddg_physical_floor 기본값과 환경변수 동작
  S-2: offtarget_dock dock_and_score 반환 키에 unphysical 존재
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest


# ── flexpep_dock._ddg_physical_floor 임포트 ────────────────────────────────────

def _import_flexpep_floor():
    """flexpep_dock._ddg_physical_floor 를 PyRosetta 없이 임포트."""
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[2] / "AG_src" / "scripts"),
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "flexpep_dock",
        Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "flexpep_dock.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # pyrosetta import 블록을 건너뛰기 위해 sys.modules 가짜 등록
    import types
    fake_pyrosetta = types.ModuleType("pyrosetta")
    sys.modules.setdefault("pyrosetta", fake_pyrosetta)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def flexpep_mod():
    return _import_flexpep_floor()


# ── global_leaderboard 임포트 ───────────────────────────────────────────────────

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from pyrosetta_flow.global_leaderboard import (
    GlobalSelectivityLeaderboard,
    _ddg_physical_floor,
    _is_non_robust_single,
    _is_pose_uncertain,
    load_ddg_robust_stats,
)


# ══════════════════════════════════════════════════════════════════════════════
# F: physical floor 기본 동작
# ══════════════════════════════════════════════════════════════════════════════

class TestFloor:
    """F-1 ~ F-7: DDG_PHYSICAL_FLOOR 환경변수 및 집계 제외."""

    def test_f1_default_floor(self):
        """F-1: 기본값 -150 반환."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DDG_PHYSICAL_FLOOR", None)
            assert _ddg_physical_floor() == -150.0

    def test_f2_env_override(self):
        """F-2: 환경변수로 -200 설정 시 반영."""
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-200"}):
            assert _ddg_physical_floor() == -200.0

    def _write_experiment_log(self, rows, tmpdir: str) -> Path:
        p = Path(tmpdir) / "experiment_log.jsonl"
        with open(p, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return p

    def test_f3_outlier_excluded(self, tmp_path):
        """F-3: -510 outlier는 load_ddg_robust_stats 수집에서 제외."""
        rows = [
            {"sequence": "AGCKNFFWKTFTSC", "ddg": -30.0},
            {"sequence": "AGCKNFFWKTFTSC", "ddg": -510.0},  # 비물리
            {"sequence": "AGCKNFFWKTFTSC", "ddg": -32.0},
        ]
        log_path = self._write_experiment_log(rows, str(tmp_path))
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-150"}):
            stats = load_ddg_robust_stats(log_path)
        result = stats.get("AGCKNFFWKTFTSC", {})
        # n_converged 는 2 (정상 2건, outlier 1건 제외)
        assert result.get("ddg_n_converged") == 2
        # median 은 -31.0 (-30, -32 의 중간)
        assert result.get("ddg_median") == pytest.approx(-31.0, abs=0.01)

    def test_f4_normal_collected(self, tmp_path):
        """F-4: 정상값(-30, -32)은 수집됨."""
        rows = [
            {"sequence": "SGCKNFFWKTFTSC", "ddg": -30.0},
            {"sequence": "SGCKNFFWKTFTSC", "ddg": -32.0},
        ]
        log_path = self._write_experiment_log(rows, str(tmp_path))
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-150"}):
            stats = load_ddg_robust_stats(log_path)
        result = stats.get("SGCKNFFWKTFTSC", {})
        assert result.get("ddg_n_converged") == 2

    def test_f5_only_outliers_gives_no_stats(self, tmp_path):
        """F-5: 비물리 값만 있는 서열은 통계 없음."""
        rows = [
            {"sequence": "GGCKNFFWKTFTSC", "ddg": -300.0},
            {"sequence": "GGCKNFFWKTFTSC", "ddg": -510.0},
        ]
        log_path = self._write_experiment_log(rows, str(tmp_path))
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-150"}):
            stats = load_ddg_robust_stats(log_path)
        assert "GGCKNFFWKTFTSC" not in stats

    def test_f6_add_measurement_unphysical_flag(self):
        """F-6: add_measurement — floor 이하 ddg 는 unphysical=True."""
        lb = GlobalSelectivityLeaderboard(capacity=10)
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-150"}):
            lb.add_measurement(
                seq="AGCKNFFWKTFTSC",
                ddg=-510.0,
                margin=5.0,
                delta_margin=1.0,
            )
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTFTSC"), None)
        assert entry is not None
        assert entry.get("unphysical") is True

    def test_f7_add_measurement_normal_not_unphysical(self):
        """F-7: 정상 ddg(-30)는 unphysical=False."""
        lb = GlobalSelectivityLeaderboard(capacity=10)
        with patch.dict(os.environ, {"DDG_PHYSICAL_FLOOR": "-150"}):
            lb.add_measurement(
                seq="AGCKNFFWKTDTSC",
                ddg=-30.0,
                margin=5.0,
                delta_margin=1.0,
            )
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTDTSC"), None)
        assert entry is not None
        assert entry.get("unphysical") is False


# ══════════════════════════════════════════════════════════════════════════════
# R: robust nstruct 단일값 판정 + canonical 순위
# ══════════════════════════════════════════════════════════════════════════════

class TestRobustPending:
    """R-1 ~ R-7: nstruct=1 단일값 canonical 자격 박탈."""

    def test_r1_single_is_non_robust(self):
        """R-1: n=1, sd=0 → _is_non_robust_single=True."""
        entry = {"ddg_median": -30.0, "ddg_n_converged": 1, "ddg_sd": 0.0}
        assert _is_non_robust_single(entry) is True

    def test_r2_multi_is_robust(self):
        """R-2: n=5, sd>0 → _is_non_robust_single=False."""
        entry = {"ddg_median": -30.0, "ddg_n_converged": 5, "ddg_sd": 2.5}
        assert _is_non_robust_single(entry) is False

    def test_r3_no_median_is_non_robust(self):
        """R-3: ddg_median=None → _is_non_robust_single=True."""
        entry = {"ddg_median": None}
        assert _is_non_robust_single(entry) is True

    def test_r4_single_ddg_robust_pending_true(self):
        """R-4: add_measurement — ddg_median 없으면 robust_pending=True."""
        lb = GlobalSelectivityLeaderboard(capacity=10)
        lb.add_measurement(
            seq="AGCKNFFWKTFTSA",
            ddg=-25.0,
            margin=5.0,
            delta_margin=1.0,
            extra={},
        )
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTFTSA"), None)
        assert entry is not None
        assert entry.get("robust_pending") is True

    def test_r5_multi_nstruct_robust_pending_false(self):
        """R-5: extra에 n_converged=5 있으면 robust_pending=False."""
        lb = GlobalSelectivityLeaderboard(capacity=10)
        lb.add_measurement(
            seq="AGCKNFFWKTFTSV",
            ddg=-25.0,
            margin=5.0,
            delta_margin=1.0,
            extra={
                "ddg_median": -25.0,
                "ddg_sd": 2.0,
                "ddg_n_converged": 5,
            },
        )
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTFTSV"), None)
        assert entry is not None
        assert entry.get("robust_pending") is False

    def test_r6_robust_pending_sorted_after_robust(self):
        """R-6: robust_pending 후보는 robust 후보보다 정렬 후순위."""
        lb = GlobalSelectivityLeaderboard(capacity=10, use_robust_ddg=True)
        # robust 후보 (n=5, 더 약한 ddg=-20)
        lb.add_measurement(
            seq="SGCKNFFWKTFTSC",
            ddg=-20.0,
            margin=5.0,
            delta_margin=1.0,
            extra={"ddg_median": -20.0, "ddg_sd": 2.0, "ddg_n_converged": 5},
        )
        # non-robust 단일값 후보 (더 강한 ddg=-35, 단 nstruct=1)
        lb.add_measurement(
            seq="AGCKNFFWKAFTSC",
            ddg=-35.0,
            margin=6.0,
            delta_margin=2.0,
            extra={},  # ddg_median 없음 → robust_pending=True
        )
        # robust 후보가 앞에 와야 함
        assert lb.entries[0]["sequence"] == "SGCKNFFWKTFTSC"
        assert lb.entries[1].get("robust_pending") is True

    def test_r7_apply_robust_recalculates_pending(self):
        """R-7: apply_robust_ddg_stats 후 robust_pending 재계산."""
        lb = GlobalSelectivityLeaderboard(capacity=10)
        lb.add_measurement(
            seq="AGCKNFFWKTFTSW",
            ddg=-25.0,
            margin=5.0,
            delta_margin=1.0,
            extra={},
        )
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTFTSW"), None)
        assert entry is not None
        assert entry.get("robust_pending") is True

        # 이제 robust 통계 적용 (n=5)
        lb.apply_robust_ddg_stats({
            "AGCKNFFWKTFTSW": {
                "ddg_median": -26.0,
                "ddg_sd": 2.5,
                "ddg_n_converged": 5,
            }
        })
        entry = next((e for e in lb.entries if e["sequence"] == "AGCKNFFWKTFTSW"), None)
        assert entry is not None
        assert entry.get("robust_pending") is False


# ══════════════════════════════════════════════════════════════════════════════
# M: mmgbsa_daemon TOP_N
# ══════════════════════════════════════════════════════════════════════════════

class TestMMGBSATopN:
    """M-1: mmgbsa_daemon.TOP_N == 100."""

    def test_m1_top_n_100(self):
        """M-1: scripts/mmgbsa_daemon.py의 TOP_N 상수가 100."""
        spec_path = (
            Path(__file__).resolve().parents[2] / "scripts" / "mmgbsa_daemon.py"
        )
        src = spec_path.read_text(encoding="utf-8")
        # "TOP_N: int = 100" 패턴 확인
        import re
        m = re.search(r"^TOP_N\s*:\s*int\s*=\s*(\d+)", src, re.MULTILINE)
        assert m is not None, "TOP_N 상수를 찾을 수 없습니다"
        assert int(m.group(1)) == 100, f"TOP_N이 100이 아닙니다: {m.group(1)}"


# ══════════════════════════════════════════════════════════════════════════════
# S: flexpep_dock / offtarget_dock 결과 키 검증 (PyRosetta 없이 AST/소스 확인)
# ══════════════════════════════════════════════════════════════════════════════

class TestDockOutputKeys:
    """S-1 ~ S-2: dock 스크립트 출력 키 검증 (소스 파싱)."""

    def test_s1_flexpep_floor_in_source(self):
        """S-1: flexpep_dock.py가 DDG_PHYSICAL_FLOOR 환경변수를 읽는다."""
        src_path = (
            Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "flexpep_dock.py"
        )
        src = src_path.read_text(encoding="utf-8")
        assert "DDG_PHYSICAL_FLOOR" in src

    def test_s2_unphysical_key_in_result(self):
        """S-2: flexpep_dock.py main()의 result dict에 'unphysical' 키가 있다."""
        src_path = (
            Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "flexpep_dock.py"
        )
        src = src_path.read_text(encoding="utf-8")
        # result dict 에 "unphysical" 키 존재
        assert '"unphysical"' in src or "'unphysical'" in src

    def test_s3_offtarget_unphysical_key(self):
        """S-3: offtarget_dock.py dock_and_score 반환에 'unphysical' 키가 있다."""
        src_path = (
            Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "offtarget_dock.py"
        )
        src = src_path.read_text(encoding="utf-8")
        assert '"unphysical"' in src or "'unphysical'" in src

    def test_s4_n_unphysical_renamed(self):
        """S-4: offtarget_dock.py에서 _unphysical_records 변수명 사용."""
        src_path = (
            Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "offtarget_dock.py"
        )
        src = src_path.read_text(encoding="utf-8")
        assert "_unphysical_records" in src

    def test_s5_flexpep_physical_records_selection(self):
        """S-5: flexpep_dock.py 대표 pose 선택 시 physical_records 필터 사용."""
        src_path = (
            Path(__file__).resolve().parents[2] / "AG_src" / "scripts" / "flexpep_dock.py"
        )
        src = src_path.read_text(encoding="utf-8")
        assert "physical_records" in src


# ══════════════════════════════════════════════════════════════════════════════
# AST 문법 검사
# ══════════════════════════════════════════════════════════════════════════════

class TestASTSyntax:
    """수정된 .py 파일 AST 문법 검사."""

    @pytest.mark.parametrize("rel_path", [
        "AG_src/scripts/flexpep_dock.py",
        "AG_src/scripts/offtarget_dock.py",
        "pyrosetta_flow/global_leaderboard.py",
        "scripts/mmgbsa_daemon.py",
    ])
    def test_ast_syntax(self, rel_path):
        """각 파일이 AST 파싱 가능한지 확인."""
        import ast
        full_path = Path(__file__).resolve().parents[2] / rel_path
        src = full_path.read_text(encoding="utf-8")
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{rel_path} AST 문법 오류: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# P: 포즈-인지 수렴 게이트 (2026-07-06) — 수렴 고-sd(floppy) 후보 헤드라인 승격 차단
# ══════════════════════════════════════════════════════════════════════════════

class TestPoseGate:
    """P-1 ~ P-6: pose_uncertain(floppy) 판정 + 정렬 penalty + high_confidence 부스트 차단."""

    def test_p1_high_sd_converged_is_pose_uncertain(self):
        """P-1: n>=2 수렴 + sd>임계(15) → pose_uncertain=True."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "1", "POSE_FLOPPY_SD_THRESHOLD": "15.0"}):
            assert _is_pose_uncertain({"ddg_median": -30.0, "ddg_n_converged": 5, "ddg_sd": 20.0}) is True

    def test_p2_low_sd_converged_not_uncertain(self):
        """P-2: n>=2 + sd<임계 → pose_uncertain=False (native sd~12.3 오탐 없음)."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "1", "POSE_FLOPPY_SD_THRESHOLD": "15.0"}):
            assert _is_pose_uncertain({"ddg_median": -30.0, "ddg_n_converged": 8, "ddg_sd": 12.3}) is False

    def test_p3_single_pose_not_uncertain(self):
        """P-3: n=1 단일포즈는 robust_pending 소관 → pose_uncertain=False."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "1"}):
            assert _is_pose_uncertain({"ddg_median": -50.0, "ddg_n_converged": 1, "ddg_sd": 0.0}) is False

    def test_p4_gate_disabled_never_uncertain(self):
        """P-4: POSE_GATE_ENABLED=0 → 항상 False (기존 동작 보존)."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "0"}):
            assert _is_pose_uncertain({"ddg_median": -30.0, "ddg_n_converged": 5, "ddg_sd": 99.0}) is False

    def test_p5_floppy_demoted_below_clean(self):
        """P-5: _resort — clean-converged(저sd)가 floppy(고sd) 위, floppy는 단일포즈 위."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "1", "POSE_FLOPPY_SD_THRESHOLD": "15.0",
                                     "MMGBSA_LEADERBOARD_BOOST": "0"}):
            lb = GlobalSelectivityLeaderboard(capacity=10)
            # floppy: 더 강한 median(-40)이지만 sd 큼(25)
            lb.add_measurement(seq="FLOPPYCFWKTFTC", ddg=-40.0, margin=5.0, delta_margin=2.0,
                               extra={"ddg_median": -40.0, "ddg_sd": 25.0, "ddg_n_converged": 6})
            # clean: median -35, 저sd
            lb.add_measurement(seq="CLEANCFFWKTFTC", ddg=-35.0, margin=5.0, delta_margin=2.0,
                               extra={"ddg_median": -35.0, "ddg_sd": 6.0, "ddg_n_converged": 6})
            seqs = [e["sequence"] for e in lb.entries]
            # clean 이 floppy 보다 위 (floppy 가 median 더 강해도 penalty로 후순위)
            assert seqs.index("CLEANCFFWKTFTC") < seqs.index("FLOPPYCFWKTFTC")
            floppy = next(e for e in lb.entries if e["sequence"] == "FLOPPYCFWKTFTC")
            assert floppy.get("pose_uncertain") is True

    def test_p6_floppy_high_confidence_no_boost(self):
        """P-6: floppy + high_confidence 여도 부스트 차단 → clean 위로 못 올라옴."""
        with patch.dict(os.environ, {"POSE_GATE_ENABLED": "1", "POSE_FLOPPY_SD_THRESHOLD": "15.0",
                                     "MMGBSA_LEADERBOARD_BOOST": "3.0"}):
            lb = GlobalSelectivityLeaderboard(capacity=10)
            lb.add_measurement(seq="FLOPPYCFWKTFTC", ddg=-40.0, margin=5.0, delta_margin=2.0,
                               extra={"ddg_median": -40.0, "ddg_sd": 25.0, "ddg_n_converged": 6})
            lb.add_measurement(seq="CLEANCFFWKTFTC", ddg=-35.0, margin=5.0, delta_margin=2.0,
                               extra={"ddg_median": -35.0, "ddg_sd": 6.0, "ddg_n_converged": 6})
            # floppy 에 high_confidence 부여 후 재정렬
            for e in lb.entries:
                if e["sequence"] == "FLOPPYCFWKTFTC":
                    e["consensus_flag"] = "high_confidence"
            lb._resort()
            seqs = [e["sequence"] for e in lb.entries]
            assert seqs.index("CLEANCFFWKTFTC") < seqs.index("FLOPPYCFWKTFTC")
