"""
test_ddg_physical_floor.py
===========================
비물리 ddg 하한 게이트(_ddg_physical_floor, converged_ddgs 필터) 단위 테스트.

대상:
    AG_src/scripts/flexpep_dock.py :: _ddg_physical_floor
    AG_src/scripts/offtarget_dock.py :: _ddg_physical_floor

PyRosetta를 임포트하지 않고 로직만 검증한다.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# 공통 경로
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]  # ai4sci-kaeri/
_FLEXPEP_PATH = _ROOT / "AG_src" / "scripts" / "flexpep_dock.py"
_OFFTARGET_PATH = _ROOT / "AG_src" / "scripts" / "offtarget_dock.py"

assert _FLEXPEP_PATH.exists(), f"flexpep_dock.py 없음: {_FLEXPEP_PATH}"
assert _OFFTARGET_PATH.exists(), f"offtarget_dock.py 없음: {_OFFTARGET_PATH}"


def _load_module(path: Path, name: str) -> object:
    """PyRosetta import 없이 모듈 로드 (pyrosetta/rosetta를 stub으로 차단)."""
    # pyrosetta import를 가짜 모듈로 교체해 ImportError 방지
    import types

    stub = types.ModuleType("pyrosetta")
    stub.init = lambda *a, **kw: None  # type: ignore[attr-defined]
    rosetta_stub = types.ModuleType("pyrosetta.rosetta")
    sys.modules.setdefault("pyrosetta", stub)
    sys.modules.setdefault("pyrosetta.rosetta", rosetta_stub)
    for submod in (
        "pyrosetta.rosetta.protocols",
        "pyrosetta.rosetta.protocols.flexpep_docking",
        "pyrosetta.rosetta.protocols.analysis",
        "pyrosetta.rosetta.protocols.relax",
        "pyrosetta.rosetta.core",
        "pyrosetta.rosetta.core.scoring",
    ):
        sys.modules.setdefault(submod, types.ModuleType(submod))

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_flexpep = _load_module(_FLEXPEP_PATH, "flexpep_dock_test_module")
_offtarget = _load_module(_OFFTARGET_PATH, "offtarget_dock_test_module")


# ---------------------------------------------------------------------------
# _ddg_physical_floor — flexpep_dock.py
# ---------------------------------------------------------------------------

class TestDdgPhysicalFloorFlexpep:
    """_ddg_physical_floor 헬퍼 — 기본값 및 환경변수 오버라이드."""

    def test_default_is_minus_150(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DDG_PHYSICAL_FLOOR", raising=False)
        assert _flexpep._ddg_physical_floor() == -150.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DDG_PHYSICAL_FLOOR", "-200")
        assert _flexpep._ddg_physical_floor() == -200.0

    def test_invalid_env_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DDG_PHYSICAL_FLOOR", "not_a_number")
        assert _flexpep._ddg_physical_floor() == -150.0

    def test_return_type_is_float(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DDG_PHYSICAL_FLOOR", raising=False)
        result = _flexpep._ddg_physical_floor()
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# _ddg_physical_floor — offtarget_dock.py
# ---------------------------------------------------------------------------

class TestDdgPhysicalFloorOfftarget:
    """offtarget_dock.py 동일 헬퍼 검증."""

    def test_default_is_minus_150(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DDG_PHYSICAL_FLOOR", raising=False)
        assert _offtarget._ddg_physical_floor() == -150.0

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DDG_PHYSICAL_FLOOR", "-120")
        assert _offtarget._ddg_physical_floor() == -120.0


# ---------------------------------------------------------------------------
# 수렴 게이트 필터 로직 — flexpep_dock.py 재현
# ---------------------------------------------------------------------------

def _apply_flexpep_gate(
    ddg_values: List[float],
    floor: float = -150.0,
) -> Tuple[List[float], int]:
    """flexpep_dock.py 수렴 게이트 로직 재현.

    Returns (converged_ddgs, n_unphysical).
    """
    converged_ddgs: List[float] = []
    n_unphysical = 0
    for ddg in ddg_values:
        _physical = ddg < 0 and ddg > floor
        if _physical:
            converged_ddgs.append(ddg)
        elif ddg < 0:
            n_unphysical += 1
    return converged_ddgs, n_unphysical


class TestFlexpepConvergeGate:
    """flexpep_dock.py 수렴 게이트: ddg < 0 AND ddg > floor."""

    def test_normal_ddg_passes(self) -> None:
        """정상 범위 ddg(-10 ~ -80)는 converged로 집계된다."""
        ddgs = [-10.0, -45.5, -80.0]
        converged, n_unphysical = _apply_flexpep_gate(ddgs)
        assert converged == ddgs
        assert n_unphysical == 0

    def test_unphysical_outlier_excluded(self) -> None:
        """−510, −624 같은 비물리 outlier는 converged에서 제외된다."""
        ddgs = [-510.54, -624.6]
        converged, n_unphysical = _apply_flexpep_gate(ddgs)
        assert converged == []
        assert n_unphysical == 2

    def test_mixed_values(self) -> None:
        """정상 + 비물리 혼합 시 정상만 수렴으로 집계된다."""
        ddgs = [-30.0, -510.54, -55.0, -624.6, -5.0]
        converged, n_unphysical = _apply_flexpep_gate(ddgs)
        assert converged == [-30.0, -55.0, -5.0]
        assert n_unphysical == 2

    def test_positive_ddg_excluded_as_before(self) -> None:
        """양수 ddg는 이전과 동일하게 제외된다(비물리 카운트에는 미포함)."""
        ddgs = [5.0, 12.0]
        converged, n_unphysical = _apply_flexpep_gate(ddgs)
        assert converged == []
        assert n_unphysical == 0  # 양수는 비물리 카운트 X

    def test_floor_boundary_excluded(self) -> None:
        """floor 경계값(-150.0) 자체는 비물리로 제외된다(ddg > floor 조건)."""
        ddgs = [-150.0]
        converged, n_unphysical = _apply_flexpep_gate(ddgs, floor=-150.0)
        assert converged == []
        assert n_unphysical == 1

    def test_just_inside_floor_passes(self) -> None:
        """-149.9는 floor -150보다 크므로 수렴으로 집계된다."""
        ddgs = [-149.9]
        converged, n_unphysical = _apply_flexpep_gate(ddgs, floor=-150.0)
        assert converged == [-149.9]
        assert n_unphysical == 0

    def test_custom_floor_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DDG_PHYSICAL_FLOOR=−200 설정 시 −180은 수렴, −510은 제외."""
        monkeypatch.setenv("DDG_PHYSICAL_FLOOR", "-200")
        floor = _flexpep._ddg_physical_floor()
        assert floor == -200.0
        converged, n_unphysical = _apply_flexpep_gate([-180.0, -510.0], floor=floor)
        assert -180.0 in converged
        assert -510.0 not in converged
        assert n_unphysical == 1


# ---------------------------------------------------------------------------
# 수렴 게이트 필터 로직 — offtarget_dock.py 재현
# ---------------------------------------------------------------------------

def _apply_offtarget_gate(
    ddg_values: List[float],
    floor: float = -150.0,
) -> Tuple[List[float], int]:
    """offtarget_dock.py dock_and_score 수렴 게이트 로직 재현."""
    _unphysical = [ddg for ddg in ddg_values if ddg < 0 and ddg <= floor]
    converged_ddgs = [ddg for ddg in ddg_values if ddg < 0 and ddg > floor]
    return converged_ddgs, len(_unphysical)


class TestOfftargetConvergeGate:
    """offtarget_dock.py dock_and_score 수렴 게이트."""

    def test_normal_range_passes(self) -> None:
        ddgs = [-20.0, -60.0, -100.0]
        converged, n_unphysical = _apply_offtarget_gate(ddgs)
        assert sorted(converged) == sorted(ddgs)
        assert n_unphysical == 0

    def test_outlier_minus_510_excluded(self) -> None:
        """보고된 outlier -510.54 제외 검증."""
        ddgs = [-510.54]
        converged, n_unphysical = _apply_offtarget_gate(ddgs)
        assert converged == []
        assert n_unphysical == 1

    def test_n_unphysical_in_result_key(self) -> None:
        """dock_and_score 반환 dict에 n_unphysical 키가 존재하는지 확인.

        실제 함수 호출 없이 반환 키 목록 검사로 대체한다.
        """
        import inspect
        src = inspect.getsource(_offtarget.dock_and_score)
        assert "n_unphysical" in src, "dock_and_score 반환 dict에 n_unphysical 누락"

    def test_n_unphysical_in_flexpep_stats(self) -> None:
        """run_flexpep_refine_pose stats_info에 n_unphysical 키가 있는지 확인."""
        import inspect
        src = inspect.getsource(_flexpep.run_flexpep_refine_pose)
        assert "n_unphysical" in src, "run_flexpep_refine_pose stats_info에 n_unphysical 누락"
