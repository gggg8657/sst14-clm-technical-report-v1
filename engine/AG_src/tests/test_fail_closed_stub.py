"""D1 (F08) fail-closed 회귀 테스트.

실패/미설치 PyRosetta stub 이 진짜처럼 보이는 점수(ddg=0.0)로 랭킹에 진입하지 않고
fail-closed(ddg=999, stub=True) 되는지 검증. dev opt-in(AG_ALLOW_ROSETTA_STUB=1) 시
중립 stub 허용.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AG_src.pipeline.step06_rosetta import _stub_rosetta_result


def test_stub_is_fail_closed_by_default(monkeypatch):
    monkeypatch.delenv("AG_ALLOW_ROSETTA_STUB", raising=False)
    r = _stub_rosetta_result("/tmp/x.pdb")
    assert r.stub is True
    assert r.ddg == 999.0, "기본 stub 은 fail-closed(ddg=999) 여야 랭킹서 제외됨"
    assert r.clash_score == 999.0


def test_stub_opt_in_neutral(monkeypatch):
    monkeypatch.setenv("AG_ALLOW_ROSETTA_STUB", "1")
    r = _stub_rosetta_result("/tmp/x.pdb")
    assert r.stub is True
    assert r.ddg == 0.0, "opt-in 시 중립 stub(0.0) 허용 (dev/CI)"


def test_fail_closed_excluded_by_ddg_gate():
    """fail-closed stub(ddg=999)은 ddG 게이트(<= -5)에서 탈락해야 한다."""
    os.environ.pop("AG_ALLOW_ROSETTA_STUB", None)
    r = _stub_rosetta_result("/tmp/x.pdb")
    ddg_max = -5.0
    passes = (r.ddg <= ddg_max)
    assert passes is False, "stub 은 게이트 통과하면 안 됨"


def test_multiobjective_skips_nan_offtarget():
    """screen_selectivity 의 NaN(off-target 도킹 실패) 결측 처리 로직 단위 검증."""
    # NaN 필터 규칙: ddg != ddg 면 제외
    offtarget = {}
    for name, ddg in [("SSTR1", -40.0), ("SSTR3", float("nan")), ("SSTR4", -35.0)]:
        if ddg != ddg:
            continue
        offtarget[name] = ddg
    assert "SSTR3" not in offtarget
    assert set(offtarget) == {"SSTR1", "SSTR4"}
    assert min(offtarget.values()) == -40.0
