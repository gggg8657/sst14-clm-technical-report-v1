"""test_surrogate_v2_equivalence.py — surrogate 원본 vs _v2 수치 동등성 검증.

인프라 Phase (2026-06-25): _v2 사본은 동작이 원본과 완전히 동일해야 한다.
정확도 개선은 이번 Phase 범위 밖 — 동일 입력에 동일 출력이 나오면 PASS.

대상:
  - predict_half_life: AG_src.pipeline.step08_stability (원본)
    vs pyrosetta_flow.surrogate_v2.step08_stability_v2 (_v2)
  - ensemble_halflife: pyrosetta_flow.halflife_ensemble (원본)
    vs pyrosetta_flow.halflife_ensemble_v2 (_v2)
  - pepadmet_toxicity / pepadmet_toxicity_v2 는 subprocess 의존(실 추론 환경 필요),
    여기서는 모듈 임포트 + 공개 API 시그니처 동일성만 검사.
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# REPO 루트를 sys.path에 추가 (conftest 미적용 환경 대비)
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# 테스트 서열 목록 (다양한 유형: native, 짧은 것, Phe/Trp 없는 것)
# ---------------------------------------------------------------------------
_TEST_SEQS: List[str] = [
    "AGCKNFFWKTFTSC",   # native SST-14
    "AGCKNFFWKTDTSC",   # F11D 변이 (pos11 = Asp)
    "AGCKNAFWKTFTSC",   # F7A 변이
    "ACDEF",            # 짧은 서열
    "AGCKNFFWKTFTSC",   # 중복 — 캐시 동작 확인용
    "ACKNFFWKTFTSA",    # C1A/C14A (SS bond 없음)
    "MAGICWAND",        # 임의 서열 (알파벳 대부분 포함)
]

_TEST_MODS_SETS: List[List[Any]] = [
    [],
    ["cyclization"],
    ["d_amino_acid", "cyclization"],
    ["fatty_acid"],
    ["pegylation", "d_amino_acid"],
]


# ---------------------------------------------------------------------------
# 1. predict_half_life 동등성
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def step08_orig():
    """원본 predict_half_life."""
    from AG_src.pipeline.step08_stability import predict_half_life
    return predict_half_life


@pytest.fixture(scope="module")
def step08_v2():
    """_v2 predict_half_life."""
    from pyrosetta_flow.surrogate_v2.step08_stability_v2 import predict_half_life
    return predict_half_life


def _has_d_amino_acid(mods: list) -> bool:
    """modifications에 D-aa 항목이 포함되는지 확인 (동등성 skip 판정용)."""
    return any(
        "d_amino" in str(m).lower() or "d-amino" in str(m).lower()
        for m in mods
    )


@pytest.mark.parametrize("seq", _TEST_SEQS)
@pytest.mark.parametrize("mods", _TEST_MODS_SETS)
def test_predict_half_life_equivalence(step08_orig, step08_v2, seq: str, mods: list):
    """원본 predict_half_life == _v2 predict_half_life (동일 입력).

    2026-06-25 변경: D-aa 포함 케이스는 _v2에서 잔기수 비례 계수로 개선되어
    의도적으로 orig와 달라짐 → 해당 케이스는 동등성 검증 대신 skip.
    D-aa 정확도 개선 검증은 test_daa_halflife_improvement_v2.py 에서 수행.
    """
    if _has_d_amino_acid(mods):
        pytest.skip(
            "D-aa 케이스는 _v2에서 잔기수 비례 계수로 의도적으로 개선됨 "
            "(test_daa_halflife_improvement_v2.py 에서 별도 검증)"
        )

    orig_val = step08_orig(seq, mods)
    v2_val = step08_v2(seq, mods)

    # float 반환: 절대 오차 1e-9 이하 (부동소수점 재현성)
    assert isinstance(orig_val, float), f"원본 반환값이 float가 아님: {type(orig_val)}"
    assert isinstance(v2_val, float), f"_v2 반환값이 float가 아님: {type(v2_val)}"

    if math.isnan(orig_val) and math.isnan(v2_val):
        return  # 둘 다 NaN → 동등
    assert not (math.isnan(orig_val) ^ math.isnan(v2_val)), (
        f"NaN 불일치: orig={orig_val}, v2={v2_val} (seq={seq}, mods={mods})"
    )
    assert abs(orig_val - v2_val) < 1e-9, (
        f"predict_half_life 불일치: orig={orig_val}, v2={v2_val} "
        f"(seq={seq}, mods={mods})"
    )


# ---------------------------------------------------------------------------
# 2. ensemble_halflife 동등성
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ensemble_orig():
    """원본 ensemble_halflife."""
    from pyrosetta_flow.halflife_ensemble import ensemble_halflife
    return ensemble_halflife


@pytest.fixture(scope="module")
def ensemble_v2():
    """_v2 ensemble_halflife."""
    from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife
    return ensemble_halflife


_ENSEMBLE_KEYS = [
    "half_life_h",
    "half_life_heuristic_h",
    "half_life_rf_h",
    "stability_norm",
    "halflife_source",
]


@pytest.mark.parametrize("seq", _TEST_SEQS)
@pytest.mark.parametrize("mods", [[], ["d_amino_acid", "cyclization"], ["fatty_acid"]])
def test_ensemble_halflife_equivalence(ensemble_orig, ensemble_v2, seq: str, mods: list):
    """ensemble_halflife 원본 == _v2 모든 키 수치 동등 검증.

    2026-06-25 변경: D-aa 포함 케이스는 _v2에서 의도적 개선(휴리스틱+RF has_D)으로
    orig와 달라짐 → skip. 검증은 test_daa_halflife_improvement_v2.py 참조.
    """
    if _has_d_amino_acid(mods):
        pytest.skip(
            "D-aa 케이스는 _v2에서 의도적으로 개선됨 "
            "(test_daa_halflife_improvement_v2.py 에서 별도 검증)"
        )
    orig: Dict[str, Any] = ensemble_orig(seq, modifications=mods)
    v2: Dict[str, Any] = ensemble_v2(seq, modifications=mods)

    # 반환 키 집합 동일
    assert set(orig.keys()) == set(v2.keys()), (
        f"키 집합 불일치: orig={set(orig.keys())}, v2={set(v2.keys())}"
    )

    for key in _ENSEMBLE_KEYS:
        ov = orig[key]
        vv = v2[key]
        if isinstance(ov, float) and isinstance(vv, float):
            if math.isnan(ov) and math.isnan(vv):
                continue
            assert abs(ov - vv) < 1e-9, (
                f"ensemble_halflife[{key}] 불일치: orig={ov}, v2={vv} "
                f"(seq={seq}, mods={mods})"
            )
        else:
            assert ov == vv, (
                f"ensemble_halflife[{key}] 불일치: orig={ov!r}, v2={vv!r} "
                f"(seq={seq}, mods={mods})"
            )


# ---------------------------------------------------------------------------
# 3. pepadmet_toxicity / pepadmet_toxicity_v2 모듈 API 동일성 (import + 시그니처)
# ---------------------------------------------------------------------------

def test_pepadmet_toxicity_v2_api():
    """pepadmet_toxicity_v2가 원본과 동일한 공개 함수를 노출하는지 확인."""
    import pyrosetta_flow.pepadmet_toxicity as orig_mod
    import pyrosetta_flow.pepadmet_toxicity_v2 as v2_mod

    public_funcs = ["predict_toxicity", "batch_predict_toxicity"]
    for fn in public_funcs:
        assert hasattr(orig_mod, fn), f"원본에 {fn} 없음"
        assert hasattr(v2_mod, fn), f"_v2에 {fn} 없음"

    # 내부 헬퍼도 동일하게 존재
    assert hasattr(v2_mod, "_normalize_runner_row"), "_v2에 _normalize_runner_row 없음"


# ---------------------------------------------------------------------------
# 4. import 순환 없음 확인
# ---------------------------------------------------------------------------

def test_no_circular_import_halflife_v2():
    """halflife_ensemble_v2 → surrogate_v2.step08_stability_v2 단방향 확인."""
    # 이미 import된 경우 캐시에서 반환되므로, sys.modules에서 직접 확인
    import pyrosetta_flow.halflife_ensemble_v2 as hev2
    import pyrosetta_flow.surrogate_v2.step08_stability_v2 as s08v2

    # step08_stability_v2가 halflife_ensemble_v2를 역참조하지 않음
    s08v2_src = Path(s08v2.__file__).read_text()
    assert "halflife_ensemble_v2" not in s08v2_src, (
        "step08_stability_v2.py가 halflife_ensemble_v2를 import함 — 순환 위험"
    )
    assert "halflife_ensemble" not in s08v2_src, (
        "step08_stability_v2.py가 halflife_ensemble을 import함 — 불필요 의존"
    )


# ---------------------------------------------------------------------------
# 5. surrogate_v2 __init__.py 패키지 인식 확인
# ---------------------------------------------------------------------------

def test_surrogate_v2_package():
    """surrogate_v2가 패키지로 임포트 가능한지 확인."""
    import pyrosetta_flow.surrogate_v2 as sv2_pkg
    assert sv2_pkg is not None
    # __init__.py가 빈 파일이므로 별도 속성 없음
    assert hasattr(sv2_pkg, "__path__"), "패키지 __path__ 없음"
