# FROZEN_COPY_V2 (2026-06-25): halflife_ensemble.py 검증 기준선 사본.
# 원본은 pyrosetta_flow/halflife_ensemble.py (동결).
# 이 사본에서만 정확도 개선 수정 허용.
"""halflife_ensemble.py — 반감기 앙상블 (2026-06-09).

두 상보적 추정기를 결합:
  A) 휴리스틱  : step08.predict_half_life — SST-14 절대 스케일·문헌 벤치(Spearman 0.86) 우수.
  C) RF(PEPlife2): halflife_model — in-domain CV Spearman 0.78 / R²log 0.64 (데이터 일반화).
두 값을 log10 [0.02h, 200h] 로 정규화 후 평균 → stability_norm 앙상블. RF/sklearn 미가용 시
휴리스틱 단독으로 graceful fallback. 표시용 half_life_h 는 휴리스틱(절대 스케일 신뢰) 사용.

honest: 둘 다 surrogate (임상 t½ 아님). 앙상블은 단일보다 강건성 향상을 노린 것이며 추가 검증 권장.

2026-06-25 개선 (_v2):
  - RF rf_hours() 호출 시 rec에 has_D 정보 자동 주입 (버그수정: rec=None → has_D=0 강제).
    수정 전: ensemble_halflife(seq, mods) → rf_hours(seq, rec=None) → has_D=0 무조건.
    수정 후: modifications 또는 서열 소문자에서 D-aa 감지 → rec["chiral"]="D" or "L".
  - 휴리스틱은 step08_stability_v2.predict_half_life (D-aa 잔기수 비례 계수 개선판).
"""
from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pyrosetta_flow.halflife_ensemble_v2")

_HL_LO, _HL_HI = math.log10(0.02), math.log10(200.0)  # 정규화 범위

# --- A: 휴리스틱 ---
try:
    from pyrosetta_flow.surrogate_v2.step08_stability_v2 import predict_half_life as _heuristic_hl
    _HAS_HEUR = True
except Exception:  # pragma: no cover
    _heuristic_hl = None
    _HAS_HEUR = False

# --- C: RF (joblib) ---
_RF_BUNDLE = None
_RF_TRIED = False
_MODEL_PATH = Path(__file__).resolve().parent / "halflife_model" / "halflife_gbr.joblib"


def _load_rf():
    global _RF_BUNDLE, _RF_TRIED
    if _RF_TRIED:
        return _RF_BUNDLE
    _RF_TRIED = True
    try:
        import joblib
        _RF_BUNDLE = joblib.load(_MODEL_PATH)
    except Exception as exc:  # sklearn/joblib/모델 부재 → fallback
        logger.warning("half-life RF 모델 로드 실패(%s) → 휴리스틱 단독", exc)
        _RF_BUNDLE = None
    return _RF_BUNDLE


def rf_hours(sequence: str, rec: Optional[Dict] = None) -> Optional[float]:
    b = _load_rf()
    if b is None:
        return None
    try:
        import numpy as np
        from .halflife_model.features import featurize
        f = featurize(sequence, rec or {})
        if f is None:
            return None
        x = np.array([[f[c] for c in b["feat_cols"]]], float)
        return float(np.expm1(b["model"].predict(x)[0]))
    except Exception as exc:  # pragma: no cover
        logger.warning("half-life RF 추론 실패(%s)", exc)
        return None


def heuristic_hours(sequence: str, modifications=None) -> Optional[float]:
    if not _HAS_HEUR:
        return None
    try:
        return float(_heuristic_hl(sequence, modifications or []))
    except Exception:
        return None


def _norm_log(hl: Optional[float]) -> Optional[float]:
    if hl is None or hl != hl or hl <= 0:
        return None
    return min(1.0, max(0.0, (math.log10(hl) - _HL_LO) / (_HL_HI - _HL_LO)))


def _detect_d_aa(sequence: str, modifications: Optional[List] = None) -> bool:
    """서열 또는 modifications에서 D-aa 포함 여부를 감지한다.

    감지 우선순위:
      1) modifications 리스트에 'd_amino_acid' 또는 'd-amino' 문자열 포함.
      2) 서열에 소문자 알파벳 포함 (D-aa 소문자 표기 관행).

    Args:
        sequence:      펩타이드 서열 (대/소문자 혼용 가능).
        modifications: modification 유형 목록.

    Returns:
        D-aa 포함 여부 (bool).
    """
    mods = modifications or []
    for mod in mods:
        m_lower = str(mod).lower()
        if "d_amino" in m_lower or "d-amino" in m_lower:
            return True
    # 서열 소문자 토큰
    return any(c.islower() and c.upper().isalpha() for c in sequence)


def ensemble_halflife(sequence: str, modifications=None, rec: Optional[Dict] = None) -> Dict[str, Any]:
    """반감기 앙상블. Returns:
        half_life_h          : 표시용(휴리스틱, 절대 스케일 신뢰; 없으면 RF)
        half_life_heuristic_h, half_life_rf_h
        stability_norm       : 두 정규화 값 평균(없으면 가용한 쪽)
        halflife_source      : 'ensemble' | 'heuristic' | 'rf' | 'none'

    2026-06-25: rec=None 일 때 D-aa 정보를 자동 주입 (has_D 피처 활성화 버그 수정).
    """
    # RF rec에 D-aa 정보 자동 주입 (rec이 None이거나 chiral 키 없을 때만)
    # caller가 rec을 이미 제공했다면 그대로 사용 (override 방지).
    _rec = dict(rec) if rec is not None else {}
    if "chiral" not in _rec:
        _rec["chiral"] = "D" if _detect_d_aa(sequence, modifications) else "L"

    h = heuristic_hours(sequence, modifications)
    r = rf_hours(sequence, _rec)
    nh, nr = _norm_log(h), _norm_log(r)
    norms = [v for v in (nh, nr) if v is not None]
    if nh is not None and nr is not None:
        source = "ensemble"
    elif nh is not None:
        source = "heuristic"
    elif nr is not None:
        source = "rf"
    else:
        source = "none"
    stability_norm = round(sum(norms) / len(norms), 4) if norms else 0.0
    display_h = h if h is not None else r
    return {
        "half_life_h": round(display_h, 3) if display_h is not None else float("nan"),
        "half_life_heuristic_h": round(h, 3) if h is not None else None,
        "half_life_rf_h": round(r, 3) if r is not None else None,
        "stability_norm": stability_norm,
        "halflife_source": source,
    }
