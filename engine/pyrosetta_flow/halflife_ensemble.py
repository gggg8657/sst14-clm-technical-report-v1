# FROZEN (2026-06-25): 검증된 기준선 보존용 수정 금지. 개선은 halflife_ensemble_v2.py 에서. 발굴 import 전환됨.
"""halflife_ensemble.py — 반감기 앙상블 (2026-06-09).

두 상보적 추정기를 결합:
  A) 휴리스틱  : step08.predict_half_life — SST-14 절대 스케일·문헌 벤치(Spearman 0.86) 우수.
  C) RF(PEPlife2): halflife_model — in-domain CV Spearman 0.78 / R²log 0.64 (데이터 일반화).
두 값을 log10 [0.02h, 200h] 로 정규화 후 평균 → stability_norm 앙상블. RF/sklearn 미가용 시
휴리스틱 단독으로 graceful fallback. 표시용 half_life_h 는 휴리스틱(절대 스케일 신뢰) 사용.

honest: 둘 다 surrogate (임상 t½ 아님). 앙상블은 단일보다 강건성 향상을 노린 것이며 추가 검증 권장.
"""
from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_HL_LO, _HL_HI = math.log10(0.02), math.log10(200.0)  # 정규화 범위

# --- A: 휴리스틱 ---
try:
    from AG_src.pipeline.step08_stability import predict_half_life as _heuristic_hl
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


def ensemble_halflife(sequence: str, modifications=None, rec: Optional[Dict] = None) -> Dict[str, Any]:
    """반감기 앙상블. Returns:
        half_life_h          : 표시용(휴리스틱, 절대 스케일 신뢰; 없으면 RF)
        half_life_heuristic_h, half_life_rf_h
        stability_norm       : 두 정규화 값 평균(없으면 가용한 쪽)
        halflife_source      : 'ensemble' | 'heuristic' | 'rf' | 'none'
    """
    h = heuristic_hours(sequence, modifications)
    r = rf_hours(sequence, rec)
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
