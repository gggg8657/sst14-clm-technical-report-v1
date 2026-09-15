"""test_hemolysis_aliphatic.py — Lane C 물성 기반 용혈 휴리스틱 단위 테스트.

검증 항목:
  1. AI > 120 직쇄 L-aa → HIGH, applies=True
  2. AI <= 120 직쇄 L-aa → LOW, applies=True
  3. cyclic SST-14 계열 (Cys3-Cys14 보존) → NA, applies=False
  4. 비표준 잔기(D-aa 표기 등) → NA, applies=False
  5. poly-L 고AI → HIGH, applies=True
  6. soft penalty: HIGH면 admet_score 감점, 탈락(reject) 없음
  7. applies=False: penalty 없음
  8. enrich_candidates에서 native SST-14(cyclic)는 페널티 없음
  9. apply_hemolysis_penalty 함수 단독 검증
 10. 빈 서열 안전 처리 (applies=False, crash 없음)
"""
from __future__ import annotations

import math
import pytest

from pyrosetta_flow.multiobjective import (
    NATIVE_SST14,
    hemolysis_aliphatic_flag,
    apply_hemolysis_penalty,
    _AI_HIGH_THRESHOLD,
    _HEMOLYSIS_SOFT_PENALTY,
)


# ---------------------------------------------------------------------------
# 1. AI > 120 직쇄 L-aa → HIGH, applies=True
# ---------------------------------------------------------------------------

def test_high_ai_linear_laa_is_high():
    """Aliphatic Index > 120 직쇄 L-aa 표준서열 → HIGH, applies=True."""
    # AAAA...VAL(I/L)이 많은 서열 → AI 높음
    # Aliphatic Index = 100 × xA + 2.9 × xV + 3.9 × (xI + xL)
    # LLLLLLLLLLL → AI ≈ 100*0 + 2.9*0 + 3.9*(11/11)*100 = 390 >> 120
    result = hemolysis_aliphatic_flag("LLLLLLLLLLL")
    assert result["applies"] is True, "직쇄 L-aa는 applies=True이어야 함"
    assert result["hemolysis_risk"] == "HIGH", f"AI={result['aliphatic_index']} > 120이어야 HIGH"
    assert result["aliphatic_index"] > _AI_HIGH_THRESHOLD


# ---------------------------------------------------------------------------
# 2. AI <= 120 직쇄 L-aa → LOW, applies=True
# ---------------------------------------------------------------------------

def test_low_ai_linear_laa_is_low():
    """Aliphatic Index ≤ 120 직쇄 L-aa → LOW, applies=True."""
    # 주로 극성/하전 아미노산(K,R,E,D,G,S) → AI 낮음
    # GGGGGGGGGGG → aliphatic_index ≈ 0 (Gly는 aliphatic side-chain 없음)
    result = hemolysis_aliphatic_flag("GKRDESNPQTH")
    assert result["applies"] is True
    assert result["hemolysis_risk"] == "LOW", f"AI={result['aliphatic_index']} ≤ 120이어야 LOW"
    assert result["aliphatic_index"] <= _AI_HIGH_THRESHOLD


# ---------------------------------------------------------------------------
# 3. cyclic SST-14 계열 (Cys3-Cys14) → NA, applies=False
# ---------------------------------------------------------------------------

def test_cyclic_sst14_native_is_na():
    """native SST-14 (AGCKNFFWKTFTSC, Cys3-Cys14 cyclic) → applies=False, risk=NA."""
    result = hemolysis_aliphatic_flag(NATIVE_SST14)
    assert result["applies"] is False, "SST-14 cyclic은 applies=False이어야 함"
    assert result["hemolysis_risk"] == "NA"
    assert "cyclic" in result["reason"].lower() or "SST-14" in result["reason"] or "ss-bond" in result["reason"].lower()


def test_cyclic_variant_sst14_is_na():
    """SST-14 변이체 중 Cys3-Cys14 보존 서열 → applies=False."""
    # F11D 변이체: AGCKNFFWKTDTSC (길이 14, Cys[2]=C, Cys[13]=C)
    result = hemolysis_aliphatic_flag("AGCKNFFWKTDTSC")
    assert result["applies"] is False
    assert result["hemolysis_risk"] == "NA"


# ---------------------------------------------------------------------------
# 4. 비표준 잔기 포함 → NA, applies=False
# ---------------------------------------------------------------------------

def test_nonstandard_residue_is_na():
    """소문자 또는 표준 20종 외 문자 포함 서열 → applies=False, risk=NA."""
    # 'X'는 표준 20종에 없음 (D-aa 표기 등 가능성)
    result = hemolysis_aliphatic_flag("LLLLXLLLLL")
    assert result["applies"] is False
    assert result["hemolysis_risk"] == "NA"
    assert "비표준" in result["reason"] or "NONE" in result["reason"]


def test_mixed_case_d_aa_marker_is_na():
    """혼합 대소문자 소문자 표준 AA(D-aa marker) → applies=False, risk=NA."""
    result = hemolysis_aliphatic_flag("AGCkNFFWKTFTSC")
    assert result["applies"] is False
    assert result["hemolysis_risk"] == "NA"
    assert "D-aa" in result["reason"] or "MIXED" in result["reason"]


# ---------------------------------------------------------------------------
# 5. poly-L 고AI 직쇄 → HIGH, applies=True
# ---------------------------------------------------------------------------

def test_poly_leu_high_ai():
    """LLLLLLLLLLLLLL (14잔기, 직쇄 L-aa, 초고AI) → HIGH, applies=True."""
    result = hemolysis_aliphatic_flag("LLLLLLLLLLLLLL")
    assert result["applies"] is True
    assert result["hemolysis_risk"] == "HIGH"
    assert result["aliphatic_index"] > 300  # poly-Leu는 AI ≫ 120


# ---------------------------------------------------------------------------
# 6. soft penalty: HIGH면 admet_score 감점, 탈락(reject) 없음
# ---------------------------------------------------------------------------

def test_soft_penalty_reduces_admet_not_zero():
    """HIGH risk → admet_score 감점이지만 0이 되지 않음 (탈락 아님)."""
    hemo = {"applies": True, "hemolysis_risk": "HIGH", "aliphatic_index": 200.0, "reason": "test"}
    extra = {"admet_score": 0.80}
    apply_hemolysis_penalty(extra, hemo)
    assert extra["admet_score"] < 0.80, "HIGH는 감점이어야 함"
    assert extra["admet_score"] > 0.0, "감점이지만 0보다 커야 함 (탈락 아님)"
    expected = round(0.80 * _HEMOLYSIS_SOFT_PENALTY, 4)
    assert abs(extra["admet_score"] - expected) < 1e-6


def test_soft_penalty_factor_is_mild():
    """penalty factor 0.90~0.95 범위 확인 (너무 가혹하지 않음)."""
    assert 0.85 <= _HEMOLYSIS_SOFT_PENALTY <= 0.99, \
        f"soft penalty {_HEMOLYSIS_SOFT_PENALTY}는 0.85~0.99 범위이어야 함"


# ---------------------------------------------------------------------------
# 7. applies=False → penalty 없음
# ---------------------------------------------------------------------------

def test_no_penalty_when_applies_false():
    """applies=False (cyclic/비표준) → admet_score 변경 없음."""
    hemo = {"applies": False, "hemolysis_risk": "NA", "aliphatic_index": float("nan"), "reason": "cyclic"}
    extra = {"admet_score": 0.75}
    apply_hemolysis_penalty(extra, hemo)
    assert extra["admet_score"] == 0.75, "applies=False는 penalty 없음"
    assert extra["hemolysis_risk"] == "NA"
    assert extra["hemolysis_applies"] is False


def test_no_penalty_when_low_risk():
    """applies=True + LOW risk → admet_score 변경 없음."""
    hemo = {"applies": True, "hemolysis_risk": "LOW", "aliphatic_index": 80.0, "reason": "LOW"}
    extra = {"admet_score": 0.70}
    apply_hemolysis_penalty(extra, hemo)
    assert extra["admet_score"] == 0.70, "LOW risk는 penalty 없음"


# ---------------------------------------------------------------------------
# 8. enrich_candidates에서 native SST-14(cyclic)는 페널티 없음
# ---------------------------------------------------------------------------

def test_enrich_native_sst14_no_penalty():
    """enrich_candidates로 native SST-14 처리 시 hemolysis 페널티 없음 (cyclic → applies=False)."""
    from pyrosetta_flow.multiobjective import enrich_candidates
    cands = [{"sequence": NATIVE_SST14, "ddg": -19.8, "clash_score": 5.0}]
    enrich_candidates(cands)
    c = cands[0]
    es = c.get("extra_scores", {})
    # cyclic이므로 applies=False, risk=NA
    assert es.get("hemolysis_applies") is False
    assert es.get("hemolysis_risk") == "NA"
    # admet_score는 페널티 없이 원본 그대로여야 함 (직접 비교: 동일 호출 결과와 일치)
    from pyrosetta_flow.multiobjective import cheap_objectives
    orig = cheap_objectives(NATIVE_SST14)
    # 페널티 없으므로 enrich 후 admet_score == 원본
    assert abs(c.get("admet_score", 0) - orig["admet_score"]) < 1e-6, \
        "native SST-14는 hemolysis 페널티 없어야 함"


# ---------------------------------------------------------------------------
# 9. apply_hemolysis_penalty: extra_scores 키 기록 검증
# ---------------------------------------------------------------------------

def test_apply_hemolysis_penalty_records_keys():
    """apply_hemolysis_penalty가 hemolysis_risk, hemolysis_applies, hemolysis_reason을 기록."""
    hemo = {
        "applies": True,
        "hemolysis_risk": "HIGH",
        "aliphatic_index": 180.0,
        "reason": "AI=180.0 > 120 → 고용혈 위험",
    }
    extra: dict = {"admet_score": 0.50}
    apply_hemolysis_penalty(extra, hemo)
    assert "hemolysis_risk" in extra
    assert "hemolysis_aliphatic_index" in extra
    assert "hemolysis_applies" in extra
    assert "hemolysis_reason" in extra
    assert extra["hemolysis_aliphatic_index"] == 180.0


# ---------------------------------------------------------------------------
# 10. 빈 서열 안전 처리
# ---------------------------------------------------------------------------

def test_empty_sequence_safe():
    """빈 서열도 applies=False로 안전하게 처리되고 crash 없음."""
    result = hemolysis_aliphatic_flag("")
    assert result["applies"] is False
    assert result["hemolysis_risk"] == "NA"


def test_none_ai_provided():
    """aliphatic_index=None으로 호출해도 정상 처리 (내부 계산 or graceful)."""
    result = hemolysis_aliphatic_flag("GKRDESNPQTH", aliphatic_index=None)
    # applies 여부와 무관하게 crash 없이 반환
    assert isinstance(result, dict)
    assert "hemolysis_risk" in result
    assert "applies" in result


def test_precomputed_ai_used():
    """aliphatic_index 미리 제공 시 해당 값이 결과에 사용됨."""
    # 인위적으로 AI=200 제공 → HIGH (직쇄 표준 서열)
    result = hemolysis_aliphatic_flag("GKRDESNPQTH", aliphatic_index=200.0)
    assert result["aliphatic_index"] == 200.0
    # GKRDESNPQTH는 직쇄 L-aa → applies=True (비표준 없음, cyclic 아님)
    assert result["applies"] is True
    assert result["hemolysis_risk"] == "HIGH"
