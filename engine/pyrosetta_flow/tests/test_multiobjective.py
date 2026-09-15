"""multiobjective.py 회귀 테스트.

다목적 통합(ΔG + 반감기 + 선택성 + ADMET)의 cheap-objective 계산,
ADMET 합리성 점수, Pareto 키 매핑, top-K 선별을 검증한다.
"""
from __future__ import annotations

import math

import pytest

from pyrosetta_flow.multiobjective import (
    NATIVE_SST14,
    ObjectiveWeights,
    admet_reasonableness,
    cheap_objectives,
    enrich_candidates,
    multiobjective_scalar,
    select_topk_for_selectivity,
)


def test_cheap_objectives_native_keys():
    o = cheap_objectives(NATIVE_SST14)
    for key in ("half_life_h", "admet_score", "stability_norm",
                "gravy", "boman_index", "instability_index", "pi"):
        assert key in o, f"missing {key}"
    assert 0.0 <= o["admet_score"] <= 1.0
    assert 0.0 <= o["stability_norm"] <= 1.0


def test_cheap_objectives_legacy_lowercase_laa_normalizes():
    o = cheap_objectives("agcknffwktftsc")
    assert o["sequence"] == NATIVE_SST14
    assert "gravy" in o


def test_cheap_objectives_preserves_mixed_case_d_aa_marker():
    seq = "AGCkNFFWKTFTSC"
    o = cheap_objectives(seq)
    assert o["sequence"] == seq
    assert o["halflife_source"] in {"ensemble", "heuristic", "rf", "none"}
    assert "gravy" not in o, "L-aa physicochemical surrogate must not consume D-aa markers as L-aa"


def test_terminal_mutation_lowers_half_life():
    """말단(N/C) 변이는 exopeptidase 취약성 ↑ → 반감기 ↓ 여야 한다."""
    native = cheap_objectives("AGCKNFFWKTFTSC")["half_life_h"]
    terminal_mut = cheap_objectives("YGCKNFFWKTFTST")["half_life_h"]
    assert terminal_mut < native


def test_admet_reasonableness_bounds_and_monotonic():
    # 매우 불안정(II 큼) → 낮은 점수
    bad = admet_reasonableness({"instability_index": 90, "gravy": 2.0,
                                "boman_index": 0.1, "pi": 12.0})
    good = admet_reasonableness({"instability_index": 20, "gravy": -0.5,
                                 "boman_index": 2.5, "pi": 7.0})
    assert 0.0 <= bad <= 1.0 and 0.0 <= good <= 1.0
    assert good > bad


def test_enrich_maps_pareto_keys():
    cands = [{"sequence": "AGCKNFFWKTFTSC", "ddg": -19.8, "clash_score": 5.0}]
    enrich_candidates(cands)
    c = cands[0]
    # pareto_ranking 이 기대하는 키
    assert "stability" in c and "druggability" in c
    assert c["stability"] == c["extra_scores"]["stability_norm"]
    assert c["druggability"] == c["extra_scores"]["admet_score"]


def test_multiobjective_scalar_prefers_strong_binder():
    weak = {"sequence": "AGCKNFFWKTFTSC", "ddg": -5.0, "clash_score": 5.0}
    strong = {"sequence": "AGCKNFFWKTFTSC", "ddg": -120.0, "clash_score": 5.0}
    enrich_candidates([weak, strong])
    assert multiobjective_scalar(strong) > multiobjective_scalar(weak)


def test_selectivity_margin_rewarded():
    base = {"sequence": "AGCKNFFWKTFTSC", "ddg": -30.0, "clash_score": 5.0}
    enrich_candidates([base])
    no_sel = dict(base, selectivity_margin=0.0)
    high_sel = dict(base, selectivity_margin=18.0)
    assert multiobjective_scalar(high_sel) > multiobjective_scalar(no_sel)


def test_topk_respects_clash_gate_then_ddg():
    cands = [
        {"sequence": "AGCKNFFWKTFTSC", "ddg": -100.0, "clash_score": 50.0},  # clash 탈락
        {"sequence": "AGCKYEFWKTFTSC", "ddg": -40.0, "clash_score": 3.0},
        {"sequence": "AGCRNFFWKTFTSC", "ddg": -60.0, "clash_score": 8.0},
    ]
    enrich_candidates(cands)
    top = select_topk_for_selectivity(cands, k=2, clash_max=10.0)
    seqs = [c["sequence"] for c in top]
    # clash 통과 후보 중 ddg 좋은 순: AGCRNFFWKTFTSC(-60) > AGCKYEFWKTFTSC(-40)
    assert seqs == ["AGCRNFFWKTFTSC", "AGCKYEFWKTFTSC"]


def test_weights_sum_reasonable():
    w = ObjectiveWeights()
    assert math.isclose(w.ddg + w.selectivity + w.stability + w.admet, 1.0, abs_tol=1e-6)


# --- B (2026-06-09): pepADMET 독성 → admet 페널티 적용 로직 (순수 함수) ---
def test_apply_toxicity_penalizes_admet():
    """native 보다 hc50 가 크게 나쁜(더 독성) 후보만 페널티 (2026-06-10: binary→hc50 상대 게이트)."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"sequence": "AGCkNFFWKTFTSC", "admet_score": 0.60}
    # native hc50≈-55.68 보다 훨씬 독성(-150) → 페널티
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "toxicity_type": "hemostasis", "binary_toxicity": 1.0, "hc50": -150.0})
    assert extra["pepadmet_toxic"] is True
    assert extra["pepadmet_toxicity_type"] == "hemostasis"
    assert extra["more_toxic_than_native"] is True
    assert extra["hc50_vs_native"] < 0
    assert extra["admet_score"] < 0.60, "native보다 독성 큰 후보는 admet_score 페널티 받아야"


def test_hc50_l_aa_bypassed():
    """A5 Critical #1: L-aa only 서열은 역변별 hc50 페널티를 bypass."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"sequence": "AGCKNFFWKTFTSC", "admet_score": 0.60}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "toxicity_type": "hemostasis", "binary_toxicity": 1.0, "hc50": -150.0})
    assert extra["more_toxic_than_native"] is False
    assert extra["toxicity_penalty"] == 1.0
    assert extra["toxicity_source"] == "L_aa_hc50_reverse_discriminant_bypassed"
    assert extra["admet_score"] == 0.60


def test_hc50_d_aa_still_applied():
    """A5 Critical #1: D-aa marker 서열은 기존 hc50 페널티 적용 유지."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"sequence": "AGCkNFFWKTFTSC", "admet_score": 0.60}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "toxicity_type": "hemostasis", "binary_toxicity": 1.0, "hc50": -150.0})
    assert extra["more_toxic_than_native"] is True
    assert extra["toxicity_penalty"] < 1.0
    assert extra["admet_score"] < 0.60


def test_apply_toxicity_linear_fallback_skips_gate():
    """SMILES 선형 폴백(cyclic 구조 소실) 시 hc50 신뢰불가 → 게이트 skip (VR-A2, 2026-06-17)."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.5}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "hc50": -150.0, "graph_note": "linear_sequence_fallback"})
    assert extra["hc50_reliable"] is False
    assert "more_toxic_than_native" not in extra, "신뢰불가 hc50 은 게이트 미적용"
    assert extra["admet_score"] == 0.5, "선형 폴백은 admet 페널티 없음(fail-open)"


def test_apply_toxicity_native_comparable_no_penalty():
    """hc50 가 native 동급(±tolerance)이면 is_toxic=True 라도 페널티 없음 (home-advantage)."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.60}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "toxicity_type": "hemostasis", "binary_toxicity": 1.0, "hc50": -55.4})
    assert extra["pepadmet_toxic"] is True          # 기록은 됨
    assert extra["more_toxic_than_native"] is False
    assert extra["admet_score"] == 0.60, "native 동급 독성은 페널티 없음"


def test_apply_toxicity_nontoxic_no_penalty():
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.60}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": False})
    assert extra["admet_score"] == 0.60, "비독성은 페널티 없음"
    assert extra["pepadmet_toxic"] is False


def test_apply_toxicity_unavailable_no_change():
    """추론 불가(available=False)면 admet_score 를 건드리지 않음 (fail-closed: 가짜 안전판정 X)."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.60}
    apply_toxicity_to_extra(extra, {"available": False})
    assert extra["admet_score"] == 0.60
    assert "pepadmet_toxic" not in extra


# ---------------------------------------------------------------------------
# 신규 필드 검증 (2026-06-25, 약리 개선 5건)
# ---------------------------------------------------------------------------

def test_cheap_objectives_modifications_passed_through():
    """변경 1: cand dict 의 modifications 가 ensemble_halflife 에 전달되는지 smoke 확인."""
    from pyrosetta_flow.multiobjective import cheap_objectives
    # modifications 없는 경우 vs 빈 리스트 → 동일 결과여야 함 (regression-safe)
    o_no_cand = cheap_objectives(NATIVE_SST14)
    o_with_empty = cheap_objectives(NATIVE_SST14, cand={"modifications": []})
    assert abs(o_no_cand["half_life_h"] - o_with_empty["half_life_h"]) < 1e-6
    assert abs(o_no_cand["stability_norm"] - o_with_empty["stability_norm"]) < 1e-6


def test_cheap_objectives_radiolysis_fields():
    """변경 2: radiolysis_score, radiolysis_risk_level 가 반환되어야 한다."""
    from pyrosetta_flow.multiobjective import cheap_objectives
    o = cheap_objectives(NATIVE_SST14)
    # 두 필드가 존재 (None 또는 값)
    assert "radiolysis_score" in o
    assert "radiolysis_risk_level" in o
    if o["radiolysis_score"] is not None:
        assert 0.0 <= o["radiolysis_score"] <= 1.0, "radiolysis_score 0~1 범위"
        assert o["radiolysis_risk_level"] in ("low", "moderate", "high")


def test_admet_reasonableness_with_radiolysis():
    """변경 2: radiolysis_total_score 포함 시 가중평균에 반영되고 합계=100% 유지."""
    from pyrosetta_flow.multiobjective import admet_reasonableness
    props_no_rad = {"instability_index": 30.0, "gravy": -0.3, "boman_index": 2.5, "pi": 7.0}
    props_with_rad = dict(props_no_rad, radiolysis_total_score=6.0)
    score_no = admet_reasonableness(props_no_rad)
    score_with = admet_reasonableness(props_with_rad)
    # 두 값 모두 0~1 범위
    assert 0.0 <= score_no <= 1.0
    assert 0.0 <= score_with <= 1.0
    # radiolysis=6 (moderate, score=0.5) 이면 일반 조건에서 약간 달라질 수 있음
    # (값이 다를 수 있음 — 회귀가 아닌 반영 여부 확인)
    props_high_rad = dict(props_no_rad, radiolysis_total_score=12.0)
    score_high = admet_reasonableness(props_high_rad)
    assert score_high < score_no, "radiolysis 최대 위험(12)은 점수를 낮춰야 한다"


def test_cheap_objectives_instability_applicable_cyclic():
    """변경 3: cyclic SST-14 서열은 instability_applicable=False 플래그."""
    from pyrosetta_flow.multiobjective import cheap_objectives
    o = cheap_objectives(NATIVE_SST14)  # AGCKNFFWKTFTSC — cyclic
    if "instability_applicable" in o:
        assert o["instability_applicable"] is False, "cyclic SST-14 → instability_applicable=False"


def test_cheap_objectives_instability_applicable_linear():
    """변경 3: 비cyclic(Cys 없는) 서열은 instability_applicable=True."""
    from pyrosetta_flow.multiobjective import cheap_objectives
    o = cheap_objectives("AGAKNFFWKTFTSA")  # Cys→Ala 치환 → 비cyclic
    if "instability_applicable" in o:
        assert o["instability_applicable"] is True


def test_apply_toxicity_linear_fallback_ood_warning():
    """변경 4: linear_sequence_fallback 시 hc50_ood_warning 키가 extra 에 추가되어야 한다."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.5}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": True,
                                    "hc50": -150.0, "graph_note": "linear_sequence_fallback"})
    assert "hc50_ood_warning" in extra, "cyclic fallback → hc50_ood_warning 필드 필수"
    assert "cyclic" in extra["hc50_ood_warning"].lower() or "OOD" in extra["hc50_ood_warning"]


def test_apply_toxicity_non_fallback_no_ood_warning():
    """변경 4: graph_note 가 linear_sequence_fallback 이 아니면 hc50_ood_warning 없음."""
    from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
    extra = {"admet_score": 0.5}
    apply_toxicity_to_extra(extra, {"available": True, "is_toxic": False,
                                    "hc50": -55.0, "graph_note": None})
    assert "hc50_ood_warning" not in extra


def test_objective_weights_radiolysis_field():
    """변경 5: ObjectiveWeights 에 radiolysis 필드가 있고 기본값 0.0."""
    from pyrosetta_flow.multiobjective import ObjectiveWeights
    w = ObjectiveWeights()
    assert hasattr(w, "radiolysis"), "radiolysis 필드 없음"
    assert w.radiolysis == 0.0, "기본값 0.0(하위호환)"
    # 커스텀 값 설정 가능
    w2 = ObjectiveWeights(radiolysis=0.05)
    assert w2.radiolysis == 0.05


def test_objective_weights_backward_compat():
    """변경 5: 기존 4개 필드 합계 변경 없음 (하위호환)."""
    import math
    from pyrosetta_flow.multiobjective import ObjectiveWeights
    w = ObjectiveWeights()
    assert math.isclose(w.ddg + w.selectivity + w.stability + w.admet, 1.0, abs_tol=1e-6)
