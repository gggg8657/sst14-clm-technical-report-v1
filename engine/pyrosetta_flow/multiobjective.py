"""multiobjective.py
=================
SSTR2 방사성의약품 스크리닝 다목적 통합 (ΔG + 반감기 + 선택성 + ADMET).

목표(goal): SST-14(AGCKNFFWKTFTSC) 변이체 중
  - ddG ↓               : SSTR2에 강하게 결합
  - half_life_h ↑        : 혈중 반감기가 길다
  - selectivity_margin ↑ : SSTR1/3/4/5(off-target) 대비 SSTR2 선택적
  - admet_score ↑        : ADMET 프로파일이 합리적 (용해도/안정성/결합경향)

비용 계층화 (cost-tiered) — 실제 도킹은 비싸므로:
  Layer 0 (모든 후보, 서열만, μs):  half_life + ADMET surrogate
  Layer 1 (top-K, 실제 PyRosetta):  selectivity off-target docking

honest disclaimer (VR-cycle-09 / H-06):
  half_life_h, admet_score 는 **랭킹용 surrogate** 다. 임상 반감기·임상 ADMET 수치가
  아니며, in-vitro 혈청 안정성/투과도 assay 로 검증되지 않았다. ddG·selectivity_margin
  은 실제 PyRosetta FlexPepDock 결과(REU/kcal·mol)지만 절대 친화도(Ki/Kd)가 아니다.

이 모듈은 pyrosetta_flow 후보 dict(키: sequence, ddg, total_score, clash_score, …)를
입력받아 extra_scores 를 채우고, pareto_ranking 이 기대하는 키(stability, druggability)로
매핑한다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NATIVE_SST14 = "AGCKNFFWKTFTSC"

# 표준 20종 L-아미노산 one-letter 코드 집합
_STANDARD_L_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _coerce_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def _has_d_aa_marker(sequence: str) -> bool:
    """Return True when lowercase standard AA letters mark D-amino acids."""
    seq = sequence or ""
    has_lower_aa = any(ch.islower() and ch.upper() in _STANDARD_L_AA for ch in seq)
    has_upper_aa = any(ch in _STANDARD_L_AA for ch in seq)
    return has_lower_aa and has_upper_aa


def _normalize_sequence_preserving_d_aa(sequence: str) -> str:
    """Uppercase legacy L-only inputs, but keep lowercase D-aa markers intact."""
    seq = (sequence or "").strip()
    return seq if _has_d_aa_marker(seq) else seq.upper()


def _l_only_sequence(sequence: str) -> Optional[str]:
    """Return uppercase sequence only when no lowercase D-aa marker is present."""
    seq = (sequence or "").strip()
    if _has_d_aa_marker(seq):
        return None
    return seq.upper()


def robust_ddg_value(cand: Dict[str, Any], fallback: float = 999.0) -> float:
    """Return robust ddG for ranking, preferring measured median over single-run ddG.

    P1 may provide ``ddg_median`` at the top level or in ``extra_scores``. When it is
    absent, fall back to the legacy single ``ddg``/``ddG`` so older artifacts still
    rank deterministically.
    """
    extra = cand.get("extra_scores") if isinstance(cand.get("extra_scores"), dict) else {}
    for key in ("ddg_median", "ddG_median"):
        val = _coerce_float(cand.get(key))
        if val is not None:
            return val
        val = _coerce_float(extra.get(key))
        if val is not None:
            return val
    val = _coerce_float(cand.get("ddg", cand.get("ddG")))
    return val if val is not None else fallback

# ---------------------------------------------------------------------------
# 의존 모듈 (지연 import — 일부 환경에서 PyRosetta/AG_src 부재 가능)
# ---------------------------------------------------------------------------

try:  # 반감기 surrogate (서열 기반)
    from pyrosetta_flow.surrogate_v2.step08_stability_v2 import predict_half_life as _predict_half_life
    _HAS_HALFLIFE = True
except Exception:  # pragma: no cover
    _predict_half_life = None  # type: ignore[assignment]
    _HAS_HALFLIFE = False

try:  # ADMET-reasonableness surrogate (문헌 기반 물성)
    from AG_src.pipeline.pharma_properties import PharmaProperties as _PharmaProperties
    _HAS_PHARMA = True
except Exception:  # pragma: no cover
    _PharmaProperties = None  # type: ignore[assignment]
    _HAS_PHARMA = False


# ---------------------------------------------------------------------------
# ADMET-reasonableness scoring
# ---------------------------------------------------------------------------

# "합리적(reasonable)" 범위 — 문헌 휴리스틱. 점수는 0(나쁨)~1(좋음).
#   Instability Index < 40  → 안정 (Guruprasad 1990)
#   GRAVY < 0               → 수용성(친수성), 펩타이드 약물에 유리
#   Boman 2.48 부근~높음    → 단백질 결합 경향(수용체 결합에 유리하나 과도하면 비특이)
#   pI 중성 근처(6~8)        → 제형/용해 안정


def admet_reasonableness(props: Dict[str, Any]) -> float:
    """4개 물성(+radiolysis 선택적)에서 0~1 합리성 점수를 합성한다 (가중 평균).

    각 항목을 0~1 부분점수로 변환 후 평균. surrogate 임을 잊지 말 것.

    HEURISTIC proxy: radiolysis_total_score 가 props 에 있으면 W/M 방사손상 가중평균에
    선택적으로 반영한다(가중 10%). 절대값 아님 — 임상 방사안정성 지표가 아닌 잔기 취약성
    추정치다.
    """
    ii = props.get("instability_index", 50.0)
    gravy = props.get("gravy", 0.0)
    boman = props.get("boman_index", 0.0)
    pi = props.get("pi", 7.0)

    # Instability: <40 만점, 40~80 선형 감점, >80 0점
    s_ii = 1.0 if ii < 40 else max(0.0, 1.0 - (ii - 40.0) / 40.0)
    # GRAVY: <=-1 만점(매우 친수), 0 에서 0.5, >=1 0점 (소수성 과다 → 응집/저용해)
    s_gravy = min(1.0, max(0.0, (1.0 - gravy) / 2.0))
    # Boman: 2.0~3.5 이상이 단백질 결합 펩타이드에 적정. 1.0 미만이면 결합경향 약함.
    if boman < 1.0:
        s_boman = max(0.0, boman / 1.0) * 0.5
    elif boman <= 4.0:
        s_boman = 0.5 + 0.5 * (boman - 1.0) / 3.0
    else:
        s_boman = 1.0
    # pI: 6~8 만점, 멀어질수록 감점
    s_pi = max(0.0, 1.0 - abs(pi - 7.0) / 5.0)

    radiolysis_total = props.get("radiolysis_total_score")
    if radiolysis_total is not None:
        # radiolysis 포함 시: 기존 4항목을 90%로 축소, radiolysis 10% 추가 (합계=100% 유지)
        s_rad = max(0.0, 1.0 - float(radiolysis_total) / 12.0)
        score = (0.35 * s_ii + 0.30 * s_gravy + 0.15 * s_boman + 0.20 * s_pi) * 0.90
        score += s_rad * 0.10
    else:
        # 기존 가중평균 (radiolysis 없음)
        score = 0.35 * s_ii + 0.30 * s_gravy + 0.15 * s_boman + 0.20 * s_pi
    return round(score, 4)


# ---------------------------------------------------------------------------
# Layer 0 — cheap objectives (모든 후보)
# ---------------------------------------------------------------------------

# 반감기 정규화: native(~16h) 대비. 0~1 로 사용(stability objective).
_HALFLIFE_REF_H = 16.0


def cheap_objectives(
    sequence: str,
    reference_seq: str = NATIVE_SST14,
    cand: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """서열만으로 계산 가능한 저비용 목적값 + ADMET surrogate.

    Args:
        sequence: 펩타이드 서열. L-only 입력은 대소문자 무관, 소문자 표준 AA는 D-aa marker.
        reference_seq: ADMET 기준 서열 (기본 native SST-14).
        cand: 후보 dict (있으면 modifications 필드를 ensemble_halflife 에 전달).

    Returns dict keys:
        half_life_h, gravy, boman_index, instability_index, instability_applicable,
        aliphatic_index, pi, admet_score, stability_norm, radiolysis_score,
        radiolysis_risk_level  (모두 surrogate / ranking 용)
    """
    seq = _normalize_sequence_preserving_d_aa(sequence)
    l_seq = _l_only_sequence(sequence)
    out: Dict[str, Any] = {"sequence": seq}

    # candidate dict 에서 실제 modification 정보 추출 (변경 1: 빈 리스트 고정 해제)
    mods: List[Any] = list((cand or {}).get("modifications") or [])

    # 반감기 — 앙상블 (휴리스틱 A + RF C, log10 정규화 평균). RF 미가용 시 휴리스틱 단독 fallback.
    # 2026-06-09: 단일 추정기 대신 두 상보적 모델 결합 (A=SST-14 절대 스케일, C=PEPlife2 데이터 일반화).
    if seq:
        try:
            from .halflife_ensemble_v2 import ensemble_halflife
            ens = ensemble_halflife(seq, modifications=mods)
            out["half_life_h"] = ens["half_life_h"]
            out["half_life_heuristic_h"] = ens["half_life_heuristic_h"]
            out["half_life_rf_h"] = ens["half_life_rf_h"]
            out["halflife_source"] = ens["halflife_source"]
            out["stability_norm"] = ens["stability_norm"]
        except Exception as exc:  # pragma: no cover — 앙상블 실패 시 레거시 휴리스틱
            logger.warning("halflife ensemble 실패(%s) → 휴리스틱 단독", exc)
            hl = float(_predict_half_life(seq, mods)) if _HAS_HALFLIFE else float("nan")
            out["half_life_h"] = hl
            import math as _m
            out["stability_norm"] = (round(min(1.0, max(0.0, (_m.log10(hl) - _m.log10(0.02)) /
                                     (_m.log10(200.0) - _m.log10(0.02)))), 4)
                                     if hl == hl and hl > 0 else 0.0)
    else:
        out["half_life_h"] = float("nan")
        out["stability_norm"] = 0.0

    # ADMET-reasonableness (물성 surrogate)
    rads: Optional[Dict[str, Any]] = None
    if _HAS_PHARMA and l_seq:
        try:
            pp = _PharmaProperties(reference_seq=reference_seq)
            # Cys3-Cys14 SS bond → pI 계산 시 해당 Cys 제외 (0-indexed)
            cys = {i for i, a in enumerate(l_seq) if a == "C"}
            ss = {min(cys), max(cys)} if len(cys) >= 2 else None
            props: Dict[str, Any] = {
                "gravy": round(pp.calculate_gravy(l_seq), 4),
                "boman_index": round(pp.calculate_boman_index(l_seq), 4),
                "instability_index": round(pp.calculate_instability_index(l_seq), 4),
                "aliphatic_index": round(pp.calculate_aliphatic_index(l_seq), 4),
                "pi": pp.calculate_pi(l_seq, ss_bond_cysteines=ss),
            }
            # 변경 3: instability_index cyclic 가드 (Guruprasad 1990 globular 훈련 — cyclic 미검증)
            if _is_cyclic_sst14(l_seq):
                props["instability_applicable"] = False
                logger.warning(
                    "instability_index는 cyclic SST-14에 미검증(Guruprasad 1990 globular 훈련). seq=%s",
                    l_seq[:8],
                )
            else:
                props["instability_applicable"] = True

            # 변경 2: radiolysis susceptibility (방사성의약품 방사손상 HEURISTIC proxy)
            try:
                rads = pp.calculate_radiolysis_susceptibility(l_seq)
                props["radiolysis_total_score"] = rads["total_score"]
            except Exception as _rads_exc:  # pragma: no cover
                logger.debug("radiolysis 계산 skip(%s)", _rads_exc)
                rads = None
        except Exception as exc:  # pragma: no cover
            logger.warning("PharmaProperties 실패(%s)", exc)
            props = {}
    else:
        props = {}
    out.update(props)
    out["admet_score"] = admet_reasonableness(props) if props else 0.0

    # 변경 2: radiolysis_score + radiolysis_risk_level (cheap_objectives 반환값에 포함)
    if rads is not None:
        rs_raw = rads.get("total_score", 0.0)
        # 0~12 범위 → 0~1 정규화 반전 (score=0 → 1.0, score>=12 → 0.0)
        out["radiolysis_score"] = round(max(0.0, min(1.0, 1.0 - rs_raw / 12.0)), 4)
        out["radiolysis_risk_level"] = rads.get("risk_level")
    else:
        out["radiolysis_score"] = None
        out["radiolysis_risk_level"] = None
    return out


# ---------------------------------------------------------------------------
# pepADMET 실제 독성 ML 추론 (B, 2026-06-09) — physicochemical surrogate 보강
# ---------------------------------------------------------------------------
# pepADMET GNN(toxicity_early_stop.pth)을 pepadmet conda env subprocess 로 배치 추론.
# 독성 후보는 admet_score 에 페널티 → 안전성을 다목적 랭킹에 반영.
_TOXIC_ADMET_PENALTY = 0.4   # native보다 심하게 독성↑ 후보의 admet_score 곱셈 페널티 하한
# 2026-06-10: pepADMET binary(is_toxic)는 비변별적 — oxytocin·native SST-14 까지 전부 toxic 판정.
# 따라서 게이트는 hc50 **연속값을 native 대비 상대(home-advantage)** 로 평가한다 (Δmargin 과 대칭).
_HC50_NATIVE_TOLERANCE = 5.0   # native hc50 ±이 밴드는 "동급"으로 간주 (페널티 없음)
_HC50_PENALTY_SCALE = 200.0    # native 초과 독성분(hc50)당 선형 감점 스케일
_NATIVE_HC50: Optional[float] = None  # native SST-14 hc50 기준선 (지연 로드)


def _native_hc50_baseline() -> Optional[float]:
    """native SST-14 의 pepADMET hc50 기준선 (home-advantage). 1회 로드 후 캐시."""
    global _NATIVE_HC50
    if _NATIVE_HC50 is not None:
        return _NATIVE_HC50
    import json as _json
    from pathlib import Path as _P
    # multiobjective.py -> pyrosetta_flow -> repo_root
    path = _P(__file__).resolve().parents[1] / "data/somatostatin_receptor/curated/native_toxicity_baseline.json"
    try:
        if path.exists():
            _NATIVE_HC50 = float(_json.loads(path.read_text()).get("hc50"))
    except Exception as exc:  # pragma: no cover
        logger.warning("native hc50 baseline 로드 실패(%s)", exc)
    return _NATIVE_HC50


def predict_toxicity_for_sequences(sequences: List[str]) -> Dict[str, Dict[str, Any]]:
    """pepADMET 독성 배치 추론. {sequence: result}. 미설치/실패 시 빈 dict (graceful)."""
    seqs = [s for s in dict.fromkeys(sequences) if s]  # 중복 제거, 순서 유지
    if not seqs:
        return {}
    try:
        from .pepadmet_runner import predict_toxicity_batch
    except Exception as exc:  # pragma: no cover
        logger.warning("pepadmet_runner import 실패(%s) — 독성 skip", exc)
        return {}
    try:
        results = predict_toxicity_batch(seqs)
    except Exception as exc:
        logger.warning("pepADMET 독성 추론 실패(%s) — skip", exc)
        return {}
    return {r.get("sequence"): r for r in results if isinstance(r, dict) and r.get("sequence")}


def apply_toxicity_to_extra(extra: Dict[str, Any], tox: Dict[str, Any]) -> None:
    """pepADMET 독성 결과를 extra_scores 에 기록하고 admet_score 에 페널티 반영 (in-place).

    available=False(추론 불가)면 admet_score 를 건드리지 않는다 (fail-closed: 가짜 안전판정 X).

    2026-06-10 수정: binary is_toxic 는 비변별적(전부 True)이라 게이트에 **쓰지 않는다**. 대신
    hc50 을 native SST-14 기준선 대비(home-advantage)로 평가한다 — Δmargin 과 동일 철학.
      - hc50_vs_native = hc50 − native_hc50  (>0 = native보다 안전, <0 = 더 독성)
      - native ±_HC50_NATIVE_TOLERANCE 밴드 = "동급" → 페널티 없음
      - 그보다 독성↑ 일 때만 초과분에 선형 비례 페널티(하한 _TOXIC_ADMET_PENALTY)
    """
    if not tox or not tox.get("available"):
        return
    extra["pepadmet_toxic"] = bool(tox.get("is_toxic"))          # 기록만 (비변별적, 게이트 미사용)
    extra["pepadmet_toxicity_type"] = tox.get("toxicity_type")
    extra["pepadmet_binary_toxicity"] = tox.get("binary_toxicity")
    hc50 = tox.get("hc50")
    extra["pepadmet_hc50"] = hc50
    seq = str(extra.get("sequence") or tox.get("sequence") or "")

    # 2026-06-17 (VR-A2): SMILES 파싱 실패로 선형 폴백된 경우 cyclic(SS bond) 구조가 소실되어
    # hc50 이 신뢰 불가 → hc50 게이트를 건너뛴다(fail-open: 가짜 독성판정 방지). 플래그만 기록.
    reliable = tox.get("graph_note") != "linear_sequence_fallback"
    extra["hc50_reliable"] = bool(reliable)
    # 변경 4: cyclic SMILES fallback → OOD 마킹 (신뢰불가 명시)
    if not reliable:
        extra["hc50_ood_warning"] = "cyclic SMILES fallback: hc50 신뢰불가(OOD 마킹)"
    native_hc50 = _native_hc50_baseline()
    if not reliable or not isinstance(hc50, (int, float)) or native_hc50 is None:
        return
    delta = hc50 - native_hc50                                   # >0 안전, <0 독성↑
    extra["hc50_vs_native"] = round(delta, 3)

    # A5 Critical #1 (2026-08-03): pepADMET hc50 는 L-aa에서 역변별(AUC=0.146)이 확인되어
    # L-only 서열에는 production ADMET 페널티/게이트로 쓰지 않는다. D-aa marker가 있을 때만 유지.
    if not _has_d_aa_marker(seq):
        extra["more_toxic_than_native"] = False
        extra["toxicity_penalty"] = 1.0
        extra["toxicity_source"] = "L_aa_hc50_reverse_discriminant_bypassed"
        return

    more_toxic = delta < -_HC50_NATIVE_TOLERANCE
    extra["more_toxic_than_native"] = bool(more_toxic)
    if more_toxic:
        excess = -(delta + _HC50_NATIVE_TOLERANCE)               # native 초과 독성분 (>0)
        factor = max(_TOXIC_ADMET_PENALTY, 1.0 - excess / _HC50_PENALTY_SCALE)
        extra["toxicity_penalty"] = round(factor, 4)
        base = extra.get("admet_score")
        if isinstance(base, (int, float)):
            extra["admet_score"] = round(base * factor, 4)
    else:
        extra["toxicity_penalty"] = 1.0


def enrich_candidates(
    candidates: List[Dict[str, Any]],
    reference_seq: str = NATIVE_SST14,
) -> List[Dict[str, Any]]:
    """각 후보 dict 에 cheap_objectives + hemolysis_aliphatic_flag 결과를 병합한다 (in-place + 반환).

    후보는 'sequence' 키를 가져야 한다. 결과는 후보 dict 최상위와
    extra_scores(존재 시)에 모두 기록한다.

    Silo B hemolysis 통합:
      - cheap_objectives에서 aliphatic_index가 이미 계산되므로 재사용.
      - hemolysis_aliphatic_flag 로 risk 판정 후 extra_scores에 기록.
      - applies=True + HIGH면 admet_score에 soft penalty(×0.90) 적용 (하드 reject 아님).
      - native SST-14(cyclic Cys3-14)는 applies=False → 영향 없음.
    """
    for c in candidates:
        seq = c.get("sequence") or c.get("seq") or ""
        obj = cheap_objectives(seq, reference_seq=reference_seq)
        for k, v in obj.items():
            if k == "sequence":
                continue
            c[k] = v
        es = c.setdefault("extra_scores", {})
        if isinstance(es, dict):
            es.update({k: v for k, v in obj.items() if k != "sequence"})

        # Lane C: hemolysis soft flag (aliphatic_index 재사용)
        ai_val: Optional[float] = obj.get("aliphatic_index")
        hemo = hemolysis_aliphatic_flag(seq, aliphatic_index=ai_val)
        if isinstance(es, dict):
            apply_hemolysis_penalty(es, hemo)
            # admet_score가 페널티로 갱신된 경우 최상위 키에도 반영
            if hemo.get("applies") and hemo.get("hemolysis_risk") == "HIGH":
                penalized = es.get("admet_score")
                if penalized is not None:
                    c["admet_score"] = penalized

        # pareto_ranking 키 매핑: stability(반감기), druggability(ADMET)
        # druggability는 페널티 반영 후 admet_score 사용
        c.setdefault("stability", obj["stability_norm"])
        c["druggability"] = c.get("admet_score", obj["admet_score"])
    return candidates


# ---------------------------------------------------------------------------
# 다목적 스칼라 점수 (UI 표시 / 단일 랭킹용 보조)
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveWeights:
    """다목적 스칼라 점수 가중치.

    가중치 임의설정·검증없음. radiolysis / mmgbsa 기본 0.0(하위호환).
    mmgbsa 는 있으면 정합성 지표로 가산(회의록 4월 A-04 통합 요구).
    합계가 1.0이 아닐 수 있으므로 정규화 없이 가중합만 사용.
    """

    ddg: float = 0.40           # 결합 (최우선, FlexPepDock ddG)
    selectivity: float = 0.25   # SSTR2 선택성
    stability: float = 0.20     # 반감기
    admet: float = 0.15         # ADMET 합리성
    radiolysis: float = 0.0     # 방사손상 취약성 (기본 0.0, 하위호환; 활성화 시 가중 조정 필요)
    mmgbsa: float = 0.0         # MM-GBSA dg_bind 정합성 (기본 0.0=미보유 시 무영향).
                                # ObjectiveWeightsWithMMGBSA 프리셋으로 활성화.


# 회의록 4월 A-04 대응 프리셋 (dg + sel + stab + admet + mmgbsa 5축 통합)
# 원 가중치를 유지하며 mmgbsa 가산 = 하위호환 (MM-GBSA 없으면 원 점수와 동일)
OBJECTIVE_WEIGHTS_WITH_MMGBSA = ObjectiveWeights(
    ddg=0.35, selectivity=0.25, stability=0.15, admet=0.10,
    radiolysis=0.0, mmgbsa=0.15,
)


def multiobjective_scalar(
    cand: Dict[str, Any],
    weights: ObjectiveWeights = ObjectiveWeights(),
    ddg_ref: float = 0.0,
    ddg_scale: float = 50.0,
    mmgbsa_ref: float = 0.0,
    mmgbsa_scale: float = 100.0,
) -> float:
    """후보의 다목적 스칼라 점수(높을수록 좋음). UI 정렬 보조용.

    ddg 는 음수가 좋으므로 (ddg_ref - ddg)/ddg_scale 로 0~1 근사.
    selectivity_margin 은 양수가 좋음 → 0~1 포화.
    stability/admet 은 이미 0~1.
    mmgbsa 도 음수가 좋음 (dg_bind). cand['mmgbsa_dg_bind'] 있을 때만 반영.
      weights.mmgbsa > 0 이지만 mmgbsa_dg_bind 결측이면 0 처리(가중 무효).
    """
    ddg = robust_ddg_value(cand, fallback=0.0)
    s_ddg = max(0.0, min(1.0, (ddg_ref - ddg) / ddg_scale))
    margin = float(cand.get("selectivity_margin", 0.0))
    s_sel = max(0.0, min(1.0, margin / 20.0)) if margin == margin else 0.0
    s_stab = float(cand.get("stability_norm", cand.get("stability", 0.0)))
    s_admet = float(cand.get("admet_score", cand.get("druggability", 0.0)))
    # MM-GBSA: 결측 시 0(무영향)
    mm = cand.get("mmgbsa_dg_bind")
    if mm is None or mm != mm:  # None or NaN
        s_mm = 0.0
    else:
        s_mm = max(0.0, min(1.0, (mmgbsa_ref - float(mm)) / mmgbsa_scale))
    score = (
        weights.ddg * s_ddg
        + weights.selectivity * s_sel
        + weights.stability * s_stab
        + weights.admet * s_admet
        + weights.mmgbsa * s_mm
    )
    return round(score, 4)


def select_topk_for_selectivity(
    candidates: List[Dict[str, Any]],
    k: int = 5,
    clash_max: float = 10.0,
) -> List[Dict[str, Any]]:
    """선택성(비싼 off-target 도킹) 대상 top-K 선별.

    clash 게이트 통과 후보를 robust ddG median 오름차순(좋은 순)으로 정렬해 상위 K.
    ``ddg_median`` 이 없으면 legacy 단일 ddG 로 폴백한다.
    """
    feasible = [
        c for c in candidates
        if float(c.get("clash_score", 999.0)) <= clash_max
    ]
    pool = feasible or candidates
    pool = sorted(pool, key=lambda c: robust_ddg_value(c, fallback=999.0))
    return pool[:k]


# ---------------------------------------------------------------------------
# Lane C — 물성 기반 용혈 휴리스틱 (Aliphatic Index > 120 규칙)
# ---------------------------------------------------------------------------
# 근거: Lane C (lane_C_heuristic.md, 2026-06-22) — L-aa 직쇄에서 정밀도1.00·재현율0.96·F1=0.98.
# 정직성 제약 (반드시 준수):
#   - L-aa 직쇄 전용. D-aa/MIXED는 적용 불가(NONE).
#   - SST-14 SS-bond 계열(cyclic, Cys3-14 보존)은 LOW 신뢰 → applies=False, risk="NA".
#   - 하드 탈락(reject) 금지. soft penalty + flag 용도로만.

_AI_HIGH_THRESHOLD = 120.0   # Aliphatic Index 고용혈 위험 컷오프 (Chen et al. 2019)
_HEMOLYSIS_SOFT_PENALTY = 0.90  # HIGH 시 admet_score 곱셈 페널티 (0.9~0.95, 탈락 아님)

def _is_cyclic_sst14(sequence: str) -> bool:
    """Cys3-Cys14 (1-indexed) SS-bond 패턴 → SST-14 계열 cyclic 여부.

    조건: 길이 14이고 3번째·14번째 잔기가 모두 Cys('C')인 경우.
    이보다 길이가 다르더라도 첫 번째와 마지막 Cys가 3번·말단에 위치하면 cyclic으로 간주.
    단, 엄밀하게는 AGCKNFFWKTFTSC 구조를 기준으로 판정한다.
    """
    seq = _normalize_sequence_preserving_d_aa(sequence)
    n = len(seq)
    if n < 6:
        return False
    # 3번째(idx=2)와 마지막(idx=-1) 위치 Cys 보존 여부 (SST-14 Cys3-Cys14 패턴)
    if n == 14:
        return seq[2] == "C" and seq[13] == "C"
    # 14aa 아닌 경우: 3번째·N번째(말단) Cys — de novo 서열에는 해당 없음
    return False


def _has_nonstandard_residue(sequence: str) -> bool:
    """비표준 잔기(D-아미노산 표기 등) 포함 여부.

    표준 20종 대문자 one-letter 코드 이외의 문자가 있으면 True.
    """
    seq = (sequence or "").strip()
    return any(ch not in _STANDARD_L_AA for ch in seq)


def hemolysis_aliphatic_flag(
    sequence: str,
    aliphatic_index: Optional[float] = None,
) -> Dict[str, Any]:
    """Lane C 물성 기반 용혈 위험 플래그 (soft, 하드게이트 아님).

    Args:
        sequence: 펩타이드 서열. L-only 입력은 대소문자 무관, 소문자 표준 AA는 D-aa marker.
        aliphatic_index: 미리 계산된 Aliphatic Index 값(있으면 재사용, 없으면 자체 계산).

    Returns:
        {
            "aliphatic_index": float,       # 계산된 AI 값
            "hemolysis_risk": "HIGH" | "LOW" | "NA",
            "applies": bool,                # True = 규칙 적용 가능 (직쇄 L-aa 표준)
            "reason": str,                  # 판정 근거 요약
        }

    applies=False → risk="NA" (cyclic SST-14 계열 또는 비표준 잔기).
    applies=True  → AI > 120 이면 risk="HIGH", 이하이면 risk="LOW".
    하드 탈락 금지 — soft penalty / flag 용도로만 사용할 것.

    정직성 제약 (Lane C 원칙):
      - SST-14 SS-bond 계열(cyclic, Cys3-14 보존): applies=False, risk="NA" (과대추정 주의).
      - D-aa/MIXED 또는 비표준 잔기 포함: applies=False, risk="NA".
      - L-aa 직쇄 표준서열만 applies=True.
    """
    seq = _normalize_sequence_preserving_d_aa(sequence)
    l_seq = _l_only_sequence(sequence)

    # 비표준 잔기(D-aa/MIXED) → applies=False. AI surrogate도 L-aa 전용이므로 계산하지 않는다.
    if _has_nonstandard_residue(seq):
        return {
            "aliphatic_index": float("nan") if aliphatic_index is None else float(aliphatic_index),
            "hemolysis_risk": "NA",
            "applies": False,
            "reason": "비표준잔기 포함(D-aa/MIXED 추정) — 규칙 적용 불가(NONE 신뢰)",
        }

    # AI 계산 (미제공 시)
    if aliphatic_index is None:
        ai: float = float("nan")
        if _HAS_PHARMA and l_seq:
            try:
                pp = _PharmaProperties(reference_seq=NATIVE_SST14)
                ai = round(pp.calculate_aliphatic_index(l_seq), 4)
            except Exception as exc:
                logger.warning("hemolysis_aliphatic_flag: AI 계산 실패(%s)", exc)
    else:
        ai = float(aliphatic_index)

    # cyclic SST-14 계열 → applies=False (LOW 신뢰, 과대추정 위험)
    if _is_cyclic_sst14(seq):
        return {
            "aliphatic_index": ai,
            "hemolysis_risk": "NA",
            "applies": False,
            "reason": "cyclic SST-14 계열(Cys3-Cys14 SS-bond) — 규칙 적용 보류(LOW 신뢰)",
        }

    # AI 계산 실패 → 판정 불가
    if ai != ai:  # NaN check
        return {
            "aliphatic_index": ai,
            "hemolysis_risk": "NA",
            "applies": False,
            "reason": "Aliphatic Index 계산 실패 — 판정 불가",
        }

    # L-aa 직쇄 표준서열 → 규칙 적용
    risk = "HIGH" if ai > _AI_HIGH_THRESHOLD else "LOW"
    return {
        "aliphatic_index": ai,
        "hemolysis_risk": risk,
        "applies": True,
        "reason": (
            f"AI={ai:.1f} > {_AI_HIGH_THRESHOLD} → 고용혈 위험(L-aa 직쇄 전용 규칙)"
            if risk == "HIGH"
            else f"AI={ai:.1f} <= {_AI_HIGH_THRESHOLD} → 저용혈 위험"
        ),
    }


def apply_hemolysis_penalty(extra: Dict[str, Any], hemo: Dict[str, Any]) -> None:
    """hemolysis_aliphatic_flag 결과를 extra_scores에 기록하고 admet_score에 soft penalty 적용 (in-place).

    - applies=False(cyclic/비표준/계산실패): extra에 기록만, admet_score 변경 없음.
    - applies=True + risk="HIGH": admet_score *= _HEMOLYSIS_SOFT_PENALTY (탈락 아님).
    - applies=True + risk="LOW": penalty 없음.

    하드 탈락 절대 금지 — 이 함수는 flag 기록 + 경미한 감점만 수행한다.
    """
    extra["hemolysis_risk"] = hemo.get("hemolysis_risk", "NA")
    extra["hemolysis_aliphatic_index"] = hemo.get("aliphatic_index")
    extra["hemolysis_applies"] = hemo.get("applies", False)
    extra["hemolysis_reason"] = hemo.get("reason", "")

    if hemo.get("applies") and hemo.get("hemolysis_risk") == "HIGH":
        base = extra.get("admet_score")
        if isinstance(base, (int, float)):
            extra["admet_score"] = round(base * _HEMOLYSIS_SOFT_PENALTY, 4)
            logger.debug(
                "hemolysis soft penalty 적용: admet %.4f → %.4f (AI=%.1f)",
                base, extra["admet_score"], hemo.get("aliphatic_index", float("nan")),
            )


# ---------------------------------------------------------------------------
# Layer 1 — selectivity (top-K, 실제 off-target PyRosetta 도킹) — 비쌈
# ---------------------------------------------------------------------------

# 2026-06-09: 큐레이션된 단일체인 off-target 수용체 (SSTR2 프레임에 0.93~0.95 사전정렬).
# 원본 *_aligned.pdb 는 G단백질 포함 멀티체인이라 부적합 → CA-overlap 으로 수용체 체인
# 식별·추출(SSTR1=D, SSTR3=A, SSTR4=R, SSTR5=R). offtarget_dock.py --pre-aligned 로 사용.
DEFAULT_OFFTARGET_RECEPTORS = {
    "SSTR1": "data/somatostatin_receptor/curated/SSTR1_receptor.pdb",
    "SSTR3": "data/somatostatin_receptor/curated/SSTR3_receptor.pdb",
    "SSTR4": "data/somatostatin_receptor/curated/SSTR4_receptor.pdb",
    "SSTR5": "data/somatostatin_receptor/curated/SSTR5_receptor.pdb",
}

_NATIVE_BASELINE_CACHE: Dict[str, Any] = {}


def _native_selectivity_baseline(root) -> Optional[float]:
    """native SST-14 의 동일프로토콜 selectivity_margin (home-advantage 기준선). 캐시."""
    import json as _json
    from pathlib import Path as _P
    key = str(root)
    if key in _NATIVE_BASELINE_CACHE:
        return _NATIVE_BASELINE_CACHE[key]
    path = _P(root) / "data/somatostatin_receptor/curated/native_selectivity_baseline.json"
    val = None
    try:
        if path.exists():
            val = float(_json.loads(path.read_text()).get("margin"))
    except Exception as exc:  # pragma: no cover
        logger.warning("native baseline 로드 실패(%s)", exc)
    _NATIVE_BASELINE_CACHE[key] = val
    return val


def screen_selectivity(
    sstr2_complex_pdb: str,
    on_target_ddg: float,
    offtarget_receptors: Optional[Dict[str, str]] = None,
    repo_root: Optional[str] = None,
    conda_env: str = "bio-tools",
    timeout: int = 600,
    margin_min: float = 10.0,
    offtarget_max_allowed: float = -15.0,
) -> Dict[str, Any]:
    """한 후보의 SSTR2 정밀화 복합체를 SSTR1/3/4/5 에 off-target 도킹하여 선택성 계산.

    실제 PyRosetta(offtarget_dock.py via step05b.dock_against_offtarget)를 호출하므로
    비싸다(수용체당 수분). top-K 후보에만 사용할 것.

    Returns dict:
        offtarget_ddg: {receptor: ddg}, selectivity_margin, worst_offtarget,
        is_selective(bool), gate_pass(bool)
      selectivity_margin = min(offtarget_ddg) - sstr2_ddg
        (양수 = SSTR2 에 더 강하게 결합 = 선택적, G-2 SSOT)
    """
    import os
    from pathlib import Path as _P

    offtarget_receptors = offtarget_receptors or DEFAULT_OFFTARGET_RECEPTORS
    root = _P(repo_root) if repo_root else _P(__file__).resolve().parents[1]

    try:
        from AG_src.pipeline.step05b_selectivity import (
            dock_against_offtarget,
            compute_selectivity_margin,
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("selectivity import 실패(%s) — skip", exc)
        return {"selectivity_margin": None, "error": str(exc)}

    cfg = {
        "selectivity": {"offtarget_timeout_sec": timeout},
        "rosetta": {"conda_env": conda_env},
    }
    # 2026-06-10: on-target SSTR2(동일 프로토콜, baseline) + off-target SSTR1/3/4/5 를 **병렬** 도킹.
    #   - SSTR2 도 off-target 과 동일 transplant+pre-relax 로 재서 margin 편향 제거(이전 아티팩트 수정).
    #   - 수용체 병렬화로 in-loop 비용 절감(순차 ~25분 → 병렬 ~6분/후보).
    from concurrent.futures import ThreadPoolExecutor as _TPE
    receptors: Dict[str, str] = {
        "SSTR2": str(root / "data/somatostatin_receptor/curated/SSTR2_receptor.pdb")
    }
    for name, rel in offtarget_receptors.items():
        receptors[name] = rel if os.path.isabs(rel) else str(root / rel)
    receptors = {n: p for n, p in receptors.items() if _P(p).exists()}

    def _dock_one(item):
        name, rpath = item
        try:
            v = float(dock_against_offtarget(
                candidate_pdb=sstr2_complex_pdb, receptor_pdb=rpath, engine="pyrosetta",
                config=cfg, on_target_score=on_target_ddg, sstr2_complex_pdb=sstr2_complex_pdb,
            ))
            return name, (round(v, 4) if v == v else None)   # NaN(fail-closed) → None
        except Exception as exc:
            logger.warning("도킹 실패 %s: %s", name, exc)
            return name, None

    with _TPE(max_workers=min(6, len(receptors)) or 1) as _ex:
        dock_results = dict(_ex.map(_dock_one, list(receptors.items())))

    sstr2_ddg_same = dock_results.pop("SSTR2", None)
    offtarget_ddg = {n: v for n, v in dock_results.items() if v is not None}
    if not offtarget_ddg:
        return {"selectivity_margin": None, "offtarget_ddg": {}}

    # baseline: 동일 프로토콜 우선, 실패 시 루프 ddg 폴백
    baseline = sstr2_ddg_same if sstr2_ddg_same is not None else on_target_ddg
    worst = min(offtarget_ddg.values())                 # 가장 강한(낮은) off-target
    margin = worst - baseline                             # 양수 = SSTR2 가 더 강함(선택적)
    # 2026-06-10 home-advantage 보정: native SST-14 도 동일 프로토콜에서 +margin(SSTR2 수용체가
    # source 복합체 유래) → 절대 margin 은 편향. native baseline 대비 Δmargin 이 진짜 선택성 신호.
    nat_margin = _native_selectivity_baseline(root)
    delta_margin = round(margin - nat_margin, 4) if nat_margin is not None else None
    out: Dict[str, Any] = {
        "offtarget_ddg": offtarget_ddg,
        "worst_offtarget": worst,
        "sstr2_ddg_sameprotocol": sstr2_ddg_same,
        "sstr2_ddg_loop": on_target_ddg,
        "selectivity_margin": round(margin, 4),
        "native_margin": nat_margin,
        "delta_margin": delta_margin,                 # >0 = native 보다 SSTR2-선택적 (home-adv 보정)
        "more_selective_than_native": (delta_margin is not None and delta_margin > 0),
        "is_selective": margin >= margin_min,
    }
    try:
        out["selectivity_detail"] = compute_selectivity_margin(
            seq_id="cand", sstr2_score=baseline, offtarget_scores=offtarget_ddg,
            margin_min=margin_min, offtarget_max_allowed=offtarget_max_allowed,
        )
    except Exception:
        pass
    return out
