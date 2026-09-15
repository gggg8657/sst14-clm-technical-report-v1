#!/usr/bin/env python3
"""Silo A + Silo B 통합 리더보드 빌드 스크립트.

출력: docs/unified_leaderboard.json

척도 차이 주의:
  - Silo B: ddg_median (nstruct robust median, REU), 14aa SST-14 변이
  - Silo A: ddg (단일값 non-robust, REU), 16aa de novo
  직접 순위 비교는 제한적임. source/robust 구분 필수.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# ── 반감기 surrogate (import 실패 시 graceful fallback) ─────────────────────────
# REPO_ROOT를 sys.path에 추가해야 scripts/ 외부의 pyrosetta_flow 패키지를 찾을 수 있음
_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

try:
    from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife as _ensemble_halflife
    _HAS_HALFLIFE = True
except Exception as _hl_import_err:
    _ensemble_halflife = None  # type: ignore[assignment]
    _HAS_HALFLIFE = False
    # 경고는 main()에서 한 번만 출력 (import 시점에 stderr 오염 방지)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
ROOT = REPO_ROOT.parent.parent.parent  # SST14-M_scr

SILO_B_LB = REPO_ROOT / "runs" / "pyrosetta_flow" / "global_selectivity_leaderboard.json"
SILO_A_LB = REPO_ROOT / "runs" / "silo_a_flow" / "silo_a_leaderboard.json"
MMGBSA_CONSENSUS = REPO_ROOT / "runs" / "pyrosetta_flow" / "mmgbsa_consensus.json"
CROSS_VAL_JSON = REPO_ROOT / "runs" / "silo_a_flow" / "cross_validation.json"
OUT_PATH = ROOT / "docs" / "unified_leaderboard.json"

# ── 반감기 계산 헬퍼 ──────────────────────────────────────────────────────────

def _get_halflife_fields(seq: str) -> dict:
    """서열로부터 반감기 필드를 반환. surrogate 불가 시 None 필드로 graceful fallback.

    Returns:
        dict with keys: half_life_h, stability_norm, halflife_source
    """
    if not _HAS_HALFLIFE or _ensemble_halflife is None:
        return {"half_life_h": None, "stability_norm": None, "halflife_source": "unavailable"}
    try:
        result = _ensemble_halflife(seq)
        hl = result.get("half_life_h")
        # NaN → None (JSON-safe)
        if hl is not None and isinstance(hl, float) and hl != hl:
            hl = None
        return {
            "half_life_h": hl,
            "stability_norm": result.get("stability_norm"),
            "halflife_source": result.get("halflife_source", "unknown"),
        }
    except Exception as exc:
        print(f"  [halflife][WARN] 반감기 계산 실패({seq[:10]}...): {exc}", file=sys.stderr)
        return {"half_life_h": None, "stability_norm": None, "halflife_source": "error"}


# ── 저복잡도 판정 (silo_a_flow._is_low_complexity 과 동일 기준) ──────────────
_LC_MONO_FRAC = 0.6
_LC_MIN_UNIQ = 3
_LC_MIN_ENTROPY = 1.5


def _is_low_complexity(seq: str) -> tuple[bool, str]:
    """서열 저복잡도 아티팩트 판정 (silo_a_flow 와 동일 로직)."""
    if not seq:
        return True, "빈 서열"
    n = len(seq)
    cnt: Counter = Counter(seq.upper())
    max_frac = max(cnt.values()) / n
    if max_frac > _LC_MONO_FRAC:
        aa = max(cnt, key=cnt.get)  # type: ignore[arg-type]
        return True, f"단일잔기과다({aa}={max_frac:.0%}>{_LC_MONO_FRAC:.0%})"
    n_unique = len(cnt)
    if n_unique <= _LC_MIN_UNIQ:
        return True, f"고유잔기부족({n_unique}<={_LC_MIN_UNIQ})"
    entropy = -sum((v / n) * math.log2(v / n) for v in cnt.values())
    if entropy < _LC_MIN_ENTROPY:
        return True, f"shannon_entropy낮음({entropy:.2f}<{_LC_MIN_ENTROPY})"
    return False, ""


# ── experiment_log 기반 세 통계(median/mean/best) 재집계 ──────────────────────

EXPERIMENT_LOG_PATH = (
    SCRIPT_DIR.parent / "runs" / "pyrosetta_flow" / "experiment_log.jsonl"
)

_OUTLIER_SD_MULTIPLIER = 1.0   # |best − median| > sd * N → outlier 경보


def load_experiment_log_stats(
    log_path: Optional[Path] = None,
) -> dict[str, dict]:
    """experiment_log.jsonl → 서열별 {median, mean, best(min), worst(max), sd, n_converged}.

    수렴 기준: ddg < 0 (ddg_median 우선, 없으면 ddg 사용).
    성능: 스트리밍 읽기(31 k 줄 기준 ~0.5 초), 결과는 호출 측에서 캐시 권장.

    Returns:
        {sequence: {ddg_median, ddg_mean, ddg_best, ddg_worst, ddg_sd, n_converged}}
    """
    path = Path(log_path or EXPERIMENT_LOG_PATH)
    by_seq: dict[str, list[float]] = {}
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = row.get("sequence") or row.get("seq")
                if not seq:
                    continue
                # ddg_median 우선, 없으면 ddg
                raw = row.get("ddg_median") or row.get("ddG_median")
                if raw is None:
                    raw = row.get("ddg") or row.get("ddG")
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if val < 0:
                    by_seq.setdefault(seq, []).append(val)
    except (FileNotFoundError, OSError):
        return {}

    result: dict[str, dict] = {}
    for seq, vals in by_seq.items():
        n = len(vals)
        med = round(statistics.median(vals), 4)
        mean = round(statistics.mean(vals), 4)
        best = round(min(vals), 4)   # 가장 음수 = 결합력 강한 값
        worst = round(max(vals), 4)  # 0에 가장 가까움 = 결합력 약한 값
        sd = round(statistics.stdev(vals), 4) if n > 1 else 0.0
        # outlier 경보: best 와 median 의 차이가 sd 초과이거나 n < 3
        outlier_warn = (
            abs(best - med) > sd * _OUTLIER_SD_MULTIPLIER
            or n < 3
        )
        result[seq] = {
            "ddg_median": med,
            "ddg_mean": mean,
            "ddg_best": best,
            "ddg_worst": worst,
            "ddg_sd": sd,
            "n_converged": n,
            "outlier_warn": outlier_warn,
        }
    return result


def enrich_entries_with_log_stats(
    entries: list[dict],
    log_stats: dict[str, dict],
) -> None:
    """통합 entry 목록에 experiment_log 재집계 통계를 in-place 보강.

    Silo B entry 에만 적용. ddg_mean/ddg_best/ddg_worst/outlier_warn 필드를 채운다.
    기존 ddg_median/ddg_sd/ddg_n_converged 는 global_leaderboard 값 우선(변경 없음).
    """
    for e in entries:
        if e.get("source") != "silo_b":
            continue
        seq = e.get("sequence", "")
        stats = log_stats.get(seq)
        if not stats:
            continue
        # ddg_mean/ddg_best 는 로그 재집계 값으로 채움(없으면 그대로 None 유지)
        if e.get("ddg_mean") is None:
            e["ddg_mean"] = stats["ddg_mean"]
        if e.get("ddg_min") is None:
            e["ddg_min"] = stats["ddg_best"]
        # outlier 경보 — log n 기준 (global lb n 보다 누적이 더 많을 수 있음)
        e["ddg_best"] = stats["ddg_best"]
        e["ddg_worst"] = stats["ddg_worst"]
        e["ddg_log_n"] = stats["n_converged"]
        e["outlier_warn"] = stats["outlier_warn"]


# ── Silo B 로드 ────────────────────────────────────────────────────────────────

def load_silo_b(mmgbsa_results: dict, silo_b_cv_map: Optional[dict] = None) -> list[dict]:
    """global_selectivity_leaderboard.json → 통합 entry 목록.

    Args:
        mmgbsa_results: mmgbsa_consensus.json 결과 dict (서열 키)
        silo_b_cv_map: cross_validation.json에서 Silo B 서열 → 검증 결과 dict
    """
    with open(SILO_B_LB, encoding="utf-8") as f:
        raw = json.load(f)

    if silo_b_cv_map is None:
        silo_b_cv_map = {}

    entries_raw: list[dict] = raw.get("entries", [])
    out: list[dict] = []
    for e in entries_raw:
        seq: str = e.get("sequence", "")
        # MM-GBSA consensus 조인
        mm = mmgbsa_results.get(seq, {})
        consensus_flag: Optional[str] = e.get("consensus_flag") or mm.get("consensus_flag")

        # cross-silo 검증 결과 조인 (Silo B 서열 기준)
        cv: dict = silo_b_cv_map.get(seq, {})
        cross_validated: Optional[str] = cv.get("verdict") if cv else None
        robust_delta_margin: Optional[float] = cv.get("robust_delta_margin") if cv else None
        offtarget_ddg: dict = cv.get("offtarget_ddg", {}) if cv else {}
        cross_robust_ddg: Optional[float] = cv.get("siloB_robust_ddg_median") if cv else None
        cross_n_converged: Optional[int] = cv.get("n_converged") if cv else None
        binding_artifact: bool = bool(cv.get("binding_artifact")) if cv else False
        binding_artifact_reason: str = cv.get("binding_artifact_reason", "") if cv else ""

        hl_fields = _get_halflife_fields(seq)
        entry: dict = {
            "source": "silo_b",
            "sequence": seq,
            "length": len(seq),
            "ddg": e.get("ddg_median") if e.get("ddg_median") is not None else e.get("ddg"),
            "ddg_raw": e.get("ddg"),
            "ddg_median": e.get("ddg_median"),
            "ddg_mean": e.get("ddg_mean"),
            "ddg_min": e.get("ddg_min"),   # best(min) — outlier 비교 기준
            "ddg_sd": e.get("ddg_sd"),
            "ddg_n_converged": e.get("ddg_n_converged"),
            "robust": True,
            "delta_margin": e.get("delta_margin"),
            # cross-silo robust selectivity (독립 재현성 검증 후 계산)
            "robust_delta_margin": robust_delta_margin,
            "offtarget_ddg": offtarget_ddg,
            "hc50": e.get("hc50"),
            "hc50_vs_native": e.get("hc50_vs_native"),
            "more_toxic_than_native": e.get("more_toxic_than_native"),
            # 반감기 surrogate
            "half_life_h": hl_fields["half_life_h"],
            "stability_norm": hl_fields["stability_norm"],
            "halflife_source": hl_fields["halflife_source"],
            "mmgbsa_dg": e.get("mmgbsa_dg") or mm.get("dg_bind"),
            "mmgbsa_consensus_flag": consensus_flag,
            # cross-silo 검증 결과
            "cross_validated": cross_validated,
            "cross_robust_ddg": cross_robust_ddg,
            "cross_n_converged": cross_n_converged,
            # C-tail artifact (intracellular tail 결합 여부)
            "binding_artifact": binding_artifact,
            "binding_artifact_reason": binding_artifact_reason,
            # Silo A 전용 필드 — B에서는 null
            "plddt": None,
            "plddt_pass": None,
            "low_complexity": False,
            "low_complexity_reason": "",
            "candidate_class": "mutation",
            "mutation_source": e.get("run_id", ""),
        }
        out.append(entry)
    return out


# ── Cross-Silo 검증 결과 로드 ─────────────────────────────────────────────────

def load_cross_validation() -> tuple[dict, dict]:
    """cross_validation.json 로드.

    Returns:
        (candidate_id_to_result, sequence_to_result) — 두 인덱스 모두 반환.
        파일 없으면 빈 dict, 빈 dict 반환 (하위 호환).
    """
    if not CROSS_VAL_JSON.exists():
        return {}, {}
    try:
        with open(CROSS_VAL_JSON, encoding="utf-8") as f:
            raw = json.load(f)
        results: dict = raw.get("results", {})
        # 서열 → 검증 결과 인덱스 (Silo B 조인용)
        seq_map: dict = {}
        for cv_entry in results.values():
            seq = cv_entry.get("sequence", "")
            src = cv_entry.get("source", "silo_a")
            if seq and src == "silo_b":
                seq_map[seq] = cv_entry
        return results, seq_map
    except (json.JSONDecodeError, OSError):
        return {}, {}


# ── Silo A 로드 ────────────────────────────────────────────────────────────────

def load_silo_a(cross_val: Optional[dict] = None) -> list[dict]:
    """silo_a_leaderboard.json → 통합 entry 목록 (cross_val 조인 포함).

    cross_val: {candidate_id: 검증 결과 dict} — cross_validation.json 내용.
               None 이면 cross-silo 미적용.
    """
    if cross_val is None:
        cross_val = {}

    with open(SILO_A_LB, encoding="utf-8") as f:
        raw = json.load(f)

    entries_raw: list[dict] = raw.get("entries", [])
    out: list[dict] = []
    for e in entries_raw:
        seq: str = e.get("sequence", "")
        lc_flag, lc_reason = _is_low_complexity(seq)

        cid: str = e.get("candidate_id", "")
        cv: dict = cross_val.get(cid, {})

        # cross-silo 검증 결과 반영
        cross_validated: Optional[str] = cv.get("verdict") if cv else None
        robust_ddg: Optional[float] = cv.get("siloB_robust_ddg_median") if cv else None
        mmgbsa_dg_cv: Optional[float] = cv.get("mmgbsa_dg") if cv else None
        # robust selectivity (off-target 재도킹 결과)
        robust_delta_margin: Optional[float] = cv.get("robust_delta_margin") if cv else None
        offtarget_ddg: dict = cv.get("offtarget_ddg", {}) if cv else {}
        binding_artifact: bool = bool(cv.get("binding_artifact")) if cv else False
        binding_artifact_reason: str = cv.get("binding_artifact_reason", "") if cv else ""

        # confirmed 계열 모두 robust로 승격 (binding_artifact=True는 이미 refuted로 강등됨)
        is_robust = cross_validated in ("confirmed", "confirmed_selective", "confirmed_nonselective")

        hl_fields_a = _get_halflife_fields(seq)
        entry: dict = {
            "source": "silo_a",
            "candidate_id": cid,
            "sequence": seq,
            "length": len(seq),
            "ddg": e.get("ddg"),
            "ddg_raw": e.get("ddg"),
            "ddg_median": None,   # A는 단일값
            "ddg_sd": None,
            "ddg_n_converged": None,
            "robust": is_robust,
            "delta_margin": e.get("delta_margin"),
            # cross-silo robust selectivity
            "robust_delta_margin": robust_delta_margin,
            "offtarget_ddg": offtarget_ddg,
            "hc50": e.get("hc50"),
            "hc50_vs_native": None,  # A에는 없음
            "more_toxic_than_native": None,
            # 반감기 surrogate
            "half_life_h": hl_fields_a["half_life_h"],
            "stability_norm": hl_fields_a["stability_norm"],
            "halflife_source": hl_fields_a["halflife_source"],
            "mmgbsa_dg": mmgbsa_dg_cv,  # cross-silo MM-GBSA 우선
            "mmgbsa_consensus_flag": None,
            "plddt": e.get("plddt"),
            "plddt_pass": e.get("plddt_pass"),
            "low_complexity": lc_flag,
            "low_complexity_reason": lc_reason,
            "candidate_class": e.get("candidate_class", "de_novo"),
            "mutation_source": e.get("mutation_source", "silo_a"),
            # cross-silo 전용 필드
            "cross_validated": cross_validated,
            "robust_ddg": robust_ddg,
            "cross_ddg_sd": cv.get("ddg_sd") if cv else None,
            "cross_n_converged": cv.get("n_converged") if cv else None,
            # C-tail artifact (intracellular tail 결합 여부)
            "binding_artifact": binding_artifact,
            "binding_artifact_reason": binding_artifact_reason,
        }
        out.append(entry)
    return out


# ── 통합 + 정렬 + 순위 부여 ────────────────────────────────────────────────────

def _sort_key_unified(e: dict) -> tuple:
    """통합 리더보드 정렬 키 (selectivity 반영).

    tier:
      0: confirmed_selective (cross-silo 확정 + Δmargin > 0) — 최우선
      0: confirmed / confirmed_nonselective (cross-silo 확정)
      0: Silo B (항상 robust)
      1: 미검증 (cross_validated=None)
      2: uncertain (수렴 부족)
      3: refuted / 저복잡도 (강등)

    동일 tier 내: robust_ddg / ddg 오름차순 (낮을수록 우선)
    동일 tier 0 내 selectivity 보너스: Δmargin을 secondary 키로 사용 (높을수록 우선 → 부호 반전)

    반환: (tier, selectivity_secondary, ddg_value) — 오름차순 정렬
    """
    cv: Optional[str] = e.get("cross_validated")
    lc: bool = bool(e.get("low_complexity"))
    ddg: Optional[float] = e.get("ddg")
    dm: Optional[float] = e.get("robust_delta_margin")
    # selectivity secondary key: Δmargin 높을수록 우선 → 부호 반전 (낮은 값 = 우선)
    dm_sec: float = -(dm if dm is not None else -999.0)

    if e["source"] == "silo_b":
        # Silo B: tier 0, selectivity 반영
        b_cv = e.get("cross_validated")  # None or verdict (독립 재현성 검증 결과)
        if b_cv == "confirmed_selective":
            return (0, dm_sec, ddg if ddg is not None else 0.0)
        return (0, 0.0, ddg if ddg is not None else 0.0)

    if lc:
        return (3, 0.0, ddg if ddg is not None else 0.0)

    if cv in ("confirmed_selective", "confirmed", "confirmed_nonselective"):
        robust = e.get("robust_ddg")
        val = robust if robust is not None else (ddg if ddg is not None else 0.0)
        tier = 0
        if cv == "confirmed_selective":
            return (tier, dm_sec, val)
        return (tier, 0.0, val)
    elif cv == "refuted":
        return (3, 0.0, ddg if ddg is not None else 0.0)
    elif cv == "uncertain":
        return (2, 0.0, ddg if ddg is not None else 0.0)
    else:
        # 미검증 → tier 1
        return (1, 0.0, ddg if ddg is not None else 0.0)


def build_unified(
    silo_b: list[dict],
    silo_a: list[dict],
    top_per_silo: int = 50,
    top_unified: int = 100,
) -> dict:
    """두 리더보드 병합 → cross-silo verdict 반영 정렬 → top_unified 반환.

    정렬:
      - Silo B: tier 0 (항상 robust)
      - Silo A confirmed: tier 0 (robust ddg 기준)
      - Silo A 미검증: tier 1
      - Silo A uncertain: tier 2
      - Silo A refuted / 저복잡도: tier 3 (강등)
    """
    # 각 Silo 상위 top_per_silo 만 포함
    b_sorted = sorted(
        (e for e in silo_b if e["ddg"] is not None),
        key=lambda x: x["ddg"],
    )[:top_per_silo]
    a_sorted = sorted(
        (e for e in silo_a if e["ddg"] is not None),
        key=lambda x: x["ddg"],
    )[:top_per_silo]

    combined = b_sorted + a_sorted
    # cross-silo 반영 정렬
    combined.sort(key=_sort_key_unified)
    combined = combined[:top_unified]

    for i, e in enumerate(combined, start=1):
        e["rank"] = i

    # 통계
    n_silo_a = sum(1 for e in combined if e["source"] == "silo_a")
    n_silo_b = sum(1 for e in combined if e["source"] == "silo_b")
    n_low_complexity = sum(1 for e in combined if e.get("low_complexity"))
    n_mmgbsa_high = sum(
        1 for e in combined if e.get("mmgbsa_consensus_flag") == "high_confidence"
    )
    n_cv_confirmed = sum(
        1 for e in combined
        if e.get("cross_validated") in ("confirmed", "confirmed_selective", "confirmed_nonselective")
    )
    n_cv_selective = sum(
        1 for e in combined
        if e.get("cross_validated") == "confirmed_selective"
    )
    n_cv_refuted = sum(
        1 for e in combined
        if e.get("cross_validated") == "refuted"
    )
    n_cv_uncertain = sum(
        1 for e in combined
        if e.get("cross_validated") == "uncertain"
    )
    n_robust_delta_margin = sum(
        1 for e in combined
        if e.get("robust_delta_margin") is not None
    )

    import datetime
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    caveat = (
        "【척도 차이 주의】 "
        "Silo A(de novo·16aa): ddG는 단일값(non-robust), 저복잡도 서열 존재 가능. "
        "Silo B(변이·14aa): ddG_median은 nstruct robust 통계(n≥1). "
        "두 Silo의 ddG는 측정 방식·서열 길이·기원이 달라 직접 순위 비교가 제한적입니다. "
        "MM-GBSA consensus(고신뢰 배지)는 Silo B 전용. "
        "Silo A 저복잡도 서열(poly-G/L 등)은 아티팩트 위험 — low_complexity 컬럼 참조. "
        "cross_validated: confirmed_selective=선택성 검증 완료(Δmargin>0), "
        "confirmed_nonselective=확정but비선택적, confirmed=확정(selectivity 미계산), "
        "refuted=반증됨(강등), uncertain=수렴 부족. "
        "robust_delta_margin: 독립 재도킹 후 off-target(SSTR1/3/4/5) 평균 ddG - SSTR2 ddG "
        "(양수=SSTR2 선호, 신뢰도 높음)."
    )

    return {
        "generated_at": generated_at,
        "caveat": caveat,
        "caveat_en": (
            "[Scale difference] Silo A (de novo, 16aa): single ddG value, non-robust, "
            "low-complexity artifacts possible. "
            "Silo B (mutation, 14aa): ddG_median = nstruct robust statistic. "
            "Direct rank comparison across silos is limited. "
            "MM-GBSA consensus available for Silo B only. "
            "cross_validated: confirmed_selective=selectivity verified (Δmargin>0), "
            "confirmed_nonselective=confirmed but not selective, "
            "confirmed=confirmed (selectivity not computed), "
            "refuted=disproved (demoted), uncertain=insufficient convergence. "
            "robust_delta_margin: mean(off-target ddG) - SSTR2 ddG after independent redocking "
            "(positive = SSTR2-preferential, high confidence)."
        ),
        "n_total": len(combined),
        "n_silo_a": n_silo_a,
        "n_silo_b": n_silo_b,
        "n_low_complexity_in_top": n_low_complexity,
        "n_mmgbsa_high_confidence": n_mmgbsa_high,
        "n_cross_validated_confirmed": n_cv_confirmed,
        "n_cross_validated_selective": n_cv_selective,
        "n_cross_validated_refuted": n_cv_refuted,
        "n_cross_validated_uncertain": n_cv_uncertain,
        "n_robust_delta_margin": n_robust_delta_margin,
        "top_per_silo_included": top_per_silo,
        "entries": combined,
    }


# ── 패널 도입 전/후 비교 (★C: 5-전문가 패널+모델이질성 도입, 커밋 96f299f1) ─────
#
# baseline: 패널 도입 직전 스냅샷 (2026-07-01T02:12:13Z).
# 정직성 원칙: 데이터가 아직 적으면("데이터 축적 중") 그대로 표시하고 가짜 개선을
# 만들지 않는다. 비교는 실제 데이터가 쌓이면 autopush 재생성 시 자동 갱신된다.
PANEL_BASELINE_DIR = (
    REPO_ROOT / "runs" / "pyrosetta_flow" / "archives" / "pre_hetero_panel_20260701T021213Z"
)
PANEL_BASELINE_CUTOFF_TS = "2026-07-01T02:12:13"
PANEL_BASELINE_LB_PATH = PANEL_BASELINE_DIR / "runs_pyrosetta_flow_global_selectivity_leaderboard.json"
PANEL_BASELINE_LOG_PATH = PANEL_BASELINE_DIR / "runs_pyrosetta_flow_experiment_log.jsonl"

# native 통계유의 능가 판정 기준 (cross_silo_validation.py::_NATIVE_BEATS_MIN_DIFF_REU 와 동일).
_NATIVE_STATSIG_MIN_DIFF_REU: float = 15.0
# 최소 표시 임계 — 이 미만의 신규 candidate 수면 "데이터 축적 중"으로 정직 표시.
_MIN_NEW_CANDIDATES_FOR_TREND: int = 50


def _count_new_candidates_since(log_path: Path, cutoff_ts: str) -> int:
    """experiment_log.jsonl에서 cutoff_ts 이후 신규 candidate 레코드 수를 센다.

    파일이 없거나 파싱 실패 시 0 반환(graceful).
    """
    if not log_path.exists():
        return 0
    n = 0
    try:
        with log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("record_type") != "candidate":
                    continue
                ts = rec.get("ts") or ""
                if ts > cutoff_ts:
                    n += 1
    except OSError:
        return 0
    return n


def _count_statsig_beats_native(
    entries: list[dict], native_ddg: float, min_diff: float = _NATIVE_STATSIG_MIN_DIFF_REU
) -> tuple[int, set]:
    """robust median 기준 native를 통계유의(≥min_diff REU) 능가하는 후보 수/서열 집합.

    n_converged>=1인 robust median만 대상(단일값 Silo A는 척도가 달라 제외).
    """
    seqs: set = set()
    for e in entries:
        med = e.get("ddg_median")
        n_conv = e.get("ddg_n_converged") or 0
        if med is None or n_conv < 1:
            continue
        if (native_ddg - float(med)) >= min_diff:
            seqs.add(e.get("sequence", ""))
    return len(seqs), seqs


def build_panel_comparison(
    current_silo_b_lb: dict,
    native_ddg: float,
) -> dict:
    """★C: 새 패널(5전문가+모델이질성+confidence-gate) 도입 전/후 비교 지표 산출.

    ★정직성: baseline 스냅샷 미존재 또는 신규 데이터가 부족(<_MIN_NEW_CANDIDATES_FOR_TREND)
    하면 "데이터 축적 중"으로 표시하고 임의로 개선 수치를 만들지 않는다.

    비교 지표:
      1. n_new_evaluations   — baseline cutoff 이후 신규 candidate 평가 수
      2. best_ddg_median_before/after — Silo B 리더보드 best robust median 변화
      3. n_statsig_beats_native_before/after — native(≥15 REU) 통계유의 능가 후보 수 변화
      4. new_top_candidates  — baseline top50에 없던 신규 서열(top50 진입 기준)

    Returns:
        panel_comparison dict — dashboard Tab 7에 별도 섹션으로 노출.
    """
    baseline_available = PANEL_BASELINE_LB_PATH.exists()
    if not baseline_available:
        return {
            "available": False,
            "status": "baseline_missing",
            "message": (
                f"baseline 스냅샷 없음 ({PANEL_BASELINE_LB_PATH}) — 비교 불가."
            ),
        }

    try:
        with open(PANEL_BASELINE_LB_PATH, encoding="utf-8") as f:
            baseline_lb = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "status": "baseline_read_error",
            "message": f"baseline 리더보드 읽기 실패: {exc}",
        }

    # 1. 신규 평가 수 (experiment_log 타임스탬프 기준)
    current_log_path = REPO_ROOT / "runs" / "pyrosetta_flow" / "experiment_log.jsonl"
    n_new_evals = _count_new_candidates_since(current_log_path, PANEL_BASELINE_CUTOFF_TS)

    data_sufficient = n_new_evals >= _MIN_NEW_CANDIDATES_FOR_TREND

    # 2. best ddg_median 변화 (Silo B)
    best_before = baseline_lb.get("best_ddg_median")
    best_after = current_silo_b_lb.get("best_ddg_median")
    best_delta: Optional[float] = None
    if best_before is not None and best_after is not None:
        best_delta = round(float(best_after) - float(best_before), 4)

    # 3. native 통계유의 능가 후보 수 (top-N 리더보드 스냅샷 기준 — 참고용, 전수 아님)
    n_statsig_before, seqs_before = _count_statsig_beats_native(
        baseline_lb.get("entries", []), native_ddg
    )
    n_statsig_after, seqs_after = _count_statsig_beats_native(
        current_silo_b_lb.get("entries", []), native_ddg
    )

    # 4. 신규 top-N 진입 서열 (baseline top-N에 없던 서열)
    baseline_seqs = {e.get("sequence") for e in baseline_lb.get("entries", [])}
    current_seqs_top = [
        e.get("sequence") for e in current_silo_b_lb.get("entries", [])
        if e.get("sequence") not in baseline_seqs
    ]

    return {
        "available": True,
        "status": "sufficient" if data_sufficient else "accumulating",
        "message": (
            f"신규 평가 {n_new_evals}건 축적됨 (임계 {_MIN_NEW_CANDIDATES_FOR_TREND}건 미만) — "
            "패널 효과 판단에는 데이터 축적이 더 필요합니다. 가짜 개선 수치가 아닌 "
            "실측값을 그대로 표시합니다."
            if not data_sufficient
            else f"신규 평가 {n_new_evals}건 누적 — 추세 판단 가능 단계."
        ),
        "baseline_cutoff": PANEL_BASELINE_CUTOFF_TS,
        "baseline_archive": str(PANEL_BASELINE_DIR.relative_to(REPO_ROOT)),
        "n_new_evaluations": n_new_evals,
        "min_new_for_trend": _MIN_NEW_CANDIDATES_FOR_TREND,
        "best_ddg_median_before": best_before,
        "best_ddg_median_after": best_after,
        "best_ddg_median_delta": best_delta,
        "native_ddg_used": native_ddg,
        "native_statsig_min_diff_reu": _NATIVE_STATSIG_MIN_DIFF_REU,
        "n_statsig_beats_native_before": n_statsig_before,
        "n_statsig_beats_native_after": n_statsig_after,
        "n_statsig_beats_native_delta": n_statsig_after - n_statsig_before,
        "new_top_candidates": current_seqs_top[:20],
        "n_new_top_candidates": len(current_seqs_top),
        "caveat": (
            "robust median끼리만 비교(단일값 Silo A와 혼동 금지). "
            "native 능가는 통계유의(≥15 REU) 기준 유지. "
            "top-N(리더보드 capacity) 스냅샷 비교이므로 전체 모집단 통계가 아님 — 참고용."
        ),
    }


# ── 메인 ───────────────────────────────────────────────────────────────────────

NATIVE_SEQUENCE = "AGCKNFFWKTFTSC"

# ── canonical native ddG 소스 (v2 검증값) ─────────────────────────────────────
# native_robust_baseline_v2.json: 진짜 native PDB + nstruct=10, n_converged=8/10, HIGH
# 이전 −41.99는 n_converged=2 + 입력PDB 불명 → LOW reliability (오염값, 사용 금지).
NATIVE_ROBUST_BASELINE_V2_PATH = (
    REPO_ROOT / "runs" / "pyrosetta_flow" / "native_robust_baseline_v2.json"
)
# fallback: v2 파일 미존재 + mmgbsa_consensus.json native_ddg 결측 시 최후 방어값
_NATIVE_DDG_FALLBACK: float = -20.28   # v2 ddg_median 대표값
_NATIVE_DG_FALLBACK: float = -71.05    # MM-GBSA native ΔG (도킹 ddG와 별개 — 건드리지 말 것)


def _load_native_ddg_v2_leaderboard() -> tuple[Optional[float], str]:
    """native_robust_baseline_v2.json에서 canonical native ddG를 읽어 반환.

    우선순위:
      1. native_robust_baseline_v2.json::ddg_median (진짜 native PDB + nstruct=10, HIGH)
      2. mmgbsa_consensus.json::native_ddg (데몬이 이미 v2를 읽어 갱신한 경우)
      3. _NATIVE_DDG_FALLBACK (-20.28)

    Returns:
        (native_ddg_value, source_label) — source_label은 출처 추적용 문자열.
    """
    # 1순위: v2 파일 직접 읽기
    if NATIVE_ROBUST_BASELINE_V2_PATH.exists():
        try:
            with open(NATIVE_ROBUST_BASELINE_V2_PATH, encoding="utf-8") as fv:
                v2 = json.load(fv)
            val = v2.get("ddg_median")
            result = float(val)
            n_conv = v2.get("n_converged", "?")
            rel = v2.get("reliability", "?")
            print(
                f"  [native-ddg] v2 파일 로드 완료: {result:.4f} REU "
                f"(n_converged={n_conv}, reliability={rel})",
                file=sys.stderr,
            )
            return result, "robust_v2_nstruct10"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            print(
                f"  [native-ddg][WARN] v2 파일 파싱 실패: {exc} "
                "→ mmgbsa_consensus fallback 시도",
                file=sys.stderr,
            )
    else:
        print(
            f"  [native-ddg][WARN] v2 파일 미존재: {NATIVE_ROBUST_BASELINE_V2_PATH} "
            "→ mmgbsa_consensus fallback 시도",
            file=sys.stderr,
        )

    # 2순위: mmgbsa_consensus.json::native_ddg_source 확인 후 사용
    # (데몬이 이미 v2를 읽어 갱신했다면 "robust_v2_nstruct10" 소스임)
    try:
        with open(MMGBSA_CONSENSUS, encoding="utf-8") as fc:
            consensus = json.load(fc)
        val = consensus.get("native_ddg")
        src_label = consensus.get("native_ddg_source", "mmgbsa_consensus_unknown")
        result = float(val)
        if src_label == "robust_v2_nstruct10":
            print(
                f"  [native-ddg] mmgbsa_consensus 경유 v2값 사용: {result:.4f} REU "
                f"(source={src_label})",
                file=sys.stderr,
            )
        else:
            print(
                f"  [native-ddg][WARN] mmgbsa_consensus native_ddg={result:.4f} REU "
                f"(source={src_label}) — v2 아님, 신뢰 불확실",
                file=sys.stderr,
            )
        return result, src_label
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    # 3순위: 하드코딩 fallback
    print(
        f"  [native-ddg][ERROR] 모든 소스 실패 — fallback {_NATIVE_DDG_FALLBACK} 사용",
        file=sys.stderr,
    )
    return _NATIVE_DDG_FALLBACK, "hardcoded_fallback_v2_representative"


def build_native_baseline_entry(mm_raw: dict) -> dict:
    """canonical native ddG(v2 우선)를 읽어 기준선 엔트리를 반환.

    native_ddg 결정 순서:
      1. native_robust_baseline_v2.json::ddg_median (진짜 native PDB + nstruct=10, HIGH)
      2. mmgbsa_consensus.json::native_ddg (데몬이 v2 읽어 갱신했을 경우)
      3. _NATIVE_DDG_FALLBACK (-20.28)

    MM-GBSA native_dg(-71.05) 는 도킹 ddG와 별개 채점값 — 그대로 유지.

    Args:
        mm_raw: mmgbsa_consensus.json 전체 dict (이미 로드된 상태).
            native_ddg는 이 함수 내에서 v2 우선 재결정. mm_raw는 native_dg 참조에만 사용.

    Returns:
        is_baseline=True 플래그가 포함된 native 엔트리 dict.
    """
    # native_ddg: v2 파일 우선
    native_ddg, ddg_source = _load_native_ddg_v2_leaderboard()

    # native_dg (MM-GBSA): 건드리지 말 것 — 도킹 ddG와 별개
    native_dg: Optional[float] = mm_raw.get("native_dg")
    if native_dg is None:
        print(
            f"[WARN] native_dg 없음 — fallback {_NATIVE_DG_FALLBACK} 사용",
            file=sys.stderr,
        )
        native_dg = _NATIVE_DG_FALLBACK

    # native 반감기 (SST-14 서열 기준)
    native_hl = _get_halflife_fields(NATIVE_SEQUENCE)

    return {
        "source": "native",
        "sequence": NATIVE_SEQUENCE,
        "length": len(NATIVE_SEQUENCE),
        "ddg": native_ddg,
        "ddg_raw": native_ddg,
        "ddg_median": native_ddg,
        "ddg_mean": native_ddg,
        "ddg_min": native_ddg,
        "ddg_best": native_ddg,
        "ddg_worst": native_ddg,
        "ddg_sd": 0.0,
        "ddg_n_converged": None,
        "robust": True,
        "delta_margin": 0.0,
        "robust_delta_margin": 0.0,
        "hc50": None,
        "hc50_vs_native": 0.0,
        "more_toxic_than_native": False,
        # 반감기 surrogate (native baseline 참조값)
        "half_life_h": native_hl["half_life_h"],
        "stability_norm": native_hl["stability_norm"],
        "halflife_source": native_hl["halflife_source"],
        "mmgbsa_dg": native_dg,
        "mmgbsa_consensus_flag": "baseline",
        "plddt": None,
        "plddt_pass": None,
        "low_complexity": False,
        "low_complexity_reason": "",
        "candidate_class": "baseline",
        "mutation_source": "native",
        "native_ddg_source": ddg_source,    # 출처 추적 필드
        "cross_validated": None,
        "robust_ddg": None,
        "cross_ddg_sd": None,
        "cross_n_converged": None,
        "binding_artifact": False,
        "binding_artifact_reason": "",
        "offtarget_ddg": {},
        # 기준선 전용 플래그 — 정렬/필터에서 별도 취급
        "is_baseline": True,
        "rank": 0,
    }


def main() -> None:
    t0 = time.perf_counter()

    # 반감기 surrogate import 상태 보고
    if _HAS_HALFLIFE:
        print("  [halflife] ensemble_halflife_v2 로드 완료 — 반감기 필드 채움", file=sys.stderr)
    else:
        print(
            "  [halflife][WARN] halflife_ensemble_v2 import 실패 "
            "→ half_life_h/stability_norm/halflife_source=None으로 채움",
            file=sys.stderr,
        )

    # 파일 존재 확인
    for p in (SILO_B_LB, SILO_A_LB, MMGBSA_CONSENSUS):
        if not p.exists():
            print(f"[ERROR] 파일 없음: {p}", file=sys.stderr)
            sys.exit(1)

    # MM-GBSA consensus 결과 로드
    with open(MMGBSA_CONSENSUS, encoding="utf-8") as f:
        mm_raw = json.load(f)
    mm_results: dict = mm_raw.get("results", {})

    # cross-silo 검증 결과 로드 (파일 없으면 빈 dict → 하위 호환)
    cross_val, silo_b_cv_map = load_cross_validation()
    if cross_val:
        n_cv = len(cross_val)
        n_sel = sum(1 for v in cross_val.values() if v.get("verdict") == "confirmed_selective")
        print(
            f"  [cross-silo] {n_cv}건 교차 검증 결과 로드 완료 (선택성 확인: {n_sel}건)",
            file=sys.stderr,
        )
    else:
        print(
            "  [cross-silo] cross_validation.json 없음 — cross-silo 배지 미표시",
            file=sys.stderr,
        )

    # 각 Silo 로드
    silo_b = load_silo_b(mm_results, silo_b_cv_map=silo_b_cv_map)
    silo_a = load_silo_a(cross_val=cross_val)

    # experiment_log 재집계 → Silo B entry에 mean/best/outlier_warn 보강
    if EXPERIMENT_LOG_PATH.exists():
        t_log = time.perf_counter()
        log_stats = load_experiment_log_stats()
        enrich_entries_with_log_stats(silo_b, log_stats)
        n_enriched = sum(1 for e in silo_b if e.get("ddg_best") is not None)
        print(
            f"  [log-stats] experiment_log 재집계: {len(log_stats)}서열 "
            f"→ {n_enriched}건 보강 ({time.perf_counter()-t_log:.2f}s)",
            file=sys.stderr,
        )
    else:
        print(
            f"  [log-stats] {EXPERIMENT_LOG_PATH} 없음 — mean/best 보강 스킵",
            file=sys.stderr,
        )

    # 통합
    unified = build_unified(silo_b, silo_a)

    # native 기준선 엔트리 주입 — entries 맨 앞에 추가, 기존 순위 보존
    native_entry = build_native_baseline_entry(mm_raw)
    unified["entries"] = [native_entry] + unified["entries"]
    # 통합 리더보드에도 native_ddg_source 기록
    unified["native_ddg"] = native_entry["ddg"]
    unified["native_ddg_source"] = native_entry["native_ddg_source"]
    print(
        f"  [native baseline] AGCKNFFWKTFTSC 엔트리 주입 완료 "
        f"(ddG={native_entry['ddg']:.4f} REU, source={native_entry['native_ddg_source']}, "
        f"MM-GBSA ΔG={native_entry['mmgbsa_dg']})",
        file=sys.stderr,
    )

    # ★C: 패널 도입 전/후 비교 (5-전문가 패널+모델이질성, 커밋 96f299f1)
    # Silo B 원본 리더보드(raw)를 다시 읽어 비교 — load_silo_b()의 변환 entry가 아닌
    # ddg_median/ddg_n_converged 원본 필드가 필요.
    try:
        with open(SILO_B_LB, encoding="utf-8") as f:
            _silo_b_raw_for_cmp = json.load(f)
        panel_comparison = build_panel_comparison(
            current_silo_b_lb=_silo_b_raw_for_cmp,
            native_ddg=native_entry["ddg"],
        )
    except (OSError, json.JSONDecodeError) as exc:
        panel_comparison = {
            "available": False,
            "status": "current_read_error",
            "message": f"현재 Silo B 리더보드 읽기 실패: {exc}",
        }
    unified["panel_comparison"] = panel_comparison
    print(
        f"  [panel-comparison] status={panel_comparison.get('status')} "
        f"n_new_evaluations={panel_comparison.get('n_new_evaluations', '—')}",
        file=sys.stderr,
    )

    # 출력 디렉토리 생성
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(unified, f, ensure_ascii=False, indent=2)

    elapsed = time.perf_counter() - t0
    n_confirmed = unified.get("n_cross_validated_confirmed", 0)
    n_selective = unified.get("n_cross_validated_selective", 0)
    n_refuted = unified.get("n_cross_validated_refuted", 0)
    n_uncertain = unified.get("n_cross_validated_uncertain", 0)
    n_rdm = unified.get("n_robust_delta_margin", 0)
    print(
        f"[build_unified_leaderboard] 완료: {OUT_PATH}\n"
        f"  Silo B: {len(silo_b)}건 → top50 포함\n"
        f"  Silo A: {len(silo_a)}건 → top50 포함\n"
        f"  통합 top{unified['n_total']}건 (A:{unified['n_silo_a']}, B:{unified['n_silo_b']})\n"
        f"  저복잡도 포함: {unified['n_low_complexity_in_top']}건\n"
        f"  MM-GBSA 고신뢰 포함: {unified['n_mmgbsa_high_confidence']}건\n"
        f"  Cross-silo confirmed={n_confirmed}(선택성={n_selective}) "
        f"refuted={n_refuted} uncertain={n_uncertain}\n"
        f"  robust Δmargin 계산 완료: {n_rdm}건\n"
        f"  소요시간: {elapsed:.2f}s"
    )

    # top5 출력
    print("\n  ── 통합 top5 ──")
    for e in unified["entries"][:5]:
        lc = " [저복잡도!]" if e.get("low_complexity") else ""
        rb = "robust" if e["robust"] else "non-robust"
        cv = e.get("cross_validated")
        cv_str = f" [CV:{cv}]" if cv else ""
        dm = e.get("delta_margin")
        dm_str = f"{dm:.4f}" if dm is not None else "N/A"
        rdm = e.get("robust_delta_margin")
        rdm_str = f"{rdm:+.4f}" if rdm is not None else "N/A"
        print(
            f"  #{e['rank']:>2} [{e['source']}] {e['sequence']:<20} "
            f"ddG={e['ddg']:.2f} Δmargin={dm_str} robust_Δmargin={rdm_str} "
            f"({rb}){lc}{cv_str}"
        )


if __name__ == "__main__":
    main()
