"""
Cluster classification for SST-14 analogue candidates.

Assigns each candidate to one of five clusters (A–E) based on the evaluation
criteria defined in the meet_log_backup.md specification.  Classification is
deterministic and uses only the numeric/structured fields already computed by
the runner pipeline and pharmacology module — no external dependencies.

Cluster definitions
-------------------
A – High Affinity Core
    ddG ≤ -8.0 AND clash_score ≤ 5 AND pLDDT ≥ 75 AND FWKT contact maintained
B – Selectivity-Optimised
    SSTR2 ddG low (implicitly via selectivity_margin) AND selectivity_margin ≥ 3.0
C – Stability-Enhanced
    instability_index < 30 AND blosum62 total_score high (≥ 0) AND
    reduced protease sites (total_sites ≤ 8 compared with SST-14 native of 9)
D – Radiochemistry-Optimal
    GRAVY ∈ [-1.0, +0.5] AND |net_charge_ph74| ≤ 1.0 AND chelator site available
    (metal_coordination.n_strong ≥ 1)
E – Exploratory Candidates
    All remaining candidates not assigned to A–D.

Priority: A > B > C > D > E.  A candidate satisfying multiple cluster criteria
is assigned to the highest-priority cluster only.

Usage
-----
    from pyrosetta_flow.cluster_report import classify_cluster, batch_classify
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ─────────────────────────── constants ────────────────────────────────────────

_CLUSTER_NAMES: Dict[str, str] = {
    "A": "High Affinity Core",
    "B": "Selectivity-Optimised",
    "C": "Stability-Enhanced",
    "D": "Radiochemistry-Optimal",
    "E": "Exploratory Candidates",
}

_CLUSTER_PRIORITY: Dict[str, int] = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
}

# SST-14 native protease site count used as baseline for cluster C comparison
_SST14_PROTEASE_BASELINE: int = 9


# ─────────────────────────── helpers ──────────────────────────────────────────


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Coerce value to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fwkt_contact_maintained(candidate: Dict[str, Any]) -> bool:
    """Return True if structural_rules indicates FWKT pharmacophore passes.

    The structural_rules dict (from check_structural_rules / check_structural_rules
    in pharma_properties.py) contains ``rules.fwkt_pharmacophore.pass``.
    Accepts both nested and flat formats to stay forward-compatible.
    """
    sr = candidate.get("structural_rules")
    if sr is None:
        return False
    if isinstance(sr, dict):
        rules = sr.get("rules", sr)
        fwkt = rules.get("fwkt_pharmacophore", {})
        if isinstance(fwkt, dict):
            return bool(fwkt.get("pass", False))
        # flat bool format
        return bool(fwkt)
    return False


def _blosum62_total(candidate: Dict[str, Any]) -> float:
    """Extract BLOSUM62 total score from various dict shapes."""
    b = candidate.get("blosum62")
    if b is None:
        return 0.0
    if isinstance(b, dict):
        # pharmacology.py: "total_blosum62_score"
        # pharma_properties.py: "total_score"
        score = b.get("total_blosum62_score", b.get("total_score", 0))
        return _safe_float(score, 0.0)
    return 0.0


def _protease_total(candidate: Dict[str, Any]) -> int:
    """Extract total protease site count."""
    ps = candidate.get("protease_sites")
    if ps is None:
        return 0
    if isinstance(ps, dict):
        # pharmacology.py uses "total_sites"; pharma_properties.py uses "total"
        total = ps.get("total_sites", ps.get("total", 0))
        return int(total)
    return 0


def _metal_n_strong(candidate: Dict[str, Any]) -> int:
    """Extract number of strong metal-coordinating residues."""
    mc = candidate.get("metal_coordination")
    if mc is None:
        return 0
    if isinstance(mc, dict):
        return int(mc.get("n_strong", 0))
    return 0


def _chelator_site_from_candidate(candidate: Dict[str, Any]) -> bool:
    """DOTA chelator 부착 가능 site 판정 — compute_chelator_site 위임 + n_strong fallback.

    약리학적 기준 (reviewer-pharma 정의, 2026-05-14):
      Primary:   N-terminus free α-NH₂ (non-Pro) → DOTA-NHS ester coupling 가능
                 DOTATATE 선례: N-terminus 우선 (de Jong 2002; Maecke 2005)
      Secondary: Lys ε-NH₂ (K)                   → DOTA ε-amine coupling 가능
      Forbidden: SS-bond Cys thiol (disulfide 상태 → chelation 불가)

    수정 이유 (P2-1):
      기존 _metal_n_strong() 기반 로직은 SS-bond 상태의 Cys thiol을 n_strong에
      포함시켜 chelation site 존재를 과대 평가했음.
      예) "PGCPNFFWRTFTSC" (Pro N-term, Lys→Arg): Cys SS-bond → n_strong≥1이지만
          실제 chelation 부위 없음 → 오분류.

    확장 (2026-06-25):
      backend.pharmacophore.compute_chelator_site() 위임으로 상세 정보 활용.
      K5(idx 4) 포켓 인접 위험, fatty_acid/PEG competition_risk 판정 포함.
      단, cluster_report 내부에서는 bool만 필요하므로 site != "none" 로 변환.

    Fallback:
      sequence 없을 때 n_strong >= 1 (기존 로직 유지, 하위 호환).

    문헌: Krenning 1992 Lancet; de Jong 2002 J Nucl Med; Maecke 2005 J Nucl Med
    """
    seq: str = (candidate.get("sequence") or "").upper().strip()

    if seq:
        try:
            # backend.pharmacophore.compute_chelator_site 위임 (상세 판정)
            from backend.pharmacophore import compute_chelator_site as _compute_cs
            mod_list: Optional[List[str]] = candidate.get("mod_list")
            detail = _compute_cs(seq, mod_list=mod_list)
            if detail is None:
                return False
            return detail["site"] != "none"
        except ImportError:
            # ImportError 시 인라인 fallback (하위 호환)
            n_term_ok: bool = seq[0] != "P"
            has_lys: bool = "K" in seq
            cys_count: int = seq.count("C")
            return (n_term_ok or has_lys) and cys_count >= 2
    else:
        # Fallback: sequence 없을 때 n_strong >= 1 (하위 호환)
        return _metal_n_strong(candidate) >= 1


# ─────────────────────────── cluster criteria ─────────────────────────────────


def _criteria_a(candidate: Dict[str, Any]) -> Dict[str, bool]:
    """Cluster A — High Affinity Core.

    pLDDT is optional: when ESMFold is not available (PyRosetta-only mode),
    pLDDT will be None/0.  In that case the criterion is marked True
    (not penalised) so that candidates can still qualify for Cluster A
    based on the remaining three criteria (ddG, clash, FWKT).
    """
    ddg = _safe_float(candidate.get("ddG"))
    clash = _safe_float(candidate.get("clash_score"))
    plddt = candidate.get("pLDDT")
    plddt_val = _safe_float(plddt) if plddt is not None else float("nan")
    fwkt = _fwkt_contact_maintained(candidate)

    import math

    ddg_ok = (not math.isnan(ddg)) and ddg <= -8.0
    clash_ok = (not math.isnan(clash)) and clash <= 5.0
    # pLDDT unavailable (None, 0, NaN) → skip (True)
    plddt_available = (not math.isnan(plddt_val)) and plddt_val > 0
    plddt_ok = plddt_val >= 75.0 if plddt_available else True
    fwkt_ok = fwkt

    return {
        "ddG_lte_minus8": ddg_ok,
        "clash_lte_5": clash_ok,
        "pLDDT_gte_75": plddt_ok,
        "pLDDT_available": plddt_available,  # informational flag
        "fwkt_contact": fwkt_ok,
    }


def _criteria_b(candidate: Dict[str, Any]) -> Dict[str, bool]:
    """Cluster B — Selectivity-Optimised."""
    sm = candidate.get("selectivity_margin")
    sm_val = _safe_float(sm) if sm is not None else float("nan")
    ddg = _safe_float(candidate.get("ddG"))

    import math

    sm_ok = (not math.isnan(sm_val)) and sm_val >= 3.0
    # "SSTR2 ddG low" is operationalised as ddG < -5.0 (reasonable binding)
    ddg_ok = (not math.isnan(ddg)) and ddg < -5.0

    return {
        "selectivity_margin_gte_3": sm_ok,
        "ddG_binding_present": ddg_ok,
    }


def _criteria_c(candidate: Dict[str, Any]) -> Dict[str, bool]:
    """Cluster C — Stability-Enhanced."""
    ii = _safe_float(candidate.get("instability_index"))
    blosum = _blosum62_total(candidate)
    protease_total = _protease_total(candidate)

    import math

    ii_ok = (not math.isnan(ii)) and ii < 30.0
    blosum_ok = blosum >= 0
    # "protease_sites 감소" — fewer than or equal to native SST-14 baseline
    protease_ok = protease_total <= _SST14_PROTEASE_BASELINE

    return {
        "instability_lt_30": ii_ok,
        "blosum62_nonnegative": blosum_ok,
        "protease_sites_reduced": protease_ok,
    }


def _criteria_d(candidate: Dict[str, Any]) -> Dict[str, bool]:
    """Cluster D — Radiochemistry-Optimal."""
    gravy = _safe_float(candidate.get("gravy"))
    charge = _safe_float(candidate.get("net_charge_ph74"))

    import math

    gravy_ok = (not math.isnan(gravy)) and -1.0 <= gravy <= 0.5
    charge_ok = (not math.isnan(charge)) and abs(charge) <= 1.0
    # 수정 (P2-1): sequence-based 로직 우선 (N-term + Lys ε-NH₂), n_strong fallback
    # 기존: chelator_ok = _metal_n_strong(candidate) >= 1  ← SS-bond Cys 오포함
    chelator_ok = _chelator_site_from_candidate(candidate)

    return {
        "gravy_in_range": gravy_ok,
        "net_charge_low": charge_ok,
        "chelator_site_available": chelator_ok,
    }


# ─────────────────────────── public API ───────────────────────────────────────


def classify_cluster(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Classify a single candidate into cluster A–E.

    Cluster priority: A > B > C > D > E.  The first cluster whose *all*
    defining criteria are met wins.  Cluster E is the fallback.

    Parameters
    ----------
    candidate : dict
        Must contain the keys produced by runner.py + pharmacology module:
        ddG, clash_score, pLDDT (optional), structural_rules, instability_index,
        blosum62, protease_sites, gravy, net_charge_ph74, metal_coordination,
        selectivity_margin (optional).

    Returns
    -------
    dict
        cluster, cluster_name, priority, criteria_met, note.
    """
    crit_a = _criteria_a(candidate)
    if all(crit_a.values()):
        return {
            "cluster": "A",
            "cluster_name": _CLUSTER_NAMES["A"],
            "priority": _CLUSTER_PRIORITY["A"],
            "criteria_met": {"A": crit_a},
            "note": (
                "All four A-criteria satisfied: strong ddG, low clash, "
                "high pLDDT, FWKT contact maintained."
            ),
        }

    crit_b = _criteria_b(candidate)
    if all(crit_b.values()):
        return {
            "cluster": "B",
            "cluster_name": _CLUSTER_NAMES["B"],
            "priority": _CLUSTER_PRIORITY["B"],
            "criteria_met": {"A": crit_a, "B": crit_b},
            "note": (
                "Selectivity margin ≥ 3.0 with confirmed SSTR2 binding; "
                "prioritised for isoform selectivity profiling."
            ),
        }

    crit_c = _criteria_c(candidate)
    if all(crit_c.values()):
        return {
            "cluster": "C",
            "cluster_name": _CLUSTER_NAMES["C"],
            "priority": _CLUSTER_PRIORITY["C"],
            "criteria_met": {"A": crit_a, "B": crit_b, "C": crit_c},
            "note": (
                "Low instability index, conservative mutations, and reduced "
                "protease sites indicate enhanced in vivo stability."
            ),
        }

    crit_d = _criteria_d(candidate)
    if all(crit_d.values()):
        return {
            "cluster": "D",
            "cluster_name": _CLUSTER_NAMES["D"],
            "priority": _CLUSTER_PRIORITY["D"],
            "criteria_met": {
                "A": crit_a, "B": crit_b, "C": crit_c, "D": crit_d,
            },
            "note": (
                "GRAVY and charge in optimal range for 68Ga/177Lu labelling; "
                "chelator attachment site confirmed."
            ),
        }

    return {
        "cluster": "E",
        "cluster_name": _CLUSTER_NAMES["E"],
        "priority": _CLUSTER_PRIORITY["E"],
        "criteria_met": {
            "A": crit_a, "B": crit_b, "C": crit_c, "D": crit_d,
        },
        "note": (
            "Does not meet A–D criteria; includes non-conservative "
            "substitutions or Tier 3 exploratory candidates."
        ),
    }


def batch_classify(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Classify a list of candidates and return per-candidate results + stats.

    Parameters
    ----------
    candidates : list of dict
        Each dict is a candidate in the format accepted by :func:`classify_cluster`.
        An optional ``"name"`` or ``"sequence"`` key is used as the identifier
        in the output; otherwise the list index is used.

    Returns
    -------
    dict
        ``results`` — list of (identifier, classify_cluster output) pairs.
        ``statistics`` — cluster distribution counts and percentages.
        ``cluster_groups`` — dict mapping cluster letter to list of identifiers.
    """
    results: List[Dict[str, Any]] = []
    cluster_groups: Dict[str, List[str]] = {c: [] for c in "ABCDE"}

    for idx, cand in enumerate(candidates):
        name = str(cand.get("name", cand.get("sequence", f"candidate_{idx}")))
        classification = classify_cluster(cand)
        cluster = classification["cluster"]
        cluster_groups[cluster].append(name)
        results.append({"id": name, "classification": classification})

    total = len(candidates)
    statistics: Dict[str, Any] = {"total": total, "distribution": {}}
    for cluster in "ABCDE":
        count = len(cluster_groups[cluster])
        pct = round(100.0 * count / total, 1) if total > 0 else 0.0
        statistics["distribution"][cluster] = {
            "count": count,
            "percent": pct,
            "name": _CLUSTER_NAMES[cluster],
        }

    return {
        "results": results,
        "statistics": statistics,
        "cluster_groups": cluster_groups,
    }
