"""
Cluster Classification 단위 테스트
====================================
대상 모듈: pyrosetta_flow/cluster_report.py

A~E 각 클러스터 mock candidate 테스트 + edge cases + batch_classify 통계 검증.
"""

from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import pytest

from pyrosetta_flow.cluster_report import classify_cluster, batch_classify


# ─── mock candidate builders ──────────────────────────────────────────────────

def _structural_rules_fwkt_pass() -> dict:
    return {
        "rules": {
            "fwkt_pharmacophore": {"pass": True, "detail": "positions 7-10 = FWKT"},
            "k9_salt_bridge": {"pass": True, "detail": "position 9 = K"},
            "cys3_cys14_disulfide": {"pass": True, "detail": "pos3=C, pos14=C"},
            "phe6_phe11_stacking": {"pass": True, "detail": "pos6=F, pos11=F"},
            "nterm_chelator": {"pass": True, "detail": "pos1=A (preferred)"},
        },
        "all_pass": True,
    }


def _structural_rules_fwkt_fail() -> dict:
    return {
        "rules": {
            "fwkt_pharmacophore": {"pass": False, "detail": "positions 7-10 = FAAT"},
        },
        "all_pass": False,
    }


def _blosum62_positive() -> dict:
    """BLOSUM62 total_score > 0 (backend pharmacology.py 포맷)."""
    return {"total_blosum62_score": 5, "n_mutations": 2}


def _blosum62_negative() -> dict:
    return {"total_blosum62_score": -8, "n_mutations": 4}


def _metal_coordination_strong() -> dict:
    return {"n_strong": 2, "total_count": 3, "chelator_interference_risk": "high"}


def _metal_coordination_none() -> dict:
    return {"n_strong": 0, "total_count": 0, "chelator_interference_risk": "low"}


def _protease_low() -> dict:
    return {"total_sites": 5, "chymotrypsin": {"count": 3}, "trypsin": {"count": 2}}


def _protease_high() -> dict:
    return {"total_sites": 15}


# ── Cluster A mock candidate ──────────────────────────────────────────────────

_CANDIDATE_A: dict = {
    "ddG": -9.5,
    "clash_score": 3,
    "pLDDT": 82.0,
    "structural_rules": _structural_rules_fwkt_pass(),
    "instability_index": 25.0,
    "blosum62": _blosum62_positive(),
    "protease_sites": _protease_low(),
    "gravy": -0.5,
    "net_charge_ph74": 0.3,
    "metal_coordination": _metal_coordination_strong(),
    "selectivity_margin": 4.0,
}

# ── Cluster B mock candidate ──────────────────────────────────────────────────
# B에는 해당하지만 A는 불통 (pLDDT 미달)

_CANDIDATE_B: dict = {
    "ddG": -6.5,
    "clash_score": 3,
    "pLDDT": 60.0,          # pLDDT < 75 → A 탈락
    "structural_rules": _structural_rules_fwkt_pass(),
    "instability_index": 25.0,
    "blosum62": _blosum62_positive(),
    "protease_sites": _protease_low(),
    "gravy": -0.5,
    "net_charge_ph74": 0.3,
    "metal_coordination": _metal_coordination_strong(),
    "selectivity_margin": 3.5,   # >= 3.0 + ddG < -5 → B 통과
}

# ── Cluster C mock candidate ──────────────────────────────────────────────────
# C에는 해당하지만 A, B 불통 (selectivity_margin 없음, pLDDT 미달)

_CANDIDATE_C: dict = {
    "ddG": -4.0,             # ddG > -5 → B 탈락
    "clash_score": 10,
    "pLDDT": 50.0,
    "structural_rules": _structural_rules_fwkt_fail(),
    "instability_index": 22.0,   # < 30 → ok
    "blosum62": _blosum62_positive(),   # >= 0 → ok
    "protease_sites": _protease_low(),  # total_sites=5 ≤ 9 → ok
    "gravy": 1.2,
    "net_charge_ph74": 3.0,
    "metal_coordination": _metal_coordination_none(),
    "selectivity_margin": None,
}

# ── Cluster D mock candidate ──────────────────────────────────────────────────
# D에는 해당하지만 A, B, C 불통

_CANDIDATE_D: dict = {
    "ddG": -3.0,
    "clash_score": 20,
    "pLDDT": 45.0,
    "structural_rules": _structural_rules_fwkt_fail(),
    "instability_index": 45.0,   # >= 30 → C 탈락
    "blosum62": _blosum62_negative(),
    "protease_sites": _protease_high(),  # total_sites=15 > 9 → C 탈락
    "gravy": -0.3,               # in [-1.0, +0.5] → D ok
    "net_charge_ph74": 0.8,      # |0.8| <= 1.0 → D ok
    "metal_coordination": _metal_coordination_strong(),  # n_strong >= 1 → D ok
    "selectivity_margin": None,
}

# ── Cluster E mock candidate ──────────────────────────────────────────────────

_CANDIDATE_E: dict = {
    "ddG": -2.0,
    "clash_score": 30,
    "pLDDT": 35.0,
    "structural_rules": _structural_rules_fwkt_fail(),
    "instability_index": 55.0,
    "blosum62": _blosum62_negative(),
    "protease_sites": _protease_high(),
    "gravy": 2.0,            # out of [-1.0, +0.5] → D 탈락
    "net_charge_ph74": 4.0,  # > 1.0 → D 탈락
    "metal_coordination": _metal_coordination_none(),
    "selectivity_margin": None,
}


# ─── cluster assignment tests ─────────────────────────────────────────────────

class TestClusterA:
    def test_basic_assignment(self):
        result = classify_cluster(_CANDIDATE_A)
        assert result["cluster"] == "A"

    def test_cluster_name(self):
        result = classify_cluster(_CANDIDATE_A)
        assert result["cluster_name"] == "High Affinity Core"

    def test_priority_is_1(self):
        result = classify_cluster(_CANDIDATE_A)
        assert result["priority"] == 1

    def test_criteria_met_present(self):
        result = classify_cluster(_CANDIDATE_A)
        assert "A" in result["criteria_met"]
        crit = result["criteria_met"]["A"]
        assert crit["ddG_lte_minus8"] is True
        assert crit["clash_lte_5"] is True
        assert crit["pLDDT_gte_75"] is True
        assert crit["fwkt_contact"] is True

    def test_note_is_string(self):
        result = classify_cluster(_CANDIDATE_A)
        assert isinstance(result["note"], str)
        assert len(result["note"]) > 0

    def test_ddg_boundary_exact(self):
        """ddG == -8.0 は A の条件境界（≤ -8.0）→ A 통과."""
        cand = dict(_CANDIDATE_A, ddG=-8.0)
        assert classify_cluster(cand)["cluster"] == "A"

    def test_ddg_just_above_boundary(self):
        """ddG == -7.99 → A 탈락."""
        cand = dict(_CANDIDATE_A, ddG=-7.99)
        result = classify_cluster(cand)
        assert result["cluster"] != "A"

    def test_plddt_boundary(self):
        """pLDDT == 75.0 → A 통과."""
        cand = dict(_CANDIDATE_A, pLDDT=75.0)
        assert classify_cluster(cand)["cluster"] == "A"

    def test_plddt_just_below_boundary(self):
        """pLDDT == 74.9 → A 탈락."""
        cand = dict(_CANDIDATE_A, pLDDT=74.9)
        result = classify_cluster(cand)
        assert result["cluster"] != "A"

    def test_clash_boundary(self):
        """clash_score == 5 → A 통과."""
        cand = dict(_CANDIDATE_A, clash_score=5)
        assert classify_cluster(cand)["cluster"] == "A"

    def test_clash_above_boundary(self):
        """clash_score == 6 → A 탈락."""
        cand = dict(_CANDIDATE_A, clash_score=6)
        result = classify_cluster(cand)
        assert result["cluster"] != "A"


class TestClusterB:
    def test_basic_assignment(self):
        result = classify_cluster(_CANDIDATE_B)
        assert result["cluster"] == "B"

    def test_cluster_name(self):
        result = classify_cluster(_CANDIDATE_B)
        assert result["cluster_name"] == "Selectivity-Optimised"

    def test_priority_is_2(self):
        assert classify_cluster(_CANDIDATE_B)["priority"] == 2

    def test_selectivity_margin_boundary(self):
        """selectivity_margin == 3.0 → B 통과."""
        cand = dict(_CANDIDATE_B, selectivity_margin=3.0)
        assert classify_cluster(cand)["cluster"] == "B"

    def test_selectivity_margin_below_boundary(self):
        """selectivity_margin == 2.99 → B 탈락."""
        cand = dict(_CANDIDATE_B, selectivity_margin=2.99)
        result = classify_cluster(cand)
        assert result["cluster"] != "B"

    def test_no_selectivity_margin_falls_through(self):
        """selectivity_margin=None → B 탈락."""
        cand = dict(_CANDIDATE_B, selectivity_margin=None)
        result = classify_cluster(cand)
        assert result["cluster"] != "B"


class TestClusterC:
    def test_basic_assignment(self):
        result = classify_cluster(_CANDIDATE_C)
        assert result["cluster"] == "C"

    def test_cluster_name(self):
        assert classify_cluster(_CANDIDATE_C)["cluster_name"] == "Stability-Enhanced"

    def test_priority_is_3(self):
        assert classify_cluster(_CANDIDATE_C)["priority"] == 3

    def test_instability_boundary(self):
        """instability_index == 29.9 → C 통과."""
        cand = dict(_CANDIDATE_C, instability_index=29.9)
        assert classify_cluster(cand)["cluster"] == "C"

    def test_instability_at_30_fails(self):
        """instability_index == 30.0 → C 탈락 (조건: < 30)."""
        cand = dict(_CANDIDATE_C, instability_index=30.0)
        result = classify_cluster(cand)
        assert result["cluster"] != "C"

    def test_blosum62_pharma_format(self):
        """pharma_properties.py 포맷(total_score 키)도 인식."""
        cand = dict(_CANDIDATE_C)
        cand["blosum62"] = {"total_score": 3}
        assert classify_cluster(cand)["cluster"] == "C"


class TestClusterD:
    def test_basic_assignment(self):
        result = classify_cluster(_CANDIDATE_D)
        assert result["cluster"] == "D"

    def test_cluster_name(self):
        assert classify_cluster(_CANDIDATE_D)["cluster_name"] == "Radiochemistry-Optimal"

    def test_priority_is_4(self):
        assert classify_cluster(_CANDIDATE_D)["priority"] == 4

    def test_gravy_lower_boundary(self):
        """gravy == -1.0 → D 통과."""
        cand = dict(_CANDIDATE_D, gravy=-1.0)
        assert classify_cluster(cand)["cluster"] == "D"

    def test_gravy_upper_boundary(self):
        """gravy == 0.5 → D 통과."""
        cand = dict(_CANDIDATE_D, gravy=0.5)
        assert classify_cluster(cand)["cluster"] == "D"

    def test_gravy_below_lower(self):
        """gravy == -1.01 → D 탈락."""
        cand = dict(_CANDIDATE_D, gravy=-1.01)
        result = classify_cluster(cand)
        assert result["cluster"] != "D"

    def test_gravy_above_upper(self):
        """gravy == 0.51 → D 탈락."""
        cand = dict(_CANDIDATE_D, gravy=0.51)
        result = classify_cluster(cand)
        assert result["cluster"] != "D"

    def test_charge_boundary_positive(self):
        """net_charge_ph74 == 1.0 → D 통과."""
        cand = dict(_CANDIDATE_D, net_charge_ph74=1.0)
        assert classify_cluster(cand)["cluster"] == "D"

    def test_charge_boundary_negative(self):
        """net_charge_ph74 == -1.0 → D 통과."""
        cand = dict(_CANDIDATE_D, net_charge_ph74=-1.0)
        assert classify_cluster(cand)["cluster"] == "D"

    def test_charge_above_boundary(self):
        """net_charge_ph74 == 1.01 → D 탈락."""
        cand = dict(_CANDIDATE_D, net_charge_ph74=1.01)
        result = classify_cluster(cand)
        assert result["cluster"] != "D"

    def test_no_chelator_site_fails(self):
        """sequence 없을 때 n_strong == 0 → fallback → D 탈락."""
        cand = dict(_CANDIDATE_D, metal_coordination=_metal_coordination_none())
        # _CANDIDATE_D에 sequence 없음 → n_strong fallback → 0 < 1 → False
        result = classify_cluster(cand)
        assert result["cluster"] != "D"


class TestClusterE:
    def test_basic_assignment(self):
        result = classify_cluster(_CANDIDATE_E)
        assert result["cluster"] == "E"

    def test_cluster_name(self):
        assert classify_cluster(_CANDIDATE_E)["cluster_name"] == "Exploratory Candidates"

    def test_priority_is_5(self):
        assert classify_cluster(_CANDIDATE_E)["priority"] == 5

    def test_criteria_met_has_all_abcd(self):
        """E는 A~D 모두 실패 → criteria_met에 A, B, C, D 키 모두 포함."""
        result = classify_cluster(_CANDIDATE_E)
        for key in ("A", "B", "C", "D"):
            assert key in result["criteria_met"]


# ─── priority ordering tests ──────────────────────────────────────────────────

class TestPriorityOrdering:

    def test_a_beats_b_criteria(self):
        """A 기준을 모두 만족하는 후보는 B 기준도 만족해도 A로 분류."""
        # _CANDIDATE_A는 selectivity_margin=4.0이므로 B 기준도 충족
        result = classify_cluster(_CANDIDATE_A)
        assert result["cluster"] == "A"

    def test_b_beats_c(self):
        """B 기준 충족 + C 기준도 충족 → B 우선."""
        cand = dict(_CANDIDATE_B,
                    instability_index=22.0,
                    blosum62=_blosum62_positive(),
                    protease_sites=_protease_low())
        result = classify_cluster(cand)
        assert result["cluster"] == "B"

    def test_c_beats_d(self):
        """C 기준 충족 + D 기준도 충족 → C 우선."""
        cand = dict(_CANDIDATE_C,
                    gravy=-0.3,
                    net_charge_ph74=0.5,
                    metal_coordination=_metal_coordination_strong())
        result = classify_cluster(cand)
        assert result["cluster"] == "C"


# ─── edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_missing_optional_fields(self):
        """pLDDT=None, selectivity_margin=None → A, B 탈락해도 오류 없음."""
        cand = {
            "ddG": -5.0,
            "clash_score": 4,
            "pLDDT": None,
            "structural_rules": _structural_rules_fwkt_pass(),
            "instability_index": 25.0,
            "blosum62": _blosum62_positive(),
            "protease_sites": _protease_low(),
            "gravy": -0.5,
            "net_charge_ph74": 0.5,
            "metal_coordination": _metal_coordination_strong(),
            "selectivity_margin": None,
        }
        result = classify_cluster(cand)
        assert result["cluster"] in "ABCDE"

    def test_missing_all_optional(self):
        """필수 키 없음 → E 반환 (nan 처리)."""
        result = classify_cluster({})
        assert result["cluster"] == "E"

    def test_empty_candidate(self):
        result = classify_cluster({})
        assert result["cluster"] == "E"
        assert result["priority"] == 5

    def test_fwkt_flat_bool(self):
        """structural_rules.rules.fwkt_pharmacophore 이 bool일 때도 처리."""
        cand = dict(_CANDIDATE_A)
        cand["structural_rules"] = {
            "rules": {"fwkt_pharmacophore": True},
            "all_pass": True,
        }
        result = classify_cluster(cand)
        assert result["cluster"] == "A"

    def test_blosum62_missing_keys(self):
        """blosum62 = {} → 0으로 처리 → C 기준 통과 가능."""
        cand = dict(_CANDIDATE_C, blosum62={})
        # total=0 이므로 nonnegative → C 기준 통과
        result = classify_cluster(cand)
        assert result["cluster"] == "C"

    def test_return_keys_complete(self):
        """반환값에 필수 키 모두 포함."""
        for cand in [_CANDIDATE_A, _CANDIDATE_B, _CANDIDATE_C,
                     _CANDIDATE_D, _CANDIDATE_E]:
            result = classify_cluster(cand)
            for key in ("cluster", "cluster_name", "priority", "criteria_met", "note"):
                assert key in result, f"Missing key '{key}' for cluster {result.get('cluster')}"


# ─── batch_classify tests ─────────────────────────────────────────────────────

class TestBatchClassify:

    _FIVE_CANDIDATES = [
        dict(_CANDIDATE_A, name="cand_A"),
        dict(_CANDIDATE_B, name="cand_B"),
        dict(_CANDIDATE_C, name="cand_C"),
        dict(_CANDIDATE_D, name="cand_D"),
        dict(_CANDIDATE_E, name="cand_E"),
    ]

    def test_results_length(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        assert len(output["results"]) == 5

    def test_each_cluster_present(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        clusters_found = {r["classification"]["cluster"] for r in output["results"]}
        assert clusters_found == {"A", "B", "C", "D", "E"}

    def test_statistics_total(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        assert output["statistics"]["total"] == 5

    def test_statistics_distribution_counts(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        dist = output["statistics"]["distribution"]
        for c in "ABCDE":
            assert dist[c]["count"] == 1

    def test_statistics_percent(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        dist = output["statistics"]["distribution"]
        for c in "ABCDE":
            assert dist[c]["percent"] == pytest.approx(20.0, abs=0.1)

    def test_cluster_groups_keys(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        assert set(output["cluster_groups"].keys()) == set("ABCDE")

    def test_cluster_groups_membership(self):
        output = batch_classify(self._FIVE_CANDIDATES)
        groups = output["cluster_groups"]
        assert "cand_A" in groups["A"]
        assert "cand_E" in groups["E"]

    def test_empty_list(self):
        output = batch_classify([])
        assert output["statistics"]["total"] == 0
        for c in "ABCDE":
            assert output["statistics"]["distribution"][c]["count"] == 0

    def test_candidate_without_name(self):
        """name 키 없는 후보 → 인덱스 기반 ID 사용."""
        output = batch_classify([_CANDIDATE_E])
        assert len(output["results"]) == 1
        assert output["results"][0]["id"].startswith("candidate_")

    def test_sequence_as_id(self):
        """name 없고 sequence 있으면 sequence를 ID로 사용."""
        cand = dict(_CANDIDATE_E, sequence="TESTSEQ")
        cand.pop("name", None)
        output = batch_classify([cand])
        assert output["results"][0]["id"] == "TESTSEQ"


# ─── chelator_site sequence-based 로직 테스트 (P2-1 fix) ─────────────────────

# _CANDIDATE_D 기반 + sequence 추가 (D 분류 통과하는 기본 수치 유지)
_D_BASE = {
    "ddG": -3.0,
    "clash_score": 20,
    "pLDDT": 45.0,
    "structural_rules": _structural_rules_fwkt_fail(),
    "instability_index": 45.0,
    "blosum62": _blosum62_negative(),
    "protease_sites": _protease_high(),
    "gravy": -0.3,
    "net_charge_ph74": 0.8,
    "metal_coordination": _metal_coordination_strong(),
    "selectivity_margin": None,
}


class TestChelatorSiteSequenceBased:
    """_chelator_site_from_candidate() sequence-based 로직 검증.

    P2-1 fix: _criteria_d()가 n_strong 대신 sequence-based 판정을 우선 사용.
    SS-bond Cys thiol은 disulfide 상태이므로 chelation 불가.
    실제 판정 기준: N-term non-Pro (α-NH₂) OR Lys ε-NH₂.
    """

    def test_non_pro_nterm_sequence_passes_cluster_d(self):
        """Non-Pro N-term 서열 → sequence-based chelator True → D 통과."""
        cand = dict(_D_BASE, sequence="AGCKNFFWKTFTSC")  # SST-14 WT
        assert classify_cluster(cand)["cluster"] == "D"

    def test_pro_nterm_without_lys_fails_cluster_d(self):
        """Pro N-term + Lys 없음 → chelator False → D 탈락.

        기존 n_strong 로직: Cys SS-bond → n_strong=2 → True (오분류)
        수정 후 sequence-based: Pro N-term + no Lys → False ✓
        """
        # Pro N-term, K→R 치환, SS-bond Cys 보유 (n_strong이 높지만 chelation 불가)
        cand = dict(_D_BASE, sequence="PGCPNFFWRTFTSC")
        result = classify_cluster(cand)
        assert result["cluster"] != "D"
        assert result["criteria_met"]["D"]["chelator_site_available"] is False

    def test_pro_nterm_with_lys_passes_cluster_d(self):
        """Pro N-term이지만 Lys 존재 → Lys ε-NH₂ 경유 chelation 가능 → D 통과."""
        cand = dict(_D_BASE, sequence="PGCKNFFWKTFTSC")  # Pro N-term + K
        assert classify_cluster(cand)["cluster"] == "D"

    def test_lys_only_nterm_pro_passes(self):
        """Pro N-term + 내부 Lys → has_lys=True → chelator True."""
        cand = dict(_D_BASE, sequence="PGCKNFFWRTFTSC")  # Pro N-term + K(idx3)
        result = classify_cluster(cand)
        assert result["criteria_met"]["D"]["chelator_site_available"] is True

    def test_no_sequence_falls_back_to_n_strong(self):
        """sequence 키 없음 → n_strong fallback → n_strong>=1 → True."""
        cand = dict(_D_BASE)  # sequence 없음
        # metal_coordination = _metal_coordination_strong (n_strong=2)
        result = classify_cluster(cand)
        assert result["criteria_met"]["D"]["chelator_site_available"] is True

    def test_empty_sequence_falls_back_to_n_strong(self):
        """sequence = '' (빈 문자열) → n_strong fallback."""
        cand = dict(_D_BASE, sequence="")
        result = classify_cluster(cand)
        # n_strong=2 → True
        assert result["criteria_met"]["D"]["chelator_site_available"] is True

    def test_sst14_wt_chelator_via_sequence(self):
        """SST-14 WT: A N-term + K4, K8 → True (sequence-based)."""
        cand = dict(_D_BASE, sequence="AGCKNFFWKTFTSC")
        result = classify_cluster(cand)
        assert result["criteria_met"]["D"]["chelator_site_available"] is True

    def test_pro_nterm_no_lys_no_chelator_criteria_d_field(self):
        """criteria_met['D']['chelator_site_available'] == False 확인."""
        cand = dict(_D_BASE, sequence="PGCPNFFWRTFTSC", gravy=-0.3,
                    net_charge_ph74=0.8)
        result = classify_cluster(cand)
        assert "D" in result["criteria_met"]
        assert result["criteria_met"]["D"]["chelator_site_available"] is False
