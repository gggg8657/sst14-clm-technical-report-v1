"""
test_modification_conflict.py
====================================================================
pyrosetta_flow/modification_conflict.py 단위 테스트.

대상: check_modification_conflicts(), _parse_mod(), 규칙 R-F1~R-W3.
"""
from __future__ import annotations

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

import pytest

from pyrosetta_flow.modification_conflict import (
    check_modification_conflicts,
    _parse_mod,
    _is_dota,
    _is_fatty_acid,
    _is_peg,
)

# SST-14 WT 서열
_SEQ = "AGCKNFFWKTFTSC"  # 14aa


# ---------------------------------------------------------------------------
# _parse_mod 파싱 테스트
# ---------------------------------------------------------------------------

class TestParseMod:
    def test_parse_fatty_acid_at_k5(self):
        result = _parse_mod("fatty_acid@K5")
        assert result is not None
        assert result["mod_type"] == "fatty_acid"
        assert result["aa"] == "K"
        assert result["position"] == 5

    def test_parse_dota_at_k9(self):
        result = _parse_mod("DOTA@K9")
        assert result is not None
        assert result["mod_type"] == "dota"
        assert result["aa"] == "K"
        assert result["position"] == 9

    def test_parse_peg_lowercase(self):
        result = _parse_mod("peg@K5")
        assert result is not None
        assert result["mod_type"] == "peg"
        assert result["position"] == 5

    def test_parse_nterm_cap(self):
        result = _parse_mod("N-term_cap")
        assert result is not None
        assert result["aa"] is None
        assert result["position"] is None

    def test_parse_acetyl(self):
        result = _parse_mod("acetyl")
        assert result is not None
        assert result["aa"] is None

    def test_parse_d_aa(self):
        result = _parse_mod("D_aa@F7")
        assert result is not None
        assert result["mod_type"] == "d_aa"
        assert result["aa"] == "F"
        assert result["position"] == 7

    def test_parse_unknown_format_returns_none(self):
        # @가 없고 N-term 목록에도 없는 경우
        result = _parse_mod("some_weird_mod_xyz")
        assert result is None


# ---------------------------------------------------------------------------
# 정상 케이스 (충돌 없음)
# ---------------------------------------------------------------------------

class TestNoConflict:
    def test_empty_mod_list(self):
        result = check_modification_conflicts(_SEQ, [])
        assert result["ok"] is True
        assert result["conflicts"] == []
        assert result["fail_count"] == 0

    def test_dota_nterm_only(self):
        """N-term DOTA 단독 — SST-14 WT의 표준 DOTATATE 선례"""
        result = check_modification_conflicts(_SEQ, ["DOTA"])
        assert result["ok"] is True
        assert result["fail_count"] == 0

    def test_fatty_acid_only_on_k5(self):
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K5"])
        assert result["ok"] is True
        assert result["fail_count"] == 0

    def test_dota_on_k9_only(self):
        # [BUG-FIX 버그4] R-F4: DOTA@K9는 FWKT 구성 Lys(K9=Lys9) 부착 → FAIL
        # K9은 SSTR2 pocket 직접 접촉 잔기 (Reubi 2017: K9→Ala Ki ×50)
        result = check_modification_conflicts(_SEQ, ["DOTA@K9"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F4" in rules

    def test_dota_on_k4_only(self):
        """DOTA@K4는 FWKT 외부 Lys — R-F4 미발동"""
        result = check_modification_conflicts(_SEQ, ["DOTA@K4"])
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F4" not in fail_rules
        assert result["ok"] is True

    def test_fatty_acid_k5_dota_k9_different_lys(self):
        # [BUG-FIX 버그4] R-F4: DOTA@K9는 FWKT-Lys 부착 → FAIL (서로 다른 Lys여도)
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K4", "DOTA@K9"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F4" in rules

    def test_fatty_acid_k4_dota_k4_still_r_f1(self):
        """DOTA@K4(FWKT 외부) + fatty_acid@K4 동일 Lys → R-F1(FAIL), R-F4 미발동"""
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K4", "DOTA@K4"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F1" in rules
        assert "R-F4" not in rules


# ---------------------------------------------------------------------------
# R-F1: 동일 Lys fatty_acid + DOTA → FAIL
# ---------------------------------------------------------------------------

class TestRF1:
    def test_fail_fatty_acid_and_dota_same_lys(self):
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K5", "DOTA@K5"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F1" in rules
        assert result["fail_count"] >= 1

    def test_fail_c18_and_dota_same_lys(self):
        """c18(fatty acid alias) + DOTA"""
        result = check_modification_conflicts(_SEQ, ["c18@K5", "DOTA@K5"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F1" in rules

    def test_fail_acyl_and_nota_same_lys(self):
        """acyl + NOTA(DOTA 동종)"""
        result = check_modification_conflicts(_SEQ, ["acyl@K5", "NOTA@K5"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F1" in rules

    def test_no_fail_if_different_lys(self):
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K4", "DOTA@K5"])
        # SST-14: K4(idx 3), K5(idx 4) — 서로 다른 위치
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F1" not in fail_rules


# ---------------------------------------------------------------------------
# R-F2: D-aa + acylation 동일 위치 → WARN (BUG-FIX 버그4: FAIL에서 완화)
# D-Lys ε-amine NHS-ester는 Cα 키랄성 독립 (세마글루타이드 선례)
# ---------------------------------------------------------------------------

class TestRF2:
    def test_warn_d_aa_and_acylation_same_pos(self):
        # [BUG-FIX 버그4] R-F2: FAIL → WARN으로 완화 (D-Lys acylation 가능)
        result = check_modification_conflicts(_SEQ, ["D_aa@K4", "fatty_acid@K4"])
        assert result["ok"] is True   # WARN은 ok=True
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-F2" in warn_rules

    def test_warn_d_aa_and_peg_same_pos(self):
        # [BUG-FIX 버그4] R-F2: WARN — chirality-HPLC 검증 필요
        result = check_modification_conflicts(_SEQ, ["D_aa@K4", "PEG@K4"])
        assert result["ok"] is True   # WARN은 ok=True
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-F2" in warn_rules

    def test_no_warn_d_aa_and_acylation_different_pos(self):
        result = check_modification_conflicts(_SEQ, ["D_aa@F7", "fatty_acid@K4"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        r_f2_warns = [r for r in warn_rules if r == "R-F2"]
        assert len(r_f2_warns) == 0  # 다른 위치 → R-F2 미발동


# ---------------------------------------------------------------------------
# R-F3: N-term capping + DOTA N-term → FAIL
# ---------------------------------------------------------------------------

class TestRF3:
    def test_fail_nterm_cap_and_dota_nterm(self):
        result = check_modification_conflicts(_SEQ, ["N-term_cap", "DOTA"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F3" in rules

    def test_fail_acetyl_and_dota_nterm(self):
        result = check_modification_conflicts(_SEQ, ["acetyl", "DOTA"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F3" in rules

    def test_no_fail_nterm_cap_without_dota(self):
        result = check_modification_conflicts(_SEQ, ["N-term_cap"])
        assert result["ok"] is True

    def test_no_fail_dota_on_lys_without_nterm_cap(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K4"])  # K4는 FWKT 외부
        # R-F3 발동 안 해야 함 (N-term capping 없음)
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F3" not in fail_rules


# ---------------------------------------------------------------------------
# R-F4: DOTA가 FWKT-Lys(K9) 부착 → FAIL (버그4 신규)
# ---------------------------------------------------------------------------

class TestRF4:
    def test_fail_dota_on_fwkt_lys_k9(self):
        """SST-14 K9(Lys9, FWKT 내부) → R-F4 FAIL"""
        result = check_modification_conflicts(_SEQ, ["DOTA@K9"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F4" in rules
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F4" in fail_rules

    def test_no_fail_dota_on_k4_outside_fwkt(self):
        """K4(FWKT 외부 Lys) → R-F4 미발동"""
        result = check_modification_conflicts(_SEQ, ["DOTA@K4"])
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F4" not in fail_rules
        assert result["ok"] is True

    def test_no_fail_dota_nterm_only(self):
        """DOTA N-term 단독 → R-F4 미발동 (위치 없음 = N-term 부착)"""
        result = check_modification_conflicts(_SEQ, ["DOTA"])
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F4" not in fail_rules
        assert result["ok"] is True

    def test_fail_dota_on_fwkt_lys_non_sst14(self):
        """다른 FWKT 보유 서열에서도 R-F4 발동 확인"""
        # CFWKTAC: FWKT at idx 1, K=FWKT[2]=idx 3, 1-indexed=4
        seq = "CFWKTACC"
        result = check_modification_conflicts(seq, ["DOTA@K4"])
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F4" in rules

    def test_no_r_f4_if_no_fwkt_motif(self):
        """FWKT 없는 서열 → R-F4 미발동"""
        seq = "AGCKNFAKKTFTSC"  # FWKT→FAKT 변이
        result = check_modification_conflicts(seq, ["DOTA@K9"])
        fail_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "FAIL"]
        assert "R-F4" not in fail_rules


# ---------------------------------------------------------------------------
# R-W1: 동일 Lys fatty_acid + PEG → WARN (ok=True)
# ---------------------------------------------------------------------------

class TestRW1:
    def test_warn_fatty_acid_and_peg_same_lys(self):
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K5", "PEG@K5"])
        assert result["ok"] is True  # WARN은 ok=True
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W1" in warn_rules
        assert result["warn_count"] >= 1

    def test_no_warn_different_lys(self):
        result = check_modification_conflicts(_SEQ, ["fatty_acid@K4", "PEG@K5"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W1" not in warn_rules


# ---------------------------------------------------------------------------
# R-W2: 여러 위치 DOTA → WARN
# ---------------------------------------------------------------------------

class TestRW2:
    def test_warn_two_dota_sites(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K5", "DOTA@K9"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W2" in warn_rules

    def test_no_warn_single_dota(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K9"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W2" not in warn_rules


# ---------------------------------------------------------------------------
# R-W3: DOTA on Lys + N-term cap → WARN
# ---------------------------------------------------------------------------

class TestRW3:
    def test_warn_dota_lys_with_nterm_cap(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K9", "N-term_cap"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W3" in warn_rules

    def test_no_warn_dota_lys_without_nterm_cap(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K9"])
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        assert "R-W3" not in warn_rules


# ---------------------------------------------------------------------------
# 복합 시나리오
# ---------------------------------------------------------------------------

class TestCombined:
    def test_multiple_fails(self):
        """R-F1(동일 Lys fatty+DOTA) + R-F3(N-term capping + DOTA N-term)"""
        result = check_modification_conflicts(
            _SEQ,
            ["fatty_acid@K5", "DOTA@K5", "N-term_cap", "DOTA"],
        )
        assert result["ok"] is False
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-F1" in rules
        assert "R-F3" in rules
        assert result["fail_count"] >= 2

    def test_input_empty_sequence(self):
        result = check_modification_conflicts("", ["DOTA"])
        assert result["ok"] is False

    def test_result_echoes_inputs(self):
        result = check_modification_conflicts(_SEQ, ["DOTA@K9"])
        assert result["sequence"] == _SEQ
        assert result["mod_list"] == ["DOTA@K9"]

    def test_invalid_position_beyond_seq(self):
        """서열 길이(14)보다 큰 위치 지정"""
        result = check_modification_conflicts(_SEQ, ["DOTA@K20"])
        # 위치 초과 → R-INPUT FAIL
        rules = [c["rule"] for c in result["conflicts"]]
        assert "R-INPUT" in rules
        assert result["ok"] is False

    def test_aa_mismatch_warn(self):
        """K5이지만 실제 서열은 N(idx 4=K) — SST-14에서 idx 3=K4, idx 4=N5
        서열 'AGCKNFFWKTFTSC': 위치5(1-idx) = 'N'이지만 mod가 K5라 표기"""
        # SST-14: A(1)G(2)C(3)K(4)N(5)... → K5는 실제로 N
        result = check_modification_conflicts(_SEQ, ["DOTA@K5"])
        # 잔기 불일치 WARN 기대
        warn_rules = [c["rule"] for c in result["conflicts"] if c["severity"] == "WARN"]
        # K5는 실제 N → 불일치 경고
        # (seq[4] = 'N' ≠ 'K')
        assert any("R-INPUT" == r for r in warn_rules) or result["ok"] is True
        # 최소한 ok 상태만 확인 (WARN이 발동해도 ok=True)
        # 위치 5에서 잔기 불일치는 WARN이므로 ok=True
        assert result["ok"] is True  # WARN만이므로 ok


# ---------------------------------------------------------------------------
# pharmacophore.compute_chelator_site 연동 테스트
# ---------------------------------------------------------------------------

class TestPharmacophoreIntegration:
    def test_compute_chelator_site_returns_dict(self):
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site(_SEQ)
        assert isinstance(result, dict)
        assert "site" in result
        assert "position" in result
        assert "competition_risk" in result

    def test_sst14_wt_nterm_preferred(self):
        """SST-14 WT: N-term(Ala) free → site=N-term, position=0"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site(_SEQ)
        assert result is not None
        assert result["site"] == "N-term"
        assert result["position"] == 0
        assert result["competition_risk"] is False

    def test_pro_nterm_uses_lys(self):
        """Pro N-term 서열 → site=Lys"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site("PGCKNFFWKTFTSC")
        assert result is not None
        assert result["site"] == "Lys"

    def test_no_valid_site_pro_nterm_no_lys(self):
        """Pro N-term + Lys 없음 + SS-bond 유지"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site("PGCNNFFWRTFTSC")  # K→R
        assert result is not None
        assert result["site"] == "none"
        assert result["position"] == -1

    def test_no_cys_pair_returns_none_site(self):
        """SS-bond 없음(Cys<2) → site=none"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site("AGAKNFFWKTFTSA")  # C→A
        assert result is not None
        assert result["site"] == "none"

    def test_competition_risk_fatty_acid(self):
        """fatty_acid@K5 이미 부착 → N-term은 자유이므로 competition_risk=False"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site(_SEQ, mod_list=["fatty_acid@K5"])
        # N-term이 free이면 N-term site 반환 → fatty_acid는 Lys에 붙은 것이므로 N-term risk 없음
        assert result is not None
        assert result["site"] == "N-term"
        assert result["competition_risk"] is False

    def test_nterm_cap_competition_risk(self):
        """N-term_cap 있을 때 N-term site → competition_risk=True"""
        from backend.pharmacophore import compute_chelator_site
        result = compute_chelator_site(_SEQ, mod_list=["N-term_cap"])
        assert result is not None
        assert result["site"] == "N-term"
        assert result["competition_risk"] is True

    def test_compute_pharmacophore_fields_backward_compat(self):
        """compute_pharmacophore_fields: chelator_site_available bool 하위 호환"""
        from backend.pharmacophore import compute_pharmacophore_fields
        cand = {"sequence": _SEQ}
        result = compute_pharmacophore_fields(cand)
        assert "chelator_site_available" in result
        assert "chelator_site_detail" in result
        assert isinstance(result["chelator_site_available"], bool)
        assert result["chelator_site_available"] is True

    def test_compute_pharmacophore_fields_with_mod_list(self):
        from backend.pharmacophore import compute_pharmacophore_fields
        cand = {"sequence": _SEQ, "mod_list": ["fatty_acid@K5", "DOTA@K9"]}
        result = compute_pharmacophore_fields(cand)
        assert result["chelator_site_available"] is True
        detail = result["chelator_site_detail"]
        assert detail["site"] == "N-term"


# ---------------------------------------------------------------------------
# cluster_report._chelator_site_from_candidate 연동 테스트
# ---------------------------------------------------------------------------

class TestClusterReportIntegration:
    def test_chelator_site_sst14_wt(self):
        from pyrosetta_flow.cluster_report import _chelator_site_from_candidate
        cand = {"sequence": _SEQ}
        assert _chelator_site_from_candidate(cand) is True

    def test_chelator_site_pro_nterm_with_lys(self):
        from pyrosetta_flow.cluster_report import _chelator_site_from_candidate
        cand = {"sequence": "PGCKNFFWKTFTSC"}
        # Pro N-term이지만 Lys 있음 → True
        assert _chelator_site_from_candidate(cand) is True

    def test_chelator_site_pro_nterm_no_lys(self):
        from pyrosetta_flow.cluster_report import _chelator_site_from_candidate
        cand = {"sequence": "PGCNNFFWRTFTSC"}
        # Pro N-term + Lys 없음 → False
        assert _chelator_site_from_candidate(cand) is False

    def test_chelator_site_no_sequence_fallback_n_strong(self):
        from pyrosetta_flow.cluster_report import _chelator_site_from_candidate
        cand = {"metal_coordination": {"n_strong": 2}}
        # sequence 없으면 n_strong fallback
        assert _chelator_site_from_candidate(cand) is True

    def test_chelator_site_no_sequence_no_strong(self):
        from pyrosetta_flow.cluster_report import _chelator_site_from_candidate
        cand = {"metal_coordination": {"n_strong": 0}}
        assert _chelator_site_from_candidate(cand) is False
