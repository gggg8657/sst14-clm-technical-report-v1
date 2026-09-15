"""
test_monitoring_key_contacts.py
build_monitoring_data._compute_key_contacts / _merge_key_contacts 단위 테스트.

D2 key-contact 5 Boolean 서열 기반 판정 로직 검증.
서열 위치 기반 proxy(원자거리 실측 아님) — pharma_properties.check_structural_rules와 동일 규칙.
"""
import importlib.util
import sys
import os
import pytest
from typing import Optional, Dict

# build_monitoring_data는 scripts/ 아래 있으므로 경로 조작 없이 spec으로 로드
_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../../../..",  # REPO root
    "scripts/build_monitoring_data.py",
)
_SCRIPT_PATH = os.path.normpath(_SCRIPT_PATH)


@pytest.fixture(scope="module")
def bmd():
    """build_monitoring_data 모듈 로드."""
    spec = importlib.util.spec_from_file_location("build_monitoring_data", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── _compute_key_contacts ──────────────────────────────────────────────────────

class TestComputeKeyContacts:

    def test_sst14_native_all_pass(self, bmd):
        """SST-14 원형 AGCKNFFWKTFTSC — FWKT/SS/K9/Phe/Chel 모두 True."""
        r = bmd._compute_key_contacts("AGCKNFFWKTFTSC")
        assert r["fwkt_conserved"] is True
        assert r["ss_bond_intact"] is True
        assert r["k9_d122_salt"] is True
        assert r["phe_stacking"] is True   # F(pos6), F(pos11)
        assert r["chelator_ready"] is True  # A(pos1)

    def test_fwkt_mutated(self, bmd):
        """pos7-10이 FWKT 아닌 경우 fwkt_conserved=False."""
        # F11D 변이 (AGCKNFFWKTDTSC) — FWKT는 pos7-10이므로 영향없음
        r = bmd._compute_key_contacts("AGCKNFFWKTDTSC")
        assert r["fwkt_conserved"] is True
        # FWKT를 직접 깨는 변이
        r2 = bmd._compute_key_contacts("AGCKNFFWKAXTSC")  # pos10 T→X 아님; 실제 K→A
        # pos9=K10=T(원형) → 'AGCKNFFWKAXTSC' : pos7-10 = FWKA
        seq = "AGCKNFFWKAFTSC"  # pos10 T→A → FWKA
        r3 = bmd._compute_key_contacts(seq)
        assert r3["fwkt_conserved"] is False, f"expected False, got {r3['fwkt_conserved']}"

    def test_phe_stacking_false_when_pos11_nonaromatic(self, bmd):
        """pos11이 비방향족(D)이면 phe_stacking=False."""
        r = bmd._compute_key_contacts("AGCKNFFWKTDTSC")
        assert r["phe_stacking"] is False  # D at pos11

    def test_phe_stacking_true_with_tryptophan(self, bmd):
        """pos6/pos11에 W(방향족)이 있어도 stacking True."""
        r = bmd._compute_key_contacts("AGCKNWFWKTWXSC")  # pos11 W
        # 14자리 서열로 맞추기
        r = bmd._compute_key_contacts("AGCKNWFWKTWCSC")
        assert r["phe_stacking"] is True  # W(pos6), W(pos11)

    def test_ss_bond_missing_cys14(self, bmd):
        """pos14가 C 아닌 경우 ss_bond_intact=False."""
        r = bmd._compute_key_contacts("AGCKNFFWKTFTSA")  # A at pos14
        assert r["ss_bond_intact"] is False

    def test_k9_salt_false(self, bmd):
        """pos9가 K 아닌 경우 k9_d122_salt=False."""
        r = bmd._compute_key_contacts("AGCKNFFWRTFTSC")  # R at pos9
        assert r["k9_d122_salt"] is False

    def test_chelator_ready_false_when_pos1_not_ag(self, bmd):
        """pos1이 A/G 아닌 경우 chelator_ready=False."""
        r = bmd._compute_key_contacts("TGCKNFFWKTFTSC")  # T at pos1
        assert r["chelator_ready"] is False

    def test_chelator_ready_true_for_g(self, bmd):
        """pos1이 G이면 chelator_ready=True."""
        r = bmd._compute_key_contacts("GGCKNFFWKTFTSC")
        assert r["chelator_ready"] is True

    def test_short_sequence_returns_none(self, bmd):
        """4잔기 이하 서열은 계산 불가 필드 None 반환."""
        r = bmd._compute_key_contacts("AGCK")
        assert r["fwkt_conserved"] is None   # n<10
        assert r["ss_bond_intact"] is None   # n<14
        # n>=4이므로 chelator_ready는 판정 가능
        assert r["chelator_ready"] is True   # A(pos1)

    def test_none_input_all_none(self, bmd):
        """None 입력 → 전 필드 None."""
        r = bmd._compute_key_contacts(None)
        assert all(v is None for v in r.values())

    def test_empty_string_all_none(self, bmd):
        """빈 문자열 입력 → 전 필드 None."""
        r = bmd._compute_key_contacts("")
        assert all(v is None for v in r.values())

    def test_lowercase_input_normalized(self, bmd):
        """소문자 입력도 upper() 처리돼 동일 결과."""
        r_lower = bmd._compute_key_contacts("agcknffwktftsc")
        r_upper = bmd._compute_key_contacts("AGCKNFFWKTFTSC")
        assert r_lower == r_upper

    def test_returns_all_five_keys(self, bmd):
        """반환 dict가 정확히 5개 키를 포함."""
        r = bmd._compute_key_contacts("AGCKNFFWKTFTSC")
        expected = {"fwkt_conserved", "ss_bond_intact", "k9_d122_salt",
                    "phe_stacking", "chelator_ready"}
        assert set(r.keys()) == expected


# ── _merge_key_contacts ────────────────────────────────────────────────────────

class TestMergeKeyContacts:

    def test_fills_missing_fields(self, bmd):
        """entry에 필드 없으면 kc 값으로 채운다."""
        entry: dict = {"sequence": "AGCKNFFWKTFTSC"}
        kc = {
            "fwkt_conserved": True, "ss_bond_intact": True,
            "k9_d122_salt": True, "phe_stacking": True, "chelator_ready": True,
        }
        bmd._merge_key_contacts(entry, kc)
        assert entry["fwkt_conserved"] is True
        assert entry["ss_bond_intact"] is True

    def test_preserves_existing_non_none_value(self, bmd):
        """기존 False 값을 True로 덮어쓰지 않는다."""
        entry: dict = {"fwkt_conserved": False}
        kc = {"fwkt_conserved": True, "ss_bond_intact": True,
              "k9_d122_salt": None, "phe_stacking": None, "chelator_ready": None}
        bmd._merge_key_contacts(entry, kc)
        assert entry["fwkt_conserved"] is False   # 기존 False 유지
        assert entry["ss_bond_intact"] is True     # None → True 채워짐

    def test_fills_when_existing_is_none(self, bmd):
        """기존 값이 None이면 kc 값으로 교체."""
        entry: dict = {"fwkt_conserved": None}
        kc = {"fwkt_conserved": True, "ss_bond_intact": None,
              "k9_d122_salt": None, "phe_stacking": None, "chelator_ready": None}
        bmd._merge_key_contacts(entry, kc)
        assert entry["fwkt_conserved"] is True
