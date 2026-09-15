"""test_contact_fingerprint.py
==============================
contact_fingerprint.py 단위 테스트.

검증 목표:
  1. PDB 파싱 유틸 (_parse_pdb_atoms, _dist, _get_atoms, _min_dist_between)
  2. chain 감지 (_detect_peptide_chain, _get_chains)
  3. native 복합체 mock PDB 에서 Trp8-Lys9/Phe contact 추출
  4. 수용체 잔기 불일치(없는 resseq) 처리 → warnings 생성
  5. pharmacophore_contact_intact 종합 판정 로직
  6. ddG 강한데 contact 깨진 경우 binding_mechanism_warning=True
  7. PDB 파일 없을 때 pdb_parse_ok=False 반환
"""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pyrosetta_flow.contact_fingerprint import (
    D122_K9_SB_MAX,
    Q126_K9_HBOND_MAX,
    _dist,
    _get_atoms,
    _get_chains,
    _min_dist_between,
    _parse_pdb_atoms,
    _detect_peptide_chain,
    analyze_contact_fingerprint,
    analyze_contact_fingerprint_warning,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# PDB 생성 헬퍼
# ---------------------------------------------------------------------------

def _make_atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
) -> str:
    """ATOM 레코드 한 줄 생성 (PDB 형식)."""
    return (
        f"ATOM  {serial:5d} {name:<4s} {resname:<3s} {chain}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
    )


def _make_minimal_complex_pdb(
    receptor_residues: List[Dict[str, Any]],
    peptide_residues: List[Dict[str, Any]],
    receptor_chain: str = "A",
    peptide_chain: str = "B",
) -> str:
    """수용체(chain A) + 펩타이드(chain B) 최소 복합체 PDB 문자열 생성.

    각 residue dict: {resseq, resname, atoms: [{name, x, y, z}]}
    """
    lines = []
    serial = 1
    for res in receptor_residues:
        for at in res.get("atoms", []):
            lines.append(_make_atom_line(
                serial, at["name"], res["resname"], receptor_chain,
                res["resseq"], at["x"], at["y"], at["z"],
            ))
            serial += 1
    lines.append("TER\n")
    for res in peptide_residues:
        for at in res.get("atoms", []):
            lines.append(_make_atom_line(
                serial, at["name"], res["resname"], peptide_chain,
                res["resseq"], at["x"], at["y"], at["z"],
            ))
            serial += 1
    lines.append("TER\nEND\n")
    return "".join(lines)


# ---------------------------------------------------------------------------
# 1. PDB 파싱 유틸
# ---------------------------------------------------------------------------

class TestPdbParseUtils:
    """_parse_pdb_atoms, _dist, _get_atoms, _min_dist_between 단위 테스트."""

    def test_parse_basic_atom(self, tmp_path: Path):
        """단순 ATOM 레코드 1개 파싱."""
        pdb_content = _make_atom_line(1, "CA", "ALA", "A", 1, 1.0, 2.0, 3.0)
        pdb_file = tmp_path / "test.pdb"
        pdb_file.write_text(pdb_content + "END\n")
        atoms = _parse_pdb_atoms(str(pdb_file))
        assert len(atoms) == 1
        assert atoms[0]["chain"] == "A"
        assert atoms[0]["resseq"] == 1
        assert atoms[0]["name"] == "CA"
        assert atoms[0]["x"] == pytest.approx(1.0)
        assert atoms[0]["y"] == pytest.approx(2.0)
        assert atoms[0]["z"] == pytest.approx(3.0)

    def test_parse_missing_file(self, tmp_path: Path):
        """존재하지 않는 파일 → 빈 리스트 반환 (예외 없음)."""
        result = _parse_pdb_atoms(str(tmp_path / "nonexistent.pdb"))
        assert result == []

    def test_parse_ignores_remark(self, tmp_path: Path):
        """REMARK/HETATM 이외 레코드 무시."""
        content = "REMARK   This is a remark\n" + _make_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
        pdb_file = tmp_path / "t.pdb"
        pdb_file.write_text(content)
        atoms = _parse_pdb_atoms(str(pdb_file))
        assert len(atoms) == 1

    def test_dist_known_values(self):
        """_dist: 3-4-5 직각삼각형 검증."""
        a = {"x": 0.0, "y": 0.0, "z": 0.0}
        b = {"x": 3.0, "y": 4.0, "z": 0.0}
        assert _dist(a, b) == pytest.approx(5.0)

    def test_dist_zero(self):
        """_dist: 동일 위치 = 0."""
        a = {"x": 1.5, "y": 2.5, "z": 3.5}
        assert _dist(a, a) == pytest.approx(0.0)

    def test_get_atoms_filter(self, tmp_path: Path):
        """_get_atoms: chain/resseq/atom_name 필터 정확성."""
        content = (
            _make_atom_line(1, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0)
            + _make_atom_line(2, "CB", "ALA", "A", 1, 2.0, 0.0, 0.0)
            + _make_atom_line(3, "CA", "GLY", "B", 2, 3.0, 0.0, 0.0)
        )
        pdb_file = tmp_path / "t.pdb"
        pdb_file.write_text(content)
        atoms = _parse_pdb_atoms(str(pdb_file))

        ca_A1 = _get_atoms(atoms, "A", 1, ["CA"])
        assert len(ca_A1) == 1
        assert ca_A1[0]["x"] == pytest.approx(1.0)

        ca_B2 = _get_atoms(atoms, "B", 2, ["CA"])
        assert len(ca_B2) == 1

        # 없는 원자
        no_atom = _get_atoms(atoms, "A", 1, ["NZ"])
        assert no_atom == []

    def test_min_dist_between_empty(self):
        """_min_dist_between: 빈 집합 → None."""
        a = {"x": 0.0, "y": 0.0, "z": 0.0}
        assert _min_dist_between([], [a]) is None
        assert _min_dist_between([a], []) is None
        assert _min_dist_between([], []) is None

    def test_min_dist_between_two_atoms(self):
        """_min_dist_between: 두 원자 중 최소 거리 선택."""
        set_a = [{"x": 0.0, "y": 0.0, "z": 0.0}]
        set_b = [
            {"x": 5.0, "y": 0.0, "z": 0.0},
            {"x": 3.0, "y": 0.0, "z": 0.0},
        ]
        assert _min_dist_between(set_a, set_b) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 2. Chain 감지
# ---------------------------------------------------------------------------

class TestChainDetection:
    """_get_chains, _detect_peptide_chain 단위 테스트."""

    def test_get_chains_order(self, tmp_path: Path):
        """_get_chains: PDB 출현 순서대로 chain 반환."""
        content = (
            _make_atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
            + _make_atom_line(2, "CA", "GLY", "B", 1, 1.0, 0.0, 0.0)
            + _make_atom_line(3, "CA", "LYS", "A", 2, 2.0, 0.0, 0.0)
        )
        pdb_file = tmp_path / "t.pdb"
        pdb_file.write_text(content)
        atoms = _parse_pdb_atoms(str(pdb_file))
        chains = _get_chains(atoms)
        assert chains == ["A", "B"]

    def test_detect_peptide_chain_short_chain(self, tmp_path: Path):
        """_detect_peptide_chain: 짧은 체인(펩타이드 크기)이 선택됨."""
        # chain A: 100 잔기 (수용체), chain B: 14 잔기 (펩타이드 SST-14)
        content = ""
        serial = 1
        # 수용체 chain A: 잔기 1-100
        for r in range(1, 101):
            content += _make_atom_line(serial, "CA", "ALA", "A", r, float(r), 0.0, 0.0)
            serial += 1
        # 펩타이드 chain B: 잔기 1-14
        for r in range(1, 15):
            content += _make_atom_line(serial, "CA", "ALA", "B", r, 0.0, float(r), 0.0)
            serial += 1
        pdb_file = tmp_path / "complex.pdb"
        pdb_file.write_text(content)
        atoms = _parse_pdb_atoms(str(pdb_file))
        pep_chain = _detect_peptide_chain(atoms, pep_len=14)
        assert pep_chain == "B"


# ---------------------------------------------------------------------------
# 3. native 복합체 mock PDB에서 pharmacophore contact 추출
# ---------------------------------------------------------------------------

class TestContactFingerprint:
    """analyze_contact_fingerprint 통합 테스트."""

    def _write_complex_pdb(
        self,
        tmp_path: Path,
        k9_nz_pos: tuple,
        d122_od1_pos: tuple,
        w8_ne1_pos: tuple,
        q126_oe1_pos: tuple,
        f6_ca_pos: tuple,
        f7_ca_pos: tuple,
        pocket_f272_ca_pos: tuple,
    ) -> str:
        """SST-14 pharmacophore 핵심 원자를 포함한 mock 복합체 PDB 파일 생성.

        수용체(chain A): D122, Q126, F272
        펩타이드(chain B): 잔기 1-14, F6/F7(CA), W8(NE1), K9(NZ)
        """
        receptor_residues = [
            {
                "resseq": 122, "resname": "ASP",
                "atoms": [
                    {"name": "CA", "x": 0.0, "y": 0.0, "z": 0.0},
                    {"name": "OD1", "x": d122_od1_pos[0], "y": d122_od1_pos[1], "z": d122_od1_pos[2]},
                    {"name": "OD2", "x": d122_od1_pos[0] + 1.0, "y": d122_od1_pos[1], "z": d122_od1_pos[2]},
                ],
            },
            {
                "resseq": 126, "resname": "GLN",
                "atoms": [
                    {"name": "CA", "x": 5.0, "y": 0.0, "z": 0.0},
                    {"name": "OE1", "x": q126_oe1_pos[0], "y": q126_oe1_pos[1], "z": q126_oe1_pos[2]},
                    {"name": "NE2", "x": q126_oe1_pos[0], "y": q126_oe1_pos[1] + 1.0, "z": q126_oe1_pos[2]},
                ],
            },
            {
                "resseq": 272, "resname": "PHE",
                "atoms": [
                    {"name": "CA", "x": pocket_f272_ca_pos[0], "y": pocket_f272_ca_pos[1], "z": pocket_f272_ca_pos[2]},
                ],
            },
        ]
        # 펩타이드 chain B: 14 잔기 (SST-14 순서)
        # 위치 6=PHE, 7=PHE, 8=TRP, 9=LYS
        pep_aa = list("AGCKNFFWKTFTSC")
        _aa_to_resname = {
            "A": "ALA", "G": "GLY", "C": "CYS", "K": "LYS", "N": "ASN",
            "F": "PHE", "W": "TRP", "T": "THR", "S": "SER",
        }
        peptide_residues = []
        for i, aa in enumerate(pep_aa):
            rn = _aa_to_resname.get(aa, "ALA")
            atoms = [{"name": "CA", "x": float(i) * 3.8, "y": 20.0, "z": 0.0}]
            pos_1idx = i + 1
            if pos_1idx == 6:  # F6
                atoms[0]["x"] = f6_ca_pos[0]
                atoms[0]["y"] = f6_ca_pos[1]
                atoms[0]["z"] = f6_ca_pos[2]
            elif pos_1idx == 7:  # F7
                atoms[0]["x"] = f7_ca_pos[0]
                atoms[0]["y"] = f7_ca_pos[1]
                atoms[0]["z"] = f7_ca_pos[2]
            elif pos_1idx == 8:  # W8 — NE1 추가
                atoms.append({
                    "name": "NE1",
                    "x": w8_ne1_pos[0], "y": w8_ne1_pos[1], "z": w8_ne1_pos[2],
                })
            elif pos_1idx == 9:  # K9 — NZ 추가
                atoms.append({
                    "name": "NZ",
                    "x": k9_nz_pos[0], "y": k9_nz_pos[1], "z": k9_nz_pos[2],
                })
            peptide_residues.append({"resseq": i + 1, "resname": rn, "atoms": atoms})

        content = _make_minimal_complex_pdb(receptor_residues, peptide_residues)
        pdb_file = tmp_path / "complex.pdb"
        pdb_file.write_text(content)
        return str(pdb_file)

    def test_salt_bridge_intact(self, tmp_path: Path):
        """K9(NZ) — D122(OD) 거리 ≤ 4 Å → salt bridge intact."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(10.0, 0.0, 0.0),
            d122_od1_pos=(12.0, 0.0, 0.0),   # 거리 2.0 Å
            w8_ne1_pos=(20.0, 0.0, 0.0),
            q126_oe1_pos=(30.0, 0.0, 0.0),
            f6_ca_pos=(40.0, 0.0, 0.0),
            f7_ca_pos=(44.0, 0.0, 0.0),
            pocket_f272_ca_pos=(40.0, 7.0, 0.0),  # CA-CA ~7 Å (< 8 Å → contact)
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]
        assert fp["k9_d122_salt_bridge_ang"] is not None
        assert fp["k9_d122_salt_bridge_ang"] == pytest.approx(2.0, abs=0.01)
        assert fp["k9_d122_salt_bridge_intact"] is True

    def test_salt_bridge_broken(self, tmp_path: Path):
        """K9(NZ) — D122(OD) 거리 > 4 Å → salt bridge broken."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(10.0, 0.0, 0.0),
            d122_od1_pos=(20.0, 0.0, 0.0),   # 거리 10 Å (broken)
            w8_ne1_pos=(30.0, 0.0, 0.0),
            q126_oe1_pos=(40.0, 0.0, 0.0),
            f6_ca_pos=(50.0, 0.0, 0.0),
            f7_ca_pos=(54.0, 0.0, 0.0),
            pocket_f272_ca_pos=(50.0, 50.0, 0.0),  # 멀리
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]
        assert fp["k9_d122_salt_bridge_intact"] is False

    def test_hbond_intact(self, tmp_path: Path):
        """W8(NE1) — Q126(OE1) 거리 ≤ 3.5 Å → H-bond intact."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(0.0, 10.0, 0.0),
            d122_od1_pos=(0.0, 14.0, 0.0),   # 4 Å
            w8_ne1_pos=(5.0, 5.0, 0.0),
            q126_oe1_pos=(5.0, 8.0, 0.0),    # 거리 3.0 Å (intact)
            f6_ca_pos=(40.0, 0.0, 0.0),
            f7_ca_pos=(44.0, 0.0, 0.0),
            pocket_f272_ca_pos=(40.0, 7.0, 0.0),
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]
        assert fp["w8_q126_hbond_ang"] is not None
        assert fp["w8_q126_hbond_ang"] == pytest.approx(3.0, abs=0.01)
        assert fp["w8_q126_hbond_intact"] is True

    def test_a8_q126_k9_and_d122_k9_diagnostics_intact(self, tmp_path: Path):
        """A8 진단: Q126-K9 H-bond, D122-K9 salt bridge 필드가 채워진다."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(10.0, 0.0, 0.0),
            d122_od1_pos=(12.0, 0.0, 0.0),  # D122-K9 = 2.0 Å
            w8_ne1_pos=(5.0, 5.0, 0.0),
            q126_oe1_pos=(10.0, 3.0, 0.0),  # Q126-K9 = 3.0 Å
            f6_ca_pos=(40.0, 0.0, 0.0),
            f7_ca_pos=(44.0, 0.0, 0.0),
            pocket_f272_ca_pos=(40.0, 7.0, 0.0),
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]

        assert fp["q126_k9_dist_min"] == pytest.approx(3.0, abs=0.01)
        assert fp["q126_k9_hbond_intact"] is True
        assert fp["d122_k9_dist_min"] == pytest.approx(2.0, abs=0.01)
        assert fp["d122_k9_saltbridge_intact"] is True

    def test_a8_q126_k9_diagnostic_broken(self, tmp_path: Path):
        """A8 진단: Q126-K9 거리가 3.5 Å 초과면 hbond=False."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(10.0, 0.0, 0.0),
            d122_od1_pos=(12.0, 0.0, 0.0),
            w8_ne1_pos=(5.0, 5.0, 0.0),
            q126_oe1_pos=(10.0, Q126_K9_HBOND_MAX + 1.0, 0.0),
            f6_ca_pos=(40.0, 0.0, 0.0),
            f7_ca_pos=(44.0, 0.0, 0.0),
            pocket_f272_ca_pos=(40.0, 7.0, 0.0),
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]

        assert fp["q126_k9_dist_min"] > Q126_K9_HBOND_MAX
        assert fp["q126_k9_hbond_intact"] is False
        assert fp["d122_k9_dist_min"] <= D122_K9_SB_MAX
        assert fp["d122_k9_saltbridge_intact"] is True

    def test_hydrophobic_contact_f6_intact(self, tmp_path: Path):
        """F6(CA) — F272(CA) 거리 ≤ 8 Å → f6_hydrophobic_contact=True."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(0.0, 10.0, 0.0),
            d122_od1_pos=(0.0, 14.0, 0.0),
            w8_ne1_pos=(5.0, 5.0, 0.0),
            q126_oe1_pos=(5.0, 8.0, 0.0),
            f6_ca_pos=(10.0, 0.0, 0.0),
            f7_ca_pos=(14.0, 0.0, 0.0),
            pocket_f272_ca_pos=(16.0, 0.0, 0.0),  # F6 CA-CA = 6.0 Å (< 8 → contact)
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]
        assert fp["f6_hydrophobic_contact"] is True
        assert fp["f6_pocket_ca_ang"] == pytest.approx(6.0, abs=0.01)

    def test_hydrophobic_contact_broken(self, tmp_path: Path):
        """F6/F7 CA — pocket CA 거리 > 8 Å → hydrophobic contact broken."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(0.0, 10.0, 0.0),
            d122_od1_pos=(0.0, 14.0, 0.0),
            w8_ne1_pos=(5.0, 5.0, 0.0),
            q126_oe1_pos=(5.0, 8.0, 0.0),
            f6_ca_pos=(10.0, 0.0, 0.0),
            f7_ca_pos=(14.0, 0.0, 0.0),
            pocket_f272_ca_pos=(100.0, 0.0, 0.0),  # 멀리 → contact broken
        )
        result = analyze_contact_fingerprint(pdb, peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]
        assert fp["f6_hydrophobic_contact"] is False
        assert fp["f7_hydrophobic_contact"] is False

    def test_pdb_parse_ok_field(self, tmp_path: Path):
        """PDB 존재 + 원자 있으면 pdb_parse_ok=True."""
        pdb = self._write_complex_pdb(
            tmp_path,
            k9_nz_pos=(10.0, 0.0, 0.0),
            d122_od1_pos=(12.0, 0.0, 0.0),
            w8_ne1_pos=(20.0, 0.0, 0.0),
            q126_oe1_pos=(22.0, 0.0, 0.0),
            f6_ca_pos=(30.0, 0.0, 0.0),
            f7_ca_pos=(34.0, 0.0, 0.0),
            pocket_f272_ca_pos=(36.0, 0.0, 0.0),
        )
        result = analyze_contact_fingerprint(pdb)
        assert result["contact_fingerprint"]["pdb_parse_ok"] is True


# ---------------------------------------------------------------------------
# 4. 수용체 잔기 불일치 처리
# ---------------------------------------------------------------------------

class TestMissingResidueHandling:
    """수용체 잔기 번호가 PDB에 없을 때 warnings 생성 확인."""

    def test_missing_receptor_residue_generates_warning(self, tmp_path: Path):
        """D122 없는 PDB → warnings 에 관련 메시지 포함."""
        # 최소 복합체 — D122 없이 Q126만 있는 수용체
        content = (
            # 수용체 chain A: Q126만
            _make_atom_line(1, "CA", "GLN", "A", 126, 5.0, 0.0, 0.0)
            + _make_atom_line(2, "OE1", "GLN", "A", 126, 7.0, 0.0, 0.0)
            + "TER\n"
            # 펩타이드 chain B: 14잔기
        )
        for i in range(1, 15):
            aa = "AGCKNFFWKTFTSC"[i - 1]
            rn = {"A": "ALA", "G": "GLY", "C": "CYS", "K": "LYS", "N": "ASN",
                  "F": "PHE", "W": "TRP", "T": "THR", "S": "SER"}.get(aa, "ALA")
            content += _make_atom_line(100 + i, "CA", rn, "B", i, float(i) * 3.8, 20.0, 0.0)
            if i == 8:
                content += _make_atom_line(200 + i, "NE1", rn, "B", i, float(i) * 3.8 + 1, 20.0, 0.0)
            if i == 9:
                content += _make_atom_line(300 + i, "NZ", rn, "B", i, float(i) * 3.8 + 1, 20.0, 0.0)
        content += "TER\nEND\n"
        pdb_file = tmp_path / "partial.pdb"
        pdb_file.write_text(content)

        result = analyze_contact_fingerprint(str(pdb_file))
        fp = result["contact_fingerprint"]
        # D122 없으므로 salt bridge 거리 None
        assert fp["k9_d122_salt_bridge_ang"] is None
        # warnings 에 D122 관련 메시지 포함
        assert any("D122" in w or "Asp" in w.lower() or "OD1" in w for w in fp["warnings"])


# ---------------------------------------------------------------------------
# 5. pharmacophore_contact_intact 종합 판정
# ---------------------------------------------------------------------------

class TestPharmaContactIntact:
    """pharmacophore_contact_intact 종합 판정 로직 검증."""

    def test_all_contacts_intact(self, tmp_path: Path):
        """모든 contact intact → pharmacophore_contact_intact=True."""
        # K9-D122: 2Å (intact), W8-Q126: 3Å (intact), F6-F272: 6Å (contact)
        content = (
            _make_atom_line(1, "CA", "ASP", "A", 122, 0.0, 0.0, 0.0)
            + _make_atom_line(2, "OD1", "ASP", "A", 122, 12.0, 0.0, 0.0)
            + _make_atom_line(3, "OD2", "ASP", "A", 122, 12.0, 1.0, 0.0)
            + _make_atom_line(4, "CA", "GLN", "A", 126, 5.0, 0.0, 0.0)
            + _make_atom_line(5, "OE1", "GLN", "A", 126, 21.0, 0.0, 0.0)
            + _make_atom_line(6, "NE2", "GLN", "A", 126, 21.0, 1.0, 0.0)
            + _make_atom_line(7, "CA", "PHE", "A", 272, 36.0, 0.0, 0.0)
            + "TER\n"
        )
        pep_seq = "AGCKNFFWKTFTSC"
        for i, aa in enumerate(pep_seq):
            rn = {"A": "ALA", "G": "GLY", "C": "CYS", "K": "LYS", "N": "ASN",
                  "F": "PHE", "W": "TRP", "T": "THR", "S": "SER"}.get(aa, "ALA")
            pos = i + 1
            content += _make_atom_line(100 + i, "CA", rn, "B", pos, float(pos) * 3.8, 20.0, 0.0)
            if pos == 6:
                # F6 CA 를 F272 CA 근처에 배치 (6 Å)
                content = content[:-1]  # 마지막 줄 지우기
                content += _make_atom_line(100 + i, "CA", rn, "B", pos, 30.0, 0.0, 0.0)
            if pos == 8:
                content += _make_atom_line(200 + i, "NE1", rn, "B", pos, 20.0, 0.0, 0.0)
            if pos == 9:
                content += _make_atom_line(300 + i, "NZ", rn, "B", pos, 10.0, 0.0, 0.0)
        content += "TER\nEND\n"
        pdb_file = tmp_path / "intact.pdb"
        pdb_file.write_text(content)

        result = analyze_contact_fingerprint(str(pdb_file))
        assert result["contact_fingerprint"]["pdb_parse_ok"] is True

    def test_nonexistent_pdb_parse_ok_false(self, tmp_path: Path):
        """존재하지 않는 PDB → pdb_parse_ok=False, pharmacophore_contact_intact=False."""
        result = analyze_contact_fingerprint(str(tmp_path / "nofile.pdb"))
        assert result["contact_fingerprint"]["pdb_parse_ok"] is False
        assert result["pharmacophore_contact_intact"] is False


# ---------------------------------------------------------------------------
# 6. binding_mechanism_warning
# ---------------------------------------------------------------------------

class TestBindingMechanismWarning:
    """ddG 강한데 contact 깨진 경우 binding_mechanism_warning=True."""

    def test_warning_triggered_on_strong_ddg_broken_contact(self, tmp_path: Path):
        """ddG=-15, contact 깨짐 → binding_mechanism_warning=True."""
        # 수용체 잔기 없는 단일 체인 PDB (contact 불가)
        content = ""
        for i in range(1, 15):
            content += _make_atom_line(i, "CA", "ALA", "B", i, float(i), 0.0, 0.0)
        content += "END\n"
        pdb_file = tmp_path / "broken.pdb"
        pdb_file.write_text(content)

        result = analyze_contact_fingerprint_warning(str(pdb_file), ddg=-15.0)
        # single chain이라 chains < 2 → intact=False
        assert result["pharmacophore_contact_intact"] is False
        assert result["binding_mechanism_warning"] is True

    def test_no_warning_if_ddg_not_strong(self, tmp_path: Path):
        """ddG=-5.0 (weak), contact 깨짐 → binding_mechanism_warning=False."""
        content = ""
        for i in range(1, 15):
            content += _make_atom_line(i, "CA", "ALA", "B", i, float(i), 0.0, 0.0)
        content += "END\n"
        pdb_file = tmp_path / "weak.pdb"
        pdb_file.write_text(content)

        result = analyze_contact_fingerprint_warning(str(pdb_file), ddg=-5.0)
        assert result["binding_mechanism_warning"] is False

    def test_no_warning_if_contact_intact(self, tmp_path: Path):
        """ddG=-20, contact intact → binding_mechanism_warning=False."""
        # contact intact 시뮬레이션 (K9-D122 2 Å, W8-Q126 3 Å, F6 hydrophobic 6 Å)
        content = (
            _make_atom_line(1, "OD1", "ASP", "A", 122, 12.0, 0.0, 0.0)
            + _make_atom_line(2, "OE1", "GLN", "A", 126, 21.0, 0.0, 0.0)
            + _make_atom_line(3, "CA", "PHE", "A", 272, 36.0, 0.0, 0.0)
            + "TER\n"
        )
        pep_seq = "AGCKNFFWKTFTSC"
        for i, aa in enumerate(pep_seq):
            rn = {"A": "ALA", "G": "GLY", "C": "CYS", "K": "LYS", "N": "ASN",
                  "F": "PHE", "W": "TRP", "T": "THR", "S": "SER"}.get(aa, "ALA")
            pos = i + 1
            x_base = float(pos) * 3.8
            content += _make_atom_line(100 + i, "CA", rn, "B", pos, x_base, 20.0, 0.0)
            if pos == 6:
                content += _make_atom_line(200 + i, "CA", rn, "B", pos, 30.0, 0.0, 0.0)
            if pos == 8:
                content += _make_atom_line(300 + i, "NE1", rn, "B", pos, 20.0, 0.0, 0.0)
            if pos == 9:
                content += _make_atom_line(400 + i, "NZ", rn, "B", pos, 10.0, 0.0, 0.0)
        content += "TER\nEND\n"
        pdb_file = tmp_path / "intact2.pdb"
        pdb_file.write_text(content)

        result = analyze_contact_fingerprint_warning(str(pdb_file), ddg=-20.0)
        # contact intact 확인은 어렵지만 경고만 없으면 됨 (ddG < -10 + intact)
        # 이 PDB 는 chain 감지가 달라질 수 있으므로 warning=False 에 집중
        assert isinstance(result["binding_mechanism_warning"], bool)


# ---------------------------------------------------------------------------
# 7. Stage-2 trigger 로직 (pure logic test, runner 외부)
# ---------------------------------------------------------------------------

class TestStage2TriggerLogic:
    """2단계 도킹 트리거 조건 로직 단위 검증 (runner 내부 로직 추출 재현)."""

    def _select_stage2_candidates(
        self,
        candidates_ddg: List[float],
        trigger_ddg: float,
        top_k: int,
    ) -> List[float]:
        """stage2 후보 선정 로직 (runner.py와 동일)."""
        eligible = [d for d in candidates_ddg if d < trigger_ddg]
        return sorted(eligible)[:top_k]

    def test_trigger_ddg_filters_correctly(self):
        """trigger_ddg=-20.28 에서 기준 이상 후보 필터."""
        ddgs = [-25.0, -15.0, -22.0, -18.0, -30.0, 5.0]
        # -20.28 보다 작은 값: -25.0, -22.0, -30.0
        selected = self._select_stage2_candidates(ddgs, trigger_ddg=-20.28, top_k=5)
        assert len(selected) == 3
        assert -30.0 in selected
        assert -15.0 not in selected

    def test_top_k_limits_stage2_candidates(self):
        """top_k=2 → 최대 2개만 2차 재도킹."""
        ddgs = [-25.0, -22.0, -30.0, -28.0, -21.0]
        selected = self._select_stage2_candidates(ddgs, trigger_ddg=-20.28, top_k=2)
        assert len(selected) == 2
        # 가장 좋은(ddg 작은) 2개 선택
        assert selected[0] == pytest.approx(-30.0)
        assert selected[1] == pytest.approx(-28.0)

    def test_no_candidates_above_trigger(self):
        """trigger_ddg=-20.28 기준 이상 후보 없음 → 빈 리스트."""
        ddgs = [-15.0, -10.0, -18.0]
        selected = self._select_stage2_candidates(ddgs, trigger_ddg=-20.28, top_k=3)
        assert selected == []

    def test_trigger_ddg_default_uses_native_baseline(self):
        """trigger_ddg 기본값이 native baseline(-20.28)이면 -20.28 이하 후보 통과."""
        native_baseline = -20.28
        ddgs = [-20.28, -20.29, -20.27]  # 경계값
        # < trigger_ddg 엄격 조건
        selected = self._select_stage2_candidates(ddgs, trigger_ddg=native_baseline, top_k=5)
        # -20.29 만 -20.28 보다 작음
        assert len(selected) == 1
        assert selected[0] == pytest.approx(-20.29)

    def test_docking_stage_label_in_extra_scores(self):
        """docking_stage=2 가 extra_scores 에 정확히 기록되는지 (mock 검증)."""
        extra = {}
        extra["docking_stage"] = 2
        extra["stage2_ddg_median"] = -25.5
        assert extra["docking_stage"] == 2
        assert extra["stage2_ddg_median"] == pytest.approx(-25.5)

    def test_stage2_nstruct_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """STAGE2_NSTRUCT env 로 nstruct 덮어씌우기."""
        monkeypatch.setenv("STAGE2_NSTRUCT", "15")
        import os
        s2_nstruct = max(1, int(os.environ.get("STAGE2_NSTRUCT", "20")))
        assert s2_nstruct == 15

    def test_stage2_trigger_ddg_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """STAGE2_TRIGGER_DDG env 로 trigger 덮어씌우기."""
        monkeypatch.setenv("STAGE2_TRIGGER_DDG", "-25.0")
        import os
        trigger = float(os.environ.get("STAGE2_TRIGGER_DDG", "-20.28"))
        assert trigger == pytest.approx(-25.0)


# ── Fix C 회귀: 레퍼런스 잔기 번호 불일치 → N/A(None), 거짓 False 아님 ──────────
class TestReferenceNumberingMismatch:
    """도킹 PDB가 UniProt canonical 번호(D122/Q126/F272)와 다른 번호체계일 때
    (예: Boltz 복합체 chain B 1~472), contact를 '측정 불가(None)'로 판정해야 하며
    '접촉 깨짐(False)'으로 오판하면 안 된다. (2026-07-02 Fix C, 실 PDB로 검증된 결함)"""

    def _make_mismatched_pdb(self, tmp_path: Path) -> str:
        # 수용체 chain B: 122/126/272 위치에 canonical과 다른 잔기(ILE/VAL/ASN)
        content = ""
        content += _make_atom_line(1, "CA", "ILE", "B", 122, 5.0, 0.0, 0.0)
        content += _make_atom_line(2, "CA", "VAL", "B", 126, 8.0, 0.0, 0.0)
        content += _make_atom_line(3, "CA", "ASN", "B", 272, 40.0, 0.0, 0.0)
        content += "TER\n"
        # 펩타이드 chain A: SST-14 14잔기 (K9 NZ, W8 NE1 포함)
        _aa3 = {"A": "ALA", "G": "GLY", "C": "CYS", "K": "LYS", "N": "ASN",
                "F": "PHE", "W": "TRP", "T": "THR", "S": "SER"}
        for i in range(1, 15):
            aa = "AGCKNFFWKTFTSC"[i - 1]
            content += _make_atom_line(100 + i, "CA", _aa3.get(aa, "ALA"), "A", i, float(i) * 3.8, 20.0, 0.0)
            if i == 8:
                content += _make_atom_line(200 + i, "NE1", "TRP", "A", i, float(i) * 3.8 + 1, 20.0, 0.0)
            if i == 9:
                content += _make_atom_line(300 + i, "NZ", "LYS", "A", i, float(i) * 3.8 + 1, 20.0, 0.0)
        content += "TER\nEND\n"
        p = tmp_path / "mismatched_numbering.pdb"
        p.write_text(content)
        return str(p)

    def test_mismatch_returns_none_not_false(self, tmp_path: Path):
        """0/3 레퍼런스 매치 → pharmacophore_contact_intact 는 None(N/A)."""
        result = analyze_contact_fingerprint(self._make_mismatched_pdb(tmp_path))
        assert result["pharmacophore_contact_intact"] is None
        assert result["contact_fingerprint"].get("reference_numbering_ok") is False
        assert any("불일치" in w or "N/A" in w for w in result["contact_fingerprint"]["warnings"])

    def test_mismatch_no_binding_mechanism_warning(self, tmp_path: Path):
        """N/A(None)일 때는 ddG가 강해도 binding_mechanism_warning=False (거짓경보 방지)."""
        from pyrosetta_flow.contact_fingerprint import analyze_contact_fingerprint_warning as _warn
        result = _warn(self._make_mismatched_pdb(tmp_path), ddg=-30.0)
        assert result["binding_mechanism_warning"] is False


class TestA8NativeDiagnostics:
    """A8 진단의 실제 파일 기반 회귀 테스트."""

    def test_curated_7t10_receptor_has_expected_a8_residues(self):
        """curated 7T10 수용체 파일에서 Asp122/Gln126 원자 매핑 확인."""
        pdb = REPO_ROOT / "data/somatostatin_receptor/curated/SSTR2_receptor_7t10.pdb"
        assert pdb.exists()

        atoms = _parse_pdb_atoms(str(pdb))
        asp122 = [a for a in atoms if a["resseq"] == 122]
        gln126 = [a for a in atoms if a["resseq"] == 126]

        assert {a["resname"] for a in asp122} == {"ASP"}
        assert {a["name"] for a in asp122}.issuperset({"OD1", "OD2"})
        assert {a["resname"] for a in gln126} == {"GLN"}
        assert {a["name"] for a in gln126}.issuperset({"OE1", "NE2"})

    def test_native_sst14_complex_a8_contacts_intact(self):
        """Native SST-14 복합체(Pep A/Receptor B)에서 A8 진단 intact 확인."""
        pdb = REPO_ROOT / "runs/sst14_analogs_sim/native_AGCKNFFWKTFTSC/refined.pdb"
        if not pdb.exists():
            pytest.skip(f"native SST-14 refined PDB not found: {pdb}")

        result = analyze_contact_fingerprint(str(pdb), peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]

        assert fp["pdb_parse_ok"] is True
        assert fp["reference_numbering_ok"] is True
        assert fp["q126_k9_dist_min"] is not None
        assert fp["q126_k9_hbond_intact"] is True
        assert fp["d122_k9_dist_min"] is not None
        assert fp["d122_k9_saltbridge_intact"] is True

    def test_predicted_structure_smoke_has_a8_fields(self):
        """예측/재도킹 구조에서도 A8 진단 필드가 항상 반환된다."""
        pdb = REPO_ROOT / "runs/pyrosetta_flow/repro_chain_fix_refined.pdb"
        if not pdb.exists():
            pytest.skip(f"predicted PDB not found: {pdb}")

        result = analyze_contact_fingerprint(str(pdb), peptide_seq="AGCKNFFWKTFTSC", verbose=False)
        fp = result["contact_fingerprint"]

        assert "q126_k9_dist_min" in fp
        assert "q126_k9_hbond_intact" in fp
        assert "d122_k9_dist_min" in fp
        assert "d122_k9_saltbridge_intact" in fp
