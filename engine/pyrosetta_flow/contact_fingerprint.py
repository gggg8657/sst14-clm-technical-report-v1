"""contact_fingerprint.py
=======================
SSTR2 binding pocket contact fingerprint 분석 모듈.

도킹 복합체 PDB에서 SST-14 pharmacophore 핵심 contact를 원자간 거리로 추출한다.
PyRosetta 의존 없이 PDB 텍스트 파싱만 사용하여 bio-tools 환경 밖에서도 호출 가능.

참조 구조: SSTR2-SST14 cryo-EM (PDB 7T11 / 7T10)
  - Trp8(W8)-Lys9(K9) motif — binding pocket 내부 깊은 위치
  - K9(NZ) — SSTR2 Asp122(OD1/OD2) salt bridge (~3.5 Å)
  - Gln126(OE1/NE2) — W8 NE1 H-bond (~3.5 Å)
  - Phe6(CZ)/Phe7(CZ) — hydrophobic pocket (CA-CA ≤ 8 Å)

SST-14 numbering (1-indexed): A=1,G=2,C=3,K=4,N=5,F=6,F=7,W=8,K=9,T=10,F=11,T=12,S=13,C=14
pharmacophore: F6,F7,W8,K9 (pos 6-9, 1-indexed)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# SSTR2 pocket 잔기 번호 (수용체 PDB 기준, 정렬/실험 구조 의존)
# cryo-EM SSTR2 (7T11) binding pocket key residues:
#   Asp122(D122), Gln126(Q126), Tyr205(Y205), Phe272(F272), His273(H273),
#   Asn276(N276), Trp281(W281), Phe294(F294)
# 기본값: 수용체 chain A의 잔기 번호 (아미노산 서열상 번호 기준).
# 실제 Pose 번호(1-indexed)로 자동 보정은 _match_receptor_residue() 사용.
# ---------------------------------------------------------------------------

# SSTR2 pharmacophore-relevant pocket residues (sequence number, 기본 — 실험 구조 7T11)
_SSTR2_POCKET_SEQ_NUMS: Dict[str, int] = {
    "ASP122": 122,
    "GLN126": 126,
    "TYR205": 205,
    "PHE272": 272,
    "HIS273": 273,
    "ASN276": 276,
}

# SST-14 pharmacophore 위치 (1-indexed): F6, F7, W8, K9
_PHARMA_1IDX = [6, 7, 8, 9]

# 거리 임계값
_SALT_BRIDGE_MAX_ANG = 4.0       # K9 NZ — D122 OD 염다리 (4 Å 이내)
_HBOND_MAX_ANG = 3.5             # W8 NE1 — Q126 OE1/NE2 H-bond
_HYDROPHOBIC_CA_MAX_ANG = 8.0    # Phe6/Phe7 CA — 소수성 pocket CA 거리

# A8 diagnostic contacts (score에는 사용하지 않는 리더보드/로그 진단 필드)
Q126_K9_HBOND_MAX = 3.5           # Q126 OE1/NE2 — K9 NZ H-bond
D122_K9_SB_MAX = 4.0              # D122 OD1/OD2 — K9 NZ salt bridge


# ---------------------------------------------------------------------------
# PDB 파싱 유틸
# ---------------------------------------------------------------------------

def _parse_pdb_atoms(pdb_path: str) -> List[Dict[str, Any]]:
    """PDB 파일에서 ATOM/HETATM 레코드를 파싱하여 원자 dict 리스트 반환.

    반환 dict 키: chain, resseq(int), resname, name(원자명), x, y, z
    """
    atoms: List[Dict[str, Any]] = []
    path = Path(pdb_path)
    if not path.exists():
        print(
            f"[contact_fingerprint] WARNING: PDB not found: {pdb_path}",
            file=sys.stderr,
        )
        return atoms

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atoms.append({
                    "chain": chain,
                    "resseq": resseq,
                    "resname": resname,
                    "name": atom_name,
                    "x": x,
                    "y": y,
                    "z": z,
                })
            except (ValueError, IndexError):
                continue
    return atoms


def _dist(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """두 원자 dict 사이 유클리드 거리 (Å)."""
    dx = a["x"] - b["x"]
    dy = a["y"] - b["y"]
    dz = a["z"] - b["z"]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _get_atoms(
    atoms: List[Dict[str, Any]],
    chain: Optional[str],
    resseq: int,
    atom_names: List[str],
) -> List[Dict[str, Any]]:
    """chain, resseq, atom_name 으로 원자 필터링 (복수 atom_names 허용)."""
    result = []
    for a in atoms:
        if chain is not None and a["chain"] != chain:
            continue
        if a["resseq"] != resseq:
            continue
        if a["name"] in atom_names:
            result.append(a)
    return result


def _min_dist_between(
    set_a: List[Dict[str, Any]],
    set_b: List[Dict[str, Any]],
) -> Optional[float]:
    """두 원자 집합 사이 최소 거리 반환. 집합 중 하나가 비면 None."""
    if not set_a or not set_b:
        return None
    return min(_dist(a, b) for a in set_a for b in set_b)


def _get_chains(atoms: List[Dict[str, Any]]) -> List[str]:
    """PDB의 chain ID 목록 반환 (순서 보존)."""
    seen: List[str] = []
    for a in atoms:
        if a["chain"] not in seen:
            seen.append(a["chain"])
    return seen


# ---------------------------------------------------------------------------
# 핵심 contact 분석 함수
# ---------------------------------------------------------------------------

def _detect_peptide_chain(atoms: List[Dict[str, Any]], pep_len: int = 14) -> Optional[str]:
    """펩타이드 chain ID 추정 (가장 잔기 수가 pep_len에 가까운 체인).

    FlexPepDock는 펩타이드를 마지막 체인(chain B 또는 C)에 배치.
    """
    chains = _get_chains(atoms)
    if not chains:
        return None
    # 체인별 유니크 resseq 수
    chain_rescount: Dict[str, int] = {}
    for a in atoms:
        c = a["chain"]
        if c not in chain_rescount:
            chain_rescount[c] = set()  # type: ignore[assignment]
        chain_rescount[c].add(a["resseq"])  # type: ignore[attr-defined]

    chain_sizes = {c: len(chain_rescount[c]) for c in chain_rescount}  # type: ignore[arg-type]
    # pep_len에 가장 가까운 체인 선택 (짧은 체인 = 펩타이드)
    pep_chain = min(chain_sizes, key=lambda c: (abs(chain_sizes[c] - pep_len), chain_sizes[c]))
    return pep_chain


def _find_receptor_residue_by_seqnum(
    atoms: List[Dict[str, Any]],
    receptor_chain: str,
    seq_num: int,
) -> Optional[int]:
    """수용체 chain의 seq_num 잔기 존재 여부 확인. 존재하면 seq_num 반환."""
    for a in atoms:
        if a["chain"] == receptor_chain and a["resseq"] == seq_num:
            return seq_num
    return None


def analyze_contact_fingerprint(
    pdb_path: str,
    peptide_seq: str = "AGCKNFFWKTFTSC",
    sstr2_asp122_seqnum: int = 122,
    sstr2_gln126_seqnum: int = 126,
    sstr2_phe272_seqnum: int = 272,
    sstr2_his273_seqnum: int = 273,
    sstr2_asn276_seqnum: int = 276,
    verbose: bool = False,
) -> Dict[str, Any]:
    """도킹 복합체 PDB에서 SSTR2-SST14 pharmacophore contact fingerprint 분석.

    핵심 contact 5종:
      1. K9(NZ) — D122(OD1/OD2) salt bridge 거리
      2. W8(NE1) — Q126(OE1/NE2) H-bond 거리
      3. F6(CA) — pocket(F272/H273/N276) CA 최소 거리
      4. F7(CA) — pocket(F272/H273/N276) CA 최소 거리
      5. F6/F7 hydrophobic contact 유지 여부 (CA-CA ≤ 8 Å)

    A8 진단 contact 2종(스코어/종합 판정 미반영):
      - Q126(OE1/NE2) — K9(NZ) H-bond 거리
      - D122(OD1/OD2) — K9(NZ) salt bridge 거리

    Args:
        pdb_path: 도킹 복합체 PDB 경로 (수용체+펩타이드 합본).
        peptide_seq: 펩타이드 서열 (F6,F7,W8,K9 위치 확인용; 기본 SST-14).
        sstr2_asp122_seqnum: SSTR2 Asp122 잔기 번호 (default 122).
        sstr2_gln126_seqnum: SSTR2 Gln126 잔기 번호 (default 126).
        sstr2_phe272_seqnum: SSTR2 Phe272 잔기 번호 (default 272).
        sstr2_his273_seqnum: SSTR2 His273 잔기 번호 (default 273).
        sstr2_asn276_seqnum: SSTR2 Asn276 잔기 번호 (default 276).
        verbose: True 이면 세부 거리값을 stderr에 출력.

    Returns:
        dict with keys:
          contact_fingerprint: {
            "k9_d122_salt_bridge_ang": float | None,
            "k9_d122_salt_bridge_intact": bool,
            "w8_q126_hbond_ang": float | None,
            "w8_q126_hbond_intact": bool,
            "f6_pocket_ca_ang": float | None,
            "f7_pocket_ca_ang": float | None,
            "f6_hydrophobic_contact": bool,
            "f7_hydrophobic_contact": bool,
            "q126_k9_dist_min": float | None,
            "q126_k9_hbond_intact": bool,
            "d122_k9_dist_min": float | None,
            "d122_k9_saltbridge_intact": bool,
            "pharmacophore_contact_intact": bool,
            "pdb_parse_ok": bool,
            "warnings": [str],
          }
          pharmacophore_contact_intact: bool (convenience key)
    """
    warnings: List[str] = []
    result: Dict[str, Any] = {
        "k9_d122_salt_bridge_ang": None,
        "k9_d122_salt_bridge_intact": False,
        "w8_q126_hbond_ang": None,
        "w8_q126_hbond_intact": False,
        "f6_pocket_ca_ang": None,
        "f7_pocket_ca_ang": None,
        "f6_hydrophobic_contact": False,
        "f7_hydrophobic_contact": False,
        "q126_k9_dist_min": None,
        "q126_k9_hbond_intact": False,
        "d122_k9_dist_min": None,
        "d122_k9_saltbridge_intact": False,
        "pharmacophore_contact_intact": False,
        "pdb_parse_ok": False,
        "warnings": warnings,
    }

    atoms = _parse_pdb_atoms(pdb_path)
    if not atoms:
        warnings.append(f"PDB 파싱 결과 원자 없음: {pdb_path}")
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": False}

    result["pdb_parse_ok"] = True

    # 체인 감지
    chains = _get_chains(atoms)
    if len(chains) < 2:
        warnings.append(f"체인 수 < 2 (found: {chains}). 복합체 PDB인지 확인 필요.")
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": False}

    pep_chain = _detect_peptide_chain(atoms, pep_len=len(peptide_seq))
    receptor_chains = [c for c in chains if c != pep_chain]
    if not receptor_chains:
        warnings.append(f"수용체 체인을 감지하지 못함 (chains={chains}, pep_chain={pep_chain}).")
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": False}

    # 수용체는 여러 체인일 수 있으므로 None으로 수용체 체인 전체 검색
    # 단, 펩타이드 체인은 제외
    receptor_atoms = [a for a in atoms if a["chain"] != pep_chain]
    peptide_atoms = [a for a in atoms if a["chain"] == pep_chain]

    if not peptide_atoms:
        warnings.append(f"펩타이드 체인({pep_chain}) 원자 없음.")
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": False}

    # ── Fix C: 레퍼런스 잔기 번호 정합성 검증 ─────────────────────────────────
    # 수용체 레퍼런스 잔기(Asp122/Gln126/Phe272)의 실제 잔기명이 기대와 다르면
    # (예: 도킹 PDB가 UniProt canonical 번호가 아닌 다른 번호체계) 접촉 측정이
    # 불가능하다. 이 경우 "접촉 깨짐(False)"이 아니라 "측정 불가(None=N/A)"로
    # 판정을 유보해 거짓 False가 로그·리더보드에 남지 않도록 한다.
    def _resname_at(seqnum: int) -> Optional[str]:
        for a in receptor_atoms:
            if a["resseq"] == seqnum:
                return a["resname"]
        return None
    _ref_expect = {
        sstr2_asp122_seqnum: "ASP",
        sstr2_gln126_seqnum: "GLN",
        sstr2_phe272_seqnum: "PHE",
    }
    _ref_matches = sum(1 for sn, exp in _ref_expect.items() if _resname_at(sn) == exp)
    result["reference_numbering_ok"] = _ref_matches >= 1
    # 레퍼런스 잔기가 하나도 안 맞으면(0/3) 번호체계 자체가 다른 것 → 측정 불가(N/A).
    # 1개 이상 맞으면 부분 누락으로 보고 측정 가능한 접촉만 계산(기존 동작 유지).
    if _ref_matches == 0:
        _found = {sn: _resname_at(sn) for sn in _ref_expect}
        warnings.append(
            "수용체 레퍼런스 잔기 번호 불일치 — 이 PDB의 번호체계가 UniProt canonical과 "
            f"다름(기대 {_ref_expect} vs 실제 {_found}). pharmacophore contact 측정 불가 → N/A."
        )
        result["pharmacophore_contact_intact"] = None  # None = 측정 불가(N/A), False(접촉깨짐) 아님
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": None}

    # 펩타이드 잔기 번호 목록 (정렬)
    pep_resseqs = sorted(set(a["resseq"] for a in peptide_atoms))
    pep_len_actual = len(pep_resseqs)

    if pep_len_actual < 9:
        warnings.append(f"펩타이드 잔기 수 부족 ({pep_len_actual}개). pharmacophore pos 9 접근 불가.")
        return {"contact_fingerprint": result, "pharmacophore_contact_intact": False}

    # pharmacophore 잔기 resseq 매핑 (1-indexed → pep_resseqs index)
    def pep_pos_to_resseq(pos_1idx: int) -> Optional[int]:
        """1-indexed 펩타이드 위치 → PDB resseq. 범위 초과면 None."""
        idx = pos_1idx - 1
        if 0 <= idx < len(pep_resseqs):
            return pep_resseqs[idx]
        return None

    f6_resseq = pep_pos_to_resseq(6)
    f7_resseq = pep_pos_to_resseq(7)
    w8_resseq = pep_pos_to_resseq(8)
    k9_resseq = pep_pos_to_resseq(9)

    if verbose:
        print(
            f"[contact_fp] pep_chain={pep_chain} pep_len={pep_len_actual} "
            f"F6_resseq={f6_resseq} F7_resseq={f7_resseq} "
            f"W8_resseq={w8_resseq} K9_resseq={k9_resseq}",
            file=sys.stderr,
        )

    # 수용체 pocket 잔기 원자 — 없으면 Warning
    def _recv_atoms_multi(resseq: int, atom_names: List[str]) -> List[Dict[str, Any]]:
        return [a for a in receptor_atoms if a["resseq"] == resseq and a["name"] in atom_names]

    # ── Contact 1: K9(NZ) — D122(OD1/OD2) salt bridge ──────────────────────
    if k9_resseq is not None:
        k9_nz = [a for a in peptide_atoms if a["resseq"] == k9_resseq and a["name"] == "NZ"]
        d122_od = _recv_atoms_multi(sstr2_asp122_seqnum, ["OD1", "OD2"])

        d_salt = _min_dist_between(k9_nz, d122_od)
        if d_salt is not None:
            result["k9_d122_salt_bridge_ang"] = round(d_salt, 3)
            result["k9_d122_salt_bridge_intact"] = d_salt <= _SALT_BRIDGE_MAX_ANG
            result["d122_k9_dist_min"] = round(d_salt, 3)
            result["d122_k9_saltbridge_intact"] = d_salt <= D122_K9_SB_MAX
        else:
            if not k9_nz:
                warnings.append(f"K9(NZ) 원자 없음 (resseq={k9_resseq}). 서열에서 K→다른 AA 변이 가능.")
            if not d122_od:
                warnings.append(
                    f"SSTR2 D{sstr2_asp122_seqnum}(OD1/OD2) 원자 없음 — "
                    "수용체 resseq 번호가 잘못되었거나 PDB에 없는 잔기."
                )

        if verbose:
            print(
                f"[contact_fp] K9(NZ)→D{sstr2_asp122_seqnum}(OD): "
                f"dist={result['k9_d122_salt_bridge_ang']} Å "
                f"intact={result['k9_d122_salt_bridge_intact']}",
                file=sys.stderr,
            )

        # A8 diagnostic: Q126 side-chain amide ↔ K9 side-chain NZ.
        # 기존 W8-Q126 pharmacophore H-bond와 별도이며 score/종합 판정에는 반영하지 않는다.
        q126_oene_for_k9 = _recv_atoms_multi(sstr2_gln126_seqnum, ["OE1", "NE2"])
        d_q126_k9 = _min_dist_between(q126_oene_for_k9, k9_nz)
        if d_q126_k9 is not None:
            result["q126_k9_dist_min"] = round(d_q126_k9, 3)
            result["q126_k9_hbond_intact"] = d_q126_k9 <= Q126_K9_HBOND_MAX
        else:
            if not q126_oene_for_k9:
                warnings.append(
                    f"SSTR2 Q{sstr2_gln126_seqnum}(OE1/NE2) 원자 없음 — "
                    "Q126-K9 A8 진단 측정 불가."
                )

        if verbose:
            print(
                f"[contact_fp] A8 Q{sstr2_gln126_seqnum}(OE1/NE2)→K9(NZ): "
                f"dist={result['q126_k9_dist_min']} Å "
                f"intact={result['q126_k9_hbond_intact']}",
                file=sys.stderr,
            )

    # ── Contact 2: W8(NE1) — Q126(OE1/NE2) H-bond ──────────────────────────
    if w8_resseq is not None:
        w8_ne1 = [a for a in peptide_atoms if a["resseq"] == w8_resseq and a["name"] == "NE1"]
        q126_oene = _recv_atoms_multi(sstr2_gln126_seqnum, ["OE1", "NE2"])

        d_hb = _min_dist_between(w8_ne1, q126_oene)
        if d_hb is not None:
            result["w8_q126_hbond_ang"] = round(d_hb, 3)
            result["w8_q126_hbond_intact"] = d_hb <= _HBOND_MAX_ANG
        else:
            if not w8_ne1:
                warnings.append(f"W8(NE1) 원자 없음 (resseq={w8_resseq}). W→다른 AA 변이 가능.")
            if not q126_oene:
                warnings.append(
                    f"SSTR2 Q{sstr2_gln126_seqnum}(OE1/NE2) 원자 없음."
                )

        if verbose:
            print(
                f"[contact_fp] W8(NE1)→Q{sstr2_gln126_seqnum}(OE1/NE2): "
                f"dist={result['w8_q126_hbond_ang']} Å "
                f"intact={result['w8_q126_hbond_intact']}",
                file=sys.stderr,
            )

    # ── Contact 3/4: F6/F7(CA) — hydrophobic pocket(F272/H273/N276) CA ──────
    pocket_ca_resseqs = [sstr2_phe272_seqnum, sstr2_his273_seqnum, sstr2_asn276_seqnum]
    pocket_ca_atoms = [
        a for a in receptor_atoms
        if a["resseq"] in pocket_ca_resseqs and a["name"] == "CA"
    ]
    if not pocket_ca_atoms:
        warnings.append(
            f"소수성 pocket CA 원자 없음 (resseqs={pocket_ca_resseqs}). "
            "수용체 resseq 확인 필요."
        )

    for pep_pos, pep_resseq, result_key_ang, result_key_contact in [
        (6, f6_resseq, "f6_pocket_ca_ang", "f6_hydrophobic_contact"),
        (7, f7_resseq, "f7_pocket_ca_ang", "f7_hydrophobic_contact"),
    ]:
        if pep_resseq is None:
            warnings.append(f"F{pep_pos} resseq 산출 불가 (펩타이드 서열 부족).")
            continue
        pep_ca = [a for a in peptide_atoms if a["resseq"] == pep_resseq and a["name"] == "CA"]
        if not pep_ca:
            # CA 없으면 CB 시도 (아미노산 대치된 경우)
            pep_ca = [a for a in peptide_atoms if a["resseq"] == pep_resseq and a["name"] == "CB"]
        d_hydro = _min_dist_between(pep_ca, pocket_ca_atoms)
        if d_hydro is not None:
            result[result_key_ang] = round(d_hydro, 3)
            result[result_key_contact] = d_hydro <= _HYDROPHOBIC_CA_MAX_ANG
        else:
            if not pep_ca:
                warnings.append(f"F{pep_pos}(CA/CB) 원자 없음 (resseq={pep_resseq}).")

        if verbose:
            print(
                f"[contact_fp] F{pep_pos}(CA)→pocket_CA: "
                f"dist={result[result_key_ang]} Å "
                f"contact={result[result_key_contact]}",
                file=sys.stderr,
            )

    # ── pharmacophore_contact_intact 종합 판정 ────────────────────────────────
    # 필수 조건:
    #   (a) K9-D122 salt bridge intact 또는 K9_NZ 원자 자체가 없음(K→다른 AA 변이) — 경고만
    #   (b) W8-Q126 H-bond intact 또는 W8_NE1 없음
    #   (c) F6 or F7 hydrophobic contact intact (둘 중 하나 이상)
    # 모두 데이터가 있을 때만 종합 판정; 원자 없어서 None인 경우는 경고 후 판정 유보.
    _sb_ok: Optional[bool] = result["k9_d122_salt_bridge_intact"] if result["k9_d122_salt_bridge_ang"] is not None else None
    _hb_ok: Optional[bool] = result["w8_q126_hbond_intact"] if result["w8_q126_hbond_ang"] is not None else None
    _hy_ok: bool = result["f6_hydrophobic_contact"] or result["f7_hydrophobic_contact"]

    _has_any_contact_data = (
        result["k9_d122_salt_bridge_ang"] is not None
        or result["w8_q126_hbond_ang"] is not None
        or result["f6_pocket_ca_ang"] is not None
        or result["f7_pocket_ca_ang"] is not None
    )

    if not _has_any_contact_data:
        # 데이터 전무 — 판정 불가 (PDB 잔기 번호 불일치 가능성)
        warnings.append(
            "모든 contact 측정 불가. 수용체 잔기 번호(Asp122 등)가 "
            "이 PDB의 실제 번호와 다를 수 있음 → N/A(판정 유보)."
        )
        result["pharmacophore_contact_intact"] = None  # 측정 불가(N/A), 접촉깨짐(False) 아님
    else:
        # 데이터가 있는 항목만 판정에 반영
        # None은 "불확실"로 처리 — 최소 1개 이상 True 이면 전체 판정 True
        _intact_flags: List[bool] = []
        if _sb_ok is not None:
            _intact_flags.append(_sb_ok)
        if _hb_ok is not None:
            _intact_flags.append(_hb_ok)
        if result["f6_pocket_ca_ang"] is not None or result["f7_pocket_ca_ang"] is not None:
            _intact_flags.append(_hy_ok)

        # 전체 intact = 측정 가능한 contact 중 과반 이상 intact
        if _intact_flags:
            n_intact = sum(1 for f in _intact_flags if f)
            result["pharmacophore_contact_intact"] = n_intact >= len(_intact_flags) * 0.5
        else:
            result["pharmacophore_contact_intact"] = False

    if warnings and verbose:
        for w in warnings:
            print(f"[contact_fp] WARN: {w}", file=sys.stderr)

    return {
        "contact_fingerprint": result,
        "pharmacophore_contact_intact": result["pharmacophore_contact_intact"],
    }


def analyze_contact_fingerprint_warning(
    pdb_path: str,
    ddg: Optional[float] = None,
    peptide_seq: str = "AGCKNFFWKTFTSC",
    **kwargs: Any,
) -> Dict[str, Any]:
    """contact fingerprint 분석 + ddG 좋은데 contact 깨진 경우 경고 생성.

    ddG 좋아도(< -10 REU) pharmacophore_contact_intact=False 이면
    result에 "binding_mechanism_warning" 플래그 추가.

    Returns:
        analyze_contact_fingerprint 결과 + "binding_mechanism_warning": bool
    """
    result = analyze_contact_fingerprint(pdb_path, peptide_seq=peptide_seq, **kwargs)
    intact = result.get("pharmacophore_contact_intact", True)
    binding_mechanism_warning = False
    # intact is None(측정 불가/N/A)이면 경고 유보 — False(명시적 접촉깨짐)일 때만 경고
    if ddg is not None and ddg < -10.0 and intact is False:
        binding_mechanism_warning = True
        print(
            f"[contact_fp] WARNING: ddG={ddg:.2f} 강한데 pharmacophore contact 깨짐 — "
            f"결합 메커니즘 검토 필요 ({pdb_path})",
            file=sys.stderr,
        )
    result["binding_mechanism_warning"] = binding_mechanism_warning
    return result
