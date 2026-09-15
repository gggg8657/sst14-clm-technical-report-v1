#!/usr/bin/env python3
"""
cryo-EM 수용체 큐레이션 스크립트
-  7T10  → SSTR2 수용체 단독(chain A), SSTR2+SST14 복합체(chain A+B)
-  8XIP  → SSTR1 수용체 단독, SSTR2에 구조정렬
-  8XIR  → SSTR3 수용체 단독, SSTR2에 구조정렬
-  7XMS  → SSTR4 수용체 단독, SSTR2에 구조정렬
-  8ZCJ  → SSTR5 수용체 단독, SSTR2에 구조정렬

출력:
  curated/SSTR2_receptor_7t10.pdb      — SSTR2 수용체 단독 (chain A)
  curated/SSTR2_SST14_complex_7t10.pdb — SSTR2+SST14 복합체 (chain A+B)
  curated/SSTR1_receptor_8xip.pdb      — SSTR1 수용체 단독, SSTR2 정렬
  curated/SSTR3_receptor_8xir.pdb      — SSTR3 수용체 단독, SSTR2 정렬
  curated/SSTR4_receptor_7xms.pdb      — SSTR4 수용체 단독, SSTR2 정렬
  curated/SSTR5_receptor_8zcj.pdb      — SSTR5 수용체 단독, SSTR2 정렬

  curated/SSTR2_receptor.pdb      — 위 SSTR2 수용체의 교체 대상 (백업 후 덮어씀)
  curated/SSTR1_receptor.pdb      — 위 SSTR1의 교체 대상
  curated/SSTR3_receptor.pdb
  curated/SSTR4_receptor.pdb
  curated/SSTR5_receptor.pdb
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

from Bio.PDB import PDBParser, PDBIO, MMCIFParser, Superimposer, Select

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "somatostatin_receptor" / "raw_cryo_em"
CURATED_DIR = REPO / "data" / "somatostatin_receptor" / "curated"


# ---------------------------------------------------------------------------
# Helper: 단일 chain만 저장
# ---------------------------------------------------------------------------
class ChainSelect(Select):
    def __init__(self, chains: list[str], include_het: bool = False):
        self.chains = set(chains)
        self.include_het = include_het

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, residue):
        if residue.id[0] == ' ':   # 표준 아미노산
            return True
        if residue.id[0] == 'W':  # 물 제외
            return False
        return self.include_het


def save_chains(structure, chains: list[str], out_path: Path,
                renumber: bool = False, new_chain_ids: list[str] | None = None,
                include_het: bool = False) -> int:
    """지정 chain을 PDB로 저장. renumber=True면 잔기번호 1부터 재번호."""
    if renumber and new_chain_ids:
        # 잔기 재번호 + chain 이름 변경
        counter = 1
        for m in structure:
            for i, cid in enumerate(chains):
                ch = m[cid]
                new_id = new_chain_ids[i] if new_chain_ids else cid
                ch.id = new_id  # chain rename
                for res in list(ch.get_residues()):
                    if res.id[0] == ' ':
                        old_id = res.id
                        res.id = (' ', counter, ' ')
                        counter += 1
        # renumber 후 chain id가 바뀌었으므로 new_chain_ids로 저장
        sel = ChainSelect(new_chain_ids if new_chain_ids else chains, include_het)
    else:
        sel = ChainSelect(chains, include_het)

    io = PDBIO()
    io.set_structure(structure)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(out_path), sel)

    # 잔기 수 반환
    count = 0
    for m in structure:
        for cid in (new_chain_ids if (renumber and new_chain_ids) else chains):
            try:
                ch = m[cid]
                count += sum(1 for r in ch if r.id[0] == ' ')
            except KeyError:
                pass
    return count


# ---------------------------------------------------------------------------
# SSTR2 큐레이션 (7T10)
# ---------------------------------------------------------------------------
def curate_7t10() -> tuple[int, int]:
    """
    7T10: chain R = SSTR2 수용체 (ICL3→KOPR chimera, 잔기 40-327, 1개 gap @188)
          chain P = SST-14 (잔기 1-14)
    → 출력:
      (a) SSTR2_receptor_7t10.pdb : chain R만, chain A로 rename
      (b) SSTR2_SST14_complex_7t10.pdb : chain R+P → chain A+B
    """
    pdb_path = RAW_DIR / "7T10.pdb"
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("7T10", str(pdb_path))

    # --- 복합체 저장 (chain R→A, P→B, 재번호) ---
    # Biopython은 직접 chain rename + 잔기 재번호를 in-place로 해야 함
    # 두 번 파싱해서 각각 처리
    struct_complex = parser.get_structure("7T10_complex", str(pdb_path))
    model = struct_complex[0]

    # chain rename: R→A, P→B
    # 주의: 기존 chain A(Gα)가 있으므로 임시 이름 사용
    for m in struct_complex:
        for c in list(m.get_chains()):
            if c.id not in ('R', 'P'):
                m.detach_child(c.id)

    # R→A, P→B 재명명 (충돌 없으므로 직접)
    for m in struct_complex:
        try:
            ch_r = m['R']
            ch_r.id = 'A'
        except KeyError:
            pass
        try:
            ch_p = m['P']
            ch_p.id = 'B'
        except KeyError:
            pass

    io = PDBIO()
    io.set_structure(struct_complex)

    complex_path = CURATED_DIR / "SSTR2_SST14_complex_7t10.pdb"
    complex_path.parent.mkdir(parents=True, exist_ok=True)

    class StdResSelect(Select):
        def accept_residue(self, r):
            return r.id[0] == ' '
        def accept_atom(self, a):
            return True

    io.save(str(complex_path), StdResSelect())
    n_complex = sum(1 for m in struct_complex for c in m for r in c if r.id[0] == ' ')

    # --- 수용체 단독 저장 (R만, chain A) ---
    struct_rec = parser.get_structure("7T10_rec", str(pdb_path))
    for m in struct_rec:
        for c in list(m.get_chains()):
            if c.id != 'R':
                m.detach_child(c.id)
        try:
            m['R'].id = 'A'
        except KeyError:
            pass

    rec_path = CURATED_DIR / "SSTR2_receptor_7t10.pdb"
    io.set_structure(struct_rec)
    io.save(str(rec_path), StdResSelect())
    n_rec = sum(1 for m in struct_rec for c in m for r in c if r.id[0] == ' ')

    print(f"[7T10] 수용체 단독: {n_rec}잔기 → {rec_path}")
    print(f"[7T10] SSTR2+SST14 복합체: {n_complex}잔기 → {complex_path}")
    return n_rec, n_complex


# ---------------------------------------------------------------------------
# Off-target 큐레이션 공통 함수
# ---------------------------------------------------------------------------
def _extract_receptor_chain(
    pdb_id: str,
    pdb_path: Path,
    receptor_chain: str,
    out_path: Path,
    is_cif: bool = False,
) -> tuple[object, int]:
    """수용체 chain만 추출 → chain A로 rename하여 저장. structure 반환."""
    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    struct = parser.get_structure(pdb_id, str(pdb_path))

    for m in struct:
        for c in list(m.get_chains()):
            if c.id != receptor_chain:
                m.detach_child(c.id)
        try:
            ch = m[receptor_chain]
            ch.id = 'A'
        except KeyError:
            pass

    class StdResSelect(Select):
        def accept_residue(self, r):
            return r.id[0] == ' '

    io = PDBIO()
    io.set_structure(struct)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io.save(str(out_path), StdResSelect())
    n = sum(1 for m in struct for c in m for r in c if r.id[0] == ' ')
    return struct, n


def _align_to_sstr2(
    mobile_struct,
    ref_struct,
    mobile_chain: str = 'A',
    ref_chain: str = 'A',
) -> float:
    """CA 원자 기반 구조정렬. RMSD 반환."""
    sup = Superimposer()

    def get_ca(struct, chain_id):
        atoms = []
        for m in struct:
            try:
                ch = m[chain_id]
                for r in ch:
                    if r.id[0] == ' ' and 'CA' in r:
                        atoms.append(r['CA'])
            except KeyError:
                pass
        return atoms

    mobile_ca = get_ca(mobile_struct, mobile_chain)
    ref_ca = get_ca(ref_struct, ref_chain)

    # 공통 길이만큼 정렬 (단순 순서 기반 — 서열 동일성 낮으면 근사)
    min_len = min(len(mobile_ca), len(ref_ca))
    if min_len < 50:
        print(f"  경고: CA 원자 수 부족 ({min_len}). 정렬 스킵.")
        return -1.0

    sup.set_atoms(ref_ca[:min_len], mobile_ca[:min_len])
    sup.apply(list(mobile_struct.get_atoms()))
    return float(sup.rms)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="cryo-EM SSTR 수용체 큐레이션")
    p.add_argument("--overwrite-curated", action="store_true",
                   help="curated/SSTR*.pdb를 cryo-EM 파일로 덮어씀 (기본: 새 이름으로만 저장)")
    args = p.parse_args()

    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: 7T10 (SSTR2) 큐레이션")
    print("=" * 60)
    n_rec, n_complex = curate_7t10()

    # SSTR2 참조 구조 로드 (정렬 기준)
    pdb_parser = PDBParser(QUIET=True)
    ref_sstr2 = pdb_parser.get_structure(
        "sstr2_ref", str(CURATED_DIR / "SSTR2_receptor_7t10.pdb")
    )

    print()
    print("=" * 60)
    print("Step 2: off-target 큐레이션 및 SSTR2 정렬")
    print("=" * 60)

    off_targets = [
        # (pdb_id, raw_file, receptor_chain, is_cif, label, out_stem)
        # 8XIP/8XIR: CIF 변환 pdb에 삽입코드 E 문제 → CIF 직접 파싱
        ("8XIP", RAW_DIR / "8XIP.cif", "A", True, "SSTR1", "SSTR1_receptor_8xip"),
        ("8XIR", RAW_DIR / "8XIR.cif", "A", True, "SSTR3", "SSTR3_receptor_8xir"),
        ("7XMS", RAW_DIR / "7XMS.pdb", "R", False, "SSTR4", "SSTR4_receptor_7xms"),
        ("8ZCJ", RAW_DIR / "8ZCJ.pdb", "G", False, "SSTR5", "SSTR5_receptor_8zcj"),
    ]

    results = {}
    for pdb_id, raw_file, rec_chain, is_cif, label, out_stem in off_targets:
        out_path = CURATED_DIR / f"{out_stem}.pdb"
        print(f"\n[{pdb_id}] {label} 큐레이션...")

        if not raw_file.exists():
            print(f"  오류: {raw_file} 없음 — 스킵")
            results[label] = {"status": "SKIP", "reason": f"{raw_file} 없음"}
            continue

        struct, n = _extract_receptor_chain(
            pdb_id, raw_file, rec_chain, out_path, is_cif=is_cif
        )
        print(f"  수용체 추출: {n}잔기 → {out_path}")

        # SSTR2와 구조정렬
        rmsd = _align_to_sstr2(struct, ref_sstr2)
        if rmsd >= 0:
            print(f"  SSTR2 정렬 RMSD: {rmsd:.2f} Å (CA, {min(n, 287)}잔기)")

            # 정렬된 구조 재저장
            class StdResSelect(Select):
                def accept_residue(self, r):
                    return r.id[0] == ' '
            io = PDBIO()
            io.set_structure(struct)
            io.save(str(out_path), StdResSelect())
            print(f"  정렬 후 저장 완료: {out_path}")

        results[label] = {"status": "OK", "n_residues": n, "rmsd_vs_sstr2": rmsd,
                          "out": str(out_path)}

    # ------------------------------------------------------------------
    # Step 3: curated/ 교체 (--overwrite-curated 옵션)
    # ------------------------------------------------------------------
    if args.overwrite_curated:
        print()
        print("=" * 60)
        print("Step 3: curated/ SSTR*.pdb 교체")
        print("=" * 60)
        import shutil

        replacements = [
            (CURATED_DIR / "SSTR2_receptor_7t10.pdb", CURATED_DIR / "SSTR2_receptor.pdb"),
            (CURATED_DIR / "SSTR1_receptor_8xip.pdb", CURATED_DIR / "SSTR1_receptor.pdb"),
            (CURATED_DIR / "SSTR3_receptor_8xir.pdb", CURATED_DIR / "SSTR3_receptor.pdb"),
            (CURATED_DIR / "SSTR4_receptor_7xms.pdb", CURATED_DIR / "SSTR4_receptor.pdb"),
            (CURATED_DIR / "SSTR5_receptor_8zcj.pdb", CURATED_DIR / "SSTR5_receptor.pdb"),
        ]
        for src, dst in replacements:
            if src.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  교체: {src.name} → {dst.name}")
            else:
                print(f"  오류: {src} 없음")
    else:
        print()
        print("  --overwrite-curated 미지정 — curated/ 교체 건너뜀.")
        print("  교체하려면: python scripts/curate_cryo_em_receptors.py --overwrite-curated")

    # ------------------------------------------------------------------
    # 요약
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("큐레이션 요약")
    print("=" * 60)
    print(f"SSTR2 수용체 단독 (7T10 chain R→A): {n_rec}잔기")
    print(f"SSTR2+SST14 복합체 (7T10 R+P→A+B): {n_complex}잔기")
    for label, r in results.items():
        if r["status"] == "OK":
            rmsd_s = f"{r['rmsd_vs_sstr2']:.2f}Å" if r['rmsd_vs_sstr2'] >= 0 else "N/A"
            print(f"{label}: {r['n_residues']}잔기, RMSD vs SSTR2={rmsd_s}")
        else:
            print(f"{label}: {r['status']} ({r.get('reason','')})")


if __name__ == "__main__":
    main()
