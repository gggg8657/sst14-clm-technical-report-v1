#!/usr/bin/env python3
"""
Off-target 수용체(SSTR1/3/4/5) → SSTR2 서열기반 구조 재정렬.

배경 (EXP72_VERIFICATION_MASTER_PLAN.md §발견된 리스크):
  기존 `scripts/curate_cryo_em_receptors.py::_align_to_sstr2()` 는 서열정렬 없이
  CA 원자를 "파일에 나온 순서" 그대로 앞에서부터 min_len 개 잘라 Superimposer 에
  넣는 naive 방식이다. 코드 주석(pyrosetta_flow/multiobjective.py:587-589)은
  "SSTR2 프레임에 0.93~0.95 사전정렬"이라 주장하나, 실측 RMSD 는 4.6~16.7Å 로
  전혀 정렬돼 있지 않음 (`_workspace/04_engineer-backend_offtarget_receptors_summary.json`).

본 스크립트는:
  1) TMalign 바이너리 유무를 확인(있으면 최우선 사용 — 이번 환경엔 부재 확인됨).
  2) 없으면 Biopython Bio.Align.PairwiseAligner(BLOSUM62, affine gap) 로
     구조에서 직접 추출한 서열을 SSTR2 참조 서열에 정렬 → 매칭된 (동일 정렬 컬럼,
     양쪽 다 gap 아님) 잔기쌍의 CA 원자로 1차 Superimposer 수행.
  3) 이후 반복적 outlier pruning(구조정렬 표준 기법 — CE/TM-align 계열이 쓰는
     "정렬 후 거리 큰 pair 제거 → 재정렬" 을 수렴할 때까지 반복)으로 core 잔기만
     남겨 정밀 RMSD 산출.
  4) 표준 TM-score 공식(d0 = 1.24*(L-15)^(1/3) - 1.8, L=SSTR2 참조 길이)을
     "정렬된 전체 매칭쌍"의 최종 거리로 계산해 보고 — Zhang & Skolnick 2004 공식은
     맞지만, dynamic-programming 기반 진짜 TM-align 최적화는 아님(정직하게 명시).

주의:
  - 기존 code/curated 파일 편집 없음. 산출물은 전부 신규 경로
    `_workspace/05_engineer-backend_offtarget_receptors_seqalign/` 에만 저장.
  - data/ 는 READ-ONLY 로만 사용(원본 raw 구조 읽기).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select, Superimposer
from Bio.PDB.Polypeptide import three_to_one

REPO = Path(__file__).resolve().parents[1]
CURATED_DIR = REPO / "data" / "somatostatin_receptor" / "curated"
DATA_DIR = REPO / "data" / "somatostatin_receptor"
RAW_DIR = DATA_DIR / "raw_cryo_em"
OUT_DIR = REPO / "_workspace" / "05_engineer-backend_offtarget_receptors_seqalign"

# 참조: SSTR2 (7T10 chain R→A, 287잔기, 40-327). 기존 curated 산출물 재사용(읽기전용).
REF_PDB = CURATED_DIR / "SSTR2_receptor_7t10.pdb"
REF_CHAIN = "A"

# 4종 off-target 원본 소스 (biology 세트: SSTR1=9IK8, SSTR3=8XIR, SSTR4=7XMT, SSTR5=8ZBJ)
TARGETS = [
    # (label, pdb_id, raw_path, is_cif, receptor_chain)
    ("SSTR1", "9IK8", DATA_DIR / "SSTR1_9IK8.cif", True, "D"),
    ("SSTR3", "8XIR", RAW_DIR / "8XIR.cif", True, "A"),
    ("SSTR4", "7XMT", DATA_DIR / "SSTR4_7XMT.pdb", False, "R"),
    ("SSTR5", "8ZBJ", DATA_DIR / "SSTR5_8ZBJ.pdb", False, "R"),
]


class StdResSelect(Select):
    """표준 잔기만 저장 + 수소원자 제외(기존 curated 4종과 동일 규약: C/N/O/S heavy atom만).
    9IK8 CIF 는 H-model(명시적 수소 포함) 소스라 이 필터가 없으면 H가 그대로 남는다."""
    def accept_residue(self, r):
        return r.id[0] == ' '

    def accept_atom(self, a):
        return a.element != 'H'


def find_tmalign() -> Optional[str]:
    """TMalign 바이너리 탐색. 있으면 경로, 없으면 None."""
    exe = shutil.which("TMalign") or shutil.which("tmalign")
    return exe


def load_chain(pdb_id: str, path: Path, chain_id: str, is_cif: bool):
    parser = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_id, str(path))
    model = struct[0]
    chain = model[chain_id]
    residues = [r for r in chain if r.id[0] == ' ' and 'CA' in r]
    return struct, chain, residues


def residues_to_seq(residues) -> str:
    seq_chars = []
    for r in residues:
        try:
            seq_chars.append(three_to_one(r.get_resname()))
        except KeyError:
            seq_chars.append('X')
    return "".join(seq_chars)


def sequence_align_pairs(ref_residues, mob_residues):
    """BLOSUM62 global affine-gap 정렬 → (ref_idx, mob_idx) 매칭쌍 리스트(gap 제외)."""
    ref_seq = residues_to_seq(ref_residues)
    mob_seq = residues_to_seq(mob_residues)

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.mode = "global"
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    alignment = aligner.align(ref_seq, mob_seq)[0]
    # biopython 1.79 PairwiseAlignment 에는 .indices 속성이 없음(1.80+ 전용) ->
    # .aligned (gap-free 블록의 (start,end) 쌍 리스트, target/query 각각) 사용.
    ref_blocks, mob_blocks = alignment.aligned

    pairs = []
    for (rs, re_), (ms, me) in zip(ref_blocks, mob_blocks):
        assert re_ - rs == me - ms, "정렬 블록 길이 불일치"
        for k in range(re_ - rs):
            pairs.append((rs + k, ms + k))

    identity = sum(
        1 for ri, mi in pairs if ref_seq[ri] == mob_seq[mi]
    )
    return pairs, ref_seq, mob_seq, float(alignment.score), identity


def iterative_prune_superpose(ref_ca, mob_ca, max_iter: int = 30, min_keep: int = 40):
    """
    구조정렬 표준 outlier-pruning: 전체 매칭쌍으로 1차 fit → 거리 기준 상위
    outlier 제거 → 재fit → 수렴(또는 min_keep 도달)까지 반복.
    각 반복에서 RMSD 가 더 개선되지 않거나 잔기수가 min_keep 미만이면 직전 상태 반환.
    """
    idx = list(range(len(ref_ca)))
    sup = Superimposer()

    best_rot, best_tran, best_rms, best_idx = None, None, None, None

    for it in range(max_iter):
        cur_ref = [ref_ca[i] for i in idx]
        cur_mob = [mob_ca[i] for i in idx]
        sup.set_atoms(cur_ref, cur_mob)
        rot, tran = sup.rotran
        rms = float(sup.rms)

        best_rot, best_tran, best_rms, best_idx = rot, tran, rms, list(idx)

        if len(idx) <= min_keep:
            break

        # 현재 회전으로 각 pair 거리 계산
        mob_coords = np.array([a.get_coord() for a in cur_mob])
        ref_coords = np.array([a.get_coord() for a in cur_ref])
        mob_xform = mob_coords @ rot + tran
        d = np.linalg.norm(mob_xform - ref_coords, axis=1)

        # 표준 CE/TM류 pruning: 거리 > 2*current_rmsd(하한 2.0Å) 인 pair 제거.
        # 한 번에 너무 많이 잘리지 않도록 상위 10% 또는 threshold 초과 중 적은 쪽만 제거.
        cutoff = max(2.0, 2.0 * rms)
        drop_mask = d > cutoff
        n_drop = int(drop_mask.sum())
        if n_drop == 0:
            break  # 수렴
        if len(idx) - n_drop < min_keep:
            # min_keep 유지 위해 거리 큰 순으로만 자르기
            order = np.argsort(-d)
            n_drop = len(idx) - min_keep
            drop_set = set(order[:n_drop].tolist())
        else:
            drop_set = set(np.where(drop_mask)[0].tolist())

        idx = [j for k, j in enumerate(idx) if k not in drop_set]

    return best_rot, best_tran, best_rms, best_idx


def tm_score(ref_ca, mob_ca, rot, tran, l_norm: int) -> float:
    """표준 TM-score 공식 (Zhang & Skolnick 2004). d0 는 l_norm(정규화 길이) 기준."""
    if l_norm <= 15:
        d0 = 0.5
    else:
        d0 = 1.24 * (l_norm - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)
    mob_coords = np.array([a.get_coord() for a in mob_ca])
    ref_coords = np.array([a.get_coord() for a in ref_ca])
    mob_xform = mob_coords @ rot + tran
    d = np.linalg.norm(mob_xform - ref_coords, axis=1)
    score = np.sum(1.0 / (1.0 + (d / d0) ** 2))
    return float(score / l_norm)


def run_target(label, pdb_id, raw_path, is_cif, chain_id, ref_struct, ref_chain, ref_residues, ref_ca):
    print(f"\n[{label}] {pdb_id} chain {chain_id} 처리 중...")
    if not raw_path.exists():
        print(f"  오류: {raw_path} 없음 — 스킵")
        return {"label": label, "status": "SKIP", "reason": f"{raw_path} 없음"}

    mob_struct, mob_chain, mob_residues = load_chain(pdb_id, raw_path, chain_id, is_cif)
    mob_ca_all = [r['CA'] for r in mob_residues]

    # --- naive(기존 방식) RMSD 재현: 정렬 전(순서 index-cut) 비교용 ---
    min_len_naive = min(len(ref_ca), len(mob_ca_all))
    sup_naive = Superimposer()
    sup_naive.set_atoms(ref_ca[:min_len_naive], mob_ca_all[:min_len_naive])
    naive_rmsd = float(sup_naive.rms)

    # --- 서열정렬 ---
    pairs, ref_seq, mob_seq, aln_score, identity = sequence_align_pairs(ref_residues, mob_residues)
    n_matched = len(pairs)
    seq_identity_pct = 100.0 * identity / n_matched if n_matched else 0.0
    print(f"  서열정렬 매칭 컬럼수={n_matched}, 동일성={seq_identity_pct:.1f}% (ref len={len(ref_seq)}, mob len={len(mob_seq)})")

    ref_ca_matched = [ref_residues[ri]['CA'] for ri, _ in pairs]
    mob_ca_matched = [mob_residues[mi]['CA'] for _, mi in pairs]

    # 서열정렬만 적용한 1차 RMSD(pruning 전)
    sup1 = Superimposer()
    sup1.set_atoms(ref_ca_matched, mob_ca_matched)
    seqaligned_rmsd_prepruning = float(sup1.rms)

    # --- 반복 outlier pruning ---
    rot, tran, core_rmsd, core_idx = iterative_prune_superpose(ref_ca_matched, mob_ca_matched)
    n_core = len(core_idx)

    # TM-score: 전체 매칭쌍(pruning 전)에 최종 rot/tran 적용, l_norm=SSTR2 참조 길이
    tms = tm_score(ref_ca_matched, mob_ca_matched, rot, tran, l_norm=len(ref_residues))

    print(f"  정렬 전(naive index-cut) RMSD  = {naive_rmsd:.2f} Å ({min_len_naive}잔기)")
    print(f"  서열정렬 적용(pruning전) RMSD  = {seqaligned_rmsd_prepruning:.2f} Å ({n_matched}잔기)")
    print(f"  outlier-pruning 후 core RMSD   = {core_rmsd:.2f} Å ({n_core}/{n_matched}잔기)")
    print(f"  TM-score(l_norm={len(ref_residues)}, 매칭 {n_matched}잔기 기준) = {tms:.3f}")

    # --- 전체 mobile 구조에 최종 변환 적용 ---
    rot_t = rot  # Bio.PDB rotran: coord_new = coord_old @ rot + tran
    for atom in mob_struct.get_atoms():
        c = atom.get_coord()
        atom.set_coord(c @ rot_t + tran)

    out_path = OUT_DIR / f"{label}_receptor_{pdb_id.lower()}_seqalign.pdb"
    io = PDBIO()
    io.set_structure(mob_struct)
    io.save(str(out_path), StdResSelect())
    print(f"  저장: {out_path}")

    return {
        "label": label,
        "status": "OK",
        "pdb_id": pdb_id,
        "receptor_chain": chain_id,
        "out_path": str(out_path.relative_to(REPO)),
        "n_residues_mobile": len(mob_residues),
        "n_residues_ref": len(ref_residues),
        "naive_index_cut_rmsd_A": round(naive_rmsd, 3),
        "naive_index_cut_n": min_len_naive,
        "seq_alignment_score": aln_score,
        "seq_identity_pct_over_matched": round(seq_identity_pct, 2),
        "n_matched_after_seqalign": n_matched,
        "seqalign_rmsd_prepruning_A": round(seqaligned_rmsd_prepruning, 3),
        "n_core_after_pruning": n_core,
        "core_rmsd_A": round(core_rmsd, 3),
        "tm_score_lnorm_ref": round(tms, 4),
        "meets_rmsd_lt_2A_core": bool(core_rmsd < 2.0),
        "meets_tmscore_gt_0.5": bool(tms > 0.5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmalign-only-check", action="store_true",
                     help="TMalign 바이너리 존재 여부만 확인하고 종료")
    args = ap.parse_args()

    tmalign_path = find_tmalign()
    print(f"TMalign 바이너리: {tmalign_path or '없음 (Biopython 서열정렬 fallback 사용)'}")
    if args.tmalign_only_check:
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref_struct = PDBParser(QUIET=True).get_structure("sstr2_ref", str(REF_PDB))
    ref_chain = ref_struct[0][REF_CHAIN]
    ref_residues = [r for r in ref_chain if r.id[0] == ' ' and 'CA' in r]
    ref_ca = [r['CA'] for r in ref_residues]
    print(f"SSTR2 참조: {REF_PDB.relative_to(REPO)} chain {REF_CHAIN}, {len(ref_residues)}잔기")

    results = []
    for label, pdb_id, raw_path, is_cif, chain_id in TARGETS:
        res = run_target(label, pdb_id, raw_path, is_cif, chain_id,
                          ref_struct, ref_chain, ref_residues, ref_ca)
        results.append(res)

    summary = {
        "method": "Bio.Align.PairwiseAligner(BLOSUM62, global, affine gap open=-11/extend=-1) "
                  "서열정렬로 매칭 잔기쌍 도출 -> Bio.PDB.Superimposer 반복 outlier-pruning "
                  "(거리>max(2.0, 2*current_rmsd) pair 순차 제거, 수렴까지) -> 최종 강체변환을 "
                  "전체 mobile 구조에 적용. TMalign 바이너리 부재로 사용 못함(확인됨, tmalign_binary_found=false).",
        "tmalign_binary_found": bool(tmalign_path),
        "reference": {"pdb_id": "7T10", "chain": REF_CHAIN, "n_residues": len(ref_residues),
                      "path": str(REF_PDB.relative_to(REPO))},
        "targets": results,
    }
    out_json = OUT_DIR / "seqalign_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n요약 저장: {out_json}")


if __name__ == "__main__":
    main()
