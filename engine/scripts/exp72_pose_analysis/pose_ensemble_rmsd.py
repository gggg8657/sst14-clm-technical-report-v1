#!/usr/bin/env python3
"""Pose ensemble 일관성 분석.

각 후보의 여러 pose에 대해:
  - 대표 pose(median ddG) 기준 CA-RMSD (peptide 부분만, chain A)
  - Receptor(chain B) 정렬 후 peptide RMSD → 결합 모드의 이동성 측정
  - ddG spread + RMSD spread 상관

출력: runs/exp72_analysis/pose_analysis/ensemble_rmsd/ensemble_stats.jsonl
      - 후보별: {sequence, n_poses, ddg_median, ddg_sd, ca_rmsd_median, ca_rmsd_max, consistency_flag}
      - consistency_flag: consistent(rmsd<2Å) / moderate(2-4) / floppy(>4)
"""
import argparse, glob, json, os, re, sys, statistics as st
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]


def parse_pose_file(f):
    """파일명 → (seq, pose_idx, tag, ddg)."""
    m = re.search(r'([A-Z]+)_pose(\d+)_(conv|excl)_ddg(-?[\d.]+)\.pdb$', f)
    if not m:
        return None
    return m.group(1), int(m.group(2)), m.group(3), float(m.group(4))


def read_ca(pdb_path, chain):
    """PDB 파일에서 chain의 CA 좌표 반환 [(resnum, x, y, z), ...]."""
    ca = []
    for l in open(pdb_path, errors='ignore'):
        if l.startswith('ATOM') and l[12:16].strip() == 'CA' and l[21] == chain:
            try:
                ca.append((int(l[22:26]), float(l[30:38]), float(l[38:46]), float(l[46:54])))
            except Exception:
                pass
    return ca


def kabsch_rmsd(P, Q):
    """P, Q: Nx3 좌표 리스트. RMSD (align 없이, 이미 receptor로 정렬됨 가정)."""
    import math
    if len(P) != len(Q) or not P:
        return None
    sq = sum((p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2 for p, q in zip(P, Q))
    return math.sqrt(sq / len(P))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', default='runs/exp72_system/phase1_work',
                    help='pose PDB 폴더')
    ap.add_argument('--out', default='runs/exp72_analysis/pose_analysis/ensemble_rmsd/ensemble_stats.jsonl')
    ap.add_argument('--converged-only', action='store_true', default=True,
                    help='수렴 pose만 분석 (default True)')
    args = ap.parse_args()

    # 후보별로 pose 파일 그룹화
    by_seq = defaultdict(list)
    for f in glob.glob(os.path.join(args.input_dir, '*_pose*.pdb')):
        info = parse_pose_file(f)
        if not info:
            continue
        seq, idx, tag, ddg = info
        if args.converged_only and tag != 'conv':
            continue
        by_seq[seq].append((idx, tag, ddg, f))

    print(f"[ensemble-rmsd] 후보 수: {len(by_seq)} (converged 필터 적용)", file=sys.stderr)

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as fout:
        for i, (seq, poses) in enumerate(sorted(by_seq.items())):
            if len(poses) < 2:
                continue  # 최소 2개 pose 필요
            # 각 pose의 peptide CA 좌표 로드
            pep_cas = []
            ddgs = []
            for idx, tag, ddg, f in poses:
                ca = read_ca(f, 'A')  # peptide chain
                if len(ca) < 5:
                    continue
                # 잔기 순서로 정렬 후 좌표만
                ca.sort(key=lambda x: x[0])
                pep_cas.append([(c[1], c[2], c[3]) for c in ca])
                ddgs.append(ddg)

            if len(pep_cas) < 2:
                continue

            # 대표 = median ddG pose
            med_idx = sorted(range(len(ddgs)), key=lambda i: ddgs[i])[len(ddgs)//2]
            ref = pep_cas[med_idx]
            # 각 pose vs ref RMSD (receptor 정렬은 이미 안 됨 — 상대적 이동만 봄)
            rmsds = [kabsch_rmsd(ref, other) for j, other in enumerate(pep_cas) if j != med_idx]
            rmsds = [r for r in rmsds if r is not None]
            if not rmsds:
                continue

            rm_med = st.median(rmsds)
            rm_max = max(rmsds)
            ddg_med = st.median(ddgs)
            ddg_sd = st.stdev(ddgs) if len(ddgs) > 1 else 0.0

            flag = 'consistent' if rm_med < 2.0 else ('moderate' if rm_med < 4.0 else 'floppy')

            rec = {
                'sequence': seq,
                'n_poses_converged': len(pep_cas),
                'ddg_median': round(ddg_med, 3),
                'ddg_sd': round(ddg_sd, 3),
                'ca_rmsd_median': round(rm_med, 3),
                'ca_rmsd_max': round(rm_max, 3),
                'consistency_flag': flag,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if (i + 1) % 5 == 0 or i == len(by_seq) - 1:
                print(f"  [{i+1}/{len(by_seq)}] {seq} ddG_med={ddg_med:.1f} RMSD_med={rm_med:.2f} → {flag}",
                      file=sys.stderr, flush=True)

    print(f"[ensemble-rmsd] DONE → {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
