#!/usr/bin/env python3
"""주말 자율 실험 - Random pose sampling (LHS).

winner 533에 대해 rigid 6D 공간을 Latin Hypercube로 30 pose씩 균등 샘플링 → refine.
총 533 × 30 = 15,990 pose 대규모 데이터셋 생산.

목적:
- XGBoost/DeepONet surrogate model 훈련 데이터
- Pose landscape 균등 커버 (BO의 지적 탐색과 상보)
- 후속 AutoML-BO의 warm-start

세션독립: setsid + 파일기반 재개.
"""
import argparse, json, os, sys, time, math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRANS_RANGE = 5.0   # ±5Å
ROT_RANGE = 30.0    # ±30°


def latin_hypercube(n_samples, n_dim, seed=42):
    """간단한 LHS 구현 (scipy.stats.qmc 없이도)."""
    import random
    random.seed(seed)
    result = []
    # 각 차원 [0,1] n_samples 구간 균등 → shuffle
    grids = []
    for d in range(n_dim):
        segs = [(i + random.random()) / n_samples for i in range(n_samples)]
        random.shuffle(segs)
        grids.append(segs)
    for i in range(n_samples):
        result.append([grids[d][i] for d in range(n_dim)])
    return result


def lhs_to_6d(lhs_point):
    """LHS [0,1]^6 → 물리 좌표계 (dx,dy,dz,rx,ry,rz)."""
    return [
        (lhs_point[0] - 0.5) * 2 * TRANS_RANGE,   # dx ∈ [-5, 5]
        (lhs_point[1] - 0.5) * 2 * TRANS_RANGE,   # dy
        (lhs_point[2] - 0.5) * 2 * TRANS_RANGE,   # dz
        (lhs_point[3] - 0.5) * 2 * ROT_RANGE,     # rx ∈ [-30, 30]
        (lhs_point[4] - 0.5) * 2 * ROT_RANGE,     # ry
        (lhs_point[5] - 0.5) * 2 * ROT_RANGE,     # rz
    ]


def load_winner_seqs():
    """Winner 533 로드 (pool_selectivity.jsonl에서 native 능가 + 이황정상 + selectivity 조건 모두)."""
    p = REPO / 'runs/exp72_analysis/pool_selectivity.jsonl'
    seqs = []
    for l in p.open():
        try:
            r = json.loads(l)
            if r.get('arm') != 'native' and r.get('delta_margin') is not None:
                seqs.append(r['sequence'])
        except Exception:
            pass
    return sorted(set(seqs))


def process_one(seq, source_pdb, work_dir, n_pose, out_path):
    """한 서열에 대해 LHS n_pose개 샘플링 + refine + 기록."""
    sys.path.insert(0, str(REPO / 'scripts'))
    from exp72_pose_bo_rigid import build_pose_rigid, score_pose

    # 재개: 이미 처리한 (seq, pose_idx) 스킵
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for l in f:
                try:
                    d = json.loads(l)
                    if d.get('sequence') == seq and d.get('pose_idx') is not None:
                        done.add(d['pose_idx'])
                except Exception:
                    pass

    # LHS 6D 30 pose (seed=hash(seq) → 서열별 다른 샘플)
    seed = hash(seq) & 0xFFFF
    lhs_points = latin_hypercube(n_pose, 6, seed=seed)

    for pose_idx, lhs in enumerate(lhs_points):
        if pose_idx in done:
            continue
        X = lhs_to_6d(lhs)
        tmp_pdb = f"{work_dir}/{seq}_lhs{pose_idx:03d}_init.pdb"
        refined_pdb = f"{work_dir}/{seq}_lhs{pose_idx:03d}_refined.pdb"
        rec = {'sequence': seq, 'pose_idx': pose_idx, 'X': X,
               'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            pose = build_pose_rigid(source_pdb, X, tmp_pdb)
            scores = score_pose(pose, refined_pdb, run_mmgbsa=False)
            rec.update(scores)
            rec['status'] = 'ok'
        except Exception as exc:
            rec['status'] = 'error'
            rec['error'] = f"{type(exc).__name__}: {exc}"
        with out_path.open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        print(f"  {seq} lhs{pose_idx+1:2d}/{n_pose} ddG={rec.get('ddg_flex')} SS={rec.get('ss_dist')}",
              file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-pose', type=int, default=30)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=32)
    ap.add_argument('--work-dir', default='runs/exp72_analysis/lhs_pose_data')
    ap.add_argument('--source-pdb-dir', default='runs/exp72_analysis/pool_work')
    args = ap.parse_args()

    import pyrosetta
    pyrosetta.init('-mute all -ex1 -ex2aro -ignore_unrecognized_res '
                   '-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0')

    all_seqs = load_winner_seqs()
    mine = [s for i, s in enumerate(all_seqs) if i % args.shard_count == args.shard_index]
    print(f"[lhs-sampling shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(all_seqs)}",
          file=sys.stderr)

    work = REPO / args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    out_path = work / f'lhs_pose_shard_{args.shard_index:02d}.jsonl'

    for seq in mine:
        src = REPO / args.source_pdb_dir / f'{seq}.pdb'
        if not src.exists():
            src = REPO / 'runs/exp72_system/phase1_work' / f'{seq}.pdb'
        if not src.exists():
            print(f"  {seq}: PDB 없음, skip", file=sys.stderr)
            continue
        process_one(seq, str(src), str(work), args.n_pose, out_path)

    print(f"[lhs shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
