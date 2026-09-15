#!/usr/bin/env python3
"""Tier 1 18 후보에 대해 backbone-flexible 재도킹.

-flexPepDocking:lowres_preoptimize 플래그 → 저해상 backbone 최적화 + 고해상 refine.
peptide backbone(CA)가 실제로 이동한다 (기존 pep_refine 은 side-chain만).

목적: '자세(pose) 다양성' 실제 탐색. 이번 실험이 답 못한 backbone level 결합 자세를 확인.
비용: ~10 min/pose × nstruct 3 × 18 = ~9 CPU hours, 8-shard 병렬 → ~1.5h wall.
출력: runs/exp72_analysis/bbflex_tier1.jsonl + poses
"""
import argparse, json, os, sys, time, glob, re
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]

def load_tier1_seqs():
    """pool_selectivity.jsonl에서 Tier 1 = 강결합(onT<-30) + 아형선택(신호>+10) 산출."""
    p = REPO / 'runs/exp72_analysis/pool_selectivity.jsonl'
    recs = [json.loads(l) for l in p.open() if l.strip()]
    seen = {}
    for r in recs:
        if r.get('delta_margin') is not None:
            seen[r['sequence']] = r
    recs = list(seen.values())
    nat = [r for r in recs if r['arm'] == 'native'][0]
    nat_m = nat['delta_margin']
    tier1 = []
    for r in recs:
        if r['arm'] == 'native': continue
        sig = r['delta_margin'] - nat_m
        if r['on_target_ddg'] < -30 and sig > 10:
            tier1.append((r['sequence'], r['on_target_ddg']))
    tier1.sort(key=lambda x: x[1])  # 강결합 우선 정렬
    return [s for s, _ in tier1]


def load_pool_seqs():
    """pool_redock.jsonl에서 풀 전체 unique 서열 (강결합 우선 정렬)."""
    from collections import OrderedDict
    p = REPO / 'runs/exp72_analysis/pool_redock.jsonl'
    best = {}  # 서열 → 최소 ddg (dedup + 정렬용)
    for l in p.open():
        try:
            r = json.loads(l)
            s = r.get('sequence'); m = r.get('ddg_median')
            if s and m is not None:
                if s not in best or m < best[s]: best[s] = m
        except Exception:
            pass
    items = sorted(best.items(), key=lambda x: x[1])  # 강한 결합 우선
    return [s for s, _ in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nstruct', type=int, default=int(os.environ.get('FLEXPEP_NSTRUCT', '3')))
    ap.add_argument('--out', default='runs/exp72_analysis/bbflex_tier1.jsonl')
    ap.add_argument('--pose-dir', default='runs/exp72_analysis/bbflex_poses')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--winners', default='_workspace/EXP72_POOL_WINNERS.json')
    ap.add_argument('--source', choices=['tier1', 'pool'], default='tier1',
                    help='tier1(18) or pool(2552 unique)')
    args = ap.parse_args()

    if args.source == 'pool':
        seqs = load_pool_seqs()
    else:
        seqs = load_tier1_seqs()
    # 재개 필터
    done = set()
    out_path = REPO / args.out
    if out_path.exists():
        for l in out_path.open():
            try:
                r = json.loads(l)
                if r.get('status') == 'ok':
                    done.add(r['sequence'])
            except Exception:
                pass
    mine = [s for i, s in enumerate(seqs) if i % args.shard_count == args.shard_index and s not in done]
    print(f"[bbflex shard {args.shard_index}/{args.shard_count}] 처리 {len(mine)} 후보 "
          f"(nstruct {args.nstruct}, 전체 {len(seqs)}, 완료 {len(done)})", file=sys.stderr)

    pose_dir = REPO / args.pose_dir
    pose_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # PyRosetta with backbone-flexible option
    import pyrosetta
    pyrosetta.init(
        '-mute all -ex1 -ex2aro -ignore_unrecognized_res '
        '-flexPepDocking:lowres_preoptimize '
        '-flexPepDocking:pep_refine '
        '-constraints:cst_fa_weight 1.0'
    )
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
    sys.path.insert(0, str(REPO))
    from AG_src.scripts.flexpep_dock import (
        _find_peptide_cys_residues, _add_disulfide_constraint,
        _check_disulfide_distance, compute_interface_ddg,
    )
    from pyrosetta_flow.pose_chain_utils import detect_peptide_chain

    # winner PDB 로드 (pool_work 대표 복합체)
    for seq in mine:
        pdb_src = REPO / f'runs/exp72_analysis/pool_work/{seq}.pdb'
        if not pdb_src.exists():
            print(f"  {seq}: source PDB 없음, skip", file=sys.stderr); continue

        ddgs_conv = []
        n_unphysical = 0
        pose_files = []
        ca_disp_max = 0.0
        t_start = time.time()

        # nstruct 반복
        for i in range(args.nstruct):
            try:
                pose = pyrosetta.pose_from_pdb(str(pdb_src))
                peptide_chain = detect_peptide_chain(pose, len(seq))
                # CA 좌표 원본(peptide) 저장 → 이후 변위 측정
                ca_before = []
                for k in range(1, pose.total_residue()+1):
                    if pose.pdb_info().chain(k) == peptide_chain:
                        xyz = pose.residue(k).xyz('CA')
                        ca_before.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

                # SS bond
                cys = _find_peptide_cys_residues(pose)
                if len(cys) == 2:
                    try: pose.conformation().detect_disulfides()
                    except Exception:
                        try: _add_disulfide_constraint(pose, cys[0], cys[1])
                        except Exception: pass

                # BB-flex apply
                fpd = FlexPepDockingProtocol()
                fpd.apply(pose)
                ddg = compute_interface_ddg(pose)
                converged = ddg < 0

                # CA 변위 측정 (peptide backbone 실제로 움직였나 검증)
                ca_after = []
                for k in range(1, pose.total_residue()+1):
                    if pose.pdb_info().chain(k) == peptide_chain:
                        xyz = pose.residue(k).xyz('CA')
                        ca_after.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
                if len(ca_after) == len(ca_before) and ca_before:
                    import math
                    disps = [math.sqrt(sum((a[j]-b[j])**2 for j in range(3))) for a, b in zip(ca_after, ca_before)]
                    ca_disp_max = max(ca_disp_max, max(disps))

                tag = 'conv' if converged else 'excl'
                pose_out = pose_dir / f'{seq}_bbflex{i+1:02d}_{tag}_ddg{ddg:.2f}.pdb'
                pose.dump_pdb(str(pose_out))
                pose_files.append(str(pose_out))
                if converged: ddgs_conv.append(ddg)
                else: n_unphysical += 1
                print(f"  [bbflex {i+1}/{args.nstruct}] {seq} ddG={ddg:.2f} ({tag}) CA_disp_max={ca_disp_max:.2f}Å",
                      file=sys.stderr, flush=True)
            except Exception as exc:
                print(f"  [bbflex {i+1}/{args.nstruct}] {seq} FAIL: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

        # 통계
        import statistics as st
        rec = {
            'sequence': seq, 'nstruct': args.nstruct,
            'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'elapsed_s': round(time.time() - t_start, 1),
            'ddg_median': round(st.median(ddgs_conv), 3) if ddgs_conv else None,
            'ddg_mean': round(st.mean(ddgs_conv), 3) if ddgs_conv else None,
            'ddg_sd': round(st.stdev(ddgs_conv), 3) if len(ddgs_conv) > 1 else 0.0,
            'ddg_min': round(min(ddgs_conv), 3) if ddgs_conv else None,
            'n_converged': len(ddgs_conv), 'n_unphysical': n_unphysical,
            'ca_disp_max': round(ca_disp_max, 3),  # peptide backbone 실제 이동 최대 (진짜 BB-flex 검증)
            'poses': pose_files, 'status': 'ok',
        }
        with out_path.open('a') as fout:
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"  ✓ {seq} med={rec['ddg_median']} sd={rec['ddg_sd']} CA_disp={rec['ca_disp_max']}Å "
              f"({rec['elapsed_s']}s)", file=sys.stderr, flush=True)

    print(f"[bbflex shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
