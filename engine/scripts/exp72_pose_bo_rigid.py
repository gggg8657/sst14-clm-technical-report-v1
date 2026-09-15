#!/usr/bin/env python3
"""Rigid Pose BO — peptide 전체를 rigid body로 receptor 대비 이동/회전.

X ∈ R^6: (dx, dy, dz, rx, ry, rz)  — 도 단위 회전각, Å 단위 이동거리
  · 이동 범위: ±5 Å (포켓 내 탐색)
  · 회전 범위: ±30°  (극단 뒤집기 배제)
Y = 0.5·norm(ddG_flex) + 0.3·norm(dg_bind_mmgbsa) + 0.1·SS + 0.1·FWKT

SS bond·내부 구조 완전 안전 (peptide는 rigid하게만 이동/회전).
FlexPepDock refine 이 새 위치에서 side-chain 재정렬 + 스코어 산출.

세션독립: setsid + 파일기반 재개.
"""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

W_DDG, W_MM, W_SS, W_FWKT = 0.5, 0.3, 0.1, 0.1
DDG_REF, DDG_SCALE = 0.0, 50.0
MM_REF,  MM_SCALE  = 0.0, 100.0
TRANS_RANGE = 5.0   # ±5Å
ROT_RANGE   = 30.0  # ±30°
MMGBSA_PYTHON = '[LOCAL_PATH]'


def normalize_negative(x, ref, scale):
    if x is None or (isinstance(x, float) and x != x): return 0.0
    v = (ref - x) / scale
    return max(0.0, min(1.0, v))


def rigid_transform_peptide(pose, dx, dy, dz, rx, ry, rz, chain=None):
    """Peptide chain을 rigid body로 이동 + Euler rotation(rx,ry,rz도) 적용."""
    import pyrosetta
    from pyrosetta.rosetta.numeric import xyzVector_double_t, xyzMatrix_double_t
    from pyrosetta_flow.pose_chain_utils import detect_peptide_chain

    chain = chain or detect_peptide_chain(pose, 14)

    # Euler → 회전 행렬 (도→라디안)
    rx_r, ry_r, rz_r = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(rx_r), math.sin(rx_r)
    cy, sy = math.cos(ry_r), math.sin(ry_r)
    cz, sz = math.cos(rz_r), math.sin(rz_r)
    # R = Rz @ Ry @ Rx
    R = [
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy,   sx*cy,            cx*cy],
    ]

    # peptide COM 계산 (회전 중심)
    com = [0.0, 0.0, 0.0]; n = 0
    for i in range(1, pose.total_residue()+1):
        if pose.pdb_info().chain(i) == chain:
            for j in range(1, pose.residue(i).natoms()+1):
                xyz = pose.residue(i).xyz(j)
                com[0] += xyz[0]; com[1] += xyz[1]; com[2] += xyz[2]
                n += 1
    if n == 0: return pose
    com = [c/n for c in com]

    # 각 원자에 R (about COM) + translation 적용
    for i in range(1, pose.total_residue()+1):
        if pose.pdb_info().chain(i) == chain:
            for j in range(1, pose.residue(i).natoms()+1):
                xyz = pose.residue(i).xyz(j)
                # 중심 이동
                v = [xyz[0]-com[0], xyz[1]-com[1], xyz[2]-com[2]]
                # 회전
                vr = [
                    R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
                    R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
                    R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
                ]
                # 복귀 + translation
                new_xyz = xyzVector_double_t(vr[0] + com[0] + dx,
                                              vr[1] + com[1] + dy,
                                              vr[2] + com[2] + dz)
                pose.residue(i).atom(j).xyz(new_xyz)
    return pose


def build_pose_rigid(source_pdb, X, out_pdb):
    """source_pdb 로드 → peptide rigid transform 적용 → out_pdb 저장."""
    import pyrosetta
    pose = pyrosetta.pose_from_pdb(source_pdb)
    rigid_transform_peptide(pose, *X)
    pose.dump_pdb(out_pdb)
    return pose


def run_mmgbsa_rescore_subprocess(refined_pdb, peptide_chain, receptor_chains):
    wrapper_code = r"""
import contextlib
import json
import sys

with contextlib.redirect_stdout(sys.stderr):
    from pyrosetta_flow.mmgbsa_rescore import mmgbsa_rescore
    mm_result = mmgbsa_rescore(
        sys.argv[1], receptor_chains=json.loads(sys.argv[3]), peptide_chain=sys.argv[2]
    )

print(json.dumps(mm_result or {}))
"""
    proc = subprocess.run(
        [MMGBSA_PYTHON, '-c', wrapper_code, refined_pdb, peptide_chain, json.dumps(receptor_chains)],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None, proc.stderr.strip() or proc.stdout.strip()
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as exc:
        msg = proc.stderr.strip() or proc.stdout.strip()
        return None, msg or f"{type(exc).__name__}: {exc}"


def score_pose(pose, refined_pdb, run_mmgbsa=True):
    from AG_src.scripts.flexpep_dock import run_flexpep_refine_pose
    from pyrosetta_flow.pose_chain_utils import chain_lengths, detect_peptide_chain
    result = {}
    peptide_chain = detect_peptide_chain(pose, 14)
    receptor_chains = [chain for chain in chain_lengths(pose) if chain != peptide_chain]
    try:
        _, info = run_flexpep_refine_pose(pose, refined_pdb, nstruct=1)
        result['ddg_flex'] = info.get('ddg_median') or info.get('ddg')
        result['ss_dist'] = info.get('sg_sg_distance')
        result['ss_ok'] = 1.0 if info.get('disulfide_intact') else 0.0
    except Exception as exc:
        result['ddg_flex'] = None
        result['ss_ok'] = 0.0
        result['error_flex'] = f"{type(exc).__name__}: {exc}"
        return result

    # FWKT 접촉
    try:
        pose2 = pose
        cas = {}
        for i in range(1, pose2.total_residue()+1):
            ch = pose2.pdb_info().chain(i)
            if pose2.residue(i).has('CA'):
                cas.setdefault(ch, []).append((pose2.pdb_info().number(i), pose2.residue(i).xyz('CA')))
        min_dist = 999.0
        for pep_resnum, pep_xyz in cas.get(peptide_chain, []):
            if pep_resnum not in (7,8,9,10): continue
            for receptor_chain in receptor_chains:
                for rec_resnum, rec_xyz in cas.get(receptor_chain, []):
                    d = ((pep_xyz[0]-rec_xyz[0])**2 + (pep_xyz[1]-rec_xyz[1])**2 + (pep_xyz[2]-rec_xyz[2])**2) ** 0.5
                    if d < min_dist: min_dist = d
        result['fwkt_dist'] = round(min_dist, 3)
        result['fwkt_ok'] = 1.0 if min_dist <= 8.0 else 0.0
    except Exception:
        result['fwkt_ok'] = 0.0; result['fwkt_dist'] = None

    if run_mmgbsa:
        try:
            mm_result, error_mm = run_mmgbsa_rescore_subprocess(
                refined_pdb, peptide_chain, receptor_chains
            )
            if error_mm:
                result['dg_bind_mm'] = None
                result['error_mm'] = error_mm
                return result
            result['dg_bind_mm'] = mm_result.get('dg_bind') if mm_result else None
        except Exception as exc:
            result['dg_bind_mm'] = None
            result['error_mm'] = f"{type(exc).__name__}: {exc}"
    else:
        result['dg_bind_mm'] = None
    return result


def objective(X, seq, source_pdb, work_dir, iter_num, mm_gbsa=True):
    tmp = f"{work_dir}/{seq}_bo{iter_num:03d}_init.pdb"
    refined = f"{work_dir}/{seq}_bo{iter_num:03d}_refined.pdb"
    try:
        pose = build_pose_rigid(source_pdb, X, tmp)
        scores = score_pose(pose, refined, run_mmgbsa=mm_gbsa)
    except Exception as exc:
        return (1.0, {'X': list(X), 'error': f"{type(exc).__name__}: {exc}"})

    s_ddg = normalize_negative(scores.get('ddg_flex'), DDG_REF, DDG_SCALE)
    s_mm  = normalize_negative(scores.get('dg_bind_mm'), MM_REF, MM_SCALE)
    s_ss  = scores.get('ss_ok', 0.0)
    s_fwkt = scores.get('fwkt_ok', 0.0)
    f = W_DDG*s_ddg + W_MM*s_mm + W_SS*s_ss + W_FWKT*s_fwkt
    scores.update({'X': list(X), 'f': round(f,4), 'iter': iter_num, 'refined_pdb': refined})
    return (-f, scores)


def load_tier1_seqs():
    p = REPO / 'runs/exp72_analysis/pool_selectivity.jsonl'
    seen = {}
    for l in p.open():
        try: r = json.loads(l)
        except: continue
        if r.get('delta_margin') is not None: seen[r['sequence']] = r
    recs = list(seen.values())
    nat = [r for r in recs if r['arm']=='native'][0]; nat_m = nat['delta_margin']
    t1 = []
    for r in recs:
        if r['arm']=='native': continue
        sig = r['delta_margin'] - nat_m
        if r['on_target_ddg']<-30 and sig>10: t1.append((r['sequence'], r['on_target_ddg']))
    t1.sort(key=lambda x:x[1])
    return [s for s,_ in t1]


def run_bo_for_seq(seq, source_pdb, work_dir, n_iter, n_initial, mm_gbsa=True):
    from skopt import Optimizer
    from skopt.space import Real
    hist = Path(work_dir) / f"{seq}_bo_history.jsonl"
    # (dx, dy, dz, rx, ry, rz)
    space = [
        Real(-TRANS_RANGE, TRANS_RANGE),   # dx
        Real(-TRANS_RANGE, TRANS_RANGE),   # dy
        Real(-TRANS_RANGE, TRANS_RANGE),   # dz
        Real(-ROT_RANGE, ROT_RANGE),       # rx
        Real(-ROT_RANGE, ROT_RANGE),       # ry
        Real(-ROT_RANGE, ROT_RANGE),       # rz
    ]
    opt = Optimizer(space, base_estimator='GP', acq_func='EI',
                    n_initial_points=n_initial, random_state=42)
    done = 0
    if hist.exists():
        for l in hist.open():
            try:
                d = json.loads(l); opt.tell(d['X'], -d['f']); done += 1
            except: pass
        print(f"  {seq} 재개: {done} iter", file=sys.stderr, flush=True)
    best = None
    for i in range(done, n_iter):
        X = opt.ask()
        y, scores = objective(X, seq, source_pdb, work_dir, i, mm_gbsa=mm_gbsa)
        opt.tell(X, y)
        f = -y
        if best is None or f > best: best = f
        with hist.open('a') as h: h.write(json.dumps(scores, default=str)+'\n')
        print(f"  {seq} iter{i+1:3d}/{n_iter} f={f:.4f} best={best:.4f} "
              f"ddG={scores.get('ddg_flex')} SS={scores.get('ss_dist')}",
              file=sys.stderr, flush=True)
    return best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', default=None)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--n-iter', type=int, default=100)
    ap.add_argument('--n-initial', type=int, default=10)
    ap.add_argument('--work-dir', default='runs/exp72_analysis/pose_bo_rigid')
    ap.add_argument('--no-mmgbsa', action='store_true')
    ap.add_argument('--source-pdb-dir', default='runs/exp72_analysis/pool_work')
    args = ap.parse_args()

    import pyrosetta
    pyrosetta.init('-mute all -ex1 -ex2aro -ignore_unrecognized_res '
                   '-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0')

    all_seqs = args.seqs if args.seqs else load_tier1_seqs()
    mine = [s for i,s in enumerate(all_seqs) if i % args.shard_count == args.shard_index]
    print(f"[rigid-bo shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(all_seqs)}", file=sys.stderr)

    work = REPO / args.work_dir; work.mkdir(parents=True, exist_ok=True)
    summary = work / 'bo_summary.jsonl'
    for seq in mine:
        src = REPO / args.source_pdb_dir / f'{seq}.pdb'
        if not src.exists(): src = REPO / 'runs/exp72_system/phase1_work' / f'{seq}.pdb'
        if not src.exists():
            print(f"  {seq}: no PDB, skip", file=sys.stderr); continue
        t0 = time.time()
        try:
            best, hist = run_bo_for_seq(seq, str(src), str(work),
                                        args.n_iter, args.n_initial, mm_gbsa=not args.no_mmgbsa)
            rec = {'sequence':seq,'best_f':best,'n_iter':args.n_iter,
                   'elapsed_s':round(time.time()-t0,1),'history':str(hist),
                   'ts':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
            with summary.open('a') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
            print(f"✓ {seq} best_f={best:.4f} ({rec['elapsed_s']:.0f}s)", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"✗ {seq} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    print(f"[rigid-bo shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
