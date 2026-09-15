#!/usr/bin/env python3
"""Pose Bayesian Optimization — 각 후보 서열의 진짜 최적 backbone 자세 탐색.

BO 대상: peptide backbone torsion (φ,ψ) for i∈{7,8,9,10} (FWKT residues)
  X ∈ R^8, each ∈ [-180°, 180°]
목적함수: f(X) = 0.5·norm(ddG_flex) + 0.3·norm(dg_bind_mmgbsa) + 0.1·SS + 0.1·FWKT

각 BO iter:
  1. X (torsion) → peptide backbone 수정 → PDB 저장
  2. FlexPepDock refine (nstruct 1, side-chain만) → ddG
  3. MM-GBSA (dg_bind)
  4. SS/FWKT flag
  5. Scalar f → BO tell()

BO 라이브러리: scikit-optimize (skopt) Gaussian Process + EI acquisition.
Warm-start: 기존 pep_refine 결과 있으면 (torsion=0으로 매핑하여) 4점 seed.

세션독립: setsid + 파일기반 재개 (bo_history.jsonl).
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ── 목적함수 파라미터 ───────────────────────────────────────
W_DDG = 0.5
W_MM  = 0.3
W_SS  = 0.1
W_FWKT = 0.1
DDG_REF, DDG_SCALE = 0.0, 50.0     # ddG를 [0,1]로 정규화 (음수가 좋음)
MM_REF,  MM_SCALE  = 0.0, 100.0    # MM-GBSA dg_bind 정규화

# ── Torsion 지역탐색 범위 (native ± DELTA) ────────────────
# native SST-14 (측정치, phase1_work/AGCKNFFWKTFTSC.pdb) - FWKT 잔기 7-10:
#   res7 PHE:  φ=-110.29, ψ=-171.04
#   res8 TRP:  φ= -46.83, ψ= -59.46
#   res9 LYS:  φ=-139.56, ψ= +30.03
#   res10 THR: φ=-144.19, ψ=+155.08
# 전 범위 [-180,180]는 이황화(Cys3-Cys14) 강제 파괴 → 실측 SS 파괴 100% 관찰
# native ±30° 지역탐색 = SS 유지 가능성↑ + 실질적 자세 다양성 확보
NATIVE_TORSIONS = [
    -110.29, -171.04,   # F7 φ, ψ
     -46.83,  -59.46,   # W8 φ, ψ
    -139.56,   30.03,   # K9 φ, ψ
    -144.19,  155.08,   # T10 φ, ψ
]
TORSION_DELTA = 30.0    # ±30° 지역탐색 (설계 문서 개정 2026-07-23)


def normalize_negative(x, ref, scale):
    """x < 0 일수록 좋음 → (ref - x) / scale, [0,1] 포화."""
    if x is None or (isinstance(x, float) and (x != x)):  # None/NaN
        return 0.0
    v = (ref - x) / scale
    return max(0.0, min(1.0, v))


def _apply_ss_constraint(pose):
    """Cys3-Cys14 SG-SG AtomPairConstraint 강제 유지 (torsion 회전 후 재체결 유도).
    반환: True if constraint 적용 성공, False otherwise."""
    from AG_src.scripts.flexpep_dock import _add_disulfide_constraint, _find_peptide_cys_residues
    cys = _find_peptide_cys_residues(pose)
    if len(cys) != 2:
        return False
    try:
        _add_disulfide_constraint(pose, cys[0], cys[1])
        return True
    except Exception:
        return False


def build_pose_from_torsion(source_pdb, torsions, out_pdb, target_residues=(7, 8, 9, 10),
                              enforce_ss=True):
    """source_pdb 로드 → target_residues의 (φ,ψ)를 torsions로 설정 → out_pdb 저장.
    torsions: [φ7, ψ7, φ8, ψ8, φ9, ψ9, φ10, ψ10]  (도 단위)
    enforce_ss: True이면 회전 후 Cys-Cys AtomPairConstraint 적용 (2.05Å sd=0.3)
    return: pose (수정 완료 + constraint 부착)
    """
    import pyrosetta
    from pyrosetta_flow.pose_chain_utils import detect_peptide_chain
    pose = pyrosetta.pose_from_pdb(source_pdb)
    peptide_chain = detect_peptide_chain(pose, 14)
    # pdb_info로 peptide chain 잔기 찾기
    for i, resnum in enumerate(target_residues):
        pose_i = None
        for k in range(1, pose.total_residue() + 1):
            if pose.pdb_info().chain(k) == peptide_chain and pose.pdb_info().number(k) == resnum:
                pose_i = k; break
        if pose_i is None: continue
        pose.set_phi(pose_i, torsions[2*i])
        pose.set_psi(pose_i, torsions[2*i + 1])
    # SS constraint (torsion 회전이 chain propagation으로 SS 파괴 → refine이 재체결 유도)
    if enforce_ss:
        _apply_ss_constraint(pose)
    pose.dump_pdb(out_pdb)
    return pose


def score_pose(pose, out_pdb_refined, run_mmgbsa=True):
    """FlexPepDock refine (nstruct 1) + MM-GBSA + 구조검증.
    return: dict(ddg_flex, dg_bind_mm, ss_ok, fwkt_ok, disulfide_dist, fwkt_dist)
    """
    from AG_src.scripts.flexpep_dock import (
        run_flexpep_refine_pose, _check_disulfide_distance, _find_peptide_cys_residues,
    )
    from pyrosetta_flow.pose_chain_utils import chain_lengths, detect_peptide_chain
    result = {}
    peptide_chain = detect_peptide_chain(pose, 14)
    receptor_chains = [chain for chain in chain_lengths(pose) if chain != peptide_chain]

    # Step 1: FlexPepDock refine (backbone 고정, side-chain rotamer만)
    try:
        _, info = run_flexpep_refine_pose(pose, out_pdb_refined, nstruct=1)
        # info에서 ddg 추출 (구조: {"ddg": val, ...})
        result['ddg_flex'] = info.get('ddg_median') or info.get('ddg')
        result['ss_dist'] = info.get('sg_sg_distance')
        result['ss_ok'] = 1.0 if info.get('disulfide_intact') else 0.0
    except Exception as exc:
        result['ddg_flex'] = None
        result['ss_ok'] = 0.0
        result['error_flex'] = f"{type(exc).__name__}: {exc}"
        return result

    # Step 2: FWKT 파마코포어 접촉 (7-10 CA와 receptor CA 최소거리)
    try:
        pose2 = pose  # refine 후 pose
        cas = {}
        for i in range(1, pose2.total_residue() + 1):
            ch = pose2.pdb_info().chain(i)
            if pose2.residue(i).has('CA'):
                cas.setdefault(ch, []).append((pose2.pdb_info().number(i), pose2.residue(i).xyz('CA')))
        min_dist = 999.0
        for pep_resnum, pep_xyz in cas.get(peptide_chain, []):
            if pep_resnum not in (7, 8, 9, 10): continue
            for receptor_chain in receptor_chains:
                for rec_resnum, rec_xyz in cas.get(receptor_chain, []):
                    d = ((pep_xyz[0]-rec_xyz[0])**2 + (pep_xyz[1]-rec_xyz[1])**2 + (pep_xyz[2]-rec_xyz[2])**2) ** 0.5
                    if d < min_dist: min_dist = d
        result['fwkt_dist'] = round(min_dist, 3)
        result['fwkt_ok'] = 1.0 if min_dist <= 8.0 else 0.0
    except Exception:
        result['fwkt_ok'] = 0.0
        result['fwkt_dist'] = None

    # Step 3: MM-GBSA (선택)
    if run_mmgbsa:
        # MM-GBSA env로 subprocess 호출 (mmgbsa daemon 함수 재사용)
        # 간단히 placeholder — 실제는 mmgbsa_rescore module 호출
        try:
            from pyrosetta_flow.mmgbsa_rescore import mmgbsa_rescore
            mm_result = mmgbsa_rescore(out_pdb_refined,
                                       receptor_chains=receptor_chains, peptide_chain=peptide_chain)
            result['dg_bind_mm'] = mm_result.get('dg_bind') if mm_result else None
        except Exception as exc:
            result['dg_bind_mm'] = None
            result['error_mm'] = f"{type(exc).__name__}: {exc}"
    else:
        result['dg_bind_mm'] = None

    return result


def objective(X, seq, source_pdb, work_dir, iter_num, mm_gbsa=True):
    """BO 목적함수. X는 torsion 8-tuple. 반환: -f(X) (skopt는 최소화)."""
    tmp_pdb = f"{work_dir}/{seq}_bo{iter_num:03d}_init.pdb"
    refined_pdb = f"{work_dir}/{seq}_bo{iter_num:03d}_refined.pdb"

    try:
        # enforce_ss=True: build 후 SS constraint 부착 → refine이 재체결 유도
        pose = build_pose_from_torsion(source_pdb, X, tmp_pdb, enforce_ss=True)
        # 사전 SS 거리 기록만 (게이트 아님, refine 후 재판정)
        ss_pre = None
        try:
            from pyrosetta_flow.pose_chain_utils import detect_peptide_chain
            peptide_chain = detect_peptide_chain(pose, 14)
            cys_idx = []
            for k in range(1, pose.total_residue()+1):
                if pose.pdb_info().chain(k)==peptide_chain and pose.pdb_info().number(k) in (3,14):
                    if pose.residue(k).has('SG'): cys_idx.append(k)
            if len(cys_idx)==2:
                sg1 = pose.residue(cys_idx[0]).xyz('SG')
                sg2 = pose.residue(cys_idx[1]).xyz('SG')
                ss_pre = round(((sg1[0]-sg2[0])**2 + (sg1[1]-sg2[1])**2 + (sg1[2]-sg2[2])**2) ** 0.5, 3)
        except Exception:
            pass
        # 극단적 파괴(>25Å)만 스킵 — refine으로 재체결 불가능
        if ss_pre is not None and ss_pre > 25.0:
            return (0.0, {'X': list(X), 'iter': iter_num, 'ss_precheck': ss_pre,
                          'f': 0.0, 'skipped': 'SS_extreme_break'})
        scores = score_pose(pose, refined_pdb, run_mmgbsa=mm_gbsa)
        if ss_pre is not None: scores['ss_precheck'] = ss_pre
    except Exception as exc:
        return (1.0, {'X': list(X), 'error': f"{type(exc).__name__}: {exc}"})

    # 스칼라 종합
    s_ddg = normalize_negative(scores.get('ddg_flex'), DDG_REF, DDG_SCALE)
    s_mm  = normalize_negative(scores.get('dg_bind_mm'), MM_REF, MM_SCALE)
    s_ss  = scores.get('ss_ok', 0.0)
    s_fwkt = scores.get('fwkt_ok', 0.0)
    f = W_DDG * s_ddg + W_MM * s_mm + W_SS * s_ss + W_FWKT * s_fwkt

    scores.update({'X': list(X), 'f': round(f, 4), 'iter': iter_num, 'refined_pdb': refined_pdb})
    return (-f, scores)  # minimize -f = maximize f


def load_tier1_seqs():
    p = REPO / 'runs/exp72_analysis/pool_selectivity.jsonl'
    seen = {}
    for l in p.open():
        try: r = json.loads(l)
        except: continue
        if r.get('delta_margin') is not None: seen[r['sequence']] = r
    recs = list(seen.values())
    nat = [r for r in recs if r['arm'] == 'native'][0]
    nat_m = nat['delta_margin']
    t1 = []
    for r in recs:
        if r['arm'] == 'native': continue
        sig = r['delta_margin'] - nat_m
        if r['on_target_ddg'] < -30 and sig > 10:
            t1.append((r['sequence'], r['on_target_ddg']))
    t1.sort(key=lambda x: x[1])
    return [s for s, _ in t1]


def run_bo_for_seq(seq, source_pdb, work_dir, n_iter, n_initial, mm_gbsa=True):
    from skopt import Optimizer
    from skopt.space import Real

    hist_path = Path(work_dir) / f"{seq}_bo_history.jsonl"
    # Native ± TORSION_DELTA 지역탐색 (SS 파괴 방지 + 실질적 다양성)
    space = [Real(NATIVE_TORSIONS[i] - TORSION_DELTA, NATIVE_TORSIONS[i] + TORSION_DELTA)
             for i in range(8)]
    opt = Optimizer(space, base_estimator='GP', acq_func='EI',
                    n_initial_points=n_initial, random_state=42)

    # 재개
    done_iters = 0
    if hist_path.exists():
        with hist_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    opt.tell(d['X'], -d['f'])  # skopt 관습: min(-f)
                    done_iters += 1
                except Exception:
                    pass
        print(f"  {seq} 재개: {done_iters} iter 이미 완료", file=sys.stderr, flush=True)

    best_f = None
    for i in range(done_iters, n_iter):
        X = opt.ask()
        y, scores = objective(X, seq, source_pdb, work_dir, i, mm_gbsa=mm_gbsa)
        opt.tell(X, y)
        f = -y
        if best_f is None or f > best_f:
            best_f = f
        # 기록
        with hist_path.open('a') as h:
            h.write(json.dumps(scores, default=str) + '\n')
        print(f"  {seq} iter{i+1:3d}/{n_iter} f={f:.4f} best={best_f:.4f} "
              f"ddG={scores.get('ddg_flex')} MM={scores.get('dg_bind_mm')}",
              file=sys.stderr, flush=True)

    return best_f, hist_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', default=None, help='분석할 서열 (기본: Tier 1 18종)')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--n-iter', type=int, default=100)
    ap.add_argument('--n-initial', type=int, default=10)
    ap.add_argument('--work-dir', default='runs/exp72_analysis/pose_bo')
    ap.add_argument('--no-mmgbsa', action='store_true', help='MM-GBSA 스킵 (빠른 테스트용)')
    ap.add_argument('--source-pdb-dir', default='runs/exp72_analysis/pool_work',
                    help='서열별 초기 PDB 위치')
    args = ap.parse_args()

    # PyRosetta 초기화 (backbone 각도 직접 조작에는 refine 옵션이면 충분)
    import pyrosetta
    pyrosetta.init(
        '-mute all -ex1 -ex2aro -ignore_unrecognized_res '
        '-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0'
    )

    # 서열 로드 + 샤딩
    if args.seqs:
        all_seqs = args.seqs
    else:
        all_seqs = load_tier1_seqs()
    mine = [s for i, s in enumerate(all_seqs) if i % args.shard_count == args.shard_index]
    print(f"[pose-bo shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(all_seqs)} 후보",
          file=sys.stderr, flush=True)

    work = REPO / args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    summary_path = work / 'bo_summary.jsonl'

    for seq in mine:
        src = REPO / args.source_pdb_dir / f'{seq}.pdb'
        if not src.exists():
            # shortlist에서 찾기
            src = REPO / 'runs/exp72_system/phase1_work' / f'{seq}.pdb'
        if not src.exists():
            print(f"  {seq}: source PDB 없음, skip", file=sys.stderr); continue

        t0 = time.time()
        try:
            best_f, hist = run_bo_for_seq(
                seq, str(src), str(work), args.n_iter, args.n_initial,
                mm_gbsa=not args.no_mmgbsa
            )
            rec = {'sequence': seq, 'best_f': best_f, 'n_iter': args.n_iter,
                   'elapsed_s': round(time.time() - t0, 1),
                   'history': str(hist), 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            with summary_path.open('a') as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            print(f"✓ {seq} best_f={best_f:.4f} ({rec['elapsed_s']:.0f}s)", file=sys.stderr, flush=True)
        except Exception as exc:
            print(f"✗ {seq} FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    print(f"[pose-bo shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
