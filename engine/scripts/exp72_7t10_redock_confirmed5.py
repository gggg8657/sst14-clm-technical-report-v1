#!/usr/bin/env python3
"""[DEPRECATED 2026-08-03] Chain 배정 버그. 새 파이프라인 사용:
    scripts/pipeline_predicted_vs_experimental.py

버그 (line ~91): `chain=='A'` 조건은 curated PDB(chain A=receptor)에서 매칭 0.
변이 미적용 상태로 재도킹되어 non-native 5서열 결과 무효.
정정 문서: `_workspace/A1_INVALIDATION_2026-08-03.md`.

원본 목적 (참고):
- Wave 1 A1: 확증 5종을 7T10 실측 cryo-EM으로 재도킹
- 실제 실행 결과는 native 6반복 재도킹으로 판명
"""
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIRMED = [
    'AICKNFFWKTYTSC',
    'AACNRLFWKTFTMC',
    'AVCKNLFWKTFTSC',
    'ARCKFFFWKTFTSC',
    'AGCKWDFWKTATSC',
    'AGCKNFFWKTFTSC',  # native (SST-14)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nstruct', type=int, default=10)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--receptor', default='data/somatostatin_receptor/curated/SSTR2_SST14_complex_7t10.pdb',
                    help='7T10 실측 cryo-EM 복합체')
    ap.add_argument('--out', default='runs/exp72_analysis/7t10_redock/results.jsonl')
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    receptor = REPO / args.receptor
    if not receptor.exists():
        print(f"ERR: receptor 없음: {receptor}", file=sys.stderr)
        sys.exit(1)

    # 재개
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
                if r.get('status') == 'ok':
                    done.add(r['sequence'])
            except Exception:
                pass

    mine = [s for i, s in enumerate(CONFIRMED)
            if i % args.shard_count == args.shard_index and s not in done]
    print(f"[7t10-redock shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(CONFIRMED)}",
          file=sys.stderr, flush=True)

    import pyrosetta
    pyrosetta.init(
        '-mute all -ex1 -ex2aro -ignore_unrecognized_res '
        '-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0'
    )
    from AG_src.scripts.flexpep_dock import run_flexpep_refine_pose
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    aa3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS','I':'ILE',
           'K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG','S':'SER',
           'T':'THR','V':'VAL','W':'TRP','Y':'TYR'}

    work = out_path.parent / 'poses'
    work.mkdir(parents=True, exist_ok=True)

    for seq in mine:
        t0 = time.time()
        rec = {'sequence': seq, 'nstruct': args.nstruct, 'receptor': args.receptor,
               'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            pose = pyrosetta.pose_from_pdb(str(receptor))
            # peptide chain A 확인, 변이 적용 (native 아니면)
            native = 'AGCKNFFWKTFTSC'
            if seq != native:
                for i, (n_aa, m_aa) in enumerate(zip(native, seq)):
                    if n_aa != m_aa:
                        for k in range(1, pose.total_residue()+1):
                            if pose.pdb_info().chain(k) == 'A' and pose.pdb_info().number(k) == i+1:
                                MutateResidue(target=k, new_res=aa3[m_aa]).apply(pose)
                                break

            out_pdb = str(work / f'{seq}_7t10_refined.pdb')
            _, info = run_flexpep_refine_pose(pose, out_pdb, nstruct=args.nstruct)
            rec.update({
                'ddg_median': info.get('ddg_median'),
                'ddg_mean': info.get('ddg_mean'),
                'ddg_sd': info.get('ddg_sd'),
                'ddg_min': info.get('ddg_min'),
                'n_converged': info.get('n_converged'),
                'sg_sg_distance': info.get('sg_sg_distance'),
                'disulfide_intact': info.get('disulfide_intact'),
                'elapsed_s': round(time.time() - t0, 1),
                'status': 'ok',
            })
        except Exception as exc:
            rec['status'] = 'error'
            rec['error'] = f"{type(exc).__name__}: {exc}"
            rec['elapsed_s'] = round(time.time() - t0, 1)
        with out_path.open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        print(f"  {seq} median={rec.get('ddg_median')} SS={rec.get('sg_sg_distance')} "
              f"status={rec['status']} ({rec.get('elapsed_s')}s)", file=sys.stderr, flush=True)

    print(f"[7t10-redock shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
