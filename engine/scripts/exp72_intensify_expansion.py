#!/usr/bin/env python3
"""INTENSIFY 확장 검증 - Tier 1 후보 각각에 1-mutation 인접 20개 생성 → refine.

목적: EXP72 최종 판정에서 INTENSIFY(LLM exploitation) 64% 선택률 = 유일한 시스템 우위
      신호였음. n=22 소수로 통계력 부족 → 확장 검증(n=360).

방식: Tier 1 18 후보 × 각 후보의 1-mutation 이웃 20개 생성 (design_pos 무작위 1자리)
      → 각 이웃 pep_refine (nstruct 5, robust) → ddG 산출.

세션독립, 재개 가능.
"""
import argparse, json, os, random, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AA = 'ACDEFGHIKLMNPQRSTVWY'  # 20 표준
DESIGN_POS = [0, 1, 3, 4, 5, 10, 11, 12]  # SST14 스캐폴드 밖 (Cys3/14 제외, FWKT7-10 보존)


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


def neighbors(seq, n_neighbors=20, seed=0):
    """seq의 1-mutation 이웃 n_neighbors개 생성 (design_pos 무작위 1자리)."""
    random.seed(hash(seq) ^ seed & 0xFFFF)
    out = set()
    tries = 0
    while len(out) < n_neighbors and tries < n_neighbors * 10:
        pos = random.choice(DESIGN_POS)
        aa = random.choice(AA)
        if seq[pos] == aa: tries += 1; continue
        new = seq[:pos] + aa + seq[pos+1:]
        if new != seq: out.add(new)
        tries += 1
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=16)
    ap.add_argument('--nstruct', type=int, default=5)
    ap.add_argument('--n-neighbors', type=int, default=20)
    ap.add_argument('--work-dir', default='runs/exp72_analysis/intensify_expansion')
    args = ap.parse_args()

    from AG_src.scripts.flexpep_dock import run_flexpep_refine_pose
    from pyrosetta_flow.pose_chain_utils import detect_peptide_chain
    import pyrosetta
    pyrosetta.init('-mute all -ex1 -ex2aro -ignore_unrecognized_res '
                   '-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0')

    tier1 = load_tier1_seqs()
    all_neigh = []
    for parent in tier1:
        for n in neighbors(parent, args.n_neighbors):
            all_neigh.append((parent, n))
    mine = [x for i, x in enumerate(all_neigh) if i % args.shard_count == args.shard_index]
    print(f"[intensify shard {args.shard_index}/{args.shard_count}] {len(mine)}/{len(all_neigh)}", file=sys.stderr)

    work = REPO / args.work_dir; work.mkdir(parents=True, exist_ok=True)
    out_path = work / f'shard_{args.shard_index:02d}.jsonl'

    # 재개
    done = set()
    if out_path.exists():
        for l in out_path.open():
            try:
                d = json.loads(l)
                if d.get('status')=='ok': done.add(d['sequence'])
            except: pass

    for parent, seq in mine:
        if seq in done: continue
        rec = {'sequence': seq, 'parent': parent, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        try:
            # native 복합체를 base로, mutation 적용
            src = REPO / 'runs/exp72_analysis/pool_work' / f'{parent}.pdb'
            if not src.exists():
                src = REPO / 'runs/exp72_system/phase1_work' / f'{parent}.pdb'
            if not src.exists():
                rec['status'] = 'skip'; rec['error'] = 'parent PDB 없음'
            else:
                pose = pyrosetta.pose_from_pdb(str(src))
                peptide_chain = detect_peptide_chain(pose, len(seq))
                # peptide chain의 mutation position 찾아서 변이
                for i, (p_aa, n_aa) in enumerate(zip(parent, seq)):
                    if p_aa != n_aa:
                        for k in range(1, pose.total_residue()+1):
                            if pose.pdb_info().chain(k)==peptide_chain and pose.pdb_info().number(k)==i+1:
                                from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
                                aa3 = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY',
                                       'H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN',
                                       'P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL',
                                       'W':'TRP','Y':'TYR'}[n_aa]
                                MutateResidue(target=k, new_res=aa3).apply(pose)
                                break
                out_pdb = str(work / f'{seq}_refined.pdb')
                _, info = run_flexpep_refine_pose(pose, out_pdb, nstruct=args.nstruct)
                rec.update({
                    'ddg_median': info.get('ddg_median'),
                    'ddg_mean': info.get('ddg_mean'),
                    'ddg_sd': info.get('ddg_sd'),
                    'n_converged': info.get('n_converged'),
                    'sg_sg_distance': info.get('sg_sg_distance'),
                    'disulfide_intact': info.get('disulfide_intact'),
                    'status': 'ok',
                })
        except Exception as exc:
            rec['status'] = 'error'
            rec['error'] = f"{type(exc).__name__}: {exc}"
        with out_path.open('a') as f: f.write(json.dumps(rec, ensure_ascii=False, default=str)+'\n')
        print(f"  {parent}→{seq} ddG={rec.get('ddg_median')} status={rec['status']}", file=sys.stderr, flush=True)

    print(f"[intensify shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
