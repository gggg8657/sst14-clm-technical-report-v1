#!/usr/bin/env python3
"""EXP72 풀 전체 robust 재도킹 (Stage A) — 강결합 풀(ddG<-30 ∪ robust-clean, ~2583).

각 후보를 nstruct robust FlexPepDock 재도킹 + 구조검증(SS) → PDB 생성(선택성/MMGBSA 전제).
provenance 무관 전 후보(random/llm/silo_a/legacy) 통합 검증 = 사용자 요청 "거의 대부분".
입력: _workspace/EXP72_POOL.json. 출력: runs/exp72_analysis/pool_redock.jsonl (재개), PDB→pool_work/.
샤딩 --shard-index/--shard-count. NO MOCK.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "AG_src" / "scripts")); sys.path.insert(0, str(REPO / "scripts"))
TEMPLATE = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstruct", type=int, default=int(os.environ.get("FLEXPEP_NSTRUCT", "10")))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--pool", default="_workspace/EXP72_POOL.json")
    ap.add_argument("--out", default="runs/exp72_analysis/pool_redock.jsonl")
    args = ap.parse_args()

    out_path = REPO / args.out; out_path.parent.mkdir(parents=True, exist_ok=True)
    work = REPO / "runs/exp72_analysis/pool_work"; work.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open(errors="ignore"):
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["sequence"])
            except Exception:
                pass

    pool = json.loads((REPO / args.pool).read_text())["candidates"]
    mine = [c for i, c in enumerate(pool) if i % args.shard_count == args.shard_index
            and c["sequence"] not in done]
    print(f"[pool-redock shard {args.shard_index}/{args.shard_count}] 처리 {len(mine)} "
          f"(완료 {len(done)}) nstruct={args.nstruct}", file=sys.stderr, flush=True)

    import flexpep_dock as fpd
    from exp72_structure_check import check_complex_structure
    fpd.init_pyrosetta()

    for c in mine:
        seq = c["sequence"]
        rec = {"sequence": seq, "provenance": c.get("provenance"),
               "prior_singlepose": c.get("ddg_singlepose"), "nstruct": args.nstruct,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            pose, _ = fpd.prepare_complex_by_mutation(TEMPLATE, seq, peptide_chain=1)
            out_pdb = str(work / f"{seq}.pdb")
            _, info = fpd.run_flexpep_refine_pose(pose, out_pdb, nstruct=args.nstruct)
            rec.update({k: info.get(k) for k in
                        ("ddg_median", "ddg_mean", "ddg_min", "ddg_sd", "n_converged") if k in info})
            try:
                chk = check_complex_structure(out_pdb, seq)
                rec.update({k: chk.get(k) for k in ("sg_sg_distance", "disulfide_flag", "disulfide_intact")})
            except Exception as ce:
                rec["struct_err"] = str(ce)
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"  {seq} med={rec.get('ddg_median')} sd={rec.get('ddg_sd')} SS={rec.get('sg_sg_distance')}",
              file=sys.stderr, flush=True)
    print(f"[pool-redock shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
