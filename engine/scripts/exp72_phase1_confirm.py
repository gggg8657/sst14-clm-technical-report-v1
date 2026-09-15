#!/usr/bin/env python3
"""EXP72 Phase 1 확증 재도킹 (R2) — 양 arm top8 + native 를 nstruct≥20 robust 재도킹.

목적: 랜덤 −43.05 vs 시스템 −36.77 의 6.3REU 차이가 도킹노이즈(sd~8–10) 밖인지 확증.
math 권고: nstruct≥20 이라야 80% 검정력으로 6.3REU 차이 검출.
구조검증(SS bond) 동시 기록 → R3(이황결합 파손 아티팩트) 재확인.

입력: `_workspace/EXP72_PHASE1_TARGETS.json` (random top8 + system llm_guided top8 + native)
출력: `runs/exp72_system/phase1_confirm.jsonl` (재개가능, seq당 1레코드; O_APPEND 원자적)
샤딩: --shard-index i --shard-count N → targets[i::N] 만 처리 (병렬 워커용)
NO MOCK: 실제 FlexPepDock nstruct=20. 실패/미수렴 정직 기록.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "AG_src" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))

TEMPLATE = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")


def load_targets():
    d = json.loads((REPO / "_workspace/EXP72_PHASE1_TARGETS.json").read_text())
    out = []
    for x in d["random_arm"]["top8_by_ddg_median_asc"]:
        out.append({"sequence": x["sequence"], "arm": "random", "prior": x.get("ddg_median")})
    for x in d["system_arm_llm_guided_only"]["top8_by_ddg_asc"]:
        out.append({"sequence": x["sequence"], "arm": "system_llm", "prior_singlepose": x.get("ddg")})
    nat = d["native_reference"]["sequence"]
    out.append({"sequence": nat, "arm": "native", "prior": d["native_reference"]["canonical_ddg_median"]})
    # dedup by sequence (native가 arm 목록에 우연히 겹치면 첫것 유지)
    seen = {}
    for t in out:
        seen.setdefault(t["sequence"], t)
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstruct", type=int, default=20)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", default="runs/exp72_system/phase1_confirm.jsonl")
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / "phase1_work"; work.mkdir(exist_ok=True)

    # 재개: 이미 완료된 seq 스킵
    done = set()
    if out_path.exists():
        for line in out_path.open(errors="ignore"):
            try:
                r = json.loads(line)
                if r.get("status") == "ok":
                    done.add(r["sequence"])
            except Exception:
                pass

    targets = load_targets()
    mine = [t for i, t in enumerate(targets) if i % args.shard_count == args.shard_index]
    mine = [t for t in mine if t["sequence"] not in done]
    print(f"[phase1 shard {args.shard_index}/{args.shard_count}] 처리 {len(mine)}개 "
          f"(전체 {len(targets)}, 완료 {len(done)}), nstruct={args.nstruct}", file=sys.stderr, flush=True)

    import flexpep_dock as fpd
    from exp72_structure_check import check_complex_structure
    fpd.init_pyrosetta()

    for t in mine:
        seq = t["sequence"]
        rec = {"sequence": seq, "arm": t["arm"], "nstruct": args.nstruct,
               "prior": t.get("prior"), "prior_singlepose": t.get("prior_singlepose"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            pose, _ = fpd.prepare_complex_by_mutation(TEMPLATE, seq, peptide_chain=1)
            out_pdb = str(work / f"{seq}.pdb")
            _, info = fpd.run_flexpep_refine_pose(pose, out_pdb, nstruct=args.nstruct)
            rec.update({k: info.get(k) for k in
                        ("ddg_median", "ddg_mean", "ddg_min", "ddg_sd", "n_converged") if k in info})
            try:
                chk = check_complex_structure(out_pdb, seq)
                rec.update({k: chk.get(k) for k in
                            ("sg_sg_distance", "disulfide_flag", "disulfide_intact")})
                pc = chk.get("pharmacophore_contact") or {}
                rec["fwkt_in_contact"] = pc.get("in_contact")
                rec["fwkt_min_dist"] = pc.get("min_distance")
            except Exception as ce:
                rec["struct_err"] = str(ce)
            rec["status"] = "ok"
        except Exception as exc:
            rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [{t['arm']}] {seq} med={rec.get('ddg_median')} sd={rec.get('ddg_sd')} "
              f"n={rec.get('n_converged')} SS={rec.get('sg_sg_distance')}({rec.get('disulfide_flag')})",
              file=sys.stderr, flush=True)

    print(f"[phase1 shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
