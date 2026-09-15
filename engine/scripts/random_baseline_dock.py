#!/usr/bin/env python3
"""랜덤 베이스라인 발굴 (72h A/B 실험, arm 1 = RANDOM).

'그냥 랜덤시드로 SST-14 변이체 만들어서 도킹' — LLM planner/panel/bandit 전혀 없이,
scaffold(Cys3/14 + FWKT7-10) 보존한 무작위 변이만 생성해 robust FlexPepDock 채점.

시스템 arm(LLM 계획)과의 대조군. 동일 변이규칙(generate_random_mutant)·동일 nstruct·
동일 template·동일 native 기준 → 변인은 "선택 방식(랜덤 vs 계획)"뿐.

출력: runs/exp72_random/experiment_log.jsonl (시스템 로그와 동일 스키마 → 동일 툴로 비교).
정지: _workspace/STOP_RANDOM_BASELINE 파일 생성 또는 --max-seconds 경과.
NO MOCK: 실제 PyRosetta FlexPepDock. 실패/미수렴은 정직 기록.
"""
from __future__ import annotations
import argparse, json, os, sys, time, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "AG_src" / "scripts"))

NATIVE = "AGCKNFFWKTFTSC"
# 변이 허용 위치(1-indexed): Cys3/14 + FWKT(7,8,9,10) 제외 = pharmacophore/이황화 보존
DESIGN_POSITIONS = [1, 2, 4, 5, 6, 11, 12, 13]
TEMPLATE = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/exp72_random")
    ap.add_argument("--max-seconds", type=int, default=72 * 3600)  # 72h
    ap.add_argument("--nstruct", type=int, default=int(os.environ.get("FLEXPEP_NSTRUCT", "5")))
    ap.add_argument("--seed", type=int, default=20260708)
    ap.add_argument("--stop-file", default="_workspace/STOP_RANDOM_BASELINE")
    args = ap.parse_args()

    out_dir = REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "experiment_log.jsonl"
    work_dir = out_dir / "work"; work_dir.mkdir(exist_ok=True)
    stop_path = REPO / args.stop_file

    import flexpep_dock as fpd
    from pyrosetta_flow.adapter import generate_random_mutant
    fpd.init_pyrosetta()

    rng = random.Random(args.seed)
    seen = set([NATIVE])
    # 재개: 기존 로그의 서열 seen 로드
    if log_path.exists():
        for line in log_path.open(errors="ignore"):
            try:
                s = json.loads(line).get("sequence")
                if s:
                    seen.add(s)
            except Exception:
                pass

    t0 = time.time()
    n_done = 0
    n_beat = 0
    NATIVE_ROBUST = -20.28  # canonical robust median native baseline
    print(f"[random-baseline] start out={out_dir} nstruct={args.nstruct} "
          f"design_pos={DESIGN_POSITIONS} 재개seen={len(seen)}", file=sys.stderr, flush=True)

    while time.time() - t0 < args.max_seconds:
        if stop_path.exists():
            print("[random-baseline] STOP file 감지 — 종료", file=sys.stderr); break
        # 1) 무작위 변이체 생성 (scaffold 보존, dedup)
        seq = None
        for _ in range(200):
            n_mut = rng.randint(1, min(5, len(DESIGN_POSITIONS)))
            cand = generate_random_mutant(NATIVE, DESIGN_POSITIONS, rng, n_mutations=n_mut)
            if cand != NATIVE and cand not in seen:
                seq = cand; break
        if seq is None:
            print("[random-baseline] dedup 포화 — 잠시 대기", file=sys.stderr); time.sleep(5); continue
        seen.add(seq)

        # 2) robust 도킹
        rec = {"record_type": "candidate", "sequence": seq, "mutation_source": "random_baseline",
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "arm": "random"}
        try:
            pose, pep_idx = fpd.prepare_complex_by_mutation(TEMPLATE, seq, peptide_chain=1)
            out_pdb = str(work_dir / f"{seq}.pdb")
            refined, info = fpd.run_flexpep_refine_pose(pose, out_pdb, nstruct=args.nstruct)
            rec.update({k: info.get(k) for k in
                        ("ddg_median", "ddg_mean", "ddg_min", "ddg_sd", "n_converged",
                         "disulfide_intact", "sg_sg_distance") if k in info})
            rec["ddg"] = info.get("ddg_median")
            rec["status"] = "ok"
            med = info.get("ddg_median")
            if isinstance(med, (int, float)) and med <= NATIVE_ROBUST - 15:  # ≥15 REU 유의 능가
                n_beat += 1
        except Exception as exc:
            rec["status"] = "error"; rec["error_summary"] = f"{type(exc).__name__}: {exc}"

        with log_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 5 == 0:
            el = (time.time() - t0) / 3600
            print(f"[random-baseline] {n_done}개 도킹 ({el:.1f}h), native능가(≥15REU) {n_beat}, "
                  f"최근 {seq} ddg_med={rec.get('ddg_median')}", file=sys.stderr, flush=True)

    print(f"[random-baseline] DONE: {n_done}개, native능가 {n_beat}, {(time.time()-t0)/3600:.1f}h", file=sys.stderr)


if __name__ == "__main__":
    main()
