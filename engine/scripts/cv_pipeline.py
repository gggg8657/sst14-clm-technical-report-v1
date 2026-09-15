#!/usr/bin/env python3
"""CV 파이프라인 — 후보별 정밀 교차검증 (wet-lab 제안용).

각 후보 × 지표(provenance 명시):
  1. 5× robust FlexPepDock nstruct5 (총 25 refines) → 25 pose ddG,
     robust_median mean/sd across 5 runs = 진짜 교차검증 안정성
  2. Selectivity (screen_selectivity) → Δmargin vs SSTR1/3/4/5 (1×, 내부 도킹 무거움)
  3. ADMET (pepADMET) → hc50 vs native −55.68 (deterministic → 1×)
  4. Half-life ensemble → half_life_h (deterministic → 1×)
  5. MM-GBSA → PDB를 docs/structure_view/pdb/ 에 드롭 → mmgbsa_daemon이 다음 사이클 처리 (async)

출력: runs/cv_analysis/cv_master.jsonl (append-only, resumable)
      각 레코드에 source_experiment 필드 (사용자 요구 — 실험 구분)

병렬: N 워커(각 후보 단일-스레드 순차 metric), setsid PPID=1 세션독립.
정지: touch _workspace/STOP_CV_PIPELINE
NO MOCK — 실제 도킹/게이트만.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib, shutil
from pathlib import Path
from statistics import median as _median, stdev, mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "AG_src" / "scripts"))

NATIVE_ROBUST = -20.28
NATIVE_HC50 = -55.6816
TEMPLATE = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")
STRUCT_VIEW_PDB = REPO.parent.parent.parent / "docs/structure_view/pdb"  # mmgbsa_daemon 큐


def _num(x): return x if isinstance(x, (int, float)) else None


def _acquire_lock(seq: str, lock_dir: Path) -> Path | None:
    """후보 처리 원자적 획득. 이미 있으면 None (다른 워커 담당)."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{seq}.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, f"pid={os.getpid()} ts={time.time()}\n".encode())
        os.close(fd)
        return lock
    except FileExistsError:
        return None


def _release_lock(lock: Path):
    try: lock.unlink()
    except Exception: pass


def already_done(seq: str, master_path: Path) -> bool:
    if not master_path.exists(): return False
    with master_path.open(errors="ignore") as f:
        for line in f:
            try:
                if json.loads(line).get("sequence") == seq: return True
            except Exception: continue
    return False


def robust_25pose_dock(seq: str, work_dir: Path, n_runs: int = 5, poses_per_run: int = 5):
    """5 outer runs × 5 poses (nstruct=1) = **25개 독립 pose 전부 PDB 저장**.
    각 pose = 독립 FlexPepDock refine 1회 → 고유 PDB.
    Within-run robust median(5개 pose로) + across-run CV stats(5 medians로)."""
    import flexpep_dock as fpd
    runs = []
    all_ddgs = []
    for i in range(n_runs):
        run_poses = []
        for j in range(poses_per_run):
            out_pdb = str(work_dir / f"{seq}__run{i:02d}_pose{j:02d}.pdb")
            pose, _ = fpd.prepare_complex_by_mutation(TEMPLATE, seq, peptide_chain=1)
            _, info = fpd.run_flexpep_refine_pose(pose, out_pdb, nstruct=1)
            # nstruct=1 이면 ddg_median = 유일한 값(수렴 시), 미수렴이면 None
            ddg = _num(info.get("ddg_median"))
            run_poses.append({"pose": j, "pdb": out_pdb, "ddg": ddg,
                              "converged": ddg is not None and ddg < 0,
                              "disulfide_intact": info.get("disulfide_intact")})
            if ddg is not None and ddg < 0:
                all_ddgs.append(ddg)
        # within-run robust: 수렴 pose들의 median/sd
        conv = [p["ddg"] for p in run_poses if p["converged"]]
        runs.append({"run": i, "poses": run_poses,
                     "n_converged": len(conv),
                     "within_run_median": round(_median(conv), 3) if conv else None,
                     "within_run_sd": round(stdev(conv), 3) if len(conv) > 1 else None,
                     "within_run_min": min(conv) if conv else None})
    # across-run CV: 5개 within-run median
    meds = [r["within_run_median"] for r in runs if r["within_run_median"] is not None]
    agg = {"n_runs": n_runs, "poses_per_run": poses_per_run,
           "total_poses": n_runs * poses_per_run,
           "n_saved_pdbs": sum(len(r["poses"]) for r in runs),
           "n_converged_total": len(all_ddgs),
           "within_run_medians": meds,
           # 전체 25 pose 기반 robust
           "global_ddg_median": round(_median(all_ddgs), 3) if all_ddgs else None,
           "global_ddg_sd": round(stdev(all_ddgs), 3) if len(all_ddgs) > 1 else None,
           "global_ddg_min": min(all_ddgs) if all_ddgs else None,
           "global_ddg_max": max(all_ddgs) if all_ddgs else None,
           # 5-run CV
           "cv_mean_of_medians": round(mean(meds), 3) if meds else None,
           "cv_sd_of_medians": round(stdev(meds), 3) if len(meds) > 1 else None,
           "runs": runs}
    return agg


def best_pose_pdb(agg: dict) -> str | None:
    """25개 pose 중 ddG 최저(수렴) → selectivity/MM-GBSA용 대표 pose."""
    best = None; best_ddg = None
    for r in agg.get("runs", []):
        for p in r.get("poses", []):
            if p.get("converged") and (best_ddg is None or p["ddg"] < best_ddg):
                best_ddg = p["ddg"]; best = p["pdb"]
    return best


def run_selectivity(pdb: str, on_target_ddg: float, timeout: int = 1800):
    from pyrosetta_flow.multiobjective import screen_selectivity
    try:
        r = screen_selectivity(sstr2_complex_pdb=pdb, on_target_ddg=on_target_ddg,
                               conda_env="bio-tools", timeout=timeout)
        return {"offtarget_ddg": r.get("offtarget_ddg"),
                "selectivity_margin": _num(r.get("selectivity_margin")),
                "worst_offtarget": r.get("worst_offtarget"),
                "is_selective": r.get("is_selective"), "gate_pass": r.get("gate_pass")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_admet(seq: str):
    from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
    tox = predict_toxicity_for_sequences([seq]).get(seq, {})
    hc50 = _num(tox.get("hc50")) if tox.get("available") else None
    return {"pepadmet_available": bool(tox.get("available")),
            "hc50": hc50,
            "hc50_vs_native": round(hc50 - NATIVE_HC50, 3) if hc50 is not None else None,
            "is_toxic_flag": tox.get("is_toxic"),
            "toxicity_type": tox.get("toxicity_type")}


def run_halflife(seq: str):
    from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife
    try:
        h = ensemble_halflife(seq)
        return {"half_life_h": _num(h.get("half_life_h")),
                "half_life_heuristic_h": _num(h.get("half_life_heuristic_h")),
                "half_life_rf_h": _num(h.get("half_life_rf_h")),
                "halflife_source": h.get("halflife_source")}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def drop_pdb_for_mmgbsa(seq: str, best_pdb: str):
    """MM-GBSA 데몬 큐(docs/structure_view/pdb/)에 대표 pose 복사. 데몬이 다음 사이클 처리."""
    if not best_pdb or not Path(best_pdb).exists(): return None
    STRUCT_VIEW_PDB.mkdir(parents=True, exist_ok=True)
    dest = STRUCT_VIEW_PDB / f"cv_{seq}.pdb"
    try:
        shutil.copy(best_pdb, dest)
        return str(dest)
    except Exception as exc:
        return f"copy_error: {exc}"


def process_one(cand: dict, master_path: Path, work_root: Path, lock_dir: Path,
                do_selectivity: bool, do_mmgbsa: bool):
    seq = cand["sequence"]
    if already_done(seq, master_path):
        return "skip_done"
    lock = _acquire_lock(seq, lock_dir)
    if lock is None:
        return "skip_locked"
    try:
        work = work_root / seq[:8]        # dedup 방지, 서열별 서브폴더
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rec = {"sequence": seq, "cv_started_utc": time.strftime("%FT%TZ", time.gmtime()),
               "provenance": cand.get("provenance", {}),
               "existing_stats": cand.get("existing_stats", {})}
        # 1) 25 pose (5 runs × 5 poses, 전부 PDB 저장)
        rec["docking_cv"] = robust_25pose_dock(seq, work)
        best_pdb = best_pose_pdb(rec["docking_cv"])
        rec["docking_cv"]["best_pdb"] = best_pdb
        # 2) Selectivity (heavy — 옵션)
        if do_selectivity and best_pdb:
            rec["selectivity"] = run_selectivity(best_pdb, rec["docking_cv"]["global_ddg_min"])
        # 3) ADMET (deterministic)
        rec["admet"] = run_admet(seq)
        # 4) Half-life (deterministic)
        rec["halflife"] = run_halflife(seq)
        # 5) MM-GBSA — 큐 드롭 (데몬 async 처리)
        if do_mmgbsa and best_pdb:
            rec["mmgbsa_queue"] = drop_pdb_for_mmgbsa(seq, best_pdb)
        rec["cv_finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
        rec["cv_duration_s"] = round(time.time() - t0, 1)
        with master_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return "ok"
    except Exception as exc:
        err_rec = {"sequence": seq, "provenance": cand.get("provenance", {}),
                   "cv_error": f"{type(exc).__name__}: {exc}",
                   "cv_finished_utc": time.strftime("%FT%TZ", time.gmtime())}
        with master_path.open("a") as f:
            f.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
        return f"err:{type(exc).__name__}"
    finally:
        _release_lock(lock)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/cv_analysis/candidates_manifest.json")
    ap.add_argument("--out", default="runs/cv_analysis/cv_master.jsonl")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1, help="병렬 워커 수 (index로 분할)")
    ap.add_argument("--n-runs", type=int, default=5, help="후보당 robust 도킹 반복 (기본 5)")
    ap.add_argument("--nstruct", type=int, default=int(os.environ.get("FLEXPEP_NSTRUCT", "5")))
    ap.add_argument("--no-selectivity", action="store_true", help="선택성 도킹 스킵(테스트용)")
    ap.add_argument("--no-mmgbsa", action="store_true", help="MM-GBSA 큐 드롭 스킵")
    ap.add_argument("--stop-file", default="_workspace/STOP_CV_PIPELINE")
    args = ap.parse_args()

    os.environ.setdefault("FLEXPEP_NSTRUCT", str(args.nstruct))

    manifest = json.loads((REPO / args.manifest).read_text())["manifest"]
    # 워커별 분할 (해시 기반 균형)
    my_cands = [c for c in manifest
                if int(hashlib.md5(c["sequence"].encode()).hexdigest(), 16) % args.n_workers == args.worker_id]

    master = REPO / args.out
    master.parent.mkdir(parents=True, exist_ok=True)
    work_root = REPO / "runs/cv_analysis/work"
    lock_dir = REPO / "runs/cv_analysis/locks"
    stop_path = REPO / args.stop_file

    # 필요 시 PyRosetta 1회 초기화
    import flexpep_dock as fpd
    fpd.init_pyrosetta()

    print(f"[cv-worker {args.worker_id}/{args.n_workers}] 담당 후보 {len(my_cands)}개, "
          f"selectivity={'off' if args.no_selectivity else 'on'}, "
          f"mmgbsa_queue={'off' if args.no_mmgbsa else 'on'}", file=sys.stderr, flush=True)

    stats = {"ok": 0, "skip": 0, "err": 0}
    for i, cand in enumerate(my_cands):
        if stop_path.exists():
            print(f"[cv-worker {args.worker_id}] STOP 감지 — 종료", file=sys.stderr); break
        status = process_one(cand, master, work_root, lock_dir,
                             do_selectivity=not args.no_selectivity,
                             do_mmgbsa=not args.no_mmgbsa)
        key = "ok" if status == "ok" else ("skip" if status.startswith("skip") else "err")
        stats[key] += 1
        print(f"[cv-worker {args.worker_id}] {i+1}/{len(my_cands)} {cand['sequence']} → {status}",
              file=sys.stderr, flush=True)
    print(f"[cv-worker {args.worker_id}] DONE stats={stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
