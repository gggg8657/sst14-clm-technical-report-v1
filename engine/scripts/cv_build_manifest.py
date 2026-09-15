#!/usr/bin/env python3
"""CV 파이프라인용 후보 매니페스트 빌더 — provenance 포함.

수집 소스(진지한 후보만, 전수 ~28K는 비현실):
  A) pyrosetta_flow 리더보드 top-50 (누적, 이미 MM-GBSA/Δmargin 통과)
  B) exp72_random arm robust-clean (native 능가 + top-K)
  C) exp72_system window top single-pose (robust 재도킹 필요)
  D) silo_a_flow 리더보드 상위 (있으면)

각 후보에 provenance 붙임: 어느 실험에서 나왔는가, 언제, 어떤 mutation_source, PDB 존재 여부.
출력: runs/cv_analysis/candidates_manifest.json

기록 시 반드시 `source_experiment` 로 실험 구분 (사용자 요구).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parents[1]

NATIVE_ROBUST = -20.28
BEAT_THRESHOLD = -35.28   # native −15REU


def _num(x):
    return x if isinstance(x, (int, float)) else None


def scan_exp_log(path: Path, filter_arm=None, skip_first_lines=0):
    """experiment_log.jsonl 스캔 → seq당 첫 등장 정보."""
    first = {}
    if not path.exists():
        return first
    with path.open(errors="ignore") as f:
        for i, line in enumerate(f):
            if i < skip_first_lines:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if filter_arm and r.get("arm") != filter_arm:
                continue
            s = r.get("sequence")
            if not s or s in first:
                continue
            first[s] = {
                "mutation_source": r.get("mutation_source"),
                "ts": r.get("ts"),
                "candidate_id": r.get("candidate_id"),
                "iteration": r.get("iteration"),
                "ddg_single": _num(r.get("ddg")),
                "ddg_median": _num(r.get("ddg_median")),
                "ddg_sd": _num(r.get("ddg_sd")),
            }
    return first


def find_pdb_for(seq: str, experiment: str, candidate_id: str = None):
    """해당 실험 dir에서 후보 서열의 도킹 PDB 후보(없을 수도 있음)."""
    # exp72_random: work/{seq}.pdb
    # pyrosetta_flow: pyrosetta_flow/iter_NN/cand_NNN.pdb (candidate_id 기반)
    # silo_a_flow: 다양
    candidates = []
    if experiment == "exp72_random":
        p = REPO / "runs/exp72_random/work" / f"{seq}.pdb"
        if p.exists(): candidates.append(str(p))
    elif experiment == "pyrosetta_flow" and candidate_id:
        try:
            cn = int(candidate_id.split("cand")[1])
            for iter_dir in (REPO / "runs/pyrosetta_flow/pyrosetta_flow").glob("iter_*"):
                p = iter_dir / f"cand_{cn:03d}.pdb"
                if p.exists(): candidates.append(str(p))
        except Exception:
            pass
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/cv_analysis/candidates_manifest.json")
    ap.add_argument("--random-topk", type=int, default=30, help="랜덤 arm robust median 상위 N")
    ap.add_argument("--system-window-topk", type=int, default=40, help="시스템 window single-pose 상위 N")
    ap.add_argument("--include-native", action="store_true", default=True,
                    help="native SST-14도 포함(공정 비교 기준선)")
    args = ap.parse_args()

    manifest = {}
    # ---- A) 리더보드 top-50 (누적 시스템, 이미 robust+게이트 통과) ----
    lb_path = REPO / "runs/pyrosetta_flow/global_selectivity_leaderboard.json"
    lb = json.loads(lb_path.read_text())
    lb_entries = lb if isinstance(lb, list) else lb.get("entries", [])
    for r in lb_entries:
        s = r.get("sequence")
        if not s:
            continue
        manifest[s] = {
            "sequence": s,
            "provenance": {
                "experiments": ["pyrosetta_flow_leaderboard"],
                "primary_source": "leaderboard_top50",
                "run_id": r.get("run_id"),
            },
            "existing_stats": {
                "ddg_median": r.get("ddg_median"), "ddg_sd": r.get("ddg_sd"),
                "ddg_n_converged": r.get("ddg_n_converged"),
                "delta_margin": r.get("delta_margin"),
                "hc50": r.get("hc50"), "mmgbsa_dg": r.get("mmgbsa_dg"),
                "pose_uncertain": r.get("pose_uncertain"),
                "consensus_flag": r.get("consensus_flag"),
            },
            "existing_pdb": [],  # 리더보드는 PDB 경로 없음 (아래서 재도킹)
        }

    # ---- B) exp72_random robust-clean top-K + native 능가 전건 ----
    rand_first = scan_exp_log(REPO / "runs/exp72_random/experiment_log.jsonl", filter_arm="random")
    rand_recs = []
    for seq, info in rand_first.items():
        med = info.get("ddg_median")
        sd = info.get("ddg_sd")
        if med is None or sd is None:
            continue
        if sd > 15 or sd == 0:  # floppy 또는 n=1 배제
            continue
        rand_recs.append({"sequence": seq, "median": med, "sd": sd, "info": info})
    rand_recs.sort(key=lambda r: r["median"])
    # native 능가 전건 + top-K 상위 median
    rand_selected = set()
    for r in rand_recs:
        if r["median"] <= BEAT_THRESHOLD:
            rand_selected.add(r["sequence"])
    for r in rand_recs[:args.random_topk]:
        rand_selected.add(r["sequence"])
    for seq in rand_selected:
        info = rand_first[seq]
        pdbs = find_pdb_for(seq, "exp72_random")
        if seq not in manifest:
            manifest[seq] = {"sequence": seq, "provenance": {"experiments": []},
                             "existing_stats": {}, "existing_pdb": []}
        entry = manifest[seq]
        entry["provenance"]["experiments"].append("exp72_random")
        entry["provenance"].setdefault("random_arm", {}).update({
            "mutation_source": info.get("mutation_source"),
            "ts": info.get("ts"),
        })
        entry["existing_stats"].setdefault("random_arm_ddg_median", info.get("ddg_median"))
        entry["existing_stats"].setdefault("random_arm_ddg_sd", info.get("ddg_sd"))
        entry["existing_pdb"].extend(pdbs)

    # ---- C) exp72_system window top-K by single-pose ddg ----
    win_start = 42280
    marker_path = REPO / "runs/exp72_system/WINDOW_MARKER.json"
    if marker_path.exists():
        win_start = json.loads(marker_path.read_text())["experiment_log_start_lines"]
    sys_first = scan_exp_log(REPO / "runs/pyrosetta_flow/experiment_log.jsonl",
                              filter_arm=None, skip_first_lines=win_start)
    # skip arm==random within same log (재사용 방어)
    sys_first = {s: v for s, v in sys_first.items() if v.get("mutation_source") != "random_baseline"}
    sys_ranked = [(s, v) for s, v in sys_first.items() if v.get("ddg_single") is not None]
    sys_ranked.sort(key=lambda x: x[1]["ddg_single"])
    for seq, info in sys_ranked[:args.system_window_topk]:
        pdbs = find_pdb_for(seq, "pyrosetta_flow", info.get("candidate_id"))
        if seq not in manifest:
            manifest[seq] = {"sequence": seq, "provenance": {"experiments": []},
                             "existing_stats": {}, "existing_pdb": []}
        entry = manifest[seq]
        entry["provenance"]["experiments"].append("exp72_system_window")
        entry["provenance"].setdefault("system_window", {}).update({
            "mutation_source": info.get("mutation_source"),
            "candidate_id": info.get("candidate_id"),
            "iteration": info.get("iteration"),
            "ts": info.get("ts"),
        })
        entry["existing_stats"].setdefault("system_window_single_ddg", info.get("ddg_single"))
        entry["existing_pdb"].extend(pdbs)

    # ---- D) Silo A 리더보드 (있으면) ----
    silo_lb = REPO / "runs/silo_a_flow/global_selectivity_leaderboard.json"
    if silo_lb.exists():
        try:
            sl = json.loads(silo_lb.read_text())
            sl_entries = sl if isinstance(sl, list) else sl.get("entries", [])
            for r in sl_entries[:30]:
                s = r.get("sequence")
                if not s: continue
                if s not in manifest:
                    manifest[s] = {"sequence": s, "provenance": {"experiments": []},
                                    "existing_stats": {}, "existing_pdb": []}
                entry = manifest[s]
                entry["provenance"]["experiments"].append("silo_a_flow")
                entry["provenance"].setdefault("silo_a", {}).update({"run_id": r.get("run_id")})
                entry["existing_stats"].setdefault("silo_a_ddg_median", r.get("ddg_median"))
        except Exception as exc:
            print(f"  Silo A 리더보드 로드 실패: {exc}", file=sys.stderr)

    # ---- native 포함 ----
    if args.include_native:
        manifest.setdefault("AGCKNFFWKTFTSC", {
            "sequence": "AGCKNFFWKTFTSC",
            "provenance": {"experiments": ["native_baseline"], "primary_source": "SST-14 wild-type"},
            "existing_stats": {"canonical_robust_ddg": NATIVE_ROBUST,
                                "canonical_hc50": -55.6816},
            "existing_pdb": [],
        })

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_candidates": len(manifest),
                                "manifest": list(manifest.values())},
                               ensure_ascii=False, indent=2))
    print(f"  후보 {len(manifest)}개 → {out}", file=sys.stderr)
    # 요약 출력
    src_count = defaultdict(int)
    pdb_have = 0
    for v in manifest.values():
        for e in v["provenance"].get("experiments", []):
            src_count[e] += 1
        if v.get("existing_pdb"):
            pdb_have += 1
    print(f"  출처별: {dict(src_count)}", file=sys.stderr)
    print(f"  기존 PDB 보유: {pdb_have}/{len(manifest)}", file=sys.stderr)


if __name__ == "__main__":
    main()
