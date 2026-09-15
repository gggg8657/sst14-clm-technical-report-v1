#!/usr/bin/env python3
"""EXP72 선택성 정밀 (Stage 1) — 후보×SSTR1/3/4/5 개별 도킹 + 포즈 보존 + 실패 표기.

요구(2026-07-14):
 1. SSTR1/3/4/5 각각 도킹 스코어 측정, 도킹 실패 시 "docking simulation failed" 명시.
 3. off-target 도킹 포즈를 후보×subtype별로 보존 → Stage 2에서 MM-GBSA 정합성 재측정.

offtarget_dock.py 직접 호출(conda run 오버헤드 제거) + 스레드핀. 포즈 = work/ot_{seq}_{subtype}.pdb.
입력: phase1_confirm(status ok) 후보 = SSTR2 복합체 포즈(phase1_work). 출력: offtarget_full.jsonl.
NO MOCK: 실제 FlexPepDock. 실패=정직 표기.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ALIGN = REPO / "_workspace/05_engineer-backend_offtarget_receptors_seqalign"
SUBTYPES = {
    "SSTR1": str(ALIGN / "SSTR1_receptor_9ik8_seqalign.pdb"),
    "SSTR3": str(ALIGN / "SSTR3_receptor_8xir_seqalign.pdb"),
    "SSTR4": str(ALIGN / "SSTR4_receptor_7xmt_seqalign.pdb"),
    "SSTR5": str(ALIGN / "SSTR5_receptor_8zbj_seqalign.pdb"),
}
DOCK = str(REPO / "AG_src/scripts/offtarget_dock.py")
BIO = "[LOCAL_PATH]"


def _num(x):
    return x if isinstance(x, (int, float)) else None


def load_targets(targets_json=None):
    # --targets 로 winner 리스트(json: [{sequence, sstr2_pose, sstr2_ddg, arm}]) 지정 가능(funnel용)
    if targets_json:
        d = json.loads((REPO / targets_json).read_text())
        items = d if isinstance(d, list) else d.get("winners", d.get("candidates", []))
        out = []
        for r in items:
            pdb = r.get("sstr2_pose") or r.get("pdb")
            if pdb and Path(pdb).exists():
                out.append({"sequence": r["sequence"], "arm": r.get("arm", "pool"),
                            "sstr2_pose": pdb, "sstr2_ddg": r.get("sstr2_ddg") or r.get("ddg_median")})
        return out
    out = []
    seen = set()
    work = REPO / "runs/exp72_system/phase1_work"
    for line in (REPO / "runs/exp72_system/phase1_confirm.jsonl").open(errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("status") != "ok" or r["sequence"] in seen:
            continue
        pdb = work / f"{r['sequence']}.pdb"
        if pdb.exists():
            out.append({"sequence": r["sequence"], "arm": r.get("arm"), "sstr2_pose": str(pdb),
                        "sstr2_ddg": r.get("ddg_median")})
            seen.add(r["sequence"])
    return out


def dock_one(sstr2_pose, subtype_pdb, out_pdb, timeout):
    """offtarget_dock 1회. 반환 (ddg or None, status). 실패=docking simulation failed."""
    try:
        p = subprocess.run([BIO, DOCK, "--sstr2-complex", sstr2_pose,
                            "--offtarget-receptor", subtype_pdb, "--output", out_pdb, "--pre-aligned"],
                           capture_output=True, text=True, timeout=timeout,
                           env={**os.environ,  # ★ 전체 env 상속(LD_LIBRARY_PATH 등 pyrosetta 필수) + 오버라이드만
                                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                                "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                                "FLEXPEP_NSTRUCT": "1"})  # 단일 pose(MM-GBSA 입력용). Δmargin robust는 v3_selectivity
    except subprocess.TimeoutExpired:
        return None, "docking simulation failed (timeout)"
    except Exception as exc:
        return None, f"docking simulation failed ({type(exc).__name__})"
    if p.returncode != 0:
        return None, f"docking simulation failed (rc={p.returncode})"
    # stdout 마지막 JSON 라인 파싱
    for ln in reversed(p.stdout.strip().splitlines()):
        try:
            j = json.loads(ln)
            ddg = _num(j.get("ddg") if "ddg" in j else j.get("ddg_median"))
            if ddg is None:
                return None, "docking simulation failed (ddg 없음)"
            return ddg, "ok"
        except Exception:
            continue
    return None, "docking simulation failed (출력 파싱 실패)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=3000)
    ap.add_argument("--out", default="runs/exp72_system/offtarget_full.jsonl")
    ap.add_argument("--targets", default=None, help="winner 리스트 json (funnel용). 미지정=shortlist")
    ap.add_argument("--poses-dir", default="runs/exp72_system/offtarget_poses")
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = REPO / args.poses_dir; work.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.open(errors="ignore"):
            try: done.add(json.loads(l)["sequence"])
            except Exception: pass

    tg = load_targets(args.targets)
    mine = [t for i, t in enumerate(tg) if i % args.shard_count == args.shard_index and t["sequence"] not in done]
    print(f"[offtarget-full shard {args.shard_index}/{args.shard_count}] 처리 {len(mine)}", file=sys.stderr, flush=True)

    for t in mine:
        seq = t["sequence"]
        rec = {"sequence": seq, "arm": t["arm"], "sstr2_ddg": t["sstr2_ddg"],
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "offtarget": {}}
        for st, pdb in SUBTYPES.items():
            pose = str(work / f"ot_{seq}_{st}.pdb")
            ddg, status = dock_one(t["sstr2_pose"], pdb, pose, args.timeout)
            rec["offtarget"][st] = {"flexpep_ddg": ddg, "status": status,
                                    "pose_pdb": pose if status == "ok" else None}
            print(f"  {seq} {st}: {ddg if ddg is not None else status}", file=sys.stderr, flush=True)
        # Δmargin(flexpep) = min(offtarget) - sstr2  (양수=SSTR2 선택적)
        offs = [v["flexpep_ddg"] for v in rec["offtarget"].values() if v["flexpep_ddg"] is not None]
        rec["n_offtarget_ok"] = len(offs)
        rec["delta_margin_flexpep"] = (min(offs) - t["sstr2_ddg"]) if offs and t["sstr2_ddg"] is not None else None
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"[offtarget-full shard {args.shard_index}] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
