#!/usr/bin/env python3
"""EXP72 선택성 정밀 (Stage 2) — off-target 도킹 포즈를 MM-GBSA로 재측정 (정합성 판정).

요구 #3: 도킹된 selectivity 결과(off-target 복합체)를 MM-GBSA로 재측정 → FlexPepDock 선택성과 정합성 판정.
입력: offtarget_full.jsonl (후보×subtype pose_pdb). mmgbsa env 실행.
출력: offtarget_mmgbsa.jsonl (후보×subtype off-target MM-GBSA dG).
체인 자동감지: 펩타이드=최소잔기 체인(≈14), 수용체=나머지. NO MOCK.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))


def detect_chains(pdb):
    """(receptor_chains, peptide_chain) 자동감지: 최소잔기 체인=펩타이드."""
    res = {}
    for ln in open(pdb, errors="ignore"):
        if ln.startswith("ATOM"):
            ch = ln[21]; rn = ln[22:26].strip()
            res.setdefault(ch, set()).add(rn)
    if not res:
        return None, None
    counts = {c: len(s) for c, s in res.items()}
    pep = min(counts, key=counts.get)
    rec = [c for c in counts if c != pep]
    return rec, pep


def main():
    from pyrosetta_flow.mmgbsa_rescore import mmgbsa_rescore
    out_path = REPO / "runs/exp72_system/offtarget_mmgbsa.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.open(errors="ignore"):
            try:
                r = json.loads(l)
                if r.get("mmgbsa_dg") is not None:
                    done.add((r["sequence"], r["subtype"]))
            except Exception:
                pass

    # Stage1 결과에서 (seq, subtype, pose) 수집
    tasks = []
    fp = REPO / "runs/exp72_system/offtarget_full.jsonl"
    for l in fp.open(errors="ignore"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        for st, v in (r.get("offtarget") or {}).items():
            if v.get("status") == "ok" and v.get("pose_pdb") and Path(v["pose_pdb"]).exists():
                if (r["sequence"], st) not in done:
                    tasks.append((r["sequence"], r.get("arm"), st, v["pose_pdb"], v.get("flexpep_ddg")))
    print(f"[offtarget-mmgbsa] 처리 {len(tasks)} (완료 {len(done)})", file=sys.stderr, flush=True)

    for seq, arm, st, pose, fp_ddg in tasks:
        rec = {"sequence": seq, "arm": arm, "subtype": st, "flexpep_ddg": fp_ddg,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            rec_chains, pep = detect_chains(pose)
            rec["chains"] = {"receptor": rec_chains, "peptide": pep}
            res = mmgbsa_rescore(pose, receptor_chains=rec_chains, peptide_chain=pep,
                                 platform_name="CUDA", max_minimize_iterations=500)
            rec["mmgbsa_dg"] = round(res["dg_bind"], 3) if res.get("ok") else None
            if not res.get("ok"):
                rec["error"] = res.get("error", "")[:150]
        except Exception as exc:
            rec["mmgbsa_dg"] = None; rec["error"] = f"{type(exc).__name__}: {exc}"
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"  {seq} {st}: flexpep={fp_ddg} mmgbsa={rec.get('mmgbsa_dg')}", file=sys.stderr, flush=True)
    print("[offtarget-mmgbsa] DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
