#!/usr/bin/env python3
"""MM-GBSA 배치 재채점 — 도킹된 복합체 PDB를 OpenMM single-snapshot MM-GBSA로
재채점해 도킹 ddG와 직교하는 결합에너지(ΔG_bind)를 산출하고 native 대비 비교한다.

실행: ~/miniforge3/envs/mmgbsa/bin/python scripts/run_mmgbsa_rescore.py
입력: docs/structure_view/pdb/{native_*,rank*}.pdb (chain A=수용체, B=펩타이드, ot_* 제외)
출력: runs/pyrosetta_flow/mmgbsa_rescore.json
caveat: single-snapshot, no entropy(TΔS), GBn2 implicit. 절대값 아닌 상대 비교용.
"""
import sys, os, json, glob, re

REPO = "[LOCAL_PATH]"
PDB_DIR = "[LOCAL_PATH]"
OUT = f"{REPO}/runs/pyrosetta_flow/mmgbsa_rescore.json"
sys.path.insert(0, REPO)
from pyrosetta_flow.mmgbsa_rescore import mmgbsa_rescore

# native + rank 변이체 복합체만 (ot_ 수용체 단독 제외)
pdbs = sorted(p for p in glob.glob(f"{PDB_DIR}/*.pdb")
              if not os.path.basename(p).startswith("ot_"))
print(f"[mmgbsa-batch] 대상 {len(pdbs)}개 복합체", flush=True)

results = []
for i, pdb in enumerate(pdbs, 1):
    name = os.path.basename(pdb).replace(".pdb", "")
    seq = name.split("_")[-1]  # rank01_AGCKNFFWKTDTSC → AGCKNFFWKTDTSC
    print(f"[mmgbsa-batch] {i}/{len(pdbs)} {name} ...", flush=True)
    try:
        r = mmgbsa_rescore(pdb, receptor_chains=["A"], peptide_chain="B")
    except Exception as e:
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    r["name"] = name
    r["sequence"] = seq
    results.append(r)
    print(f"    dg_bind={r.get('dg_bind')} ok={r.get('ok')} {r.get('elapsed_s','')}s", flush=True)

# native 대비 비교
nat = next((r for r in results if "native" in r["name"] and r.get("ok")), None)
nat_dg = nat["dg_bind"] if nat else None
ok = [r for r in results if r.get("ok") and r.get("dg_bind") is not None]
ok.sort(key=lambda r: r["dg_bind"])  # 낮을수록(음수 큼) 강결합
summary = {
    "native_mmgbsa_dg": nat_dg,
    "n_total": len(results), "n_ok": len(ok),
    "ranking_low_to_high": [{"name": r["name"], "seq": r["sequence"],
                             "dg_bind": round(r["dg_bind"], 2),
                             "beats_native": (nat_dg is not None and r["dg_bind"] < nat_dg)}
                            for r in ok],
    "caveat": "single-snapshot MM-GBSA(no entropy, GBn2), 상대비교용. cyclic/D-aa 미검증.",
}
json.dump({"summary": summary, "results": results}, open(OUT, "w"),
          ensure_ascii=False, indent=2)
print(f"[mmgbsa-batch] 완료 → {OUT}", flush=True)
print(f"  native ΔG_bind={nat_dg} | native 능가 후보: "
      f"{[r['name'] for r in ok if nat_dg is not None and r['dg_bind']<nat_dg]}", flush=True)
