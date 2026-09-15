#!/usr/bin/env python3
"""EXP72 마스터 스코어보드 추가 스코어 backfill.

기존 master_candidates.jsonl 에 미부착 스코어 추가 → master_candidates_v2.jsonl.
- radiolysis_susceptibility (방사분해 total_score/risk — 방사성의약품 핵심)  [즉시]
- admet_reasonableness (ADMET 종합 HEURISTIC, radiolysis 반영)               [즉시]
- (옵션 --with-surrogate) stability_norm + hc50 재계산 (느린 서로게이트)

radiolysis/admet 는 순수 서열기반 = 33k 전수 수초. surrogate 는 느려서 옵션.
NO MOCK: 실제 함수 호출. surrogate 파일 미수정(호출만).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="runs/exp72_analysis/master_candidates.jsonl")
    ap.add_argument("--out", default="runs/exp72_analysis/master_candidates_v2.jsonl")
    ap.add_argument("--with-surrogate", action="store_true", help="stability_norm+hc50 재계산(느림)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    from backend.pharmacology import radiolysis_susceptibility
    from pyrosetta_flow.multiobjective import admet_reasonableness
    hl_fn = tox_fn = None
    if args.with_surrogate:
        from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife as hl_fn
        from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences as tox_fn

    inp = REPO / args.inp; outp = REPO / args.out
    done = set()
    if outp.exists():
        for l in outp.open(errors="ignore"):
            try: done.add(json.loads(l)["sequence"])
            except Exception: pass

    n = 0
    with outp.open("a") as f:
        for idx, line in enumerate(inp.open(errors="ignore")):
            if idx % args.shard_count != args.shard_index: continue   # 샤딩
            try: r = json.loads(line)
            except Exception: continue
            if r["sequence"] in done: continue
            seq = r["sequence"]; m = r.setdefault("metrics", {})
            # radiolysis (즉시)
            try:
                rad = radiolysis_susceptibility(seq)
                m["radiolysis_total"] = rad.get("total_score")
                m["radiolysis_risk"] = rad.get("risk_level")
            except Exception as e:
                m["radiolysis_err"] = str(e)
            # admet_reasonableness (props = pharmacology_full + radiolysis_total_score)
            try:
                pf = m.get("pharmacology_full", {})
                # admet_reasonableness 기대 키로 매핑, None 은 제외(함수 기본값 사용)
                cand_props = {"instability_index": pf.get("instability_index"),
                              "gravy": pf.get("gravy"), "boman_index": pf.get("boman_index"),
                              "pi": pf.get("isoelectric_point"),
                              "radiolysis_total_score": m.get("radiolysis_total")}
                props = {k: v for k, v in cand_props.items() if v is not None}
                m["admet_reasonableness"] = admet_reasonableness(props)
            except Exception as e:
                m["admet_err"] = str(e)
            # surrogate (옵션, 느림) — stability_norm(HLRRS) + hc50(hemolysis ADMET, per-seq=배치버그 회피)
            if args.with_surrogate:
                try:
                    hl = hl_fn(seq); m["stability_norm"] = hl.get("stability_norm")
                    if m.get("half_life_h") is None: m["half_life_h"] = hl.get("half_life_h")
                except Exception as e:
                    m["stability_err"] = str(e)
                try:
                    tr = tox_fn([seq]).get(seq, {})   # 단일 호출=SILO_A 배치오염 회피
                    m["hc50"] = tr.get("hc50") if tr.get("available") else None
                    m["tox_available"] = bool(tr.get("available"))
                except Exception as e:
                    m["tox_err"] = str(e)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
            if n % 5000 == 0: print(f"  {n} enriched", file=sys.stderr, flush=True)
    print(f"[enrich] +{n} → {outp}", file=sys.stderr)


if __name__ == "__main__":
    main()
