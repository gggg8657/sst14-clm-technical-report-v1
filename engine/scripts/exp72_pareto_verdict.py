#!/usr/bin/env python3
"""EXP72 5축 다목적 파레토 판정 초안.

5축(shortlist 후보):
 1. FlexPepDock ddG20 (robust median, ↓강함)   ← phase1_confirm.jsonl
 2. MM-GBSA dg_bind (↓강함, 직교 결합지표)       ← v3b_mmgbsa.jsonl
 3. 선택성 Δmargin (↑좋음)                        ← v3_selectivity.jsonl
 4. 이황결합 SS (normal=제약, compressed=아티팩트 실격)  ← phase1_confirm
 5. 개발성 (GRAVY↓·aromatic%↓, 응집/합성 위험)     ← master_candidates(scoreboard)

파레토: 정상SS 후보 중 3개 결합/선택성 목적(bind_flexpep, bind_mmgbsa, selectivity)으로
비지배 전선 산출. 개발성은 제약/타이브레이크. provenance(random vs system_llm) 태깅.
판정 초안: 전선의 arm 구성, native 지배 여부, 승자 유무.
NO MOCK. 선택성 결측이면 '미완' 표기하고 가용 축으로만 잠정.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NATIVE = "AGCKNFFWKTFTSC"
NATIVE_DDG, NATIVE_MMGBSA = -20.28, -71.053


def load(p, key="sequence"):
    d = {}
    fp = REPO / p
    if not fp.exists():
        return d
    for l in fp.open(errors="ignore"):
        try:
            r = json.loads(l); d[r[key]] = r
        except Exception:
            pass
    return d


def dominates(a, b, objs):
    """a가 b를 지배? (모든 목적에서 >=, 하나에서 >). 목적은 '클수록 좋음'으로 정규화된 값."""
    ge = all(a[o] >= b[o] for o in objs)
    gt = any(a[o] > b[o] for o in objs)
    return ge and gt


def main():
    rd = load("runs/exp72_system/phase1_confirm.jsonl")
    mg = load("runs/exp72_system/v3b_mmgbsa.jsonl")
    sel = load("runs/exp72_system/v3_selectivity.jsonl")
    sb = {}
    for l in (REPO / "runs/exp72_analysis/master_candidates.jsonl").open(errors="ignore"):
        try:
            r = json.loads(l); sb[r["sequence"]] = r.get("metrics", {})
        except Exception:
            pass

    # 후보 조립
    cands = []
    for seq, r in rd.items():
        if r.get("status") != "ok" or r.get("ddg_median") is None:
            continue
        g = mg.get(seq, {}); s = sel.get(seq, {}); m = sb.get(seq, {})
        cands.append({
            "seq": seq, "arm": r.get("arm"),
            "ddg20": r.get("ddg_median"), "ss": r.get("disulfide_flag"),
            "mmgbsa": g.get("dg_bind"), "delta_margin": s.get("delta_margin"),
            "gravy": m.get("gravy"), "aromatic": m.get("aromatic_frac"),
        })

    n_sel = sum(1 for c in cands if c["delta_margin"] is not None)
    print("=" * 74)
    print(f"EXP72 5축 다목적 파레토 판정 초안  (선택성 {n_sel}/{len(cands)} 완료)")
    print("=" * 74)

    # 정상 SS만 (compressed=아티팩트 실격) + 3목적 다 있는 것
    viable = [c for c in cands if c["ss"] == "normal"
              and c["mmgbsa"] is not None and c["ddg20"] is not None]
    have_sel = [c for c in viable if c["delta_margin"] is not None]

    # 목적 정규화(클수록 좋음): -ddg20, -mmgbsa, +delta_margin
    def prep(c):
        c["_bind_fp"] = -c["ddg20"]; c["_bind_mm"] = -c["mmgbsa"]
        c["_sel"] = c["delta_margin"]
        return c
    objs = ["_bind_fp", "_bind_mm", "_sel"]

    if len(have_sel) >= 2:
        pts = [prep(dict(c)) for c in have_sel]
        front = [a for a in pts if not any(dominates(b, a, objs) for b in pts if b is not a)]
        print(f"\n[3축 파레토 전선] (정상SS·ddG·MMGBSA·Δmargin 모두 보유 {len(pts)}개 중)")
        for c in sorted(front, key=lambda x: x["ddg20"]):
            print(f"  ★ [{c['arm']:10}] {c['seq']} ddG={c['ddg20']:.1f} MMGBSA={c['mmgbsa']:.1f} "
                  f"Δmargin={c['delta_margin']:.2f} | GRAVY={c['gravy']} arom={c['aromatic']}")
        from collections import Counter
        arm_ct = Counter(c["arm"] for c in front)
        print(f"\n  전선 arm 구성: {dict(arm_ct)}")
    else:
        print(f"\n[선택성 미완 — {len(have_sel)}개만 Δmargin 보유] 2축(ddG+MMGBSA) 잠정 파레토:")
        pts = [c for c in viable]
        for c in pts:
            c["_bind_fp"] = -c["ddg20"]; c["_bind_mm"] = -c["mmgbsa"]
        objs2 = ["_bind_fp", "_bind_mm"]
        front = [a for a in pts if not any(dominates(b, a, objs2) for b in pts if b is not a)]
        for c in sorted(front, key=lambda x: x["ddg20"]):
            print(f"  ○ [{c['arm']:10}] {c['seq']} ddG={c['ddg20']:.1f} MMGBSA={c['mmgbsa']:.1f} "
                  f"| GRAVY={c['gravy']} arom={c['aromatic']} (Δmargin 대기)")

    # native 대비
    print(f"\n[native 기준] ddG={NATIVE_DDG} MMGBSA={NATIVE_MMGBSA}"
          f" Δmargin={sel.get(NATIVE,{}).get('delta_margin')}")
    beat_both = [c for c in viable if c["ddg20"] < NATIVE_DDG - 15 and c["mmgbsa"] < NATIVE_MMGBSA]
    print(f"  ddG(15REU유의)+MMGBSA 동시 native 능가(정상SS): {len(beat_both)}개"
          + (f" → {[c['seq'] for c in beat_both]}" if beat_both else " (없음)"))

    # 실격(압축SS) 요약
    dq = [c for c in cands if c["ss"] != "normal"]
    print(f"\n[아티팩트 실격] 압축/파손 SS: {len(dq)}개 "
          f"(random {sum(1 for c in dq if c['arm']=='random')} / system {sum(1 for c in dq if c['arm']=='system_llm')})")
    print("=" * 74)


if __name__ == "__main__":
    main()
