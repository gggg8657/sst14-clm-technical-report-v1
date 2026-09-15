#!/usr/bin/env python3
"""EXP72 shortlist 완전 테이블 — 전 지표 통합 + 선택성 2방법(FlexPepDock vs MM-GBSA) 정합성.

통합 소스:
 - phase1_confirm.jsonl        : ddG20(robust), SS(이황결합)
 - v3b_mmgbsa.jsonl            : on-target(SSTR2) MM-GBSA
 - v3_selectivity.jsonl        : 집계 Δmargin + per-subtype offtarget_ddg(FlexPepDock)
 - offtarget_full.jsonl        : per-subtype SSTR1/3/4/5 개별 도킹 + 실패표기(Stage1)
 - offtarget_mmgbsa.jsonl      : per-subtype off-target MM-GBSA(Stage2, 정합성)
 - master_candidates_v2.jsonl  : 개발성/ADMET/radiolysis/hc50/stability

출력: 콘솔 요약 + `_workspace/EXP72_SHORTLIST_FULL_TABLE.md`(markdown) + `.json`.
정합성: Δmargin(FlexPepDock) vs Δmargin(MM-GBSA) 부호 일치 여부.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load(p, key="sequence"):
    d = {}
    fp = REPO / p
    if fp.exists():
        for l in fp.open(errors="ignore"):
            try:
                r = json.loads(l); d[r[key]] = r
            except Exception:
                pass
    return d


def _num(x):
    return x if isinstance(x, (int, float)) else None


def main():
    rd = load("runs/exp72_system/phase1_confirm.jsonl")
    mg = load("runs/exp72_system/v3b_mmgbsa.jsonl")
    sel = load("runs/exp72_system/v3_selectivity.jsonl")
    of = load("runs/exp72_system/offtarget_full.jsonl")
    # Stage2 off-target mmgbsa: (seq,subtype) 단위 → seq별 dict 로 재구성
    otmm = {}
    fp = REPO / "runs/exp72_system/offtarget_mmgbsa.jsonl"
    if fp.exists():
        for l in fp.open(errors="ignore"):
            try:
                r = json.loads(l); otmm.setdefault(r["sequence"], {})[r["subtype"]] = _num(r.get("mmgbsa_dg"))
            except Exception:
                pass
    sb = {}
    for l in (REPO / "runs/exp72_analysis/master_candidates_v2.jsonl").open(errors="ignore"):
        try:
            r = json.loads(l); sb[r["sequence"]] = r.get("metrics", {})
        except Exception:
            pass
    # hc50/stability 는 surro_shards(별도 backfill)에 있음 → sb 에 병합
    import glob as _glob
    for f in _glob.glob(str(REPO / "runs/exp72_analysis/surro_shards/v3_shard_*.jsonl")):
        for l in open(f, errors="ignore"):
            try:
                r = json.loads(l); m = r.get("metrics", {})
                if r["sequence"] in sb:
                    if m.get("hc50") is not None: sb[r["sequence"]]["hc50"] = m["hc50"]
                    if m.get("stability_norm") is not None: sb[r["sequence"]]["stability_norm"] = m["stability_norm"]
            except Exception:
                pass

    subtypes = ["SSTR1", "SSTR3", "SSTR4", "SSTR5"]
    rows = []
    for seq, r in rd.items():
        if r.get("status") != "ok":
            continue
        arm = r.get("arm"); m = sb.get(seq, {})
        ont_mm = _num((mg.get(seq) or {}).get("dg_bind"))
        offf = (of.get(seq) or {}).get("offtarget", {})
        offmm = otmm.get(seq, {})
        sel_off = (sel.get(seq) or {}).get("offtarget_ddg") or {}   # v3_selectivity per-subtype(nstruct2 robust)
        # per-subtype FlexPepDock = v3_selectivity 사용. (Stage1 flexpep는 nstruct1 비수렴 0.0 아티팩트라 미사용,
        #  Stage1은 MM-GBSA용 포즈 생성 전용.)
        st_fp = {st: sel_off.get(st) for st in subtypes}
        st_fp_status = {st: (offf.get(st) or {}).get("status") for st in subtypes}
        st_mm = {st: offmm.get(st) for st in subtypes}   # #3 off-target MMGBSA(Stage2)
        offs_mm = [v for v in st_mm.values() if isinstance(v, (int, float))]
        dm_fp = _num((sel.get(seq) or {}).get("delta_margin"))   # v3 authoritative Δmargin(FlexPepDock)
        dm_mm = (min(offs_mm) - ont_mm) if offs_mm and ont_mm is not None else None
        # 정합성: 두 방법 Δmargin 부호 일치?
        consistency = None
        if dm_fp is not None and dm_mm is not None:
            consistency = "일치" if (dm_fp > 0) == (dm_mm > 0) else "불일치"
        rows.append({
            "sequence": seq, "arm": arm, "ddg20": r.get("ddg_median"), "ddg20_sd": r.get("ddg_sd"),
            "ss": r.get("disulfide_flag"), "mmgbsa_ontarget": ont_mm,
            "delta_margin_agg": _num((sel.get(seq) or {}).get("delta_margin")),
            "dm_flexpep": dm_fp, "dm_mmgbsa": dm_mm, "selectivity_consistency": consistency,
            "offtarget_flexpep": st_fp, "offtarget_status": st_fp_status, "offtarget_mmgbsa": st_mm,
            "gravy": m.get("gravy"), "aromatic": m.get("aromatic_frac"),
            "admet_reasonableness": m.get("admet_reasonableness"), "radiolysis": m.get("radiolysis_total"),
            "hc50": m.get("hc50"), "half_life_h": m.get("half_life_h"), "dota_site": m.get("dota_site"),
        })
    rows.sort(key=lambda x: (x["ss"] != "normal", -(x["delta_margin_agg"] or -999)))

    # markdown
    out = REPO / "_workspace/EXP72_SHORTLIST_FULL_TABLE.md"
    with out.open("w") as f:
        f.write("# EXP72 shortlist 완전 검증 테이블 (2026-07-14)\n\n")
        f.write("전 지표 통합 + 선택성 2방법(FlexPepDock vs MM-GBSA) 정합성. 정상SS 우선, Δmargin 순.\n\n")
        f.write("| 서열 | arm | ddG20(sd) | MMGBSA | SS | Δmargin(집계) | Δmg_flexpep | Δmg_mmgbsa | 정합성 | GRAVY | arom% | ADMET | radiolysis | hc50 | t½h | DOTA |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for x in rows:
            f.write(f"| {x['sequence']} | {x['arm']} | {x['ddg20']:.1f}({x['ddg20_sd']}) | "
                    f"{x['mmgbsa_ontarget']} | {x['ss']} | {x['delta_margin_agg']} | "
                    f"{round(x['dm_flexpep'],2) if x['dm_flexpep'] is not None else '-'} | "
                    f"{round(x['dm_mmgbsa'],2) if x['dm_mmgbsa'] is not None else '-'} | "
                    f"{x['selectivity_consistency'] or '-'} | {x['gravy']} | {x['aromatic']} | "
                    f"{round(x['admet_reasonableness'],3) if x['admet_reasonableness'] is not None else '-'} | "
                    f"{x['radiolysis']} | {x['hc50']} | {x['half_life_h']} | {x['dota_site']} |\n")
        # per-subtype 상세 블록
        f.write("\n## per-subtype off-target 상세 (FlexPepDock ddG / MM-GBSA / 상태)\n\n")
        f.write("| 서열 | SSTR1 | SSTR3 | SSTR4 | SSTR5 |\n|---|---|---|---|---|\n")
        for x in rows:
            def cell(st):
                fp = x["offtarget_flexpep"].get(st); mm = x["offtarget_mmgbsa"].get(st); s = x["offtarget_status"].get(st)
                if s and s != "ok":
                    return "docking simulation failed"
                return f"fp={round(fp,1) if fp is not None else '-'}/mm={round(mm,1) if mm is not None else '-'}"
            f.write(f"| {x['sequence']} | {cell('SSTR1')} | {cell('SSTR3')} | {cell('SSTR4')} | {cell('SSTR5')} |\n")

    (REPO / "_workspace/EXP72_SHORTLIST_FULL_TABLE.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

    # 콘솔 요약
    cons_ok = sum(1 for x in rows if x["selectivity_consistency"] == "일치")
    cons_no = sum(1 for x in rows if x["selectivity_consistency"] == "불일치")
    print(f"shortlist {len(rows)}개 완전테이블 → {out}")
    print(f"선택성 2방법 정합성: 일치 {cons_ok} / 불일치 {cons_no} / 미완 {len(rows)-cons_ok-cons_no}")
    print(f"{'seq':15}{'arm':11}{'ddG20':>7}{'MMGBSA':>8}{'SS':>10}{'Δmg':>7}{'정합':>6}")
    for x in rows:
        print(f"{x['sequence']:15}{str(x['arm']):11}{x['ddg20']:7.1f}"
              f"{str(x['mmgbsa_ontarget']):>8}{str(x['ss']):>10}"
              f"{str(round(x['delta_margin_agg'],1) if x['delta_margin_agg'] is not None else '-'):>7}"
              f"{str(x['selectivity_consistency'] or '-'):>6}")


if __name__ == "__main__":
    main()
