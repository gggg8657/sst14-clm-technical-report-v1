#!/usr/bin/env python3
"""CV 파이프라인 결과 집계 → wet-lab 제안 후보 통합 테이블.

입력: runs/cv_analysis/cv_master.jsonl (진행 중이어도 부분 결과 집계)
      runs/pyrosetta_flow/mmgbsa_consensus.json (MM-GBSA 데몬 결과, 있으면 병합)

출력:
  runs/cv_analysis/wetlab_candidates.json   (프로그래매틱 접근용)
  runs/cv_analysis/wetlab_candidates.csv    (검토·엑셀용)
  runs/cv_analysis/wetlab_proposal.md       (사람 읽기용, 실험 제안 초안)

각 행에 **source_experiment** 필수 명시 (사용자 요구 — 실험 구분).

Tier 기준 (multiobjective gate):
  S = global_ddg_median ≤ −35, Δmargin > 5, hc50 native 동급이상, half-life > 0.05h, MM-GBSA < −60
  A = ≤ −30, Δmargin > 0, hc50 동급, half-life > 0.03h
  B = ≤ −25, Δmargin > 0
  C = 그 외 (fail-flag 이유 명시)
"""
from __future__ import annotations
import argparse, json, csv, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
NATIVE_HC50 = -55.6816
NATIVE_ROBUST = -20.28


def _num(x): return x if isinstance(x, (int, float)) else None


def load_cv_master(path: Path):
    """seq별 최신 레코드(마지막 성공 진입, error는 최후수단)."""
    latest = {}
    if not path.exists(): return latest
    for line in path.open(errors="ignore"):
        try: r = json.loads(line)
        except Exception: continue
        s = r.get("sequence")
        if not s: continue
        # 성공 우선(cv_error 없는 것)
        if s in latest and "cv_error" in r and "cv_error" not in latest[s]:
            continue
        latest[s] = r
    return latest


def load_mmgbsa(path: Path):
    if not path.exists(): return {}
    try:
        d = json.loads(path.read_text())
        # {"per_candidate": [{sequence, mmgbsa_dg, ...}]} 형태로 가정
        by = {}
        entries = d.get("per_candidate", d if isinstance(d, list) else [])
        for e in entries:
            s = e.get("sequence")
            if s: by[s] = _num(e.get("mmgbsa_dg") or e.get("dg") or e.get("mmgbsa"))
        return by
    except Exception as exc:
        print(f"  mmgbsa 로드 스킵: {exc}", file=sys.stderr)
        return {}


def tier_and_reasons(rec, mmgbsa):
    d = rec.get("docking_cv", {})
    s = rec.get("selectivity", {}) or {}
    a = rec.get("admet", {}) or {}
    h = rec.get("halflife", {}) or {}
    med = _num(d.get("global_ddg_median"))
    dm = _num(s.get("selectivity_margin"))
    hv = _num(a.get("hc50_vs_native"))
    hl = _num(h.get("half_life_h"))
    mm = _num(mmgbsa)
    reasons = []
    # Tier S — 최우선 wet-lab
    if (med is not None and med <= -35 and dm is not None and dm > 5
        and (hv is None or hv >= -5) and (hl is None or hl > 0.05)
        and (mm is None or mm < -60)):
        return "S", ["다목적 전지표 통과"]
    # Tier A
    if (med is not None and med <= -30 and dm is not None and dm > 0
        and (hv is None or hv >= -5) and (hl is None or hl > 0.03)):
        return "A", ["binding·selectivity·tox 통과"]
    # Tier B
    if med is not None and med <= -25 and dm is not None and dm > 0:
        return "B", ["기본 binding·selectivity 통과, tox/t½ 유보"]
    # Tier C — reasons 나열
    if med is None: reasons.append("도킹 결과 없음")
    elif med > -25: reasons.append(f"약결합(med={med:.1f})")
    if dm is None: reasons.append("선택성 미측정")
    elif dm <= 0: reasons.append(f"오프타겟 우세(Δ={dm:.1f})")
    if hv is not None and hv < -5: reasons.append(f"native보다 독성↑(hc50Δ={hv:.1f})")
    if hl is not None and hl < 0.03: reasons.append(f"반감기 짧음({hl:.3f}h)")
    return "C", reasons


def build_row(seq, rec, mm):
    d = rec.get("docking_cv", {}) or {}
    s = rec.get("selectivity", {}) or {}
    a = rec.get("admet", {}) or {}
    h = rec.get("halflife", {}) or {}
    prov = rec.get("provenance", {}) or {}
    exps = prov.get("experiments", []) or []
    tier, reasons = tier_and_reasons(rec, mm)
    return {
        "sequence": seq,
        "tier": tier,
        "source_experiment": ";".join(exps) or "unknown",   # ★ 사용자 요구
        "primary_source": prov.get("primary_source") or (exps[0] if exps else None),
        # 도킹 CV (25 pose 기반)
        "docking_global_median": d.get("global_ddg_median"),
        "docking_global_sd": d.get("global_ddg_sd"),
        "docking_global_min": d.get("global_ddg_min"),
        "docking_n_converged_of_25": d.get("n_converged_total"),
        "cv_mean_of_5run_medians": d.get("cv_mean_of_medians"),
        "cv_sd_of_5run_medians": d.get("cv_sd_of_medians"),
        "best_pose_pdb": d.get("best_pdb"),
        # 선택성
        "selectivity_margin": s.get("selectivity_margin"),
        "worst_offtarget": s.get("worst_offtarget"),
        "selectivity_gate_pass": s.get("gate_pass"),
        # ADMET
        "hc50": a.get("hc50"),
        "hc50_vs_native": a.get("hc50_vs_native"),
        "pepadmet_toxic_flag": a.get("is_toxic_flag"),
        # 반감기
        "half_life_h": h.get("half_life_h"),
        "halflife_source": h.get("halflife_source"),
        # MM-GBSA (있으면)
        "mmgbsa_dg": mm,
        # 판정
        "wetlab_tier": tier,
        "tier_reasons": "; ".join(reasons),
        # 프로비넌스 세부
        "cv_started": rec.get("cv_started_utc"),
        "cv_finished": rec.get("cv_finished_utc"),
        "cv_duration_s": rec.get("cv_duration_s"),
        "cv_error": rec.get("cv_error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-master", default="runs/cv_analysis/cv_master.jsonl")
    ap.add_argument("--mmgbsa", default="runs/pyrosetta_flow/mmgbsa_consensus.json")
    ap.add_argument("--out-json", default="runs/cv_analysis/wetlab_candidates.json")
    ap.add_argument("--out-csv", default="runs/cv_analysis/wetlab_candidates.csv")
    ap.add_argument("--out-md", default="runs/cv_analysis/wetlab_proposal.md")
    args = ap.parse_args()

    cv = load_cv_master(REPO / args.cv_master)
    mm = load_mmgbsa(REPO / args.mmgbsa)
    rows = [build_row(s, r, mm.get(s)) for s, r in cv.items()]
    # 정렬: tier(S<A<B<C) → global_ddg_median
    tier_ord = {"S": 0, "A": 1, "B": 2, "C": 3}
    rows.sort(key=lambda r: (tier_ord.get(r["wetlab_tier"], 9),
                              r["docking_global_median"] if r["docking_global_median"] is not None else 999))
    # JSON
    (REPO / args.out_json).write_text(json.dumps({"n": len(rows), "rows": rows},
                                                  ensure_ascii=False, indent=2))
    # CSV
    if rows:
        cols = list(rows[0].keys())
        with (REPO / args.out_csv).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # Markdown 제안서
    tiers = Counter(r["wetlab_tier"] for r in rows)
    md_lines = [
        "# Wet-lab 후보 제안 (CV 파이프라인 집계)",
        f"- 생성 시각: {rows[0].get('cv_finished') if rows else '(데이터 없음)'}",
        f"- 총 CV 완료 후보: **{len(rows)}** (tier 분포: {dict(tiers)})",
        "- 데이터 근거: 각 후보 25 pose(5 outer runs × 5 nstruct=1) FlexPepDock + selectivity(SSTR1/3/4/5) + pepADMET + halflife + MM-GBSA(async)",
        "",
        "## Tier 정의",
        "- **S**: 다목적 전지표 통과 (med≤−35, Δmargin>5, hc50 동급, t½>0.05h, MM-GBSA<−60)",
        "- **A**: binding+selectivity+tox 통과 (med≤−30, Δmargin>0)",
        "- **B**: 기본 binding+selectivity (med≤−25, Δmargin>0)",
        "- **C**: 미통과 (이유 명시)",
        "",
        "## 상위 후보 (tier 우선, 결합력 순)",
        "",
        "| Rank | Sequence | Tier | Source | med(25p) | sd | Δmargin | hc50Δ | t½h | MMGBSA |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows[:30], 1):
        md_lines.append(f"| {i} | `{r['sequence']}` | **{r['wetlab_tier']}** | "
                        f"{r['source_experiment']} | {r['docking_global_median']} | "
                        f"{r['docking_global_sd']} | {r['selectivity_margin']} | "
                        f"{r['hc50_vs_native']} | {r['half_life_h']} | {r['mmgbsa_dg']} |")
    md_lines += ["", "## Tier C 후보 미통과 이유 (참고, 실험 배제 근거)"]
    for r in rows:
        if r["wetlab_tier"] == "C":
            md_lines.append(f"- `{r['sequence']}` ({r['source_experiment']}): {r['tier_reasons']}")
    (REPO / args.out_md).write_text("\n".join(md_lines))

    print(f"  집계 완료: {len(rows)} 후보 → "
          f"{args.out_json} / {args.out_csv} / {args.out_md}", file=sys.stderr)
    print(f"  Tier 분포: {dict(tiers)}", file=sys.stderr)


if __name__ == "__main__":
    main()
