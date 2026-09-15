#!/usr/bin/env python3
"""72h 실험 판정 — 랜덤(ddG-only) vs 시스템(다목적 계획) 공정 비교.

방법론 근거: _workspace/EXP72_RANDOM_VS_SYSTEM.md §공정 비교 방법론.

핵심 규칙 (두 arm은 게임 규칙이 다르므로):
1. **robust 대 robust만** — 랜덤은 전부 nstruct5 robust. 시스템 experiment_log는 single-pose
   스크리닝이라 직접 비교 부당. 시스템 robust는 (a)리더보드 승격분 (b)--redock로 window-new 상위
   재도킹분. single-pose ddg는 "스크리닝 분포" 참고용으로만 출력.
2. **n=1 가짜 clean 배제** — n_converged<2 (또는 sd==0.0 추정)는 non-robust로 제외 (양 arm 공통).
3. **floppy 배제** — sd>15 는 포즈 불안정. clean=robust(n>=2)+sd<=15.
4. **목적함수 비대칭 명시** — 랜덤=ddG only. 게이트(--gate) 없이는 랜덤 후보의 선택성/독성 미검증.
   "발굴 승리"는 랜덤 top이 --gate(선택성 Δmargin + hc50 + 반감기)를 통과해야만 성립.
5. 지표 = 후보당 robust-clean native능가율(throughput 중립) + best robust-clean median.

사용:
  # 기본 비교(분석만, 외부 의존 없음)
  python scripts/compare_exp72.py
  # 시스템 window-new 상위 N 재도킹(robust) 후 비교 (pyrosetta 필요)
  python scripts/compare_exp72.py --redock 10
  # 랜덤 상위 N 후보에 다목적 게이트(선택성+독성+반감기) 적용 (pyrosetta+surrogate)
  python scripts/compare_exp72.py --gate 5
NO MOCK — 실제 로그/리더보드/도킹만. 불확실은 표기.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import Counter
from statistics import median as _median

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "AG_src" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))   # exp72_structure_check

NATIVE_ROBUST = -20.28          # canonical robust median native baseline
NATIVE_HC50 = -55.6816          # data/.../native_toxicity_baseline.json


def _num(x):
    return x if isinstance(x, (int, float)) else None


def classify(med, sd, n_conv, floppy_sd):
    """robust-clean / floppy / nonrobust / no_stats 분류."""
    med = _num(med); sd = _num(sd)
    if med is None:
        return "no_stats"
    if n_conv is not None and n_conv < 2:
        return "nonrobust"          # n=1 가짜 clean
    if n_conv is None and (sd is None or sd == 0.0):
        return "nonrobust"          # sd 부재/0 → n=1 추정
    if sd is not None and sd > floppy_sd:
        return "floppy"
    return "clean"


def load_random(arm_dir: Path):
    """arm==random robust 레코드 → seq당 best(최저 median) 유지."""
    best = {}
    log = arm_dir / "experiment_log.jsonl"
    if not log.exists():
        return []
    for line in log.open(errors="ignore"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("arm") != "random" or r.get("status") != "ok":
            continue
        med = _num(r.get("ddg_median"))
        if med is None:
            continue
        seq = r.get("sequence")
        rec = {"sequence": seq, "median": med, "sd": _num(r.get("ddg_sd")),
               "n_conv": r.get("n_converged") if r.get("n_converged") is not None
                         else r.get("ddg_n_converged"),
               "pdb": str(arm_dir / "work" / f"{seq}.pdb")}
        if seq not in best or med < best[seq]["median"]:
            best[seq] = rec
    return list(best.values())


def load_system_window(sys_dir: Path, start_lines: int, source_filter=None):
    """window-new(라인 start_lines 이후, arm!=random) single-pose 스크리닝 → seq당 best ddg.

    source_filter: None=전체(모든 mutation_source). str/set=해당 mutation_source만 유지.
      ★ code감사 R1: 시스템 window의 ~67%가 시스템 자체 random(bandit 폴백)이라
        "LLM계획 vs 랜덤"을 보려면 source_filter="llm_guided"로 순수 계획분만 비교해야 함.
    srcs(전체 출처 분포)는 필터와 무관하게 항상 집계(비교 오염 진단용).
    """
    if isinstance(source_filter, str):
        source_filter = {source_filter}
    best = {}
    srcs = Counter()
    log = sys_dir / "experiment_log.jsonl"
    with log.open(errors="ignore") as f:
        for i, line in enumerate(f):
            if i < start_lines:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("arm") == "random" or not r.get("sequence"):
                continue
            src = r.get("mutation_source")
            srcs[src] += 1                       # 전체 분포는 항상 집계
            if source_filter is not None and src not in source_filter:
                continue                          # 필터: 지정 출처만 비교 대상
            d = _num(r.get("ddg"))
            if d is None:
                continue
            seq = r["sequence"]
            if seq not in best or d < best[seq]["ddg"]:
                best[seq] = {"sequence": seq, "ddg": d,
                             "mutation_source": src,
                             "candidate_id": r.get("candidate_id"),
                             "iteration": r.get("iteration")}
    return list(best.values()), dict(srcs)


def load_leaderboard(sys_dir: Path):
    p = sys_dir / "global_selectivity_leaderboard.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    e = d if isinstance(d, list) else d.get("entries", d.get("leaderboard", []))
    out = []
    cache = sys_dir / "mmgbsa_pdb_cache"
    for r in e:
        seq = r.get("sequence")
        pdb = cache / f"{seq}.pdb"
        out.append({"sequence": seq, "median": _num(r.get("ddg_median")),
                    "sd": _num(r.get("ddg_sd")), "n_conv": r.get("ddg_n_converged"),
                    "delta_margin": _num(r.get("delta_margin")), "hc50": _num(r.get("hc50")),
                    "mmgbsa": _num(r.get("mmgbsa_dg")), "pose_uncertain": r.get("pose_uncertain"),
                    "pdb": str(pdb) if pdb.exists() else None})
    return out


def summarize(recs, floppy_sd, beat_margin, label):
    buckets = Counter()
    clean = []
    for r in recs:
        c = classify(r.get("median"), r.get("sd"), r.get("n_conv"), floppy_sd)
        buckets[c] += 1
        if c == "clean":
            clean.append(r)
    clean.sort(key=lambda r: r["median"])
    beat = [r for r in clean if r["median"] <= NATIVE_ROBUST - beat_margin]
    rate = (100 * len(beat) / len(clean)) if clean else 0.0
    return {"label": label, "total": len(recs), "buckets": dict(buckets),
            "clean_n": len(clean), "clean_best": clean[0] if clean else None,
            "beat_n": len(beat), "beat_rate_pct": round(rate, 2),
            "clean_top": clean[:10]}


# ---------------- 구조검증 (R3: 스코어함정/이황결합 왜곡 탐지) ----------------

def struct_check_arm(clean_top, n, arm_label):
    """arm의 robust-clean 상위 N개 기존 도킹 PDB를 구조검증(도킹 불필요=저비용).
    이황결합(SG-SG)·FWKT 접촉·Ramachandran 요약 기록. R3 아티팩트 노출용."""
    from exp72_structure_check import check_complex_structure
    out = []
    for t in clean_top[:n]:
        seq = t["sequence"]; pdb = t.get("pdb")
        row = {"sequence": seq, "median": t.get("median"), "sd": t.get("sd"), "arm": arm_label}
        if pdb and Path(pdb).exists():
            try:
                chk = check_complex_structure(pdb, seq)
                row.update({k: chk.get(k) for k in
                            ("sg_sg_distance", "disulfide_flag", "disulfide_intact",
                             "pharmacophore_contact", "peptide_chain_autodetected")})
            except Exception as exc:
                row["struct_err"] = f"{type(exc).__name__}: {exc}"
        else:
            row["struct_err"] = "pdb 없음"
        out.append(row)
        print(f"  [struct:{arm_label}] {seq} med={row.get('median')} "
              f"SG-SG={row.get('sg_sg_distance')} {row.get('disulfide_flag')}", file=sys.stderr)
    return out


# ---------------- 무거운 옵션: 재도킹 / 게이트 ----------------

def redock_system_top(sys_window, n, sys_dir: Path):
    """시스템 window-new single-pose 상위 N을 robust(nstruct5) 재도킹 → runs/exp72_system/redock.jsonl."""
    import flexpep_dock as fpd
    fpd.init_pyrosetta()
    template = str(REPO / "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb")
    out = sys_dir.parent / "exp72_system"
    out.mkdir(parents=True, exist_ok=True)
    rd_log = out / "redock.jsonl"
    done = set()
    if rd_log.exists():
        for line in rd_log.open(errors="ignore"):
            try: done.add(json.loads(line)["sequence"])
            except Exception: pass
    work = out / "redock_work"; work.mkdir(exist_ok=True)
    targets = sorted(sys_window, key=lambda r: r["ddg"])[:n]
    results = []
    for t in targets:
        seq = t["sequence"]
        if seq in done:
            continue
        try:
            pose, _ = fpd.prepare_complex_by_mutation(template, seq, peptide_chain=1)
            _, info = fpd.run_flexpep_refine_pose(pose, str(work / f"{seq}.pdb"), nstruct=5)
            rec = {"sequence": seq, "median": _num(info.get("ddg_median")),
                   "sd": _num(info.get("ddg_sd")),
                   "n_conv": info.get("n_converged", info.get("ddg_n_converged")),
                   "pdb": str(work / f"{seq}.pdb"), "screening_ddg": t["ddg"]}
            # 0-5: 재도킹 포즈 구조검증(SS bond 기록)
            try:
                from exp72_structure_check import check_complex_structure
                chk = check_complex_structure(str(work / f"{seq}.pdb"), seq)
                rec.update({k: chk.get(k) for k in
                            ("sg_sg_distance", "disulfide_flag", "disulfide_intact")})
            except Exception as _ce:
                rec["struct_err"] = str(_ce)
        except Exception as exc:
            rec = {"sequence": seq, "error": f"{type(exc).__name__}: {exc}"}
        with rd_log.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        results.append(rec)
        print(f"  [redock] {seq} median={rec.get('median')} sd={rec.get('sd')}", file=sys.stderr)
    # 기존 + 신규 합쳐 반환
    allrecs = []
    for line in rd_log.open(errors="ignore"):
        try: allrecs.append(json.loads(line))
        except Exception: pass
    return allrecs


def gate_random_top(clean_top, n, margin_ref=None):
    """랜덤 robust-clean 상위 N에 게이트. 랜덤=ddG only 이므로 통과분만 '진짜 후보'.

    ★ pharma/code 감사 반영 (2026-07-13):
    - 승패 근거 = **선택성 Δmargin 하나만** (물리 endpoint). fail-CLOSED: 선택성 미산출=탈락.
    - hc50 **판정 제외**: pepADMET hc50은 L-aa 역변별(AUC 0.146), EXP72=100% L-aa → 가치 0.
      기록만 하고 "미검증" 병기(H-06). 이전 fail-open(hc50 결측=통과) 버그 제거.
    - 반감기: HEURISTIC tie-breaker로 **기록만**, 판정식 제외.
    - Δmargin 임계: 절대(10) 이중잣대 금지 → margin_ref(시스템 리더보드 median≈5.02) **상대기준**.
    """
    from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences, screen_selectivity
    from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife
    targets = clean_top[:n]
    seqs = [t["sequence"] for t in targets]
    tox = predict_toxicity_for_sequences(seqs)
    out = []
    for t in targets:
        seq = t["sequence"]
        row = {"sequence": seq, "ddg_median": t["median"], "ddg_sd": t["sd"]}
        # 독성 (hc50 vs native, >0 = native보다 안전)
        tr = tox.get(seq, {})
        hc50 = _num(tr.get("hc50")) if tr.get("available") else None
        row["hc50"] = hc50
        row["hc50_vs_native"] = round(hc50 - NATIVE_HC50, 3) if hc50 is not None else None
        row["tox_available"] = bool(tr.get("available"))
        row["hc50_note"] = "기록만 — L-aa 역변별로 판정 미사용(미검증)"
        # 반감기 (tie-breaker 기록만, 판정 제외)
        try:
            hl = ensemble_halflife(seq)
            row["half_life_h"] = hl.get("half_life_h")
            row["halflife_source"] = hl.get("halflife_source")
        except Exception as exc:
            row["half_life_h"] = None; row["halflife_err"] = str(exc)
        # 선택성 (도킹 PDB 필요 — 랜덤 arm work/{seq}.pdb)
        pdb = t.get("pdb")
        if pdb and Path(pdb).exists():
            try:
                sel = screen_selectivity(sstr2_complex_pdb=pdb, on_target_ddg=t["median"],
                                         conda_env="bio-tools", timeout=900)
                row["delta_margin"] = _num(sel.get("selectivity_margin"))
                row["offtarget_ddg"] = sel.get("offtarget_ddg")
            except Exception as exc:
                row["delta_margin"] = None; row["selectivity_err"] = str(exc)
        else:
            row["delta_margin"] = None; row["selectivity_err"] = "pdb 없음"
        # 판정: 선택성 Δmargin 하나만. fail-CLOSED (미산출=탈락). 상대기준(margin_ref).
        dm = row.get("delta_margin")
        bar = margin_ref if margin_ref is not None else 0.0
        row["margin_bar"] = bar
        row["pass_selectivity"] = bool(dm is not None and dm > bar)
        out.append(row)
        print(f"  [gate] {seq} Δmargin={dm} (bar>{bar}) hc50Δ={row.get('hc50_vs_native')}(미판정) "
              f"t½={row.get('half_life_h')}(tie) → pass={row['pass_selectivity']}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-random", default="runs/exp72_random")
    ap.add_argument("--arm-system", default="runs/pyrosetta_flow")
    ap.add_argument("--window-start-lines", type=int, default=None,
                    help="시스템 window 시작 라인(기본: WINDOW_MARKER.json에서 로드)")
    ap.add_argument("--floppy-sd", type=float, default=15.0)
    ap.add_argument("--beat-margin", type=float, default=15.0, help="native 대비 유의 능가 REU")
    ap.add_argument("--redock", type=int, default=0, help="시스템 window-new 상위 N robust 재도킹")
    ap.add_argument("--gate", type=int, default=0, help="랜덤 robust-clean 상위 N 선택성 게이트")
    ap.add_argument("--struct-check", type=int, default=0,
                    help="양 arm robust-clean 상위 N개 기존 PDB 구조검증(SS bond/FWKT, 도킹불필요). R3.")
    ap.add_argument("--system-source", default="llm_guided",
                    help="시스템 arm 비교 대상 mutation_source (기본 llm_guided=순수 계획분만; "
                         "'all'=전체[랜덤 폴백 포함, 오염 비교]). code감사 R1.")
    ap.add_argument("--margin-ref", type=float, default=5.02,
                    help="게이트 Δmargin 상대기준(기본=시스템 리더보드 median 5.02). 절대임계10 이중잣대 회피.")
    ap.add_argument("--out", default="runs/exp72_system/compare_report.json")
    args = ap.parse_args()

    rand_dir = REPO / args.arm_random
    sys_dir = REPO / args.arm_system
    src_filter = None if args.system_source == "all" else args.system_source

    # window 시작 라인 (code감사 R6: 폴백 시 경고)
    start_lines = args.window_start_lines
    if start_lines is None:
        mk = REPO / "runs/exp72_system/WINDOW_MARKER.json"
        if mk.exists():
            start_lines = json.loads(mk.read_text())["experiment_log_start_lines"]
        else:
            start_lines = 0
            print("⚠️ WINDOW_MARKER.json 부재 → start_lines=0 (전체 이력을 window로 간주! 확인 필요)",
                  file=sys.stderr)

    # provenance (code감사 P1: 재현성 기록)
    import subprocess, time as _time
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=str(REPO), text=True).strip()
    except Exception:
        commit = "unknown"

    # 데이터 로드
    rand = load_random(rand_dir)
    sys_window, sys_srcs = load_system_window(sys_dir, start_lines, source_filter=src_filter)
    lb = load_leaderboard(sys_dir)

    rand_sum = summarize(rand, args.floppy_sd, args.beat_margin, "RANDOM (ddG-only, nstruct5)")
    lb_sum = summarize(lb, args.floppy_sd, args.beat_margin, "SYSTEM 리더보드 robust (누적)")

    # 시스템 window single-pose 스크리닝 분포(참고, robust 아님)
    sw_ddg = sorted(r["ddg"] for r in sys_window)
    total_window = sum(sys_srcs.values())
    n_llm = sys_srcs.get("llm_guided", 0)
    n_sysrandom = sys_srcs.get("random", 0)
    sw_ref = {"source_filter": args.system_source, "n_after_filter": len(sys_window),
              "best_screening_ddg": sw_ddg[0] if sw_ddg else None,
              "median_screening_ddg": round(_median(sw_ddg), 2) if sw_ddg else None,
              "sources_full": sys_srcs, "total_window_records": total_window,
              "llm_guided_frac": round(n_llm / total_window, 3) if total_window else None,
              "system_internal_random_frac": round(n_sysrandom / total_window, 3) if total_window else None}

    report = {"provenance": {"git_commit": commit, "generated_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                             "system_source_filter": args.system_source, "margin_ref": args.margin_ref,
                             "note": "PyRosetta seed 미고정=재도킹 비결정적(P1 잔존). 요약통계는 결정적."},
              "window_start_lines": start_lines, "floppy_sd": args.floppy_sd,
              "beat_margin": args.beat_margin, "native_robust": NATIVE_ROBUST,
              "random": rand_sum, "system_leaderboard": lb_sum,
              "system_window_screening": sw_ref}

    # ---- 무거운 옵션 ----
    if args.redock > 0:
        print(f"\n[redock] 시스템 window-new 상위 {args.redock} robust 재도킹...", file=sys.stderr)
        rd = redock_system_top(sys_window, args.redock, sys_dir)
        rd_sum = summarize(rd, args.floppy_sd, args.beat_margin, "SYSTEM window-new robust 재도킹")
        report["system_window_redock"] = rd_sum
    if args.struct_check > 0:
        print(f"\n[struct-check] 양 arm 상위 {args.struct_check} 구조검증(SS bond)...", file=sys.stderr)
        report["random_struct"] = struct_check_arm(rand_sum["clean_top"], args.struct_check, "random")
        report["system_struct"] = struct_check_arm(lb_sum["clean_top"], args.struct_check, "system_lb")
    if args.gate > 0:
        print(f"\n[gate] 랜덤 robust-clean 상위 {args.gate} 선택성 게이트 (bar>{args.margin_ref})...", file=sys.stderr)
        gated = gate_random_top(rand_sum["clean_top"], args.gate, margin_ref=args.margin_ref)
        report["random_gated"] = gated
        report["random_gated_pass_n"] = sum(1 for g in gated if g.get("pass_selectivity"))

    # 저장
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    # ---- 콘솔 판정 리포트 ----
    def _fmt(s):
        cb = s["clean_best"]
        best = f"{cb['sequence']} {cb['median']:.2f} (sd {cb['sd']})" if cb else "-"
        return (f"  {s['label']}\n"
                f"    총 {s['total']} | 분류 {s['buckets']}\n"
                f"    robust-clean {s['clean_n']} | best {best}\n"
                f"    native {args.beat_margin:.0f}REU 능가 {s['beat_n']}건 (능가율 {s['beat_rate_pct']}%)")
    print("\n" + "=" * 70)
    print("72h 실험 판정 — 랜덤 vs 시스템 (robust-clean 대 robust-clean)")
    print("=" * 70)
    print(f"  provenance: commit {commit} | system-source='{args.system_source}' | margin-ref {args.margin_ref}")
    print(_fmt(rand_sum))
    print(_fmt(lb_sum))
    if "system_window_redock" in report:
        print(_fmt(report["system_window_redock"]))
    print(f"  SYSTEM window (source='{sw_ref['source_filter']}'): 필터후 n {sw_ref['n_after_filter']} / "
          f"best {sw_ref['best_screening_ddg']} (single-pose, robust아님)")
    print(f"    ★R1 오염: 전체 window {sw_ref['total_window_records']}건 중 llm_guided "
          f"{sw_ref['llm_guided_frac']} / 시스템내부random {sw_ref['system_internal_random_frac']}")
    if "random_struct" in report:
        print(f"\n  [구조검증 R3] 이황결합(SG-SG, 정상 1.9~2.3Å) — 랜덤 top이 압축/파손이면 스코어함정:")
        for lab, key in (("RANDOM", "random_struct"), ("SYSTEM", "system_struct")):
            for s in report[key]:
                pc = s.get("pharmacophore_contact") or {}
                fwkt = f"FWKT접촉={pc.get('in_contact')}(min {pc.get('min_distance')}Å)" if pc else ""
                print(f"    [{lab}] {s['sequence']} med={s.get('median')} "
                      f"SG-SG={s.get('sg_sg_distance')}Å {s.get('disulfide_flag')} {fwkt}")
    if "random_gated" in report:
        print(f"\n  [랜덤 선택성 게이트] 상위 {args.gate} 중 통과 "
              f"{report['random_gated_pass_n']}건 (Δmargin > {args.margin_ref} 상대기준; hc50/반감기 미판정):")
        for g in report["random_gated"]:
            print(f"    {g['sequence']}: Δmargin={g.get('delta_margin')} "
                  f"hc50Δ={g.get('hc50_vs_native')}(미판정) t½={g.get('half_life_h')}h(tie) "
                  f"→ {'PASS' if g.get('pass_selectivity') else 'FAIL'}")
    print("\n  ⚠️ 해석 규칙:")
    print("    - 승패 근거 = ddG + 선택성 Δmargin 2개 물리 endpoint만. hc50=L-aa역변별 미판정, 반감기=tie만.")
    print("    - 시스템 비교는 llm_guided-only (전체는 67% 시스템내부 랜덤이라 오염). single-pose(n<2)·floppy(sd>15) 배제.")
    print(f"    - 리포트 저장: {outp}")
    print("=" * 70)


if __name__ == "__main__":
    main()
