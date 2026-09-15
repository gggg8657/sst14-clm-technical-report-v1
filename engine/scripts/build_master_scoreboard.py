#!/usr/bin/env python3
"""EXP72 마스터 스코어보드 빌더 (V0 provenance + V1 서열기반 지표 전수).

범위: 역대 전체 unique 서열(사용자 승인 2026-07-13). 저비용 지표는 전수, 고비용 도킹은 별도 shortlist.
provenance 정확 구분: PURE_RANDOM / LLM_GUIDED / SYS_RANDOM(bandit) / INTENSIFY / SILO_A / LEGACY.

V0: 전 소스 experiment_log + 리더보드 통합 → seq당 provenance(전 출처 집합 + 최초 ts 기준 primary) + 기존 로그 결합지표.
V1: 각 seq에 서열기반 지표 부착 — 개발성(compute_pharmacology)+DOTA(compute_chelator_site)+독성(batch)+반감기.
출력: runs/exp72_analysis/master_candidates.jsonl (재개가능: 이미 V1 완료 seq 스킵).

NO MOCK: 실제 로그/서로게이트만. 실패는 필드 None+이유. surrogate 파일 미수정(호출만).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

WINDOW_START_LINE = 42280   # pyrosetta_flow experiment_log의 72h 실험 윈도우 시작
NATIVE = "AGCKNFFWKTFTSC"

# provenance 우선순위(중복서열의 primary 결정 시, 최초 ts 우선; 동시엔 이 순서)
PROV_PRIORITY = ["PURE_RANDOM", "LLM_GUIDED", "INTENSIFY", "SILO_A", "SYS_RANDOM", "LEGACY"]


def _num(x):
    return x if isinstance(x, (int, float)) else None


def _src_to_label(mutation_source, in_window):
    ms = (mutation_source or "").lower()
    if "llm" in ms:
        return "LLM_GUIDED"
    if "intensif" in ms:
        return "INTENSIFY"
    if ms in ("random", "bandit", "random_fallback", "bo_guided", "dedup_fallback"):
        return "SYS_RANDOM"
    return None  # 알수없음→호출측에서 window로 LEGACY 판정


def gather(sources_verbose=False):
    """V0: 전 소스 통합 → {seq: rec}."""
    cand = {}

    def touch(seq, label, ts, ddg=None, robust=None, extra=None):
        if not seq:
            return
        r = cand.get(seq)
        if r is None:
            r = {"sequence": seq, "provenances": set(), "first_ts": ts, "first_label": label,
                 "best_ddg_singlepose": None, "robust": None, "logged": {}}
            cand[seq] = r
        r["provenances"].add(label)
        if ts and (r["first_ts"] is None or ts < r["first_ts"]):
            r["first_ts"] = ts; r["first_label"] = label
        if ddg is not None and (r["best_ddg_singlepose"] is None or ddg < r["best_ddg_singlepose"]):
            r["best_ddg_singlepose"] = ddg
        if robust and robust.get("ddg_median") is not None:
            cur = r["robust"]
            if cur is None or (robust["ddg_median"] < cur.get("ddg_median", 1e9)):
                r["robust"] = robust
        if extra:
            r["logged"].update({k: v for k, v in extra.items() if v is not None})

    # 1) PURE_RANDOM
    p = REPO / "runs/exp72_random/experiment_log.jsonl"
    if p.exists():
        for line in p.open(errors="ignore"):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("arm") != "random":
                continue
            robust = {"ddg_median": _num(x.get("ddg_median")), "ddg_sd": _num(x.get("ddg_sd")),
                      "n_conv": x.get("n_converged"), "src": "exp72_random"} if _num(x.get("ddg_median")) is not None else None
            touch(x.get("sequence"), "PURE_RANDOM", x.get("ts"), _num(x.get("ddg")), robust)

    # 2) 시스템(pyrosetta_flow): mutation_source별 라벨 + window/legacy
    p = REPO / "runs/pyrosetta_flow/experiment_log.jsonl"
    if p.exists():
        for i, line in enumerate(p.open(errors="ignore")):
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get("arm") == "random":   # exp72 랜덤이 이 파일에 섞였을 리 없지만 방어
                continue
            in_window = i >= WINDOW_START_LINE
            label = _src_to_label(x.get("mutation_source"), in_window)
            if label is None:
                label = "LEGACY" if not in_window else "SYS_RANDOM"
            touch(x.get("sequence"), label, x.get("ts"), _num(x.get("ddg")), None)

    # 3) SILO_A
    p = REPO / "runs/silo_a_flow/experiment_log.jsonl"
    if p.exists():
        for line in p.open(errors="ignore"):
            try:
                x = json.loads(line)
            except Exception:
                continue
            touch(x.get("sequence"), "SILO_A", x.get("ts"), _num(x.get("ddg")), None)

    # 4) 리더보드: 권위있는 robust+선택성+독성+mmgbsa 병합
    p = REPO / "runs/pyrosetta_flow/global_selectivity_leaderboard.json"
    if p.exists():
        d = json.loads(p.read_text())
        e = d if isinstance(d, list) else d.get("entries", d.get("leaderboard", []))
        for x in e:
            seq = x.get("sequence")
            robust = {"ddg_median": _num(x.get("ddg_median")), "ddg_sd": _num(x.get("ddg_sd")),
                      "n_conv": x.get("ddg_n_converged"), "src": "leaderboard"} if _num(x.get("ddg_median")) is not None else None
            touch(seq, cand.get(seq, {}).get("first_label", "LEGACY") if seq in cand else "LEGACY",
                  None, None, robust,
                  extra={"lb_delta_margin": _num(x.get("delta_margin")), "lb_hc50": _num(x.get("hc50")),
                         "lb_mmgbsa": _num(x.get("mmgbsa_dg")), "lb_consensus": x.get("consensus_flag"),
                         "lb_pose_uncertain": x.get("pose_uncertain")})
    if sources_verbose:
        print(f"  [V0] 고유서열 {len(cand)}", file=sys.stderr)
    return cand


def primary_provenance(provs):
    for lab in PROV_PRIORITY:
        if lab in provs:
            return lab
    return "UNKNOWN"


def classify_binding(robust, floppy_sd=15.0):
    if not robust or robust.get("ddg_median") is None:
        return "no_robust"
    med, sd, n = robust.get("ddg_median"), robust.get("ddg_sd"), robust.get("n_conv")
    if n is not None and n < 2:
        return "nonrobust"
    if n is None and (sd is None or sd == 0.0):
        return "nonrobust"
    if sd is not None and sd > floppy_sd:
        return "floppy"
    return "clean"


def enrich_one(seq, tr, _pharm, _chel, _hl):
    """서열 1개 지표 계산 (tr=이 seq의 tox 결과)."""
    aromatic = set("FWY")
    row = {}
    try:
        ph = _pharm(seq)
        for k in ("gravy", "boman", "instability_index", "aliphatic_index",
                  "isoelectric_point", "net_charge"):
            if k in ph:
                row[k] = ph[k]
        row["pharmacology_full"] = {k: ph[k] for k in ph if isinstance(ph[k], (int, float, str, bool))}
    except Exception as e:
        row["pharm_err"] = str(e)
    row["aromatic_frac"] = round(sum(1 for a in seq if a in aromatic) / len(seq), 3)
    try:
        ch = _chel(seq)
        if isinstance(ch, dict):
            row["dota_site"] = ch.get("site")
            row["dota_position"] = ch.get("position")
            row["dota_competition_risk"] = ch.get("competition_risk")
        else:
            row["dota_site"] = None
    except Exception as e:
        row["dota_err"] = str(e)
    row["hc50"] = _num(tr.get("hc50")) if tr.get("available") else None
    row["tox_available"] = bool(tr.get("available"))
    try:
        hl = _hl(seq)
        row["half_life_h"] = hl.get("half_life_h")
        row["halflife_source"] = hl.get("halflife_source")
    except Exception as e:
        row["halflife_err"] = str(e)
    row["cys_intact"] = (len(seq) >= 14 and seq[2] == "C" and seq[13] == "C")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/exp72_analysis/master_candidates.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="스모크용: 앞 N개만 V1 (0=전수)")
    ap.add_argument("--v0-only", action="store_true", help="V0(provenance)만, V1 스킵")
    args = ap.parse_args()

    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    cand = gather(sources_verbose=True)
    # provenance 요약
    prov_counter = Counter(primary_provenance(r["provenances"]) for r in cand.values())
    bind_counter = Counter(classify_binding(r["robust"]) for r in cand.values())
    print(f"  primary provenance 분포: {dict(prov_counter)}", file=sys.stderr)
    print(f"  robust binding 분포: {dict(bind_counter)}", file=sys.stderr)

    seqs = list(cand.keys())
    if args.limit:
        seqs = seqs[:args.limit]

    # 재개: 이미 출력된 seq 스킵
    done = set()
    if outp.exists():
        for line in outp.open(errors="ignore"):
            try:
                done.add(json.loads(line)["sequence"])
            except Exception:
                pass
    todo = [s for s in seqs if s not in done]
    print(f"  전체 {len(seqs)} / 완료 {len(done)} / V1대상 {len(todo)}", file=sys.stderr)

    # V1 함수 준비 (v0-only면 스킵)
    if not args.v0_only:
        from backend.pharmacology import compute_pharmacology
        from backend.pharmacophore import compute_chelator_site
        from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
        from pyrosetta_flow.halflife_ensemble_v2 import ensemble_halflife

    def _row(seq):
        r = cand[seq]
        rec = {"sequence": seq, "primary_provenance": primary_provenance(r["provenances"]),
               "provenances": sorted(r["provenances"]),
               "first_ts": r["first_ts"], "first_label": r["first_label"],
               "seen_pure_random": "PURE_RANDOM" in r["provenances"],
               "seen_llm_guided": "LLM_GUIDED" in r["provenances"],
               "seen_sys_random": "SYS_RANDOM" in r["provenances"],
               "best_ddg_singlepose": r["best_ddg_singlepose"],
               "robust": r["robust"], "binding_class": classify_binding(r["robust"]),
               "logged": r["logged"]}
        return rec

    # 청크(1000)별 tox batch + 서열별 즉시 append (크래시 안전·재개 가능)
    CHUNK = 1000
    written = 0
    for c0 in range(0, len(todo), CHUNK):
        chunk = todo[c0:c0 + CHUNK]
        tox = {} if args.v0_only else predict_toxicity_for_sequences(chunk)
        with outp.open("a") as f:            # 청크마다 열고 닫아 flush 보장
            for seq in chunk:
                rec = _row(seq)
                if not args.v0_only:
                    rec["metrics"] = enrich_one(seq, tox.get(seq, {}),
                                                compute_pharmacology, compute_chelator_site, ensemble_halflife)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
        print(f"  [V1] {written}/{len(todo)} 기록됨", file=sys.stderr, flush=True)
    print(f"[master] 저장 {outp} (+{written})", file=sys.stderr)


if __name__ == "__main__":
    main()
