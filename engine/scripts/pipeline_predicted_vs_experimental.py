#!/usr/bin/env python3
"""pipeline_predicted_vs_experimental.py
==========================================
예측 구조(Boltz) vs 실측 구조(7T10 cryo-EM) 재도킹 대조 — 임의 서열 리스트 대응
일반 파이프라인 (Wave 1 A1 후속).

배경 (`_workspace/A1_7T10_REDOCK_RESULTS_2026-08-03.md`):
    예측 native ddG = −20.28 REU (Boltz-1 예측 구조, controlled nstruct 10)
    실측 native ddG = −36.37 REU (7T10 실측 cryo-EM 구조 재도킹, nstruct 10)
    → 16 REU 절대값 차이. 확증 top5 중 1건(AGCKWD)은 실측에서 native 미만으로
      뒤집혀 예측 오탐 실증.

본 스크립트는 이 대조를 임의 서열 리스트에 재현 가능하게 일반화한다:
    서열 리스트 → predicted/experimental 두 receptor로 각각 FlexPepDock refine
    → ddG median/SD/n 계산 → 순위·SEM 기반 판정(agreed/inverted/inconclusive_SEM).

★ 중요 버그 노트 (2026-08-03, engineer-backend 발견 — 신규 스크립트 설계에 직접 반영):
    구 스크립트 `scripts/exp72_7t10_redock_confirmed5.py` 는 7T10 재도킹 시
    `pose.pdb_info().chain(k) == 'A'` 로 펩타이드 위치(1~14)를 찾아
    MutateResidue 를 적용했다. 그러나 실측 curated PDB
    (`data/somatostatin_receptor/curated/SSTR2_SST14_complex_7t10.pdb`)에서는
    **chain A = 수용체(287잔기, resnum 40~327), chain B = 펩타이드(14잔기,
    resnum 1~14)** 이다 (raw PDB 컬럼 기준 실측 확인, 2026-08-03). 따라서
    resnum 1~14 에 해당하는 chain **A** 잔기는 애초에 존재하지 않아 —
    **그 스크립트는 non-native 5개 서열에 대해 단 한 번도 변이를 적용하지
    못했을 가능성이 높다.** (SD가 9~18 REU로 컸던 것은 이 가설과 부합 —
    사실상 native 펩타이드를 5번 다른 무작위 시드로 재도킹한 노이즈 표본).
    본 스크립트는 이 문제를 원천 차단하기 위해 하드코딩된 체인 문자 비교 대신
    `AG_src.scripts.flexpep_dock.prepare_complex_by_mutation()` 을 재사용한다
    — 이 함수는 펩타이드 체인을 **서열 길이 비교로 자동 판별**하므로 receptor
    파일마다 체인 배정이 달라도(예측=A펩타이드/B수용체, 실측=A수용체/B펩타이드)
    안전하다. 이 발견은 A1의 6서열 결과 신뢰도에도 영향을 주므로 재검증 필요
    (본 파이프라인의 `--seed-experimental-cache` 로 재현 시 `legacy_cache_reliable`
    플래그로 명시).

사용법:
    conda run -n bio-tools python scripts/pipeline_predicted_vs_experimental.py \
        --sequences AGCKNFFWKTFTSC --nstruct 3

    conda run -n bio-tools python scripts/pipeline_predicted_vs_experimental.py \
        --sequences seqs.txt --nstruct 10 --shard-index 0 --shard-count 2

    # 리포트만 재생성 (도킹 생략)
    conda run -n bio-tools python scripts/pipeline_predicted_vs_experimental.py \
        --report-only --out runs/pred_vs_exp/results.jsonl

산출: `--out` (기본 runs/pred_vs_exp/results.jsonl, append 전용 raw 레코드)
      + `<out>.report.json` (서열별 predicted/experimental/delta/rank/verdict 요약)

정직 고지:
    - nstruct 10은 pool 재도킹 하한(Wave 1 A3 정책). wet-lab 인계는 nstruct 20+ 필요.
    - 절대값보다 순위(rank_shift)와 SEM 판정을 우선한다. 예측-실측 절대값 차이(16 REU
      규모)는 구조 베이스라인 차이이지 서열간 우열 판정이 아니다.
    - Silo A(de novo)는 별개 흐름 — 본 파이프라인은 Silo B(pyrosetta_flow) mutation
      후보에만 적용한다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

NATIVE_SEQ = "AGCKNFFWKTFTSC"

# 2026-08-03: 작업 스펙 원문은 `data/somatostatin_receptor/curated/SSTR2_SST14_complex_boltz_1.pdb`
# 를 기본 predicted 경로로 지정했으나, 저장소 실제 위치는 curated/ 하위가 아니라
# data/somatostatin_receptor/ 바로 아래이다 (확인: ls 결과). 실제 경로로 정정.
DEFAULT_PREDICTED = "data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb"
DEFAULT_EXPERIMENTAL = "data/somatostatin_receptor/curated/SSTR2_SST14_complex_7t10.pdb"
DEFAULT_OUT = "runs/pred_vs_exp/results.jsonl"
DEFAULT_NSTRUCT = 10  # Wave 1 A3 정책상 pool 재도킹 하한 (wet-lab 인계는 20+ 필요)

SIDES = ("predicted", "experimental")


# ---------------------------------------------------------------------------
# 입력 로딩
# ---------------------------------------------------------------------------

def load_sequences(arg: str) -> List[str]:
    """--sequences 값을 파일 경로 또는 콤마 구분 문자열로 해석. 중복 제거(순서 보존)."""
    p = Path(arg)
    if p.exists():
        raw = [ln.strip() for ln in p.read_text().splitlines()]
        seqs = [s for s in raw if s and not s.startswith("#")]
    else:
        seqs = [s.strip() for s in arg.split(",") if s.strip()]
    seen = set()
    out: List[str] = []
    for s in seqs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def load_completed_by_side(out_path: Path) -> Dict[Tuple[str, str], Dict]:
    """기존 --out jsonl에서 (sequence, side) 별 최신 레코드를 로드 (재개 지원).

    같은 (sequence, side) 조합이 여러 줄 있으면 마지막 줄이 우선(최신 상태 반영).
    """
    completed: Dict[Tuple[str, str], Dict] = {}
    if not out_path.exists():
        return completed
    for line in out_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        seq = rec.get("sequence")
        side = rec.get("side")
        if not seq or side not in SIDES:
            continue
        completed[(seq, side)] = rec
    return completed


def seed_legacy_cache(legacy_path: Path) -> Dict[str, Dict]:
    """구 A1 jsonl(`scripts/exp72_7t10_redock_confirmed5.py` 산출)을 experimental
    캐시로 흡수.

    ⚠ 모듈 docstring의 버그 노트 참조: 그 스크립트는 chain 'A' 하드코딩 비교로
    인해 non-native 서열에 변이를 적용하지 못했을 가능성이 높다. 따라서
    native 이외 서열의 레코드는 `legacy_cache_reliable=False` 로 표기하여
    소비자가 판단할 수 있게 한다 (자동으로 폐기하지 않음 — 재개 지원 기능
    자체는 스펙대로 제공하되, 신뢰도 캐비어트를 명시).
    """
    seeds: Dict[str, Dict] = {}
    if not legacy_path.exists():
        return seeds
    for line in legacy_path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("status") != "ok":
            continue
        seq = rec.get("sequence")
        if not seq:
            continue
        seeds[seq] = {
            "ddg_median": rec.get("ddg_median"),
            "ddg_sd": rec.get("ddg_sd"),
            "n_converged": rec.get("n_converged"),
            "n_total": rec.get("nstruct"),
            "sg_sg_distance": rec.get("sg_sg_distance"),
            "disulfide_intact": rec.get("disulfide_intact"),
            "source": f"legacy_cache:{legacy_path}",
            "legacy_cache_reliable": (seq == NATIVE_SEQ),
        }
    return seeds


# ---------------------------------------------------------------------------
# 도킹 (PyRosetta 필요)
# ---------------------------------------------------------------------------

def dock_sequence(receptor_pdb: str, sequence: str, nstruct: int, work_dir: Path, tag: str) -> Dict:
    """서열을 receptor_pdb에 적용 후 FlexPepDock refine.

    `prepare_complex_by_mutation()`의 서열-길이 기반 자동 펩타이드 체인 탐지를
    사용 — receptor 파일마다 체인 배정(A=펩타이드 vs A=수용체)이 달라도 안전
    (모듈 docstring 버그 노트 참조).
    """
    from AG_src.scripts.flexpep_dock import (
        prepare_complex_by_mutation,
        reorder_peptide_last,
        run_flexpep_refine_pose,
    )

    pose, resolved_chain = prepare_complex_by_mutation(receptor_pdb, sequence, peptide_chain=1)
    pose = reorder_peptide_last(pose, resolved_chain)
    out_pdb = str(work_dir / f"{tag}_{sequence}_refined.pdb")
    _, info = run_flexpep_refine_pose(pose, out_pdb, nstruct=nstruct)
    return info


def run_docking(
    sequences: List[str],
    receptor: str,
    side: str,
    nstruct: int,
    work_dir: Path,
    out_path: Path,
    shard_index: int,
    shard_count: int,
    completed: Dict[Tuple[str, str], Dict],
) -> None:
    for i, seq in enumerate(sequences):
        if shard_count > 1 and i % shard_count != shard_index:
            continue
        key = (seq, side)
        if completed.get(key, {}).get("status") == "ok":
            print(f"  [skip] {seq} ({side}) — 캐시 존재", file=sys.stderr)
            continue
        t0 = time.time()
        rec: Dict = {
            "sequence": seq,
            "side": side,
            "receptor": receptor,
            "nstruct": nstruct,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source": "computed",
        }
        try:
            info = dock_sequence(receptor, seq, nstruct, work_dir, tag=side)
            rec.update(
                {
                    "ddg_median": info.get("ddg_median"),
                    "ddg_mean": info.get("ddg_mean"),
                    "ddg_sd": info.get("ddg_sd"),
                    "ddg_min": info.get("ddg_min"),
                    "n_converged": info.get("n_converged"),
                    "n_total": info.get("n_total"),
                    "sg_sg_distance": info.get("sg_sg_distance"),
                    "disulfide_intact": info.get("disulfide_intact"),
                    "elapsed_s": round(time.time() - t0, 1),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            rec["elapsed_s"] = round(time.time() - t0, 1)
        with out_path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        completed[key] = rec
        print(
            f"  {seq} [{side}] median={rec.get('ddg_median')} SS={rec.get('sg_sg_distance')} "
            f"status={rec['status']} ({rec.get('elapsed_s')}s)",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 판정 로직 (순수 함수 — PyRosetta 불필요, 단위 테스트 대상)
# ---------------------------------------------------------------------------

def compute_sem(sd: Optional[float], n: Optional[int]) -> Optional[float]:
    """SEM = SD / sqrt(n). n이 없거나 0 이하면 None (계산 불가)."""
    if sd is None or n is None or n < 1:
        return None
    return sd / math.sqrt(n)


def compute_delta_and_significance(
    pred_median: Optional[float],
    pred_sd: Optional[float],
    pred_n: Optional[int],
    exp_median: Optional[float],
    exp_sd: Optional[float],
    exp_n: Optional[int],
) -> Dict:
    """delta_abs, SEM_delta(=sqrt(SEM_pred^2 + SEM_exp^2)), 유의성 판정.

    |delta_abs| >= 3*SEM_delta 이면 유의(significant=True), 아니면 노이즈 내
    (significant=False). SEM 계산 불가(SD/n 없음)면 significant=None.
    """
    if pred_median is None or exp_median is None:
        return {"delta_abs": None, "sem_delta": None, "significant": None}

    delta_abs = abs(pred_median - exp_median)
    sem_pred = compute_sem(pred_sd, pred_n)
    sem_exp = compute_sem(exp_sd, exp_n)

    if sem_pred is None or sem_exp is None:
        return {"delta_abs": round(delta_abs, 4), "sem_delta": None, "significant": None}

    sem_delta = math.sqrt(sem_pred ** 2 + sem_exp ** 2)
    significant = (delta_abs >= 3 * sem_delta) if sem_delta > 0 else (delta_abs > 0)
    return {
        "delta_abs": round(delta_abs, 4),
        "sem_delta": round(sem_delta, 4),
        "significant": significant,
    }


def assign_ranks(medians: Dict[str, Optional[float]]) -> Dict[str, Optional[int]]:
    """ddG median 오름차순(더 음수=더 강한 결합=1위) 순위. None은 순위 없음(None)."""
    valid = [(seq, v) for seq, v in medians.items() if v is not None]
    valid.sort(key=lambda kv: kv[1])
    ranks: Dict[str, Optional[int]] = {seq: i + 1 for i, (seq, _v) in enumerate(valid)}
    for seq, v in medians.items():
        if v is None:
            ranks.setdefault(seq, None)
    return ranks


def determine_verdict(
    rank_shift: Optional[int],
    significant: Optional[bool],
    pred_n: Optional[int],
    exp_n: Optional[int],
) -> str:
    """agreed / inverted / inconclusive_SEM 판정.

    우선순위:
      1. n_converged 부재/n<2(SD·SEM 추정 불가) → inconclusive_SEM
      2. 순위 뒤집힘(rank_shift >= 2) → inverted (예측 오탐/과탐 신호 — A1의
         AGCKWD 사례: 예측 top5 → 실측 최하위)
      3. 그 외 → agreed (절대값 차이가 유의하더라도 순위가 유지되면, 그 차이는
         예측/실측 구조 베이스라인 차이로 해석 — Wave 1 A1 결론 반영)
    """
    if pred_n is None or exp_n is None:
        return "inconclusive_SEM"
    if pred_n < 2 or exp_n < 2:
        return "inconclusive_SEM"
    if significant is None:
        return "inconclusive_SEM"
    if rank_shift is not None and rank_shift >= 2:
        return "inverted"
    return "agreed"


# ---------------------------------------------------------------------------
# 리포트
# ---------------------------------------------------------------------------

def build_report(out_path: Path, sequences: Optional[List[str]]) -> Dict:
    completed = load_completed_by_side(out_path)

    if sequences is None:
        seen: List[str] = []
        for (seq, _side) in completed.keys():
            if seq not in seen:
                seen.append(seq)
        sequences = seen

    pred_medians: Dict[str, Optional[float]] = {}
    exp_medians: Dict[str, Optional[float]] = {}
    for seq in sequences:
        prec = completed.get((seq, "predicted"))
        erec = completed.get((seq, "experimental"))
        pred_medians[seq] = prec.get("ddg_median") if prec else None
        exp_medians[seq] = erec.get("ddg_median") if erec else None

    pred_ranks = assign_ranks(pred_medians)
    exp_ranks = assign_ranks(exp_medians)

    rows = []
    for seq in sequences:
        prec = completed.get((seq, "predicted")) or {}
        erec = completed.get((seq, "experimental")) or {}
        pred_m = pred_medians[seq]
        exp_m = exp_medians[seq]
        pred_sd = prec.get("ddg_sd")
        pred_n = prec.get("n_converged")
        exp_sd = erec.get("ddg_sd")
        exp_n = erec.get("n_converged")

        sig = compute_delta_and_significance(pred_m, pred_sd, pred_n, exp_m, exp_sd, exp_n)
        rp = pred_ranks.get(seq)
        rex = exp_ranks.get(seq)
        rank_shift = abs(rp - rex) if (rp is not None and rex is not None) else None
        verdict = determine_verdict(rank_shift, sig["significant"], pred_n, exp_n)

        rows.append(
            {
                "sequence": seq,
                "predicted_ddg_median": pred_m,
                "predicted_ddg_sd": pred_sd,
                "predicted_n_converged": pred_n,
                "experimental_ddg_median": exp_m,
                "experimental_ddg_sd": exp_sd,
                "experimental_n_converged": exp_n,
                "delta_abs": sig["delta_abs"],
                "sem_delta": sig["sem_delta"],
                "significant": sig["significant"],
                "rank_predicted": rp,
                "rank_experimental": rex,
                "rank_shift": rank_shift,
                "verdict": verdict,
                "experimental_source": erec.get("source"),
                "legacy_cache_reliable": erec.get("legacy_cache_reliable"),
            }
        )

    pred_receptor = next((r.get("receptor") for r in completed.values() if r.get("side") == "predicted"), None)
    exp_receptor = next((r.get("receptor") for r in completed.values() if r.get("side") == "experimental"), None)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sequences": len(sequences),
        "predicted_receptor": pred_receptor,
        "experimental_receptor": exp_receptor,
        "rows": rows,
    }


def print_report_table(report: Dict) -> None:
    print("\n" + "=" * 100, file=sys.stderr)
    print("[pred-vs-exp] 예측 vs 실측 대조 리포트", file=sys.stderr)
    print(f"  predicted  : {report.get('predicted_receptor')}", file=sys.stderr)
    print(f"  experimental: {report.get('experimental_receptor')}", file=sys.stderr)
    header = (
        f"{'sequence':<16} {'pred_med':>10} {'exp_med':>10} {'delta':>8} "
        f"{'sem_d':>8} {'rank_p':>7} {'rank_e':>7} {'shift':>6} {'verdict':<18}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for row in report.get("rows", []):
        pm = row["predicted_ddg_median"]
        em = row["experimental_ddg_median"]
        d = row["delta_abs"]
        sd = row["sem_delta"]
        caveat = "" if row.get("legacy_cache_reliable") is not False else "  [!legacy_unreliable]"
        print(
            f"{row['sequence']:<16} "
            f"{pm if pm is not None else '-':>10} "
            f"{em if em is not None else '-':>10} "
            f"{d if d is not None else '-':>8} "
            f"{sd if sd is not None else '-':>8} "
            f"{row['rank_predicted'] if row['rank_predicted'] is not None else '-':>7} "
            f"{row['rank_experimental'] if row['rank_experimental'] is not None else '-':>7} "
            f"{row['rank_shift'] if row['rank_shift'] is not None else '-':>6} "
            f"{row['verdict']:<18}{caveat}",
            file=sys.stderr,
        )
    print("=" * 100, file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="예측(Boltz) vs 실측(7T10) 구조 재도킹 대조 파이프라인 (Wave 1 A1 후속)"
    )
    ap.add_argument("--sequences", default=None, help="파일 경로(한 줄 한 서열) 또는 콤마 구분 문자열")
    ap.add_argument("--predicted", default=DEFAULT_PREDICTED, help="예측 receptor PDB (repo 상대경로)")
    ap.add_argument("--experimental", default=DEFAULT_EXPERIMENTAL, help="실측 receptor PDB (repo 상대경로)")
    ap.add_argument("--nstruct", type=int, default=DEFAULT_NSTRUCT, help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT})")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument(
        "--seed-experimental-cache",
        default=None,
        help=(
            "구 jsonl(예: runs/exp72_analysis/7t10_redock/results.jsonl)을 experimental 캐시로 흡수. "
            "⚠ native 이외 서열은 chain 'A' 버그로 변이 미적용 의심 — legacy_cache_reliable=False로 표기됨."
        ),
    )
    ap.add_argument("--predicted-only", action="store_true", help="predicted 쪽만 도킹 (experimental 생략)")
    ap.add_argument("--experimental-only", action="store_true", help="experimental 쪽만 도킹 (predicted 생략)")
    ap.add_argument("--report-only", action="store_true", help="도킹 생략, 기존 --out으로 리포트만 재생성")
    args = ap.parse_args()

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predicted = str(REPO / args.predicted)
    experimental = str(REPO / args.experimental)

    if not (REPO / args.predicted).exists():
        print(f"ERR: predicted receptor 없음: {predicted}", file=sys.stderr)
        sys.exit(1)
    if not (REPO / args.experimental).exists():
        print(f"ERR: experimental receptor 없음: {experimental}", file=sys.stderr)
        sys.exit(1)

    if args.sequences is None:
        if not args.report_only:
            print("ERR: --sequences 필요 (--report-only 모드 제외)", file=sys.stderr)
            sys.exit(1)
        sequences: Optional[List[str]] = None
    else:
        sequences = load_sequences(args.sequences)
        print(f"[pred-vs-exp] {len(sequences)}개 서열 로드", file=sys.stderr)

    completed = load_completed_by_side(out_path)

    if args.seed_experimental_cache:
        legacy_path = REPO / args.seed_experimental_cache
        seeds = seed_legacy_cache(legacy_path)
        n_seeded = 0
        for seq, seed_rec in seeds.items():
            key = (seq, "experimental")
            if completed.get(key, {}).get("status") == "ok":
                continue  # 이미 본 파이프라인이 직접 계산한 값이 있으면 legacy로 덮어쓰지 않음
            rec = {
                "sequence": seq,
                "side": "experimental",
                "receptor": args.experimental,
                "nstruct": seed_rec.get("n_total"),
                "status": "ok",
                "ddg_median": seed_rec["ddg_median"],
                "ddg_sd": seed_rec["ddg_sd"],
                "n_converged": seed_rec["n_converged"],
                "sg_sg_distance": seed_rec.get("sg_sg_distance"),
                "disulfide_intact": seed_rec.get("disulfide_intact"),
                "source": seed_rec["source"],
                "legacy_cache_reliable": seed_rec["legacy_cache_reliable"],
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with out_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            completed[key] = rec
            n_seeded += 1
            print(
                f"  [seed] {seq} legacy experimental ddg_median={rec['ddg_median']} "
                f"reliable={rec['legacy_cache_reliable']}",
                file=sys.stderr,
            )
        print(f"[pred-vs-exp] legacy cache에서 {n_seeded}건 흡수 ({legacy_path})", file=sys.stderr)

    if not args.report_only:
        import pyrosetta

        pyrosetta.init(
            "-mute all -ex1 -ex2aro -ignore_unrecognized_res "
            "-flexPepDocking:pep_refine -constraints:cst_fa_weight 1.0"
        )
        work_dir = out_path.parent / "poses"
        work_dir.mkdir(parents=True, exist_ok=True)

        if not args.experimental_only:
            print(f"[pred-vs-exp] predicted 도킹: {len(sequences)}개 서열", file=sys.stderr)
            run_docking(
                sequences, predicted, "predicted", args.nstruct, work_dir, out_path,
                args.shard_index, args.shard_count, completed,
            )
        if not args.predicted_only:
            print(f"[pred-vs-exp] experimental 도킹: {len(sequences)}개 서열", file=sys.stderr)
            run_docking(
                sequences, experimental, "experimental", args.nstruct, work_dir, out_path,
                args.shard_index, args.shard_count, completed,
            )

    report = build_report(out_path, sequences)
    report_path = out_path.with_suffix(".report.json")
    with report_path.open("w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print_report_table(report)
    print(f"[pred-vs-exp] 리포트 저장: {report_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
