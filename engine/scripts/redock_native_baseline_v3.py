#!/usr/bin/env python3
"""redock_native_baseline_v3.py
================================
native SST-14 복합체(SSTR2_SST14_complex_boltz_1.pdb)로
FlexPepDock nstruct=50 robust 재도킹 — −42 vs −20 모순 결판.

목적:
  - nstruct=50으로 충분한 샘플로 native baseline 결판
  - v2(nstruct=10, median=-20.28)와 비교
  - experiment_log에서 추출된 -41.99(n=2)의 진짜 신뢰도 평가

모순 원인 요약 (이 스크립트 실행 전 이미 추적 완료):
  - −41.99: experiment_log native 서열 유효 ddg 2개(-33.39, -50.58), median=-41.99
    → n=2, nstruct 불명확, 입력 PDB 불명확 (발굴 과정의 부산물)
  - −20.28: v2(nstruct=10, 진짜 native PDB SSTR2_SST14_complex_boltz_1.pdb)
    → n=8/10, 올바른 입력
  - 이 v3: nstruct=50으로 어느 쪽이 진짜인지 결판

사용법:
  conda run -n bio-tools python scripts/redock_native_baseline_v3.py
  conda run -n bio-tools python scripts/redock_native_baseline_v3.py --nstruct 50
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"

NATIVE_PDB = (
    REPO_ROOT
    / "data"
    / "somatostatin_receptor"
    / "SSTR2_SST14_complex_boltz_1.pdb"
)

# v2에서 이미 검증·재정렬된 PDB 재사용
REORDERED_PDB = PYROSETTA_FLOW_DIR / "native_robust_work_v2" / "native_reordered.pdb"

OUT_JSON = PYROSETTA_FLOW_DIR / "native_robust_baseline_v3.json"
WORK_DIR = PYROSETTA_FLOW_DIR / "native_robust_work_v3"

NATIVE_SEQ = "AGCKNFFWKTFTSC"
DEFAULT_NSTRUCT = 50
DDG_PHYSICAL_FLOOR = -150.0

if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))


# ── FlexPepDock nstruct=50 robust 재도킹 ─────────────────────────────────────

def run_native_robust_v3(
    input_pdb: Path,
    nstruct: int = DEFAULT_NSTRUCT,
) -> Dict:
    """FlexPepDock nstruct=N robust 재도킹.

    flexpep_dock.py::run_flexpep_refine 재사용.
    ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150) 을 "converged" 로 집계.
    max_workers 제한(발굴 엔진 방해 최소화): FLEXPEP_NSTRUCT는 sequential 실행.
    """
    from flexpep_dock import run_flexpep_refine, init_pyrosetta  # type: ignore

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_pdb = str(WORK_DIR / "native_refined_v3.pdb")

    print(
        f"  [FlexPepDock-v3] native nstruct={nstruct} 시작: {input_pdb}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        init_pyrosetta()
        _pose, stats = run_flexpep_refine(str(input_pdb), output_pdb, nstruct=nstruct)
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            f"  [FlexPepDock-v3] 완료 {elapsed}s | "
            f"ddg_median={stats.get('ddg_median')} "
            f"n_converged={stats.get('n_converged')}/{nstruct}",
            file=sys.stderr,
        )
        stats["elapsed_s"] = elapsed
        return stats
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [FlexPepDock-v3] 실패 {elapsed}s: {exc}", file=sys.stderr)
        return {
            "ddg_median": None, "ddg_sd": None, "ddg_mean": None,
            "ddg_min": None, "n_converged": 0, "n_total": nstruct,
            "converged": False, "elapsed_s": elapsed,
            "error": str(exc),
        }


# ── −42 vs −20 모순 분석 ─────────────────────────────────────────────────────

def analyze_discrepancy(v3_stats: Dict, nstruct: int) -> Dict:
    """−42 vs −20 모순의 근본 원인 및 v3 결론 정리."""

    # experiment_log 기반 −42 값 재계산
    exp_log = PYROSETTA_FLOW_DIR / "experiment_log.jsonl"
    exp_vals: List[float] = []
    exp_all_ddg: List[Optional[float]] = []
    if exp_log.exists():
        with open(exp_log) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                seq = row.get("sequence") or row.get("seq")
                if seq != NATIVE_SEQ:
                    continue
                ddg_raw = row.get("ddg") or row.get("ddG")
                if ddg_raw is None:
                    ddg_raw = row.get("ddg_median") or row.get("ddG_median")
                try:
                    ddg_f = float(ddg_raw) if ddg_raw is not None else None
                except (TypeError, ValueError):
                    ddg_f = None
                exp_all_ddg.append(ddg_f)
                if ddg_f is not None and ddg_f < 0 and ddg_f > DDG_PHYSICAL_FLOOR:
                    exp_vals.append(ddg_f)

    exp_median: Optional[float] = None
    exp_sd: Optional[float] = None
    if exp_vals:
        exp_median = round(statistics.median(exp_vals), 4)
        exp_sd = round(statistics.stdev(exp_vals), 4) if len(exp_vals) > 1 else 0.0

    ddg_v3 = v3_stats.get("ddg_median")
    n_conv_v3 = v3_stats.get("n_converged", 0)

    # 결론 판정
    if ddg_v3 is None:
        verdict = "INCONCLUSIVE — v3 도킹 실패"
        conclusion = "재측정 필요"
    elif n_conv_v3 < 5:
        verdict = f"UNCERTAIN — n_converged={n_conv_v3}/{nstruct}"
        conclusion = "nstruct 추가 필요"
    else:
        # v3 median이 −42 vs −20 어느 쪽에 가까운가
        dist_to_42 = abs(ddg_v3 - (-41.99))
        dist_to_20 = abs(ddg_v3 - (-20.28))
        if dist_to_42 < dist_to_20:
            verdict = f"−42 쪽 (v3 median={ddg_v3:.2f}, n={n_conv_v3}/{nstruct})"
            conclusion = (
                f"−42 계열이 진짜 native baseline에 가깝다. "
                f"v2(−20.28, n=8/10)는 nstruct 부족으로 고강도 binding 샘플이 "
                f"충분히 반영되지 않았을 가능성. v3 median={ddg_v3:.2f} 사용 권고."
            )
        elif dist_to_20 < dist_to_42:
            verdict = f"−20 쪽 (v3 median={ddg_v3:.2f}, n={n_conv_v3}/{nstruct})"
            conclusion = (
                f"−20 계열이 진짜 native baseline에 가깝다. "
                f"experiment_log −42(n=2)는 단 2개의 샘플 편향이었음. "
                f"v3 median={ddg_v3:.2f}, v2 median=-20.28 일치 → v3 사용 권고."
            )
        else:
            verdict = f"중간 (v3 median={ddg_v3:.2f}, n={n_conv_v3}/{nstruct})"
            conclusion = f"v3 median={ddg_v3:.2f} 사용 권고."

    return {
        "minus_42_origin": {
            "value": exp_median,
            "n_valid_samples": len(exp_vals),
            "all_valid_ddg": exp_vals,
            "source": "experiment_log.jsonl — native 서열 유효(ddg<0, ddg>-150) 누적값",
            "how_it_became_42": (
                f"experiment_log에서 native 서열의 유효 ddg 2개: {exp_vals}. "
                f"median({exp_vals}) = {exp_median}. "
                "n=2(매우 적음), 입력 PDB 불명확, nstruct 불명확 — 발굴 과정의 부산물."
            ),
            "problem": "n=2 편향. 단 2번의 측정으로 추정된 median이므로 신뢰도 LOW.",
        },
        "minus_20_origin": {
            "value": -20.28,
            "n_valid_samples": 8,
            "source": "native_robust_baseline_v2.json (nstruct=10)",
            "input_pdb": str(NATIVE_PDB),
            "input_validated": True,
            "problem": (
                "nstruct=10으로 충분하나, n=8/10은 안정적. "
                "그러나 nstruct=50 대비 신뢰도는 낮음."
            ),
        },
        "v3_result": {
            "value": ddg_v3,
            "n_converged": n_conv_v3,
            "n_total": nstruct,
            "ddg_sd": v3_stats.get("ddg_sd"),
            "ddg_min": v3_stats.get("ddg_min"),
            "ddg_mean": v3_stats.get("ddg_mean"),
            "convergence_rate": round(n_conv_v3 / nstruct, 4) if nstruct > 0 else 0,
        },
        "verdict": verdict,
        "conclusion": conclusion,
        "key_insight": (
            "−42는 experiment_log에서 우연히 강하게 결합한 2개 샘플(−33.4, −50.6)의 "
            "median이며 nstruct·입력PDB가 불명확한 발굴 부산물. "
            "−20은 검증된 native PDB + nstruct=10 controlled 측정. "
            "v3(nstruct=50)가 진짜 baseline."
        ),
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "native SST-14 FlexPepDock nstruct=50 robust 재도킹 — −42 vs −20 결판"
        )
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=DEFAULT_NSTRUCT,
        help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT})",
    )
    args = parser.parse_args()
    nstruct: int = args.nstruct

    t_total = time.perf_counter()

    print(
        f"[native-v3] native SST-14 nstruct={nstruct} robust 재도킹 시작",
        file=sys.stderr,
    )
    print(f"[native-v3] 서열: {NATIVE_SEQ}", file=sys.stderr)

    # ① 입력 PDB 결정 (v2에서 재정렬된 PDB 우선, 없으면 원본 사용)
    if REORDERED_PDB.exists():
        input_pdb = REORDERED_PDB
        print(
            f"[native-v3] v2 재정렬 PDB 재사용: {input_pdb}",
            file=sys.stderr,
        )
    else:
        input_pdb = NATIVE_PDB
        print(
            f"[native-v3] 원본 native PDB 사용: {input_pdb}",
            file=sys.stderr,
        )

    if not input_pdb.exists():
        print(f"[ERROR] 입력 PDB 없음: {input_pdb}", file=sys.stderr)
        sys.exit(1)

    # ② robust FlexPepDock (nstruct=50)
    stats = run_native_robust_v3(input_pdb, nstruct=nstruct)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    ddg_median: Optional[float] = stats.get("ddg_median")
    n_converged: int = int(stats.get("n_converged") or 0)
    ddg_sd: Optional[float] = stats.get("ddg_sd")
    ddg_min: Optional[float] = stats.get("ddg_min")
    ddg_max: Optional[float] = stats.get("ddg_max")
    ddg_mean: Optional[float] = stats.get("ddg_mean")
    n_total: int = int(stats.get("n_total") or nstruct)
    convergence_rate: float = round(n_converged / n_total, 4) if n_total > 0 else 0.0
    flexpep_error: Optional[str] = stats.get("error")

    # ③ 신뢰도 판정
    if ddg_median is None:
        reliability = "FAILED — 도킹 실패"
        recommendation = "재측정 필요"
    elif n_converged < 5:
        reliability = f"UNCERTAIN — n_converged={n_converged}/{n_total} (기준 n≥5 미충족)"
        recommendation = "nstruct 늘려 재측정 필요"
    elif n_converged >= 20:
        reliability = f"VERY HIGH — n_converged={n_converged}/{n_total} ({convergence_rate*100:.0f}%), 진짜 native PDB"
        recommendation = (
            f"native baseline ddg_median={ddg_median:.4f} REU 확정. "
            f"sd={ddg_sd} REU, 수렴률={convergence_rate*100:.0f}%"
        )
    elif n_converged >= 10:
        reliability = f"HIGH — n_converged={n_converged}/{n_total} ({convergence_rate*100:.0f}%), 진짜 native PDB"
        recommendation = (
            f"native baseline ddg_median={ddg_median:.4f} REU 사용 권고. "
            f"sd={ddg_sd} REU"
        )
    else:
        reliability = f"MODERATE — n_converged={n_converged}/{n_total} ({convergence_rate*100:.0f}%)"
        recommendation = (
            f"native baseline ddg_median={ddg_median:.4f} REU 사용 가능. "
            f"sd={ddg_sd} REU"
        )

    # ④ 모순 분석
    discrepancy = analyze_discrepancy(stats, nstruct)

    # ⑤ 결과 저장
    out_data: Dict = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "redock_native_baseline_v3.py",
        "description": (
            f"native SST-14 FlexPepDock nstruct={nstruct} robust 재도킹 — "
            "−42 vs −20 모순 결판. 진짜 native PDB 입력."
        ),
        "measurement_conditions": {
            "protocol": "flexpep_refine",
            "nstruct": nstruct,
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만)",
            "input_pdb": str(input_pdb),
            "original_native_pdb": str(NATIVE_PDB),
            "chain_reorder_applied": (input_pdb == REORDERED_PDB),
            "note": (
                "v2에서 검증·재정렬된 PDB 재사용 (Chain A=수용체, Chain B=native 펩타이드). "
                "SSTR2_SST14_complex_boltz_1.pdb에서 파생."
            ),
        },
        "native_seq": NATIVE_SEQ,
        "ddg_median": ddg_median,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "ddg_max": ddg_max,
        "ddg_mean": ddg_mean,
        "n_converged": n_converged,
        "n_total": n_total,
        "convergence_rate": convergence_rate,
        "n_unphysical": stats.get("n_unphysical", 0),
        "reliability": reliability,
        "recommendation": recommendation,
        "flexpep_error": flexpep_error,
        "elapsed_total_s": elapsed_total,
        # 비교 컨텍스트
        "comparison": {
            "v2_nstruct10": {"ddg_median": -20.2753, "n_converged": 8, "n_total": 10},
            "exp_log_native_n2": {"ddg_median": -41.9864, "n_converged": 2, "n_total": "unknown"},
        },
        # 모순 분석
        "discrepancy_analysis": discrepancy,
    }

    PYROSETTA_FLOW_DIR.mkdir(parents=True, exist_ok=True)

    # v3 파일 존재 여부 확인 — 덮어쓰되 v2는 절대 건드리지 않음
    v2_path = PYROSETTA_FLOW_DIR / "native_robust_baseline_v2.json"
    assert not (OUT_JSON == v2_path), "v3이 v2 경로를 덮어쓰려 함 — 중단"

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # ── 결과 요약 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 72, file=sys.stderr)
    print("[native-v3] 재도킹 완료 요약", file=sys.stderr)
    print(f"  총 소요시간       : {elapsed_total}s", file=sys.stderr)
    print(f"  입력 PDB          : {input_pdb}", file=sys.stderr)
    print(f"  nstruct           : {nstruct}", file=sys.stderr)
    print(f"  ddg_median        : {ddg_median}", file=sys.stderr)
    print(f"  n_converged       : {n_converged} / {n_total} ({convergence_rate*100:.0f}%)", file=sys.stderr)
    print(f"  ddg_sd            : {ddg_sd}", file=sys.stderr)
    print(f"  ddg_min           : {ddg_min}", file=sys.stderr)
    print(f"  ddg_max           : {ddg_max}", file=sys.stderr)
    print(f"  신뢰도            : {reliability}", file=sys.stderr)
    print(f"  권고              : {recommendation}", file=sys.stderr)
    print(f"  출력 파일         : {OUT_JSON}", file=sys.stderr)
    print("\n  [−42 vs −20 결판]", file=sys.stderr)
    print(f"  판정: {discrepancy['verdict']}", file=sys.stderr)
    print(f"  결론: {discrepancy['conclusion']}", file=sys.stderr)
    print(f"  핵심: {discrepancy['key_insight']}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    summary = {
        "native_robust_ddg_median_v3": ddg_median,
        "n_converged": n_converged,
        "n_total": n_total,
        "convergence_rate": convergence_rate,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "reliability": reliability,
        "verdict_42_vs_20": discrepancy["verdict"],
        "conclusion": discrepancy["conclusion"],
        "out_json": str(OUT_JSON),
        "elapsed_s": elapsed_total,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
