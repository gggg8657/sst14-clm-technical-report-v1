#!/usr/bin/env python3
"""redock_native_baseline.py
============================
native SST-14 (AGCKNFFWKTFTSC) 를 cross_silo_validation 과 **완전 동일 조건** 으로
robust FlexPepDock 재도킹하여 공정한 baseline ddG 를 산출한다.

동일 조건 (cross_silo_validation.py::run_robust_flexpep 로직과 1:1 대응):
  - 입력 PDB : find_silo_b_pdb(native_seq) 가 반환하는 PDB
              (experiment_log.jsonl 에서 native 서열 마지막 등장 docked PDB)
  - protocol : flexpep_refine  (FlexPepDockingProtocol + InterfaceAnalyzerMover)
  - nstruct  : 5  (기본값, --nstruct 로 변경 가능)
  - ddG 집계 : ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150) 을 "converged" 로 집계
               → ddg_median, ddg_sd, ddg_mean, ddg_min, n_converged, n_total

출력:
  runs/pyrosetta_flow/native_robust_baseline.json
  (기존 native_ddg = -41.9864 파일은 덮어쓰지 않음 — 별도 파일)

사용법:
  conda run -n bio-tools python scripts/redock_native_baseline.py
  conda run -n bio-tools python scripts/redock_native_baseline.py --nstruct 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
EXPERIMENT_LOG = PYROSETTA_FLOW_DIR / "experiment_log.jsonl"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"
OUT_JSON = PYROSETTA_FLOW_DIR / "native_robust_baseline.json"
WORK_DIR = PYROSETTA_FLOW_DIR / "native_robust_work"

# cross_silo_validation 임포트 경로 추가 (run_robust_flexpep 재사용)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))

NATIVE_SEQ = "AGCKNFFWKTFTSC"
_CAND_RE = re.compile(r"iter(\d+)_cand(\d+)")


def find_native_pdb() -> Optional[Path]:
    """experiment_log.jsonl 에서 native 서열의 마지막 등장 docked PDB 경로를 반환.

    cross_silo_validation.py::find_silo_b_pdb() 의 동일 로직.
    """
    if not EXPERIMENT_LOG.exists():
        print(f"[ERROR] experiment_log.jsonl 없음: {EXPERIMENT_LOG}", file=sys.stderr)
        return None

    last_pdb: Optional[Path] = None
    with open(EXPERIMENT_LOG, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if d.get("sequence") != NATIVE_SEQ:
                continue
            m = _CAND_RE.search(d.get("candidate_id", ""))
            if m is None:
                continue
            run_id: str = d.get("run_id", "")
            iter_num = int(m.group(1))
            cand_num = int(m.group(2))
            pdb = (
                PYROSETTA_FLOW_DIR
                / "archives"
                / run_id
                / f"iter_{iter_num:02d}"
                / f"cand_{cand_num:03d}.pdb"
            )
            if pdb.exists():
                last_pdb = pdb

    return last_pdb


def run_native_robust_redock(
    pdb_path: Path,
    nstruct: int = 5,
) -> dict:
    """cross_silo_validation.py::run_robust_flexpep 와 **동일 로직** 으로 재도킹.

    코드 복사를 피하기 위해 cross_silo_validation 모듈을 임포트해 직접 호출한다.
    """
    from cross_silo_validation import run_robust_flexpep  # type: ignore

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    stats = run_robust_flexpep(
        pdb_path=pdb_path,
        candidate_id="native_AGCKNFFWKTFTSC",
        nstruct=nstruct,
    )
    elapsed = round(time.perf_counter() - t0, 1)
    stats["elapsed_s"] = elapsed
    return stats


def compare_with_crossval(
    native_robust_ddg: Optional[float],
    n_converged: int,
) -> dict:
    """cross_silo_validation 후보들의 cross_ddg 와 비교해 공정 판정.

    cross_validation.json 에서 confirmed/uncertain 후보의 siloB_robust_ddg_median 수집 후
    native robust ddG 와 비교.
    """
    cross_val_json = REPO_ROOT / "runs" / "silo_a_flow" / "cross_validation.json"
    if not cross_val_json.exists():
        return {"candidates_compared": 0, "comparison": []}

    with open(cross_val_json, encoding="utf-8") as f:
        cv = json.load(f)

    results = cv.get("results", cv.get("entries", {}))
    if isinstance(results, dict):
        entries = list(results.values())
    else:
        entries = results

    comparisons = []
    for r in entries:
        seq = r.get("sequence", "")
        cross_ddg = r.get("siloB_robust_ddg_median")
        nc = r.get("n_converged", 0)
        verdict = r.get("verdict", "")
        if cross_ddg is None:
            continue

        if native_robust_ddg is not None:
            if cross_ddg < native_robust_ddg:
                judgment = "후보가_native보다_강함(더_음수)"
            elif abs(cross_ddg - native_robust_ddg) < 3.0:
                judgment = "동등(±3REU_이내)"
            else:
                judgment = "후보가_native보다_약함(덜_음수)"
        else:
            judgment = "native_ddg_없음_비교불가"

        comparisons.append({
            "sequence": seq,
            "cross_ddg": cross_ddg,
            "cross_n_converged": nc,
            "verdict": verdict,
            "judgment": judgment,
        })

    # ddg 기준 정렬 (강한 결합 = 더 음수 = 오름차순)
    comparisons.sort(key=lambda x: x["cross_ddg"])

    return {
        "candidates_compared": len(comparisons),
        "comparison": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "native SST-14 robust 재도킹 — cross_silo_validation 과 완전 동일 조건"
        )
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=5,
        help="FlexPepDock nstruct (기본 5, cross-val 동일 조건)",
    )
    args = parser.parse_args()

    nstruct: int = args.nstruct
    t_total = time.perf_counter()

    print(
        f"[native-baseline] native SST-14 robust 재도킹 시작 (nstruct={nstruct})",
        file=sys.stderr,
    )
    print(f"[native-baseline] 서열: {NATIVE_SEQ}", file=sys.stderr)
    print(
        f"[native-baseline] 기존 native_ddg_median: -41.9864 (n_converged=2, 원본 조건 불명확)",
        file=sys.stderr,
    )
    print(
        f"[native-baseline] 목표: nstruct={nstruct} 동일 조건으로 재측정",
        file=sys.stderr,
    )

    # ① 입력 PDB 탐색 (cross_silo_validation::find_silo_b_pdb 동일 로직)
    pdb_path = find_native_pdb()
    if pdb_path is None:
        print(
            "[ERROR] native 서열에 대한 docked PDB 없음 "
            "(experiment_log 에 존재하지 않거나 archives PDB 파일 없음)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[native-baseline] 입력 PDB: {pdb_path}", file=sys.stderr)

    # ② robust FlexPepDock (cross_silo_validation::run_robust_flexpep 직접 호출)
    stats = run_native_robust_redock(pdb_path, nstruct=nstruct)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    ddg_median: Optional[float] = stats.get("ddg_median")
    n_converged: int = int(stats.get("n_converged") or 0)
    ddg_sd: Optional[float] = stats.get("ddg_sd")
    ddg_min: Optional[float] = stats.get("ddg_min")
    n_total: int = int(stats.get("n_total") or nstruct)
    flexpep_error: Optional[str] = stats.get("error")

    # ③ cross-val 후보와 공정 비교
    comparison = compare_with_crossval(ddg_median, n_converged)

    # ④ 결론 판정
    if ddg_median is None:
        conclusion = "도킹실패_비교불가"
    elif n_converged == 0:
        conclusion = "수렴없음_비교불가"
    else:
        prev_native = -41.9864
        prev_n_conv = 2
        diff_from_prev = round(ddg_median - prev_native, 4)
        conclusion_parts = [
            f"native_robust_ddg={ddg_median:.4f} "
            f"(n_conv={n_converged}/{n_total})"
        ]
        if abs(diff_from_prev) < 3.0:
            conclusion_parts.append(
                f"기존_native_ddg(-41.99,n_conv=2)와_동등(Δ={diff_from_prev:+.2f}REU)"
            )
        elif ddg_median < prev_native:
            conclusion_parts.append(
                f"기존_native_ddg보다_강함(Δ={diff_from_prev:+.2f}REU)"
            )
        else:
            conclusion_parts.append(
                f"기존_native_ddg보다_약함(Δ={diff_from_prev:+.2f}REU)"
            )
        conclusion = " | ".join(conclusion_parts)

    import datetime
    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "description": (
            "native SST-14 robust 재도킹 — cross_silo_validation 완전 동일 조건 "
            "(nstruct=5, flexpep_refine, ddg<0 수렴 집계)"
        ),
        "measurement_conditions": {
            "protocol": "flexpep_refine",
            "nstruct": nstruct,
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만)",
            "input_pdb": str(pdb_path),
            "input_pdb_selection": (
                "experiment_log.jsonl 에서 native 서열(AGCKNFFWKTFTSC) "
                "마지막 등장 docked PDB — cross_silo_validation::find_silo_b_pdb 동일 로직"
            ),
            "identical_to_cross_val": True,
        },
        "native_seq": NATIVE_SEQ,
        # robust 재도킹 결과
        "ddg_median": ddg_median,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "ddg_mean": stats.get("ddg_mean"),
        "n_converged": n_converged,
        "n_total": n_total,
        # 기존 값 (비교용, 덮어쓰지 않음)
        "previous_native_ddg_median": -41.9864,
        "previous_native_ddg_n_converged": 2,
        "previous_native_ddg_note": (
            "global_selectivity_leaderboard.json::native_ddg_median — "
            "nstruct·출처 불명확, n_converged=2 (cross-val 기준 n>=3 미충족)"
        ),
        # 공정 비교 결론
        "conclusion": conclusion,
        "flexpep_error": flexpep_error,
        "elapsed_total_s": elapsed_total,
        # cross-val 후보와의 비교
        "crossval_comparison": comparison,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # ── 결과 요약 출력 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65, file=sys.stderr)
    print("[native-baseline] 재도킹 완료 요약", file=sys.stderr)
    print(f"  총 소요시간      : {elapsed_total}s", file=sys.stderr)
    print(f"  입력 PDB         : {pdb_path}", file=sys.stderr)
    print(f"  nstruct          : {nstruct}", file=sys.stderr)
    print(f"  ddg_median       : {ddg_median}", file=sys.stderr)
    print(f"  n_converged      : {n_converged} / {n_total}", file=sys.stderr)
    print(f"  ddg_sd           : {ddg_sd}", file=sys.stderr)
    print(f"  ddg_min          : {ddg_min}", file=sys.stderr)
    print(f"  기존 native_ddg  : -41.9864 (n_conv=2, 조건불명)", file=sys.stderr)
    print(f"  결론             : {conclusion}", file=sys.stderr)
    print(f"  출력 파일        : {OUT_JSON}", file=sys.stderr)

    if comparison["candidates_compared"] > 0:
        print(
            f"\n[native-baseline] cross-val 후보 {comparison['candidates_compared']}건 공정 비교:",
            file=sys.stderr,
        )
        for c in comparison["comparison"]:
            print(
                f"  {c['sequence']} | cross_ddg={c['cross_ddg']:.4f} "
                f"n_conv={c['cross_n_converged']} verdict={c['verdict']} "
                f"→ {c['judgment']}",
                file=sys.stderr,
            )

    print("=" * 65, file=sys.stderr)

    # stdout: JSON 요약 (파이프라인 캡처용)
    summary = {
        "native_robust_ddg_median": ddg_median,
        "n_converged": n_converged,
        "n_total": n_total,
        "ddg_sd": ddg_sd,
        "previous_native_ddg": -41.9864,
        "conclusion": conclusion,
        "out_json": str(OUT_JSON),
        "elapsed_s": elapsed_total,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
