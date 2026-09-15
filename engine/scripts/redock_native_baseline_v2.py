#!/usr/bin/env python3
"""redock_native_baseline_v2.py
================================
진짜 native SST-14 복합체(SSTR2_SST14_complex_boltz_1.pdb)로
FlexPepDock robust 재도킹 — nstruct=10, physical floor 적용.

기존 redock_native_baseline.py의 문제:
  - 입력이 experiment_log.jsonl에서 찾은 cand_006.pdb (AGCKNIFWKTFNSC)
    → native AGCKNFFWKTFTSC가 아닌 변이 후보 docked pose
  - nstruct=5 (노이즈 큰 native baseline에 부족)

이 스크립트의 개선:
  - 입력: data/somatostatin_receptor/SSTR2_SST14_complex_boltz_1.pdb
    (Chain A=AGCKNFFWKTFTSC 진짜 native 펩타이드, Chain B=SSTR2 수용체)
  - nstruct=10 (기본값, --nstruct로 변경 가능)
  - chain 재정렬: FlexPepDock 표준(수용체 먼저, 펩타이드 마지막)으로 자동 정렬
  - 출력: runs/pyrosetta_flow/native_robust_baseline_v2.json
    (기존 파일 덮어쓰지 않음)

사용법:
  conda run -n bio-tools python scripts/redock_native_baseline_v2.py
  conda run -n bio-tools python scripts/redock_native_baseline_v2.py --nstruct 10
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
import tempfile
from pathlib import Path
from typing import Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"

# 진짜 native 복합체 (이 스크립트의 핵심 — cand_006.pdb 아님!)
NATIVE_PDB = (
    REPO_ROOT
    / "data"
    / "somatostatin_receptor"
    / "SSTR2_SST14_complex_boltz_1.pdb"
)

OUT_JSON = PYROSETTA_FLOW_DIR / "native_robust_baseline_v2.json"
WORK_DIR = PYROSETTA_FLOW_DIR / "native_robust_work_v2"

NATIVE_SEQ = "AGCKNFFWKTFTSC"

# sys.path 보완 (flexpep_dock.py 임포트용)
if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))


# ── PDB 입력 검증 ─────────────────────────────────────────────────────────────

_AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def extract_chain_sequences(pdb_path: Path) -> dict[str, str]:
    """PDB ATOM 기록에서 chain별 아미노산 서열 추출."""
    chains: dict[str, list[tuple[int, str]]] = {}
    prev: dict[str, Optional[int]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                continue
            resname = line[17:20].strip()
            if chain not in chains:
                chains[chain] = []
                prev[chain] = None
            if resnum != prev[chain] and resname in _AA3TO1:
                chains[chain].append((resnum, _AA3TO1[resname]))
                prev[chain] = resnum
    return {ch: "".join(r[1] for r in res) for ch, res in sorted(chains.items())}


def validate_native_pdb(pdb_path: Path) -> tuple[bool, str, str]:
    """native 복합체 PDB가 진짜 native 펩타이드(AGCKNFFWKTFTSC)를 포함하는지 검증.

    Returns:
        (is_valid, peptide_chain_id, found_peptide_seq)
    """
    chain_seqs = extract_chain_sequences(pdb_path)
    for chain_id, seq in chain_seqs.items():
        if seq == NATIVE_SEQ:
            return True, chain_id, seq
    # 서열 전체 비교 실패 시 요약 출력
    summary = {ch: seq[:30] + "..." if len(seq) > 30 else seq
               for ch, seq in chain_seqs.items()}
    return False, "", str(summary)


# ── chain 재정렬 (FlexPepDock 표준: 수용체 먼저, 펩타이드 마지막) ──────────────

def reorder_chains_for_flexpep(
    pdb_path: Path, peptide_chain: str, work_dir: Path
) -> Path:
    """FlexPepDock 표준에 맞게 chain 순서 재정렬 후 임시 PDB 반환.

    SSTR2_SST14_complex_boltz_1.pdb: Chain A=펩타이드, Chain B=수용체
    FlexPepDock 요구: 수용체 먼저, 펩타이드 마지막

    이미 올바른 순서면 원본 경로 반환.
    """
    chain_blocks: dict[str, list[str]] = {}
    chain_order: list[str] = []
    header_lines: list[str] = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                cid = line[21]
                if cid not in chain_blocks:
                    chain_blocks[cid] = []
                    chain_order.append(cid)
                chain_blocks[cid].append(line)
            elif not line.startswith(("TER", "END")):
                header_lines.append(line)

    if not chain_order:
        raise ValueError(f"chain 없음: {pdb_path}")

    # 수용체 chain = 펩타이드 chain이 아닌 모든 chain
    receptor_chains = [c for c in chain_order if c != peptide_chain]
    new_order = receptor_chains + [peptide_chain]

    if new_order == chain_order:
        print(
            f"  [reorder] chain 순서 이미 표준({' '.join(chain_order)}) — 재정렬 불필요",
            file=sys.stderr,
        )
        return pdb_path

    print(
        f"  [reorder] {' '.join(chain_order)} → {' '.join(new_order)} (수용체 먼저, 펩타이드 마지막)",
        file=sys.stderr,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    new_lines = list(header_lines)
    for idx, old_cid in enumerate(new_order):
        new_cid = chr(65 + idx)  # A, B, C, ...
        for atom_line in chain_blocks[old_cid]:
            new_lines.append(atom_line[:21] + new_cid + atom_line[22:])
        new_lines.append("TER\n")
    new_lines.append("END\n")

    reordered_pdb = work_dir / "native_reordered.pdb"
    with open(reordered_pdb, "w") as fh:
        fh.writelines(new_lines)
    return reordered_pdb


# ── FlexPepDock robust 재도킹 ─────────────────────────────────────────────────

def run_native_robust_flexpep(
    input_pdb: Path,
    nstruct: int = 10,
) -> dict:
    """FlexPepDock nstruct=N robust 재도킹.

    flexpep_dock.py::run_flexpep_refine 재사용.
    ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150) 을 "converged" 로 집계.
    """
    from flexpep_dock import run_flexpep_refine, init_pyrosetta  # type: ignore

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_pdb = str(WORK_DIR / "native_refined_v2.pdb")

    print(
        f"  [FlexPepDock] native nstruct={nstruct} 시작: {input_pdb}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        init_pyrosetta()
        _pose, stats = run_flexpep_refine(str(input_pdb), output_pdb, nstruct=nstruct)
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            f"  [FlexPepDock] 완료 {elapsed}s | "
            f"ddg_median={stats.get('ddg_median')} n_converged={stats.get('n_converged')}",
            file=sys.stderr,
        )
        stats["elapsed_s"] = elapsed
        return stats
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [FlexPepDock] 실패 {elapsed}s: {exc}", file=sys.stderr)
        return {
            "ddg_median": None, "ddg_sd": None, "ddg_mean": None,
            "ddg_min": None, "n_converged": 0, "n_total": nstruct,
            "converged": False, "elapsed_s": elapsed,
            "error": str(exc),
        }


# ── 3값 불일치 원인 규명 ──────────────────────────────────────────────────────

def trace_three_values() -> dict:
    """기존 3값의 출처·조건을 추적하여 불일치 원인을 정리."""
    # 1) −41.99: global_selectivity_leaderboard.json
    lb_path = PYROSETTA_FLOW_DIR / "global_selectivity_leaderboard.json"
    lb_val: Optional[float] = None
    lb_n_conv: Optional[int] = None
    lb_input_note = "출처불명 (nstruct·입력PDB 기록 없음)"
    if lb_path.exists():
        try:
            lb = json.loads(lb_path.read_text())
            lb_val = lb.get("native_ddg_median")
            lb_n_conv = lb.get("native_ddg_n_converged")
            lb_input_note = (
                "global_selectivity_leaderboard.json — "
                "nstruct·입력PDB 기록 없음. native_ddg_sd=12.15 → n_converged=2로 sd 높음"
            )
        except Exception:
            pass

    # 2) −6.05: mmgbsa_consensus.json (native_robust_baseline.json 결과를 mmgbsa_consensus가 저장)
    nb_path = PYROSETTA_FLOW_DIR / "native_robust_baseline.json"
    nb_val: Optional[float] = None
    nb_input: Optional[str] = None
    nb_n_conv: Optional[int] = None
    nb_note = ""
    if nb_path.exists():
        try:
            nb = json.loads(nb_path.read_text())
            nb_val = nb.get("ddg_median")
            nb_input = nb.get("measurement_conditions", {}).get("input_pdb")
            nb_n_conv = nb.get("n_converged")
            nb_note = (
                "native_robust_baseline.json (redock_native_baseline.py 결과) — "
                "입력이 experiment_log.jsonl에서 찾은 cand_006.pdb: "
                "실제 서열=AGCKNIFWKTFNSC (F7→I, T13→N 변이 후보!), native 아님"
            )
        except Exception:
            pass

    # cand_006.pdb 실제 서열 확인
    cand_pdb_path = (
        PYROSETTA_FLOW_DIR
        / "archives"
        / "sst14_mutdock_3000"
        / "iter_03"
        / "cand_006.pdb"
    )
    cand_seq: Optional[str] = None
    cand_is_native = False
    if cand_pdb_path.exists():
        try:
            seqs = extract_chain_sequences(cand_pdb_path)
            # 짧은 chain이 펩타이드
            seqs_sorted = sorted(seqs.items(), key=lambda x: len(x[1]))
            if seqs_sorted:
                cand_seq = seqs_sorted[0][1]
                cand_is_native = cand_seq == NATIVE_SEQ
        except Exception:
            pass

    # 3) +0.54: baseline_cache.json
    bc_path = PYROSETTA_FLOW_DIR / "baseline_cache.json"
    bc_val: Optional[float] = None
    bc_input: Optional[str] = None
    bc_note = ""
    if bc_path.exists():
        try:
            bc = json.loads(bc_path.read_text())
            bc_val = bc.get("ddg")
            bc_input = bc.get("template_pdb")
            bc_note = (
                "baseline_cache.json — nstruct=1 단일 도킹(run_flexpep_refine_pose "
                "기본값이 nstruct=1이었던 시기). "
                "단일 도킹은 noise가 매우 크고 ddg>0은 수렴 실패를 의미"
            )
        except Exception:
            pass

    return {
        "value_minus_41_99": {
            "value": lb_val,
            "n_converged": lb_n_conv,
            "source": "global_selectivity_leaderboard.json::native_ddg_median",
            "input_pdb": "불명확 — 기록 없음",
            "nstruct": "불명확 — 기록 없음",
            "protocol": "flexpep_refine (추정)",
            "problem": (
                "n_converged=2 (cross-val 신뢰 기준 n≥3 미충족), "
                "sd=12.15 REU (매우 높음), 입력PDB·nstruct 기록 없어 재현 불가"
            ),
            "reliability": "LOW — n_converged=2, 조건불명",
        },
        "value_minus_6_05": {
            "value": nb_val,
            "n_converged": nb_n_conv,
            "source": "native_robust_baseline.json (mmgbsa_consensus에서 재인용)",
            "input_pdb": nb_input,
            "cand_006_actual_seq": cand_seq,
            "cand_006_is_native": cand_is_native,
            "nstruct": 5,
            "protocol": "flexpep_refine",
            "problem": (
                f"입력 PDB가 native 복합체가 아님: "
                f"cand_006.pdb 실제 서열={cand_seq} "
                f"(native={NATIVE_SEQ} 불일치 — F7→I, T13→N 변이). "
                f"experiment_log.jsonl에서 native 서열을 검색했으나 "
                f"해당 entry의 실제 PDB는 변이 후보였음. 잘못된 입력."
            ),
            "reliability": "INVALID — 입력이 native가 아닌 변이 후보 PDB",
        },
        "value_plus_0_54": {
            "value": bc_val,
            "source": "baseline_cache.json",
            "input_pdb": bc_input,
            "nstruct": "1 (단일 도킹)",
            "protocol": "flexpep_refine nstruct=1",
            "problem": (
                "nstruct=1 단일 도킹 — ddg>0은 수렴 실패 표시. "
                "입력 PDB는 SSTR2_SST14_complex_boltz_1.pdb(올바름)이지만 "
                "단일 도킹은 noise가 극도로 크고 unreliable"
            ),
            "reliability": "INVALID — nstruct=1 단일 도킹, ddg>0 수렴실패",
        },
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "진짜 native SST-14 복합체로 FlexPepDock robust 재도킹 — nstruct≥10"
        )
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=10,
        help="FlexPepDock nstruct (기본 10)",
    )
    args = parser.parse_args()
    nstruct: int = args.nstruct

    t_total = time.perf_counter()

    print(
        f"[native-v2] native SST-14 robust 재도킹 v2 시작 (nstruct={nstruct})",
        file=sys.stderr,
    )
    print(f"[native-v2] 서열: {NATIVE_SEQ}", file=sys.stderr)
    print(f"[native-v2] 입력 PDB: {NATIVE_PDB}", file=sys.stderr)

    # ① 입력 PDB 존재 확인
    if not NATIVE_PDB.exists():
        print(f"[ERROR] native 복합체 PDB 없음: {NATIVE_PDB}", file=sys.stderr)
        sys.exit(1)

    # ② 입력 PDB 서열 검증
    print("[native-v2] 입력 PDB 서열 검증 중...", file=sys.stderr)
    chain_seqs = extract_chain_sequences(NATIVE_PDB)
    print(
        f"[native-v2] chains: " +
        ", ".join(f"{ch}(len={len(s)})" for ch, s in chain_seqs.items()),
        file=sys.stderr,
    )
    is_valid, peptide_chain_id, found_seq = validate_native_pdb(NATIVE_PDB)

    if not is_valid:
        print(
            f"[ERROR] 입력 PDB에서 native 서열({NATIVE_SEQ}) 미발견. "
            f"발견된 chain 서열: {found_seq}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"[native-v2] native 서열 확인: Chain {peptide_chain_id} = {found_seq}",
        file=sys.stderr,
    )

    # ③ chain 재정렬 (수용체 먼저, 펩타이드 마지막 — FlexPepDock 표준)
    input_pdb = reorder_chains_for_flexpep(NATIVE_PDB, peptide_chain_id, WORK_DIR)
    print(f"[native-v2] FlexPepDock 입력 PDB: {input_pdb}", file=sys.stderr)

    # ④ 3값 불일치 원인 추적
    print("[native-v2] 기존 3값 불일치 원인 추적...", file=sys.stderr)
    discrepancy = trace_three_values()

    # ⑤ robust FlexPepDock (nstruct=10)
    stats = run_native_robust_flexpep(input_pdb, nstruct=nstruct)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    ddg_median: Optional[float] = stats.get("ddg_median")
    n_converged: int = int(stats.get("n_converged") or 0)
    ddg_sd: Optional[float] = stats.get("ddg_sd")
    ddg_min: Optional[float] = stats.get("ddg_min")
    ddg_mean: Optional[float] = stats.get("ddg_mean")
    n_total: int = int(stats.get("n_total") or nstruct)
    flexpep_error: Optional[str] = stats.get("error")

    # ⑥ 신뢰 판정
    if ddg_median is None:
        reliability = "FAILED — 도킹 실패"
        recommendation = "재측정 필요"
    elif n_converged < 3:
        reliability = f"UNCERTAIN — n_converged={n_converged}/{n_total} (기준 n≥3 미충족)"
        recommendation = "nstruct 늘려 재측정 필요"
    elif n_converged >= 5:
        reliability = f"HIGH — n_converged={n_converged}/{n_total}, 진짜 native PDB 입력"
        recommendation = (
            f"native baseline ddg_median={ddg_median:.4f} REU 사용 권고. "
            f"sd={ddg_sd} REU"
        )
    else:
        reliability = f"MODERATE — n_converged={n_converged}/{n_total}, 진짜 native PDB 입력"
        recommendation = (
            f"native baseline ddg_median={ddg_median:.4f} REU 사용 가능. "
            f"sd={ddg_sd} REU"
        )

    # ⑦ 결과 저장
    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "redock_native_baseline_v2.py",
        "description": (
            "진짜 native SST-14 복합체(SSTR2_SST14_complex_boltz_1.pdb)로 "
            f"FlexPepDock robust 재도킹 — nstruct={nstruct}, flexpep_refine, "
            "physical floor(-150) 적용"
        ),
        "measurement_conditions": {
            "protocol": "flexpep_refine",
            "nstruct": nstruct,
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만)",
            "input_pdb": str(NATIVE_PDB),
            "input_pdb_peptide_chain": peptide_chain_id,
            "input_pdb_seq_validated": is_valid,
            "flexpep_input_pdb": str(input_pdb),
            "chain_reorder_applied": (input_pdb != NATIVE_PDB),
            "why_this_input": (
                "SSTR2_SST14_complex_boltz_1.pdb = 진짜 native SST-14 Boltz-2 복합체. "
                "cand_006.pdb(기존 v1 입력)는 AGCKNIFWKTFNSC 변이 후보로 잘못된 입력이었음."
            ),
        },
        "native_seq": NATIVE_SEQ,
        "pdb_chain_sequences": chain_seqs,
        "ddg_median": ddg_median,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "ddg_mean": ddg_mean,
        "n_converged": n_converged,
        "n_total": n_total,
        "n_unphysical": stats.get("n_unphysical", 0),
        "reliability": reliability,
        "recommendation": recommendation,
        "flexpep_error": flexpep_error,
        "elapsed_total_s": elapsed_total,
        # 기존 3값 불일치 원인 규명
        "discrepancy_analysis": {
            "summary": (
                "3값 불일치 원인: "
                "(1) −41.99: n_converged=2 부족·조건불명 → LOW reliability. "
                "(2) −6.05: 입력이 native가 아닌 변이후보 cand_006.pdb(AGCKNIFWKTFNSC) → INVALID. "
                "(3) +0.54: nstruct=1 단일도킹 수렴실패 → INVALID. "
                "이 v2 측정이 진짜 native PDB + nstruct=10으로 가장 신뢰할 만함."
            ),
            "values": discrepancy,
        },
    }

    PYROSETTA_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # ── 결과 요약 출력 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70, file=sys.stderr)
    print("[native-v2] 재도킹 완료 요약", file=sys.stderr)
    print(f"  총 소요시간      : {elapsed_total}s", file=sys.stderr)
    print(f"  입력 PDB         : {NATIVE_PDB}", file=sys.stderr)
    print(f"  펩타이드 chain   : {peptide_chain_id} ({found_seq})", file=sys.stderr)
    print(f"  nstruct          : {nstruct}", file=sys.stderr)
    print(f"  ddg_median       : {ddg_median}", file=sys.stderr)
    print(f"  n_converged      : {n_converged} / {n_total}", file=sys.stderr)
    print(f"  ddg_sd           : {ddg_sd}", file=sys.stderr)
    print(f"  ddg_min          : {ddg_min}", file=sys.stderr)
    print(f"  신뢰도           : {reliability}", file=sys.stderr)
    print(f"  권고             : {recommendation}", file=sys.stderr)
    print(f"  출력 파일        : {OUT_JSON}", file=sys.stderr)
    print("\n  [불일치 원인 요약]", file=sys.stderr)
    print(
        "  −41.99: n_converged=2, 조건불명 → LOW reliability",
        file=sys.stderr,
    )
    print(
        "  −6.05 : 입력 cand_006.pdb 서열=AGCKNIFWKTFNSC (변이 후보!) → INVALID",
        file=sys.stderr,
    )
    print(
        "  +0.54 : nstruct=1 단일도킹 수렴실패 → INVALID",
        file=sys.stderr,
    )
    print("=" * 70, file=sys.stderr)

    # stdout: JSON 요약
    summary = {
        "native_robust_ddg_median": ddg_median,
        "n_converged": n_converged,
        "n_total": n_total,
        "ddg_sd": ddg_sd,
        "ddg_min": ddg_min,
        "reliability": reliability,
        "input_pdb": str(NATIVE_PDB),
        "input_validated": is_valid,
        "out_json": str(OUT_JSON),
        "elapsed_s": elapsed_total,
        "discrepancy_summary": out_data["discrepancy_analysis"]["summary"],
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
