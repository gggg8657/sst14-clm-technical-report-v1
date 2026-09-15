#!/usr/bin/env python3
"""daa_phase1_pilot.py
=======================
D-aa Phase 1 파일럿 — SST-14 native 복합체(7T10) 위에 **MutateResidue로
D-치환 변이체**를 만들어 robust FlexPepDock 재도킹하고, native(전부 L) 대비
결합 변화를 실측한다.

배경 (`_workspace/DAA_FEASIBILITY_AND_DESIGN.md` §Phase 1):
  - anchor_calibration.py(커밋 be0b5d6c)는 **실험적으로 존재하는** D-aa
    약물(octreotide 등)의 실제 SSTR2 복합체 PDB를 직접 로드해 도킹 신뢰도를
    검증했다 (native ddg_median=-35.48 vs octreotide -31.45, 동급 강결합
    확인).
  - 이 스크립트는 그와 별개로 **아직 실험 구조가 없는 gedanken D-변이체**
    (native SST-14 backbone에 특정 위치만 D-aa로 바꾼 가상 서열)를
    다룬다 — 따라서 MutateResidue(소문자=D) 방식이 유일한 접근이며,
    결과는 실험 검증 전 **가설/정성 참고용**이다 (도킹 스코어 신뢰도
    LOW~MODERATE, anchor_calibration이 보여준 "동급 강결합 인정" 그 이상의
    정밀도 주장 금지).

★scaffold 게이트 우회 안내: 발굴 루프(runner.py)는 pharmacophore
  (F7/W8/K9/T10, 1-indexed)·Cys(SS bond) 위치를 변이 금지 스캐폴드로
  보호한다. 이 파일럿은 **의도적으로 그 위치들을 D-치환**하는 실험이므로
  발굴 루프와 무관한 독립 스크립트로 실행하며, 서열은 CLI/코드에 수동
  지정한다(엔진 게이트를 우회하지도, 건드리지도 않음).

대상 서열 (소문자=D-aa, native="AGCKNFFWKTFTSC", 1-indexed):
  - native            : AGCKNFFWKTFTSC              (전부 L, 대조)
  - d_trp8            : AGCKNFFwKTFTSC   (Trp8→D-Trp) ★핵심
      근거: octreotide D-Trp⁴ 임상 성공 선례 + Bo et al. 2022 Cell
      Discovery(DOI 10.1038/s41421-022-00405-2) — D-Trp이 SSTR2 F208과
      소수성 상호작용 강화 직접 서술.
  - d_phe6            : AGCKNfFWKTFTSC   (Phe6→D-Phe, 대조 1)
  - d_phe7            : AGCKNFfWKTFTSC   (Phe7→D-Phe, 대조 2)
  - d_lys9            : AGCKNFFWkTFTSC   (Lys9→D-Lys, 대조 3)
  - d_phe7_trp8       : AGCKNFfwKTFTSC   (Phe7+Trp8 이중 D-치환, 대조 4 —
      octreotide가 D-Phe1+D-Trp4 "이중" 패턴이므로 이중 조합도 참고 관찰)

프로토콜 (native baseline v2/v3, anchor_calibration.py와 동일 재사용):
  - reference PDB: data/somatostatin_receptor/curated/
    SSTR2_SST14_complex_7t10.pdb (실험 구조, chain A=수용체, chain B=SST-14)
  - flexpep_dock.py::prepare_complex_by_mutation()으로 chain B를 목표
    서열로 MutateResidue (소문자=D-ResidueType, 커밋 be0b5d6c 지원)
  - reorder_peptide_last() 로 펩타이드를 마지막 체인으로 정렬
    (FlexPepDockingProtocol 요구사항)
  - run_flexpep_refine_pose() 로 nstruct>=12 독립 refine, ddg_median/sd/
    n_converged 집계 (native baseline과 동일 코드 경로 — 알고리즘
    재구현 없음)

사용법:
  conda run -n bio-tools python scripts/daa_phase1_pilot.py
  conda run -n bio-tools python scripts/daa_phase1_pilot.py --nstruct 12
  conda run -n bio-tools python scripts/daa_phase1_pilot.py --variants native,d_trp8

주의:
  - 엔진/continuous/experiment_log/autopush/surrogate 무접촉.
  - data/ 는 읽기 전용(원본 그대로 로드만).
  - 결과는 정성 참고·가설 취급 — anchor_calibration.py가 확립한 신뢰도
    체계(HIGH: n_converged>=10, MODERATE: >=5, LOW: <5)를 그대로 적용.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"

REFERENCE_PDB = (
    REPO_ROOT
    / "data"
    / "somatostatin_receptor"
    / "curated"
    / "SSTR2_SST14_complex_7t10.pdb"
)
REFERENCE_RECEPTOR_CHAIN = "A"
REFERENCE_PEPTIDE_CHAIN_NUM = 2  # pose 기준 1-indexed chain 번호 (A=1, B=2)
NATIVE_SEQ = "AGCKNFFWKTFTSC"

WORK_DIR = PYROSETTA_FLOW_DIR / "daa_phase1_pilot_work"
DEFAULT_NSTRUCT = 12  # robust 하한(사용자 지정: nstruct>=12)
DDG_PHYSICAL_FLOOR = -150.0

if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))


# ── 변이체 스펙 정의 ──────────────────────────────────────────────────────────


@dataclass
class VariantSpec:
    name: str
    sequence: str  # 소문자=D-aa, 대문자=L-aa
    mutated_positions: List[str]  # 사람이 읽을 수 있는 설명(1-indexed)
    hypothesis: str


def _default_variant_specs() -> Dict[str, VariantSpec]:
    return {
        "native": VariantSpec(
            name="native",
            sequence="AGCKNFFWKTFTSC",
            mutated_positions=[],
            hypothesis="대조군 — 전부 L-aa, 무변이. 동일 프로토콜 기준선.",
        ),
        "d_trp8": VariantSpec(
            name="d_trp8",
            sequence="AGCKNFFwKTFTSC",
            mutated_positions=["Trp8→D-Trp"],
            hypothesis=(
                "★핵심 가설: octreotide D-Trp4 임상 성공 선례 + Bo et al. 2022 "
                "Cell Discovery(D-Trp이 SSTR2 F208과 소수성 상호작용 강화) 근거로 "
                "native 대비 결합 유지 또는 강화 기대."
            ),
        ),
        "d_phe6": VariantSpec(
            name="d_phe6",
            sequence="AGCKNfFWKTFTSC",
            mutated_positions=["Phe6→D-Phe"],
            hypothesis="대조 1 — pharmacophore 인접 위치 단일 D-치환, 비교용.",
        ),
        "d_phe7": VariantSpec(
            name="d_phe7",
            sequence="AGCKNFfWKTFTSC",
            mutated_positions=["Phe7→D-Phe"],
            hypothesis="대조 2 — pharmacophore 인접 위치 단일 D-치환, 비교용.",
        ),
        "d_lys9": VariantSpec(
            name="d_lys9",
            sequence="AGCKNFFWkTFTSC",
            mutated_positions=["Lys9→D-Lys"],
            hypothesis="대조 3 — pharmacophore 인접 위치 단일 D-치환, 비교용.",
        ),
        "d_phe7_trp8": VariantSpec(
            name="d_phe7_trp8",
            sequence="AGCKNFfwKTFTSC",
            mutated_positions=["Phe7→D-Phe", "Trp8→D-Trp"],
            hypothesis=(
                "대조 4 — octreotide가 D-Phe1+D-Trp4 이중 D-치환 패턴이므로, "
                "이중 조합의 참고 관찰(단일 D-Trp8과 비교)."
            ),
        ),
    }


# ── MutateResidue 기반 D-변이체 robust 재도킹 ─────────────────────────────────


def run_variant_robust_flexpep(
    spec: VariantSpec, nstruct: int, work_dir: Path
) -> Dict:
    """native 복합체(7T10) 위에 MutateResidue로 spec.sequence를 적용하고
    nstruct회 robust FlexPepDock refine 실행. native baseline/anchor_calibration과
    동일 코드 경로(prepare_complex_by_mutation → reorder_peptide_last →
    run_flexpep_refine_pose)를 재사용 — 새 도킹 알고리즘 재구현 없음.
    """
    from flexpep_dock import (  # type: ignore
        init_pyrosetta,
        prepare_complex_by_mutation,
        reorder_peptide_last,
        run_flexpep_refine_pose,
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = str(work_dir / "refined.pdb")

    print(
        f"  [FlexPepDock] {spec.name}: nstruct={nstruct} 시작 "
        f"(target_seq={spec.sequence}, mutations={spec.mutated_positions})",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        init_pyrosetta()
        pose, resolved_chain = prepare_complex_by_mutation(
            str(REFERENCE_PDB),
            spec.sequence,
            REFERENCE_PEPTIDE_CHAIN_NUM,
        )
        pose = reorder_peptide_last(pose, resolved_chain)
        _refined_pose, stats = run_flexpep_refine_pose(
            pose, output_pdb, nstruct=nstruct
        )
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            f"  [FlexPepDock] {spec.name} 완료 {elapsed}s | "
            f"ddg_median={stats.get('ddg_median')} "
            f"n_converged={stats.get('n_converged')}/{nstruct}",
            file=sys.stderr,
        )
        stats["elapsed_s"] = elapsed
        return stats
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [FlexPepDock] {spec.name} 실패 {elapsed}s: {exc}", file=sys.stderr)
        return {
            "ddg_median": None, "ddg_sd": None, "ddg_mean": None,
            "ddg_min": None, "n_converged": 0, "n_total": nstruct,
            "converged": False, "elapsed_s": elapsed,
            "error": str(exc),
        }


# ── 신뢰도 판정 (anchor_calibration.py/native baseline v2/v3와 동일 기준) ─────


def _reliability_verdict(
    ddg_median: Optional[float], n_converged: int, n_total: int
) -> str:
    if ddg_median is None:
        return "FAILED"
    if n_converged < 3:
        return f"UNCERTAIN (n_converged={n_converged}/{n_total}, 기준 n>=3 미충족)"
    if n_converged >= 10:
        return f"HIGH (n_converged={n_converged}/{n_total})"
    if n_converged >= 5:
        return f"MODERATE (n_converged={n_converged}/{n_total})"
    return f"LOW (n_converged={n_converged}/{n_total})"


# ── 메인 처리 (변이체 1개) ─────────────────────────────────────────────────────


def process_variant(spec: VariantSpec, nstruct: int, work_root: Path) -> Dict:
    result: Dict = {
        "name": spec.name,
        "sequence": spec.sequence,
        "mutated_positions": spec.mutated_positions,
        "hypothesis": spec.hypothesis,
    }

    if not REFERENCE_PDB.exists():
        result.update(
            {
                "status": "REFERENCE_PDB_NOT_FOUND",
                "message": f"참조 PDB 없음: {REFERENCE_PDB}",
                "ddg_median": None,
                "n_converged": 0,
                "n_total": nstruct,
                "reliability": "NO_DATA",
            }
        )
        return result

    work_dir = work_root / spec.name
    stats = run_variant_robust_flexpep(spec, nstruct, work_dir)

    ddg_median = stats.get("ddg_median")
    n_converged = int(stats.get("n_converged") or 0)
    n_total = int(stats.get("n_total") or nstruct)
    reliability = _reliability_verdict(ddg_median, n_converged, n_total)

    if stats.get("error") is not None:
        status = "FLEXPEP_ERROR"
    elif ddg_median is not None:
        status = "DOCKED"
    else:
        status = "DOCKED_NOT_CONVERGED"  # 실행은 성공했으나 0/n_total만 ddg<0 수렴

    result.update(
        {
            "status": status,
            "reference_pdb": str(REFERENCE_PDB),
            "ddg_median": ddg_median,
            "ddg_mean": stats.get("ddg_mean"),
            "ddg_sd": stats.get("ddg_sd"),
            "ddg_min": stats.get("ddg_min"),
            "n_converged": n_converged,
            "n_total": n_total,
            "n_unphysical": stats.get("n_unphysical", 0),
            "convergence_rate": round(n_converged / n_total, 4) if n_total else 0.0,
            "elapsed_s": stats.get("elapsed_s"),
            "disulfide_intact": stats.get("disulfide_intact"),
            "sg_sg_distance": stats.get("sg_sg_distance"),
            "flexpep_error": stats.get("error"),
            "reliability": reliability,
        }
    )
    return result


# ── native 대비 Δ 판정 ────────────────────────────────────────────────────────


def compare_to_native(results: Dict[str, Dict]) -> Dict:
    native = results.get("native")
    if not native or native.get("ddg_median") is None:
        return {
            "comparable": False,
            "reason": "native 기준값(ddg_median)이 없어 비교 불가.",
        }

    native_ddg = native["ddg_median"]
    comparisons: Dict[str, Dict] = {}

    for name, r in results.items():
        if name == "native":
            continue
        if r.get("ddg_median") is None:
            comparisons[name] = {
                "delta_vs_native": None,
                "verdict": "NO_DATA",
            }
            continue
        delta = round(r["ddg_median"] - native_ddg, 4)  # 음수 = 변이체가 더 강한 결합
        # 정성 임계치: |delta| < 3 REU는 "유지"로 취급(anchor_calibration
        # native(-35.48) vs octreotide(-31.45) 차이 ~4 REU가 "동급 강결합"으로
        # 판정된 선례를 참고해, 이보다 좁은 3 REU를 "유지" 문턱으로 보수적 채택)
        if delta <= -3.0:
            verdict = "강화 (STRENGTHENED)"
        elif delta >= 3.0:
            verdict = "약화 (WEAKENED)"
        else:
            verdict = "유지 (MAINTAINED, native와 동급)"
        comparisons[name] = {
            "native_ddg_median": native_ddg,
            "variant_ddg_median": r["ddg_median"],
            "delta_vs_native": delta,
            "verdict": verdict,
            "both_reliable": (
                native.get("reliability", "").startswith(("HIGH", "MODERATE"))
                and r.get("reliability", "").startswith(("HIGH", "MODERATE"))
            ),
        }

    return {
        "comparable": True,
        "native_ddg_median": native_ddg,
        "native_reliability": native.get("reliability"),
        "comparisons": comparisons,
        "caveat": (
            "MutateResidue 기반 D-치환은 native backbone(실험 구조) 위에 "
            "사이드체인만 바꾼 것 — D-aa는 backbone 카이랄성과 무관하게 동일 "
            "backbone 좌표를 사용하므로 실제 D-치환 펩타이드의 진짜 우호적 "
            "backbone 재배열(예: octreotide의 8-mer 고리 형태)은 반영되지 "
            "않는다. anchor_calibration.py 결과(D-aa 실험 구조 도킹은 native급 "
            "신뢰)와 달리, 이 결과는 '가상 D-변이체'에 대한 1차 스코어링 "
            "신호일 뿐 실험 검증 전 가설 단계."
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D-aa Phase 1 파일럿 — native SST-14 복합체 위에 MutateResidue로 "
            "D-치환 변이체(Trp8 핵심 + F6/F7/K9 대조)를 만들어 robust "
            "FlexPepDock(nstruct>=12)으로 native 대비 결합 변화 실측"
        )
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=DEFAULT_NSTRUCT,
        help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT}, robust 하한)",
    )
    parser.add_argument(
        "--variants",
        default="native,d_trp8,d_phe6,d_phe7,d_lys9,d_phe7_trp8",
        help="쉼표구분 변이체 목록",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="출력 JSON 경로 (기본: runs/pyrosetta_flow/daa_phase1_pilot_<timestamp>.json)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    nstruct: int = args.nstruct
    if nstruct < 12:
        print(
            f"WARNING: nstruct={nstruct} < 12 — robust 하한(사용자 지정) 미충족.",
            file=sys.stderr,
        )

    requested = [s.strip() for s in args.variants.split(",") if s.strip()]
    specs = _default_variant_specs()

    unknown = [v for v in requested if v not in specs]
    if unknown:
        print(f"[ERROR] 알 수 없는 변이체: {unknown}. 지원: {sorted(specs)}", file=sys.stderr)
        sys.exit(1)

    if not REFERENCE_PDB.exists():
        print(f"[ERROR] 참조 PDB 없음: {REFERENCE_PDB}", file=sys.stderr)
        sys.exit(1)

    t_total = time.perf_counter()
    print(
        f"[daa-phase1-pilot] 변이체={requested} nstruct={nstruct} 시작",
        file=sys.stderr,
    )
    print(f"[daa-phase1-pilot] 참조 PDB={REFERENCE_PDB} (chain A=수용체, chain B=SST-14)", file=sys.stderr)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for name in requested:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[{name}] 처리 시작 (seq={specs[name].sequence})", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        results[name] = process_variant(specs[name], nstruct, WORK_DIR)

    native_comparison = compare_to_native(results)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "daa_phase1_pilot.py",
        "description": (
            "D-aa Phase 1 파일럿: native SST-14 복합체(7T10) 위에 MutateResidue로 "
            f"D-치환 변이체(Trp8 핵심 + 대조군)를 만들어 robust FlexPepDock "
            f"(nstruct={nstruct})으로 native 대비 결합 변화 실측"
        ),
        "measurement_conditions": {
            "protocol": (
                "prepare_complex_by_mutation → reorder_peptide_last → "
                "run_flexpep_refine_pose (AG_src/scripts/flexpep_dock.py 재사용, "
                "native baseline v2/v3·anchor_calibration.py와 동일 코드 경로)"
            ),
            "nstruct": nstruct,
            "reference_pdb": str(REFERENCE_PDB),
            "reference_source": "실험 구조 7T10 (Robertson 2022 NSMB), chain A=수용체 chain B=SST-14",
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만)",
            "mutation_mode": (
                "MutateResidue(소문자=D-ResidueType) — native backbone 좌표 위에 "
                "사이드체인만 D-aa로 치환 (실험 D-구조 직접 로드가 아님, "
                "가상 변이체 가설 스코어링)"
            ),
            "scaffold_gate_note": (
                "발굴 루프(runner.py)의 pharmacophore/Cys 변이 금지 스캐폴드 게이트는 "
                "이 파일럿에 적용되지 않음 — 독립 스크립트에서 수동 서열 지정으로 "
                "의도적 우회 (엔진/게이트 코드 자체는 무수정)"
            ),
        },
        "native_seq": NATIVE_SEQ,
        "variants": results,
        "native_comparison": native_comparison,
        "elapsed_total_s": elapsed_total,
        "references": [
            "_workspace/DAA_FEASIBILITY_AND_DESIGN.md §Phase 1",
            "runs/pyrosetta_flow/anchor_calibration_20260701T061533Z.json (D-aa 도킹 신뢰도 선행 검증)",
        ],
        "caveat_top_level": (
            "이 파일럿의 도킹 결과는 정성 참고·가설 취급 원칙을 따른다. "
            "MutateResidue는 native 실험 backbone 좌표를 그대로 쓰므로, D-치환이 "
            "유발할 수 있는 실제 backbone 재배열(고리 형태 변화 등)은 반영되지 "
            "않는다. '결합 유지/강화/약화' 판정은 1차 스코어링 신호이며 실험 "
            "검증(예: 합성 후 결합능 측정)이 있어야 확정된다."
        ),
    }

    if args.out_json:
        out_json_path = Path(args.out_json)
    else:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_json_path = PYROSETTA_FLOW_DIR / f"daa_phase1_pilot_{ts}.json"

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    # ── 결과 요약 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 72, file=sys.stderr)
    print("[daa-phase1-pilot] 완료 요약", file=sys.stderr)
    print(f"  총 소요시간   : {elapsed_total}s", file=sys.stderr)
    for name, r in results.items():
        if r.get("ddg_median") is not None:
            print(
                f"  {name:16s}: seq={r['sequence']:16s} ddg_median={r['ddg_median']} "
                f"n_converged={r['n_converged']}/{r['n_total']} "
                f"sd={r['ddg_sd']} reliability={r['reliability']}",
                file=sys.stderr,
            )
        else:
            reason = r.get("message") or r.get("flexpep_error") or (
                f"n_converged={r.get('n_converged')}/{r.get('n_total')} "
                "(수렴 기준 ddg<0 미충족 — 표본 확대 필요)"
            )
            print(f"  {name:16s}: {r.get('status')} — {reason}", file=sys.stderr)
    if native_comparison.get("comparable"):
        print("\n  [native 대비 판정]", file=sys.stderr)
        for name, c in native_comparison["comparisons"].items():
            print(
                f"    {name:16s}: Δ={c.get('delta_vs_native')} -> {c.get('verdict')}",
                file=sys.stderr,
            )
    print(f"  출력 파일     : {out_json_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    summary = {
        "out_json": str(out_json_path),
        "elapsed_s": elapsed_total,
        "variant_ddg": {name: r.get("ddg_median") for name, r in results.items()},
        "native_comparison_verdicts": {
            name: c.get("verdict")
            for name, c in native_comparison.get("comparisons", {}).items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
