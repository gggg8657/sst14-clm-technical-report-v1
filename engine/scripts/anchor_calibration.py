#!/usr/bin/env python3
"""anchor_calibration.py
========================
D-aa 앵커 캘리브레이션 — SSTR2에서 임상적으로 검증된 D-aa 결합제
(octreotide, lanreotide)와 native SST-14를 **동일 robust FlexPepDock
프로토콜**로 도킹해, 우리 도킹 스코어가 실험적 결합 순위(Ki/pKi)를
정성적으로 재현하는지 검증한다.

배경 (2026-07-01, `_workspace/DAA_FEASIBILITY_AND_DESIGN.md` +
`_workspace/02_researcher_octreotide_anchor_calibration.md` 참조):
  - flexpep_dock.py에 D-aa 최소 지원(소문자→D-ResidueType)이 이제 추가됨
    (AG_src/scripts/flexpep_dock.py::prepare_complex_by_mutation).
  - 그러나 D-aa "도킹이 된다"는 것과 "도킹 스코어가 신뢰할 만하다"는 것은
    별개 질문 — 이 스크립트는 후자를 검증하기 위한 것.
  - octreotide(8-mer, D-Phe1/D-Trp4, Thr-ol C말단)와 lanreotide(8-mer,
    D-2Nal1/D-Trp4, Thr-NH2 C말단)는 SST-14(14-mer, 전부 L)와 **골격
    자체가 다르다** → MutateResidue(SST-14 backbone 위에 얹기) 방식은
    부적합. 반드시 **각 리간드의 실제 SSTR2 복합체 구조(PDB)를 직접
    로드**해서 FlexPepDock refine 해야 한다.

★robust 필수(사용자 강조): 리간드별 nstruct≥10 반복 → 수렴(ddg<0) 값의
  median + sd + n_converged. native baseline v2/v3
  (`scripts/redock_native_baseline_v2.py`, `_v3.py`)와 동일 방식으로
  `flexpep_dock.run_flexpep_refine` 을 그대로 재사용한다(중복 구현 금지).

지원 리간드 (2026-07-01 기준 로컬 데이터 상태):
  - native (SST-14, 14aa, L-only) — data/somatostatin_receptor/curated/
    SSTR2_SST14_complex_7t10.pdb (2체인, chain A=수용체 chain B=펩타이드,
    확인됨: chain B=AGCKNFFWKTFTSC).
  - octreotide (8aa, D-Phe1/D-Trp4) — data/somatostatin_receptor/
    SSTR2_octreotide_7t11.pdb (6체인 원본 cryo-EM 파일: A/B/C=G-protein
    서브유닛, S=scFv16, R=수용체, P=펩타이드). 이 스크립트가 R+P만
    추출해서 사용한다.
  - lanreotide — **로컬에 구조 없음** (researcher 조사 결과, 2026-07-01
    시점 lanreotide 단독 SSTR2 cryo-EM 구조를 확보하지 못함). 구조가
    없으면 이 리간드는 "구조 필요" 안내만 출력하고 **스킵**한다
    (가짜 결과 생성 금지).

사용법:
  # 사용 가능한 리간드만 자동 검출해서 전부 도킹
  conda run -n bio-tools python scripts/anchor_calibration.py

  # nstruct 조정 (기본 10, robust 하한)
  conda run -n bio-tools python scripts/anchor_calibration.py --nstruct 15

  # 특정 리간드만
  conda run -n bio-tools python scripts/anchor_calibration.py --ligands native,octreotide

  # lanreotide 구조를 나중에 확보하면 CLI로 경로 주입 가능
  conda run -n bio-tools python scripts/anchor_calibration.py \\
      --lanreotide-pdb /path/to/lanreotide_sstr2_complex.pdb \\
      --lanreotide-peptide-chain P --lanreotide-receptor-chain R

주의:
  - 엔진/autopush 재시작 금지. surrogate 7파일 금지. experiment_log 등
    실데이터 미접촉. 출력은 전용 파일(runs/pyrosetta_flow/
    anchor_calibration_<timestamp>.json)에만 기록.
  - data/ 는 읽기 전용으로만 사용(다중체인 원본은 임시 작업 디렉토리
    runs/pyrosetta_flow/anchor_calibration_work/ 에 추출본을 쓴다).
"""

from __future__ import annotations

import argparse
import datetime
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"
DATA_DIR = REPO_ROOT / "data" / "somatostatin_receptor"

WORK_DIR = PYROSETTA_FLOW_DIR / "anchor_calibration_work"
DEFAULT_NSTRUCT = 10  # robust 하한 (사용자 지정: nstruct>=10)
DDG_PHYSICAL_FLOOR = -150.0

if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))


# ── 리간드 스펙 정의 ──────────────────────────────────────────────────────────
#
# 각 리간드는 자체 실험 복합체 PDB를 직접 로드한다(MutateResidue 부적합 —
# octreotide/lanreotide는 8-mer로 SST-14(14-mer)와 골격이 다름).
# receptor_chain / peptide_chain: 원본 PDB의 체인 ID(1-letter). 다중 체인
# 원본(예: 7T11의 6체인 cryo-EM 파일)은 extract_receptor_peptide_chains()가
# 해당 두 체인만 뽑아 FlexPepDock 표준 순서(수용체 먼저, 펩타이드 마지막)로
# 재구성한다.


@dataclass
class LigandSpec:
    name: str
    pdb_path: Optional[Path]
    receptor_chain: Optional[str]
    peptide_chain: Optional[str]
    expected_seq: Optional[str]  # 검증용 (알려진 경우만)
    ki_nM: Optional[float]  # 실험 Ki(nM), 낮을수록 강한 결합 (파라미터화)
    ki_source: str
    notes: str = ""


def _default_ligand_specs() -> Dict[str, LigandSpec]:
    return {
        "native": LigandSpec(
            name="native",
            pdb_path=DATA_DIR / "curated" / "SSTR2_SST14_complex_7t10.pdb",
            receptor_chain="A",
            peptide_chain="B",
            expected_seq="AGCKNFFWKTFTSC",
            # IUPHAR/BPS SST2, pKi 8.9-10.5 → 중심값 근사(정밀 단일값 아님,
            # §검증 필요로 문서화됨). 여기서는 순위 비교용 참고치.
            ki_nM=0.2,
            ki_source=(
                "IUPHAR/BPS Guide to Pharmacology SST2 (pKi 8.9-10.5 메타집계 "
                "중심값 근사, 개별 1차 논문 미확정) — "
                "_workspace/02_researcher_octreotide_anchor_calibration.md §3.4"
            ),
            notes="native SST-14, 14aa, 전부 L-aa. PDB 7T10 (Robertson 2022 NSMB).",
        ),
        "octreotide": LigandSpec(
            name="octreotide",
            pdb_path=DATA_DIR / "SSTR2_octreotide_7t11.pdb",
            receptor_chain="R",
            peptide_chain="P",
            expected_seq=None,  # non-standard 잔기(DPN/DTR/THO) 포함, 1-letter 검증 skip
            ki_nM=0.6,
            ki_source=(
                "WebSearch 별도 IC50 수치(1차 출처 미확정, Patel & Srikant 1994 "
                "유력 후보이나 원문 미대조) — 동 문서 §3.4, §검증 필요 항목 2"
            ),
            notes=(
                "8aa 고리, D-Phe1/D-Trp4, C말단 Thr-ol(THO, 비표준 잔기). "
                "PDB 7T11 (Robertson 2022 NSMB, 동일 논문 native와 페어)."
            ),
        ),
        "lanreotide": LigandSpec(
            name="lanreotide",
            pdb_path=None,  # 2026-07-01 시점 로컬 미보유 — researcher 재확인 필요
            receptor_chain=None,
            peptide_chain=None,
            expected_seq=None,
            ki_nM=0.8,
            ki_source=(
                "WebSearch 별도 IC50 수치(1차 출처 미확정) — 동 문서 §3.4, "
                "§검증 필요 항목 2,3"
            ),
            notes=(
                "8aa 고리, D-2Nal1/D-Trp4, C말단 Thr-NH2. 실험 SSTR2 복합체 "
                "PDB를 이번 조사에서 확보하지 못함(§검증 필요 항목 3) — "
                "--lanreotide-pdb로 확보되는 대로 주입 가능."
            ),
        ),
    }


# ── 다중 체인 PDB에서 수용체+펩타이드만 추출 ─────────────────────────────────

def extract_receptor_peptide_chains(
    src_pdb: Path,
    receptor_chain: str,
    peptide_chain: str,
    out_pdb: Path,
) -> Path:
    """원본 PDB(다중 체인 가능)에서 수용체·펩타이드 체인만 뽑아 FlexPepDock
    표준 순서(수용체=chain A, 펩타이드=chain B, 펩타이드가 마지막)로
    재구성한 새 PDB를 out_pdb에 쓰고 그 경로를 반환한다.

    G-protein 서브유닛/scFv 등 무관한 체인(예: 7T11의 A/B/C/S)은 제외한다
    — FlexPepDockingProtocol은 정확히 수용체+펩타이드 2체인 구조를
    기대하므로, 부가 체인을 남기면 잘못된 인터페이스로 도킹될 위험이 있다.
    """
    chain_blocks: Dict[str, List[str]] = {}
    header_lines: List[str] = []

    with open(src_pdb) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                cid = line[21]
                if cid in (receptor_chain, peptide_chain):
                    chain_blocks.setdefault(cid, []).append(line)
            elif not line.startswith(("TER", "END", "MODEL", "ENDMDL")):
                header_lines.append(line)

    if receptor_chain not in chain_blocks:
        raise ValueError(
            f"Receptor chain '{receptor_chain}' not found in {src_pdb}"
        )
    if peptide_chain not in chain_blocks:
        raise ValueError(
            f"Peptide chain '{peptide_chain}' not found in {src_pdb}"
        )

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    new_lines = list(header_lines)
    # 수용체 먼저(new chain A), 펩타이드 마지막(new chain B) — FlexPepDock 표준
    for new_cid, old_cid in (("A", receptor_chain), ("B", peptide_chain)):
        for atom_line in chain_blocks[old_cid]:
            new_lines.append(atom_line[:21] + new_cid + atom_line[22:])
        new_lines.append("TER\n")
    new_lines.append("END\n")

    with open(out_pdb, "w") as fh:
        fh.writelines(new_lines)

    print(
        f"  [extract] {src_pdb.name}: chain {receptor_chain}(receptor)+"
        f"{peptide_chain}(peptide) -> {out_pdb} "
        f"(receptor={len(chain_blocks[receptor_chain])} atoms, "
        f"peptide={len(chain_blocks[peptide_chain])} atoms)",
        file=sys.stderr,
    )
    return out_pdb


# ── robust FlexPepDock 재도킹 (native_robust_baseline_v2/v3.py 프로토콜 재사용) ──

def run_ligand_robust_flexpep(input_pdb: Path, nstruct: int, work_dir: Path) -> Dict:
    """flexpep_dock.run_flexpep_refine 을 그대로 재사용해 nstruct회 robust
    재도킹하고 median/sd/n_converged 통계를 반환한다.

    새 도킹 알고리즘을 재구현하지 않는다 — native baseline v2/v3와 동일한
    코드 경로(run_flexpep_refine)를 사용해야 native/octreotide/lanreotide가
    서로 비교 가능한 동일 프로토콜 하에 있다고 주장할 수 있다.
    """
    from flexpep_dock import run_flexpep_refine, init_pyrosetta  # type: ignore

    work_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = str(work_dir / "refined.pdb")

    print(
        f"  [FlexPepDock] nstruct={nstruct} 시작: {input_pdb}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        init_pyrosetta()
        _pose, stats = run_flexpep_refine(str(input_pdb), output_pdb, nstruct=nstruct)
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            f"  [FlexPepDock] 완료 {elapsed}s | "
            f"ddg_median={stats.get('ddg_median')} "
            f"n_converged={stats.get('n_converged')}/{nstruct}",
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


# ── 신뢰도 판정 (native_robust_baseline_v2/v3.py와 동일 기준) ─────────────────

def _reliability_verdict(ddg_median: Optional[float], n_converged: int, n_total: int) -> Tuple[str, str]:
    if ddg_median is None:
        return "FAILED", "도킹 실패 — 재측정 필요"
    if n_converged < 3:
        return (
            f"UNCERTAIN (n_converged={n_converged}/{n_total}, 기준 n>=3 미충족)",
            "nstruct 늘려 재측정 필요",
        )
    if n_converged >= 10:
        return (
            f"HIGH (n_converged={n_converged}/{n_total})",
            f"ddg_median={ddg_median:.4f} REU 사용 권고",
        )
    if n_converged >= 5:
        return (
            f"MODERATE (n_converged={n_converged}/{n_total})",
            f"ddg_median={ddg_median:.4f} REU 사용 가능, 편차 유의",
        )
    return (
        f"LOW (n_converged={n_converged}/{n_total})",
        "표본 부족 — 결과는 참고용",
    )


# ── 메인 도킹 파이프라인 (리간드 1개) ─────────────────────────────────────────

def process_ligand(
    spec: LigandSpec, nstruct: int, work_root: Path
) -> Dict:
    """리간드 1개에 대한 전체 처리: 구조 확인 -> (다중체인이면) 추출 ->
    robust FlexPepDock -> 통계 반환. 구조가 없으면 명시적으로 스킵."""

    result: Dict = {
        "name": spec.name,
        "notes": spec.notes,
        "ki_nM": spec.ki_nM,
        "ki_source": spec.ki_source,
    }

    if spec.pdb_path is None or spec.receptor_chain is None or spec.peptide_chain is None:
        result.update(
            {
                "status": "SKIPPED_NO_STRUCTURE",
                "message": (
                    f"'{spec.name}' 리간드의 SSTR2 실험 복합체 PDB가 없습니다. "
                    f"MutateResidue(SST-14 backbone) 방식은 골격이 다른 리간드에 "
                    f"부적합하므로 사용하지 않습니다. "
                    f"구조를 확보하면 --{spec.name}-pdb / "
                    f"--{spec.name}-receptor-chain / --{spec.name}-peptide-chain "
                    f"CLI 파라미터로 주입하십시오."
                ),
                "ddg_median": None,
                "ddg_sd": None,
                "n_converged": 0,
                "n_total": nstruct,
                "reliability": "NO_DATA",
            }
        )
        print(f"[{spec.name}] SKIP — {result['message']}", file=sys.stderr)
        return result

    if not spec.pdb_path.exists():
        result.update(
            {
                "status": "SKIPPED_PDB_NOT_FOUND",
                "message": f"PDB 경로가 존재하지 않음: {spec.pdb_path}",
                "ddg_median": None,
                "ddg_sd": None,
                "n_converged": 0,
                "n_total": nstruct,
                "reliability": "NO_DATA",
            }
        )
        print(f"[{spec.name}] SKIP — {result['message']}", file=sys.stderr)
        return result

    work_dir = work_root / spec.name
    extracted_pdb = work_dir / f"{spec.name}_receptor_peptide.pdb"

    try:
        input_pdb = extract_receptor_peptide_chains(
            spec.pdb_path, spec.receptor_chain, spec.peptide_chain, extracted_pdb
        )
    except Exception as exc:
        result.update(
            {
                "status": "EXTRACT_FAILED",
                "message": str(exc),
                "ddg_median": None,
                "ddg_sd": None,
                "n_converged": 0,
                "n_total": nstruct,
                "reliability": "NO_DATA",
            }
        )
        print(f"[{spec.name}] EXTRACT FAILED — {exc}", file=sys.stderr)
        return result

    stats = run_ligand_robust_flexpep(input_pdb, nstruct, work_dir)

    ddg_median = stats.get("ddg_median")
    n_converged = int(stats.get("n_converged") or 0)
    n_total = int(stats.get("n_total") or nstruct)
    reliability, recommendation = _reliability_verdict(ddg_median, n_converged, n_total)

    result.update(
        {
            "status": "DOCKED",
            "source_pdb": str(spec.pdb_path),
            "receptor_chain": spec.receptor_chain,
            "peptide_chain": spec.peptide_chain,
            "extracted_pdb": str(input_pdb),
            "expected_seq": spec.expected_seq,
            "ddg_median": ddg_median,
            "ddg_mean": stats.get("ddg_mean"),
            "ddg_sd": stats.get("ddg_sd"),
            "ddg_min": stats.get("ddg_min"),
            "n_converged": n_converged,
            "n_total": n_total,
            "n_unphysical": stats.get("n_unphysical", 0),
            "convergence_rate": round(n_converged / n_total, 4) if n_total else 0.0,
            "elapsed_s": stats.get("elapsed_s"),
            "flexpep_error": stats.get("error"),
            "reliability": reliability,
            "recommendation": recommendation,
        }
    )
    return result


# ── 순위 비교 (도킹 ddG vs 실험 Ki) ───────────────────────────────────────────

def compare_ranking(results: Dict[str, Dict]) -> Dict:
    """도킹된(ddg_median 有) 리간드들의 ddG 순위와 실험 Ki 순위를 비교.

    ddG는 더 음수일수록 강한 결합(우리 관례), Ki는 더 낮을수록 강한 결합
    (약리학 관례) — 방향을 맞춰서 순위를 비교한다.
    """
    docked = {
        name: r
        for name, r in results.items()
        if r.get("status") == "DOCKED" and r.get("ddg_median") is not None
    }

    if len(docked) < 2:
        return {
            "comparable": False,
            "reason": (
                f"도킹 성공(ddg_median 有) 리간드가 {len(docked)}개뿐 — "
                "순위 비교에는 최소 2개 필요."
            ),
            "n_dockable": len(docked),
        }

    # ddG 오름차순(더 음수 = 더 강한 결합 = 순위 1위)
    ddg_rank = sorted(docked.items(), key=lambda kv: kv[1]["ddg_median"])
    # Ki 오름차순(더 낮은 Ki = 더 강한 결합 = 순위 1위); Ki 없는 항목은 제외
    ki_available = {
        name: r for name, r in docked.items() if r.get("ki_nM") is not None
    }
    ki_rank = sorted(ki_available.items(), key=lambda kv: kv[1]["ki_nM"])

    ddg_order = [name for name, _ in ddg_rank]
    ki_order = [name for name, _ in ki_rank]

    # 두 순위 리스트를 공통 항목만으로 비교
    common = [n for n in ddg_order if n in ki_order]
    ddg_common_rank = {n: i for i, n in enumerate(n for n in ddg_order if n in common)}
    ki_common_rank = {n: i for i, n in enumerate(n for n in ki_order if n in common)}

    rank_matches = all(
        ddg_common_rank[n] == ki_common_rank[n] for n in common
    ) if common else False

    return {
        "comparable": True,
        "n_dockable": len(docked),
        "ddg_rank_order": ddg_order,
        "ddg_values": {n: r["ddg_median"] for n, r in ddg_rank},
        "ki_rank_order": ki_order,
        "ki_values_nM": {n: r["ki_nM"] for n, r in ki_rank},
        "ki_sources": {n: r["ki_source"] for n, r in ki_rank},
        "exact_rank_match": rank_matches,
        "verdict": (
            "도킹 ddG 순위가 실험 Ki 순위와 정확히 일치"
            if rank_matches
            else (
                "도킹 ddG 순위가 실험 Ki 순위와 불일치 — "
                "단, pKi 범위가 문헌 간 크게 겹치므로(researcher 조사, "
                "_workspace/02_researcher_octreotide_anchor_calibration.md §3.4: "
                "SST-14 pKi 8.9-10.5, Octreotide 8.7-9.9, Lanreotide 8.7-9.6) "
                "'정밀 순위 재현 실패'가 곧 '도킹이 틀렸다'를 의미하지 않음. "
                "이 셋은 실험적으로도 sub-nM~저nM 범위의 동일 등급 강결합체."
            )
        ),
        "caveat": (
            "Ki 값은 개별 1차 논문 출처가 확정되지 않은 참고치("
            "researcher §검증 필요 항목 2) — 통계적 유의성 주장 금지, "
            "정성적 순위 참고용으로만 사용."
        ),
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D-aa 앵커 캘리브레이션 — native/octreotide/lanreotide를 동일 "
            "robust FlexPepDock 프로토콜(nstruct>=10)로 도킹해 실험 결합 "
            "순위(Ki) 재현 여부를 검증"
        )
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=DEFAULT_NSTRUCT,
        help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT}, robust 하한 준수 위해 10 미만 비권장)",
    )
    parser.add_argument(
        "--ligands",
        default="native,octreotide,lanreotide",
        help="쉼표구분 리간드 목록 (기본: native,octreotide,lanreotide)",
    )
    # 구조가 아직 없는 리간드(현재 lanreotide)를 나중에 채울 수 있도록
    # 범용 오버라이드 제공. 향후 신규 리간드 확장도 동일 패턴으로 가능.
    for lig in ("native", "octreotide", "lanreotide"):
        parser.add_argument(f"--{lig}-pdb", default=None, help=f"{lig} 복합체 PDB 경로 오버라이드")
        parser.add_argument(f"--{lig}-receptor-chain", default=None, help=f"{lig} PDB의 수용체 체인 ID")
        parser.add_argument(f"--{lig}-peptide-chain", default=None, help=f"{lig} PDB의 펩타이드 체인 ID")
        parser.add_argument(f"--{lig}-ki-nm", type=float, default=None, help=f"{lig} 실험 Ki(nM) 오버라이드")
    parser.add_argument(
        "--out-json",
        default=None,
        help="출력 JSON 경로 (기본: runs/pyrosetta_flow/anchor_calibration_<UTC timestamp>.json)",
    )
    return parser


def _apply_cli_overrides(specs: Dict[str, LigandSpec], args: argparse.Namespace) -> None:
    for lig in ("native", "octreotide", "lanreotide"):
        pdb_override = getattr(args, f"{lig}_pdb")
        if pdb_override:
            specs[lig].pdb_path = Path(pdb_override)
        rc = getattr(args, f"{lig}_receptor_chain")
        if rc:
            specs[lig].receptor_chain = rc
        pc = getattr(args, f"{lig}_peptide_chain")
        if pc:
            specs[lig].peptide_chain = pc
        ki = getattr(args, f"{lig}_ki_nm")
        if ki is not None:
            specs[lig].ki_nM = ki


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    nstruct: int = args.nstruct
    if nstruct < 10:
        print(
            f"WARNING: nstruct={nstruct} < 10 — robust 하한(사용자 지정) 미충족. "
            "결과의 n_converged가 낮으면 UNCERTAIN/LOW로 표시됩니다.",
            file=sys.stderr,
        )

    requested_ligands = [s.strip() for s in args.ligands.split(",") if s.strip()]

    specs = _default_ligand_specs()
    _apply_cli_overrides(specs, args)

    unknown = [l for l in requested_ligands if l not in specs]
    if unknown:
        print(f"[ERROR] 알 수 없는 리간드: {unknown}. 지원: {sorted(specs)}", file=sys.stderr)
        sys.exit(1)

    t_total = time.perf_counter()
    print(
        f"[anchor-calibration] 리간드={requested_ligands} nstruct={nstruct} 시작",
        file=sys.stderr,
    )

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for lig_name in requested_ligands:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"[{lig_name}] 처리 시작", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        results[lig_name] = process_ligand(specs[lig_name], nstruct, WORK_DIR)

    ranking = compare_ranking(results)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "anchor_calibration.py",
        "description": (
            "D-aa 앵커 캘리브레이션: native SST-14 + 임상 검증 D-aa SSTR2 "
            f"결합제(octreotide/lanreotide)를 동일 robust FlexPepDock "
            f"프로토콜(nstruct={nstruct})로 도킹해 실험 Ki 순위 재현 여부 검증"
        ),
        "measurement_conditions": {
            "protocol": "flexpep_refine (AG_src/scripts/flexpep_dock.py::run_flexpep_refine 재사용)",
            "nstruct": nstruct,
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만, native baseline v2/v3와 동일)",
            "input_mode": (
                "각 리간드의 실제 SSTR2 실험 복합체 PDB를 직접 로드 "
                "(MutateResidue 방식 사용 안 함 — octreotide/lanreotide는 "
                "SST-14와 골격이 다른 8-mer)"
            ),
        },
        "ligands": results,
        "ranking_comparison": ranking,
        "elapsed_total_s": elapsed_total,
        "references": [
            "_workspace/DAA_FEASIBILITY_AND_DESIGN.md",
            "_workspace/02_researcher_octreotide_anchor_calibration.md",
        ],
    }

    if args.out_json:
        out_json_path = Path(args.out_json)
    else:
        ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_json_path = PYROSETTA_FLOW_DIR / f"anchor_calibration_{ts}.json"

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    # ── 결과 요약 출력 ────────────────────────────────────────────────────────
    print("\n" + "=" * 72, file=sys.stderr)
    print("[anchor-calibration] 완료 요약", file=sys.stderr)
    print(f"  총 소요시간   : {elapsed_total}s", file=sys.stderr)
    for name, r in results.items():
        if r.get("status") == "DOCKED":
            print(
                f"  {name:12s}: ddg_median={r['ddg_median']} "
                f"n_converged={r['n_converged']}/{r['n_total']} "
                f"sd={r['ddg_sd']} reliability={r['reliability']}",
                file=sys.stderr,
            )
        else:
            print(f"  {name:12s}: {r['status']} — {r.get('message', '')}", file=sys.stderr)
    print(f"  순위 비교     : {ranking.get('verdict', ranking.get('reason'))}", file=sys.stderr)
    print(f"  출력 파일     : {out_json_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    summary = {
        "out_json": str(out_json_path),
        "elapsed_s": elapsed_total,
        "ligand_status": {name: r.get("status") for name, r in results.items()},
        "ranking_verdict": ranking.get("verdict", ranking.get("reason")),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
