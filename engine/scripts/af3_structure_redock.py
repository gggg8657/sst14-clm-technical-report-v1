#!/usr/bin/env python3
"""af3_structure_redock.py
===========================
AF3(AlphaFold3)가 예측한 SSTR2-AICLNWFWKTVISC 복합체 **구조 자체**에
우리 robust FlexPepDock 프로토콜을 그대로 재적용해, 우리 도킹 스코어가
"어떤 초기 구조를 넣어도" 강한 결합(ddG 매우 음수)을 재현하는지
교차검증한다.

3중 교차검증 배경:
  (1) 우리 원래 도킹: AICLNWFWKTVISC, boltz/native 기반 초기 구조 →
      ddG_median = -48.8 REU (기존 리더보드/파이프라인 산출값, 이 스크립트가
      재계산하지 않음 — CLI로 참고치 주입).
  (2) AlphaFold3 독립 예측: 같은 서열 조합에 대해 AF3가 자체적으로
      구조를 예측 → iPTM 0.76, has_clash 0 (fold_2026_02_25_17_15
      summary_confidences_0.json).
  (3) ★이 스크립트: (2)의 AF3 예측 구조(cif)를 PDB로 변환해 **그 구조를
      입력으로** 우리 FlexPepDock robust 재도킹을 수행 → ddG_median 산출.
      (1)과 (3)이 비슷한 크기의 강한 음수 ddG로 수렴하면 우리 도킹이
      구조 의존적이지 않고 신뢰할 만하다는 근거가 되고, 크게 다르면
      우리 도킹이 특정 초기 구조에 민감하다는 것을 노출한다.

새 도킹 알고리즘을 만들지 않는다 — scripts/anchor_calibration.py의
extract_receptor_peptide_chains() / run_ligand_robust_flexpep()
(= AG_src/scripts/flexpep_dock.py::run_flexpep_refine 재사용)을 그대로
import해서 쓴다.

입력:
  _workspace/fold_2026_02_25_17_15/fold_2026_02_25_17_15_model_0.cif
  (chain A = 펩타이드 AICLNWFWKTVISC 14aa, chain B = human SSTR2 수용체
  369aa — entity_poly 태그 및 gemmi 파싱으로 확인됨. auth_asym_id 기준
  A=펩타이드, B=수용체이므로 anchor_calibration의 "수용체 먼저" 표준
  순서로 재배열이 필요하다.)

사용법:
  conda run -n bio-tools python scripts/af3_structure_redock.py
  conda run -n bio-tools python scripts/af3_structure_redock.py --nstruct 15

주의:
  - 엔진/continuous/experiment_log/autopush/surrogate 무접촉.
  - cif/pdb 읽기 및 runs/pyrosetta_flow/af3_structure_redock_work/ 내
    임시 변환 파일만 생성. data/ 등 READ-ONLY 경계 미접촉
    (_workspace/ 입력은 읽기 전용으로만 사용).
  - flexpep_dock.py 로직 변경 없음(그대로 재사용).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
AG_SRC_SCRIPTS = REPO_ROOT / "AG_src" / "scripts"

WORK_DIR = PYROSETTA_FLOW_DIR / "af3_structure_redock_work"
DEFAULT_NSTRUCT = 12  # robust 하한(사용자 지정 nstruct>=12) 준수

DEFAULT_CIF = (
    REPO_ROOT
    / "_workspace"
    / "fold_2026_02_25_17_15"
    / "fold_2026_02_25_17_15_model_0.cif"
)
DEFAULT_CONFIDENCE_JSON = (
    REPO_ROOT
    / "_workspace"
    / "fold_2026_02_25_17_15"
    / "fold_2026_02_25_17_15_summary_confidences_0.json"
)

EXPECTED_PEPTIDE_SEQ = "AICLNWFWKTVISC"

# 비교 대상 참고치 (사용자 프롬프트 지정값 — 이 스크립트가 재계산하지
# 않고, 다른 산출물의 기존 결과를 "참고 문자열"로만 병기한다)
OUR_ORIGINAL_DOCKING_DDG = -48.8
OUR_ORIGINAL_DOCKING_SOURCE = (
    "기존 파이프라인 도킹 결과(boltz/native 구조 기반 초기 포즈), "
    "AICLNWFWKTVISC robust FlexPepDock ddG_median 참고치 — "
    "사용자 제공, 이 스크립트가 재계산한 값 아님"
)

if str(AG_SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(AG_SRC_SCRIPTS))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


# ── cif -> pdb 변환 (gemmi) ──────────────────────────────────────────────────

def convert_cif_to_pdb(cif_path: Path, out_pdb: Path) -> Path:
    """AF3 출력 cif를 표준 PDB로 변환한다(gemmi 사용, bio-tools env 가용).

    변환만 수행 — 체인 재배열/추출은 별도로
    anchor_calibration.extract_receptor_peptide_chains()가 담당한다
    (기존 로직 재사용, 중복 구현 금지).
    """
    import gemmi

    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    st.write_pdb(str(out_pdb))

    print(f"  [cif->pdb] {cif_path.name} -> {out_pdb}", file=sys.stderr)
    return out_pdb


def inspect_chains(pdb_path: Path) -> Dict[str, int]:
    """PDB의 체인별 ATOM 라인 수를 세어 반환(추출 전 검증용)."""
    counts: Dict[str, int] = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                cid = line[21]
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def _three_to_one(res3: str) -> str:
    table = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return table.get(res3, "X")


def extract_chain_seq(pdb_path: Path, chain_id: str) -> str:
    """PDB 한 체인의 CA 원자 순서로 1-letter 서열을 복원(검증용)."""
    seq_chars = []
    seen_res = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM") and line[21] == chain_id and line[12:16].strip() == "CA":
                res_key = line[22:27]
                if res_key in seen_res:
                    continue
                seen_res.add(res_key)
                seq_chars.append(_three_to_one(line[17:20].strip()))
    return "".join(seq_chars)


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AF3 예측 구조(cif) 자체에 우리 robust FlexPepDock을 재적용해 "
            "3중 교차검증(우리 도킹 vs AF3 iPTM vs AF3구조+우리도킹)을 수행"
        )
    )
    parser.add_argument("--cif", default=str(DEFAULT_CIF), help="AF3 model cif 경로")
    parser.add_argument(
        "--confidence-json",
        default=str(DEFAULT_CONFIDENCE_JSON),
        help="AF3 summary_confidences json 경로 (iptm/has_clash 추출용)",
    )
    parser.add_argument(
        "--nstruct",
        type=int,
        default=DEFAULT_NSTRUCT,
        help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT}, robust 하한 준수)",
    )
    parser.add_argument(
        "--our-original-ddg",
        type=float,
        default=OUR_ORIGINAL_DOCKING_DDG,
        help="비교 대상 (1) 우리 원래 도킹 ddG_median 참고치 오버라이드",
    )
    parser.add_argument(
        "--out-json",
        default=str(PYROSETTA_FLOW_DIR / "af3_structure_redock.json"),
        help="출력 JSON 경로",
    )
    args = parser.parse_args()

    if args.nstruct < 10:
        print(
            f"WARNING: nstruct={args.nstruct} < 10 — robust 하한 미충족. "
            "n_converged가 낮으면 UNCERTAIN/LOW로 표시됩니다.",
            file=sys.stderr,
        )

    from anchor_calibration import (  # type: ignore
        extract_receptor_peptide_chains,
        run_ligand_robust_flexpep,
        _reliability_verdict,
    )

    cif_path = Path(args.cif)
    confidence_path = Path(args.confidence_json)

    if not cif_path.exists():
        print(f"[ERROR] cif 파일 없음: {cif_path}", file=sys.stderr)
        sys.exit(1)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print(f"[af3-structure-redock] 입력: {cif_path}", file=sys.stderr)
    print(f"[af3-structure-redock] nstruct={args.nstruct}", file=sys.stderr)

    # 1) cif -> pdb 변환
    raw_pdb = WORK_DIR / "af3_raw.pdb"
    convert_cif_to_pdb(cif_path, raw_pdb)

    chain_counts = inspect_chains(raw_pdb)
    print(f"  [chains] raw pdb chain atom counts: {chain_counts}", file=sys.stderr)

    # cif 상 auth_asym_id: A=펩타이드(entity 1, 14aa), B=수용체(entity 2, 369aa)
    # (entity_poly.pdbx_strand_id 태그 + gemmi 파싱으로 확인).
    # anchor_calibration 표준 입력은 "수용체 먼저" 이므로 여기서는
    # receptor_chain=B, peptide_chain=A 로 지정해 재배열한다.
    peptide_seq_raw = extract_chain_seq(raw_pdb, "A")
    receptor_seq_len = len(extract_chain_seq(raw_pdb, "B"))
    print(
        f"  [chains] chain A(peptide) seq={peptide_seq_raw} "
        f"(len={len(peptide_seq_raw)}), chain B(receptor) len={receptor_seq_len}",
        file=sys.stderr,
    )

    seq_match = peptide_seq_raw == EXPECTED_PEPTIDE_SEQ
    if not seq_match:
        print(
            f"  [WARN] 펩타이드 서열 불일치: 기대={EXPECTED_PEPTIDE_SEQ} "
            f"실제={peptide_seq_raw}",
            file=sys.stderr,
        )

    # 2) 표준 순서(수용체=새 chain A, 펩타이드=새 chain B)로 재배열
    #    (anchor_calibration.extract_receptor_peptide_chains 재사용)
    reordered_pdb = WORK_DIR / "af3_receptor_peptide.pdb"
    try:
        extract_receptor_peptide_chains(
            raw_pdb, receptor_chain="B", peptide_chain="A", out_pdb=reordered_pdb
        )
    except Exception as exc:
        print(f"[ERROR] 체인 재배열 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    # 3) robust FlexPepDock 재도킹 (anchor_calibration.run_ligand_robust_flexpep
    #    = flexpep_dock.run_flexpep_refine 재사용, 새 알고리즘 없음)
    redock_dir = WORK_DIR / "redock"
    stats = run_ligand_robust_flexpep(reordered_pdb, args.nstruct, redock_dir)

    ddg_median = stats.get("ddg_median")
    n_converged = int(stats.get("n_converged") or 0)
    n_total = int(stats.get("n_total") or args.nstruct)
    reliability, recommendation = _reliability_verdict(ddg_median, n_converged, n_total)

    # 4) AF3 confidence 값 로드
    af3_confidence: Dict = {}
    if confidence_path.exists():
        with open(confidence_path) as fh:
            af3_confidence = json.load(fh)
    else:
        print(f"  [WARN] confidence json 없음: {confidence_path}", file=sys.stderr)

    af3_iptm = af3_confidence.get("iptm")
    af3_has_clash = af3_confidence.get("has_clash")

    # 5) 3중 교차검증 판정
    delta_vs_original = None
    signal_agreement = "UNKNOWN"
    agreement_note = ""
    if ddg_median is not None:
        delta_vs_original = round(ddg_median - args.our_original_ddg, 2)
        abs_delta = abs(delta_vs_original)
        # 판정 기준: 둘 다 강한 음수(<0, 특히 < -20 REU 수준)이고
        # 차이가 원래 값 크기의 상당 부분(예: 50%) 이내면 "재현"으로 판정.
        both_strong_negative = ddg_median < -10 and args.our_original_ddg < -10
        relative_diff = (
            abs_delta / abs(args.our_original_ddg)
            if args.our_original_ddg
            else float("inf")
        )
        if both_strong_negative and relative_diff <= 0.5:
            signal_agreement = "CONSISTENT"
            agreement_note = (
                f"AF3 구조 기반 ddG_median({ddg_median:.2f})이 우리 원래 도킹"
                f"({args.our_original_ddg:.2f})과 같은 강한 음수 등급이며 "
                f"상대 차이 {relative_diff*100:.1f}% <= 50% — 구조 비의존적 "
                f"신뢰 신호로 판정. AF3 iPTM({af3_iptm})·clash({af3_has_clash})"
                f"도 우호적 구조를 시사하므로 세 신호가 방향 일치."
            )
        elif both_strong_negative:
            signal_agreement = "PARTIAL"
            agreement_note = (
                f"AF3 구조 기반 ddG_median({ddg_median:.2f})과 우리 원래 도킹"
                f"({args.our_original_ddg:.2f}) 모두 강한 음수(<-10 REU)로 "
                f"'강결합' 방향은 일치하나 절대 크기 차이가 "
                f"{relative_diff*100:.1f}%(>50%)로 커서 구조 의존성이 있음 — "
                f"정성적 재현이나 정량 재현은 아님."
            )
        else:
            signal_agreement = "DIVERGENT"
            agreement_note = (
                f"AF3 구조 기반 ddG_median({ddg_median:.2f})이 우리 원래 도킹"
                f"({args.our_original_ddg:.2f})과 강한 음수 등급을 공유하지 "
                f"않음 — 우리 도킹이 초기 구조(AF3 predicted vs boltz/native)"
                f"에 민감하게 반응한다는 구조 의존성 노출."
            )
    else:
        signal_agreement = "DOCKING_FAILED"
        agreement_note = (
            "AF3 구조 기반 재도킹이 실패/미수렴하여 비교 불가 — "
            f"error={stats.get('error')}"
        )

    elapsed_total = round(time.perf_counter() - t_total, 1)

    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "script": "af3_structure_redock.py",
        "description": (
            "AF3(AlphaFold3)가 예측한 SSTR2-AICLNWFWKTVISC 복합체 구조 자체에 "
            "우리 robust FlexPepDock(anchor_calibration.py 재사용)을 재적용해 "
            "3중 교차검증(우리 도킹 vs AF3 iPTM vs AF3구조+우리도킹) 수행"
        ),
        "input": {
            "cif_path": str(cif_path),
            "confidence_json_path": str(confidence_path),
            "raw_pdb": str(raw_pdb),
            "reordered_pdb": str(reordered_pdb),
            "chain_atom_counts_raw": chain_counts,
            "peptide_seq_extracted": peptide_seq_raw,
            "peptide_seq_expected": EXPECTED_PEPTIDE_SEQ,
            "peptide_seq_match": seq_match,
            "receptor_chain_length": receptor_seq_len,
        },
        "measurement_conditions": {
            "protocol": "flexpep_refine (AG_src/scripts/flexpep_dock.py::run_flexpep_refine 재사용, anchor_calibration.py 경유)",
            "nstruct": args.nstruct,
            "ddg_convergence_criterion": "ddg < 0 AND ddg > DDG_PHYSICAL_FLOOR(-150)",
            "ddg_aggregation": "ddg_median (수렴한 것만, native baseline v2/v3와 동일 프로토콜)",
        },
        "three_way_cross_validation": {
            "signal_1_our_original_docking": {
                "ligand": "AICLNWFWKTVISC",
                "ddg_median": args.our_original_ddg,
                "structure_basis": "boltz/native 기반 초기 포즈",
                "source": OUR_ORIGINAL_DOCKING_SOURCE,
            },
            "signal_2_alphafold3_prediction": {
                "iptm": af3_iptm,
                "ptm": af3_confidence.get("ptm"),
                "has_clash": af3_has_clash,
                "ranking_score": af3_confidence.get("ranking_score"),
                "chain_iptm": af3_confidence.get("chain_iptm"),
                "source": str(confidence_path),
            },
            "signal_3_af3_structure_plus_our_docking": {
                "ligand": "AICLNWFWKTVISC",
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
                "structure_basis": "AF3 predicted complex (fold_2026_02_25_17_15_model_0.cif)",
            },
            "delta_signal3_minus_signal1": delta_vs_original,
            "signal_agreement": signal_agreement,
            "agreement_note": agreement_note,
        },
        "elapsed_total_s": elapsed_total,
        "caveats": [
            "도킹 결과는 정성 참고용 — n_converged/nstruct, sd를 함께 확인할 것.",
            "signal_1(우리 원래 도킹)은 이 스크립트가 재계산한 값이 아니라 "
            "사용자/기존 산출물이 제공한 참고치이며, 서로 다른 초기 구조 "
            "(boltz/native 기반 vs AF3 예측)에서 나온 값이라는 점이 이 "
            "교차검증의 핵심 변수임.",
            "AF3 iPTM은 구조 예측 신뢰도이지 결합 자유에너지가 아니므로 "
            "ddG와 척도가 다름 — 방향성(강결합 시사 여부)만 비교.",
        ],
    }

    out_json_path = Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as fh:
        json.dump(out_data, fh, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72, file=sys.stderr)
    print("[af3-structure-redock] 완료 요약", file=sys.stderr)
    print(f"  총 소요시간            : {elapsed_total}s", file=sys.stderr)
    print(f"  (1) 우리 원래 도킹 ddG : {args.our_original_ddg}", file=sys.stderr)
    print(
        f"  (2) AF3 iPTM/clash     : iptm={af3_iptm} has_clash={af3_has_clash}",
        file=sys.stderr,
    )
    print(
        f"  (3) AF3구조+우리도킹   : ddg_median={ddg_median} "
        f"n_converged={n_converged}/{n_total} sd={stats.get('ddg_sd')} "
        f"reliability={reliability}",
        file=sys.stderr,
    )
    print(f"  신호 일치 판정         : {signal_agreement}", file=sys.stderr)
    print(f"  {agreement_note}", file=sys.stderr)
    print(f"  출력 파일              : {out_json_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)

    summary = {
        "out_json": str(out_json_path),
        "elapsed_s": elapsed_total,
        "ddg_median_af3_structure": ddg_median,
        "our_original_ddg": args.our_original_ddg,
        "af3_iptm": af3_iptm,
        "signal_agreement": signal_agreement,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
