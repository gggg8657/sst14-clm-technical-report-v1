#!/usr/bin/env python3
"""
exp72_structure_check.py
=========================
EXP72 (랜덤 vs 시스템 72h 실험) 분석 전용, **독립** 구조 건전성 검증 헬퍼.

목적 (`_workspace/EXP72_ANALYSIS_PLAN.md` Phase0-5, Phase1-1 / `EXP72_ANALYSIS_biology.md` §1):
랜덤 top 서열 `ALCKFFFWKTYLAC`의 ddG 우위(−43.05)가 이황화결합(SG-SG) 압축 등
"스코어 함정(score-exploitation)"인지, 아니면 구조적으로 정상인지를 재분석한다.
72h 로그 85,296건 중 `sg_sg_distance` 기록이 0건이었기 때문에(biology report §0.1),
이미 존재하는 도킹 산출물 PDB를 **retroactive**로 재분석하는 용도다.

기존 파이프라인 코드(`AG_src/scripts/flexpep_dock.py`, `AG_src/pipeline/structure_validation.py`)는
**읽기 전용으로 참고/재사용**만 하며 이 파일은 그것들을 편집하지 않는다 (병렬 작업 충돌 방지).

핵심 발견(설계 근거, 스모크로 재확인):
  두 arm의 펩타이드 체인 배정이 서로 다르다 —
    RANDOM 산출물: 펩타이드=chain A(1), 수용체=chain B(2), 472 residues
    SYSTEM 산출물: 펩타이드=chain B(2), 수용체=chain A(1), 472 residues
  즉 `AG_src/scripts/flexpep_dock.py:_find_peptide_cys_residues()`의
  "마지막 체인=펩타이드" 가정이 arm마다 다르게 어긋난다(biology report §0.1 재현).
  본 모듈은 이 문제를 피하기 위해 **서열 일치로 펩타이드 체인을 자동 탐지**한다
  (`peptide_chain` 인자는 우선 시도값일 뿐, 불일치 시 전체 체인 스캔으로 재탐지).

필수 함수:
    check_complex_structure(pdb_path, sequence, peptide_chain=1) -> dict

CLI:
    conda run -n bio-tools python scripts/exp72_structure_check.py <pdb> <sequence> [--peptide-chain N]

주의:
  - PyRosetta(`bio-tools` conda env) 필수. 가용하지 않으면 MOCK/stub을 반환하지 않고
    즉시 RuntimeError로 실패한다(환각 금지 원칙, PROMPT_PRST_N_FM_EXAMPLE.md §3).
  - 이 파일은 신규 생성물이며 다른 파일을 편집하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 상수 / 판정 임계값
# ---------------------------------------------------------------------------

# SG-SG 이황화결합 거리 판정 window.
# 근거: Rosetta 이상적 disulfide 거리 2.04 A (AG_src/pipeline/structure_validation.py:379,
# `_IDEAL_SS_DISTANCE`). 기존 flexpep_dock.py:_check_disulfide_distance()는 단측
# `distance < 3.0` 만 확인하여 1.60A(압축) 같은 비정상 구조도 "INTACT"로 통과시키는
# 설계상 허점이 있음 (biology report §0.2 지적). 본 모듈은 EXP72_ANALYSIS_biology.md §1-A
# 권고에 따라 양측 window로 교체한다:
#   compressed : d <  1.9   (비정상 압축 — 클래시에 가까운 왜곡, score-exploitation 의심)
#   normal     : 1.9 <= d <= 2.3   (정상 이황화결합, 문헌 관측 2.02-2.05A 포함)
#   stretched  : 2.3 <  d <  2.5   (약간 늘어남, 경계)
#   broken     : d >= 2.5   (사실상 파괴)
SS_COMPRESSED_MAX = 1.9
SS_NORMAL_MAX = 2.3
SS_STRETCHED_MAX = 2.5

# FWKT pharmacophore heavy-atom 접촉 판정 cutoff.
# 근거: 표준 heavy-atom 반데르발스 접촉 거리(합 ~3.4-4.0A)에 여유를 둔 통용 cutoff.
# `pyrosetta_flow/contact_fingerprint.py`는 특정 SSTR2 잔기 번호(Asp122/Gln126 등)에
# 의존하는 salt-bridge/H-bond fingerprint라 수용체 넘버링이 다른 임의 PDB에는 재사용
# 불가(biology report §0.3, 번호체계 불일치 시 None 반환하는 "Fix C" 가드가 이미 있음).
# 본 함수는 그 대신 **넘버링에 의존하지 않는 범용 최소거리 접촉 판정**을 제공한다
# (특정 salt-bridge/H-bond 유무가 아니라 "포켓에 실제로 닿아 있는가"만 판별 — 한계 명시).
PHARMACOPHORE_CONTACT_CUTOFF = 4.5

# FWKT(SST-14 pos7-10)의 서열 내 1-indexed 위치.
PHARMACOPHORE_START = 7
PHARMACOPHORE_END = 10

# Cys3-Cys14 (SST-14 pharmacophore 이황화결합)의 1-indexed 위치.
CYS_POS_1 = 3
CYS_POS_2 = 14


def _check_pyrosetta_available() -> bool:
    try:
        import pyrosetta  # noqa: F401
        return True
    except ImportError:
        return False


def _ensure_pyrosetta_init() -> None:
    import pyrosetta

    if not pyrosetta.rosetta.basic.was_init_called():
        pyrosetta.init(
            options="-mute all -ex1 -ex2aro -ignore_unrecognized_res",
            silent=True,
        )


def _classify_ss_distance(distance: Optional[float]) -> str:
    """SG-SG 거리를 정성 분류한다. 임계 근거는 모듈 상단 상수 주석 참고."""
    if distance is None:
        return "none"
    if distance < SS_COMPRESSED_MAX:
        return "compressed"
    if distance <= SS_NORMAL_MAX:
        return "normal"
    if distance < SS_STRETCHED_MAX:
        return "stretched"
    return "broken"


def _chain_residue_indices(pose: "pyrosetta.Pose", chain_num: int) -> List[int]:
    """주어진 pose chain 번호에 속하는 전체(global) residue index 리스트(오름차순)."""
    begin = pose.chain_begin(chain_num)
    end = pose.chain_end(chain_num)
    return list(range(begin, end + 1))


def _chain_one_letter_sequence(pose: "pyrosetta.Pose", residue_indices: List[int]) -> str:
    chars = []
    for i in residue_indices:
        res = pose.residue(i)
        chars.append(res.name1() if res.is_protein() else "X")
    return "".join(chars)


def _detect_peptide_chain(
    pose: "pyrosetta.Pose",
    sequence: str,
    peptide_chain_hint: int,
) -> Tuple[int, List[int], bool]:
    """펩타이드가 위치한 pose chain 번호를 결정한다.

    두 exp72 arm이 서로 다른 체인 배정을 쓰는 것으로 실측 확인됨(모듈 docstring 참고).
    따라서 `peptide_chain_hint`를 먼저 시도하되, 해당 체인의 서열이 주어진
    `sequence`와 일치하지 않으면 전체 체인을 스캔해 정확히 일치하는 체인으로 재탐지한다.

    Returns:
        (chain_num, residue_indices, autodetected)
    """
    n_chains = pose.num_chains()
    seq_upper = sequence.strip().upper()

    # 1) hint 우선 시도
    if 1 <= peptide_chain_hint <= n_chains:
        residues = _chain_residue_indices(pose, peptide_chain_hint)
        if _chain_one_letter_sequence(pose, residues) == seq_upper:
            return peptide_chain_hint, residues, False

    # 2) 전체 체인 스캔 — 정확히 일치하는 체인 탐색
    for c in range(1, n_chains + 1):
        residues = _chain_residue_indices(pose, c)
        if _chain_one_letter_sequence(pose, residues) == seq_upper:
            return c, residues, (c != peptide_chain_hint)

    # 3) 정확 일치 실패 시, 길이만 같은 체인으로 최선 시도(부분 변이 허용, 경고 기록은 호출부에서)
    for c in range(1, n_chains + 1):
        residues = _chain_residue_indices(pose, c)
        if len(residues) == len(seq_upper):
            return c, residues, True

    # 4) 완전 실패 — hint 그대로 반환 (호출부에서 length mismatch로 드러남)
    residues = _chain_residue_indices(pose, peptide_chain_hint) if 1 <= peptide_chain_hint <= n_chains else []
    return peptide_chain_hint, residues, True


def _min_heavy_atom_distance(
    pose: "pyrosetta.Pose",
    query_residues: List[int],
    target_residues: List[int],
) -> Optional[Tuple[float, int, int]]:
    """query_residues 대 target_residues 간 최소 heavy-atom 거리.

    Returns (min_dist, query_resnum, target_resnum) or None if no atoms compared.
    """
    best: Optional[Tuple[float, int, int]] = None
    for qi in query_residues:
        q_res = pose.residue(qi)
        for ti in target_residues:
            t_res = pose.residue(ti)
            for qa in range(1, q_res.nheavyatoms() + 1):
                q_xyz = q_res.xyz(qa)
                for ta in range(1, t_res.nheavyatoms() + 1):
                    t_xyz = t_res.xyz(ta)
                    d = (q_xyz - t_xyz).norm()
                    if best is None or d < best[0]:
                        best = (d, qi, ti)
    return best


def check_complex_structure(
    pdb_path: str,
    sequence: str,
    peptide_chain: int = 1,
) -> Dict[str, Any]:
    """도킹된 수용체-펩타이드 복합체 PDB의 구조 건전성을 판정한다.

    EXP72 랜덤 top `ALCKFFFWKTYLAC`가 이황화결합 압축(SG-SG 1.60A) 등
    스코어 함정인지 검증하기 위한 독립 재분석 함수. NO MOCK — PyRosetta로
    실제 PDB 좌표를 파싱해 계산한다.

    Args:
        pdb_path:      복합체 PDB 경로 (수용체 + 펩타이드).
        sequence:      펩타이드 1-letter 서열 (14aa SST-14 유사체 가정, Cys3/Cys14 SS bond,
                       FWKT pos7-10 pharmacophore).
        peptide_chain: pose 상 펩타이드 체인 번호(1-indexed) 우선 시도값. 두 exp72 arm의
                       체인 배정이 서로 다름이 실측 확인되어(모듈 docstring), 이 값과
                       실제 서열이 불일치하면 자동으로 올바른 체인을 재탐지한다.

    Returns:
        dict, 최소 키:
          sg_sg_distance, disulfide_intact, disulfide_flag, pharmacophore_contact,
          structure_validation_summary (가능 시) / structure_validation_note (불가 시 이유),
          + 진단용 부가 필드 (peptide_chain_used, peptide_chain_autodetected,
            receptor_chain_used, sequence_matched, warnings).
    """
    if not _check_pyrosetta_available():
        raise RuntimeError(
            "PyRosetta를 임포트할 수 없습니다. `conda run -n bio-tools python ...` 로 실행하세요. "
            "(NO MOCK 원칙 — stub 반환 대신 즉시 실패)"
        )

    import pyrosetta

    _ensure_pyrosetta_init()

    pdb_file = Path(pdb_path)
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

    warnings: List[str] = []
    sequence = sequence.strip().upper()

    pose = pyrosetta.pose_from_pdb(str(pdb_file))

    pep_chain_num, pep_residues, autodetected = _detect_peptide_chain(
        pose, sequence, peptide_chain
    )
    if autodetected:
        warnings.append(
            f"peptide_chain hint={peptide_chain} 서열 불일치 → chain {pep_chain_num} 자동 재탐지"
        )

    actual_seq = _chain_one_letter_sequence(pose, pep_residues)
    sequence_matched = actual_seq == sequence
    if not sequence_matched:
        warnings.append(
            f"펩타이드 체인 서열({actual_seq!r})이 입력 sequence({sequence!r})와 정확히 일치하지 않음"
        )
    if len(pep_residues) != len(sequence):
        warnings.append(
            f"펩타이드 체인 residue 수({len(pep_residues)}) != len(sequence)({len(sequence)}) "
            "— pos3/pos14/pos7-10 매핑이 부정확할 수 있음"
        )

    # 수용체 체인 = 펩타이드가 아닌 나머지 전체 residue.
    all_residues = list(range(1, pose.total_residue() + 1))
    receptor_residues = [i for i in all_residues if i not in set(pep_residues)]
    receptor_chain_nums = sorted({pose.chain(i) for i in receptor_residues})

    result: Dict[str, Any] = {
        "pdb_path": str(pdb_file),
        "sequence": sequence,
        "peptide_chain_hint": peptide_chain,
        "peptide_chain_used": pep_chain_num,
        "peptide_chain_autodetected": autodetected,
        "peptide_chain_sequence_observed": actual_seq,
        "sequence_matched": sequence_matched,
        "receptor_chain_used": receptor_chain_nums,
        "n_peptide_residues": len(pep_residues),
        "n_receptor_residues": len(receptor_residues),
    }

    # -----------------------------------------------------------------
    # 1) SG-SG 이황화결합 (Cys3-Cys14)
    # -----------------------------------------------------------------
    sg_sg_distance: Optional[float] = None
    disulfide_flag = "none"
    cys1_resnum: Optional[int] = None
    cys2_resnum: Optional[int] = None

    has_cys_at_expected_positions = (
        len(sequence) >= CYS_POS_2
        and sequence[CYS_POS_1 - 1] == "C"
        and sequence[CYS_POS_2 - 1] == "C"
    )

    if not has_cys_at_expected_positions:
        warnings.append(
            f"sequence[{CYS_POS_1}]/[{CYS_POS_2}]가 Cys가 아님 — 이황화결합 판정 불가(disulfide_flag='none')"
        )
    elif len(pep_residues) < CYS_POS_2:
        warnings.append(
            f"펩타이드 체인 residue 수({len(pep_residues)})가 pos{CYS_POS_2}보다 적음 — SS 판정 불가"
        )
    else:
        cys1_resnum = pep_residues[CYS_POS_1 - 1]
        cys2_resnum = pep_residues[CYS_POS_2 - 1]
        res1 = pose.residue(cys1_resnum)
        res2 = pose.residue(cys2_resnum)
        if res1.name3().strip() != "CYS" or res2.name3().strip() != "CYS":
            warnings.append(
                f"pos{CYS_POS_1}/pos{CYS_POS_2} 매핑 residue가 CYS가 아님 "
                f"({res1.name3()}{cys1_resnum}/{res2.name3()}{cys2_resnum}) — 매핑 오류 의심"
            )
        elif not (res1.has("SG") and res2.has("SG")):
            warnings.append("Cys 잔기에 SG 원자 없음 — SS 판정 불가")
        else:
            sg1 = res1.xyz("SG")
            sg2 = res2.xyz("SG")
            sg_sg_distance = round((sg1 - sg2).norm(), 3)
            disulfide_flag = _classify_ss_distance(sg_sg_distance)

    disulfide_intact = disulfide_flag == "normal"

    result["cys_pos1_resnum"] = cys1_resnum
    result["cys_pos2_resnum"] = cys2_resnum
    result["sg_sg_distance"] = sg_sg_distance
    result["disulfide_intact"] = disulfide_intact
    result["disulfide_flag"] = disulfide_flag
    result["disulfide_threshold_note"] = (
        f"compressed<{SS_COMPRESSED_MAX}, normal=[{SS_COMPRESSED_MAX},{SS_NORMAL_MAX}], "
        f"stretched<{SS_STRETCHED_MAX}, broken>={SS_STRETCHED_MAX} "
        "(EXP72_ANALYSIS_biology.md §1-A 권고, Rosetta ideal 2.04A 기준 양측 window)"
    )

    # -----------------------------------------------------------------
    # 2) FWKT pharmacophore(pos7-10) 수용체 접촉
    # -----------------------------------------------------------------
    pharmacophore_contact: Dict[str, Any] = {
        "positions": [PHARMACOPHORE_START, PHARMACOPHORE_END],
        "cutoff_angstrom": PHARMACOPHORE_CONTACT_CUTOFF,
        "method": (
            "min heavy-atom distance, peptide pos7-10 vs 전체 수용체 residue "
            "(salt-bridge/H-bond 특이적 fingerprint 아님 — contact_fingerprint.py의 "
            "SSTR2 특정 잔기번호(Asp122/Gln126) 의존 로직과 달리 수용체 넘버링에 "
            "의존하지 않는 범용 근접성 판별. '실제 접촉 여부'만 판별 가능하고 "
            "결합 모드의 특이성(살트브릿지/수소결합 종류)은 판별 불가 — 한계 명시)"
        ),
    }

    if len(pep_residues) < PHARMACOPHORE_END or not receptor_residues:
        pharmacophore_contact["in_contact"] = None
        pharmacophore_contact["min_distance"] = None
        pharmacophore_contact["reason"] = (
            "펩타이드 residue 수 부족 또는 수용체 체인 없음 — 접촉 판정 불가"
        )
    else:
        motif_residues = pep_residues[PHARMACOPHORE_START - 1: PHARMACOPHORE_END]
        motif_seq = sequence[PHARMACOPHORE_START - 1: PHARMACOPHORE_END] if len(sequence) >= PHARMACOPHORE_END else ""
        pharmacophore_contact["motif_sequence_observed"] = motif_seq
        pharmacophore_contact["motif_matches_fwkt"] = motif_seq == "FWKT"

        best = _min_heavy_atom_distance(pose, motif_residues, receptor_residues)
        if best is None:
            pharmacophore_contact["in_contact"] = None
            pharmacophore_contact["min_distance"] = None
            pharmacophore_contact["reason"] = "원자 비교 실패(heavy atom 없음)"
        else:
            min_dist, q_resnum, t_resnum = best
            pharmacophore_contact["min_distance"] = round(min_dist, 3)
            pharmacophore_contact["in_contact"] = min_dist <= PHARMACOPHORE_CONTACT_CUTOFF
            pharmacophore_contact["closest_peptide_resnum"] = q_resnum
            pharmacophore_contact["closest_receptor_resnum"] = t_resnum

    result["pharmacophore_contact"] = pharmacophore_contact

    # -----------------------------------------------------------------
    # 3) structure_validation.py 배선 시도 (Ramachandran/rotamer/backbone 요약)
    #    — 존재하고 배선 가능하면 요약 포함, 아니면 None + 이유 (환각 금지).
    # -----------------------------------------------------------------
    structure_validation_summary: Optional[Dict[str, Any]] = None
    structure_validation_note: Optional[str] = None
    try:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from AG_src.pipeline.structure_validation import validate_structure  # type: ignore

        disulfide_pairs = (
            [(cys1_resnum, cys2_resnum)] if cys1_resnum and cys2_resnum else None
        )
        full_report = validate_structure(
            str(pdb_file),
            peptide_chain=pep_chain_num,
            disulfide_pairs=disulfide_pairs,
        )
        structure_validation_summary = {
            "quality_grade": full_report.get("quality_grade"),
            "quality_issues": full_report.get("quality_issues"),
            "ramachandran_pct_favored": full_report.get("ramachandran", {}).get("pct_favored"),
            "ramachandran_n_outlier": full_report.get("ramachandran", {}).get("n_outlier"),
            "rotamer_n_outliers": full_report.get("rotamer_quality", {}).get("n_outliers"),
            "backbone_n_bond_outliers": full_report.get("backbone_geometry", {}).get("n_bond_outliers"),
            "backbone_n_angle_outliers": full_report.get("backbone_geometry", {}).get("n_angle_outliers"),
            "disulfide_bonds_module_result": full_report.get("disulfide_bonds"),
        }
    except ImportError as e:
        structure_validation_note = f"AG_src.pipeline.structure_validation import 실패: {e}"
    except Exception as e:  # noqa: BLE001 - 배선 실패를 조용히 삼키지 않고 이유를 기록
        structure_validation_note = f"structure_validation.validate_structure() 실행 중 예외: {e}"

    result["structure_validation_summary"] = structure_validation_summary
    result["structure_validation_note"] = structure_validation_note

    result["warnings"] = warnings

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "EXP72 독립 구조 건전성 검증 — SG-SG 이황화결합 위상 + FWKT pharmacophore "
            "접촉 + (가능 시) structure_validation.py 요약."
        )
    )
    parser.add_argument("pdb_path", help="복합체 PDB 경로")
    parser.add_argument("sequence", help="펩타이드 1-letter 서열 (예: ALCKFFFWKTYLAC)")
    parser.add_argument(
        "--peptide-chain", type=int, default=1,
        help="펩타이드 pose chain 번호 우선 시도값(1-indexed, 기본 1). "
             "서열 불일치 시 자동 재탐지됨.",
    )
    args = parser.parse_args()

    report = check_complex_structure(
        args.pdb_path, args.sequence, peptide_chain=args.peptide_chain
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
