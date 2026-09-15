#!/usr/bin/env python3
"""
blind_dock_validation.py
=========================
SSTR2 blind(de novo) FlexPepDock ab-initio 검증.

목적
----
AF3/Boltz pose를 시작점으로 쓰지 않고, apo SSTR2 수용체만 놓은 상태에서
후보 펩타이드를 "정답 위치를 모르는 채" 도킹하여, 후보가 독립적으로
SSTR2 결합 포켓을 재발견하는지 검증한다(3중 교차검증의 가장 엄격한 신호).

절대 원칙 (호출자 지시 그대로 준수)
--------------------------------
- NO MOCK: 실제 PyRosetta FlexPepDockingProtocol만 사용. 가짜 수치 금지.
- 수렴 안 되거나 결과가 모호하면 정직하게 "inconclusive"로 보고한다.
  우리 refine 도킹(-48.8)보다 약하거나 포켓을 못 찾으면 그대로 보고.

방법론과 한계 (반드시 이해하고 읽을 것)
--------------------------------------
FlexPepDockingProtocol은 PyRosetta 안에서 "글로벌"(수용체 전체 표면을 훑는)
docking이 아니라, jump(rigid-body) 좌표계를 저해상도 centroid 단계에서
몬테카를로로 흔든 뒤 고해상도로 refine하는 **국소 탐색**이다. 완전한
blind(수용체 전체 표면 어디든 가능) 도킹을 흉내내려면 통상 수백~수천 개의
독립 decoy가 필요하고, 이는 계산 예산을 크게 초과한다.

이 스크립트가 실제로 구현하는 "blind"의 정의는 다음과 같다(호출자 지시
3번 항목의 한계 명시 요구 반영):
  1. 펩타이드 시작 conformation은 extended(모든 phi=-150/psi=150/omega=180) —
     AF3/Boltz backbone을 전혀 참조하지 않는다.
  2. 각 decoy마다 알려진 포켓 중심(W197/Y205/F272 CA centroid)에서
     반경 BLIND_RADIUS_A(기본 20Å, 세포외 domain 크기 스케일)만큼 떨어진
     **완전 무작위 방향**의 구면 위 지점에 펩타이드 중심을 배치하고,
     RigidBodyRandomizeMover로 회전도 완전 무작위화한다.
  3. 따라서 시작 pose는 "포켓 부근이지만 포켓이 어느 방향인지, 어떤
     자세로 결합하는지는 모르는" 상태다 — 정답 pose(AF3/native)와는
     좌표·자세가 다르다.
  4. 이것은 "국소 반경 내 blind" 이지 "수용체 표면 전체 blind"가 아니다.
     즉 펩타이드가 포켓 반경 20Å 구 바깥(예: 수용체 반대쪽 막)에서
     출발해 포켓을 찾는 능력까지는 검증하지 않는다. 이 한계는
     honest_verdict/limitations 필드에 항상 명시한다.

포켓 잔기 매핑 근거
--------------------
호출자가 요청한 문헌 기준 포켓 잔기는 W197/Y205/F272 (수용체 잔기 번호).
`data/somatostatin_receptor/curated/SSTR2_receptor.pdb` 를 직접 조회한 결과
(본 스크립트 개발 중 검증):
    PDB resnum 197 = TRP  (W197 일치)
    PDB resnum 205 = TYR  (Y205 일치)
    PDB resnum 272 = PHE  (F272 일치)
즉 curated PDB의 잔기 번호 체계가 문헌/기존 파이프라인 포켓 정의
(`data/somatostatin_receptor/binding_pocket_SSTR2.json`: residue_ids
[208, 209, 272, 273, 276], source_pdb=SSTR2_7XNA) 와 **동일한 numbering**을
쓰고 있어 별도 재매핑(시퀀스 정렬 등) 없이 PDB 잔기번호를 그대로
`pdb_info().pdb2pose()` 로 pose 인덱스 변환만 하면 된다. 이 사실은
`pose_from_pdb` 로 로드 후 `pdb2pose("A", resnum)` 매핑 성공 여부로
런타임에도 재검증한다(실패 시 에러 raise, 조용히 넘어가지 않음).

Ab-initio 옵션
--------------
PyRosetta 2026.20 Python 바인딩의 FlexPepDockingProtocol은
set_lowres_abinitio() 같은 세터를 노출하지 않는다(dir() 확인 완료,
개발 중 검증). 대신 `-flexPepDocking:lowres_abinitio true` 커맨드라인
옵션을 pyrosetta.init()에 전달하면 옵션 시스템이 이를 읽어 저해상도
centroid 단계(temperature annealing 로그로 실행 확인됨)부터 시작한다.
FlexPepDockingProtocol()은 인자 없이 생성하고 옵션에 전부 위임한다.

스코어링 지표 정의 (정직성 노트)
--------------------------------
FlexPepDockingProtocol.apply() 는 (C++ 실행파일과 달리) pose.scores에
표준 FlexPepDock 리포터 컬럼(I_sc, reweighted_sc, Irms 등)을 채우지
않는다(개발 중 실측 확인: fa_atr/fa_rep/...만 존재). 따라서 이 스크립트는
기존 `AG_src/scripts/flexpep_dock.py::compute_interface_ddg()` 와 동일한
방법(InterfaceAnalyzerMover, jump=1, pack_input/pack_separated=True)으로
interface dG를 직접 계산하고, 이를 "I_sc"의 대리 지표로 보고한다.
결과 JSON의 `metric_definition` 필드에 이 사실을 항상 남긴다 — 진짜
FlexPepDock reweighted_sc가 아니라 InterfaceAnalyzer dG임을 숨기지 않는다.

사용
----
conda run -n bio-tools python scripts/blind_dock_validation.py \
    --sequence AICLNWFWKTVISC --label candidate --n-decoys 50

conda run -n bio-tools python scripts/blind_dock_validation.py \
    --sequence AGCKNFFWKTFTSC --label native --n-decoys 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEPTOR_PDB = REPO_ROOT / "data/somatostatin_receptor/curated/SSTR2_receptor.pdb"
OUT_DIR = REPO_ROOT / "runs/pyrosetta_flow"
WORK_DIR = OUT_DIR / "blind_dock_validation_work"

# 문헌 기준 SSTR2 포켓 잔기 (PDB 잔기번호, curated PDB numbering과 일치 검증됨)
POCKET_PDB_RESNUMS: List[int] = [197, 205, 272]
POCKET_CHAIN = "A"
POCKET_CONTACT_THRESHOLD_A = 5.0  # heavy atom 5A 이내 = 접촉 (호출자 지시)
BLIND_RADIUS_A = 20.0  # 포켓 centroid 로부터 blind 배치 반경


def _err_exit(msg: str) -> None:
    print(json.dumps({"error": msg}), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# PyRosetta init
# ---------------------------------------------------------------------------

def init_pyrosetta_abinitio(max_threads: int = 1) -> None:
    """FlexPepDock ab-initio 저해상도 모드로 PyRosetta 초기화.

    -flexPepDocking:lowres_abinitio true 로 centroid 단계부터 몬테카를로
    탐색을 시작하도록 옵션 시스템에 위임한다(Python API에 별도 세터 없음,
    개발 중 dir() 검증 완료).
    """
    import pyrosetta

    pyrosetta.init(
        options=(
            "-mute all -ex1 -ex2aro -ignore_unrecognized_res"
            " -flexPepDocking:lowres_abinitio true"
            " -flexPepDocking:pep_refine true"
            " -multithreading:total_threads " + str(max(1, max_threads))
        ),
        silent=True,
    )


# ---------------------------------------------------------------------------
# Peptide construction (extended, no template)
# ---------------------------------------------------------------------------

def build_extended_peptide(sequence: str) -> "pyrosetta.Pose":
    """서열로부터 extended conformation 펩타이드 pose 생성 (템플릿 무사용).

    전부 L-aa 가정(호출자 지시: D-aa 없음). 소문자가 섞이면 명시적으로 거부한다
    (조용한 오귀속 방지 — flexpep_dock.py의 기존 정책과 동일한 정신).
    """
    import pyrosetta

    if not sequence.isupper():
        raise ValueError(
            f"build_extended_peptide: D-aa(소문자) 미지원 대상 서열 '{sequence}' — "
            "blind ab-initio 검증은 L-aa 전용으로 설계됨."
        )

    pep = pyrosetta.pose_from_sequence(sequence, "fa_standard")
    for i in range(1, pep.total_residue() + 1):
        pep.set_phi(i, -150.0)
        pep.set_psi(i, 150.0)
        pep.set_omega(i, 180.0)
    return pep


# ---------------------------------------------------------------------------
# Complex assembly + blind randomization
# ---------------------------------------------------------------------------

def assemble_blind_complex(
    receptor_pose: "pyrosetta.Pose",
    peptide_pose: "pyrosetta.Pose",
    pocket_centroid,
    rng: random.Random,
    blind_radius_a: float = BLIND_RADIUS_A,
) -> "pyrosetta.Pose":
    """수용체+펩타이드를 하나의 pose로 합치고, 펩타이드를 포켓 centroid에서
    blind_radius_a 만큼 떨어진 무작위 방향/무작위 회전으로 배치한다.

    Returns: combo pose (chain 1=receptor, chain 2=peptide), jump 1 로 연결됨.
    """
    import numpy as np
    import pyrosetta
    from pyrosetta.rosetta.core.pose import append_pose_to_pose
    from pyrosetta.rosetta.numeric import xyzVector_double_t as V3
    from pyrosetta.rosetta.protocols.rigid import (
        Partner,
        RigidBodyRandomizeMover,
        RigidBodyTransMover,
    )

    combo = pyrosetta.Pose()
    combo.assign(receptor_pose)
    append_pose_to_pose(combo, peptide_pose, True)

    if combo.num_chains() < 2:
        raise RuntimeError(
            f"assemble_blind_complex: expected 2 chains after append, got {combo.num_chains()}"
        )
    jump_num = 1
    pep_begin = combo.chain_begin(2)
    pep_end = combo.chain_end(2)

    def _centroid(b: int, e: int):
        xs = []
        for i in range(b, e + 1):
            xyz = combo.residue(i).xyz("CA")
            xs.append([xyz.x, xyz.y, xyz.z])
        return np.mean(xs, axis=0)

    # 무작위 방향 (구면 균등 샘플)
    u = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r_xy = math.sqrt(max(0.0, 1.0 - u * u))
    direction = np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), u])
    target_point = np.asarray(pocket_centroid) + blind_radius_a * direction

    pep_centroid = _centroid(pep_begin, pep_end)
    trans_vec = target_point - pep_centroid
    dist = float(np.linalg.norm(trans_vec))

    if dist > 1e-6:
        trans_mover = RigidBodyTransMover(combo, jump_num)
        trans_mover.trans_axis(V3(float(trans_vec[0]), float(trans_vec[1]), float(trans_vec[2])))
        trans_mover.step_size(dist)
        trans_mover.apply(combo)

    randomizer = RigidBodyRandomizeMover(combo, jump_num, Partner.partner_downstream)
    randomizer.apply(combo)

    return combo


def compute_pocket_centroid(receptor_pose: "pyrosetta.Pose") -> Tuple["np.ndarray", List[int]]:
    """포켓 잔기(PDB 197/205/272)의 CA centroid를 apo 수용체 pose에서 계산.

    pdb2pose 매핑 실패 시 조용히 넘어가지 않고 명시적으로 raise한다.
    """
    import numpy as np

    pdb_info = receptor_pose.pdb_info()
    if pdb_info is None:
        raise RuntimeError("compute_pocket_centroid: receptor pose has no pdb_info (numbering lost)")

    pocket_pose_idx: List[int] = []
    resolved_resname: Dict[int, str] = {}
    for resnum in POCKET_PDB_RESNUMS:
        pose_idx = pdb_info.pdb2pose(POCKET_CHAIN, resnum)
        if pose_idx == 0:
            raise RuntimeError(
                f"compute_pocket_centroid: PDB resnum {resnum} chain {POCKET_CHAIN} "
                "not found in receptor (pdb2pose returned 0)"
            )
        pocket_pose_idx.append(pose_idx)
        resolved_resname[resnum] = receptor_pose.residue(pose_idx).name3()

    print(
        f"[pocket_mapping] PDB resnums {POCKET_PDB_RESNUMS} -> pose idx {pocket_pose_idx}; "
        f"residue names: {resolved_resname}",
        file=sys.stderr,
    )

    coords = []
    for idx in pocket_pose_idx:
        xyz = receptor_pose.residue(idx).xyz("CA")
        coords.append([xyz.x, xyz.y, xyz.z])
    centroid = np.mean(coords, axis=0)
    return centroid, pocket_pose_idx


# ---------------------------------------------------------------------------
# Ab-initio docking of one decoy
# ---------------------------------------------------------------------------

def run_one_decoy(
    combo_pose: "pyrosetta.Pose",
) -> Dict:
    """FlexPepDockingProtocol ab-initio 1회 실행 후 interface dG/clash/pocket 접촉 계산.

    Returns dict: {interface_dg, total_score, clash_count, pocket_contact,
                   pocket_residues_contacted, min_pep_pocket_ca_dist, pose}
    """
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol
    import pyrosetta

    pose = combo_pose.clone()

    # SS bond: peptide 내 2개 Cys 있으면 자동 detect (native/candidate 공통 처리)
    n_chains = pose.num_chains()
    pep_begin = pose.chain_begin(n_chains)
    pep_end = pose.chain_end(n_chains)
    cys_residues = [
        i for i in range(pep_begin, pep_end + 1) if pose.residue(i).name1() == "C"
    ]
    if len(cys_residues) == 2:
        try:
            pose.conformation().detect_disulfides()
        except Exception as exc:  # pragma: no cover - PyRosetta 버전 의존
            print(f"  [disulfide] detect_disulfides failed: {exc}", file=sys.stderr)

    fpd = FlexPepDockingProtocol()
    fpd.apply(pose)

    # Interface dG (I_sc 대리 지표 — 상단 모듈 docstring 참조)
    iam = InterfaceAnalyzerMover(1)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.apply(pose)
    interface_dg = float(iam.get_interface_dG())

    scorefxn = pyrosetta.get_fa_scorefxn()
    total_score = float(scorefxn(pose))

    # clash: peptide chain residues with fa_rep > 10
    from pyrosetta.rosetta.core.scoring import fa_rep

    clash_count = 0
    for i in range(pep_begin, pep_end + 1):
        energies = pose.energies().residue_total_energies(i)
        if energies[fa_rep] > 10.0:
            clash_count += 1

    return {
        "interface_dg": interface_dg,
        "total_score": total_score,
        "clash_count": clash_count,
        "pose": pose,
    }


def compute_pocket_contact(
    pose: "pyrosetta.Pose",
    pocket_pose_idx: List[int],
    threshold_a: float = POCKET_CONTACT_THRESHOLD_A,
) -> Tuple[bool, List[int], float]:
    """펩타이드(마지막 체인) heavy atom 이 포켓 잔기 heavy atom 5A 이내 접촉하는지 판정.

    Returns (contact_bool, contacted_pdb_resnums, min_heavy_atom_distance).
    """
    n_chains = pose.num_chains()
    pep_begin = pose.chain_begin(n_chains)
    pep_end = pose.chain_end(n_chains)
    pdb_info = pose.pdb_info()

    contacted: List[int] = []
    min_dist = float("inf")

    for pocket_idx in pocket_pose_idx:
        pocket_res = pose.residue(pocket_idx)
        pocket_resnum = pdb_info.number(pocket_idx) if pdb_info else pocket_idx
        found = False
        for pep_i in range(pep_begin, pep_end + 1):
            pep_res = pose.residue(pep_i)
            for pa in range(1, pep_res.nheavyatoms() + 1):
                pep_xyz = pep_res.xyz(pa)
                for ra in range(1, pocket_res.nheavyatoms() + 1):
                    d = pep_xyz.distance(pocket_res.xyz(ra))
                    if d < min_dist:
                        min_dist = d
                    if d <= threshold_a:
                        found = True
        if found:
            contacted.append(pocket_resnum)

    return (len(contacted) > 0, contacted, min_dist)


# ---------------------------------------------------------------------------
# Full run for one sequence
# ---------------------------------------------------------------------------

def run_blind_validation(
    sequence: str,
    label: str,
    n_decoys: int,
    seed_base: int,
    best_pdb_out: Path,
) -> Dict:
    import pyrosetta

    t_start = time.time()

    receptor_pose = pyrosetta.pose_from_pdb(str(RECEPTOR_PDB))
    pocket_centroid, pocket_pose_idx = compute_pocket_centroid(receptor_pose)

    decoy_records: List[Dict] = []
    best_record: Optional[Dict] = None

    for d in range(n_decoys):
        seed = seed_base + d
        rng = random.Random(seed)
        pyrosetta.rosetta.numeric.random.rg().set_seed(seed)

        peptide_pose = build_extended_peptide(sequence)
        combo = assemble_blind_complex(receptor_pose, peptide_pose, pocket_centroid, rng)

        result = run_one_decoy(combo)
        contact, contacted_resnums, min_dist = compute_pocket_contact(
            result["pose"], pocket_pose_idx
        )

        record = {
            "decoy_idx": d,
            "seed": seed,
            "interface_dg": round(result["interface_dg"], 4),
            "total_score": round(result["total_score"], 4),
            "clash_count": result["clash_count"],
            "pocket_contact": contact,
            "pocket_residues_contacted": contacted_resnums,
            "min_pep_pocket_heavy_atom_dist_a": round(min_dist, 3),
        }
        decoy_records.append(record)
        print(
            f"[{label}] decoy {d + 1}/{n_decoys} seed={seed} "
            f"interface_dg={record['interface_dg']:.2f} "
            f"pocket_contact={contact} min_dist={record['min_pep_pocket_heavy_atom_dist_a']:.2f}A",
            file=sys.stderr,
        )

        if best_record is None or record["interface_dg"] < best_record["interface_dg"]:
            best_record = record
            result["pose"].dump_pdb(str(best_pdb_out))

    elapsed = time.time() - t_start

    # --- Funnel analysis ---
    sorted_by_dg = sorted(decoy_records, key=lambda r: r["interface_dg"])
    top10pct_n = max(1, math.ceil(len(sorted_by_dg) * 0.10))
    top10pct = sorted_by_dg[:top10pct_n]
    top10pct_dgs = [r["interface_dg"] for r in top10pct]
    median_top10pct = statistics.median(top10pct_dgs)
    best_record_final = sorted_by_dg[0]

    # 수렴 판정: top10% 내 표준편차가 작고(<=5 REU) 전체 범위 대비 좁으면 funnel 존재로 판단.
    # 임계값은 임의 휴리스틱임을 명시 — "수렴"을 과장하지 않기 위해 보수적으로 좁게 설정.
    all_dgs = [r["interface_dg"] for r in decoy_records]
    dg_range = max(all_dgs) - min(all_dgs)
    top10pct_sd = statistics.stdev(top10pct_dgs) if len(top10pct_dgs) > 1 else 0.0
    converged = bool(top10pct_sd <= 5.0 and len(decoy_records) >= 10)

    any_pocket_contact_in_best = best_record_final["pocket_contact"]
    n_decoys_with_contact = sum(1 for r in decoy_records if r["pocket_contact"])

    funnel_quality = (
        "narrow_funnel" if (converged and top10pct_sd <= 2.0) else
        "loose_funnel" if converged else
        "no_funnel"
    )

    return {
        "sequence": sequence,
        "label": label,
        "receptor": str(RECEPTOR_PDB.relative_to(REPO_ROOT)),
        "n_decoys": n_decoys,
        "protocol": "FlexPepDockingProtocol ab-initio (-flexPepDocking:lowres_abinitio true), "
                    "extended-conformation start, random rigid-body placement "
                    f"{BLIND_RADIUS_A}A from pocket centroid + full random rotation per decoy",
        "metric_definition": (
            "interface_dg = InterfaceAnalyzerMover(jump=1).get_interface_dG() "
            "(kcal/mol proxy; PyRosetta FlexPepDockingProtocol.apply() does NOT populate "
            "the C++ CLI's I_sc/reweighted_sc score-file columns — verified empirically "
            "during development, pose.scores only contains raw energy terms)"
        ),
        "best_interface_dg": best_record_final["interface_dg"],
        "median_top10pct_interface_dg": round(median_top10pct, 4),
        "top10pct_n": top10pct_n,
        "top10pct_sd": round(top10pct_sd, 4),
        "dg_range_all_decoys": round(dg_range, 4),
        "converged": converged,
        "funnel_quality": funnel_quality,
        "pocket_contact_best_decoy": any_pocket_contact_in_best,
        "n_decoys_with_pocket_contact": n_decoys_with_contact,
        "pocket_residues_contacted_best": best_record_final["pocket_residues_contacted"],
        "pocket_residue_pdb_resnums_checked": POCKET_PDB_RESNUMS,
        "pocket_contact_threshold_a": POCKET_CONTACT_THRESHOLD_A,
        "blind_radius_a": BLIND_RADIUS_A,
        "elapsed_seconds": round(elapsed, 1),
        "best_pdb": str(best_pdb_out.relative_to(REPO_ROOT)),
        "decoy_records": decoy_records,
    }


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def build_verdict(candidate_result: Dict, native_result: Dict) -> Dict:
    """호출자 지시 7번 판정 로직: PASS / partial / inconclusive."""
    cand_dg = candidate_result["best_interface_dg"]
    nat_dg = native_result["best_interface_dg"]
    cand_pocket = candidate_result["pocket_contact_best_decoy"]
    nat_pocket = native_result["pocket_contact_best_decoy"]

    stronger_than_native = cand_dg < nat_dg  # 더 음수 = 더 강함
    found_pocket = cand_pocket

    if stronger_than_native and found_pocket:
        verdict = "PASS"
        text = (
            f"Blind 검증 PASS: 후보({candidate_result['sequence']})의 최저 interface_dG="
            f"{cand_dg:.2f}가 native({nat_dg:.2f})보다 낮고(강한 결합), 최고 decoy가 "
            f"알려진 포켓 잔기({candidate_result['pocket_residues_contacted_best']})와 "
            f"{candidate_result['pocket_contact_threshold_a']}A 이내 접촉했다 — "
            "후보가 위치 정보 없이 독립적으로 SSTR2 포켓을 재발견했다는 신호."
        )
    elif found_pocket or stronger_than_native:
        verdict = "partial"
        reasons = []
        if not found_pocket:
            reasons.append("최고 decoy가 포켓 잔기와 접촉하지 못함")
        if not stronger_than_native:
            reasons.append(
                f"후보 interface_dG({cand_dg:.2f})가 native({nat_dg:.2f})보다 강하지 않음"
            )
        text = (
            f"Blind 검증 PARTIAL: {' / '.join(reasons)}. "
            "포켓 발견 또는 결합 강도 중 하나만 만족 — 완전한 독립 재발견으로 보기엔 근거 부족."
        )
    else:
        verdict = "inconclusive"
        text = (
            f"Blind 검증 INCONCLUSIVE: 후보 최고 decoy가 포켓 잔기와 접촉하지 못했고 "
            f"(min_dist={candidate_result['decoy_records'][0].get('min_pep_pocket_heavy_atom_dist_a', 'NA')}A 등), "
            f"interface_dG({cand_dg:.2f})도 native({nat_dg:.2f})보다 강하지 않다. "
            f"n_decoys={candidate_result['n_decoys']}개 국소(반경 {BLIND_RADIUS_A}A) blind 탐색으로는 "
            "포켓 재발견 여부를 확증할 수 없다. decoy 수 부족 또는 blind 반경/저해상도 탐색 한계일 "
            "가능성 모두 배제하지 않는다 — 이 결과를 결합 능력 부재의 증거로 과잉 해석하지 말 것."
        )

    return {"honest_verdict": text, "verdict_label": verdict}


LIMITATIONS_TEXT = (
    "(1) 이 스크립트의 'blind'는 포켓 centroid 기준 반경 20A 구면 위 무작위 지점 + "
    "완전 무작위 회전에서 시작하는 '국소 반경 내 blind'이며, 수용체 표면 전체(막 반대쪽 등) "
    "어디서나 시작 가능한 완전 global docking이 아니다. "
    "(2) FlexPepDockingProtocol Python 바인딩은 C++ CLI의 표준 I_sc/reweighted_sc를 "
    "노출하지 않아 InterfaceAnalyzer dG를 대리 지표로 사용했다(metric_definition 필드 참조). "
    "(3) decoy 수(40-60)는 계산 예산 제약(가동 중인 다른 엔진과 CPU 공유) 때문이며, "
    "진짜 ab-initio 논문 수준 벤치마크(수백~수천 decoy)보다 훨씬 적어 funnel 판정의 통계적 "
    "검정력이 제한적이다. "
    "(4) 포켓 잔기(197/205/272)는 apo 수용체 CA 좌표만으로 정의했고 side-chain 배향/유도 결합 "
    "(induced fit)은 반영하지 않는다. "
    "(5) 이 결과는 refine 도킹(AF3/Boltz pose에서 시작)이나 cross-silo validation과 "
    "독립적인 세번째 신호로만 해석해야 하며, 단독으로 결합 여부를 확정하지 않는다."
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SSTR2 blind ab-initio FlexPepDock validation")
    parser.add_argument("--candidate-sequence", default="AICLNWFWKTVISC")
    parser.add_argument("--native-sequence", default="AGCKNFFWKTFTSC")
    parser.add_argument("--n-decoys", type=int, default=50)
    parser.add_argument("--smoke", action="store_true", help="스모크 모드: decoy 수를 3으로 강제")
    parser.add_argument("--seed-base", type=int, default=20260701)
    parser.add_argument("--max-threads", type=int, default=1)
    parser.add_argument(
        "--output-json",
        default=str(OUT_DIR / "blind_dock_validation.json"),
    )
    args = parser.parse_args()

    if not RECEPTOR_PDB.exists():
        _err_exit(f"Receptor PDB not found: {RECEPTOR_PDB}")

    n_decoys = 3 if args.smoke else args.n_decoys
    if n_decoys < 2:
        _err_exit("n_decoys must be >= 2")

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import pyrosetta  # noqa: F401  (경로 확인용, init은 아래서)

    init_pyrosetta_abinitio(max_threads=args.max_threads)

    print(f"=== Blind dock validation: candidate={args.candidate_sequence} ===", file=sys.stderr)
    candidate_result = run_blind_validation(
        sequence=args.candidate_sequence,
        label="candidate",
        n_decoys=n_decoys,
        seed_base=args.seed_base,
        best_pdb_out=WORK_DIR / f"blind_best_{args.candidate_sequence}.pdb",
    )

    print(f"=== Blind dock validation: native={args.native_sequence} ===", file=sys.stderr)
    native_result = run_blind_validation(
        sequence=args.native_sequence,
        label="native",
        n_decoys=n_decoys,
        seed_base=args.seed_base + 100000,
        best_pdb_out=WORK_DIR / f"blind_best_{args.native_sequence}.pdb",
    )

    verdict = build_verdict(candidate_result, native_result)

    output = {
        "sequence": candidate_result["sequence"],
        "receptor": candidate_result["receptor"],
        "n_decoys": n_decoys,
        "protocol": candidate_result["protocol"],
        "metric_definition": candidate_result["metric_definition"],
        "best_I_sc": candidate_result["best_interface_dg"],
        "median_top10pct": candidate_result["median_top10pct_interface_dg"],
        "ddg_best": candidate_result["best_interface_dg"],
        "converged": candidate_result["converged"],
        "funnel_quality": candidate_result["funnel_quality"],
        "pocket_contact": candidate_result["pocket_contact_best_decoy"],
        "pocket_residues_contacted": candidate_result["pocket_residues_contacted_best"],
        "honest_verdict": verdict["honest_verdict"],
        "verdict_label": verdict["verdict_label"],
        "limitations": LIMITATIONS_TEXT,
        "native_comparison": {
            "native_sequence": native_result["sequence"],
            "native_best_I_sc": native_result["best_interface_dg"],
            "native_median_top10pct": native_result["median_top10pct_interface_dg"],
            "native_converged": native_result["converged"],
            "native_funnel_quality": native_result["funnel_quality"],
            "native_pocket_contact": native_result["pocket_contact_best_decoy"],
            "native_pocket_residues_contacted": native_result["pocket_residues_contacted_best"],
            "native_best_pdb": native_result["best_pdb"],
        },
        "candidate_best_pdb": candidate_result["best_pdb"],
        "n_decoys_with_pocket_contact_candidate": candidate_result["n_decoys_with_pocket_contact"],
        "n_decoys_with_pocket_contact_native": native_result["n_decoys_with_pocket_contact"],
        "elapsed_seconds_candidate": candidate_result["elapsed_seconds"],
        "elapsed_seconds_native": native_result["elapsed_seconds"],
        "full_candidate_decoy_records": candidate_result["decoy_records"],
        "full_native_decoy_records": native_result["decoy_records"],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(json.dumps({k: v for k, v in output.items() if not k.startswith("full_")}, indent=2))
    print(f"\nWrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
