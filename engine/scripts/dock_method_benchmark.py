#!/usr/bin/env python3
"""
dock_method_benchmark.py
=========================
FlexPepDock 3-수용체 x 8-조건 도킹 방법론 벤치마크 (2026-07-02 3차 재설계).

이전 버전(단일 Boltz 복합체 3-조건 -> 7XNA/7T10 2-수용체 5-조건)에 이어,
**AF3 예측 복합체를 3번째 수용체 축으로 추가**했다(예측계열 7XNA·AF3 vs
실험 cryo-EM 7T10 비교 완성용). NO MOCK: 전부 실제 PyRosetta
FlexPepDockingProtocol 실행. 실패 run은 정직하게 스킵/기록한다.

수용체 3종
----------
- **7XNA** (`data/somatostatin_receptor/SSTR2_7XNA.pdb`): SSTR2+xylanase 융합체.
  chain A = 수용체(세포외 실제 도메인 + 세포내 융합부 1000-1184, 후자는
  세포외 포켓과 무관하므로 그대로 둠). chain B = "YKTC" 4잔기 조각(SST14 아님,
  결합 파트너 아님) -> **본 스크립트가 로드 전 텍스트 필터로 반드시 제거**.
  SST14 펩타이드가 아예 없으므로 조건 1/2에서는 7T10에서 후보 펩타이드를
  이식(graft)한다(아래 "펩타이드 이식" 참조). -> **2조건**(blind, pocket_known).
- **7T10** (`data/somatostatin_receptor/curated/SSTR2_SST14_complex_7t10.pdb`):
  실험 cryo-EM 복합체. chain A = 수용체(40-327), chain B = SST14
  "AGCKNFFWKTFTSC" (14aa). `prepare_complex_by_mutation`으로 chain B를
  후보 서열로 치환. -> **3조건**(blind, pocket_known, local_refine).
- **AF3** (`data/somatostatin_receptor/SSTR2_AF3analog_complex.pdb`, 2026-07-02
  추가): AF3 예측 복합체. **chain A = 펩타이드**(1-14, SST14 유사체
  "AGCKFDFWKTITSC" — native SST14 아님, FWKT pharmacophore(F7/W8/K9/T10)와
  Cys3-Cys14는 보존. 향후 native SST14 AF3로 교체 예정, 결과표 각주 필수
  명시), **chain B = 수용체**(1-369). 이미 펩타이드가 포함된 복합체라
  7XNA처럼 이식(transplant) 불필요 — 7T10과 동일하게
  `prepare_complex_by_mutation` + `reorder_peptide_last`로 처리(단
  peptide_chain=1, reorder가 실제로 chain 재배치를 수행함 — 7T10/7XNA는
  peptide가 이미 마지막 chain이라 no-op이었던 것과 다름. reorder 후에는
  수용체가 항상 새 chain 'A'로 재배정되므로 이후 pdb2pose 로직은 7T10과
  동일하게 재사용 가능). -> **3조건**(blind, pocket_known, local_refine).

포켓 잔기 numbering (pdb2pose 필수, 하드코딩 pose-index 금지)
--------------------------------------------------------------
문헌/JSON 기준 포켓 잔기는 PDB 번호 208/209/272/273/276 (기대 잔기명
F/I/F/Y/N — `data/somatostatin_receptor/curated/SSTR2_receptor.pdb`에서
이미 검증된 값과 동일). 두 수용체 PDB 모두 이 잔기명이 chain A에 그대로
존재함을 실측 확인했으나(2026-07-02 개발 중 offline 검증), 두 구조의 pose
내부 번호(1-indexed 연속 인덱스)는 서로 다르다(7XNA는 세포내 융합부
삽입으로 208/209 이후 pose 인덱스가 크게 점프함, 7T10은 연속적). 따라서
`pose.pdb_info().pdb2pose("A", resnum)` 로 매 실행마다 동적으로 매핑하고
(`compute_pocket_pose_ids`), 매핑된 잔기명이 기대값과 정확히 일치하는지
`validate_pocket_residue_identity()`가 재검증한다(하나라도 안 맞으면
RuntimeError로 즉시 중단 — 조용한 오귀속 방지).

포켓 중심(COM) — pocket_known 조건에 사용
-------------------------------------------
포켓 중심 = 잔기 208/209/272/273/276의 CA 좌표 평균(COM). 매 수용체마다
로드 직후 실측 좌표로 직접 계산한다(`compute_pocket_centroid`). 7XNA의
경우 `data/somatostatin_receptor/binding_pocket_SSTR2.json`에 이미 기록된
center(-4.6638,-28.535,50.8738)와 근사적으로 일치하는지 로그로 교차검증한다
(참고용, 실제 배치는 실측 COM을 사용).

펩타이드 이식 (조건 1,2 — 7XNA용) — 2026-07-02 2차 수정: 중심이동 -> 구조중첩
------------------------------------------------------------------------------
1차 구현("중심만 pocket_centroid로 이동, 회전은 7T10 좌표계 그대로 유지")은
스모크에서 7xna_pocket_known ddG=+1751 REU(심각한 clash) + disulfide BROKEN을
유발했다 — 7XNA와 7T10은 독립적으로 solve된 서로 다른 좌표계라 "중심만 맞추고
방향은 유지"하면 수용체 표면에 펩타이드가 임의의 각도로 박히기 때문이다.

수정된 방식(구조 중첩, superposition):
  1. 두 수용체 chain A에서 PDB 잔기번호 40-327 범위 중 **양쪽에 다 존재하고
     잔기명까지 일치**하는 CA 쌍을 전부 수집한다(`build_ca_alignment_pairs`) —
     2026-07-02 개발 중 실측: 269쌍 매칭, Kabsch 알고리즘으로 7T10->7XNA
     회전+병진(R,t) 계산 시 전체 RMSD 2.47A, 포켓 잔기 5개만 놓고 보면
     잔차 0.9~2.0A로 오히려 더 정확하게 겹침(포켓 영역이 구조적으로 잘
     보존됨을 시사).
  2. 7T10에서 이식한 후보 펩타이드(mutated, split_by_chain으로 분리)에 이
     (R,t)를 그대로 적용(`apply_rigid_transform_to_pose`) — 펩타이드는
     "7T10 수용체에 대해 결합해 있던 상대 위치/방향"을 유지한 채 7XNA
     좌표계로 통째로 이동한다. 이는 이제 "위치만 맞춘 배치"가 아니라
     "구조적으로 대응되는 실제 결합 자세로 옮겨진 배치"다.
  3. `7xna_pocket_known`은 이 중첩 결과를 그대로 사용(추가 강제 이동 없음,
     이미 포켓 위치/방향이 맞음) + tether constraint만 추가.
  4. `7xna_blind`은 중첩 결과에서 시작하되 포켓 중심 기준 ~20A 무작위 변위
     + 무작위 회전으로 의도적으로 흩뜨린다(변경 없음, 기존 로직 재사용).

5 조건
------
1. 7xna_blind        — 7XNA + 구조중첩 이식 펩타이드, 포켓 중심에서 ~20A
                        무작위 변위 + 무작위 회전 + lowres_abinitio 전역 탐색.
2. 7xna_pocket_known — 7XNA + 구조중첩 이식 펩타이드(그대로, 추가 강제이동
                        없음) + pharmacophore(Trp8 자리)-포켓 AtomPair
                        harmonic tether(8A, sd=2) + lowres_abinitio 탐색.
                        constraint 실패 시 "중첩 배치 + lowres"만 적용 각주.
3. 7t10_blind        — 7T10 SST14->후보 변이, 포켓 중심에서 ~20A 무작위 변위
                        + 무작위 회전 + lowres_abinitio 전역 탐색.
4. 7t10_pocket_known — 7T10 변이 펩타이드를 포켓 중심에 배치/유지(원래도
                        native 근처) + tether + lowres_abinitio 탐색.
5. 7t10_local_refine — 7T10 SST14->후보 변이, native 위치에서 highres refine
                        only (lowres_abinitio 미적용).

pocket_contacts — 게이트 제외, 리포트 전용 지표로 격하 (2026-07-02 2차 수정)
------------------------------------------------------------------------------
1차 스모크에서 `7t10_local_refine`이 pocket_contacts=0으로 게이트 실패했으나,
원인 진단 결과 **raw native(미도킹) 7T10 구조 자체가 이미 CA-CA 8A 임계값을
20쌍 중 1쌍(7.02A)만 겨우 통과**하는 것으로 확인됐다(numbering 버그 아님,
메트릭이 이 pharmacophore/pocket 조합에는 너무 타이트함). 따라서:
  - 게이트는 이제 pocket_contacts가 아니라 **물리적 타당성**(ddG 수렴 여부 +
    이황화 유지 + 비-clash)으로 판정한다(`aggregate_condition`의 n_converged
    정의 확장 + 스모크 게이트 재정의, 아래 참조).
  - pocket_contacts는 **다중 임계값(8/10/12A) 카운트 + CA 최근접거리 +
    side-chain heavy-atom 최근접거리**로 리포트 전용 지표화한다
    (`compute_pocket_metrics`). 반드시 **native SST14(raw, 미도킹 7T10)
    baseline과 나란히 비교**해서 "native 대비 얼마나 포켓을 유지하는가"로
    해석한다(`compute_native_baseline`).

양의 극단 clash도 unphysical로 포착 (2026-07-02 2차 수정)
------------------------------------------------------------
`_ddg_physical_floor()`(기본 -150)는 음의 극단만 걸러 +1751 REU 같은 심각한
clash를 통과시켰다. `POSITIVE_CLASH_REU_THRESHOLD`(기본 100.0 REU) 이상이면
`unphysical=True`로 마킹하고 n_converged 집계에서 제외한다. n_converged는
추가로 disulfide_intact가 명시적으로 False(끊어짐 확인됨)인 run도 제외한다
(None=미측정은 배제하지 않음).

FlexPepDockingProtocol Python 바인딩 관련 기존 발견 (2026-07-02, 이전 스모크)
------------------------------------------------------------------------------
이 PyRosetta 빌드(2026.20)의 FlexPepDockingProtocol에는 set_lowres_preoptimize()
같은 세터가 없다(dir() 확인, AttributeError). blind_dock_validation.py가 이미
검증한 방식대로 `-flexPepDocking:lowres_abinitio` 커맨드라인 옵션에 위임하되,
5개 조건이 한 프로세스에서 순차 실행되므로 매 run마다
`pyrosetta.rosetta.basic.options.set_boolean_option()`으로 런타임 토글한다.

시스템 부하 각주
----------------
실행 시점에 Silo A/B 무한 발굴 엔진이 CPU를 동시 점유 중이었다(`uptime`
load average 확인). 절대 wall_time은 부하 영향을 받으나, 80개 run을
순차(비동시) 실행했으므로 조건 간 상대 비교는 유효하다.

사용
----
conda run -n bio-tools python scripts/dock_method_benchmark.py --smoke
conda run -n bio-tools python scripts/dock_method_benchmark.py
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
sys.path.insert(0, str(REPO_ROOT))

from AG_src.scripts.flexpep_dock import (  # noqa: E402
    _add_disulfide_constraint,
    _check_disulfide_distance,
    _ddg_physical_floor,
    _find_peptide_cys_residues,
    compute_interface_ddg,
    init_pyrosetta,
    prepare_complex_by_mutation,
    reorder_peptide_last,
)
# 참고: flexpep_dock.py::compute_pocket_contacts()(단일 8A 임계값, CA-CA)는 이
# 벤치마크에서 native조차 임계값을 겨우 통과하는 것으로 확인되어(모듈 docstring
# "pocket_contacts — 게이트 제외" 참조) 다중 임계값 + heavy-atom 버전인
# compute_pocket_metrics() 로 대체했다. import하지 않음(사용 안 함).

SEVENXNA_PDB = REPO_ROOT / "data/somatostatin_receptor/SSTR2_7XNA.pdb"
SEVENXNA_CHAIN_TO_STRIP = "B"  # "YKTC" 4잔기 조각, SST14 아님
SEVENT10_COMPLEX_PDB = REPO_ROOT / "data/somatostatin_receptor/curated/SSTR2_SST14_complex_7t10.pdb"
AF3_COMPLEX_PDB = REPO_ROOT / "data/somatostatin_receptor/SSTR2_AF3analog_complex.pdb"
AF3_PEPTIDE_CHAIN = 1  # AF3: chain A(=1번째)=펩타이드, chain B=수용체 (7T10과 반대)
AF3_ANALOG_SEQUENCE = "AGCKFDFWKTITSC"  # native SST14 아님 — FWKT+Cys3/14 보존 유사체 (각주 필수)
BINDING_POCKET_JSON = REPO_ROOT / "data/somatostatin_receptor/binding_pocket_SSTR2.json"

POCKET_PDB_RESNUMS: List[int] = [208, 209, 272, 273, 276]
POCKET_RECEPTOR_CHAIN = "A"
_POCKET_EXPECTED_RESNAMES: Dict[int, str] = {208: "PHE", 209: "ILE", 272: "PHE", 273: "TYR", 276: "ASN"}
PHARMACOPHORE_ANCHOR_0IDX = 7  # 0-indexed peptide 내 위치 (1-indexed pos8, "Trp8" 자리)
PHARMACOPHORE_POSITIONS_0IDX: List[int] = [7, 8, 9, 10]  # W8,K9,T10,F11 (1-indexed), flexpep_dock.py와 동일 정의

# 7XNA<->7T10 구조 중첩(superposition)에 쓸 공통 CA 매칭 범위 (PDB 잔기번호, chain A)
ALIGNMENT_RESNUM_RANGE: Tuple[int, int] = (40, 327)

# 양의 극단 clash도 unphysical로 포착 (2026-07-02, 1751 REU 사고 이후 추가)
POSITIVE_CLASH_REU_THRESHOLD = 100.0

# pocket_contacts 리포트용 다중 임계값 (Å)
POCKET_CONTACT_THRESHOLDS: List[float] = [8.0, 10.0, 12.0]

SEQUENCES: List[str] = ["AICLNWFWKTVISC", "AMCKNFFWKTGTSC"]
CONDITION_SPECS: List[Dict[str, str]] = [
    {"label": "7xna_blind", "receptor": "7xna", "mode": "blind"},
    {"label": "7xna_pocket_known", "receptor": "7xna", "mode": "pocket_known"},
    {"label": "7t10_blind", "receptor": "7t10", "mode": "blind"},
    {"label": "7t10_pocket_known", "receptor": "7t10", "mode": "pocket_known"},
    {"label": "7t10_local_refine", "receptor": "7t10", "mode": "local_refine"},
    {"label": "af3_blind", "receptor": "af3", "mode": "blind"},
    {"label": "af3_pocket_known", "receptor": "af3", "mode": "pocket_known"},
    {"label": "af3_local_refine", "receptor": "af3", "mode": "local_refine"},
]
DEFAULT_NSTRUCT = 5
BLIND_RADIUS_A = 20.0

OUT_JSON = REPO_ROOT / "runs/pyrosetta_flow/dock_method_benchmark.json"
SCRATCH_DIR = REPO_ROOT / "runs/pyrosetta_flow/dock_method_benchmark_work"

_FOOTNOTES: List[str] = []


def _log(msg: str) -> None:
    print(msg, file=sys.stderr)


def _note(msg: str) -> None:
    if msg not in _FOOTNOTES:
        _FOOTNOTES.append(msg)
    _log(f"[FOOTNOTE] {msg}")


# ---------------------------------------------------------------------------
# 수용체 로드 + 정제
# ---------------------------------------------------------------------------

def _strip_chain(pdb_path: Path, chain_to_remove: str, out_path: Path) -> None:
    """PDB 텍스트에서 특정 chain의 ATOM/HETATM 라인을 제거하여 out_path에 기록."""
    with pdb_path.open() as fh:
        lines = fh.readlines()
    out_lines = [
        line
        for line in lines
        if not (line.startswith(("ATOM", "HETATM")) and line[21] == chain_to_remove)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(out_lines))
    _log(f"[strip_chain] {pdb_path.name}: chain {chain_to_remove} 제거 -> {out_path} ({len(out_lines)} lines)")


def load_7xna_receptor() -> "pyrosetta.Pose":
    """7XNA에서 chain B("YKTC" 조각) 제거 후 로드. chain A(수용체)만 남는다."""
    import pyrosetta

    cleaned = SCRATCH_DIR / "7xna_chainA_only.pdb"
    _strip_chain(SEVENXNA_PDB, SEVENXNA_CHAIN_TO_STRIP, cleaned)
    pose = pyrosetta.pose_from_pdb(str(cleaned))
    _log(f"[load_7xna] num_chains={pose.num_chains()} total_residue={pose.total_residue()}")
    if pose.num_chains() != 1:
        _note(
            f"7XNA 정제 후 chain 수={pose.num_chains()} (기대값 1) — chain B 제거가 "
            "의도대로 안 됐을 수 있음, 결과 해석 시 주의"
        )
    return pose


def load_7t10_template() -> "pyrosetta.Pose":
    """포켓 pose-index/centroid 계산용 원본(미변이) 7T10 복합체 로드."""
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(str(SEVENT10_COMPLEX_PDB))
    _log(f"[load_7t10_template] num_chains={pose.num_chains()} total_residue={pose.total_residue()}")
    return pose


def load_af3_template() -> "pyrosetta.Pose":
    """포켓 pose-index/centroid 계산용 원본(미변이) AF3 복합체 로드.

    AF3는 chain A=펩타이드(1), chain B=수용체 — 7T10/7XNA와 반대다.
    `reorder_peptide_last(pose, 1)`을 미리 적용해 "수용체=새 chain A,
    펩타이드=마지막 chain" 레이아웃으로 정규화한다. `build_af3_pose()`도
    매 서열마다 동일한 mutate+reorder 절차를 거치므로 여기서 계산한
    pocket_pose_ids를 그대로 재사용할 수 있다(레이아웃이 항상 동일하게
    결정론적으로 재현됨).
    """
    import pyrosetta

    pose = pyrosetta.pose_from_pdb(str(AF3_COMPLEX_PDB))
    _log(f"[load_af3_template] raw num_chains={pose.num_chains()} total_residue={pose.total_residue()}")
    pose = reorder_peptide_last(pose, AF3_PEPTIDE_CHAIN)
    _log(f"[load_af3_template] post-reorder num_chains={pose.num_chains()} total_residue={pose.total_residue()}")
    return pose


# ---------------------------------------------------------------------------
# 포켓 numbering (pdb2pose 필수) + 검증 + centroid
# ---------------------------------------------------------------------------

def compute_pocket_pose_ids(pose: "pyrosetta.Pose", label: str) -> List[int]:
    """pdb_info().pdb2pose()로 POCKET_PDB_RESNUMS -> pose 1-indexed 매핑.

    하드코딩 pose-index 금지(수용체마다 numbering이 다름) — 매 수용체 로드 후
    이 함수로 동적 계산한다.
    """
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise RuntimeError(f"[{label}] pose has no pdb_info; cannot map pocket residues")

    pocket_pose_ids: List[int] = []
    for resnum in POCKET_PDB_RESNUMS:
        pose_idx = pdb_info.pdb2pose(POCKET_RECEPTOR_CHAIN, resnum)
        if pose_idx == 0:
            raise RuntimeError(
                f"[{label}] pdb2pose({POCKET_RECEPTOR_CHAIN!r}, {resnum}) returned 0 "
                "(residue not found) — pocket numbering mismatch, 중단"
            )
        pocket_pose_ids.append(pose_idx)
    _log(f"[{label}] pocket PDB resnums {POCKET_PDB_RESNUMS} -> pose ids {pocket_pose_ids}")
    return pocket_pose_ids


def validate_pocket_residue_identity(pose: "pyrosetta.Pose", pocket_pose_ids: List[int], label: str) -> None:
    """pocket_pose_ids가 실제로 기대 잔기명(F/I/F/Y/N)을 가리키는지 검증.

    하나라도 안 맞으면 numbering이 어긋난 것 -> 조용히 진행하지 않고 즉시
    RuntimeError로 중단한다(호출자 지시: "0이면... 멈추고 보고"의 사전 예방).
    """
    mismatches = []
    for resnum, pose_idx in zip(POCKET_PDB_RESNUMS, pocket_pose_ids):
        expected = _POCKET_EXPECTED_RESNAMES[resnum]
        actual = pose.residue(pose_idx).name3()
        if actual != expected:
            mismatches.append(f"resnum {resnum} (pose {pose_idx}): expected {expected}, got {actual}")
    if mismatches:
        raise RuntimeError(f"[{label}] POCKET numbering 불일치: " + "; ".join(mismatches))
    _log(f"[{label}] [pocket_validation] OK — 전부 기대 잔기명과 일치")


def compute_pocket_centroid(pose: "pyrosetta.Pose", pocket_pose_ids: List[int]):
    import numpy as np

    coords = []
    for idx in pocket_pose_ids:
        xyz = pose.residue(idx).xyz("CA")
        coords.append([xyz.x, xyz.y, xyz.z])
    return np.mean(coords, axis=0)


# ---------------------------------------------------------------------------
# 7XNA<->7T10 구조 중첩 (Kabsch superposition) — 펩타이드 이식 좌표계 보정
# ---------------------------------------------------------------------------

def build_ca_alignment_pairs(
    pose_a: "pyrosetta.Pose",
    pose_b: "pyrosetta.Pose",
    chain: str = POCKET_RECEPTOR_CHAIN,
    resnum_range: Tuple[int, int] = ALIGNMENT_RESNUM_RANGE,
) -> List[Tuple[int, int, int]]:
    """두 pose에서 PDB 잔기번호가 겹치고 잔기명까지 일치하는 CA 쌍을 수집.

    Returns list of (resnum, pose_idx_in_a, pose_idx_in_b).
    """
    pdb_a = pose_a.pdb_info()
    pdb_b = pose_b.pdb_info()
    pairs: List[Tuple[int, int, int]] = []
    for resnum in range(resnum_range[0], resnum_range[1] + 1):
        idx_a = pdb_a.pdb2pose(chain, resnum)
        idx_b = pdb_b.pdb2pose(chain, resnum)
        if idx_a == 0 or idx_b == 0:
            continue
        res_a = pose_a.residue(idx_a)
        res_b = pose_b.residue(idx_b)
        if not res_a.has("CA") or not res_b.has("CA"):
            continue
        if res_a.name3() != res_b.name3():
            continue
        pairs.append((resnum, idx_a, idx_b))
    return pairs


def compute_ca_alignment_transform(
    pose_mobile: "pyrosetta.Pose",
    pose_target: "pyrosetta.Pose",
    label: str = "",
) -> Tuple["np.ndarray", "np.ndarray", float, int]:
    """Kabsch superposition: pose_mobile(예: 7T10) -> pose_target(예: 7XNA) 좌표계.

    Returns (R, t, rmsd, n_pairs) such that for a point p in pose_mobile's
    frame, R @ p + t ≈ 해당 위치 in pose_target's frame.
    """
    import numpy as np

    pairs = build_ca_alignment_pairs(pose_target, pose_mobile)
    if len(pairs) < 3:
        raise RuntimeError(
            f"[{label}] alignment pairs too few (n={len(pairs)}) — 구조 중첩 불가, 최소 3개 필요"
        )

    P = np.array([[pose_mobile.residue(ib).xyz("CA").x, pose_mobile.residue(ib).xyz("CA").y, pose_mobile.residue(ib).xyz("CA").z] for (_, ia, ib) in pairs])
    Q = np.array([[pose_target.residue(ia).xyz("CA").x, pose_target.residue(ia).xyz("CA").y, pose_target.residue(ia).xyz("CA").z] for (_, ia, ib) in pairs])

    centroid_p = P.mean(axis=0)
    centroid_q = Q.mean(axis=0)
    Pc = P - centroid_p
    Qc = Q - centroid_q
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = centroid_q - R @ centroid_p

    pred = (R @ P.T).T + t
    rmsd = float(np.sqrt(np.mean(np.sum((pred - Q) ** 2, axis=1))))
    _log(f"[{label}] CA superposition: n_pairs={len(pairs)} RMSD={rmsd:.3f}A (range={ALIGNMENT_RESNUM_RANGE})")

    # 포켓 잔기 5개만 놓고 fit 잔차 로그 (진단용)
    for resnum in POCKET_PDB_RESNUMS:
        ia = pose_target.pdb_info().pdb2pose(POCKET_RECEPTOR_CHAIN, resnum)
        ib = pose_mobile.pdb_info().pdb2pose(POCKET_RECEPTOR_CHAIN, resnum)
        if ia == 0 or ib == 0:
            continue
        p = np.array([pose_mobile.residue(ib).xyz("CA").x, pose_mobile.residue(ib).xyz("CA").y, pose_mobile.residue(ib).xyz("CA").z])
        q = np.array([pose_target.residue(ia).xyz("CA").x, pose_target.residue(ia).xyz("CA").y, pose_target.residue(ia).xyz("CA").z])
        residual = float(np.linalg.norm((R @ p + t) - q))
        _log(f"[{label}]   pocket resnum {resnum} fit residual = {residual:.3f}A")

    return R, t, rmsd, len(pairs)


def apply_rigid_transform_to_pose(pose: "pyrosetta.Pose", R: "np.ndarray", t: "np.ndarray") -> None:
    """pose의 모든 원자 좌표에 x -> R @ x + t 를 적용(in-place)."""
    import numpy as np
    from pyrosetta.rosetta.core.id import AtomID
    from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

    for i in range(1, pose.total_residue() + 1):
        res = pose.residue(i)
        for a in range(1, res.natoms() + 1):
            xyz = res.xyz(a)
            v = np.array([xyz.x, xyz.y, xyz.z])
            new_v = R @ v + t
            pose.set_xyz(AtomID(a, i), V3(float(new_v[0]), float(new_v[1]), float(new_v[2])))


# ---------------------------------------------------------------------------
# 포켓 접촉 지표 (리포트 전용, 다중 임계값 + heavy-atom) — native baseline과 비교
# ---------------------------------------------------------------------------

def compute_pocket_metrics(pose: "pyrosetta.Pose", pocket_pose_ids: List[int]) -> Dict:
    """pharmacophore(W8/K9/T10/F11, 0-indexed 7-10) vs pocket 잔기 간
    CA-CA 다중임계값(8/10/12A) 카운트 + CA 최근접거리 + heavy-atom 최근접거리.

    native baseline과 동일 함수로 계산해야 "포켓 유지 정도"를 정직하게 비교 가능.
    """
    n_chains = pose.num_chains()
    pep_begin = pose.chain_begin(n_chains)
    pep_end = pose.chain_end(n_chains)

    pharma_ids = [pep_begin + off for off in PHARMACOPHORE_POSITIONS_0IDX if pep_begin + off <= pep_end]
    result: Dict = {
        "pocket_min_ca_dist": None,
        "pocket_min_heavy_dist": None,
        **{f"pocket_contacts_{int(th)}": 0 for th in POCKET_CONTACT_THRESHOLDS},
    }
    if not pharma_ids or not pocket_pose_ids:
        return result

    ca_dists: List[float] = []
    heavy_dists: List[float] = []
    for p in pharma_ids:
        pep_res = pose.residue(p)
        if pep_res.has("CA"):
            pep_ca = pep_res.xyz("CA")
            for q in pocket_pose_ids:
                pocket_res = pose.residue(q)
                if pocket_res.has("CA"):
                    ca_dists.append(pep_ca.distance(pocket_res.xyz("CA")))
        for pa in range(1, pep_res.nheavyatoms() + 1):
            pxyz = pep_res.xyz(pa)
            for q in pocket_pose_ids:
                pocket_res = pose.residue(q)
                for qa in range(1, pocket_res.nheavyatoms() + 1):
                    heavy_dists.append(pxyz.distance(pocket_res.xyz(qa)))

    if ca_dists:
        result["pocket_min_ca_dist"] = round(min(ca_dists), 3)
        for th in POCKET_CONTACT_THRESHOLDS:
            result[f"pocket_contacts_{int(th)}"] = sum(1 for d in ca_dists if d <= th)
    if heavy_dists:
        result["pocket_min_heavy_dist"] = round(min(heavy_dists), 3)
    return result


# ---------------------------------------------------------------------------
# 이황화 처리 (flexpep_dock.py 기존 패턴)
# ---------------------------------------------------------------------------

def _handle_disulfide(pose: "pyrosetta.Pose", tag: str) -> None:
    cys_residues = _find_peptide_cys_residues(pose)
    if len(cys_residues) != 2:
        return
    try:
        pose.conformation().detect_disulfides()
        _log(f"  [{tag}] [disulfide] detect_disulfides() OK")
    except Exception as exc:
        _log(f"  [{tag}] [disulfide] detect_disulfides() failed ({exc}), manual fallback")
        try:
            _add_disulfide_constraint(pose, cys_residues[0], cys_residues[1])
        except Exception as exc2:
            _log(f"  [{tag}] [disulfide] manual constraint also failed ({exc2})")


# ---------------------------------------------------------------------------
# 7XNA용 펩타이드 이식
# ---------------------------------------------------------------------------

def build_7xna_combo(
    receptor_pose: "pyrosetta.Pose",
    sequence: str,
    alignment_R: "np.ndarray",
    alignment_t: "np.ndarray",
) -> "pyrosetta.Pose":
    """7T10에서 후보 펩타이드를 MutateResidue로 이식 -> 독립 Pose 분리 ->
    (alignment_R, alignment_t)로 7XNA 좌표계로 구조 중첩(rigid transform) ->
    7XNA 수용체에 append. 결과: chain1=수용체, chain2(마지막)=펩타이드.

    2026-07-02 2차 수정: 이전에는 펩타이드 중심만 pocket_centroid로 이동시켜
    ddG=+1751 REU 급 clash를 유발했다. 이제는 7T10<->7XNA 수용체 구조 중첩으로
    구한 (R,t)를 펩타이드에 그대로 적용해 "7T10에 결합해 있던 상대 자세"를
    유지한 채 7XNA 좌표계로 옮긴다(모듈 docstring "펩타이드 이식" 참조).
    """
    import pyrosetta
    from pyrosetta.rosetta.core.pose import append_pose_to_pose

    mut_pose, resolved_chain = prepare_complex_by_mutation(str(SEVENT10_COMPLEX_PDB), sequence, 2)
    pep_pose = mut_pose.split_by_chain()[resolved_chain]

    # 이식 이황화 유지 확인 (Cys3-Cys14) — rigid transform은 내부 거리를 보존하므로
    # transform 전/후 값은 동일하다. transform 전에 1회 확인.
    cys = [i for i in range(1, pep_pose.total_residue() + 1) if pep_pose.residue(i).name1() == "C"]
    if len(cys) == 2:
        try:
            pep_pose.conformation().detect_disulfides()
            intact, dist = _check_disulfide_distance(pep_pose, cys[0], cys[1])
            _log(f"[build_7xna_combo] seq={sequence} 이식 펩타이드 이황화 intact={intact} dist={dist:.3f}A")
            if not intact:
                _note(f"7XNA 이식({sequence}): 이식 직후 이황화가 이미 끊어짐(dist={dist:.3f}A)")
        except Exception as exc:
            _log(f"[build_7xna_combo] disulfide check failed: {exc}")
    else:
        _note(f"7XNA 이식({sequence}): 펩타이드에서 Cys 2개를 찾지 못함(found={len(cys)})")

    # 구조 중첩: 7T10 좌표계 -> 7XNA 좌표계
    apply_rigid_transform_to_pose(pep_pose, alignment_R, alignment_t)

    combo = pyrosetta.Pose()
    combo.assign(receptor_pose)
    append_pose_to_pose(combo, pep_pose, True)
    if combo.num_chains() != 2:
        raise RuntimeError(f"build_7xna_combo: expected 2 chains after append, got {combo.num_chains()}")
    return combo


def build_7t10_pose(sequence: str) -> "pyrosetta.Pose":
    pose, resolved_chain = prepare_complex_by_mutation(str(SEVENT10_COMPLEX_PDB), sequence, 2)
    pose = reorder_peptide_last(pose, resolved_chain)
    return pose


def build_af3_pose(sequence: str) -> "pyrosetta.Pose":
    """AF3 복합체의 펩타이드(chain A, AF3_ANALOG_SEQUENCE)를 후보 서열로 치환.

    이미 펩타이드가 포함된 복합체이므로 7XNA처럼 이식(transplant)이 필요
    없다 — 7T10과 동일한 mutate+reorder 절차, 단 peptide_chain=1(AF3는
    chain A가 펩타이드).
    """
    pose, resolved_chain = prepare_complex_by_mutation(str(AF3_COMPLEX_PDB), sequence, AF3_PEPTIDE_CHAIN)
    pose = reorder_peptide_last(pose, resolved_chain)
    return pose


# ---------------------------------------------------------------------------
# 펩타이드 배치 (blind: 무작위 변위+회전 / pocket_known: 중심 배치)
# ---------------------------------------------------------------------------

def place_peptide(
    pose: "pyrosetta.Pose",
    target_point,
    randomize_orientation: bool,
    seed: int,
    tag: str,
) -> str:
    """펩타이드(마지막 chain) 중심을 target_point(절대 xyz)로 이동.
    randomize_orientation=True면 추가로 RigidBodyRandomizeMover 적용.

    RigidBody 계열이 실패하면 저수준 좌표 직접 이동으로 폴백.
    Returns: "rigid_body" 또는 "manual_coord_fallback" 또는 "failed_no_placement".
    """
    import numpy as np
    import pyrosetta
    from pyrosetta.rosetta.numeric import xyzVector_double_t as V3

    n_chains = pose.num_chains()
    jump_num = 1
    pep_begin = pose.chain_begin(n_chains)
    pep_end = pose.chain_end(n_chains)

    try:
        from pyrosetta.rosetta.protocols.rigid import (
            Partner,
            RigidBodyRandomizeMover,
            RigidBodyTransMover,
        )

        coords = []
        for i in range(pep_begin, pep_end + 1):
            xyz = pose.residue(i).xyz("CA")
            coords.append([xyz.x, xyz.y, xyz.z])
        pep_centroid = np.mean(coords, axis=0)
        trans_vec = np.asarray(target_point) - pep_centroid
        dist = float(np.linalg.norm(trans_vec))

        if dist > 1e-6:
            direction = trans_vec / dist
            trans_mover = RigidBodyTransMover(pose, jump_num)
            trans_mover.trans_axis(V3(float(direction[0]), float(direction[1]), float(direction[2])))
            trans_mover.step_size(dist)
            trans_mover.apply(pose)

        if randomize_orientation:
            randomizer = RigidBodyRandomizeMover(pose, jump_num, Partner.partner_downstream)
            randomizer.apply(pose)

        _log(f"  [{tag}] [place] rigid_body OK dist={dist:.2f}A randomize={randomize_orientation}")
        return "rigid_body"
    except Exception as exc:
        _note(
            f"{tag}: RigidBody{{Trans,Randomize}}Mover 실패({type(exc).__name__}: {exc}) -> "
            "펩타이드 원자 좌표 직접 이동으로 폴백"
        )

    try:
        from pyrosetta.rosetta.core.id import AtomID

        rng = random.Random(seed + 999)
        centroid = np.zeros(3)
        n_res = pep_end - pep_begin + 1
        for i in range(pep_begin, pep_end + 1):
            xyz = pose.residue(i).xyz("CA")
            centroid += np.array([xyz.x, xyz.y, xyz.z])
        centroid /= max(1, n_res)
        shift = np.asarray(target_point) - centroid

        for i in range(pep_begin, pep_end + 1):
            res = pose.residue(i)
            for a in range(1, res.natoms() + 1):
                xyz = res.xyz(a)
                v = np.array([xyz.x, xyz.y, xyz.z]) - centroid
                if randomize_orientation:
                    v = np.array([-v[1], v[0], v[2]])  # 90도 축 순열 근사 회전
                new_xyz = centroid + v + shift
                pose.set_xyz(
                    AtomID(a, i),
                    V3(float(new_xyz[0]), float(new_xyz[1]), float(new_xyz[2])),
                )
        _log(f"  [{tag}] [place] manual_coord_fallback applied")
        return "manual_coord_fallback"
    except Exception as exc2:
        _note(f"{tag}: 저수준 좌표 폴백도 실패({type(exc2).__name__}: {exc2}) -> 배치 없이 진행")
        return "failed_no_placement"


def _random_unit_direction(seed: int):
    import numpy as np

    rng = random.Random(seed)
    u = rng.uniform(-1.0, 1.0)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    r_xy = math.sqrt(max(0.0, 1.0 - u * u))
    return np.array([r_xy * math.cos(theta), r_xy * math.sin(theta), u])


# ---------------------------------------------------------------------------
# pocket_known용 pharmacophore -> pocket AtomPair constraint
# ---------------------------------------------------------------------------

def add_pocket_tether_constraints(pose: "pyrosetta.Pose", pocket_pose_ids: List[int], tag: str) -> bool:
    try:
        from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
        from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
        from pyrosetta.rosetta.core.id import AtomID

        n_chains = pose.num_chains()
        pep_begin = pose.chain_begin(n_chains)
        pep_end = pose.chain_end(n_chains)
        anchor_resnum = pep_begin + PHARMACOPHORE_ANCHOR_0IDX
        if anchor_resnum > pep_end or not pose.residue(anchor_resnum).has("CA"):
            raise RuntimeError(f"pharmacophore anchor {anchor_resnum} invalid (pep range {pep_begin}-{pep_end})")

        n_added = 0
        for pocket_id in pocket_pose_ids:
            if not pose.residue(pocket_id).has("CA"):
                continue
            a1 = AtomID(pose.residue(anchor_resnum).atom_index("CA"), anchor_resnum)
            a2 = AtomID(pose.residue(pocket_id).atom_index("CA"), pocket_id)
            pose.add_constraint(AtomPairConstraint(a1, a2, HarmonicFunc(8.0, 2.0)))
            n_added += 1

        if n_added == 0:
            raise RuntimeError("no valid pocket residues to tether")
        _log(f"  [{tag}] [pocket_tether] anchor={anchor_resnum} -> {n_added} pocket residues, harmonic(8.0A, sd=2.0)")
        return True
    except Exception as exc:
        _log(f"  [{tag}] [pocket_tether] FAILED ({type(exc).__name__}: {exc}) -> fallback (no constraint)")
        return False


# ---------------------------------------------------------------------------
# 단일 run 실행
# ---------------------------------------------------------------------------

def run_single(
    base_pose: "pyrosetta.Pose",
    condition_label: str,
    receptor: str,
    mode: str,
    run_idx: int,
    seed_base: int,
    pocket_pose_ids: List[int],
    pocket_centroid,
) -> Dict:
    """조건별 단일 FlexPepDock run. 성공/실패 모두 dict로 반환 (raise 안 함)."""
    from pyrosetta.rosetta.protocols.flexpep_docking import FlexPepDockingProtocol

    tag = f"{condition_label}#{run_idx}"
    record: Dict = {
        "run_idx": run_idx,
        "condition": condition_label,
        "ok": False,
        "error": None,
        "fallback_used": None,
    }

    pose = base_pose.clone()
    _handle_disulfide(pose, tag)

    seed = seed_base + run_idx
    if mode == "blind":
        target = pocket_centroid + BLIND_RADIUS_A * _random_unit_direction(seed)
        record["fallback_used"] = place_peptide(pose, target, randomize_orientation=True, seed=seed, tag=tag)
    elif mode == "pocket_known":
        # 2026-07-02 3차 수정: 펩타이드는 이미 포켓 결합 위치에 있다
        # (7T10=native 결합 자세, 7XNA=superposition으로 포켓 안착).
        # pocket_centroid는 포켓 잔기(208/209/272/273/276) CA의 COM = TM 다발
        # '내부' 깊숙한 점이라, 거기로 펩타이드 COM을 옮기면 수용체에 파묻혀
        # 심각한 clash(관측: +372~+1751 REU, disulfide 파괴)가 난다.
        # 따라서 어느 수용체든 이동 없이 결합 위치를 유지하고 tether constraint만
        # 추가한다. (blind만 pocket 밖 무작위 시작을 쓴다)
        # 2026-07-05 조사(이슈#3): pocket_known은 발산 서열(예: AICLNWFWKTVISC)에서 0/5 미수렴.
        # 원인=(1)발산 side-chain이 native backbone 기하와 부적합→repack으로도 잔여 clash 미해결
        #      (2)lowres_abinitio 재접힘이 발산서열에서 좋은 pose로 수렴 실패.
        # → 버그 아닌 '방법 한계'. repack 선행은 일반적 해결책 아니라 미적용.
        # 신뢰할 방법은 local_refine(참조 복합체 정제)이며 발굴 파이프라인이 이를 사용.
        _log(f"  [{tag}] [place] pocket_known: 결합 위치 유지(이동 없음) + tether constraint")
        record["fallback_used"] = "keep_bound_pose"
        ok = add_pocket_tether_constraints(pose, pocket_pose_ids, tag)
        if not ok:
            _note(f"{condition_label}: constraint 추가 실패 -> '배치 유지 + lowres'만 적용(constraint 없음)")
            record["fallback_used"] = (record["fallback_used"] or "") + "+no_pocket_constraint"
    # mode == "local_refine": 배치 변경 없음 (native 위치 유지)

    floor_val = _ddg_physical_floor()
    t0 = time.time()
    try:
        # FlexPepDockingProtocol Python 바인딩에는 set_lowres_preoptimize() 세터가
        # 없다(이 PyRosetta 2026.20 빌드에서 AttributeError 확인). 대신
        # blind_dock_validation.py가 검증한 -flexPepDocking:lowres_abinitio 옵션을
        # 런타임 토글한다(모듈 docstring 참조).
        import pyrosetta.rosetta.basic.options as _opts

        _lowres_abinitio = mode != "local_refine"
        _opts.set_boolean_option("flexPepDocking:lowres_abinitio", _lowres_abinitio)
        fpd = FlexPepDockingProtocol()
        fpd.apply(pose)
    except (RuntimeError, ValueError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_time_s"] = round(time.time() - t0, 2)
        _log(f"  [{tag}] FAILED: {record['error']}")
        return record

    wall_time = time.time() - t0
    ddg = compute_interface_ddg(pose.clone())
    pocket_metrics = compute_pocket_metrics(pose, pocket_pose_ids)

    cys_residues = _find_peptide_cys_residues(pose)
    disulfide_intact: Optional[bool] = None
    sg_sg_distance: Optional[float] = None
    if len(cys_residues) == 2:
        try:
            disulfide_intact, sg_sg_distance = _check_disulfide_distance(pose, cys_residues[0], cys_residues[1])
            sg_sg_distance = round(sg_sg_distance, 4)
        except Exception as exc:
            _log(f"  [{tag}] disulfide distance check failed: {exc}")

    # 2026-07-02 2차 수정: 음의 극단(floor) 뿐 아니라 양의 극단(clash, 예: +1751 REU)도
    # unphysical로 포착 — POSITIVE_CLASH_REU_THRESHOLD 참조.
    unphysical = (ddg <= floor_val) or (ddg > POSITIVE_CLASH_REU_THRESHOLD)

    record.update(
        {
            "ok": True,
            "ddg": round(ddg, 4),
            "disulfide_intact": disulfide_intact,
            "sg_sg_distance": sg_sg_distance,
            "wall_time_s": round(wall_time, 2),
            "unphysical": bool(unphysical),
        }
    )
    record.update(pocket_metrics)
    _log(
        f"  [{tag}] OK ddg={ddg:.2f} pocket_contacts_8={pocket_metrics.get('pocket_contacts_8')} "
        f"pocket_min_ca_dist={pocket_metrics.get('pocket_min_ca_dist')} "
        f"disulfide_intact={disulfide_intact} wall_time={wall_time:.1f}s unphysical={unphysical}"
    )
    return record


# ---------------------------------------------------------------------------
# 조건별 집계
# ---------------------------------------------------------------------------

def aggregate_condition(records: List[Dict]) -> Dict:
    ok_records = [r for r in records if r["ok"]]
    n_total = len(records)
    n_failed = n_total - len(ok_records)

    # 2026-07-02 2차 수정: "유효 run" = ddG 수렴(floor<ddg<0) AND not unphysical
    # (floor 이하 또는 +clash 극단 아님) AND disulfide가 명시적으로 끊어지지
    # 않음(None=미측정은 배제하지 않음).
    floor_val = _ddg_physical_floor()
    valid_records = [
        r
        for r in ok_records
        if r["ddg"] is not None
        and floor_val < r["ddg"] < 0
        and not r["unphysical"]
        and r.get("disulfide_intact") is not False
    ]
    physical_ddgs = [r["ddg"] for r in valid_records]
    all_wall_times = [r["wall_time_s"] for r in records if r.get("wall_time_s") is not None]
    disulfide_flags = [r["disulfide_intact"] for r in ok_records if r["disulfide_intact"] is not None]

    min_ca_vals = [r["pocket_min_ca_dist"] for r in ok_records if r.get("pocket_min_ca_dist") is not None]
    min_heavy_vals = [r["pocket_min_heavy_dist"] for r in ok_records if r.get("pocket_min_heavy_dist") is not None]

    agg: Dict = {
        "n_total": n_total,
        "n_ok": len(ok_records),
        "n_failed": n_failed,
        "n_converged": len(physical_ddgs),
        "ddg_median": round(statistics.median(physical_ddgs), 4) if physical_ddgs else None,
        "ddg_mean": round(statistics.mean(physical_ddgs), 4) if physical_ddgs else None,
        "ddg_sd": round(statistics.stdev(physical_ddgs), 4) if len(physical_ddgs) > 1 else (0.0 if physical_ddgs else None),
        "ddg_min": round(min(physical_ddgs), 4) if physical_ddgs else None,
        "pocket_min_ca_dist_mean": round(statistics.mean(min_ca_vals), 3) if min_ca_vals else None,
        "pocket_min_heavy_dist_mean": round(statistics.mean(min_heavy_vals), 3) if min_heavy_vals else None,
        "disulfide_intact_ratio": round(sum(1 for f in disulfide_flags if f) / len(disulfide_flags), 3)
        if disulfide_flags
        else None,
        "total_wall_time_s": round(sum(all_wall_times), 2) if all_wall_times else None,
        "mean_wall_time_per_run_s": round(statistics.mean(all_wall_times), 2) if all_wall_times else None,
    }
    for th in POCKET_CONTACT_THRESHOLDS:
        key = f"pocket_contacts_{int(th)}"
        vals = [r[key] for r in ok_records if key in r]
        agg[f"{key}_mean"] = round(statistics.mean(vals), 3) if vals else None
    return agg


# ---------------------------------------------------------------------------
# native SST14 baseline (raw, 미도킹) — pocket_contacts 리포트 비교 기준
# ---------------------------------------------------------------------------

def compute_native_baseline(template_pose: "pyrosetta.Pose", pocket_pose_ids: List[int]) -> Dict:
    """7T10 raw(미변이, 미도킹) 복합체에서 ddG/pocket 지표를 그대로 계산.

    FlexPepDockingProtocol을 전혀 적용하지 않은 "있는 그대로의 실험 구조"
    기준선 — "native조차 이 메트릭으로 pocket_contacts가 낮다"는 사실을
    본 벤치마크 조건들과 나란히 비교할 수 있게 한다.
    """
    label = "native_sst14_7t10_raw"
    pose = template_pose.clone()
    if pose.num_chains() != 2:
        raise RuntimeError(f"[{label}] expected 2 chains, got {pose.num_chains()}")

    _handle_disulfide(pose, label)
    cys_residues = _find_peptide_cys_residues(pose)
    disulfide_intact: Optional[bool] = None
    sg_sg_distance: Optional[float] = None
    if len(cys_residues) == 2:
        try:
            disulfide_intact, sg_sg_distance = _check_disulfide_distance(pose, cys_residues[0], cys_residues[1])
            sg_sg_distance = round(sg_sg_distance, 4)
        except Exception as exc:
            _log(f"[{label}] disulfide distance check failed: {exc}")

    t0 = time.time()
    ddg = compute_interface_ddg(pose.clone())
    wall_time = time.time() - t0
    pocket_metrics = compute_pocket_metrics(pose, pocket_pose_ids)

    floor_val = _ddg_physical_floor()
    unphysical = (ddg <= floor_val) or (ddg > POSITIVE_CLASH_REU_THRESHOLD)

    record: Dict = {
        "label": label,
        "sequence": "AGCKNFFWKTFTSC",
        "note": "raw native cryo-EM 구조, FlexPepDock 미적용(도킹/리파인 없음) — 순수 기하/에너지 기준선",
        "ddg": round(ddg, 4),
        "disulfide_intact": disulfide_intact,
        "sg_sg_distance": sg_sg_distance,
        "wall_time_s": round(wall_time, 2),
        "unphysical": bool(unphysical),
    }
    record.update(pocket_metrics)
    _log(
        f"[{label}] ddg={ddg:.2f} pocket_contacts_8={pocket_metrics.get('pocket_contacts_8')} "
        f"pocket_min_ca_dist={pocket_metrics.get('pocket_min_ca_dist')} disulfide_intact={disulfide_intact}"
    )
    return record


# ---------------------------------------------------------------------------
# 서열 1개 x 조건 1개 실행
# ---------------------------------------------------------------------------

def run_condition_for_sequence(
    sequence: str,
    spec: Dict[str, str],
    nstruct: int,
    seed_base: int,
    receptor_bases: Dict[str, "pyrosetta.Pose"],
    receptor_pocket_ids: Dict[str, List[int]],
    receptor_pocket_centroids: Dict[str, object],
) -> Dict:
    label = spec["label"]
    receptor = spec["receptor"]
    mode = spec["mode"]
    _log(f"=== sequence={sequence} condition={label} (receptor={receptor}, mode={mode}) nstruct={nstruct} ===")

    base_pose = receptor_bases[receptor]
    pocket_ids = receptor_pocket_ids[receptor]
    pocket_centroid = receptor_pocket_centroids[receptor]

    records: List[Dict] = []
    for i in range(nstruct):
        rec = run_single(base_pose, label, receptor, mode, i, seed_base, pocket_ids, pocket_centroid)
        records.append(rec)

    agg = aggregate_condition(records)
    return {"sequence": sequence, "condition": label, "receptor": receptor, "mode": mode, "runs": records, "aggregate": agg}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FlexPepDock 3-수용체 x 8-조건 도킹 방법론 벤치마크")
    parser.add_argument("--smoke", action="store_true", help="스모크: nstruct=1, 서열 1개, 8조건 전부")
    parser.add_argument("--nstruct", type=int, default=DEFAULT_NSTRUCT)
    parser.add_argument("--seed-base", type=int, default=20260702)
    parser.add_argument("--output-json", default=str(OUT_JSON))
    args = parser.parse_args()

    if not SEVENXNA_PDB.exists():
        print(json.dumps({"error": f"7XNA PDB not found: {SEVENXNA_PDB}"}), file=sys.stderr)
        sys.exit(1)
    if not SEVENT10_COMPLEX_PDB.exists():
        print(json.dumps({"error": f"7T10 complex PDB not found: {SEVENT10_COMPLEX_PDB}"}), file=sys.stderr)
        sys.exit(1)
    if not AF3_COMPLEX_PDB.exists():
        print(json.dumps({"error": f"AF3 complex PDB not found: {AF3_COMPLEX_PDB}"}), file=sys.stderr)
        sys.exit(1)

    _note(
        f"AF3 축의 펩타이드는 native SST14(AGCKNFFWKTFTSC)가 아니라 유사체 "
        f"({AF3_ANALOG_SEQUENCE}, FWKT pharmacophore + Cys3-Cys14 보존)이다. "
        "AF3 축의 의미는 '이 유사체에 대해 AF3가 예측한 수용체 conformation'이며, "
        "향후 native SST14 AF3 예측으로 교체 예정."
    )

    sequences = ["AMCKNFFWKTGTSC"] if args.smoke else SEQUENCES
    nstruct = 1 if args.smoke else args.nstruct

    _log(
        f"### dock_method_benchmark start: smoke={args.smoke} sequences={sequences} "
        f"conditions={[s['label'] for s in CONDITION_SPECS]} nstruct={nstruct} ###"
    )
    _log(
        "### 시스템 부하 각주: 동시 실행 중인 Silo A/B 엔진으로 인해 절대 wall_time은 "
        "부하 영향을 받음. 작업 순차 실행으로 상대 비교는 유효. ###"
    )

    init_pyrosetta()

    # --- 수용체 1회 로드 + 포켓 numbering/centroid 계산 ---
    recA_7xna = load_7xna_receptor()
    pocket_ids_7xna = compute_pocket_pose_ids(recA_7xna, "7xna")
    validate_pocket_residue_identity(recA_7xna, pocket_ids_7xna, "7xna")
    centroid_7xna = compute_pocket_centroid(recA_7xna, pocket_ids_7xna)
    _log(f"[7xna] pocket centroid (실측) = {centroid_7xna.tolist()}")
    if BINDING_POCKET_JSON.exists():
        try:
            ref = json.loads(BINDING_POCKET_JSON.read_text())
            _log(
                f"[7xna] JSON 참고 center = ({ref.get('center_x')},{ref.get('center_y')},{ref.get('center_z')}) "
                "(source=7XNA, 교차검증 참고용)"
            )
        except Exception:
            pass

    template_7t10 = load_7t10_template()
    pocket_ids_7t10 = compute_pocket_pose_ids(template_7t10, "7t10")
    validate_pocket_residue_identity(template_7t10, pocket_ids_7t10, "7t10")
    centroid_7t10 = compute_pocket_centroid(template_7t10, pocket_ids_7t10)
    _log(f"[7t10] pocket centroid (실측) = {centroid_7t10.tolist()}")

    template_af3 = load_af3_template()
    pocket_ids_af3 = compute_pocket_pose_ids(template_af3, "af3")
    validate_pocket_residue_identity(template_af3, pocket_ids_af3, "af3")
    centroid_af3 = compute_pocket_centroid(template_af3, pocket_ids_af3)
    _log(f"[af3] pocket centroid (실측) = {centroid_af3.tolist()}")

    receptor_pocket_ids = {"7xna": pocket_ids_7xna, "7t10": pocket_ids_7t10, "af3": pocket_ids_af3}
    receptor_pocket_centroids = {"7xna": centroid_7xna, "7t10": centroid_7t10, "af3": centroid_af3}

    # --- (A) 7T10<->7XNA 구조 중첩(superposition) — 펩타이드 이식 좌표계 보정 ---
    alignment_R, alignment_t, alignment_rmsd, alignment_n_pairs = compute_ca_alignment_transform(
        pose_mobile=template_7t10, pose_target=recA_7xna, label="7t10->7xna"
    )

    # --- native SST14 baseline (raw, 미도킹) ---
    native_baseline = compute_native_baseline(template_7t10, pocket_ids_7t10)

    t_start = time.time()
    results: List[Dict] = []

    for sequence in sequences:
        _log(f"### building base poses for sequence={sequence} ###")
        try:
            combo_7xna = build_7xna_combo(recA_7xna, sequence, alignment_R, alignment_t)
        except Exception as exc:
            _log(f"### BASE-POSE FAILURE (7xna) sequence={sequence}: {type(exc).__name__}: {exc} ###")
            combo_7xna = None
        try:
            pose_7t10 = build_7t10_pose(sequence)
        except Exception as exc:
            _log(f"### BASE-POSE FAILURE (7t10) sequence={sequence}: {type(exc).__name__}: {exc} ###")
            pose_7t10 = None
        try:
            pose_af3 = build_af3_pose(sequence)
        except Exception as exc:
            _log(f"### BASE-POSE FAILURE (af3) sequence={sequence}: {type(exc).__name__}: {exc} ###")
            pose_af3 = None

        receptor_bases = {"7xna": combo_7xna, "7t10": pose_7t10, "af3": pose_af3}

        for spec in CONDITION_SPECS:
            if receptor_bases[spec["receptor"]] is None:
                results.append(
                    {
                        "sequence": sequence,
                        "condition": spec["label"],
                        "receptor": spec["receptor"],
                        "mode": spec["mode"],
                        "runs": [],
                        "aggregate": {"n_total": 0, "n_ok": 0, "error": "base pose construction failed"},
                    }
                )
                continue
            try:
                res = run_condition_for_sequence(
                    sequence,
                    spec,
                    nstruct,
                    args.seed_base,
                    receptor_bases,
                    receptor_pocket_ids,
                    receptor_pocket_centroids,
                )
            except Exception as exc:
                _log(
                    f"### CONDITION-LEVEL FAILURE sequence={sequence} condition={spec['label']}: "
                    f"{type(exc).__name__}: {exc} ###"
                )
                res = {
                    "sequence": sequence,
                    "condition": spec["label"],
                    "receptor": spec["receptor"],
                    "mode": spec["mode"],
                    "runs": [],
                    "aggregate": {"n_total": 0, "n_ok": 0, "error": f"{type(exc).__name__}: {exc}"},
                }
            results.append(res)

    elapsed_total = time.time() - t_start

    output = {
        "sevenxna_pdb": str(SEVENXNA_PDB),
        "sevent10_complex_pdb": str(SEVENT10_COMPLEX_PDB),
        "af3_complex_pdb": str(AF3_COMPLEX_PDB),
        "af3_analog_sequence": AF3_ANALOG_SEQUENCE,
        "pocket_pdb_resnums": POCKET_PDB_RESNUMS,
        "pocket_pose_ids": {"7xna": pocket_ids_7xna, "7t10": pocket_ids_7t10, "af3": pocket_ids_af3},
        "pocket_centroid": {
            "7xna": centroid_7xna.tolist(),
            "7t10": centroid_7t10.tolist(),
            "af3": centroid_af3.tolist(),
        },
        "alignment_7t10_to_7xna": {
            "rmsd_angstrom": round(alignment_rmsd, 3),
            "n_pairs": alignment_n_pairs,
            "resnum_range": list(ALIGNMENT_RESNUM_RANGE),
            "note": "Kabsch CA superposition, 7T10 수용체 -> 7XNA 수용체 좌표계. "
            "이 (R,t)를 7T10에서 이식한 후보 펩타이드에 적용해 7XNA 포켓에 안착.",
        },
        "positive_clash_reu_threshold": POSITIVE_CLASH_REU_THRESHOLD,
        "ddg_physical_floor": _ddg_physical_floor(),
        "pocket_contact_thresholds_angstrom": POCKET_CONTACT_THRESHOLDS,
        "pharmacophore_anchor_note": "anchor = peptide 0-indexed pos7 (1-indexed pos8, 'Trp8' 자리; 서열 무관 위치기반)",
        "native_baseline": native_baseline,
        "sequences": sequences,
        "condition_specs": CONDITION_SPECS,
        "nstruct": nstruct,
        "smoke": args.smoke,
        "seed_base": args.seed_base,
        "elapsed_seconds_total": round(elapsed_total, 1),
        "system_load_caveat": (
            "실행 시점에 Silo A/B 무한 발굴 엔진이 CPU를 동시 점유(uptime load average 확인). "
            "절대 wall_time은 부하 영향을 받으나, 작업을 순차(비동시) 실행했으므로 "
            "조건 간 상대 비교는 유효하다."
        ),
        "footnotes": list(_FOOTNOTES),
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    _log(f"### DONE elapsed_total={elapsed_total:.1f}s -> {out_path} ###")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, indent=2))

    # --- 스모크 전용 gate (2026-07-02 2차 수정): pocket_contacts 게이트 폐지.
    # 대신 물리적 타당성(ddG 수렴 가능 범위 + not-clash + disulfide 유지)을 게이트.
    if args.smoke:
        gate_failures = []
        for res in results:
            agg = res.get("aggregate", {})
            if "error" in agg:
                gate_failures.append(f"{res['condition']}: base/condition 실패 - {agg['error']}")
                continue
            for r in res.get("runs", []):
                if not r.get("ok"):
                    gate_failures.append(f"{res['condition']} run#{r['run_idx']}: FlexPepDock 실패 - {r.get('error')}")
                    continue
                if r.get("unphysical"):
                    gate_failures.append(
                        f"{res['condition']} run#{r['run_idx']}: unphysical ddg={r.get('ddg')} "
                        f"(floor={_ddg_physical_floor()}, clash_threshold={POSITIVE_CLASH_REU_THRESHOLD})"
                    )
                if r.get("disulfide_intact") is False:
                    gate_failures.append(f"{res['condition']} run#{r['run_idx']}: disulfide BROKEN (dist={r.get('sg_sg_distance')}A)")
        if native_baseline.get("unphysical"):
            gate_failures.append(f"native_baseline: unphysical ddg={native_baseline.get('ddg')} (예상 밖)")

        if gate_failures:
            _log("### SMOKE GATE FAILED: 물리적으로 타당하지 않은 run 발견 ###")
            for f in gate_failures:
                _log(f"  - {f}")
            sys.exit(2)

        _log("### SMOKE GATE PASSED: 8조건 전부 unphysical/disulfide-broken 없음 ###")
        _log(
            f"[native baseline] pocket_contacts_8={native_baseline.get('pocket_contacts_8')} "
            f"min_ca_dist={native_baseline.get('pocket_min_ca_dist')} ddg={native_baseline.get('ddg')}"
        )
        for res in results:
            ok_runs = [r for r in res.get("runs", []) if r.get("ok")]
            if ok_runs:
                r = ok_runs[0]
                _log(
                    f"[{res['condition']}] ddg={r.get('ddg')} pocket_contacts_8={r.get('pocket_contacts_8')} "
                    f"min_ca_dist={r.get('pocket_min_ca_dist')} unphysical={r.get('unphysical')}"
                )


if __name__ == "__main__":
    main()
