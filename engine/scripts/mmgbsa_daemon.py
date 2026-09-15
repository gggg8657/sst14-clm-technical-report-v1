#!/usr/bin/env python3
"""
mmgbsa_daemon.py — 주기 MM-GBSA 재채점 데몬 (v3: Silo A de novo 후보 포함).

동작:
  1. 주기(기본 10분)마다 Silo B 리더보드 top N(기본 100) + Silo A top M(기본 5) 채점
     [Silo B]
     a) docs/structure_view/pdb/ 에 있으면 그대로 사용 (docked, 정확)
     b) 없으면 PyRosetta MutateResidue로 native ref(chain B)에 변이 적용
        → runs/pyrosetta_flow/mmgbsa_pdb_cache/ 저장 (mutated_approx, 근사)
     c) D-aa/DOTA OOD 후보, cyclic 파괴 변이 → skip + 사유 기록
     [Silo A]
     d) runs/silo_a_flow/epoch_{ts}/silo_a_{ts}_bb{bb}_sq{sq}.pdb 직접 사용
        (de novo 복합체 — MutateResidue 불가, native scaffold 미사용)
     e) 복합체 PDB 없으면 skip + 사유 기록
  2. native도 매 주기 채점 (baseline 분산 추적)
  3. CUDA_VISIBLE_DEVICES=2 (GPU2)로 OpenMM MM-GBSA 채점
  4. 결과를 runs/pyrosetta_flow/mmgbsa_consensus.json에 원자적 기록
     - structure_source: "docked" | "mutated_approx" | "native" | "silo_a_docked"
     - source: "silo_b" | "silo_a" | "native"
  5. fail-closed: CUDA OOM/PyRosetta 실패 시 skip·재시도

실행(setsid 세션독립):
  setsid ~/miniforge3/envs/mmgbsa/bin/python \\
    scripts/mmgbsa_daemon.py >> logs/mmgbsa_daemon.log 2>&1 &

환경:
  mmgbsa env: openmm 8.5+, pdbfixer (MM-GBSA 채점)
  bio-tools env: PyRosetta (MutateResidue 구조 생성, Silo B만)
  → PyRosetta는 subprocess 호출(bio-tools env)로 격리하여 환경 충돌 방지.
  CUDA_VISIBLE_DEVICES=2 는 이 스크립트 내에서 설정됨.

caveat(구조):
  mutated_approx: MutateResidue(side-chain 치환, backbone 미재도킹) 근사 구조.
  도킹 pose가 아님 — 상대비교 참고용.
  docked: docs/structure_view/pdb/ 실제 도킹 pose. 더 신뢰성 높음.
  silo_a_docked: Silo A de novo 복합체 — chain B=16~20aa(SST-14 14aa 상이),
    ddG non-robust(단일 스냅샷, Silo B 5회 중앙값 아님).
    Silo B와 직접 비교 불가. consensus_flag 계산 시 주의 요망.

불가침:
  - surrogate 7파일 수정 금지.
  - 리더보드 JSON 쓰기 금지(데몬은 mmgbsa_consensus.json과 pdb_cache만 씀).
  - ROOT([LOCAL_PATH]) 밖 접근 금지.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path("[LOCAL_PATH]")
REPO = ROOT / "AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"
PDB_DIR = ROOT / "docs/structure_view/pdb"
PDB_CACHE_DIR = REPO / "runs/pyrosetta_flow/mmgbsa_pdb_cache"

LEADERBOARD_PATH = REPO / "runs/pyrosetta_flow/global_selectivity_leaderboard.json"
SILO_A_LEADERBOARD_PATH = REPO / "runs/silo_a_flow/silo_a_leaderboard.json"
SILO_A_FLOW_DIR = REPO / "runs/silo_a_flow"
CONSENSUS_PATH = REPO / "runs/pyrosetta_flow/mmgbsa_consensus.json"
LOG_PATH = REPO / "logs/mmgbsa_daemon.log"

# native 복합체 ref PDB (chain A=수용체, chain B=SST-14 펩타이드)
NATIVE_PDB_PATH = PDB_DIR / "native_AGCKNFFWKTFTSC.pdb"

# bio-tools env Python (PyRosetta 보유)
BIOTOOLS_PYTHON = "[LOCAL_PATH]"

# ── GPU 설정 (GPU2 전용, CUDA_VISIBLE_DEVICES 기준 device_index="0") ──────────
CUDA_DEVICE_ENV = "2"
DEVICE_INDEX = "0"

# ── 주기 설정 ──────────────────────────────────────────────────────────────────
CYCLE_SECONDS: int = 600         # 10분 (GPU2 유휴 활용, 전체 리더보드 커버)
TOP_N: int = 100                 # Silo B 리더보드 top N (2026-06-29: 50 → 100 확대)
SILO_A_TOP_N: int = 5            # Silo A 리더보드 top M (de novo, non-robust)

# ── canonical native ddG baseline (v2 검증값) ─────────────────────────────────
# native_robust_baseline_v2.json — 진짜 native PDB + nstruct=10 + physical floor 적용
# n_converged=8/10, 신뢰도=HIGH. 데몬이 매 사이클 이 값을 우선 사용.
# 파일 없을 시: Silo B 리더보드 native_ddg_median fallback + 경고 출력.
NATIVE_ROBUST_BASELINE_V2_PATH = REPO / "runs/pyrosetta_flow/native_robust_baseline_v2.json"
_NATIVE_DDG_V2_FALLBACK: float = -20.28   # v2 대표값 (파일 읽기 실패 최후 fallback)

# ── native 서열 (Cys3-Cys14 SS bond 참조) ──────────────────────────────────────
NATIVE_SEQ = "AGCKNFFWKTFTSC"
NATIVE_PDB_NAME = "native_AGCKNFFWKTFTSC"

# ── 표준 20종 AA (MutateResidue 대상, D-aa/비표준 제외) ────────────────────────
STANDARD_AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")

# ── 방법론 caveat ──────────────────────────────────────────────────────────────
CAVEAT = (
    "single-snapshot MM-GBSA(no MD, no TΔS): "
    "AMBER14+GBn2 implicit solvent, single pose, "
    "entropy not included, variance unknown. "
    "Cyclic SS bond via pdbfixer. 상대비교 전용."
)
CAVEAT_MUTATED = (
    "mutated_approx: MutateResidue(side-chain 치환, backbone 미재도킹) 근사 구조. "
    "도킹 pose가 아님 — 도킹 구조 대비 구조 오차 존재. "
    "relative 비교 참고용, 절대값 해석 금지."
)
CAVEAT_SILO_A = (
    "Silo A de novo: RFdiffusion/DiffPepBuilder 설계, 16~20aa(SST-14 14aa 상이). "
    "도킹 ddG 단일 스냅샷 non-robust(Silo B 5회 중앙값 아님). "
    "MutateResidue 불가(native scaffold 미사용). "
    "Silo B와 직접 비교 불가 — consensus_flag 계산 시 주의. "
    "MM-GBSA는 chain B(펩타이드)·chain A(수용체) 기준이나 "
    "16aa 구조이므로 SST-14 14aa 대비 결합 계면 상이."
)

# candidate_id 파싱 패턴: silo_a_{timestamp}_bb{bb}_sq{sq}
_SILO_A_CAND_RE = re.compile(r"silo_a_(\d{8}T\d{6}Z)_bb(\d+)_sq(\d+)$")

sys.path.insert(0, str(REPO))


# ── 로깅 ────────────────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    """로그 설정 (파일 전용 — stdout은 setsid 기동 시 파일로 리다이렉트됨)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s [mmgbsa-daemon] %(message)s")
    fh = logging.FileHandler(str(LOG_PATH), encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    root.setLevel(logging.INFO)


# ── JSON 유틸 ────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Optional[dict]:
    """JSON 파일 graceful 로드."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logging.warning("JSON 로드 실패 %s: %s", path, e)
        return None


def _save_json(path: Path, data: dict) -> None:
    """JSON 원자적 쓰기 (tmp → rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.rename(path)


# ── OOD / skip 판정 ──────────────────────────────────────────────────────────────

def _is_ood_skip(sequence: str, leaderboard_entry: Optional[dict]) -> tuple[bool, str]:
    """MutateResidue 불가 후보 판정.

    skip 조건:
      1. D-aa/DOTA OOD 플래그 (extra_scores.is_ood + ood_reasons에 d_amino_acid|dota 포함)
      2. 서열에 비표준 AA (표준 20종 외 문자) 포함
      3. cyclic Cys SS bond 파괴: pos3 또는 pos14 가 비-Cys ('C' 아님)
         → MutateResidue로 Cys 제거 시 disulfide topology 파괴 → pdbfixer 실패 가능
         ※ cyclic 자체는 skip 아님(native도 cyclic), Cys를 다른 AA로 바꾸는 경우만.

    Returns:
        (should_skip: bool, reason: str)
    """
    if not sequence or len(sequence) < 14:
        return True, "서열 길이 부족(<14)"

    # 표준 AA 외 문자
    non_std = [aa for aa in sequence if aa not in STANDARD_AA_SET]
    if non_std:
        return True, f"비표준AA: {non_std}"

    # Cys SS bond 파괴 여부 (pos3=idx2, pos14=idx13)
    if sequence[2] != "C" or sequence[13] != "C":
        return True, (
            f"CysSS 파괴 (pos3={sequence[2]}, pos14={sequence[13]}) "
            "→ pdbfixer disulfide 처리 불가"
        )

    # D-aa/DOTA OOD (leaderboard entry extra_scores)
    if leaderboard_entry:
        es = leaderboard_entry.get("extra_scores") or {}
        is_ood = leaderboard_entry.get("is_ood") or es.get("is_ood")
        ood_reasons = leaderboard_entry.get("ood_reasons") or es.get("ood_reasons") or []
        if is_ood:
            ood_str = ", ".join(ood_reasons) if ood_reasons else "OOD"
            return True, f"D-aa/DOTA OOD: {ood_str}"

    return False, ""


# ── PyRosetta MutateResidue (subprocess, bio-tools env) ──────────────────────────

_MUTATE_SCRIPT = """
import sys, json
import pyrosetta
pyrosetta.init('-mute all', silent=True)
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta import pose_from_pdb

native_pdb = sys.argv[1]
target_seq = sys.argv[2]
out_pdb    = sys.argv[3]
native_seq = "AGCKNFFWKTFTSC"

aa_1to3 = {
    'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS',
    'Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE',
    'L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO',
    'S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL',
}

pose = pose_from_pdb(native_pdb)
chain_b_start = pose.chain_begin(2)
chain_b_end   = pose.chain_end(2)
pep_len = chain_b_end - chain_b_start + 1

if pep_len != len(target_seq):
    print(json.dumps({"ok": False, "error": f"길이 불일치: pep={pep_len} target={len(target_seq)}"}))
    sys.exit(1)

for i, (nat, tgt) in enumerate(zip(native_seq, target_seq)):
    if nat != tgt:
        res_idx = chain_b_start + i
        aa3 = aa_1to3.get(tgt)
        if aa3 is None:
            print(json.dumps({"ok": False, "error": f"알 수 없는 AA: {tgt}"}))
            sys.exit(1)
        MutateResidue(res_idx, aa3).apply(pose)

pose.dump_pdb(out_pdb)

chain_b_start2 = pose.chain_begin(2)
chain_b_end2   = pose.chain_end(2)
result_seq = ''.join(pose.residue(i).name1() for i in range(chain_b_start2, chain_b_end2+1))
ok = result_seq == target_seq
print(json.dumps({"ok": ok, "result_seq": result_seq, "target_seq": target_seq}))
"""


def _build_complex_pdb_by_mutation(sequence: str) -> Optional[Path]:
    """PyRosetta MutateResidue로 native → target 복합체 PDB 생성.

    bio-tools env의 Python을 subprocess로 호출(환경 분리 — mmgbsa env와 충돌 방지).
    생성 파일: PDB_CACHE_DIR/{sequence}.pdb

    Args:
        sequence: 목표 서열 (14aa, 표준 AA, Cys3·Cys14 보존)

    Returns:
        생성된 PDB 경로, 실패 시 None
    """
    PDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_pdb = PDB_CACHE_DIR / f"{sequence}.pdb"

    # 이미 캐시에 있으면 재사용
    if out_pdb.exists():
        logging.info("pdb_cache 캐시 재사용: %s", out_pdb.name)
        return out_pdb

    if not NATIVE_PDB_PATH.exists():
        logging.error("native PDB 미존재: %s", NATIVE_PDB_PATH)
        return None

    # 인라인 스크립트를 임시 파일로 작성 후 subprocess 실행
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(_MUTATE_SCRIPT)
        tmp_script = tf.name

    try:
        proc = subprocess.run(
            [
                BIOTOOLS_PYTHON, tmp_script,
                str(NATIVE_PDB_PATH), sequence, str(out_pdb),
            ],
            capture_output=True, text=True, timeout=120,
        )
        # 마지막 줄에서 JSON 결과 파싱
        stdout_lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if not stdout_lines:
            logging.error(
                "MutateResidue subprocess 출력 없음 (seq=%s) stderr=%s",
                sequence, proc.stderr[:300],
            )
            return None

        result = json.loads(stdout_lines[-1])
        if not result.get("ok"):
            logging.error("MutateResidue 실패 (seq=%s): %s", sequence, result.get("error"))
            return None

        logging.info("MutateResidue 성공: %s → %s", sequence, out_pdb.name)
        return out_pdb

    except subprocess.TimeoutExpired:
        logging.error("MutateResidue 타임아웃 (seq=%s)", sequence)
        return None
    except Exception as exc:
        logging.error("MutateResidue 예외 (seq=%s): %s", sequence, exc)
        return None
    finally:
        try:
            os.unlink(tmp_script)
        except OSError:
            pass


# ── PDB 확보 (docked 우선, 없으면 mutated_approx) ─────────────────────────────────

def _find_or_build_pdb(
    sequence: str,
    leaderboard_entry: Optional[dict],
) -> tuple[Optional[Path], str]:
    """복합체 PDB 경로 반환.

    우선순위:
      1. docs/structure_view/pdb/ (docked, 정확) → structure_source="docked"
      2. MutateResidue 생성 (mutated_approx, 근사) → structure_source="mutated_approx"
      3. skip 사유(OOD/비표준AA/CysSS파괴/PyRosetta실패) → (None, skip_reason)

    Returns:
        (pdb_path, structure_source) — 실패 시 (None, reason)
    """
    # 1) docs/structure_view/pdb/ 탐색 (native 포함)
    if PDB_DIR.exists():
        for p in PDB_DIR.glob("*.pdb"):
            if p.stem.startswith("ot_"):
                continue
            if sequence in p.stem:
                return p, "docked"

    # 2) OOD/비표준/CysSS 파괴 판정 → skip
    should_skip, skip_reason = _is_ood_skip(sequence, leaderboard_entry)
    if should_skip:
        return None, f"skip:{skip_reason}"

    # 3) MutateResidue로 생성
    pdb_path = _build_complex_pdb_by_mutation(sequence)
    if pdb_path is None:
        return None, "PyRosetta MutateResidue 실패"
    return pdb_path, "mutated_approx"


# ── Silo A PDB 탐색 ──────────────────────────────────────────────────────────────

def _find_silo_a_pdb(candidate_id: str) -> Optional[Path]:
    """Silo A candidate_id에서 FlexPepDock 복합체 PDB를 찾아 반환.

    매핑 규칙:
      candidate_id = silo_a_{ts}_bb{bb}_sq{sq}
      epoch_dir    = runs/silo_a_flow/epoch_{ts}/
      복합체 PDB   = silo_a_{ts}_bb{bb}_sq{sq}.pdb  (FlexPepDock 산출)
      fallback     = diffpep_{ts}_bb{bb}_sq{sq}.pdb  (DiffPepBuilder 산출)

    Returns:
        복합체 PDB 경로(원본 epoch 디렉토리), 못 찾으면 None
    """
    m = _SILO_A_CAND_RE.match(candidate_id)
    if m is None:
        logging.warning("Silo A candidate_id 파싱 실패: %s", candidate_id)
        return None

    ts, bb, sq = m.group(1), m.group(2), m.group(3)
    epoch_dir = SILO_A_FLOW_DIR / f"epoch_{ts}"
    if not epoch_dir.exists():
        logging.warning("Silo A epoch 디렉토리 없음: %s", epoch_dir)
        return None

    # FlexPepDock 복합체 우선, fallback DiffPepBuilder
    for prefix in ("silo_a", "diffpep"):
        candidate_pdb = epoch_dir / f"{prefix}_{ts}_bb{bb}_sq{sq}.pdb"
        if candidate_pdb.exists():
            return candidate_pdb

    logging.warning(
        "Silo A 복합체 PDB 없음 (epoch=%s, bb=%s, sq=%s)", ts, bb, sq
    )
    return None


# ── Silo A 리더보드 로드 ─────────────────────────────────────────────────────────

def _load_silo_a_top(n: int) -> list[dict]:
    """silo_a_leaderboard.json에서 top N 엔트리 반환.

    정렬: ddg 오름차순(낮을수록 강결합=상위).

    Returns:
        list of dict with candidate_id, sequence, ddg, plddt, selectivity_margin, ...
    """
    lb = _load_json(SILO_A_LEADERBOARD_PATH)
    if lb is None:
        logging.warning("Silo A 리더보드 로드 실패: %s", SILO_A_LEADERBOARD_PATH)
        return []
    entries = lb.get("entries", [])
    if not entries:
        logging.warning("Silo A 리더보드 entries 비어있음")
        return []

    def _sort_key(e: dict) -> float:
        v = e.get("ddg")
        return v if isinstance(v, (int, float)) else float("inf")

    return sorted(entries, key=_sort_key)[:n]


# ── MM-GBSA 채점 ─────────────────────────────────────────────────────────────────

def _run_one(
    pdb_path: Path,
    name: str,
    sequence: str,
    structure_source: str,
    source: str = "silo_b",
) -> Optional[dict]:
    """단일 복합체 MM-GBSA 실행. 실패 시 None 반환(fail-closed).

    Args:
        pdb_path: 복합체 PDB 경로
        name: 후보 식별자
        sequence: 펩타이드 서열
        structure_source: "docked" | "mutated_approx" | "native" | "silo_a_docked"
        source: "silo_b" | "silo_a" | "native"
    """
    from pyrosetta_flow.mmgbsa_rescore import mmgbsa_rescore

    logging.info(
        "MM-GBSA 시작: %s (%s) [%s, source=%s]",
        name, pdb_path.name, structure_source, source,
    )
    try:
        result = mmgbsa_rescore(
            str(pdb_path),
            receptor_chains=["A"],   # chain A = SSTR2 수용체
            peptide_chain="B",       # chain B = 펩타이드
            platform_name="CUDA",
            device_index=DEVICE_INDEX,
            max_minimize_iterations=500,
        )
    except Exception as exc:
        logging.error("mmgbsa_rescore 예외 %s: %s", name, exc)
        return None

    if not result.get("ok"):
        logging.warning("MM-GBSA 실패 %s: %s", name, result.get("error", "")[:200])
        return None

    dg = result["dg_bind"]
    elapsed = result["elapsed_s"]
    logging.info(
        "완료 %s [%s/%s]: ΔG_bind=%.2f kcal/mol (%.1fs)",
        name, structure_source, source, dg, elapsed,
    )

    # caveat 조합
    caveat_used = CAVEAT
    if structure_source == "mutated_approx":
        caveat_used = CAVEAT + " | " + CAVEAT_MUTATED
    if source == "silo_a":
        caveat_used = caveat_used + " | " + CAVEAT_SILO_A

    dg_sd = result.get("dg_bind_sd")

    return {
        "name": name,
        "sequence": sequence,
        "source": source,
        "dg_bind": round(dg, 3),
        "dg_bind_sd": round(dg_sd, 4) if dg_sd is not None else None,
        "n_snapshots": result.get("n_snapshots", 1),
        "ss_bond_verified": result.get("ss_bond_verified", False),
        "e_complex": round(result["e_complex"], 3) if result.get("e_complex") is not None else None,
        "e_receptor": round(result["e_receptor"], 3) if result.get("e_receptor") is not None else None,
        "e_peptide": round(result["e_peptide"], 3) if result.get("e_peptide") is not None else None,
        "elapsed_s": round(elapsed, 1),
        "platform_used": result.get("platform_used"),
        "structure_source": structure_source,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caveat": caveat_used,
    }


# ── Consensus 플래그 ─────────────────────────────────────────────────────────────

def _compute_consensus_flag(
    entry_dg: float,
    entry_ddg: Optional[float],
    native_dg: Optional[float],
    native_ddg: Optional[float],
    dg_bind_sd: Optional[float] = None,
    uncertain_sd_threshold: float = 5.0,
) -> str:
    """도킹 ddG + MM-GBSA ΔG_bind 합의 플래그.

    high_confidence : 도킹 ddG < native_ddg AND MM-GBSA dg_bind < native_dg (분산 기준 미달 시)
    docking_only    : 도킹만 native 능가
    mmgbsa_only     : MM-GBSA만 native 능가
    uncertain       : MM-GBSA dg_bind_sd > uncertain_sd_threshold (분산 과대 — 신뢰 불가)
    none            : 둘 다 native 미달 또는 데이터 없음

    Args:
        entry_dg: 후보 MM-GBSA ΔG_bind, kcal/mol
        entry_ddg: 후보 도킹 ddG, kcal/mol (없으면 None)
        native_dg: native MM-GBSA ΔG_bind, kcal/mol (없으면 None)
        native_ddg: native 도킹 ddG, kcal/mol (없으면 None)
        dg_bind_sd: MM-GBSA ΔG_bind stdev (n_snapshots>=2일 때). None이면 분산 무시.
        uncertain_sd_threshold: 이 값(kcal/mol) 초과 시 uncertain 분류 (기본 5.0)
    """
    # 분산 과대 → uncertain (신뢰 불가, 다른 플래그보다 우선)
    if dg_bind_sd is not None and dg_bind_sd > uncertain_sd_threshold:
        return "uncertain"

    docking_beats = (
        entry_ddg is not None
        and native_ddg is not None
        and entry_ddg < native_ddg
    )
    mmgbsa_beats = native_dg is not None and entry_dg < native_dg
    if docking_beats and mmgbsa_beats:
        return "high_confidence"
    if docking_beats:
        return "docking_only"
    if mmgbsa_beats:
        return "mmgbsa_only"
    return "none"


# ── canonical native ddG v2 로더 ────────────────────────────────────────────────

def _load_native_ddg_v2() -> float:
    """native_robust_baseline_v2.json에서 canonical native ddG를 읽어 반환.

    우선순위:
      1. NATIVE_ROBUST_BASELINE_V2_PATH::ddg_median (진짜 native PDB + nstruct=10, HIGH)
      2. 파일 없거나 ddg_median 결측: Silo B 리더보드 native_ddg_median fallback + 경고
      3. 모두 실패: _NATIVE_DDG_V2_FALLBACK (-20.28) 반환 + ERROR 로그

    Returns:
        canonical native ddG (float, 항상 음수 ≤ 0 기대)

    Side-effects:
        logging.warning / logging.error 출력 (파일 없거나 fallback 사용 시)
    """
    # 1순위: v2 파일
    if NATIVE_ROBUST_BASELINE_V2_PATH.exists():
        v2 = _load_json(NATIVE_ROBUST_BASELINE_V2_PATH)
        if v2 is not None:
            val = v2.get("ddg_median")
            try:
                result = float(val)
                logging.info(
                    "native_ddg canonical v2 로드 완료: %.4f REU "
                    "(source=native_robust_baseline_v2.json, n_converged=%s, reliability=%s)",
                    result,
                    v2.get("n_converged", "?"),
                    v2.get("reliability", "?"),
                )
                return result
            except (TypeError, ValueError):
                logging.warning(
                    "native_robust_baseline_v2.json ddg_median 파싱 실패 (val=%r) "
                    "→ Silo B 리더보드 fallback 시도",
                    val,
                )
        else:
            logging.warning(
                "native_robust_baseline_v2.json 로드 실패 → Silo B 리더보드 fallback 시도"
            )
    else:
        logging.warning(
            "native_robust_baseline_v2.json 미존재 (%s) → Silo B 리더보드 fallback 시도",
            NATIVE_ROBUST_BASELINE_V2_PATH,
        )

    # 2순위: Silo B 리더보드 native_ddg_median
    lb = _load_json(LEADERBOARD_PATH)
    if lb is not None:
        val = lb.get("native_ddg_median")
        try:
            result = float(val)
            logging.warning(
                "[FALLBACK] native_ddg = Silo B 리더보드 native_ddg_median=%.4f REU "
                "(v2 파일 없음 — 오염 위험: n_converged 불명, 신뢰 낮음)",
                result,
            )
            return result
        except (TypeError, ValueError):
            pass

    # 3순위: 하드코딩 fallback
    logging.error(
        "[FALLBACK-FINAL] native_ddg = _NATIVE_DDG_V2_FALLBACK=%.2f REU "
        "(v2 파일 미존재 + Silo B 리더보드 native_ddg_median 결측)",
        _NATIVE_DDG_V2_FALLBACK,
    )
    return _NATIVE_DDG_V2_FALLBACK


# ── consensus 로드 ────────────────────────────────────────────────────────────────

def _load_consensus() -> dict:
    """기존 consensus JSON 로드, 없으면 빈 구조 반환."""
    existing = _load_json(CONSENSUS_PATH)
    if existing is None:
        return {
            "caveat": CAVEAT,
            "generated_at": None,
            "native_dg": None,
            "results": {},
        }
    return existing


# ── 메인 사이클 ──────────────────────────────────────────────────────────────────

def run_cycle(top_n: int = TOP_N, silo_a_top_n: int = SILO_A_TOP_N) -> None:
    """1회 주기 실행.

    native + Silo B top N + Silo A top M 복합체 PDB 확보 →
    MM-GBSA 채점 → mmgbsa_consensus.json 원자적 기록.

    Args:
        top_n: Silo B 리더보드 top N
        silo_a_top_n: Silo A 리더보드 top M
    """
    logging.info(
        "=== MM-GBSA 데몬 사이클 시작 (Silo B top_n=%d, Silo A top_n=%d) ===",
        top_n, silo_a_top_n,
    )

    # ── canonical native ddG: v2 파일 우선, 없으면 Silo B 리더보드 fallback ────
    native_ddg: Optional[float] = _load_native_ddg_v2()

    # Silo B 리더보드 로드 (읽기 전용)
    lb = _load_json(LEADERBOARD_PATH)
    if lb is None:
        logging.error("Silo B 리더보드 로드 실패 → 사이클 스킵")
        return

    entries = lb.get("entries", [])

    # Silo B entry → leaderboard dict 맵 (OOD 판정용)
    entry_map: dict[str, dict] = {
        e.get("sequence", ""): e for e in entries if e.get("sequence")
    }

    # Silo B 처리 대상: (name, sequence, "silo_b") — native 항상 포함
    targets: list[tuple[str, str, str]] = [(NATIVE_PDB_NAME, NATIVE_SEQ, "native")]
    for e in entries[:top_n]:
        seq = e.get("sequence", "")
        if seq and seq != NATIVE_SEQ:
            targets.append((f"top_{seq}", seq, "silo_b"))

    # Silo A 처리 대상: (candidate_id, sequence, "silo_a")
    silo_a_entries = _load_silo_a_top(silo_a_top_n)
    silo_a_targets: list[tuple[str, str, str]] = []
    for e in silo_a_entries:
        cid = e.get("candidate_id", "")
        seq = e.get("sequence", "")
        if cid and seq:
            silo_a_targets.append((cid, seq, "silo_a"))

    logging.info(
        "Silo B 대상 %d개 (native 1 + top%d) + Silo A %d개: %s …",
        len(targets), top_n, len(silo_a_targets),
        ", ".join(t[1][:8] for t in targets[:3]),
    )

    # 기존 consensus 로드
    consensus = _load_consensus()
    results_map: dict[str, dict] = consensus.get("results") or {}
    native_dg_prev: Optional[float] = consensus.get("native_dg")

    new_count = 0
    skip_count = 0
    native_dg_new: Optional[float] = native_dg_prev

    # ── Silo B + native 채점 ────────────────────────────────────────────────────
    for name, seq, src in targets:
        lb_entry = entry_map.get(seq)
        pdb_path, structure_source = _find_or_build_pdb(seq, lb_entry)

        if pdb_path is None:
            logging.warning("Silo B skip [%s]: %s", seq, structure_source)
            skip_count += 1
            if seq not in results_map:
                results_map[seq] = {
                    "name": name, "sequence": seq,
                    "source": src,
                    "dg_bind": None,
                    "structure_source": "skipped",
                    "skip_reason": structure_source,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            continue

        r = _run_one(pdb_path, name, seq, structure_source, source=src)
        if r is None:
            logging.warning("MM-GBSA 실패 → skip: %s", name)
            skip_count += 1
            continue

        if seq == NATIVE_SEQ:
            native_dg_new = r["dg_bind"]
            logging.info("native ΔG_bind 갱신: %.3f kcal/mol", native_dg_new)

        results_map[seq] = r
        new_count += 1

    # ── Silo A 채점 ─────────────────────────────────────────────────────────────
    silo_a_scored = 0
    silo_a_skipped = 0
    for cid, seq, src in silo_a_targets:
        # Silo A key: candidate_id 기준 (서열 중복 가능)
        result_key = f"silo_a:{cid}"
        pdb_path = _find_silo_a_pdb(cid)

        if pdb_path is None:
            logging.warning("Silo A skip [%s]: 복합체 PDB 없음", cid)
            silo_a_skipped += 1
            if result_key not in results_map:
                results_map[result_key] = {
                    "name": cid,
                    "candidate_id": cid,
                    "sequence": seq,
                    "source": "silo_a",
                    "dg_bind": None,
                    "structure_source": "skipped",
                    "skip_reason": "Silo A 복합체 PDB 없음",
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "caveat": CAVEAT_SILO_A,
                }
            continue

        r = _run_one(pdb_path, cid, seq, "silo_a_docked", source="silo_a")
        if r is None:
            logging.warning("Silo A MM-GBSA 실패 → skip: %s", cid)
            silo_a_skipped += 1
            continue

        r["candidate_id"] = cid
        results_map[result_key] = r
        silo_a_scored += 1
        new_count += 1

    logging.info(
        "Silo A: 채점=%d 스킵=%d", silo_a_scored, silo_a_skipped
    )

    # ── consensus 플래그 계산 ────────────────────────────────────────────────────
    for key, r in results_map.items():
        if r.get("dg_bind") is None:
            continue
        seq = r.get("sequence", "")
        if seq == NATIVE_SEQ and r.get("source") in ("native", "silo_b"):
            r["beats_native"] = False
            r["consensus_flag"] = "native"
            continue

        entry_ddg: Optional[float] = None
        src = r.get("source", "silo_b")

        if src == "silo_b":
            lb_e = entry_map.get(seq)
            if lb_e:
                entry_ddg = lb_e.get("ddg_median") or lb_e.get("ddg")
        elif src == "silo_a":
            # Silo A는 ddg_median 없음 — single ddg 사용 (non-robust)
            # consensus_flag 계산은 하되 silo_a_caveat 항상 부착
            # entry_ddg 찾기: Silo A leaderboard는 candidate_id 기준
            # (이미 caveat에 non-robust 명시됨)
            pass  # entry_ddg = None → docking_only/none 로 처리됨

        dg = r["dg_bind"]
        dg_sd = r.get("dg_bind_sd")
        r["beats_native"] = bool(native_dg_new is not None and dg < native_dg_new)
        flag = _compute_consensus_flag(
            dg, entry_ddg, native_dg_new, native_ddg, dg_bind_sd=dg_sd
        )
        # Silo A는 non-robust ddg → high_confidence 불가, 상한을 mmgbsa_only로
        if src == "silo_a" and flag == "high_confidence":
            flag = "mmgbsa_only_silo_a_nonrobust"
        r["consensus_flag"] = flag

    # ── consensus_ranking (전체, skipped 제외, native 제외) ─────────────────────
    scored = [
        r for r in results_map.values()
        if r.get("dg_bind") is not None
        and r.get("sequence") != NATIVE_SEQ
    ]
    scored.sort(key=lambda r: r["dg_bind"])

    # native_ddg_source: v2 파일 존재 여부로 출처 명시 (추적 가능)
    _v2_src = (
        "robust_v2_nstruct10"
        if NATIVE_ROBUST_BASELINE_V2_PATH.exists()
        else "silo_b_leaderboard_fallback"
    )
    consensus_out: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "native_dg": native_dg_new,
        "native_ddg": native_ddg,
        "native_ddg_source": _v2_src,
        "n_total": len(results_map),
        "n_scored": new_count,
        "n_skipped": skip_count + silo_a_skipped,
        "n_silo_b_scored": sum(
            1 for r in scored if r.get("source") == "silo_b"
        ),
        "n_silo_a_scored": silo_a_scored,
        "n_silo_a_skipped": silo_a_skipped,
        "n_beats_native": sum(1 for r in scored if r.get("beats_native")),
        "n_high_confidence": sum(
            1 for r in scored if r.get("consensus_flag") == "high_confidence"
        ),
        "n_uncertain": sum(
            1 for r in scored if r.get("consensus_flag") == "uncertain"
        ),
        "n_docked": sum(
            1 for r in scored if r.get("structure_source") == "docked"
        ),
        "n_mutated_approx": sum(
            1 for r in scored if r.get("structure_source") == "mutated_approx"
        ),
        "n_silo_a_docked": sum(
            1 for r in scored if r.get("structure_source") == "silo_a_docked"
        ),
        "caveat": CAVEAT,
        "caveat_mutated": CAVEAT_MUTATED,
        "caveat_silo_a": CAVEAT_SILO_A,
        "results": results_map,
        "consensus_ranking": [
            {
                "rank": i + 1,
                "sequence": r["sequence"],
                "source": r.get("source", "silo_b"),
                "candidate_id": r.get("candidate_id"),
                "mmgbsa_dg": r["dg_bind"],
                "mmgbsa_dg_sd": r.get("dg_bind_sd"),
                "n_snapshots": r.get("n_snapshots", 1),
                "ss_bond_verified": r.get("ss_bond_verified", False),
                "beats_native": r.get("beats_native", False),
                "consensus_flag": r.get("consensus_flag", "none"),
                "structure_source": r.get("structure_source", "unknown"),
            }
            for i, r in enumerate(scored)
        ],
    }

    _save_json(CONSENSUS_PATH, consensus_out)
    logging.info(
        "=== 사이클 완료: 채점=%d(SiloB=%d SiloA=%d) 스킵=%d "
        "high_confidence=%d (docked=%d mutated=%d silo_a=%d) native_dg=%.3f ===",
        new_count,
        consensus_out["n_silo_b_scored"],
        silo_a_scored,
        skip_count + silo_a_skipped,
        consensus_out["n_high_confidence"],
        consensus_out["n_docked"],
        consensus_out["n_mutated_approx"],
        consensus_out["n_silo_a_docked"],
        native_dg_new if native_dg_new is not None else float("nan"),
    )


# ── 메인 루프 ────────────────────────────────────────────────────────────────────

def main() -> None:
    """데몬 메인 루프."""
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE_ENV
    _setup_logging()

    logging.info(
        "MM-GBSA 데몬 v3 시작 | GPU CUDA_VISIBLE_DEVICES=%s | "
        "주기=%ds | Silo B top_n=%d | Silo A top_n=%d",
        CUDA_DEVICE_ENV, CYCLE_SECONDS, TOP_N, SILO_A_TOP_N,
    )
    logging.info("Silo B 리더보드: %s", LEADERBOARD_PATH)
    logging.info("Silo A 리더보드: %s", SILO_A_LEADERBOARD_PATH)
    logging.info("consensus 출력: %s", CONSENSUS_PATH)
    logging.info("pdb_cache: %s", PDB_CACHE_DIR)

    cycle_num = 0
    while True:
        cycle_num += 1
        logging.info("--- 사이클 #%d 시작 ---", cycle_num)
        try:
            run_cycle(top_n=TOP_N)
        except KeyboardInterrupt:
            logging.info("KeyboardInterrupt — 데몬 종료")
            break
        except Exception as exc:
            logging.error("사이클 #%d 예외(계속): %s", cycle_num, exc, exc_info=True)

        logging.info("--- 사이클 #%d 완료, %ds 대기 ---", cycle_num, CYCLE_SECONDS)
        time.sleep(CYCLE_SECONDS)


if __name__ == "__main__":
    main()
