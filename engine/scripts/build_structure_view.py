#!/usr/bin/env python3
"""build_structure_view.py — 현 리더보드 top N을 읽어 구조 뷰어용 JSON 자동 생성.

변경 사항 (2026-06-25):
  - 기존: candidates.json 고정·일회성 읽기 + 옛 rank PDB만 사용
  - 신규: global_selectivity_leaderboard.json top N 읽기 →
          candidates.json 자동 갱신 + PDB 없으면 PyRosetta MutateResidue로 생성
  - 2026-06-25: Silo A(de novo) 상위 후보 포함
      silo_a_leaderboard.json top N 읽기 →
      runs/silo_a_flow/epoch_{ts}/silo_a_{ts}_bb{bb}_sq{sq}.pdb 복사 → pdb/silo_a_{cand_id}.pdb
      source="silo_a", caveat 명시(de novo·non-robust·16aa, 직접 비교 불가)

실행:
    source [LOCAL_PATH]
    python scripts/build_structure_view.py [--top N] [--silo-a-top M]

출력:
    docs/structure_view/candidates.json  ← Silo B top N + Silo A top M 반영
    docs/structure_view/pdb/rank{NN}_{seq}.pdb  ← Silo B 없으면 생성
    docs/structure_view/pdb/silo_a_{cand_id}.pdb  ← Silo A de novo 복합체
    docs/structure_view/structure_data.json

caveat (Silo A):
  Silo A는 de novo(RFdiffusion/DiffPepBuilder) 설계 펩타이드로:
  - 16~20aa (SST-14 14aa와 길이 상이 — 직접 비교 불가)
  - 도킹 ddG 단일 스냅샷 non-robust (Silo B의 5회 중앙값 아님)
  - MutateResidue 불가 (native SST-14 scaffold 미사용)
  - plddt 기반 품질 필터링 적용
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

# ── 경로 ────────────────────────────────────────────────────────────────────────
ROOT = Path("[LOCAL_PATH]")
REPO = ROOT / "AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"
DOCS_DIR = ROOT / "docs/structure_view"
PDB_DIR = DOCS_DIR / "pdb"
OUTPUT_JSON = DOCS_DIR / "structure_data.json"
CANDIDATES_JSON = DOCS_DIR / "candidates.json"

LEADERBOARD_PATH = REPO / "runs/pyrosetta_flow/global_selectivity_leaderboard.json"
SILO_A_LEADERBOARD_PATH = REPO / "runs/silo_a_flow/silo_a_leaderboard.json"
SILO_A_FLOW_DIR = REPO / "runs/silo_a_flow"
NATIVE_PDB_PATH = PDB_DIR / "native_AGCKNFFWKTFTSC.pdb"

# bio-tools env Python (PyRosetta 보유)
BIOTOOLS_PYTHON = "[LOCAL_PATH]"

DEFAULT_TOP_N = 10
DEFAULT_SILO_A_TOP_N = 5

CONTACT_THRESHOLD_A = 5.0  # 5Å 접촉 기준

# Silo A de novo caveat (구조·길이·방법론 한계)
SILO_A_CAVEAT = (
    "Silo A de novo 후보: RFdiffusion/DiffPepBuilder 설계, 16~20aa(SST-14 14aa 대비 길이 상이). "
    "도킹 ddG 단일 스냅샷 non-robust (Silo B 5회 중앙값 아님). "
    "MutateResidue 불가(native scaffold 미사용). Silo B와 직접 비교 불가."
)

# candidate_id 파싱 패턴: silo_a_{timestamp}_bb{bb}_sq{sq}
_SILO_A_CAND_RE = re.compile(r"silo_a_(\d{8}T\d{6}Z)_bb(\d+)_sq(\d+)$")

# FWKT pharmacophore: 펩타이드 pos 7-10 (1-indexed)
FWKT_POSITIONS = {7, 8, 9, 10}
# Cys SS bond: pos 3, 14
CYS_POSITIONS = {3, 14}

# 표준 20종 AA
STANDARD_AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")

NATIVE_SEQ = "AGCKNFFWKTFTSC"

AA_MAP = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H', 'MSE': 'M', 'CYX': 'C', 'CYM': 'C',
}

# PyRosetta MutateResidue 인라인 스크립트 (mmgbsa_daemon._MUTATE_SCRIPT 동일 로직)
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


# ── OOD 판정 (mmgbsa_daemon._is_ood_skip 동일 로직) ────────────────────────────

def _is_ood_skip(sequence: str, lb_entry: Optional[dict]) -> tuple[bool, str]:
    """MutateResidue 불가 후보 판정.

    skip 조건:
      1. 서열에 소문자(D-aa 표기) 또는 비표준 AA 포함
      2. Cys3·Cys14 파괴(SS bond 파괴)
      3. DOTA/OOD 플래그

    Returns:
        (should_skip: bool, reason: str)
    """
    if not sequence or len(sequence) < 14:
        return True, "서열 길이 부족(<14)"

    # D-aa: 소문자 포함
    lower_chars = [c for c in sequence if c.islower()]
    if lower_chars:
        return True, f"D-아미노산(소문자): {lower_chars}"

    # 비표준 AA
    non_std = [aa for aa in sequence.upper() if aa not in STANDARD_AA_SET]
    if non_std:
        return True, f"비표준AA: {non_std}"

    # Cys SS bond 파괴 여부 (pos3=idx2, pos14=idx13)
    if len(sequence) >= 14 and (sequence[2] != "C" or sequence[13] != "C"):
        return True, (
            f"CysSS 파괴 (pos3={sequence[2]}, pos14={sequence[13]})"
        )

    # DOTA/OOD 플래그 (리더보드 엔트리)
    if lb_entry:
        es = lb_entry.get("extra_scores") or {}
        is_ood = lb_entry.get("is_ood") or es.get("is_ood")
        ood_reasons = lb_entry.get("ood_reasons") or es.get("ood_reasons") or []
        if is_ood:
            ood_str = ", ".join(ood_reasons) if ood_reasons else "OOD"
            return True, f"D-aa/DOTA OOD: {ood_str}"

    return False, ""


# ── PyRosetta MutateResidue (bio-tools subprocess) ─────────────────────────────

def _build_complex_pdb_by_mutation(sequence: str, out_pdb: Path) -> bool:
    """PyRosetta MutateResidue로 native → target 복합체 PDB 생성.

    bio-tools env Python을 subprocess로 호출(환경 분리).

    Args:
        sequence: 목표 서열 (14aa, 표준 AA, Cys3·Cys14 보존)
        out_pdb: 출력 PDB 경로

    Returns:
        성공 여부
    """
    if not NATIVE_PDB_PATH.exists():
        print(f"  [WARN] native PDB 미존재: {NATIVE_PDB_PATH}")
        return False

    if not Path(BIOTOOLS_PYTHON).exists():
        print(f"  [WARN] bio-tools Python 미존재: {BIOTOOLS_PYTHON}")
        return False

    out_pdb.parent.mkdir(parents=True, exist_ok=True)

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
        stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
        if not stdout_lines:
            stderr_excerpt = proc.stderr[:300] if proc.stderr else "(없음)"
            print(f"  [WARN] MutateResidue 출력 없음 (seq={sequence}) stderr={stderr_excerpt}")
            return False

        result = json.loads(stdout_lines[-1])
        if not result.get("ok"):
            print(f"  [WARN] MutateResidue 실패 (seq={sequence}): {result.get('error')}")
            return False

        print(f"  [OK] MutateResidue 성공: {sequence} → {out_pdb.name}")
        return True

    except subprocess.TimeoutExpired:
        print(f"  [WARN] MutateResidue 타임아웃 (seq={sequence})")
        return False
    except Exception as exc:
        print(f"  [WARN] MutateResidue 예외 (seq={sequence}): {exc}")
        return False
    finally:
        try:
            os.unlink(tmp_script)
        except OSError:
            pass


# ── PDB 확보: docked 우선, 없으면 MutateResidue ────────────────────────────────

def _find_or_build_pdb(
    sequence: str,
    rank: int,
    lb_entry: Optional[dict],
) -> tuple[Optional[Path], str]:
    """복합체 PDB 경로 반환.

    우선순위:
      1. docs/structure_view/pdb/ 내 서열 매칭 파일 (docked)
      2. MutateResidue 생성 → pdb/rank{NN}_{seq}.pdb (mutated_approx)
      3. OOD/실패 → (None, skip_reason)

    Returns:
        (pdb_path_relative_to_DOCS_DIR, structure_source)
        pdb_path는 docs/structure_view 기준 상대경로(str) 또는 None
    """
    # native 처리
    if sequence == NATIVE_SEQ:
        if NATIVE_PDB_PATH.exists():
            return NATIVE_PDB_PATH, "docked"
        return None, "native PDB 미존재"

    # 1) 기존 PDB 탐색 (서열 포함 파일명)
    if PDB_DIR.exists():
        for p in PDB_DIR.glob("*.pdb"):
            if p.stem.startswith("ot_"):
                continue
            if sequence in p.stem:
                return p, "docked"

    # 2) OOD 판정
    should_skip, skip_reason = _is_ood_skip(sequence, lb_entry)
    if should_skip:
        return None, f"skip:{skip_reason}"

    # 3) MutateResidue로 생성
    rank_str = f"{rank:02d}"
    out_pdb = PDB_DIR / f"rank{rank_str}_{sequence}.pdb"
    success = _build_complex_pdb_by_mutation(sequence, out_pdb)
    if not success:
        # graceful: 기존 PDB 그대로, 메타만 갱신
        return None, "PyRosetta MutateResidue 실패(graceful skip)"

    return out_pdb, "mutated_approx"


# ── Silo A PDB 탐색 ──────────────────────────────────────────────────────────

def _find_silo_a_pdb(candidate_id: str) -> Optional[Path]:
    """Silo A candidate_id로부터 FlexPepDock 복합체 PDB를 찾아 pdb/ 에 복사.

    매핑 규칙:
      candidate_id = silo_a_{ts}_bb{bb}_sq{sq}
      epoch_dir    = runs/silo_a_flow/epoch_{ts}/
      복합체 PDB   = silo_a_{ts}_bb{bb}_sq{sq}.pdb  (FlexPepDock 산출)
      fallback     = diffpep_{ts}_bb{bb}_sq{sq}.pdb  (DiffPepBuilder 산출)

    복사 대상: docs/structure_view/pdb/silo_a_{candidate_id}.pdb

    Returns:
        복사된 PDB 경로(pdb/ 아래), 못 찾으면 None
    """
    m = _SILO_A_CAND_RE.match(candidate_id)
    if m is None:
        print(f"  [WARN] Silo A candidate_id 파싱 실패: {candidate_id}")
        return None

    ts, bb, sq = m.group(1), m.group(2), m.group(3)
    epoch_dir = SILO_A_FLOW_DIR / f"epoch_{ts}"
    if not epoch_dir.exists():
        print(f"  [WARN] Silo A epoch 디렉토리 없음: {epoch_dir}")
        return None

    # 복합체 PDB 우선순위: silo_a_*.pdb > diffpep_*.pdb
    src_pdb: Optional[Path] = None
    for prefix in ("silo_a", "diffpep"):
        candidate_pdb = epoch_dir / f"{prefix}_{ts}_bb{bb}_sq{sq}.pdb"
        if candidate_pdb.exists():
            src_pdb = candidate_pdb
            break

    if src_pdb is None:
        print(f"  [WARN] Silo A 복합체 PDB 없음 (epoch={ts}, bb={bb}, sq={sq})")
        return None

    # docs/structure_view/pdb/ 로 복사 (이미 있으면 재사용)
    # candidate_id 자체가 "silo_a_" 접두어를 포함하므로 그대로 사용
    dest_name = f"{candidate_id}.pdb"
    dest_pdb = PDB_DIR / dest_name
    if not dest_pdb.exists():
        PDB_DIR.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src_pdb, dest_pdb)
            print(f"  [OK] Silo A PDB 복사: {src_pdb.name} → {dest_name}")
        except OSError as e:
            print(f"  [WARN] Silo A PDB 복사 실패: {e}")
            return None
    else:
        print(f"  [OK] Silo A PDB 재사용: {dest_name}")

    return dest_pdb


# ── Silo A 리더보드 로드 ────────────────────────────────────────────────────────

def _load_silo_a_leaderboard_top(n: int) -> list[dict]:
    """silo_a_leaderboard.json에서 top N 엔트리 반환.

    Silo A 리더보드 구조: {"source": "silo_a", "entries": [...], ...}
    정렬: ddg 오름차순 (낮을수록 강결합=상위).

    Returns:
        list of dict with candidate_id, sequence, ddg, selectivity_margin, plddt, ...
    """
    if not SILO_A_LEADERBOARD_PATH.exists():
        print(f"[WARN] Silo A 리더보드 파일 미존재: {SILO_A_LEADERBOARD_PATH}")
        return []

    with open(SILO_A_LEADERBOARD_PATH, encoding="utf-8") as f:
        lb = json.load(f)

    entries = lb.get("entries", [])
    if not entries:
        print("[WARN] Silo A 리더보드 entries 비어있음")
        return []

    def _sort_key(e: dict) -> float:
        v = e.get("ddg")
        return v if isinstance(v, (int, float)) else float("inf")

    entries_sorted = sorted(entries, key=_sort_key)
    return entries_sorted[:n]


# ── 리더보드 로드 ─────────────────────────────────────────────────────────────

def _load_leaderboard_top(n: int) -> list[dict]:
    """global_selectivity_leaderboard.json에서 top N 엔트리 반환.

    엔트리 정렬: 리더보드 canonical(ranking_mode=robust_ddg_median)과 일치.
    ddg_median 오름차순(낮을수록 강결합=상위). mutation_network/monitoring과 동일 순위.

    Returns:
        list of dict with keys: sequence, delta_margin, ddg_median, hc50, ...
    """
    if not LEADERBOARD_PATH.exists():
        print(f"[WARN] 리더보드 파일 미존재: {LEADERBOARD_PATH}")
        return []

    with open(LEADERBOARD_PATH, encoding="utf-8") as f:
        lb = json.load(f)

    entries = lb.get("entries", [])
    if not entries:
        print("[WARN] 리더보드 entries 비어있음")
        return []

    # 리더보드 entries는 이미 robust_ddg_median 정렬되어 저장됨 → 그 순위 보존.
    # 안전망: ddg_median 오름차순 재정렬(None/양수는 뒤로) — 리더보드와 동일 기준.
    def _sort_key(e):
        v = e.get("ddg_median", e.get("ddg"))
        return v if isinstance(v, (int, float)) else float("inf")
    entries_sorted = sorted(entries, key=_sort_key)
    return entries_sorted[:n]


# ── candidates.json 자동 생성 ───────────────────────────────────────────────────

def build_candidates_from_leaderboard(
    top_n: int,
    silo_a_top_n: int = DEFAULT_SILO_A_TOP_N,
) -> list[dict]:
    """리더보드 top N (Silo B) + Silo A top M + native를 기반으로 candidates 목록 생성.

    Silo B 각 후보에 대해:
      - PDB 확보(docked / mutated_approx)
      - OOD skip 시 pdb_file=None, skip_reason 기록
      - source="silo_b"

    Silo A 각 후보에 대해:
      - de novo 복합체 PDB(runs/silo_a_flow/epoch_*/silo_a_*.pdb) 복사·참조
      - 복합체 PDB 없으면 graceful skip(메타만)
      - source="silo_a", caveat 명시

    Returns:
        candidates list (native + Silo B rank 순서 + Silo A rank 순서)
    """
    top_entries = _load_leaderboard_top(top_n)
    if not top_entries:
        print("[ERROR] Silo B 리더보드에서 엔트리를 로드할 수 없음. 기존 candidates.json 유지.")
        if CANDIDATES_JSON.exists():
            with open(CANDIDATES_JSON, encoding="utf-8") as f:
                return json.load(f)
        return []

    candidates: list[dict] = []

    # native 추가 (rank 0, source=silo_b 기준)
    native_pdb_rel = f"pdb/native_{NATIVE_SEQ}.pdb"
    native_cand = {
        "rank": 0,
        "sequence": NATIVE_SEQ,
        "label": "SST-14 Native (참조)",
        "source": "native",
        "pdb_file": native_pdb_rel,
        "structure_source": "docked",
        "ddg_median": None,
        "delta_margin": 0.0,
        "hc50": None,
        "mutations": [],
        "run_id": "baseline_cached",
        "ood_skip": False,
        "skip_reason": "",
        "caveat": "",
    }
    candidates.append(native_cand)

    pdb_gen_count = 0
    pdb_skip_count = 0

    # ── Silo B top N ─────────────────────────────────────────────────────────
    print(f"\n[Silo B] top {top_n} 후보 처리...")
    for rank_idx, entry in enumerate(top_entries, start=1):
        seq = entry.get("sequence", "")
        if not seq:
            continue

        if seq == NATIVE_SEQ:
            continue

        t0 = time.time()
        print(f"\n  [Silo B rank {rank_idx:02d}] {seq}")

        pdb_path, source = _find_or_build_pdb(seq, rank_idx, entry)
        elapsed = time.time() - t0

        if source.startswith("skip:") or pdb_path is None:
            ood_skip = True
            skip_reason = source
            pdb_rel = None
            pdb_skip_count += 1
            print(f"    → OOD/skip ({skip_reason}) [{elapsed:.1f}s]")
        else:
            ood_skip = False
            skip_reason = ""
            try:
                pdb_rel = "pdb/" + pdb_path.name
            except Exception:
                pdb_rel = str(pdb_path)
            if source == "mutated_approx":
                pdb_gen_count += 1
            print(f"    → {source}: {pdb_rel} [{elapsed:.1f}s]")

        mutations: list[str] = []
        if len(seq) == len(NATIVE_SEQ):
            for i, (nat, tgt) in enumerate(zip(NATIVE_SEQ, seq)):
                if nat != tgt:
                    mutations.append(f"{nat}{i+1}{tgt}")

        cand = {
            "rank": rank_idx,
            "sequence": seq,
            "label": f"B-rank{rank_idx:02d}",
            "source": "silo_b",
            "pdb_file": pdb_rel,
            "structure_source": source if not ood_skip else "none",
            "ddg_median": entry.get("ddg_median"),
            "delta_margin": entry.get("delta_margin"),
            "hc50": entry.get("hc50"),
            "more_toxic_than_native": entry.get("more_toxic_than_native", False),
            "mutations": mutations,
            "run_id": entry.get("run_id", ""),
            "ood_skip": ood_skip,
            "skip_reason": skip_reason,
            "consensus_flag": entry.get("consensus_flag", False),
            "caveat": "",
        }
        candidates.append(cand)

    print(f"\n  [Silo B] PDB 신규 생성: {pdb_gen_count}개, OOD skip: {pdb_skip_count}개")

    # ── Silo A top M ─────────────────────────────────────────────────────────
    print(f"\n[Silo A] top {silo_a_top_n} de novo 후보 처리...")
    silo_a_entries = _load_silo_a_leaderboard_top(silo_a_top_n)
    silo_a_found = 0
    silo_a_skipped = 0

    for a_rank_idx, entry in enumerate(silo_a_entries, start=1):
        cand_id = entry.get("candidate_id", "")
        seq = entry.get("sequence", "")
        if not seq or not cand_id:
            continue

        t0 = time.time()
        print(f"\n  [Silo A rank {a_rank_idx:02d}] {cand_id}  seq={seq}")

        pdb_path = _find_silo_a_pdb(cand_id)
        elapsed = time.time() - t0

        if pdb_path is None:
            pdb_rel = None
            structure_src = "none"
            silo_a_skipped += 1
            skip_reason = "Silo A 복합체 PDB 없음 (graceful skip)"
            print(f"    → PDB 없음, 메타만 [{elapsed:.1f}s]")
        else:
            pdb_rel = "pdb/" + pdb_path.name
            structure_src = "silo_a_docked"
            silo_a_found += 1
            skip_reason = ""
            print(f"    → {structure_src}: {pdb_rel} [{elapsed:.1f}s]")

        cand = {
            "rank": a_rank_idx,
            "sequence": seq,
            "label": f"A-rank{a_rank_idx:02d}",
            "source": "silo_a",
            "candidate_id": cand_id,
            "pdb_file": pdb_rel,
            "structure_source": structure_src,
            # Silo A는 ddg_median 아님(단일 스냅샷 non-robust)
            "ddg": entry.get("ddg"),
            "ddg_median": None,
            "delta_margin": entry.get("delta_margin"),
            "selectivity_margin": entry.get("selectivity_margin"),
            "plddt": entry.get("plddt"),
            "hc50": entry.get("hc50"),
            "mutations": [],  # de novo — 변이 목록 없음
            "run_id": entry.get("candidate_id", ""),
            "ood_skip": pdb_rel is None,
            "skip_reason": skip_reason,
            "consensus_flag": False,
            "caveat": SILO_A_CAVEAT,
        }
        candidates.append(cand)

    print(
        f"\n  [Silo A] PDB 포함: {silo_a_found}개, "
        f"메타만(PDB 없음): {silo_a_skipped}개"
    )
    return candidates


# ── PDB 파싱 ──────────────────────────────────────────────────────────────────

def parse_pdb_atoms(pdb_path: Path) -> dict:
    """PDB 파일에서 ATOM 레코드를 파싱해 chain별 딕셔너리 반환."""
    chains: dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue
            try:
                chain = line[21]
                resnum = int(line[22:26].strip())
                resname = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except (ValueError, IndexError):
                continue

            if chain not in chains:
                chains[chain] = {}
            if resnum not in chains[chain]:
                chains[chain][resnum] = {
                    'resname': resname,
                    'aa': AA_MAP.get(resname, 'X'),
                    'atoms': [],
                }
            chains[chain][resnum]['atoms'].append((None, x, y, z))

    return chains


def min_distance(res_a: dict, res_b: dict) -> float:
    """두 잔기 간 최소 원자간 거리(Å)."""
    min_d = float('inf')
    for _, ax, ay, az in res_a['atoms']:
        for _, bx, by, bz in res_b['atoms']:
            d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
            if d < min_d:
                min_d = d
    return min_d


def compute_contacts(
    chains: dict,
    chain_receptor: str = 'A',
    chain_peptide: str = 'B',
    threshold: float = CONTACT_THRESHOLD_A,
) -> dict:
    """5Å 이내 접촉잔기 계산."""
    if chain_receptor not in chains or chain_peptide not in chains:
        return {"receptor_contacts": [], "peptide_contacts": []}

    rec = chains[chain_receptor]
    pep = chains[chain_peptide]

    pep_resnums = sorted(pep.keys())
    pep_pos_map = {rn: i + 1 for i, rn in enumerate(pep_resnums)}

    receptor_contacts: dict[int, float] = {}
    peptide_contacts: dict[int, float] = {}

    for pep_rn in pep_resnums:
        pep_res = pep[pep_rn]
        for rec_rn, rec_res in rec.items():
            d = min_distance(pep_res, rec_res)
            if d <= threshold:
                if rec_rn not in receptor_contacts or d < receptor_contacts[rec_rn]:
                    receptor_contacts[rec_rn] = d
                if pep_rn not in peptide_contacts or d < peptide_contacts[pep_rn]:
                    peptide_contacts[pep_rn] = d

    rec_result = []
    for rn in sorted(receptor_contacts.keys()):
        rec_result.append({
            "resnum": rn,
            "aa": rec[rn]['aa'],
            "resname": rec[rn]['resname'],
            "min_dist": round(receptor_contacts[rn], 3),
        })

    pep_result = []
    for rn in pep_resnums:
        if rn in peptide_contacts:
            pos = pep_pos_map[rn]
            pep_result.append({
                "resnum": rn,
                "aa": pep[rn]['aa'],
                "resname": pep[rn]['resname'],
                "pos": pos,
                "min_dist": round(peptide_contacts[rn], 3),
                "is_fwkt": pos in FWKT_POSITIONS,
                "is_cys_ss": pos in CYS_POSITIONS,
            })

    return {"receptor_contacts": rec_result, "peptide_contacts": pep_result}


def get_chain_b_seq(chains: dict) -> str:
    """chain B 서열 반환."""
    if 'B' not in chains:
        return ""
    pep = chains['B']
    return ''.join(pep[rn]['aa'] for rn in sorted(pep.keys()))


def process_candidate(candidate: dict) -> dict:
    """후보 1개 처리: PDB 있으면 접촉잔기 계산, 없으면 메타만."""
    pdb_rel = candidate.get("pdb_file")
    if not pdb_rel:
        # OOD skip — 접촉 없이 메타만
        return {
            **candidate,
            "seq_verified": candidate.get("sequence", ""),
            "contacts": {"receptor_contacts": [], "peptide_contacts": []},
            "n_receptor_contacts": 0,
            "n_peptide_contacts": 0,
            "error": candidate.get("skip_reason", "PDB 없음"),
        }

    pdb_file = DOCS_DIR / pdb_rel
    if not pdb_file.exists():
        print(f"  MISSING PDB: {pdb_file}")
        return {
            **candidate,
            "seq_verified": candidate.get("sequence", ""),
            "contacts": {"receptor_contacts": [], "peptide_contacts": []},
            "n_receptor_contacts": 0,
            "n_peptide_contacts": 0,
            "error": "PDB 파일 미존재",
        }

    try:
        chains = parse_pdb_atoms(pdb_file)
    except Exception as e:
        print(f"  ERROR parsing {pdb_file.name}: {e}")
        return {
            **candidate,
            "seq_verified": "",
            "contacts": {"receptor_contacts": [], "peptide_contacts": []},
            "n_receptor_contacts": 0,
            "n_peptide_contacts": 0,
            "error": f"PDB 파싱 실패: {e}",
        }

    seq = get_chain_b_seq(chains)
    contacts = compute_contacts(chains)

    return {
        **candidate,
        "seq_verified": seq,
        "contacts": contacts,
        "n_receptor_contacts": len(contacts["receptor_contacts"]),
        "n_peptide_contacts": len(contacts["peptide_contacts"]),
    }


# ── stale PDB 정리 ─────────────────────────────────────────────────────────────

def _cleanup_stale_pdbs(current_seqs: set[str]) -> list[str]:
    """현 top에 없는 rank PDB를 실제 삭제하고 목록 반환.

    삭제 대상: rank{NN}_{seq}.pdb 중 seq가 현재 top에 없는 것.
    보존 대상: native_*.pdb, ot_SSTR*.pdb, silo_a_*.pdb (이름 패턴으로 보호).

    중복 접두어(같은 rank번호에 다른 서열 PDB 공존) 문제를 해소해
    뷰어가 pdb_file 필드로 정확히 로드되도록 한다.
    """
    stale: list[str] = []
    if not PDB_DIR.exists():
        return stale

    for p in PDB_DIR.glob("rank*.pdb"):
        # 파일명에서 서열 추출 (rank{NN}_{seq}.pdb)
        parts = p.stem.split("_", 1)
        if len(parts) < 2:
            continue
        file_seq = parts[1]
        if file_seq not in current_seqs:
            stale.append(p.name)
            try:
                p.unlink()
                print(f"  [CLEANUP] stale PDB 삭제: {p.name}")
            except OSError as e:
                print(f"  [WARN] stale PDB 삭제 실패 ({p.name}): {e}")

    if stale:
        print(f"\n  [INFO] stale rank PDB 정리 완료 ({len(stale)}개 삭제): {stale}")
    else:
        print("\n  [INFO] stale rank PDB 없음 (정리 불필요)")
    return stale


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="structure_view 빌드 (Silo B top N + Silo A top M 반영)"
    )
    parser.add_argument(
        "--top", type=int, default=DEFAULT_TOP_N,
        help=f"Silo B 리더보드 top N (기본 {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--silo-a-top", type=int, default=DEFAULT_SILO_A_TOP_N,
        help=f"Silo A 리더보드 top M (기본 {DEFAULT_SILO_A_TOP_N})",
    )
    args = parser.parse_args()
    top_n = args.top
    silo_a_top_n = args.silo_a_top

    t_total_start = time.time()
    print(f"=== build_structure_view.py (Silo B top_n={top_n}, Silo A top_n={silo_a_top_n}) ===")
    print(f"  Silo B Leaderboard: {LEADERBOARD_PATH}")
    print(f"  Silo A Leaderboard: {SILO_A_LEADERBOARD_PATH}")
    print(f"  PDB dir:            {PDB_DIR}")
    print(f"  Contact threshold:  {CONTACT_THRESHOLD_A} Å")
    print(f"  Silo A caveat:      {SILO_A_CAVEAT[:80]}...")

    # 1) candidates.json 자동 생성 (Silo B top N + Silo A top M 반영)
    print(f"\n[1] 리더보드 읽기 + PDB 확보 (Silo B={top_n}, Silo A={silo_a_top_n})...")
    candidates = build_candidates_from_leaderboard(top_n, silo_a_top_n)

    if not candidates:
        print("[ERROR] candidates 빌드 실패. 종료.")
        sys.exit(1)

    # candidates.json 원자적 쓰기
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_cand = CANDIDATES_JSON.with_suffix(".tmp")
    with open(tmp_cand, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    tmp_cand.rename(CANDIDATES_JSON)
    print(f"\n  candidates.json 갱신: {len(candidates)}개")

    # stale PDB 로깅
    current_seqs = {c["sequence"] for c in candidates}
    _cleanup_stale_pdbs(current_seqs)

    # 2) 접촉잔기 계산
    print(f"\n[2] 접촉잔기 계산...")
    results = []
    for cand in candidates:
        seq = cand.get("sequence", "")
        rank = cand.get("rank", "?")
        print(f"\n  rank {rank}: {seq}")
        result = process_candidate(cand)
        n_rec = result.get("n_receptor_contacts", 0)
        n_pep = result.get("n_peptide_contacts", 0)
        ood = cand.get("ood_skip", False)
        if ood:
            print(f"    OOD skip — contacts 없음 (skip_reason={cand.get('skip_reason', '')})")
        else:
            print(f"    contacts: receptor={n_rec}, peptide={n_pep}")
        results.append(result)

    # 3) off-target 수용체 단독 PDB 메타
    off_targets = []
    for sstr_n in [1, 2, 3, 4, 5]:
        ot_pdb = PDB_DIR / f"ot_SSTR{sstr_n}_receptor.pdb"
        if ot_pdb.exists():
            try:
                chains = parse_pdb_atoms(ot_pdb)
                chain_list = sorted(chains.keys())
                total_res = sum(len(v) for v in chains.values())
                off_targets.append({
                    "sstr": sstr_n,
                    "label": f"SSTR{sstr_n}",
                    "pdb_file": f"pdb/ot_SSTR{sstr_n}_receptor.pdb",
                    "chains": chain_list,
                    "total_residues": total_res,
                })
                print(f"\n  Off-target SSTR{sstr_n}: chains={chain_list}, res={total_res}")
            except Exception as e:
                print(f"\n  Off-target SSTR{sstr_n} error: {e}")

    # source별 통계
    n_silo_b = sum(1 for c in results if c.get("source") == "silo_b")
    n_silo_a = sum(1 for c in results if c.get("source") == "silo_a")
    n_silo_a_with_pdb = sum(
        1 for c in results
        if c.get("source") == "silo_a" and c.get("pdb_file") is not None
    )

    # 4) structure_data.json 저장
    output = {
        "version": "2.1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "leaderboard_path": str(LEADERBOARD_PATH),
        "silo_a_leaderboard_path": str(SILO_A_LEADERBOARD_PATH),
        "top_n": top_n,
        "silo_a_top_n": silo_a_top_n,
        "n_silo_b": n_silo_b,
        "n_silo_a": n_silo_a,
        "n_silo_a_with_pdb": n_silo_a_with_pdb,
        "contact_threshold_angstrom": CONTACT_THRESHOLD_A,
        "native_sequence": NATIVE_SEQ,
        "fwkt_positions": list(FWKT_POSITIONS),
        "cys_ss_positions": list(CYS_POSITIONS),
        "silo_a_caveat": SILO_A_CAVEAT,
        "candidates": results,
        "off_targets": off_targets,
    }

    tmp_out = OUTPUT_JSON.with_suffix(".tmp")
    with open(tmp_out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    tmp_out.rename(OUTPUT_JSON)

    t_elapsed = time.time() - t_total_start
    print(f"\n=== 완료 ===")
    print(f"  Saved: {OUTPUT_JSON}")
    print(f"  candidates: {len(results)}개 (Silo B: {n_silo_b}, Silo A: {n_silo_a} [PDB있음: {n_silo_a_with_pdb}], off-target: {len(off_targets)}개)")
    print(f"  Silo A caveat: {SILO_A_CAVEAT[:80]}...")
    print(f"  소요시간: {t_elapsed:.1f}초")


if __name__ == '__main__':
    main()
