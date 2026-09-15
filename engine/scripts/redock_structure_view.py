#!/usr/bin/env python3
"""redock_structure_view.py — 리더보드 top N 후보를 FlexPepDock으로 재도킹해
docs/structure_view/pdb/ 에 진짜 복합체 구조를 저장하고 candidates.json 갱신.

문제: 기존 build_structure_view.py 는 MutateResidue(side-chain 치환)만 해서
backbone이 native와 100% 동일 → structure_source="docked" 가 거짓 라벨.

해결:
  1. top N 서열 → MutateResidue로 시작 구조 생성 → FlexPepDock refine(nstruct=3)
  2. 재도킹 성공 → structure_source="docked" (backbone 실제 변경)
  3. OOD(D-aa/DOTA/CysBreak) → MutateResidue fallback → structure_source="mutated_approx"
  4. candidates.json structure_source 정직화
  5. 검증: 재도킹 PDB의 chain B CA RMSD vs native > 0.5Å 확인

실행:
    [LOCAL_PATH] \
        scripts/redock_structure_view.py [--top N] [--nstruct 3] [--dry-run]

제약:
  - flexpep_dock.py 수정 금지 — subprocess 호출만
  - 발굴 엔진 파일 수정 금지
  - max_workers 8~16 (엔진 방해 방지)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT = Path("[LOCAL_PATH]")
REPO = ROOT / "AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri"
DOCS_DIR = ROOT / "docs/structure_view"
PDB_DIR = DOCS_DIR / "pdb"
CANDIDATES_JSON = DOCS_DIR / "candidates.json"

LEADERBOARD_PATH = REPO / "runs/pyrosetta_flow/global_selectivity_leaderboard.json"
# structure_view/pdb/ 의 native (chain A=receptor, B=peptide)
NATIVE_PDB = PDB_DIR / "native_AGCKNFFWKTFTSC.pdb"
# AG_src/scripts/flexpep_dock.py (읽기/호출만, 수정 금지)
FLEXPEP_DOCK_PY = REPO / "AG_src/scripts/flexpep_dock.py"
BIOTOOLS_PYTHON = "[LOCAL_PATH]"

NATIVE_SEQ = "AGCKNFFWKTFTSC"
STANDARD_AA_SET = frozenset("ACDEFGHIKLMNPQRSTVWY")

DEFAULT_TOP_N = 10
DEFAULT_NSTRUCT = 3  # 속도 vs 품질 균형 (엔진 자원 경쟁 고려)
DEFAULT_MAX_WORKERS = 8  # 엔진(128 workers) 방해 최소화


# ── OOD 판정 ──────────────────────────────────────────────────────────────────

def is_ood(sequence: str, lb_entry: Optional[dict]) -> tuple[bool, str]:
    """FlexPepDock 불가 후보 판정.

    D-aa(소문자), 비표준 AA, Cys3/Cys14 파괴, DOTA/OOD 플래그 중 하나라도 해당하면 True.
    """
    if not sequence or len(sequence) < 14:
        return True, "서열 길이 부족"

    lower_chars = [c for c in sequence if c.islower()]
    if lower_chars:
        return True, f"D-아미노산(소문자): {lower_chars}"

    non_std = [aa for aa in sequence.upper() if aa not in STANDARD_AA_SET]
    if non_std:
        return True, f"비표준AA: {non_std}"

    if len(sequence) >= 14 and (sequence[2] != "C" or sequence[13] != "C"):
        return True, f"CysSS 파괴 (pos3={sequence[2]}, pos14={sequence[13]})"

    if lb_entry:
        es = lb_entry.get("extra_scores") or {}
        is_ood_flag = lb_entry.get("is_ood") or es.get("is_ood")
        ood_reasons = lb_entry.get("ood_reasons") or es.get("ood_reasons") or []
        if is_ood_flag:
            reason_str = ", ".join(ood_reasons) if ood_reasons else "OOD"
            return True, f"D-aa/DOTA OOD: {reason_str}"

    return False, ""


# ── CA RMSD 계산 ──────────────────────────────────────────────────────────────

def get_chain_b_ca(pdb_path: Path) -> list[tuple[float, float, float]]:
    """PDB에서 chain B CA 좌표 반환."""
    cas = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "B":
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    cas.append((x, y, z))
                except ValueError:
                    continue
    return cas


def ca_rmsd(pdb_a: Path, pdb_b: Path) -> Optional[float]:
    """chain B CA RMSD(Å) 계산. 길이 불일치 시 None 반환."""
    cas_a = get_chain_b_ca(pdb_a)
    cas_b = get_chain_b_ca(pdb_b)
    if not cas_a or not cas_b or len(cas_a) != len(cas_b):
        return None
    msd = sum((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
              for a, b in zip(cas_a, cas_b)) / len(cas_a)
    return math.sqrt(msd)


def ca_mean_displacement(pdb_a: Path, pdb_b: Path) -> Optional[float]:
    """chain B CA 평균 이동량(Å). 각 잔기의 이동 평균."""
    cas_a = get_chain_b_ca(pdb_a)
    cas_b = get_chain_b_ca(pdb_b)
    if not cas_a or not cas_b or len(cas_a) != len(cas_b):
        return None
    dists = [math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
             for a, b in zip(cas_a, cas_b)]
    return sum(dists) / len(dists)


# ── 리더보드 로드 ─────────────────────────────────────────────────────────────

def load_leaderboard_top(n: int) -> list[dict]:
    """global_selectivity_leaderboard.json top N 반환 (ddg_median 오름차순)."""
    with open(LEADERBOARD_PATH, encoding="utf-8") as f:
        lb = json.load(f)
    entries = lb.get("entries", [])

    def sort_key(e: dict) -> float:
        v = e.get("ddg_median", e.get("ddg"))
        return v if isinstance(v, (int, float)) else float("inf")

    return sorted(entries, key=sort_key)[:n]


# ── FlexPepDock 재도킹 (subprocess 호출) ─────────────────────────────────────

def _dock_one_sequence(
    seq: str,
    rank: int,
    nstruct: int,
) -> dict:
    """단일 서열을 FlexPepDock으로 재도킹.

    flexpep_dock.py를 subprocess(bio-tools env)로 호출.
    --reference-complex NATIVE_PDB --target-sequence SEQ --nstruct N

    Returns dict with keys: seq, rank, success, pdb_path, ddg, rmsd, mean_disp, error
    """
    rank_str = f"{rank:02d}"
    out_pdb = PDB_DIR / f"rank{rank_str}_{seq}.pdb"

    t0 = time.time()

    cmd = [
        BIOTOOLS_PYTHON,
        str(FLEXPEP_DOCK_PY),
        "--input", str(NATIVE_PDB),
        "--reference-complex", str(NATIVE_PDB),
        "--target-sequence", seq,
        "--peptide-chain", "2",  # native: chain B = peptide (chain index 2)
        "--output", str(out_pdb),
        "--protocol", "flexpep_refine",
        "--nstruct", str(nstruct),
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10분 제한 (nstruct=3 기준)
        )
    except subprocess.TimeoutExpired:
        return {
            "seq": seq, "rank": rank, "success": False,
            "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
            "error": "FlexPepDock 타임아웃(600s)",
        }
    except Exception as exc:
        return {
            "seq": seq, "rank": rank, "success": False,
            "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
            "error": f"subprocess 예외: {exc}",
        }

    elapsed = time.time() - t0

    # flexpep_dock.py stdout = JSON 한 줄
    json_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not json_lines:
        stderr_excerpt = proc.stderr[-500:] if proc.stderr else "(없음)"
        return {
            "seq": seq, "rank": rank, "success": False,
            "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
            "error": f"FlexPepDock 출력 없음 (rc={proc.returncode}). stderr={stderr_excerpt}",
        }

    try:
        result = json.loads(json_lines[-1])
    except json.JSONDecodeError as exc:
        return {
            "seq": seq, "rank": rank, "success": False,
            "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
            "error": f"JSON 파싱 실패: {exc}. stdout={json_lines[-1][:200]}",
        }

    # PDB 생성 확인
    if not out_pdb.exists():
        return {
            "seq": seq, "rank": rank, "success": False,
            "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
            "error": f"FlexPepDock 완료했으나 PDB 미생성: {out_pdb}",
        }

    # CA RMSD vs native 계산
    rmsd_val = ca_rmsd(NATIVE_PDB, out_pdb)
    mean_disp_val = ca_mean_displacement(NATIVE_PDB, out_pdb)

    ddg_val = result.get("ddg") or result.get("ddg_median")

    return {
        "seq": seq,
        "rank": rank,
        "success": True,
        "pdb_path": str(out_pdb),
        "pdb_name": out_pdb.name,
        "ddg": ddg_val,
        "rmsd": round(rmsd_val, 4) if rmsd_val is not None else None,
        "mean_disp": round(mean_disp_val, 4) if mean_disp_val is not None else None,
        "elapsed_sec": round(elapsed, 1),
        "flexpep_result": result,
        "error": None,
    }


# ── MutateResidue fallback (OOD용) ───────────────────────────────────────────

_MUTATE_SCRIPT = """
import sys, json
import pyrosetta
pyrosetta.init('-mute all', silent=True)
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta import pose_from_pdb

native_pdb  = sys.argv[1]
target_seq  = sys.argv[2]
out_pdb     = sys.argv[3]
native_seq  = "AGCKNFFWKTFTSC"
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
result_seq = ''.join(pose.residue(i).name1() for i in range(chain_b_start, chain_b_end+1))
print(json.dumps({"ok": result_seq == target_seq, "result_seq": result_seq}))
"""


def build_mutate_approx(seq: str, rank: int) -> Optional[Path]:
    """MutateResidue로 backbone=native 근사 PDB 생성 (OOD 표준 A 이외 서열용)."""
    rank_str = f"{rank:02d}"
    out_pdb = PDB_DIR / f"rank{rank_str}_{seq}.pdb"

    if out_pdb.exists():
        return out_pdb

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(_MUTATE_SCRIPT)
        tmp_script = tf.name

    try:
        proc = subprocess.run(
            [BIOTOOLS_PYTHON, tmp_script, str(NATIVE_PDB), seq, str(out_pdb)],
            capture_output=True, text=True, timeout=120,
        )
        json_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
        if not json_lines:
            return None
        result = json.loads(json_lines[-1])
        return out_pdb if result.get("ok") and out_pdb.exists() else None
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_script)
        except OSError:
            pass


# ── candidates.json 갱신 ─────────────────────────────────────────────────────

def update_candidates_json(
    dock_results: list[dict],
    ood_results: list[dict],
) -> list[dict]:
    """현재 candidates.json을 읽어 structure_source 정직화 후 재저장.

    재도킹 성공 → structure_source="docked", pdb_file 갱신
    OOD/실패 → structure_source="mutated_approx" 또는 "none"
    """
    # 현재 candidates.json 로드
    if CANDIDATES_JSON.exists():
        with open(CANDIDATES_JSON, encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        candidates = []

    # rank → dock_result 맵핑
    dock_map: dict[int, dict] = {r["rank"]: r for r in dock_results}
    ood_map: dict[int, dict] = {r["rank"]: r for r in ood_results}

    updated = []
    for cand in candidates:
        rank = cand.get("rank")
        source = cand.get("source", "")

        if source != "silo_b" or rank == 0:
            # native / Silo A 는 건드리지 않음
            updated.append(cand)
            continue

        # 재도킹 성공 여부 확인
        if rank in dock_map:
            dr = dock_map[rank]
            if dr["success"]:
                cand = dict(cand)
                cand["structure_source"] = "docked"
                cand["pdb_file"] = "pdb/" + dr["pdb_name"]
                cand["redock_ddg"] = dr.get("ddg")
                cand["redock_rmsd_vs_native"] = dr.get("rmsd")
                cand["redock_mean_disp"] = dr.get("mean_disp")
                cand["caveat"] = ""
            else:
                # 도킹 실패 → mutated_approx fallback
                cand = dict(cand)
                if rank in ood_map and ood_map[rank].get("pdb_path"):
                    cand["structure_source"] = "mutated_approx"
                    cand["caveat"] = (
                        "근사 구조 — FlexPepDock 실패 후 MutateResidue fallback. "
                        "backbone은 native와 동일(side-chain만 치환). 재도킹 아님."
                    )
                else:
                    cand["structure_source"] = "none"
                    cand["caveat"] = f"FlexPepDock 실패: {dr.get('error','')}"
                cand["redock_error"] = dr.get("error")
        elif rank in ood_map:
            oor = ood_map[rank]
            cand = dict(cand)
            if oor.get("pdb_path"):
                cand["structure_source"] = "mutated_approx"
                cand["pdb_file"] = "pdb/" + Path(oor["pdb_path"]).name
                cand["caveat"] = (
                    "근사 구조 — 실제 재도킹 아님. backbone은 native와 동일(side-chain만 치환). "
                    f"OOD 이유: {oor.get('ood_reason','')}"
                )
            else:
                cand["structure_source"] = "none"
                cand["caveat"] = f"OOD skip: {oor.get('ood_reason','')}"
        updated.append(cand)

    # 원자적 쓰기
    tmp_path = CANDIDATES_JSON.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    tmp_path.rename(CANDIDATES_JSON)
    return updated


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="리더보드 top N → FlexPepDock 재도킹 → structure_view 갱신"
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N,
                        help=f"처리할 Silo B top N (기본 {DEFAULT_TOP_N})")
    parser.add_argument("--nstruct", type=int, default=DEFAULT_NSTRUCT,
                        help=f"FlexPepDock nstruct (기본 {DEFAULT_NSTRUCT})")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"병렬 워커 수 (기본 {DEFAULT_MAX_WORKERS}, 엔진 방해 최소화)")
    parser.add_argument("--dry-run", action="store_true",
                        help="OOD 판정만, 실제 도킹 없음")
    args = parser.parse_args()

    print(f"=== redock_structure_view.py ===")
    print(f"  top_n={args.top}, nstruct={args.nstruct}, max_workers={args.max_workers}")
    print(f"  native PDB: {NATIVE_PDB}")
    print(f"  FlexPepDock: {FLEXPEP_DOCK_PY}")
    print(f"  dry_run: {args.dry_run}")

    # 경로 검증
    if not NATIVE_PDB.exists():
        print(f"[ERROR] native PDB 없음: {NATIVE_PDB}")
        sys.exit(1)
    if not FLEXPEP_DOCK_PY.exists():
        print(f"[ERROR] flexpep_dock.py 없음: {FLEXPEP_DOCK_PY}")
        sys.exit(1)
    if not Path(BIOTOOLS_PYTHON).exists():
        print(f"[ERROR] bio-tools Python 없음: {BIOTOOLS_PYTHON}")
        sys.exit(1)

    PDB_DIR.mkdir(parents=True, exist_ok=True)

    # 1) 리더보드 top N 로드
    print(f"\n[1] 리더보드 top {args.top} 로드...")
    top_entries = load_leaderboard_top(args.top)
    if not top_entries:
        print("[ERROR] 리더보드 엔트리 없음")
        sys.exit(1)

    # 2) OOD 분류
    dock_targets: list[tuple[str, int, dict]] = []  # (seq, rank, lb_entry)
    ood_list: list[dict] = []

    for rank_idx, entry in enumerate(top_entries, start=1):
        seq = entry.get("sequence", "")
        if not seq or seq == NATIVE_SEQ:
            continue
        ood_flag, ood_reason = is_ood(seq, entry)
        if ood_flag:
            ood_list.append({"seq": seq, "rank": rank_idx, "ood_reason": ood_reason})
            print(f"  rank{rank_idx:02d} {seq} → OOD: {ood_reason}")
        else:
            dock_targets.append((seq, rank_idx, entry))
            print(f"  rank{rank_idx:02d} {seq} → 재도킹 대상")

    print(f"\n  재도킹 대상: {len(dock_targets)}개")
    print(f"  OOD (MutateResidue fallback): {len(ood_list)}개")

    # 3) OOD → MutateResidue fallback PDB 생성
    ood_results: list[dict] = []
    if ood_list and not args.dry_run:
        print(f"\n[2] OOD 후보 MutateResidue fallback 생성 ({len(ood_list)}개)...")
        for oor in ood_list:
            seq, rank = oor["seq"], oor["rank"]
            # OOD 중 표준 AA 이며 14aa 이면 MutateResidue 시도
            if (len(seq) == 14
                    and all(c in STANDARD_AA_SET for c in seq)):
                print(f"  [{rank:02d}] MutateResidue fallback: {seq}")
                pdb_path = build_mutate_approx(seq, rank)
                ood_results.append({
                    "seq": seq, "rank": rank,
                    "pdb_path": str(pdb_path) if pdb_path else None,
                    "ood_reason": oor["ood_reason"],
                })
                if pdb_path:
                    print(f"    → mutated_approx: {pdb_path.name}")
                else:
                    print(f"    → MutateResidue 실패 (structure_source=none)")
            else:
                # D-aa 등 MutateResidue 자체 불가
                ood_results.append({
                    "seq": seq, "rank": rank,
                    "pdb_path": None,
                    "ood_reason": oor["ood_reason"],
                })
                print(f"  [{rank:02d}] {seq} → D-aa/비표준, MutateResidue 불가 (structure_source=none)")

    # 4) FlexPepDock 재도킹 (병렬)
    dock_results: list[dict] = []
    if dock_targets and not args.dry_run:
        print(f"\n[3] FlexPepDock 재도킹 ({len(dock_targets)}개, "
              f"nstruct={args.nstruct}, max_workers={args.max_workers})...")

        workers = min(args.max_workers, len(dock_targets))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_dock_one_sequence, seq, rank, args.nstruct): (seq, rank)
                for seq, rank, _ in dock_targets
            }
            for fut in as_completed(futures):
                seq, rank = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "seq": seq, "rank": rank, "success": False,
                        "pdb_path": None, "ddg": None, "rmsd": None, "mean_disp": None,
                        "error": f"future 예외: {exc}",
                    }
                dock_results.append(result)
                status = "성공" if result["success"] else "실패"
                rmsd_str = f"RMSD={result['rmsd']:.4f}Å" if result.get("rmsd") is not None else "RMSD=N/A"
                disp_str = f"mean_disp={result['mean_disp']:.4f}Å" if result.get("mean_disp") is not None else ""
                ddg_str = f"ddG={result['ddg']:.2f}" if result.get("ddg") is not None else "ddG=N/A"
                print(f"  rank{rank:02d} {seq} → {status} | {ddg_str} | {rmsd_str} {disp_str}")
                if not result["success"]:
                    print(f"    ERROR: {result.get('error','')}")

    # 5) candidates.json 갱신
    print(f"\n[4] candidates.json 갱신...")
    updated = update_candidates_json(dock_results, ood_results)
    n_docked = sum(1 for c in updated if c.get("structure_source") == "docked" and c.get("source") == "silo_b")
    n_approx = sum(1 for c in updated if c.get("structure_source") == "mutated_approx")
    n_none = sum(1 for c in updated if c.get("structure_source") == "none")
    print(f"  docked(진짜): {n_docked}개")
    print(f"  mutated_approx(근사): {n_approx}개")
    print(f"  none(skip): {n_none}개")

    # 6) 검증: CA RMSD 성공 기준 (> 0.5Å)
    print(f"\n[5] 검증 결과:")
    success_count = 0
    fail_count = 0
    for r in dock_results:
        if not r["success"]:
            continue
        rmsd = r.get("rmsd")
        mean_disp = r.get("mean_disp")
        passed = rmsd is not None and rmsd > 0.5
        sym = "PASS" if passed else "FAIL(RMSD<=0.5A, backbone 미변경 의심)"
        print(f"  rank{r['rank']:02d} {r['seq']}: RMSD={rmsd} mean_disp={mean_disp} → {sym}")
        if passed:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n=== 완료 ===")
    print(f"  재도킹 성공(CA 이동 확인): {success_count}개 / {len(dock_results)}개 시도")
    print(f"  CA RMSD < 0.5Å (backbone 미변경 의심): {fail_count}개")
    print(f"  OOD (mutated_approx/none): {len(ood_results)}개")
    print(f"  candidates.json 갱신 완료: {CANDIDATES_JSON}")


if __name__ == "__main__":
    main()
