#!/usr/bin/env python3
"""cross_silo_validation.py
============================
Silo A(de novo) + Silo B(mutation) 상위 후보를 교차 검증.

목표:
  - Silo A 리더보드 top N(기본 5) + Silo B 리더보드 top N(기본 5) 읽기
  - 이미 검증한 candidate_id는 상태파일(cross_validated.state)로 dedup
  - 각 후보에 대해:
      ① robust FlexPepDock (nstruct=5) → ddg_median/sd/n_converged
      ② MM-GBSA (mmgbsa_rescore, GPU2)
      ③ 저복잡도 필터 (_is_low_complexity)
      ④ off-target robust 재도킹 (SSTR1/3/4/5, nstruct=10) → robust Δmargin
  - verdict 판정: confirmed / refuted / uncertain (selectivity 반영)
  - 출력: runs/silo_a_flow/cross_validation.json

verdict 기준:
  confirmed  : robust ddg_median < 0, n_converged >= 3, 저복잡도 아님,
               |robust_median - 원본_single| / |원본_single| <= 0.5
  refuted    : robust ddg_median >= 0 또는 저복잡도 아티팩트
  uncertain  : n_converged < 3 (수렴 부족)

selectivity 가점/감점 (verdict가 confirmed일 때):
  robust_delta_margin > 0 → confirmed_selective
  robust_delta_margin <= 0 → confirmed_nonselective
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
SILO_A_FLOW_DIR = REPO_ROOT / "runs" / "silo_a_flow"
SILO_A_LB = SILO_A_FLOW_DIR / "silo_a_leaderboard.json"
SILO_B_LB = REPO_ROOT / "runs" / "pyrosetta_flow" / "global_selectivity_leaderboard.json"
EXPERIMENT_LOG = REPO_ROOT / "runs" / "pyrosetta_flow" / "experiment_log.jsonl"
PYROSETTA_FLOW_DIR = REPO_ROOT / "runs" / "pyrosetta_flow"
CURATED_DIR = REPO_ROOT / "data" / "somatostatin_receptor" / "curated"
NATIVE_SEL_BASELINE = CURATED_DIR / "native_selectivity_baseline.json"
STATE_FILE = SILO_A_FLOW_DIR / "cross_validated.state"
OUT_JSON = SILO_A_FLOW_DIR / "cross_validation.json"
WORK_DIR = SILO_A_FLOW_DIR / "cross_validation_work"
MMGBSA_CONSENSUS = REPO_ROOT / "runs" / "pyrosetta_flow" / "mmgbsa_consensus.json"
# 검증된 native baseline v2 (nstruct=10, HIGH 신뢰) — 우선 사용
NATIVE_ROBUST_BASELINE_V2 = REPO_ROOT / "runs" / "pyrosetta_flow" / "native_robust_baseline_v2.json"
# native 도킹 기준 최소 차이 권장값 (sd=12.26 REU 고려, 통계적 유의 threshold)
_NATIVE_BEATS_MIN_DIFF_REU: float = 15.0

# ── off-target 수용체 목록 ──────────────────────────────────────────────────────
OFFTARGET_RECEPTORS: dict[str, Path] = {
    "SSTR1": CURATED_DIR / "SSTR1_receptor.pdb",
    "SSTR3": CURATED_DIR / "SSTR3_receptor.pdb",
    "SSTR4": CURATED_DIR / "SSTR4_receptor.pdb",
    "SSTR5": CURATED_DIR / "SSTR5_receptor.pdb",
}

# ── native selectivity baseline 로드 ──────────────────────────────────────────
def _load_native_selectivity_baseline() -> dict:
    """native_selectivity_baseline.json 로드. 없으면 빈 dict."""
    if NATIVE_SEL_BASELINE.exists():
        with open(NATIVE_SEL_BASELINE, encoding="utf-8") as f:
            return json.load(f)
    return {"sstr2": None, "offtarget": {}, "margin": None}

_NATIVE_SEL: dict = _load_native_selectivity_baseline()

# ── Silo A candidate_id 파싱 정규식 ───────────────────────────────────────────
_SILO_A_CAND_RE = re.compile(r"silo_a_(\d{8}T\d{6}Z)_bb(\d+)_sq(\d+)$")

# ── Silo B candidate_id 패턴 ──────────────────────────────────────────────────
_SILO_B_CAND_RE = re.compile(r"iter(\d+)_cand(\d+)")

# ── 저복잡도 판정 기준 ─────────────────────────────────────────────────────────
_LC_MONO_FRAC = 0.6
_LC_MIN_UNIQ = 3
_LC_MIN_ENTROPY = 1.5


def _is_low_complexity(seq: str) -> tuple[bool, str]:
    """서열 저복잡도 아티팩트 판정.

    기준 (OR):
      (a) 단일 잔기 비율 > 0.6
      (b) 고유 잔기 수 <= 3
      (c) Shannon entropy < 1.5 bits

    Returns:
        (is_artifact, reason)
    """
    if not seq:
        return True, "빈 서열"
    n = len(seq)
    cnt: Counter = Counter(seq.upper())
    max_frac = max(cnt.values()) / n
    if max_frac > _LC_MONO_FRAC:
        aa = max(cnt, key=cnt.get)  # type: ignore[arg-type]
        return True, f"단일잔기과다({aa}={max_frac:.0%}>{_LC_MONO_FRAC:.0%})"
    n_unique = len(cnt)
    if n_unique <= _LC_MIN_UNIQ:
        return True, f"고유잔기부족({n_unique}<={_LC_MIN_UNIQ})"
    entropy = -sum((v / n) * math.log2(v / n) for v in cnt.values())
    if entropy < _LC_MIN_ENTROPY:
        return True, f"shannon_entropy낮음({entropy:.2f}<{_LC_MIN_ENTROPY})"
    return False, ""


# ── Silo A PDB 탐색 ────────────────────────────────────────────────────────────

def _find_silo_a_pdb(candidate_id: str) -> Optional[Path]:
    """Silo A candidate_id로부터 복합체 PDB를 탐색.

    매핑:
      candidate_id = silo_a_{ts}_bb{bb}_sq{sq}
      epoch_dir    = runs/silo_a_flow/epoch_{ts}/
      PDB 우선순위: silo_a_{ts}_bb{bb}_sq{sq}.pdb > diffpep_{ts}_bb{bb}_sq{sq}.pdb

    Returns:
        원본 PDB 경로 (복사 없음), 없으면 None
    """
    m = _SILO_A_CAND_RE.match(candidate_id)
    if m is None:
        print(f"  [WARN] candidate_id 파싱 실패: {candidate_id}", file=sys.stderr)
        return None

    ts, bb, sq = m.group(1), m.group(2), m.group(3)
    epoch_dir = SILO_A_FLOW_DIR / f"epoch_{ts}"
    if not epoch_dir.exists():
        print(f"  [WARN] epoch 디렉토리 없음: {epoch_dir}", file=sys.stderr)
        return None

    for prefix in ("silo_a", "diffpep"):
        candidate_pdb = epoch_dir / f"{prefix}_{ts}_bb{bb}_sq{sq}.pdb"
        if candidate_pdb.exists():
            return candidate_pdb

    print(
        f"  [WARN] 복합체 PDB 없음 (epoch={ts}, bb={bb}, sq={sq})",
        file=sys.stderr,
    )
    return None


# ── Silo B PDB 탐색 ────────────────────────────────────────────────────────────

def _build_silo_b_seq_to_pdb_map() -> dict[str, Path]:
    """experiment_log.jsonl에서 서열 → 최신 PDB 경로 매핑 구성.

    패턴: run_id=sst14_mutdock_XXXXXX, candidate_id=iterNN_candMMM
    → runs/pyrosetta_flow/archives/{run_id}/iter_{NN}/cand_{MMM}.pdb

    Returns:
        {sequence: pdb_path} — 서열마다 가장 최근 (마지막 등장) PDB 경로
    """
    if not EXPERIMENT_LOG.exists():
        return {}

    seq_to_pdb: dict[str, Path] = {}
    try:
        with open(EXPERIMENT_LOG, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq: str = d.get("sequence", "")
                run_id: str = d.get("run_id", "")
                cand_id: str = d.get("candidate_id", "")
                if not (seq and run_id and cand_id):
                    continue
                m = _SILO_B_CAND_RE.search(cand_id)
                if m is None:
                    continue
                iter_num = int(m.group(1))
                cand_num = int(m.group(2))
                pdb_path = (
                    PYROSETTA_FLOW_DIR
                    / "archives"
                    / run_id
                    / f"iter_{iter_num:02d}"
                    / f"cand_{cand_num:03d}.pdb"
                )
                if pdb_path.exists():
                    seq_to_pdb[seq] = pdb_path  # 마지막 등장 = 최신
    except OSError:
        pass
    return seq_to_pdb


# 모듈 수준 캐시 (초기화 지연)
_SILO_B_SEQ_MAP: Optional[dict[str, Path]] = None


def _get_silo_b_seq_map() -> dict[str, Path]:
    """Silo B 서열-PDB 매핑 캐시 반환."""
    global _SILO_B_SEQ_MAP
    if _SILO_B_SEQ_MAP is None:
        _SILO_B_SEQ_MAP = _build_silo_b_seq_to_pdb_map()
        print(
            f"  [Silo B PDB map] {len(_SILO_B_SEQ_MAP)}개 서열 매핑 로드",
            file=sys.stderr,
        )
    return _SILO_B_SEQ_MAP


def find_silo_b_pdb(sequence: str) -> Optional[Path]:
    """Silo B 후보 서열로부터 refined 복합체 PDB를 탐색.

    Args:
        sequence: 14aa 펩타이드 서열

    Returns:
        PDB 경로 (없으면 None)
    """
    mapping = _get_silo_b_seq_map()
    pdb = mapping.get(sequence)
    if pdb is None:
        print(f"  [WARN] Silo B PDB 없음: {sequence}", file=sys.stderr)
    return pdb


# ── off-target robust 재도킹 ───────────────────────────────────────────────────

def run_offtarget_robust(
    complex_pdb: Path,
    candidate_id: str,
    nstruct: int = 10,
) -> dict:
    """SSTR1/3/4/5 off-target robust 재도킹 → robust Δmargin 계산.

    각 수용체에 대해 offtarget_dock.py를 subprocess로 호출(bio-tools conda env).
    robust Δmargin = offtarget_ddg_mean - sstr2_robust_ddg_median.

    Args:
        complex_pdb: SSTR2 복합체 PDB (Silo A or B)
        candidate_id: 로깅용 식별자
        nstruct: off-target 재도킹 nstruct (기본 10, cross-silo pool 재도킹 정책)

    Returns:
        dict with:
          offtarget_ddg: {receptor_name: ddg_median}
          robust_delta_margin: min(off_target ddg) - sstr2_robust_ddg (양수=선택적)
          offtarget_elapsed_s: 총 소요시간
          offtarget_error: 오류 메시지 (있을 경우)
    """
    offtarget_dock_script = REPO_ROOT / "AG_src" / "scripts" / "offtarget_dock.py"
    if not offtarget_dock_script.exists():
        return {
            "offtarget_ddg": {},
            "robust_delta_margin": None,
            "offtarget_elapsed_s": 0.0,
            "offtarget_error": f"offtarget_dock.py 없음: {offtarget_dock_script}",
        }

    offtarget_ddgs: dict[str, Optional[float]] = {}
    t0 = time.perf_counter()

    for receptor_name, receptor_pdb in OFFTARGET_RECEPTORS.items():
        if not receptor_pdb.exists():
            print(
                f"  [off-target] {receptor_name} PDB 없음: {receptor_pdb}",
                file=sys.stderr,
            )
            offtarget_ddgs[receptor_name] = None
            continue

        out_pdb = WORK_DIR / f"{candidate_id}_ot_{receptor_name}.pdb"
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        cmd = [
            "conda", "run", "-n", "bio-tools", "--no-capture-output",
            "python",
            str(offtarget_dock_script),
            "--sstr2-complex", str(complex_pdb),
            "--offtarget-receptor", str(receptor_pdb),
            "--output", str(out_pdb),
            "--pre-aligned",
        ]
        env_override = {"FLEXPEP_NSTRUCT": str(nstruct)}

        print(
            f"  [off-target] {candidate_id} vs {receptor_name} nstruct={nstruct} 시작",
            file=sys.stderr,
        )
        t_rec = time.perf_counter()
        try:
            import os
            run_env = {**os.environ, **env_override}
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=run_env,
            )
            elapsed_rec = round(time.perf_counter() - t_rec, 1)
            # stdout은 JSON only
            stdout = proc.stdout.strip()
            if not stdout:
                print(
                    f"  [off-target] {receptor_name} stdout 없음 (rc={proc.returncode})",
                    file=sys.stderr,
                )
                offtarget_ddgs[receptor_name] = None
                continue
            # 마지막 JSON 줄 파싱 (conda run이 앞에 출력을 끼울 수 있음)
            result_json: Optional[dict] = None
            for raw in stdout.split("\n"):
                raw = raw.strip()
                if raw.startswith("{"):
                    try:
                        result_json = json.loads(raw)
                    except json.JSONDecodeError:
                        pass
            if result_json is None:
                print(
                    f"  [off-target] {receptor_name} JSON 파싱 실패: {stdout[:200]}",
                    file=sys.stderr,
                )
                offtarget_ddgs[receptor_name] = None
                continue
            if "error" in result_json:
                print(
                    f"  [off-target] {receptor_name} 오류: {result_json['error']}",
                    file=sys.stderr,
                )
                offtarget_ddgs[receptor_name] = None
                continue
            ddg_val = result_json.get("ddg_median") or result_json.get("ddg")
            offtarget_ddgs[receptor_name] = float(ddg_val) if ddg_val is not None else None
            print(
                f"  [off-target] {receptor_name} 완료 {elapsed_rec}s | ddg_median={ddg_val}",
                file=sys.stderr,
            )
        except subprocess.TimeoutExpired:
            elapsed_rec = round(time.perf_counter() - t_rec, 1)
            print(
                f"  [off-target] {receptor_name} 타임아웃 {elapsed_rec}s",
                file=sys.stderr,
            )
            offtarget_ddgs[receptor_name] = None
        except Exception as exc:
            elapsed_rec = round(time.perf_counter() - t_rec, 1)
            print(
                f"  [off-target] {receptor_name} 예외 {elapsed_rec}s: {exc}",
                file=sys.stderr,
            )
            offtarget_ddgs[receptor_name] = None

    elapsed_total = round(time.perf_counter() - t0, 1)

    # robust Δmargin 계산: off-target ddg 평균 - SSTR2 ddg (값 있는 것만)
    # Δmargin > 0: SSTR2에 더 강하게 결합 (선택적)
    # 양수가 좋음 = (SSTR2 ddg) < (off-target ddg 평균) 이면 양수
    valid_ot_ddgs = [v for v in offtarget_ddgs.values() if v is not None]
    robust_delta_margin: Optional[float] = None
    if valid_ot_ddgs:
        # off_target_mean - sstr2_ddg: sstr2_ddg 음수, off_target 덜 음수면 양수
        # (표준 정의: margin = sstr2_ddg - offtarget_ddg → 음수끼리 비교)
        # 기존 파이프라인 delta_margin = off_target_mean_ddg - sstr2_ddg (양수=SSTR2선호)
        ot_mean = statistics.mean(valid_ot_ddgs)
        # robust_delta_margin은 본 함수 호출자가 sstr2_ddg를 주입해야 함
        # 여기서는 offtarget_ddg만 반환하고, 호출자가 delta_margin 계산
        pass

    return {
        "offtarget_ddg": offtarget_ddgs,
        "robust_delta_margin": None,  # 호출자에서 sstr2 ddg와 조합 후 계산
        "offtarget_elapsed_s": elapsed_total,
        "offtarget_error": None,
    }


def compute_robust_delta_margin(
    sstr2_robust_ddg: Optional[float],
    offtarget_ddgs: dict[str, Optional[float]],
) -> Optional[float]:
    """robust Δmargin 계산.

    Δmargin = mean(off_target_ddg) - sstr2_ddg
    양수: SSTR2에 더 강하게 결합 (선택적), 음수: off-target에도 강하게 결합 (비선택적)

    Args:
        sstr2_robust_ddg: SSTR2 robust ddg_median
        offtarget_ddgs: {receptor: ddg_median}

    Returns:
        Δmargin float 또는 None (계산 불가)
    """
    if sstr2_robust_ddg is None:
        return None
    valid_ot = [v for v in offtarget_ddgs.values() if v is not None]
    if not valid_ot:
        return None
    ot_mean = statistics.mean(valid_ot)
    return round(ot_mean - sstr2_robust_ddg, 4)


# ── 상태 파일 관리 ─────────────────────────────────────────────────────────────

def load_state() -> dict:
    """cross_validated.state 로드. 없으면 빈 dict."""
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"validated": {}}


def save_state(state: dict) -> None:
    """상태 파일 저장."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ── 기존 결과 로드 ─────────────────────────────────────────────────────────────

def load_existing_results() -> dict:
    """cross_validation.json 로드. 없으면 빈 구조."""
    if OUT_JSON.exists():
        with open(OUT_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {"results": {}, "generated_at": None}


# ── robust FlexPepDock 재도킹 ─────────────────────────────────────────────────

def run_robust_flexpep(
    pdb_path: Path,
    candidate_id: str,
    nstruct: int = 5,
) -> dict:
    """FlexPepDock nstruct=N robust 재도킹 실행.

    bio-tools conda 환경에서 pyrosetta import 필요.

    Returns:
        dict with: ddg_median, ddg_sd, ddg_mean, ddg_min, n_converged, n_total, converged
    """
    try:
        import pyrosetta
    except ImportError:
        return {
            "ddg_median": None, "ddg_sd": None, "ddg_mean": None,
            "ddg_min": None, "n_converged": 0, "n_total": nstruct,
            "converged": False, "error": "pyrosetta not available",
        }

    # flexpep_dock.py 임포트 (sys.path 보완)
    ag_src = REPO_ROOT / "AG_src" / "scripts"
    if str(ag_src) not in sys.path:
        sys.path.insert(0, str(ag_src))

    from flexpep_dock import run_flexpep_refine, init_pyrosetta  # type: ignore

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    output_pdb = str(WORK_DIR / f"{candidate_id}_refined.pdb")

    print(f"  [FlexPepDock] {candidate_id} nstruct={nstruct} 시작", file=sys.stderr)
    t0 = time.perf_counter()
    try:
        init_pyrosetta()
        _pose, stats = run_flexpep_refine(str(pdb_path), output_pdb, nstruct=nstruct)
        elapsed = round(time.perf_counter() - t0, 1)
        print(
            f"  [FlexPepDock] 완료 {elapsed}s | "
            f"ddg_median={stats.get('ddg_median')} n_converged={stats.get('n_converged')}",
            file=sys.stderr,
        )
        stats["elapsed_s"] = elapsed
        stats.pop("error", None)
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


# ── MM-GBSA 재스코어링 ─────────────────────────────────────────────────────────

def run_mmgbsa(
    pdb_path: Path,
    candidate_id: str,
    device_index: str = "2",
) -> dict:
    """MM-GBSA 재스코어링.

    Silo A PDB 구조:
      - chain A = SSTR2 수용체 (de novo 파이프라인 표준)
      - chain B = 펩타이드
    단, 원본 PDB chain 구성이 다를 수 있으니 chain 목록을 확인 후 적용.

    Returns:
        mmgbsa_rescore 결과 dict
    """
    mmgbsa_src = REPO_ROOT / "pyrosetta_flow"
    if str(mmgbsa_src) not in sys.path:
        sys.path.insert(0, str(mmgbsa_src))

    try:
        from mmgbsa_rescore import mmgbsa_rescore  # type: ignore
    except ImportError as e:
        return {"ok": False, "dg_bind": None, "error": f"mmgbsa_rescore import 실패: {e}"}

    # Silo A PDB에서 chain 정보 파악
    receptor_chains, peptide_chain = _detect_chains(pdb_path)
    if receptor_chains is None:
        return {
            "ok": False, "dg_bind": None,
            "error": f"chain 감지 실패: {pdb_path}",
        }

    print(
        f"  [MM-GBSA] {candidate_id} receptor={receptor_chains} peptide={peptide_chain}",
        file=sys.stderr,
    )
    t0 = time.perf_counter()
    try:
        result = mmgbsa_rescore(
            complex_pdb=str(pdb_path),
            receptor_chains=receptor_chains,
            peptide_chain=peptide_chain,
            platform_name="CUDA",
            device_index=device_index,
            n_snapshots=1,
        )
        elapsed = round(time.perf_counter() - t0, 1)
        result["elapsed_s"] = elapsed
        print(
            f"  [MM-GBSA] 완료 {elapsed}s | ok={result.get('ok')} dg_bind={result.get('dg_bind')}",
            file=sys.stderr,
        )
        return result
    except Exception as exc:
        elapsed = round(time.perf_counter() - t0, 1)
        print(f"  [MM-GBSA] 실패 {elapsed}s: {exc}", file=sys.stderr)
        return {"ok": False, "dg_bind": None, "elapsed_s": elapsed, "error": str(exc)}


def _detect_chains(pdb_path: Path) -> tuple[Optional[list[str]], Optional[str]]:
    """PDB 파일에서 chain 목록을 읽어 수용체/펩타이드 chain 추정.

    실측 확인(2026-06-26):
      Silo A PDB (silo_a_flow/epoch_*/): chain A=SSTR2 수용체(472aa), chain B=펩타이드(16aa)
      cross_validation_work/*_refined.pdb: FlexPepDock 출력 → 잔기 수 기반 자동 판별
    판별 방식: 잔기 수(ATOM 레코드 행 수, 중복 포함) 기준 — 짧은 쪽=펩타이드.
    하드코딩 없음. 잔기 수 기반 자동 판별이므로 chain A/B 순서에 무관.

    Returns:
        (receptor_chains, peptide_chain) — 감지 실패 시 (None, None)
    """
    chains: dict[str, int] = {}  # chain_id → residue count
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    ch = line[21]
                    if ch.strip():
                        chains[ch] = chains.get(ch, 0) + 1
    except OSError:
        return None, None

    if not chains:
        return None, None

    if len(chains) == 1:
        ch = list(chains.keys())[0]
        return [ch], ch  # 단일 chain — 그대로 사용(정보 부족)

    # 가장 짧은 chain = 펩타이드
    pep_chain = min(chains, key=lambda c: chains[c])
    rec_chains = [c for c in chains if c != pep_chain]
    return rec_chains, pep_chain


# ── C-tail artifact 판정 ──────────────────────────────────────────────────────
# SSTR2(P30874) transmembrane domain C-terminal boundary ≈ 잔기 320
# 이 값 이후 수용체 잔기는 intracellular tail — orthosteric pocket 아님
_SSTR2_TM_END_RESIDUE = 320  # 수용체 PDB 잔기 번호 기준 (intracellular if > this)


def detect_ctail_artifact(
    pdb_path: Path,
    receptor_residue_threshold: int = _SSTR2_TM_END_RESIDUE,
    ctail_contact_fraction_threshold: float = 0.5,
    contact_distance_angstrom: float = 6.0,
) -> tuple[bool, str]:
    """펩타이드 접촉 잔기가 수용체 C-tail(intracellular) 편중인지 판정.

    수용체 잔기번호 > receptor_residue_threshold인 잔기와의 접촉이
    전체 수용체 접촉 잔기 중 ctail_contact_fraction_threshold(50%) 초과이면
    orthosteric pocket 이 아닌 intracellular tail에 결합 → artifact.

    CA-CA 거리 <= contact_distance_angstrom 을 접촉으로 정의.

    Returns:
        (is_artifact, reason) — artifact 아니면 (False, "")
    """
    try:
        # PDB에서 CA 좌표 수집 (chain, resnum) -> (x, y, z)
        all_ca: dict[tuple[str, int], tuple[float, float, float]] = {}

        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                if len(line) < 54:
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                chain = line[21]
                try:
                    resnum = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                all_ca[(chain, resnum)] = (x, y, z)

        if not all_ca:
            return False, "CA 좌표 없음"

        # chain 길이로 수용체/펩타이드 chain 구분
        chain_residue_counts: dict[str, set[int]] = {}
        for (ch, rn) in all_ca:
            chain_residue_counts.setdefault(ch, set()).add(rn)

        if len(chain_residue_counts) < 2:
            return False, "단일 chain — 판정 불가"

        sorted_chains_ct = sorted(chain_residue_counts.items(), key=lambda x: len(x[1]))
        pep_chain_letter = sorted_chains_ct[0][0]   # 짧은 쪽 = 펩타이드
        rec_chain_letters = {c for c, _ in sorted_chains_ct[1:]}

        # 펩타이드/수용체 CA 분리
        pep_ca = {k: v for k, v in all_ca.items() if k[0] == pep_chain_letter}
        rec_ca_only = {k: v for k, v in all_ca.items() if k[0] in rec_chain_letters}

        if not pep_ca or not rec_ca_only:
            return False, "chain CA 분리 실패"

        # 접촉 수용체 잔기번호 수집 (CA-CA <= threshold)
        contacted_rec_resnums: list[int] = []
        for (_pch, _prn), (px, py, pz) in pep_ca.items():
            for (_rch, rrn), (rx, ry, rz) in rec_ca_only.items():
                dist = ((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2) ** 0.5
                if dist <= contact_distance_angstrom:
                    contacted_rec_resnums.append(rrn)

        if not contacted_rec_resnums:
            return False, "접촉 잔기 없음 (거리 기준 초과 — 미결합 가능성)"

        # C-tail 접촉 비율 계산
        ctail_contacts = [r for r in contacted_rec_resnums if r > receptor_residue_threshold]
        ctail_fraction = len(ctail_contacts) / len(contacted_rec_resnums)

        if ctail_fraction > ctail_contact_fraction_threshold:
            reason = (
                f"C-tail 접촉 과다 ({ctail_fraction:.0%}>{ctail_contact_fraction_threshold:.0%}, "
                f"잔기>{receptor_residue_threshold}): 수용체 intracellular tail 결합 의심 "
                f"(orthosteric pocket 아님)"
            )
            return True, reason

        return False, ""

    except OSError as e:
        return False, f"PDB 읽기 실패: {e}"
    except Exception as e:
        return False, f"예외 발생: {e}"


# ── verdict 판정 ──────────────────────────────────────────────────────────────

def determine_verdict(
    orig_ddg: Optional[float],
    robust_ddg_median: Optional[float],
    n_converged: int,
    low_complexity: bool,
    robust_delta_margin: Optional[float] = None,
) -> str:
    """Cross-silo 검증 verdict 판정 (selectivity 반영).

    confirmed_selective   : confirmed + robust Δmargin > 0
    confirmed_nonselective: confirmed + robust Δmargin <= 0
    confirmed             : confirmed (off-target 미계산)
    refuted               : robust ddg_median >= 0 또는 저복잡도 아티팩트
    uncertain             : n_converged < 3 (수렴 부족)

    Returns:
        "confirmed_selective" | "confirmed_nonselective" | "confirmed" |
        "refuted" | "uncertain"
    """
    # 저복잡도 아티팩트 → 즉시 refuted
    if low_complexity:
        return "refuted"

    # 도킹 결과 없음 → uncertain
    if robust_ddg_median is None:
        return "uncertain"

    # 수렴 부족 → uncertain
    if n_converged < 3:
        return "uncertain"

    # robust ddG 양수 → refuted (결합 불안정)
    if robust_ddg_median >= 0:
        return "refuted"

    # 원본 단일값과 부합 여부 (상대 오차 <= 50%)
    if orig_ddg is not None and abs(orig_ddg) > 1e-6:
        rel_err = abs(robust_ddg_median - orig_ddg) / abs(orig_ddg)
        if rel_err > 0.5:
            # 값이 크게 다름 — uncertain으로 강등
            return "uncertain"

    # selectivity 반영
    if robust_delta_margin is not None:
        if robust_delta_margin > 0:
            return "confirmed_selective"
        else:
            return "confirmed_nonselective"

    return "confirmed"


# ── 단일 후보 검증 ────────────────────────────────────────────────────────────

def validate_candidate(
    entry: dict,
    nstruct: int = 5,
    ot_nstruct: int = 3,
    mmgbsa_device: str = "2",
    skip_mmgbsa: bool = False,
    skip_offtarget: bool = False,
    source: str = "silo_a",
) -> dict:
    """후보 1건을 robust 파이프라인으로 재검증.

    source="silo_a": Silo A de novo 후보 → _find_silo_a_pdb 사용
    source="silo_b": Silo B mutation 후보 → find_silo_b_pdb 사용 (독립 seed 재현성)

    Args:
        entry: 리더보드 entry dict
        nstruct: SSTR2 FlexPepDock nstruct
        ot_nstruct: off-target 재도킹 nstruct
        mmgbsa_device: GPU device 인덱스
        skip_mmgbsa: MM-GBSA 건너뜀
        skip_offtarget: off-target 재도킹 건너뜀
        source: "silo_a" | "silo_b"

    Returns:
        검증 결과 dict (cross_validation.json entries 형식)
    """
    candidate_id: str = entry["candidate_id"]
    seq: str = entry.get("sequence", "")
    orig_ddg: Optional[float] = entry.get("ddg")

    print(f"\n[cross-silo] 검증 시작: {candidate_id} [{source}]", file=sys.stderr)
    print(f"  서열: {seq}  원본 ddG: {orig_ddg}", file=sys.stderr)

    t_start = time.perf_counter()

    # ① 저복잡도 필터
    low_complexity, lc_reason = _is_low_complexity(seq)
    if low_complexity:
        print(f"  [LC] 저복잡도 아티팩트: {lc_reason}", file=sys.stderr)

    # ② PDB 탐색 (소스별 분기)
    if source == "silo_b":
        pdb_path = find_silo_b_pdb(seq)
    else:
        pdb_path = _find_silo_a_pdb(candidate_id)

    # C-tail artifact 초기값 (PDB 없을 때도 _make_result 인자로 전달)
    ctail_artifact: bool = False
    ctail_reason: str = ""

    if pdb_path is None:
        elapsed = round(time.perf_counter() - t_start, 1)
        return _make_result(
            entry=entry,
            source=source,
            low_complexity=low_complexity,
            lc_reason=lc_reason,
            flexpep_stats={
                "ddg_median": None, "ddg_sd": None, "n_converged": 0,
                "error": "PDB 없음 — epoch 디렉토리 또는 복합체 PDB 미발견",
            },
            mmgbsa_result={"ok": False, "dg_bind": None},
            offtarget_result={"offtarget_ddg": {}, "robust_delta_margin": None},
            elapsed_s=elapsed,
            ctail_artifact=ctail_artifact,
            ctail_reason=ctail_reason,
        )

    # ②-b C-tail artifact 판정 (PDB 존재 시)
    ctail_artifact, ctail_reason = detect_ctail_artifact(pdb_path)
    if ctail_artifact:
        print(f"  [C-tail] 아티팩트 감지: {ctail_reason}", file=sys.stderr)

    # ③ robust FlexPepDock
    flexpep_stats = run_robust_flexpep(pdb_path, candidate_id, nstruct=nstruct)

    # ④ MM-GBSA (skip_mmgbsa 옵션 또는 저복잡도 시 건너뜀)
    if skip_mmgbsa or low_complexity:
        mmgbsa_result: dict = {"ok": False, "dg_bind": None, "skipped": True}
    else:
        mmgbsa_result = run_mmgbsa(pdb_path, candidate_id, device_index=mmgbsa_device)

    # ⑤ off-target robust 재도킹
    if skip_offtarget or low_complexity:
        offtarget_result: dict = {
            "offtarget_ddg": {}, "robust_delta_margin": None, "skipped": True,
        }
    else:
        offtarget_result = run_offtarget_robust(pdb_path, candidate_id, nstruct=ot_nstruct)
        # robust Δmargin 계산 (SSTR2 robust ddg 주입)
        sstr2_robust_ddg: Optional[float] = flexpep_stats.get("ddg_median")
        offtarget_result["robust_delta_margin"] = compute_robust_delta_margin(
            sstr2_robust_ddg=sstr2_robust_ddg,
            offtarget_ddgs=offtarget_result.get("offtarget_ddg", {}),
        )
        print(
            f"  [selectivity] robust Δmargin={offtarget_result['robust_delta_margin']}",
            file=sys.stderr,
        )

    elapsed = round(time.perf_counter() - t_start, 1)
    return _make_result(
        entry=entry,
        source=source,
        low_complexity=low_complexity,
        lc_reason=lc_reason,
        flexpep_stats=flexpep_stats,
        mmgbsa_result=mmgbsa_result,
        offtarget_result=offtarget_result,
        elapsed_s=elapsed,
        ctail_artifact=ctail_artifact,
        ctail_reason=ctail_reason,
    )


def _make_result(
    entry: dict,
    source: str,
    low_complexity: bool,
    lc_reason: str,
    flexpep_stats: dict,
    mmgbsa_result: dict,
    offtarget_result: dict,
    elapsed_s: float,
    ctail_artifact: bool = False,
    ctail_reason: str = "",
) -> dict:
    """검증 결과 dict 생성."""
    import datetime

    candidate_id: str = entry["candidate_id"]
    seq: str = entry.get("sequence", "")
    orig_ddg: Optional[float] = entry.get("ddg")

    robust_ddg_median: Optional[float] = flexpep_stats.get("ddg_median")
    ddg_sd: Optional[float] = flexpep_stats.get("ddg_sd")
    n_converged: int = int(flexpep_stats.get("n_converged") or 0)
    mmgbsa_dg: Optional[float] = mmgbsa_result.get("dg_bind")
    robust_delta_margin: Optional[float] = offtarget_result.get("robust_delta_margin")
    offtarget_ddg: dict = offtarget_result.get("offtarget_ddg", {})

    verdict = determine_verdict(
        orig_ddg=orig_ddg,
        robust_ddg_median=robust_ddg_median,
        n_converged=n_converged,
        low_complexity=low_complexity,
        robust_delta_margin=robust_delta_margin,
    )

    # C-tail artifact → verdict 강등 (orthosteric pocket 결합 아님)
    if ctail_artifact and verdict != "refuted":
        print(
            f"  [C-tail] verdict 강등: {candidate_id} {verdict}→refuted (C-tail artifact)",
            file=sys.stderr,
        )
        verdict = "refuted"

    print(
        f"  [verdict] {candidate_id}: {verdict} "
        f"(robust_ddg={robust_ddg_median}, n_conv={n_converged}, "
        f"lc={low_complexity}, orig_ddg={orig_ddg}, "
        f"robust_Δmargin={robust_delta_margin}, "
        f"ctail_artifact={ctail_artifact})",
        file=sys.stderr,
    )

    return {
        "candidate_id": candidate_id,
        "source": source,
        "sequence": seq,
        # 원본 값 (소스 구분 없이 orig_ddg_single로 통일)
        "orig_ddg_single": orig_ddg,
        # 하위 호환: Silo A 필드 유지
        "silo_a_ddg_single": orig_ddg if source == "silo_a" else None,
        # robust 재도킹 결과
        "siloB_robust_ddg_median": robust_ddg_median,
        "ddg_sd": ddg_sd,
        "ddg_mean": flexpep_stats.get("ddg_mean"),
        "ddg_min": flexpep_stats.get("ddg_min"),
        "n_converged": n_converged,
        "n_total": int(flexpep_stats.get("n_total") or 0),
        # MM-GBSA
        "mmgbsa_dg": mmgbsa_dg,
        "mmgbsa_ok": bool(mmgbsa_result.get("ok")),
        # off-target selectivity
        "robust_delta_margin": robust_delta_margin,
        "offtarget_ddg": offtarget_ddg,
        # 저복잡도
        "low_complexity": low_complexity,
        "low_complexity_reason": lc_reason,
        # verdict
        "verdict": verdict,
        # C-tail artifact (intracellular tail 결합 여부)
        "binding_artifact": ctail_artifact,
        "binding_artifact_reason": ctail_reason,
        # 오류
        "flexpep_error": flexpep_stats.get("error"),
        "mmgbsa_error": mmgbsa_result.get("error"),
        "offtarget_error": offtarget_result.get("offtarget_error"),
        "elapsed_s": elapsed_s,
        "validated_at": datetime.datetime.utcnow().isoformat() + "Z",
        # 원본 메타 (소스 무관 공통)
        "orig_plddt": entry.get("plddt"),
        "orig_selectivity_margin": entry.get("selectivity_margin"),
        "orig_delta_margin": entry.get("delta_margin"),
        "orig_hc50": entry.get("hc50"),
        # 하위 호환: Silo A 전용 필드 유지
        "silo_a_plddt": entry.get("plddt") if source == "silo_a" else None,
        "silo_a_selectivity_margin": entry.get("selectivity_margin") if source == "silo_a" else None,
        "silo_a_delta_margin": entry.get("delta_margin") if source == "silo_a" else None,
        "silo_a_hc50": entry.get("hc50") if source == "silo_a" else None,
    }


# ── native baseline 로드 ──────────────────────────────────────────────────────

def _load_native_baselines() -> tuple[Optional[float], Optional[float], str]:
    """native 도킹 기준(native_ddg) 및 MM-GBSA 기준(native_dg) 반환.

    우선순위:
      1. native_robust_baseline_v2.json — nstruct=10, n_converged=8, HIGH 신뢰
         (검증된 진짜 native PDB: SSTR2_SST14_complex_boltz_1.pdb)
      2. mmgbsa_consensus.json::native_ddg — fallback (출처불명·LOW 신뢰 경고)

    MM-GBSA native_dg 는 항상 mmgbsa_consensus.json 에서 읽는다(별개 채점).

    Returns:
        (native_ddg, native_dg, native_ddg_source)
          native_ddg_source: "robust_v2_nstruct10" | "mmgbsa_consensus_fallback" | "unknown"
    """
    # ── 1순위: native_robust_baseline_v2.json ──
    if NATIVE_ROBUST_BASELINE_V2.exists():
        try:
            with open(NATIVE_ROBUST_BASELINE_V2, encoding="utf-8") as f:
                v2 = json.load(f)
            native_ddg: Optional[float] = v2.get("ddg_median")
            if native_ddg is not None:
                print(
                    f"[native-baseline] native_robust_baseline_v2 사용: "
                    f"ddg_median={native_ddg:.4f} REU "
                    f"(n_conv={v2.get('n_converged')}, sd={v2.get('ddg_sd'):.4f}, "
                    f"reliability={v2.get('reliability', '?')})",
                    file=sys.stderr,
                )
                # MM-GBSA native_dg 는 별도 파일에서
                native_dg: Optional[float] = None
                if MMGBSA_CONSENSUS.exists():
                    try:
                        with open(MMGBSA_CONSENSUS, encoding="utf-8") as f2:
                            mc = json.load(f2)
                        native_dg = mc.get("native_dg")
                    except (OSError, json.JSONDecodeError) as e2:
                        print(
                            f"[WARN] mmgbsa_consensus.json (native_dg) 읽기 실패: {e2}",
                            file=sys.stderr,
                        )
                return native_ddg, native_dg, "robust_v2_nstruct10"
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"[WARN] native_robust_baseline_v2.json 읽기 실패: {e} — mmgbsa fallback 사용",
                file=sys.stderr,
            )

    # ── 2순위 fallback: mmgbsa_consensus.json ──
    print(
        "[WARN] native_robust_baseline_v2.json 없음 또는 읽기 실패 — "
        "mmgbsa_consensus.json::native_ddg(-41.99) fallback 사용. "
        "이 값은 n_converged=2(LOW 신뢰)로 과도하게 강한 기준임. "
        "native_robust_baseline_v2.json 재생성 권고.",
        file=sys.stderr,
    )
    if not MMGBSA_CONSENSUS.exists():
        print(f"[WARN] mmgbsa_consensus.json 없음: {MMGBSA_CONSENSUS}", file=sys.stderr)
        return None, None, "unknown"
    try:
        with open(MMGBSA_CONSENSUS, encoding="utf-8") as f:
            mc = json.load(f)
        native_ddg_fb: Optional[float] = mc.get("native_ddg")
        native_dg_fb: Optional[float] = mc.get("native_dg")
        return native_ddg_fb, native_dg_fb, "mmgbsa_consensus_fallback"
    except (OSError, json.JSONDecodeError) as e:
        print(f"[WARN] mmgbsa_consensus.json 읽기 실패: {e}", file=sys.stderr)
        return None, None, "unknown"


# ── native 능가 후보 수집 (--beats-native 모드) ───────────────────────────────

def _collect_candidates_beats_native() -> list[dict]:
    """native SST-14 를 능가한 후보 전부를 수집.

    선정 기준 (합집합, 단일값 noise 제외):
      A. global_selectivity_leaderboard 에서:
           ddg_median < native_ddg  AND  ddg_n_converged >= 3
         (robust 통계 신호로 native 결합 에너지 능가)
      B. mmgbsa_consensus.json 에서:
           beats_native == True  또는  dg_bind < native_dg
         (MM-GBSA 직교 신호로 native 능가, 단일값이어도 직교 검증으로 의미 있음)

    단일값-only 제외 규칙:
      리더보드에 ddg_n_converged < 3 이고 mmgbsa beats_native 도 아닌 후보는 대상에서 제외.

    Returns:
        [{..., 'source': 'silo_b', 'candidate_id': ...}]  (Silo B 후보 리스트)
    """
    native_ddg, native_dg, native_ddg_source = _load_native_baselines()

    # native_ddg 출처·신뢰도 caveat 출력
    if native_ddg is not None:
        diff_threshold = _NATIVE_BEATS_MIN_DIFF_REU
        print(
            f"[beats-native] native 도킹 기준: {native_ddg:.4f} REU "
            f"(출처={native_ddg_source}). "
            f"'능가' 판정 권장 차이: >= {diff_threshold} REU "
            f"(native sd=12.26 REU, 통계적 유의 threshold). "
            f"ddg_median < native_ddg 조건만으로는 noise 범위 내 포함 가능.",
            file=sys.stderr,
        )

    candidates: list[dict] = []
    seen_seqs: set[str] = set()

    # ── Silo B 리더보드 로드 ──
    if not SILO_B_LB.exists():
        print(f"[WARN] Silo B 리더보드 없음: {SILO_B_LB}", file=sys.stderr)
        return []

    with open(SILO_B_LB, encoding="utf-8") as f:
        lb_raw = json.load(f)
    lb_entries: list[dict] = lb_raw.get("entries", [])
    lb_by_seq: dict[str, dict] = {e["sequence"]: e for e in lb_entries if e.get("sequence")}

    # ── mmgbsa_consensus.json 로드 ──
    mmgbsa_beats_seqs: set[str] = set()
    if MMGBSA_CONSENSUS.exists():
        try:
            with open(MMGBSA_CONSENSUS, encoding="utf-8") as f:
                mc = json.load(f)
            mc_results: dict = mc.get("results", {})
            for seq_key, val in mc_results.items():
                dg_bind: Optional[float] = val.get("dg_bind")
                # beats_native 플래그 또는 dg_bind < native_dg 로 판단
                if val.get("beats_native", False):
                    mmgbsa_beats_seqs.add(seq_key)
                elif (
                    native_dg is not None
                    and dg_bind is not None
                    and dg_bind < native_dg
                ):
                    mmgbsa_beats_seqs.add(seq_key)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[WARN] mmgbsa_consensus.json 로드 실패: {e}", file=sys.stderr)

    print(
        f"[beats-native] native_ddg={native_ddg} (출처={native_ddg_source}), "
        f"native_dg={native_dg} (출처=mmgbsa_consensus)",
        file=sys.stderr,
    )
    print(
        f"[beats-native] MM-GBSA 신호 능가 후보: {len(mmgbsa_beats_seqs)}건",
        file=sys.stderr,
    )

    # ── 기준 A: robust 도킹 신호 (n_conv >= 3) ──
    robust_count = 0
    for e in lb_entries:
        seq: str = e.get("sequence", "")
        if not seq or seq in seen_seqs:
            continue
        ddg_median: Optional[float] = e.get("ddg_median")
        n_conv: int = int(e.get("ddg_n_converged") or 0)
        if (
            native_ddg is not None
            and ddg_median is not None
            and n_conv >= 3
            and ddg_median < native_ddg
        ):
            run_id = e.get("run_id", "silo_b")
            cid = f"silob_{run_id}_{seq}"
            seen_seqs.add(seq)
            margin_from_native = native_ddg - ddg_median  # 양수 = native 대비 개선폭
            is_stat_sig = margin_from_native >= _NATIVE_BEATS_MIN_DIFF_REU
            if not is_stat_sig:
                print(
                    f"  [beats-native] {seq}: ddg={ddg_median:.2f} REU, "
                    f"native 대비 개선={margin_from_native:.2f} REU "
                    f"< 권장 {_NATIVE_BEATS_MIN_DIFF_REU} REU — noise 범위 내 가능성 (포함은 유지, caveat 기록)",
                    file=sys.stderr,
                )
            candidates.append({
                **e,
                "source": "silo_b",
                "candidate_id": cid,
                "ddg": ddg_median,
                "beats_native_signal": "robust_ddg",
                "native_ddg_used": native_ddg,
                "native_ddg_source": native_ddg_source,
                "beats_native_margin_reu": round(margin_from_native, 4),
                "beats_native_stat_sig": is_stat_sig,
            })
            robust_count += 1

    # ── 기준 B: MM-GBSA 직교 신호 ──
    mmgbsa_only_count = 0
    for seq in mmgbsa_beats_seqs:
        if seq in seen_seqs:
            continue  # 이미 기준 A에서 추가됨
        # 리더보드에서 entry 찾기 (candidate_id 생성 위해 필요)
        lb_e: Optional[dict] = lb_by_seq.get(seq)
        if lb_e is None:
            # 리더보드에 없는 서열 — candidate_id 생성 불가, 건너뜀
            print(
                f"  [beats-native] MM-GBSA 능가 후보 '{seq}' 리더보드 미등재 → 건너뜀",
                file=sys.stderr,
            )
            continue
        run_id = lb_e.get("run_id", "silo_b")
        cid = f"silob_{run_id}_{seq}"
        seen_seqs.add(seq)
        seq_ddg = lb_e.get("ddg_median") or lb_e.get("ddg")
        candidates.append({
            **lb_e,
            "source": "silo_b",
            "candidate_id": cid,
            "ddg": seq_ddg,
            "beats_native_signal": "mmgbsa",
            "native_ddg_used": native_ddg,
            "native_ddg_source": native_ddg_source,
            # MM-GBSA 신호이므로 robust_ddg 기반 margin은 None
            "beats_native_margin_reu": (
                round(native_ddg - seq_ddg, 4)
                if native_ddg is not None and seq_ddg is not None
                else None
            ),
            "beats_native_stat_sig": None,  # MM-GBSA 경로: robust_ddg margin 미산출
        })
        mmgbsa_only_count += 1

    print(
        f"[beats-native] 수집 완료: robust_ddg={robust_count}건 + mmgbsa_only={mmgbsa_only_count}건 = {len(candidates)}건",
        file=sys.stderr,
    )
    return candidates


# ── 메인 ──────────────────────────────────────────────────────────────────────

def _collect_candidates(
    top_n_a: int,
    top_n_b: int,
) -> list[dict]:
    """Silo A + Silo B 상위 후보 수집 및 dedup (서열 기준).

    Args:
        top_n_a: Silo A 상위 N건
        top_n_b: Silo B 상위 N건

    Returns:
        [{..., 'source': 'silo_a'|'silo_b', 'candidate_id': ...}]
    """
    candidates: list[dict] = []
    seen_ids: set[str] = set()

    # Silo A
    if SILO_A_LB.exists():
        with open(SILO_A_LB, encoding="utf-8") as f:
            lb_raw = json.load(f)
        entries_a = sorted(
            (e for e in lb_raw.get("entries", []) if e.get("ddg") is not None),
            key=lambda x: x["ddg"],
        )
        for e in entries_a[:top_n_a]:
            cid = e["candidate_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append({**e, "source": "silo_a"})
        print(
            f"[cross-silo] Silo A: {len(entries_a)}건 → 상위 {top_n_a}건",
            file=sys.stderr,
        )
    else:
        print(f"[WARN] Silo A 리더보드 없음: {SILO_A_LB}", file=sys.stderr)

    # Silo B
    if SILO_B_LB.exists() and top_n_b > 0:
        with open(SILO_B_LB, encoding="utf-8") as f:
            lb_b_raw = json.load(f)
        entries_b = lb_b_raw.get("entries", [])  # 이미 ddg 오름차순 정렬
        count_b = 0
        for e in entries_b:
            if count_b >= top_n_b:
                break
            seq = e.get("sequence", "")
            # Silo B는 candidate_id가 없으므로 run_id + 서열로 생성
            run_id = e.get("run_id", "silo_b")
            cid = f"silob_{run_id}_{seq}"
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            candidates.append({
                **e,
                "source": "silo_b",
                "candidate_id": cid,
                "ddg": e.get("ddg_median") or e.get("ddg"),
            })
            count_b += 1
        print(
            f"[cross-silo] Silo B: {len(entries_b)}건 → 상위 {count_b}건",
            file=sys.stderr,
        )
    elif top_n_b > 0:
        print(f"[WARN] Silo B 리더보드 없음: {SILO_B_LB}", file=sys.stderr)

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Silo A + Silo B 상위 후보 교차 검증 (selectivity 포함)",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Silo A 검증 상위 N건 (기본 5)")
    parser.add_argument("--top-n-b", type=int, default=5, help="Silo B 검증 상위 N건 (기본 5)")
    parser.add_argument(
        "--beats-native",
        action="store_true",
        help=(
            "native SST-14 능가 후보 전부를 대상에 추가 (--top-n 과 합집합).\n"
            "선정 기준: robust(n_conv>=3) ddg_median < native_ddg  OR  MM-GBSA beats_native.\n"
            "단일값-only noise 제외: robust 통계 또는 MM-GBSA 직교 신호가 있는 것만."
        ),
    )
    parser.add_argument("--nstruct", type=int, default=5, help="SSTR2 FlexPepDock nstruct (기본 5)")
    parser.add_argument("--ot-nstruct", type=int, default=10, help="off-target FlexPepDock nstruct (기본 10)")
    parser.add_argument("--mmgbsa-device", default="2", help="MM-GBSA GPU device 인덱스 (기본 2)")
    parser.add_argument("--skip-mmgbsa", action="store_true", help="MM-GBSA 단계 건너뜀")
    parser.add_argument("--skip-offtarget", action="store_true", help="off-target 재도킹 건너뜀")
    parser.add_argument("--force-rerun", action="store_true", help="dedup 무시 — 전부 재실행")
    parser.add_argument("--dry-run", action="store_true", help="PDB 탐색 + LC 판정만, 실제 도킹 없음")
    args = parser.parse_args()

    t_total = time.perf_counter()

    # ── 후보 수집 (Silo A + Silo B top-N) ──
    all_candidates = _collect_candidates(top_n_a=args.top_n, top_n_b=args.top_n_b)

    # ── beats-native 모드: native 능가 후보 추가 (top-N 과 합집합) ──
    if args.beats_native:
        bn_candidates = _collect_candidates_beats_native()
        # candidate_id 기준 dedup (top-N 에서 이미 추가된 것 제외)
        existing_cids: set[str] = {c["candidate_id"] for c in all_candidates}
        added = 0
        for c in bn_candidates:
            if c["candidate_id"] not in existing_cids:
                all_candidates.append(c)
                existing_cids.add(c["candidate_id"])
                added += 1
        print(
            f"[beats-native] top-N 에 {added}건 신규 추가 (합집합 총 {len(all_candidates)}건)",
            file=sys.stderr,
        )

    if not all_candidates:
        print("[ERROR] 검증할 후보 없음 — 리더보드 파일 확인 필요", file=sys.stderr)
        sys.exit(1)
    print(
        f"[cross-silo] 총 {len(all_candidates)}건 후보 수집 완료",
        file=sys.stderr,
    )

    # ── dedup (상태파일) ──
    state = load_state()
    validated_ids: dict = state.get("validated", {})

    existing = load_existing_results()
    results_dict: dict = existing.get("results", {})

    to_validate: list[dict] = []
    skipped: list[str] = []
    for e in all_candidates:
        cid = e["candidate_id"]
        if not args.force_rerun and cid in validated_ids:
            skipped.append(cid)
            print(f"  [dedup] 이미 검증됨 → 건너뜀: {cid}", file=sys.stderr)
        else:
            to_validate.append(e)

    print(
        f"  dedup: {len(skipped)}건 건너뜀 / {len(to_validate)}건 신규 검증 예정",
        file=sys.stderr,
    )

    # ── 검증 실행 ──
    new_results: list[dict] = []
    for entry in to_validate:
        src = entry.get("source", "silo_a")
        cid = entry["candidate_id"]
        seq = entry.get("sequence", "")

        if args.dry_run:
            if src == "silo_b":
                pdb_found = find_silo_b_pdb(seq)
            else:
                pdb_found = _find_silo_a_pdb(cid)
            lc, lc_r = _is_low_complexity(seq)
            print(
                f"  [dry-run][{src}] {cid} seq={seq} | "
                f"PDB={'있음' if pdb_found else '없음'} "
                f"| LC={lc}({lc_r})",
                file=sys.stderr,
            )
            continue

        result = validate_candidate(
            entry=entry,
            nstruct=args.nstruct,
            ot_nstruct=args.ot_nstruct,
            mmgbsa_device=args.mmgbsa_device,
            skip_mmgbsa=args.skip_mmgbsa,
            skip_offtarget=args.skip_offtarget,
            source=src,
        )
        new_results.append(result)
        # 상태 업데이트 (실패해도 등록)
        validated_ids[cid] = {
            "verdict": result["verdict"],
            "source": src,
            "validated_at": result["validated_at"],
        }

    if args.dry_run:
        print("[dry-run] 완료 — 실제 도킹 미실행", file=sys.stderr)
        return

    # ── 결과 병합 ──
    for r in new_results:
        results_dict[r["candidate_id"]] = r

    # ── 출력 JSON 생성 ──
    import datetime
    # native baseline 메타 (결과 JSON 명시용)
    _nb_ddg, _nb_dg, _nb_source = _load_native_baselines()
    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "top_n_silo_a_requested": args.top_n,
        "top_n_silo_b_requested": args.top_n_b,
        "n_validated_total": len(results_dict),
        "n_new_this_run": len(new_results),
        "n_skipped_dedup": len(skipped),
        # native baseline 출처·신뢰도 명시
        "native_baseline": {
            "native_ddg": _nb_ddg,
            "native_ddg_source": _nb_source,
            "native_ddg_caveat": (
                f"sd=12.26 REU(nstruct=10). 통계적 유의 권장 차이>={_NATIVE_BEATS_MIN_DIFF_REU} REU."
                if _nb_source == "robust_v2_nstruct10"
                else "LOW 신뢰(n_converged=2, 조건불명) — robust_v2 재생성 권고."
            ),
            "native_dg_mmgbsa": _nb_dg,
            "native_dg_source": "mmgbsa_consensus",
        },
        "results": results_dict,
        "entries": list(results_dict.values()),
    }

    SILO_A_FLOW_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    # 상태 저장
    state["validated"] = validated_ids
    save_state(state)

    elapsed_total = round(time.perf_counter() - t_total, 1)

    # ── 결과 요약 출력 ──
    print("\n" + "=" * 60, file=sys.stderr)
    print("[cross-silo] 검증 완료 요약", file=sys.stderr)
    print(f"  총 소요시간  : {elapsed_total}s", file=sys.stderr)
    print(f"  신규 검증    : {len(new_results)}건", file=sys.stderr)
    print(f"  dedup 건너뜀 : {len(skipped)}건", file=sys.stderr)
    print(f"  누적 결과    : {len(results_dict)}건", file=sys.stderr)
    print(f"  출력 파일    : {OUT_JSON}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    if new_results:
        print("\n[cross-silo] 후보별 verdict:", file=sys.stderr)
        for r in new_results:
            orig = r.get("orig_ddg_single")
            robust = r["siloB_robust_ddg_median"]
            nc = r["n_converged"]
            v = r["verdict"]
            lc = r["low_complexity"]
            seq = r["sequence"]
            dm = r.get("robust_delta_margin")
            src = r.get("source", "?")
            print(
                f"  [{src}] {r['candidate_id']}\n"
                f"    서열: {seq}\n"
                f"    원본 ddG        : {orig}\n"
                f"    Robust ddG      : {robust} (n_conv={nc})\n"
                f"    MM-GBSA dG_bind : {r['mmgbsa_dg']}\n"
                f"    Robust Δmargin  : {dm}\n"
                f"    저복잡도        : {lc} ({r['low_complexity_reason']})\n"
                f"    verdict         : {v.upper()}\n",
                file=sys.stderr,
            )

    # stdout에도 JSON 요약 출력 (스크립트 결과 캡처용)
    summary = {
        "elapsed_total_s": elapsed_total,
        "n_new": len(new_results),
        "n_skipped": len(skipped),
        "verdicts": {r["candidate_id"]: r["verdict"] for r in new_results},
        "selectivity": {
            r["candidate_id"]: r.get("robust_delta_margin")
            for r in new_results
            if r.get("robust_delta_margin") is not None
        },
        "out_json": str(OUT_JSON),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
