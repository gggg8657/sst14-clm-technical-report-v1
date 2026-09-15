"""
boltz_consensus.py
==================
Boltz-2 직교 결합 검증(컨센서스) 모듈.

두 사일로 리더보드(Silo B PyRosetta ddG / Silo A de-novo) 상위 후보를
Boltz-2(GPU2 전용)로 SSTR2 복합체 구조를 독립 예측하고 iPTM/confidence를
PyRosetta ddG와 교차 비교하여 불일치(아티팩트) 의심 후보를 플래그한다.

⚠️ 정직 disclaimer (HEURISTIC-PARTIAL) — 반드시 보고서에 포함:
    - Boltz-2 affinity_kcal_mol: **펩타이드(protein chain)에는 지원 불가**.
      Boltz-2 로컬의 affinity head는 소분자(ligand) 전용이며,
      단백질-단백질/펩타이드 복합체에서는 ValueError가 발생한다.
      (실측 확인: 2026-06-19, "Chain B is not a ligand! Affinity is currently only supported for ligands.")
      따라서 본 모듈의 독립 지표는 **iPTM(구조 geometry 신뢰도)** 와
      **confidence_score**만 사용한다.
    - iPTM: 구조 geometry 신뢰도이며 결합 친화도 순위 proxy 아님.
      SST-14 vs SSTR1~5 실측: iPTM-Ki Spearman ρ ≈ −0.3 (step05c_boltz_cross.py).
    - 아티팩트 탐지는 iPTM 임계값 기반이며 이진 분류가 아닌 "의심 신호"다.
    - 결측 결과(Boltz 실패)는 점수 0 또는 추정치로 대체하지 않는다.

독립성 원칙:
    PyRosetta ddG와 Boltz-2 iPTM은 완전히 다른 물리 모델 기반이므로
    두 방법이 동시에 낮은 값(강한 결합)을 가리키면 신뢰↑,
    한쪽만 극단적이면 아티팩트 의심 신호로 해석한다.

GPU 격리:
    CUDA_VISIBLE_DEVICES=2 (GPU0+1=72B vLLM, GPU3=Silo A 무간섭)

Public API:
    load_leaderboard_top_k(path, k) -> List[LeaderboardEntry]
    run_boltz_single(sequence, sstr2_receptor_seq, work_dir, boltz_env, cuda_device, timeout)
        -> BoltzResult
    compare_ddg_boltz(entry, boltz_result) -> ConsensusRecord
    detect_artifact(ddg, boltz_affinity, ipTM) -> Tuple[bool, str]
    run_consensus_batch(silo_b_path, silo_a_path, work_dir, k, boltz_env, cuda_device)
        -> ConsensusReport
    save_report(report, out_dir) -> Dict[str, str]
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSTR2 수용체 서열 (step05c_boltz_cross.py에서 검증된 canonical isoform 1)
# ---------------------------------------------------------------------------

SSTR2_SEQUENCE: str = (
    "MDMADEPLNGSHTWLSIPFDLNGSVVSTNTSNQTEPYYDLTSNAVLTFIYFVVCIIGLCGNTLVIYVILRYAKMKTITNIYILNL"
    "AIADELFMLGLPFLAMQVALVHWPFGKAICRVVMTVDGINQFTSIFCLTVMSIDRYLAVVHPIKSAKWRRPRTAKMITMAVWGVS"
    "LLVILPIMIYAGLRSNQWGRSSCTINWPGESGAWYTGFIIYTFILGFLVPLTIICLCYLFIIIKVKSSGIRVGSSKRKKSEKKVT"
    "RMVSIVVAVFIFCWLPFYIFNVSSVSMAISPTPALKGMFDFVVVLTYANSCANPILYAFLSDNFKKSFQNVLCLVKVSGTDDGER"
    "SDSKQDKSRLNETTETQRTLLNGDLQTSI"
)

# ---------------------------------------------------------------------------
# 아티팩트 탐지 임계값
# ---------------------------------------------------------------------------
# ddG가 매우 강한데(< -30 kcal/mol) Boltz affinity가 약하면(> -5 kcal/mol) 의심.
# poly-G나 Rosetta 점수 함수 artifact 사례에서 관찰된 패턴.
ARTIFACT_DDG_THRESHOLD: float = -30.0       # 이보다 강해야 "극단적 ddG"
ARTIFACT_AFFINITY_WEAK: float = -5.0        # Boltz affinity가 이보다 약하면 의심
ARTIFACT_IPTM_LOW: float = 0.6              # iPTM이 이보다 낮으면 구조 신뢰도 부족


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class LeaderboardEntry:
    """리더보드 항목 (Silo A/B 공통 최소 필드)."""
    sequence: str
    ddg: Optional[float]            # PyRosetta FlexPepDock ddG (kcal/mol)
    delta_margin: Optional[float]   # Silo B selectivity Δmargin
    source: str                     # "silo_b" | "silo_a"
    candidate_id: Optional[str]     # Silo A candidate_id (없으면 None)
    run_id: Optional[str]           # Silo B run_id (없으면 None)


@dataclass
class BoltzResult:
    """Boltz-2 예측 결과."""
    affinity_kcal_mol: Optional[float]  # Boltz-2 predicted ΔG (kcal/mol) — 없으면 None
    ipTM: Optional[float]               # Boltz-2 iPTM (구조 신뢰도)
    pTM: Optional[float]
    elapsed_s: float                    # 실행 시간 (초)
    error: Optional[str]                # 에러 메시지 (없으면 None)


@dataclass
class ConsensusRecord:
    """단일 후보의 컨센서스 비교 레코드."""
    sequence: str
    source: str
    candidate_id: Optional[str]
    run_id: Optional[str]

    # PyRosetta
    ddg: Optional[float]
    delta_margin: Optional[float]

    # Boltz-2
    boltz_affinity: Optional[float]
    ipTM: Optional[float]
    pTM: Optional[float]
    boltz_elapsed_s: Optional[float]
    boltz_error: Optional[str]

    # 비교 결과
    sign_agree: Optional[bool]          # 부호 일치 (ddg<0 ↔ affinity<0)
    rank_diff: Optional[float]          # 현재 배치 내 순위 차이 (None = 단일 후보)
    artifact_flag: bool                 # True = 아티팩트 의심
    artifact_reason: str                # 의심 사유 (없으면 "")
    consensus_label: str                # "consistent" | "discrepant" | "artifact" | "missing"

    # disclaimer
    disclaimer: str = (
        "Boltz-2 affinity_kcal_mol은 펩타이드(protein chain)에서 지원 불가 — "
        "실측 확인(2026-06-19): affinity head는 소분자(ligand) 전용. "
        "iPTM은 geometry 신뢰도이며 Ki/Kd 친화도 proxy 아님(SST-14 실측 ρ≈−0.3). "
        "컨센서스 라벨은 아티팩트 '의심' 신호이며 확정 근거 아님. "
        "최종 확인은 FEP 또는 실측 radioligand binding assay 필요."
    )


@dataclass
class ConsensusReport:
    """전체 배치 컨센서스 보고서."""
    generated_at: str                       # ISO timestamp
    silo_b_path: str
    silo_a_path: str
    k_top: int
    cuda_device: int
    boltz_env: str
    records: List[ConsensusRecord]

    n_total: int = 0
    n_consistent: int = 0
    n_discrepant: int = 0
    n_artifact: int = 0
    n_missing: int = 0

    disclaimer: str = (
        "이 보고서는 독립 Boltz-2 재채점으로 PyRosetta ddG 결과를 교차 검증한다. "
        "두 surrogate의 불일치는 아티팩트 '의심' 신호이며 확정 근거가 아니다. "
        "최종 판단은 FEP / 실측 radioligand binding assay가 필요하다."
    )

    def compute_summary(self) -> None:
        self.n_total = len(self.records)
        self.n_consistent = sum(1 for r in self.records if r.consensus_label == "consistent")
        self.n_discrepant = sum(1 for r in self.records if r.consensus_label == "discrepant")
        self.n_artifact = sum(1 for r in self.records if r.consensus_label == "artifact")
        self.n_missing = sum(1 for r in self.records if r.consensus_label == "missing")


# ---------------------------------------------------------------------------
# 리더보드 로딩
# ---------------------------------------------------------------------------

def load_leaderboard_top_k(path: str, k: int, source_label: str) -> List[LeaderboardEntry]:
    """리더보드 JSON에서 상위 K개 후보를 읽기 전용으로 로드.

    Silo B 형식: entries[].{sequence, ddg, delta_margin, run_id}
    Silo A 형식: entries[].{candidate_id, sequence, ddg, delta_margin}
    두 형식 모두 지원한다.

    Args:
        path: 리더보드 JSON 파일 경로 (읽기 전용)
        k: 상위 K개
        source_label: "silo_b" | "silo_a"

    Returns:
        LeaderboardEntry 리스트 (최대 K개, ddg 없음=정렬 맨 뒤)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning("[BoltzConsensus] 리더보드 파일 없음: %s", path)
        return []

    try:
        with open(path_obj, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("[BoltzConsensus] 리더보드 로드 실패 %s: %s", path, exc)
        return []

    raw_entries: List[Dict[str, Any]] = data.get("entries", [])
    if not raw_entries:
        logger.warning("[BoltzConsensus] 리더보드 entries 없음: %s", path)
        return []

    entries: List[LeaderboardEntry] = []
    for e in raw_entries:
        seq = e.get("sequence", "")
        if not seq:
            continue
        ddg_raw = e.get("ddg")
        ddg = float(ddg_raw) if ddg_raw is not None else None
        dm_raw = e.get("delta_margin")
        dm = float(dm_raw) if dm_raw is not None else None
        entries.append(LeaderboardEntry(
            sequence=seq,
            ddg=ddg,
            delta_margin=dm,
            source=source_label,
            candidate_id=e.get("candidate_id"),
            run_id=e.get("run_id"),
        ))

    # ddg 오름차순 정렬 (더 낮을수록 강한 결합), ddg=None은 뒤로
    entries.sort(key=lambda e: (e.ddg is None, e.ddg if e.ddg is not None else 0.0))
    result = entries[:k]
    logger.info("[BoltzConsensus] %s에서 상위 %d개 로드 (요청 k=%d)", path, len(result), k)
    return result


# ---------------------------------------------------------------------------
# Boltz-2 단일 예측 (GPU2 전용)
# ---------------------------------------------------------------------------

def _write_boltz_yaml(
    peptide_seq: str,
    receptor_seq: str,
    out_dir: Path,
    name: str = "complex",
) -> Path:
    """Boltz-2 입력 YAML 작성 (single-seq msa empty, 로컬 모드)."""
    yaml_text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: \"{receptor_seq}\"\n"
        '      msa: "empty"\n'
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: \"{peptide_seq}\"\n"
        '      msa: "empty"\n'
    )
    yaml_path = out_dir / f"{name}.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def _parse_boltz_confidence(out_dir: Path, name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Boltz-2 confidence JSON에서 affinity_kcal_mol, ipTM, pTM 파싱.

    반환: (affinity_kcal_mol, ipTM, pTM) — 없으면 None.
    """
    # 알려진 경로 패턴
    known = (
        out_dir
        / f"boltz_results_{name}"
        / "predictions"
        / name
        / f"confidence_{name}_model_0.json"
    )
    candidates: List[Path] = []
    if known.exists():
        candidates = [known]
    else:
        candidates = sorted(
            out_dir.rglob("confidence_*_model_0.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not candidates:
        logger.warning("[BoltzConsensus] confidence JSON 없음: %s", out_dir)
        return None, None, None

    conf_path = candidates[0]
    try:
        raw = json.loads(conf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("[BoltzConsensus] confidence JSON 파싱 실패 %s: %s", conf_path, exc)
        return None, None, None

    # affinity: boltz2는 affinity_kcal_mol 키 사용
    affinity = raw.get("affinity_kcal_mol") or raw.get("binding_affinity") or raw.get("predicted_affinity")
    iptm = raw.get("iptm") or raw.get("ipTM")
    ptm = raw.get("ptm") or raw.get("pTM")

    return (
        float(affinity) if affinity is not None else None,
        float(iptm) if iptm is not None else None,
        float(ptm) if ptm is not None else None,
    )


def run_boltz_single(
    sequence: str,
    sstr2_receptor_seq: str,
    work_dir: Path,
    boltz_env: str = "boltz",
    cuda_device: int = 2,
    timeout: int = 600,
    name: Optional[str] = None,
) -> BoltzResult:
    """단일 펩타이드+SSTR2 페어를 Boltz-2로 예측.

    CUDA_VISIBLE_DEVICES=cuda_device 로 GPU를 격리하여 실행.
    conda run -n {boltz_env}를 통해 boltz CLI 호출.

    Args:
        sequence: 펩타이드 아미노산 서열
        sstr2_receptor_seq: SSTR2 수용체 서열
        work_dir: 작업 디렉토리 (YAML + 출력 저장)
        boltz_env: conda 환경 이름 (기본 "boltz")
        cuda_device: GPU 번호 (기본 2)
        timeout: subprocess timeout 초 (기본 600)
        name: 예측 이름 (None=서열 해시 사용)

    Returns:
        BoltzResult (에러 시 error 필드 설정, 점수 None)
    """
    if not sequence:
        return BoltzResult(
            affinity_kcal_mol=None, ipTM=None, pTM=None,
            elapsed_s=0.0, error="빈 서열"
        )

    # 안전한 파일명 생성
    safe_name = name or f"pep_{abs(hash(sequence)) % 10**8:08d}"
    pair_dir = work_dir / safe_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = _write_boltz_yaml(
        peptide_seq=sequence,
        receptor_seq=sstr2_receptor_seq,
        out_dir=pair_dir,
        name=safe_name,
    )
    boltz_out_dir = pair_dir / "boltz_out"
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "--no-capture-output", "-n", boltz_env,
        "boltz", "predict", str(yaml_path),
        "--out_dir", str(boltz_out_dir),
        "--recycling_steps", "1",
        "--sampling_steps", "50",
        "--diffusion_samples", "1",
        "--output_format", "pdb",
        "--override",
        "--num_workers", "0",
        "--no_kernels",
    ]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cuda_device)}

    t0 = time.time()
    logger.info(
        "[BoltzConsensus] Boltz-2 실행 시작: %s (CUDA_VISIBLE_DEVICES=%s)",
        safe_name, cuda_device
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        elapsed = time.time() - t0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logger.error("[BoltzConsensus] TIMEOUT (%ds) %s", timeout, safe_name)
        return BoltzResult(
            affinity_kcal_mol=None, ipTM=None, pTM=None,
            elapsed_s=elapsed, error=f"timeout after {timeout}s"
        )
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("[BoltzConsensus] subprocess 오류 %s: %s", safe_name, exc)
        return BoltzResult(
            affinity_kcal_mol=None, ipTM=None, pTM=None,
            elapsed_s=elapsed, error=str(exc)
        )

    if proc.returncode != 0:
        stderr_tail = proc.stderr[-500:] if proc.stderr else ""
        logger.warning(
            "[BoltzConsensus] boltz predict 실패 (rc=%d) %s: %s",
            proc.returncode, safe_name, stderr_tail,
        )
        return BoltzResult(
            affinity_kcal_mol=None, ipTM=None, pTM=None,
            elapsed_s=elapsed,
            error=f"boltz exit {proc.returncode}: {stderr_tail[:200]}"
        )

    # confidence 파싱
    affinity, iptm, ptm = _parse_boltz_confidence(boltz_out_dir, safe_name)

    logger.info(
        "[BoltzConsensus] %s 완료 (%.1fs): affinity=%s iPTM=%.3f",
        safe_name,
        elapsed,
        f"{affinity:.3f}" if affinity is not None else "N/A(펩타이드 affinity head 불가)",
        iptm if iptm is not None else float("nan"),
    )

    return BoltzResult(
        affinity_kcal_mol=affinity,
        ipTM=iptm,
        pTM=ptm,
        elapsed_s=elapsed,
        error=None,
    )


# ---------------------------------------------------------------------------
# 비교 로직 및 아티팩트 탐지
# ---------------------------------------------------------------------------

def detect_artifact(
    ddg: Optional[float],
    boltz_affinity: Optional[float],
    ipTM: Optional[float],
    ddg_threshold: float = ARTIFACT_DDG_THRESHOLD,
    affinity_weak: float = ARTIFACT_AFFINITY_WEAK,
    iptm_low: float = ARTIFACT_IPTM_LOW,
) -> Tuple[bool, str]:
    """아티팩트 의심 패턴을 탐지.

    탐지 규칙 (OR 조합):
    1. poly-G/artifact rule: ddg < ddg_threshold (극단 강함) AND
       boltz_affinity > affinity_weak (Boltz는 약함) → surrogate 불일치
    2. 구조 신뢰도 부족: iPTM < iptm_low → 구조 자체 불신뢰

    Args:
        ddg: PyRosetta ddG (kcal/mol), None=미측정
        boltz_affinity: Boltz-2 affinity_kcal_mol, None=예측 실패
        ipTM: Boltz-2 iPTM, None=예측 실패
        ddg_threshold: ddG가 이보다 낮을 때 "극단적 ddG"로 판단 (기본 -30)
        affinity_weak: Boltz affinity가 이보다 클 때 "약함"으로 판단 (기본 -5)
        iptm_low: iPTM이 이보다 낮을 때 "구조 불신뢰" (기본 0.6)

    Returns:
        (is_artifact, reason)
    """
    reasons: List[str] = []

    # Rule 1: PyRosetta 극단 강 + Boltz 약함
    if (
        ddg is not None
        and boltz_affinity is not None
        and ddg < ddg_threshold
        and boltz_affinity > affinity_weak
    ):
        reasons.append(
            f"poly-G/artifact 의심: ddG={ddg:.2f}(극단 강) "
            f"but Boltz_affinity={boltz_affinity:.2f}(약함)"
        )

    # Rule 2: iPTM 낮음 → 구조 신뢰도 부족
    if ipTM is not None and ipTM < iptm_low:
        reasons.append(f"iPTM={ipTM:.3f}<{iptm_low} (구조 신뢰도 부족)")

    is_artifact = len(reasons) > 0
    return is_artifact, "; ".join(reasons)


def compare_ddg_boltz(
    entry: LeaderboardEntry,
    boltz_result: BoltzResult,
) -> ConsensusRecord:
    """LeaderboardEntry + BoltzResult를 ConsensusRecord로 변환.

    비교 전략:
    1. Boltz-2 affinity_kcal_mol: 펩타이드(protein chain)에서 지원 불가.
       affinity가 있으면 부호 일치(ddG<0 ↔ affinity<0)도 판단하지만,
       없어도 iPTM 기반으로 라벨을 결정할 수 있다.
    2. iPTM < ARTIFACT_IPTM_LOW → 구조 신뢰도 부족 → artifact.
    3. Boltz 자체 실패(error) → missing.
    4. iPTM 획득 + 아티팩트 없음 → consistent (ddG/iPTM 두 방법 모두 유효).
    """
    ddg = entry.ddg
    affinity = boltz_result.affinity_kcal_mol
    iptm = boltz_result.ipTM

    # 부호 일치 판단 (affinity 있을 때만)
    sign_agree: Optional[bool] = None
    if ddg is not None and affinity is not None:
        sign_agree = (ddg < 0) == (affinity < 0)

    # 아티팩트 탐지 (iPTM 기반 rule2가 주 탐지 경로)
    is_artifact, artifact_reason = detect_artifact(ddg, affinity, iptm)

    # 결과 라벨 결정
    # - Boltz 예측 자체 실패(process error) → missing
    # - iPTM도 없음(파싱 실패) → missing
    # - 아티팩트 탐지 → artifact
    # - affinity 있고 부호 불일치 → discrepant
    # - 그 외 (iPTM 정상, 아티팩트 없음) → consistent
    if boltz_result.error is not None or iptm is None:
        label = "missing"
    elif is_artifact:
        label = "artifact"
    elif sign_agree is False:
        label = "discrepant"
    else:
        label = "consistent"

    return ConsensusRecord(
        sequence=entry.sequence,
        source=entry.source,
        candidate_id=entry.candidate_id,
        run_id=entry.run_id,
        ddg=ddg,
        delta_margin=entry.delta_margin,
        boltz_affinity=affinity,
        ipTM=iptm,
        pTM=boltz_result.pTM,
        boltz_elapsed_s=boltz_result.elapsed_s,
        boltz_error=boltz_result.error,
        sign_agree=sign_agree,
        rank_diff=None,  # 배치 내 순위 비교는 run_consensus_batch에서 설정
        artifact_flag=is_artifact,
        artifact_reason=artifact_reason,
        consensus_label=label,
    )


# ---------------------------------------------------------------------------
# 배치 실행
# ---------------------------------------------------------------------------

def run_consensus_batch(
    silo_b_path: str,
    silo_a_path: str,
    work_dir: Path,
    k: int = 10,
    boltz_env: str = "boltz",
    cuda_device: int = 2,
    timeout_per_pair: int = 600,
    sstr2_receptor_seq: str = SSTR2_SEQUENCE,
) -> ConsensusReport:
    """두 리더보드 상위 K후보를 Boltz-2로 배치 예측하고 컨센서스 보고서 반환.

    - Silo B + Silo A 각 상위 K개 합산 (중복 서열 제거).
    - CUDA_VISIBLE_DEVICES=cuda_device (기본 2) 격리.
    - 실패 후보는 missing으로 기록 (점수 지어내기 없음).

    Args:
        silo_b_path: global_selectivity_leaderboard.json 경로 (읽기 전용)
        silo_a_path: silo_a_leaderboard.json 경로 (읽기 전용)
        work_dir: Boltz 예측 작업 디렉토리 (runs/boltz_consensus/ 이하)
        k: 각 사일로에서 가져올 상위 후보 수
        boltz_env: conda 환경 이름
        cuda_device: GPU 번호 (기본 2)
        timeout_per_pair: 후보당 Boltz timeout (초)
        sstr2_receptor_seq: SSTR2 수용체 서열

    Returns:
        ConsensusReport
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat()

    # 리더보드 로딩 (읽기 전용)
    silo_b_entries = load_leaderboard_top_k(silo_b_path, k, source_label="silo_b")
    silo_a_entries = load_leaderboard_top_k(silo_a_path, k, source_label="silo_a")

    # 중복 서열 제거 (첫 번째 출처 우선)
    seen_seqs: set = set()
    all_entries: List[LeaderboardEntry] = []
    for e in silo_b_entries + silo_a_entries:
        if e.sequence not in seen_seqs:
            seen_seqs.add(e.sequence)
            all_entries.append(e)

    logger.info(
        "[BoltzConsensus] 배치 시작: %d 후보 (SiloB=%d, SiloA=%d, 중복제거)",
        len(all_entries), len(silo_b_entries), len(silo_a_entries),
    )

    records: List[ConsensusRecord] = []
    for idx, entry in enumerate(all_entries):
        safe_name = f"cand_{idx:03d}_{abs(hash(entry.sequence)) % 10**6:06d}"
        logger.info(
            "[BoltzConsensus] [%d/%d] %s (source=%s seq=%s ddG=%s)",
            idx + 1, len(all_entries), safe_name,
            entry.source, entry.sequence,
            f"{entry.ddg:.2f}" if entry.ddg is not None else "None",
        )

        boltz_result = run_boltz_single(
            sequence=entry.sequence,
            sstr2_receptor_seq=sstr2_receptor_seq,
            work_dir=work_dir,
            boltz_env=boltz_env,
            cuda_device=cuda_device,
            timeout=timeout_per_pair,
            name=safe_name,
        )
        record = compare_ddg_boltz(entry, boltz_result)
        records.append(record)

    # 배치 내 순위 차이 계산 (ddg·affinity 동시 있는 후보)
    _compute_rank_diff(records)

    report = ConsensusReport(
        generated_at=now_iso,
        silo_b_path=str(silo_b_path),
        silo_a_path=str(silo_a_path),
        k_top=k,
        cuda_device=cuda_device,
        boltz_env=boltz_env,
        records=records,
    )
    report.compute_summary()

    logger.info(
        "[BoltzConsensus] 배치 완료: total=%d consistent=%d discrepant=%d artifact=%d missing=%d",
        report.n_total, report.n_consistent, report.n_discrepant,
        report.n_artifact, report.n_missing,
    )
    return report


def _compute_rank_diff(records: List[ConsensusRecord]) -> None:
    """배치 내 ddG 순위 vs Boltz affinity 순위 차이를 각 레코드에 기록.

    순위: 낮을수록 = 더 낮은 값 = 더 강한 결합 = 순위 1위.
    rank_diff = abs(ddg_rank - boltz_rank). 0 = 완전 일치.
    """
    # ddg 유효한 후보 순위
    ddg_valid = [(i, r) for i, r in enumerate(records) if r.ddg is not None]
    ddg_sorted = sorted(ddg_valid, key=lambda x: x[1].ddg)  # type: ignore[arg-type]
    ddg_rank = {orig_i: rank + 1 for rank, (orig_i, _) in enumerate(ddg_sorted)}

    # affinity 유효한 후보 순위
    aff_valid = [(i, r) for i, r in enumerate(records) if r.boltz_affinity is not None]
    aff_sorted = sorted(aff_valid, key=lambda x: x[1].boltz_affinity)  # type: ignore[arg-type]
    aff_rank = {orig_i: rank + 1 for rank, (orig_i, _) in enumerate(aff_sorted)}

    for i, rec in enumerate(records):
        dr = ddg_rank.get(i)
        ar = aff_rank.get(i)
        if dr is not None and ar is not None:
            rec.rank_diff = abs(dr - ar)


# ---------------------------------------------------------------------------
# 보고서 저장
# ---------------------------------------------------------------------------

def save_report(report: ConsensusReport, out_dir: Path) -> Dict[str, str]:
    """ConsensusReport를 JSON + Markdown으로 저장.

    args:
        report: ConsensusReport 인스턴스
        out_dir: 저장 디렉토리 (runs/boltz_consensus/)

    Returns:
        저장 경로 dict {"json": ..., "markdown": ...}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- JSON (Silo별 분리) ----
    silo_b_records = [r for r in report.records if r.source == "silo_b"]
    silo_a_records = [r for r in report.records if r.source == "silo_a"]

    def _record_to_dict(r: ConsensusRecord) -> Dict[str, Any]:
        d = asdict(r)
        return d

    def _save_silo_json(records: List[ConsensusRecord], silo: str) -> str:
        out_path = out_dir / f"consensus_{silo}.json"
        payload = {
            "generated_at": report.generated_at,
            "silo": silo,
            "disclaimer": report.disclaimer,
            "n_total": len(records),
            "candidates": [_record_to_dict(r) for r in records],
        }
        _atomic_write_json(out_path, payload)
        return str(out_path)

    saved: Dict[str, str] = {}
    if silo_b_records:
        saved["silo_b_json"] = _save_silo_json(silo_b_records, "silo_b")
    if silo_a_records:
        saved["silo_a_json"] = _save_silo_json(silo_a_records, "silo_a")

    # ---- 전체 JSON ----
    all_path = out_dir / "consensus_all.json"
    _atomic_write_json(all_path, {
        "generated_at": report.generated_at,
        "silo_b_path": report.silo_b_path,
        "silo_a_path": report.silo_a_path,
        "k_top": report.k_top,
        "cuda_device": report.cuda_device,
        "boltz_env": report.boltz_env,
        "n_total": report.n_total,
        "n_consistent": report.n_consistent,
        "n_discrepant": report.n_discrepant,
        "n_artifact": report.n_artifact,
        "n_missing": report.n_missing,
        "disclaimer": report.disclaimer,
        "candidates": [_record_to_dict(r) for r in report.records],
    })
    saved["all_json"] = str(all_path)

    # ---- Markdown 요약 ----
    md_path = out_dir / "consensus_report.md"
    md_path.write_text(_build_markdown(report), encoding="utf-8")
    saved["markdown"] = str(md_path)

    logger.info("[BoltzConsensus] 보고서 저장 완료: %s", out_dir)
    return saved


def _atomic_write_json(path: Path, data: Any) -> None:
    """원자적 JSON 쓰기 (tmp → rename)."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _build_markdown(report: ConsensusReport) -> str:
    """ConsensusReport → Markdown 요약 문자열."""
    lines: List[str] = []
    lines.append("# Boltz-2 직교 결합 검증 컨센서스 보고서")
    lines.append(f"\n생성 시각: {report.generated_at}")
    lines.append(f"Silo B 소스: `{report.silo_b_path}`")
    lines.append(f"Silo A 소스: `{report.silo_a_path}`")
    lines.append(f"상위 K: {report.k_top} / GPU: CUDA_VISIBLE_DEVICES={report.cuda_device}")
    lines.append("")
    lines.append("## 정직 disclaimer")
    lines.append(f"> {report.disclaimer}")
    lines.append("")
    lines.append("## 요약")
    lines.append(f"| 항목 | 값 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 전체 후보 | {report.n_total} |")
    lines.append(f"| consistent | {report.n_consistent} |")
    lines.append(f"| discrepant | {report.n_discrepant} |")
    lines.append(f"| artifact 의심 | {report.n_artifact} |")
    lines.append(f"| Boltz 실패(missing) | {report.n_missing} |")
    lines.append("")
    lines.append("## 후보별 결과")
    lines.append(
        "| # | 서열 | 출처 | ddG | Boltz affinity | iPTM | 부호일치 | 순위차 | 플래그 |"
    )
    lines.append("|---|------|------|-----|----------------|------|---------|--------|--------|")
    for i, r in enumerate(report.records, 1):
        ddg_str = f"{r.ddg:.2f}" if r.ddg is not None else "N/A"
        aff_str = f"{r.boltz_affinity:.2f}" if r.boltz_affinity is not None else "N/A"
        iptm_str = f"{r.ipTM:.3f}" if r.ipTM is not None else "N/A"
        sa_str = "O" if r.sign_agree else ("X" if r.sign_agree is False else "-")
        rd_str = f"{int(r.rank_diff)}" if r.rank_diff is not None else "-"
        flag = r.consensus_label.upper()
        if r.artifact_flag:
            flag = f"**{flag}**"
        lines.append(
            f"| {i} | `{r.sequence}` | {r.source} | {ddg_str} | {aff_str} | {iptm_str} "
            f"| {sa_str} | {rd_str} | {flag} |"
        )

    lines.append("")
    lines.append("## 아티팩트 의심 후보 상세")
    artifacts = [r for r in report.records if r.artifact_flag]
    if artifacts:
        for r in artifacts:
            lines.append(f"- `{r.sequence}` ({r.source}): {r.artifact_reason}")
    else:
        lines.append("(없음)")

    lines.append("")
    lines.append("## iPTM 한계 (재확인)")
    lines.append(
        "> iPTM은 구조 geometry 신뢰도이며 결합 친화도 순위 proxy **아님**. "
        "SST-14 vs SSTR1~5 실측: Ki-iPTM Spearman ρ≈−0.3 (step05c_boltz_cross.py §disclaimer)."
    )

    return "\n".join(lines) + "\n"
