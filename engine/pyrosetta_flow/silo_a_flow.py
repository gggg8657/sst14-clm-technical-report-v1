"""silo_a_flow.py — Silo A de novo 발굴 엔진 (GPU3 전용, Silo B 완전 분리).

RFdiffusion → ProteinMPNN → ESMFold pLDDT 검증 → FlexPepDock 온타겟 도킹
→ 선택성(오프타겟) 스코어링 → 저비용 surrogate(반감기/ADMET/독성)를 거쳐
de novo 펩타이드 후보를 생성하고 채점한다.

분리 원칙 (MUST):
- 출력: runs/silo_a_flow/ 만 사용. runs/pyrosetta_flow/ 절대 미접근.
- 저장: silo_a_leaderboard.json + experiment_log.jsonl (별도).
- 태그: candidate_class='de_novo', mutation_source='silo_a'.
- scaffold 게이트(FWKT/Cys) 미적용.
- GPU3 전용: 호출 시 CUDA_VISIBLE_DEVICES=3 설정.

환각 0 원칙: 실측 불가 지표는 None/결측 기록. 가짜 수치 금지.
"""
from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

_SILO_A_OUTPUT_DIR = "runs/silo_a_flow"
_SILO_A_LEADERBOARD_NAME = "silo_a_leaderboard.json"
_SILO_A_EXPLOG_NAME = "experiment_log.jsonl"
_PLDDT_GATE = 0.5           # ESMFold pLDDT 임계 (이하 거름)
_LC_MONO_FRAC = 0.6         # 단일 잔기 비율 임계
_LC_MIN_UNIQ  = 3           # 고유 잔기 최소 수
_LC_MIN_ENTROPY = 1.5       # Shannon entropy(bits) 최소


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


# SILO_A_PLDDT_FAILCLOSED: "1"(기본)=fail-closed(측정 실패 → 게이트 탈락),
# "0"=fail-open(하위호환). ESMFold 환경 문제로 상시 실패 시 Silo A 전멸을 방지하려면
# 명시적으로 "0"으로 세팅하되, 리더보드 오염 위험을 인지하고 사용할 것.
_PLDDT_FAILCLOSED: bool = os.environ.get('SILO_A_PLDDT_FAILCLOSED', '1') != '0'

_NSF_ENABLED = os.environ.get('SILO_A_NONSPECIFIC_FILTER', '1') != '0'
_NSF_GRAVY_MAX = _env_float('SILO_A_GRAVY_MAX', 1.5)
_NSF_HYDRO_FRAC_MAX = _env_float('SILO_A_HYDROPHOBIC_FRAC_MAX', 0.65)
_NSF_AMP_CHARGE_MIN = _env_float('SILO_A_AMP_CHARGE_MIN', 3.5)  # H-H 근사 오차 고려(이론+4 ≈ 실측3.9)
_NSF_AMP_MUH_MIN = _env_float('SILO_A_AMP_MUH_MIN', 0.6)
_NSF_CHARGE_MIN = _env_float('SILO_A_NET_CHARGE_MIN', -2.0)  # SVIDKLL류(charge≈-2) 포함
_NSF_CHARGE_MAX = _env_float('SILO_A_NET_CHARGE_MAX', 4.0)
_HYDROPHOBIC_AA = frozenset('ACFILMVWY')
_DEFAULT_CONDA_ENV = "bio-tools"
_PIPELINE_LOCAL_STEPS_ROOT: Optional[Path] = None  # 지연 resolve

# ---------------------------------------------------------------------------
# DiffPepBuilder arm 상수
# ---------------------------------------------------------------------------

_DIFFPEP_REPO: Optional[Path] = None  # 지연 resolve
_DIFFPEP_CONDA_ENV = "diffpepbuilder"  # DiffPepBuilder conda env 이름
_DIFFPEP_HOTSPOT_CHAIN_A = [            # chain A 수용체 기준 핫스팟 (process_receptor 용)
    "A150", "A154", "A197", "A200", "A250", "A294"
]


def _diffpep_repo() -> Path:
    """DiffPepBuilder 로컬 모델 디렉토리 경로."""
    global _DIFFPEP_REPO
    if _DIFFPEP_REPO is None:
        # ROOT/local_models/DiffPepBuilder
        _DIFFPEP_REPO = _repo_root().parents[2] / "local_models" / "DiffPepBuilder"
    return _DIFFPEP_REPO


def _is_low_complexity(seq: str) -> tuple[bool, str]:
    """서열이 저복잡도 아티팩트인지 판정.

    판정 기준 (OR):
      (a) 단일 잔기 비율 > _LC_MONO_FRAC (예: GGGGGGGGGGGGG = 1.0 > 0.6)
      (b) 고유 잔기 수 <= _LC_MIN_UNIQ   (예: GAG 조합 = 2 unique)
      (c) Shannon entropy < _LC_MIN_ENTROPY bits

    반환: (is_artifact: bool, reason: str)
    """
    if not seq:
        return True, "빈 서열"
    n = len(seq)
    cnt = Counter(seq.upper())
    # (a) 단일 잔기 비율
    max_frac = max(cnt.values()) / n
    if max_frac > _LC_MONO_FRAC:
        aa = max(cnt, key=cnt.get)
        return True, f"단일잔기과다({aa}={max_frac:.0%}>{_LC_MONO_FRAC:.0%})"
    # (b) 고유 잔기 수
    n_unique = len(cnt)
    if n_unique <= _LC_MIN_UNIQ:
        return True, f"고유잔기부족({n_unique}<={_LC_MIN_UNIQ})"
    # (c) Shannon entropy
    entropy = -sum((v / n) * math.log2(v / n) for v in cnt.values())
    if entropy < _LC_MIN_ENTROPY:
        return True, f"shannon_entropy낮음({entropy:.2f}<{_LC_MIN_ENTROPY})"
    return False, ""


def _is_nonspecific_artifact(seq: str) -> tuple[bool, str]:
    """서열 기반 비특이 아티팩트 필터. 계산 실패 시 fail-open."""
    if not _NSF_ENABLED or not seq:
        return False, ""

    try:
        seq_u = seq.upper()
        n = len(seq_u)

        # repo root를 sys.path에 추가하여 'AG_src.pipeline.pharma_properties' 접근
        _repo = str(Path(__file__).resolve().parents[1])
        if _repo not in sys.path:
            sys.path.insert(0, _repo)
        from AG_src.pipeline.pharma_properties import PharmaProperties

        pp = PharmaProperties()
        gravy = pp.calculate_gravy(seq_u)
        if gravy > _NSF_GRAVY_MAX:
            return True, f"GRAVY과다({gravy:.2f}>{_NSF_GRAVY_MAX:.2f})"

        hydro_frac = sum(1 for aa in seq_u if aa in _HYDROPHOBIC_AA) / n
        if hydro_frac >= _NSF_HYDRO_FRAC_MAX:
            return True, f"소수성잔기과다({hydro_frac:.0%}>={_NSF_HYDRO_FRAC_MAX:.0%})"

        net_charge = pp.calculate_net_charge(seq_u, ph=7.4)
        mu_h = pp.calculate_hydrophobic_moment(seq_u, angle=100.0)
        if net_charge >= _NSF_AMP_CHARGE_MIN and mu_h > _NSF_AMP_MUH_MIN:
            return True, (
                f"AMP패턴(charge={net_charge:.1f}>={_NSF_AMP_CHARGE_MIN:.1f},"
                f"μH={mu_h:.2f}>{_NSF_AMP_MUH_MIN:.2f})"
            )

        if net_charge < _NSF_CHARGE_MIN:
            return True, f"과음전하(charge={net_charge:.1f}<{_NSF_CHARGE_MIN:.1f})"
        if net_charge > _NSF_CHARGE_MAX:
            return True, f"과양전하(charge={net_charge:.1f}>{_NSF_CHARGE_MAX:.1f})"
    except Exception:
        return False, ""

    return False, ""


def _repo_root() -> Path:
    """pyrosetta_flow/silo_a_flow.py 기준 repo root."""
    return Path(__file__).resolve().parent.parent


def _pipeline_local_root() -> Path:
    """pipeline_local 패키지 루트 (repo_root 상위의 SST14-M_scr/pipeline_local)."""
    # repo_root: AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri
    # pipeline_local: SST14-M_scr/pipeline_local
    return _repo_root().parents[2] / "pipeline_local"


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SiloAConfig:
    """Silo A 연속 발굴 설정."""
    receptor_pdb: str                          # SSTR2 수용체 PDB (chain B 기준)
    run_id: str = "silo_a"
    n_backbone: int = 2                        # 백본 생성 수
    k_seq_per_backbone: int = 2               # 백본당 서열 수
    diffusion_steps: int = 50
    contigs: str = "B1-369/0 12-16"           # RFdiffusion contig (수용체 chain B + 바인더 길이)
    hotspot_res: List[str] = field(default_factory=lambda: [
        "B150", "B154", "B197", "B200", "B250", "B294"
    ])
    plddt_threshold: float = _PLDDT_GATE
    conda_env: str = _DEFAULT_CONDA_ENV        # FlexPepDock 용 conda env
    rfdiffusion_env: str = "rfdiffusion"
    esmfold_env: str = "esmfold"
    proteinmpnn_env: str = "proteinmpnn"
    output_base_dir: str = _SILO_A_OUTPUT_DIR  # runs/silo_a_flow
    device: str = "cuda:0"                     # CUDA_VISIBLE_DEVICES=3 외부 설정 → 내부 cuda:0
    script_timeout: int = 900                  # FlexPepDock timeout(s)
    selectivity_enabled: bool = True           # off-target 도킹 수행 여부
    selectivity_timeout: int = 900
    max_selectivity_per_epoch: int = 2         # epoch 당 선택성 도킹 최대 수 (비용 제어)
    # DiffPepBuilder arm 설정 (GPU2 전용)
    diffpep_arm_enabled: bool = False           # DiffPepBuilder arm 활성화 여부
    diffpep_cuda_device: str = "2"             # DiffPepBuilder arm GPU (CUDA_VISIBLE_DEVICES 값)
    diffpep_n_samples: int = 3                 # epoch당 DiffPepBuilder 생성 서열 수
    diffpep_min_length: int = 10               # 생성 펩타이드 최소 길이
    diffpep_max_length: int = 14               # 생성 펩타이드 최대 길이
    diffpep_num_t: int = 50                    # diffusion 스텝 수
    diffpep_hotspot_res: List[str] = field(default_factory=lambda: _DIFFPEP_HOTSPOT_CHAIN_A.copy())
    diffpep_timeout: int = 600                 # DiffPepBuilder 서브프로세스 타임아웃(s)


@dataclass
class SiloACandidateResult:
    """Silo A 단일 후보 결과."""
    candidate_id: str
    sequence: str
    backbone_idx: int
    seq_idx: int
    plddt: Optional[float]                     # ESMFold pLDDT (None=측정 실패)
    plddt_pass: bool                           # pLDDT 게이트 통과 여부
    ddg: Optional[float]                       # 온타겟 ΔG (FlexPepDock, None=실패)
    clash_score: Optional[float]
    selectivity_margin: Optional[float]        # SSTR2 vs min(off-target) margin
    delta_margin: Optional[float]              # Δmargin vs native baseline
    hc50: Optional[float]                      # pepADMET HC50 (None=결측)
    half_life_h: Optional[float]               # 반감기 surrogate
    admet_score: Optional[float]               # ADMET 합리성 surrogate
    fail_reason: str = ""
    candidate_class: str = "de_novo"
    mutation_source: str = "silo_a"
    extra_scores: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidate_class"] = self.candidate_class
        d["mutation_source"] = self.mutation_source
        return d


# ---------------------------------------------------------------------------
# 수용체 PDB chain B 변환 헬퍼
# ---------------------------------------------------------------------------

def _ensure_chain_b_receptor(receptor_pdb: str, work_dir: Path) -> str:
    """수용체 PDB를 읽어 모든 ATOM/HETATM 레코드를 chain B로 재레이블.

    RFdiffusion이 바인더를 chain A로 생성하려면 수용체가 chain B여야 한다.
    이미 chain B인 파일은 그대로 반환(복사본 생성 없이 원본 경로).
    """
    lines = Path(receptor_pdb).read_text(encoding="utf-8").splitlines()
    atom_lines = [l for l in lines if l.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        raise ValueError(f"수용체 PDB에 ATOM/HETATM 레코드 없음: {receptor_pdb}")

    # 현재 chain 확인
    chains = {l[21] for l in atom_lines if len(l) >= 22}
    if chains == {"B"}:
        return receptor_pdb  # 이미 chain B

    # chain B 재레이블
    new_lines = []
    for l in lines:
        if l.startswith(("ATOM", "HETATM")) and len(l) >= 22:
            l = l[:21] + "B" + l[22:]
        new_lines.append(l)
    out_path = work_dir / "receptor_chainB.pdb"
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    logger.info("[SiloA] 수용체 chain %s → B 재레이블: %s", chains, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# ESMFold pLDDT 검증
# ---------------------------------------------------------------------------

def _esmfold_plddt(sequence: str, esmfold_env: str, work_dir: Path, timeout: int = 300) -> Optional[float]:
    """ESMFold로 서열의 mean_plddt를 측정. 실패 시 None (fail-closed)."""
    pipeline_local = _pipeline_local_root()
    wrapper = pipeline_local / "wrapper_scripts" / "run_esmfold.py"
    if not wrapper.exists():
        logger.warning("[SiloA] ESMFold wrapper 없음: %s — pLDDT 측정 생략", wrapper)
        return None

    # conda Python 해석기 찾기
    python_exec = _resolve_conda_python(esmfold_env)
    if not python_exec:
        logger.warning("[SiloA] esmfold conda env Python 미발견 — pLDDT 측정 생략")
        return None

    payload = {"sequence": sequence}
    payload_file = work_dir / f"esmfold_payload_{sequence[:8]}.json"
    payload_file.write_text(json.dumps(payload), encoding="utf-8")

    cmd = [python_exec, str(wrapper), "--input-json", str(payload_file), "--output-dir", str(work_dir)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "3")},
        )
    except subprocess.TimeoutExpired:
        logger.warning("[SiloA] ESMFold 타임아웃(seq=%s...)", sequence[:8])
        return None
    if proc.returncode != 0:
        logger.warning("[SiloA] ESMFold 실패(rc=%d): %s", proc.returncode, (proc.stderr or "")[:200])
        return None
    lines = (proc.stdout or "").strip().splitlines()
    for line in reversed(lines):
        try:
            data = json.loads(line)
            plddt = data.get("mean_plddt")
            if plddt is not None:
                return float(plddt)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# conda Python 경로 해석
# ---------------------------------------------------------------------------

def _resolve_conda_python(conda_env: str) -> str:
    """conda 환경의 Python 실행파일 경로. 없으면 빈 문자열."""
    if not conda_env:
        return sys.executable
    for base in [
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]:
        env_python = base / "envs" / conda_env / "bin" / "python"
        if env_python.exists():
            return str(env_python)
    return ""


# ---------------------------------------------------------------------------
# DiffPepBuilder arm — GPU2 전용 de novo 생성
# ---------------------------------------------------------------------------

def _prepare_diffpep_receptor(
    receptor_pdb: str,
    work_dir: Path,
    hotspot_res: List[str],
    diffpep_conda_env: str,
    timeout: int = 120,
) -> Optional[Path]:
    """DiffPepBuilder process_receptor.py로 수용체 전처리.

    반환: metadata_test.csv 경로 또는 실패 시 None.
    수용체는 chain A 기준 (curated SSTR2_receptor.pdb 형식).
    """
    diffpep = _diffpep_repo()
    process_script = diffpep / "experiments" / "process_receptor.py"
    if not process_script.exists():
        logger.error("[SiloA-DiffPep] process_receptor.py 없음: %s", process_script)
        return None

    # 수용체 이름에 underscore 금지 (DiffPepBuilder 제약)
    recv_name = Path(receptor_pdb).stem.replace("_", "")
    recv_copy = work_dir / f"{recv_name}.pdb"
    # PDB 복사 (underscore 제거한 이름)
    import shutil
    shutil.copy2(receptor_pdb, recv_copy)

    # hotspot 문자열 (예: A150, A154, ...) → DiffPepBuilder JSON 형식
    hotspot_str = ", ".join(hotspot_res)
    cases_json = {recv_name: {"hotspots": hotspot_str}}
    cases_path = work_dir / "receptor_cases.json"
    cases_path.write_text(json.dumps(cases_json, indent=2), encoding="utf-8")

    processed_dir = work_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # conda Python 찾기
    python_exec = _resolve_conda_python(diffpep_conda_env)
    if not python_exec:
        logger.error("[SiloA-DiffPep] conda env '%s' Python 없음", diffpep_conda_env)
        return None

    cmd = [
        python_exec,
        str(process_script),
        "--pdb_dir", str(work_dir),
        "--write_dir", str(processed_dir),
        "--receptor_info_path", str(cases_path),
        "--max_batch_size", "8",
    ]
    env = {
        **os.environ,
        "BASE_PATH": str(diffpep),
        "CUDA_VISIBLE_DEVICES": "2",  # 전처리는 CPU 위주지만 ESM 임베딩이 GPU 사용
    }
    try:
        proc = subprocess.run(
            cmd, cwd=str(diffpep), capture_output=True, text=True,
            check=False, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("[SiloA-DiffPep] process_receptor 타임아웃(%ds)", timeout)
        return None

    if proc.returncode != 0:
        logger.error(
            "[SiloA-DiffPep] process_receptor 실패(rc=%d): %s",
            proc.returncode, (proc.stderr or "")[:300],
        )
        return None

    csv_path = processed_dir / "metadata_test.csv"
    if not csv_path.exists():
        logger.error("[SiloA-DiffPep] metadata_test.csv 미생성: %s", processed_dir)
        return None

    logger.info("[SiloA-DiffPep] 수용체 전처리 완료: %s", csv_path)
    return csv_path


def _run_diffpep_inference(
    metadata_csv: Path,
    work_dir: Path,
    config: "SiloAConfig",
) -> List[Dict[str, Any]]:
    """DiffPepBuilder run_inference.py를 subprocess로 실행하고 생성된 서열 목록 반환.

    반환: [{"sequence": str, "pdb_path": str, "length": int}]
    실패 시 빈 리스트 (정직 실패).
    """
    diffpep = _diffpep_repo()
    run_inference = diffpep / "experiments" / "run_inference.py"
    if not run_inference.exists():
        logger.error("[SiloA-DiffPep] run_inference.py 없음: %s", run_inference)
        return []

    # mock stub 스크립트 (pyrosetta/openmm 없이 실행)
    stub_script = _build_diffpep_stub(work_dir)
    inference_out = work_dir / "inference_out"
    inference_out.mkdir(parents=True, exist_ok=True)

    python_exec = _resolve_conda_python(_DIFFPEP_CONDA_ENV)
    if not python_exec:
        logger.error("[SiloA-DiffPep] diffpepbuilder conda env Python 없음")
        return []

    ckpt = diffpep / "experiments" / "checkpoints" / "diffpepdock_v1.pth"
    if not ckpt.exists():
        logger.error("[SiloA-DiffPep] 체크포인트 없음: %s", ckpt)
        return []

    cmd = [
        python_exec,
        str(stub_script),
        f"data.val_csv_path={metadata_csv}",
        "experiment.use_ddp=False",
        "experiment.num_gpus=1",
        "experiment.eval_batch_size=1",
        f"experiment.eval_ckpt_path={ckpt}",
        f"experiment.eval_dir={inference_out}",
        f"inference.denoising.num_t={config.diffpep_num_t}",
        f"inference.sampling.samples_per_length={config.diffpep_n_samples}",
        f"inference.sampling.min_length={config.diffpep_min_length}",
        f"inference.sampling.max_length={config.diffpep_max_length}",
        "inference.ss_bond.build_ss_bond=False",
        "inference.ss_bond.save_entropy=True",  # entropy 없으면 버그(upstream 코드 문제)
    ]
    env = {
        **os.environ,
        "BASE_PATH": str(diffpep),
        "CUDA_VISIBLE_DEVICES": config.diffpep_cuda_device,
    }

    try:
        proc = subprocess.run(
            cmd, cwd=str(diffpep), capture_output=True, text=True,
            check=False, timeout=config.diffpep_timeout, env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error("[SiloA-DiffPep] inference 타임아웃(%ds)", config.diffpep_timeout)
        return []

    if proc.returncode != 0:
        logger.error(
            "[SiloA-DiffPep] inference 실패(rc=%d): %s",
            proc.returncode, (proc.stderr or "")[-400:],
        )
        return []

    # 생성된 PDB 수집 및 서열 파싱
    results = []
    for pdb_path in sorted(inference_out.rglob("*.pdb")):
        seq = _extract_sequence_from_pdb(pdb_path, chain="A")
        if seq:
            results.append({
                "sequence": seq,
                "pdb_path": str(pdb_path),
                "length": len(seq),
            })
            logger.info("[SiloA-DiffPep] 생성 서열: %s (len=%d)", seq, len(seq))

    logger.info("[SiloA-DiffPep] 총 %d개 서열 생성", len(results))
    return results


def _build_diffpep_stub(work_dir: Path) -> Path:
    """pyrosetta/openmm 없이 DiffPepBuilder inference를 실행하는 stub 스크립트 생성.

    GPUtil은 cuda:0 고정 반환으로 mock(CUDA_VISIBLE_DEVICES=2 환경에서 내부 cuda:0 사용).
    """
    stub_path = work_dir / "_diffpep_inference_stub.py"
    if stub_path.exists():
        return stub_path  # 재사용

    diffpep_root = str(_diffpep_repo())
    stub_content = f'''"""DiffPepBuilder inference stub (pyrosetta/openmm 없이 실행)."""
import sys, os, types

diffpep_root = {repr(diffpep_root)}
if diffpep_root not in sys.path:
    sys.path.insert(0, diffpep_root)

def _mk(name, attrs=None):
    m = types.ModuleType(name)
    if attrs:
        for k, v in attrs.items():
            setattr(m, k, v)
    return m

class _GPU:
    id = 0
    memoryUsed = 0
    memoryTotal = 10000

gputil_mock = _mk("GPUtil")
gputil_mock.getAvailable = lambda **kw: ["0"]
gputil_mock.getGPUs = lambda: [_GPU()]
sys.modules["GPUtil"] = gputil_mock

class _PDBFixer:
    def __init__(self, *a, **kw): pass
sys.modules["pdbfixer"] = _mk("pdbfixer", {{"PDBFixer": _PDBFixer}})

class _AmberRelaxation:
    def __init__(self, *a, **kw): pass
    def process(self, *a, **kw): return None, None

sys.modules["analysis.amber_minimize"] = _mk("analysis.amber_minimize", {{"AmberRelaxation": _AmberRelaxation}})
sys.modules["analysis.cleanup"] = _mk("analysis.cleanup")

class _Postprocess:
    def __init__(self, *a, **kw): pass
    def process(self, *a, **kw): return {{}}

sys.modules["analysis.postprocess"] = _mk("analysis.postprocess", {{"Postprocess": _Postprocess}})
sys.modules["analysis.postprocess_utils"] = _mk("analysis.postprocess_utils", {{
    "summarize_statistics": lambda *a, **kw: {{}}
}})

pyr = _mk("pyrosetta")
pyr.init = lambda *a, **kw: None
pyr.pose_from_pdb = lambda *a, **kw: None
pyr.get_fa_scorefxn = lambda *a, **kw: None
sys.modules["pyrosetta"] = pyr

import runpy
os.environ.setdefault("BASE_PATH", diffpep_root)
sys.argv = [diffpep_root + "/experiments/run_inference.py"] + sys.argv[1:]
runpy.run_path(diffpep_root + "/experiments/run_inference.py", run_name="__main__")
'''
    stub_path.write_text(stub_content, encoding="utf-8")
    return stub_path


def _extract_sequence_from_pdb(pdb_path: Path, chain: str = "A") -> str:
    """PDB 파일의 지정 chain ATOM 레코드에서 1문자 아미노산 서열 추출."""
    _AA3TO1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seq = []
    prev_res = -99999
    try:
        for line in pdb_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("ATOM"):
                continue
            if len(line) < 26:
                continue
            ch = line[21]
            if ch != chain:
                continue
            res_num = int(line[22:26].strip())
            if res_num == prev_res:
                continue
            aa3 = line[17:20].strip()
            seq.append(_AA3TO1.get(aa3, "X"))
            prev_res = res_num
    except Exception as exc:
        logger.warning("[SiloA-DiffPep] PDB 서열 추출 실패(%s): %s", pdb_path.name, exc)
        return ""
    return "".join(seq)


def _generate_diffpep_sequences(
    receptor_pdb: str,
    config: "SiloAConfig",
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """DiffPepBuilder arm: SSTR2 수용체 대상 de novo 펩타이드 서열 생성.

    분리 원칙:
    - CUDA_VISIBLE_DEVICES={config.diffpep_cuda_device} (GPU2 전용)
    - mutation_source='silo_a_diffpep' 태그 부여
    - RFdiffusion arm(GPU3)과 완전 독립 실행

    반환: [{"sequence": str, "backbone_idx": 0, "seq_idx": N, "backbone_pdb": None,
             "mutation_source": "silo_a_diffpep"}]
    실패 시 빈 리스트 (정직 실패, 환각 0).
    """
    diffpep = _diffpep_repo()
    if not diffpep.exists():
        logger.error("[SiloA-DiffPep] DiffPepBuilder 저장소 없음: %s", diffpep)
        return []

    ckpt = diffpep / "experiments" / "checkpoints" / "diffpepdock_v1.pth"
    if not ckpt.exists():
        logger.error("[SiloA-DiffPep] 체크포인트 없음(다운로드 필요): %s", ckpt)
        return []

    logger.info(
        "[SiloA-DiffPep] DiffPepBuilder arm 시작 (GPU%s, len=%d~%d, samples=%d)",
        config.diffpep_cuda_device,
        config.diffpep_min_length, config.diffpep_max_length,
        config.diffpep_n_samples,
    )

    # 1) 수용체 전처리
    metadata_csv = _prepare_diffpep_receptor(
        receptor_pdb=receptor_pdb,
        work_dir=work_dir,
        hotspot_res=config.diffpep_hotspot_res,
        diffpep_conda_env=_DIFFPEP_CONDA_ENV,
    )
    if metadata_csv is None:
        logger.error("[SiloA-DiffPep] 수용체 전처리 실패")
        return []

    # 2) de novo 추론
    generated = _run_diffpep_inference(metadata_csv, work_dir, config)
    if not generated:
        logger.error("[SiloA-DiffPep] de novo 생성 결과 없음")
        return []

    # 3) silo_a_flow 형식으로 변환 (mutation_source 태그 포함)
    results = []
    for i, item in enumerate(generated):
        results.append({
            "sequence": item["sequence"],
            "backbone_idx": 0,   # DiffPepBuilder는 백본 구조를 pdb_path에 직접 제공
            "seq_idx": i,
            "seq_id": f"diffpep_len{item['length']}_s{i}",
            "backbone_pdb": item["pdb_path"],  # 생성된 복합체 PDB (chain A=펩타이드)
            "mutation_source": "silo_a_diffpep",
        })
    return results


# ---------------------------------------------------------------------------
# de novo 백본 + 서열 생성
# ---------------------------------------------------------------------------

def _generate_de_novo_sequences(
    receptor_pdb_chainB: str,
    config: SiloAConfig,
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """RFdiffusion + ProteinMPNN으로 de novo 서열 목록 생성.

    반환: [{"sequence": ..., "backbone_idx": ..., "seq_idx": ..., "backbone_pdb": ...}]
    실패 시 빈 리스트 (정직 실패 — 환각 없음).
    """
    # pipeline_local 경로를 sys.path에 추가 (임시)
    pipeline_local = _pipeline_local_root()
    if str(pipeline_local) not in sys.path:
        sys.path.insert(0, str(pipeline_local.parent))

    try:
        from pipeline_local.steps.step02_backbone import generate_backbones
        from pipeline_local.steps.step03_sequence import design_sequences
    except ImportError as exc:
        logger.error("[SiloA] step02/step03 import 실패: %s", exc)
        return []

    step02_config: Dict[str, Any] = {
        "run_id": config.run_id,
        "output_base_dir": str(work_dir),
        "contigs": config.contigs,
        "hotspot_res": config.hotspot_res,
        "iteration": {
            "n_backbone": config.n_backbone,
            "diffusion_steps": config.diffusion_steps,
        },
        "device": config.device,
    }

    # CUDA_VISIBLE_DEVICES=3 강제 (환경변수로 전달)
    cuda_env = dict(os.environ)
    cuda_env["CUDA_VISIBLE_DEVICES"] = "3"

    logger.info("[SiloA] step02: 백본 %d개 생성 시작 (contigs='%s')", config.n_backbone, config.contigs)
    try:
        step02_out = generate_backbones(receptor_pdb_chainB, {}, step02_config)
    except Exception as exc:
        logger.error("[SiloA] 백본 생성 실패: %s", exc)
        return []

    if not step02_out.backbone_pdbs:
        logger.error("[SiloA] 백본 생성 0개 — 서열 생성 불가")
        return []

    logger.info("[SiloA] step02 완료: %d개 백본", step02_out.n_generated)

    step03_config: Dict[str, Any] = {
        "run_id": config.run_id,
        "output_base_dir": str(work_dir),
        "iteration": {
            "k_seq_per_backbone": config.k_seq_per_backbone,
        },
        "sequence_constraints": {"enabled": False},  # de novo: 제약 없음
    }

    logger.info("[SiloA] step03: 서열 설계 시작 (k=%d/backbone)", config.k_seq_per_backbone)
    try:
        step03_out = design_sequences(step02_out.backbone_pdbs, step03_config)
    except Exception as exc:
        logger.error("[SiloA] 서열 설계 실패: %s", exc)
        return []

    results = []
    for entry in step03_out.sequences:
        # 백본 PDB 경로 연결
        bb_path = step02_out.backbone_pdbs[entry.backbone_idx] if entry.backbone_idx < len(step02_out.backbone_pdbs) else None
        results.append({
            "sequence": entry.sequence,
            "backbone_idx": entry.backbone_idx,
            "seq_idx": entry.seq_idx,
            "seq_id": entry.seq_id,
            "backbone_pdb": bb_path,
        })

    logger.info("[SiloA] de novo 서열 %d개 생성 완료", len(results))
    return results


# ---------------------------------------------------------------------------
# FlexPepDock 온타겟 도킹
# ---------------------------------------------------------------------------

def _dock_de_novo(
    sequence: str,
    backbone_pdb: str,
    candidate_id: str,
    iter_dir: Path,
    config: SiloAConfig,
    repo_root: Path,
) -> Dict[str, Any]:
    """de novo 서열을 backbone PDB 복합체에 FlexPepDock refine으로 도킹.

    de novo 백본은 chain A(바인더) + chain B(수용체) 복합체 포맷이어야 한다.
    RFdiffusion chain 규약(chain A=바인더) 준수 확인 후 도킹 수행.

    반환: {"ddg": float, "total_score": float, "clash_score": float} 또는 fail dict.
    """
    flexpep_script = repo_root / "AG_src" / "scripts" / "flexpep_dock.py"
    if not flexpep_script.exists():
        return {"ddg": None, "fail_reason": f"flexpep_dock.py 없음: {flexpep_script}"}
    if not backbone_pdb or not Path(backbone_pdb).exists():
        return {"ddg": None, "fail_reason": f"backbone PDB 없음: {backbone_pdb}"}

    out_pdb = iter_dir / f"{candidate_id}.pdb"
    python_exec = _resolve_conda_python(config.conda_env)
    if not python_exec:
        cmd_prefix = ["conda", "run", "-n", config.conda_env, "python"]
    else:
        cmd_prefix = [python_exec]

    cmd = cmd_prefix + [
        str(flexpep_script),
        "--input", backbone_pdb,          # de novo 복합체(chain A+B)
        "--output", str(out_pdb),
        "--protocol", "flexpep_refine",
        "--reference-complex", backbone_pdb,
        "--target-sequence", sequence,
        "--peptide-chain", "1",            # chain A = 바인더 (PDB chain index 1)
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=str(repo_root), capture_output=True, text=True,
            check=False, timeout=config.script_timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ddg": None, "fail_reason": f"FlexPepDock 타임아웃({config.script_timeout}s)"}
    if proc.returncode != 0:
        stderr = (proc.stderr or "")[:300]
        return {"ddg": None, "fail_reason": f"FlexPepDock 실패(rc={proc.returncode}): {stderr}"}

    lines = (proc.stdout or "").strip().splitlines()
    for line in reversed(lines):
        try:
            data = json.loads(line)
            return {
                "ddg": data.get("ddg"),
                "total_score": data.get("total_score"),
                "clash_score": data.get("clash_score"),
                "pdb_path": str(out_pdb) if out_pdb.exists() else None,
            }
        except Exception:
            continue
    return {"ddg": None, "fail_reason": "FlexPepDock stdout JSON 파싱 실패"}


# ---------------------------------------------------------------------------
# 저비용 surrogate (반감기/ADMET)
# ---------------------------------------------------------------------------

def _cheap_scores(sequence: str) -> Dict[str, Any]:
    """반감기/ADMET surrogate (cheap_objectives 재사용). 실패 시 결측 반환."""
    try:
        from pyrosetta_flow.multiobjective import cheap_objectives
        return cheap_objectives(sequence, reference_seq=sequence)  # de novo: 자기 자신 기준
    except Exception as exc:
        logger.warning("[SiloA] cheap_objectives 실패(%s) — 결측", exc)
        return {}


def _hemolysis_flag(sequence: str, aliphatic_index: Optional[float] = None) -> Dict[str, Any]:
    """Lane C 물성 기반 용혈 휴리스틱 (multiobjective.hemolysis_aliphatic_flag 래퍼).

    실패 시 applies=False, risk="NA" graceful 반환 (fail-open).
    """
    try:
        from pyrosetta_flow.multiobjective import hemolysis_aliphatic_flag
        return hemolysis_aliphatic_flag(sequence, aliphatic_index=aliphatic_index)
    except Exception as exc:
        logger.warning("[SiloA] hemolysis_aliphatic_flag 실패(%s) — skip", exc)
        return {"aliphatic_index": aliphatic_index, "hemolysis_risk": "NA", "applies": False, "reason": f"계산오류:{exc}"}


def _toxicity_scores(sequences: List[str]) -> Dict[str, Dict[str, Any]]:
    """pepADMET 독성 배치 추론. 실패 시 빈 dict."""
    try:
        from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
        return predict_toxicity_for_sequences(sequences)
    except Exception as exc:
        logger.warning("[SiloA] predict_toxicity_for_sequences 실패(%s) — 결측", exc)
        return {}


# ---------------------------------------------------------------------------
# 선택성 측정 (off-target 도킹)
# ---------------------------------------------------------------------------

def _selectivity_score(
    candidate_pdb: str,
    ddg: float,
    config: SiloAConfig,
    repo_root: Path,
) -> Dict[str, Any]:
    """screen_selectivity 재사용. de novo 복합체 포맷(chain A+B)으로 호환 여부 불확실 — 시도 후 정직 기록."""
    try:
        from pyrosetta_flow.multiobjective import screen_selectivity
        result = screen_selectivity(
            sstr2_complex_pdb=candidate_pdb,
            on_target_ddg=ddg,
            repo_root=str(repo_root),
            conda_env=config.conda_env,
            timeout=config.selectivity_timeout,
        )
        return result
    except Exception as exc:
        logger.warning("[SiloA] screen_selectivity 실패(%s) — 결측", exc)
        return {"selectivity_margin": None, "delta_margin": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# 메인 에포크 함수
# ---------------------------------------------------------------------------

def generate_and_score_silo_a(
    config: SiloAConfig,
) -> List[SiloACandidateResult]:
    """Silo A 1 epoch: de novo 생성 → pLDDT 검증 → 도킹 → 스코어링 → 결과.

    반환: SiloACandidateResult 리스트 (성공 + 실패 모두 포함, fail_reason 으로 구분).
    작동 / 미작동 항목은 fail_reason 필드로 정직 기록.

    환각 금지: 실측 불가 지표는 None.
    """
    repo_root = _repo_root()
    t_start = time.time()

    # 출력 디렉토리 구성
    out_base = repo_root / config.output_base_dir
    iter_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    iter_dir = out_base / f"epoch_{iter_ts}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # 수용체 chain B 확보
    try:
        receptor_chainB = _ensure_chain_b_receptor(config.receptor_pdb, iter_dir)
    except Exception as exc:
        logger.error("[SiloA] 수용체 PDB chain B 변환 실패: %s", exc)
        return []

    # de novo 생성 (step02 + step03, RFdiffusion arm — GPU3)
    raw_seqs = _generate_de_novo_sequences(receptor_chainB, config, iter_dir)

    # DiffPepBuilder arm (GPU2) — arm 활성화 시 추가 서열 생성 후 병합
    diffpep_work_dir = iter_dir / "diffpep_arm"
    if config.diffpep_arm_enabled:
        diffpep_work_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[SiloA] DiffPepBuilder arm 시작 (GPU%s)", config.diffpep_cuda_device)
        diffpep_seqs = _generate_diffpep_sequences(
            receptor_pdb=config.receptor_pdb,  # chain A 수용체 (원본 경로 사용)
            config=config,
            work_dir=diffpep_work_dir,
        )
        if diffpep_seqs:
            logger.info("[SiloA] DiffPepBuilder arm: %d개 서열 추가", len(diffpep_seqs))
            raw_seqs = raw_seqs + diffpep_seqs
        else:
            logger.warning("[SiloA] DiffPepBuilder arm: 서열 생성 실패 — RFdiffusion 결과만 사용")
    else:
        diffpep_seqs = []

    if not raw_seqs:
        logger.error("[SiloA] 모든 arm에서 서열 생성 실패 — 이 epoch 결과 없음")
        return []

    results: List[SiloACandidateResult] = []
    sequences_for_toxicity = [s["sequence"] for s in raw_seqs]

    # 독성 배치 추론 (전체 서열 대상)
    toxicity_map = _toxicity_scores(sequences_for_toxicity)

    # ESMFold pLDDT 검증
    # SILO_A_PLDDT_FAILCLOSED=1(기본): 측정 실패(None) → measurement_missing → 게이트 탈락(fail-closed)
    # SILO_A_PLDDT_FAILCLOSED=0: 측정 실패 → 게이트 통과(fail-open, 하위호환, 위험)
    plddt_map: Dict[str, Optional[float]] = {}
    for item in raw_seqs:
        seq = item["sequence"]
        if seq not in plddt_map:
            plddt = _esmfold_plddt(seq, config.esmfold_env, iter_dir)
            plddt_map[seq] = plddt
            if plddt is None:
                if _PLDDT_FAILCLOSED:
                    logger.warning(
                        "[SiloA] pLDDT 측정 불가(seq=%s...) — fail-closed: 게이트 탈락(measurement_missing). "
                        "ESMFold 환경 점검 필요(SILO_A_PLDDT_FAILCLOSED=1). "
                        "하위호환 필요 시 SILO_A_PLDDT_FAILCLOSED=0 설정.",
                        seq[:8],
                    )
                else:
                    logger.warning(
                        "[SiloA] pLDDT 측정 불가(seq=%s...) — fail-open 모드(SILO_A_PLDDT_FAILCLOSED=0): 게이트 통과. "
                        "리더보드 오염 위험. 운영 환경에서는 SILO_A_PLDDT_FAILCLOSED=1 권장.",
                        seq[:8],
                    )

    # 선택성 도킹 대상 제한용 카운터
    selectivity_done = 0

    for item in raw_seqs:
        seq = item["sequence"]
        bb_idx = item["backbone_idx"]
        sq_idx = item["seq_idx"]
        bb_pdb = item.get("backbone_pdb")
        # arm 구분 태그: DiffPepBuilder arm이면 'silo_a_diffpep', 아니면 'silo_a'
        item_mutation_source: str = item.get("mutation_source", "silo_a")
        cid_prefix = "diffpep" if item_mutation_source == "silo_a_diffpep" else "silo_a"
        cid = f"{cid_prefix}_{iter_ts}_bb{bb_idx:02d}_sq{sq_idx:02d}"

        plddt = plddt_map.get(seq)
        # fail-closed(기본): plddt=None(측정 실패)은 게이트 탈락(measurement_missing)
        # fail-open(SILO_A_PLDDT_FAILCLOSED=0): plddt=None이면 통과 — 하위호환 전용
        if plddt is None:
            plddt_pass = not _PLDDT_FAILCLOSED  # fail-closed=False, fail-open=True
        else:
            plddt_pass = plddt >= config.plddt_threshold

        if not plddt_pass:
            if plddt is None:
                # measurement_missing: ESMFold 환경 문제로 측정 자체 불가
                _fail_reason = "esmfold_unavailable(measurement_missing)"
                logger.info(
                    "[SiloA] pLDDT 게이트 탈락(seq=%s...) — measurement_missing(ESMFold 미작동). "
                    "리더보드 제외됨.",
                    seq[:8],
                )
            else:
                _fail_reason = f"pLDDT 게이트 탈락(plddt={plddt:.3f}<{config.plddt_threshold})"
                logger.info("[SiloA] pLDDT 게이트 탈락(seq=%s..., plddt=%.3f)", seq[:8], plddt)
            result = SiloACandidateResult(
                candidate_id=cid,
                sequence=seq,
                backbone_idx=bb_idx,
                seq_idx=sq_idx,
                plddt=plddt,
                plddt_pass=False,
                ddg=None,
                clash_score=None,
                selectivity_margin=None,
                delta_margin=None,
                hc50=None,
                half_life_h=None,
                admet_score=None,
                fail_reason=_fail_reason,
                mutation_source=item_mutation_source,
                extra_scores={
                    "candidate_class": "de_novo",
                    "mutation_source": item_mutation_source,
                    "plddt": plddt,
                    "plddt_status": "measurement_missing" if plddt is None else "below_threshold",
                },
            )
            results.append(result)
            continue

        # 저복잡도 아티팩트 게이트
        lc_fail, lc_reason = _is_low_complexity(seq)
        if lc_fail:
            logger.info("[SiloA] 복잡도 게이트 탈락(seq=%s..., reason=%s)", seq[:8], lc_reason)
            result = SiloACandidateResult(
                candidate_id=cid,
                sequence=seq,
                backbone_idx=bb_idx,
                seq_idx=sq_idx,
                plddt=plddt,
                plddt_pass=plddt_pass,
                ddg=None,
                clash_score=None,
                selectivity_margin=None,
                delta_margin=None,
                hc50=None,
                half_life_h=None,
                admet_score=None,
                fail_reason=f"low_complexity_artifact({lc_reason})",
                mutation_source=item_mutation_source,
                extra_scores={
                    "candidate_class": "de_novo",
                    "mutation_source": item_mutation_source,
                    "plddt": plddt,
                    "low_complexity_reason": lc_reason,
                },
            )
            results.append(result)
            continue

        nsf_fail, nsf_reason = _is_nonspecific_artifact(seq)
        if nsf_fail:
            logger.info('[SiloA] 비특이필터 탈락(seq=%s..., reason=%s)', seq[:8], nsf_reason)
            result = SiloACandidateResult(
                candidate_id=cid, sequence=seq, backbone_idx=bb_idx, seq_idx=sq_idx,
                plddt=plddt, plddt_pass=plddt_pass, ddg=None, clash_score=None,
                selectivity_margin=None, delta_margin=None, hc50=None,
                half_life_h=None, admet_score=None,
                fail_reason=f'nonspecific_artifact({nsf_reason})',
                mutation_source=item_mutation_source,
                extra_scores={
                    'candidate_class': 'de_novo',
                    'mutation_source': item_mutation_source,
                    'plddt': plddt,
                    'nonspecific_reason': nsf_reason,
                },
            )
            results.append(result)
            continue

        # FlexPepDock 온타겟 도킹
        dock_result = _dock_de_novo(seq, bb_pdb, cid, iter_dir, config, repo_root)
        ddg = dock_result.get("ddg")
        clash = dock_result.get("clash_score")
        pdb_path = dock_result.get("pdb_path")
        dock_fail = dock_result.get("fail_reason", "")

        # 저비용 surrogate
        cheap = _cheap_scores(seq)
        half_life = cheap.get("half_life_h")
        admet = cheap.get("admet_score")

        # Lane C: hemolysis soft flag (Aliphatic Index > 120 규칙, L-aa 직쇄 전용)
        # de novo 서열은 대부분 직쇄 L-aa → 규칙 유효. soft penalty + flag만, 하드 reject 금지.
        hemo = _hemolysis_flag(seq, aliphatic_index=cheap.get("aliphatic_index"))
        if hemo.get("applies") and hemo.get("hemolysis_risk") == "HIGH":
            # admet_score에 soft penalty 반영 (탈락 아님)
            _HEMOLYSIS_SOFT_PENALTY = 0.90
            if admet is not None:
                admet = round(float(admet) * _HEMOLYSIS_SOFT_PENALTY, 4)

        # 독성
        tox = toxicity_map.get(seq, {})
        hc50 = tox.get("hc50") if tox else None

        # 선택성 (온타겟 ΔG 실측 성공 + PDB 존재 + 미달 횟수 여유 시)
        sel_margin: Optional[float] = None
        delta_margin: Optional[float] = None
        sel_note = "미수행"

        if (
            ddg is not None
            and pdb_path
            and config.selectivity_enabled
            and selectivity_done < config.max_selectivity_per_epoch
        ):
            sel = _selectivity_score(pdb_path, float(ddg), config, repo_root)
            sel_margin = sel.get("selectivity_margin")
            delta_margin = sel.get("delta_margin")
            sel_note = sel.get("error", "성공") if sel_margin is None else "성공"
            selectivity_done += 1
        elif ddg is None:
            sel_note = f"온타겟 도킹 실패로 선택성 미수행({dock_fail[:80]})"
        elif not config.selectivity_enabled:
            sel_note = "selectivity_enabled=False"
        else:
            sel_note = "epoch 선택성 한도 초과(max_selectivity_per_epoch 도달)"

        extra: Dict[str, Any] = {
            "candidate_class": "de_novo",
            "mutation_source": item_mutation_source,
            "plddt": plddt,
            "plddt_pass": plddt_pass,
            "selectivity_note": sel_note,
            "backbone_pdb": str(bb_pdb) if bb_pdb else None,
            "iter_dir": str(iter_dir),
        }
        extra.update({k: v for k, v in cheap.items() if k not in ("sequence",)})
        if tox:
            extra["pepadmet_hc50"] = hc50
            extra["pepadmet_toxic"] = tox.get("is_toxic")
            extra["hc50_reliable"] = tox.get("graph_note") != "linear_sequence_fallback"

        # Lane C: hemolysis flag extra_scores 기록 (admet_score는 위에서 이미 페널티 반영)
        extra["hemolysis_risk"] = hemo.get("hemolysis_risk", "NA")
        extra["hemolysis_aliphatic_index"] = hemo.get("aliphatic_index")
        extra["hemolysis_applies"] = hemo.get("applies", False)
        extra["hemolysis_reason"] = hemo.get("reason", "")
        # 페널티 반영된 admet_score를 extra에도 동기화
        if admet is not None:
            extra["admet_score"] = admet

        result = SiloACandidateResult(
            candidate_id=cid,
            sequence=seq,
            backbone_idx=bb_idx,
            seq_idx=sq_idx,
            plddt=plddt,
            plddt_pass=plddt_pass,
            ddg=float(ddg) if ddg is not None else None,
            clash_score=float(clash) if clash is not None else None,
            selectivity_margin=float(sel_margin) if sel_margin is not None else None,
            delta_margin=float(delta_margin) if delta_margin is not None else None,
            hc50=float(hc50) if hc50 is not None else None,
            half_life_h=float(half_life) if half_life is not None else None,
            admet_score=float(admet) if admet is not None else None,
            fail_reason=dock_fail if ddg is None else "",
            mutation_source=item_mutation_source,
            extra_scores=extra,
        )
        results.append(result)

    elapsed = round(time.time() - t_start, 1)
    n_pass = sum(1 for r in results if not r.fail_reason)
    n_dock_ok = sum(1 for r in results if r.ddg is not None)
    n_sel_ok = sum(1 for r in results if r.selectivity_margin is not None)
    logger.info(
        "[SiloA] epoch 완료: 생성=%d, pLDDT pass=%d, 도킹 성공=%d, 선택성=%d, %.1f초",
        len(results), n_pass, n_dock_ok, n_sel_ok, elapsed,
    )
    return results


# ---------------------------------------------------------------------------
# Silo A 전용 리더보드
# ---------------------------------------------------------------------------

class SiloALeaderboard:
    """Silo A de novo 후보 영속 리더보드. 별도 JSON, Silo B와 완전 분리."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.entries: List[Dict[str, Any]] = []
        self.screened_seqs: set = set()
        self.n_total: int = 0

    @classmethod
    def load(cls, path: Path, capacity: int = 100) -> "SiloALeaderboard":
        lb = cls(capacity=capacity)
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                lb.entries = data.get("entries", [])
                lb.screened_seqs = set(data.get("screened_seqs", []))
                lb.n_total = int(data.get("n_total", 0))
                for e in lb.entries:
                    lb.screened_seqs.add(e.get("sequence", ""))
        except Exception as exc:
            logger.warning("[SiloALB] 로드 실패(%s) — 빈 리더보드로 시작", exc)
        return lb

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": "silo_a",
            "candidate_class": "de_novo",
            "capacity": self.capacity,
            "n_total": self.n_total,
            "n_unique": len(self.entries),
            "best_ddg": self._best_ddg(),
            "best_selectivity_margin": self._best_sel_margin(),
            "entries": self.entries,
            "screened_seqs": sorted(self.screened_seqs),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def add(self, result: SiloACandidateResult) -> None:
        self.n_total += 1
        seq = result.sequence
        self.screened_seqs.add(seq)

        # 기존 동일 서열 제거 (더 좋은 결과로 교체)
        existing = [e for e in self.entries if e.get("sequence") == seq]
        if existing:
            old_ddg = existing[0].get("ddg")
            new_ddg = result.ddg
            if old_ddg is not None and new_ddg is not None and new_ddg >= old_ddg:
                return  # 기존이 더 좋음 — 업데이트 안 함
            self.entries = [e for e in self.entries if e.get("sequence") != seq]

        self.entries.append(result.to_dict())
        # ddG 오름차순 (낮을수록 좋음) + delta_margin 내림차순
        self.entries.sort(key=lambda e: (
            float(e.get("ddg") or 999.0),
            -float(e.get("selectivity_margin") or -999.0),
        ))
        self.entries = self.entries[:self.capacity]

    def _best_ddg(self) -> Optional[float]:
        ddgs = [e.get("ddg") for e in self.entries if e.get("ddg") is not None]
        return min(ddgs) if ddgs else None

    def _best_sel_margin(self) -> Optional[float]:
        margins = [e.get("selectivity_margin") for e in self.entries if e.get("selectivity_margin") is not None]
        return max(margins) if margins else None

    def count_passing(self, ddg_max: float = -5.0) -> int:
        return sum(1 for e in self.entries if (e.get("ddg") or 999.0) <= ddg_max)


# ---------------------------------------------------------------------------
# experiment_log 기록 헬퍼
# ---------------------------------------------------------------------------

def append_silo_a_records(
    log_path: Path,
    results: List[SiloACandidateResult],
    epoch: int,
    run_id: str,
) -> None:
    """experiment_log.jsonl에 Silo A 결과를 JSONL로 추가."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        for r in results:
            rec = r.to_dict()
            rec["epoch"] = epoch
            rec["run_id"] = run_id
            rec["logged_at"] = ts
            # 필수 식별자 보장
            rec.setdefault("candidate_class", "de_novo")
            rec.setdefault("mutation_source", "silo_a")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
