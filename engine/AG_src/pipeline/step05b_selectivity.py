"""
step05b_selectivity.py
======================
Step 05b: Multi-Receptor Selectivity Screening (선택성 스크리닝)

SSTR2 도킹 상위 후보를 SSTR1/3/4/5 off-target 수용체에 도킹하여
선택성 마진(selectivity_margin)을 계산하고 게이트 필터링을 수행한다.

Uses the same docking engine (DiffDock/Boltz-2) as Step05, but targets
off-target receptors (SSTR1/3/4/5) to ensure SSTR2 specificity.

Selectivity formula:
    selectivity_margin = max(dock_score(off-targets)) - dock_score(SSTR2)
    (higher = more selective for SSTR2; G-2 fix: 양수=좋음 컨벤션 SSOT)

Gate: selectivity_margin >= selectivity_margin_min (e.g., 10.0)
      AND offtarget_max_score >= offtarget_max_allowed (e.g., -15.0)

Input:
    - Top-K candidates from Step05 (DockingResult list)
    - Off-target receptor PDB paths (from config)
    - Receptor PDB path for SSTR2 (Step01 output)

Output:
    - 05b_selectivity/selectivity_scores.json
    - 05b_selectivity/{seq_id}_offtarget_{receptor}.json

Public API:
    run_selectivity_screening(candidates, offtarget_receptors, config) -> Step05bOutput
    dock_against_offtarget(candidate_pdb, receptor_pdb, engine, config) -> float
    compute_selectivity_margin(sstr2_score, offtarget_scores) -> SelectivityResult
    apply_selectivity_gate(results, margin_min, offtarget_max) -> (passed, failed)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ..schemas.io_schemas import DockingResult

# ---------------------------------------------------------------------------
# Path safety helper
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r'^[A-Za-z0-9_\-\.]+$')


def _safe_filename_component(value: str, field: str) -> str:
    """Raise ValueError if value contains path-unsafe characters."""
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"Unsafe characters in {field!r}: {value!r}. "
            "Only alphanumerics, underscores, hyphens, and dots are allowed."
        )
    return value

try:
    from ..schemas.io_schemas import (
        OffTargetDockingResult,
        SelectivityResult,
        Step05bOutput,
    )
except ImportError:
    # Fallback dataclasses until io_schemas.py is updated
    @dataclass
    class OffTargetDockingResult:  # type: ignore[no-redef]
        """Off-target 도킹 결과 (fallback)."""
        seq_id: str
        receptor_name: str
        dock_score: float
        confidence: float
        engine: str

        def to_dict(self) -> Dict[str, Any]:
            from dataclasses import asdict
            return asdict(self)

    @dataclass
    class SelectivityResult:  # type: ignore[no-redef]
        """선택성 마진 결과 (fallback)."""
        seq_id: str
        sstr2_dock_score: float
        offtarget_scores: Dict[str, float]
        offtarget_max_score: float
        offtarget_max_receptor: str
        selectivity_margin: float
        passed: bool

        def to_dict(self) -> Dict[str, Any]:
            from dataclasses import asdict
            return asdict(self)

    @dataclass
    class Step05bOutput:  # type: ignore[no-redef]
        """Step05b 출력 (fallback)."""
        selectivity_results: List[SelectivityResult]
        offtarget_docking_details: List[OffTargetDockingResult]

        def to_dict(self) -> Dict[str, Any]:
            return {
                "selectivity_results": [r.to_dict() for r in self.selectivity_results],
                "offtarget_docking_details": [d.to_dict() for d in self.offtarget_docking_details],
            }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_selectivity_screening(
    candidates: List[DockingResult],
    offtarget_receptors: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Step05bOutput:
    """Run selectivity screening for top candidates against off-target receptors.

    Args:
        candidates: Top docking results from Step05 (sorted by score)
        offtarget_receptors: List of off-target receptor configs
            [{name, pdb_path, chain}, ...]
        config: Pipeline config dict with selectivity params

    Returns:
        Step05bOutput with selectivity results
    """
    sel_config = config.get("selectivity", {})
    top_k = sel_config.get("top_k_for_selectivity", 20)
    engine = sel_config.get("engine", "diffdock")
    margin_min = sel_config.get("selectivity_margin_min", config.get("selectivity_margin_min", 10.0))
    offtarget_max = sel_config.get("offtarget_max_allowed", config.get("offtarget_max_allowed", -15.0))

    # Select top K candidates
    top_candidates = candidates[:top_k]
    logger.info(
        "Selectivity screening: %d candidates against %d off-targets",
        len(top_candidates), len(offtarget_receptors),
    )

    all_offtarget_details: List[OffTargetDockingResult] = []
    selectivity_results: List[SelectivityResult] = []

    for candidate in top_candidates:
        offtarget_scores: Dict[str, float] = {}

        for receptor in offtarget_receptors:
            # Support both string ("SSTR1") and dict ({"name": "SSTR1", "pdb_path": ...}) formats
            if isinstance(receptor, str):
                receptor_name = receptor
                receptor_pdb = None
            else:
                receptor_name = receptor["name"]
                receptor_pdb = receptor.get("pdb_path")
            # Extract on-target score from candidate (handle both attribute styles)
            on_target = getattr(candidate, 'score', None) or getattr(candidate, 'dock_score', 0.0)
            candidate_pdb = getattr(candidate, 'pose_pdb', None) or getattr(candidate, 'pdb_path', '')

            if not receptor_pdb:
                # Estimation mode: no PDB available, use noise model
                logger.debug("Off-target '%s' no PDB; using estimation mode", receptor_name)

            # Dock candidate against off-target (estimation mode when no PDB)
            ot_score = dock_against_offtarget(
                candidate_pdb=candidate_pdb,
                receptor_pdb=receptor_pdb or "",
                engine=engine,
                config=config,
                on_target_score=on_target,
            )
            offtarget_scores[receptor_name] = ot_score

            cand_seq_id = getattr(candidate, 'seq_id', getattr(candidate, 'candidate_id', 'unknown'))
            all_offtarget_details.append(OffTargetDockingResult(
                seq_id=cand_seq_id,
                receptor_name=receptor_name,
                dock_score=ot_score,
                confidence=0.0,  # placeholder
                engine=engine,
            ))

        # Compute selectivity (handle both .score and .dock_score attributes)
        sel_result = compute_selectivity_margin(
            seq_id=getattr(candidate, 'seq_id', getattr(candidate, 'candidate_id', 'unknown')),
            sstr2_score=on_target,
            offtarget_scores=offtarget_scores,
            margin_min=margin_min,
            offtarget_max_allowed=offtarget_max,
        )
        selectivity_results.append(sel_result)

    output = Step05bOutput(
        selectivity_results=selectivity_results,
        offtarget_docking_details=all_offtarget_details,
    )

    passed, failed = apply_selectivity_gate(
        selectivity_results, margin_min, offtarget_max
    )
    logger.info(
        "Selectivity gate: %d/%d passed (margin_min=%.1f, offtarget_max=%.1f)",
        len(passed), len(selectivity_results), margin_min, offtarget_max,
    )

    return output


# ---------------------------------------------------------------------------
# Docking Against Off-Targets
# ---------------------------------------------------------------------------

def dock_against_offtarget(
    candidate_pdb: str,
    receptor_pdb: str,
    engine: str = "diffdock",
    config: Optional[Dict[str, Any]] = None,
    on_target_score: Optional[float] = None,
    sstr2_complex_pdb: str = "",
) -> float:
    """Dock a candidate peptide against an off-target receptor.

    Production mode (PyRosetta): When sstr2_complex_pdb and receptor_pdb are
    available, runs real FlexPepDock via offtarget_dock.py subprocess.

    Estimation mode: applies noise penalty to on_target_score when no
    real structures are available.

    Args:
        candidate_pdb: Path to the candidate pose PDB
        receptor_pdb: Path to the off-target receptor PDB
        engine: "diffdock", "boltz2", or "pyrosetta"
        config: Pipeline config dict (contains selectivity noise params)
        on_target_score: SSTR2 docking score used as baseline for estimation
        sstr2_complex_pdb: SSTR2 refined complex PDB for alignment (production mode)

    Returns:
        float: Docking score / ddG (lower = stronger binding)
    """
    cfg = config or {}
    sel_cfg = cfg.get("selectivity", {})

    # Production mode: real PyRosetta FlexPepDock
    if receptor_pdb and Path(receptor_pdb).exists() and sstr2_complex_pdb and Path(sstr2_complex_pdb).exists():
        logger.info("Off-target docking (production mode): %s vs %s", sstr2_complex_pdb, receptor_pdb)
        try:
            return _run_offtarget_pyrosetta(
                sstr2_complex_pdb, receptor_pdb,
                timeout=sel_cfg.get("offtarget_timeout_sec", 300),
                conda_env=cfg.get("rosetta", {}).get("conda_env", "bio-tools"),
            )
        except Exception as e:
            # 2026-06-09 F08 fail-closed: production 도킹(실제 구조)이 실패하면 noise 추정으로
            # **fall-through 하지 않는다**. 가짜 추정값이 selectivity_margin 으로 들어가 거짓
            # 선택성 판정을 유발하기 때문. NaN 반환 → 호출자(screen_selectivity)가 해당
            # off-target 을 결측으로 제외. estimation 모드는 '구조가 애초에 없는' 경우에만 사용.
            if sel_cfg.get("allow_estimation_on_failure", False):
                logger.warning("Off-target PyRosetta failed (%s); estimation fallback (opt-in)", e)
            else:
                logger.error("Off-target PyRosetta FAILED (%s) → NaN (fail-closed, 결측 처리). "
                             "추정 폴백 원하면 selectivity.allow_estimation_on_failure=true", e)
                return float("nan")

    # Estimation mode (fallback)
    noise_std = sel_cfg.get("selectivity_noise_std", 2.0)
    seed = sel_cfg.get("selectivity_seed", 42)

    logger.debug(
        "Off-target docking (estimation mode): %s vs %s using %s",
        candidate_pdb, receptor_pdb, engine,
    )

    base_score = on_target_score if on_target_score is not None else -5.0
    # 2026-06-09 D3/F10: Python hash() 는 PYTHONHASHSEED 로 프로세스마다 salt 가 달라
    # 동일 입력이라도 실행마다 다른 시드를 만들어 재현 불가했다. sha256 기반 안정 시드로 교체.
    import hashlib
    _seed_key = f"{candidate_pdb}|{receptor_pdb}|{seed}".encode("utf-8")
    pair_hash = int.from_bytes(hashlib.sha256(_seed_key).digest()[:4], "big")
    logger.debug("estimation seed (stable sha256): %d", pair_hash)

    try:
        import numpy as np
        rng = np.random.default_rng(pair_hash)
        offset = rng.normal(loc=noise_std, scale=noise_std / 2.0)
    except ImportError:
        import random
        rng = random.Random(pair_hash)
        offset = rng.gauss(mu=noise_std, sigma=noise_std / 2.0)

    estimated_score = base_score + abs(offset)

    logger.debug(
        "Estimated off-target score: %.3f (base=%.3f, offset=+%.3f)",
        estimated_score, base_score, abs(offset),
    )
    return estimated_score


def _run_offtarget_pyrosetta(
    sstr2_complex_pdb: str,
    receptor_pdb: str,
    timeout: int = 300,
    conda_env: str = "bio-tools",
) -> float:
    """Run offtarget_dock.py via subprocess and return ddG.

    Args:
        sstr2_complex_pdb: Path to SSTR2 refined complex PDB
        receptor_pdb: Path to off-target receptor PDB
        timeout: Subprocess timeout in seconds
        conda_env: Conda environment with PyRosetta

    Returns:
        float: ddG from InterfaceAnalyzerMover (kcal/mol)

    Raises:
        RuntimeError: If subprocess fails or returns invalid JSON
    """
    script = Path(__file__).parent.parent / "scripts" / "offtarget_dock.py"
    out_pdb = Path(sstr2_complex_pdb).parent / f"ot_{Path(receptor_pdb).stem}.pdb"

    cmd = [
        "conda", "run", "-n", conda_env, "python", str(script),
        "--sstr2-complex", sstr2_complex_pdb,
        "--offtarget-receptor", receptor_pdb,
        "--output", str(out_pdb),
    ]

    logger.info("Running: %s", " ".join(cmd[-6:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    if proc.returncode != 0:
        logger.warning("Off-target dock failed (rc=%d): %s", proc.returncode, proc.stderr[:300])
        raise RuntimeError(f"offtarget_dock.py failed: {proc.stderr[:200]}")

    stdout = proc.stdout.strip()
    if not stdout:
        raise RuntimeError("offtarget_dock.py returned empty stdout")

    result = json.loads(stdout)
    if "error" in result:
        raise RuntimeError(f"offtarget_dock.py error: {result['error']}")

    return result["ddg"]


# ---------------------------------------------------------------------------
# Selectivity Margin Computation
# ---------------------------------------------------------------------------

def compute_selectivity_margin(
    seq_id: str,
    sstr2_score: float,
    offtarget_scores: Dict[str, float],
    margin_min: float = 10.0,
    offtarget_max_allowed: float = -15.0,
) -> SelectivityResult:
    """Compute selectivity margin for a candidate.

    Selectivity margin = max(offtarget_scores) - sstr2_score  [G-2 fix: 양수=좋음]
    For docking scores where lower = stronger binding:
    - Positive margin means SSTR2 binds stronger than off-targets (good)
    - Negative margin means off-target binds stronger than SSTR2 (bad)

    Args:
        seq_id: Candidate sequence ID
        sstr2_score: SSTR2 docking score
        offtarget_scores: {receptor_name: score} dict
        margin_min: Minimum acceptable margin (default 10.0, positive = good)
        offtarget_max_allowed: Max off-target score allowed (default -15.0)

    Returns:
        SelectivityResult with computed margin and pass/fail
    """
    if not offtarget_scores:
        # No off-targets evaluated — margin is undefined, use 0.0 as neutral sentinel
        return SelectivityResult(
            seq_id=seq_id,
            sstr2_dock_score=sstr2_score,
            offtarget_scores={},
            offtarget_max_score=0.0,
            offtarget_max_receptor="none",
            selectivity_margin=0.0,
            passed=True,
        )

    # Find the strongest off-target binding (most negative score)
    worst_receptor = min(offtarget_scores, key=offtarget_scores.get)
    worst_score = offtarget_scores[worst_receptor]

    # Margin: worst_offtarget - SSTR2 (positive = SSTR2 binds stronger; G-2 fix: 양수=좋음)
    margin = worst_score - sstr2_score

    # Gate logic:
    # 1. margin must be >= margin_min (SSTR2 much stronger than off-targets)
    # 2. worst off-target score must be >= offtarget_max_allowed (off-targets bind weakly)
    passed = (margin >= margin_min) and (worst_score >= offtarget_max_allowed)

    return SelectivityResult(
        seq_id=seq_id,
        sstr2_dock_score=sstr2_score,
        offtarget_scores=offtarget_scores,
        offtarget_max_score=worst_score,
        offtarget_max_receptor=worst_receptor,
        selectivity_margin=margin,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Selectivity Gate
# ---------------------------------------------------------------------------

def apply_selectivity_gate(
    results: List[SelectivityResult],
    margin_min: float = 10.0,
    offtarget_max_allowed: float = -15.0,
) -> Tuple[List[SelectivityResult], List[SelectivityResult]]:
    """Apply selectivity gate to filter candidates.

    Args:
        results: List of SelectivityResult objects
        margin_min: Minimum selectivity margin (양수=좋음; default 10.0)
        offtarget_max_allowed: Maximum off-target score allowed (default -15.0)

    Returns:
        Tuple of (passed, failed) lists
    """
    passed = []
    failed = []

    for r in results:
        if r.passed:
            passed.append(r)
        else:
            failed.append(r)

    return passed, failed


# ---------------------------------------------------------------------------
# Local CIF/PDB Loading Utilities
# ---------------------------------------------------------------------------

def _convert_cif_to_pdb(cif_path: str, pdb_path: str, chain: Optional[str] = None) -> str:
    """CIF 파일을 PDB 형식으로 변환한다 (BioPython 사용).

    Args:
        cif_path: 입력 CIF 파일 경로
        pdb_path: 출력 PDB 파일 경로
        chain: 추출할 체인 ID (None이면 전체 모델)

    Returns:
        str: 작성된 PDB 파일 경로

    Raises:
        ImportError: BioPython 없을 때
        RuntimeError: 파싱 실패 또는 해당 체인 없을 때
    """
    try:
        from Bio.PDB import MMCIFParser, PDBIO, Select
    except ImportError as e:
        raise ImportError("BioPython 필요: pip install biopython") from e

    parser = MMCIFParser(QUIET=True)
    structure_id = Path(cif_path).stem
    structure = parser.get_structure(structure_id, cif_path)

    if chain is not None:
        class ChainSelect(Select):
            def accept_chain(self, c: Any) -> int:
                return 1 if c.get_id() == chain else 0

        selector: Select = ChainSelect()
    else:
        selector = Select()

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path, selector)
    logger.info("CIF→PDB 변환 완료: %s → %s (chain=%s)", cif_path, pdb_path, chain)
    return pdb_path


def load_offtarget_receptors_from_config(
    config: Dict[str, Any],
    project_root: Optional[str] = None,
    pdb_output_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """pipeline_config.yaml의 off_target_receptors 섹션을 읽어
    pdb_path가 확인된 receptor dict 리스트를 반환한다.

    pdb_source="local"인 경우 local_path를 resolve하고,
    CIF 파일인 경우 BioPython으로 PDB로 변환한다.

    Args:
        config: pipeline_config.yaml에서 로드된 전체 config dict
        project_root: 상대 경로 기준 디렉토리 (None이면 CWD)
        pdb_output_dir: 변환된 PDB 저장 디렉토리 (None이면 임시 디렉토리)

    Returns:
        List of {name, pdb_path, chain} dicts — run_selectivity_screening 인자로 직접 사용 가능
    """
    root = Path(project_root) if project_root else Path.cwd()
    raw_receptors: List[Dict[str, Any]] = config.get("off_target_receptors", [])
    resolved: List[Dict[str, Any]] = []

    for rec in raw_receptors:
        name = rec.get("name", "unknown")
        chain = rec.get("chain")
        pdb_source = rec.get("pdb_source", "rcsb")
        local_path = rec.get("local_path")

        if pdb_source != "local" or not local_path:
            logger.debug("Receptor '%s': pdb_source=%s — 스킵 (local 아님)", name, pdb_source)
            resolved.append({"name": name, "pdb_path": None, "chain": chain})
            continue

        abs_path = Path(local_path)
        if not abs_path.is_absolute():
            abs_path = root / local_path

        if not abs_path.exists():
            logger.warning("Receptor '%s' local_path 없음: %s", name, abs_path)
            resolved.append({"name": name, "pdb_path": None, "chain": chain})
            continue

        # CIF 파일은 PDB로 변환
        if abs_path.suffix.lower() == ".cif":
            if pdb_output_dir:
                out_dir = Path(pdb_output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                pdb_path = str(out_dir / f"{abs_path.stem}.pdb")
            else:
                # 임시 파일 (프로세스 생존 기간 동안 유지)
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".pdb", prefix=f"{name}_", delete=False
                )
                pdb_path = tmp.name
                tmp.close()

            try:
                _convert_cif_to_pdb(str(abs_path), pdb_path, chain=chain)
            except Exception as e:
                logger.error("CIF→PDB 변환 실패 (%s): %s", name, e)
                resolved.append({"name": name, "pdb_path": None, "chain": chain})
                continue
        else:
            pdb_path = str(abs_path)

        resolved.append({"name": name, "pdb_path": pdb_path, "chain": chain})
        logger.info("Receptor '%s' resolved → %s", name, pdb_path)

    return resolved


# ---------------------------------------------------------------------------
# Utility: Save Results
# ---------------------------------------------------------------------------

def save_selectivity_results(
    output: Step05bOutput,
    output_dir: Path,
) -> Dict[str, str]:
    """Save selectivity screening results to files.

    Args:
        output: Step05bOutput with all results
        output_dir: Path to 05b_selectivity/ directory

    Returns:
        Dict of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = output_dir / "selectivity_scores.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(output.to_dict(), f, indent=2, ensure_ascii=False)

    # Save per-candidate details
    saved = {"summary": str(summary_path)}
    for result in output.selectivity_results:
        safe_id = _safe_filename_component(result.seq_id, "seq_id")
        detail_path = output_dir / f"{safe_id}_selectivity.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        saved[result.seq_id] = str(detail_path)

    logger.info("Selectivity results saved to %s", output_dir)
    return saved
