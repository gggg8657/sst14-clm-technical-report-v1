"""
step06_rosetta.py
=================
Step 06: 상위 M개 정밀 정제/에너지 (Rosetta Refinement via PyRosetta MCP)

PyRosetta MCP 서버를 통해 Step05에서 선발된 상위 M개 후보를
FastRelax -> FlexPepDock -> ddG 계산 프로토콜로 정밀 정제하고
결합 자유 에너지(ddG)와 에너지 분해(per-residue energy)를 계산한다.

Uses the PyRosetta MCP server to run:
  FastRelax (cartesian) -> FlexPepDock refinement -> ddG calculation

Gate criteria (from gate_thresholds.yaml):
    ddG <= -5.0 kcal/mol   (rosetta_ddg_max)
    clash_score == 0        (rosetta_clash_max)
    constraint_violations == 0

Input:
    - Top-M docking candidates (DockingResult list from Step05)
    - receptor PDB path

Output:
    - 06_rosetta/refined_{seq_id}.pdb
    - 06_rosetta/energy_table.json

Public API:
    run_rosetta_refinement(candidates, receptor_pdb, config) -> Step06Output
    refine_single(complex_pdb, protocol)                     -> RosettaResult
    compute_binding_ddg(complex_pdb)                         -> float
    energy_decomposition(complex_pdb)                        -> Dict[int, float]
    apply_rosetta_gate(results, ddg_threshold,
                       clash_max)                            -> (passed, failed)
    enhanced_score(complex_pdb, score_fn, with_constraints)  -> Dict
    run_interface_analysis(complex_pdb)                       -> Dict
    run_structure_validation(complex_pdb, peptide_chain)      -> Dict
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import RosettaResult, Step06Output, DockingResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DDG_MAX: float = -5.0
_DEFAULT_CLASH_MAX: int = 10
_DEFAULT_PROTOCOL: str = "flexpep_refine"
_PYROSETTA_CONDA_ENV: str = "bio-tools"


# ---------------------------------------------------------------------------
# Result cache — keyed by sha256(pdb_content + sequence + protocol)
# ---------------------------------------------------------------------------

class _ResultCache:
    """In-process cache for PyRosetta refinement results.

    Cache key is sha256(pdb_content || sequence || protocol).
    Optionally persists to a JSON sidecar in the output directory so that
    restarted runs skip already-computed candidates.
    """

    def __init__(self) -> None:
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._disk_path: Optional[Path] = None

    def set_disk_path(self, out_dir: Path) -> None:
        self._disk_path = out_dir / ".rosetta_cache.json"
        if self._disk_path.exists():
            try:
                cached = json.loads(self._disk_path.read_text(encoding="utf-8"))
                self._mem.update(cached)
                logger.info("[Step06][Cache] Loaded %d entries from %s", len(cached), self._disk_path)
            except (json.JSONDecodeError, OSError):
                pass

    @staticmethod
    def _make_key(pdb_content: str, sequence: str, protocol: str) -> str:
        blob = f"{pdb_content}\x00{sequence}\x00{protocol}".encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:24]

    def get(self, pdb_content: str, sequence: str, protocol: str) -> Optional[RosettaResult]:
        key = self._make_key(pdb_content, sequence, protocol)
        entry = self._mem.get(key)
        if entry is None:
            return None
        logger.info("[Step06][Cache] HIT key=%s", key)
        return RosettaResult(
            seq_id=entry.get("seq_id", ""),
            ddg=float(entry["ddg"]),
            total_score=float(entry["total_score"]),
            clash_score=float(entry["clash_score"]),
            constraint_violations=int(entry["constraint_violations"]),
            refined_pdb=entry.get("refined_pdb", ""),
            pre_score=float(entry.get("pre_score", 0.0)),
            score_delta=float(entry.get("score_delta", 0.0)),
        )

    def put(self, pdb_content: str, sequence: str, protocol: str, result: RosettaResult) -> None:
        key = self._make_key(pdb_content, sequence, protocol)
        self._mem[key] = result.to_dict()
        logger.info("[Step06][Cache] STORE key=%s (ddG=%.2f)", key, result.ddg)
        self._flush()

    def _flush(self) -> None:
        if self._disk_path is None:
            return
        try:
            self._disk_path.write_text(
                json.dumps(self._mem, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            logger.warning("[Step06][Cache] Disk flush failed: %s", exc)


_cache = _ResultCache()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def is_pyrosetta_available(conda_env: str = _PYROSETTA_CONDA_ENV) -> bool:
    """Check whether PyRosetta is callable via the specified conda environment.

    Runs a lightweight subprocess probe. Result is cached for the process lifetime.
    """
    if hasattr(is_pyrosetta_available, "_cached"):
        return is_pyrosetta_available._cached

    try:
        proc = subprocess.run(
            ["conda", "run", "-n", conda_env, "python", "-c",
             "import pyrosetta; print('OK')"],
            capture_output=True, text=True, timeout=30,
        )
        available = proc.returncode == 0 and "OK" in proc.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        available = False

    is_pyrosetta_available._cached = available
    logger.info("[Step06] PyRosetta availability: %s", available)
    return available


def run_rosetta_refinement(
    candidates: List[DockingResult],
    receptor_pdb: str,
    config: Dict[str, Any],
) -> Step06Output:
    """상위 M개 도킹 후보를 Rosetta로 정밀 정제하고 ddG를 계산한다.

    Orchestration entry for Step 06.  Tries to invoke PyRosetta via the
    MCP server subprocess; falls back to a lightweight stub when PyRosetta
    is not available (useful for CI / dry-run).

    Args:
        candidates:   Top-M DockingResult candidates from Step05.
        receptor_pdb: Path to receptor PDB (for complex assembly).
        config:       Full pipeline configuration dict.

    Returns:
        Step06Output with RosettaResult for each candidate.
    """
    run_id: str = config.get("run_id", "default_run")
    output_base: Path = Path(config.get("output_base_dir", "runs")) / run_id
    out_dir: Path = output_base / "06_rosetta"
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_cfg: Dict[str, Any] = config.get("gate_thresholds", {})
    ddg_max: float = float(gate_cfg.get("rosetta_ddg_max", _DEFAULT_DDG_MAX))
    clash_max: int = int(gate_cfg.get("rosetta_clash_max", _DEFAULT_CLASH_MAX))
    top_m: int = int(config.get("iteration", {}).get("top_m_rosetta", 10))

    rosetta_cfg: Dict[str, Any] = config.get("rosetta", {})
    timeout_per = int(rosetta_cfg.get("timeout_per_candidate_sec", 1800))
    os.environ.setdefault("ROSETTA_TIMEOUT", str(timeout_per))

    # Initialize result cache (loads disk sidecar if present)
    _cache.set_disk_path(out_dir)

    # Limit to top_m candidates
    candidates = candidates[:top_m]
    receptor_content = Path(receptor_pdb).read_text(encoding="utf-8")

    pyrosetta_ok = is_pyrosetta_available()
    logger.info(
        "[Step06] Rosetta refinement for %d candidates (ddG<=%s, clash<=%d, pyrosetta=%s).",
        len(candidates),
        ddg_max,
        clash_max,
        pyrosetta_ok,
    )

    sequence_map: Dict[str, str] = config.get("sequence_map", {})

    rosetta_results: List[RosettaResult] = []
    for idx, candidate in enumerate(candidates, 1):
        seq_id = candidate.seq_id
        pose_pdb_path = candidate.pose_pdb
        target_sequence = sequence_map.get(seq_id, "")
        logger.info("[Step06] [%d/%d] Refining: %s (seq=%s)",
                     idx, len(candidates), seq_id, target_sequence[:20] or "N/A")

        if not pose_pdb_path or not Path(pose_pdb_path).exists():
            logger.warning("[Step06] No pose PDB for %s; building from receptor.", seq_id)
            complex_pdb_text = receptor_content
        else:
            pose_content = Path(pose_pdb_path).read_text(encoding="utf-8")
            complex_pdb_text = _assemble_complex(receptor_content, pose_content)

        # Check cache before running expensive subprocess
        cached_result = _cache.get(complex_pdb_text, target_sequence, _DEFAULT_PROTOCOL)
        if cached_result is not None:
            cached_result.seq_id = seq_id
            rosetta_results.append(cached_result)
            logger.info("[Step06] %s: CACHED ddG=%.2f, clash=%.1f", seq_id, cached_result.ddg, cached_result.clash_score)
            continue

        # Write temporary complex PDB for Rosetta
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(complex_pdb_text)
            tmp_path = tmp.name

        try:
            t0 = time.monotonic()
            result = refine_single(tmp_path, protocol=_DEFAULT_PROTOCOL, target_sequence=target_sequence)
            elapsed = time.monotonic() - t0
            result.seq_id = seq_id

            # Save refined PDB
            refined_path = out_dir / f"refined_{seq_id}.pdb"
            if result.refined_pdb and Path(result.refined_pdb).exists():
                import shutil
                shutil.copy(result.refined_pdb, refined_path)
                result = RosettaResult(
                    seq_id=seq_id,
                    ddg=result.ddg,
                    total_score=result.total_score,
                    clash_score=result.clash_score,
                    constraint_violations=result.constraint_violations,
                    refined_pdb=str(refined_path),
                    pre_score=result.pre_score,
                    score_delta=result.score_delta,
                )

            # Store in cache
            _cache.put(complex_pdb_text, target_sequence, _DEFAULT_PROTOCOL, result)

            rosetta_results.append(result)
            logger.info(
                "[Step06] %s: ddG=%.2f, total=%.2f, clash=%.1f, cv=%d (%.1fs)",
                seq_id,
                result.ddg,
                result.total_score,
                result.clash_score,
                result.constraint_violations,
                elapsed,
            )
        except Exception as exc:
            logger.error("[Step06] Rosetta refinement failed for %s: %s", seq_id, exc)
            rosetta_results.append(
                RosettaResult(
                    seq_id=seq_id,
                    ddg=0.0,
                    total_score=0.0,
                    clash_score=999.0,
                    constraint_violations=999,
                    refined_pdb="",
                )
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    passed, failed = apply_rosetta_gate(rosetta_results, ddg_max, clash_max)
    logger.info(
        "[Step06] Rosetta gate: %d passed / %d failed.",
        len(passed),
        len(failed),
    )

    _save_energy_table(rosetta_results, out_dir)
    return Step06Output(rosetta_results=rosetta_results)


def refine_single(
    complex_pdb: str,
    protocol: str = _DEFAULT_PROTOCOL,
    target_sequence: str = "",
) -> RosettaResult:
    """단일 복합체 PDB에 Rosetta 정제 프로토콜을 적용한다.

    Dispatches to the PyRosetta MCP server subprocess.  If PyRosetta is
    not installed, falls back to a stub that returns placeholder scores.

    Args:
        complex_pdb: File path to the assembled receptor-peptide complex PDB.
        protocol:    Refinement protocol name.  Currently supports
                     ``"flexpep_refine"`` and ``"fast_relax"``.
        target_sequence: Peptide amino acid sequence (1-letter code) for
                     MutateResidue approach using reference complex.

    Returns:
        RosettaResult with energy scores and path to the refined PDB.
    """
    if not is_pyrosetta_available():
        logger.warning("[Step06] PyRosetta not available; using energy stub.")
        return _stub_rosetta_result(complex_pdb)

    try:
        logger.info("[Step06] Calling PyRosetta subprocess (protocol=%s, pdb=%s, seq=%s)",
                     protocol, Path(complex_pdb).name, target_sequence[:20] or "N/A")
        return _run_pyrosetta_subprocess(complex_pdb, protocol, target_sequence)
    except (FileNotFoundError, RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(
            "[Step06] PyRosetta subprocess failed (%s); using energy stub.", exc
        )
        return _stub_rosetta_result(complex_pdb)


def compute_binding_ddg(complex_pdb: str) -> float:
    """수용체-펩타이드 복합체의 결합 ΔΔG를 계산한다.

    Attempts to call PyRosetta's ddG protocol via subprocess.  Returns a
    stub value (0.0) when PyRosetta is unavailable.

    Args:
        complex_pdb: Path to the refined complex PDB.

    Returns:
        Binding ΔΔG in kcal/mol (more negative = stronger binding).
    """
    try:
        script = _get_pyrosetta_script("ddg_calc")
        result = _run_script(script, args=["--pdb", complex_pdb])
        return float(result.strip())
    except Exception as exc:
        logger.warning("[Step06] compute_binding_ddg stub (PyRosetta unavailable): %s", exc)
        return 0.0


def energy_decomposition(complex_pdb: str) -> Dict[int, float]:
    """복합체 PDB의 잔기별 에너지를 딕셔너리로 반환한다.

    Attempts per-residue energy breakdown via PyRosetta subprocess.
    Returns an empty dict when PyRosetta is unavailable.

    Args:
        complex_pdb: Path to the refined complex PDB.

    Returns:
        Dict mapping residue number (int) -> energy score (float).
    """
    try:
        script = _get_pyrosetta_script("energy_decomp")
        result = _run_script(script, args=["--pdb", complex_pdb])
        return json.loads(result)
    except Exception as exc:
        logger.warning("[Step06] energy_decomposition unavailable: %s", exc)
        return {}


def apply_rosetta_gate(
    results: List[RosettaResult],
    ddg_threshold: float = _DEFAULT_DDG_MAX,
    clash_max: int = _DEFAULT_CLASH_MAX,
) -> Tuple[List[RosettaResult], List[RosettaResult]]:
    """ddG, clash, constraint 기준으로 Rosetta 게이트를 적용한다.

    A candidate passes if ALL of the following hold:
        ``ddg <= ddg_threshold``
        ``clash_score <= clash_max``
        ``constraint_violations == 0``

    Args:
        results:        List of RosettaResult to evaluate.
        ddg_threshold:  Maximum allowed ddG (default -5.0 kcal/mol).
        clash_max:      Maximum allowed clash score (default 0).

    Returns:
        Tuple ``(passed, failed)``.
    """
    passed: List[RosettaResult] = []
    failed: List[RosettaResult] = []
    for r in results:
        if (
            r.ddg <= ddg_threshold
            and r.clash_score <= clash_max
            and r.constraint_violations == 0
        ):
            passed.append(r)
        else:
            failed.append(r)
    return passed, failed


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_ca_coords(pdb_text: str) -> List[Tuple[float, float, float]]:
    """PDB 텍스트에서 CA 원자의 (x, y, z) 좌표를 추출한다."""
    coords = []
    for line in pdb_text.splitlines():
        if line[:6].strip() in ("ATOM",) and line[12:16].strip() == "CA":
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
            except (ValueError, IndexError):
                pass
    return coords


def _center_of_mass(coords: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    """좌표 리스트의 중심점을 계산한다."""
    n = len(coords)
    if n == 0:
        return (0.0, 0.0, 0.0)
    cx = sum(c[0] for c in coords) / n
    cy = sum(c[1] for c in coords) / n
    cz = sum(c[2] for c in coords) / n
    return (cx, cy, cz)


def _translate_pdb_atoms(
    pdb_text: str, dx: float, dy: float, dz: float
) -> str:
    """PDB의 ATOM/HETATM 레코드의 좌표를 (dx, dy, dz)만큼 이동한다."""
    out_lines = []
    for line in pdb_text.splitlines():
        if line[:6].strip() in ("ATOM", "HETATM") and len(line) >= 54:
            try:
                x = float(line[30:38]) + dx
                y = float(line[38:46]) + dy
                z = float(line[46:54]) + dz
                line = f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
            except (ValueError, IndexError):
                pass
        out_lines.append(line)
    return "\n".join(out_lines)


def _get_reference_peptide_com() -> Optional[Tuple[float, float, float]]:
    """참조 복합체(AlphaFold3)에서 펩타이드 체인(A)의 CA 중심점을 가져온다."""
    ref_paths = [
        Path(__file__).parent.parent.parent / "PRST_N_FM" / "data" / "fold_test1" / "fold_test1_model_0.pdb",
    ]
    for ref_path in ref_paths:
        if ref_path.exists():
            ref_text = ref_path.read_text(encoding="utf-8")
            # Chain A = peptide in reference complex
            pep_cas = []
            for line in ref_text.splitlines():
                if (line[:6].strip() == "ATOM"
                        and line[12:16].strip() == "CA"
                        and len(line) >= 54
                        and line[21] == "A"):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        pep_cas.append((x, y, z))
                    except (ValueError, IndexError):
                        pass
            if pep_cas:
                return _center_of_mass(pep_cas)
    return None


def _assemble_complex(receptor_pdb: str, peptide_pdb: str) -> str:
    """수용체 PDB와 펩타이드 PDB를 합쳐 복합체 PDB 텍스트를 반환한다.

    Receptor lines are remapped to chain A, peptide lines to chain B.
    The peptide is translated to match the reference complex's peptide
    binding position (from AlphaFold3 prediction) to provide a proper
    starting configuration for FlexPepDock.
    """
    # Translate peptide to the reference binding position
    pep_cas = _parse_ca_coords(peptide_pdb)
    ref_pep_com = _get_reference_peptide_com()

    if pep_cas and ref_pep_com:
        pep_com = _center_of_mass(pep_cas)
        dx = ref_pep_com[0] - pep_com[0]
        dy = ref_pep_com[1] - pep_com[1]
        dz = ref_pep_com[2] - pep_com[2]
        peptide_pdb = _translate_pdb_atoms(peptide_pdb, dx, dy, dz)
        logger.info("[Step06] Peptide translated to reference binding site (dx=%.1f, dy=%.1f, dz=%.1f)", dx, dy, dz)
    elif pep_cas:
        # Fallback: place near receptor center
        rec_cas = _parse_ca_coords(receptor_pdb)
        if rec_cas:
            rec_com = _center_of_mass(rec_cas)
            pep_com = _center_of_mass(pep_cas)
            dx = rec_com[0] + 15.0 - pep_com[0]
            dy = rec_com[1] - pep_com[1]
            dz = rec_com[2] - pep_com[2]
            peptide_pdb = _translate_pdb_atoms(peptide_pdb, dx, dy, dz)
            logger.warning("[Step06] Reference complex not found; using receptor COM + 15A offset.")

    # Remap receptor chain to 'A'
    rec_lines = []
    for l in receptor_pdb.splitlines():
        tag = l[:6].strip()
        if tag in ("ATOM", "HETATM"):
            if len(l) >= 22:
                l = l[:21] + "A" + l[22:]
            rec_lines.append(l)
        elif tag in ("TER", "REMARK", "CRYST1"):
            rec_lines.append(l)

    # Remap peptide chain to 'B'
    pep_lines = []
    for l in peptide_pdb.splitlines():
        tag = l[:6].strip()
        if tag in ("ATOM", "HETATM"):
            if len(l) >= 22:
                l = l[:21] + "B" + l[22:]
            pep_lines.append(l)

    return "\n".join(rec_lines + ["TER"] + pep_lines + ["END\n"])


def _get_reference_complex_path() -> Optional[str]:
    """참조 복합체 PDB 경로를 반환한다 (AlphaFold3 예측 구조)."""
    ref_paths = [
        Path(__file__).parent.parent.parent / "PRST_N_FM" / "data" / "fold_test1" / "fold_test1_model_0.pdb",
    ]
    for p in ref_paths:
        if p.exists():
            return str(p)
    return None


def _run_pyrosetta_subprocess(
    complex_pdb: str,
    protocol: str,
    target_sequence: str = "",
) -> RosettaResult:
    """conda run -n bio-tools で PyRosetta スクリプトを実行する."""
    script_map = {
        "flexpep_refine": "flexpep_dock.py",
        "flexpep_abinitio": "flexpep_dock.py",
        "fast_relax": "fast_design.py",
        "fast_design": "fast_design.py",
    }
    script_name = script_map.get(protocol, "flexpep_dock.py")
    # Look for the script next to this module or in PRST_N_FM/bionemo/
    script_candidates = [
        Path(__file__).parent.parent / "scripts" / script_name,
        Path(__file__).parent.parent.parent / "PRST_N_FM" / "bionemo" / script_name,
        Path(__file__).parent.parent / "tools" / "mcp" / script_name,
    ]
    script_path: Optional[Path] = None
    for candidate in script_candidates:
        if candidate.exists():
            script_path = candidate
            break

    if script_path is None:
        raise FileNotFoundError(
            f"[Step06] PyRosetta script '{script_name}' not found in search paths."
        )

    out_pdb = Path(complex_pdb).parent / (Path(complex_pdb).stem + "_refined.pdb")
    cmd = [
        "conda", "run", "-n", _PYROSETTA_CONDA_ENV,
        "python", str(script_path),
        "--input", complex_pdb,
        "--output", str(out_pdb),
        "--protocol", protocol,
    ]

    # Use MutateResidue approach when reference complex is available
    ref_complex = _get_reference_complex_path()
    if ref_complex and target_sequence and script_name == "flexpep_dock.py":
        cmd.extend([
            "--reference-complex", ref_complex,
            "--target-sequence", target_sequence,
            "--peptide-chain", "1",
        ])
        logger.info("[Step06] Using MutateResidue mode (ref=%s, seq=%s)",
                     Path(ref_complex).name, target_sequence)
    timeout = int(os.environ.get("ROSETTA_TIMEOUT", "1800"))
    logger.info("[Step06] Subprocess cmd: %s (timeout=%ds)", " ".join(cmd[:6]) + " ...", timeout)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - t0

    if proc.returncode != 0:
        logger.error("[Step06] PyRosetta stderr (last 500 chars): %s", proc.stderr[-500:])
        raise RuntimeError(
            f"[Step06] PyRosetta script failed (code {proc.returncode}, {elapsed:.1f}s): "
            f"{proc.stderr[:500]}"
        )

    logger.info("[Step06] PyRosetta subprocess completed in %.1fs", elapsed)

    # Parse scores from stdout JSON
    try:
        scores = json.loads(proc.stdout.strip())
    except json.JSONDecodeError:
        logger.warning("[Step06] Could not parse JSON from stdout: %s", proc.stdout[:200])
        scores = {}

    result = RosettaResult(
        seq_id="",
        ddg=float(scores.get("ddg", 0.0)),
        total_score=float(scores.get("total_score", 0.0)),
        clash_score=float(scores.get("clash_score", 0.0)),
        constraint_violations=int(scores.get("constraint_violations", 0)),
        refined_pdb=str(out_pdb) if out_pdb.exists() else "",
        pre_score=float(scores.get("pre_score", 0.0)),
        score_delta=float(scores.get("score_delta", 0.0)),
    )
    logger.info("[Step06] PyRosetta result: ddG=%.4f, total=%.4f, clash=%.1f, pre=%.4f, delta=%.4f",
                 result.ddg, result.total_score, result.clash_score, result.pre_score, result.score_delta)
    return result


def _stub_rosetta_result(complex_pdb: str) -> RosettaResult:
    """PyRosetta 미설치/실패 시 반환하는 플레이스홀더 결과.

    2026-06-09 F08 fail-closed: 기본적으로 stub 은 **실패값(ddg=999)** 으로 반환해
    ddG 게이트(<= -5)에서 탈락 → 랭킹 진입 차단. 이전엔 ddg=0.0 을 반환해 "계산 안 됨"이
    "중립 결합"으로 오인되어 가짜 후보가 순위에 오를 수 있었다. 개발/CI 에서 stub 을
    통과시키려면 환경변수 AG_ALLOW_ROSETTA_STUB=1 로 명시 opt-in. 어느 경우든 stub=True 태깅.
    """
    import os
    allow = os.environ.get("AG_ALLOW_ROSETTA_STUB", "").lower() in ("1", "true", "yes")
    if allow:
        logger.warning("[Step06] STUB (dev opt-in) for %s — ddg=0.0 is NOT real.", complex_pdb)
        return RosettaResult(seq_id="", ddg=0.0, total_score=0.0, clash_score=0.0,
                             constraint_violations=0, refined_pdb=complex_pdb, stub=True)
    logger.error(
        "[Step06] PyRosetta unavailable/failed for %s → FAIL-CLOSED (ddg=999, ranking 제외). "
        "dev/CI 에서 중립 stub 필요 시 AG_ALLOW_ROSETTA_STUB=1.", complex_pdb,
    )
    return RosettaResult(
        seq_id="", ddg=999.0, total_score=999.0, clash_score=999.0,
        constraint_violations=999, refined_pdb=complex_pdb, stub=True,
    )


def _get_pyrosetta_script(script_name: str) -> str:
    """공통 PyRosetta 스크립트 경로를 반환한다."""
    candidates = [
        Path(__file__).parent.parent / "scripts" / f"{script_name}.py",
        Path(__file__).parent.parent.parent / "PRST_N_FM" / "bionemo" / f"{script_name}.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"PyRosetta script '{script_name}.py' not found.")


def _run_script(script_path: str, args: List[str]) -> str:
    """지정 Python 스크립트를 bio-tools conda 환경에서 실행하고 stdout을 반환한다."""
    cmd = ["conda", "run", "-n", _PYROSETTA_CONDA_ENV, "python", script_path] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[:400])
    return proc.stdout.strip()


def _save_energy_table(results: List[RosettaResult], out_dir: Path) -> None:
    """에너지 표를 JSON으로 저장한다."""
    table = {
        "total": len(results),
        "results": [r.to_dict() for r in results],
    }
    path = out_dir / "energy_table.json"
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("[Step06] Energy table written -> %s", path)


# ---------------------------------------------------------------------------
# Enhanced scoring (ref2015 / beta_nov16 support)
# ---------------------------------------------------------------------------


def enhanced_score(
    complex_pdb: str,
    score_function: str = "ref2015",
    with_constraints: bool = False,
) -> Dict[str, Any]:
    """Score a complex with ref2015 or beta_nov16, optionally with constraints.

    Runs via the interface_analysis module for per-residue decomposition,
    and adds interface ddG with and without constraint weights.

    Args:
        complex_pdb:    Path to complex PDB.
        score_function: "ref2015" or "beta_nov16".
        with_constraints: If True, enable atom_pair_constraint weight.

    Returns:
        Dict with total_score, ddg, ddg_constrained, per_residue_energies, etc.
    """
    try:
        from .interface_analysis import per_residue_energy, compute_buried_surface_area
    except ImportError:
        from AG_src.pipeline.interface_analysis import per_residue_energy, compute_buried_surface_area

    energy_report = per_residue_energy(complex_pdb, score_function=score_function)
    bsa_report = compute_buried_surface_area(complex_pdb)

    result: Dict[str, Any] = {
        "score_function": score_function,
        "total_score": energy_report.get("total_score", 0.0),
        "interface_dG": bsa_report.get("interface_dG", 0.0),
        "delta_sasa": bsa_report.get("delta_sasa", 0.0),
        "n_residues": len(energy_report.get("residues", [])),
    }

    # Per-residue breakdown (top 10 most stabilizing interface residues)
    residues = energy_report.get("residues", [])
    interface_residues = sorted(residues, key=lambda r: r.get("total_energy", 0.0))
    result["top_stabilizing"] = interface_residues[:10]

    if with_constraints:
        # ddG with constraints — uses subprocess to pass constraint weight
        try:
            script = _get_pyrosetta_script("ddg_calc")
            ddg_cst = _run_script(
                script, args=["--pdb", complex_pdb, "--constraints"]
            )
            result["ddg_constrained"] = float(ddg_cst.strip())
        except Exception as exc:
            logger.warning("[Step06] ddG with constraints failed: %s", exc)
            result["ddg_constrained"] = result["interface_dG"]
    else:
        result["ddg_constrained"] = result["interface_dG"]

    return result


def run_interface_analysis(complex_pdb: str) -> Dict[str, Any]:
    """Run full interface analysis on a refined complex PDB.

    Convenience wrapper that delegates to the interface_analysis module.

    Args:
        complex_pdb: Path to complex PDB.

    Returns:
        Full interface analysis report dict.
    """
    try:
        from .interface_analysis import analyze_interface
    except ImportError:
        from AG_src.pipeline.interface_analysis import analyze_interface

    return analyze_interface(complex_pdb)


def run_structure_validation(
    complex_pdb: str,
    peptide_chain: Optional[int] = None,
    disulfide_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Run full structure validation on a refined complex PDB.

    Convenience wrapper that delegates to the structure_validation module.

    Args:
        complex_pdb:    Path to complex PDB.
        peptide_chain:  Chain to focus on (None for all).
        disulfide_pairs: Expected disulfide pairs.

    Returns:
        Full structure validation report dict.
    """
    try:
        from .structure_validation import validate_structure
    except ImportError:
        from AG_src.pipeline.structure_validation import validate_structure

    return validate_structure(
        complex_pdb,
        peptide_chain=peptide_chain,
        disulfide_pairs=disulfide_pairs,
    )


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step06: Rosetta refinement standalone test")
    parser.add_argument("--complex-pdb", required=True, help="Path to receptor-peptide complex PDB")
    parser.add_argument("--seq-id", default="bb00_seq00")
    parser.add_argument("--output-dir", default="runs/test_run")
    args = parser.parse_args()

    dummy = DockingResult(
        seq_id=args.seq_id,
        engine="diffdock",
        score=-1.5,
        confidence=-8.0,
        pose_pdb=args.complex_pdb,
        rank=1,
    )
    cfg: Dict[str, Any] = {
        "run_id": "test_run",
        "output_base_dir": args.output_dir,
        "gate_thresholds": {"rosetta_ddg_max": -5.0, "rosetta_clash_max": 10},
        "iteration": {"top_m_rosetta": 10},
    }
    result = run_rosetta_refinement([dummy], args.complex_pdb, cfg)
    for r in result.rosetta_results:
        print(f"{r.seq_id}: ddG={r.ddg:.2f}, clash={r.clash_score:.1f}")
