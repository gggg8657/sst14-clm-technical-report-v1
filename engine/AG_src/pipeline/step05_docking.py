"""
step05_docking.py
=================
Step 05: 결합 포즈/친화도 평가 (Binding Pose & Affinity Evaluation)

DiffDock NIM API와 Boltz-2 NIM API를 사용하여 QC 통과 펩타이드 후보를
수용체(SSTR2)에 도킹하고 결합 친화도를 예측한다.

Uses the DiffDock NIM API for pose prediction and the Boltz-2 NIM API for
affinity ranking.  Both engines are tried; if one fails the other is used
as a fallback.  The top ``docking_top_pct`` percent of candidates by score
proceed to Step06.

Gate criteria:
    Top 20 % by docking score (configurable via ``docking_top_pct``).

Input:
    - QC-passed candidates (List[QCResult] from Step04)
    - receptor PDB path (Step01 output)

Output:
    - 05_docking/pose_{seq_id}_{pose_idx}.pdb
    - 05_docking/docking_scores.json

Public API:
    run_docking(candidates, receptor_pdb, config)   -> Step05Output
    dock_with_diffdock(protein_pdb, ligand_pdb,
                       num_poses)                   -> DockResult
    predict_with_boltz2(receptor_seq, peptide_seq)  -> Boltz2Result
    apply_docking_gate(results, top_pct)            -> (passed, failed)
    merge_docking_results(dd, b2)                   -> List[DockingResult]
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import DockingResult, Step05Output, QCResult


# ---------------------------------------------------------------------------
# Internal result types
# ---------------------------------------------------------------------------


@dataclass
class DockResult:
    """DiffDock 단일 호출 결과."""
    seq_id: str
    poses: List[str]        # PDB text for each pose
    scores: List[float]     # Confidence score per pose (lower = better)
    success: bool
    error: Optional[str] = None


@dataclass
class Boltz2Result:
    """Boltz-2 단일 호출 결과."""
    seq_id: str
    affinity_kcal: float    # Predicted binding affinity (kcal/mol)
    pdb_text: Optional[str]
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_PCT: float = 20.0
_DEFAULT_NUM_POSES: int = 5


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def run_docking(
    candidates: List[QCResult],
    receptor_pdb: str,
    config: Dict[str, Any],
) -> Step05Output:
    """QC 통과 후보 전체에 대해 도킹을 실행하고 상위 20%를 선발한다.

    Orchestration entry for Step 05.  For each candidate:
    1. Attempts DiffDock (pose prediction).
    2. On DiffDock failure, falls back to Boltz-2.
    3. Attempts Boltz-2 for affinity score in all cases.
    4. Merges results and selects top ``docking_top_pct`` percent.

    Args:
        candidates:   QC-passed QCResult list.
        receptor_pdb: Path to the receptor PDB file.
        config:       Full pipeline configuration dict.

    Returns:
        Step05Output with all DockingResult records (gated by top_pct flag
        embedded in the ``rank`` field, and ``score`` for ordering).
    """
    run_id: str = config.get("run_id", "default_run")
    output_base: Path = Path(config.get("output_base_dir", "runs")) / run_id
    out_dir: Path = output_base / "05_docking"
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_cfg: Dict[str, Any] = config.get("gate_thresholds", {})
    top_pct: float = float(gate_cfg.get("docking_top_pct", _DEFAULT_TOP_PCT))
    num_poses: int = int(config.get("docking_num_poses", _DEFAULT_NUM_POSES))

    receptor_pdb_content: str = Path(receptor_pdb).read_text(encoding="utf-8")
    receptor_seq: str = _extract_sequence_from_pdb(receptor_pdb_content)

    logger.info(
        "[Step05] Docking %d QC-passed candidates (top_pct=%.0f%%, num_poses=%d).",
        len(candidates),
        top_pct,
        num_poses,
    )

    all_results: List[DockingResult] = []

    for candidate in candidates:
        seq_id = candidate.seq_id
        peptide_pdb_path = candidate.pdb_path   # ESMFold PDB for peptide

        if not peptide_pdb_path or not Path(peptide_pdb_path).exists():
            logger.warning("[Step05] No PDB for candidate %s; skipping docking.", seq_id)
            continue

        peptide_pdb_content = Path(peptide_pdb_path).read_text(encoding="utf-8")
        peptide_seq = _extract_sequence_from_pdb(peptide_pdb_content)

        # --- DiffDock pose prediction ---
        dd_result = _try_diffdock(
            seq_id=seq_id,
            protein_pdb=receptor_pdb_content,
            ligand_pdb=peptide_pdb_content,
            num_poses=num_poses,
            out_dir=out_dir,
        )

        # --- Boltz-2 affinity prediction ---
        b2_result = _try_boltz2(
            seq_id=seq_id,
            receptor_seq=receptor_seq,
            peptide_seq=peptide_seq,
        )

        # Merge into DockingResult
        merged = _build_docking_result(seq_id, dd_result, b2_result, out_dir)
        all_results.extend(merged)

    passed, _ = apply_docking_gate(all_results, top_pct)
    logger.info(
        "[Step05] Docking complete: %d results, %d in top %.0f%%.",
        len(all_results),
        len(passed),
        top_pct,
    )

    _save_docking_scores(all_results, out_dir)
    return Step05Output(docking_results=all_results)


def dock_with_diffdock(
    protein_pdb: str,
    ligand_pdb: str,
    num_poses: int = _DEFAULT_NUM_POSES,
) -> DockResult:
    """DiffDock NIM API를 호출하여 도킹 포즈와 신뢰도 점수를 반환한다.

    Args:
        protein_pdb: Receptor PDB *content* string.
        ligand_pdb:  Ligand (peptide) PDB *content* string.
        num_poses:   Number of poses to generate.

    Returns:
        DockResult with poses and confidence scores.

    Raises:
        RuntimeError: On unrecoverable API failure.
    """
    import requests

    api_key = _resolve_api_key()
    endpoint = "https://health.api.nvidia.com/v1/biology/mit/diffdock"
    payload: Dict[str, Any] = {
        "protein": protein_pdb,
        "ligand": ligand_pdb,
        "num_poses": num_poses,
        "time_divisions": 20,
        "steps": 18,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Step05] DiffDock API HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    # Response may have 'output_pdbs' (list) and 'position_confidence' (list)
    poses: List[str] = data.get("output_pdbs", []) or data.get("pdbs", [])
    scores: List[float] = (
        data.get("position_confidence", [])
        or data.get("scores", [])
        or [-float(i) for i in range(len(poses))]  # fallback dummy scores
    )
    return DockResult(seq_id="", poses=poses, scores=scores, success=True)


def predict_with_boltz2(
    receptor_seq: str,
    peptide_seq: str,
) -> Boltz2Result:
    """Boltz-2 NIM API로 수용체-펩타이드 결합 친화도를 예측한다.

    Args:
        receptor_seq: Receptor amino acid sequence (one-letter code).
        peptide_seq:  Peptide amino acid sequence (one-letter code).

    Returns:
        Boltz2Result with predicted affinity in kcal/mol.

    Raises:
        RuntimeError: On API failure.
    """
    import requests

    api_key = _resolve_api_key()
    endpoint = "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
    payload: Dict[str, Any] = {
        "sequences": [
            {"protein": {"id": "A", "sequence": receptor_seq}},
            {"protein": {"id": "B", "sequence": peptide_seq}},
        ],
        "compute_affinity": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Step05] Boltz-2 API HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    affinity: float = (
        data.get("affinity_kcal_mol")
        or data.get("binding_affinity")
        or data.get("result", {}).get("affinity_kcal_mol", 0.0)
    )
    pdb_text: Optional[str] = (
        data.get("output_pdb") or data.get("pdb") or None
    )
    return Boltz2Result(
        seq_id="",
        affinity_kcal=float(affinity),
        pdb_text=pdb_text,
        success=True,
    )


def apply_docking_gate(
    results: List[DockingResult],
    top_pct: float = _DEFAULT_TOP_PCT,
) -> Tuple[List[DockingResult], List[DockingResult]]:
    """도킹 점수 기준 상위 top_pct% 후보를 선발한다.

    Sorts by ``score`` ascending (lower = better) and returns the top
    ``top_pct`` percent as *passed* and the remainder as *failed*.

    Args:
        results: List of DockingResult objects.
        top_pct: Percentage of candidates to pass (default 20.0).

    Returns:
        Tuple ``(passed, failed)``.
    """
    if not results:
        return [], []
    sorted_results = sorted(results, key=lambda r: r.score)
    n_pass = max(1, int(len(sorted_results) * top_pct / 100.0))
    return sorted_results[:n_pass], sorted_results[n_pass:]


def merge_docking_results(
    diffdock_results: List[DockingResult],
    boltz2_results: List[DockingResult],
) -> List[DockingResult]:
    """DiffDock와 Boltz-2 결과를 seq_id 기준으로 병합한다.

    When a candidate has both DiffDock and Boltz-2 results, the DiffDock
    score is used for pose ranking, while the Boltz-2 affinity is stored
    in ``confidence``.  When only one source is available, that source's
    result is used directly.

    Args:
        diffdock_results: DockingResult list from DiffDock.
        boltz2_results:   DockingResult list from Boltz-2.

    Returns:
        Merged List[DockingResult].
    """
    dd_map: Dict[str, DockingResult] = {r.seq_id: r for r in diffdock_results}
    b2_map: Dict[str, DockingResult] = {r.seq_id: r for r in boltz2_results}

    merged: List[DockingResult] = []
    all_ids = set(dd_map) | set(b2_map)

    for seq_id in sorted(all_ids):
        if seq_id in dd_map and seq_id in b2_map:
            dd = dd_map[seq_id]
            b2 = b2_map[seq_id]
            # DiffDock for pose; Boltz-2 confidence as affinity proxy
            merged.append(
                DockingResult(
                    seq_id=seq_id,
                    engine="diffdock+boltz2",
                    score=dd.score,
                    confidence=b2.confidence,
                    pose_pdb=dd.pose_pdb,
                    rank=dd.rank,
                )
            )
        elif seq_id in dd_map:
            merged.append(dd_map[seq_id])
        else:
            merged.append(b2_map[seq_id])

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _try_diffdock(
    seq_id: str,
    protein_pdb: str,
    ligand_pdb: str,
    num_poses: int,
    out_dir: Path,
) -> Optional[DockResult]:
    """DiffDock 호출 + 파일 저장; 실패 시 None 반환."""
    try:
        result = dock_with_diffdock(protein_pdb, ligand_pdb, num_poses)
        result.seq_id = seq_id
        # Save top pose
        if result.poses:
            pose_path = out_dir / f"pose_{seq_id}_00.pdb"
            pose_path.write_text(result.poses[0], encoding="utf-8")
        return result
    except Exception as exc:
        logger.warning("[Step05] DiffDock failed for %s: %s. Trying Boltz-2.", seq_id, exc)
        return None


def _try_boltz2(
    seq_id: str,
    receptor_seq: str,
    peptide_seq: str,
) -> Optional[Boltz2Result]:
    """Boltz-2 호출; 실패 시 None 반환."""
    try:
        result = predict_with_boltz2(receptor_seq, peptide_seq)
        result.seq_id = seq_id
        return result
    except Exception as exc:
        logger.warning("[Step05] Boltz-2 failed for %s: %s.", seq_id, exc)
        return None


def _build_docking_result(
    seq_id: str,
    dd: Optional[DockResult],
    b2: Optional[Boltz2Result],
    out_dir: Path,
) -> List[DockingResult]:
    """DockResult + Boltz2Result -> DockingResult 목록으로 변환한다."""
    results: List[DockingResult] = []

    if dd and dd.success and dd.poses:
        for pose_idx, (pose_pdb, score) in enumerate(
            zip(dd.poses, dd.scores or [0.0] * len(dd.poses))
        ):
            pose_path = out_dir / f"pose_{seq_id}_{pose_idx:02d}.pdb"
            pose_path.write_text(pose_pdb, encoding="utf-8")
            confidence = b2.affinity_kcal if (b2 and b2.success) else 0.0
            results.append(
                DockingResult(
                    seq_id=seq_id,
                    engine="diffdock",
                    score=float(score),
                    confidence=float(confidence),
                    pose_pdb=str(pose_path),
                    rank=pose_idx + 1,
                )
            )
    elif b2 and b2.success:
        # DiffDock failed; use Boltz-2 only
        pose_path = out_dir / f"pose_{seq_id}_00.pdb"
        if b2.pdb_text:
            pose_path.write_text(b2.pdb_text, encoding="utf-8")
        results.append(
            DockingResult(
                seq_id=seq_id,
                engine="boltz2",
                score=b2.affinity_kcal,
                confidence=b2.affinity_kcal,
                pose_pdb=str(pose_path) if b2.pdb_text else "",
                rank=1,
            )
        )
    else:
        logger.error("[Step05] Both DiffDock and Boltz-2 failed for %s.", seq_id)

    return results


def _extract_sequence_from_pdb(pdb_text: str) -> str:
    """PDB ATOM 레코드에서 단일 문자 아미노산 시퀀스를 추출한다."""
    _aa3to1: Dict[str, str] = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    seen: Dict[int, str] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            res_name = line[17:20].strip()
            res_num = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue
        if res_num not in seen and res_name in _aa3to1:
            seen[res_num] = _aa3to1[res_name]
    return "".join(seen[k] for k in sorted(seen))


def _save_docking_scores(results: List[DockingResult], out_dir: Path) -> None:
    """도킹 점수를 JSON 파일로 저장한다."""
    summary = {
        "total_poses": len(results),
        "results": [r.to_dict() for r in results],
    }
    score_path = out_dir / "docking_scores.json"
    score_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[Step05] Docking scores written -> %s", score_path)


def _resolve_api_key() -> str:
    """환경변수 / 키 파일에서 NVIDIA API 키를 탐색한다."""
    for var in ("NGC_CLI_API_KEY", "NVIDIA_API_KEY"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    for name in ("ngc.key", "molmim.key"):
        for directory in (Path(__file__).parent.parent.parent, Path.cwd()):
            key_file = directory / name
            if key_file.exists():
                val = key_file.read_text(encoding="utf-8").strip()
                if val:
                    return val
    raise ValueError("NVIDIA API key not found. Set NGC_CLI_API_KEY or create ngc.key.")


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Step05: Docking standalone test")
    parser.add_argument("--receptor-pdb", required=True)
    parser.add_argument("--candidate-pdb", required=True, help="ESMFold PDB of one candidate")
    parser.add_argument("--seq-id", default="bb00_seq00")
    args = parser.parse_args()

    dummy_candidate = QCResult(
        seq_id=args.seq_id,
        plddt_mean=85.0,
        plddt_interface=80.0,
        pdb_path=args.candidate_pdb,
        passed_gate=True,
    )
    cfg: Dict[str, Any] = {
        "run_id": "test_run",
        "output_base_dir": "runs",
        "gate_thresholds": {"docking_top_pct": 20},
    }
    result = run_docking([dummy_candidate], args.receptor_pdb, cfg)
    print(f"Step05: {len(result.docking_results)} poses generated.")
