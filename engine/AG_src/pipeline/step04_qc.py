"""
step04_qc.py
============
Step 04: 빠른 구조 QC (Fast Structure Quality Control via ESMFold)

ESMFold NIM API를 사용하여 Step03에서 설계된 펩타이드 서열의 구조를
빠르게 예측하고 pLDDT 기반 품질 관리(QC) 게이트를 적용한다.

Uses the ESMFold NIM API to rapidly predict structures of sequences from
Step03 and applies pLDDT-based QC gates.  Candidates failing either the
mean pLDDT or interface pLDDT threshold are rejected.

Gate criteria (from gate_thresholds.yaml):
    mean pLDDT    >= 75   (esmfold_plddt_min)
    interface pLDDT >= 70  (esmfold_interface_plddt_min)

Input:
    - sequences: List[SequenceEntry] from Step03
    - pocket_residues: interface residue numbers for interface pLDDT

Output:
    - 04_qc/esmfold_{seq_id}.pdb    (one per sequence)
    - 04_qc/qc_summary.json

Public API:
    run_qc(sequences, config)                              -> Step04Output
    predict_and_evaluate(sequence, seq_id)                 -> QCResult
    apply_plddt_gate(results, plddt_threshold,
                     interface_threshold)                  -> (passed, failed)
    compute_interface_plddt(pdb_text, interface_residues)  -> float
    save_qc_results(results, output_dir)                   -> paths
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import QCResult, Step04Output, SequenceEntry


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PLDDT_MIN: float = 75.0
_DEFAULT_INTERFACE_PLDDT_MIN: float = 70.0


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def run_qc(
    sequences: List[SequenceEntry],
    config: Dict[str, Any],
) -> Step04Output:
    """모든 서열에 대해 ESMFold 구조 예측 + pLDDT QC 게이트를 실행한다.

    Orchestration entry for Step 04.

    Args:
        sequences: All SequenceEntry items from Step03Output.sequences.
        config:    Full pipeline configuration dict.

    Returns:
        Step04Output with QCResult for every evaluated sequence.
    """
    run_id: str = config.get("run_id", "default_run")
    output_base: Path = Path(config.get("output_base_dir", "runs")) / run_id
    out_dir: Path = output_base / "04_qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    gate_cfg: Dict[str, Any] = config.get("gate_thresholds", {})
    plddt_min: float = float(gate_cfg.get("esmfold_plddt_min", _DEFAULT_PLDDT_MIN))
    interface_min: float = float(
        gate_cfg.get("esmfold_interface_plddt_min", _DEFAULT_INTERFACE_PLDDT_MIN)
    )
    pocket_residues: List[int] = config.get("receptor", {}).get("pocket_residues", [])

    # Disulfide bond gate configuration
    gates_enabled: Dict[str, bool] = gate_cfg.get("gates_enabled", {})
    disulfide_gate_enabled: bool = gates_enabled.get("disulfide", True)
    disulfide_max_dist: float = float(gate_cfg.get("disulfide_bond_max_distance", 2.5))
    disulfide_cys_pos: List[int] = gate_cfg.get("disulfide_cys_positions", [])

    logger.info(
        "[Step04] Running ESMFold QC on %d sequences (pLDDT>=%s, iface>=%s).",
        len(sequences),
        plddt_min,
        interface_min,
    )

    qc_results: List[QCResult] = []
    for entry in sequences:
        logger.info("[Step04] ESMFold predict: %s", entry.seq_id)
        try:
            result = predict_and_evaluate(
                sequence=entry.sequence,
                seq_id=entry.seq_id,
                out_dir=out_dir,
                pocket_residues=pocket_residues,
                plddt_min=plddt_min,
                interface_min=interface_min,
                disulfide_cys_positions=disulfide_cys_pos if disulfide_cys_pos else None,
                disulfide_max_distance=disulfide_max_dist,
                disulfide_gate_enabled=disulfide_gate_enabled,
            )
            qc_results.append(result)
        except Exception as exc:
            logger.error("[Step04] ESMFold failed for %s: %s", entry.seq_id, exc)
            qc_results.append(
                QCResult(
                    seq_id=entry.seq_id,
                    plddt_mean=0.0,
                    plddt_interface=0.0,
                    pdb_path="",
                    passed_gate=False,
                )
            )

    passed, failed = apply_plddt_gate(qc_results, plddt_min, interface_min)
    logger.info(
        "[Step04] QC gate: %d passed / %d failed.",
        len(passed),
        len(failed),
    )

    save_qc_results(qc_results, out_dir)
    return Step04Output(qc_results=qc_results)


def predict_and_evaluate(
    sequence: str,
    seq_id: str,
    out_dir: Optional[Path] = None,
    pocket_residues: Optional[List[int]] = None,
    plddt_min: float = _DEFAULT_PLDDT_MIN,
    interface_min: float = _DEFAULT_INTERFACE_PLDDT_MIN,
    disulfide_cys_positions: Optional[List[int]] = None,
    disulfide_max_distance: float = 2.5,
    disulfide_gate_enabled: bool = True,
) -> QCResult:
    """단일 서열에 대해 ESMFold 예측을 수행하고 QCResult를 반환한다.

    Args:
        sequence:         Amino acid sequence (one-letter code).
        seq_id:           Unique sequence identifier for file naming.
        out_dir:          Directory to write the predicted PDB.
        pocket_residues:  Residue numbers defining the binding interface.
        plddt_min:        Mean pLDDT threshold.
        interface_min:    Interface pLDDT threshold.

    Returns:
        QCResult with pLDDT scores and gate decision.
    """
    import requests

    api_key = _resolve_api_key()
    endpoint = "https://health.api.nvidia.com/v1/biology/nvidia/esmfold"
    payload = {"sequence": sequence}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(
            f"[Step04] ESMFold API HTTP {resp.status_code}: {resp.text[:400]}"
        )

    data = resp.json()
    pdb_text: str = (
        data.get("pdbs", [""])[0]
        or data.get("pdb", "")
        or data.get("output_pdb", "")
        or data.get("result", {}).get("pdb", "")
    )
    if not pdb_text:
        raise RuntimeError(
            f"[Step04] ESMFold returned no PDB. Keys: {list(data.keys())}"
        )

    # Save PDB
    pdb_path = ""
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdb_file = out_dir / f"esmfold_{seq_id}.pdb"
        pdb_file.write_text(pdb_text, encoding="utf-8")
        pdb_path = str(pdb_file)

    # Parse pLDDT from B-factor column
    plddt_mean = _extract_mean_plddt(pdb_text)
    plddt_iface = (
        compute_interface_plddt(pdb_text, pocket_residues or [])
        if pocket_residues
        else plddt_mean
    )

    passed = bool(plddt_mean >= plddt_min and plddt_iface >= interface_min)
    logger.info(
        "[Step04] %s: pLDDT=%.1f, iface_pLDDT=%.1f -> %s",
        seq_id,
        plddt_mean,
        plddt_iface,
        "PASS" if passed else "FAIL",
    )

    # Disulfide bond distance check
    disulfide_intact: Optional[bool] = None
    disulfide_distance: Optional[float] = None
    if disulfide_cys_positions and disulfide_gate_enabled:
        disulfide_intact, disulfide_distance = check_disulfide_bond(
            pdb_text, disulfide_cys_positions, disulfide_max_distance,
        )
        if disulfide_intact is False:
            passed = False
            logger.info(
                "[Step04] %s: FAILED disulfide gate (distance=%.2f A > %.1f A)",
                seq_id, disulfide_distance or 0.0, disulfide_max_distance,
            )

    return QCResult(
        seq_id=seq_id,
        plddt_mean=round(plddt_mean, 2),
        plddt_interface=round(plddt_iface, 2),
        pdb_path=pdb_path,
        passed_gate=passed,
        disulfide_intact=disulfide_intact,
        disulfide_distance=disulfide_distance,
    )


def apply_plddt_gate(
    results: List[QCResult],
    plddt_threshold: float = _DEFAULT_PLDDT_MIN,
    interface_threshold: float = _DEFAULT_INTERFACE_PLDDT_MIN,
) -> Tuple[List[QCResult], List[QCResult]]:
    """pLDDT 임계값 기반 게이트를 적용하여 통과/실패 목록을 반환한다.

    A candidate passes if BOTH of the following hold:
        ``plddt_mean >= plddt_threshold``  AND
        ``plddt_interface >= interface_threshold``

    Args:
        results:             List of QCResult objects to evaluate.
        plddt_threshold:     Mean pLDDT minimum (default 75).
        interface_threshold: Interface pLDDT minimum (default 70).

    Returns:
        Tuple ``(passed, failed)`` where each element is a List[QCResult].
    """
    passed: List[QCResult] = []
    failed: List[QCResult] = []
    for r in results:
        plddt_ok = r.plddt_mean >= plddt_threshold and r.plddt_interface >= interface_threshold
        # passed_gate includes disulfide bond check from predict_and_evaluate()
        gate_ok = getattr(r, "passed_gate", True)
        if plddt_ok and gate_ok:
            passed.append(r)
        else:
            failed.append(r)
    return passed, failed


def compute_interface_plddt(
    pdb_text: str,
    interface_residues: List[int],
) -> float:
    """PDB B-factor 컬럼에서 계면 잔기의 평균 pLDDT를 계산한다.

    ESMFold writes per-residue pLDDT into the B-factor (tempFactor) column.
    This function extracts those values for the specified *interface_residues*
    and returns their mean.

    Args:
        pdb_text:            Full PDB text from ESMFold.
        interface_residues:  List of residue sequence numbers to include.

    Returns:
        Mean pLDDT of interface residues, or overall mean if no match found.
    """
    if not interface_residues:
        return _extract_mean_plddt(pdb_text)

    res_set = set(interface_residues)
    values: List[float] = []

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            res_num = int(line[22:26].strip())
            b_factor = float(line[60:66].strip())
        except (ValueError, IndexError):
            continue
        if res_num in res_set:
            values.append(b_factor)

    if not values:
        logger.debug(
            "[Step04] compute_interface_plddt: no matching residues found for %s; "
            "returning overall pLDDT.",
            interface_residues[:5],
        )
        return _extract_mean_plddt(pdb_text)

    return sum(values) / len(values)


def check_disulfide_bond(
    pdb_text: str,
    cys_positions: List[int],
    max_distance: float = 2.5,
) -> Tuple[bool, Optional[float]]:
    """PDB에서 시스테인 SG 원자 간 거리를 측정하여 이황화 결합 유지 여부를 확인한다.

    Checks whether the sulfur-sulfur (SG-SG) distance between two cysteine
    residues is within covalent bond range.  A typical S-S bond is ~2.05 A;
    distances exceeding *max_distance* indicate a broken disulfide.

    Args:
        pdb_text:       Full PDB text (e.g. from ESMFold).
        cys_positions:  Two residue sequence numbers (1-indexed) to check.
                        e.g. [3, 13] for Cys3-Cys13.
        max_distance:   Maximum SG-SG distance in Angstroms (default 2.5).

    Returns:
        Tuple of (intact: bool, distance: float or None).
        ``intact`` is True when distance <= max_distance.
        ``distance`` is None when SG atoms could not be found.
    """
    if len(cys_positions) != 2:
        logger.warning("[Step04] check_disulfide_bond: expected 2 positions, got %d", len(cys_positions))
        return True, None  # Cannot check, assume intact

    pos_a, pos_b = cys_positions[0], cys_positions[1]
    sg_coords: Dict[int, Tuple[float, float, float]] = {}

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "SG":
            continue
        try:
            res_num = int(line[22:26].strip())
        except (ValueError, IndexError):
            continue
        if res_num in (pos_a, pos_b):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                sg_coords[res_num] = (x, y, z)
            except (ValueError, IndexError):
                continue

    if pos_a not in sg_coords or pos_b not in sg_coords:
        logger.warning(
            "[Step04] check_disulfide_bond: SG atoms not found for positions %d, %d. "
            "Found SG at: %s. Marking as BROKEN (Cys may be mutated).",
            pos_a, pos_b, list(sg_coords.keys()),
        )
        return False, None  # Cannot measure -> assume broken (Cys likely mutated)

    xa, ya, za = sg_coords[pos_a]
    xb, yb, zb = sg_coords[pos_b]
    distance = math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
    intact = distance <= max_distance

    logger.info(
        "[Step04] Disulfide check Cys%d-Cys%d: SG-SG distance=%.2f A -> %s (max=%.1f)",
        pos_a, pos_b, distance, "INTACT" if intact else "BROKEN", max_distance,
    )
    return intact, round(distance, 3)


def save_qc_results(results: List[QCResult], output_dir: Path) -> Dict[str, str]:
    """QC 결과를 qc_summary.json 파일로 저장하고 경로 딕셔너리를 반환한다.

    Args:
        results:    List of QCResult objects.
        output_dir: Directory to write qc_summary.json.

    Returns:
        Dict mapping ``"qc_summary"`` to the written JSON path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.passed_gate),
        "failed": sum(1 for r in results if not r.passed_gate),
        "results": [r.to_dict() for r in results],
    }
    summary_path = output_dir / "qc_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("[Step04] QC summary written -> %s", summary_path)
    return {"qc_summary": str(summary_path)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_mean_plddt(pdb_text: str) -> float:
    """PDB ATOM 레코드의 B-factor 컬럼에서 평균 pLDDT를 추출한다."""
    values: List[float] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            b_factor = float(line[60:66].strip())
            values.append(b_factor)
        except (ValueError, IndexError):
            continue
    return sum(values) / len(values) if values else 0.0


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
    parser = argparse.ArgumentParser(description="Step04: ESMFold QC standalone test")
    parser.add_argument(
        "--fasta",
        required=True,
        help="FASTA file with sequences to evaluate",
    )
    parser.add_argument("--output-dir", default="runs/test_run")
    args = parser.parse_args()

    # Parse sequences from FASTA
    entries: List[SequenceEntry] = []
    fasta_path = Path(args.fasta)
    lines = fasta_path.read_text().splitlines()
    seq_id = ""
    seq_buf: List[str] = []
    idx = 0
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            if seq_id and seq_buf:
                entries.append(
                    SequenceEntry(
                        backbone_idx=idx // 8,
                        seq_idx=idx % 8,
                        sequence="".join(seq_buf),
                        fasta_path=args.fasta,
                        seq_id=seq_id,
                    )
                )
                idx += 1
            seq_id = line[1:].split()[0]
            seq_buf = []
        else:
            seq_buf.append(line)
    if seq_id and seq_buf:
        entries.append(
            SequenceEntry(
                backbone_idx=idx // 8,
                seq_idx=idx % 8,
                sequence="".join(seq_buf),
                fasta_path=args.fasta,
                seq_id=seq_id,
            )
        )

    cfg: Dict[str, Any] = {
        "run_id": "test_run",
        "output_base_dir": args.output_dir,
    }
    result = run_qc(entries, cfg)
    print(f"Step04: {len(result.passed())} passed / {len(result.qc_results)} total")
