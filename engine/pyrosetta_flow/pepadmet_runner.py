"""
pepADMET 독성 예측 — subprocess로 pepadmet conda env 호출.

bio-tools env에서 실행하는 runner.py가 pepadmet env (DGL 0.4.3)를
subprocess로 호출하여 독성 예측 결과를 받아옴.

Usage:
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch
    results = predict_toxicity_batch(["AGCKNFFWKTFTSC", "AGCKYEFWKTVTSC"])
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

def _default_pepadmet_repo() -> Path:
    """PRST_N_FM/local_models/pepadmet/repo — infer 스크립트 기본 경로와 일치."""
    # pepadmet_runner.py 위치: .../PRST_N_FM/AgenticAI4SCIENCE_pyrosetta_track/repos/ai4sci-kaeri/pyrosetta_flow/
    root = Path(__file__).resolve().parent.parent.parent.parent.parent
    return root / "local_models" / "pepadmet" / "repo"


_PEPADMET_REPO = (
    Path(os.environ["PEPADMET_REPO"])
    if os.environ.get("PEPADMET_REPO")
    else _default_pepadmet_repo()
)
_CONDA_ENV = "pepadmet"
# inline script 대신 파일 직접 호출 (hang 방지)
_INFER_SCRIPT = Path(__file__).resolve().parent / "pepadmet_infer_script.py"


def predict_toxicity_batch(
    sequences: list[str],
    smiles_list: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run pepADMET toxicity prediction via subprocess.

    Parameters
    ----------
    sequences : list[str]
        Amino acid sequences.
    smiles_list : list[str] or None
        SMILES strings. If None, attempts conversion via RDKit.

    Returns
    -------
    list[dict]
        Per-sequence toxicity predictions.
    """
    if not _PEPADMET_REPO.exists():
        return [{"sequence": s, "available": False, "error": "pepADMET repo not found"} for s in sequences]

    if not _INFER_SCRIPT.exists():
        return [{"sequence": s, "available": False, "error": "pepadmet_infer_script.py not found"} for s in sequences]

    # SMILES 변환 (bio-tools env의 smiles_converter 사용)
    if smiles_list is None:
        try:
            from .smiles_converter import sequence_to_smiles
            smiles_list = [sequence_to_smiles(s) or "" for s in sequences]
        except ImportError:
            smiles_list = [""] * len(sequences)

    input_data = [{"sequence": s, "smiles": sm} for s, sm in zip(sequences, smiles_list)]
    input_json = json.dumps(input_data)

    # PEPADMET_REPO 경로를 환경변수로 전달
    env = os.environ.copy()
    env["PEPADMET_REPO"] = str(_PEPADMET_REPO)

    try:
        # inline script(-c) 대신 파일 직접 호출 — conda run 버퍼링 hang 방지
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", _CONDA_ENV,
             "python3", str(_INFER_SCRIPT), input_json],
            capture_output=True, text=True, timeout=120, env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr[-500:] if result.stderr else "unknown error"
            return [{"sequence": s, "available": False, "error": error_msg[:200]} for s in sequences]

        # stdout에서 JSON 파싱 (마지막 줄)
        for line in reversed(result.stdout.strip().split("\n")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        return [{"sequence": s, "available": False, "error": "no JSON output"} for s in sequences]

    except subprocess.TimeoutExpired:
        return [{"sequence": s, "available": False, "error": "timeout"} for s in sequences]
    except Exception as exc:
        return [{"sequence": s, "available": False, "error": str(exc)[:200]} for s in sequences]
