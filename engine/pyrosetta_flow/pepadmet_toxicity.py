# FROZEN (2026-06-25): 검증된 기준선 보존용 수정 금지. 개선은 pepadmet_toxicity_v2.py 에서. 발굴 import 전환됨.
"""
pepADMET 독성 모델 추론 래퍼.

GitHub: ifyoungnet/pepADMET (JCIM 2026, 66, 936-946)
모델: toxicity_early_stop.pth — MLR-GAT (RGCN + MLP + Attention)
4개 태스크: binary toxicity, 6-class type, 4-class neurotoxicity, HC50

실제 추론은 pepadmet conda env에서 `pepadmet_infer_script.py`가 수행하고,
본 모듈은 `pepadmet_runner.predict_toxicity_batch`를 통해 subprocess 결과를 노출한다.

Usage:
    from pyrosetta_flow.pepadmet_toxicity import predict_toxicity
    result = predict_toxicity("AGCKNFFWKTFTSC", smiles="C[C@H]...")
"""
from __future__ import annotations

from typing import Any

from .pepadmet_runner import predict_toxicity_batch


def _normalize_runner_row(row: dict[str, Any], sequence: str) -> dict[str, Any]:
    """runner 출력 한 건을 레거시 API 형태로 맞춘다.

    HC50 단위 면책(VR-A2, 2026-06-17): pepADMET task_3 HC50 출력은 모델의 회귀 타깃
    (hemolytic conc., log-scale 추정)이며 **절대 단위(μg/mL vs μM)가 원논문 대비 미검증**이다.
    따라서 hc50 절대값 해석 금지 — native baseline 대비 상대 비교로만 사용(multiobjective).
    또한 SMILES 파싱 실패로 선형 폴백(graph_note="linear_sequence_fallback")된 경우
    이황화/고리 구조가 소실되므로 cyclic 펩타이드의 hc50 은 신뢰 불가 → hc50_reliable=False.
    """
    graph_note = row.get("graph_note")
    base: dict[str, Any] = {
        "available": row.get("available", False),
        "model": "pepADMET-MLR-GAT",
        "sequence": row.get("sequence", sequence),
        "binary_toxicity": row.get("binary_toxicity"),
        "toxicity_type": row.get("toxicity_type"),
        "neurotoxicity_type": row.get("neurotoxicity_type"),
        "hc50": row.get("hc50"),
        "hc50_unit": "pepADMET-model-output (절대단위 미검증; 상대비교 전용)",
        "graph_note": graph_note,
        # 선형 폴백이면 cyclic 구조 소실 → hc50 신뢰 불가
        "hc50_reliable": (graph_note != "linear_sequence_fallback"),
    }
    if "is_toxic" in row:
        base["is_toxic"] = row["is_toxic"]
    if "toxicity_type_confidence" in row:
        base["toxicity_type_confidence"] = row["toxicity_type_confidence"]
    if "neurotoxicity_confidence" in row:
        base["neurotoxicity_confidence"] = row["neurotoxicity_confidence"]
    if row.get("error"):
        base["error"] = row["error"]
    smi = row.get("smiles")
    if isinstance(smi, str) and smi:
        base["smiles"] = smi[:80] + "..." if len(smi) > 80 else smi
    return base


def predict_toxicity(
    sequence: str,
    smiles: str | None = None,
) -> dict[str, Any]:
    """Predict toxicity via pepadmet_runner (conda-isolated inference)."""
    smiles_arg: list[str] | None
    if smiles is None:
        smiles_arg = None
    else:
        smiles_arg = [smiles]
    rows = predict_toxicity_batch([sequence], smiles_arg)
    if not rows:
        return {
            "available": False,
            "model": "pepADMET-MLR-GAT",
            "error": "empty batch result",
            "binary_toxicity": None,
            "toxicity_type": None,
            "neurotoxicity_type": None,
            "hc50": None,
        }
    return _normalize_runner_row(rows[0], sequence)


def batch_predict_toxicity(
    sequences: list[str],
    smiles_list: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Batch toxicity prediction — delegates to pepadmet_runner."""
    raw = predict_toxicity_batch(sequences, smiles_list)
    return [_normalize_runner_row(r, seq) for r, seq in zip(raw, sequences)]
