# FROZEN_COPY_V2 (2026-06-25): pepadmet_toxicity.py 검증 기준선 사본.
# 원본은 pyrosetta_flow/pepadmet_toxicity.py (동결).
# 이 사본에서만 정확도 개선 수정 허용.
"""
pepADMET 독성 모델 추론 래퍼 (v2 — cyclic-aware 통합).

기본 동작 (PEPADMET_CYCLIC_AWARE=1, 기본값):
  ESM2 + 구조 feature 4d 기반 cyclic-aware 용혈 분류기를 우선 사용.
  모델 파일 누락/GPU 에러/임포트 실패 시 기존 pepADMET-MLR-GAT 로 자동 폴백.

환경 변수:
  PEPADMET_CYCLIC_AWARE=0  — cyclic-aware 모델을 건너뛰고 pepADMET만 사용.

cyclic-aware 모델 caveat (필수 고지):
  - ML surrogate (ESM2 640d + struct_proj 32d concat → 이진 분류헤드).
  - SST-14 native 예측값 ≈ 0.46 (threshold 0.5 근처, 마진 얇음).
    → "비용혈 확신"이 아니라 "용혈 아닐 가능성 높음" 수준으로 해석할 것.
  - HemoPI2 학습 데이터: CC-BY-NC-ND — 내부 학술 용도만 허용.
  - 단일 포인트 추정 (분산·신뢰구간 없음).
  - cyclic 구조 플래그(is_cyclic, has_ss_bond)를 서열에서 휴리스틱으로 추정하므로,
    BILN/HELM 없이 서열만 있는 경우 Cys≥2이면 SS bond로 간주 (보수적).

GitHub: ifyoungnet/pepADMET (JCIM 2026, 66, 936-946) — 폴백용.
모델 경로: runs/surrogate_finetune/models/esm2_cyclic_aware_best.pt (567 MB, git 제외)
ESM2: facebook/esm2_t30_150M_UR50D (30-layer, hidden 640)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .pepadmet_runner import predict_toxicity_batch

logger = logging.getLogger(__name__)

# ── cyclic-aware 모델 경로 ────────────────────────────────────────────────────
# 학습 스크립트(runs/surrogate_finetune/esm2_cyclic_aware_finetune.py)와 일치하는 경로
_REPO_ROOT = Path(__file__).parent.parent  # ai4sci-kaeri/
_CYCLIC_MODEL_PATH = (
    _REPO_ROOT / "runs" / "surrogate_finetune" / "models" / "esm2_cyclic_aware_best.pt"
)
_ESM2_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
_MODEL_CACHE = Path("/tmp/hf_cache")
_MAX_LEN = 50
_STRUCT_DIM = 4  # [is_cyclic, has_ss_bond, n_cys_norm, cys_spacing_norm]

# ── 싱글턴: 모델/토크나이저 lazy 로드 ────────────────────────────────────────
_cyclic_model: Any = None   # CyclicAwareESM2Classifier | None
_cyclic_tokenizer: Any = None
_cyclic_device: Any = None
_cyclic_load_failed: bool = False  # 한 번 실패하면 재시도 않음


def _cyclic_aware_enabled() -> bool:
    """환경 변수 PEPADMET_CYCLIC_AWARE=0 이면 False, 그 외 True (기본 on)."""
    return os.environ.get("PEPADMET_CYCLIC_AWARE", "1").strip() not in ("0", "false", "False")


def _compute_struct_features_from_sequence(sequence: str) -> "list[float]":
    """서열에서 cyclic 구조 feature 4d 를 휴리스틱으로 추정한다.

    학습 스크립트(esm2_cyclic_aware_finetune.py::extract_struct_features)의
    Cys 기반 로직과 일관성을 유지한다. BILN/HELM 없이 서열만 있는 경우:
      - Cys >= 2 → is_cyclic=1, has_ss_bond=1 (SST-14 계열 보수적 적용)
      - Cys  < 2 → 0 벡터 (선형 또는 SS 없음)

    이 휴리스틱은 우리 타겟(cyclic Cys3-Cys14 SST-14)에 적합하게 설계되었다.
    """
    seq = sequence.upper()
    n_cys = seq.count("C")
    has_ss = n_cys >= 2  # 보수적: Cys 둘 이상이면 SS bond로 간주
    is_cyclic = has_ss   # cyclic SST-14처럼 SS bond가 고리를 형성

    cys_spacing = 0.0
    if has_ss and len(seq) > 0:
        cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
        if len(cys_positions) >= 2:
            max_span = max(cys_positions) - min(cys_positions)
            cys_spacing = max_span / len(seq)

    return [
        float(is_cyclic),
        float(has_ss),
        min(n_cys, 10) / 10.0,
        cys_spacing,
    ]


def _load_cyclic_model() -> bool:
    """cyclic-aware 모델과 토크나이저를 lazy 로드한다.

    Returns:
        True if successfully loaded, False otherwise.
    """
    global _cyclic_model, _cyclic_tokenizer, _cyclic_device, _cyclic_load_failed

    if _cyclic_load_failed:
        return False
    if _cyclic_model is not None:
        return True

    if not _CYCLIC_MODEL_PATH.exists():
        logger.warning(
            "cyclic-aware 모델 파일 없음: %s — pepADMET 폴백 사용", _CYCLIC_MODEL_PATH
        )
        _cyclic_load_failed = True
        return False

    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, EsmModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── 모델 아키텍처 재구성 (학습 스크립트와 동일) ──────────────────────
        class _CyclicAwareESM2Classifier(nn.Module):
            """ESM2 [CLS] + struct_proj(4d→32d) concat → 이진 분류헤드."""

            def __init__(self, esm2_model_name: str, dropout: float = 0.3):
                super().__init__()
                self.esm2 = EsmModel.from_pretrained(
                    esm2_model_name, cache_dir=str(_MODEL_CACHE)
                )
                hidden_size = self.esm2.config.hidden_size  # 640
                self.struct_proj = nn.Sequential(
                    nn.Linear(_STRUCT_DIM, 32),
                    nn.GELU(),
                    nn.Linear(32, 32),
                )
                fused_size = hidden_size + 32  # 672
                self.classifier = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(fused_size, 256),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 64),
                    nn.GELU(),
                    nn.Dropout(dropout / 2),
                    nn.Linear(64, 1),
                )

            def forward(
                self,
                input_ids: "torch.Tensor",
                attention_mask: "torch.Tensor",
                struct_feat: "torch.Tensor",
            ) -> "torch.Tensor":
                out = self.esm2(input_ids=input_ids, attention_mask=attention_mask)
                cls_emb = out.last_hidden_state[:, 0, :]
                struct_emb = self.struct_proj(struct_feat)
                fused = torch.cat([cls_emb, struct_emb], dim=-1)
                return self.classifier(fused).squeeze(-1)

        tokenizer = AutoTokenizer.from_pretrained(
            _ESM2_MODEL_NAME, cache_dir=str(_MODEL_CACHE)
        )
        model = _CyclicAwareESM2Classifier(_ESM2_MODEL_NAME).to(device)
        state_dict = torch.load(str(_CYCLIC_MODEL_PATH), map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        _cyclic_model = model
        _cyclic_tokenizer = tokenizer
        _cyclic_device = device
        logger.info("cyclic-aware 모델 로드 완료: %s (device=%s)", _CYCLIC_MODEL_PATH, device)
        return True

    except Exception as exc:
        logger.warning("cyclic-aware 모델 로드 실패: %s — pepADMET 폴백 사용", exc)
        _cyclic_load_failed = True
        return False


def _predict_hemolysis_cyclic_aware(sequence: str) -> Optional[float]:
    """cyclic-aware ESM2 모델로 용혈 확률을 추정한다.

    Args:
        sequence: 아미노산 서열 (1문자 코드).

    Returns:
        0.0 ~ 1.0 용혈 확률, 모델 미사용 시 None.

    Note (caveat):
        cyclic-aware ML proxy.
        SST-14 native ≈ 0.46 (threshold 0.5 근처, 마진 얇음 — 확신 금지).
        HemoPI2 학습 데이터: CC-BY-NC-ND (내부 학술만).
        단일 포인트 추정 (분산 없음).
    """
    if not _cyclic_aware_enabled():
        return None
    if not _load_cyclic_model():
        return None

    try:
        import torch

        seq = sequence.upper()
        struct_feats = _compute_struct_features_from_sequence(seq)

        spaced = " ".join(list(seq[:_MAX_LEN]))
        enc = _cyclic_tokenizer(
            spaced,
            max_length=_MAX_LEN + 2,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        struct_tensor = torch.tensor(
            [struct_feats], dtype=torch.float
        ).to(_cyclic_device)

        with torch.no_grad():
            logit = _cyclic_model(
                enc["input_ids"].to(_cyclic_device),
                enc["attention_mask"].to(_cyclic_device),
                struct_tensor,
            )
        prob: float = torch.sigmoid(logit).item()
        return float(prob)

    except Exception as exc:
        logger.warning("cyclic-aware 추론 실패 (서열=%s): %s", sequence[:20], exc)
        return None


# ── 기존 인터페이스 호환 레이어 ───────────────────────────────────────────────

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


def _attach_cyclic_aware(result: dict[str, Any], sequence: str) -> dict[str, Any]:
    """result dict에 cyclic-aware 용혈 확률 필드를 추가한다.

    추가 필드:
      hemolysis_prob_cyclic_aware  : float | None — 용혈 확률 (0~1)
      hemolysis_model_used         : str   — 사용된 모델 이름
      hemolysis_cyclic_struct_feat : list[float] | None — [is_cyclic, has_ss_bond, n_cys_norm, cys_spacing_norm]
      hemolysis_caveat             : str   — 필수 고지

    caveat 내용:
      cyclic-aware ML proxy. SST-14 native ≈ 0.46 (threshold 0.5 근처, 마진 얇음).
      HemoPI2 데이터 CC-BY-NC-ND (내부 학술). 단일 포인트 추정 (분산 없음).
    """
    CAVEAT = (
        "cyclic-aware ML proxy (ESM2+struct_proj). "
        "SST-14 native ≈ 0.46 — threshold 0.5 근처, 마진 얇음(확신 금지). "
        "HemoPI2 학습 데이터: CC-BY-NC-ND(내부 학술 전용). "
        "단일 포인트 추정(분산·신뢰구간 없음)."
    )
    prob = _predict_hemolysis_cyclic_aware(sequence)
    struct_feats = _compute_struct_features_from_sequence(sequence) if prob is not None else None

    result["hemolysis_prob_cyclic_aware"] = prob
    result["hemolysis_model_used"] = (
        "esm2_cyclic_aware (esm2_cyclic_aware_best.pt)"
        if prob is not None
        else "pepADMET-fallback (cyclic-aware unavailable)"
    )
    result["hemolysis_cyclic_struct_feat"] = struct_feats
    result["hemolysis_caveat"] = CAVEAT
    return result


def predict_toxicity(
    sequence: str,
    smiles: Optional[str] = None,
) -> dict[str, Any]:
    """Predict toxicity via pepadmet_runner + cyclic-aware hemolysis overlay.

    기존 시그니처 유지. cyclic-aware 모델이 가용하면 hemolysis_prob_cyclic_aware
    필드를 추가하고, 폴백 시에는 None으로 채워 하위 호환을 보장한다.
    """
    smiles_arg: Optional[list[str]]
    if smiles is None:
        smiles_arg = None
    else:
        smiles_arg = [smiles]
    rows = predict_toxicity_batch([sequence], smiles_arg)
    if not rows:
        base: dict[str, Any] = {
            "available": False,
            "model": "pepADMET-MLR-GAT",
            "error": "empty batch result",
            "binary_toxicity": None,
            "toxicity_type": None,
            "neurotoxicity_type": None,
            "hc50": None,
        }
        return _attach_cyclic_aware(base, sequence)
    result = _normalize_runner_row(rows[0], sequence)
    return _attach_cyclic_aware(result, sequence)


def batch_predict_toxicity(
    sequences: list[str],
    smiles_list: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Batch toxicity prediction — delegates to pepadmet_runner + cyclic-aware overlay."""
    raw = predict_toxicity_batch(sequences, smiles_list)
    results = [_normalize_runner_row(r, seq) for r, seq in zip(raw, sequences)]
    return [_attach_cyclic_aware(res, seq) for res, seq in zip(results, sequences)]
