#!/usr/bin/env python3
"""A7 (Wave 2) — XGBoost pose surrogate baseline 학습 파이프라인.

목적: Pose(6D rigid-body translation/rotation + pharmacophore contact feature)
     -> ddg_flex 예측 대리모델을 학습하여, 추후 BO exploration 후보의 "순위"
     재현(residual signal)에 사용한다.

★ 절대 원칙 ★
  - 이 모델은 절대 ddG(REU)를 신뢰도 있게 예측하지 않는다. 목적은 "pose 순위
    재현"이며, nstruct=1 단일 도킹 결과라 REU 자체에 통계적 노이즈가 있다.
  - Spearman rho < 0.4 이면 보고서에 "pose 순위 재현 부족"이라고 명시한다
    (임의로 좋게 포장하지 않는다).
  - surrogate 관련 기존 파일(pepadmet_toxicity_v2.py, step08_stability.py 등)은
    이 스크립트에서 절대 import/수정하지 않는다 — 완전히 독립된 신규 모듈이다.

데이터 소스 (Explore 조사로 확정, 총 14,009 유효 레코드 / 534 서열):
  - runs/exp72_analysis/lhs_pose_data/lhs_pose_shard_{00..31}.jsonl  (12,611 valid)
  - runs/exp72_analysis/pose_bo_rigid/bo_summary.jsonl + *_bo_history.jsonl (1,389 valid)
  - runs/exp72_analysis/pose_bo_rigid_proto/*_bo_history.jsonl (9 valid, 선택적)

사용법:
  conda run -n bio-tools python scripts/train_xgboost_pose_surrogate.py [--smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# 상수 (scripts/exp72_pose_bo_rigid.py 와 동일 — 정규화 범위)
# ----------------------------------------------------------------------------
TRANS_RANGE = 5.0   # ±5 A
ROT_RANGE = 30.0    # ±30 deg
NATIVE_SEQ = "AGCKNFFWKTFTSC"  # SST-14 native (Cys3-Cys14 SS bond, FWKT pharmacophore)

LHS_DIR = os.path.join(REPO_ROOT, "runs/exp72_analysis/lhs_pose_data")
BO_DIR = os.path.join(REPO_ROOT, "runs/exp72_analysis/pose_bo_rigid")
BO_PROTO_DIR = os.path.join(REPO_ROOT, "runs/exp72_analysis/pose_bo_rigid_proto")

OUT_MODEL_DIR = os.path.join(REPO_ROOT, "_workspace/models")
OUT_MODEL_PATH = os.path.join(OUT_MODEL_DIR, "xgb_pose_surrogate_v1.joblib")
OUT_REPORT_PATH = os.path.join(REPO_ROOT, "_workspace/A7_XGB_TRAINING_REPORT_2026-08-03.md")
OUT_PRED_PATH = os.path.join(REPO_ROOT, "_workspace/A7_xgb_test_predictions.jsonl")
OUT_FI_PLOT_PATH = os.path.join(REPO_ROOT, "_workspace/A7_xgb_feature_importance.png")

SEED = 42
DDG_ABS_CAP = 300.0  # |ddg_flex| >= 300 은 물리적으로 비정상(클래시 아웃라이어)로 취급, 제외


@dataclass
class PoseRecord:
    sequence: str
    pose_idx: str
    X: list  # 6D [dx, dy, dz, rx, ry, rz]
    ddg_flex: float
    ss_dist: Optional[float]
    fwkt_dist: Optional[float]
    source: str


def hamming_distance(seq: str, ref: str = NATIVE_SEQ) -> int:
    """native 서열과의 Hamming distance (같은 길이 가정, 다르면 길이 페널티 포함)."""
    if len(seq) != len(ref):
        # 길이가 다른 변형(삽입/결실)은 드물지만 방어적으로 처리
        n = min(len(seq), len(ref))
        return sum(a != b for a, b in zip(seq[:n], ref[:n])) + abs(len(seq) - len(ref))
    return sum(a != b for a, b in zip(seq, ref))


# ----------------------------------------------------------------------------
# 데이터 로딩
# ----------------------------------------------------------------------------

def _valid_ddg(v) -> bool:
    return v is not None and abs(v) < DDG_ABS_CAP


def load_lhs_shards(lhs_dir: str = LHS_DIR) -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(lhs_dir, "lhs_pose_shard_*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                if not _valid_ddg(r.get("ddg_flex")):
                    continue
                if not r.get("X") or len(r["X"]) != 6:
                    continue
                records.append(
                    PoseRecord(
                        sequence=r["sequence"],
                        pose_idx=f"lhs{r.get('pose_idx')}",
                        X=r["X"],
                        ddg_flex=float(r["ddg_flex"]),
                        ss_dist=r.get("ss_dist"),
                        fwkt_dist=r.get("fwkt_dist"),
                        source="lhs",
                    )
                )
    return records


def load_bo_history_dir(bo_dir: str, source_label: str) -> list:
    """*_bo_history.jsonl 파일명에서 서열을 추출(파일 자체엔 sequence 필드 없음)."""
    records = []
    for path in sorted(glob.glob(os.path.join(bo_dir, "*_bo_history.jsonl"))):
        seq = os.path.basename(path).replace("_bo_history.jsonl", "")
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not _valid_ddg(r.get("ddg_flex")):
                    continue
                if not r.get("X") or len(r["X"]) != 6:
                    continue
                records.append(
                    PoseRecord(
                        sequence=seq,
                        pose_idx=f"{source_label}{r.get('iter')}",
                        X=r["X"],
                        ddg_flex=float(r["ddg_flex"]),
                        ss_dist=r.get("ss_dist"),
                        fwkt_dist=r.get("fwkt_dist"),
                        source=source_label,
                    )
                )
    return records


def load_all_records() -> list:
    records = []
    records += load_lhs_shards()
    records += load_bo_history_dir(BO_DIR, "bo")
    if os.path.isdir(BO_PROTO_DIR):
        records += load_bo_history_dir(BO_PROTO_DIR, "bo_proto")
    return records


# ----------------------------------------------------------------------------
# Feature 구성
# ----------------------------------------------------------------------------

def build_feature_matrix(records: list):
    """6D pose(raw+normalized) + fwkt_dist(있으면) + native Hamming distance.

    ss_dist 는 현재 확보된 전 데이터에서 100% null (LHS 32 shard 전체 + BO 18
    시퀀스 전체 확인됨) 이라 feature 로 포함하지 않는다(전부 결측이면 트리
    모델 학습에 무의미). 다시 채워지면 이 함수만 손보면 된다.
    """
    n = len(records)
    dx = np.array([r.X[0] for r in records])
    dy = np.array([r.X[1] for r in records])
    dz = np.array([r.X[2] for r in records])
    rx = np.array([r.X[3] for r in records])
    ry = np.array([r.X[4] for r in records])
    rz = np.array([r.X[5] for r in records])

    ndx, ndy, ndz = dx / TRANS_RANGE, dy / TRANS_RANGE, dz / TRANS_RANGE
    nrx, nry, nrz = rx / ROT_RANGE, ry / ROT_RANGE, rz / ROT_RANGE

    fwkt_present = any(r.fwkt_dist is not None for r in records)
    ss_present = any(r.ss_dist is not None for r in records)

    feature_cols = ["dx", "dy", "dz", "rx", "ry", "rz",
                     "ndx", "ndy", "ndz", "nrx", "nry", "nrz"]
    cols = [dx, dy, dz, rx, ry, rz, ndx, ndy, ndz, nrx, nry, nrz]

    if fwkt_present:
        fwkt = np.array([r.fwkt_dist if r.fwkt_dist is not None else np.nan for r in records])
        cols.append(fwkt)
        feature_cols.append("fwkt_dist")
    if ss_present:
        ssd = np.array([r.ss_dist if r.ss_dist is not None else np.nan for r in records])
        cols.append(ssd)
        feature_cols.append("ss_dist")

    hamming = np.array([hamming_distance(r.sequence) for r in records], dtype=float)
    cols.append(hamming)
    feature_cols.append("hamming_to_native")

    X = np.column_stack(cols)
    y = np.array([r.ddg_flex for r in records])
    return X, y, feature_cols


# ----------------------------------------------------------------------------
# 서열 기준 group split (train/val/test 이 서열 단위로 겹치지 않게)
# ----------------------------------------------------------------------------

def group_split_by_sequence(records: list, seed: int = SEED,
                             ratios=(0.8, 0.1, 0.1)):
    sequences = sorted({r.sequence for r in records})
    rng = random.Random(seed)
    rng.shuffle(sequences)
    n = len(sequences)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train_seqs = set(sequences[:n_train])
    val_seqs = set(sequences[n_train:n_train + n_val])
    test_seqs = set(sequences[n_train + n_val:])

    idx_train = [i for i, r in enumerate(records) if r.sequence in train_seqs]
    idx_val = [i for i, r in enumerate(records) if r.sequence in val_seqs]
    idx_test = [i for i, r in enumerate(records) if r.sequence in test_seqs]
    return idx_train, idx_val, idx_test, (train_seqs, val_seqs, test_seqs)


# ----------------------------------------------------------------------------
# 평가
# ----------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    rho, pval = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "spearman_rho": rho, "spearman_p": pval, "r2": r2}


def per_sequence_spearman(records: list, idx: list, y_true: np.ndarray, y_pred: np.ndarray,
                           min_poses: int = 3) -> dict:
    """서열별 pose 순위 재현 Spearman (min_poses 개 미만인 서열은 제외 — 상관계수 정의 불가)."""
    from scipy.stats import spearmanr
    from collections import defaultdict

    by_seq_true = defaultdict(list)
    by_seq_pred = defaultdict(list)
    for local_i, global_i in enumerate(idx):
        seq = records[global_i].sequence
        by_seq_true[seq].append(y_true[local_i])
        by_seq_pred[seq].append(y_pred[local_i])

    rhos = []
    n_seq_used = 0
    n_seq_skipped = 0
    for seq, t in by_seq_true.items():
        p = by_seq_pred[seq]
        if len(t) < min_poses or len(set(t)) < 2 or len(set(p)) < 2:
            n_seq_skipped += 1
            continue
        rho, _ = spearmanr(t, p)
        if rho is not None and not np.isnan(rho):
            rhos.append(rho)
            n_seq_used += 1
        else:
            n_seq_skipped += 1

    return {
        "mean_per_sequence_spearman": float(np.mean(rhos)) if rhos else None,
        "median_per_sequence_spearman": float(np.median(rhos)) if rhos else None,
        "n_sequences_evaluated": n_seq_used,
        "n_sequences_skipped_insufficient_poses": n_seq_skipped,
    }


def intra_sequence_pose_ranking_ceiling(records: list, feature_cols: list, seed: int = SEED,
                                        min_poses: int = 30) -> Optional[dict]:
    """(옵션) pose-내삽 능력 상한 참고치.

    본 train/test 는 '서열 단위' group split 이라 test는 전부 unseen-sequence
    (서열-외삽) 평가다. 하지만 실제 BO 사용 시나리오는 '이미 아는 서열 하나에
    대해 다음 pose 후보 순위를 매기는 것'에 가깝다 — 즉 pose-내삽
    (same-sequence unseen pose) 능력이 실사용과 더 밀접하다.

    이 함수는 완전히 별도의 미니 실험으로, 포즈가 풍부한(>=min_poses) 서열들
    (BO 18개 계열, 각 최대 100 iter)만 모아 그 안에서 pose를 80:20 랜덤 split
    (서열은 양쪽에 섞여도 됨 — 이건 '서열을 아는 상태에서 pose 순위 재현'을
    보는 것이 목적이라 group split을 적용하지 않는다). 메인 리포트 수치와
    혼동하지 않도록 별도 항목으로만 표기한다.
    """
    from collections import defaultdict
    import xgboost as xgb

    by_seq = defaultdict(list)
    for i, r in enumerate(records):
        by_seq[r.sequence].append(i)

    rich_seqs = [s for s, idxs in by_seq.items() if len(idxs) >= min_poses]
    if not rich_seqs:
        return None

    rng = random.Random(seed)
    train_idx, test_idx = [], []
    for s in rich_seqs:
        idxs = by_seq[s][:]
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * 0.2)))
        test_idx += idxs[:n_test]
        train_idx += idxs[n_test:]

    X_all, y_all, _ = build_feature_matrix(records)
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = compute_metrics(y_te, y_pred)
    seq_metrics = per_sequence_spearman(records, test_idx, y_te, y_pred, min_poses=3)
    return {
        "n_rich_sequences": len(rich_seqs),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "overall": metrics,
        "per_sequence": seq_metrics,
    }


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1000 sample 로 first-pass 스모크만 실행")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    import xgboost as xgb
    import joblib

    print(f"[load] LHS shards from {LHS_DIR}")
    records = load_all_records()
    print(f"[load] total valid records: {len(records)} / sequences: {len({r.sequence for r in records})}")

    if args.smoke:
        rng = random.Random(args.seed)
        records = rng.sample(records, min(1000, len(records)))
        print(f"[smoke] subsampled to {len(records)} records")

    idx_train, idx_val, idx_test, (train_seqs, val_seqs, test_seqs) = group_split_by_sequence(
        records, seed=args.seed
    )
    print(f"[split] train={len(idx_train)} recs / {len(train_seqs)} seqs, "
          f"val={len(idx_val)} recs / {len(val_seqs)} seqs, "
          f"test={len(idx_test)} recs / {len(test_seqs)} seqs")

    X_all, y_all, feature_cols = build_feature_matrix(records)
    print(f"[features] {feature_cols}")

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_test, y_test = X_all[idx_test], y_all[idx_test]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=args.seed,
        objective="reg:squarederror",
        early_stopping_rounds=20,
        eval_metric="mae",
    )

    print("[train] fitting XGBRegressor (early_stopping_rounds=20, eval=val)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_iter = getattr(model, "best_iteration", None)
    print(f"[train] done. best_iteration={best_iter}")

    y_pred_test = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred_test)
    test_seq_metrics = per_sequence_spearman(records, idx_test, y_test, y_pred_test)

    y_pred_val = model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_pred_val)

    y_pred_train = model.predict(X_train)
    train_metrics = compute_metrics(y_train, y_pred_train)

    print("[eval] test:", test_metrics)
    print("[eval] per-sequence (test):", test_seq_metrics)

    interp_result = None
    if not args.smoke:
        print("[interp] running intra-sequence pose-ranking ceiling (option)...")
        interp_result = intra_sequence_pose_ranking_ceiling(records, feature_cols, seed=args.seed)
        print("[interp] result:", interp_result)

    # ---- 산출물 저장 ----
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_REPORT_PATH), exist_ok=True)

    if not args.smoke:
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "native_seq": NATIVE_SEQ, "trans_range": TRANS_RANGE,
                     "rot_range": ROT_RANGE}, OUT_MODEL_PATH)
        print(f"[save] model -> {OUT_MODEL_PATH}")

        with open(OUT_PRED_PATH, "w") as fh:
            for local_i, global_i in enumerate(idx_test):
                r = records[global_i]
                fh.write(json.dumps({
                    "sequence": r.sequence,
                    "pose_idx": r.pose_idx,
                    "source": r.source,
                    "X": r.X,
                    "y_true": float(y_test[local_i]),
                    "y_pred": float(y_pred_test[local_i]),
                }) + "\n")
        print(f"[save] test predictions -> {OUT_PRED_PATH}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            importances = model.feature_importances_
            order = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
            ax.set_xlabel("XGBoost feature importance (gain-normalized)")
            ax.set_title("A7 pose surrogate v1 — feature importance")
            fig.tight_layout()
            fig.savefig(OUT_FI_PLOT_PATH, dpi=150)
            plt.close(fig)
            print(f"[save] feature importance plot -> {OUT_FI_PLOT_PATH}")
        except Exception as e:  # matplotlib 미설치 등 — 정직 보고, 조용히 넘기지 않음
            print(f"[warn] feature importance plot 생략: {e}")

        write_report(
            records, feature_cols, train_seqs, val_seqs, test_seqs,
            idx_train, idx_val, idx_test,
            train_metrics, val_metrics, test_metrics, test_seq_metrics,
            model, best_iter, interp_result,
        )
        print(f"[save] report -> {OUT_REPORT_PATH}")
    else:
        print("[smoke] skip save (model/report/predictions) — --smoke 는 first-pass 검증 전용")


def write_report(records, feature_cols, train_seqs, val_seqs, test_seqs,
                  idx_train, idx_val, idx_test,
                  train_metrics, val_metrics, test_metrics, test_seq_metrics,
                  model, best_iter, interp_result):
    honest_verdict = (
        "pose 순위 재현 **충분** (rho >= 0.4)"
        if (test_metrics["spearman_rho"] is not None and test_metrics["spearman_rho"] >= 0.4)
        else "pose 순위 재현 **부족** (rho < 0.4) — BO 통합 전 feature 보강 또는 데이터 확충 필요"
    )

    lines = []
    lines.append("# A7 (Wave 2) — XGBoost Pose Surrogate 학습 리포트\n")
    lines.append(f"- 생성일: 2026-08-03\n")
    lines.append(f"- 스크립트: `scripts/train_xgboost_pose_surrogate.py`\n")
    lines.append(f"- 모델 파일: `_workspace/models/xgb_pose_surrogate_v1.joblib`\n\n")

    lines.append("## 0. 목적 및 정직 원칙\n")
    lines.append(
        "- 본 모델은 **절대 ddG(REU) 예측이 아니다**. 목적은 pose(6D rigid-body "
        "perturbation)의 **순위 재현**이며, 향후 BO exploration 후보 랭킹의 "
        "residual signal로만 사용한다.\n"
    )
    lines.append(
        "- 학습/평가 데이터는 **nstruct=1** (단일 FlexPepDock refine) 결과다. "
        "ddg_flex 자체에 도킹 노이즈가 존재하고 표준편차 추정이 불가하다 — "
        "즉 라벨(y) 자체가 완벽한 ground truth가 아니라는 한계를 감안해야 한다.\n"
    )
    lines.append(f"- **판정**: {honest_verdict}\n\n")

    lines.append("## 1. 데이터\n")
    lines.append(f"- 총 유효 레코드: {len(records)}\n")
    lines.append(f"- 총 서열 수: {len({r.sequence for r in records})}\n")
    from collections import Counter
    src_counts = Counter(r.source for r in records)
    lines.append(f"- 출처별: {dict(src_counts)}\n")
    lines.append(
        f"- Split (서열 단위 group split, 서열이 train/val/test 간 겹치지 않음): "
        f"train {len(idx_train)}건/{len(train_seqs)}서열, "
        f"val {len(idx_val)}건/{len(val_seqs)}서열, "
        f"test {len(idx_test)}건/{len(test_seqs)}서열\n\n"
    )

    lines.append("## 2. Feature\n")
    lines.append(f"- 사용된 feature ({len(feature_cols)}개): `{feature_cols}`\n")
    lines.append(
        "- `ss_dist`는 확보된 전체 데이터(LHS 32 shard + BO 18계열)에서 100% "
        "결측이라 feature에서 제외했다 (전부 null이면 트리 분기에 무의미). "
        "재수집 시 `build_feature_matrix()`가 자동으로 포함하도록 구현됨.\n"
    )
    lines.append(
        "- `hamming_to_native`: native SST-14(`AGCKNFFWKTFTSC`) 대비 서열 "
        "Hamming distance 1개 스칼라로 534개 서열의 카디널리티를 대체 "
        "(스펙 지시대로 원-핫 대신 최소 feature로 시작).\n\n"
    )

    lines.append("## 3. 모델 & 하이퍼파라미터\n")
    lines.append(
        "- `xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, "
        "subsample=0.9, colsample_bytree=0.9, early_stopping_rounds=20, eval_metric='mae')`\n"
    )
    lines.append(f"- best_iteration (early stopping, val 기준): {best_iter}\n\n")

    def fmt_metrics(m):
        return (f"MAE={m['mae']:.3f} REU, RMSE={m['rmse']:.3f} REU, "
                f"Spearman rho={m['spearman_rho']:.3f} (p={m['spearman_p']:.2e}), "
                f"R2={m['r2']:.3f}")

    lines.append("## 4. 결과 — 메인 (서열-외삽 / unseen-sequence)\n")
    lines.append("이 test set의 534개 서열 중 10%는 학습에 전혀 등장하지 않은 **완전히 새로운 서열**이다 — "
                  "즉 이 수치가 \"새 후보 서열에 대해 이 모델이 얼마나 쓸모 있는가\"의 정직한 답이다.\n\n")
    lines.append(f"- Train: {fmt_metrics(train_metrics)}\n")
    lines.append(f"- Val:   {fmt_metrics(val_metrics)}\n")
    lines.append(f"- **Test: {fmt_metrics(test_metrics)}**\n\n")
    lines.append("### 4.1 서열별 pose 순위 재현 (test, 우리 실제 목적)\n")
    lines.append(f"- 평가 대상 서열 수: {test_seq_metrics['n_sequences_evaluated']} "
                  f"(pose 3개 미만 또는 순위 정의 불가 서열 {test_seq_metrics['n_sequences_skipped_insufficient_poses']}개 제외)\n")
    if test_seq_metrics["mean_per_sequence_spearman"] is not None:
        lines.append(f"- 서열별 Spearman rho 평균: {test_seq_metrics['mean_per_sequence_spearman']:.3f}\n")
        lines.append(f"- 서열별 Spearman rho 중앙값: {test_seq_metrics['median_per_sequence_spearman']:.3f}\n\n")
    else:
        lines.append("- test set 내 pose 3개 이상 보유 서열이 없어 서열별 순위 평가 불가 "
                      "(LHS 서열은 대부분 서열당 pose 수가 적음 — 데이터 특성상 한계).\n\n")

    if interp_result is not None:
        lines.append("## 5. (옵션/참고) Pose-내삽 순위 재현 상한 — 별도 미니 실험\n")
        lines.append(
            "**주의**: 아래는 메인 리포트 수치(§4)와 완전히 별개의 실험이다. "
            "메인 평가는 서열 단위 group split이라 test = 100% unseen-sequence다. "
            "하지만 실제 BO 사용 맥락은 **이미 아는 서열 하나**에 대해 다음 pose 후보 순위를 "
            "매기는 것에 가깝다. 그래서 포즈가 풍부한(>=30 pose) BO 계열 서열만 모아 "
            "서열 내 pose를 80:20으로 나눠(서열 자체는 양쪽에 공유) 추가로 학습/평가했다 — "
            "이건 '순위 재현 능력의 낙관적 상한'이지 실사용 모델의 정식 성능이 아니다.\n\n"
        )
        lines.append(f"- 대상 서열 수(pose>=30): {interp_result['n_rich_sequences']}\n")
        lines.append(f"- train pose {interp_result['n_train']}건 / test pose {interp_result['n_test']}건\n")
        lines.append(f"- 전체: {fmt_metrics(interp_result['overall'])}\n")
        sm = interp_result["per_sequence"]
        if sm["mean_per_sequence_spearman"] is not None:
            lines.append(f"- 서열별 Spearman rho 평균: {sm['mean_per_sequence_spearman']:.3f} "
                         f"(평가 서열 {sm['n_sequences_evaluated']}개)\n\n")
        else:
            lines.append("\n")
    else:
        lines.append("## 5. (옵션) Pose-내삽 실험: 생략\n대상 서열(pose>=30) 없음.\n\n")

    lines.append("## 6. Feature Importance\n")
    lines.append("![feature importance](A7_xgb_feature_importance.png)\n\n")

    lines.append("## 7. 다음 단계 (BO 통합) 권장\n")
    if test_metrics["spearman_rho"] >= 0.4:
        lines.append(
            "- 서열-외삽 rho가 0.4 이상이므로, BO acquisition 후보 pre-filter/re-rank의 "
            "약한 prior로 통합 검토 가능. 단 최종 선택은 반드시 실제 FlexPepDock 재도킹으로 확정.\n"
        )
    else:
        lines.append(
            "- 서열-외삽 rho가 0.4 미만이라 **현재 상태로 BO 후보 랭킹에 바로 투입 비권장**. "
            "다음 중 하나 이상 필요: (a) fwkt_dist 외 추가 contact feature 확보(예: "
            "contact_fingerprint.py 5-contact 재적용), (b) 서열 encoding을 Hamming 1D에서 "
            "ESM-2 임베딩 등으로 승격, (c) 데이터 추가 수집(현재 LHS는 서열당 pose 30개 "
            "미만이 대부분이라 서열별 순위 학습 신호가 약함).\n"
        )
    lines.append(
        "- 어느 경우든 이 모델은 **exploration 후보를 좁히는 1차 필터**로만 쓰고, "
        "최종 ddG 판정은 항상 실제 도킹(FlexPepDock)으로 재확인한다 — surrogate가 "
        "실제 물리 계산을 대체하지 않는다.\n"
    )

    with open(OUT_REPORT_PATH, "w") as fh:
        fh.writelines(lines)


if __name__ == "__main__":
    main()
