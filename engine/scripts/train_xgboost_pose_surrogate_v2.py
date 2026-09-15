#!/usr/bin/env python3
"""A13 (Wave 3) — XGBoost pose surrogate v2 (pharmacophore 5-contact 추가).

목적: A7(v1) baseline(14 feature: 6D pose raw+normalized + fwkt_dist + hamming,
     Test Spearman rho=0.297)에 pharmacophore 5-contact feature를 추가하여
     pose 순위 재현력(rho)이 임계 0.4를 넘는지 검증한다.

★ 절대 원칙 ★ (v1과 동일 — scripts/train_xgboost_pose_surrogate.py 참조)
  - 이 모델은 절대 ddG(REU)를 신뢰도 있게 예측하지 않는다. 목적은 "pose 순위
    재현"이다.
  - Spearman rho < 0.4 이면 "pose 순위 재현 부족"이라고 정직하게 명시한다.
  - v2가 v1보다 나빠도 결과를 그대로 보고한다 (임의로 좋게 포장하지 않는다).
  - surrogate 관련 기존 파일(pepadmet_toxicity_v2.py, step08_stability.py 등)은
    이 스크립트에서 절대 import/수정하지 않는다.

★★ 중요 — Feature 설계 변경 사유 (반드시 읽을 것) ★★
  최초 지시는 `pyrosetta_flow/contact_fingerprint.py::analyze_contact_fingerprint()`
  (수용체 잔기번호 Asp122/Gln126/Phe272/His273/Asn276 특이적 salt-bridge/H-bond
  fingerprint)를 그대로 재사용하는 것이었다. 그러나 실제 15,990+1,800개 pose PDB에
  대해 이 함수를 직접 호출해 검증한 결과, **무작위 표본 전부(10/10, 그리고 본 스크립트의
  진단 서브샘플에서도 재확인됨)에서 `reference_numbering_ok=False`,
  `pharmacophore_contact_intact=None`** 이었다 — 즉 salt-bridge/H-bond 4개 raw
  거리값이 전량 결측(all-NaN)이라 ML feature로 전혀 쓸모가 없다.

  원인: 이 pose 데이터셋의 수용체 PDB(chain B, 1-472, Boltz-2 예측 구조 유래,
  `data/somatostatin_receptor/SSTR2_SST14_complex_boltz_*.pdb` 계열)는 UniProt
  canonical 잔기번호 체계를 쓰지 않는다(122번=ILE, 126번=VAL, 272번=ASN — 기대값
  ASP/GLN/PHE와 불일치). 이는 이미 프로젝트 내에서 알려진 한계로,
  `scripts/exp72_structure_check.py:63-68`에 "수용체 넘버링에 의존하지 않는 범용
  최소거리 접촉 판정"으로 대체 사용한다는 주석이 명시되어 있고, v1의 `fwkt_dist`
  feature 자체도 이미 이 방식(pos7-10 CA vs 전체 수용체 CA 최소거리, 원본 계산은
  `scripts/exp72_pose_bo_rigid.py:129-147`)으로 만들어졌다.

  따라서 본 스크립트는 **동일한 원칙(수용체 잔기번호 비의존, CA-CA 최소거리)을
  5개 개별 pharmacophore 위치(F6,F7,W8,K9,T10)로 세분화**하여 5-contact feature를
  구성한다:
    - f6_ca_mindist, f7_ca_mindist, w8_ca_mindist, k9_ca_mindist, t10_ca_mindist
  이 중 F6은 v1의 fwkt_dist(pos7-10 집계값)에는 전혀 포함되지 않았던 신규 정보다.
  literal salt-bridge/H-bond fingerprint(analyze_contact_fingerprint)는 본
  스크립트에서도 진단(diagnostic) 목적으로 서브샘플에 한해 호출하여 결측률을
  실측·보고하지만, 전량 결측이므로 학습 feature에는 포함하지 않는다.

데이터 소스 (v1과 동일):
  - runs/exp72_analysis/lhs_pose_data/lhs_pose_shard_{00..31}.jsonl
  - runs/exp72_analysis/pose_bo_rigid/bo_summary.jsonl + *_bo_history.jsonl
  - runs/exp72_analysis/pose_bo_rigid_proto/*_bo_history.jsonl

사용법:
  conda run -n bio-tools python scripts/train_xgboost_pose_surrogate_v2.py [--smoke]
  conda run -n bio-tools python scripts/train_xgboost_pose_surrogate_v2.py --workers 8
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ----------------------------------------------------------------------------
# v1 모듈 동적 import (scripts/ 디렉터리가 site-packages의 동명 패키지와 충돌하는
# 환경이 있어(예: bio-tools conda env) `from scripts.xxx import`가 실패할 수 있다.
# 파일 경로 기반 importlib으로 우회한다 — v1 파일은 절대 수정하지 않는다.)
# ----------------------------------------------------------------------------
_V1_PATH = os.path.join(REPO_ROOT, "scripts", "train_xgboost_pose_surrogate.py")
_v1_spec = importlib.util.spec_from_file_location("xgb_pose_surrogate_v1", _V1_PATH)
v1 = importlib.util.module_from_spec(_v1_spec)
sys.modules[_v1_spec.name] = v1
_v1_spec.loader.exec_module(v1)

from pyrosetta_flow.contact_fingerprint import (  # noqa: E402
    _parse_pdb_atoms,
    _get_chains,
    _detect_peptide_chain,
    _dist,
    analyze_contact_fingerprint,
)

# ----------------------------------------------------------------------------
# 상수 (v1과 동일 데이터 경로/정규화 범위 재사용)
# ----------------------------------------------------------------------------
NATIVE_SEQ = v1.NATIVE_SEQ
TRANS_RANGE = v1.TRANS_RANGE
ROT_RANGE = v1.ROT_RANGE
LHS_DIR = v1.LHS_DIR
BO_DIR = v1.BO_DIR
BO_PROTO_DIR = v1.BO_PROTO_DIR
SEED = v1.SEED

OUT_MODEL_DIR = os.path.join(REPO_ROOT, "_workspace/models")
OUT_MODEL_PATH = os.path.join(OUT_MODEL_DIR, "xgb_pose_surrogate_v2.joblib")
OUT_REPORT_PATH = os.path.join(REPO_ROOT, "_workspace/A13_XGB_V2_TRAINING_REPORT_2026-08-03.md")
OUT_PRED_PATH = os.path.join(REPO_ROOT, "_workspace/A13_xgb_v2_test_predictions.jsonl")
OUT_FI_PLOT_PATH = os.path.join(REPO_ROOT, "_workspace/A13_xgb_v2_feature_importance.png")
CACHE_PATH = os.path.join(OUT_MODEL_DIR, "pose_pharmacophore_cache.jsonl")
V1_REPORT_PATH = os.path.join(REPO_ROOT, "_workspace/A7_XGB_TRAINING_REPORT_2026-08-03.md")

# pharmacophore 5-contact 위치 (1-indexed, SST-14 F6,F7,W8,K9,T10)
_CONTACT_POSITIONS_1IDX = [6, 7, 8, 9, 10]
_CONTACT_FEATURE_NAMES = [
    "f6_ca_mindist", "f7_ca_mindist", "w8_ca_mindist", "k9_ca_mindist", "t10_ca_mindist",
]
_CONTACT_DIST_CAP = 50.0  # 물리적으로 무의미한 극단값(unbound/파손 pose) 방어 캡 (A)


@dataclass
class PoseRecordV2:
    sequence: str
    pose_idx: str
    X: list
    ddg_flex: float
    ss_dist: Optional[float]
    fwkt_dist: Optional[float]
    source: str
    pdb_path: Optional[str]


# ----------------------------------------------------------------------------
# 데이터 로딩 (v1 로더 + pdb_path 해석 추가)
# ----------------------------------------------------------------------------

_LHS_FILENAME_RE = re.compile(r"^(?P<seq>[A-Za-z]+)_lhs(?P<idx>\d+)_refined\.pdb$")


def build_lhs_pdb_index(lhs_dir: str = LHS_DIR) -> dict:
    """LHS 디렉터리 파일명을 스캔해 (sequence, pose_idx) -> 절대경로 인덱스 구축.

    zero-padding 폭을 가정하지 않기 위해 실제 파일 목록에서 역산한다.
    """
    index: dict = {}
    if not os.path.isdir(lhs_dir):
        return index
    for fn in os.listdir(lhs_dir):
        m = _LHS_FILENAME_RE.match(fn)
        if not m:
            continue
        key = (m.group("seq"), int(m.group("idx")))
        index[key] = os.path.join(lhs_dir, fn)
    return index


def load_lhs_shards_v2(lhs_dir: str = LHS_DIR) -> list:
    index = build_lhs_pdb_index(lhs_dir)
    records = []
    n_missing = 0
    for path in sorted(glob.glob(os.path.join(lhs_dir, "lhs_pose_shard_*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("status") != "ok":
                    continue
                if not v1._valid_ddg(r.get("ddg_flex")):
                    continue
                if not r.get("X") or len(r["X"]) != 6:
                    continue
                idx = r.get("pose_idx")
                pdb_path = index.get((r["sequence"], idx))
                if pdb_path is None:
                    n_missing += 1
                records.append(PoseRecordV2(
                    sequence=r["sequence"],
                    pose_idx=f"lhs{idx}",
                    X=r["X"],
                    ddg_flex=float(r["ddg_flex"]),
                    ss_dist=r.get("ss_dist"),
                    fwkt_dist=r.get("fwkt_dist"),
                    source="lhs",
                    pdb_path=pdb_path,
                ))
    if n_missing:
        print(f"[load][lhs] WARNING: {n_missing}건 pdb_path 해석 실패 (pharmacophore feature NaN 처리됨)")
    return records


def load_bo_history_dir_v2(bo_dir: str, source_label: str) -> list:
    """*_bo_history.jsonl 은 자체에 refined_pdb 절대경로 필드를 가지므로 그대로 사용."""
    records = []
    n_missing = 0
    for path in sorted(glob.glob(os.path.join(bo_dir, "*_bo_history.jsonl"))):
        seq = os.path.basename(path).replace("_bo_history.jsonl", "")
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if not v1._valid_ddg(r.get("ddg_flex")):
                    continue
                if not r.get("X") or len(r["X"]) != 6:
                    continue
                pdb_path = r.get("refined_pdb")
                if pdb_path and not os.path.isfile(pdb_path):
                    pdb_path = None
                if pdb_path is None:
                    n_missing += 1
                records.append(PoseRecordV2(
                    sequence=seq,
                    pose_idx=f"{source_label}{r.get('iter')}",
                    X=r["X"],
                    ddg_flex=float(r["ddg_flex"]),
                    ss_dist=r.get("ss_dist"),
                    fwkt_dist=r.get("fwkt_dist"),
                    source=source_label,
                    pdb_path=pdb_path,
                ))
    if n_missing:
        print(f"[load][{source_label}] WARNING: {n_missing}건 pdb_path 해석 실패")
    return records


def load_all_records_v2() -> list:
    records = []
    records += load_lhs_shards_v2()
    records += load_bo_history_dir_v2(BO_DIR, "bo")
    if os.path.isdir(BO_PROTO_DIR):
        records += load_bo_history_dir_v2(BO_PROTO_DIR, "bo_proto")
    return records


# ----------------------------------------------------------------------------
# Pharmacophore 5-contact 계산 (수용체 잔기번호 비의존 — 위 사유 참조)
# ----------------------------------------------------------------------------

def compute_pharmacophore_contacts(pdb_path: str, peptide_seq: str) -> dict:
    """peptide 위치 F6/F7/W8/K9/T10 각각의 CA(또는 CB) — 수용체 전체 CA 최소거리(Å).

    contact_fingerprint.py 의 salt-bridge/H-bond 특이적 fingerprint와 달리 수용체
    잔기번호에 의존하지 않는다 (scripts/exp72_pose_bo_rigid.py의 fwkt_dist와 동일 원리를
    5개 개별 위치로 세분화).
    """
    out = {name: None for name in _CONTACT_FEATURE_NAMES}
    out["ok"] = False
    out["error"] = None

    atoms = _parse_pdb_atoms(pdb_path)
    if not atoms:
        out["error"] = "pdb_parse_empty"
        return out

    chains = _get_chains(atoms)
    if len(chains) < 2:
        out["error"] = "chains<2"
        return out

    pep_chain = _detect_peptide_chain(atoms, pep_len=len(peptide_seq))
    peptide_atoms = [a for a in atoms if a["chain"] == pep_chain]
    receptor_atoms = [a for a in atoms if a["chain"] != pep_chain]
    if not peptide_atoms or not receptor_atoms:
        out["error"] = "missing_chain_atoms"
        return out

    receptor_ca = [a for a in receptor_atoms if a["name"] == "CA"]
    if not receptor_ca:
        out["error"] = "no_receptor_ca"
        return out

    pep_resseqs = sorted(set(a["resseq"] for a in peptide_atoms))

    def pep_pos_to_resseq(pos_1idx: int) -> Optional[int]:
        idx = pos_1idx - 1
        if 0 <= idx < len(pep_resseqs):
            return pep_resseqs[idx]
        return None

    any_computed = False
    for pos, fname in zip(_CONTACT_POSITIONS_1IDX, _CONTACT_FEATURE_NAMES):
        resseq = pep_pos_to_resseq(pos)
        if resseq is None:
            continue
        pep_ca = [a for a in peptide_atoms if a["resseq"] == resseq and a["name"] == "CA"]
        if not pep_ca:
            pep_ca = [a for a in peptide_atoms if a["resseq"] == resseq and a["name"] == "CB"]
        if not pep_ca:
            continue
        d = min(_dist(pa, ra) for pa in pep_ca for ra in receptor_ca)
        out[fname] = round(min(d, _CONTACT_DIST_CAP), 3)
        any_computed = True

    out["ok"] = any_computed
    if not any_computed:
        out["error"] = "no_pharmacophore_position_resolved"
    return out


def _compute_worker(args):
    pdb_path, seq = args
    try:
        res = compute_pharmacophore_contacts(pdb_path, seq)
    except Exception as e:  # 방어적 — 개별 pdb 실패가 전체를 죽이지 않게
        res = {name: None for name in _CONTACT_FEATURE_NAMES}
        res["ok"] = False
        res["error"] = f"{type(e).__name__}: {e}"
    res["pdb_path"] = pdb_path
    return res


def load_cache(path: str = CACHE_PATH) -> dict:
    cache = {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "pdb_path" in r:
                    cache[r["pdb_path"]] = r
    return cache


def save_cache(cache: dict, path: str = CACHE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for v in cache.values():
            fh.write(json.dumps(v) + "\n")
    os.replace(tmp, path)


def get_pharmacophore_cache(records: list, workers: int = 1, cache_path: str = CACHE_PATH) -> dict:
    """cache miss 만 신규 계산 (Cache 도입, 병렬화 옵션)."""
    cache = load_cache(cache_path)
    seq_by_path = {}
    for r in records:
        if r.pdb_path:
            seq_by_path[r.pdb_path] = r.sequence

    to_compute = sorted(p for p in seq_by_path if p not in cache)
    print(f"[pharmacophore] cache hit {len(seq_by_path) - len(to_compute)} / "
          f"신규 계산 {len(to_compute)} (총 unique pdb {len(seq_by_path)})")

    if not to_compute:
        return cache

    jobs = [(p, seq_by_path[p]) for p in to_compute]
    n_done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_compute_worker, job): job[0] for job in jobs}
            for fut in as_completed(futs):
                res = fut.result()
                cache[res["pdb_path"]] = res
                n_done += 1
                if n_done % 2000 == 0:
                    print(f"[pharmacophore] {n_done}/{len(to_compute)} 계산 완료 (workers={workers})")
    else:
        for job in jobs:
            res = _compute_worker(job)
            cache[res["pdb_path"]] = res
            n_done += 1
            if n_done % 2000 == 0:
                print(f"[pharmacophore] {n_done}/{len(to_compute)} 계산 완료 (단일 프로세스)")

    save_cache(cache, cache_path)
    print(f"[pharmacophore] cache 저장 -> {cache_path} (총 {len(cache)}건)")
    return cache


def diagnose_literal_pharmacophore(records: list, sample_size: int = 200, seed: int = SEED) -> dict:
    """analyze_contact_fingerprint() 문자 그대로 재사용 시도 — 결측률 실측(진단 전용).

    본 함수의 결과는 학습 feature로 사용되지 않는다. 왜 5-contact를 수용체
    잔기번호 비의존 방식으로 재설계했는지에 대한 실증 근거를 리포트에 남기기 위함.
    """
    resolvable = [r for r in records if r.pdb_path]
    rng = random.Random(seed)
    sample = rng.sample(resolvable, min(sample_size, len(resolvable)))
    n_ref_ok = 0
    n_intact_true = 0
    n_intact_false = 0
    n_intact_none = 0
    for r in sample:
        try:
            res = analyze_contact_fingerprint(r.pdb_path, peptide_seq=r.sequence)
        except Exception:
            continue
        cf = res.get("contact_fingerprint", {})
        if cf.get("reference_numbering_ok"):
            n_ref_ok += 1
        intact = res.get("pharmacophore_contact_intact")
        if intact is True:
            n_intact_true += 1
        elif intact is False:
            n_intact_false += 1
        else:
            n_intact_none += 1
    return {
        "n_sampled": len(sample),
        "n_reference_numbering_ok": n_ref_ok,
        "n_intact_true": n_intact_true,
        "n_intact_false": n_intact_false,
        "n_intact_none_na": n_intact_none,
    }


# ----------------------------------------------------------------------------
# Feature 구성 (v1 14개 + pharmacophore 5-contact)
# ----------------------------------------------------------------------------

def build_feature_matrix_v2(records: list, cache: dict):
    # v1.build_feature_matrix 는 r.X/r.fwkt_dist/r.ss_dist/r.sequence/r.ddg_flex 속성만
    # 사용하는 duck-typing 구조라 PoseRecordV2 를 그대로 넣어도 동작한다 (동일 필드명).
    X_base, y, feature_cols = v1.build_feature_matrix(records)
    cols = [X_base[:, i] for i in range(X_base.shape[1])]
    feature_cols = list(feature_cols)

    for name in _CONTACT_FEATURE_NAMES:
        vals = np.full(len(records), np.nan)
        for i, r in enumerate(records):
            entry = cache.get(r.pdb_path) if r.pdb_path else None
            if entry and entry.get("ok") and entry.get(name) is not None:
                vals[i] = entry[name]
        cols.append(vals)
        feature_cols.append(name)

    X = np.column_stack(cols)
    return X, y, feature_cols


def intra_sequence_pose_ranking_ceiling_v2(records: list, cache: dict, seed: int = SEED,
                                            min_poses: int = 30) -> Optional[dict]:
    """v1과 동일한 pose-내삽 상한 참고 실험 (v2 feature 사용 버전)."""
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

    X_all, y_all, _ = build_feature_matrix_v2(records, cache)
    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
    X_te, y_te = X_all[test_idx], y_all[test_idx]

    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.9, colsample_bytree=0.9, random_state=seed,
        objective="reg:squarederror",
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    metrics = v1.compute_metrics(y_te, y_pred)
    seq_metrics = v1.per_sequence_spearman(records, test_idx, y_te, y_pred, min_poses=3)
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
    parser.add_argument("--workers", type=int, default=1, help="pharmacophore 5-contact 계산 병렬 프로세스 수")
    parser.add_argument("--diag-sample", type=int, default=200,
                        help="analyze_contact_fingerprint() literal 결측률 진단 서브샘플 크기")
    args = parser.parse_args()

    import xgboost as xgb
    import joblib

    print(f"[load] LHS shards from {LHS_DIR}")
    records = load_all_records_v2()
    print(f"[load] total valid records: {len(records)} / sequences: {len({r.sequence for r in records})}")
    n_pdb_resolved = sum(1 for r in records if r.pdb_path)
    print(f"[load] pdb_path 해석됨: {n_pdb_resolved}/{len(records)} "
          f"({n_pdb_resolved / len(records) * 100:.1f}%)")

    if args.smoke:
        rng = random.Random(args.seed)
        records = rng.sample(records, min(1000, len(records)))
        print(f"[smoke] subsampled to {len(records)} records")

    print(f"[diag] analyze_contact_fingerprint() literal 재사용 결측률 진단 "
          f"(sample={args.diag_sample})...")
    diag = diagnose_literal_pharmacophore(records, sample_size=args.diag_sample, seed=args.seed)
    print(f"[diag] {diag}")

    cache = get_pharmacophore_cache(records, workers=args.workers)

    idx_train, idx_val, idx_test, (train_seqs, val_seqs, test_seqs) = v1.group_split_by_sequence(
        records, seed=args.seed
    )
    print(f"[split] train={len(idx_train)} recs / {len(train_seqs)} seqs, "
          f"val={len(idx_val)} recs / {len(val_seqs)} seqs, "
          f"test={len(idx_test)} recs / {len(test_seqs)} seqs")

    X_all, y_all, feature_cols = build_feature_matrix_v2(records, cache)
    print(f"[features] {feature_cols}")

    # pharmacophore feature 결측률(학습 데이터 전체 기준) — 리포트용 실측치
    contact_coverage = {}
    for name in _CONTACT_FEATURE_NAMES:
        j = feature_cols.index(name)
        col = X_all[:, j]
        contact_coverage[name] = float(np.mean(~np.isnan(col)))
    print(f"[features] pharmacophore 5-contact 결측 아닌 비율: {contact_coverage}")

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
    test_metrics = v1.compute_metrics(y_test, y_pred_test)
    test_seq_metrics = v1.per_sequence_spearman(records, idx_test, y_test, y_pred_test)

    y_pred_val = model.predict(X_val)
    val_metrics = v1.compute_metrics(y_val, y_pred_val)

    y_pred_train = model.predict(X_train)
    train_metrics = v1.compute_metrics(y_train, y_pred_train)

    print("[eval] test:", test_metrics)
    print("[eval] per-sequence (test):", test_seq_metrics)

    interp_result = None
    if not args.smoke:
        print("[interp] running intra-sequence pose-ranking ceiling (option)...")
        interp_result = intra_sequence_pose_ranking_ceiling_v2(records, cache, seed=args.seed)
        print("[interp] result:", interp_result)

    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_REPORT_PATH), exist_ok=True)

    if not args.smoke:
        joblib.dump({"model": model, "feature_cols": feature_cols,
                     "native_seq": NATIVE_SEQ, "trans_range": TRANS_RANGE,
                     "rot_range": ROT_RANGE, "contact_feature_names": _CONTACT_FEATURE_NAMES},
                    OUT_MODEL_PATH)
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
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
            ax.set_xlabel("XGBoost feature importance (gain-normalized)")
            ax.set_title("A13 pose surrogate v2 — feature importance")
            fig.tight_layout()
            fig.savefig(OUT_FI_PLOT_PATH, dpi=150)
            plt.close(fig)
            print(f"[save] feature importance plot -> {OUT_FI_PLOT_PATH}")
        except Exception as e:
            print(f"[warn] feature importance plot 생략: {e}")

        write_report(
            records, feature_cols, train_seqs, val_seqs, test_seqs,
            idx_train, idx_val, idx_test,
            train_metrics, val_metrics, test_metrics, test_seq_metrics,
            model, best_iter, interp_result, diag, contact_coverage, n_pdb_resolved,
        )
        print(f"[save] report -> {OUT_REPORT_PATH}")
    else:
        print("[smoke] skip save (model/report/predictions) — --smoke 는 first-pass 검증 전용")


def _load_v1_test_rho() -> Optional[str]:
    """v1 리포트(A7)에서 Test Spearman rho 문자열을 그대로 읽어와 비교용으로 인용.

    수치를 손으로 다시 타이핑해 환각을 일으키지 않기 위해 원본 파일에서 정규식으로
    추출한다.
    """
    if not os.path.exists(V1_REPORT_PATH):
        return None
    with open(V1_REPORT_PATH) as fh:
        text = fh.read()
    m = re.search(r"\*\*Test: (.*?)\*\*", text)
    return m.group(1) if m else None


def write_report(records, feature_cols, train_seqs, val_seqs, test_seqs,
                  idx_train, idx_val, idx_test,
                  train_metrics, val_metrics, test_metrics, test_seq_metrics,
                  model, best_iter, interp_result, diag, contact_coverage, n_pdb_resolved):
    honest_verdict = (
        "pose 순위 재현 **충분** (rho >= 0.4)"
        if (test_metrics["spearman_rho"] is not None and test_metrics["spearman_rho"] >= 0.4)
        else "pose 순위 재현 **여전히 부족** (rho < 0.4)"
    )
    v1_test_line = _load_v1_test_rho()

    lines = []
    lines.append("# A13 (Wave 3) — XGBoost Pose Surrogate v2 학습 리포트 (pharmacophore 5-contact)\n")
    lines.append("- 생성일: 2026-08-03\n")
    lines.append("- 스크립트: `scripts/train_xgboost_pose_surrogate_v2.py`\n")
    lines.append("- 모델 파일: `_workspace/models/xgb_pose_surrogate_v2.joblib`\n")
    lines.append("- Cache 파일: `_workspace/models/pose_pharmacophore_cache.jsonl`\n\n")

    lines.append("## 0. 목적 및 정직 원칙\n")
    lines.append(
        "- v1(A7) baseline: 6D pose(raw+normalized) + fwkt_dist + hamming_to_native, "
        "14 feature, Test Spearman rho=0.297 (임계 0.4 미달).\n"
    )
    lines.append(
        "- Phase 1(SYSTEM_RELIMITS N5) 판정: 개선 우선순위 pharmacophore contact > "
        "pose 수 확충 > ESM-2 임베딩. 본 실험은 그 1순위(pharmacophore contact 추가)를 검증한다.\n"
    )
    lines.append(f"- **판정**: {honest_verdict}\n\n")

    lines.append("## 1. ★ Feature 설계 변경 사유 (실측 근거) ★\n")
    lines.append(
        "최초 지시는 `pyrosetta_flow/contact_fingerprint.py::analyze_contact_fingerprint()` "
        "(수용체 잔기번호 Asp122/Gln126/Phe272 특이적 salt-bridge/H-bond fingerprint)의 "
        "재사용이었다. 그러나 본 스크립트가 학습 직전 진단 서브샘플에서 이 함수를 "
        "**문자 그대로** 호출해 실측한 결과:\n\n"
    )
    lines.append(
        f"- 서브샘플 크기: {diag['n_sampled']}건\n"
        f"- `reference_numbering_ok=True` (수용체 잔기번호 체계가 기대값과 일치): "
        f"{diag['n_reference_numbering_ok']}/{diag['n_sampled']}\n"
        f"- `pharmacophore_contact_intact`: True={diag['n_intact_true']}, "
        f"False={diag['n_intact_false']}, None(측정불가/N-A)={diag['n_intact_none_na']}\n\n"
    )
    lines.append(
        "즉 이 pose 데이터셋(수용체 PDB는 `data/somatostatin_receptor/"
        "SSTR2_SST14_complex_boltz_*.pdb` 계열, Boltz-2 예측 구조 유래 chain B 1-472 넘버링)은 "
        "UniProt canonical 잔기번호 체계를 쓰지 않는다 — 122번=ILE, 126번=VAL, 272번=ASN으로, "
        "기대값 ASP/GLN/PHE와 불일치한다. 이는 이미 알려진 한계로 "
        "`scripts/exp72_structure_check.py:63-68` 주석에도 명시되어 있고, v1의 `fwkt_dist` "
        "자체도 이미 수용체 잔기번호 비의존 방식(pos7-10 CA vs 전체 수용체 CA 최소거리, "
        "원본은 `scripts/exp72_pose_bo_rigid.py:129-147`)으로 계산된 값이다.\n\n"
    )
    lines.append(
        "**따라서 본 v2는 동일 원칙(수용체 잔기번호 비의존, CA-CA 최소거리)을 5개 개별 "
        "pharmacophore 위치(F6,F7,W8,K9,T10)로 세분화한 feature를 사용한다** — literal "
        "salt-bridge/H-bond fingerprint를 그대로 쓰면 4개 raw 거리값이 전량 결측(all-NaN)이라 "
        "트리 분기에 전혀 기여하지 못하기 때문이다 (이 사실 자체를 감추지 않고 위와 같이 실측·명시함).\n\n"
    )

    lines.append("## 2. 데이터\n")
    lines.append(f"- 총 유효 레코드: {len(records)}\n")
    lines.append(f"- 총 서열 수: {len({r.sequence for r in records})}\n")
    from collections import Counter
    src_counts = Counter(r.source for r in records)
    lines.append(f"- 출처별: {dict(src_counts)}\n")
    lines.append(f"- pdb_path 해석 성공: {n_pdb_resolved}/{len(records)} "
                  f"({n_pdb_resolved / len(records) * 100:.1f}%)\n")
    lines.append(
        f"- Split (서열 단위 group split, v1과 동일 seed): "
        f"train {len(idx_train)}건/{len(train_seqs)}서열, "
        f"val {len(idx_val)}건/{len(val_seqs)}서열, "
        f"test {len(idx_test)}건/{len(test_seqs)}서열\n\n"
    )

    lines.append("## 3. Feature\n")
    lines.append(f"- 사용된 feature ({len(feature_cols)}개): `{feature_cols}`\n")
    lines.append(
        "- 신규 5개(pharmacophore 5-contact, Å 단위 raw 거리 — XGBoost는 트리 기반이라 "
        "scale-invariant하므로 별도 정규화 없이 raw 값을 사용. 극단값(unbound/파손 pose)만 "
        f"{_CONTACT_DIST_CAP}Å로 캡): `{_CONTACT_FEATURE_NAMES}`\n"
    )
    lines.append(f"- 5-contact 결측 아닌 비율(전체 학습 데이터 기준): {contact_coverage}\n\n")

    lines.append("## 4. 모델 & 하이퍼파라미터\n")
    lines.append(
        "- `xgboost.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, "
        "subsample=0.9, colsample_bytree=0.9, early_stopping_rounds=20, eval_metric='mae')` "
        "(v1과 동일)\n"
    )
    lines.append(f"- best_iteration (early stopping, val 기준): {best_iter}\n\n")

    def fmt_metrics(m):
        return (f"MAE={m['mae']:.3f} REU, RMSE={m['rmse']:.3f} REU, "
                f"Spearman rho={m['spearman_rho']:.3f} (p={m['spearman_p']:.2e}), "
                f"R2={m['r2']:.3f}")

    lines.append("## 5. 결과 — 메인 (서열-외삽 / unseen-sequence)\n")
    if v1_test_line:
        lines.append(f"- **v1(A7) baseline Test**: {v1_test_line} (원본: `_workspace/A7_XGB_TRAINING_REPORT_2026-08-03.md`)\n")
    lines.append(f"- Train: {fmt_metrics(train_metrics)}\n")
    lines.append(f"- Val:   {fmt_metrics(val_metrics)}\n")
    lines.append(f"- **v2 Test: {fmt_metrics(test_metrics)}**\n\n")

    lines.append("### 5.1 서열별 pose 순위 재현 (test, 우리 실제 목적)\n")
    lines.append(f"- 평가 대상 서열 수: {test_seq_metrics['n_sequences_evaluated']} "
                  f"(pose 3개 미만 또는 순위 정의 불가 서열 {test_seq_metrics['n_sequences_skipped_insufficient_poses']}개 제외)\n")
    if test_seq_metrics["mean_per_sequence_spearman"] is not None:
        lines.append(f"- 서열별 Spearman rho 평균: {test_seq_metrics['mean_per_sequence_spearman']:.3f}\n")
        lines.append(f"- 서열별 Spearman rho 중앙값: {test_seq_metrics['median_per_sequence_spearman']:.3f}\n\n")
    else:
        lines.append("- test set 내 pose 3개 이상 보유 서열이 없어 평가 불가.\n\n")

    if interp_result is not None:
        lines.append("## 6. (옵션/참고) Pose-내삽 순위 재현 상한 — 별도 미니 실험\n")
        lines.append(
            "**주의**: v1과 동일하게 메인 리포트 수치(§5)와 완전히 별개의 낙관적 상한 참고치다.\n\n"
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
        lines.append("## 6. (옵션) Pose-내삽 실험: 생략\n대상 서열(pose>=30) 없음.\n\n")

    lines.append("## 7. Feature Importance (v1 대비)\n")
    lines.append("![feature importance](A13_xgb_v2_feature_importance.png)\n\n")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    lines.append("전체 feature importance 순위(gain-normalized, 내림차순):\n\n")
    for rank, i in enumerate(order, 1):
        marker = " ← 신규(pharmacophore 5-contact)" if feature_cols[i] in _CONTACT_FEATURE_NAMES else ""
        lines.append(f"{rank}. `{feature_cols[i]}` = {importances[i]:.4f}{marker}\n")
    lines.append("\n")

    lines.append("## 8. 결론 및 다음 단계\n")
    v1_rho = 0.297
    v2_rho = test_metrics["spearman_rho"]
    delta = v2_rho - v1_rho if v2_rho is not None else None
    if delta is not None:
        direction = "개선" if delta > 0 else ("악화" if delta < 0 else "변화 없음")
        lines.append(f"- v1 Test rho={v1_rho:.3f} → v2 Test rho={v2_rho:.3f} ({direction}, Δ={delta:+.3f}).\n")
    if v2_rho is not None and v2_rho >= 0.4:
        lines.append(
            "- rho가 0.4 이상으로 개선되어 BO acquisition 후보 pre-filter/re-rank의 약한 prior로 "
            "통합 검토 가능. 단 최종 선택은 반드시 실제 FlexPepDock 재도킹으로 확정.\n"
        )
    else:
        lines.append(
            "- rho가 여전히 0.4 미만이라 **BO 후보 랭킹에 바로 투입 비권장**. pharmacophore "
            "5-contact(수용체 넘버링 비의존 CA-CA 최소거리)만으로는 신호 보강이 불충분함을 실증. "
            "다음 우선순위(Phase 1 판정 순서상 2순위: pose 수 확충, 3순위: ESM-2 임베딩)로 이동 검토 필요. "
            "또한 §1에서 실측했듯 literal salt-bridge/H-bond fingerprint는 이 데이터셋에서 원천적으로 "
            "사용 불가(수용체 PDB 재넘버링 또는 UniProt 정합 매핑 별도 작업 필요 — 본 스크립트 범위 밖).\n"
        )
    lines.append(
        "- 어느 경우든 이 모델은 **exploration 후보를 좁히는 1차 필터**로만 쓰고, "
        "최종 ddG 판정은 항상 실제 도킹(FlexPepDock)으로 재확인한다.\n"
    )

    with open(OUT_REPORT_PATH, "w") as fh:
        fh.writelines(lines)


if __name__ == "__main__":
    main()
