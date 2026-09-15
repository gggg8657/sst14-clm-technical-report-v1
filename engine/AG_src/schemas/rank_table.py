"""
rank_table.py
=============
순위 테이블 구축 및 유틸리티 함수
Ranking table construction and utility functions for the SSTR2 pipeline.

모든 단계(Step 04~07) 결과를 통합하여 최종 순위를 계산하고
CSV로 내보내며 게이트 필터링을 수행합니다.
Merges results from steps 04-07, computes final weighted scores,
exports to CSV, and filters by gate thresholds.
"""

from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Optional, Tuple

from .io_schemas import (
    DockingResult,
    QCResult,
    RankTableRow,
    RosettaResult,
    Step04Output,
    Step05Output,
    Step06Output,
    Step07Output,
)


# =============================================================================
# Score Normalization (점수 정규화)
# =============================================================================

def normalize_scores(
    scores: List[float],
    method: str = "min_max",
) -> List[float]:
    """
    점수 목록을 정규화합니다.
    Normalize a list of scores.

    Args:
        scores:  원시 점수 목록 / List of raw scores.
        method:  "min_max" (0~1 범위) 또는 "z_score" (표준화).

    Returns:
        정규화된 점수 목록 / Normalized scores in the same order.

    Notes:
        - min_max: (x - min) / (max - min); 상수 리스트면 모두 0.5 반환.
        - z_score: (x - mean) / std; std == 0 이면 모두 0.0 반환.
    """
    if not scores:
        return []

    if method == "min_max":
        lo, hi = min(scores), max(scores)
        if math.isclose(lo, hi):
            return [0.5] * len(scores)
        return [(x - lo) / (hi - lo) for x in scores]

    elif method == "z_score":
        n = len(scores)
        mean = sum(scores) / n
        variance = sum((x - mean) ** 2 for x in scores) / n
        std = math.sqrt(variance)
        if math.isclose(std, 0.0):
            return [0.0] * n
        return [(x - mean) / std for x in scores]

    else:
        raise ValueError(f"Unknown normalization method: {method!r}. Use 'min_max' or 'z_score'.")


# =============================================================================
# Final Score Computation (최종 점수 계산)
# =============================================================================

def compute_final_score(
    plddt: float,
    dock_score: float,
    ddg: float,
    lddt: float,
    selectivity_margin: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    가중 최종 점수를 계산합니다 (값이 클수록 좋음).
    Compute the weighted final score (higher = better).

    점수 방향 처리 (direction handling):
    - plddt: higher is better  -> 그대로 사용
    - dock_score: lower is better -> 부호 반전 후 사용
    - ddg: lower is better -> 부호 반전 후 사용
    - lddt: higher is better -> 그대로 사용
    - selectivity_margin: lower is better (더 음수 = 더 선택적) -> 부호 반전 후 사용

    Args:
        plddt:      ESMFold 평균 pLDDT (0~100).
        dock_score: 도킹 점수 (더 음수 = 더 좋음).
        ddg:        Rosetta ΔΔG kcal/mol (더 음수 = 더 좋음).
        lddt:       FoldMason lDDT (0~1).
        selectivity_margin: 선택성 마진 (더 음수 = 더 좋음, 기본 0.0).
        weights:    {"plddt": w1, "dock_score": w2, "ddg": w3, "lddt": w4, "selectivity": w5}.
                    기본값: gate_thresholds.yaml과 동일한 가중치.

    Returns:
        최종 점수 (float) / Final weighted score.
    """
    if weights is None:
        weights = {
            "plddt": 0.15,
            "dock_score": 0.25,
            "ddg": 0.25,
            "lddt": 0.15,
            "selectivity": 0.20,
        }

    # pLDDT를 0~1 범위로 정규화 (100점 만점 -> 1.0)
    plddt_norm = plddt / 100.0

    # dock_score와 ddg는 낮을수록 좋으므로 부호 반전 후 정규화
    # (부호 반전만 적용; 실제 사용 시 배치 정규화를 권장)
    dock_contrib = -dock_score * weights.get("dock_score", 0.25)
    ddg_contrib = -ddg * weights.get("ddg", 0.25)

    # selectivity_margin: 낮을수록 좋음 (부호 반전)
    sel_contrib = -selectivity_margin * weights.get("selectivity", 0.20)

    score = (
        plddt_norm * weights.get("plddt", 0.15)
        + dock_contrib
        + ddg_contrib
        + lddt * weights.get("lddt", 0.15)
        + sel_contrib
    )
    return round(score, 6)


# =============================================================================
# Rank Table Builder (순위 테이블 구축)
# =============================================================================

def build_rank_table(
    step04_output: Step04Output,
    step05_output: Step05Output,
    step06_output: Step06Output,
    step07_output: Step07Output,
    weights: Optional[Dict[str, float]] = None,
) -> List[RankTableRow]:
    """
    각 단계 결과를 seq_id 기준으로 병합하여 최종 순위 테이블을 생성합니다.
    Merge step results by seq_id and compute final ranked table.

    Args:
        step04_output: ESMFold QC 결과.
        step05_output: 도킹 결과.
        step06_output: Rosetta 정제 결과.
        step07_output: 분석 결과 (lDDT는 별도 JSON에서 로드되므로 여기서는 경로만 참조).
        weights:       최종 점수 가중치.

    Returns:
        최종 점수 기준 내림차순 정렬된 RankTableRow 목록.
        List of RankTableRow sorted by final_score descending.
    """
    # seq_id 기준 조회 딕셔너리 구축
    qc_map: Dict[str, QCResult] = {r.seq_id: r for r in step04_output.qc_results}
    dock_map: Dict[str, DockingResult] = {r.seq_id: r for r in step05_output.docking_results}
    rosetta_map: Dict[str, RosettaResult] = {r.seq_id: r for r in step06_output.rosetta_results}

    # lDDT 맵은 Step07Output에 경로만 있으므로 기본값 사용
    # (실제 사용 시 lddt_table_path에서 로드 후 주입)
    lddt_map: Dict[str, float] = {}

    rows: List[RankTableRow] = []

    # Rosetta 결과 기준으로 병합 (Rosetta까지 통과한 후보만 최종 테이블에 포함)
    for seq_id, rosetta in rosetta_map.items():
        qc = qc_map.get(seq_id)
        dock = dock_map.get(seq_id)

        if qc is None or dock is None:
            continue  # 데이터 누락 시 건너뜀

        lddt_val = lddt_map.get(seq_id, float("nan"))

        # 백본 인덱스 파싱 (seq_id 형식: "bb{bb_idx:02d}_seq{seq_idx:02d}")
        try:
            backbone_id = int(seq_id.split("_")[0].replace("bb", ""))
        except (IndexError, ValueError):
            backbone_id = -1

        final_score = compute_final_score(
            plddt=qc.plddt_mean,
            dock_score=dock.score,
            ddg=rosetta.ddg,
            lddt=lddt_val if not math.isnan(lddt_val) else 0.0,
            selectivity_margin=0.0,  # 선택성 데이터는 별도 주입 필요
            weights=weights,
        )

        row = RankTableRow(
            backbone_id=backbone_id,
            seq_id=seq_id,
            sequence="",            # 시퀀스는 Step03 결과에서 주입 필요
            plddt_mean=qc.plddt_mean,
            plddt_interface=qc.plddt_interface,
            dock_score=dock.score,
            dock_engine=dock.engine,
            ddg=rosetta.ddg,
            lddt=lddt_val,
            final_score=final_score,
            pass_fail="PASS",       # 이 단계까지 온 후보는 임시 PASS
            fail_reason="",
        )
        rows.append(row)

    # 최종 점수 기준 내림차순 정렬
    rows.sort(key=lambda r: r.final_score, reverse=True)
    return rows


# =============================================================================
# Gate Filtering (게이트 필터링)
# =============================================================================

def filter_by_gates(
    rank_table: List[RankTableRow],
    thresholds: Dict[str, Any],
) -> List[RankTableRow]:
    """
    게이트 임계값에 따라 각 후보를 PASS/FAIL로 표시합니다.
    Mark each candidate PASS or FAIL according to gate thresholds.

    Args:
        rank_table:   RankTableRow 목록.
        thresholds:   gate_thresholds.yaml 로드 결과 딕셔너리.

    Returns:
        pass_fail 및 fail_reason 필드가 갱신된 새 RankTableRow 목록.
    """
    plddt_min = thresholds.get("esmfold_plddt_min", 75)
    iface_min = thresholds.get("esmfold_interface_plddt_min", 70)
    ddg_max = thresholds.get("rosetta_ddg_max", -5.0)
    clash_max = thresholds.get("rosetta_clash_max", 0)
    lddt_min = thresholds.get("foldmason_lddt_min", 0.6)
    selectivity_margin_min = thresholds.get("selectivity_margin_min", None)

    updated: List[RankTableRow] = []
    for row in rank_table:
        reasons: List[str] = []

        if row.plddt_mean < plddt_min:
            reasons.append(f"pLDDT_mean={row.plddt_mean:.1f}<{plddt_min}")
        if row.plddt_interface < iface_min:
            reasons.append(f"pLDDT_iface={row.plddt_interface:.1f}<{iface_min}")
        if row.ddg > ddg_max:
            reasons.append(f"ddG={row.ddg:.2f}>{ddg_max}")
        if not math.isnan(row.lddt) and row.lddt < lddt_min:
            reasons.append(f"lDDT={row.lddt:.3f}<{lddt_min}")

        # Gate 4: Selectivity (선택성 게이트)
        if selectivity_margin_min is not None and row.selectivity_margin != 0.0:
            if row.selectivity_margin > selectivity_margin_min:
                reasons.append(
                    f"selectivity_margin={row.selectivity_margin:.2f}>{selectivity_margin_min}"
                )
                row.pass_selectivity = "FAIL"
            else:
                row.pass_selectivity = "PASS"

        # RankTableRow는 frozen이 아니므로 직접 변경
        row.pass_fail = "FAIL" if reasons else "PASS"
        row.fail_reason = "; ".join(reasons)
        updated.append(row)

    return updated


# =============================================================================
# CSV Export (CSV 내보내기)
# =============================================================================

# CSV 컬럼 순서 정의 (rank_table_template.csv 헤더와 일치)
_CSV_FIELDS: List[str] = [
    "backbone_id",
    "seq_id",
    "sequence",
    "plddt_mean",
    "plddt_interface",
    "dock_score",
    "dock_engine",
    "ddg",
    "lddt",
    "selectivity_margin",
    "offtarget_max_score",
    "pass_selectivity",
    "final_score",
    "rank",
    "pass_fail",
    "fail_reason",
]


def export_csv(rank_table: List[RankTableRow], output_path: str) -> None:
    """
    순위 테이블을 CSV 파일로 내보냅니다 (rank 열 자동 추가).
    Export rank table to CSV with auto-assigned rank column.

    Args:
        rank_table:   RankTableRow 목록 (final_score 내림차순 정렬 권장).
        output_path:  출력 CSV 파일 경로.
    """
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for rank_idx, row in enumerate(rank_table, start=1):
            row_dict = row.to_dict()
            row_dict["rank"] = rank_idx
            # 필드 순서를 _CSV_FIELDS 기준으로 맞춤
            writer.writerow({k: row_dict.get(k, "") for k in _CSV_FIELDS})
