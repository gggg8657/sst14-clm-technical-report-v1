"""
qc_ranker.py
SSTR2 펩타이드 바인더 Co-Scientist - QC & Ranker 에이전트
Role: 품질관리 및 랭킹 (Quality Control & Ranking)

QC & Ranker는 ESMFold pLDDT, FoldMason lDDT, Docking score, Rosetta ddG를
통합하여 단일 랭킹 테이블을 생성하고, 게이트 조건을 적용해 최종 후보를 선별한다.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """단일 펩타이드 후보의 점수 데이터.

    Attributes:
        candidate_id: 후보 고유 식별자 (예: 'bb03_seq02')
        backbone_id: 백본 인덱스 (RFdiffusion 생성 순서)
        seq_id: 서열 인덱스 (MPNN 생성 순서)
        sequence: 아미노산 서열 (1-letter code)
        pdb_path: 최종 구조 파일 경로
        plddt_mean: ESMFold 전체 평균 pLDDT (0~100)
        plddt_interface: ESMFold 계면 잔기 평균 pLDDT (0~100)
        dock_score: 도킹 점수 (더 음수 = 더 좋음)
        ddg: Rosetta ΔΔG (kcal/mol, 더 음수 = 더 강한 결합)
        clash_count: 클래시 위반 수
        constraint_violations: 제약 위반 수
        lddt: FoldMason lDDT (0~1)
        final_score: 가중 합산 최종 점수 (높을수록 좋음)
        pass_gates: 게이트 통과 여부
        fail_reasons: 게이트 실패 이유 목록
    """
    candidate_id: str
    backbone_id: int
    seq_id: int
    sequence: str = ""
    pdb_path: str = ""
    plddt_mean: float = 0.0
    plddt_interface: float = 0.0
    dock_score: float = 0.0
    ddg: float = 0.0
    clash_count: int = 0
    constraint_violations: int = 0
    lddt: float = 0.0
    selectivity_margin: float = 0.0
    offtarget_max_score: float = 0.0
    final_score: float = 0.0
    pass_gates: bool = False
    fail_reasons: list[str] = field(default_factory=list)


@dataclass
class RankTable:
    """랭킹 테이블 - QC 통과 후보의 순위 목록.

    Attributes:
        run_id: 실행 식별자
        iteration: 반복 번호
        ranked_candidates: 순위 순으로 정렬된 Candidate 목록
        weights: 점수 가중치 딕셔너리
        created_at: 생성 시각
    """
    run_id: str
    iteration: int
    ranked_candidates: list[Candidate]
    weights: dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class QCReport:
    """QC 실행 결과 보고서.

    Attributes:
        run_id: 실행 식별자
        total_input: 입력 후보 수
        passed_count: 모든 게이트 통과 수
        failed_count: 게이트 실패 수
        failure_breakdown: 실패 유형별 카운트
        gates_applied: 적용된 게이트 조건 딕셔너리
        pass_rate: 통과율 (0~1)
    """
    run_id: str
    total_input: int
    passed_count: int
    failed_count: int
    failure_breakdown: dict[str, int]
    gates_applied: dict[str, Any]
    pass_rate: float = 0.0


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "plddt": 0.15,
    "dock_score": 0.25,
    "ddg": 0.25,
    "lddt": 0.15,
    "selectivity": 0.20,
}

PYROSETTA_ONLY_WEIGHTS: dict[str, float] = {
    "plddt": 0.0,
    "dock_score": 0.0,
    "ddg": 0.70,
    "lddt": 0.0,
    "selectivity": 0.0,
    "total_score": 0.20,
    "clash": 0.10,
}


# ---------------------------------------------------------------------------
# QC & Ranker agent
# ---------------------------------------------------------------------------

class QCRankerAgent(BaseAgent):
    """품질관리 및 랭킹 에이전트.

    역할:
        1. ESMFold pLDDT, FoldMason lDDT, Docking score, ddG를 단일 랭킹 테이블로 통합
        2. 게이트 조건 적용 (단계별 AND 조건)
        3. 가중 합산으로 최종 점수 계산
        4. 통과/실패 결정 및 실패 이유 기록

    게이트 적용 순서:
        1. ESMFold pLDDT 게이트: mean >= 75 AND interface >= 70
        2. 도킹 게이트: 전체 상위 20%
        3. Rosetta 게이트: ddG <= threshold AND clash == 0 AND violations == 0

    최종 점수 공식:
        final_score = w1*norm(plddt) + w2*norm(-dock_score) + w3*norm(-ddg) + w4*norm(lddt)
        (dock_score, ddg는 낮을수록 좋으므로 부호 반전 후 정규화)
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        llm_provider: str = "claude",
    ) -> None:
        super().__init__(
            name="QCRanker",
            role="품질관리 및 랭킹",
            description=(
                "ESMFold pLDDT, FoldMason lDDT, Docking score, Rosetta ddG를 "
                "단일 랭킹 테이블로 통합하고, 게이트 조건을 적용해 최종 후보를 선별한다."
            ),
            llm_provider=llm_provider,
        )
        self.weights: dict[str, float] = weights or DEFAULT_WEIGHTS.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_gates(
        self,
        candidates: list[Candidate],
        thresholds: dict[str, Any],
    ) -> tuple[list[Candidate], list[Candidate]]:
        """게이트 조건을 순서대로 적용하여 통과/실패 목록을 반환한다.

        게이트 순서 (AND 조건):
            Gate 1 - ESMFold pLDDT:
                plddt_mean >= esmfold_plddt_min (기본 75)
                plddt_interface >= esmfold_interface_plddt_min (기본 70)
            Gate 2 - Docking:
                dock_score 기준 상위 docking_top_pct% 해당 후보만 통과
            Gate 3 - Rosetta:
                ddg <= rosetta_ddg_max
                clash_count <= rosetta_clash_max
                constraint_violations <= rosetta_constraint_violations_max

        Args:
            candidates: 평가할 Candidate 목록
            thresholds: 임계값 딕셔너리 (gate_thresholds.yaml 구조 대응)

        Returns:
            (passed, failed): 통과/실패 Candidate 목록 튜플
        """
        self.log(f"게이트 적용 시작: {len(candidates)}개 후보")
        for c in candidates:
            c.fail_reasons = []

        gates_enabled: dict[str, bool] = thresholds.get("gates_enabled", {})

        # Gate 1: ESMFold pLDDT
        if gates_enabled.get("plddt", True):
            plddt_min = thresholds.get("esmfold_plddt_min", 75)
            iface_min = thresholds.get("esmfold_interface_plddt_min", 70)
            for c in candidates:
                if c.plddt_mean < plddt_min:
                    c.fail_reasons.append(f"pLDDT_mean {c.plddt_mean:.1f} < {plddt_min}")
                if c.plddt_interface < iface_min:
                    c.fail_reasons.append(f"pLDDT_interface {c.plddt_interface:.1f} < {iface_min}")
            after_gate1 = [c for c in candidates if not c.fail_reasons]
            self.log(f"  Gate1 (pLDDT): {len(after_gate1)}/{len(candidates)} 통과")
        else:
            after_gate1 = list(candidates)
            self.log("  Gate1 (pLDDT): DISABLED - 전체 통과")

        # Gate 2: Docking - 상위 top_pct% (dock_score 낮을수록 좋음)
        if gates_enabled.get("docking", True):
            top_pct = thresholds.get("docking_top_pct", 20) / 100.0
            if after_gate1:
                sorted_by_dock = sorted(after_gate1, key=lambda c: c.dock_score)
                cutoff_idx = max(1, int(len(sorted_by_dock) * top_pct))
                top_dock_ids = {c.candidate_id for c in sorted_by_dock[:cutoff_idx]}
                for c in after_gate1:
                    if c.candidate_id not in top_dock_ids:
                        c.fail_reasons.append(
                            f"dock_score {c.dock_score:.2f} 상위 {int(top_pct*100)}% 미달"
                        )
            after_gate2 = [c for c in after_gate1 if not c.fail_reasons]
            self.log(f"  Gate2 (Docking): {len(after_gate2)}/{len(after_gate1)} 통과")
        else:
            after_gate2 = list(after_gate1)
            self.log("  Gate2 (Docking): DISABLED - 전체 통과")

        # Gate 3: Rosetta ddG / clash / constraint
        if gates_enabled.get("rosetta", True):
            ddg_max = thresholds.get("rosetta_ddg_max", -5.0)
            clash_max = thresholds.get("rosetta_clash_max", 10)
            viol_max = thresholds.get("rosetta_constraint_violations_max", 0)
            for c in after_gate2:
                if c.ddg > ddg_max:
                    c.fail_reasons.append(f"ddG {c.ddg:.2f} > {ddg_max}")
                if c.clash_count > clash_max:
                    c.fail_reasons.append(f"clash {c.clash_count} > {clash_max}")
                if c.constraint_violations > viol_max:
                    c.fail_reasons.append(f"constraint_violations {c.constraint_violations} > {viol_max}")
            after_gate3 = [c for c in after_gate2 if not c.fail_reasons]
            self.log(f"  Gate3 (Rosetta): {len(after_gate3)}/{len(after_gate2)} 통과")
        else:
            after_gate3 = list(after_gate2)
            self.log("  Gate3 (Rosetta): DISABLED - 전체 통과")

        # Gate 4: Selectivity (선택성 게이트)
        if gates_enabled.get("selectivity", True):
            sel_margin_min = thresholds.get("selectivity_margin_min", None)
            offtarget_max_allowed = thresholds.get("offtarget_max_allowed", None)
            if sel_margin_min is not None:
                for c in after_gate3:
                    # G-2: 양수=좋음 컨벤션 — margin < sel_margin_min 이면 탈락
                    if c.selectivity_margin != 0.0 and c.selectivity_margin < sel_margin_min:
                        c.fail_reasons.append(
                            f"selectivity_margin {c.selectivity_margin:.2f} < {sel_margin_min}"
                        )
                    if (
                        offtarget_max_allowed is not None
                        and c.offtarget_max_score != 0.0
                        and c.offtarget_max_score < offtarget_max_allowed
                    ):
                        c.fail_reasons.append(
                            f"offtarget_max_score {c.offtarget_max_score:.2f} < {offtarget_max_allowed}"
                        )
            after_gate4 = [c for c in after_gate3 if not c.fail_reasons]
            self.log(f"  Gate4 (Selectivity): {len(after_gate4)}/{len(after_gate3)} 통과")
        else:
            after_gate4 = list(after_gate3)
            self.log("  Gate4 (Selectivity): DISABLED - 전체 통과")

        passed = after_gate4
        failed = [c for c in candidates if c.fail_reasons]

        for c in passed:
            c.pass_gates = True
        for c in failed:
            c.pass_gates = False

        self.log(f"게이트 완료: {len(passed)} 통과 / {len(failed)} 실패")
        return passed, failed

    def compute_rankings(
        self,
        candidates: list[Candidate],
        weights: Optional[dict[str, float]] = None,
    ) -> RankTable:
        """게이트 통과 후보에 대해 가중 합산 점수를 계산하고 순위를 매긴다.

        점수 공식 (min-max 정규화 후 가중 합산):
            normalized_plddt   = (plddt - min) / (max - min)         # 높을수록 좋음
            normalized_dock    = 1 - (dock - min) / (max - min)      # 낮을수록 좋음 (반전)
            normalized_ddg     = 1 - (ddg - min) / (max - min)       # 낮을수록 좋음 (반전)
            normalized_lddt    = (lddt - min) / (max - min)          # 높을수록 좋음
            final_score = w1*norm_plddt + w2*norm_dock + w3*norm_ddg + w4*norm_lddt

        Args:
            candidates: 게이트 통과 Candidate 목록
            weights: 가중치 딕셔너리 (없으면 self.weights 사용)

        Returns:
            RankTable: 순위 순으로 정렬된 결과
        """
        w = weights or self.weights
        run_id = "unknown"

        if not candidates:
            self.log("랭킹 계산: 후보 없음", level="warning")
            return RankTable(run_id=run_id, iteration=0, ranked_candidates=[], weights=w)

        # min-max 정규화 헬퍼
        def _normalize(values: list[float], invert: bool = False) -> list[float]:
            mn, mx = min(values), max(values)
            if mx == mn:
                return [1.0] * len(values)
            normed = [(v - mn) / (mx - mn) for v in values]
            return [1.0 - n for n in normed] if invert else normed

        plddt_vals = [c.plddt_mean for c in candidates]
        dock_vals = [c.dock_score for c in candidates]
        ddg_vals = [c.ddg for c in candidates]
        lddt_vals = [c.lddt for c in candidates]
        sel_vals = [c.selectivity_margin for c in candidates]

        norm_plddt = _normalize(plddt_vals, invert=False)
        norm_dock = _normalize(dock_vals, invert=True)   # 낮을수록 좋음
        norm_ddg = _normalize(ddg_vals, invert=True)     # 낮을수록 좋음
        norm_lddt = _normalize(lddt_vals, invert=False)
        norm_sel = _normalize(sel_vals, invert=False)    # G-2: 높을수록 좋음 (양수=더 선택적)

        for i, c in enumerate(candidates):
            c.final_score = (
                w.get("plddt", 0.15) * norm_plddt[i]
                + w.get("dock_score", 0.25) * norm_dock[i]
                + w.get("ddg", 0.25) * norm_ddg[i]
                + w.get("lddt", 0.15) * norm_lddt[i]
                + w.get("selectivity", 0.20) * norm_sel[i]
            )

        ranked = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        self.log(f"랭킹 완료: {len(ranked)}개 후보 순위 결정")

        return RankTable(
            run_id=run_id,
            iteration=0,
            ranked_candidates=ranked,
            weights=w,
        )

    def get_top_k(self, rank_table: RankTable, k: int) -> list[Candidate]:
        """랭킹 테이블에서 상위 k개 후보를 반환한다.

        Args:
            rank_table: compute_rankings() 반환값
            k: 반환할 후보 수

        Returns:
            상위 k개 Candidate 목록
        """
        top = rank_table.ranked_candidates[:k]
        self.log(f"Top-{k} 후보 선별: {[c.candidate_id for c in top]}")
        return top

    def get_top_by_ddg(self, candidates: list[Candidate], k: int = 5) -> list[Candidate]:
        """deltaG(ddG) 기준으로 상위 k개 후보를 반환한다.

        ddG가 더 음수일수록 결합이 강하므로, 오름차순 정렬 후 상위 k개를 선택한다.

        Args:
            candidates: 평가할 Candidate 목록
            k: 반환할 후보 수 (기본 5)

        Returns:
            ddG 기준 상위 k개 Candidate 목록
        """
        sorted_by_ddg = sorted(candidates, key=lambda c: c.ddg)
        top = sorted_by_ddg[:k]
        self.log(
            f"Top-{k} by ddG: {[(c.candidate_id, f'{c.ddg:.2f}') for c in top]}"
        )
        return top

    def generate_qc_report(
        self,
        rank_table: RankTable,
        gates_applied: dict[str, Any],
        all_candidates: Optional[list[Candidate]] = None,
    ) -> QCReport:
        """QC 결과 보고서를 생성한다.

        Args:
            rank_table: 랭킹 테이블
            gates_applied: 적용된 게이트 조건
            all_candidates: 게이트 전 전체 후보 목록 (실패 분석용)

        Returns:
            QCReport
        """
        passed = rank_table.ranked_candidates
        total = len(all_candidates) if all_candidates else len(passed)
        failed_cnt = total - len(passed)

        # 실패 유형 분류
        breakdown: dict[str, int] = {}
        if all_candidates:
            for c in all_candidates:
                for reason in c.fail_reasons:
                    # 이유 첫 키워드 추출
                    key = reason.split(" ")[0]
                    breakdown[key] = breakdown.get(key, 0) + 1

        pass_rate = len(passed) / total if total > 0 else 0.0

        report = QCReport(
            run_id=rank_table.run_id,
            total_input=total,
            passed_count=len(passed),
            failed_count=failed_cnt,
            failure_breakdown=breakdown,
            gates_applied=gates_applied,
            pass_rate=pass_rate,
        )
        self.log(
            f"QC 보고서: {len(passed)}/{total} 통과 ({pass_rate*100:.1f}%)"
        )
        return report

    def save_rank_table_csv(self, rank_table: RankTable, output_path: str) -> str:
        """랭킹 테이블을 CSV 파일로 저장한다.

        컬럼: backbone_id, seq_id, candidate_id, sequence,
               plddt_mean, plddt_interface, dock_score, ddg,
               clash_count, constraint_violations, lddt, final_score,
               pass_gates, fail_reasons

        Args:
            rank_table: 저장할 RankTable
            output_path: 저장 경로 (CSV)

        Returns:
            저장된 파일 경로 문자열
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "backbone_id", "seq_id", "candidate_id", "sequence",
            "plddt_mean", "plddt_interface", "dock_score", "ddg",
            "clash_count", "constraint_violations", "lddt",
            "final_score", "pass_gates", "fail_reasons",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in rank_table.ranked_candidates:
                writer.writerow({
                    "backbone_id": c.backbone_id,
                    "seq_id": c.seq_id,
                    "candidate_id": c.candidate_id,
                    "sequence": c.sequence,
                    "plddt_mean": f"{c.plddt_mean:.2f}",
                    "plddt_interface": f"{c.plddt_interface:.2f}",
                    "dock_score": f"{c.dock_score:.4f}",
                    "ddg": f"{c.ddg:.4f}",
                    "clash_count": c.clash_count,
                    "constraint_violations": c.constraint_violations,
                    "lddt": f"{c.lddt:.4f}",
                    "final_score": f"{c.final_score:.6f}",
                    "pass_gates": c.pass_gates,
                    "fail_reasons": "; ".join(c.fail_reasons),
                })

        self.log(f"랭킹 CSV 저장: {path}")
        return str(path)

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """파이프라인 컨텍스트에서 후보 목록을 받아 QC 및 랭킹을 수행한다.

        context 키:
            - candidates (list[Candidate]): 평가할 후보 목록
            - thresholds (dict): 게이트 임계값
            - run_id (str): 실행 식별자
            - iteration (int): 반복 번호
            - output_dir (str, optional): CSV 저장 디렉터리

        Returns:
            {
                'status': str,
                'rank_table': RankTable,
                'qc_report': QCReport,
                'top_candidates': list[Candidate],
            }
        """
        candidates: list[Candidate] = context.get("candidates", [])
        thresholds: dict[str, Any] = context.get("thresholds", {})
        run_id: str = context.get("run_id", "unknown")
        iteration: int = context.get("iteration", 0)
        ranking_mode: str = thresholds.get("ranking_mode", "ddg_primary")
        top_k_ddg: int = int(thresholds.get("top_k_by_ddg", 5))

        passed, failed = self.apply_gates(candidates, thresholds)
        rank_table = self.compute_rankings(passed, self.weights)
        rank_table.run_id = run_id
        rank_table.iteration = iteration

        qc_report = self.generate_qc_report(rank_table, thresholds, candidates)

        if output_dir := context.get("output_dir"):
            csv_path = str(Path(output_dir) / "08_reports" / "rank_table.csv")
            self.save_rank_table_csv(rank_table, csv_path)

        if ranking_mode == "ddg_primary":
            top_candidates = self.get_top_by_ddg(passed, top_k_ddg)
            self.log(f"랭킹 모드: ddg_primary - ddG 기준 상위 {top_k_ddg}개 선정")
        else:
            top_k = context.get("top_k", 10)
            top_candidates = self.get_top_k(rank_table, top_k)
            self.log(f"랭킹 모드: weighted - 가중합 기준 상위 {top_k}개 선정")

        return {
            "status": "ok",
            "rank_table": rank_table,
            "qc_report": qc_report,
            "top_candidates": top_candidates,
        }
