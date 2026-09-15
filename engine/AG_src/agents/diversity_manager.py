"""
diversity_manager.py
SSTR2 펩타이드 바인더 Co-Scientist - Diversity Manager 에이전트 (신규)
Role: 구조 다양성 관리 (Structural Diversity Management)

Diversity Manager는 QC&Ranker의 게이트 필터링 이후, 최종 랭킹 전에
후보 구조들이 충분히 다양한 구조 공간을 차지하는지 확인한다.
유사 구조의 중복 제출을 방지하고 실험 효율을 높인다.

통합 위치: QC&Ranker의 apply_gates() 이후, compute_rankings() 이전
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .base_agent import BaseAgent
from .qc_ranker import Candidate


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """구조 클러스터 - 유사한 구조들의 그룹.

    Attributes:
        cluster_id: 클러스터 고유 번호
        representative: 클러스터 대표 후보 (중심 구조)
        members: 클러스터에 속하는 모든 후보 목록
        intra_similarity: 클러스터 내 평균 유사도 (0~1, 높을수록 유사)
        method: 클러스터링에 사용된 방법 ('foldmason' | 'sequence' | 'hybrid')
    """
    cluster_id: int
    representative: Candidate
    members: list[Candidate]
    intra_similarity: float = 0.0
    method: str = "foldmason"


@dataclass
class DiversityReport:
    """다양성 분석 결과 보고서.

    Attributes:
        total_candidates: 분석 입력 후보 수
        n_clusters: 발견된 클러스터 수
        diversity_score: 전체 구조 다양성 점수 (0~1, 높을수록 다양)
        redundant_ids: 중복으로 판정된 후보 ID 목록
        cluster_sizes: 클러스터별 크기 목록
        method: 사용된 클러스터링 방법
        created_at: 보고서 생성 시각
    """
    total_candidates: int
    n_clusters: int
    diversity_score: float
    redundant_ids: list[str]
    cluster_sizes: list[int]
    method: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Diversity Manager agent
# ---------------------------------------------------------------------------

class DiversityManagerAgent(BaseAgent):
    """구조 다양성 관리 에이전트 (Co-Scientist 개선 사항).

    역할:
        1. 후보 구조를 구조적 유사도 기반으로 클러스터링 (FoldMason lDDT 활용)
        2. 각 클러스터에서 대표 구조 선택
        3. 최종 선별 후보가 다양한 구조 공간을 커버하는지 보장
        4. 중복/유사 후보 탐지 및 플래그

    통합 위치:
        QC & Ranker의 게이트 통과 후보 목록을 받아
        다양성 기반 선별을 수행한 뒤 QC & Ranker에 반환.

        [QC Gate 통과] -> [Diversity Manager] -> [최종 Ranking]

    클러스터링 방법:
        - 'foldmason': FoldMason lDDT 기반 구조 클러스터링 (권장)
        - 'sequence': 서열 동일성 기반 클러스터링 (빠른 근사)
        - 'hybrid': 구조 + 서열 동시 고려

    Attributes:
        similarity_threshold: 동일 클러스터 판정 유사도 상한 (0~1)
        clustering_method: 사용할 클러스터링 방법
        tool_fn: FoldMason 호출 callable (외부 주입, None이면 서열 기반 근사)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        clustering_method: str = "foldmason",
        tool_fn: Optional[Any] = None,
        llm_provider: str = "claude",
    ) -> None:
        super().__init__(
            name="DiversityManager",
            role="구조 다양성 관리",
            description=(
                "QC 게이트 통과 후보를 구조적 유사도 기반으로 클러스터링하여 "
                "다양한 구조 공간을 커버하는 대표 집합을 선별하고 중복을 제거한다."
            ),
            llm_provider=llm_provider,
        )
        self.similarity_threshold = similarity_threshold
        self.clustering_method = clustering_method
        self.tool_fn = tool_fn  # run_foldmason() 등 외부 주입

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster_candidates(
        self,
        candidates: list[Candidate],
        method: Optional[str] = None,
    ) -> list[Cluster]:
        """후보 목록을 구조적 유사도 기반으로 클러스터링한다.

        FoldMason이 가용한 경우 lDDT 기반 클러스터링을 수행하고,
        그렇지 않으면 서열 동일성 기반 근사 클러스터링으로 대체한다.

        Args:
            candidates: 클러스터링할 Candidate 목록
            method: 클러스터링 방법 (None이면 self.clustering_method 사용)

        Returns:
            Cluster 목록 (클러스터 크기 내림차순)
        """
        m = method or self.clustering_method
        self.log(f"클러스터링 시작: {len(candidates)}개 후보, 방법={m}, 임계값={self.similarity_threshold}")

        if not candidates:
            return []

        if m == "foldmason" and self.tool_fn is not None:
            clusters = self._cluster_by_foldmason(candidates)
        elif m == "hybrid" and self.tool_fn is not None:
            clusters = self._cluster_hybrid(candidates)
        else:
            # 서열 동일성 기반 근사 클러스터링 (fallback)
            if m == "foldmason":
                self.log("FoldMason tool 미등록 - 서열 기반 근사 클러스터링으로 대체", level="warning")
            clusters = self._cluster_by_sequence(candidates)

        clusters.sort(key=lambda c: len(c.members), reverse=True)
        self.log(f"클러스터링 완료: {len(clusters)}개 클러스터")
        return clusters

    def select_diverse_set(
        self,
        clusters: list[Cluster],
        n: int,
    ) -> list[Candidate]:
        """클러스터에서 대표 구조를 선택하여 다양한 후보 집합을 구성한다.

        선택 전략:
            1. 각 클러스터에서 대표(representative)를 우선 선택
            2. n에 달할 때까지 큰 클러스터부터 2번째 멤버를 추가
            3. final_score 내림차순으로 정렬하여 반환

        Args:
            clusters: cluster_candidates() 반환값
            n: 선택할 최대 후보 수

        Returns:
            최대 n개의 다양한 Candidate 목록 (final_score 내림차순)
        """
        if not clusters:
            return []

        selected: list[Candidate] = []
        seen_ids: set[str] = set()

        # 1단계: 각 클러스터 대표 선택
        for cluster in clusters:
            if len(selected) >= n:
                break
            rep = cluster.representative
            if rep.candidate_id not in seen_ids:
                selected.append(rep)
                seen_ids.add(rep.candidate_id)

        # 2단계: 부족하면 큰 클러스터의 2번째 멤버 추가
        if len(selected) < n:
            for cluster in clusters:
                for member in cluster.members[1:]:  # 대표(index 0) 제외
                    if len(selected) >= n:
                        break
                    if member.candidate_id not in seen_ids:
                        selected.append(member)
                        seen_ids.add(member.candidate_id)
                if len(selected) >= n:
                    break

        # final_score 내림차순 정렬
        selected.sort(key=lambda c: c.final_score, reverse=True)
        self.log(f"다양성 기반 선별 완료: {len(selected)}/{n}개")
        return selected

    def compute_diversity_score(self, candidates: list[Candidate]) -> float:
        """후보 집합의 전체 구조 다양성 점수를 계산한다 (0~1).

        서열 기반 근사:
            - 모든 쌍의 서열 동일성을 계산하고 평균 비유사도를 반환
            - diversity = 1 - mean_pairwise_identity
            - FoldMason lDDT가 있으면 해당 값을 사용 (lddt 필드 활용)

        Args:
            candidates: 평가할 Candidate 목록

        Returns:
            다양성 점수 (0~1, 1에 가까울수록 다양)
        """
        if len(candidates) <= 1:
            return 1.0 if len(candidates) == 1 else 0.0

        # lDDT 값이 채워진 경우: 구조 다양성으로 계산
        lddt_values = [c.lddt for c in candidates if c.lddt > 0]
        if len(lddt_values) == len(candidates):
            # 높은 lDDT = 구조 보존 = 낮은 다양성
            mean_lddt = sum(lddt_values) / len(lddt_values)
            diversity = 1.0 - mean_lddt
            self.log(f"lDDT 기반 다양성: {diversity:.3f} (평균 lDDT={mean_lddt:.3f})")
            return round(diversity, 4)

        # 서열 기반 근사
        total_identity = 0.0
        pair_count = 0
        seqs = [c.sequence for c in candidates]

        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                identity = self._sequence_identity(seqs[i], seqs[j])
                total_identity += identity
                pair_count += 1

        mean_identity = total_identity / pair_count if pair_count > 0 else 0.0
        diversity = 1.0 - mean_identity
        self.log(f"서열 기반 다양성: {diversity:.3f} (평균 서열 동일성={mean_identity:.3f})")
        return round(diversity, 4)

    def flag_redundant(
        self,
        candidates: list[Candidate],
        similarity_threshold: Optional[float] = None,
    ) -> list[str]:
        """유사도 임계값 이상인 중복 후보 ID 목록을 반환한다.

        탐지 방법:
            - 모든 쌍에 대해 서열 동일성 계산
            - identity >= threshold인 쌍에서 final_score가 낮은 후보를 중복으로 표시

        Args:
            candidates: 검사할 Candidate 목록
            similarity_threshold: 중복 판정 임계값 (None이면 self.similarity_threshold)

        Returns:
            중복으로 판정된 후보 ID 목록
        """
        threshold = similarity_threshold or self.similarity_threshold
        redundant_ids: set[str] = set()

        # final_score 내림차순 정렬 (높은 점수를 우선 보존)
        sorted_cands = sorted(candidates, key=lambda c: c.final_score, reverse=True)

        for i, c1 in enumerate(sorted_cands):
            if c1.candidate_id in redundant_ids:
                continue
            for c2 in sorted_cands[i + 1:]:
                if c2.candidate_id in redundant_ids:
                    continue
                identity = self._sequence_identity(c1.sequence, c2.sequence)
                if identity >= threshold:
                    redundant_ids.add(c2.candidate_id)  # 낮은 점수 쪽 제거
                    self.log(
                        f"  중복 탐지: {c1.candidate_id} <-> {c2.candidate_id} "
                        f"(identity={identity:.2f} >= {threshold})"
                    )

        self.log(f"중복 탐지 완료: {len(redundant_ids)}개 중복 후보")
        return list(redundant_ids)

    def generate_diversity_report(
        self,
        candidates: list[Candidate],
        clusters: list[Cluster],
        redundant_ids: list[str],
        method: str,
    ) -> DiversityReport:
        """다양성 분석 결과 보고서를 생성한다.

        Args:
            candidates: 전체 입력 후보
            clusters: 클러스터 목록
            redundant_ids: 중복 후보 ID 목록
            method: 사용된 클러스터링 방법

        Returns:
            DiversityReport
        """
        diversity_score = self.compute_diversity_score(candidates)
        report = DiversityReport(
            total_candidates=len(candidates),
            n_clusters=len(clusters),
            diversity_score=diversity_score,
            redundant_ids=redundant_ids,
            cluster_sizes=[len(c.members) for c in clusters],
            method=method,
        )
        self.log(
            f"다양성 보고서: {len(clusters)}개 클러스터, "
            f"다양성 점수={diversity_score:.3f}, "
            f"중복={len(redundant_ids)}개"
        )
        return report

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """QC 게이트 통과 후보를 받아 다양성 기반 선별을 수행한다.

        context 키:
            - candidates (list[Candidate]): QC 게이트 통과 후보
            - n_select (int): 선별할 후보 수 (기본 20)
            - method (str, optional): 클러스터링 방법

        Returns:
            {
                'status': str,
                'diverse_candidates': list[Candidate],
                'clusters': list[Cluster],
                'diversity_report': DiversityReport,
                'redundant_ids': list[str],
            }
        """
        candidates: list[Candidate] = context.get("candidates", [])
        n_select: int = context.get("n_select", 20)
        method: str = context.get("method", self.clustering_method)

        if not candidates:
            self.log("입력 후보 없음 - 다양성 관리 건너뜀", level="warning")
            return {
                "status": "ok",
                "diverse_candidates": [],
                "clusters": [],
                "diversity_report": DiversityReport(
                    total_candidates=0, n_clusters=0, diversity_score=0.0,
                    redundant_ids=[], cluster_sizes=[], method=method,
                ),
                "redundant_ids": [],
            }

        clusters = self.cluster_candidates(candidates, method=method)
        redundant_ids = self.flag_redundant(candidates)
        diverse_set = self.select_diverse_set(clusters, n=n_select)
        report = self.generate_diversity_report(candidates, clusters, redundant_ids, method)

        return {
            "status": "ok",
            "diverse_candidates": diverse_set,
            "clusters": clusters,
            "diversity_report": report,
            "redundant_ids": redundant_ids,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_by_sequence(self, candidates: list[Candidate]) -> list[Cluster]:
        """서열 동일성 기반 greedy 클러스터링 (FoldMason 없을 때 fallback)."""
        threshold = self.similarity_threshold
        assigned: set[str] = set()
        clusters: list[Cluster] = []
        cluster_id = 0

        # final_score 내림차순으로 처리 (고득점 후보가 클러스터 대표)
        sorted_cands = sorted(candidates, key=lambda c: c.final_score, reverse=True)

        for seed in sorted_cands:
            if seed.candidate_id in assigned:
                continue

            members = [seed]
            assigned.add(seed.candidate_id)

            for other in sorted_cands:
                if other.candidate_id in assigned:
                    continue
                identity = self._sequence_identity(seed.sequence, other.sequence)
                if identity >= threshold:
                    members.append(other)
                    assigned.add(other.candidate_id)

            intra_sim = self._intra_cluster_similarity(members)
            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    representative=members[0],
                    members=members,
                    intra_similarity=intra_sim,
                    method="sequence",
                )
            )
            cluster_id += 1

        return clusters

    def _cluster_by_foldmason(self, candidates: list[Candidate]) -> list[Cluster]:
        """FoldMason lDDT 기반 구조 클러스터링.

        tool_fn(pdb_paths, ref_pdb) -> lDDT 행렬을 반환한다고 가정.
        실제 통합 시 tool_fn의 반환 형식에 맞게 파싱 로직을 조정한다.
        """
        self.log("FoldMason 구조 클러스터링 시작")
        pdb_paths = [c.pdb_path for c in candidates]

        try:
            # tool_fn: run_foldmason(models, ref_model, out_dir, params)
            # 반환: {'lddt_matrix': [[float]], 'candidate_ids': [str]}
            result = self.tool_fn(
                models=pdb_paths,
                ref_model=pdb_paths[0],
                out_dir="/tmp/foldmason_cluster",
                params={},
            )
            lddt_matrix: list[list[float]] = result.get("lddt_matrix", [])
        except Exception as exc:
            self.log(f"FoldMason 호출 실패: {exc} - 서열 기반으로 대체", level="warning")
            return self._cluster_by_sequence(candidates)

        if not lddt_matrix or len(lddt_matrix) != len(candidates):
            self.log("lDDT 행렬 형식 오류 - 서열 기반으로 대체", level="warning")
            return self._cluster_by_sequence(candidates)

        # lDDT 행렬 기반 greedy 클러스터링
        threshold = self.similarity_threshold
        assigned: set[str] = set()
        clusters: list[Cluster] = []
        cluster_id = 0

        sorted_cands = sorted(candidates, key=lambda c: c.final_score, reverse=True)
        id_to_idx = {c.candidate_id: i for i, c in enumerate(candidates)}

        for seed in sorted_cands:
            if seed.candidate_id in assigned:
                continue
            seed_idx = id_to_idx[seed.candidate_id]
            members = [seed]
            assigned.add(seed.candidate_id)

            for other in sorted_cands:
                if other.candidate_id in assigned:
                    continue
                other_idx = id_to_idx[other.candidate_id]
                # lDDT is symmetric; use the higher index for row
                r, col = (seed_idx, other_idx) if seed_idx < other_idx else (other_idx, seed_idx)
                try:
                    sim = lddt_matrix[col][r]
                except IndexError:
                    sim = 0.0
                if sim >= threshold:
                    members.append(other)
                    assigned.add(other.candidate_id)

            intra_sim = self._intra_cluster_similarity_lddt(members, lddt_matrix, id_to_idx)
            clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    representative=members[0],
                    members=members,
                    intra_similarity=intra_sim,
                    method="foldmason",
                )
            )
            cluster_id += 1

        return clusters

    def _cluster_hybrid(self, candidates: list[Candidate]) -> list[Cluster]:
        """구조(FoldMason) + 서열 동일성 혼합 클러스터링.

        두 방법의 클러스터 결과를 교집합 기준으로 병합한다.
        """
        struct_clusters = self._cluster_by_foldmason(candidates)
        seq_clusters = self._cluster_by_sequence(candidates)

        # 두 결과 중 더 많은 클러스터를 생성한 것을 사용 (보수적 선택)
        if len(struct_clusters) >= len(seq_clusters):
            self.log(f"Hybrid: 구조 기반 선택 ({len(struct_clusters)}개 클러스터)")
            return struct_clusters
        self.log(f"Hybrid: 서열 기반 선택 ({len(seq_clusters)}개 클러스터)")
        return seq_clusters

    @staticmethod
    def _sequence_identity(seq1: str, seq2: str) -> float:
        """두 서열의 동일성을 계산한다 (0~1).

        단순 위치별 비교 (정렬 없음). 길이가 다르면 짧은 쪽 기준.
        """
        if not seq1 or not seq2:
            return 0.0
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return 0.0
        matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
        # 길이 차이 패널티 포함
        return matches / max_len

    @staticmethod
    def _intra_cluster_similarity(members: list[Candidate]) -> float:
        """클러스터 내 평균 서열 동일성을 계산한다."""
        if len(members) <= 1:
            return 1.0
        total = 0.0
        pairs = 0
        seqs = [m.sequence for m in members]
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                si, sj = seqs[i], seqs[j]
                if si and sj:
                    min_l = min(len(si), len(sj))
                    max_l = max(len(si), len(sj))
                    matches = sum(a == b for a, b in zip(si[:min_l], sj[:min_l]))
                    total += matches / max_l if max_l > 0 else 0.0
                    pairs += 1
        return round(total / pairs, 4) if pairs > 0 else 1.0

    @staticmethod
    def _intra_cluster_similarity_lddt(
        members: list[Candidate],
        lddt_matrix: list[list[float]],
        id_to_idx: dict[str, int],
    ) -> float:
        """lDDT 행렬 기반 클러스터 내 평균 유사도를 계산한다."""
        if len(members) <= 1:
            return 1.0
        total = 0.0
        pairs = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                idx_i = id_to_idx.get(members[i].candidate_id, -1)
                idx_j = id_to_idx.get(members[j].candidate_id, -1)
                if idx_i < 0 or idx_j < 0:
                    continue
                r, c = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
                try:
                    total += lddt_matrix[c][r]
                except IndexError:
                    pass
                pairs += 1
        return round(total / pairs, 4) if pairs > 0 else 1.0
