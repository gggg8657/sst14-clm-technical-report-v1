"""Tests for _apply_alternative_scoring integration in runner.py.

각 단계(GNINA, ECR, Pareto, BO)의 graceful degradation 및
정상 동작을 검증합니다.

주의: 실행 환경에 따라 _HAS_GNINA / _HAS_PARETO / _HAS_BO 플래그가
False일 수 있으므로 각 테스트에서 필요한 플래그를 명시적으로 patch합니다.
"""
from __future__ import annotations

import os
# 2026-06-09 B: 이 단위 테스트들은 GNINA/ECR/Pareto/BO 로직 검증용.
# Step 0.5 pepADMET 독성(실제 subprocess)은 비활성화해 빠르고 env-독립적으로 유지.
os.environ.setdefault("SST_DISABLE_PEPADMET_TOX", "1")

import math
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from pyrosetta_flow.schema import CandidateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    idx: int,
    ddg: float = -10.0,
    clash: float = 2.0,
    fail: str = "",
    iteration: int = 1,
    seq: str = "AGCKNFFWKTFTSC",
) -> CandidateResult:
    return CandidateResult(
        iteration=iteration,
        candidate_id=f"iter{iteration:02d}_cand{idx:03d}",
        sequence=seq,
        ddg=ddg,
        total_score=ddg * 10,
        clash_score=clash,
        fail_reason=fail,
    )


def _import_fn():
    from pyrosetta_flow.runner import _apply_alternative_scoring
    return _apply_alternative_scoring


# Pareto rank_candidates 실제 구현 (pymoo mock용)
def _mock_pareto_rank(candidates, clash_threshold=10.0):
    """NSGA-II 없이 단순 ddG 순위를 pareto_rank로 할당."""
    sorted_by_ddg = sorted(range(len(candidates)), key=lambda i: candidates[i].get("ddG", 0))
    for rank, idx in enumerate(sorted_by_ddg):
        candidates[idx]["pareto_rank"] = rank
        candidates[idx]["crowding_distance"] = float(len(candidates) - rank)
    return candidates


# ---------------------------------------------------------------------------
# 1. Empty candidates: no-op
# ---------------------------------------------------------------------------

class TestApplyAlternativeScoringEmpty:
    def test_empty_list_returns_empty(self, tmp_path):
        fn = _import_fn()
        # B2 패치 후: Tuple[List, List[int]] 반환
        result, bo_pos = fn([], iter_dir=tmp_path, iteration=1)
        assert result == []
        assert bo_pos == []


# ---------------------------------------------------------------------------
# 2. Failed candidates: GNINA skip, Pareto에서 hard_violations=1
# ---------------------------------------------------------------------------

class TestFailedCandidatesSkipped:
    def test_failed_candidates_not_gnina_rescored(self, tmp_path):
        fn = _import_fn()
        failed = _make_candidate(1, ddg=999.0, fail="dock error")
        ok = _make_candidate(2, ddg=-8.0)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=[]) as mock_gnina:
            result, _ = fn([failed, ok], iter_dir=tmp_path, iteration=1)
        # failed candidate has no PDB → batch_gnina_rescore receives 0 paths
        assert mock_gnina.call_count <= 1
        # returned candidates count preserved
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 3. _HAS_GNINA=False → graceful skip
# ---------------------------------------------------------------------------

class TestGninaUnavailable:
    def test_no_gnina_flag_skips_quietly(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        assert len(result) == 3

    def test_gnina_skipped_no_extra_gnina_keys(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(1, ddg=-5.0)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        for cand in result:
            assert "gnina_cnn_score" not in cand.extra_scores


# ---------------------------------------------------------------------------
# 4. _HAS_PARETO=False → graceful skip
# ---------------------------------------------------------------------------

class TestParetoUnavailable:
    def test_no_pareto_flag_skips_quietly(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        assert len(result) == 3

    def test_pareto_skipped_no_pareto_rank(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(1, ddg=-5.0)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        for cand in result:
            assert "pareto_rank" not in cand.extra_scores


# ---------------------------------------------------------------------------
# 5. _HAS_BO=False → graceful skip
# ---------------------------------------------------------------------------

class TestBOUnavailable:
    def test_no_bo_flag_skips_quietly(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1, bo_optimizer=None)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# 6. Pareto ranking: extra_scores에 pareto_rank 기록
# ---------------------------------------------------------------------------

class TestParetoIntegration:
    def test_pareto_rank_added_to_extra_scores(self, tmp_path):
        fn = _import_fn()
        candidates = [
            _make_candidate(1, ddg=-12.0, clash=1.0),
            _make_candidate(2, ddg=-8.0, clash=3.0),
            _make_candidate(3, ddg=-5.0, clash=6.0),
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates", side_effect=_mock_pareto_rank):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        for cand in result:
            assert "pareto_rank" in cand.extra_scores, (
                f"{cand.candidate_id} missing pareto_rank"
            )
            assert isinstance(cand.extra_scores["pareto_rank"], int)
            assert cand.extra_scores["pareto_rank"] >= 0

    def test_best_ddg_gets_lowest_pareto_rank(self, tmp_path):
        fn = _import_fn()
        best = _make_candidate(1, ddg=-20.0, clash=0.5)
        worst = _make_candidate(2, ddg=5.0, clash=15.0)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates", side_effect=_mock_pareto_rank):
            result, _ = fn([best, worst], iter_dir=tmp_path, iteration=1)
        best_res = next(c for c in result if c.candidate_id == best.candidate_id)
        worst_res = next(c for c in result if c.candidate_id == worst.candidate_id)
        assert best_res.extra_scores["pareto_rank"] <= worst_res.extra_scores["pareto_rank"]

    def test_crowding_distance_populated(self, tmp_path):
        fn = _import_fn()
        candidates = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates", side_effect=_mock_pareto_rank):
            result, _ = fn(candidates, iter_dir=tmp_path, iteration=1)
        for cand in result:
            assert "crowding_distance" in cand.extra_scores


# ---------------------------------------------------------------------------
# 7. GNINA dry-run 모드
# ---------------------------------------------------------------------------

class TestGninaDryRun:
    def test_gnina_dry_run_when_no_pdb(self, tmp_path):
        """PDB 파일 없으면 batch 빈 리스트 → gnina 미호출."""
        fn = _import_fn()
        cand = _make_candidate(1, ddg=-8.0)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=[]) as mock_gnina:
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)
        # pdb_paths가 비어있어 batch_gnina_rescore 미호출
        mock_gnina.assert_not_called()
        assert len(result) == 1

    def test_gnina_dry_run_with_mock_pdb(self, tmp_path):
        """PDB 파일 존재 + gnina mock → dry-run 스코어 extra_scores에 저장."""
        fn = _import_fn()
        pdb_file = tmp_path / "cand_001.pdb"
        pdb_file.write_text("ATOM  ...\n")

        cand = _make_candidate(1, ddg=-8.0)
        dry_run_score = {
            "gnina_cnn_score": 0.0,
            "gnina_cnn_affinity": 0.0,
            "gnina_vina_score": 0.0,
            "gnina_dry_run": 1.0,
        }
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=[dry_run_score]):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(result) == 1
        assert result[0].extra_scores.get("gnina_cnn_score") == 0.0
        assert result[0].extra_scores.get("gnina_dry_run") == 1.0


# ---------------------------------------------------------------------------
# 8. ECR 스코어가 extra_scores에 저장
# ---------------------------------------------------------------------------

class TestECRIntegration:
    def test_ecr_score_in_extra_scores(self, tmp_path):
        fn = _import_fn()
        for i in [1, 2]:
            (tmp_path / f"cand_{i:03d}.pdb").write_text("ATOM  ...\n")

        cands = [
            _make_candidate(1, ddg=-10.0),
            _make_candidate(2, ddg=-5.0),
        ]
        gnina_scores = [
            {"gnina_cnn_score": 0.8, "gnina_cnn_affinity": 7.0, "gnina_vina_score": -8.0},
            {"gnina_cnn_score": 0.4, "gnina_cnn_affinity": 5.0, "gnina_vina_score": -6.0},
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=gnina_scores):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)

        for cand in result:
            assert "ecr_score" in cand.extra_scores, (
                f"{cand.candidate_id} missing ecr_score"
            )
            assert cand.extra_scores["ecr_score"] > 0.0

    def test_gnina_scores_stored_in_extra_scores(self, tmp_path):
        fn = _import_fn()
        (tmp_path / "cand_001.pdb").write_text("ATOM  ...\n")

        cand = _make_candidate(1, ddg=-8.0)
        gnina_score = {
            "gnina_cnn_score": 0.75,
            "gnina_cnn_affinity": 6.5,
            "gnina_vina_score": -7.2,
        }
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=[gnina_score]):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)

        assert result[0].extra_scores["gnina_cnn_score"] == pytest.approx(0.75)
        assert result[0].extra_scores["gnina_cnn_affinity"] == pytest.approx(6.5)


# ---------------------------------------------------------------------------
# 9. BO optimizer: fit/suggest 호출 검증
# ---------------------------------------------------------------------------

class TestBOIntegration:
    def test_bo_suggest_called_with_optimizer(self, tmp_path):
        fn = _import_fn()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 5)]

        mock_optimizer = MagicMock()
        mock_optimizer.suggest.return_value = [
            {"sequence": "AGCKNFFWKTFTSC", "position": 3, "mutation": "A", "acquisition_value": 0.9}
        ]

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", True):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        mock_optimizer.fit.assert_called_once()
        mock_optimizer.suggest.assert_called_once()
        assert len(result) == 4

    def test_bo_not_called_with_none_optimizer(self, tmp_path):
        fn = _import_fn()
        cands = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", True):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=None)
        assert len(result) == 3

    def test_bo_skipped_when_less_than_2_valid(self, tmp_path):
        """유효 관측치 < 2개면 BO fit을 호출하지 않음."""
        fn = _import_fn()
        cands = [_make_candidate(1, ddg=999.0, fail="failed")]

        mock_optimizer = MagicMock()
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", True):
            fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        mock_optimizer.fit.assert_not_called()

    def test_bo_skipped_when_optimizer_none(self, tmp_path):
        """2026-06-09 P1 계약 변경: BO 단계는 bo_optimizer 가 None 이면 skip 한다.
        (이전엔 _HAS_BO 플래그가 게이트였으나, scoring_pipeline 분리 후 게이트는 'optimizer
        가 전달되었는가'로 단순화. runner 가 _HAS_BO 일 때만 optimizer 를 생성·전달하므로
        BO 불가 = bo_optimizer None 전달.) None 전달 시 mock.fit 호출 안 함을 확인."""
        fn = _import_fn()
        cands = [_make_candidate(i) for i in range(1, 4)]
        mock_optimizer = MagicMock()
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=None)
        mock_optimizer.fit.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Exception resilience
# ---------------------------------------------------------------------------

class TestExceptionResilience:
    def test_gnina_exception_continues_pareto(self, tmp_path):
        fn = _import_fn()
        pdb_file = tmp_path / "cand_001.pdb"
        pdb_file.write_text("ATOM  ...\n")

        cand = _make_candidate(1, ddg=-8.0)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", side_effect=RuntimeError("gnina crash")), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates", side_effect=_mock_pareto_rank):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(result) == 1
        # GNINA 실패해도 Pareto는 실행
        assert "pareto_rank" in result[0].extra_scores

    def test_pareto_exception_continues_gracefully(self, tmp_path):
        fn = _import_fn()
        cand = _make_candidate(1, ddg=-8.0)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates", side_effect=RuntimeError("pareto crash")):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)
        assert len(result) == 1
        # Pareto 실패 → extra_scores에 pareto_rank 없음 (graceful degradation)
        assert "pareto_rank" not in result[0].extra_scores

    def test_bo_exception_continues_gracefully(self, tmp_path):
        fn = _import_fn()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]
        mock_optimizer = MagicMock()
        mock_optimizer.fit.side_effect = RuntimeError("bo crash")

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", True):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        assert len(result) == 3

    def test_ecr_exception_continues_gracefully(self, tmp_path):
        fn = _import_fn()
        (tmp_path / "cand_001.pdb").write_text("ATOM  ...\n")
        cand = _make_candidate(1, ddg=-8.0)

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore", return_value=[{"gnina_cnn_score": 0.5, "gnina_cnn_affinity": 5.0, "gnina_vina_score": -6.0}]), \
             patch("pyrosetta_flow.scoring_pipeline.exponential_rank_consensus", side_effect=RuntimeError("ecr crash")):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(result) == 1
        # ECR 실패해도 GNINA 스코어는 extra_scores에 남아있어야 함
        assert "gnina_cnn_score" in result[0].extra_scores


# ---------------------------------------------------------------------------
# 11. extra_scores 필드 기본값 테스트 (schema 검증)
# ---------------------------------------------------------------------------

class TestExtraScoresField:
    def test_extra_scores_default_empty_dict(self):
        cand = _make_candidate(1)
        assert cand.extra_scores == {}

    def test_extra_scores_independent_instances(self):
        """각 인스턴스가 독립적인 extra_scores를 가져야 함 (mutable default 공유 방지)."""
        c1 = _make_candidate(1)
        c2 = _make_candidate(2)
        c1.extra_scores["test"] = 1
        assert "test" not in c2.extra_scores


# ---------------------------------------------------------------------------
# 12. _compute_mean_pairwise_hamming 단위 테스트
# ---------------------------------------------------------------------------

class TestComputeMeanPairwiseHamming:
    """작업 1: Pareto diversity 실효화 — Hamming 계산 함수 단위 검증."""

    def _fn(self):
        from pyrosetta_flow.scoring_pipeline import _compute_mean_pairwise_hamming
        return _compute_mean_pairwise_hamming

    def test_empty_returns_zero(self):
        fn = self._fn()
        assert fn([]) == pytest.approx(0.0)

    def test_single_seq_returns_zero(self):
        fn = self._fn()
        assert fn(["AGCKNFFWKTFTSC"]) == pytest.approx(0.0)

    def test_identical_seqs_hamming_zero(self):
        fn = self._fn()
        # 동일 서열 2개 → 거리 0
        assert fn(["AGCKNFFWKTFTSC", "AGCKNFFWKTFTSC"]) == pytest.approx(0.0)

    def test_fully_different_seqs(self):
        fn = self._fn()
        # 길이 4, 전부 다름 → Hamming=4
        assert fn(["AAAA", "BBBB"]) == pytest.approx(4.0)

    def test_single_mutation_hamming_one(self):
        fn = self._fn()
        # pos0만 다름
        assert fn(["AGCKNFFWKTFTSC", "BGCKNFFWKTFTSC"]) == pytest.approx(1.0)

    def test_three_seqs_average(self):
        fn = self._fn()
        # seq1 vs seq2: 1 (pos0), seq1 vs seq3: 2 (pos0,pos1), seq2 vs seq3: 1 (pos1)
        # 평균 = (1+2+1)/3 = 4/3
        assert fn(["AAAA", "BAAA", "ABAA"]) == pytest.approx(4.0 / 3.0)

    def test_different_length_seqs(self):
        fn = self._fn()
        # "ABC" vs "ABCD": min_len=3 → 0 mismatch + |3-4|=1 → Hamming=1
        assert fn(["ABC", "ABCD"]) == pytest.approx(1.0)

    def test_return_type_float(self):
        fn = self._fn()
        result = fn(["AGCKNFFWKTFTSC", "AGCKNFFWKTFTSA"])
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 13. Pareto diversity 실제 값 반영 통합 테스트
# ---------------------------------------------------------------------------

class TestParetoDiversityIntegration:
    """작업 1: diversity 컬럼이 0.0 상수가 아닌 Hamming 거리로 채워지는지 검증."""

    def test_diversity_nonzero_for_different_seqs(self, tmp_path):
        """서로 다른 서열이면 Pareto 입력 diversity > 0."""
        fn = _import_fn()
        # 두 서열은 pos12만 다름 (T vs A)
        cand1 = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        cand2 = _make_candidate(2, seq="AGCKNFFWKTFASC", ddg=-8.0)

        captured_inputs: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured_inputs.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand1, cand2], iter_dir=tmp_path, iteration=1)

        assert len(captured_inputs) == 2
        # 두 서열이 다르므로 diversity > 0
        for entry in captured_inputs:
            assert entry["diversity"] > 0.0, (
                f"diversity should be > 0 for different sequences, got {entry['diversity']}"
            )

    def test_diversity_zero_for_single_valid_cand(self, tmp_path):
        """valid 후보 1개(fail 아님)면 다른 후보 없으므로 diversity=0."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["diversity"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 14. OOD 정책 cyclic_ss_bond 정합성 테스트
# ---------------------------------------------------------------------------

class TestCyclicOODPolicyNormal:
    """작업 2: _SCORING_OOD_POLICY['cyclic_ss_bond']가 normal로 수정됐는지 검증."""

    def test_cyclic_pareto_treatment_is_normal(self):
        from pyrosetta_flow.scoring_pipeline import _SCORING_OOD_POLICY
        policy = _SCORING_OOD_POLICY.get("cyclic_ss_bond", {})
        assert policy.get("pareto_treatment") == "normal", (
            "cyclic_ss_bond는 우리 타겟이라 pareto_treatment=normal이어야 함"
        )

    def test_cyclic_recommended_for_decision_true(self):
        from pyrosetta_flow.scoring_pipeline import _SCORING_OOD_POLICY
        policy = _SCORING_OOD_POLICY.get("cyclic_ss_bond", {})
        assert policy.get("recommended_for_decision") is True

    def test_cyclic_surrogate_reliability_normal(self):
        from pyrosetta_flow.scoring_pipeline import _SCORING_OOD_POLICY
        policy = _SCORING_OOD_POLICY.get("cyclic_ss_bond", {})
        assert policy.get("surrogate_reliability") == "normal"

    def test_d_amino_still_hard_violation(self):
        """D-AA 정책은 변경 없이 hard_violation 유지."""
        from pyrosetta_flow.scoring_pipeline import _SCORING_OOD_POLICY
        assert _SCORING_OOD_POLICY["d_amino_acid"]["pareto_treatment"] == "hard_violation"

    def test_dota_still_hard_violation(self):
        """DOTA 정책은 변경 없이 hard_violation 유지."""
        from pyrosetta_flow.scoring_pipeline import _SCORING_OOD_POLICY
        assert _SCORING_OOD_POLICY["dota_chelator"]["pareto_treatment"] == "hard_violation"


# ---------------------------------------------------------------------------
# 15. pocket_contacts soft penalty 테스트
# ---------------------------------------------------------------------------

class TestPocketContactsSoftPenalty:
    """작업 3: pocket_contacts==0이면 hard_violations=0.5 soft penalty."""

    def test_pocket_contacts_zero_gives_soft_penalty(self, tmp_path):
        """pocket_contacts=0 → Pareto 입력 hard_violations=0.5."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        cand.extra_scores["pocket_contacts"] = 0

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 1
                c["crowding_distance"] = 0.5
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["hard_violations"] == pytest.approx(0.5), (
            f"pocket_contacts=0이면 hard_violations=0.5여야 함, got {captured[0]['hard_violations']}"
        )

    def test_pocket_contacts_nonzero_no_penalty(self, tmp_path):
        """pocket_contacts > 0 → hard_violations=0 (정상)."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        cand.extra_scores["pocket_contacts"] = 5

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["hard_violations"] == 0

    def test_pocket_contacts_key_absent_graceful(self, tmp_path):
        """pocket_contacts 키 없으면 hard_violations=0 (graceful, 기존 동작)."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        # pocket_contacts 키 없음

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["hard_violations"] == 0

    def test_pocket_contacts_zero_ood_cand_stays_hard_violation(self, tmp_path):
        """OOD 후보(D-AA)에 pocket_contacts=0이 있어도 hard_violations=1 유지."""
        fn = _import_fn()
        # D-AA 소문자 서열 → is_ood=True
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSa", ddg=-10.0)
        cand.extra_scores["pocket_contacts"] = 0

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 999
                c["crowding_distance"] = 0.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["hard_violations"] == 1, (
            "OOD 후보는 pocket_contacts=0이어도 hard_violations=1이어야 함"
        )


# ---------------------------------------------------------------------------
# 16. 이슈6: GNINA dry-run enforce — dry-run 점수가 ECR에서 제외되는지 검증
# ---------------------------------------------------------------------------

class TestGninaDryRunEnforce:
    """이슈6: GNINA dry-run==1.0 후보는 ECR에서 제외 (순위에 기여 금지)."""

    def test_dry_run_excluded_from_ecr(self, tmp_path):
        """dry-run 점수(gnina_dry_run=1.0)는 ecr_score가 설정되지 않아야 한다."""
        fn = _import_fn()
        for i in [1, 2]:
            (tmp_path / f"cand_{i:03d}.pdb").write_text("ATOM  ...\n")

        cands = [
            _make_candidate(1, ddg=-10.0),  # dry-run 후보
            _make_candidate(2, ddg=-5.0),   # dry-run 후보
        ]
        dry_run_scores = [
            {"gnina_cnn_score": 0.0, "gnina_cnn_affinity": 0.0,
             "gnina_vina_score": 0.0, "gnina_dry_run": 1.0},
            {"gnina_cnn_score": 0.0, "gnina_cnn_affinity": 0.0,
             "gnina_vina_score": 0.0, "gnina_dry_run": 1.0},
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline._GNINA_DRY_RUN_ENFORCE", True), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore",
                   return_value=dry_run_scores):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)

        # dry-run enforce=True → ecr_score가 설정되지 않아야 함 (ECR 제외)
        for cand in result:
            assert "ecr_score" not in cand.extra_scores, (
                f"{cand.candidate_id}: dry-run 후보는 ecr_score 없어야 함 "
                f"(got {cand.extra_scores.get('ecr_score')})"
            )
        # gnina 점수는 extra_scores에 기록됨 (추적용)
        assert result[0].extra_scores.get("gnina_dry_run") == 1.0

    def test_live_gnina_keeps_ecr(self, tmp_path):
        """live GNINA(gnina_dry_run 없음) 후보는 ecr_score가 설정되어야 한다."""
        fn = _import_fn()
        for i in [1, 2]:
            (tmp_path / f"cand_{i:03d}.pdb").write_text("ATOM  ...\n")

        cands = [
            _make_candidate(1, ddg=-10.0),
            _make_candidate(2, ddg=-5.0),
        ]
        live_scores = [
            {"gnina_cnn_score": 0.85, "gnina_cnn_affinity": 6.5, "gnina_vina_score": -8.0},
            {"gnina_cnn_score": 0.60, "gnina_cnn_affinity": 5.0, "gnina_vina_score": -6.5},
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline._GNINA_DRY_RUN_ENFORCE", True), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore",
                   return_value=live_scores):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)

        # live GNINA → ecr_score 계산됨
        for cand in result:
            assert "ecr_score" in cand.extra_scores, (
                f"{cand.candidate_id}: live GNINA 후보는 ecr_score 있어야 함"
            )
            assert cand.extra_scores["ecr_score"] > 0.0

    def test_mixed_dry_live_only_live_in_ecr(self, tmp_path):
        """dry-run + live 혼합 시, live 후보만 ECR에 포함된다."""
        fn = _import_fn()
        for i in [1, 2]:
            (tmp_path / f"cand_{i:03d}.pdb").write_text("ATOM  ...\n")

        cands = [
            _make_candidate(1, ddg=-10.0),  # live
            _make_candidate(2, ddg=-5.0),   # dry-run
        ]
        mixed_scores = [
            {"gnina_cnn_score": 0.85, "gnina_cnn_affinity": 6.5, "gnina_vina_score": -8.0},
            {"gnina_cnn_score": 0.0, "gnina_cnn_affinity": 0.0,
             "gnina_vina_score": 0.0, "gnina_dry_run": 1.0},
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline._GNINA_DRY_RUN_ENFORCE", True), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore",
                   return_value=mixed_scores):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)

        # cand_001(live) → ecr_score 있음, cand_002(dry) → ecr_score 없음
        live_cand = next(c for c in result if "001" in c.candidate_id)
        dry_cand = next(c for c in result if "002" in c.candidate_id)
        assert "ecr_score" in live_cand.extra_scores, "live 후보는 ecr_score 있어야 함"
        assert "ecr_score" not in dry_cand.extra_scores, "dry-run 후보는 ecr_score 없어야 함"

    def test_enforce_disabled_allows_dry_run_ecr(self, tmp_path):
        """GNINA_DRY_RUN_ENFORCE=False 시 dry-run도 ECR에 포함(하위호환)."""
        fn = _import_fn()
        for i in [1, 2]:
            (tmp_path / f"cand_{i:03d}.pdb").write_text("ATOM  ...\n")

        cands = [
            _make_candidate(1, ddg=-10.0),
            _make_candidate(2, ddg=-5.0),
        ]
        dry_run_scores = [
            {"gnina_cnn_score": 0.0, "gnina_cnn_affinity": 0.0,
             "gnina_vina_score": 0.0, "gnina_dry_run": 1.0},
            {"gnina_cnn_score": 0.0, "gnina_cnn_affinity": 0.0,
             "gnina_vina_score": 0.0, "gnina_dry_run": 1.0},
        ]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", True), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.scoring_pipeline._GNINA_DRY_RUN_ENFORCE", False), \
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore",
                   return_value=dry_run_scores):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)

        # enforce 비활성화 → 이전 동작: ecr_score 설정됨
        for cand in result:
            assert "ecr_score" in cand.extra_scores, (
                f"enforce 꺼짐 → dry-run도 ecr_score 포함(하위호환)"
            )


# ---------------------------------------------------------------------------
# 17. 이슈9: half-life source enforce — halflife_source='none'이면 stability=0
# ---------------------------------------------------------------------------

class TestHalflifeSourceEnforce:
    """이슈9: halflife_source='none'(모든 추정기 실패) → Pareto stability 기여 0.

    cheap_objectives(Step 0)가 실행되면 halflife_source를 덮어쓰므로,
    Step 0를 mock하여 원하는 halflife_source가 Pareto 단계에 도달하도록 한다.
    """

    def _mock_cheap_objectives_with_source(self, source: str, stability_norm: float):
        """cheap_objectives를 mock하여 지정한 halflife_source/stability_norm을 반환."""
        def _mock(seq, reference_seq="AGCKNFFWKTFTSC", cand=None):
            return {
                "sequence": seq,
                "half_life_h": 0.0 if source == "none" else 10.0,
                "stability_norm": stability_norm,
                "halflife_source": source,
                "admet_score": 0.5,
                "gravy": 0.0,
                "boman_index": 2.0,
                "instability_index": 30.0,
                "aliphatic_index": 100.0,
                "pi": 7.0,
                "radiolysis_score": None,
                "radiolysis_risk_level": None,
            }
        return _mock

    def test_halflife_source_none_sets_stability_zero(self, tmp_path):
        """halflife_source='none'이면 Pareto 입력 stability=0.0이어야 한다."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        mock_cheap = self._mock_cheap_objectives_with_source("none", 0.5)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", side_effect=mock_cheap), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["stability"] == pytest.approx(0.0), (
            f"halflife_source=none → stability=0.0이어야 함, got {captured[0]['stability']}"
        )
        # enforce 기록 플래그 확인
        assert cand.extra_scores.get("stability_norm_enforced") is not None

    def test_halflife_source_heuristic_uses_value(self, tmp_path):
        """halflife_source='heuristic'이면 stability_norm 값 그대로 사용한다."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        mock_cheap = self._mock_cheap_objectives_with_source("heuristic", 0.72)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", side_effect=mock_cheap), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["stability"] == pytest.approx(0.72), (
            f"halflife_source=heuristic → 실측값 0.72 사용, got {captured[0]['stability']}"
        )

    def test_halflife_source_ensemble_uses_value(self, tmp_path):
        """halflife_source='ensemble'이면 stability_norm 값 그대로 사용한다."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)

        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        mock_cheap = self._mock_cheap_objectives_with_source("ensemble", 0.85)
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", side_effect=mock_cheap), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        assert captured[0]["stability"] == pytest.approx(0.85)

    def test_halflife_source_absent_uses_clash_proxy(self, tmp_path):
        """halflife_source 자체가 없으면(Step 0 skip) clash proxy 사용."""
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0, clash=5.0)
        # Step 0 전체 skip — halflife_source가 extra_scores에 설정되지 않도록 cheap_objectives 실패 mock
        captured: list = []

        def capture_pareto(candidates, clash_threshold=10.0):
            captured.extend(candidates)
            for c in candidates:
                c["pareto_rank"] = 0
                c["crowding_distance"] = 1.0
            return candidates

        def raise_cheap(seq, reference_seq="AGCKNFFWKTFTSC", cand=None):
            raise RuntimeError("cheap-objectives forced skip")

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.runner._HAS_BO", False), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", side_effect=raise_cheap), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=capture_pareto):
            fn([cand], iter_dir=tmp_path, iteration=1)

        assert len(captured) == 1
        # halflife_source 없음 + stability_norm 없음 → clash proxy: (40-5)/40 = 0.875
        expected = max(0.0, 40.0 - 5.0) / 40.0
        assert captured[0]["stability"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 18. 이슈10: ADMET available=False/hc50_reliable=False 처리가 이미 안전한지 확인
# ---------------------------------------------------------------------------

class TestAdmetEnforceAlreadySafe:
    """이슈10: apply_toxicity_to_extra의 available=False/hc50_reliable=False 처리 확인.
    이미 안전하게 처리되어 있음을 회귀 테스트로 고정한다.
    """

    def test_available_false_does_not_change_admet_score(self):
        """available=False → admet_score 변경 없음 (fail-closed)."""
        from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
        extra = {"admet_score": 0.75}
        apply_toxicity_to_extra(extra, {"available": False, "is_toxic": True, "hc50": -200.0})
        assert extra["admet_score"] == pytest.approx(0.75), (
            "available=False → admet_score 변경 금지 (가짜 안전판정 방지)"
        )
        assert "pepadmet_toxic" not in extra, "available=False → 필드 기록도 없어야 함"

    def test_hc50_reliable_false_skips_penalty(self):
        """hc50_reliable=False(cyclic fallback) → 페널티 없음."""
        from pyrosetta_flow.multiobjective import apply_toxicity_to_extra
        extra = {"admet_score": 0.75}
        apply_toxicity_to_extra(extra, {
            "available": True, "is_toxic": True,
            "hc50": -300.0, "graph_note": "linear_sequence_fallback"
        })
        # hc50_reliable=False → more_toxic_than_native 설정 없음
        assert "more_toxic_than_native" not in extra, (
            "hc50_reliable=False → 페널티 게이트 미적용"
        )
        assert extra["admet_score"] == pytest.approx(0.75), (
            "신뢰불가 hc50은 admet_score에 페널티 없음"
        )
        assert extra.get("hc50_reliable") is False
        assert "hc50_ood_warning" in extra

    def test_pepadmet_skip_does_not_apply_penalty(self, tmp_path):
        """pepADMET subprocess 전체 skip 시 독성 페널티 없음(admet_score에 추가 감점 없음).

        Step 0 cheap_objectives는 실행되므로 physicochemical admet_score는 재계산됨.
        확인 포인트: pepADMET skip 후 admet_score가 Step 0 계산값 그대로이고
        is_toxic/pepadmet_hc50 등 pepADMET 전용 키가 없어야 함.
        """
        fn = _import_fn()
        cand = _make_candidate(1, seq="AGCKNFFWKTFTSC", ddg=-10.0)
        # pepADMET는 SST_DISABLE_PEPADMET_TOX=1로 이미 비활성화 (conftest 최상단)
        # GNINA/Pareto 모두 skip
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.runner._HAS_BO", False):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)

        # pepADMET skip → pepADMET 전용 키 없음
        assert "pepadmet_toxic" not in result[0].extra_scores, (
            "pepADMET skip → pepadmet_toxic 키 없어야 함"
        )
        assert "pepadmet_hc50" not in result[0].extra_scores, (
            "pepADMET skip → pepadmet_hc50 키 없어야 함"
        )
        # admet_score는 Step 0 physicochemical 계산값이 있어야 함 (None 아님)
        admet = result[0].extra_scores.get("admet_score")
        assert admet is not None and 0.0 <= admet <= 1.0, (
            f"pepADMET skip → physicochemical admet_score 유지({admet})"
        )
