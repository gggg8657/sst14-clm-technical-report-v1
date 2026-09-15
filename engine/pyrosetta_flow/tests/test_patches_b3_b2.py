"""Tests for B3 OOD gate and B2 BO guidance connection patches.

B3: _detect_ood_flags() 단위 테스트 + scoring_pipeline OOD 통합
B2: _apply_alternative_scoring Tuple 반환 + bo_suggested_positions guidance 전달

주의: pepADMET 독성(실제 subprocess)은 비활성화.
"""
from __future__ import annotations

import os
os.environ.setdefault("SST_DISABLE_PEPADMET_TOX", "1")

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

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


def _import_scoring():
    from pyrosetta_flow.scoring_pipeline import _apply_alternative_scoring
    return _apply_alternative_scoring


def _import_detect():
    from pyrosetta_flow.scoring_pipeline import _detect_ood_flags
    return _detect_ood_flags


def _mock_pareto_rank(candidates, clash_threshold=10.0):
    sorted_idx = sorted(range(len(candidates)), key=lambda i: candidates[i].get("ddG", 0))
    for rank, idx in enumerate(sorted_idx):
        candidates[idx]["pareto_rank"] = rank
        candidates[idx]["crowding_distance"] = float(len(candidates) - rank)
    return candidates


# ===========================================================================
# B3: _detect_ood_flags 단위 테스트
# ===========================================================================

class TestDetectOODFlags:
    """B3: _detect_ood_flags() — OOD 감지 로직 검증."""

    def test_l_aa_sequence_no_daa_dota_ood(self):
        """순수 L-AA 대문자 서열(Cys 없음) → D-AA/DOTA OOD 없음."""
        detect = _import_detect()
        # Cys 없는 서열: D-AA/DOTA OOD 없음 → is_ood=False
        result = detect("AGAKNFFWKTFTSA")
        assert result["is_ood"] is False
        assert result["ood_reasons"] == []
        assert result["recommended_for_decision"] is True

    def test_sst14_reference_cys_pattern_is_ood(self):
        """SST-14 참조 서열(Cys3-Cys14)은 cyclic이지만 OOD 아님 (도메인 보정 2026-06-23).

        현 정책: cyclic SS-bond는 OOD 트리거에서 제외됨.
        우리 타겟이 cyclic SST-14이므로 cyclic을 OOD로 판정하면 핵심 후보가 모두 배제됨.
        OOD는 D-AA(소문자)/DOTA 변형에만 적용.
        """
        detect = _import_detect()
        # SST-14: AGCKNFFWKTFTSC — 순수 대문자 L-AA, D-AA/DOTA 없음 → OOD 아님
        result = detect("AGCKNFFWKTFTSC")
        assert result["is_ood"] is False
        assert result["ood_reasons"] == []
        assert result["recommended_for_decision"] is True

    def test_d_aa_lowercase_token_is_ood(self):
        """소문자 토큰 포함 서열 → D-AA 감지 → is_ood=True."""
        detect = _import_detect()
        # 소문자 'a' = D-Ala 관례
        result = detect("AGCKNFFWKTFTSa")
        assert result["is_ood"] is True
        assert any("D-AA" in r for r in result["ood_reasons"])
        assert result["recommended_for_decision"] is False

    def test_dota_flag_in_extra_scores_is_ood(self):
        """extra_scores에 has_dota=True → DOTA OOD."""
        detect = _import_detect()
        result = detect("AGCKNFFWKTFTSC", extra_scores={"has_dota": True})
        assert result["is_ood"] is True
        assert any("DOTA" in r for r in result["ood_reasons"])

    def test_dota_chelator_key_is_ood(self):
        """extra_scores에 dota_chelator=True → DOTA OOD."""
        detect = _import_detect()
        result = detect("AGCKNFFWKTFTSC", extra_scores={"dota_chelator": True})
        assert result["is_ood"] is True

    def test_smiles_ss_bond_is_ood(self):
        """extra_scores.smiles에 'SS' 포함 — SMILES SS-bond는 현 정책에서 OOD 트리거 아님.

        현 정책 (도메인 보정 2026-06-23): cyclic SS-bond는 OOD에서 제외.
        smiles에 SS가 포함된 경우도 D-AA/DOTA 없이는 OOD 판정 안 됨.
        """
        detect = _import_detect()
        result = detect("AGCKNFFWKTFTSC", extra_scores={"smiles": "C(C)SSC(C)"})
        # SMILES SS-bond만으로는 OOD 아님 — D-AA/DOTA 기준만 적용
        assert result["is_ood"] is False
        assert result["recommended_for_decision"] is True

    def test_cys_heuristic_sst14_pattern_is_ood(self):
        """Cys3 + Cys14(SST-14 패턴)은 cyclic이지만 OOD 아님 (도메인 보정 2026-06-23).

        현 정책: cyclic Cys-Cys heuristic은 OOD 트리거에서 제외됨.
        D-AA/DOTA가 없으면 Cys 패턴만으로 OOD 판정하지 않음.
        """
        detect = _import_detect()
        # SST-14: AGCKNFFWKTFTSC — Cys3-Cys14 cyclic, 대문자 L-AA, D-AA/DOTA 없음
        result = detect("AGCKNFFWKTFTSC")
        assert result["is_ood"] is False
        assert result["ood_reasons"] == []
        assert result["recommended_for_decision"] is True

    def test_single_cys_not_ood_via_heuristic(self):
        """Cys 1개만 있으면 heuristic SS-bond 미발동."""
        detect = _import_detect()
        # C 1개 → gap 계산 불가
        result = detect("AGAKNFFWKTFTSA")
        assert result["is_ood"] is False

    def test_multiple_ood_reasons_accumulated(self):
        """D-AA + DOTA 동시 → 복수 사유 누적."""
        detect = _import_detect()
        result = detect("AGCKNFFWKTFTSa", extra_scores={"has_dota": True})
        assert result["is_ood"] is True
        assert len(result["ood_reasons"]) >= 2

    def test_none_extra_scores_safe(self):
        """extra_scores=None 전달 시 오류 없음."""
        detect = _import_detect()
        result = detect("AGCKNFFWKTFTSC", extra_scores=None)
        # SS-bond heuristic은 발동 가능 — 오류 없어야 함
        assert isinstance(result["is_ood"], bool)

    def test_empty_sequence_not_ood(self):
        """빈 서열 → is_ood=False (예외 없음)."""
        detect = _import_detect()
        result = detect("")
        assert result["is_ood"] is False


# ===========================================================================
# B3: scoring_pipeline 통합 — OOD 플래그 기록 및 surrogate 감쇠
# ===========================================================================

class TestOODGatePipelineIntegration:
    """B3: Step 0b OOD 게이트 통합 테스트."""

    def test_ood_flag_recorded_in_extra_scores(self, tmp_path):
        """모든 후보에 is_ood 키가 extra_scores에 기록된다."""
        fn = _import_scoring()
        cands = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result, _ = fn(cands, iter_dir=tmp_path, iteration=1)
        for cand in result:
            assert "is_ood" in cand.extra_scores
            assert "ood_reasons" in cand.extra_scores
            assert "recommended_for_decision" in cand.extra_scores

    def test_l_aa_uppercase_recommended_true(self, tmp_path):
        """L-AA 대문자(SST-14 참조 서열 변이) → recommended_for_decision 판정."""
        fn = _import_scoring()
        # SST-14 참조 서열 — Cys3/14 heuristic 발동으로 is_ood=True 가능 (설계 의도)
        # 중요: 대문자 non-Cys 서열은 D-AA/DOTA OOD 아님
        cand = _make_candidate(1, seq="AGAKNFFWKTFTSA")  # Cys 없음
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result, _ = fn([cand], iter_dir=tmp_path, iteration=1)
        assert result[0].extra_scores["is_ood"] is False
        assert result[0].extra_scores["recommended_for_decision"] is True

    def test_d_aa_surrogate_damped(self, tmp_path):
        """D-AA 후보(소문자 토큰)의 stability_norm, admet_score가 감쇠된다."""
        fn = _import_scoring()
        d_aa_cand = _make_candidate(1, seq="AGCKNFFWKTFTSa")  # D-Ala

        # cheap_objectives mock: stability_norm=0.8, admet_score=0.9 반환
        mock_obj = {
            "stability_norm": 0.8,
            "admet_score": 0.9,
            "half_life_h": 4.0,
        }
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.scoring_pipeline._OOD_DAMPING", 0.5), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", return_value=mock_obj):
            result, _ = fn([d_aa_cand], iter_dir=tmp_path, iteration=1)

        cand = result[0]
        assert cand.extra_scores["is_ood"] is True
        # stability_norm, admet_score가 0.5배로 감쇠되어야 함
        assert cand.extra_scores.get("stability_norm_ood_damped") is True
        assert cand.extra_scores.get("admet_score_ood_damped") is True
        assert cand.extra_scores["stability_norm"] == pytest.approx(0.8 * 0.5)
        assert cand.extra_scores["admet_score"] == pytest.approx(0.9 * 0.5)

    def test_ood_candidate_pareto_hard_violation(self, tmp_path):
        """OOD 후보는 Pareto hard_violations=1 → pareto_rank != 0 (front-0 배제)."""
        fn = _import_scoring()
        ood_cand = _make_candidate(1, seq="AGCKNFFWKTFTSa", ddg=-15.0)  # D-AA, 좋은 ddG
        ok_cand = _make_candidate(2, seq="AGAKNFFWKTFTSA", ddg=-5.0)    # L-AA, 나쁜 ddG

        # Pareto mock: hard_violations=1이면 rank=999 반환
        def pareto_with_violation(candidates, clash_threshold=10.0):
            for cand in candidates:
                if cand.get("hard_violations", 0) == 1:
                    cand["pareto_rank"] = 999
                    cand["crowding_distance"] = 0.0
                else:
                    cand["pareto_rank"] = 0
                    cand["crowding_distance"] = 1.0
            return candidates

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=pareto_with_violation):
            result, _ = fn([ood_cand, ok_cand], iter_dir=tmp_path, iteration=1)

        ood_res = next(c for c in result if "a" in c.sequence)
        ok_res = next(c for c in result if "a" not in c.sequence)
        # OOD 후보는 rank=999 (front-0 배제)
        assert ood_res.extra_scores.get("pareto_rank", 0) != 0
        # 정상 후보는 rank=0
        assert ok_res.extra_scores.get("pareto_rank") == 0

    def test_non_ood_surrogate_not_damped(self, tmp_path):
        """OOD 아닌 후보의 surrogate 값은 감쇠되지 않는다."""
        fn = _import_scoring()
        ok_cand = _make_candidate(1, seq="AGAKNFFWKTFTSA")  # L-AA, Cys 없음

        mock_obj = {"stability_norm": 0.7, "admet_score": 0.8, "half_life_h": 3.0}
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False), \
             patch("pyrosetta_flow.multiobjective.cheap_objectives", return_value=mock_obj):
            result, _ = fn([ok_cand], iter_dir=tmp_path, iteration=1)

        assert result[0].extra_scores.get("stability_norm_ood_damped") is not True
        assert result[0].extra_scores.get("admet_score_ood_damped") is not True
        assert result[0].extra_scores["stability_norm"] == pytest.approx(0.7)


# ===========================================================================
# B2: _apply_alternative_scoring Tuple 반환
# ===========================================================================

class TestBOTupleReturn:
    """B2: _apply_alternative_scoring가 Tuple[List, List[int]] 반환."""

    def test_returns_tuple(self, tmp_path):
        """반환값이 (list, list) 형태여야 한다."""
        fn = _import_scoring()
        cands = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result = fn(cands, iter_dir=tmp_path, iteration=1)
        assert isinstance(result, tuple), "반환값이 tuple이어야 함"
        assert len(result) == 2
        candidates_out, bo_positions = result
        assert isinstance(candidates_out, list)
        assert isinstance(bo_positions, list)

    def test_empty_candidates_returns_empty_tuple(self, tmp_path):
        """빈 candidates → ([], []) 반환."""
        fn = _import_scoring()
        result = fn([], iter_dir=tmp_path, iteration=1)
        assert result == ([], [])

    def test_no_bo_optimizer_returns_empty_positions(self, tmp_path):
        """bo_optimizer=None → bo_positions 빈 리스트."""
        fn = _import_scoring()
        cands = [_make_candidate(i) for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            _, bo_positions = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=None)
        assert bo_positions == []

    def test_bo_suggest_positions_returned(self, tmp_path):
        """bo_optimizer.suggest()가 포지션 반환 시 bo_positions에 수집된다."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 5)]

        mock_optimizer = MagicMock()
        mock_optimizer.suggest.return_value = [
            {"sequence": "AGCKNFFWKTFTSC", "position": 3, "mutation": "A", "acquisition_value": 0.9},
            {"sequence": "AGCKNFFWKTFTSC", "position": 7, "mutation": "K", "acquisition_value": 0.7},
            {"sequence": "AGCKNFFWKTFTSC", "position": 11, "mutation": "S", "acquisition_value": 0.5},
        ]

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            _, bo_positions = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        assert bo_positions == [3, 7, 11]

    def test_bo_suggest_partial_none_positions_filtered(self, tmp_path):
        """suggest()에서 position=None이 포함되면 필터링된다."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]

        mock_optimizer = MagicMock()
        mock_optimizer.suggest.return_value = [
            {"position": 5, "acquisition_value": 0.8},
            {"position": None, "acquisition_value": 0.6},  # None 포지션
            {"position": 9, "acquisition_value": 0.4},
        ]

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            _, bo_positions = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        assert None not in bo_positions
        assert 5 in bo_positions
        assert 9 in bo_positions

    def test_bo_acquisition_value_in_extra_scores(self, tmp_path):
        """suggest() 성공 시 bo_acquisition_value가 첫 유효 후보 extra_scores에 기록된다."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]

        mock_optimizer = MagicMock()
        mock_optimizer.suggest.return_value = [
            {"position": 3, "acquisition_value": 0.95}
        ]

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result_cands, _ = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        # 첫 유효 후보에 bo_acquisition_value 기록
        acq_vals = [c.extra_scores.get("bo_acquisition_value") for c in result_cands]
        assert any(v == pytest.approx(0.95) for v in acq_vals if v is not None)

    def test_bo_exception_returns_empty_positions(self, tmp_path):
        """BO suggest 예외 발생 시 bo_positions 빈 리스트 (graceful)."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]

        mock_optimizer = MagicMock()
        mock_optimizer.fit.side_effect = RuntimeError("bo crash")

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result_cands, bo_positions = fn(cands, iter_dir=tmp_path, iteration=1,
                                             bo_optimizer=mock_optimizer)

        assert bo_positions == []
        assert len(result_cands) == 3  # 후보는 보존

    def test_bo_suggest_empty_returns_empty_positions(self, tmp_path):
        """suggest()가 빈 리스트 반환 시 bo_positions 빈 리스트."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i)) for i in range(1, 4)]

        mock_optimizer = MagicMock()
        mock_optimizer.suggest.return_value = []

        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            _, bo_positions = fn(cands, iter_dir=tmp_path, iteration=1, bo_optimizer=mock_optimizer)

        assert bo_positions == []


# ===========================================================================
# 기존 테스트 하위호환 확인 — Tuple 반환으로 인한 언팩 확인
# ===========================================================================

class TestBackwardCompatibility:
    """B2 반환 타입 변경 후 기존 동작 보존 확인."""

    def test_candidates_list_unchanged_count(self, tmp_path):
        """Tuple 반환 후에도 candidates 리스트 길이 보존."""
        fn = _import_scoring()
        cands = [_make_candidate(i) for i in range(1, 6)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result_cands, _ = fn(cands, iter_dir=tmp_path, iteration=1)
        assert len(result_cands) == 5

    def test_failed_candidate_preserved(self, tmp_path):
        """fail_reason 있는 후보도 반환 리스트에 포함된다."""
        fn = _import_scoring()
        ok = _make_candidate(1, ddg=-10.0)
        failed = _make_candidate(2, ddg=999.0, fail="dock error")
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", False):
            result_cands, _ = fn([ok, failed], iter_dir=tmp_path, iteration=1)
        assert len(result_cands) == 2

    def test_pareto_rank_still_recorded_after_b2(self, tmp_path):
        """B2 패치 후에도 pareto_rank가 extra_scores에 기록된다."""
        fn = _import_scoring()
        cands = [_make_candidate(i, ddg=float(-5 - i), seq="AGAKNFFWKTFTSA") for i in range(1, 4)]
        with patch("pyrosetta_flow.scoring_pipeline._HAS_GNINA", False), \
             patch("pyrosetta_flow.scoring_pipeline._HAS_PARETO", True), \
             patch("pyrosetta_flow.scoring_pipeline.pareto_rank_candidates",
                   side_effect=_mock_pareto_rank):
            result_cands, _ = fn(cands, iter_dir=tmp_path, iteration=1)
        for cand in result_cands:
            assert "pareto_rank" in cand.extra_scores

    def test_gnina_scores_preserved_after_b2(self, tmp_path):
        """B2 패치 후에도 GNINA 스코어가 extra_scores에 저장된다."""
        fn = _import_scoring()
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
             patch("pyrosetta_flow.scoring_pipeline.batch_gnina_rescore",
                   return_value=[dry_run_score]):
            result_cands, _ = fn([cand], iter_dir=tmp_path, iteration=1)
        assert result_cands[0].extra_scores.get("gnina_dry_run") == 1.0
