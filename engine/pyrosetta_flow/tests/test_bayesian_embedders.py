"""PropertyEmbedder / BLOSUM62Embedder 단위 테스트.

검증 항목:
- embed() 출력 차원
- embed_batch() 출력 shape
- 정규화 범위 [0,1]
- 물리화학 속성 단순 논리 검증 (예: Trp > Ala 소수성 아니지만 부피는 큼)
- 기존 OneHotEmbedder 하위 호환 (변경 없음 확인)
- BayesianPeptideOptimizer와 PropertyEmbedder 연동 fit/suggest 스모크 테스트
- _BOTORCH_AVAILABLE 상태 확인
"""
from __future__ import annotations

import numpy as np
import pytest

from pyrosetta_flow.bayesian_optimizer import (
    AMINO_ACIDS,
    AA_PROP_DIM,
    AA_PROPERTIES,
    BLOSUM62Embedder,
    BayesianPeptideOptimizer,
    OneHotEmbedder,
    PropertyEmbedder,
    _BOTORCH_AVAILABLE,
    _BLOSUM62_MATRIX,
)

SST14 = "AGCKNFFWKTFTSC"  # SST-14 기준 서열 (14aa)


# ---------------------------------------------------------------------------
# 환경 메타 테스트
# ---------------------------------------------------------------------------

def test_botorch_available_flag_is_bool() -> None:
    """_BOTORCH_AVAILABLE 이 bool 타입이어야 한다."""
    assert isinstance(_BOTORCH_AVAILABLE, bool)


def test_botorch_available_reported() -> None:
    """_BOTORCH_AVAILABLE 값을 로깅 (단순 출력, 항상 통과)."""
    print(f"\n[INFO] _BOTORCH_AVAILABLE = {_BOTORCH_AVAILABLE}")


# ---------------------------------------------------------------------------
# AA_PROPERTIES 테이블 무결성
# ---------------------------------------------------------------------------

class TestAAPropertiesTable:
    def test_all_standard_aa_present(self) -> None:
        """20종 표준 AA가 모두 AA_PROPERTIES에 존재해야 한다."""
        for aa in AMINO_ACIDS:
            assert aa in AA_PROPERTIES, f"{aa} missing from AA_PROPERTIES"

    def test_each_entry_has_7_dims(self) -> None:
        """각 AA 항목이 정확히 7차원 벡터여야 한다."""
        for aa, props in AA_PROPERTIES.items():
            assert len(props) == AA_PROP_DIM, (
                f"{aa}: expected {AA_PROP_DIM} dims, got {len(props)}"
            )

    def test_aromaticity_correct_residues(self) -> None:
        """방향족 잔기 (F, H, W, Y)만 aromaticity=1.0이어야 한다."""
        aromatic = {"F", "H", "W", "Y"}
        for aa, props in AA_PROPERTIES.items():
            expected = 1.0 if aa in aromatic else 0.0
            assert props[3] == expected, (
                f"{aa}: aromaticity={props[3]}, expected={expected}"
            )

    def test_charged_residues(self) -> None:
        """전하 검증: D/E = -1, K/R = +1, 나머지 = 0."""
        neg = {"D", "E"}
        pos = {"K", "R"}
        for aa, props in AA_PROPERTIES.items():
            charge = props[0]
            if aa in neg:
                assert charge == -1.0, f"{aa}: charge={charge}"
            elif aa in pos:
                assert charge == 1.0, f"{aa}: charge={charge}"
            else:
                assert charge == 0.0, f"{aa}: charge={charge}"

    def test_volume_norm_trp_max(self) -> None:
        """Trp(W)이 가장 큰 부피(vol_norm=1.0)여야 한다."""
        assert AA_PROPERTIES["W"][2] == pytest.approx(1.0), (
            f"W volume_norm={AA_PROPERTIES['W'][2]}"
        )

    def test_volume_norm_nonneg(self) -> None:
        """모든 부피 정규화 값이 0 이상이어야 한다."""
        for aa, props in AA_PROPERTIES.items():
            assert props[2] >= 0.0, f"{aa}: volume_norm={props[2]} < 0"


# ---------------------------------------------------------------------------
# PropertyEmbedder
# ---------------------------------------------------------------------------

class TestPropertyEmbedder:
    def test_embed_dim_with_max_len(self) -> None:
        """max_len=14 → 출력 차원 = 14 * 7 = 98."""
        emb = PropertyEmbedder(max_len=14)
        vec = emb.embed(SST14)
        assert vec.shape == (98,), f"shape={vec.shape}"

    def test_embed_dim_without_max_len(self) -> None:
        """max_len=None → 출력 차원 = len(seq) * 7."""
        emb = PropertyEmbedder()
        seq = "AGCK"
        vec = emb.embed(seq)
        assert vec.shape == (len(seq) * AA_PROP_DIM,), f"shape={vec.shape}"

    def test_embed_dtype_float32(self) -> None:
        """embed() 반환 dtype이 float32여야 한다."""
        emb = PropertyEmbedder(max_len=14)
        vec = emb.embed(SST14)
        assert vec.dtype == np.float32

    def test_embed_batch_shape(self) -> None:
        """embed_batch() 결과 shape = (n, 98)."""
        emb = PropertyEmbedder(max_len=14)
        seqs = [SST14, "AGCKNFFWKTFTSA", "AGCKNFFWKTFTSV"]
        mat = emb.embed_batch(seqs)
        assert mat.shape == (3, 98), f"shape={mat.shape}"

    def test_normalize_range(self) -> None:
        """normalize=True 시 모든 임베딩 값이 [0,1] 범위여야 한다."""
        emb = PropertyEmbedder(max_len=14, normalize=True)
        # 모든 20종 AA 포함 배치
        seqs = [aa * 14 for aa in AMINO_ACIDS]
        mat = emb.embed_batch(seqs)
        assert mat.min() >= -1e-6, f"min={mat.min()}"
        assert mat.max() <= 1.0 + 1e-6, f"max={mat.max()}"

    def test_normalize_false_raw_scale(self) -> None:
        """normalize=False 시 전하 속성이 원본 스케일(-1~+1)로 나와야 한다."""
        emb = PropertyEmbedder(max_len=1, normalize=False)
        # Arg(R): formal_charge=+1
        vec_R = emb.embed("R")
        assert vec_R[0] == pytest.approx(1.0), f"R charge={vec_R[0]}"
        # Asp(D): formal_charge=-1
        vec_D = emb.embed("D")
        assert vec_D[0] == pytest.approx(-1.0), f"D charge={vec_D[0]}"

    def test_unknown_aa_zero_vector(self) -> None:
        """비표준 잔기('X')는 0 벡터로 처리되어야 한다."""
        emb = PropertyEmbedder(max_len=1)
        vec = emb.embed("X")
        assert np.all(vec == 0.0), f"X embedding should be all zeros, got {vec}"

    def test_padding_zero(self) -> None:
        """max_len보다 짧은 시퀀스의 패딩 부분이 0이어야 한다."""
        emb = PropertyEmbedder(max_len=14)
        vec = emb.embed("A")  # 1잔기
        # 2번째 잔기 이후 (인덱스 7~98)가 모두 0
        assert np.all(vec[AA_PROP_DIM:] == 0.0), "padding should be 0"

    def test_n_properties(self) -> None:
        """n_properties property가 7을 반환해야 한다."""
        emb = PropertyEmbedder()
        assert emb.n_properties == 7

    def test_prop_names_length(self) -> None:
        """prop_names가 7개 이름을 반환해야 한다."""
        emb = PropertyEmbedder()
        assert len(emb.prop_names) == 7

    def test_two_similar_aa_closer_than_dissimilar(self) -> None:
        """물리화학 유사 AA 쌍(I vs L)이 이질 쌍(R vs G)보다 L2 거리가 작아야 한다."""
        emb = PropertyEmbedder(max_len=1, normalize=True)
        v_I = emb.embed("I")
        v_L = emb.embed("L")
        v_R = emb.embed("R")
        v_G = emb.embed("G")
        dist_IL = float(np.linalg.norm(v_I - v_L))
        dist_RG = float(np.linalg.norm(v_R - v_G))
        assert dist_IL < dist_RG, (
            f"I-L dist={dist_IL:.4f} should < R-G dist={dist_RG:.4f}"
        )


# ---------------------------------------------------------------------------
# BLOSUM62Embedder
# ---------------------------------------------------------------------------

class TestBLOSUM62Embedder:
    def test_embed_dim_with_max_len(self) -> None:
        """max_len=14 → 출력 차원 = 14 * 20 = 280."""
        emb = BLOSUM62Embedder(max_len=14)
        vec = emb.embed(SST14)
        assert vec.shape == (280,), f"shape={vec.shape}"

    def test_embed_dtype_float32(self) -> None:
        """embed() 반환 dtype이 float32여야 한다."""
        emb = BLOSUM62Embedder(max_len=14)
        vec = emb.embed(SST14)
        assert vec.dtype == np.float32

    def test_embed_batch_shape(self) -> None:
        """embed_batch() 결과 shape = (3, 280)."""
        emb = BLOSUM62Embedder(max_len=14)
        seqs = [SST14, "AGCKNFFWKTFTSA", "AGCKNFFWKTFTSV"]
        mat = emb.embed_batch(seqs)
        assert mat.shape == (3, 280), f"shape={mat.shape}"

    def test_normalize_range(self) -> None:
        """normalize=True 시 모든 값이 [0,1] 범위여야 한다."""
        emb = BLOSUM62Embedder(max_len=14, normalize=True)
        seqs = [aa * 14 for aa in AMINO_ACIDS]
        mat = emb.embed_batch(seqs)
        assert mat.min() >= -1e-6, f"min={mat.min()}"
        assert mat.max() <= 1.0 + 1e-6, f"max={mat.max()}"

    def test_normalize_false_raw_score(self) -> None:
        """normalize=False 시 Trp-Trp 자기 점수가 11이어야 한다 (BLOSUM62 최댓값)."""
        emb = BLOSUM62Embedder(max_len=1, normalize=False)
        vec_W = emb.embed("W")
        # W 행에서 W 열(AMINO_ACIDS.index('W')=18번째)이 11
        w_idx = AMINO_ACIDS.index("W")
        assert vec_W[w_idx] == pytest.approx(11.0), (
            f"W self-score={vec_W[w_idx]}"
        )

    def test_self_score_highest_per_aa(self) -> None:
        """각 AA의 자기 자신 BLOSUM62 점수가 행 내 최댓값이어야 한다 (모든 AA)."""
        emb = BLOSUM62Embedder(max_len=1, normalize=False)
        for aa in AMINO_ACIDS:
            vec = emb.embed(aa)
            self_idx = AMINO_ACIDS.index(aa)
            self_score = vec[self_idx]
            assert self_score == vec.max(), (
                f"{aa}: self_score={self_score} not max of {vec}"
            )

    def test_blosum62_matrix_completeness(self) -> None:
        """_BLOSUM62_MATRIX에 20종 AA 행이 모두 있어야 한다."""
        for aa in AMINO_ACIDS:
            assert aa in _BLOSUM62_MATRIX, f"{aa} missing from _BLOSUM62_MATRIX"
            assert len(_BLOSUM62_MATRIX[aa]) == 20, (
                f"{aa}: expected 20 columns, got {len(_BLOSUM62_MATRIX[aa])}"
            )

    def test_unknown_aa_zero_vector(self) -> None:
        """비표준 잔기 'X'는 0 벡터여야 한다."""
        emb = BLOSUM62Embedder(max_len=1)
        vec = emb.embed("X")
        assert np.all(vec == 0.0)


# ---------------------------------------------------------------------------
# OneHotEmbedder 하위 호환 (기존 동작 보존 확인)
# ---------------------------------------------------------------------------

class TestOneHotEmbedderBackcompat:
    def test_embed_dim(self) -> None:
        """max_len=14 → 280차원 그대로."""
        emb = OneHotEmbedder(max_len=14)
        vec = emb.embed(SST14)
        assert vec.shape == (280,)

    def test_values_binary(self) -> None:
        """출력값이 0 또는 1만 포함해야 한다."""
        emb = OneHotEmbedder(max_len=14)
        vec = emb.embed(SST14)
        unique_vals = set(vec.tolist())
        assert unique_vals <= {0.0, 1.0}, f"unexpected values: {unique_vals - {0.0, 1.0}}"

    def test_each_position_one_hot(self) -> None:
        """각 잔기 위치에서 정확히 1개만 1이어야 한다."""
        emb = OneHotEmbedder(max_len=14)
        mat = emb.embed(SST14).reshape(14, 20)
        for i in range(14):
            assert mat[i].sum() == pytest.approx(1.0), (
                f"position {i}: sum={mat[i].sum()}"
            )


# ---------------------------------------------------------------------------
# BayesianPeptideOptimizer + PropertyEmbedder 연동 스모크 테스트
# ---------------------------------------------------------------------------

class TestBayesianOptimizerWithPropertyEmbedder:
    """PropertyEmbedder를 임베더로 사용하는 BO 연동 테스트."""

    @pytest.fixture
    def sample_candidates(self) -> list:
        """5개 관측 후보 (fit에 최소 필요)."""
        seqs = [
            "AGCKNFFWKTFTSC",
            "AGCKNFFWKTFTSA",
            "AGCKNFFWKTFTSV",
            "AGCKNFFWKTFTSG",
            "AGCKNFFWKTFTSI",
        ]
        return [
            {"sequence": s, "ddg": float(-10 - i), "selectivity": float(0.5 + i * 0.1)}
            for i, s in enumerate(seqs)
        ]

    def test_fit_and_suggest_smoke(self, sample_candidates: list) -> None:
        """PropertyEmbedder로 fit → suggest 호출이 오류 없이 완료해야 한다."""
        emb = PropertyEmbedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=emb,
            objectives=["ddg", "selectivity"],
            maximize=[False, True],
        )
        bo.fit(sample_candidates)
        assert bo.is_fitted

        suggestions = bo.suggest(n=3, reference_seq=SST14)
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
        for s in suggestions:
            assert "sequence" in s
            assert "acquisition_value" in s

    def test_fit_and_suggest_blosum62(self, sample_candidates: list) -> None:
        """BLOSUM62Embedder로 fit → suggest 호출이 오류 없이 완료해야 한다."""
        emb = BLOSUM62Embedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=emb,
            objectives=["ddg", "selectivity"],
            maximize=[False, True],
        )
        bo.fit(sample_candidates)
        assert bo.is_fitted

        suggestions = bo.suggest(n=3, reference_seq=SST14)
        assert len(suggestions) <= 3

    def test_acquisition_values_shape(self, sample_candidates: list) -> None:
        """acquisition_values() 반환 shape이 입력 개수와 일치해야 한다."""
        emb = PropertyEmbedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=emb,
            objectives=["ddg", "selectivity"],
            maximize=[False, True],
        )
        bo.fit(sample_candidates)
        test_cands = [{"sequence": SST14}, {"sequence": "AGCKNFFWKTFTSA"}]
        acq = bo.acquisition_values(test_cands)
        assert acq.shape == (2,), f"shape={acq.shape}"

    def test_embed_dim_in_bo_is_property_dim(self, sample_candidates: list) -> None:
        """PropertyEmbedder(max_len=14) 사용 시 X 행렬이 (n, 98) 형태여야 한다."""
        emb = PropertyEmbedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=emb,
            objectives=["ddg", "selectivity"],
            maximize=[False, True],
        )
        bo.fit(sample_candidates)
        assert bo._X is not None
        assert bo._X.shape[1] == 98, f"X.shape={bo._X.shape}"
