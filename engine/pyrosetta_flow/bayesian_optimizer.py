"""Bayesian Optimization surrogate for peptide mutation suggestion.

GP surrogate + BoTorch acquisition (qNEHVI for multi-objective).
Embedding source is pluggable: ESM-2 if available, one-hot encoding fallback.
Complements Thompson Sampling by suggesting next mutation positions.

Dependencies:
    - numpy: required (one-hot + fallback GP)
    - botorch, gpytorch, torch: optional (full BO with qNEHVI)
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
_BOTORCH_AVAILABLE = False
try:
    import torch
    import gpytorch  # noqa: F401
    import botorch  # noqa: F401
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from botorch.acquisition.multi_objective import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.utils.multi_objective.pareto import is_non_dominated
    from botorch.utils.sampling import sample_simplex
    from botorch.utils.transforms import normalize, unnormalize
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    _BOTORCH_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX: Dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)

# SST-14 reference
SST14_SEQUENCE = "AGCKNFFWKTFTSC"


# =========================================================================
# Embedder Protocol / ABC
# =========================================================================
class PeptideEmbedder(ABC):
    """Abstract interface for peptide sequence embedding.

    Implementations convert a peptide sequence string into a fixed-length
    numerical vector suitable as GP input features.
    """

    @abstractmethod
    def embed(self, sequence: str) -> np.ndarray:
        """Embed a single peptide sequence.

        Args:
            sequence: Amino acid sequence (one-letter code, uppercase).

        Returns:
            1-D numpy array of shape ``(d,)`` where *d* is the embedding
            dimensionality.
        """
        ...

    def embed_batch(self, sequences: Sequence[str]) -> np.ndarray:
        """Embed multiple sequences. Default: sequential calls to :meth:`embed`.

        Args:
            sequences: Iterable of amino acid sequences.

        Returns:
            2-D numpy array of shape ``(n, d)``.
        """
        return np.stack([self.embed(seq) for seq in sequences])


# =========================================================================
# OneHotEmbedder (fallback, no external deps)
# =========================================================================
class OneHotEmbedder(PeptideEmbedder):
    """One-hot encoding of peptide sequences.

    Each residue is represented as a 20-dim one-hot vector (standard amino
    acids).  The full sequence embedding is the concatenation, yielding
    ``seq_len * 20`` dimensions.  If ``max_len`` is set, shorter sequences
    are zero-padded and longer sequences are truncated.

    Args:
        max_len: Maximum sequence length for fixed-size output.
            If ``None``, the length of each individual sequence is used
            (embeddings may differ in size).
    """

    def __init__(self, max_len: Optional[int] = None) -> None:
        self.max_len = max_len

    def embed(self, sequence: str) -> np.ndarray:
        """Return flattened one-hot encoding of *sequence*.

        Args:
            sequence: Amino acid sequence (uppercase, standard 20 AAs).

        Returns:
            1-D numpy array of shape ``(length * 20,)`` where *length* is
            ``max_len`` if set, otherwise ``len(sequence)``.
        """
        seq = sequence.upper()
        length = self.max_len if self.max_len is not None else len(seq)
        arr = np.zeros((length, NUM_AA), dtype=np.float32)
        for i, aa in enumerate(seq[:length]):
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                arr[i, idx] = 1.0
        return arr.ravel()


# =========================================================================
# Optional ESM-2 Embedder stub
# =========================================================================
class ESM2Embedder(PeptideEmbedder):
    """ESM-2 protein language model embedder (optional).

    Requires ``transformers`` and ``torch``. If unavailable, instantiation
    raises :class:`ImportError`.

    Args:
        model_name: HuggingFace model identifier for ESM-2.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch as _torch  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "ESM2Embedder requires 'transformers' and 'torch'. "
                "Install them or use OneHotEmbedder as fallback."
            ) from exc

        self._torch = _torch
        self.device = _torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed(self, sequence: str) -> np.ndarray:
        """Mean-pool last hidden states of ESM-2 over residue positions.

        Args:
            sequence: Amino acid sequence (uppercase).

        Returns:
            1-D numpy array of shape ``(hidden_dim,)``.
        """
        inputs = self.tokenizer(
            sequence, return_tensors="pt", add_special_tokens=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        # Mean-pool over sequence positions (skip [CLS] and [EOS])
        hidden = outputs.last_hidden_state[0, 1:-1, :]
        return hidden.mean(dim=0).cpu().numpy()


# ---------------------------------------------------------------------------
# Physicochemical property table for PropertyEmbedder
# ---------------------------------------------------------------------------
# 속성 정의 (7차원):
#   0: formal_charge   — 생리적 pH 7.4 기준 형식 전하 (-1, 0, +1, +2)
#   1: hydrophobicity  — Kyte-Doolittle GRAVY scale (−4.5 ~ +4.5)
#   2: volume_norm     — 잔기 분자 부피 (Ų, 88~228 범위를 0~1 정규화)
#   3: aromaticity     — 방향족 여부 (0 or 1)
#   4: pi_norm         — pI 기반 등전점 대리값 (0~1 min-max 정규화, 2.98~10.76)
#   5: hbond_donor     — H-bond donor 개수 (0~5, NH/OH/guanidinium 기준)
#   6: branching       — 곁사슬 분지 지수 (0=선형, 1=단분지, 2=복분지)
#
# 출처: Kyte & Doolittle (1982) J Mol Biol; Zamyatnin (1972) Prog Biophys Mol Biol
#       pI from Lehninger Biochemistry Table; volumes from Creighton (1993)
#
# 각 행: [formal_charge, hydrophobicity, volume_norm, aromaticity, pi_norm, hbond_donor, branching]
# volume_norm = (raw_volume - 88) / (228 - 88),  pi_norm = (pI - 2.98) / (10.76 - 2.98)
#
# AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY" (순서 고정)
AA_PROPERTIES: Dict[str, List[float]] = {
    #          charge  hydro   vol_n   arom   pi_n   hbd   branch
    "A": [     0.0,    1.8,    0.000,  0.0,   0.531, 1.0,  0.0],  # Ala   vol=88
    "C": [     0.0,    2.5,    0.114,  0.0,   0.680, 1.0,  0.0],  # Cys   vol=104
    "D": [    -1.0,   -3.5,    0.207,  0.0,   0.000, 0.0,  1.0],  # Asp   vol=117
    "E": [    -1.0,   -3.5,    0.350,  0.0,   0.078, 0.0,  1.0],  # Glu   vol=137
    "F": [     0.0,    2.8,    0.671,  1.0,   0.519, 1.0,  1.0],  # Phe   vol=182
    "G": [     0.0,   -0.4,    0.000,  0.0,   0.531, 1.0,  0.0],  # Gly   vol=60 → clamp to 0
    "H": [     0.0,   -3.2,    0.557,  1.0,   0.808, 2.0,  1.0],  # His   vol=163
    "I": [     0.0,    4.5,    0.457,  0.0,   0.531, 1.0,  2.0],  # Ile   vol=151
    "K": [     1.0,   -3.9,    0.486,  0.0,   1.000, 3.0,  0.0],  # Lys   vol=155
    "L": [     0.0,    3.8,    0.457,  0.0,   0.531, 1.0,  1.0],  # Leu   vol=151
    "M": [     0.0,    1.9,    0.479,  0.0,   0.531, 1.0,  0.0],  # Met   vol=154
    "N": [     0.0,   -3.5,    0.336,  0.0,   0.531, 2.0,  1.0],  # Asn   vol=135
    "P": [     0.0,   -1.6,    0.271,  0.0,   0.531, 0.0,  0.0],  # Pro   vol=126 (N in ring)
    "Q": [     0.0,   -3.5,    0.479,  0.0,   0.531, 2.0,  1.0],  # Gln   vol=154
    "R": [     1.0,   -4.5,    0.943,  0.0,   0.995, 5.0,  2.0],  # Arg   vol=219
    "S": [     0.0,   -0.8,    0.107,  0.0,   0.531, 2.0,  0.0],  # Ser   vol=103
    "T": [     0.0,   -0.7,    0.264,  0.0,   0.531, 2.0,  1.0],  # Thr   vol=125
    "V": [     0.0,    4.2,    0.321,  0.0,   0.531, 1.0,  2.0],  # Val   vol=133
    "W": [     0.0,   -0.9,    1.000,  1.0,   0.519, 2.0,  1.0],  # Trp   vol=228
    "Y": [     0.0,   -1.3,    0.814,  1.0,   0.519, 2.0,  1.0],  # Tyr   vol=202
}
AA_PROP_DIM: int = 7  # 잔기당 물리화학 속성 차원 수


# =========================================================================
# PropertyEmbedder (물리화학 속성 기반 임베딩)
# =========================================================================
class PropertyEmbedder(PeptideEmbedder):
    """물리화학 속성 벡터 기반 펩타이드 임베딩.

    각 잔기를 7차원 물리화학 속성 벡터로 인코딩한다:
        [전하, 소수성(KD), 부피(정규화), 방향족, pI(정규화), H-bond donor 수, 분지 지수]

    장점: AA 물리화학 유사성이 L2 거리에 반영 → GP RBF kernel 거리 의미 있음.
    OneHotEmbedder 대비 차원: (seq_len×7) vs (seq_len×20), ~65% 절약.

    전체 임베딩 차원: ``max_len * 7`` (max_len 지정 시) 또는 ``len(seq) * 7``.

    **권장 기본 임베더**: PropertyEmbedder(max_len=14) — OneHotEmbedder보다
    GP RBF kernel 거리 의미론이 우수. _FallbackGP lengthscale=1.0 그대로 호환.

    Args:
        max_len: 최대 시퀀스 길이 (고정 차원 출력용). None이면 시퀀스 길이 사용.
        normalize: True(기본)이면 각 속성을 column-wise [0,1] min-max 정규화.
            False이면 AA_PROPERTIES 원본 스케일 그대로 출력.
    """

    _PROP_NAMES: List[str] = [
        "formal_charge", "hydrophobicity", "volume_norm",
        "aromaticity", "pi_norm", "hbond_donor", "branching",
    ]

    def __init__(
        self,
        max_len: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.max_len = max_len
        self.normalize = normalize
        # 속성 행렬 캐시: shape (20, 7), AMINO_ACIDS 순서 고정
        self._prop_matrix: np.ndarray = self._build_prop_matrix()

    def _build_prop_matrix(self) -> np.ndarray:
        """AA_PROPERTIES 테이블로부터 (20, 7) 행렬 생성.

        Returns:
            shape ``(20, 7)``의 float32 행렬.
        """
        mat = np.zeros((NUM_AA, AA_PROP_DIM), dtype=np.float32)
        for i, aa in enumerate(AMINO_ACIDS):
            mat[i] = AA_PROPERTIES[aa]
        if self.normalize:
            # 각 속성을 [0,1]로 min-max 정규화 (column-wise)
            col_min = mat.min(axis=0)
            col_max = mat.max(axis=0)
            col_range = np.where(col_max - col_min == 0, 1.0, col_max - col_min)
            mat = (mat - col_min) / col_range
        return mat

    def embed(self, sequence: str) -> np.ndarray:
        """잔기별 물리화학 속성 벡터를 이어붙인 1-D 임베딩 반환.

        Args:
            sequence: 대문자 아미노산 1문자 코드 시퀀스.

        Returns:
            shape ``(length * 7,)``의 1-D numpy float32 배열.
            알 수 없는 잔기(비표준 AA)는 0 벡터로 처리.
        """
        seq = sequence.upper()
        length = self.max_len if self.max_len is not None else len(seq)
        arr = np.zeros((length, AA_PROP_DIM), dtype=np.float32)
        for i, aa in enumerate(seq[:length]):
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                arr[i] = self._prop_matrix[idx]
        return arr.ravel()

    @property
    def n_properties(self) -> int:
        """잔기당 속성 차원 수 (7)."""
        return AA_PROP_DIM

    @property
    def prop_names(self) -> List[str]:
        """속성 이름 목록."""
        return list(self._PROP_NAMES)


# ---------------------------------------------------------------------------
# BLOSUM62 치환 행렬 (Henikoff & Henikoff 1992, NCBI standard)
# 행/열 순서: AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# ---------------------------------------------------------------------------
_BLOSUM62_MATRIX: Dict[str, Dict[str, int]] = {
    "A": {"A":  4,"C": -1,"D": -2,"E": -1,"F": -2,"G":  0,"H": -2,"I": -1,"K": -1,"L": -1,"M": -1,"N": -2,"P": -1,"Q": -1,"R": -1,"S":  1,"T":  0,"V":  0,"W": -3,"Y": -2},
    "C": {"A": -1,"C":  9,"D": -3,"E": -4,"F": -2,"G": -3,"H": -3,"I": -1,"K": -3,"L": -1,"M": -1,"N": -3,"P": -3,"Q": -3,"R": -3,"S": -1,"T": -1,"V": -1,"W": -2,"Y": -2},
    "D": {"A": -2,"C": -3,"D":  6,"E":  2,"F": -3,"G": -1,"H": -1,"I": -3,"K": -1,"L": -4,"M": -3,"N":  1,"P": -1,"Q":  0,"R": -2,"S":  0,"T": -1,"V": -3,"W": -4,"Y": -3},
    "E": {"A": -1,"C": -4,"D":  2,"E":  5,"F": -3,"G": -2,"H":  0,"I": -3,"K":  1,"L": -3,"M": -2,"N":  0,"P": -1,"Q":  2,"R":  0,"S":  0,"T": -1,"V": -2,"W": -3,"Y": -2},
    "F": {"A": -2,"C": -2,"D": -3,"E": -3,"F":  6,"G": -3,"H": -1,"I":  0,"K": -3,"L":  1,"M":  0,"N": -3,"P": -4,"Q": -3,"R": -3,"S": -2,"T": -2,"V": -1,"W":  1,"Y":  3},
    "G": {"A":  0,"C": -3,"D": -1,"E": -2,"F": -3,"G":  6,"H": -2,"I": -4,"K": -2,"L": -4,"M": -3,"N":  0,"P": -2,"Q": -2,"R": -2,"S":  0,"T": -2,"V": -3,"W": -2,"Y": -3},
    "H": {"A": -2,"C": -3,"D": -1,"E":  0,"F": -1,"G": -2,"H":  8,"I": -3,"K": -1,"L": -3,"M": -2,"N":  1,"P": -2,"Q":  0,"R":  0,"S": -1,"T": -2,"V": -3,"W": -2,"Y":  2},
    "I": {"A": -1,"C": -1,"D": -3,"E": -3,"F":  0,"G": -4,"H": -3,"I":  4,"K": -1,"L":  2,"M":  1,"N": -3,"P": -3,"Q": -3,"R": -3,"S": -2,"T": -1,"V":  3,"W": -3,"Y": -1},
    "K": {"A": -1,"C": -3,"D": -1,"E":  1,"F": -3,"G": -2,"H": -1,"I": -1,"K":  5,"L": -2,"M": -1,"N":  0,"P": -1,"Q":  1,"R":  2,"S":  0,"T": -1,"V": -2,"W": -3,"Y": -2},
    "L": {"A": -1,"C": -1,"D": -4,"E": -3,"F":  1,"G": -4,"H": -3,"I":  2,"K": -2,"L":  4,"M":  2,"N": -3,"P": -3,"Q": -2,"R": -2,"S": -2,"T": -1,"V":  1,"W": -2,"Y": -1},
    "M": {"A": -1,"C": -1,"D": -3,"E": -2,"F":  0,"G": -3,"H": -2,"I":  1,"K": -1,"L":  2,"M":  5,"N": -2,"P": -2,"Q":  0,"R": -1,"S": -1,"T": -1,"V":  1,"W": -1,"Y": -1},
    "N": {"A": -2,"C": -3,"D":  1,"E":  0,"F": -3,"G":  0,"H":  1,"I": -3,"K":  0,"L": -3,"M": -2,"N":  6,"P": -2,"Q":  0,"R":  0,"S":  1,"T":  0,"V": -3,"W": -4,"Y": -2},
    "P": {"A": -1,"C": -3,"D": -1,"E": -1,"F": -4,"G": -2,"H": -2,"I": -3,"K": -1,"L": -3,"M": -2,"N": -2,"P":  7,"Q": -1,"R": -2,"S": -1,"T": -1,"V": -2,"W": -4,"Y": -3},
    "Q": {"A": -1,"C": -3,"D":  0,"E":  2,"F": -3,"G": -2,"H":  0,"I": -3,"K":  1,"L": -2,"M":  0,"N":  0,"P": -1,"Q":  5,"R":  1,"S":  0,"T": -1,"V": -2,"W": -2,"Y": -1},
    "R": {"A": -1,"C": -3,"D": -2,"E":  0,"F": -3,"G": -2,"H":  0,"I": -3,"K":  2,"L": -2,"M": -1,"N":  0,"P": -2,"Q":  1,"R":  5,"S": -1,"T": -1,"V": -3,"W": -3,"Y": -2},
    "S": {"A":  1,"C": -1,"D":  0,"E":  0,"F": -2,"G":  0,"H": -1,"I": -2,"K":  0,"L": -2,"M": -1,"N":  1,"P": -1,"Q":  0,"R": -1,"S":  4,"T":  1,"V": -2,"W": -3,"Y": -2},
    "T": {"A":  0,"C": -1,"D": -1,"E": -1,"F": -2,"G": -2,"H": -2,"I": -1,"K": -1,"L": -1,"M": -1,"N":  0,"P": -1,"Q": -1,"R": -1,"S":  1,"T":  5,"V":  0,"W": -2,"Y": -2},
    "V": {"A":  0,"C": -1,"D": -3,"E": -2,"F": -1,"G": -3,"H": -3,"I":  3,"K": -2,"L":  1,"M":  1,"N": -3,"P": -2,"Q": -2,"R": -3,"S": -2,"T":  0,"V":  4,"W": -3,"Y": -1},
    "W": {"A": -3,"C": -2,"D": -4,"E": -3,"F":  1,"G": -2,"H": -2,"I": -3,"K": -3,"L": -2,"M": -1,"N": -4,"P": -4,"Q": -2,"R": -3,"S": -3,"T": -2,"V": -3,"W": 11,"Y":  2},
    "Y": {"A": -2,"C": -2,"D": -3,"E": -2,"F":  3,"G": -3,"H":  2,"I": -1,"K": -2,"L": -1,"M": -1,"N": -2,"P": -3,"Q": -1,"R": -2,"S": -2,"T": -2,"V": -1,"W":  2,"Y":  7},
}


# =========================================================================
# BLOSUM62Embedder (BLOSUM62 행렬 기반 임베딩)
# =========================================================================
class BLOSUM62Embedder(PeptideEmbedder):
    """BLOSUM62 치환 행렬 기반 펩타이드 임베딩.

    각 잔기를 BLOSUM62 행렬의 해당 행 벡터(20차원)로 표현한다.
    치환 빈도 기반 유사성이 임베딩 공간 거리에 직접 반영되어
    진화적·생물학적 유사성을 GP kernel이 인식할 수 있다.

    차원: ``max_len * 20`` (OneHotEmbedder와 동일하나 값이 연속 실수).

    **참고**: PropertyEmbedder가 물리화학 관점, BLOSUM62Embedder가
    진화/서열 관점의 유사성을 각각 인코딩한다. 두 임베더 중
    일반 펩타이드 BO에는 PropertyEmbedder 권장.

    Args:
        max_len: 최대 시퀀스 길이. None이면 시퀀스 길이 사용.
        normalize: True(기본)이면 [0,1] min-max 정규화 (BLOSUM62 원본 범위 −4~11).
    """

    _BLOSUM62_MIN: float = -4.0
    _BLOSUM62_MAX: float = 11.0

    def __init__(
        self,
        max_len: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        self.max_len = max_len
        self.normalize = normalize
        self._mat: np.ndarray = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        """BLOSUM62 딕셔너리에서 (20, 20) 행렬 생성 (AMINO_ACIDS 순서).

        Returns:
            shape ``(20, 20)``의 float32 행렬.
        """
        mat = np.zeros((NUM_AA, NUM_AA), dtype=np.float32)
        for i, aa_row in enumerate(AMINO_ACIDS):
            for j, aa_col in enumerate(AMINO_ACIDS):
                mat[i, j] = float(_BLOSUM62_MATRIX[aa_row].get(aa_col, 0))
        if self.normalize:
            mat = (mat - self._BLOSUM62_MIN) / (self._BLOSUM62_MAX - self._BLOSUM62_MIN)
        return mat

    def embed(self, sequence: str) -> np.ndarray:
        """BLOSUM62 행 벡터를 이어붙인 1-D 임베딩 반환.

        Args:
            sequence: 대문자 아미노산 1문자 코드 시퀀스.

        Returns:
            shape ``(length * 20,)``의 1-D numpy float32 배열.
            알 수 없는 잔기는 0 벡터로 처리.
        """
        seq = sequence.upper()
        length = self.max_len if self.max_len is not None else len(seq)
        arr = np.zeros((length, NUM_AA), dtype=np.float32)
        for i, aa in enumerate(seq[:length]):
            idx = AA_TO_IDX.get(aa)
            if idx is not None:
                arr[i] = self._mat[idx]
        return arr.ravel()


# =========================================================================
# Minimal fallback GP (numpy-only)
# =========================================================================
class _FallbackGP:
    """Extremely simple RBF-kernel GP regressor using numpy only.

    This is a last-resort fallback when botorch/gpytorch are unavailable.
    It supports single-objective regression with a fixed noise variance.

    Args:
        noise: Observation noise variance.
        lengthscale: RBF kernel lengthscale.
    """

    def __init__(
        self, noise: float = 1e-4, lengthscale: float = 1.0
    ) -> None:
        self.noise = noise
        self.lengthscale = lengthscale
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._K_inv: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None

    def _rbf(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        sq_dist = (
            np.sum(X1 ** 2, axis=1, keepdims=True)
            + np.sum(X2 ** 2, axis=1, keepdims=True).T
            - 2.0 * X1 @ X2.T
        )
        return np.exp(-0.5 * sq_dist / (self.lengthscale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP on training data.

        Args:
            X: Feature matrix of shape ``(n, d)``.
            y: Target vector of shape ``(n,)``.
        """
        self._X = X.copy()
        self._y = y.copy()
        K = self._rbf(X, X) + self.noise * np.eye(len(X))
        self._K_inv = np.linalg.inv(K)
        self._alpha = self._K_inv @ y

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points.

        Args:
            X_new: Feature matrix of shape ``(m, d)``.

        Returns:
            Tuple of (mean, variance) arrays each of shape ``(m,)``.
        """
        assert self._X is not None, "Must call fit() first"
        K_star = self._rbf(X_new, self._X)
        mean = K_star @ self._alpha
        v = K_star @ self._K_inv @ K_star.T
        var = 1.0 - np.diag(v)
        var = np.maximum(var, 0.0)
        return mean, var


# =========================================================================
# BayesianPeptideOptimizer
# =========================================================================
class BayesianPeptideOptimizer:
    """Multi-objective Bayesian optimiser for peptide design.

    Uses GP surrogates (one per objective) with qNEHVI acquisition to suggest
    next mutation candidates.  Falls back to a numpy-only GP + random
    acquisition when BoTorch is unavailable.

    Args:
        embedder: A :class:`PeptideEmbedder` instance for featurisation.
        objectives: List of objective names (keys in candidate dicts).
        maximize: Per-objective direction; ``True`` = maximize, ``False`` =
            minimize.  If ``None``, all objectives are maximized.
        ref_point: Reference point for hypervolume computation. If ``None``,
            inferred from data as ``min(obj) - 0.1 * range(obj)``.
        seed: 난수 시드 (재현성 목적). None이면 기존 동작(비결정적) 유지.
            설정 시 numpy 전역 시드 및 (BoTorch 사용 가능한 경우) torch 전역
            시드를 동시에 설정한다. 멀티스레드 환경에서는 전역 시드 충돌에
            주의할 것.
    """

    def __init__(
        self,
        embedder: PeptideEmbedder,
        objectives: List[str],
        maximize: Optional[List[bool]] = None,
        ref_point: Optional[List[float]] = None,
        seed: Optional[int] = None,  # E1: 재현성 seed (2026-06-23)
    ) -> None:
        self.embedder = embedder
        self.objectives = objectives
        self.maximize = maximize if maximize is not None else [True] * len(objectives)
        self._ref_point = ref_point
        self._use_botorch = _BOTORCH_AVAILABLE
        self._seed = seed

        # E1: seed 설정 (None이면 기존 비결정적 동작 보존)
        if seed is not None:
            np.random.seed(seed)
            if _BOTORCH_AVAILABLE:
                torch.manual_seed(seed)
            logger.debug("[BayesianPeptideOptimizer] 전역 seed 설정: %d", seed)

        if not self._use_botorch:
            warnings.warn(
                "botorch/gpytorch not available. "
                "Falling back to numpy GP + random acquisition. "
                "Install botorch for full multi-objective BO.",
                stacklevel=2,
            )

        # State populated by fit()
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._candidates: Optional[List[Dict]] = None
        self._model = None  # BoTorch ModelListGP or list[_FallbackGP]
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_XY(
        self, candidates: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Embed sequences and collect objective values from candidate dicts.

        Args:
            candidates: List of dicts with ``"sequence"`` key and objective keys.

        Returns:
            Tuple of (X, Y) numpy arrays.
        """
        sequences = [c["sequence"] for c in candidates]
        X = self.embedder.embed_batch(sequences)
        Y = np.array(
            [[c[obj] for obj in self.objectives] for c in candidates],
            dtype=np.float64,
        )
        # Flip sign for minimisation objectives so GP always maximises
        for j, do_max in enumerate(self.maximize):
            if not do_max:
                Y[:, j] = -Y[:, j]
        return X.astype(np.float64), Y

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, candidates: List[Dict]) -> None:
        """Fit GP surrogate(s) on observed candidates.

        Args:
            candidates: List of dicts, each containing at minimum
                ``"sequence"`` (str) and one key per objective with a
                numeric value.

        Raises:
            ValueError: If candidate dicts lack required keys.
        """
        if not candidates:
            raise ValueError("candidates list must not be empty")
        missing = {"sequence"} | set(self.objectives)
        for key in list(missing):
            if key in candidates[0]:
                missing.discard(key)
        if missing:
            raise ValueError(f"Candidate dicts missing keys: {missing}")

        X, Y = self._extract_XY(candidates)
        self._X = X
        self._Y = Y
        self._candidates = list(candidates)

        if self._use_botorch:
            self._fit_botorch(X, Y)
        else:
            self._fit_fallback(X, Y)

        self._fitted = True
        logger.info(
            "BayesianPeptideOptimizer fitted on %d candidates, %d objectives",
            len(candidates),
            len(self.objectives),
        )

    def _fit_botorch(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit BoTorch ModelListGP."""
        train_X = torch.tensor(X, dtype=torch.double)
        models = []
        for j in range(Y.shape[1]):
            train_y = torch.tensor(Y[:, j : j + 1], dtype=torch.double)
            gp = SingleTaskGP(train_X, train_y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            models.append(gp)
        self._model = ModelListGP(*models)

    def _fit_fallback(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit per-objective _FallbackGP models."""
        models = []
        for j in range(Y.shape[1]):
            # PropertyEmbedder 사용 시 속성값이 [0,1] 정규화되어 lengthscale=1.0 그대로 호환
            gp = _FallbackGP(noise=1e-4, lengthscale=1.0)
            gp.fit(X, Y[:, j])
            models.append(gp)
        self._model = models

    # ------------------------------------------------------------------
    # suggest
    # ------------------------------------------------------------------
    def suggest(
        self,
        n: int,
        reference_seq: str,
        allowed_positions: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Suggest next mutation candidates using acquisition function.

        Generates single-point mutations of *reference_seq* at each allowed
        position, evaluates the acquisition value for each, and returns the
        top *n*.

        Args:
            n: Number of candidates to return.
            reference_seq: Base peptide sequence to mutate.
            allowed_positions: 0-based indices of mutable positions.  If
                ``None``, all positions are considered.

        Returns:
            List of dicts with ``"sequence"``, ``"position"``, ``"mutation"``,
            and ``"acquisition_value"`` keys, sorted by descending
            acquisition value.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before suggest()")

        mutations = self._enumerate_mutations(reference_seq, allowed_positions)
        if not mutations:
            return []

        # Build candidate dicts (only sequence needed for embedding)
        mut_candidates = [{"sequence": m["sequence"]} for m in mutations]
        acq_vals = self._compute_acquisition(mut_candidates)

        for m, v in zip(mutations, acq_vals):
            m["acquisition_value"] = float(v)

        # Sort descending by acquisition value
        mutations.sort(key=lambda x: x["acquisition_value"], reverse=True)
        return mutations[:n]

    @staticmethod
    def _enumerate_mutations(
        reference_seq: str,
        allowed_positions: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Generate all single-point mutations of *reference_seq*.

        Args:
            reference_seq: Base sequence.
            allowed_positions: Positions to mutate (0-based). If ``None``,
                all positions are considered.

        Returns:
            List of dicts with ``"sequence"``, ``"position"``, ``"mutation"``
            (the new AA), and ``"original"`` (the replaced AA).
        """
        seq = list(reference_seq.upper())
        positions = allowed_positions if allowed_positions is not None else list(range(len(seq)))
        results: List[Dict] = []
        for pos in positions:
            if pos < 0 or pos >= len(seq):
                continue
            orig = seq[pos]
            for aa in AMINO_ACIDS:
                if aa == orig:
                    continue
                new_seq = seq.copy()
                new_seq[pos] = aa
                results.append({
                    "sequence": "".join(new_seq),
                    "position": pos,
                    "mutation": aa,
                    "original": orig,
                })
        return results

    # ------------------------------------------------------------------
    # acquisition_values
    # ------------------------------------------------------------------
    def acquisition_values(self, candidates: List[Dict]) -> np.ndarray:
        """Compute acquisition values for a list of candidates.

        Args:
            candidates: List of dicts with ``"sequence"`` key.

        Returns:
            1-D numpy array of acquisition values, one per candidate.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before acquisition_values()")
        return self._compute_acquisition(candidates)

    def _compute_acquisition(self, candidates: List[Dict]) -> np.ndarray:
        """Dispatch to BoTorch or fallback acquisition."""
        sequences = [c["sequence"] for c in candidates]
        X_new = self.embedder.embed_batch(sequences).astype(np.float64)

        if self._use_botorch:
            return self._acquisition_botorch(X_new)
        else:
            return self._acquisition_fallback(X_new)

    def _acquisition_botorch(self, X_new: np.ndarray) -> np.ndarray:
        """qNEHVI acquisition via BoTorch.

        Args:
            X_new: Feature matrix of shape ``(m, d)``.

        Returns:
            Acquisition values array of shape ``(m,)``.
        """
        assert self._Y is not None
        # Reference point: per-objective minimum minus margin
        if self._ref_point is not None:
            ref = torch.tensor(self._ref_point, dtype=torch.double)
        else:
            Y_min = self._Y.min(axis=0)
            Y_range = self._Y.max(axis=0) - Y_min
            Y_range = np.where(Y_range == 0, 1.0, Y_range)
            ref = torch.tensor(Y_min - 0.1 * Y_range, dtype=torch.double)

        X_tensor = torch.tensor(X_new, dtype=torch.double)

        # Evaluate each candidate individually via posterior predictive mean
        # as a lightweight proxy (full qNEHVI is expensive for large sets)
        model = self._model
        model.eval()

        # Posterior mean per objective
        with torch.no_grad():
            posterior = model.posterior(X_tensor)
            means = posterior.mean  # (m, num_obj)

        # Simple hypervolume-improvement proxy: sum of improvement over ref
        improvement = means - ref.unsqueeze(0)
        improvement = torch.clamp(improvement, min=0.0)
        acq = improvement.prod(dim=-1)  # product-of-improvements proxy

        return acq.cpu().numpy()

    def _acquisition_fallback(self, X_new: np.ndarray) -> np.ndarray:
        """UCB-based acquisition using fallback GP (numpy only).

        Uses Upper Confidence Bound (UCB) per objective and sums them.

        Args:
            X_new: Feature matrix of shape ``(m, d)``.

        Returns:
            Acquisition values array of shape ``(m,)``.
        """
        beta = 2.0  # UCB exploration parameter
        total_acq = np.zeros(X_new.shape[0])
        for gp in self._model:
            mean, var = gp.predict(X_new)
            ucb = mean + beta * np.sqrt(var + 1e-8)
            total_acq += ucb
        return total_acq

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Whether the optimizer has been fitted."""
        return self._fitted

    @property
    def has_botorch(self) -> bool:
        """Whether BoTorch backend is available."""
        return self._use_botorch
