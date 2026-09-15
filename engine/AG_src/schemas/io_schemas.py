"""
io_schemas.py
=============
SSTR2 펩타이드 바인더 파이프라인 각 단계별 입출력 스키마 정의
Step-wise I/O schema definitions for the SSTR2 peptide binder pipeline.

각 dataclass는 to_dict() / from_dict() 메서드를 포함하여
JSON 직렬화 및 단계 간 데이터 전달을 지원합니다.
Each dataclass includes to_dict() / from_dict() for JSON serialization
and inter-step data transfer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# =============================================================================
# Step 01 - Receptor Preparation (수용체 준비)
# OpenFold3로 SSTR2 구조 예측 또는 기존 PDB 로드
# =============================================================================

@dataclass
class Step01Output:
    """
    Step 01 출력: 수용체(SSTR2) 준비 결과
    Output of receptor preparation step.
    """
    receptor_pdb_path: str          # 정제된 수용체 PDB 파일 경로
    pocket_residues: List[int]      # 포켓 잔기 번호 목록 (정수)
    chain_id: str                   # 수용체 체인 ID (예: "B")
    pocket_json_path: str           # 포켓 잔기 정보 JSON 파일 경로

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step01Output":
        return cls(**d)


# =============================================================================
# Step 02 - Backbone Design (백본 설계)
# RFdiffusion으로 de novo 바인더 백본 생성
# =============================================================================

@dataclass
class Step02Output:
    """
    Step 02 출력: RFdiffusion 백본 설계 결과
    Output of RFdiffusion backbone design step.
    """
    backbone_pdbs: List[str]        # 생성된 백본 PDB 파일 경로 목록
    design_params: Dict[str, Any]   # 사용된 설계 파라미터 (contigs, hotspot_res 등)
    n_generated: int                # 실제 생성된 백본 수

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step02Output":
        return cls(**d)


# =============================================================================
# Step 03 - Sequence Design (시퀀스 설계)
# ProteinMPNN 역폴딩으로 백본당 여러 시퀀스 생성
# =============================================================================

@dataclass
class SequenceEntry:
    """백본-시퀀스 쌍 단위 항목 / Individual backbone-sequence pair."""
    backbone_idx: int       # 백본 인덱스 (0-based)
    seq_idx: int            # 시퀀스 인덱스 (백본 내, 0-based)
    sequence: str           # 아미노산 시퀀스 (one-letter code)
    fasta_path: str         # FASTA 파일 경로
    seq_id: str = ""        # 고유 식별자: "bb{backbone_idx:02d}_seq{seq_idx:02d}"

    def __post_init__(self):
        if not self.seq_id:
            self.seq_id = f"bb{self.backbone_idx:02d}_seq{self.seq_idx:02d}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SequenceEntry":
        return cls(**d)


@dataclass
class Step03Output:
    """
    Step 03 출력: ProteinMPNN 시퀀스 설계 결과
    Output of ProteinMPNN inverse folding step.
    """
    sequences: List[SequenceEntry]  # 생성된 모든 시퀀스 항목

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequences": [s.to_dict() for s in self.sequences]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step03Output":
        return cls(
            sequences=[SequenceEntry.from_dict(s) for s in d["sequences"]]
        )


# =============================================================================
# Step 04 - ESMFold QC (구조 예측 및 품질 관리)
# ESMFold로 빠른 구조 예측 후 pLDDT 게이트 적용
# =============================================================================

@dataclass
class QCResult:
    """단일 후보의 ESMFold QC 결과 / ESMFold QC result for one candidate."""
    seq_id: str             # 시퀀스 고유 ID
    plddt_mean: float       # 전체 평균 pLDDT (0~100)
    plddt_interface: float  # 계면 잔기 평균 pLDDT (0~100)
    pdb_path: str           # ESMFold 예측 PDB 파일 경로
    passed_gate: bool       # 게이트 통과 여부
    disulfide_intact: Optional[bool] = None    # 이황화 결합 유지 여부 (None=미검사)
    disulfide_distance: Optional[float] = None # SG-SG 원자 간 거리 (Angstroms, None=미측정)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QCResult":
        return cls(
            seq_id=d["seq_id"],
            plddt_mean=d["plddt_mean"],
            plddt_interface=d["plddt_interface"],
            pdb_path=d["pdb_path"],
            passed_gate=d["passed_gate"],
            disulfide_intact=d.get("disulfide_intact"),
            disulfide_distance=d.get("disulfide_distance"),
        )


@dataclass
class Step04Output:
    """
    Step 04 출력: ESMFold QC 결과 전체
    Output of ESMFold QC step.
    """
    qc_results: List[QCResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "qc_results": [r.to_dict() for r in self.qc_results]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step04Output":
        return cls(
            qc_results=[QCResult.from_dict(r) for r in d["qc_results"]]
        )

    def passed(self) -> List[QCResult]:
        """게이트 통과 후보만 반환 / Return only gate-passing candidates."""
        return [r for r in self.qc_results if r.passed_gate]


# =============================================================================
# Step 05 - Docking (분자 도킹)
# DiffDock 또는 Boltz2로 수용체-펩타이드 도킹
# =============================================================================

@dataclass
class DockingResult:
    """단일 후보의 도킹 결과 / Docking result for one candidate."""
    seq_id: str             # 시퀀스 고유 ID
    engine: str             # 도킹 엔진: "diffdock" 또는 "boltz2"
    score: float            # 도킹 점수 (더 음수 = 더 좋음)
    confidence: float       # 신뢰도 점수
    pose_pdb: str           # 도킹 포즈 PDB 파일 경로
    rank: int               # 도킹 내부 포즈 순위 (1-based)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DockingResult":
        return cls(**d)


@dataclass
class Step05Output:
    """
    Step 05 출력: 도킹 결과 전체
    Output of docking step.
    """
    docking_results: List[DockingResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "docking_results": [r.to_dict() for r in self.docking_results]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step05Output":
        return cls(
            docking_results=[DockingResult.from_dict(r) for r in d["docking_results"]]
        )

    def top_pct(self, pct: float = 20.0) -> List[DockingResult]:
        """
        상위 pct% 도킹 결과만 반환 (score 기준 오름차순)
        Return top pct% results sorted by score ascending (lower = better).
        """
        sorted_results = sorted(self.docking_results, key=lambda r: r.score)
        n = max(1, int(len(sorted_results) * pct / 100.0))
        return sorted_results[:n]


# =============================================================================
# Step 05b - Selectivity Screening (선택성 스크리닝)
# Off-target 수용체(SSTR1/3/4/5) 대비 SSTR2 결합 특이성 평가
# =============================================================================

@dataclass
class OffTargetDockingResult:
    """단일 후보의 off-target 도킹 결과."""
    seq_id: str                  # 시퀀스 고유 ID
    receptor_name: str           # off-target 수용체 이름 (예: "SSTR1")
    dock_score: float            # 도킹 점수 (더 음수 = 더 강한 결합)
    confidence: float            # 신뢰도 점수
    engine: str                  # 도킹 엔진 이름

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OffTargetDockingResult":
        return cls(**d)


@dataclass
class SelectivityResult:
    """단일 후보의 선택성 평가 결과."""
    seq_id: str                           # 시퀀스 고유 ID
    sstr2_dock_score: float               # SSTR2 도킹 점수
    offtarget_scores: Dict[str, float]    # {receptor_name: dock_score}
    offtarget_max_score: float            # off-target 중 최고(최저) 결합 점수
    offtarget_max_receptor: str           # 가장 강하게 결합하는 off-target 이름
    selectivity_margin: float             # sstr2_score - offtarget_max_score
    passed: bool                          # 선택성 게이트 통과 여부

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelectivityResult":
        return cls(**d)


@dataclass
class Step05bOutput:
    """Step 05b 출력: 선택성 스크리닝 결과 전체."""
    selectivity_results: List[SelectivityResult]
    offtarget_docking_details: List[OffTargetDockingResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selectivity_results": [r.to_dict() for r in self.selectivity_results],
            "offtarget_docking_details": [r.to_dict() for r in self.offtarget_docking_details],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step05bOutput":
        return cls(
            selectivity_results=[SelectivityResult.from_dict(r) for r in d["selectivity_results"]],
            offtarget_docking_details=[OffTargetDockingResult.from_dict(r) for r in d["offtarget_docking_details"]],
        )

    def passed_candidates(self) -> List[SelectivityResult]:
        """선택성 게이트를 통과한 후보만 반환."""
        return [r for r in self.selectivity_results if r.passed]


# =============================================================================
# Step 06 - Rosetta Refinement (Rosetta 정제 및 ddG 계산)
# PyRosetta FlexPepDock + ddG 계산
# =============================================================================

@dataclass
class RosettaResult:
    """단일 후보의 Rosetta 정제 결과 / Rosetta refinement result for one candidate."""
    seq_id: str                     # 시퀀스 고유 ID
    ddg: float                      # ΔΔG (결합 자유 에너지 변화, kcal/mol)
    total_score: float              # Rosetta 전체 에너지 점수 (정제 후)
    clash_score: float              # 클래시(충돌) 점수
    constraint_violations: int      # 구조 제약 위반 횟수
    refined_pdb: str                # 정제된 복합체 PDB 파일 경로
    pre_score: float = 0.0          # 정제 전 전체 에너지 점수
    score_delta: float = 0.0        # 에너지 변화량 (total_score - pre_score)
    stub: bool = False              # True=실제 계산 아닌 플레이스홀더(랭킹 제외 대상). 2026-06-09 F08

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RosettaResult":
        return cls(**d)


@dataclass
class Step06Output:
    """
    Step 06 출력: Rosetta 정제 결과 전체
    Output of Rosetta refinement step.
    """
    rosetta_results: List[RosettaResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rosetta_results": [r.to_dict() for r in self.rosetta_results]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step06Output":
        return cls(
            rosetta_results=[RosettaResult.from_dict(r) for r in d["rosetta_results"]]
        )


# =============================================================================
# Step 07 - Analysis & Reporting (분석 및 보고)
# FoldMason 구조 비교, PyMOL 렌더링, 순위 테이블 작성
# =============================================================================

@dataclass
class Step07Output:
    """
    Step 07 출력: 분석 및 보고 결과
    Output of analysis and reporting step.
    """
    lddt_table_path: str            # FoldMason lDDT 결과 JSON/CSV 경로
    pymol_renders: Dict[str, str]   # 렌더 유형 -> PNG 파일 경로 매핑
                                    # e.g. {"overview": "...", "closeup": "..."}
    rank_table_csv: str             # 최종 순위 테이블 CSV 경로
    summary_md: str                 # 실험 요약 마크다운 파일 경로

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step07Output":
        return cls(**d)


# =============================================================================
# Rank Table Row (순위 테이블 행)
# 모든 단계 결과를 통합한 최종 순위 테이블의 행 단위 스키마
# =============================================================================

@dataclass
class RankTableRow:
    """
    최종 순위 테이블의 단일 행
    One row in the final ranking table, aggregating results from all steps.
    """
    backbone_id: int                # 백본 인덱스
    seq_id: str                     # 시퀀스 고유 ID (예: "bb00_seq03")
    sequence: str                   # 아미노산 시퀀스
    plddt_mean: float               # ESMFold 평균 pLDDT
    plddt_interface: float          # ESMFold 계면 pLDDT
    dock_score: float               # 도킹 점수
    dock_engine: str                # 도킹 엔진 이름
    ddg: float                      # Rosetta ΔΔG (kcal/mol)
    lddt: float                     # FoldMason lDDT
    final_score: float              # 가중 최종 점수
    pass_fail: str                  # "PASS" 또는 "FAIL"
    fail_reason: str                # 실패 이유 (통과 시 빈 문자열)
    selectivity_margin: float = 0.0     # 선택성 마진 (SSTR2 vs off-targets)
    offtarget_max_score: float = 0.0    # off-target 최대 결합 점수
    pass_selectivity: str = ""          # "PASS" or "FAIL" or "" (not evaluated)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RankTableRow":
        return cls(**d)


# =============================================================================
# Iteration Record (반복 실험 기록)
# 각 반복(iteration) 전체의 메타데이터 및 요약 기록
# =============================================================================

@dataclass
class IterationRecord:
    """
    단일 반복 실험의 메타데이터 및 결과 요약
    Metadata and result summary for one pipeline iteration.
    """
    run_id: str                             # 실행 ID (YYYYMMDD_HHMM_iterXX)
    iteration: int                          # 반복 번호 (1-based)
    hypothesis: str                         # 이번 반복의 가설 (자유 텍스트)
    parameter_changes: Dict[str, Any]       # 이전 반복 대비 변경된 파라미터
    results_summary: Dict[str, Any]         # 단계별 주요 결과 요약
    next_actions: List[str]                 # 다음 반복 계획 (목록)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "IterationRecord":
        return cls(**d)

    def to_json(self, indent: int = 2) -> str:
        """JSON 문자열로 직렬화 / Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "IterationRecord":
        """JSON 문자열에서 역직렬화 / Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Step 03b - BLOSUM62 Text-Level Mutation (Approach B)
# BLOSUM62 치환 행렬 기반 시퀀스 변이체 생성
# =============================================================================

@dataclass
class VariantEntry:
    """BLOSUM62 변이체 단위 항목 / Individual BLOSUM62 variant entry."""
    variant_id: str             # 고유 식별자: "var_001", "var_002", ...
    sequence: str               # 변이 시퀀스 (one-letter code)
    parent_sequence: str        # 원본(seed) 시퀀스
    mutations: List[str]        # 변이 목록: ["A1G", "K4R", ...]
    n_mutations: int            # 변이 수
    blosum_total_score: int     # 전체 BLOSUM62 점수 합
    source: str                 # "single_mutant" | "combinatorial"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariantEntry":
        return cls(**d)


@dataclass
class Step03bOutput:
    """
    Step 03b 출력: BLOSUM62 변이체 생성 결과
    Output of BLOSUM62 text-level mutation step (Approach B).
    """
    variants: List[VariantEntry]
    seed_sequence: str
    fixed_positions: Dict[int, str]
    total_generated: int
    strategy: str               # "approach_b"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variants": [v.to_dict() for v in self.variants],
            "seed_sequence": self.seed_sequence,
            "fixed_positions": self.fixed_positions,
            "total_generated": self.total_generated,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Step03bOutput":
        return cls(
            variants=[VariantEntry.from_dict(v) for v in d["variants"]],
            seed_sequence=d["seed_sequence"],
            fixed_positions={int(k): v for k, v in d["fixed_positions"].items()},
            total_generated=d["total_generated"],
            strategy=d["strategy"],
        )
