"""
agents/__init__.py
SSTR2 펩타이드 바인더 Co-Scientist - 에이전트 패키지
Agent package for the SSTR2 peptide binder Co-Scientist pipeline.

6개의 에이전트 클래스와 공통 데이터 구조를 공개(export)한다.

에이전트 역할 요약:
    PlannerAgent        - 연구 설계 / 실험 기획 (Research Design & Experiment Planning)
    BuilderAgent        - 실행 오케스트레이터 (Execution Orchestrator)
    QCRankerAgent       - 품질관리 및 랭킹 (Quality Control & Ranking)
    ScientistCriticAgent - 비판적 검토 / 원인 분석 (Critical Review & Root Cause Analysis)
    ReporterAgent       - 최종 리포트 / 그림 자동화 (Final Report & Automated Visualization)
    DiversityManagerAgent - 구조 다양성 관리 (Structural Diversity Management)

파이프라인 메시지 흐름:
    Planner -> Builder -> QCRanker -> DiversityManager -> ScientistCritic -> Reporter
                ^                                               |
                +----------- (다음 iteration 계획 갱신) --------+
"""

from .base_agent import AgentMessage, BaseAgent, MessageType
from .builder import (
    Action,
    BuilderAgent,
    BuilderStepResult,
    FALLBACK_PATHS,
    PipelineResult,
)
from .critic import (
    CriticAnalysis,
    FAILURE_ACTION_MAP,
    FailureType,
    ParameterChange,
    ScientistCriticAgent,
)
from .diversity_manager import (
    Cluster,
    DiversityManagerAgent,
    DiversityReport,
)
from .planner import (
    ExperimentPlan,
    PlannerAgent,
    StepConfig,
)
from .qc_ranker import (
    Candidate,
    DEFAULT_WEIGHTS,
    QCRankerAgent,
    QCReport,
    RankTable,
)
from .reporter import (
    RenderPaths,
    ReporterAgent,
)

__all__ = [
    # Base
    "BaseAgent",
    "AgentMessage",
    "MessageType",
    # Planner
    "PlannerAgent",
    "ExperimentPlan",
    "StepConfig",
    # Builder
    "BuilderAgent",
    "BuilderStepResult",
    "PipelineResult",
    "Action",
    "FALLBACK_PATHS",
    # QC & Ranker
    "QCRankerAgent",
    "Candidate",
    "RankTable",
    "QCReport",
    "DEFAULT_WEIGHTS",
    # Scientist Critic
    "ScientistCriticAgent",
    "CriticAnalysis",
    "ParameterChange",
    "FailureType",
    "FAILURE_ACTION_MAP",
    # Reporter
    "ReporterAgent",
    "RenderPaths",
    # Diversity Manager
    "DiversityManagerAgent",
    "Cluster",
    "DiversityReport",
]
