"""
builder.py
SSTR2 펩타이드 바인더 Co-Scientist - Builder (실행 오케스트레이터) 에이전트
Role: 실행 오케스트레이터 (Execution Orchestrator)

Builder는 ExperimentPlan을 받아 Step01~07을 순서대로 실행하고,
실패 시 재시도/대체 경로를 관리하며, 모든 실행 기록을 저장한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from .base_agent import BaseAgent, MessageType
from .planner import ExperimentPlan, StepConfig


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Action(str, Enum):
    """실패 처리 액션."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ABORT = "abort"
    SKIP = "skip"


@dataclass
class BuilderStepResult:
    """단일 스텝 실행 결과.

    Attributes:
        step_id: 스텝 식별자
        status: 'success' | 'failed' | 'skipped'
        output_paths: 생성된 파일 경로 목록
        metrics: 스텝별 핵심 수치 (pLDDT, dock score 등)
        error_message: 실패 시 오류 메시지
        retry_count: 최종 성공까지 시도한 횟수
        duration_seconds: 실행 소요 시간 (초)
        fallback_used: 대체 경로 사용 여부
    """
    step_id: str
    status: str  # 'success' | 'failed' | 'skipped'
    output_paths: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    duration_seconds: float = 0.0
    fallback_used: bool = False


@dataclass
class PipelineResult:
    """전체 파이프라인 실행 결과.

    Attributes:
        run_id: 실행 식별자
        iteration: 반복 번호
        status: 'success' | 'partial' | 'failed'
        step_results: 각 스텝별 BuilderStepResult 목록
        top_candidates: QC 통과 후보 식별자 목록
        total_duration_seconds: 전체 소요 시간
        log_path: 실행 로그 저장 경로
    """
    run_id: str
    iteration: int
    status: str
    step_results: list[BuilderStepResult] = field(default_factory=list)
    top_candidates: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    log_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Retry / fallback policy constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2  # exponential backoff: 2^retry_count 초

# 대체 경로 매핑: (스텝 id, 실패 조건) -> 대체 스텝 설정
FALLBACK_PATHS: dict[str, dict[str, Any]] = {
    "step05_docking": {
        "condition": "always",
        "fallback_tool": "run_docking",
        "fallback_params": {"engine": "boltz2"},
        "description": "DiffDock 실패 시 Boltz-2로 대체",
    },
    "step06_pyrosetta": {
        "condition": "always",
        "fallback_tool": None,  # 스킵
        "fallback_params": {},
        "description": "PyRosetta 실패 시 정제 단계 건너뛰고 리포트에 flag",
    },
    "step01_openfold3": {
        "condition": "always",
        "fallback_tool": "load_preexisting_structure",
        "fallback_params": {"source": "data/sstr2_reference.pdb"},
        "description": "OpenFold3 실패 시 data/ 디렉터리의 기존 SSTR2 구조 사용",
    },
}


# ---------------------------------------------------------------------------
# Builder agent
# ---------------------------------------------------------------------------

class BuilderAgent(BaseAgent):
    """실행 오케스트레이터 에이전트.

    역할:
        1. Step01~07 실행 순서 관리
        2. 실패 시 최대 3회 재시도 (지수 백오프)
        3. 재시도 실패 시 대체 경로 (fallback) 수행
        4. 모든 실행 로그/설정을 runs/{run_id}/00_config/ 에 저장
        5. Step02 백본 생성은 seed 분할로 병렬 실행 지원

    Attributes:
        runs_base_dir: 실험 결과를 저장할 루트 디렉터리
        tool_registry: 도구 이름 -> 실제 callable 매핑 (외부 주입)
        dry_run: True이면 실제 도구 호출 없이 로직만 검증
    """

    def __init__(
        self,
        runs_base_dir: str = "runs",
        tool_registry: Optional[dict[str, Callable[..., Any]]] = None,
        dry_run: bool = False,
        llm_provider: str = "claude",
    ) -> None:
        super().__init__(
            name="Builder",
            role="실행 오케스트레이터",
            description=(
                "ExperimentPlan을 받아 Step01~07을 순서대로 실행하며, "
                "실패 시 재시도·대체 경로를 관리하고 모든 실행 기록을 저장한다."
            ),
            llm_provider=llm_provider,
        )
        self.runs_base_dir = Path(runs_base_dir)
        self.tool_registry: dict[str, Callable[..., Any]] = tool_registry or {}
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_pipeline(self, plan: ExperimentPlan) -> PipelineResult:
        """ExperimentPlan 전체를 순서대로 실행한다.

        Step02(RFdiffusion)는 seed를 분할하여 병렬로 실행 가능.
        각 스텝 결과는 누적되어 다음 스텝의 입력으로 전달된다.

        Args:
            plan: Planner가 생성한 ExperimentPlan

        Returns:
            PipelineResult: 전체 실행 요약
        """
        self.log(f"파이프라인 실행 시작: {plan.run_id} (iteration {plan.iteration})")
        start = time.time()

        run_dir = self.runs_base_dir / plan.run_id
        self._ensure_dir_structure(run_dir)
        self.save_execution_log(plan.run_id, "pipeline_start", plan.parameters, {})

        step_results: list[BuilderStepResult] = []
        shared_state: dict[str, Any] = {
            "run_id": plan.run_id,
            "run_dir": str(run_dir),
            "parameters": plan.parameters,
            "gates": plan.gates,
        }

        overall_status = "success"

        for step_config in plan.steps_config:
            if not step_config.enabled:
                self.log(f"스텝 건너뜀 (disabled): {step_config.step_id}")
                step_results.append(BuilderStepResult(step_id=step_config.step_id, status="skipped"))
                continue

            result = self._run_step_with_retry(step_config, shared_state, plan)
            step_results.append(result)

            if result.status == "failed":
                self.log(
                    f"스텝 실패로 파이프라인 중단: {step_config.step_id}",
                    level="error",
                )
                overall_status = "failed"
                break

            if result.status == "skipped" and step_config.step_id == "step06_pyrosetta":
                # PyRosetta 스킵은 partial로 기록하지만 계속 진행
                overall_status = "partial"

            # 실행 상태를 shared_state에 반영
            shared_state[f"{step_config.step_id}_result"] = result
            self.save_execution_log(plan.run_id, step_config.step_id, step_config.params, result.metrics)

        elapsed = time.time() - start
        log_path = str(run_dir / "00_config" / "execution_log.json")

        pipeline_result = PipelineResult(
            run_id=plan.run_id,
            iteration=plan.iteration,
            status=overall_status,
            step_results=step_results,
            top_candidates=shared_state.get("top_candidates", []),
            total_duration_seconds=elapsed,
            log_path=log_path,
        )

        self.log(f"파이프라인 완료: {overall_status} ({elapsed:.1f}초)")
        return pipeline_result

    def execute_step(
        self,
        step_name: str,
        input_data: dict[str, Any],
        config: dict[str, Any],
    ) -> BuilderStepResult:
        """단일 스텝을 실행하고 결과를 반환한다.

        tool_registry에 등록된 callable을 호출한다.
        dry_run=True이면 실제 호출 없이 더미 결과를 반환한다.

        Args:
            step_name: 스텝 식별자 (예: 'step02_rfdiffusion')
            input_data: 스텝 입력 데이터
            config: 스텝별 실행 파라미터

        Returns:
            BuilderStepResult: 실행 결과
        """
        self.log(f"  스텝 실행: {step_name}")
        t0 = time.time()

        if self.dry_run:
            self.log(f"  [DRY RUN] {step_name} 실행 시뮬레이션")
            return BuilderStepResult(
                step_id=step_name,
                status="success",
                output_paths=[f"<dry_run>/{step_name}/output"],
                metrics={"dry_run": True},
                duration_seconds=time.time() - t0,
            )

        tool_fn = self.tool_registry.get(step_name)
        if tool_fn is None:
            # tool이 등록되지 않은 경우 - 실제 환경에서는 오류
            self.log(f"  도구 미등록: {step_name}", level="warning")
            return BuilderStepResult(
                step_id=step_name,
                status="failed",
                error_message=f"tool_registry에 '{step_name}' 미등록",
                duration_seconds=time.time() - t0,
            )

        try:
            raw_result = tool_fn(input_data=input_data, config=config)
            return BuilderStepResult(
                step_id=step_name,
                status="success",
                output_paths=raw_result.get("output_paths", []),
                metrics=raw_result.get("metrics", {}),
                duration_seconds=time.time() - t0,
            )
        except Exception as exc:
            return BuilderStepResult(
                step_id=step_name,
                status="failed",
                error_message=str(exc),
                duration_seconds=time.time() - t0,
            )

    def handle_failure(
        self,
        step_name: str,
        error: Exception,
        retry_count: int,
    ) -> Action:
        """스텝 실패 시 취할 액션을 결정한다.

        정책:
        - retry_count < MAX_RETRIES: RETRY (지수 백오프 대기)
        - 대체 경로(fallback) 존재: FALLBACK
        - 그 외: ABORT

        Args:
            step_name: 실패한 스텝 이름
            error: 발생한 예외
            retry_count: 현재까지 시도 횟수

        Returns:
            Action: 다음에 취할 액션
        """
        self.log(
            f"실패 처리: {step_name} (시도 {retry_count}/{MAX_RETRIES}): {error}",
            level="warning",
        )

        if retry_count < MAX_RETRIES:
            wait = BACKOFF_BASE_SECONDS ** retry_count
            self.log(f"  재시도 대기: {wait}초")
            if not self.dry_run:
                time.sleep(wait)
            return Action.RETRY

        if step_name in FALLBACK_PATHS:
            fb = FALLBACK_PATHS[step_name]
            self.log(f"  대체 경로 사용: {fb['description']}")
            if fb["fallback_tool"] is None:
                return Action.SKIP
            return Action.FALLBACK

        self.log(f"  복구 불가 - 파이프라인 중단", level="error")
        return Action.ABORT

    def save_execution_log(
        self,
        run_id: str,
        step_name: str,
        config: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """실행 설정과 결과를 JSON 로그 파일에 기록한다.

        저장 경로: runs/{run_id}/00_config/{step_name}_log.json

        Args:
            run_id: 실행 식별자
            step_name: 스텝 이름
            config: 실행 파라미터 딕셔너리
            result: 결과 딕셔너리
        """
        log_dir = self.runs_base_dir / run_id / "00_config"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{step_name}_log.json"

        entry = {
            "run_id": run_id,
            "step": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "result_summary": result,
        }

        # append-friendly: 기존 파일이 있으면 리스트로 합산
        if log_file.exists():
            with open(log_file, "r") as f:
                try:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                except json.JSONDecodeError:
                    existing = []
            existing.append(entry)
            payload = existing
        else:
            payload = [entry]

        with open(log_file, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """컨텍스트에서 ExperimentPlan을 꺼내 파이프라인을 실행한다.

        context 키:
            - plan (ExperimentPlan): Planner가 생성한 계획

        Returns:
            {'status': str, 'pipeline_result': PipelineResult}
        """
        plan: ExperimentPlan = context["plan"]
        result = self.execute_pipeline(plan)
        return {"status": result.status, "pipeline_result": result}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_step_with_retry(
        self,
        step_config: StepConfig,
        shared_state: dict[str, Any],
        plan: ExperimentPlan,
    ) -> BuilderStepResult:
        """재시도·대체 경로 로직을 포함한 스텝 실행 래퍼."""
        retry_count = 0
        input_data = self._build_input(step_config, shared_state)

        while True:
            result = self.execute_step(step_config.step_id, input_data, step_config.params)

            if result.status == "success":
                result.retry_count = retry_count
                return result

            action = self.handle_failure(
                step_config.step_id, Exception(result.error_message or "unknown"), retry_count
            )

            if action == Action.RETRY:
                retry_count += 1
                continue

            if action == Action.FALLBACK:
                fb = FALLBACK_PATHS[step_config.step_id]
                fb_config = {**step_config.params, **fb["fallback_params"]}
                fb_result = self.execute_step(step_config.step_id, input_data, fb_config)
                fb_result.retry_count = retry_count
                fb_result.fallback_used = True
                return fb_result

            if action == Action.SKIP:
                self.log(f"  스텝 건너뜀 (fallback=skip): {step_config.step_id}", level="warning")
                return BuilderStepResult(
                    step_id=step_config.step_id,
                    status="skipped",
                    retry_count=retry_count,
                    fallback_used=True,
                    error_message=f"대체 경로: 스킵 처리 ({FALLBACK_PATHS[step_config.step_id]['description']})",
                )

            # ABORT
            result.retry_count = retry_count
            return result

    @staticmethod
    def _build_input(step_config: StepConfig, shared_state: dict[str, Any]) -> dict[str, Any]:
        """shared_state에서 스텝이 필요로 하는 입력 딕셔너리를 구성한다."""
        return {
            "expected_inputs": step_config.input_schema,
            "shared_state": shared_state,
        }

    def _ensure_dir_structure(self, run_dir: Path) -> None:
        """runs/{run_id}/ 하위 표준 폴더 구조를 생성한다."""
        subdirs = [
            "00_config", "01_receptor", "02_backbone", "03_sequence",
            "04_qc", "05_docking", "06_rosetta", "07_viz", "08_reports",
        ]
        for sub in subdirs:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        self.log(f"디렉터리 구조 생성: {run_dir}")
