"""
orchestrator.py
===============
SSTR2 펩타이드 바인더 파이프라인 오케스트레이터
Main Pipeline Orchestrator for the SSTR2 Peptide Binder Co-Scientist System.

이 모듈이 전체 파이프라인의 핵심(CORE)이다.
This module is the CORE of the agentic pipeline.  It coordinates:

  * Step01 – Receptor preparation
  * Step02 – Backbone generation (RFdiffusion)
  * Step03 – Sequence design (ProteinMPNN)
  * Step04 – Fast QC (ESMFold)
  * Step05 – Docking (DiffDock + Boltz-2)
  * Step06 – Rosetta refinement
  * Step07 – Analysis & visualization

Agent roles invoked per iteration:
  1. Planner        – creates / updates the design plan
  2. Builder        – executes Steps 01-07
  3. QC & Ranker    – applies gates and ranks candidates
  4. Diversity Mgr  – ensures structural diversity
  5. Critic         – analyses results and proposes changes (max 2 changes)
  6. Reporter       – generates the iteration report

Convergence criteria:
  top-candidate ddG improvement < ``convergence_ddg_delta`` for
  ``convergence_patience`` consecutive iterations.

State persistence:
  Full pipeline state is serialised to JSON after each step for resume
  capability.  Resume from the last checkpoint with ``run(resume=True)``.

Usage (CLI):
    python -m AG_SRC.pipeline.orchestrator --config AG_SRC/config/pipeline_config.yaml
    python -m AG_SRC.pipeline.orchestrator --config ... --resume --run-id 20260217_1430_iter01
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema imports
# ---------------------------------------------------------------------------

from ..schemas.io_schemas import (
    Step01Output,
    Step02Output,
    Step03Output,
    Step03bOutput,
    VariantEntry,
    Step04Output,
    Step05Output,
    Step05bOutput,
    SelectivityResult,
    Step06Output,
    Step07Output,
    IterationRecord,
    RankTableRow,
    SequenceEntry,
)

# ---------------------------------------------------------------------------
# Pipeline step imports
# ---------------------------------------------------------------------------

from . import (
    step01_receptor,
    step02_backbone,
    step03_sequence,
    step03b_blosum_mutation,
    step04_qc,
    step05_docking,
    step05b_selectivity,
    step06_rosetta,
    step07_analysis,
)

# ---------------------------------------------------------------------------
# Agent imports (P0-1: replace rule-based stubs with real agent delegation)
# ---------------------------------------------------------------------------

from ..agents.base_agent import BaseAgent
from ..agents.planner import PlannerAgent, ExperimentPlan
from ..agents.qc_ranker import QCRankerAgent, Candidate, RankTable, QCReport
from ..agents.diversity_manager import DiversityManagerAgent
from ..agents.critic import ScientistCriticAgent, CriticAnalysis, ParameterChange
from ..agents.reporter import ReporterAgent
from .agent_output_validator import validate_agent_output
from ..llm import create_provider


# ---------------------------------------------------------------------------
# Result dataclasses (not in io_schemas to keep orchestrator concerns separate)
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field, asdict


@dataclass
class StepResult:
    """단일 파이프라인 단계 실행 결과 컨테이너."""
    step_name: str
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Serialize step output to dict if it has to_dict()
        if self.output and hasattr(self.output, "to_dict"):
            d["output"] = self.output.to_dict()
        return d


@dataclass
class IterationResult:
    """단일 반복(iteration) 전체 실행 결과."""
    iteration: int
    run_id: str
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    top_ddg: float = 0.0
    n_passed_final: int = 0
    hypothesis: str = ""
    next_actions: List[str] = field(default_factory=list)
    converged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "run_id": self.run_id,
            "step_results": {k: v.to_dict() for k, v in self.step_results.items()},
            "top_ddg": self.top_ddg,
            "n_passed_final": self.n_passed_final,
            "hypothesis": self.hypothesis,
            "next_actions": self.next_actions,
            "converged": self.converged,
        }


@dataclass
class AgentResponse:
    """에이전트 응답 컨테이너 (BaseAgent.execute() 위임 + stub fallback)."""
    agent_name: str
    content: Dict[str, Any]
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FinalResult:
    """전체 파이프라인 최종 결과."""
    run_id: str
    total_iterations: int
    iteration_records: List[IterationResult]
    best_candidates: List[Dict[str, Any]]
    converged: bool
    final_report_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_iterations": self.total_iterations,
            "iteration_records": [r.to_dict() for r in self.iteration_records],
            "best_candidates": self.best_candidates,
            "converged": self.converged,
            "final_report_path": self.final_report_path,
        }


# ---------------------------------------------------------------------------
# PipelineOrchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """SSTR2 펩타이드 바인더 설계 파이프라인의 메인 오케스트레이터.

    Coordinates all seven pipeline steps, six agent roles, and the
    iteration loop with convergence detection and state persistence.

    Attributes:
        config:          Merged pipeline configuration (from YAML files).
        gate_thresholds: QC gate threshold values.
        tool_registry:   Tool endpoint and capability registry.
        output_base:     Root output directory (``runs/``).
        _logger:         Module-level logger.
    """

    def __init__(self, config_path: str) -> None:
        """오케스트레이터를 초기화하고 설정 파일을 로드한다.

        Args:
            config_path: Path to pipeline_config.yaml.  The orchestrator
                         also looks for gate_thresholds.yaml and
                         tool_registry.yaml in the same directory.
        """
        self._logger = logging.getLogger(f"{__name__}.PipelineOrchestrator")
        self._configure_logging()

        config_path_obj = Path(config_path)
        config_dir = config_path_obj.parent

        self.config: Dict[str, Any] = self._load_yaml(config_path_obj)
        self.gate_thresholds: Dict[str, Any] = self._load_yaml(
            config_dir / "gate_thresholds.yaml", default={}
        )
        self.tool_registry: Dict[str, Any] = self._load_yaml(
            config_dir / "tool_registry.yaml", default={}
        )

        # Merge gate thresholds into config for downstream access
        self.config["gate_thresholds"] = self.gate_thresholds

        self.output_base: Path = Path(
            self.config.get("output_base_dir", "runs")
        )
        self._logger.info(
            "PipelineOrchestrator initialised from %s", config_path
        )

        # Initialize agent instances (P0-1)
        self._init_agents()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        max_iterations: Optional[int] = None,
        resume: bool = False,
        resume_run_id: Optional[str] = None,
    ) -> FinalResult:
        """파이프라인을 실행한다. 최대 max_iterations 반복.

        Args:
            max_iterations:  Override for the ``iteration.max_iterations``
                             config value.
            resume:          If True, load saved state from the last
                             checkpoint and continue from there.
            resume_run_id:   Explicit run_id to resume; required when
                             ``resume=True``.

        Returns:
            FinalResult summarising all iterations and best candidates.
        """
        max_iter: int = max_iterations or int(
            self.config.get("iteration", {}).get("max_iterations", 5)
        )
        convergence_delta: float = float(
            self.config.get("convergence_ddg_delta", 0.5)
        )
        convergence_patience: int = int(
            self.config.get("convergence_patience", 2)
        )

        start_iteration = 1
        state: Dict[str, Any] = {}
        iteration_records: List[IterationResult] = []
        results_history: List[IterationResult] = []

        if resume and resume_run_id:
            state = self._load_state(resume_run_id)
            start_iteration = state.get("next_iteration", 1)
            iteration_records = [
                IterationResult(**r) for r in state.get("iteration_records", [])
            ]
            results_history = iteration_records.copy()
            self._logger.info("Resuming from iteration %d (run=%s)", start_iteration, resume_run_id)

        no_improvement_count = 0
        # P1-1 fix: 0.0 causes first iteration to always register as "improvement"
        # Use inf so the first real ddG is always accepted as initial baseline
        previous_best_ddg: float = float("inf")

        for iteration in range(start_iteration, max_iter + 1):
            run_id, out_dirs = self._setup_run(iteration)
            self.config["run_id"] = run_id
            self.config["output_dirs"] = out_dirs

            self._logger.info(
                "=" * 60 + f"\n  ITERATION {iteration}/{max_iter}  run_id={run_id}\n" + "=" * 60
            )

            previous_results: Optional[IterationResult] = (
                results_history[-1] if results_history else None
            )

            try:
                iter_result = self.run_single_iteration(
                    iteration=iteration,
                    previous_results=previous_results,
                )
            except Exception as exc:
                self._logger.error(
                    "Iteration %d failed with unhandled exception: %s", iteration, exc,
                    exc_info=True,
                )
                # Save error state and stop
                self._save_state(
                    run_id=run_id,
                    iteration=iteration,
                    state={
                        "error": str(exc),
                        "iteration_records": [r.to_dict() for r in iteration_records],
                        "next_iteration": iteration,
                    },
                )
                break

            iteration_records.append(iter_result)
            results_history.append(iter_result)

            # Convergence check (P1-1: uses inf initial so first iter sets baseline)
            ddg_improvement = abs(previous_best_ddg - iter_result.top_ddg)
            if previous_best_ddg != float("inf") and ddg_improvement < convergence_delta:
                no_improvement_count += 1
                self._logger.info(
                    "No significant ddG improvement (delta=%.3f < %.3f). "
                    "Patience %d/%d.",
                    ddg_improvement,
                    convergence_delta,
                    no_improvement_count,
                    convergence_patience,
                )
            else:
                no_improvement_count = 0
            previous_best_ddg = iter_result.top_ddg

            # Persist state after each iteration
            self._save_state(
                run_id=run_id,
                iteration=iteration,
                state={
                    "iteration_records": [r.to_dict() for r in iteration_records],
                    "next_iteration": iteration + 1,
                    "previous_best_ddg": previous_best_ddg,
                },
            )

            if self._check_convergence(results_history):
                self._logger.info(
                    "Convergence detected at iteration %d.", iteration
                )
                iter_result.converged = True
                break

            if no_improvement_count >= convergence_patience:
                self._logger.info(
                    "Stopped after %d iterations without improvement.", convergence_patience
                )
                break

        # Aggregate final result
        best_candidates = self._aggregate_best_candidates(iteration_records)
        final_report_path = self._write_final_report(
            iteration_records, best_candidates, run_id=self.config.get("run_id", "final")
        )
        converged = any(r.converged for r in iteration_records)

        return FinalResult(
            run_id=self.config.get("run_id", "final"),
            total_iterations=len(iteration_records),
            iteration_records=iteration_records,
            best_candidates=best_candidates,
            converged=converged,
            final_report_path=final_report_path,
        )

    # ------------------------------------------------------------------
    # Single iteration
    # ------------------------------------------------------------------

    def run_single_iteration(
        self,
        iteration: int,
        previous_results: Optional[IterationResult] = None,
    ) -> IterationResult:
        """단일 반복을 실행하고 IterationResult를 반환한다.

        Sequence:
          1. Planner   -> update hypothesis / plan
          2. Builder   -> Step01 .. Step07
          3. QC&Ranker -> apply all gates, rank survivors
          4. Diversity -> ensure structural diversity
          5. Critic    -> propose changes for next iteration
          6. Reporter  -> write iteration report

        Args:
            iteration:        1-based iteration number.
            previous_results: IterationResult from the previous iteration
                              (None on first iteration).

        Returns:
            IterationResult populated with all step results.
        """
        run_id: str = self.config.get("run_id", "default_run")
        iter_result = IterationResult(iteration=iteration, run_id=run_id)

        # ------------------------------------------------------------------
        # 1. Planner agent
        # ------------------------------------------------------------------
        planner_resp = self._invoke_agent(
            "planner",
            context={
                "iteration": iteration,
                "previous_results": previous_results.to_dict() if previous_results else {},
                "config": self.config,
            },
        )
        iter_result.hypothesis = planner_resp.content.get(
            "hypothesis", f"Iteration {iteration}: default hypothesis"
        )
        # Apply parameter updates from planner
        param_updates = planner_resp.content.get("parameter_updates", {})
        if param_updates:
            self._apply_parameter_updates(param_updates)
            self._logger.info("[Planner] Applied parameter updates: %s", list(param_updates.keys()))

        # ------------------------------------------------------------------
        # 2. Builder: Steps 01 – 07
        # ------------------------------------------------------------------

        # Step 01: Receptor preparation
        step01_out: Optional[Step01Output] = None
        step01_result = self._execute_step(
            "step01_receptor",
            lambda: step01_receptor.prepare_receptor(self.config),
        )
        iter_result.step_results["step01"] = step01_result
        if not step01_result.success:
            self._logger.error("[Builder] Step01 failed; aborting iteration.")
            iter_result.next_actions = ["Fix receptor preparation before next iteration."]
            return iter_result
        step01_out = step01_result.output

        # Check if Approach B is enabled
        approach_b_cfg = self.config.get("approach_b", {})
        approach_b_enabled = approach_b_cfg.get("enabled", False)

        if approach_b_enabled:
            # ── Approach B: BLOSUM62 Text-Level Mutation ──
            self._logger.info("[Builder] Approach B enabled: skipping Step02/03, running Step03b.")

            # Step 03b: BLOSUM62 mutation
            step03b_result = self._execute_step(
                "step03b_blosum_mutation",
                lambda: step03b_blosum_mutation.run_approach_b(self.config),
            )
            iter_result.step_results["step03b"] = step03b_result
            if not step03b_result.success:
                self._logger.error("[Builder] Step03b failed; aborting iteration.")
                return iter_result
            step03b_out: Step03bOutput = step03b_result.output

            if not step03b_out.variants:
                self._logger.warning("[Builder] No variants generated; aborting iteration.")
                return iter_result

            # Stability pre-screening (if enabled)
            if approach_b_cfg.get("stability_prescreen", True):
                from .step08_stability import predict_half_life as _predict_hl
                min_hl = float(approach_b_cfg.get("stability_prescreen_min_hours", 50.0))
                pre_pass = []
                for v in step03b_out.variants:
                    hl = _predict_hl(v.sequence, [])
                    if hl >= min_hl:
                        pre_pass.append(v)
                self._logger.info(
                    "[Step03b-QC] Stability pre-screen: %d/%d passed (>= %.0fh)",
                    len(pre_pass), len(step03b_out.variants), min_hl,
                )
                variants_to_use = pre_pass
            else:
                variants_to_use = step03b_out.variants

            # Convert VariantEntry list to Step03Output for downstream compatibility
            seq_entries = [
                SequenceEntry(
                    backbone_idx=0,
                    seq_idx=i,
                    sequence=v.sequence,
                    fasta_path="",
                    seq_id=v.variant_id,
                )
                for i, v in enumerate(variants_to_use)
            ]
            step03_out = Step03Output(sequences=seq_entries)

        else:
            # ── Approach A: RFdiffusion → ProteinMPNN ──

            # Step 02: Backbone generation
            step02_result = self._execute_step(
                "step02_backbone",
                lambda: step02_backbone.generate_backbones(
                    receptor_pdb=step01_out.receptor_pdb_path,
                    pocket_info={"pocket_residues": step01_out.pocket_residues},
                    config=self.config,
                ),
            )
            iter_result.step_results["step02"] = step02_result
            if not step02_result.success:
                self._logger.error("[Builder] Step02 failed; aborting iteration.")
                return iter_result
            step02_out: Step02Output = step02_result.output

            if not step02_out.backbone_pdbs:
                self._logger.warning("[Builder] No backbones generated; aborting iteration.")
                return iter_result

            # Step 03: Sequence design
            step03_result = self._execute_step(
                "step03_sequence",
                lambda: step03_sequence.design_sequences(
                    backbones=step02_out.backbone_pdbs,
                    config=self.config,
                ),
            )
            iter_result.step_results["step03"] = step03_result
            if not step03_result.success:
                return iter_result
            step03_out: Step03Output = step03_result.output

        # Step 04: ESMFold QC
        step04_result = self._execute_step(
            "step04_qc",
            lambda: step04_qc.run_qc(
                sequences=step03_out.sequences,
                config=self.config,
            ),
        )
        iter_result.step_results["step04"] = step04_result
        if not step04_result.success:
            return iter_result
        step04_out: Step04Output = step04_result.output
        qc_passed = step04_out.passed()

        if not qc_passed:
            self._logger.warning(
                "[QC&Ranker] 0/%d candidates passed QC gate. Aborting iteration.",
                len(step04_out.qc_results),
            )
            return iter_result

        # ------------------------------------------------------------------
        # 3. QC & Ranker – log gate stats
        # ------------------------------------------------------------------
        qc_ranker_resp = self._invoke_agent(
            "qc_ranker",
            context={
                "qc_results": [r.to_dict() for r in step04_out.qc_results],
                "passed": len(qc_passed),
                "failed": len(step04_out.qc_results) - len(qc_passed),
            },
        )
        self._logger.info(
            "[QC&Ranker] %d/%d passed ESMFold gate.",
            len(qc_passed),
            len(step04_out.qc_results),
        )

        # Step 05: Docking
        step05_result = self._execute_step(
            "step05_docking",
            lambda: step05_docking.run_docking(
                candidates=qc_passed,
                receptor_pdb=step01_out.receptor_pdb_path,
                config=self.config,
            ),
        )
        iter_result.step_results["step05"] = step05_result
        if not step05_result.success:
            return iter_result
        step05_out: Step05Output = step05_result.output
        top_docking = step05_out.top_pct(
            pct=float(self.gate_thresholds.get("docking_top_pct", 20.0))
        )

        if not top_docking:
            self._logger.warning("[QC&Ranker] No candidates passed docking gate.")
            return iter_result

        # --- Step 05b: Selectivity Screening ---
        step05b_output: Optional[Step05bOutput] = None
        try:
            selectivity_config = {
                **self.config.get("selectivity", {}),
                "selectivity_margin_min": self.gate_thresholds.get("selectivity_margin_min", 10.0),   # G-2: 양수=좋음
                "offtarget_max_allowed": self.gate_thresholds.get("offtarget_max_allowed", -15.0),
            }
            step05b_output = step05b_selectivity.run_selectivity_screening(
                candidates=step05_out.docking_results if hasattr(step05_out, 'docking_results') else [],
                offtarget_receptors=self.config.get("off_target_receptors", []),
                config=selectivity_config,
            )
            self._logger.info(
                "[Step05b] %d/%d passed selectivity gate",
                len(step05b_output.passed_candidates()),
                len(step05b_output.selectivity_results),
            )
        except Exception as e:
            self._logger.warning("Step 05b selectivity screening failed: %s. Continuing without selectivity filter.", e)
            step05b_output = None

        # ------------------------------------------------------------------
        # 4. Diversity Manager – remove near-duplicate sequences
        # ------------------------------------------------------------------
        diversity_resp = self._invoke_agent(
            "diversity_manager",
            context={"docking_results": [r.to_dict() for r in top_docking]},
        )
        # P1-2 fix: next() without default raises StopIteration if seq_id
        # not found; use dict lookup for O(1) safety
        _docking_by_sid = {r.seq_id: r for r in top_docking}
        diverse_top = [
            _docking_by_sid[sid]
            for sid in diversity_resp.content.get(
                "selected_seq_ids", [r.seq_id for r in top_docking]
            )
            if sid in _docking_by_sid
        ]
        self._logger.info(
            "[Diversity] %d -> %d candidates after diversity filter.",
            len(top_docking),
            len(diverse_top),
        )

        # Step 06: Rosetta refinement
        step06_result = self._execute_step(
            "step06_rosetta",
            lambda: step06_rosetta.run_rosetta_refinement(
                candidates=diverse_top,
                receptor_pdb=step01_out.receptor_pdb_path,
                config=self.config,
            ),
        )
        iter_result.step_results["step06"] = step06_result
        if not step06_result.success:
            return iter_result
        step06_out: Step06Output = step06_result.output
        rosetta_passed, _ = step06_rosetta.apply_rosetta_gate(
            step06_out.rosetta_results,
            ddg_threshold=float(self.gate_thresholds.get("rosetta_ddg_max", -5.0)),
            clash_max=int(self.gate_thresholds.get("rosetta_clash_max", 0)),
        )

        if rosetta_passed:
            best_ddg = min(r.ddg for r in rosetta_passed)
            iter_result.top_ddg = best_ddg
            iter_result.n_passed_final = len(rosetta_passed)
            self._logger.info(
                "[QC&Ranker] %d candidates passed Rosetta gate. Best ddG=%.2f.",
                len(rosetta_passed),
                best_ddg,
            )
        else:
            self._logger.warning("[QC&Ranker] 0 candidates passed Rosetta gate.")

        # Step 07: Analysis & Visualization
        step07_result = self._execute_step(
            "step07_analysis",
            lambda: step07_analysis.run_analysis(
                candidates=step06_out.rosetta_results,
                receptor_pdb=step01_out.receptor_pdb_path,
                config=self.config,
            ),
        )
        iter_result.step_results["step07"] = step07_result

        # ------------------------------------------------------------------
        # 5. Critic agent
        # ------------------------------------------------------------------
        critic_resp = self._invoke_agent(
            "critic",
            context={
                "iteration": iteration,
                "hypothesis": iter_result.hypothesis,
                "top_ddg": iter_result.top_ddg,
                "n_passed_final": iter_result.n_passed_final,
                "step_results": {k: v.to_dict() for k, v in iter_result.step_results.items()},
                "previous_results": previous_results.to_dict() if previous_results else {},
            },
        )
        iter_result.next_actions = critic_resp.content.get("next_actions", [])[:2]

        # ------------------------------------------------------------------
        # 6. Reporter agent
        # ------------------------------------------------------------------
        reporter_resp = self._invoke_agent(
            "reporter",
            context={
                "iteration": iteration,
                "run_id": run_id,
                "iter_result": iter_result.to_dict(),
            },
        )
        report_path = reporter_resp.content.get("report_path", "")
        self._logger.info("[Reporter] Iteration report: %s", report_path)

        return iter_result

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_run(self, iteration: int) -> Tuple[str, Dict[str, str]]:
        """실행 ID와 출력 디렉토리 구조를 생성한다.

        Args:
            iteration: 1-based iteration counter.

        Returns:
            Tuple ``(run_id, out_dirs)`` where ``out_dirs`` maps step names
            to their absolute directory paths.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        run_id = f"{timestamp}_iter{iteration:02d}"

        out_base = self.output_base / run_id
        step_dirs = {
            "00_config": "00_config",
            "01_receptor": "01_receptor",
            "02_backbone": "02_backbone",
            "03_sequence": "03_sequence",
            "03b_blosum": "03b_blosum",
            "04_qc": "04_qc",
            "05_docking": "05_docking",
            "05b_selectivity": "05b_selectivity",
            "06_rosetta": "06_rosetta",
            "07_viz": "07_viz",
            "08_reports": "08_reports",
        }
        out_dirs: Dict[str, str] = {}
        for key, sub in step_dirs.items():
            d = out_base / sub
            d.mkdir(parents=True, exist_ok=True)
            out_dirs[key] = str(d)

        # Copy config files to 00_config
        config_src = Path(__file__).parent.parent / "config"
        config_dst = out_base / "00_config"
        for fname in ("pipeline_config.yaml", "gate_thresholds.yaml", "tool_registry.yaml"):
            src = config_src / fname
            if src.exists():
                shutil.copy(src, config_dst / fname)

        self._logger.info("[Setup] Run %s initialised at %s", run_id, out_base)
        return run_id, out_dirs

    # ------------------------------------------------------------------
    # Step execution with error handling
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step_name: str,
        step_fn: Any,
        max_retries: int = 1,
    ) -> StepResult:
        """파이프라인 단계를 실행하고 StepResult로 래핑한다.

        Args:
            step_name:   Human-readable step identifier for logging.
            step_fn:     Zero-argument callable that executes the step.
            max_retries: Number of retry attempts on failure.

        Returns:
            StepResult with success/failure status and output or error.
        """
        self._logger.info("[Step] Starting %s ...", step_name)
        t0 = time.monotonic()

        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                output = step_fn()
                elapsed = time.monotonic() - t0
                self._logger.info(
                    "[Step] %s completed in %.1fs.", step_name, elapsed
                )
                return StepResult(
                    step_name=step_name,
                    success=True,
                    output=output,
                    elapsed_sec=round(elapsed, 2),
                )
            except Exception as exc:
                last_exc = exc
                elapsed = time.monotonic() - t0
                if attempt < max_retries:
                    wait = 2 ** attempt
                    self._logger.warning(
                        "[Step] %s attempt %d/%d failed (%s). Retrying in %ds.",
                        step_name, attempt + 1, max_retries + 1, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    self._logger.error(
                        "[Step] %s failed after %d attempts: %s",
                        step_name, max_retries + 1, exc, exc_info=True,
                    )

        return StepResult(
            step_name=step_name,
            success=False,
            error=str(last_exc),
            elapsed_sec=round(time.monotonic() - t0, 2),
        )

    # ------------------------------------------------------------------
    # Agent registry initialisation (P0-1)
    # ------------------------------------------------------------------

    def _init_agents(self) -> None:
        """에이전트 인스턴스를 초기화하고 레지스트리에 등록한다."""
        self._agents: Dict[str, BaseAgent] = {}
        self._last_critic_analysis: Optional[CriticAnalysis] = None
        self._last_rank_table: Optional[RankTable] = None
        self._last_qc_report: Optional[QCReport] = None

        # Create LLM providers — M3 (2026-05-20):
        # agent별 override 지원 (`llm.agents.<name>`). 미지정 agent는 기본 llm 공유.
        self._llm_provider = create_provider(self.config)
        llm_planner = create_provider(self.config, agent_name="planner")
        llm_critic = create_provider(self.config, agent_name="critic")
        llm_reporter = create_provider(self.config, agent_name="reporter")
        llm = self._llm_provider  # 로그 표시용 (qc_ranker/diversity_manager 공유)

        try:
            self._agents["planner"] = PlannerAgent(llm_provider=llm_planner)
            # qc_ranker / diversity_manager는 code-based agent (LLM 미사용) — 기본 provider 공유
            self._agents["qc_ranker"] = QCRankerAgent(llm_provider=llm)
            self._agents["diversity_manager"] = DiversityManagerAgent(llm_provider=llm)
            self._agents["critic"] = ScientistCriticAgent(llm_provider=llm_critic)
            self._agents["reporter"] = ReporterAgent(
                runs_base_dir=str(self.output_base),
                llm_provider=llm_reporter,
            )
            self._logger.info(
                "Agent registry initialised: %s (LLM: %s)",
                list(self._agents.keys()), llm,
            )
        except Exception as e:
            self._logger.warning(
                "Agent init failed: %s. Using rule-based stubs.", e
            )
            self._agents = {}

    # ------------------------------------------------------------------
    # Agent invocation – real delegation with stub fallback (P0-1)
    # ------------------------------------------------------------------

    def _invoke_agent(
        self,
        agent_name: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """Invoke the specified agent via BaseAgent.execute() delegation.

        Adapts orchestrator context to each agent's expected format,
        calls agent.execute(), and maps the result back to AgentResponse.
        Falls back to rule-based stub on failure.

        Args:
            agent_name: Logical agent name (``planner``, ``critic``, etc.).
            context:    Dict of contextual data passed to the agent.

        Returns:
            AgentResponse with a content dict.
        """
        self._logger.debug(
            "[Agent] Invoking '%s' with context keys: %s",
            agent_name, list(context.keys()),
        )

        agent = getattr(self, "_agents", {}).get(agent_name)
        if agent is None:
            self._logger.debug(
                "[Agent] No instance for '%s'; using stub.", agent_name
            )
            return self._invoke_agent_stub(agent_name, context)

        try:
            adapted_ctx = self._adapt_agent_context(agent_name, context)
            result = agent.execute(adapted_ctx)

            # 에이전트 출력 스키마 검증 (P0-2)
            valid, validation_errors = validate_agent_output(agent_name, result)
            if not valid:
                self._logger.warning(
                    "[Agent] '%s' 출력 스키마 검증 실패 - stub으로 fallback. 오류: %s",
                    agent_name,
                    "; ".join(validation_errors),
                )
                return self._invoke_agent_stub(agent_name, context)

            content = self._map_agent_result(agent_name, result, context)
            self._logger.info("[Agent] '%s' executed successfully.", agent_name)
            return AgentResponse(agent_name=agent_name, content=content)
        except Exception as e:
            self._logger.warning(
                "[Agent] '%s' execution failed: %s. Falling back to stub.",
                agent_name, e,
            )
            return self._invoke_agent_stub(agent_name, context)

    # ------------------------------------------------------------------
    # Context adapters – orchestrator dict → agent-specific dict
    # ------------------------------------------------------------------

    def _adapt_agent_context(
        self, agent_name: str, context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform orchestrator context to each agent's expected format."""
        if agent_name == "planner":
            cfg = context.get("config", {})
            prev = context.get("previous_results", {})
            critic_fb = None
            if self._last_critic_analysis is not None:
                critic_fb = {
                    "proposed_changes": [
                        {"parameter_name": pc.parameter_name,
                         "new_value": pc.new_value,
                         "rationale": pc.rationale}
                        for pc in self._last_critic_analysis.proposed_changes
                    ],
                    "hypothesis": self._last_critic_analysis.hypothesis,
                }
            return {
                "iteration": context.get("iteration", 1),
                "receptor_config": cfg.get("receptor", {}),
                "constraints": cfg.get("constraints", {}),
                "critic_feedback": critic_fb,
                "previous_results": prev,
            }

        if agent_name == "qc_ranker":
            candidates = self._candidates_from_dicts(
                context.get("qc_results", [])
            )
            return {
                "candidates": candidates,
                "thresholds": self.gate_thresholds,
                "run_id": self.config.get("run_id", "default"),
                "iteration": context.get("iteration", 1),
            }

        if agent_name == "diversity_manager":
            candidates = self._candidates_from_dicts(
                context.get("docking_results", [])
            )
            n_select = int(
                self.config.get("iteration", {}).get("diversity_top_n", 20)
            )
            return {
                "candidates": candidates,
                "n_select": n_select,
            }

        if agent_name == "critic":
            rank_table = self._last_rank_table or self._build_empty_rank_table(
                context
            )
            qc_report = self._last_qc_report or self._build_empty_qc_report(
                context
            )
            return {
                "rank_table": rank_table,
                "qc_report": qc_report,
                "iteration": context.get("iteration", 1),
                "current_params": self.config.get("iteration", {}),
            }

        if agent_name == "reporter":
            return {
                "run_id": context.get("run_id", "default"),
                "iteration": context.get("iteration", 1),
                "rank_table": self._last_rank_table or self._build_empty_rank_table(context),
                "top_candidates": [],
                "receptor_pdb": self.config.get("receptor", {}).get("pdb_path", ""),
                "output_dir": str(
                    self.output_base / context.get("run_id", "default")
                ),
                "critic_analysis": self._last_critic_analysis,
            }

        return context

    # ------------------------------------------------------------------
    # Response mappers – agent result → orchestrator content dict
    # ------------------------------------------------------------------

    def _map_agent_result(
        self,
        agent_name: str,
        result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map agent execute() result to the content dict the orchestrator expects."""
        if agent_name == "planner":
            plan: Optional[ExperimentPlan] = result.get("plan")
            if plan is None:
                raise ValueError("Planner returned no plan")
            param_updates: Dict[str, Any] = {}
            if plan.changes_from_prev:
                param_updates = {
                    f"iteration.{k}": v for k, v in plan.parameters.items()
                }
            return {
                "hypothesis": plan.hypothesis,
                "parameter_updates": param_updates,
            }

        if agent_name == "qc_ranker":
            rank_table = result.get("rank_table")
            qc_report = result.get("qc_report")
            if rank_table is not None:
                self._last_rank_table = rank_table
            if qc_report is not None:
                self._last_qc_report = qc_report
            passed = qc_report.passed_count if qc_report else context.get("passed", 0)
            failed = qc_report.failed_count if qc_report else context.get("failed", 0)
            return {
                "ranking_comment": f"QC gate: {passed} passed, {failed} failed.",
            }

        if agent_name == "diversity_manager":
            diverse = result.get("diverse_candidates", [])
            if diverse:
                return {
                    "selected_seq_ids": [c.seq_id for c in diverse],
                }
            fallback = context.get("docking_results", [])
            return {
                "selected_seq_ids": [r["seq_id"] for r in fallback],
            }

        if agent_name == "critic":
            analysis: Optional[CriticAnalysis] = result.get("critic_analysis")
            if analysis is None:
                raise ValueError("Critic returned no analysis")
            self._last_critic_analysis = analysis
            next_actions = [
                f"{pc.parameter_name}: {pc.old_value} -> {pc.new_value} ({pc.rationale})"
                for pc in analysis.proposed_changes
            ]
            if not next_actions:
                next_actions = [analysis.hypothesis]
            return {"next_actions": next_actions[:2]}

        if agent_name == "reporter":
            report_paths = result.get("report_paths", {})
            first_path = next(iter(report_paths.values()), "") if report_paths else ""
            return {"report_path": first_path}

        return result

    # ------------------------------------------------------------------
    # Helpers – build typed objects from orchestrator dicts
    # ------------------------------------------------------------------

    @staticmethod
    def _candidates_from_dicts(records: List[Dict[str, Any]]) -> List[Candidate]:
        """dict 리스트에서 최소한의 Candidate 객체를 생성한다."""
        candidates: List[Candidate] = []
        for r in records:
            candidates.append(Candidate(
                candidate_id=str(r.get("candidate_id", r.get("seq_id", "unknown"))),
                backbone_id=int(r.get("backbone_id", 0)),
                seq_id=int(r.get("seq_id", 0)) if str(r.get("seq_id", "0")).isdigit()
                    else hash(str(r.get("seq_id", ""))) % 10000,
                sequence=str(r.get("sequence", "")),
                pdb_path=str(r.get("pdb_path", "")),
                plddt_mean=float(r.get("plddt_mean", 0.0)),
                plddt_interface=float(r.get("plddt_interface", 0.0)),
                dock_score=float(r.get("dock_score", r.get("score", 0.0))),
                ddg=float(r.get("ddg", 0.0)),
                lddt=float(r.get("lddt", 0.0)),
            ))
        return candidates

    def _build_empty_rank_table(self, context: Dict[str, Any]) -> RankTable:
        """컨텍스트에서 빈 RankTable을 생성한다."""
        return RankTable(
            run_id=self.config.get("run_id", context.get("run_id", "default")),
            iteration=context.get("iteration", 1),
            ranked_candidates=[],
            weights=dict(self.gate_thresholds.get("final_score_weights", {})),
        )

    def _build_empty_qc_report(self, context: Dict[str, Any]) -> QCReport:
        """컨텍스트에서 빈 QCReport을 생성한다."""
        n_passed = context.get("n_passed_final", 0)
        step_results = context.get("step_results", {})
        total = n_passed
        for sr in step_results.values():
            if sr.get("output") and isinstance(sr["output"], dict):
                total = max(total, sr["output"].get("total_count", total))
        return QCReport(
            run_id=self.config.get("run_id", "default"),
            total_input=total,
            passed_count=n_passed,
            failed_count=max(0, total - n_passed),
            failure_breakdown={},
            gates_applied=dict(self.gate_thresholds),
            pass_rate=n_passed / total if total > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Rule-based stub (legacy fallback)
    # ------------------------------------------------------------------

    def _invoke_agent_stub(
        self,
        agent_name: str,
        context: Dict[str, Any],
    ) -> AgentResponse:
        """Rule-based stub fallback for agent invocation.

        Preserved as a safety net when real agent execution fails.
        """
        content: Dict[str, Any] = {}

        if agent_name == "planner":
            iteration = context.get("iteration", 1)
            prev = context.get("previous_results", {})
            prev_actions = prev.get("next_actions", []) if prev else []
            hypothesis = (
                f"Iteration {iteration}: "
                + (prev_actions[0] if prev_actions else "Initial de-novo binder design for SSTR2.")
            )
            content = {
                "hypothesis": hypothesis,
                "parameter_updates": {},
            }

        elif agent_name == "qc_ranker":
            content = {
                "ranking_comment": (
                    f"QC gate: {context.get('passed', 0)} passed, "
                    f"{context.get('failed', 0)} failed."
                )
            }

        elif agent_name == "diversity_manager":
            results = context.get("docking_results", [])
            content = {
                "selected_seq_ids": [r["seq_id"] for r in results],
            }

        elif agent_name == "critic":
            top_ddg = context.get("top_ddg", 0.0)
            next_actions: List[str] = []
            if top_ddg >= 0:
                next_actions.append(
                    "Increase diffusion steps to 100 and hotspot weight to improve binding."
                )
            elif top_ddg > -5.0:
                next_actions.append(
                    "Tighten contigs to focus on TM2-TM5 pocket; reduce binder length to 15-20."
                )
            else:
                next_actions.append(
                    "Results look promising. Consider expanding to 15 backbones."
                )
            content = {"next_actions": next_actions[:2]}

        elif agent_name == "reporter":
            run_id = context.get("run_id", "unknown")
            iteration = context.get("iteration", 1)
            report_dir = self.output_base / run_id / "08_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"lab_notebook_iter{iteration:02d}.md"
            iter_result_dict = context.get("iter_result", {})
            top_ddg = iter_result_dict.get("top_ddg", 0.0)
            n_passed = iter_result_dict.get("n_passed_final", 0)
            report_lines = [
                f"# Lab Notebook – Iteration {iteration:02d}",
                f"**Run ID:** {run_id}",
                f"**Hypothesis:** {iter_result_dict.get('hypothesis', '')}",
                "",
                f"**Best ddG:** {top_ddg:.2f} kcal/mol",
                f"**Candidates passing all gates:** {n_passed}",
                "",
                "## Next Actions",
            ] + [f"- {a}" for a in iter_result_dict.get("next_actions", [])]
            report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            content = {"report_path": str(report_path)}

        else:
            self._logger.warning("[Agent] Unknown agent '%s'; returning empty response.", agent_name)

        return AgentResponse(agent_name=agent_name, content=content)

    # ------------------------------------------------------------------
    # Convergence detection
    # ------------------------------------------------------------------

    def _check_convergence(self, results_history: List[IterationResult]) -> bool:
        """수렴 기준을 평가한다.

        Convergence is declared when the best ddG improvement across the
        last two iterations is below ``convergence_ddg_delta``.

        Args:
            results_history: All IterationResult records so far.

        Returns:
            True when convergence criteria are met.
        """
        patience: int = int(self.config.get("convergence_patience", 2))
        delta: float = float(self.config.get("convergence_ddg_delta", 0.5))

        if len(results_history) < patience + 1:
            return False

        recent = results_history[-(patience + 1):]
        ddg_values = [r.top_ddg for r in recent if r.top_ddg < 0]
        if len(ddg_values) < patience:
            return False

        improvements = [
            abs(ddg_values[i] - ddg_values[i + 1]) for i in range(len(ddg_values) - 1)
        ]
        converged = all(imp < delta for imp in improvements)
        if converged:
            self._logger.info(
                "Convergence check: improvements=%s all < %.3f -> CONVERGED",
                [round(x, 3) for x in improvements],
                delta,
            )
        return converged

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_state(
        self,
        run_id: str,
        iteration: int,
        state: Dict[str, Any],
    ) -> str:
        """파이프라인 상태를 JSON 체크포인트 파일로 저장한다.

        Args:
            run_id:    Current run identifier.
            iteration: Current iteration number.
            state:     Serialisable state dict.

        Returns:
            Path to the written checkpoint file.
        """
        state_dir = self.output_base / run_id / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / f"checkpoint_iter{iteration:02d}.json"
        full_state = {
            "run_id": run_id,
            "iteration": iteration,
            "saved_at": datetime.utcnow().isoformat(),
            **state,
        }
        state_path.write_text(
            json.dumps(full_state, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        self._logger.info("[State] Checkpoint saved -> %s", state_path)
        return str(state_path)

    def _load_state(self, run_id: str) -> Dict[str, Any]:
        """가장 최근 체크포인트 파일에서 파이프라인 상태를 로드한다.

        Args:
            run_id: Run identifier whose checkpoints to search.

        Returns:
            State dict from the latest checkpoint.

        Raises:
            FileNotFoundError: When no checkpoint is found for *run_id*.
        """
        state_dir = self.output_base / run_id / "state"
        checkpoints = sorted(state_dir.glob("checkpoint_iter*.json"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoints found for run_id='{run_id}' in {state_dir}"
            )
        latest = checkpoints[-1]
        self._logger.info("[State] Loading checkpoint from %s", latest)
        return json.loads(latest.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Parameter update (from Planner)
    # ------------------------------------------------------------------

    # P1-3 fix: whitelist of keys Planner is allowed to modify
    _ALLOWED_PARAM_KEYS: frozenset = frozenset({
        "iteration.n_backbone", "iteration.k_seq", "iteration.contigs",
        "iteration.hotspot_res", "iteration.peptide_length_min",
        "iteration.peptide_length_max", "iteration.mpnn_temperature",
        "iteration.mpnn_sampling_n", "iteration.docking_engine",
        "iteration.rosetta_relax_cycles", "iteration.seed",
        "iteration.diversity_top_n",
    })

    def _apply_parameter_updates(self, updates: Dict[str, Any]) -> None:
        """Planner 에이전트가 제안한 파라미터 변경을 config에 적용한다.

        Supports flat key updates (``"key": value``) and nested updates
        using dot-notation (``"iteration.n_backbone": 15``).

        P1-3 fix: validates keys against whitelist to prevent arbitrary
        config injection from LLM outputs.

        Args:
            updates: Dict of config key -> new value pairs.
        """
        for key, value in updates.items():
            if key not in self._ALLOWED_PARAM_KEYS:
                self._logger.warning(
                    "[Config] Blocked disallowed parameter update: %s = %s", key, value
                )
                continue
            if "." in key:
                parts = key.split(".", 1)
                section = parts[0]
                sub_key = parts[1]
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][sub_key] = value
                self._logger.info("[Config] Updated %s.%s = %s", section, sub_key, value)
            else:
                self.config[key] = value
                self._logger.info("[Config] Updated %s = %s", key, value)

    # ------------------------------------------------------------------
    # Final result aggregation
    # ------------------------------------------------------------------

    def _aggregate_best_candidates(
        self, iteration_records: List[IterationResult]
    ) -> List[Dict[str, Any]]:
        """모든 반복에서 최상위 후보를 집계한다.

        Args:
            iteration_records: All IterationResult records.

        Returns:
            List of dicts describing the best candidates (sorted by ddG).
        """
        best: List[Dict[str, Any]] = []
        for record in iteration_records:
            step06 = record.step_results.get("step06")
            if not step06 or not step06.success or not step06.output:
                continue
            step06_out = step06.output
            if not hasattr(step06_out, "rosetta_results"):
                continue
            for r in step06_out.rosetta_results:
                best.append({
                    "iteration": record.iteration,
                    "seq_id": r.seq_id,
                    "ddg": r.ddg,
                    "total_score": r.total_score,
                    "clash_score": r.clash_score,
                    "refined_pdb": r.refined_pdb,
                })

        return sorted(best, key=lambda x: x.get("ddg", 0.0))[:10]

    def _write_final_report(
        self,
        iteration_records: List[IterationResult],
        best_candidates: List[Dict[str, Any]],
        run_id: str,
    ) -> str:
        """전체 실험 최종 보고서를 마크다운으로 생성한다."""
        report_dir = self.output_base / run_id / "08_reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "final_report.md"

        lines = [
            "# SSTR2 Peptide Binder Design – Final Report",
            f"**Run ID:** {run_id}",
            f"**Total Iterations:** {len(iteration_records)}",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "## Summary",
            f"Completed {len(iteration_records)} iteration(s).",
            f"Best ddG: {best_candidates[0]['ddg']:.2f} kcal/mol ({best_candidates[0]['seq_id']})"
            if best_candidates else "No passing candidates found.",
            "",
            "## Top Candidates",
            "",
            "| Rank | seq_id | ddG (kcal/mol) | Iteration | refined_pdb |",
            "|------|--------|---------------|-----------|-------------|",
        ]
        for rank, c in enumerate(best_candidates, 1):
            lines.append(
                f"| {rank} | {c['seq_id']} | {c['ddg']:.2f} | {c['iteration']} | "
                f"`{Path(c['refined_pdb']).name if c['refined_pdb'] else 'N/A'}` |"
            )

        lines += ["", "## Iteration History", ""]
        for record in iteration_records:
            lines.append(
                f"- **Iteration {record.iteration}**: "
                f"hypothesis='{record.hypothesis}', "
                f"best ddG={record.top_ddg:.2f}, "
                f"n_passed={record.n_passed_final}"
            )

        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._logger.info("[Reporter] Final report written -> %s", report_path)
        return str(report_path)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_logging() -> None:
        """루트 로거에 콘솔 핸들러를 설정한다 (중복 방지)."""
        root = logging.getLogger()
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root.addHandler(handler)
            root.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # YAML helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """YAML 파일을 로드하고 dict를 반환한다. 파일 없으면 default 반환."""
        if not path.exists():
            logger.warning("Config file not found: %s. Using default.", path)
            return default or {}
        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI 진입점: argparse로 파이프라인을 실행한다."""
    parser = argparse.ArgumentParser(
        description="SSTR2 Peptide Binder Design Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"),
        help="Path to pipeline_config.yaml",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override max_iterations from config",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to resume (required with --resume)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s][%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.resume and not args.run_id:
        parser.error("--run-id is required when using --resume")

    orchestrator = PipelineOrchestrator(config_path=args.config)
    result = orchestrator.run(
        max_iterations=args.max_iterations,
        resume=args.resume,
        resume_run_id=args.run_id,
    )

    print("\n" + "=" * 60)
    print(f"Pipeline complete.")
    print(f"  Run ID          : {result.run_id}")
    print(f"  Total iterations: {result.total_iterations}")
    print(f"  Converged       : {result.converged}")
    print(f"  Best candidates : {len(result.best_candidates)}")
    if result.best_candidates:
        top = result.best_candidates[0]
        print(f"  Top ddG         : {top['ddg']:.2f} kcal/mol ({top['seq_id']})")
    print(f"  Final report    : {result.final_report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
