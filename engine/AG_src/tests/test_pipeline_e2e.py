"""
test_pipeline_e2e.py
====================
End-to-end dry-run test for the SSTR2 Peptide Binder Design pipeline.

Mocks all 7 pipeline steps (step01-step07 + step05b) to return valid test
data without calling real APIs, then exercises the full PipelineOrchestrator
run() flow for 2 iterations.

Verifies:
  - All 5 agents are invoked (Planner, QCRanker, DiversityMgr, Critic, Reporter)
  - All 7 steps execute in order
  - QC gates filter correctly (4 pass, 2 fail in step04)
  - Convergence detection works
  - Final report is generated
  - Step05b selectivity screening runs

Run with:
  python -m pytest AG_src/tests/test_pipeline_e2e.py -v
  python -m unittest AG_src.tests.test_pipeline_e2e
"""

from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import schemas
# ---------------------------------------------------------------------------
from AG_src.schemas.io_schemas import (
    DockingResult,
    OffTargetDockingResult,
    QCResult,
    RosettaResult,
    SelectivityResult,
    SequenceEntry,
    Step01Output,
    Step02Output,
    Step03Output,
    Step04Output,
    Step05Output,
    Step05bOutput,
    Step06Output,
    Step07Output,
)

# ---------------------------------------------------------------------------
# Mock return values
# ---------------------------------------------------------------------------

def _make_step01_output() -> Step01Output:
    return Step01Output(
        receptor_pdb_path="/tmp/test_sstr2.pdb",
        pocket_residues=[122, 127, 184],
        chain_id="B",
        pocket_json_path="/tmp/pocket_residues.json",
    )


def _make_step02_output() -> Step02Output:
    return Step02Output(
        backbone_pdbs=[
            "/tmp/backbone_00.pdb",
            "/tmp/backbone_01.pdb",
            "/tmp/backbone_02.pdb",
        ],
        design_params={"contigs": "B1-369/0 10-30", "hotspot_res": ["B122", "B127", "B184"]},
        n_generated=3,
    )


def _make_step03_output() -> Step03Output:
    sequences = []
    for bb_idx in range(3):
        for seq_idx in range(2):
            seq_id = f"bb{bb_idx:02d}_seq{seq_idx:02d}"
            sequences.append(SequenceEntry(
                backbone_idx=bb_idx,
                seq_idx=seq_idx,
                sequence=f"ACDEFGHIKLMN"[: 10 + bb_idx + seq_idx],
                fasta_path=f"/tmp/{seq_id}.fasta",
                seq_id=seq_id,
            ))
    return Step03Output(sequences=sequences)


def _make_step04_output() -> Step04Output:
    """4 pass (pLDDT > 75), 2 fail (pLDDT < 75)."""
    qc_results = [
        QCResult(seq_id="bb00_seq00", plddt_mean=85.0, plddt_interface=80.0,
                 pdb_path="/tmp/bb00_seq00_esm.pdb", passed_gate=True),
        QCResult(seq_id="bb00_seq01", plddt_mean=78.5, plddt_interface=76.0,
                 pdb_path="/tmp/bb00_seq01_esm.pdb", passed_gate=True),
        QCResult(seq_id="bb01_seq00", plddt_mean=82.1, plddt_interface=79.3,
                 pdb_path="/tmp/bb01_seq00_esm.pdb", passed_gate=True),
        QCResult(seq_id="bb01_seq01", plddt_mean=77.0, plddt_interface=74.0,
                 pdb_path="/tmp/bb01_seq01_esm.pdb", passed_gate=True),
        QCResult(seq_id="bb02_seq00", plddt_mean=60.0, plddt_interface=58.0,
                 pdb_path="/tmp/bb02_seq00_esm.pdb", passed_gate=False),
        QCResult(seq_id="bb02_seq01", plddt_mean=55.5, plddt_interface=52.0,
                 pdb_path="/tmp/bb02_seq01_esm.pdb", passed_gate=False),
    ]
    return Step04Output(qc_results=qc_results)


def _make_step05_output(qc_passed: List[QCResult]) -> Step05Output:
    """Docking scores ranging -3.0 to -9.0 for passed QC candidates."""
    base_scores = [-9.0, -7.5, -5.0, -3.0]
    docking_results = []
    for i, qc in enumerate(qc_passed):
        score = base_scores[i] if i < len(base_scores) else -4.0
        docking_results.append(DockingResult(
            seq_id=qc.seq_id,
            engine="diffdock",
            score=score,
            confidence=0.85 - i * 0.05,
            pose_pdb=f"/tmp/{qc.seq_id}_pose_1.pdb",
            rank=1,
        ))
    return Step05Output(docking_results=docking_results)


def _make_step05b_output(docking_results: List[DockingResult]) -> Step05bOutput:
    """3 pass selectivity, 1 fail."""
    selectivity_results = []
    offtarget_details = []
    for i, dr in enumerate(docking_results):
        passed = (i < 3)  # first 3 pass, last fails
        margin = 15.0 if passed else 0.5  # G-2: 양수=좋음; 15.0 >= 10.0 PASS, 0.5 < 10.0 FAIL
        selectivity_results.append(SelectivityResult(
            seq_id=dr.seq_id,
            sstr2_dock_score=dr.score,
            offtarget_scores={"SSTR1": dr.score + 3.0, "SSTR3": dr.score + 2.5},
            offtarget_max_score=dr.score + 3.0,
            offtarget_max_receptor="SSTR1",
            selectivity_margin=margin,
            passed=passed,
        ))
        offtarget_details.append(OffTargetDockingResult(
            seq_id=dr.seq_id,
            receptor_name="SSTR1",
            dock_score=dr.score + 3.0,
            confidence=0.70,
            engine="diffdock",
        ))
    return Step05bOutput(
        selectivity_results=selectivity_results,
        offtarget_docking_details=offtarget_details,
    )


def _make_step06_output(docking_candidates: List[DockingResult]) -> Step06Output:
    """ddG ranging -2.0 to -8.0. Top candidates get best ddG."""
    ddg_values = [-8.0, -6.5, -5.5, -2.0]
    rosetta_results = []
    for i, dr in enumerate(docking_candidates):
        ddg = ddg_values[i] if i < len(ddg_values) else -3.0
        rosetta_results.append(RosettaResult(
            seq_id=dr.seq_id,
            ddg=ddg,
            total_score=-120.0 + i * 5,
            clash_score=0,
            constraint_violations=0,
            refined_pdb=f"/tmp/{dr.seq_id}_refined.pdb",
        ))
    return Step06Output(rosetta_results=rosetta_results)


def _make_step07_output() -> Step07Output:
    return Step07Output(
        lddt_table_path="/tmp/foldmason_lddt.json",
        pymol_renders={"overview": "/tmp/overview.png", "closeup": "/tmp/closeup.png"},
        rank_table_csv="/tmp/rank_table.csv",
        summary_md="/tmp/summary.md",
    )


# ---------------------------------------------------------------------------
# E2E Test
# ---------------------------------------------------------------------------

class TestPipelineE2E(unittest.TestCase):
    """End-to-end dry-run test exercising the full PipelineOrchestrator.run() flow."""

    def _build_orchestrator(self, tmp_dir: str):
        """Build a PipelineOrchestrator with a minimal in-memory config."""
        from AG_src.pipeline.orchestrator import PipelineOrchestrator

        orch = object.__new__(PipelineOrchestrator)
        orch._logger = logging.getLogger("test_e2e.orchestrator")
        orch.output_base = Path(tmp_dir) / "runs"
        orch.output_base.mkdir(parents=True, exist_ok=True)
        orch.gate_thresholds = {
            "esmfold_plddt_min": 75.0,
            "esmfold_interface_plddt_min": 70.0,
            "docking_top_pct": 50.0,          # 50% -> 2 of 4 pass docking gate
            "rosetta_ddg_max": -5.0,
            "rosetta_clash_max": 0,
            "selectivity_margin_min": 10.0,   # G-2: 양수=좋음
            "offtarget_max_allowed": -15.0,
            "final_score_weights": {},
        }
        orch.tool_registry = {}
        orch.config = {
            "run_id": "test_run",
            "output_base_dir": str(orch.output_base),
            "iteration": {
                "max_iterations": 2,
                "n_backbone": 3,
                "k_seq_per_backbone": 2,
                "diversity_top_n": 20,
            },
            "convergence_ddg_delta": 0.5,
            "convergence_patience": 3,
            "gate_thresholds": orch.gate_thresholds,
            "receptor": {"pdb_path": "/tmp/test_sstr2.pdb", "chain": "B"},
            "off_target_receptors": [{"name": "SSTR1"}, {"name": "SSTR3"}],
            "selectivity": {"enabled": True, "engine": "diffdock"},
        }
        # Initialize agents (they will use stub fallback since no LLM)
        orch._agents = {}
        orch._last_critic_analysis = None
        orch._last_rank_table = None
        orch._last_qc_report = None
        return orch

    def _make_step_mocks(self, step04_out, step05_out, step05b_out, step06_out):
        """Return a dict of patch targets -> return values for all steps."""
        # step05_docking.run_docking receives qc_passed candidates; we need
        # a mock that returns the right Step05Output regardless of input.
        return {
            "AG_src.pipeline.step01_receptor.prepare_receptor": MagicMock(
                return_value=_make_step01_output()
            ),
            "AG_src.pipeline.step02_backbone.generate_backbones": MagicMock(
                return_value=_make_step02_output()
            ),
            "AG_src.pipeline.step03_sequence.design_sequences": MagicMock(
                return_value=_make_step03_output()
            ),
            "AG_src.pipeline.step04_qc.run_qc": MagicMock(
                return_value=step04_out
            ),
            "AG_src.pipeline.step05_docking.run_docking": MagicMock(
                return_value=step05_out
            ),
            "AG_src.pipeline.step05b_selectivity.run_selectivity_screening": MagicMock(
                return_value=step05b_out
            ),
            "AG_src.pipeline.step06_rosetta.run_rosetta_refinement": MagicMock(
                return_value=step06_out
            ),
            "AG_src.pipeline.step06_rosetta.apply_rosetta_gate": MagicMock(
                return_value=(step06_out.rosetta_results[:2], step06_out.rosetta_results[2:])
            ),
            "AG_src.pipeline.step07_analysis.run_analysis": MagicMock(
                return_value=_make_step07_output()
            ),
        }

    def test_full_pipeline_run_2_iterations(self):
        """Full pipeline runs 2 iterations, all steps execute, final report generated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            step04_out = _make_step04_output()
            qc_passed = step04_out.passed()
            step05_out = _make_step05_output(qc_passed)
            # top 50% docking -> 2 candidates
            top_docking = step05_out.top_pct(pct=50.0)
            step05b_out = _make_step05b_output(step05_out.docking_results)
            step06_out = _make_step06_output(top_docking)

            mocks = self._make_step_mocks(step04_out, step05_out, step05b_out, step06_out)

            with patch.multiple("AG_src.pipeline.step01_receptor",
                                prepare_receptor=mocks["AG_src.pipeline.step01_receptor.prepare_receptor"]), \
                 patch.multiple("AG_src.pipeline.step02_backbone",
                                generate_backbones=mocks["AG_src.pipeline.step02_backbone.generate_backbones"]), \
                 patch.multiple("AG_src.pipeline.step03_sequence",
                                design_sequences=mocks["AG_src.pipeline.step03_sequence.design_sequences"]), \
                 patch.multiple("AG_src.pipeline.step04_qc",
                                run_qc=mocks["AG_src.pipeline.step04_qc.run_qc"]), \
                 patch.multiple("AG_src.pipeline.step05_docking",
                                run_docking=mocks["AG_src.pipeline.step05_docking.run_docking"]), \
                 patch.multiple("AG_src.pipeline.step05b_selectivity",
                                run_selectivity_screening=mocks["AG_src.pipeline.step05b_selectivity.run_selectivity_screening"]), \
                 patch.multiple("AG_src.pipeline.step06_rosetta",
                                run_rosetta_refinement=mocks["AG_src.pipeline.step06_rosetta.run_rosetta_refinement"],
                                apply_rosetta_gate=mocks["AG_src.pipeline.step06_rosetta.apply_rosetta_gate"]), \
                 patch.multiple("AG_src.pipeline.step07_analysis",
                                run_analysis=mocks["AG_src.pipeline.step07_analysis.run_analysis"]):

                orch = self._build_orchestrator(tmp_dir)
                # Also patch _setup_run to avoid real directory creation outside tmp_dir
                from AG_src.pipeline.orchestrator import PipelineOrchestrator
                original_setup = PipelineOrchestrator._setup_run

                def _mock_setup_run(self_inner, iteration):
                    run_id = f"test_iter{iteration:02d}"
                    out_base = self_inner.output_base / run_id
                    step_dirs = [
                        "00_config", "01_receptor", "02_backbone", "03_sequence",
                        "04_qc", "05_docking", "05b_selectivity", "06_rosetta",
                        "07_viz", "08_reports",
                    ]
                    out_dirs: Dict[str, str] = {}
                    for sub in step_dirs:
                        d = out_base / sub
                        d.mkdir(parents=True, exist_ok=True)
                        out_dirs[sub] = str(d)
                    return run_id, out_dirs

                orch._setup_run = lambda iteration: _mock_setup_run(orch, iteration)

                result = orch.run(max_iterations=2)

                # --- Core assertions ---
                self.assertIsNotNone(result, "run() must return a FinalResult")
                self.assertEqual(result.total_iterations, 2,
                                 "Should have run exactly 2 iterations")

                # All 7 steps executed in each iteration
                for iter_rec in result.iteration_records:
                    step_keys = set(iter_rec.step_results.keys())
                    for step in ["step01", "step02", "step03", "step04", "step05", "step06", "step07"]:
                        self.assertIn(step, step_keys,
                                      f"Iteration {iter_rec.iteration} missing {step}")
                    # All steps succeeded
                    for step_name, step_res in iter_rec.step_results.items():
                        self.assertTrue(step_res.success,
                                        f"{step_name} reported failure in iteration {iter_rec.iteration}")

                # Final report file must exist
                self.assertTrue(
                    result.final_report_path,
                    "final_report_path should be non-empty",
                )
                self.assertTrue(
                    Path(result.final_report_path).exists(),
                    f"Final report file not found: {result.final_report_path}",
                )

                # Best candidates collected (from step06 rosetta_results)
                self.assertGreater(len(result.best_candidates), 0,
                                   "Should have at least 1 best candidate")
                # Best candidate must have a ddG field
                top = result.best_candidates[0]
                self.assertIn("ddg", top)
                self.assertLess(top["ddg"], 0, "Best ddG should be negative (binding)")

    def test_qc_gate_filters_correctly(self):
        """Step04 QC gate: 4 pass, 2 fail."""
        step04_out = _make_step04_output()
        passed = step04_out.passed()
        failed = [r for r in step04_out.qc_results if not r.passed_gate]

        self.assertEqual(len(passed), 4, "Exactly 4 candidates should pass QC")
        self.assertEqual(len(failed), 2, "Exactly 2 candidates should fail QC")
        for r in passed:
            self.assertGreaterEqual(r.plddt_mean, 75.0,
                                    f"{r.seq_id} pLDDT {r.plddt_mean} should be >= 75")
        for r in failed:
            self.assertLess(r.plddt_mean, 75.0,
                            f"{r.seq_id} pLDDT {r.plddt_mean} should be < 75")

    def test_docking_top_pct_filter(self):
        """Step05 top_pct: top 50% of 4 candidates = 2 survivors."""
        step04_out = _make_step04_output()
        step05_out = _make_step05_output(step04_out.passed())
        top = step05_out.top_pct(pct=50.0)
        self.assertEqual(len(top), 2, "top_pct(50) of 4 results should yield 2")
        # Top candidate should have best (lowest) score
        self.assertLessEqual(top[0].score, top[1].score)

    def test_step05b_selectivity_gate(self):
        """Step05b: 3 of 4 candidates pass selectivity."""
        step04_out = _make_step04_output()
        step05_out = _make_step05_output(step04_out.passed())
        step05b_out = _make_step05b_output(step05_out.docking_results)
        passed = step05b_out.passed_candidates()
        self.assertEqual(len(passed), 3, "3 of 4 should pass selectivity gate")
        for r in passed:
            self.assertTrue(r.passed)
            self.assertGreaterEqual(r.selectivity_margin, 10.0,
                                    "Passed candidates should have margin >= 10.0 (G-2: 양수=좋음)")

    def test_step06_rosetta_gate(self):
        """Rosetta gate: only candidates with ddG <= -5.0 and 0 clashes pass."""
        from AG_src.pipeline import step06_rosetta
        step04_out = _make_step04_output()
        step05_out = _make_step05_output(step04_out.passed())
        top_docking = step05_out.top_pct(pct=50.0)
        step06_out = _make_step06_output(top_docking)

        # ddg values are [-8.0, -6.5] for 2 candidates, both pass -5.0 threshold
        passed, failed = step06_rosetta.apply_rosetta_gate(
            step06_out.rosetta_results,
            ddg_threshold=-5.0,
            clash_max=0,
        )
        for r in passed:
            self.assertLessEqual(r.ddg, -5.0,
                                 f"{r.seq_id} ddG {r.ddg} should pass threshold -5.0")
            self.assertEqual(r.clash_score, 0)

    def test_convergence_detection(self):
        """_check_convergence returns True when ddG improvements are below delta."""
        from AG_src.pipeline.orchestrator import PipelineOrchestrator, IterationResult

        orch = object.__new__(PipelineOrchestrator)
        orch._logger = logging.getLogger("test_convergence")
        orch.config = {
            "convergence_patience": 2,
            "convergence_ddg_delta": 0.5,
        }

        # Three iterations with tiny improvement -> should converge
        records = [
            IterationResult(iteration=1, run_id="r", top_ddg=-8.0),
            IterationResult(iteration=2, run_id="r", top_ddg=-8.1),
            IterationResult(iteration=3, run_id="r", top_ddg=-8.15),
        ]
        converged = orch._check_convergence(records)
        self.assertTrue(converged, "Should detect convergence when improvement < delta")

        # Two iterations with large improvement -> should NOT converge
        records2 = [
            IterationResult(iteration=1, run_id="r", top_ddg=-5.0),
            IterationResult(iteration=2, run_id="r", top_ddg=-8.0),
            IterationResult(iteration=3, run_id="r", top_ddg=-9.5),
        ]
        not_converged = orch._check_convergence(records2)
        self.assertFalse(not_converged, "Should NOT converge when improvement > delta")

    def test_all_agents_invoked(self):
        """All 5 agents are invoked during a single iteration run."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            step04_out = _make_step04_output()
            qc_passed = step04_out.passed()
            step05_out = _make_step05_output(qc_passed)
            top_docking = step05_out.top_pct(pct=50.0)
            step05b_out = _make_step05b_output(step05_out.docking_results)
            step06_out = _make_step06_output(top_docking)

            agent_invocations: List[str] = []

            orch = self._build_orchestrator(tmp_dir)

            original_invoke = orch._invoke_agent

            def tracking_invoke(agent_name: str, context: Dict[str, Any]):
                agent_invocations.append(agent_name)
                return original_invoke(agent_name, context)

            orch._invoke_agent = tracking_invoke

            def _mock_setup_run(iteration):
                run_id = f"test_iter{iteration:02d}"
                out_base = orch.output_base / run_id
                step_dirs = [
                    "00_config", "01_receptor", "02_backbone", "03_sequence",
                    "04_qc", "05_docking", "05b_selectivity", "06_rosetta",
                    "07_viz", "08_reports",
                ]
                out_dirs: Dict[str, str] = {}
                for sub in step_dirs:
                    d = out_base / sub
                    d.mkdir(parents=True, exist_ok=True)
                    out_dirs[sub] = str(d)
                return run_id, out_dirs

            orch._setup_run = _mock_setup_run

            with patch.multiple("AG_src.pipeline.step01_receptor",
                                prepare_receptor=MagicMock(return_value=_make_step01_output())), \
                 patch.multiple("AG_src.pipeline.step02_backbone",
                                generate_backbones=MagicMock(return_value=_make_step02_output())), \
                 patch.multiple("AG_src.pipeline.step03_sequence",
                                design_sequences=MagicMock(return_value=_make_step03_output())), \
                 patch.multiple("AG_src.pipeline.step04_qc",
                                run_qc=MagicMock(return_value=step04_out)), \
                 patch.multiple("AG_src.pipeline.step05_docking",
                                run_docking=MagicMock(return_value=step05_out)), \
                 patch.multiple("AG_src.pipeline.step05b_selectivity",
                                run_selectivity_screening=MagicMock(return_value=step05b_out)), \
                 patch.multiple("AG_src.pipeline.step06_rosetta",
                                run_rosetta_refinement=MagicMock(return_value=step06_out),
                                apply_rosetta_gate=MagicMock(
                                    return_value=(step06_out.rosetta_results, [])
                                )), \
                 patch.multiple("AG_src.pipeline.step07_analysis",
                                run_analysis=MagicMock(return_value=_make_step07_output())):

                orch.run(max_iterations=1)

            expected_agents = {"planner", "qc_ranker", "diversity_manager", "critic", "reporter"}
            invoked_set = set(agent_invocations)
            missing = expected_agents - invoked_set
            self.assertFalse(
                missing,
                f"The following agents were never invoked: {missing}. "
                f"Agents invoked: {agent_invocations}",
            )

    def test_step01_failure_aborts_iteration(self):
        """If step01 fails, the iteration aborts early without running later steps."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            orch = self._build_orchestrator(tmp_dir)

            run_id = "test_abort_iter01"
            (orch.output_base / run_id / "08_reports").mkdir(parents=True, exist_ok=True)
            orch.config["run_id"] = run_id

            step07_mock = MagicMock(return_value=_make_step07_output())

            with patch.multiple("AG_src.pipeline.step01_receptor",
                                prepare_receptor=MagicMock(side_effect=RuntimeError("step01 failed"))), \
                 patch.multiple("AG_src.pipeline.step07_analysis",
                                run_analysis=step07_mock):

                iter_result = orch.run_single_iteration(iteration=1)

            # step07 should not have been called
            step07_mock.assert_not_called()
            # step01 in results should be a failure
            self.assertIn("step01", iter_result.step_results)
            self.assertFalse(iter_result.step_results["step01"].success)

    def test_final_report_content(self):
        """Final report markdown is written and contains expected sections."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            from AG_src.pipeline.orchestrator import IterationResult, PipelineOrchestrator

            orch = self._build_orchestrator(tmp_dir)
            run_id = "test_report_run"
            orch.config["run_id"] = run_id

            step04_out = _make_step04_output()
            step05_out = _make_step05_output(step04_out.passed())
            top_docking = step05_out.top_pct(pct=50.0)
            step06_out = _make_step06_output(top_docking)

            # Build synthetic iteration records
            from AG_src.pipeline.orchestrator import IterationResult, StepResult
            iter_rec = IterationResult(
                iteration=1,
                run_id=run_id,
                top_ddg=-8.0,
                n_passed_final=2,
                hypothesis="Test hypothesis",
                next_actions=["Continue optimizing"],
            )
            iter_rec.step_results["step06"] = StepResult(
                step_name="step06_rosetta",
                success=True,
                output=step06_out,
            )

            best = orch._aggregate_best_candidates([iter_rec])
            report_path = orch._write_final_report([iter_rec], best, run_id=run_id)

            self.assertTrue(Path(report_path).exists(), "Report file must exist")
            content = Path(report_path).read_text(encoding="utf-8")
            self.assertIn("SSTR2 Peptide Binder Design", content)
            self.assertIn("Top Candidates", content)
            self.assertIn("Iteration History", content)
            # Best candidate ddG should appear
            self.assertIn("-8.00", content)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
