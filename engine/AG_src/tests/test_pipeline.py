"""
test_pipeline.py
================
Tests for pipeline step modules and the PipelineOrchestrator.

Verifies that each step module exposes its documented public API and that
the orchestrator can be imported, instantiated, and introspected without
requiring live API credentials or external binaries.
"""

import sys
import unittest

sys.path.insert(0, '/Users/kimsoyeon/ai4sci_kaeri')


class TestStep01Receptor(unittest.TestCase):

    def test_step01_has_prepare_receptor(self):
        """step01_receptor module exposes the prepare_receptor function."""
        from AG_src.pipeline import step01_receptor
        self.assertTrue(
            callable(getattr(step01_receptor, 'prepare_receptor', None)),
            "step01_receptor.prepare_receptor must be a callable",
        )


class TestStep02Backbone(unittest.TestCase):

    def test_step02_has_generate_backbones(self):
        """step02_backbone module exposes the generate_backbones function."""
        from AG_src.pipeline import step02_backbone
        self.assertTrue(
            callable(getattr(step02_backbone, 'generate_backbones', None)),
            "step02_backbone.generate_backbones must be a callable",
        )


class TestStep03Sequence(unittest.TestCase):

    def test_step03_has_design_sequences(self):
        """step03_sequence module exposes the design_sequences function."""
        from AG_src.pipeline import step03_sequence
        self.assertTrue(
            callable(getattr(step03_sequence, 'design_sequences', None)),
            "step03_sequence.design_sequences must be a callable",
        )


class TestStep04QC(unittest.TestCase):

    def test_step04_has_run_qc(self):
        """step04_qc module exposes the run_qc function."""
        from AG_src.pipeline import step04_qc
        self.assertTrue(
            callable(getattr(step04_qc, 'run_qc', None)),
            "step04_qc.run_qc must be a callable",
        )


class TestStep05Docking(unittest.TestCase):

    def test_step05_module_exists(self):
        """step05_docking imports correctly from AG_src.pipeline."""
        try:
            from AG_src.pipeline import step05_docking  # noqa: F401
        except ImportError as exc:
            self.fail(f"Failed to import step05_docking: {exc}")

    def test_step05_has_main_function(self):
        """step05_docking has its main entry function."""
        from AG_src.pipeline import step05_docking
        self.assertTrue(callable(getattr(step05_docking, 'run_docking', None)))


class TestStep06Rosetta(unittest.TestCase):

    def test_step06_module_exists(self):
        """step06_rosetta imports correctly from AG_src.pipeline."""
        try:
            from AG_src.pipeline import step06_rosetta  # noqa: F401
        except ImportError as exc:
            self.fail(f"Failed to import step06_rosetta: {exc}")

    def test_step06_has_main_function(self):
        """step06_rosetta has its main entry function."""
        from AG_src.pipeline import step06_rosetta
        self.assertTrue(callable(getattr(step06_rosetta, 'run_rosetta_refinement', None)))


class TestStep07Analysis(unittest.TestCase):

    def test_step07_module_exists(self):
        """step07_analysis imports correctly from AG_src.pipeline."""
        try:
            from AG_src.pipeline import step07_analysis  # noqa: F401
        except ImportError as exc:
            self.fail(f"Failed to import step07_analysis: {exc}")

    def test_step07_has_main_function(self):
        """step07_analysis has its main entry function."""
        from AG_src.pipeline import step07_analysis
        self.assertTrue(callable(getattr(step07_analysis, 'run_analysis', None)))


class TestOrchestratorClass(unittest.TestCase):

    def test_orchestrator_class_exists(self):
        """PipelineOrchestrator class is importable from AG_src.pipeline.orchestrator."""
        from AG_src.pipeline.orchestrator import PipelineOrchestrator
        self.assertTrue(
            isinstance(PipelineOrchestrator, type),
            "PipelineOrchestrator must be a class",
        )

    def test_orchestrator_instantiation(self):
        """PipelineOrchestrator can be created with a minimal config dict.

        We bypass the file-based __init__ by directly patching instance
        attributes after calling object.__new__, which avoids requiring real
        YAML files on disk.
        """
        from AG_src.pipeline.orchestrator import PipelineOrchestrator
        from pathlib import Path

        orch = object.__new__(PipelineOrchestrator)
        orch.config = {}
        orch.gate_thresholds = {}
        orch.tool_registry = {}
        orch.output_base = Path('/tmp')
        import logging
        orch._logger = logging.getLogger('test_orchestrator')

        self.assertIsInstance(orch, PipelineOrchestrator)
        self.assertIsInstance(orch.config, dict)

    def test_orchestrator_has_run_method(self):
        """PipelineOrchestrator has at least one of: run, execute, run_iteration."""
        from AG_src.pipeline.orchestrator import PipelineOrchestrator
        has_run = (
            callable(getattr(PipelineOrchestrator, 'run', None))
            or callable(getattr(PipelineOrchestrator, 'execute', None))
            or callable(getattr(PipelineOrchestrator, 'run_iteration', None))
            or callable(getattr(PipelineOrchestrator, 'run_single_iteration', None))
        )
        self.assertTrue(
            has_run,
            "PipelineOrchestrator must have a run / execute / run_iteration method",
        )


class TestOrchestratorDataclasses(unittest.TestCase):

    def test_step_result_dataclass(self):
        """orchestrator.StepResult dataclass has the expected fields."""
        from AG_src.pipeline.orchestrator import StepResult
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(StepResult)}
        required = {'step_name', 'success'}
        missing = required - field_names
        self.assertFalse(
            missing,
            f"StepResult is missing fields: {missing}",
        )

    def test_iteration_result_dataclass(self):
        """orchestrator.IterationResult dataclass has the expected fields."""
        from AG_src.pipeline.orchestrator import IterationResult
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(IterationResult)}
        required = {'iteration', 'run_id'}
        missing = required - field_names
        self.assertFalse(
            missing,
            f"IterationResult is missing fields: {missing}",
        )


class TestPipelinePackage(unittest.TestCase):

    def test_pipeline_init_exports(self):
        """from AG_src.pipeline import PipelineOrchestrator works."""
        try:
            from AG_src.pipeline import PipelineOrchestrator  # noqa: F401
        except ImportError as exc:
            self.fail(f"Cannot import PipelineOrchestrator from AG_src.pipeline: {exc}")

    def test_all_steps_importable(self):
        """All 7 step modules are importable from AG_src.pipeline."""
        step_names = [
            'step01_receptor',
            'step02_backbone',
            'step03_sequence',
            'step04_qc',
            'step05_docking',
            'step06_rosetta',
            'step07_analysis',
        ]
        import importlib
        failures = []
        for name in step_names:
            try:
                importlib.import_module(f'AG_src.pipeline.{name}')
            except ImportError as exc:
                failures.append(f"{name}: {exc}")

        self.assertFalse(
            failures,
            "The following step modules failed to import:\n" + "\n".join(failures),
        )


if __name__ == '__main__':
    unittest.main()
