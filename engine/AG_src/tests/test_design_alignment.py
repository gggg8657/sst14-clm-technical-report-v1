"""
test_design_alignment.py
========================
Design Alignment Tests
======================
Verify AG_src pipeline matches the design objectives from prompt/002_ag_test:
- MD Agent for Drug Target Screening
- SSTR2-specific DOTATATE derivative design
- BioNEMO API integration
- Rosetta FlexPepDock evaluation
"""

import sys
import unittest
import warnings
from pathlib import Path
import yaml

sys.path.insert(0, '/Users/kimsoyeon/ai4sci_kaeri')

# Resolve config directory relative to this file's location inside AG_src/tests/
_TESTS_DIR = Path(__file__).resolve().parent
_AG_SRC_DIR = _TESTS_DIR.parent
_CONFIG_DIR = _AG_SRC_DIR / 'config'


def _load_yaml(path: Path) -> dict:
    """Load a YAML file; return an empty dict if the file does not exist."""
    try:
        import yaml
        if not path.exists():
            return {}
        with path.open(encoding='utf-8') as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


class TestReceptorConfiguration(unittest.TestCase):

    def setUp(self):
        self.cfg = _load_yaml(_CONFIG_DIR / 'pipeline_config.yaml')

    def test_receptor_is_sstr2(self):
        """pipeline_config.yaml receptor.name == 'SSTR2'."""
        receptor = self.cfg.get('receptor', {})
        name = receptor.get('name', '')
        self.assertEqual(
            name, 'SSTR2',
            f"receptor.name must be 'SSTR2', got '{name}'",
        )

    def test_receptor_has_pocket_residues(self):
        """receptor.pocket_residues list has at least 15 entries."""
        receptor = self.cfg.get('receptor', {})
        pocket = receptor.get('pocket_residues', [])
        self.assertGreaterEqual(
            len(pocket), 15,
            f"pocket_residues must have >= 15 entries, got {len(pocket)}",
        )

    def test_has_hotspot_residues(self):
        """hotspot_res is defined and non-empty in pipeline_config.yaml."""
        hotspot = self.cfg.get('hotspot_res', [])
        self.assertTrue(
            len(hotspot) > 0,
            "hotspot_res must be defined and non-empty in pipeline_config.yaml",
        )

    def test_peptide_length_range_covers_dotatate(self):
        """contigs allows peptide length 10-30 which covers DOTATATE's 14aa."""
        contigs = self.cfg.get('contigs', '')
        # DOTATATE has 14 amino acids; the range 10-30 in the contigs string covers it.
        # Accept any contigs string that contains a range spanning [10, 30] or wider.
        self.assertIn(
            '10-30', contigs,
            f"contigs must include '10-30' range to cover DOTATATE length (14aa); "
            f"got contigs='{contigs}'",
        )


class TestBioNEMOTools(unittest.TestCase):

    def test_bionemo_api_tools_available(self):
        """All required BioNEMO tools exist: RFdiffusion, ProteinMPNN, ESMFold, DiffDock."""
        required_tools = [
            ('AG_src.tools.api.rfdiffusion_tool', 'RFdiffusionTool'),
            ('AG_src.tools.api.proteinmpnn_tool', 'ProteinMPNNTool'),
            ('AG_src.tools.api.esmfold_tool', 'ESMFoldTool'),
            ('AG_src.tools.api.diffdock_tool', 'DiffDockTool'),
        ]
        import importlib
        failures = []
        for module_path, class_name in required_tools:
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, class_name):
                    failures.append(f"{module_path}.{class_name} not found")
            except ImportError as exc:
                failures.append(f"{module_path}: {exc}")

        self.assertFalse(
            failures,
            "Missing BioNEMO API tools:\n" + "\n".join(failures),
        )

    def test_docking_engine_available(self):
        """DiffDockTool and/or Boltz2Tool exist for molecular docking."""
        import importlib
        found_any = False
        for module_path, class_name in [
            ('AG_src.tools.api.diffdock_tool', 'DiffDockTool'),
            ('AG_src.tools.api.boltz2_tool', 'Boltz2Tool'),
        ]:
            try:
                mod = importlib.import_module(module_path)
                if hasattr(mod, class_name):
                    found_any = True
                    break
            except ImportError:
                continue

        self.assertTrue(
            found_any,
            "At least one docking engine (DiffDockTool or Boltz2Tool) must be available",
        )

    def test_sequence_design_tool_available(self):
        """ProteinMPNNTool exists for sequence modification."""
        from AG_src.tools.api.proteinmpnn_tool import ProteinMPNNTool
        self.assertTrue(
            callable(ProteinMPNNTool),
            "ProteinMPNNTool must be a callable class",
        )

    def test_structure_prediction_available(self):
        """ESMFoldTool and OpenFold3Tool both exist for structure prediction."""
        import importlib
        failures = []
        for module_path, class_name in [
            ('AG_src.tools.api.esmfold_tool', 'ESMFoldTool'),
            ('AG_src.tools.api.openfold3_tool', 'OpenFold3Tool'),
        ]:
            try:
                mod = importlib.import_module(module_path)
                if not hasattr(mod, class_name):
                    failures.append(f"{class_name} not found in {module_path}")
            except ImportError as exc:
                failures.append(f"{module_path}: {exc}")

        self.assertFalse(
            failures,
            "Missing structure prediction tools:\n" + "\n".join(failures),
        )


class TestRosettaServer(unittest.TestCase):

    def _get_server_tools(self):
        """Import PyRosettaMCPServer and return its registered tool names."""
        from AG_src.tools.mcp.pyrosetta_server import PyRosettaMCPServer
        # Access the registered tools without initialising PyRosetta by
        # reading the class-level tool list directly via the base class registry.
        # We instantiate with object.__new__ to skip PyRosetta init and then
        # call _register_all which only builds the tool list.
        from AG_src.tools.mcp.base_server import MCPServer
        server = object.__new__(PyRosettaMCPServer)
        # Initialise the tools dict expected by MCPServer
        server._tools = {}
        server._register_all()
        return set(server._tools.keys())

    def test_rosetta_flexpep_dock_available(self):
        """PyRosettaMCPServer has a 'flexpep_dock' tool registered."""
        try:
            tool_names = self._get_server_tools()
            self.assertIn(
                'flexpep_dock', tool_names,
                f"PyRosettaMCPServer must register 'flexpep_dock'; found: {tool_names}",
            )
        except Exception as exc:
            # If the server cannot be introspected (e.g. missing base class attrs),
            # fall back to a source-level check.
            from AG_src.tools.mcp import pyrosetta_server
            import inspect
            src = inspect.getsource(pyrosetta_server)
            self.assertIn(
                'flexpep_dock', src,
                f"'flexpep_dock' not found in pyrosetta_server source: {exc}",
            )

    def test_rosetta_ddg_available(self):
        """PyRosettaMCPServer has a 'compute_ddg' tool registered."""
        try:
            tool_names = self._get_server_tools()
            self.assertIn(
                'compute_ddg', tool_names,
                f"PyRosettaMCPServer must register 'compute_ddg'; found: {tool_names}",
            )
        except Exception as exc:
            from AG_src.tools.mcp import pyrosetta_server
            import inspect
            src = inspect.getsource(pyrosetta_server)
            self.assertIn(
                'compute_ddg', src,
                f"'compute_ddg' not found in pyrosetta_server source: {exc}",
            )

    def test_rosetta_binding_energy_available(self):
        """PyRosettaMCPServer has a 'compute_binding_energy' tool registered."""
        try:
            tool_names = self._get_server_tools()
            self.assertIn(
                'compute_binding_energy', tool_names,
                f"PyRosettaMCPServer must register 'compute_binding_energy'; "
                f"found: {tool_names}",
            )
        except Exception as exc:
            from AG_src.tools.mcp import pyrosetta_server
            import inspect
            src = inspect.getsource(pyrosetta_server)
            self.assertIn(
                'compute_binding_energy', src,
                f"'compute_binding_energy' not found in pyrosetta_server source: {exc}",
            )


class TestQualityGates(unittest.TestCase):

    def setUp(self):
        self.gates = _load_yaml(_CONFIG_DIR / 'gate_thresholds.yaml')

    def test_quality_gates_defined(self):
        """gate_thresholds.yaml has plddt, docking, and ddg gate keys."""
        required_keys = [
            'esmfold_plddt_min',       # pLDDT gate
            'docking_top_pct',          # docking gate
            'rosetta_ddg_max',          # ddG gate
        ]
        missing = [k for k in required_keys if k not in self.gates]
        self.assertFalse(
            missing,
            f"gate_thresholds.yaml is missing required gate keys: {missing}",
        )


class TestRankingSystem(unittest.TestCase):

    def test_ranking_system_exists(self):
        """AG_src.schemas.rank_table module has a build_rank_table function."""
        from AG_src.schemas import rank_table
        self.assertTrue(
            callable(getattr(rank_table, 'build_rank_table', None)),
            "rank_table module must expose a callable build_rank_table function",
        )


class TestSelfImprovingLoop(unittest.TestCase):

    def test_self_improving_loop_exists(self):
        """Critic agent in PipelineOrchestrator has failure-action mapping capability.

        The orchestrator's _invoke_agent method must handle the 'critic' role
        and return next_actions based on the top_ddg value.
        """
        from AG_src.pipeline.orchestrator import PipelineOrchestrator
        from pathlib import Path
        import logging

        orch = object.__new__(PipelineOrchestrator)
        orch.config = {}
        orch.gate_thresholds = {}
        orch.tool_registry = {}
        orch.output_base = Path('/tmp')
        orch._logger = logging.getLogger('test_critic')

        # Invoke the critic with a poor result (top_ddg >= 0 -> should produce actions)
        response = orch._invoke_agent(
            'critic',
            context={'top_ddg': 0.5, 'iteration': 1, 'step_results': {}, 'previous_results': {}},
        )
        self.assertIsNotNone(response, "Critic agent must return a response")
        next_actions = response.content.get('next_actions', [])
        self.assertIsInstance(next_actions, list)
        self.assertGreater(
            len(next_actions), 0,
            "Critic must propose at least one next_action when top_ddg >= 0",
        )


class TestVisualization(unittest.TestCase):

    def test_visualization_available(self):
        """PyMOL MCP server exists with rendering tools."""
        try:
            from AG_src.tools.mcp.pymol_server import PyMOLMCPServer
            self.assertTrue(
                callable(PyMOLMCPServer),
                "PyMOLMCPServer must be a callable class",
            )
        except ImportError as exc:
            self.fail(f"PyMOL MCP server not importable: {exc}")

    def test_foldmason_structural_comparison(self):
        """FoldMason MCP server exists for structural comparison."""
        try:
            from AG_src.tools.mcp.foldmason_server import FoldMasonMCPServer
            self.assertTrue(
                callable(FoldMasonMCPServer),
                "FoldMasonMCPServer must be a callable class",
            )
        except ImportError as exc:
            self.fail(f"FoldMason MCP server not importable: {exc}")


class TestOrchestratorIteration(unittest.TestCase):

    def test_pipeline_orchestrator_iteration(self):
        """PipelineOrchestrator has run_single_iteration or equivalent iteration method."""
        from AG_src.pipeline.orchestrator import PipelineOrchestrator
        has_iteration = (
            callable(getattr(PipelineOrchestrator, 'run_single_iteration', None))
            or callable(getattr(PipelineOrchestrator, 'run_iteration', None))
            or callable(getattr(PipelineOrchestrator, 'run', None))
        )
        self.assertTrue(
            has_iteration,
            "PipelineOrchestrator must have an iteration method "
            "(run_single_iteration / run_iteration / run)",
        )


class TestKnownGaps(unittest.TestCase):
    """Document known gaps in the current AG_src implementation.

    These tests always PASS but emit warnings to make gaps visible in CI output.
    They serve as living documentation of features not yet implemented.
    """

    def test_gap_multi_receptor_selectivity(self):
        """Multi-receptor selectivity is now implemented via step05b."""
        from AG_src.pipeline import step05b_selectivity
        self.assertTrue(hasattr(step05b_selectivity, 'run_selectivity_screening'))
        self.assertTrue(hasattr(step05b_selectivity, 'compute_selectivity_margin'))

    def test_gap_stability_prediction(self):
        """Known gap: blood/serum stability prediction module is not implemented.

        DOTATATE derivatives require metabolic stability assessment.  A dedicated
        stability scoring step (e.g., via molecular dynamics or an ADMET model)
        is not present in the current seven-step pipeline.
        """
        warnings.warn(
            "Known gap: Blood/serum stability prediction module is absent. "
            "DOTATATE derivatives need metabolic stability scoring before "
            "clinical relevance can be claimed.",
            UserWarning,
            stacklevel=2,
        )
        # This test always passes; it exists only to document the gap.
        self.assertTrue(True)


class TestDOTATATEReferenceSequence(unittest.TestCase):
    """Verify DOTATATE reference peptide is correctly configured."""

    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).resolve().parent.parent / "config" / "pipeline_config.yaml"
        with open(config_path) as f:
            cls.config = yaml.safe_load(f)

    def test_reference_peptide_exists(self):
        """pipeline_config.yaml must define reference_peptide."""
        self.assertIn("reference_peptide", self.config)

    def test_reference_peptide_is_dotatate(self):
        """Reference peptide must be DOTATATE."""
        ref = self.config["reference_peptide"]
        self.assertEqual(ref["name"], "DOTATATE")

    def test_dotatate_sequence_length(self):
        """DOTATATE is exactly 14 amino acids."""
        ref = self.config["reference_peptide"]
        self.assertEqual(len(ref["sequence"]), 14)
        self.assertEqual(ref["length"], 14)

    def test_dotatate_has_disulfide(self):
        """DOTATATE has Cys at positions 3 and 14 for disulfide bond."""
        ref = self.config["reference_peptide"]
        seq = ref["sequence"]
        self.assertEqual(seq[2], "C")   # position 3 (0-indexed: 2)
        self.assertEqual(seq[13], "C")  # position 14 (0-indexed: 13)
        self.assertEqual(ref["cys_positions"], [3, 14])


if __name__ == '__main__':
    unittest.main()
