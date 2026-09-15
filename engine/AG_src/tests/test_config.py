"""Tests for AG_src config YAML files."""
import sys
import unittest
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / 'AG_src' / 'config'


class TestPipelineConfigLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_DIR / 'pipeline_config.yaml', 'r', encoding='utf-8') as fh:
            cls.config = yaml.safe_load(fh)

    def test_pipeline_config_loads(self):
        self.assertIsInstance(self.config, dict)
        for key in ('run_id_format', 'iteration', 'receptor', 'contigs', 'hotspot_res', 'output_base_dir'):
            self.assertIn(key, self.config, f"Missing key: {key}")

    def test_pipeline_config_receptor(self):
        receptor = self.config['receptor']
        self.assertEqual(receptor['name'], 'SSTR2')
        self.assertEqual(receptor['chain'], 'B')

    def test_pipeline_config_iteration_params(self):
        iteration = self.config['iteration']
        for param in ('n_backbone', 'k_seq_per_backbone', 'top_m_rosetta', 'max_iterations'):
            self.assertIn(param, iteration, f"Missing iteration param: {param}")
            value = iteration[param]
            self.assertIsInstance(value, int, f"{param} should be an int, got {type(value)}")
            self.assertGreater(value, 0, f"{param} should be a positive integer")


class TestGateThresholdsLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_DIR / 'gate_thresholds.yaml', 'r', encoding='utf-8') as fh:
            cls.config = yaml.safe_load(fh)

    def test_gate_thresholds_loads(self):
        self.assertIsInstance(self.config, dict)
        for key in ('esmfold_plddt_min', 'docking_top_pct', 'rosetta_ddg_max'):
            self.assertIn(key, self.config, f"Missing key: {key}")

    def test_gate_thresholds_values(self):
        plddt_min = self.config['esmfold_plddt_min']
        self.assertGreaterEqual(plddt_min, 50, "plddt_min should be >= 50")

        docking_top_pct = self.config['docking_top_pct']
        self.assertGreaterEqual(docking_top_pct, 1, "docking_top_pct should be >= 1")
        self.assertLessEqual(docking_top_pct, 100, "docking_top_pct should be <= 100")

        ddg_max = self.config['rosetta_ddg_max']
        self.assertLess(ddg_max, 0, "rosetta_ddg_max should be < 0 (negative binding energy)")

    def test_gate_weights_sum(self):
        weights_section = self.config.get('final_score_weights', {})
        self.assertTrue(weights_section, "final_score_weights should not be empty")
        total = sum(v['weight'] for v in weights_section.values())
        self.assertAlmostEqual(total, 1.0, places=6, msg=f"Weights sum to {total}, expected 1.0")


class TestToolRegistryLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_DIR / 'tool_registry.yaml', 'r', encoding='utf-8') as fh:
            cls.config = yaml.safe_load(fh)

    def test_tool_registry_loads(self):
        self.assertIsInstance(self.config, dict)
        self.assertIn('tools', self.config)
        tools = self.config['tools']
        self.assertIn('api', tools, "tool_registry should have an 'api' category")
        self.assertIn('mcp', tools, "tool_registry should have an 'mcp' category")

    def test_tool_registry_has_required_tools(self):
        tools = self.config['tools']
        api_tools = tools.get('api', {})
        mcp_tools = tools.get('mcp', {})

        # Combine api and mcp tool names for lookup
        all_tool_names = set(api_tools.keys()) | set(mcp_tools.keys())

        required_tools = ['rfdiffusion', 'proteinmpnn', 'esmfold', 'diffdock', 'pyrosetta', 'foldmason', 'pymol']
        for tool in required_tools:
            self.assertIn(tool, all_tool_names, f"Required tool '{tool}' not found in tool registry")


if __name__ == '__main__':
    unittest.main()
