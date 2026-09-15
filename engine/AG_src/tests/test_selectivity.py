"""
Tests for multi-receptor selectivity screening feature.
Verifies Step05b selectivity logic, config, schemas, and agent integration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
import warnings
import yaml


class TestSelectivityConfig(unittest.TestCase):
    """Test selectivity configuration in pipeline_config.yaml and gate_thresholds.yaml."""

    @classmethod
    def setUpClass(cls):
        config_dir = Path(__file__).resolve().parent.parent / "config"
        with open(config_dir / "pipeline_config.yaml") as f:
            cls.pipeline_config = yaml.safe_load(f)
        with open(config_dir / "gate_thresholds.yaml") as f:
            cls.gate_config = yaml.safe_load(f)

    def test_off_target_receptors_defined(self):
        """pipeline_config must define off_target_receptors."""
        self.assertIn("off_target_receptors", self.pipeline_config)
        receptors = self.pipeline_config["off_target_receptors"]
        self.assertIsInstance(receptors, list)
        self.assertGreaterEqual(len(receptors), 4)  # SSTR1, 3, 4, 5

    def test_off_target_receptor_names(self):
        """Off-target receptors must include SSTR1, SSTR3, SSTR4, SSTR5."""
        names = {r["name"] for r in self.pipeline_config["off_target_receptors"]}
        for expected in ["SSTR1", "SSTR3", "SSTR4", "SSTR5"]:
            self.assertIn(expected, names, f"Missing off-target: {expected}")

    def test_selectivity_params_defined(self):
        """pipeline_config must have selectivity section."""
        self.assertIn("selectivity", self.pipeline_config)
        sel = self.pipeline_config["selectivity"]
        self.assertIn("enabled", sel)
        self.assertIn("top_k_for_selectivity", sel)
        self.assertTrue(sel["enabled"])

    def test_selectivity_gate_thresholds(self):
        """gate_thresholds must have selectivity-specific gates."""
        self.assertIn("selectivity_margin_min", self.gate_config)
        self.assertIn("offtarget_max_allowed", self.gate_config)
        self.assertGreater(self.gate_config["selectivity_margin_min"], 0)  # G-2: 양수=좋음

    def test_selectivity_in_final_weights(self):
        """Final score weights must include selectivity."""
        weights = self.gate_config["final_score_weights"]
        self.assertIn("selectivity", weights)
        self.assertGreater(weights["selectivity"]["weight"], 0)

    def test_weights_still_sum_to_one(self):
        """Final score weights must sum to 1.0 after adding selectivity."""
        weights = self.gate_config["final_score_weights"]
        total = sum(w["weight"] for w in weights.values())
        self.assertAlmostEqual(total, 1.0, places=2)


class TestSelectivitySchemas(unittest.TestCase):
    """Test selectivity-related schema classes."""

    def test_selectivity_result_import(self):
        from AG_src.schemas.io_schemas import SelectivityResult
        self.assertTrue(callable(SelectivityResult))

    def test_offtarget_docking_result_import(self):
        from AG_src.schemas.io_schemas import OffTargetDockingResult
        self.assertTrue(callable(OffTargetDockingResult))

    def test_step05b_output_import(self):
        from AG_src.schemas.io_schemas import Step05bOutput
        self.assertTrue(callable(Step05bOutput))

    def test_selectivity_result_creation(self):
        from AG_src.schemas.io_schemas import SelectivityResult
        r = SelectivityResult(
            seq_id="bb00_seq01",
            sstr2_dock_score=-8.5,
            offtarget_scores={"SSTR1": -3.0, "SSTR3": -2.5, "SSTR4": -1.0, "SSTR5": -2.0},
            offtarget_max_score=-3.0,
            offtarget_max_receptor="SSTR1",
            selectivity_margin=-5.5,
            passed=True,
        )
        self.assertEqual(r.seq_id, "bb00_seq01")
        self.assertAlmostEqual(r.selectivity_margin, -5.5)
        self.assertTrue(r.passed)
        d = r.to_dict()
        self.assertIn("selectivity_margin", d)

    def test_step05b_output_passed_candidates(self):
        from AG_src.schemas.io_schemas import SelectivityResult, Step05bOutput
        passed_r = SelectivityResult("s1", -8.0, {"SSTR1": -2.0}, -2.0, "SSTR1", -6.0, True)
        failed_r = SelectivityResult("s2", -4.0, {"SSTR1": -3.5}, -3.5, "SSTR1", -0.5, False)
        out = Step05bOutput(selectivity_results=[passed_r, failed_r], offtarget_docking_details=[])
        self.assertEqual(len(out.passed_candidates()), 1)
        self.assertEqual(out.passed_candidates()[0].seq_id, "s1")

    def test_rank_table_row_has_selectivity_fields(self):
        from AG_src.schemas.io_schemas import RankTableRow
        import inspect
        sig = inspect.signature(RankTableRow)
        params = list(sig.parameters.keys())
        self.assertIn("selectivity_margin", params)
        self.assertIn("offtarget_max_score", params)
        self.assertIn("pass_selectivity", params)


class TestSelectivityLogic(unittest.TestCase):
    """Test the selectivity computation functions."""

    def test_compute_selectivity_margin_good(self):
        """Strong SSTR2 binding, weak off-target binding = PASS. (G-2: 양수=좋음)"""
        from AG_src.pipeline.step05b_selectivity import compute_selectivity_margin
        result = compute_selectivity_margin(
            seq_id="test01",
            sstr2_score=-20.0,
            offtarget_scores={"SSTR1": -5.0, "SSTR3": -4.0, "SSTR4": -3.0, "SSTR5": -4.5},
            margin_min=10.0,
            offtarget_max_allowed=-15.0,
        )
        self.assertEqual(result.seq_id, "test01")
        self.assertAlmostEqual(result.selectivity_margin, 15.0)  # -5 - (-20) = 15.0
        self.assertEqual(result.offtarget_max_receptor, "SSTR1")
        self.assertTrue(result.passed)

    def test_compute_selectivity_margin_bad(self):
        """SSTR2 binding similar to off-target = FAIL. (G-2: margin 낮음 → FAIL)"""
        from AG_src.pipeline.step05b_selectivity import compute_selectivity_margin
        result = compute_selectivity_margin(
            seq_id="test02",
            sstr2_score=-5.0,
            offtarget_scores={"SSTR1": -4.5, "SSTR3": -3.0},
            margin_min=10.0,
            offtarget_max_allowed=-15.0,
        )
        self.assertAlmostEqual(result.selectivity_margin, 0.5)  # -4.5 - (-5) = 0.5
        self.assertFalse(result.passed)  # margin 0.5 < 10.0

    def test_compute_selectivity_margin_offtarget_too_strong(self):
        """Off-target binds too strongly even with good margin = FAIL. (G-2 컨벤션)"""
        from AG_src.pipeline.step05b_selectivity import compute_selectivity_margin
        result = compute_selectivity_margin(
            seq_id="test03",
            sstr2_score=-30.0,
            offtarget_scores={"SSTR1": -16.0},
            margin_min=10.0,
            offtarget_max_allowed=-15.0,
        )
        # margin = -16 - (-30) = 14.0 (>= 10, good), but offtarget -16 < -15 (too strong → FAIL)
        self.assertAlmostEqual(result.selectivity_margin, 14.0)
        self.assertFalse(result.passed)

    def test_compute_selectivity_margin_empty_offtargets(self):
        """No off-targets = auto-pass."""
        from AG_src.pipeline.step05b_selectivity import compute_selectivity_margin
        result = compute_selectivity_margin("test04", -8.0, {})
        self.assertTrue(result.passed)

    def test_apply_selectivity_gate(self):
        from AG_src.pipeline.step05b_selectivity import apply_selectivity_gate
        from AG_src.schemas.io_schemas import SelectivityResult
        results = [
            SelectivityResult("s1", -8.0, {}, -2.0, "SSTR1", -6.0, True),
            SelectivityResult("s2", -4.0, {}, -3.5, "SSTR1", -0.5, False),
            SelectivityResult("s3", -9.0, {}, -1.0, "SSTR3", -8.0, True),
        ]
        passed, failed = apply_selectivity_gate(results)
        self.assertEqual(len(passed), 2)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].seq_id, "s2")


class TestCriticSelectivity(unittest.TestCase):
    """Test that Critic agent handles selectivity failures."""

    def test_poor_selectivity_failure_type_exists(self):
        from AG_src.agents.critic import FailureType
        self.assertTrue(hasattr(FailureType, "POOR_SELECTIVITY"))
        self.assertEqual(FailureType.POOR_SELECTIVITY, "poor_selectivity")

    def test_poor_selectivity_in_failure_action_map(self):
        from AG_src.agents.critic import FAILURE_ACTION_MAP, FailureType
        self.assertIn(FailureType.POOR_SELECTIVITY, FAILURE_ACTION_MAP)
        entry = FAILURE_ACTION_MAP[FailureType.POOR_SELECTIVITY]
        self.assertIn("actions", entry)
        self.assertGreater(len(entry["actions"]), 0)


class TestDesignAlignmentSelectivity(unittest.TestCase):
    """Verify that multi-receptor selectivity gap is now RESOLVED."""

    def test_multi_receptor_selectivity_implemented(self):
        """Multi-receptor selectivity is no longer a gap - it's implemented."""
        from AG_src.pipeline import step05b_selectivity
        self.assertTrue(hasattr(step05b_selectivity, 'run_selectivity_screening'))
        self.assertTrue(hasattr(step05b_selectivity, 'compute_selectivity_margin'))
        self.assertTrue(hasattr(step05b_selectivity, 'apply_selectivity_gate'))


class TestQCRankerGate4G2(unittest.TestCase):
    """G-2 컨벤션 하에서 QCRankerAgent Gate4 방향 검증.

    K-1: margin < threshold → FAIL, margin >= threshold → PASS
    K-2: 높은 margin 후보의 final_score > 낮은 margin 후보 (invert=False)
    """

    def _make_agent(self):
        from AG_src.agents.qc_ranker import QCRankerAgent
        return QCRankerAgent(llm_provider=None)

    def _make_candidate(self, cid, margin):
        from AG_src.agents.qc_ranker import Candidate
        return Candidate(
            candidate_id=cid, backbone_id=0, seq_id=0,
            plddt_mean=80.0, plddt_interface=75.0,
            dock_score=-8.0, ddg=-10.0, clash_count=0,
            constraint_violations=0, lddt=0.7,
            selectivity_margin=margin, offtarget_max_score=-5.0,
        )

    def _thresholds(self):
        return {
            "gates_enabled": {
                "plddt": False, "docking": False,
                "rosetta": False, "selectivity": True,
            },
            "selectivity_margin_min": 10.0,
            "offtarget_max_allowed": -15.0,
        }

    def test_k1_high_margin_passes_gate4(self):
        """K-1: margin=+15 >= 10.0 → Gate4 PASS (G-2 컨벤션)."""
        agent = self._make_agent()
        c = self._make_candidate("good", 15.0)
        passed, failed = agent.apply_gates([c], self._thresholds())
        self.assertIn(c, passed, f"margin=+15 는 Gate4 통과해야 함. fail_reasons={c.fail_reasons}")
        self.assertEqual(len(failed), 0)

    def test_k1_low_margin_fails_gate4(self):
        """K-1: margin=+5 < 10.0 → Gate4 FAIL (G-2 컨벤션)."""
        agent = self._make_agent()
        c = self._make_candidate("bad", 5.0)
        passed, failed = agent.apply_gates([c], self._thresholds())
        self.assertIn(c, failed, f"margin=+5 는 Gate4 탈락해야 함")
        self.assertTrue(any("selectivity_margin" in r for r in c.fail_reasons))

    def test_k2_higher_margin_gets_higher_final_score(self):
        """K-2: margin=+15 후보가 margin=+5 후보보다 높은 final_score를 받아야 한다 (invert=False)."""
        from AG_src.agents.qc_ranker import QCRankerAgent
        agent = QCRankerAgent(llm_provider=None)
        c_good = self._make_candidate("good", 15.0)
        c_bad = self._make_candidate("bad", 5.0)
        # selectivity 가중치만 비교하기 위해 나머지 값은 동일하게 설정
        rank_table = agent.compute_rankings([c_good, c_bad])
        self.assertGreater(
            c_good.final_score, c_bad.final_score,
            f"margin=+15 후보({c_good.final_score:.4f})가 margin=+5 후보({c_bad.final_score:.4f})보다 final_score 높아야 함"
        )
        self.assertEqual(rank_table.ranked_candidates[0].candidate_id, "good")


if __name__ == '__main__':
    unittest.main()
