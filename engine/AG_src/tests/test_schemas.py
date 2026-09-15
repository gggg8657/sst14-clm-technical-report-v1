"""Tests for AG_src.schemas - I/O dataclasses, ranking, lab notebook."""
import sys
sys.path.insert(0, '/Users/kimsoyeon/ai4sci_kaeri')
import unittest
import csv
import math
import os
import tempfile

from AG_src.schemas.io_schemas import (
    Step01Output,
    Step02Output,
    Step03Output,
    Step04Output,
    Step05Output,
    Step06Output,
    Step07Output,
    SequenceEntry,
    QCResult,
    DockingResult,
    RosettaResult,
    RankTableRow,
    IterationRecord,
)
from AG_src.schemas.rank_table import (
    normalize_scores,
    compute_final_score,
    build_rank_table,
    filter_by_gates,
    export_csv,
)
from AG_src.schemas.lab_notebook import (
    generate_notebook,
    generate_decision_log,
)


class TestStepOutputsInstantiate(unittest.TestCase):
    def test_step_outputs_instantiate(self):
        s1 = Step01Output(
            receptor_pdb_path="/tmp/sstr2.pdb",
            pocket_residues=[122, 127, 184],
            chain_id="B",
            pocket_json_path="/tmp/pocket.json",
        )
        self.assertEqual(s1.chain_id, "B")
        d = s1.to_dict()
        self.assertIn("receptor_pdb_path", d)

        s2 = Step02Output(
            backbone_pdbs=["/tmp/bb00.pdb"],
            design_params={"contigs": "B1-369/0 10-30"},
            n_generated=1,
        )
        self.assertEqual(s2.n_generated, 1)
        self.assertIn("backbone_pdbs", s2.to_dict())

        entry = SequenceEntry(backbone_idx=0, seq_idx=0, sequence="ACDEFGHIKL", fasta_path="/tmp/s.fasta")
        s3 = Step03Output(sequences=[entry])
        d3 = s3.to_dict()
        self.assertIn("sequences", d3)
        self.assertEqual(len(d3["sequences"]), 1)

        qc = QCResult(seq_id="bb00_seq00", plddt_mean=80.0, plddt_interface=78.0, pdb_path="/tmp/q.pdb", passed_gate=True)
        s4 = Step04Output(qc_results=[qc])
        self.assertIn("qc_results", s4.to_dict())

        dock = DockingResult(seq_id="bb00_seq00", engine="diffdock", score=-7.5, confidence=-1.2, pose_pdb="/tmp/p.pdb", rank=1)
        s5 = Step05Output(docking_results=[dock])
        self.assertIn("docking_results", s5.to_dict())

        ros = RosettaResult(seq_id="bb00_seq00", ddg=-8.0, total_score=-300.0, clash_score=0.0, constraint_violations=0, refined_pdb="/tmp/r.pdb")
        s6 = Step06Output(rosetta_results=[ros])
        self.assertIn("rosetta_results", s6.to_dict())

        s7 = Step07Output(
            lddt_table_path="/tmp/lddt.json",
            pymol_renders={"overview": "/tmp/ov.png", "closeup": "/tmp/cu.png"},
            rank_table_csv="/tmp/rank.csv",
            summary_md="/tmp/summary.md",
        )
        d7 = s7.to_dict()
        self.assertIn("lddt_table_path", d7)
        self.assertIn("pymol_renders", d7)


class TestQCResultFields(unittest.TestCase):
    def test_qc_result_fields(self):
        qc = QCResult(
            seq_id="bb01_seq02",
            plddt_mean=82.5,
            plddt_interface=79.3,
            pdb_path="/tmp/esmfold.pdb",
            passed_gate=True,
        )
        self.assertEqual(qc.seq_id, "bb01_seq02")
        self.assertAlmostEqual(qc.plddt_mean, 82.5)
        self.assertAlmostEqual(qc.plddt_interface, 79.3)
        self.assertEqual(qc.pdb_path, "/tmp/esmfold.pdb")
        self.assertTrue(qc.passed_gate)
        d = qc.to_dict()
        self.assertEqual(d["seq_id"], "bb01_seq02")
        self.assertEqual(d["passed_gate"], True)


class TestDockingResultFields(unittest.TestCase):
    def test_docking_result_fields(self):
        dock = DockingResult(
            seq_id="bb00_seq01",
            engine="diffdock",
            score=-8.2,
            confidence=-1.5,
            pose_pdb="/tmp/pose_1.pdb",
            rank=1,
        )
        self.assertEqual(dock.seq_id, "bb00_seq01")
        self.assertEqual(dock.engine, "diffdock")
        self.assertAlmostEqual(dock.score, -8.2)
        self.assertAlmostEqual(dock.confidence, -1.5)
        self.assertEqual(dock.rank, 1)
        d = dock.to_dict()
        self.assertIn("pose_pdb", d)
        self.assertEqual(d["engine"], "diffdock")


class TestRosettaResultFields(unittest.TestCase):
    def test_rosetta_result_fields(self):
        ros = RosettaResult(
            seq_id="bb02_seq03",
            ddg=-9.1,
            total_score=-312.5,
            clash_score=0.0,
            constraint_violations=0,
            refined_pdb="/tmp/refined.pdb",
        )
        self.assertEqual(ros.seq_id, "bb02_seq03")
        self.assertAlmostEqual(ros.ddg, -9.1)
        self.assertAlmostEqual(ros.total_score, -312.5)
        self.assertEqual(ros.clash_score, 0.0)
        self.assertEqual(ros.constraint_violations, 0)
        d = ros.to_dict()
        self.assertIn("refined_pdb", d)
        self.assertEqual(d["ddg"], -9.1)


class TestRankTableRow(unittest.TestCase):
    def test_rank_table_row(self):
        row = RankTableRow(
            backbone_id=0,
            seq_id="bb00_seq00",
            sequence="ACDEFGHIKL",
            plddt_mean=85.0,
            plddt_interface=80.0,
            dock_score=-7.5,
            dock_engine="diffdock",
            ddg=-8.5,
            lddt=0.75,
            final_score=1.234,
            pass_fail="PASS",
            fail_reason="",
        )
        self.assertEqual(row.backbone_id, 0)
        self.assertEqual(row.seq_id, "bb00_seq00")
        self.assertEqual(row.pass_fail, "PASS")
        d = row.to_dict()
        self.assertIn("final_score", d)
        self.assertIn("dock_engine", d)
        self.assertEqual(d["sequence"], "ACDEFGHIKL")


class TestIterationRecord(unittest.TestCase):
    def test_iteration_record(self):
        rec = IterationRecord(
            run_id="20260217_1430_iter01",
            iteration=1,
            hypothesis="Increase backbone diversity to improve binding",
            parameter_changes={"n_backbone": {"old": 5, "new": 10}},
            results_summary={"n_passed": 3, "best_ddg": -9.1},
            next_actions=["Try longer peptides", "Adjust hotspot weights"],
        )
        self.assertEqual(rec.run_id, "20260217_1430_iter01")
        self.assertEqual(rec.iteration, 1)
        self.assertEqual(len(rec.next_actions), 2)
        d = rec.to_dict()
        self.assertIn("hypothesis", d)
        self.assertIn("parameter_changes", d)
        json_str = rec.to_json()
        self.assertIn("20260217_1430_iter01", json_str)
        rec2 = IterationRecord.from_json(json_str)
        self.assertEqual(rec2.iteration, 1)


class TestNormalizeScores(unittest.TestCase):
    def test_normalize_scores(self):
        scores = [10.0, 20.0, 30.0, 40.0, 50.0]
        normalized = normalize_scores(scores, method="min_max")
        self.assertEqual(len(normalized), 5)
        self.assertAlmostEqual(min(normalized), 0.0)
        self.assertAlmostEqual(max(normalized), 1.0)
        for v in normalized:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_normalize_scores_constant(self):
        scores = [5.0, 5.0, 5.0]
        normalized = normalize_scores(scores, method="min_max")
        for v in normalized:
            self.assertAlmostEqual(v, 0.5)

    def test_normalize_scores_empty(self):
        self.assertEqual(normalize_scores([]), [])

    def test_normalize_scores_z_score(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        normalized = normalize_scores(scores, method="z_score")
        self.assertAlmostEqual(sum(normalized) / len(normalized), 0.0, places=10)


class TestComputeFinalScore(unittest.TestCase):
    def test_compute_final_score(self):
        score = compute_final_score(
            plddt=80.0,
            dock_score=-5.0,
            ddg=-6.0,
            lddt=0.8,
            weights={"plddt": 0.2, "dock_score": 0.3, "ddg": 0.3, "lddt": 0.2},
        )
        # plddt_norm = 0.8, dock_contrib = 5.0*0.3=1.5, ddg_contrib=6.0*0.3=1.8, lddt=0.8*0.2=0.16
        expected = 0.8 * 0.2 + 5.0 * 0.3 + 6.0 * 0.3 + 0.8 * 0.2
        self.assertAlmostEqual(score, round(expected, 6), places=5)

    def test_compute_final_score_default_weights(self):
        score = compute_final_score(plddt=75.0, dock_score=-4.0, ddg=-5.0, lddt=0.7)
        self.assertIsInstance(score, float)

    def test_compute_final_score_better_with_lower_ddg(self):
        score_good = compute_final_score(plddt=80.0, dock_score=-5.0, ddg=-10.0, lddt=0.8)
        score_poor = compute_final_score(plddt=80.0, dock_score=-5.0, ddg=-1.0, lddt=0.8)
        self.assertGreater(score_good, score_poor)


class TestBuildRankTable(unittest.TestCase):
    def _make_outputs(self):
        qc1 = QCResult(seq_id="bb00_seq00", plddt_mean=82.0, plddt_interface=78.0, pdb_path="/tmp/q1.pdb", passed_gate=True)
        qc2 = QCResult(seq_id="bb00_seq01", plddt_mean=76.0, plddt_interface=72.0, pdb_path="/tmp/q2.pdb", passed_gate=True)
        step04 = Step04Output(qc_results=[qc1, qc2])

        dock1 = DockingResult(seq_id="bb00_seq00", engine="diffdock", score=-7.5, confidence=-1.2, pose_pdb="/tmp/p1.pdb", rank=1)
        dock2 = DockingResult(seq_id="bb00_seq01", engine="diffdock", score=-6.0, confidence=-0.9, pose_pdb="/tmp/p2.pdb", rank=1)
        step05 = Step05Output(docking_results=[dock1, dock2])

        ros1 = RosettaResult(seq_id="bb00_seq00", ddg=-8.0, total_score=-300.0, clash_score=0.0, constraint_violations=0, refined_pdb="/tmp/r1.pdb")
        ros2 = RosettaResult(seq_id="bb00_seq01", ddg=-5.5, total_score=-280.0, clash_score=0.0, constraint_violations=0, refined_pdb="/tmp/r2.pdb")
        step06 = Step06Output(rosetta_results=[ros1, ros2])

        step07 = Step07Output(
            lddt_table_path="/tmp/lddt.json",
            pymol_renders={},
            rank_table_csv="/tmp/rank.csv",
            summary_md="/tmp/summary.md",
        )
        return step04, step05, step06, step07

    def test_build_rank_table(self):
        step04, step05, step06, step07 = self._make_outputs()
        rows = build_rank_table(step04, step05, step06, step07)
        self.assertEqual(len(rows), 2)
        # Sorted by final_score descending
        self.assertGreaterEqual(rows[0].final_score, rows[1].final_score)
        seq_ids = {r.seq_id for r in rows}
        self.assertIn("bb00_seq00", seq_ids)
        self.assertIn("bb00_seq01", seq_ids)


class TestFilterByGates(unittest.TestCase):
    def _make_row(self, seq_id, plddt_mean, plddt_interface, ddg, lddt):
        return RankTableRow(
            backbone_id=0,
            seq_id=seq_id,
            sequence="ACDEFGHIKL",
            plddt_mean=plddt_mean,
            plddt_interface=plddt_interface,
            dock_score=-5.0,
            dock_engine="diffdock",
            ddg=ddg,
            lddt=lddt,
            final_score=1.0,
            pass_fail="PASS",
            fail_reason="",
        )

    def test_filter_by_gates(self):
        thresholds = {
            "esmfold_plddt_min": 75,
            "esmfold_interface_plddt_min": 70,
            "rosetta_ddg_max": -5.0,
            "foldmason_lddt_min": 0.6,
        }
        row_pass = self._make_row("bb00_seq00", plddt_mean=82.0, plddt_interface=78.0, ddg=-8.0, lddt=0.75)
        row_fail_plddt = self._make_row("bb00_seq01", plddt_mean=60.0, plddt_interface=78.0, ddg=-8.0, lddt=0.75)
        row_fail_ddg = self._make_row("bb00_seq02", plddt_mean=82.0, plddt_interface=78.0, ddg=-2.0, lddt=0.75)

        result = filter_by_gates([row_pass, row_fail_plddt, row_fail_ddg], thresholds)
        self.assertEqual(len(result), 3)

        id_map = {r.seq_id: r for r in result}
        self.assertEqual(id_map["bb00_seq00"].pass_fail, "PASS")
        self.assertEqual(id_map["bb00_seq00"].fail_reason, "")
        self.assertEqual(id_map["bb00_seq01"].pass_fail, "FAIL")
        self.assertIn("pLDDT_mean", id_map["bb00_seq01"].fail_reason)
        self.assertEqual(id_map["bb00_seq02"].pass_fail, "FAIL")
        self.assertIn("ddG", id_map["bb00_seq02"].fail_reason)


class TestExportCSV(unittest.TestCase):
    def test_export_csv(self):
        rows = [
            RankTableRow(
                backbone_id=0,
                seq_id="bb00_seq00",
                sequence="ACDEFGHIKL",
                plddt_mean=82.0,
                plddt_interface=78.0,
                dock_score=-7.5,
                dock_engine="diffdock",
                ddg=-8.0,
                lddt=0.75,
                final_score=3.0,
                pass_fail="PASS",
                fail_reason="",
            ),
            RankTableRow(
                backbone_id=0,
                seq_id="bb00_seq01",
                sequence="LKIHGFEDCA",
                plddt_mean=76.0,
                plddt_interface=72.0,
                dock_score=-6.0,
                dock_engine="diffdock",
                ddg=-5.5,
                lddt=0.65,
                final_score=2.5,
                pass_fail="PASS",
                fail_reason="",
            ),
        ]
        out_path = "/tmp/test_rank_table.csv"
        export_csv(rows, out_path)

        self.assertTrue(os.path.exists(out_path))
        with open(out_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames
            data_rows = list(reader)

        expected_headers = [
            "backbone_id", "seq_id", "sequence", "plddt_mean", "plddt_interface",
            "dock_score", "dock_engine", "ddg", "lddt", "final_score",
            "rank", "pass_fail", "fail_reason",
        ]
        for h in expected_headers:
            self.assertIn(h, fieldnames)

        self.assertEqual(len(data_rows), 2)
        self.assertEqual(data_rows[0]["seq_id"], "bb00_seq00")
        self.assertEqual(data_rows[0]["rank"], "1")
        self.assertEqual(data_rows[1]["rank"], "2")


class TestGenerateNotebook(unittest.TestCase):
    def test_generate_notebook(self):
        config = {
            "iteration": {
                "n_backbone": 10,
                "k_seq_per_backbone": 8,
                "top_m_rosetta": 10,
            },
            "receptor": {"name": "SSTR2", "chain": "B"},
            "contigs": "B1-369/0 10-30",
            "hotspot_res": ["B122", "B127"],
        }
        results = {
            "step04": {"n_total": 80, "n_passed": 60, "plddt_gate": 75},
            "step05": {"dock_engine": "diffdock", "dock_top_pct": 20, "n_docking_passed": 12},
            "step06": {"n_refined": 12, "ddg_gate": -5.0, "n_ddg_passed": 5},
            "step07": {"lddt_min": 0.62, "lddt_max": 0.91},
            "top_candidates": [
                {
                    "seq_id": "bb00_seq00",
                    "sequence": "ACDEFGHIKL",
                    "plddt_mean": 82.0,
                    "dock_score": -7.5,
                    "ddg": -8.0,
                    "lddt": 0.75,
                    "final_score": 3.0,
                    "pass_fail": "PASS",
                }
            ],
        }
        decisions = {
            "hypothesis": "Longer peptides improve binding to SSTR2 pocket",
            "parameter_changes": {
                "n_backbone": {"previous": 5, "new": 10, "rationale": "More diversity"}
            },
            "critic_notes": "Step04 pLDDT distribution was bimodal.",
            "next_plan": "Increase k_seq_per_backbone to 12.",
        }

        md = generate_notebook(
            run_id="20260217_1430_iter01",
            iteration=1,
            config=config,
            results=results,
            decisions=decisions,
        )

        self.assertIsInstance(md, str)
        self.assertIn("# Lab Notebook", md)
        self.assertIn("## Iteration 1", md)
        self.assertIn("### Hypothesis", md)
        self.assertIn("### Results Summary", md)
        self.assertIn("### Top Candidates", md)
        self.assertIn("### Critic Analysis", md)
        self.assertIn("### Next Iteration Plan", md)
        self.assertIn("SSTR2", md)
        self.assertIn("20260217_1430_iter01", md)


class TestGenerateDecisionLog(unittest.TestCase):
    def test_generate_decision_log(self):
        critic_analysis = {
            "failures": [
                {
                    "type": "low_pLDDT",
                    "count": 20,
                    "root_cause": "Backbone too flexible",
                    "action": "Constrain backbone design",
                },
            ]
        }
        parameter_changes = {
            "n_backbone": {"previous": 5, "new": 10, "rationale": "Increase diversity"},
        }
        hypothesis = "Stiffer backbones will improve pLDDT scores in next iteration"

        md = generate_decision_log(
            run_id="20260217_1430_iter01",
            iteration=1,
            critic_analysis=critic_analysis,
            parameter_changes=parameter_changes,
            hypothesis=hypothesis,
        )

        self.assertIsInstance(md, str)
        self.assertIn("# Decision Log", md)
        self.assertIn("### Failure Analysis", md)
        self.assertIn("### Parameter Adjustments", md)
        self.assertIn("### Hypothesis for Next Iteration", md)
        self.assertIn("low_pLDDT", md)
        self.assertIn("n_backbone", md)
        self.assertIn("Stiffer backbones", md)
        # Should show iteration -> next_iteration
        self.assertIn("1 -> 2", md)


if __name__ == '__main__':
    unittest.main()
