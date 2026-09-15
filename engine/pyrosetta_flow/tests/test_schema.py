"""Tests for schema.py: dataclass defaults, serialization, round-trip."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from pyrosetta_flow.schema import (
    CandidateResult,
    FlowArtifacts,
    FlowConfig,
    IterationSummary,
)


# ===================================================================
# FlowConfig tests  (Medium priority #9)
# ===================================================================

class TestFlowConfig:

    def test_defaults(self):
        cfg = FlowConfig(template_pdb="/tmp/test.pdb")
        assert cfg.original_sequence == "AGCKNFFWKTFTSC"
        assert cfg.n_candidates == 8
        assert cfg.max_iterations == 2
        assert cfg.objective_mode == "auto"
        assert cfg.max_parallel_workers == 160  # 2026-06-25: nstruct 평탄화 병렬화로 상향 (구: 32)
        assert cfg.llm_model_override is None

    def test_custom_values(self):
        cfg = FlowConfig(
            template_pdb="/tmp/test.pdb",
            n_candidates=16,
            max_iterations=5,
            objective_mode="ddg_only",
        )
        assert cfg.n_candidates == 16
        assert cfg.max_iterations == 5

    def test_design_positions_default(self):
        cfg = FlowConfig(template_pdb="/tmp/test.pdb")
        # 2026-06: Cys3·Cys14(이황화 파트너) 모두 제외 — 변이 시 SS bond 파괴 방지
        assert 3 not in cfg.design_positions   # Cys3 excluded
        assert 14 not in cfg.design_positions  # Cys14 excluded (이황화 보존)

    def test_asdict_roundtrip(self):
        cfg = FlowConfig(template_pdb="/tmp/test.pdb")
        d = asdict(cfg)
        assert d["template_pdb"] == "/tmp/test.pdb"
        assert isinstance(d["design_positions"], list)


# ===================================================================
# CandidateResult tests  (Medium priority #9)
# ===================================================================

class TestCandidateResult:

    def test_defaults(self):
        c = CandidateResult(
            iteration=1, candidate_id="c1", sequence="AAA",
            ddg=-5.0, total_score=-100.0, clash_score=1.0,
        )
        assert c.source == "mutate_then_dock"
        assert c.selected is False
        assert c.fail_reason == ""

    def test_custom_values(self):
        c = CandidateResult(
            iteration=2, candidate_id="c2", sequence="BBB",
            ddg=-10.0, total_score=-200.0, clash_score=0.5,
            selected=True, fail_reason="test",
        )
        assert c.selected is True
        assert c.fail_reason == "test"

    def test_asdict(self):
        c = CandidateResult(
            iteration=1, candidate_id="c1", sequence="AAA",
            ddg=-5.0, total_score=-100.0, clash_score=1.0,
        )
        d = asdict(c)
        assert d["ddg"] == -5.0
        assert "candidate_id" in d


# ===================================================================
# IterationSummary tests
# ===================================================================

class TestIterationSummary:

    def test_defaults(self):
        s = IterationSummary(
            iteration=1, run_id="r1", hypothesis="h",
            objective_mode="ddg_only", n_candidates=5,
            best_ddg=-10.0, mean_ddg=-7.0,
        )
        assert s.selected_ids == []
        assert s.critic_hypothesis == ""
        assert s.report_paths == {}


# ===================================================================
# FlowArtifacts tests  (Medium priority #9)
# ===================================================================

class TestFlowArtifacts:

    def test_from_parts(self, flow_config):
        arts = FlowArtifacts.from_parts(
            run_id="test_run",
            config=flow_config,
            notebook_mapping=[],
            baseline={"ddg": -5.0},
            iterations=[],
            final_candidates=[],
            summary={"status": "ok"},
        )
        assert arts.run_id == "test_run"
        assert arts.created_at is not None
        assert arts.config["template_pdb"] == flow_config.template_pdb

    def test_to_dict(self, flow_config):
        arts = FlowArtifacts.from_parts(
            run_id="test",
            config=flow_config,
            notebook_mapping=[],
            baseline={},
            iterations=[],
            final_candidates=[],
            summary={},
        )
        d = arts.to_dict()
        assert isinstance(d, dict)
        assert d["run_id"] == "test"

    def test_write_json(self, tmp_path, flow_config):
        arts = FlowArtifacts.from_parts(
            run_id="test",
            config=flow_config,
            notebook_mapping=[{"notebook": "n", "pipeline": "p"}],
            baseline={"ddg": -5.0},
            iterations=[],
            final_candidates=[],
            summary={"mode": "test"},
        )
        out = tmp_path / "sub" / "artifacts.json"
        arts.write_json(str(out))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["run_id"] == "test"
        assert loaded["summary"]["mode"] == "test"

    def test_serialization_roundtrip(self, flow_config):
        arts = FlowArtifacts.from_parts(
            run_id="roundtrip_test",
            config=flow_config,
            notebook_mapping=[],
            baseline={"ddg": -3.0},
            iterations=[{"summary": {}, "candidates": []}],
            final_candidates=[{"id": "c1", "sequence": "AAA"}],
            summary={"mode": "test"},
        )
        json_str = json.dumps(arts.to_dict(), ensure_ascii=False)
        reloaded = json.loads(json_str)
        assert reloaded["run_id"] == "roundtrip_test"
        assert reloaded["baseline"]["ddg"] == -3.0
        assert len(reloaded["final_candidates"]) == 1
