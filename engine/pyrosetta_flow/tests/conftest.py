"""Shared fixtures for pyrosetta_flow tests.

Design note: Fixtures are parameterized where possible so they can be
reused for future 3-ARM pipeline tests (different pipeline types,
different configs, etc.).  Extend via @pytest.fixture(params=...) when
adding new pipeline variants.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.schema import CandidateResult, FlowConfig


# ---------------------------------------------------------------------------
# File / PDB helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_pdb(tmp_path: Path) -> Path:
    """Create a minimal valid PDB file for template_pdb config field."""
    pdb = tmp_path / "template.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
        "END\n"
    )
    return pdb


# ---------------------------------------------------------------------------
# FlowConfig
# ---------------------------------------------------------------------------

@pytest.fixture()
def flow_config(tmp_path: Path, tmp_pdb: Path) -> FlowConfig:
    """Minimal valid FlowConfig pointing at tmp_pdb.

    Parameterise this fixture (via indirect) when testing different
    pipeline types or config variations.
    """
    return FlowConfig(
        template_pdb=str(tmp_pdb),
        original_sequence="AGCKNFFWKTFTSC",
        design_positions=[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14],
        n_candidates=4,
        seed_base=42,
        output_dir=str(tmp_path / "runs"),
        max_iterations=2,
        top_k=3,
    )


# ---------------------------------------------------------------------------
# Mock helpers for runner.py externals
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_run_script():
    """Patch runner._run_script to return a mock CandidateResult dict.

    Usage in tests:
        def test_something(mock_run_script):
            mock_run_script.return_value = {"ddg": -10.0, ...}
    """
    with patch("pyrosetta_flow.runner._run_script") as m:
        m.return_value = {
            "ddg": -8.5,
            "total_score": -120.0,
            "clash_score": 2.0,
        }
        yield m


@pytest.fixture()
def mock_status_emitter():
    """A lightweight fake StatusEmitter that records method calls."""
    emitter = MagicMock()
    emitter.start_step.return_value = 0.0
    emitter.start_rosetta_substep.return_value = 0.0
    return emitter


@pytest.fixture()
def mock_agents():
    """Patch all external agent classes used by runner.py."""
    patches = {
        "planner": patch("pyrosetta_flow.runner.PlannerAgent"),
        "critic": patch("pyrosetta_flow.runner.ScientistCriticAgent"),
        "qcranker": patch("pyrosetta_flow.runner.QCRankerAgent"),
        "reporter": patch("pyrosetta_flow.runner.ReporterAgent"),
        "create_provider": patch("pyrosetta_flow.runner.create_provider"),
        "status_emitter": patch("pyrosetta_flow.runner.StatusEmitter"),
    }
    mocks: Dict[str, Any] = {}
    started = []
    for key, p in patches.items():
        m = p.start()
        started.append(p)
        mocks[key] = m

    # Configure PlannerAgent.execute return
    plan_mock = MagicMock()
    plan_mock.hypothesis = "Test hypothesis"
    plan_mock.parameters = {"mutation_guidance": {}}
    mocks["planner"].return_value.execute.return_value = {"plan": plan_mock}

    # Configure CriticAgent.execute return
    critic_analysis = MagicMock()
    critic_analysis.hypothesis = "Critic feedback"
    critic_analysis.proposed_changes = []
    mocks["critic"].return_value.execute.return_value = {"critic_analysis": critic_analysis}

    # Configure QCRankerAgent.execute return
    mocks["qcranker"].return_value.execute.return_value = {
        "top_candidates": [],
        "rank_table": [],
        "qc_report": {},
    }

    # Configure ReporterAgent.execute return
    mocks["reporter"].return_value.execute.return_value = {
        "report_paths": {"summary_md": "/tmp/summary.md"},
    }

    # Configure StatusEmitter
    emitter_inst = MagicMock()
    emitter_inst.start_step.return_value = 0.0
    emitter_inst.start_rosetta_substep.return_value = 0.0
    mocks["status_emitter"].return_value = emitter_inst
    mocks["emitter_instance"] = emitter_inst

    yield mocks

    for p in started:
        p.stop()


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_experiment_records() -> List[Dict[str, Any]]:
    """Historical JSONL-style records for ranking / bandit tests."""
    return [
        {
            "record_type": "candidate",
            "status": "success",
            "run_id": "run_001",
            "iteration": 1,
            "candidate_id": "iter01_cand001",
            "sequence": "AGCKNFFWKTFTSC",
            "ddg": -5.0,
            "total_score": -100.0,
            "clash_score": 1.0,
            "selected": True,
            "final_score": 5.0,
            "error_summary": "",
        },
        {
            "record_type": "candidate",
            "status": "success",
            "run_id": "run_001",
            "iteration": 1,
            "candidate_id": "iter01_cand002",
            "sequence": "AGCKAEFWKTFTSC",
            "ddg": -12.3,
            "total_score": -150.0,
            "clash_score": 0.5,
            "selected": True,
            "final_score": 12.3,
            "error_summary": "",
        },
        {
            "record_type": "candidate",
            "status": "success",
            "run_id": "run_001",
            "iteration": 1,
            "candidate_id": "iter01_cand003",
            "sequence": "AGCKNFGWKTFTSC",
            "ddg": 3.2,
            "total_score": -80.0,
            "clash_score": 5.0,
            "selected": False,
            "final_score": -3.2,
            "error_summary": "",
        },
        {
            "record_type": "candidate",
            "status": "failed",
            "run_id": "run_001",
            "iteration": 1,
            "candidate_id": "iter01_cand004",
            "sequence": "AGCKNFFWKTFASC",
            "ddg": 999.0,
            "total_score": 999.0,
            "clash_score": 999.0,
            "selected": False,
            "final_score": -999.0,
            "error_summary": "FlexPepDock crashed",
        },
    ]


@pytest.fixture()
def sample_candidates() -> List[CandidateResult]:
    """List of CandidateResult objects for helper function tests."""
    return [
        CandidateResult(
            iteration=1, candidate_id="iter01_cand001",
            sequence="AGCKAEFWKTFTSC", ddg=-12.3,
            total_score=-150.0, clash_score=0.5, selected=True,
        ),
        CandidateResult(
            iteration=1, candidate_id="iter01_cand002",
            sequence="AGCKNFGWKTFTSC", ddg=3.2,
            total_score=-80.0, clash_score=5.0, selected=False,
            fail_reason="clash too high",
        ),
        CandidateResult(
            iteration=1, candidate_id="iter01_cand003",
            sequence="AGCKNFFWKTFASC", ddg=-7.0,
            total_score=-110.0, clash_score=1.5, selected=True,
        ),
    ]
