"""Tests for runner.py helper functions: _run_script, _summarize_iteration, _emit_candidates."""
from __future__ import annotations

import json
import subprocess
import statistics
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.schema import CandidateResult, IterationSummary


# ===================================================================
# _run_script tests  (Critical priority #1)
# ===================================================================

class TestRunScript:
    """Test subprocess execution wrapper."""

    def _import_run_script(self):
        from pyrosetta_flow.runner import _run_script
        return _run_script

    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="/usr/bin/python")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_success(self, mock_subprocess, mock_conda):
        _run_script = self._import_run_script()
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='some log\n{"ddg": -10.5, "total_score": -120.0}\n',
            stderr="",
        )
        result = _run_script(
            script_path=Path("/scripts/dock.py"),
            args=["--input", "test.pdb"],
            conda_env="bio-tools",
            cwd=Path("/work"),
        )
        assert result == {"ddg": -10.5, "total_score": -120.0}
        mock_subprocess.assert_called_once()

    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="/usr/bin/python")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_timeout(self, mock_subprocess, mock_conda):
        _run_script = self._import_run_script()
        mock_subprocess.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=300)
        with pytest.raises(RuntimeError, match="timed out"):
            _run_script(
                script_path=Path("/scripts/dock.py"),
                args=[],
                conda_env="bio-tools",
                cwd=Path("/work"),
            )

    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="/usr/bin/python")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_nonzero_exit(self, mock_subprocess, mock_conda):
        _run_script = self._import_run_script()
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Segfault in librosetta",
        )
        with pytest.raises(RuntimeError, match="Script failed"):
            _run_script(
                script_path=Path("/scripts/dock.py"),
                args=[],
                conda_env="bio-tools",
                cwd=Path("/work"),
            )

    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="/usr/bin/python")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_json_parse_error(self, mock_subprocess, mock_conda):
        _run_script = self._import_run_script()
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="not valid json at all\n",
            stderr="",
        )
        with pytest.raises(RuntimeError, match="Failed to parse JSON"):
            _run_script(
                script_path=Path("/scripts/dock.py"),
                args=[],
                conda_env="bio-tools",
                cwd=Path("/work"),
            )

    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="/usr/bin/python")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_empty_stdout(self, mock_subprocess, mock_conda):
        _run_script = self._import_run_script()
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )
        result = _run_script(
            script_path=Path("/scripts/dock.py"),
            args=[],
            conda_env="bio-tools",
            cwd=Path("/work"),
        )
        assert result == {}

    # 2026-06-09 P1: _run_script/_resolve_conda_python 가 docking_executor 로 이동 → patch 대상 갱신
    @patch("pyrosetta_flow.docking_executor._resolve_conda_python", return_value="")
    @patch("pyrosetta_flow.docking_executor.subprocess.run")
    def test_run_script_conda_run_fallback(self, mock_subprocess, mock_conda):
        """When no direct env python found, uses 'conda run -n' prefix."""
        _run_script = self._import_run_script()
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout='{"ddg": -5.0}\n',
            stderr="",
        )
        _run_script(
            script_path=Path("/scripts/dock.py"),
            args=["--flag"],
            conda_env="bio-tools",
            cwd=Path("/work"),
        )
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0] if call_args[0] else call_args[1]["cmd"]
        # Should start with conda run -n bio-tools python ...
        assert cmd[0] == "conda"
        assert cmd[1] == "run"
        assert cmd[3] == "bio-tools"


# ===================================================================
# _summarize_iteration tests  (Medium priority #11)
# ===================================================================

class TestSummarizeIteration:

    def _import_summarize(self):
        from pyrosetta_flow.runner import _summarize_iteration
        return _summarize_iteration

    def test_basic_summary(self, sample_candidates):
        _summarize_iteration = self._import_summarize()
        selected = [c for c in sample_candidates if c.selected]
        summary = _summarize_iteration(
            iteration=1,
            run_id="test_run",
            hypothesis="Test hypothesis",
            objective_mode="ddg_only",
            selected=selected,
            report_paths={"summary_md": "/tmp/summary.md"},
            critic_hypothesis="Critic says improve",
        )
        assert isinstance(summary, IterationSummary)
        assert summary.iteration == 1
        assert summary.run_id == "test_run"
        assert summary.n_candidates == 2
        expected_best = min(c.ddg for c in selected)
        assert summary.best_ddg == round(expected_best, 4)
        expected_mean = statistics.mean(c.ddg for c in selected)
        assert summary.mean_ddg == round(expected_mean, 4)

    def test_empty_selected(self):
        _summarize_iteration = self._import_summarize()
        summary = _summarize_iteration(
            iteration=1, run_id="r", hypothesis="h",
            objective_mode="ddg_only", selected=[],
            report_paths={}, critic_hypothesis="",
        )
        assert summary.best_ddg == 0.0
        assert summary.mean_ddg == 0.0
        assert summary.n_candidates == 0


# ===================================================================
# _emit_candidates tests  (Medium priority #11)
# ===================================================================

class TestEmitCandidates:

    def _import_emit(self):
        from pyrosetta_flow.runner import _emit_candidates
        return _emit_candidates

    def test_emit_candidates_sorting(self, sample_candidates):
        _emit_candidates = self._import_emit()
        emitter = MagicMock()
        _emit_candidates(emitter, sample_candidates, ddg_threshold=-5.0)
        emitter.set_candidates.assert_called_once()
        rows = emitter.set_candidates.call_args[0][0]
        # Should be sorted by ddg ascending
        ddg_values = [r["ddG"] for r in rows]
        assert ddg_values == sorted(ddg_values)

    def test_emit_pass_fail_logic(self, sample_candidates):
        _emit_candidates = self._import_emit()
        emitter = MagicMock()
        _emit_candidates(emitter, sample_candidates, ddg_threshold=-5.0)
        rows = emitter.set_candidates.call_args[0][0]
        results = {r["id"]: r["result"] for r in rows}
        # selected=True → PASS
        assert results["iter01_cand001"] == "PASS"
        # ddg=3.2, not selected, ddg > threshold → FAIL
        assert results["iter01_cand002"] == "FAIL"
        # selected=True → PASS
        assert results["iter01_cand003"] == "PASS"

    def test_emit_fail_reason_passthrough(self, sample_candidates):
        _emit_candidates = self._import_emit()
        emitter = MagicMock()
        _emit_candidates(emitter, sample_candidates, ddg_threshold=-5.0)
        rows = emitter.set_candidates.call_args[0][0]
        fail_row = next(r for r in rows if r["id"] == "iter01_cand002")
        assert fail_row["failReason"] == "clash too high"
