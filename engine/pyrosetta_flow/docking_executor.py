"""docking_executor.py
====================
PyRosetta 도킹 subprocess 실행 레이어 (god-object runner.py 에서 분리, 2026-06-09 P1).

conda 환경의 Python 으로 도킹 스크립트(flexpep_dock.py 등)를 subprocess 실행하고
마지막 줄 JSON 을 파싱한다. 순수 함수 — runner 의 거대 오케스트레이터에서 분리해
단위 테스트/재사용을 쉽게 한다. (runner.py 는 이 함수들을 re-export 하여 하위호환 유지.)
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _resolve_conda_python(conda_env: str) -> str:
    """Resolve the Python executable path for a conda environment.

    Tries: conda env python directly > conda run > sys.executable fallback.
    """
    if not conda_env:
        return sys.executable
    # Try direct path to env python (works without conda on PATH)
    for base in [
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]:
        env_python = base / "envs" / conda_env / "bin" / "python"
        if env_python.exists():
            return str(env_python)
    # Fallback: try conda run (requires conda on PATH)
    return ""


def _run_script(
    script_path: Path,
    args: List[str],
    conda_env: str,
    cwd: Path,
    timeout: int = 300,
) -> Dict[str, Any]:
    env_python = _resolve_conda_python(conda_env)
    if env_python:
        cmd = [env_python, str(script_path), *args]
    elif conda_env:
        cmd = ["conda", "run", "-n", conda_env, "python", str(script_path), *args]
    else:
        cmd = [sys.executable, str(script_path), *args]
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, check=False,
            timeout=timeout,  # C3: prevent infinite hangs
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Script timed out after {timeout}s: {script_path.name}")
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        details = stderr or stdout or "no stderr/stdout"
        cmd_str = " ".join(cmd)
        raise RuntimeError(f"Script failed: {script_path.name} :: {details} :: cmd={cmd_str}")
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return {}
    # C4: protect JSON parsing from malformed stdout
    try:
        return json.loads(lines[-1])
    except (json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to parse JSON output from {script_path.name}: {exc}"
            f"\nstdout last line: {lines[-1][:200]}"
        )
