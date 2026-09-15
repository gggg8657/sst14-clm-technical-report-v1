"""
pepadmet_runner.py 단위 테스트.

pepadmet env / pepADMET repo 없이도 실행 가능한 mock 기반 테스트.
실제 conda 환경은 integration test에서 별도 검증.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── 경로 상수 검증 ──────────────────────────────────────────────────────────
def test_infer_script_path_is_sibling():
    """_INFER_SCRIPT가 runner.py와 같은 디렉터리에 위치해야 함."""
    from pyrosetta_flow.pepadmet_runner import _INFER_SCRIPT, _PEPADMET_REPO

    assert _INFER_SCRIPT.parent == Path(__file__).resolve().parent.parent, (
        "_INFER_SCRIPT는 pyrosetta_flow/ 내에 있어야 함"
    )
    assert _INFER_SCRIPT.name == "pepadmet_infer_script.py"


def test_infer_script_file_exists():
    """pepadmet_infer_script.py 파일이 실제로 존재해야 함."""
    from pyrosetta_flow.pepadmet_runner import _INFER_SCRIPT

    assert _INFER_SCRIPT.exists(), f"파일 없음: {_INFER_SCRIPT}"


# ── repo 없을 때 graceful fallback ──────────────────────────────────────────
def test_predict_toxicity_batch_no_repo():
    """pepADMET repo가 없으면 available=False 반환."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    with patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", Path("/nonexistent/repo")):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"])

    assert len(results) == 1
    assert results[0]["available"] is False
    assert "not found" in results[0]["error"]


def test_predict_toxicity_batch_no_infer_script():
    """infer script가 없으면 available=False 반환."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", Path("/")),  # 존재하는 경로
        patch("pyrosetta_flow.pepadmet_runner._INFER_SCRIPT", Path("/nonexistent/script.py")),
    ):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"])

    assert results[0]["available"] is False
    assert "pepadmet_infer_script.py not found" in results[0]["error"]


# ── subprocess mock 기반 테스트 ─────────────────────────────────────────────
_MOCK_OUTPUT = json.dumps([
    {
        "sequence": "AGCKNFFWKTFTSC",
        "available": True,
        "binary_toxicity": 0.1234,
        "is_toxic": False,
        "toxicity_type": "cytolysis",
        "toxicity_type_confidence": 0.8,
        "neurotoxicity_type": "AChR_inhibitor",
        "neurotoxicity_confidence": 0.6,
        "hc50": 3.14,
    }
])


@pytest.fixture()
def mock_repo(tmp_path):
    """임시 pepADMET repo 디렉터리 생성."""
    repo = tmp_path / "local_models" / "pepadmet" / "repo"
    repo.mkdir(parents=True)
    return repo


def test_subprocess_called_with_script_file(mock_repo):
    """subprocess.run이 -c 대신 스크립트 파일 경로로 호출되는지 검증."""
    from pyrosetta_flow.pepadmet_runner import _INFER_SCRIPT, predict_toxicity_batch

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _MOCK_OUTPUT
    mock_result.stderr = ""

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", return_value=mock_result) as mock_run,
    ):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"], smiles_list=["C"])

    cmd = mock_run.call_args[0][0]
    # -c 플래그 없음
    assert "-c" not in cmd, "inline script(-c) 사용 금지"
    # 스크립트 파일 경로 포함
    assert str(_INFER_SCRIPT) in cmd
    # --no-capture-output 포함
    assert "--no-capture-output" in cmd
    # conda run -n pepadmet 포함
    assert "conda" in cmd
    assert "-n" in cmd
    assert "pepadmet" in cmd


def test_json_output_parsed_correctly(mock_repo):
    """subprocess stdout JSON이 올바르게 파싱되는지 검증."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "some log line\n" + _MOCK_OUTPUT
    mock_result.stderr = ""

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", return_value=mock_result),
    ):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"], smiles_list=["C"])

    assert len(results) == 1
    r = results[0]
    assert r["available"] is True
    assert r["sequence"] == "AGCKNFFWKTFTSC"
    assert r["binary_toxicity"] == pytest.approx(0.1234)
    assert r["is_toxic"] is False
    assert r["hc50"] == pytest.approx(3.14)


def test_subprocess_nonzero_returncode(mock_repo):
    """subprocess 실패 시 error 필드 포함 결과 반환."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "ModuleNotFoundError: No module named 'dgl'"

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", return_value=mock_result),
    ):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"], smiles_list=["C"])

    assert results[0]["available"] is False
    assert "ModuleNotFoundError" in results[0]["error"]


def test_timeout_returns_error(mock_repo):
    """subprocess timeout 시 error 반환."""
    import subprocess as sp
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", side_effect=sp.TimeoutExpired(cmd="conda", timeout=120)),
    ):
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"], smiles_list=["C"])

    assert results[0]["available"] is False
    assert results[0]["error"] == "timeout"


def test_env_var_pepadmet_repo_passed(mock_repo):
    """subprocess 호출 시 PEPADMET_REPO 환경변수가 전달되는지 검증."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = _MOCK_OUTPUT
    mock_result.stderr = ""

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", return_value=mock_result) as mock_run,
    ):
        predict_toxicity_batch(["AGCKNFFWKTFTSC"], smiles_list=["C"])

    kwargs = mock_run.call_args[1]
    assert "env" in kwargs
    assert kwargs["env"]["PEPADMET_REPO"] == str(mock_repo)


def test_smiles_fallback_on_import_error(mock_repo):
    """smiles_converter import 실패 시 빈 SMILES로 진행."""
    from pyrosetta_flow.pepadmet_runner import predict_toxicity_batch

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps([
        {"sequence": "AGCKNFFWKTFTSC", "error": "graph build failed", "available": False}
    ])
    mock_result.stderr = ""

    with (
        patch("pyrosetta_flow.pepadmet_runner._PEPADMET_REPO", mock_repo),
        patch("pyrosetta_flow.pepadmet_runner.subprocess.run", return_value=mock_result),
        patch("pyrosetta_flow.pepadmet_runner.__builtins__", {}),  # ImportError 유발
    ):
        # smiles_list=None 전달 → ImportError 발생하면 [""] 로 대체
        results = predict_toxicity_batch(["AGCKNFFWKTFTSC"])

    # 에러 없이 결과 반환
    assert len(results) == 1
