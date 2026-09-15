"""P5 — 통합 스모크 테스트 (real integration boundaries).

목적: 단위 테스트가 mock 하는 **실제 통합 경계**를 검증한다 —
  FastAPI → subprocess/conda → PyRosetta/vLLM → 상태 파일 → 폴링 루프.
환경/서비스가 없으면 **graceful skip** (CI 깨짐 방지). 호스트에서 `pytest -m smoke`로 실행.

codex P5 권고 반영: vLLM 헬스, bio-tools PyRosetta import, 상태 원자성, 백엔드 엔드포인트
(pipeline_local 의존 포함), 다목적 enrichment, (옵션) 실제 FlexPepDock 1회.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

# 경로
AI4SCI = Path(__file__).resolve().parents[2]          # ai4sci-kaeri
PROJECT_ROOT = AI4SCI.parents[2]                       # SST14-M_scr
BIO_TOOLS_PY = Path.home() / "miniforge3" / "envs" / "bio-tools" / "bin" / "python"
VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
TEMPLATE_PDB = AI4SCI / "data" / "somatostatin_receptor" / "SSTR2_SST14_complex_boltz_1.pdb"

if str(AI4SCI) not in sys.path:
    sys.path.insert(0, str(AI4SCI))


# ---------------------------------------------------------------------------
# 헬퍼: 서비스 가용성
# ---------------------------------------------------------------------------
def _vllm_up() -> bool:
    try:
        with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. vLLM 통합
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _vllm_up(), reason="vLLM(:8000) 미가동 — skip")
def test_smoke_vllm_health_and_generate():
    # /v1/models 에 서빙 모델 존재
    with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=5) as r:
        models = [m["id"] for m in json.loads(r.read())["data"]]
    assert models, "vLLM 에 서빙 모델 없음"
    # 실제 generate (JSON, no-think)
    from AG_src.llm.provider import VLLMProvider
    p = VLLMProvider(model=models[0], base_url=VLLM_URL, enable_thinking=False)
    out = p.generate("Reply with the single word OK.", max_tokens=16, temperature=0.0)
    assert out and isinstance(out, str), f"vLLM generate 실패: {out!r}"


# ---------------------------------------------------------------------------
# 2. bio-tools PyRosetta import (실 도킹 env)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not BIO_TOOLS_PY.exists(), reason="bio-tools conda env 없음 — skip")
def test_smoke_biotools_pyrosetta_import():
    proc = subprocess.run(
        [str(BIO_TOOLS_PY), "-c", "import pyrosetta; print('pyrosetta_ok')"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0 and "pyrosetta_ok" in proc.stdout, \
        f"bio-tools PyRosetta import 실패: {proc.stderr[:300]}"


# ---------------------------------------------------------------------------
# 3. 상태 파일 원자성 (D2) — 라운드트립
# ---------------------------------------------------------------------------
def test_smoke_atomic_status_roundtrip(tmp_path):
    from backend.state import atomic_write_json
    p = tmp_path / "status.json"
    atomic_write_json(p, {"run_id": "smoke", "iteration": 1, "candidates": []})
    assert json.loads(p.read_text())["run_id"] == "smoke"
    assert not (p.with_suffix(".json.tmp")).exists()


# ---------------------------------------------------------------------------
# 4. 백엔드 엔드포인트 (in-process TestClient) — pipeline_local 의존 해결 포함
# ---------------------------------------------------------------------------
def _testclient():
    try:
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("fastapi TestClient(httpx) 미설치 — skip")
    from backend.main import app
    return TestClient(app)


def test_smoke_backend_status_endpoint():
    client = _testclient()
    r = client.get("/api/status")
    assert r.status_code == 200, f"/api/status {r.status_code}"


def test_smoke_backend_flexpepdock_jobs_resolves_pipeline_local():
    """flexpepdock 라우터는 pipeline_local.scripts.flexpepdock_worker 를 import 한다
    (consolidation 의 OUTER_REPO_ROOT/sys.path 수정이 유효해야 200)."""
    client = _testclient()
    r = client.get("/api/flexpepdock/jobs")
    assert r.status_code == 200, f"/api/flexpepdock/jobs {r.status_code} — pipeline_local 의존 미해결?"


# ---------------------------------------------------------------------------
# 5. 다목적 enrichment (서비스 불필요, 빠름) — 반감기/ADMET surrogate 변별력
# ---------------------------------------------------------------------------
def test_smoke_multiobjective_enrichment():
    from pyrosetta_flow.multiobjective import cheap_objectives
    native = cheap_objectives("AGCKNFFWKTFTSC")
    terminal_mut = cheap_objectives("YGCKNFFWKTFTST")
    assert native["half_life_h"] == native["half_life_h"]  # not NaN
    assert 0.0 <= native["admet_score"] <= 1.0
    # 말단 변이가 반감기를 낮춰야 (exopeptidase 취약성)
    assert terminal_mut["half_life_h"] < native["half_life_h"]
    # 재보정(2026-06-09): SST-14 가 과대예측되지 않음 (<1h)
    assert native["half_life_h"] < 1.0


_PEPADMET_REPO = PROJECT_ROOT / "local_models" / "pepadmet" / "repo" / "model" / "toxicity_early_stop.pth"
_PEPADMET_ENV = Path.home() / "miniforge3" / "envs" / "pepadmet" / "bin" / "python"


@pytest.mark.slow
@pytest.mark.skipif(not (_PEPADMET_REPO.exists() and _PEPADMET_ENV.exists()),
                    reason="pepADMET repo/env 없음 — skip")
def test_smoke_pepadmet_real_toxicity_inference():
    """B (2026-06-09): pepADMET GNN 실제 독성 추론 (pepadmet env subprocess)."""
    from pyrosetta_flow.multiobjective import predict_toxicity_for_sequences
    tox = predict_toxicity_for_sequences(["AGCKNFFWKTFTSC"])
    assert "AGCKNFFWKTFTSC" in tox, "pepADMET 추론 결과 없음"
    r = tox["AGCKNFFWKTFTSC"]
    assert r.get("available") is True, f"pepADMET 추론 실패: {r}"
    assert "binary_toxicity" in r and "toxicity_type" in r


# ---------------------------------------------------------------------------
# 6. (옵션, 느림) 실제 FlexPepDock 1회 — 실 ΔG + 이황화결합 검증
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not (BIO_TOOLS_PY.exists() and TEMPLATE_PDB.exists()),
                    reason="bio-tools 또는 템플릿 PDB 없음 — skip")
def test_smoke_real_flexpepdock_one_dock(tmp_path):
    script = AI4SCI / "AG_src" / "scripts" / "flexpep_dock.py"
    out_pdb = tmp_path / "refined.pdb"
    proc = subprocess.run(
        [str(BIO_TOOLS_PY), str(script), "--input", str(TEMPLATE_PDB),
         "--output", str(out_pdb), "--protocol", "flexpep_refine", "--peptide-chain", "1"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"flexpep_dock 실패: {proc.stderr[-400:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert isinstance(result["ddg"], (int, float)), "ddg 가 수치 아님"
    assert result["disulfide_intact"] is True, "Cys3-Cys14 이황화결합 유실"
    assert result["clash_score"] <= 14, "펩타이드 clash(<=14 residues) 비정상"
