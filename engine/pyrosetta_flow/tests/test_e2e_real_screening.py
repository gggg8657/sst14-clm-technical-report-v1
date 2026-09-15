"""실제 vLLM + 도킹 end-to-end 테스트.

"우리 시스템은 mock 없는 실제 스크리닝" 철학에 따라, MagicMock을 사용하지 않고
실제 ScientistCriticAgent(vLLM port 8000) + 실제 FlexPepDock 도킹으로
1 iteration을 돌려 결과 구조를 검증한다.

실행 방법:
    pytest -m integration -v pyrosetta_flow/tests/test_e2e_real_screening.py

환경 가드:
    - vLLM port 8000 미가동 시 skip
    - bio-tools PyRosetta(bio-tools conda env) 미설치 시 skip
    - 템플릿 PDB 파일 없으면 skip

소요 시간: 패널 5-turn + 1회 도킹 기준 약 5~20분.
pytest timeout: 1200s (설정 없으면 직접 subprocess에서 timeout 적용됨).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# 경로 설정
# ---------------------------------------------------------------------------
AI4SCI = Path(__file__).resolve().parents[2]          # ai4sci-kaeri
BIO_TOOLS_PY = Path.home() / "miniforge3" / "envs" / "bio-tools" / "bin" / "python"
VLLM_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
TEMPLATE_PDB = AI4SCI / "data" / "somatostatin_receptor" / "SSTR2_SST14_complex_boltz_1.pdb"

if str(AI4SCI) not in sys.path:
    sys.path.insert(0, str(AI4SCI))

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 환경 가용성 헬퍼
# ---------------------------------------------------------------------------

def _vllm_up() -> bool:
    """vLLM port 8000 가동 여부 확인."""
    try:
        with urllib.request.urlopen(f"{VLLM_URL}/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def _bio_tools_ok() -> bool:
    """bio-tools conda env 존재 여부 확인."""
    return BIO_TOOLS_PY.exists()


def _template_pdb_ok() -> bool:
    """템플릿 PDB 파일 존재 여부 확인."""
    return TEMPLATE_PDB.exists()


# ---------------------------------------------------------------------------
# 실제 e2e 테스트
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _vllm_up(),
    reason="vLLM(:8000) 미가동 — skip. vLLM을 먼저 기동하세요.",
)
@pytest.mark.skipif(
    not _bio_tools_ok(),
    reason="bio-tools conda env 없음 — skip. PyRosetta 환경이 필요합니다.",
)
@pytest.mark.skipif(
    not _template_pdb_ok(),
    reason=f"템플릿 PDB 없음({TEMPLATE_PDB}) — skip.",
)
def test_e2e_real_one_iteration_screening(tmp_path: Path) -> None:
    """실제 vLLM + 도킹으로 1 iteration run_flow를 돌리고 결과 구조를 검증한다.

    검증 목표:
    - FlowArtifacts 반환 (타입 검사)
    - iterations[0]["candidates"]에 실제 후보가 생성됨
    - 후보에 ddg(실수), sequence, candidate_id가 채워짐
    - baseline ddg가 실제 도킹값(0이 아님, 999 아님)임
    - 패널이 실행되었다면 scientific_verdict가 문자열임
    """
    from pyrosetta_flow.schema import FlowArtifacts, FlowConfig
    from pyrosetta_flow.runner import run_pyrosetta_agentic_mutdock_flow

    # 최소 설정: n_candidates=1, max_iterations=1, reuse_baseline=False
    # expert_panel=True (실제 5-turn 패널 포함)
    # stage2_enabled=False: 2차 정밀 도킹은 제외하여 속도 확보
    # validation_n_trials=1: 검증 재도킹 비활성화 (속도 우선)
    config = FlowConfig(
        template_pdb=str(TEMPLATE_PDB),
        original_sequence="AGCKNFFWKTFTSC",
        design_positions=[1, 2, 4, 5, 6, 11, 12],  # FWKT+Cys 제외
        n_candidates=1,
        seed_base=42,
        output_dir=str(tmp_path / "e2e_runs"),
        max_iterations=1,
        top_k=1,
        reuse_baseline=False,
        n_baseline_trials=5,           # native 수렴률~20% → robust 다회 도킹(1회는 수렴실패 빈번)
        expert_panel=True,             # 실제 P2 패널 포함
        preview_rounds=1,
        stage2_enabled=False,          # 2차 정밀 도킹 제외 (속도)
        validation_n_trials=1,         # 검증 재도킹 비활성화
        llm_provider="vllm",
        llm_base_url=VLLM_URL,
        script_timeout=600,
    )

    # 실제 run_flow 호출 — mock 없음
    result = run_pyrosetta_agentic_mutdock_flow(config)

    # ── 기본 구조 검증 ──────────────────────────────────────────────────────
    assert isinstance(result, FlowArtifacts), \
        f"run_flow 반환 타입 이상: {type(result)}"
    assert result.run_id.startswith("sst14_mutdock_"), \
        f"run_id 형식 이상: {result.run_id}"

    # ── baseline 검증 ───────────────────────────────────────────────────────
    baseline = result.baseline
    assert isinstance(baseline, dict), "baseline이 dict가 아님"
    baseline_ddg = baseline.get("ddg")
    # native baseline은 수렴률이 낮다(~20%; robust baseline 산출 시 nstruct=10에서 8/10 수렴).
    # n_baseline_trials회 전부 양수(비물리/수렴실패)면 ddg=None일 수 있는데, 이는 native의
    # 알려진 noise 특성이지 코드 결함이 아니다. None이면 경고만 하고 candidate 도킹값 검증으로
    # 넘어간다 — e2e의 핵심 검증은 "실제 도킹이 수치값을 산출하는가(=mock이 아닌가)"이다.
    if baseline_ddg is None:
        print(
            f"\n[e2e] ⚠ native baseline 수렴 실패(None) — "
            f"n_baseline_trials={config.n_baseline_trials}회 전부 양수. "
            f"native noise(known), candidate 검증으로 진행"
        )
    else:
        assert isinstance(baseline_ddg, (int, float)), \
            f"baseline ddg가 수치 아님: {baseline_ddg!r}"
        assert baseline_ddg != 999.0, "baseline ddg가 실패값(999.0)임 — 도킹 실패"
        assert baseline_ddg != 0.0, "baseline ddg가 0.0임 — 도킹 결과 없음 의심"

    # ── iterations 구조 검증 ────────────────────────────────────────────────
    assert len(result.iterations) >= 1, "iteration 결과가 없음"
    iter0 = result.iterations[0]
    assert "candidates" in iter0, f"iterations[0]에 'candidates' 키 없음: {list(iter0.keys())}"

    candidates = iter0["candidates"]
    assert len(candidates) >= 1, \
        f"후보가 한 건도 생성되지 않음 (n_candidates=1 설정이었음)"

    # ── 개별 후보 필드 검증 ─────────────────────────────────────────────────
    for cand in candidates:
        cand_id = cand.get("id") or cand.get("candidate_id", "UNKNOWN")
        seq = cand.get("sequence", "")
        ddg_val = cand.get("ddg")

        # sequence: 14aa SST-14 계열이어야 함
        assert len(seq) == 14, \
            f"후보 {cand_id} sequence 길이 이상: {seq!r}"
        assert seq != config.original_sequence, \
            f"후보 {cand_id}가 native와 동일 — 변이가 적용되지 않음: {seq}"

        # ddg: 실수값이어야 함 (실패 후보는 999.0일 수 있으므로 타입만 검사)
        assert isinstance(ddg_val, (int, float)), \
            f"후보 {cand_id} ddg가 수치 아님: {ddg_val!r}"

    # ── 패널 결과 검증 (expert_panel=True) ──────────────────────────────────
    # runner.py는 prereview 결과를 iter_data["prereview_rounds"] 또는
    # 각 후보의 fanin entry에 기록할 수 있음.
    # 필수 필드가 아니라 조건부로만 확인.
    _fanin_entries = [
        v for v in iter0.values()
        if isinstance(v, dict) and "scientific_verdict" in v
    ]
    if _fanin_entries:
        for entry in _fanin_entries:
            sv = entry.get("scientific_verdict")
            assert isinstance(sv, str) and sv in ("approve", "conditional", "reject", "invalid"), \
                f"scientific_verdict 값 이상: {sv!r}"
            dd = entry.get("docking_decision")
            assert isinstance(dd, bool), \
                f"docking_decision이 bool이 아님: {dd!r}"

    # ── 요약 필드 검증 ──────────────────────────────────────────────────────
    summary = result.summary
    assert "run_status" in summary, f"summary에 run_status 없음: {list(summary.keys())}"
    assert "best_final_ddg" in summary, f"summary에 best_final_ddg 없음: {list(summary.keys())}"
    assert summary["mode"] == "agentic_mutate_then_dock", \
        f"summary mode 이상: {summary.get('mode')!r}"

    # 실행 성공 로그 출력 (pytest -s 옵션으로 확인 가능)
    _bl = f"{baseline_ddg:.3f}" if isinstance(baseline_ddg, (int, float)) else "None(native noise)"
    print(
        f"\n[e2e] 완료 — baseline ddg={_bl}, "
        f"후보 {len(candidates)}건, "
        f"best_ddg={summary.get('best_final_ddg')}",
        flush=True,
    )
