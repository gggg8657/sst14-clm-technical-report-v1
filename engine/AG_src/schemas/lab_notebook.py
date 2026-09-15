"""
lab_notebook.py
===============
실험 노트북 마크다운 생성기
Lab notebook markdown generator for the SSTR2 peptide binder pipeline.

각 반복(iteration) 실험의 결과를 구조화된 마크다운 형식으로 기록합니다.
Records each iteration's results in structured Markdown format.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional


# =============================================================================
# Lab Notebook Generator (실험 노트북 생성기)
# =============================================================================

def generate_notebook(
    run_id: str,
    iteration: int,
    config: Dict[str, Any],
    results: Dict[str, Any],
    decisions: Dict[str, Any],
) -> str:
    """
    실험 노트북 마크다운 문자열을 생성합니다.
    Generate a lab notebook Markdown string for one iteration.

    Args:
        run_id:     실행 ID (예: "20260217_1430_iter01").
        iteration:  반복 번호 (1-based).
        config:     파이프라인 설정 딕셔너리 (pipeline_config.yaml 로드 결과).
        results:    각 단계별 결과 요약 딕셔너리.
                    Expected keys:
                        "step04": {"n_total", "n_passed", "plddt_gate"}
                        "step05": {"dock_engine", "dock_top_pct", "n_docking_passed"}
                        "step06": {"n_refined", "ddg_gate", "n_ddg_passed"}
                        "step07": {"lddt_min", "lddt_max"}
                        "top_candidates": list of dicts (rank_table rows)
        decisions:  결정 관련 딕셔너리.
                    Expected keys:
                        "hypothesis", "parameter_changes", "critic_notes", "next_plan"

    Returns:
        마크다운 형식의 노트북 문자열 / Markdown notebook string.
    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    # --- 파라미터 변경 테이블 ---
    param_changes = decisions.get("parameter_changes", {})
    if param_changes:
        param_table_lines = [
            "| Parameter | Previous | New | Rationale |",
            "|-----------|----------|-----|-----------|",
        ]
        for param, info in param_changes.items():
            prev = info.get("previous", "-")
            new = info.get("new", "-")
            rationale = info.get("rationale", "-")
            param_table_lines.append(f"| `{param}` | {prev} | {new} | {rationale} |")
        param_table_str = "\n".join(param_table_lines)
    else:
        param_table_str = "_최초 반복 - 이전 파라미터 없음 / First iteration - no prior parameters._"

    # --- 설정 요약 ---
    iter_cfg = config.get("iteration", {})
    receptor_cfg = config.get("receptor", {})
    config_summary = (
        f"- **n_backbone**: {iter_cfg.get('n_backbone', '?')}\n"
        f"- **k_seq_per_backbone**: {iter_cfg.get('k_seq_per_backbone', '?')}\n"
        f"- **top_m_rosetta**: {iter_cfg.get('top_m_rosetta', '?')}\n"
        f"- **receptor**: {receptor_cfg.get('name', '?')} chain {receptor_cfg.get('chain', '?')}\n"
        f"- **contigs**: `{config.get('contigs', '?')}`\n"
        f"- **hotspot_res**: {config.get('hotspot_res', [])}"
    )

    # --- Step별 결과 ---
    s04 = results.get("step04", {})
    s05 = results.get("step05", {})
    s06 = results.get("step06", {})
    s07 = results.get("step07", {})

    # --- 상위 후보 테이블 ---
    top_candidates = results.get("top_candidates", [])
    if top_candidates:
        top_lines = [
            "| Rank | seq_id | sequence | pLDDT | dock_score | ddG | lDDT | final_score | PASS/FAIL |",
            "|------|--------|----------|-------|------------|-----|------|-------------|-----------|",
        ]
        for i, cand in enumerate(top_candidates[:10], start=1):
            seq = cand.get("sequence", "-")
            if len(seq) > 20:
                seq = seq[:20] + "..."
            top_lines.append(
                f"| {i} "
                f"| {cand.get('seq_id', '-')} "
                f"| `{seq}` "
                f"| {cand.get('plddt_mean', '-')} "
                f"| {cand.get('dock_score', '-')} "
                f"| {cand.get('ddg', '-')} "
                f"| {cand.get('lddt', '-')} "
                f"| {cand.get('final_score', '-')} "
                f"| {cand.get('pass_fail', '-')} |"
            )
        rank_table_top10 = "\n".join(top_lines)
    else:
        rank_table_top10 = "_결과 없음 / No candidates available._"

    # --- 마크다운 조합 ---
    md = textwrap.dedent(f"""\
    # Lab Notebook - {run_id}
    ## Iteration {iteration}
    ### Date: {date_str}

    ---

    ### Hypothesis
    {decisions.get('hypothesis', '_가설 미입력 / No hypothesis provided._')}

    ---

    ### Parameter Changes from Previous Iteration

    {param_table_str}

    ---

    ### Pipeline Configuration

    {config_summary}

    ---

    ### Results Summary

    #### Step 04 - ESMFold QC
    - Total candidates: **{s04.get('n_total', '?')}**
    - Passed pLDDT gate (>={s04.get('plddt_gate', '?')}): **{s04.get('n_passed', '?')}**

    #### Step 05 - Docking
    - Engine: **{s05.get('dock_engine', '?')}**
    - Top {s05.get('dock_top_pct', '?')}% candidates: **{s05.get('n_docking_passed', '?')}**

    #### Step 06 - Rosetta Refinement
    - Candidates refined: **{s06.get('n_refined', '?')}**
    - Passed ddG gate (<={s06.get('ddg_gate', '?')}): **{s06.get('n_ddg_passed', '?')}**

    #### Step 07 - Analysis
    - FoldMason lDDT range: **{s07.get('lddt_min', '?')}** - **{s07.get('lddt_max', '?')}**

    ---

    ### Top Candidates

    {rank_table_top10}

    ---

    ### Critic Analysis

    {decisions.get('critic_notes', '_비평 분석 미입력 / No critic analysis provided._')}

    ---

    ### Next Iteration Plan

    {decisions.get('next_plan', '_다음 계획 미입력 / No next plan provided._')}
    """)

    return md


# =============================================================================
# Decision Log Generator (결정 로그 생성기)
# =============================================================================

def generate_decision_log(
    run_id: str,
    iteration: int,
    critic_analysis: Dict[str, Any],
    parameter_changes: Dict[str, Any],
    hypothesis: str,
) -> str:
    """
    반복 간 결정 로그 마크다운을 생성합니다.
    Generate a decision log Markdown string between iterations.

    Args:
        run_id:             실행 ID.
        iteration:          현재 반복 번호 (다음 반복은 iteration + 1).
        critic_analysis:    Critic 에이전트의 분석 결과.
                            Expected keys: "failures" (list of dicts with
                            "type", "count", "root_cause", "action")
        parameter_changes:  변경할 파라미터 딕셔너리.
                            Expected keys: param_name -> {"previous", "new", "rationale"}
        hypothesis:         다음 반복의 가설 문자열.

    Returns:
        마크다운 결정 로그 문자열 / Markdown decision log string.
    """
    next_iteration = iteration + 1

    # --- 실패 분석 테이블 ---
    failures = critic_analysis.get("failures", [])
    if failures:
        fail_lines = [
            "| Failure Type | Count | Root Cause | Action |",
            "|---|---|---|---|",
        ]
        for f in failures:
            fail_lines.append(
                f"| {f.get('type', '-')} "
                f"| {f.get('count', '-')} "
                f"| {f.get('root_cause', '-')} "
                f"| {f.get('action', '-')} |"
            )
        failure_table = "\n".join(fail_lines)
    else:
        failure_table = "_실패 없음 / No failures recorded._"

    # --- 파라미터 조정 테이블 ---
    if parameter_changes:
        param_lines = [
            "| Parameter | Previous | New | Rationale |",
            "|---|---|---|---|",
        ]
        for param, info in parameter_changes.items():
            param_lines.append(
                f"| `{param}` "
                f"| {info.get('previous', '-')} "
                f"| {info.get('new', '-')} "
                f"| {info.get('rationale', '-')} |"
            )
        param_table = "\n".join(param_lines)
    else:
        param_table = "_파라미터 변경 없음 / No parameter changes._"

    md = textwrap.dedent(f"""\
    # Decision Log - {run_id}
    ## Iteration {iteration} -> {next_iteration}

    ---

    ### Failure Analysis

    {failure_table}

    ---

    ### Parameter Adjustments

    {param_table}

    ---

    ### Hypothesis for Next Iteration

    {hypothesis if hypothesis else '_가설 미입력 / No hypothesis provided._'}
    """)

    return md
