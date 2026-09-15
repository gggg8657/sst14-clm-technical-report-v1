"""
reporter.py
SSTR2 펩타이드 바인더 Co-Scientist - Reporter 에이전트
Role: 최종 리포트 / 그림 자동화 (Final Report & Automated Visualization)

Reporter는 최종 상위 후보에 대해 PyMOL 렌더 스크립트를 생성하고,
실험 기록(설정/결과/랭킹/선정 이유)을 Markdown 문서로 자동화한다.
"""

from __future__ import annotations

import csv
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .base_agent import BaseAgent
from .critic import CriticAnalysis
from .qc_ranker import Candidate, RankTable
from ..llm.prompts import get_system_prompt, format_reporter_prompt


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RenderPaths:
    """PyMOL 4-panel 렌더 결과 경로 모음.

    Attributes:
        overview: 전체 복합체 cartoon 렌더 (PNG)
        closeup: 결합 포켓 zoom 렌더 (PNG)
        interface_contacts: H-bond/salt bridge/소수성 접촉 렌더 (PNG)
        electrostatics: APBS 정전기 포텐셜 표면 렌더 (PNG)
        pymol_session: PyMOL 세션 파일 (.pse)
        pymol_script: 생성된 PyMOL 스크립트 경로 (.pml)
    """
    overview: str = ""
    closeup: str = ""
    interface_contacts: str = ""
    electrostatics: str = ""
    pymol_session: str = ""
    pymol_script: str = ""


# ---------------------------------------------------------------------------
# Reporter agent
# ---------------------------------------------------------------------------

class ReporterAgent(BaseAgent):
    """최종 리포트 및 그림 자동화 에이전트.

    역할:
        1. PyMOL 4-panel 스크립트 자동 생성 (overview/closeup/interface/electrostatics)
        2. 실험 기록 문서화 (설정/결과/랭킹/선정 이유)
        3. Lab notebook 항목 생성 (Markdown)
        4. 반복 비교 그림 생성 보조
        5. 최종 rank_table.csv 및 summary.md 저장

    Attributes:
        runs_base_dir: 실험 결과 루트 디렉터리
    """

    def __init__(
        self,
        runs_base_dir: str = "runs",
        llm_provider: str = "claude",
    ) -> None:
        super().__init__(
            name="Reporter",
            role="최종 리포트/그림 자동화",
            description=(
                "상위 후보에 대한 PyMOL 렌더 스크립트 생성, 실험 기록 문서화, "
                "lab notebook 작성, rank_table.csv 저장을 담당한다."
            ),
            llm_provider=llm_provider,
        )
        self.runs_base_dir = Path(runs_base_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pymol_renders(
        self,
        top_candidates: list[Candidate],
        receptor_pdb: str,
        output_dir: str,
    ) -> dict[str, RenderPaths]:
        """상위 후보 각각에 대해 PyMOL 4-panel 렌더 스크립트를 생성한다.

        생성 패널:
            1. Overview:      전체 복합체 cartoon (수용체=회색, 펩타이드=청색)
            2. Closeup:       결합 포켓 zoom (포켓 residue 강조)
            3. Interface contacts: H-bond, salt bridge, 소수성 접촉 표시
            4. Electrostatics: APBS 기반 정전기 포텐셜 표면

        Args:
            top_candidates: 렌더할 상위 Candidate 목록
            receptor_pdb: 수용체 PDB 파일 경로
            output_dir: 렌더 파일 저장 디렉터리

        Returns:
            {candidate_id: RenderPaths} 딕셔너리
        """
        self.log(f"PyMOL 렌더 스크립트 생성: {len(top_candidates)}개 후보")
        out = Path(output_dir)
        viz_dir = out / "07_viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, RenderPaths] = {}

        for cand in top_candidates:
            cid = cand.candidate_id
            script_path = viz_dir / f"{cid}_render.pml"
            pse_path = viz_dir / f"{cid}.pse"

            script = self._build_pymol_script(
                candidate=cand,
                receptor_pdb=receptor_pdb,
                output_dir=str(viz_dir),
                pse_path=str(pse_path),
            )

            script_path.write_text(script, encoding="utf-8")
            self.log(f"  스크립트 저장: {script_path}")

            results[cid] = RenderPaths(
                overview=str(viz_dir / f"{cid}_1_overview.png"),
                closeup=str(viz_dir / f"{cid}_2_closeup.png"),
                interface_contacts=str(viz_dir / f"{cid}_3_interface.png"),
                electrostatics=str(viz_dir / f"{cid}_4_electrostatics.png"),
                pymol_session=str(pse_path),
                pymol_script=str(script_path),
            )

        return results

    def generate_rank_table_csv(
        self,
        rank_table: RankTable,
        output_path: str,
    ) -> str:
        """랭킹 테이블을 CSV 파일로 저장하고 경로를 반환한다.

        CSV 컬럼:
            rank, backbone_id, seq_id, candidate_id, sequence,
            plddt_mean, plddt_interface, dock_score, ddg,
            clash_count, constraint_violations, lddt,
            final_score, pass_gates, fail_reasons

        Args:
            rank_table: 저장할 RankTable
            output_path: CSV 저장 경로

        Returns:
            저장된 파일 경로
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "rank", "backbone_id", "seq_id", "candidate_id", "sequence",
            "plddt_mean", "plddt_interface", "dock_score", "ddg",
            "clash_count", "constraint_violations", "lddt",
            "final_score", "pass_gates", "fail_reasons",
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, c in enumerate(rank_table.ranked_candidates, start=1):
                writer.writerow({
                    "rank": rank,
                    "backbone_id": c.backbone_id,
                    "seq_id": c.seq_id,
                    "candidate_id": c.candidate_id,
                    "sequence": c.sequence,
                    "plddt_mean": f"{c.plddt_mean:.2f}",
                    "plddt_interface": f"{c.plddt_interface:.2f}",
                    "dock_score": f"{c.dock_score:.4f}",
                    "ddg": f"{c.ddg:.4f}",
                    "clash_count": c.clash_count,
                    "constraint_violations": c.constraint_violations,
                    "lddt": f"{c.lddt:.4f}",
                    "final_score": f"{c.final_score:.6f}",
                    "pass_gates": c.pass_gates,
                    "fail_reasons": "; ".join(c.fail_reasons),
                })

        self.log(f"rank_table.csv 저장: {path}")
        return str(path)

    def generate_summary_report(
        self,
        run_id: str,
        iteration: int,
        rank_table: RankTable,
        critic_analysis: Optional[CriticAnalysis] = None,
    ) -> str:
        """iteration 요약 Markdown 보고서를 생성하고 문자열로 반환한다.

        포함 내용:
            - Run 정보 (run_id, iteration, 생성 시각)
            - 랭킹 상위 10개 후보 테이블
            - QC 통과율
            - Critic 분석 및 다음 iteration 제안 (있는 경우)

        Args:
            run_id: 실행 식별자
            iteration: 반복 번호
            rank_table: 랭킹 결과
            critic_analysis: Critic 분석 결과 (선택)

        Returns:
            Markdown 형식 보고서 문자열
        """
        # --- LLM 분기: LLM이 있으면 프롬프트 기반 요약 시도 ---
        if self.has_llm:
            llm_report = self._summarize_via_llm(
                run_id, iteration, rank_table, critic_analysis,
            )
            if llm_report is not None:
                self.log("LLM 기반 요약 보고서 생성 완료")
                return llm_report
            self.log("LLM 요약 실패, 템플릿 기반 폴백", level="warning")

        # --- 템플릿 기반 폴백 (기존 로직) ---
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        top10 = rank_table.ranked_candidates[:10]

        # 랭킹 Markdown 테이블
        table_header = (
            "| Rank | ID | pLDDT | Dock | ddG | lDDT | Score |\n"
            "|------|-----|-------|------|-----|------|-------|\n"
        )
        table_rows = "".join(
            f"| {i+1} | {c.candidate_id} | {c.plddt_mean:.1f} | "
            f"{c.dock_score:.2f} | {c.ddg:.2f} | {c.lddt:.3f} | {c.final_score:.4f} |\n"
            for i, c in enumerate(top10)
        )

        # Critic 제안 섹션
        critic_section = ""
        if critic_analysis:
            changes_md = "\n".join(
                f"- **{ch.parameter_name}**: `{ch.old_value}` → `{ch.new_value}`  \n"
                f"  근거: {ch.rationale}  \n"
                f"  예상 효과: {ch.expected_effect}"
                for ch in critic_analysis.proposed_changes
            )
            critic_section = textwrap.dedent(f"""
            ## Scientist Critic 분석

            **구조적 인사이트**
            {critic_analysis.structural_insights}

            **실패 유형 분포**
            {self._dict_to_md_list(critic_analysis.failure_summary)}

            **다음 Iteration 파라미터 변경 제안 (최대 2개)**
            {changes_md if changes_md else '변경 없음'}

            **다음 Iteration 가설**
            > {critic_analysis.hypothesis}
            """).strip()

        report = textwrap.dedent(f"""
        # SSTR2 펩타이드 바인더 실험 요약

        - **Run ID**: `{run_id}`
        - **Iteration**: {iteration}
        - **생성 시각**: {now}
        - **총 통과 후보**: {len(rank_table.ranked_candidates)}개

        ## 상위 후보 랭킹 (Top 10)

        {table_header}{table_rows}

        {critic_section}
        """).strip()

        return report

    def generate_lab_notebook(
        self,
        run_id: str,
        all_iterations: list[dict[str, Any]],
    ) -> str:
        """전체 iteration에 걸친 Lab Notebook을 Markdown으로 생성한다.

        각 항목:
            - iteration 번호 및 run_id
            - 가설 (hypothesis)
            - 주요 파라미터
            - QC 통과율
            - 상위 후보 ID 및 점수
            - 변경 사항 및 관찰 사항
            - 다음 iteration 계획

        Args:
            run_id: 최신 실행 식별자
            all_iterations: 각 iteration의 결과 딕셔너리 목록
                각 항목: {'iteration', 'run_id', 'hypothesis',
                           'parameters', 'qc_pass_rate',
                           'top_candidates', 'changes_from_prev'}

        Returns:
            Markdown 형식 Lab Notebook 문자열
        """
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        sections: list[str] = [
            f"# SSTR2 Peptide Binder - Lab Notebook\n\n"
            f"**프로젝트**: SSTR2 de novo 펩타이드 바인더 설계  \n"
            f"**최신 Run**: `{run_id}`  \n"
            f"**기록 시각**: {now}  \n"
            f"**총 Iteration 수**: {len(all_iterations)}\n\n"
            f"---\n"
        ]

        for entry in all_iterations:
            itr = entry.get("iteration", "?")
            itr_run_id = entry.get("run_id", "unknown")
            hypothesis = entry.get("hypothesis", "")
            params = entry.get("parameters", {})
            pass_rate = entry.get("qc_pass_rate", 0.0)
            top = entry.get("top_candidates", [])
            changes = entry.get("changes_from_prev", [])

            param_lines = "\n".join(
                f"  - {k}: `{v}`" for k, v in params.items()
            )
            change_lines = (
                "\n".join(f"  - {c}" for c in changes) if changes else "  - (변경 없음)"
            )
            top_lines = (
                "\n".join(
                    f"  - {c.get('id','?')}: score={c.get('score','?')}" for c in top[:5]
                )
                if top else "  - 없음"
            )

            sections.append(textwrap.dedent(f"""
            ## Iteration {itr} — `{itr_run_id}`

            **가설**
            > {hypothesis}

            **주요 파라미터**
            {param_lines}

            **이전 대비 변경 사항**
            {change_lines}

            **QC 통과율**: {pass_rate*100:.1f}%

            **상위 후보 (Top 5)**
            {top_lines}

            ---
            """).strip())

        return "\n\n".join(sections)

    def save_all_reports(
        self,
        run_id: str,
        iteration: int,
        rank_table: RankTable,
        critic_analysis: Optional[CriticAnalysis],
        output_dir: str,
    ) -> dict[str, str]:
        """summary.md 및 rank_table.csv를 파일로 저장하고 경로를 반환한다.

        Args:
            run_id: 실행 식별자
            iteration: 반복 번호
            rank_table: 랭킹 테이블
            critic_analysis: Critic 분석 결과
            output_dir: 저장 디렉터리 (runs/{run_id}/)

        Returns:
            {'summary_md': path, 'rank_csv': path}
        """
        reports_dir = Path(output_dir) / "08_reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        summary_md = self.generate_summary_report(run_id, iteration, rank_table, critic_analysis)
        summary_path = reports_dir / "summary.md"
        summary_path.write_text(summary_md, encoding="utf-8")

        csv_path = self.generate_rank_table_csv(
            rank_table, str(reports_dir / "rank_table.csv")
        )

        self.log(f"보고서 저장 완료: {reports_dir}")
        return {"summary_md": str(summary_path), "rank_csv": csv_path}

    def create_comparison_panel(
        self,
        candidates: list[Candidate],
        receptor_pdb: str,
        output_path: str,
    ) -> str:
        """복수 후보의 비교 PyMOL 스크립트를 생성하고 경로를 반환한다.

        모든 후보를 단일 세션에 로드하여 receptor에 align 후
        각자 다른 색으로 표시하는 비교 패널을 생성한다.

        Args:
            candidates: 비교할 Candidate 목록
            receptor_pdb: 수용체 PDB 경로
            output_path: 저장 경로 (.pml)

        Returns:
            저장된 스크립트 경로
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        colors = [
            "blue", "red", "green", "yellow", "magenta",
            "cyan", "orange", "violet", "salmon", "lime",
        ]
        lines = [
            "# SSTR2 Peptide Binder Comparison Panel",
            "reinitialize",
            f"load {receptor_pdb}, receptor",
            "color gray80, receptor",
            "show cartoon, receptor",
            "",
        ]

        for i, cand in enumerate(candidates):
            color = colors[i % len(colors)]
            obj_name = f"pep_{cand.candidate_id}"
            lines += [
                f"load {cand.pdb_path}, {obj_name}",
                f"color {color}, {obj_name}",
                f"show cartoon, {obj_name}",
                f"align {obj_name}, receptor",
            ]

        lines += [
            "",
            "# 전체 뷰 최적화",
            "zoom all",
            "ray 1200, 900",
            f"png {out.parent / 'comparison_panel.png'}, dpi=150",
            "save comparison_panel.pse",
        ]

        script = "\n".join(lines)
        out.write_text(script, encoding="utf-8")
        self.log(f"비교 패널 스크립트 저장: {out}")
        return str(out)

    # ------------------------------------------------------------------
    # Abstract method implementation
    # ------------------------------------------------------------------

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """컨텍스트에서 필요한 데이터를 받아 리포트 일체를 생성한다.

        context 키:
            - run_id (str)
            - iteration (int)
            - rank_table (RankTable)
            - top_candidates (list[Candidate])
            - receptor_pdb (str)
            - output_dir (str)
            - critic_analysis (CriticAnalysis, optional)
            - all_iterations (list[dict], optional): lab notebook용

        Returns:
            {'status': str, 'report_paths': dict, 'render_paths': dict}
        """
        run_id: str = context["run_id"]
        iteration: int = context.get("iteration", 1)
        rank_table: RankTable = context["rank_table"]
        top_candidates: list[Candidate] = context.get("top_candidates", [])
        receptor_pdb: str = context.get("receptor_pdb", "")
        output_dir: str = context.get("output_dir", str(self.runs_base_dir / run_id))
        critic_analysis: Optional[CriticAnalysis] = context.get("critic_analysis")

        report_paths = self.save_all_reports(
            run_id, iteration, rank_table, critic_analysis, output_dir
        )

        render_paths: dict[str, RenderPaths] = {}
        if top_candidates and receptor_pdb:
            render_paths = self.generate_pymol_renders(
                top_candidates, receptor_pdb, output_dir
            )

        if all_iterations := context.get("all_iterations"):
            notebook_md = self.generate_lab_notebook(run_id, all_iterations)
            nb_path = Path(output_dir) / "08_reports" / "lab_notebook.md"
            nb_path.write_text(notebook_md, encoding="utf-8")
            report_paths["lab_notebook"] = str(nb_path)

        self.log("리포트 생성 완료")
        return {
            "status": "ok",
            "report_paths": report_paths,
            "render_paths": {k: vars(v) for k, v in render_paths.items()},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _summarize_via_llm(
        self,
        run_id: str,
        iteration: int,
        rank_table: RankTable,
        critic_analysis: Optional[CriticAnalysis] = None,
    ) -> Optional[str]:
        """LLM을 통해 Markdown 요약 보고서를 생성한다. 실패 시 None 반환."""
        try:
            top_cands = [
                {
                    "id": c.candidate_id,
                    "plddt": round(c.plddt_mean, 1),
                    "dock_score": round(c.dock_score, 2),
                    "ddg": round(c.ddg, 1),
                }
                for c in rank_table.ranked_candidates[:10]
            ]
            rank_summary = {
                "total": len(rank_table.ranked_candidates),
                "passed": len(rank_table.ranked_candidates),
                "top_candidates": top_cands,
            }
            critic_dict = None
            if critic_analysis:
                critic_dict = {
                    "overall_assessment": critic_analysis.structural_insights,
                    "primary_failure_type": next(
                        iter(critic_analysis.failure_summary), "none",
                    ),
                }

            prompt = format_reporter_prompt(
                iteration=iteration,
                run_id=run_id,
                rank_table_summary=rank_summary,
                critic_analysis=critic_dict,
            )
            system = get_system_prompt("reporter")
            result = self.llm_generate_json(prompt, system_prompt=system)
            if result is None:
                return None

            # JSON 결과를 Markdown 보고서로 변환
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
            title = result.get("title", f"Iteration {iteration} Summary")
            summary = result.get("summary", "")
            recs = result.get("recommendations", [])
            recs_md = "\n".join(f"- {r}" for r in recs) if recs else "- 없음"

            report = textwrap.dedent(f"""
            # {title}

            - **Run ID**: `{run_id}`
            - **Iteration**: {iteration}
            - **생성 시각**: {now}
            - **생성 방식**: LLM (Qwen 2.5 7B)

            ## Summary

            {summary}

            ## Recommendations

            {recs_md}
            """).strip()

            return report
        except Exception as exc:
            self.log(f"LLM 요약 예외: {exc}", level="error")
            return None

    def _build_pymol_script(
        self,
        candidate: Candidate,
        receptor_pdb: str,
        output_dir: str,
        pse_path: str,
    ) -> str:
        """단일 후보에 대한 PyMOL 4-panel 렌더 스크립트를 생성한다."""
        cid = candidate.candidate_id
        out = Path(output_dir)

        script = textwrap.dedent(f"""
        # PyMOL 4-Panel Auto-Render: {cid}
        # 생성: {datetime.utcnow().isoformat()}
        reinitialize

        # --- 구조 로드 ---
        load {receptor_pdb}, receptor
        load {candidate.pdb_path}, peptide

        # 기본 스타일
        bg_color white
        set ray_opaque_background, off
        set antialias, 2
        set ray_trace_mode, 1

        # 수용체: 회색 cartoon
        color gray70, receptor
        show cartoon, receptor
        hide lines, receptor

        # 펩타이드: 청색 cartoon + stick
        color marine, peptide
        show cartoon, peptide
        show sticks, peptide
        set stick_radius, 0.15

        # 수용체-펩타이드 정렬
        align peptide, receptor

        # ================================================================
        # Panel 1: Overview (전체 복합체 cartoon)
        # ================================================================
        zoom all
        set_view [\\
            1.0, 0.0, 0.0, \\
            0.0, 1.0, 0.0, \\
            0.0, 0.0, 1.0, \\
            0.0, 0.0, -80.0, \\
            0.0, 0.0, 0.0, 40.0, 200.0, -20.0]
        ray 1200, 900
        png {out}/{cid}_1_overview.png, dpi=150

        # ================================================================
        # Panel 2: Closeup (결합 포켓 zoom)
        # ================================================================
        hide everything
        show cartoon, receptor
        show sticks, peptide
        show cartoon, peptide
        # 펩타이드 주변 8A 이내 수용체 residue 표시
        select pocket_res, receptor within 8 of peptide
        show sticks, pocket_res
        color yellow, pocket_res
        zoom peptide, 8
        ray 1200, 900
        png {out}/{cid}_2_closeup.png, dpi=150

        # ================================================================
        # Panel 3: Interface contacts (H-bond, salt bridge, hydrophobic)
        # ================================================================
        hide everything
        show cartoon, receptor
        show cartoon, peptide
        show sticks, peptide
        show sticks, pocket_res
        # H-bonds
        distance hbonds, receptor, peptide, 3.5, mode=2
        color yellow, hbonds
        # salt bridges
        distance salt_bridges, receptor, peptide, 5.0, mode=1
        color red, salt_bridges
        set dash_width, 3.0
        set label_size, 12
        zoom peptide, 10
        ray 1200, 900
        png {out}/{cid}_3_interface.png, dpi=150

        # ================================================================
        # Panel 4: Electrostatics (APBS surface)
        # ================================================================
        hide everything
        show surface, receptor
        show cartoon, peptide
        show sticks, peptide
        # APBS 정전기 포텐셜 계산 (PyMOL APBS 플러그인 필요)
        # apbs_run receptor
        # ramp_new e_lvl, apbs_map, [-5, 0, 5], [red, white, blue]
        # set surface_color, e_lvl, receptor
        # --- APBS 미설치 환경 대안: 소수성 표면 ---
        spectrum count, blue_white_red, receptor, minimum=0, maximum=10
        set transparency, 0.3, receptor
        zoom peptide, 12
        ray 1200, 900
        png {out}/{cid}_4_electrostatics.png, dpi=150

        # ================================================================
        # 세션 저장
        # ================================================================
        save {pse_path}
        """).strip()

        return script

    @staticmethod
    def _dict_to_md_list(d: dict[str, Any]) -> str:
        """딕셔너리를 Markdown 불릿 리스트로 변환한다."""
        if not d:
            return "- 없음"
        return "\n".join(f"- {k}: {v}" for k, v in d.items())
