#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyrosetta_flow import FlowConfig, run_pyrosetta_agentic_mutdock_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run agentic SST14 mutate->dock PyRosetta optimization flow."
    )
    parser.add_argument("--input", required=True, help="Template receptor-peptide complex PDB path")
    parser.add_argument("--n-candidates", type=int, default=8, help="Number of mutation candidates per iteration")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base random seed")
    parser.add_argument("--conda-env", default="bio-tools", help="Conda env with PyRosetta installed")
    parser.add_argument(
        "--output-json",
        default="runs/pyrosetta_flow/pyrosetta_flow_artifacts.json",
        help="Artifact JSON output path",
    )
    parser.add_argument(
        "--peptide-chain",
        type=int,
        default=1,
        help="Peptide chain number (1-indexed). Chain A=1 in fold_test1_model_0.pdb",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Agentic loop iteration count",
    )
    parser.add_argument(
        "--objective-mode",
        choices=["auto", "ddg_only", "ddg_plus_constraints"],
        default="auto",
        help="Objective selection mode for planner loop",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k candidates passed to critic/reporter per iteration",
    )
    parser.add_argument(
        "--rosetta-ddg-max",
        type=float,
        default=-5.0,
        help="ddG gate threshold when constraint mode is active",
    )
    parser.add_argument(
        "--rosetta-clash-max",
        type=int,
        default=10,
        help="clash gate threshold when constraint mode is active",
    )
    parser.add_argument(
        "--planner-mode",
        choices=["default", "pyrosetta-only", "pyrosetta_only"],
        default="pyrosetta-only",
        help="Planner prompt mode for this run",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel FlexPepDock processes per iteration",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM model name (e.g. qwen3:8b). Priority: CLI > LLM_MODEL env > config file",
    )
    parser.add_argument(
        "--enable-selectivity", action="store_true",
        help="최종 top-K 후보에 off-target(SSTR1/3/4/5) 선택성 도킹 수행 (비쌈, 기본 OFF)",
    )
    parser.add_argument(
        "--selectivity-top-k", type=int, default=3,
        help="최종(post-loop) 선택성 도킹 대상 후보 수 (기본 3)",
    )
    parser.add_argument(
        "--inloop-selectivity", action="store_true",
        help="매 iteration 유망(ddG 강한) 후보만 off-target 도킹→Δmargin→Planner/Critic 피드백 (조건부 게이트)",
    )
    parser.add_argument(
        "--selectivity-max-per-iter", type=int, default=2,
        help="in-loop 선택성: iteration 당 최대 도킹 후보 수 (비용 제어, 기본 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FlowConfig(
        template_pdb=args.input,
        n_candidates=args.n_candidates,
        seed_base=args.seed_base,
        conda_env=args.conda_env,
        peptide_chain=args.peptide_chain,
        max_iterations=args.max_iterations,
        objective_mode=args.objective_mode,
        top_k=args.top_k,
        rosetta_ddg_max=args.rosetta_ddg_max,
        rosetta_clash_max=args.rosetta_clash_max,
        planner_mode="pyrosetta_only" if args.planner_mode in {"pyrosetta-only", "pyrosetta_only"} else "default",
        max_parallel_workers=args.max_workers,
        llm_model_override=args.llm_model,
        enable_selectivity=args.enable_selectivity,
        selectivity_top_k=args.selectivity_top_k,
        inloop_selectivity=args.inloop_selectivity,
        selectivity_max_per_iter=args.selectivity_max_per_iter,
    )
    artifacts = run_pyrosetta_agentic_mutdock_flow(cfg)
    artifacts.write_json(args.output_json)
    print(f"[pyrosetta_flow] done -> {args.output_json}")


if __name__ == "__main__":
    main()
