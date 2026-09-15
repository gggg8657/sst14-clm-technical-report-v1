#!/usr/bin/env python3
"""무한 발굴 엔진 CLI — STOP 파일이 생길 때까지 epoch 를 반복한다.

사용:
  # 무한 (STOP 파일로 정지)
  python scripts/run_continuous_discovery.py --input <complex.pdb>
  # 정지: 다른 터미널에서  touch _workspace/STOP_DISCOVERY
  # 조절: _workspace/discovery_control.json 편집 (다음 epoch부터 반영)

  # 테스트(2 epoch 만)
  python scripts/run_continuous_discovery.py --input <complex.pdb> --max-epochs 2 \
      --n-candidates 6 --max-iterations 2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyrosetta_flow import FlowConfig
from pyrosetta_flow.continuous import run_continuous_discovery


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SST14 무한 선택성 발굴 엔진 (continuous discovery)")
    p.add_argument("--input", required=True, help="Template receptor-peptide complex PDB")
    p.add_argument("--conda-env", default="bio-tools")
    p.add_argument("--peptide-chain", type=int, default=1)
    p.add_argument("--planner-mode", default="pyrosetta-only",
                   choices=["default", "pyrosetta-only", "pyrosetta_only"])
    p.add_argument("--max-workers", type=int, default=64)
    p.add_argument("--llm-model", default=None)
    # epoch(=단발 run) 기본 규모
    p.add_argument("--n-candidates", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=4, help="epoch 당 agentic iteration 수")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--objective-mode", default="auto",
                   choices=["auto", "ddg_only", "ddg_plus_constraints"])
    p.add_argument("--selectivity-max-per-iter", type=int, default=2)
    p.add_argument("--selectivity-top-k", type=int, default=3)
    p.add_argument("--enable-selectivity", action="store_true",
                   help="epoch 종료 시 post-loop top-K 선택성 도킹도 수행 (in-loop 와 별개, 비쌈)")
    # 무한 루프 제어
    p.add_argument("--max-epochs", type=int, default=None,
                   help="None(기본)이면 STOP 파일까지 무한. 정수면 그만큼만.")
    p.add_argument("--control-file", default="_workspace/discovery_control.json",
                   help="사람이 조절하는 knobs JSON (매 epoch 재로드)")
    p.add_argument("--stop-file", default="_workspace/STOP_DISCOVERY",
                   help="이 파일이 생기면 graceful 종료")
    p.add_argument("--status-file", default="runs/pyrosetta_flow/discovery_status.json",
                   help="진행 상황 기록 파일")
    p.add_argument("--epoch-pause", type=float, default=0.0, help="epoch 간 대기(초)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base = FlowConfig(
        template_pdb=args.input,
        n_candidates=args.n_candidates,
        conda_env=args.conda_env,
        peptide_chain=args.peptide_chain,
        max_iterations=args.max_iterations,
        objective_mode=args.objective_mode,
        top_k=args.top_k,
        planner_mode="pyrosetta_only" if args.planner_mode in {"pyrosetta-only", "pyrosetta_only"} else "default",
        max_parallel_workers=args.max_workers,
        llm_model_override=args.llm_model,
        enable_selectivity=args.enable_selectivity,
        selectivity_top_k=args.selectivity_top_k,
        inloop_selectivity=True,                      # 무한 엔진은 in-loop 선택성 필수
        selectivity_max_per_iter=args.selectivity_max_per_iter,
        # A/B 격리: PYROSETTA_OUTPUT_DIR 로 출력 dir 분리(experiment_log·리더보드).
        # 미지정 시 기존 기본값 "runs/pyrosetta_flow" 100% 보존.
        output_dir=os.environ.get("PYROSETTA_OUTPUT_DIR", "runs/pyrosetta_flow"),
    )

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (repo_root / pp)

    result = run_continuous_discovery(
        base_config=base,
        repo_root=repo_root,
        control_path=_resolve(args.control_file),
        stop_path=_resolve(args.stop_file),
        status_path=_resolve(args.status_file),
        max_epochs=args.max_epochs,
        epoch_pause_s=args.epoch_pause,
    )
    print(f"[continuous] DONE: {result['stop_reason']}, epochs={result['epochs_done']}, "
          f"best Δmargin={result['global_best_delta_margin']}, passing={result['passing_count']}")


if __name__ == "__main__":
    main()
