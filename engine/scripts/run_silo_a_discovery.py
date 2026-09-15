#!/usr/bin/env python3
"""Silo A 무한 de novo 발굴 엔진 CLI.

RFdiffusion → ProteinMPNN → ESMFold pLDDT 검증 → FlexPepDock 도킹 → 스코어링을
epoch 루프로 반복한다. STOP 파일이 생길 때까지(또는 --max-epochs 도달까지) 무한 실행.

피드백 루프 (silo_a_planner):
  epoch 완료 후 리더보드를 읽어 다음 epoch 생성 파라미터를 조정한다.
  LLM(vLLM GPU2) 경로 → 실패 시 규칙 기반 fallback.
  조정 내역은 runs/silo_a_flow/silo_a_planner_e{NN}.json에 기록(provenance).

사용:
  # GPU3 전용, 무한 루프 (피드백 루프 활성)
  CUDA_VISIBLE_DEVICES=3 python scripts/run_silo_a_discovery.py \\
      --receptor-pdb data/somatostatin_receptor/curated/SSTR2_receptor.pdb

  # 피드백 루프 비활성 (기존 동작)
  CUDA_VISIBLE_DEVICES=3 python scripts/run_silo_a_discovery.py \\
      --receptor-pdb data/somatostatin_receptor/curated/SSTR2_receptor.pdb \\
      --no-feedback

  # 스모크 테스트 (1 epoch, backbone=1, seq=2, steps=20)
  CUDA_VISIBLE_DEVICES=3 python scripts/run_silo_a_discovery.py \\
      --receptor-pdb data/somatostatin_receptor/curated/SSTR2_receptor.pdb \\
      --max-epochs 1 --n-backbone 1 --k-seq 2 --diffusion-steps 20

  # 정지: touch runs/silo_a_flow/STOP_SILO_A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# GPU3 전용 강제 (기동 스크립트 레벨에서)
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print("[SiloA] CUDA_VISIBLE_DEVICES=3 (default 설정)", file=sys.stderr)
else:
    print(f"[SiloA] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (환경변수)", file=sys.stderr)

# repo root sys.path 추가
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from pyrosetta_flow.silo_a_flow import (
    SiloAConfig,
    SiloALeaderboard,
    append_silo_a_records,
    generate_and_score_silo_a,
)
from pyrosetta_flow.silo_a_planner import (
    plan_next_epoch,
    update_stagnation_count,
)


# ---------------------------------------------------------------------------
# 인자 파싱
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silo A de novo 무한 발굴 엔진 (GPU3 전용, Silo B 완전 분리)"
    )
    p.add_argument("--receptor-pdb", required=True,
                   help="SSTR2 수용체 PDB 경로 (chain B로 자동 변환)")
    # 생성 파라미터
    p.add_argument("--n-backbone", type=int, default=2, help="epoch당 RFdiffusion 백본 수")
    p.add_argument("--k-seq", type=int, default=2, help="백본당 ProteinMPNN 서열 수")
    p.add_argument("--diffusion-steps", type=int, default=50)
    p.add_argument("--contigs", default="B40-187/0 B189-327/0 12-16",
                   help="RFdiffusion contig 문자열. SiloA wrapper가 수용체를 chain B로 변환한 "
                        "PDB(receptor_chainB.pdb)를 RFdiffusion에 넘기므로 chain은 B. 잔기는 40~327이며 "
                        "188번 결손(GPCR ICL3 gap)이므로 'B40-187/0 B189-327' 두 세그먼트로 끊고 "
                        "'/0' chain break 뒤 바인더 길이(12-16). 과거 'B1-369/472'는 잔기 1이 PDB에 "
                        "없어('B1 not in pdb') 백본 생성 100% 실패했다.")
    # L3-b 수정: 기본 hotspot을 SSTR2 포켓 잔기로 교체.
    # 임의 잔기(B150/B200/B250 등) → 일관 탐색 불가. 포켓 잔기로 고정.
    # ECL2(B192/197) + TM5(B205/209) + TM6(B272/279) + ECL3(B284) 대표값.
    # --hotspot-res B192 B197 ... 로 오버라이드 가능.
    p.add_argument("--hotspot-res", nargs="*",
                   default=["B192", "B197", "B205", "B209", "B272", "B284"],
                   help="핫스팟 잔기 목록 — SSTR2 포켓 잔기만 사용 (ECL2/TM5/TM6/ECL3)")
    # 환경 파라미터
    p.add_argument("--conda-env", default="bio-tools", help="FlexPepDock conda env")
    p.add_argument("--rfdiffusion-env", default="rfdiffusion")
    p.add_argument("--esmfold-env", default="esmfold")
    p.add_argument("--proteinmpnn-env", default="proteinmpnn")
    # 스코어링 파라미터
    p.add_argument("--plddt-threshold", type=float, default=0.5, help="ESMFold pLDDT 게이트")
    p.add_argument("--no-selectivity", action="store_true", help="선택성 도킹 비활성화")
    p.add_argument("--max-selectivity-per-epoch", type=int, default=2)
    p.add_argument("--selectivity-timeout", type=int, default=900)
    p.add_argument("--script-timeout", type=int, default=900, help="FlexPepDock 타임아웃(초)")
    # 루프 제어
    p.add_argument("--max-epochs", type=int, default=None,
                   help="최대 epoch 수 (None=STOP 파일까지 무한)")
    p.add_argument("--epoch-pause", type=float, default=5.0, help="epoch 간 대기(초)")
    p.add_argument("--stop-file", default="runs/silo_a_flow/STOP_SILO_A",
                   help="이 파일이 생기면 graceful 종료")
    p.add_argument("--status-file", default="runs/silo_a_flow/discovery_status.json",
                   help="진행 상황 JSON 파일")
    p.add_argument("--leaderboard-file", default="runs/silo_a_flow/silo_a_leaderboard.json")
    p.add_argument("--log-file", default="runs/silo_a_flow/experiment_log.jsonl")
    p.add_argument("--output-base-dir", default="runs/silo_a_flow")
    # 피드백 루프 제어
    p.add_argument("--no-feedback", action="store_true",
                   help="피드백 루프 비활성화 (매 epoch 동일 파라미터 사용, 기존 동작)")
    p.add_argument("--feedback-patience", type=int, default=50,
                   help="이 epoch 수 이상 정체 시 파라미터 조정 트리거 (기본 50). "
                        "사다리 단계: 1×→steps50, 2×→100, 4×→150, 8×→200")
    p.add_argument("--vllm-url", default="http://localhost:8000",
                   help="LLM Planner용 vLLM 서버 URL (기본 GPU2 http://localhost:8000)")
    # L3-a 수정: vLLM 8000 서빙명으로 교체 (HuggingFace 경로 "Qwen/Qwen3-32B" → 404).
    # curl 검증: curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v1/chat/completions
    #   -d '{"model":"qwen3-32b",...}' → 200 (72B로 라우팅됨)
    p.add_argument("--vllm-model", default="qwen3-32b",
                   help="LLM Planner 모델명 (vLLM 서빙명, 기본 'qwen3-32b' = Qwen2.5-72B-Instruct)")
    p.add_argument("--vllm-timeout", type=int, default=60,
                   help="LLM 호출 타임아웃(초)")
    # de novo 맞춤 사전검토 (prereview)
    p.add_argument("--no-silo-a-prereview", action="store_true",
                   help="Silo A 사전검토 비활성화 (기본 활성 — prereview=True). "
                        "0=off 하위호환: --no-silo-a-prereview로 비활성.")
    # DiffPepBuilder arm 파라미터
    p.add_argument("--arm", choices=["rfdiffusion", "diffpep", "both"], default="rfdiffusion",
                   help="생성 arm 선택: rfdiffusion=RFdiffusion only(GPU3), "
                        "diffpep=DiffPepBuilder only(GPU2), both=두 arm 병렬(기본 rfdiffusion)")
    p.add_argument("--diffpep-cuda", default="2",
                   help="DiffPepBuilder arm GPU CUDA_VISIBLE_DEVICES 값 (기본 2)")
    p.add_argument("--diffpep-n-samples", type=int, default=3,
                   help="DiffPepBuilder arm epoch당 생성 서열 수 (기본 3)")
    p.add_argument("--diffpep-min-length", type=int, default=10,
                   help="DiffPepBuilder arm 최소 펩타이드 길이 (기본 10)")
    p.add_argument("--diffpep-max-length", type=int, default=14,
                   help="DiffPepBuilder arm 최대 펩타이드 길이 (기본 14)")
    p.add_argument("--diffpep-num-t", type=int, default=50,
                   help="DiffPepBuilder arm diffusion 스텝 수 (기본 50)")
    p.add_argument("--diffpep-timeout", type=int, default=600,
                   help="DiffPepBuilder arm 서브프로세스 타임아웃(초) (기본 600)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 파일 경로 resolve 헬퍼
# ---------------------------------------------------------------------------

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


# ---------------------------------------------------------------------------
# status 파일 기록
# ---------------------------------------------------------------------------

def _write_status(
    status_path: Path,
    epoch: int,
    total_candidates: int,
    best_ddg: float,
    best_sel_margin: float,
    stop_reason: str,
    t_start: float,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    import math
    payload = {
        "source": "silo_a",
        "candidate_class": "de_novo",
        "epoch": epoch,
        "total_candidates": total_candidates,
        "best_ddg": None if (math.isinf(best_ddg) or math.isnan(best_ddg)) else best_ddg,
        "best_selectivity_margin": None if (math.isinf(best_sel_margin) or math.isnan(best_sel_margin)) else best_sel_margin,
        "stop_reason": stop_reason,
        "elapsed_sec": round(time.time() - t_start, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
    }
    tmp = Path(str(status_path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(status_path)


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 경로 resolve
    receptor_pdb = _resolve(args.receptor_pdb)
    stop_path = _resolve(args.stop_file)
    status_path = _resolve(args.status_file)
    leaderboard_path = _resolve(args.leaderboard_file)
    log_path = _resolve(args.log_file)

    if not receptor_pdb.exists():
        print(f"[SiloA] 수용체 PDB 없음: {receptor_pdb}", file=sys.stderr)
        sys.exit(1)

    # arm 선택 로직
    arm = getattr(args, "arm", "rfdiffusion")
    diffpep_arm_enabled = arm in ("diffpep", "both")
    rfdiffusion_arm_enabled = arm in ("rfdiffusion", "both")

    # RFdiffusion arm은 wrapper가 수용체를 chain B로 변환한 PDB를 사용하므로 hotspot도
    # chain B(args.hotspot_res) 그대로 전달한다. DiffPepBuilder arm은 원본(chain A) 수용체를
    # 쓰므로 B→A로 변환해 넘긴다. (RFdiffusion 백본 실패의 원인은 chain이 아니라 contig 잔기
    # 범위/gap이었음 — contigs 기본값 참조.)
    diffpep_hotspot_res = [
        r.replace("B", "A", 1) if r.startswith("B") else r
        for r in args.hotspot_res
    ]

    # SiloAConfig 구성
    base_config = SiloAConfig(
        receptor_pdb=str(receptor_pdb),
        run_id="silo_a",
        n_backbone=args.n_backbone if rfdiffusion_arm_enabled else 0,
        k_seq_per_backbone=args.k_seq,
        diffusion_steps=args.diffusion_steps,
        contigs=args.contigs,
        hotspot_res=args.hotspot_res,
        plddt_threshold=args.plddt_threshold,
        conda_env=args.conda_env,
        rfdiffusion_env=args.rfdiffusion_env,
        esmfold_env=args.esmfold_env,
        proteinmpnn_env=args.proteinmpnn_env,
        output_base_dir=args.output_base_dir,
        device="cuda:0",                   # CUDA_VISIBLE_DEVICES=3 → 내부 cuda:0
        script_timeout=args.script_timeout,
        selectivity_enabled=not args.no_selectivity,
        selectivity_timeout=args.selectivity_timeout,
        max_selectivity_per_epoch=args.max_selectivity_per_epoch,
        # DiffPepBuilder arm 설정
        diffpep_arm_enabled=diffpep_arm_enabled,
        diffpep_cuda_device=getattr(args, "diffpep_cuda", "2"),
        diffpep_n_samples=getattr(args, "diffpep_n_samples", 3),
        diffpep_min_length=getattr(args, "diffpep_min_length", 10),
        diffpep_max_length=getattr(args, "diffpep_max_length", 14),
        diffpep_num_t=getattr(args, "diffpep_num_t", 50),
        diffpep_hotspot_res=diffpep_hotspot_res,
        diffpep_timeout=getattr(args, "diffpep_timeout", 600),
    )

    print(
        f"[SiloA] arm={arm}: "
        f"RFdiffusion={'활성(GPU3)' if rfdiffusion_arm_enabled else '비활성'} | "
        f"DiffPepBuilder={'활성(GPU' + getattr(args, 'diffpep_cuda', '2') + ')' if diffpep_arm_enabled else '비활성'}",
        file=sys.stderr,
    )

    # 리더보드 로드 (run 간 누적)
    leaderboard = SiloALeaderboard.load(leaderboard_path)
    t_start = time.time()
    epoch = 0
    total_candidates = leaderboard.n_total
    stop_reason = "max_epochs"

    # 피드백 루프 상태
    feedback_enabled = not args.no_feedback
    stagnation_count = 0
    prev_best_ddg: float = leaderboard._best_ddg() or float("inf")
    # provenance 저장 디렉토리 (leaderboard_path와 같은 디렉토리)
    planner_output_dir = leaderboard_path.parent

    # 현재 파라미터 추적 (피드백 루프가 epoch마다 업데이트)
    current_params: dict = {
        "contigs": base_config.contigs,
        "hotspot_res": list(base_config.hotspot_res),
        "diffusion_steps": base_config.diffusion_steps,
        "n_backbone": base_config.n_backbone,
        "k_seq_per_backbone": base_config.k_seq_per_backbone,
    }

    print(f"[SiloA] 발굴 엔진 시작 (GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}, "
          f"receptor={receptor_pdb.name}, "
          f"n_backbone={args.n_backbone}, k_seq={args.k_seq}, steps={args.diffusion_steps})",
          file=sys.stderr)
    print(f"[SiloA] 이전 누적: {len(leaderboard.entries)}건 리더보드, {leaderboard.n_total}건 총 처리",
          file=sys.stderr)
    print(f"[SiloA] 피드백 루프: {'활성 (patience=' + str(args.feedback_patience) + ')' if feedback_enabled else '비활성 (--no-feedback)'}",
          file=sys.stderr)
    _prereview_on = not getattr(args, "no_silo_a_prereview", False)
    print(f"[SiloA] 사전검토(prereview): {'활성 (구조+다양성 2-관점)' if _prereview_on else '비활성 (--no-silo-a-prereview)'}",
          file=sys.stderr)

    while True:
        # STOP 파일 체크
        if stop_path.exists():
            stop_reason = "stop_file"
            print(f"[SiloA] STOP 파일 감지({stop_path}) — graceful 종료", file=sys.stderr)
            break
        # max_epochs 체크
        if args.max_epochs is not None and epoch >= args.max_epochs:
            stop_reason = "max_epochs"
            break

        epoch += 1

        # ----------------------------------------------------------------
        # 피드백 루프: 직전 리더보드 기반 파라미터 조정 (epoch 2부터)
        # ----------------------------------------------------------------
        if feedback_enabled and epoch > 1:
            try:
                # prereview 설정: --no-silo-a-prereview 플래그로 비활성화
                _prereview_enabled = not getattr(args, "no_silo_a_prereview", False)
                # discussion_log 경로: leaderboard와 같은 디렉토리
                _discussion_log = leaderboard_path.parent / "silo_a_discussion_log.jsonl"
                updated_params, decision = plan_next_epoch(
                    epoch=epoch,
                    leaderboard_path=leaderboard_path,
                    current_params=current_params,
                    stagnation_count=stagnation_count,
                    patience_epochs=args.feedback_patience,
                    vllm_url=args.vllm_url,
                    vllm_model=args.vllm_model,
                    vllm_timeout=args.vllm_timeout,
                    output_dir=planner_output_dir,
                    # Silo A experiment_log 경로 — 궤적 주입용 (silo_a_flow 전용)
                    experiment_log_path=log_path,
                    prereview_enabled=_prereview_enabled,
                    discussion_log_path=_discussion_log,
                )
                current_params = updated_params
                print(
                    f"[SiloA] 피드백 플래너(epoch={epoch}, stagnation={stagnation_count}): "
                    f"[{decision.adjustment_type}] {decision.hypothesis[:80]}",
                    file=sys.stderr,
                )
            except Exception as exc:
                print(f"[SiloA] 피드백 플래너 예외 — 현재 파라미터 유지: {exc}", file=sys.stderr)

        print(f"\n[SiloA] === Epoch {epoch} 시작 (contigs='{current_params['contigs']}', "
              f"steps={current_params['diffusion_steps']}) ===", file=sys.stderr)

        # SiloAConfig를 현재 파라미터로 구성
        epoch_config = SiloAConfig(
            receptor_pdb=base_config.receptor_pdb,
            run_id=f"silo_a_e{epoch:04d}",
            n_backbone=current_params.get("n_backbone", base_config.n_backbone),
            k_seq_per_backbone=current_params.get("k_seq_per_backbone", base_config.k_seq_per_backbone),
            diffusion_steps=current_params.get("diffusion_steps", base_config.diffusion_steps),
            contigs=current_params.get("contigs", base_config.contigs),
            hotspot_res=list(current_params.get("hotspot_res", base_config.hotspot_res)),
            plddt_threshold=base_config.plddt_threshold,
            conda_env=base_config.conda_env,
            rfdiffusion_env=base_config.rfdiffusion_env,
            esmfold_env=base_config.esmfold_env,
            proteinmpnn_env=base_config.proteinmpnn_env,
            output_base_dir=base_config.output_base_dir,
            device=base_config.device,
            script_timeout=base_config.script_timeout,
            selectivity_enabled=base_config.selectivity_enabled,
            selectivity_timeout=base_config.selectivity_timeout,
            max_selectivity_per_epoch=base_config.max_selectivity_per_epoch,
            # DiffPepBuilder arm 설정 (base_config에서 그대로 전달)
            diffpep_arm_enabled=base_config.diffpep_arm_enabled,
            diffpep_cuda_device=base_config.diffpep_cuda_device,
            diffpep_n_samples=base_config.diffpep_n_samples,
            diffpep_min_length=base_config.diffpep_min_length,
            diffpep_max_length=base_config.diffpep_max_length,
            diffpep_num_t=base_config.diffpep_num_t,
            diffpep_hotspot_res=base_config.diffpep_hotspot_res,
            diffpep_timeout=base_config.diffpep_timeout,
        )

        try:
            results = generate_and_score_silo_a(epoch_config)
        except Exception as exc:
            print(f"[SiloA] Epoch {epoch} 예외: {exc}", file=sys.stderr)
            _write_status(status_path, epoch, total_candidates,
                          leaderboard._best_ddg() or float("inf"),
                          leaderboard._best_sel_margin() or float("-inf"),
                          f"epoch_error:{exc}", t_start)
            # 에러가 있어도 루프 계속 (환경 문제일 수 있음)
            if args.epoch_pause > 0:
                time.sleep(args.epoch_pause)
            continue

        if not results:
            print(f"[SiloA] Epoch {epoch}: 결과 없음 — 다음 epoch으로", file=sys.stderr)
        else:
            # SILO_A_PLDDT_FAILCLOSED=1(기본): plddt_pass=False 후보는 리더보드 제외
            # (measurement_missing 포함 — ESMFold 미작동 후보의 리더보드 오염 방지)
            _failclosed = os.environ.get('SILO_A_PLDDT_FAILCLOSED', '1') != '0'

            # 리더보드 + log 업데이트
            for r in results:
                # fail-closed 모드: plddt_pass=False인 후보는 리더보드 진입 금지
                if _failclosed and not r.plddt_pass:
                    _status = (r.extra_scores or {}).get("plddt_status", "gate_fail")
                    print(
                        f"[SiloA] Epoch {epoch}: 리더보드 제외 "
                        f"(seq={r.sequence[:8]}..., fail_reason={r.fail_reason}, plddt_status={_status})",
                        file=sys.stderr,
                    )
                    continue

                # mutation_source 태그: DiffPepBuilder arm은 'silo_a_diffpep' 유지
                # RFdiffusion arm은 피드백 경로 구분
                if r.mutation_source != "silo_a_diffpep":
                    if feedback_enabled and epoch > 1:
                        r.mutation_source = "silo_a_llm_guided" if (
                            # planner 결정 기록이 있으면 태그 분기
                            "decision" in dir() and decision.adjustment_type == "llm_guided"  # type: ignore[possibly-undefined]
                        ) else "silo_a_random"
                leaderboard.add(r)
                total_candidates += 1
            append_silo_a_records(log_path, results, epoch, epoch_config.run_id)
            leaderboard.save(leaderboard_path)

            n_ok = sum(1 for r in results if r.ddg is not None)
            n_sel = sum(1 for r in results if r.selectivity_margin is not None)
            best_ddg_now = leaderboard._best_ddg()
            best_sel_now = leaderboard._best_sel_margin()
            print(
                f"[SiloA] Epoch {epoch}: {len(results)}건 시도, 도킹={n_ok}, 선택성={n_sel} | "
                f"글로벌 best ddG={best_ddg_now}, sel_margin={best_sel_now}",
                file=sys.stderr,
            )

            # 피드백 루프: stagnation_count 업데이트
            if feedback_enabled:
                new_best_ddg = leaderboard._best_ddg()
                stagnation_count, improved = update_stagnation_count(
                    prev_best_ddg=prev_best_ddg if prev_best_ddg != float("inf") else None,
                    current_best_ddg=new_best_ddg,
                    prev_stagnation_count=stagnation_count,
                )
                prev_best_ddg = new_best_ddg if new_best_ddg is not None else float("inf")
                if improved:
                    print(f"[SiloA] 피드백: best ddG 개선 → stagnation 리셋", file=sys.stderr)
                else:
                    # 사다리 단계 계산하여 출력 (탈출 시점 가시화)
                    from pyrosetta_flow.silo_a_planner import _get_ladder_steps
                    _ladder = _get_ladder_steps(stagnation_count, args.feedback_patience)
                    _ratio = (stagnation_count // args.feedback_patience
                              if args.feedback_patience > 0 else 0)
                    _ladder_info = (
                        f" [사다리 {_ratio}×→steps={_ladder}]" if _ladder is not None
                        else " [사다리 미발동]"
                    )
                    print(
                        f"[SiloA] 피드백: 정체 ({stagnation_count}/{args.feedback_patience})"
                        f"{_ladder_info}",
                        file=sys.stderr,
                    )

        _write_status(
            status_path, epoch, total_candidates,
            leaderboard._best_ddg() or float("inf"),
            leaderboard._best_sel_margin() or float("-inf"),
            "running", t_start,
        )

        if args.epoch_pause > 0 and not (args.max_epochs is not None and epoch >= args.max_epochs):
            time.sleep(args.epoch_pause)

    # 최종 상태 기록
    _write_status(
        status_path, epoch, total_candidates,
        leaderboard._best_ddg() or float("inf"),
        leaderboard._best_sel_margin() or float("-inf"),
        stop_reason, t_start,
    )
    elapsed = round(time.time() - t_start, 1)
    print(
        f"\n[SiloA] 종료: {stop_reason}, epochs={epoch}, "
        f"total_candidates={total_candidates}, "
        f"best_ddg={leaderboard._best_ddg()}, "
        f"best_sel_margin={leaderboard._best_sel_margin()}, "
        f"elapsed={elapsed}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
