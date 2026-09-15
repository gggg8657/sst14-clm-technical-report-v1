"""continuous.py — 무한 발굴 오케스트레이터 (2026-06-10).

기존 단발 run(run_pyrosetta_agentic_mutdock_flow) 위에 얇은 epoch 루프를 씌워
**STOP 파일이 생길 때까지 무한 발굴**한다. run 간 학습은 experiment_log.jsonl(서열
dedup·bandit) + global_selectivity_leaderboard.json(Δmargin)으로 누적된다.

설계 원칙
- **사람이 조절**: control 파일(JSON)을 매 epoch 시작 시 다시 읽어 knobs 반영(재시작 불필요).
- **STOP 파일**: 존재하면 현재 epoch까지 마치고 graceful 종료.
- **수렴→다양성 탈출**: 글로벌 best Δmargin 이 patience epoch 동안 정체하면 변이 다양성↑
  (max_random_mutations 상향, seed 교체)로 local optimum 탈출. 개선 시 base 로 리셋.
- **진행 가시화**: status 파일에 epoch·역대 best·통과 후보 수·다양성 레벨 기록.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schema import FlowConfig
from .global_leaderboard import GlobalSelectivityLeaderboard


# control 파일에서 override 허용하는 FlowConfig 필드 (안전 화이트리스트)
_CONTROL_FIELDS = {
    "n_candidates", "max_iterations", "top_k", "selectivity_max_per_iter",
    "max_random_mutations", "rosetta_ddg_max", "objective_mode",
    "design_positions", "validation_n_trials", "selectivity_top_k",
}


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _apply_control(base: FlowConfig, control: Dict[str, Any]) -> FlowConfig:
    """control 파일의 화이트리스트 필드만 base config 에 덮어쓴다."""
    overrides = {k: v for k, v in control.items() if k in _CONTROL_FIELDS}
    return replace(base, **overrides) if overrides else base


def _one_sided_t_improved(
    new_ddg_median: float,
    new_ddg_sd: float,
    new_n_converged: int,
    reference_ddg: float,
    t_threshold: float = -1.64,
) -> bool:
    """단측 t근사로 new_ddg_median 이 reference_ddg 를 통계적으로 능가하는지 판정.

    귀무가설 H0: mean >= reference_ddg (능가 아님)
    단측 검정 t = (new_ddg_median - reference_ddg) / (new_ddg_sd / sqrt(n))
    t < t_threshold (기본 -1.64, α≈0.05 단측)이면 귀무가설 기각 → 통계적 개선.

    graceful fallback:
    - n_converged < 2 또는 ddg_sd == 0: 단순 point 비교(< reference_ddg)로 후퇴.
    """
    if new_n_converged < 2 or new_ddg_sd == 0.0:
        # nstruct 부족 또는 sd=0 → point estimate 비교
        return new_ddg_median < reference_ddg - 1e-9

    se = new_ddg_sd / math.sqrt(new_n_converged)
    t_stat = (new_ddg_median - reference_ddg) / se
    return t_stat < t_threshold


def _position_entropy(focus_sequences: List[str], seq_len: int) -> float:
    """최근 N iter focus 서열들의 위치별 Shannon 엔트로피 평균.

    엔트로피가 낮을수록 변이가 특정 위치에 집중됨 → diversity boost 조기 발동 신호.
    서열 목록이 비거나 seq_len=0이면 0.0 반환.
    """
    if not focus_sequences or seq_len == 0:
        return 0.0

    # 각 위치별 아미노산 빈도 수집
    per_pos: List[Dict[str, int]] = [{} for _ in range(seq_len)]
    for seq in focus_sequences:
        for i, aa in enumerate(seq[:seq_len]):
            per_pos[i][aa] = per_pos[i].get(aa, 0) + 1

    # 위치별 엔트로피 계산
    total_entropy = 0.0
    n = len(focus_sequences)
    for freq_map in per_pos:
        pos_entropy = 0.0
        for cnt in freq_map.values():
            p = cnt / n
            if p > 0:
                pos_entropy -= p * math.log2(p)
        total_entropy += pos_entropy

    return total_entropy / seq_len  # 위치 평균 엔트로피


class DiversityPolicy:
    """글로벌 best Δmargin 정체 + 통계적 판정 + position entropy 로 다양성 탈출.

    변경점(작업3, 작업4):
    - update()에 ddg_sd/n_converged 수신 → 단측 t근사로 noise 내 허위 개선 차단.
    - add_focus_sequence()로 최근 N iter 위치를 누적 → 엔트로피 낮으면 diversity boost 조기 발동.
    """

    def __init__(self, patience: int = 3, base_mutations: int = 3,
                 max_mutations_cap: int = 6,
                 entropy_window: int = 5,
                 entropy_low_threshold: float = 0.5):
        self.patience = patience
        self.base_mutations = base_mutations
        self.max_mutations_cap = max_mutations_cap
        # [작업4] position entropy 추적
        self.entropy_window = entropy_window
        self.entropy_low_threshold = entropy_low_threshold
        self._focus_seqs: List[str] = []   # 최근 window iter 의 best 서열
        self._seq_len: int = 0

        self.level = 0                       # 다양성 레벨 (0=base)
        self._best: Optional[float] = None
        self._best_ddg_median: Optional[float] = None   # [작업3] 이전 best ddg_median
        self._stale = 0

    def add_focus_sequence(self, seq: str) -> None:
        """[작업4] 이번 epoch 의 focus 서열 추가(window 유지)."""
        if seq:
            if self._seq_len == 0:
                self._seq_len = len(seq)
            self._focus_seqs.append(seq)
            if len(self._focus_seqs) > self.entropy_window:
                self._focus_seqs = self._focus_seqs[-self.entropy_window:]

    def update(
        self,
        global_best: Optional[float],
        best_ddg_median: Optional[float] = None,
        best_ddg_sd: Optional[float] = None,
        best_n_converged: Optional[int] = None,
    ) -> Dict[str, Any]:
        """epoch 결과 반영 후 다음 epoch 다양성 파라미터 반환.

        Args:
            global_best: 글로벌 best Δmargin (선택성 지표).
            best_ddg_median: 이번 epoch best 후보 ddg_median (extra_scores에서 추출).
            best_ddg_sd: 이번 epoch best 후보 ddg_sd.
            best_n_converged: 이번 epoch best 후보 n_converged.
        """
        improved = False

        # [작업3] 통계적 개선 판정: ddg_median/sd/n_converged 사용 가능하면 단측 t근사,
        # 없으면 Δmargin 단순 비교로 graceful fallback.
        if (best_ddg_median is not None
                and best_ddg_sd is not None
                and best_n_converged is not None
                and self._best_ddg_median is not None):
            # ddg 축: 더 낮은 값이 개선. 단측 t근사로 통계적 능가 여부 판정.
            stat_improved = _one_sided_t_improved(
                new_ddg_median=best_ddg_median,
                new_ddg_sd=best_ddg_sd,
                new_n_converged=best_n_converged,
                reference_ddg=self._best_ddg_median,
            )
        else:
            # fallback: Δmargin 단순 비교 (기존 동작 보존)
            stat_improved = (
                global_best is not None
                and (self._best is None or global_best > self._best + 1e-9)
            )

        if stat_improved and global_best is not None and (
            self._best is None or global_best > self._best + 1e-9
        ):
            # Δmargin도 함께 개선되어야 진짜 개선으로 인정 (noise ddg 단독 갱신 방지)
            self._best = global_best
            if best_ddg_median is not None:
                self._best_ddg_median = best_ddg_median
            self._stale = 0
            self.level = 0                   # 개선 → 탐색 집중(base 복귀)
            improved = True
        elif (global_best is not None
              and self._best is None):
            # 최초 epoch 초기화 (개선으로 처리하지 않음, improved=False 유지)
            self._best = global_best
            if best_ddg_median is not None:
                self._best_ddg_median = best_ddg_median
        else:
            self._stale += 1
            if self._stale >= self.patience:
                self.level += 1              # 정체 → 다양성 단계 상승
                self._stale = 0

        # [작업4] position entropy 낮으면 diversity boost 조기 발동
        entropy = _position_entropy(self._focus_seqs, self._seq_len)
        entropy_triggered = (
            len(self._focus_seqs) >= 2
            and entropy < self.entropy_low_threshold
            and self.level == 0  # base level 에서만 조기 발동
        )
        if entropy_triggered:
            self.level = max(self.level, 1)  # 최소 1단계로 올림

        mutations = min(self.max_mutations_cap, self.base_mutations + self.level)
        return {
            "improved": improved,
            "level": self.level,
            "stale": self._stale,
            "max_random_mutations": mutations,
            "position_entropy": round(entropy, 4),
            "entropy_triggered": entropy_triggered,
        }


def run_continuous_discovery(
    base_config: FlowConfig,
    repo_root: Path,
    control_path: Path,
    stop_path: Path,
    status_path: Path,
    max_epochs: Optional[int] = None,
    poll_on_stop: bool = False,
    epoch_pause_s: float = 0.0,
) -> Dict[str, Any]:
    """STOP 파일이 생길 때까지(또는 max_epochs 도달까지) 무한 발굴.

    Args:
        base_config: 기준 FlowConfig (inloop_selectivity 강제 ON).
        control_path: 사람이 조절하는 JSON knobs (매 epoch 재로드).
        stop_path: 이 파일이 존재하면 graceful 종료.
        status_path: 진행 상황 기록 파일.
        max_epochs: None 이면 무한. 정수면 그만큼만.
        poll_on_stop: True 면 STOP 제거 시 재개를 위해 대기(미사용 기본).
    """
    from . import run_pyrosetta_agentic_mutdock_flow  # 지연 import (무거운 의존성)

    # 무한 엔진: in-loop 선택성 필수 + native baseline 첫 epoch만 측정 후 캐시 재사용
    base_config = replace(base_config, inloop_selectivity=True, reuse_baseline=True)
    global_lb_path = repo_root / base_config.output_dir / "global_selectivity_leaderboard.json"

    control0 = _read_json(control_path)
    policy = DiversityPolicy(
        patience=int(control0.get("patience", 3)),
        base_mutations=int(control0.get("base_mutations", base_config.max_random_mutations)),
        max_mutations_cap=int(control0.get("max_mutations_cap", 6)),
        entropy_window=int(control0.get("entropy_window", 5)),
        entropy_low_threshold=float(control0.get("entropy_low_threshold", 0.5)),
    )

    epoch = 0
    history = []
    t0 = time.time()
    stop_reason = "max_epochs"

    while True:
        if stop_path.exists():
            stop_reason = "stop_file"
            print(f"[continuous] STOP 파일 감지({stop_path}) — graceful 종료", file=sys.stderr)
            break
        if max_epochs is not None and epoch >= max_epochs:
            stop_reason = "max_epochs"
            break

        epoch += 1
        control = _read_json(control_path)            # 매 epoch 재로드 (사람 조절 반영)
        cfg = _apply_control(base_config, control)

        # 다양성: epoch 마다 새 seed 로 탐색 영역 이동 + 직전 정책이 정한 변이 수 적용
        cfg = replace(cfg, seed_base=base_config.seed_base + epoch * 1000)
        # 직전 epoch 결과로 결정된 다양성 레벨 적용
        if history:
            cfg = replace(cfg, max_random_mutations=history[-1]["next_max_mutations"])

        # 선택적 목표-도달 자동 정지
        target = control.get("target_pass_count")
        if target and control.get("stop_on_target", False):
            lb_now = GlobalSelectivityLeaderboard.load(global_lb_path)
            if lb_now.count_passing(ddg_max=cfg.rosetta_ddg_max) >= int(target):
                stop_reason = "target_reached"
                print(f"[continuous] 목표 도달(통과 {target}건) — 종료", file=sys.stderr)
                epoch -= 1
                break

        print(f"\n[continuous] ===== EPOCH {epoch} 시작 (seed={cfg.seed_base}, "
              f"mut≤{cfg.max_random_mutations}, n_cand={cfg.n_candidates}, "
              f"max_iter={cfg.max_iterations}) =====", file=sys.stderr)

        epoch_t0 = time.time()
        artifacts = None
        try:
            artifacts = run_pyrosetta_agentic_mutdock_flow(cfg)
            ran_ok = True
            err = ""
        except Exception as exc:  # epoch 실패는 무한 루프를 죽이지 않는다
            ran_ok = False
            err = repr(exc)
            print(f"[continuous] EPOCH {epoch} 실패(continue): {err}", file=sys.stderr)

        # [작업3] artifacts에서 best 후보의 ddg_median/sd/n_converged 추출
        # final_candidates 또는 iterations 내 최저 ddg 후보 기준
        _best_ddg_median: Optional[float] = None
        _best_ddg_sd: Optional[float] = None
        _best_n_converged: Optional[int] = None
        _best_seq: Optional[str] = None
        if ran_ok and artifacts is not None:
            try:
                _cands = []
                art_dict = artifacts.to_dict() if hasattr(artifacts, "to_dict") else {}
                # final_candidates 우선
                for fc in art_dict.get("final_candidates", []):
                    _cands.append(fc)
                if not _cands:
                    for it in art_dict.get("iterations", []):
                        _cands.extend(it.get("candidates", []))
                # best = 최저 ddg
                _valid = [c for c in _cands if float(c.get("ddg", 999)) < 900]
                if _valid:
                    _best_c = min(_valid, key=lambda c: float(c.get("ddg", 999)))
                    _es = _best_c.get("extra_scores") or {}
                    _best_ddg_median = _es.get("ddg_median")
                    _best_ddg_sd = _es.get("ddg_sd")
                    _n = _es.get("n_converged")
                    _best_n_converged = int(_n) if _n is not None else None
                    _best_seq = _best_c.get("sequence")
            except Exception as _ext_exc:
                print(f"[continuous] best 후보 추출 실패(non-fatal): {_ext_exc}", file=sys.stderr)

        # [작업4] focus sequence 등록 (position entropy 추적용)
        if _best_seq:
            policy.add_focus_sequence(_best_seq)

        # 글로벌 리더보드(runner 가 이미 저장) 재로드 → 진행 판정
        lb = GlobalSelectivityLeaderboard.load(global_lb_path)
        global_best = lb.best_delta()
        # [작업3] ddg_median/sd/n_converged 전달 → 단측 t근사 능가 판정
        decision = policy.update(
            global_best,
            best_ddg_median=_best_ddg_median,
            best_ddg_sd=_best_ddg_sd,
            best_n_converged=_best_n_converged,
        )

        rec = {
            "epoch": epoch,
            "seed_base": cfg.seed_base,
            "max_random_mutations": cfg.max_random_mutations,
            "ran_ok": ran_ok,
            "error": err,
            "global_best_delta": global_best,
            "n_unique_screened": len(lb.entries),
            "passing_count": lb.count_passing(ddg_max=cfg.rosetta_ddg_max),
            "diversity_level": decision["level"],
            "stale_epochs": decision["stale"],
            "improved": decision["improved"],
            "next_max_mutations": decision["max_random_mutations"],
            "position_entropy": decision.get("position_entropy"),
            "entropy_triggered": decision.get("entropy_triggered", False),
            "epoch_seconds": round(time.time() - epoch_t0, 1),
        }
        history.append(rec)

        _atomic_write_json(status_path, {
            "running": True,
            "epochs_done": epoch,
            "elapsed_seconds": round(time.time() - t0, 1),
            "global_best_delta_margin": global_best,
            "passing_count": rec["passing_count"],
            "diversity_level": decision["level"],
            "position_entropy": decision.get("position_entropy"),
            "top": lb.top(10),
            "last_epoch": rec,
            "history": history[-50:],
        })
        print(f"[continuous] EPOCH {epoch} 완료: 역대 best Δ={global_best}, "
              f"통과 {rec['passing_count']}건, 다양성 레벨={decision['level']} "
              f"(개선={decision['improved']}, entropy={decision.get('position_entropy')},"
              f" entropy_trig={decision.get('entropy_triggered', False)}), "
              f"{rec['epoch_seconds']}s", file=sys.stderr)

        if epoch_pause_s > 0:
            time.sleep(epoch_pause_s)

    # 종료 상태 기록
    lb = GlobalSelectivityLeaderboard.load(global_lb_path)
    final = {
        "running": False,
        "stop_reason": stop_reason,
        "epochs_done": epoch,
        "elapsed_seconds": round(time.time() - t0, 1),
        "global_best_delta_margin": lb.best_delta(),
        "passing_count": lb.count_passing(ddg_max=base_config.rosetta_ddg_max),
        "top": lb.top(10),
        "history": history,
    }
    _atomic_write_json(status_path, final)
    print(f"[continuous] 종료({stop_reason}): {epoch} epochs, 역대 best Δ={lb.best_delta()}, "
          f"통과 {final['passing_count']}건", file=sys.stderr)
    return final
