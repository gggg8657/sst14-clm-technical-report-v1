"""selectivity_loop.py — in-loop 선택성 측정 (조건부 게이트, 2026-06-10).

문제: 선택성(off-target 도킹)은 비싸서(후보×5수용체) 매 iteration 전체 후보에 못 돌린다.
해결(사용자 제안): **기존 selectivity 리더보드(top-K)보다 유망한(ddG가 더 강한) 후보만** 도킹.
  - 선택성은 도킹 전엔 모르므로, **ddG**(루프 내 실측, 강한 신호)를 유망도 프록시로 사용.
  - 리더보드 미충원이거나 후보 ddG가 리더보드 최약체보다 강하면 → 선택성 도킹(병렬, ~6분).
지표: **Δmargin = margin − native_margin** (home-advantage 보정, native baseline). >0 = native 초과 선택성.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class SelectivityLeaderboard:
    """선택성 측정된 후보의 top-K (Δmargin 기준). ddG 게이트로 도킹 대상 제한."""

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.entries: List[Dict[str, Any]] = []   # {seq, delta_margin, margin, ddg, cid}
        self.screened_seqs: set = set()

    def seed_from_global(self, payload: Dict[str, Any]) -> None:
        """글로벌 리더보드로 warm-start (2026-06-10 무한 엔진).

        - screened_seqs: 역대 도킹한 서열 → 이번 epoch 재도킹 회피.
        - entries: 역대 top-K 를 in-loop 게이트 기준선으로 적재 → '역대 best보다
          유망한 후보만' 도킹하도록 worst_ddg 임계를 끌어올린다.
        """
        self.screened_seqs |= set(payload.get("screened_seqs", set()))
        seeded = []
        for e in payload.get("entries", [])[: self.capacity]:
            ddg = e.get("ddg_median", e.get("ddg"))
            if ddg is None:
                continue
            seeded.append({"seq": e.get("sequence"), "ddg": ddg,
                           "ddg_median": e.get("ddg_median"),
                           "ddg_sd": e.get("ddg_sd"),
                           "ddg_n_converged": e.get("ddg_n_converged"),
                           "margin": e.get("margin"), "delta_margin": e.get("delta_margin"),
                           "cid": "global"})
        if seeded:
            self.entries = seeded
            self.entries.sort(key=lambda x: (-(x["delta_margin"] if x["delta_margin"] is not None else -1e9), x["ddg"]))
            self.entries = self.entries[: self.capacity]

    def worst_ddg(self) -> Optional[float]:
        if not self.entries:
            return None
        return max(e["ddg"] for e in self.entries)   # 최약 binder(가장 높은 ddG)

    def should_screen(self, seq: str, ddg: float, ddg_cutoff: float = -10.0) -> bool:
        """이 후보를 선택성 도킹할 가치가 있나? (유망도 프록시 = ddG)"""
        if seq in self.screened_seqs:
            return False
        if ddg > ddg_cutoff:          # 너무 약한 binder는 후보 자격 미달
            return False
        if len(self.entries) < self.capacity:
            return True               # 리더보드 미충원 → 채운다
        return ddg < self.worst_ddg()  # 기존 top-K 최약체보다 강하면 도킹

    def update(self, seq: str, ddg: float, margin: Optional[float],
               delta_margin: Optional[float], cid: str = "") -> None:
        self.screened_seqs.add(seq)
        self.entries.append({"seq": seq, "ddg": ddg, "margin": margin,
                             "delta_margin": delta_margin, "cid": cid})
        # Δmargin 내림차순(선택성 우수 순), 동률은 ddG
        self.entries.sort(key=lambda e: (-(e["delta_margin"] if e["delta_margin"] is not None else -1e9), e["ddg"]))
        self.entries = self.entries[:self.capacity]

    def summary(self) -> List[Dict[str, Any]]:
        return [{"sequence": e["seq"], "delta_margin": e["delta_margin"],
                 "margin": e["margin"], "ddg": round(e["ddg"], 2)} for e in self.entries]

    def best_delta(self) -> Optional[float]:
        deltas = [e["delta_margin"] for e in self.entries if e["delta_margin"] is not None]
        return max(deltas) if deltas else None


def screen_iteration_candidates(
    candidates: List[Any],
    iter_dir: Path,
    leaderboard: SelectivityLeaderboard,
    original_sequence: str,
    conda_env: str = "bio-tools",
    max_screen_per_iter: int = 2,
    clash_max: float = 10.0,
    timeout: int = 600,
) -> List[Dict[str, Any]]:
    """이번 iteration 후보 중 게이트 통과분만 선택성 도킹(병렬). Δmargin 을 extra_scores 에 기록.

    Returns: 이번에 도킹한 후보들의 결과 요약 리스트.
    """
    try:
        from .multiobjective import screen_selectivity
    except Exception as exc:  # pragma: no cover
        print(f"  [sel-loop] screen_selectivity import 실패: {exc}", file=sys.stderr)
        return []

    # Cys 위치(이황화) 보존 후보만 (clash 통과 + fail 아님 + PDB 존재)
    ref_cys = {i + 1 for i, a in enumerate(original_sequence.upper()) if a == "C"}

    def _disulfide_ok(seq: str) -> bool:
        return all(p <= len(seq) and seq[p - 1] == "C" for p in ref_cys)

    eligible = []
    for c in candidates:
        if getattr(c, "fail_reason", ""):
            continue
        if float(getattr(c, "clash_score", 999)) > clash_max:
            continue
        seq = getattr(c, "sequence", "")
        if not _disulfide_ok(seq):
            continue
        cid = getattr(c, "candidate_id", "")
        try:
            pdb = iter_dir / f"cand_{int(cid.split('cand')[1]):03d}.pdb" if "cand" in cid else None
        except Exception:
            pdb = None
        if not pdb or not pdb.exists():
            continue
        eligible.append((c, str(pdb)))

    # ddG 강한 순으로 게이트 적용, max_screen_per_iter 개까지
    eligible.sort(key=lambda x: float(getattr(x[0], "ddg", 999)))
    screened = []
    for c, pdb in eligible:
        if len(screened) >= max_screen_per_iter:
            break
        seq = c.sequence
        ddg = float(c.ddg)
        if not leaderboard.should_screen(seq, ddg):
            continue
        sel = screen_selectivity(sstr2_complex_pdb=pdb, on_target_ddg=ddg,
                                 conda_env=conda_env, timeout=timeout)
        margin = sel.get("selectivity_margin")
        delta = sel.get("delta_margin")
        c.extra_scores["selectivity_margin"] = margin
        c.extra_scores["delta_margin"] = delta
        c.extra_scores["offtarget_ddg"] = sel.get("offtarget_ddg")
        c.extra_scores["sstr2_ddg_sameprotocol"] = sel.get("sstr2_ddg_sameprotocol")
        leaderboard.update(seq, ddg, margin, delta, cid=c.candidate_id)
        screened.append({"sequence": seq, "ddg": round(ddg, 2), "margin": margin,
                         "delta_margin": delta,
                         "more_selective_than_native": sel.get("more_selective_than_native")})
        print(f"  [sel-loop] {seq}: margin={margin} Δ={delta} "
              f"({'native↑' if sel.get('more_selective_than_native') else 'native이하'})", file=sys.stderr)
    return screened
