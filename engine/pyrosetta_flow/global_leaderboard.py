"""global_leaderboard.py — run 간 영속되는 글로벌 선택성 리더보드 (2026-06-10).

무한 발굴 엔진의 "기억": 매 epoch(run)의 선택성 측정(Δmargin)을 디스크에 누적해
다음 epoch이 ① 이미 도킹한 서열 재도킹 회피 ② in-loop 리더보드 warm-start
③ 역대 best Δmargin 단조 추적 에 사용한다. experiment_log.jsonl(서열 dedup·bandit)
과 상보 — 이쪽은 **선택성 축** 전담.

Δmargin = selectivity_margin − native_margin(home-advantage). >0 = native 초과 선택성.

2026-06-25 개선:
- enrich_from_mmgbsa_consensus(): mmgbsa_consensus.json 을 소비해 리더보드 엔트리에
  mmgbsa_dg / consensus_flag 를 붙이고, _resort() 에서 high_confidence 에 랭킹 부스트
  적용. caveat: MM-GBSA 는 45분 주기 단일 스냅샷 비동기 신호 — 도킹 대체 아님.
- 환경변수 MMGBSA_LEADERBOARD_BOOST(기본 3.0): high_confidence 에 적용할 ddG 부스트
  (낮출수록 랭킹 상승). 0 이면 부스트 없음.

2026-06-29 개선:
- physical floor 적용 (DDG_PHYSICAL_FLOOR 환경변수, 기본 -150):
  raw experiment_log 에서 비물리 outlier(예: -510) 를 수집 단계부터 제외.
  ddg_median/mean/min 등 모든 대표값이 floor 이하이면 unphysical 마킹.
- nstruct=1 단일값은 리더보드 canonical 자격 없음:
  ddg_n_converged==1 이고 ddg_sd==0.0 이며 ddg_median 이 단일값인 경우
  robust_pending 플래그를 부여 → 리더보드 정렬 시 canonical 순위 후순위 배치.
  기존 robust 통계(ddg_median/sd/n_converged) 계산 로직은 변경 없음.
"""
from __future__ import annotations

import json
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from .multiobjective import _has_d_aa_marker

DEFAULT_FILENAME = "global_selectivity_leaderboard.json"
NATIVE_SST14 = "AGCKNFFWKTFTSC"

_NATIVE_MARGIN_CACHE: Optional[float] = None


def _ddg_physical_floor() -> float:
    """DDG_PHYSICAL_FLOOR 환경변수에서 비물리 하한값 반환 (기본 -150 kcal/mol).

    이 값 이하의 ddG는 비물리 outlier로 간주하여 통계·순위 집계에서 제외한다.
    """
    try:
        return float(os.environ.get("DDG_PHYSICAL_FLOOR", "-150"))
    except ValueError:
        return -150.0


def _is_non_robust_single(entry: Dict[str, Any]) -> bool:
    """후보 엔트리가 nstruct=1 단일 측정(non-robust)인지 판정.

    조건: ddg_n_converged==1 이고 ddg_sd==0.0 (또는 None) — 즉 단일 pose 값만 있음.
    이 경우 robust_pending=True 플래그를 통해 canonical 순위에서 후순위 배치한다.

    기존 robust 통계 계산 로직(ddg_median/sd/n_converged)은 변경하지 않고
    판정만 추가한다.
    """
    n = entry.get("ddg_n_converged")
    sd = entry.get("ddg_sd")
    med = entry.get("ddg_median")
    if med is None:
        # ddg_median 자체가 없으면 robust 통계 미산출 → non-robust
        return True
    if n is None or n <= 1:
        if sd is None or sd == 0.0:
            return True
    return False


def _coerce_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


# ---------------------------------------------------------------------------
# 포즈-인지 수렴 게이트 (2026-07-06)
# 도킹 점수는 포즈/시작위치 의존(sd~10). 수렴(nstruct>=2)했더라도 ddg_sd 가 매우 크면
# = 포즈에 따라 점수가 요동하는 floppy 후보 → 헤드라인 승격 차단(단일값 허수 방지와 별개 축).
# 근거: report.md §6.5, 도킹방법 벤치마크. 단일포즈(nstruct=1)는 기존 robust_pending 이 담당.
# ---------------------------------------------------------------------------

def _pose_gate_enabled() -> bool:
    """POSE_GATE_ENABLED(기본 1). 0이면 기존 동작 100% 보존(floppy 플래그·penalty 없음)."""
    return os.environ.get("POSE_GATE_ENABLED", "1") not in ("0", "false", "False", "")


def _pose_floppy_sd_threshold() -> float:
    """floppy 판정 sd 임계(REU). 기본 15.0 (native robust sd~12.3 위 → native 오탐 없음)."""
    try:
        return float(os.environ.get("POSE_FLOPPY_SD_THRESHOLD", "15.0"))
    except ValueError:
        return 15.0


def _pose_floppy_penalty() -> float:
    """floppy 후보 정렬 penalty. 기본 5e5 (단일포즈 robust_pending 1e6 보다 작음:
    floppy 는 다중 측정이라 단일포즈보단 낫지만, clean-converged 아래로 후순위)."""
    try:
        return float(os.environ.get("POSE_FLOPPY_PENALTY", "5e5"))
    except ValueError:
        return 5e5


def _is_pose_uncertain(entry: Dict[str, Any]) -> bool:
    """수렴은 했으나(nstruct>=2, median 존재) ddg_sd 가 임계 초과 = floppy(포즈 불안정).

    단일포즈(robust_pending)와는 별개 축 — 여러 번 도킹해도 포즈마다 점수가 크게 달라지는 경우.
    게이트 비활성(POSE_GATE_ENABLED=0)이면 항상 False.
    """
    if not _pose_gate_enabled():
        return False
    med = entry.get("ddg_median")
    n = entry.get("ddg_n_converged")
    sd = _coerce_float(entry.get("ddg_sd"))
    if med is None or sd is None:
        return False
    try:
        n_int = int(n) if n is not None else 0
    except (TypeError, ValueError):
        n_int = 0
    if n_int < 2:  # 단일포즈는 robust_pending 소관
        return False
    return sd > _pose_floppy_sd_threshold()


def _robust_ddg_key(e: Dict[str, Any]) -> float:
    """Ranking key for binding strength: median ddG first, legacy single ddG fallback."""
    ddg = _coerce_float(e.get("ddg_median"))
    if ddg is None:
        ddg = _coerce_float(e.get("ddg"))
    return ddg if ddg is not None else 1e9


def _extract_entry_ddg_stats(ddg: Optional[float], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """P1-compatible robust ddG fields from a measurement payload.

    If P1 already supplies ddg_median/ddg_sd/ddg_n_converged, trust those fields.
    Otherwise use the current single measurement only when it is a converged ddG<0.

    ddg_mean  : 수렴 trial 들의 산술 평균 (noise·outlier 비교용)
    ddg_min   : 수렴 trial 들의 최솟값 = best (outlier 경보 비교 기준)
    단일 측정인 경우 mean/min 모두 median 과 동일값으로 채운다.
    """
    extra = extra or {}
    med = _coerce_float(extra.get("ddg_median", extra.get("ddG_median")))
    sd = _coerce_float(extra.get("ddg_sd", extra.get("ddG_sd")))
    n_raw = extra.get("ddg_n_converged", extra.get("ddG_n_converged"))
    try:
        n = int(n_raw) if n_raw is not None else None
    except (TypeError, ValueError):
        n = None
    if med is not None:
        mean_val = _coerce_float(extra.get("ddg_mean", extra.get("ddG_mean")))
        min_val = _coerce_float(extra.get("ddg_min", extra.get("ddG_min")))
        # 단일 측정(n=1)이면 mean/min = median
        if mean_val is None:
            mean_val = med
        if min_val is None:
            min_val = med
        return {
            "ddg_median": med,
            "ddg_mean": mean_val,
            "ddg_min": min_val,
            "ddg_sd": sd if sd is not None else 0.0,
            "ddg_n_converged": n if n is not None else 1,
        }
    single = _coerce_float(ddg)
    if single is not None and single < 0:
        return {"ddg_median": single, "ddg_mean": single, "ddg_min": single,
                "ddg_sd": 0.0, "ddg_n_converged": 1}
    return {"ddg_median": None, "ddg_mean": None, "ddg_min": None,
            "ddg_sd": None, "ddg_n_converged": 0}


def load_ddg_robust_stats(experiment_log_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-sequence robust ddG stats from experiment_log.jsonl.

    Only measured converged values with ddG<0 are used. Failed rows and positive
    sentinel/unstable values are excluded to avoid hallucinated binding strength.

    2026-06-29: DDG_PHYSICAL_FLOOR 이하의 비물리 outlier(예: -510)도 수집 단계부터
    제외한다. 이를 통해 통계값(median/mean/sd)이 noise에 오염되지 않는다.
    """
    _floor = _ddg_physical_floor()
    by_seq: Dict[str, List[float]] = {}
    try:
        with Path(experiment_log_path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                seq = row.get("sequence") or row.get("seq")
                if not seq:
                    continue
                ddg = _coerce_float(row.get("ddg", row.get("ddG")))
                if ddg is None:
                    ddg = _coerce_float(row.get("ddg_median", row.get("ddG_median")))
                # physical floor 적용: 비물리 outlier 제외
                if ddg is not None and ddg < 0 and ddg > _floor:
                    by_seq.setdefault(seq, []).append(ddg)
    except FileNotFoundError:
        return {}
    except OSError:
        return {}

    stats: Dict[str, Dict[str, Any]] = {}
    for seq, vals in by_seq.items():
        vals = sorted(vals)
        stats[seq] = {
            "ddg_median": round(float(statistics.median(vals)), 4),
            "ddg_sd": round(float(statistics.stdev(vals)), 4) if len(vals) > 1 else 0.0,
            "ddg_n_converged": len(vals),
        }
    return stats


def backup_leaderboard(path: Path) -> Optional[Path]:
    """Create ``.bak`` next to an existing leaderboard before robust re-save."""
    p = Path(path)
    if not p.exists():
        return None
    bak = Path(str(p) + ".bak")
    if not bak.exists():
        shutil.copy2(p, bak)
    return bak


def _native_margin() -> Optional[float]:
    """native SST-14 동일프로토콜 selectivity_margin (home-advantage 기준). Δ backfill 용."""
    global _NATIVE_MARGIN_CACHE
    if _NATIVE_MARGIN_CACHE is not None:
        return _NATIVE_MARGIN_CACHE
    path = Path(__file__).resolve().parents[1] / "data/somatostatin_receptor/curated/native_selectivity_baseline.json"
    try:
        if path.exists():
            _NATIVE_MARGIN_CACHE = float(json.loads(path.read_text()).get("margin"))
    except Exception:
        pass
    return _NATIVE_MARGIN_CACHE


class GlobalSelectivityLeaderboard:
    """선택성 측정 후보의 run-간 누적 리더보드.

    기본 정렬은 robust ddG median 오름차순이다. P1/P2 이전 동작이 필요하면
    ``use_robust_ddg=False`` 로 Δmargin 내림차순 정렬을 유지할 수 있다.
    """

    def __init__(self, capacity: int = 50, use_robust_ddg: bool = True):
        self.capacity = capacity
        self.use_robust_ddg = use_robust_ddg
        self.entries: List[Dict[str, Any]] = []   # dedup-by-sequence, Δmargin desc
        self.screened_seqs: set = set()
        self.n_ingested_total: int = 0            # 누적 측정 횟수(중복 포함)
        self.native_ddg_stats: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ I/O
    @classmethod
    def load(cls, path: Path, capacity: int = 50, use_robust_ddg: bool = True,
             experiment_log_path: Optional[Path] = None) -> "GlobalSelectivityLeaderboard":
        lb = cls(capacity=capacity, use_robust_ddg=use_robust_ddg)
        try:
            if Path(path).exists():
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                lb.entries = data.get("entries", []) or []
                lb.screened_seqs = set(data.get("screened_seqs", []) or [])
                lb.n_ingested_total = int(data.get("n_ingested_total", len(lb.entries)))
                # 로드분도 screened 에 반영 (방어적)
                for e in lb.entries:
                    if e.get("sequence"):
                        lb.screened_seqs.add(e["sequence"])
        except Exception:
            pass  # 손상 시 빈 리더보드로 시작 (fail-open: 발굴은 계속)
        if use_robust_ddg:
            exp_path = experiment_log_path or Path(path).parent / "experiment_log.jsonl"
            stats = load_ddg_robust_stats(exp_path)
            lb.native_ddg_stats = stats.get(NATIVE_SST14)
            lb.apply_robust_ddg_stats(stats)
            lb._resort()
        return lb

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "capacity": self.capacity,
            "ranking_mode": "robust_ddg_median" if self.use_robust_ddg else "legacy_delta_margin",
            "best_delta_margin": self.best_delta(),
            "best_ddg_median": self.best_ddg_median(),
            "native_ddg_median": (self.native_ddg_stats or {}).get("ddg_median"),
            "native_ddg_sd": (self.native_ddg_stats or {}).get("ddg_sd"),
            "native_ddg_n_converged": (self.native_ddg_stats or {}).get("ddg_n_converged"),
            "n_unique": len(self.entries),
            "n_screened_unique": len(self.screened_seqs),
            "n_ingested_total": self.n_ingested_total,
            "entries": self.entries,
            "screened_seqs": sorted(self.screened_seqs),
        }
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)   # atomic

    # ------------------------------------------------------------- mutation
    def add_measurement(self, seq: str, ddg: Optional[float], margin: Optional[float],
                        delta_margin: Optional[float], extra: Optional[Dict[str, Any]] = None,
                        run_id: str = "", ts: str = "") -> bool:
        """선택성 1건 반영. 신규 best 갱신이면 True. 동일 서열은 더 좋은 Δ만 유지."""
        if not seq or margin is None:
            return False
        self.n_ingested_total += 1
        self.screened_seqs.add(seq)
        dm = _coerce_float(delta_margin)
        if dm is None:  # post-loop 경로 등 Δ 미계산 시 native baseline 으로 backfill
            nat = _native_margin()
            m = _coerce_float(margin)
            if nat is not None and m is not None:
                dm = round(m - nat, 4)

        # 비물리 ddG 마킹: floor 이하 단일값은 unphysical 플래그 부여
        _floor = _ddg_physical_floor()
        _ddg_val = _coerce_float(ddg)
        _ddg_unphysical = _ddg_val is not None and _ddg_val <= _floor

        rec = {
            "sequence": seq,
            "ddg": _ddg_val,
            "margin": _coerce_float(margin),
            "delta_margin": dm,
            "hc50": (extra or {}).get("pepadmet_hc50"),
            "hc50_vs_native": (extra or {}).get("hc50_vs_native"),
            "more_toxic_than_native": (extra or {}).get("more_toxic_than_native"),
            "run_id": run_id,
            "ts": ts,
            "unphysical": _ddg_unphysical,
        }
        rec.update(_extract_entry_ddg_stats(ddg, extra))

        # nstruct=1 단일값은 robust_pending 마킹 (canonical 순위 후순위)
        rec["robust_pending"] = _is_non_robust_single(rec)
        # 수렴했으나 고-sd(floppy) 후보는 pose_uncertain 마킹 (헤드라인 승격 차단)
        rec["pose_uncertain"] = _is_pose_uncertain(rec)
        prev_best = self.best_delta()
        existing = next((e for e in self.entries if e["sequence"] == seq), None)
        if existing is not None:
            # 같은 서열이면 Δmargin 더 높은 측정으로만 갱신
            if dm is not None and (existing.get("delta_margin") is None or dm > existing["delta_margin"]):
                self.entries.remove(existing)
                self.entries.append(rec)
            else:
                existing.update({k: v for k, v in rec.items()
                                 if k in {"ddg_median", "ddg_mean", "ddg_min",
                                          "ddg_sd", "ddg_n_converged"}})
        else:
            self.entries.append(rec)
        self._resort()
        new_best = self.best_delta()
        return (new_best is not None) and (prev_best is None or new_best > prev_best + 1e-9)

    def ingest_artifacts(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """한 run 의 artifacts(dict)에서 선택성 측정을 모두 수집. 반환: 요약."""
        run_id = artifacts.get("run_id", "")
        n_new = 0
        improved = False

        def _scan(cand_list):
            nonlocal n_new, improved
            for c in cand_list or []:
                es = c.get("extra_scores", {}) or {}
                margin = es.get("selectivity_margin")
                if margin is None:
                    continue
                dm = es.get("delta_margin")
                if dm is None:  # post-loop 경로는 Δ 미계산 — margin 만 있을 때 그대로 둠(None)
                    pass
                if self.add_measurement(
                    seq=c.get("sequence", ""), ddg=c.get("ddg"), margin=margin,
                    delta_margin=dm, extra=es, run_id=run_id, ts=c.get("ts", ""),
                ):
                    improved = True
                n_new += 1

        for it in artifacts.get("iterations", []):
            _scan(it.get("candidates", []))
        _scan(artifacts.get("final_candidates", []))
        return {"n_measurements": n_new, "improved_best": improved,
                "best_delta_margin": self.best_delta(), "n_unique": len(self.entries)}

    def _resort(self) -> None:
        # 환경변수 MMGBSA_LEADERBOARD_BOOST: high_confidence 항목의 ddG 키를
        # boost 만큼 낮춰 랭킹 상승(더 작은 ddG = 더 높은 순위).
        # 0 이면 부스트 없음. 기본 3.0 (kcal/mol 단위 — 도킹 noise sd~9 의 1/3).
        # caveat: MM-GBSA 는 45분 주기 단일 스냅샷 비동기 신호, 도킹 대체 아님.
        _boost = float(os.environ.get("MMGBSA_LEADERBOARD_BOOST", "3.0"))

        def _sort_key_robust(e: Dict[str, Any]) -> tuple:
            base = _robust_ddg_key(e)
            _floppy = e.get("pose_uncertain", False)
            # high_confidence 부스트 — floppy(포즈 불안정)는 부스트 차단(헤드라인 승격 방지)
            if _boost > 0 and e.get("consensus_flag") == "high_confidence" and not _floppy:
                base -= _boost
            # robust_pending(nstruct=1 단일값)은 canonical 후순위: 정렬 1순위 키를 1e6 offset
            _pending_penalty = 1e6 if e.get("robust_pending", False) else 0.0
            # pose_uncertain(수렴 고-sd floppy)은 clean 아래로 후순위(단일포즈보단 위)
            _floppy_penalty = _pose_floppy_penalty() if _floppy else 0.0
            _sd = _coerce_float(e.get("ddg_sd"))
            return (
                _pending_penalty + _floppy_penalty + base,
                -(e["delta_margin"] if e.get("delta_margin") is not None else -1e9),
                _sd if _sd is not None else 1e9,  # 일관성 tiebreaker: 낮은 sd(포즈 안정) 우선
            )

        def _sort_key_legacy(e: Dict[str, Any]) -> tuple:
            dm = -(e["delta_margin"] if e.get("delta_margin") is not None else -1e9)
            ddg = e.get("ddg") if e.get("ddg") is not None else 1e9
            _floppy = e.get("pose_uncertain", False)
            if _boost > 0 and e.get("consensus_flag") == "high_confidence" and not _floppy:
                ddg -= _boost
            # robust_pending / pose_uncertain 후순위
            _pending_penalty = 1e6 if e.get("robust_pending", False) else 0.0
            _floppy_penalty = _pose_floppy_penalty() if _floppy else 0.0
            _sd = _coerce_float(e.get("ddg_sd"))
            return (dm, _pending_penalty + _floppy_penalty + ddg, _sd if _sd is not None else 1e9)

        if self.use_robust_ddg:
            self.entries.sort(key=_sort_key_robust)
        else:
            self.entries.sort(key=_sort_key_legacy)
        self.entries = self.entries[: self.capacity]

    def enrich_from_mmgbsa_consensus(self, mmgbsa_path: Path) -> Dict[str, Any]:
        """mmgbsa_consensus.json 을 읽어 리더보드 엔트리에 MM-GBSA 필드를 붙인다.

        엔트리의 sequence 가 mmgbsa results 에 있으면 mmgbsa_dg / consensus_flag
        를 추가하고 _resort() 로 랭킹을 재계산한다.

        caveat:
        - MM-GBSA 는 45분 주기 단일 스냅샷 비동기 신호, 엔트로피/MD 미포함.
        - 도킹 ddG 를 대체하지 않고 보조 정렬 키로만 사용.
        - 파일 없거나 파싱 실패 시 graceful skip (리더보드 변경 없음).

        Returns:
            {enriched: int, skipped: int, generated_at: str}
        """
        result: Dict[str, Any] = {"enriched": 0, "skipped": 0, "generated_at": ""}
        try:
            if not Path(mmgbsa_path).exists():
                return result
            raw = json.loads(Path(mmgbsa_path).read_text(encoding="utf-8"))
            result["generated_at"] = raw.get("generated_at", "")
            mmgbsa_results: Dict[str, Any] = raw.get("results", {})
            if not mmgbsa_results:
                return result
            changed = False
            for entry in self.entries:
                seq = entry.get("sequence", "")
                if seq in mmgbsa_results:
                    rec = mmgbsa_results[seq]
                    entry["mmgbsa_dg"] = _coerce_float(rec.get("dg_bind", rec.get("mmgbsa_dg")))
                    entry["consensus_flag"] = rec.get("consensus_flag", "")
                    result["enriched"] += 1
                    changed = True
                else:
                    result["skipped"] += 1
            if changed:
                self._resort()
        except Exception as exc:
            result["error"] = str(exc)
        return result

    def apply_robust_ddg_stats(self, stats_by_seq: Dict[str, Dict[str, Any]]) -> None:
        """Attach experiment-log median/sd/n to existing entries by exact sequence.

        2026-06-29: robust 통계 적용 후 robust_pending 플래그를 재계산한다.
        nstruct>=5 의 median 이 있으면 robust_pending=False 로 전환.
        """
        for e in self.entries:
            seq = e.get("sequence")
            if seq in stats_by_seq:
                e.update(stats_by_seq[seq])
            elif "ddg_median" not in e:
                e.update(_extract_entry_ddg_stats(e.get("ddg"), None))
            # robust 통계 적용 후 robust_pending / pose_uncertain 재계산
            e["robust_pending"] = _is_non_robust_single(e)
            e["pose_uncertain"] = _is_pose_uncertain(e)

    # --------------------------------------------------------------- query
    def best_delta(self) -> Optional[float]:
        ds = [e["delta_margin"] for e in self.entries if e.get("delta_margin") is not None]
        return max(ds) if ds else None

    def top(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.entries[:n]

    def best_ddg_median(self) -> Optional[float]:
        vals = [_coerce_float(e.get("ddg_median")) for e in self.entries]
        vals = [v for v in vals if v is not None]
        return min(vals) if vals else None

    def count_passing(self, ddg_max: float = -15.0) -> int:
        """엄격 기준 충족: Δmargin>0 & robust ddG median≤ddg_max & 독성≤native.

        D-aa marker 서열의 독성은 `more_toxic_than_native is False`(명시적 측정)만 통과로 인정한다.
        L-aa only 서열은 pepADMET hc50 역변별(AUC=0.146) 확인으로 해당 판정을 skip 한다.
        """
        n = 0
        for e in self.entries:
            dm = e.get("delta_margin"); ddg = e.get("ddg_median") if self.use_robust_ddg else e.get("ddg")
            tox_ok = True
            if _has_d_aa_marker(str(e.get("sequence") or "")):
                tox_ok = e.get("more_toxic_than_native") is False
            if dm is not None and dm > 0 and ddg is not None and ddg <= ddg_max and tox_ok:
                n += 1
        return n

    def warm_start_payload(self) -> Dict[str, Any]:
        """in-loop SelectivityLeaderboard 에 넘길 warm-start 데이터."""
        return {"screened_seqs": set(self.screened_seqs), "entries": list(self.entries)}
