from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def append_experiment_records(log_path: Path, records: List[Dict[str, Any]]) -> None:
    """Append experiment records to JSONL log."""
    if not records:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        for row in records:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # 2026-07-01: 데이터 무결성 — 매 append 후 high-water mark 갱신.
    # (재시작 시 외부 요인으로 파일이 축소되는 경우를 다음 load 시점에 탐지하기 위함)
    _update_high_water_mark(log_path)


def _hwm_path(log_path: Path) -> Path:
    """experiment_log.jsonl 옆에 두는 high-water-mark sidecar 경로."""
    return log_path.with_suffix(log_path.suffix + ".hwm.json")


def _update_high_water_mark(log_path: Path) -> None:
    """현재 파일의 줄 수를 sidecar에 기록 (append 직후 호출).

    이 값은 "지금까지 이 파일이 도달한 최대 줄 수"를 추적한다.
    다음 프로세스 시작 시 load_experiment_records_guarded() 가 이 값과
    실제 파일 줄 수를 비교해 축소(=데이터 유실)를 탐지한다.
    """
    try:
        with log_path.open("r", encoding="utf-8") as fh:
            line_count = sum(1 for line in fh if line.strip())
        hwm_path = _hwm_path(log_path)
        prior = 0
        if hwm_path.exists():
            try:
                prior = int(json.loads(hwm_path.read_text(encoding="utf-8")).get("line_count", 0))
            except Exception:
                prior = 0
        # high-water mark 는 단조 증가만 허용 (절대 감소 기록하지 않음)
        new_count = max(prior, line_count)
        hwm_path.write_text(
            json.dumps({"line_count": new_count}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover — 방어적, HWM 갱신 실패가 본 파이프라인을 막으면 안 됨
        print(f"  [ranking] high-water-mark 갱신 실패(무시): {exc}", file=sys.stderr)


def check_experiment_log_shrinkage(log_path: Path) -> Optional[Dict[str, Any]]:
    """재시작 시점에 experiment_log.jsonl 이 이전 high-water-mark보다 줄었는지 검사.

    Returns:
        축소가 감지되면 {"prior_line_count": int, "current_line_count": int, "missing": int} 를 반환.
        정상(축소 없음)이거나 HWM sidecar가 없으면 None.
    """
    hwm_path = _hwm_path(log_path)
    if not hwm_path.exists():
        return None
    try:
        prior_count = int(json.loads(hwm_path.read_text(encoding="utf-8")).get("line_count", 0))
    except Exception:
        return None
    if not log_path.exists():
        current_count = 0
    else:
        with log_path.open("r", encoding="utf-8") as fh:
            current_count = sum(1 for line in fh if line.strip())
    if current_count < prior_count:
        return {
            "prior_line_count": prior_count,
            "current_line_count": current_count,
            "missing": prior_count - current_count,
        }
    return None


def load_experiment_records(log_path: Path) -> List[Dict[str, Any]]:
    """Load experiment records from JSONL log."""
    if not log_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_experiment_records_guarded(log_path: Path) -> List[Dict[str, Any]]:
    """load_experiment_records() + 축소(데이터 유실) 탐지를 결합한 안전 로더.

    재시작(warm-start) 시 이 함수로 로드하면, 이전에 기록된 high-water-mark보다
    파일이 줄어든 경우 stderr에 강하게 경고하고 `<log_path>.SHRINKAGE_DETECTED.json`
    인시던트 마커를 남긴다 (파이프라인 자체는 죽이지 않음 — fail-open으로 무한정지
    시키지 않되, 조용히 넘어가지 않도록 가시화).
    """
    shrinkage = check_experiment_log_shrinkage(log_path)
    records = load_experiment_records(log_path)
    if shrinkage is not None:
        msg = (
            f"[CRITICAL] experiment_log 데이터 유실 감지: "
            f"이전 high-water-mark={shrinkage['prior_line_count']}줄, "
            f"현재={shrinkage['current_line_count']}줄, "
            f"손실={shrinkage['missing']}줄. "
            f"복구: scripts/recover_experiment_log.py 참조."
        )
        print(f"  [ranking] {msg}", file=sys.stderr)
        try:
            incident_path = log_path.parent / (log_path.name + ".SHRINKAGE_DETECTED.json")
            incident_path.write_text(
                json.dumps({**shrinkage, "message": msg}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover
            print(f"  [ranking] 인시던트 마커 기록 실패: {exc}", file=sys.stderr)
    else:
        # 정상 로드 시에도 HWM을 현재 크기로 갱신(최초 실행 시 sidecar 생성 포함)
        _update_high_water_mark(log_path)
    return records


def extract_historical_sequences(records: List[Dict[str, Any]]) -> set[str]:
    """Extract all previously attempted sequences for cross-run deduplication."""
    return {
        r["sequence"]
        for r in records
        if r.get("record_type") == "candidate" and r.get("sequence")
    }


def summarize_top_hits(records: List[Dict[str, Any]], top_n: int = 10) -> List[Dict[str, Any]]:
    """Return top-N successful candidates sorted by ddG for planner context."""
    successes = [
        r for r in records
        if r.get("record_type") == "candidate"
        and r.get("status") == "success"
        and float(r.get("ddg", 999)) < 0
    ]
    ranked = sorted(successes, key=lambda r: float(r.get("ddg", 999)))[:top_n]
    return [
        {
            "sequence": r["sequence"],
            "ddg": round(float(r["ddg"]), 2),
            "run_id": r.get("run_id", ""),
            "iteration": r.get("iteration", 0),
        }
        for r in ranked
    ]


def build_historical_candidates(records: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    """Build aggregated ranking list from success/failure experiment records."""
    candidate_rows: List[Dict[str, Any]] = [r for r in records if r.get("record_type") == "candidate"]

    def _sort_key(row: Dict[str, Any]) -> tuple:
        status = row.get("status", "failed")
        is_failed = 0 if status == "success" else 1
        ddg = float(row.get("ddg", 999.0))
        return (is_failed, ddg)

    ranked = sorted(candidate_rows, key=_sort_key)[:limit]
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ranked, start=1):
        status = row.get("status", "failed")
        fail_reason = row.get("error_summary", "") if status != "success" else row.get("fail_reason", "")
        ddg = float(row.get("ddg", 0.0))
        out.append(
            {
                "rank": idx,
                "id": row.get("candidate_id", f"log_{idx:03d}"),
                "sequence": row.get("sequence", ""),
                "ddG": ddg,
                "totalScore": float(row.get("total_score", 0.0)),
                "clashScore": float(row.get("clash_score", 0.0)),
                "finalScore": round(-ddg, 3) if ddg < 0 else 0.0,
                "result": "PASS" if status == "success" else "FAIL",
                "failReason": fail_reason,
                "runId": row.get("run_id", ""),
            }
        )
    return out
