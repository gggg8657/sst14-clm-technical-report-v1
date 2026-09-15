#!/usr/bin/env python3
"""experiment_log.jsonl 데이터 유실 복구 스크립트.

## 배경 (2026-07-01 사고)

continuous(Silo B) 엔진의 "hetero_panel" 기능 배포를 위한 재시작 과정에서,
`runs/pyrosetta_flow/experiment_log.jsonl` 이 git 워킹트리 조작(예: `git checkout --`
/ `git reset --hard` / stash 등 — experiment_log.jsonl 이 git 추적 대상이라 발생)에
의해 마지막 git 커밋 시점(2026-06-30T21:11:39Z autopush, 34,808줄)으로 되돌아갔다.
재시작 직전에 archive 스냅샷(`runs/pyrosetta_flow/archives/pre_hetero_panel_*`)이
35,352줄로 정상 보존되어 있었으므로, 그 사이 4시간(22:27~02:08)동안 기록된 544개
고유 서열 레코드가 현재 experiment_log.jsonl 에서만 유실되었다.

append_experiment_records()/load_experiment_records() (pyrosetta_flow/ranking.py)
자체는 append-only 이며 이번 유실의 원인이 아니다 (runner.py 의 flush 로직도 마찬가지).
유실은 파이프라인 외부의 워킹트리 조작에서 발생했다. 본 스크립트는 archive 스냅샷에만
존재하는 레코드를 현재 로그에 안전 병합해 복구한다.

## 사용법

    # dry-run (기본값) — 실제로 아무것도 쓰지 않음, 복구 예정 건수만 출력
    python3 scripts/recover_experiment_log.py \
        --archive runs/pyrosetta_flow/archives/pre_hetero_panel_20260701T021213Z/runs_pyrosetta_flow_experiment_log.jsonl \
        --current runs/pyrosetta_flow/experiment_log.jsonl

    # 실제 병합 적용 (--apply 플래그 필요)
    python3 scripts/recover_experiment_log.py \
        --archive runs/pyrosetta_flow/archives/pre_hetero_panel_20260701T021213Z/runs_pyrosetta_flow_experiment_log.jsonl \
        --current runs/pyrosetta_flow/experiment_log.jsonl \
        --apply

## 안전장치

- 기본은 dry-run: `--apply` 없이는 어떤 파일도 수정하지 않는다.
- `--apply` 시에도 먼저 현재 로그를 `<current>.pre_recovery_backup_<timestamp>.jsonl` 로
  백업한 뒤에만 병합 결과를 쓴다.
- 병합 키는 (sequence, ts) 조합 — 같은 서열이라도 다른 timestamp(다른 iteration/run)의
  레코드는 별개로 취급해 보존한다. 완전히 동일한 (sequence, ts, run_id, candidate_id)
  레코드는 중복으로 간주해 1건만 남긴다.
- 최종 결과는 ts(timestamp) 오름차순으로 정렬해 기록한다 (원본의 시계열 순서 유지 원칙).
- archive/current 파일은 읽기만 하며 절대 rewrite하지 않는다 (원본 파일 무결성 보존).

## 실행 주체

이 스크립트는 **직접 실행하지 않는다** — 엔진(continuous.py/runner.py) 정지 →
본 스크립트 --apply 실행 → 엔진 재시작 순서는 메인 세션이 조율한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일을 로드한다 (malformed 라인은 건너뜀). 읽기 전용 — 절대 write 하지 않음."""
    if not path.exists():
        raise FileNotFoundError(f"파일이 존재하지 않음: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path.name}:{line_no} JSON 파싱 실패(건너뜀): {exc}", file=sys.stderr)
    return rows


def _record_identity(record: Dict[str, Any]) -> Tuple[Any, ...]:
    """완전 중복 판정을 위한 identity 키.

    (record_type, run_id, candidate_id, sequence, ts) 조합 — 이 5개가 모두 같으면
    동일 레코드(같은 flush가 두 번 기록된 경우 등)로 간주해 1건만 남긴다.
    candidate_id 가 없는(예: summary) 레코드는 전체 dict의 정렬된 JSON 문자열로 대체 식별.
    """
    if record.get("candidate_id") is not None:
        return (
            record.get("record_type"),
            record.get("run_id"),
            record.get("candidate_id"),
            record.get("sequence"),
            record.get("ts"),
        )
    return ("__no_candidate_id__", json.dumps(record, sort_keys=True, ensure_ascii=False))


def _sequence_of(record: Dict[str, Any]) -> Optional[str]:
    if record.get("record_type") == "candidate":
        return record.get("sequence")
    return None


def compute_recovery_plan(
    archive_records: List[Dict[str, Any]],
    current_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """archive 에만 있고 current 에는 없는 레코드를 계산한다 (순수 함수, side-effect 없음).

    Returns:
        dict with keys:
            missing_records: archive에만 존재하는 레코드 리스트 (병합 대상)
            missing_unique_sequences: 위 레코드들의 고유 서열 집합
            current_unique_sequences_before: 현재 로그의 고유 서열 수
            merged_records: current + missing_records 를 ts 오름차순 정렬 + 완전중복 제거한 최종 리스트
    """
    current_identities = {_record_identity(r) for r in current_records}
    current_sequences = {
        s for r in current_records if (s := _sequence_of(r)) is not None
    }

    missing_records: List[Dict[str, Any]] = []
    for r in archive_records:
        ident = _record_identity(r)
        if ident in current_identities:
            continue
        missing_records.append(r)

    missing_unique_sequences = {
        s for r in missing_records if (s := _sequence_of(r)) is not None
    } - current_sequences

    # 병합: current + missing, 완전 중복(identity) 제거, ts 오름차순 정렬
    combined = list(current_records) + missing_records
    seen_identity = set()
    deduped: List[Dict[str, Any]] = []
    for r in combined:
        ident = _record_identity(r)
        if ident in seen_identity:
            continue
        seen_identity.add(ident)
        deduped.append(r)

    def _ts_key(r: Dict[str, Any]) -> str:
        # ts 가 없는 레코드는 맨 뒤로 (문자열 비교 위해 매우 큰 값 대용)
        return r.get("ts") or "9999"

    deduped.sort(key=_ts_key)

    return {
        "missing_records": missing_records,
        "missing_unique_sequences": missing_unique_sequences,
        "current_unique_sequences_before": current_sequences,
        "merged_records": deduped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="experiment_log.jsonl 데이터 유실 복구 (archive → current 안전 병합)",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="온전한 과거 스냅샷 경로 (예: runs/pyrosetta_flow/archives/pre_hetero_panel_.../runs_pyrosetta_flow_experiment_log.jsonl)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="현재(유실된) experiment_log.jsonl 경로",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="실제로 병합 결과를 --current 경로에 기록한다. 지정하지 않으면 dry-run(미리보기만).",
    )
    args = parser.parse_args()

    archive_records = _load_jsonl(args.archive)
    current_records = _load_jsonl(args.current)

    plan = compute_recovery_plan(archive_records, current_records)

    print(f"[recover_experiment_log] archive 총 레코드: {len(archive_records)}줄")
    print(f"[recover_experiment_log] current 총 레코드: {len(current_records)}줄")
    print(f"[recover_experiment_log] 복구 대상 레코드(archive에만 존재): {len(plan['missing_records'])}건")
    print(f"[recover_experiment_log] 복구 대상 고유 서열: {len(plan['missing_unique_sequences'])}개")
    print(f"[recover_experiment_log] 병합 후 예상 총 레코드: {len(plan['merged_records'])}줄")

    if plan["missing_records"]:
        ts_values = sorted(
            r.get("ts", "") for r in plan["missing_records"] if r.get("ts")
        )
        if ts_values:
            print(f"[recover_experiment_log] 복구 레코드 timestamp 범위: {ts_values[0]} ~ {ts_values[-1]}")
        source_counts: Dict[str, int] = {}
        for r in plan["missing_records"]:
            src = r.get("mutation_source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        print(f"[recover_experiment_log] mutation_source 분포: {source_counts}")

    if not args.apply:
        print("\n[recover_experiment_log] dry-run 모드 — 아무 파일도 수정하지 않았습니다.")
        print("[recover_experiment_log] 실제 적용하려면 --apply 플래그를 추가하세요.")
        return 0

    if not plan["missing_records"]:
        print("\n[recover_experiment_log] 복구할 레코드가 없습니다 — 종료.")
        return 0

    # --apply: 백업 먼저, 그 다음에만 병합 결과 기록
    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = args.current.with_name(f"{args.current.name}.pre_recovery_backup_{ts_tag}.jsonl")
    backup_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in current_records) + ("\n" if current_records else ""),
        encoding="utf-8",
    )
    print(f"[recover_experiment_log] 현재 로그 백업 완료: {backup_path}")

    merged_text = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in plan["merged_records"]
    ) + "\n"
    args.current.write_text(merged_text, encoding="utf-8")
    print(f"[recover_experiment_log] 병합 완료: {args.current} ({len(plan['merged_records'])}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
