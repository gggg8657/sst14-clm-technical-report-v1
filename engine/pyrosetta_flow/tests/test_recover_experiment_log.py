"""scripts/recover_experiment_log.py 의 순수 로직(compute_recovery_plan) 테스트.

2026-07-01 데이터 유실 사고(warm-start 재시작 시 experiment_log.jsonl 이 git 워킹트리
조작으로 544개 고유서열 손실) 복구 스크립트 회귀 테스트.

scripts/ 는 패키지가 아니므로 importlib.util 로 파일 경로 기반 import 한다.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Dict

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "recover_experiment_log.py"
_spec = importlib.util.spec_from_file_location("recover_experiment_log", _SCRIPT_PATH)
recover_experiment_log = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(recover_experiment_log)

compute_recovery_plan = recover_experiment_log.compute_recovery_plan
_load_jsonl = recover_experiment_log._load_jsonl
main = recover_experiment_log.main


def _candidate(sequence: str, ts: str, run_id: str = "run1", candidate_id: str | None = None) -> Dict[str, Any]:
    return {
        "record_type": "candidate",
        "status": "success",
        "run_id": run_id,
        "candidate_id": candidate_id or f"{run_id}_{sequence}",
        "sequence": sequence,
        "ddg": -10.0,
        "ts": ts,
    }


class TestComputeRecoveryPlan:

    def test_no_missing_when_identical(self):
        records = [_candidate("AAA", "2026-01-01T00:00:00Z")]
        plan = compute_recovery_plan(records, records)
        assert plan["missing_records"] == []
        assert plan["missing_unique_sequences"] == set()
        assert len(plan["merged_records"]) == 1

    def test_detects_missing_suffix(self):
        """archive 사고 재현: archive는 A,B,C / current는 A만(B,C 유실)."""
        archive = [
            _candidate("AAA", "2026-01-01T00:00:00Z"),
            _candidate("BBB", "2026-01-01T00:01:00Z"),
            _candidate("CCC", "2026-01-01T00:02:00Z"),
        ]
        current = [_candidate("AAA", "2026-01-01T00:00:00Z")]
        plan = compute_recovery_plan(archive, current)
        assert plan["missing_unique_sequences"] == {"BBB", "CCC"}
        assert len(plan["missing_records"]) == 2
        assert len(plan["merged_records"]) == 3

    def test_merge_preserves_all_current_records(self):
        """복구 병합 후에도 current 에 이미 있던(archive에 없는, 재시작 후 신규) 레코드가 보존되어야 한다."""
        archive = [_candidate("AAA", "2026-01-01T00:00:00Z")]
        current = [
            _candidate("AAA", "2026-01-01T00:00:00Z"),
            _candidate("NEWSEQ", "2026-01-02T00:00:00Z", run_id="run2"),
        ]
        plan = compute_recovery_plan(archive, current)
        merged_sequences = {r["sequence"] for r in plan["merged_records"]}
        assert "NEWSEQ" in merged_sequences  # 재시작 후 신규 레코드도 보존
        assert "AAA" in merged_sequences

    def test_merged_records_sorted_by_ts(self):
        archive = [
            _candidate("BBB", "2026-01-01T00:05:00Z"),
            _candidate("AAA", "2026-01-01T00:01:00Z"),
        ]
        current: list = []
        plan = compute_recovery_plan(archive, current)
        ts_list = [r["ts"] for r in plan["merged_records"]]
        assert ts_list == sorted(ts_list)

    def test_exact_duplicate_not_double_counted(self):
        """완전히 동일한 (run_id, candidate_id, sequence, ts) 레코드는 병합 후에도 1건만 남는다."""
        rec = _candidate("AAA", "2026-01-01T00:00:00Z")
        archive = [rec]
        current = [dict(rec)]  # 완전 동일 사본
        plan = compute_recovery_plan(archive, current)
        assert plan["missing_records"] == []
        assert len(plan["merged_records"]) == 1

    def test_realistic_544_scenario_shape(self):
        """실제 사고 규모를 축소 재현: archive가 current보다 뒤쪽에 N건 더 많은 상황."""
        archive = [_candidate(f"SEQ{i}", f"2026-06-30T{i:02d}:00:00Z") for i in range(20)]
        current = archive[:15]  # 마지막 5건 유실 시뮬레이션
        plan = compute_recovery_plan(archive, current)
        assert len(plan["missing_records"]) == 5
        assert plan["missing_unique_sequences"] == {f"SEQ{i}" for i in range(15, 20)}
        assert len(plan["merged_records"]) == 20


class TestDryRunSafety:

    def test_dry_run_does_not_modify_current_file(self, tmp_path, capsys):
        archive_path = tmp_path / "archive.jsonl"
        current_path = tmp_path / "current.jsonl"
        archive_records = [
            _candidate("AAA", "2026-01-01T00:00:00Z"),
            _candidate("BBB", "2026-01-01T00:01:00Z"),
        ]
        current_records = [_candidate("AAA", "2026-01-01T00:00:00Z")]
        archive_path.write_text("\n".join(json.dumps(r) for r in archive_records) + "\n")
        current_path.write_text("\n".join(json.dumps(r) for r in current_records) + "\n")

        before_mtime = current_path.stat().st_mtime
        before_content = current_path.read_text()

        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "recover_experiment_log.py",
                "--archive", str(archive_path),
                "--current", str(current_path),
            ]  # --apply 없음 → dry-run
            rc = main()
        finally:
            sys.argv = old_argv

        assert rc == 0
        assert current_path.stat().st_mtime == before_mtime
        assert current_path.read_text() == before_content
        # 백업 파일도 생성되지 않아야 함
        backups = list(tmp_path.glob("current.jsonl.pre_recovery_backup_*.jsonl"))
        assert backups == []

        captured = capsys.readouterr()
        assert "복구 대상 레코드" in captured.out
        assert "1건" in captured.out  # BBB 1건 누락

    def test_apply_creates_backup_before_writing(self, tmp_path):
        archive_path = tmp_path / "archive.jsonl"
        current_path = tmp_path / "current.jsonl"
        archive_records = [
            _candidate("AAA", "2026-01-01T00:00:00Z"),
            _candidate("BBB", "2026-01-01T00:01:00Z"),
        ]
        current_records = [_candidate("AAA", "2026-01-01T00:00:00Z")]
        archive_path.write_text("\n".join(json.dumps(r) for r in archive_records) + "\n")
        current_path.write_text("\n".join(json.dumps(r) for r in current_records) + "\n")

        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "recover_experiment_log.py",
                "--archive", str(archive_path),
                "--current", str(current_path),
                "--apply",
            ]
            rc = main()
        finally:
            sys.argv = old_argv

        assert rc == 0
        backups = list(tmp_path.glob("current.jsonl.pre_recovery_backup_*.jsonl"))
        assert len(backups) == 1
        # 백업에는 원래 current(AAA 1건)만 있어야 함
        backup_records = _load_jsonl(backups[0])
        assert len(backup_records) == 1
        assert backup_records[0]["sequence"] == "AAA"

        # 최종 current 는 병합된 2건
        final_records = _load_jsonl(current_path)
        assert len(final_records) == 2
        final_sequences = {r["sequence"] for r in final_records}
        assert final_sequences == {"AAA", "BBB"}
