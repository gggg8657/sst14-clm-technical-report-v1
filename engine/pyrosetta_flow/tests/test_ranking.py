"""Tests for ranking.py: JSONL I/O, historical candidate aggregation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pyrosetta_flow.ranking import (
    append_experiment_records,
    build_historical_candidates,
    check_experiment_log_shrinkage,
    extract_historical_sequences,
    load_experiment_records,
    load_experiment_records_guarded,
    summarize_top_hits,
)


# ===================================================================
# load_experiment_records tests  (Critical priority #3)
# ===================================================================

class TestLoadExperimentRecords:

    def test_missing_file(self, tmp_path):
        result = load_experiment_records(tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")
        assert load_experiment_records(f) == []

    def test_valid_records(self, tmp_path):
        f = tmp_path / "log.jsonl"
        records = [
            {"record_type": "candidate", "sequence": "AAA", "ddg": -5.0},
            {"record_type": "candidate", "sequence": "BBB", "ddg": -10.0},
        ]
        f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = load_experiment_records(f)
        assert len(loaded) == 2
        assert loaded[0]["sequence"] == "AAA"

    def test_malformed_json_tolerance(self, tmp_path):
        """Malformed lines should be silently skipped."""
        f = tmp_path / "log.jsonl"
        content = (
            '{"sequence": "AAA", "ddg": -5.0}\n'
            "THIS IS NOT JSON\n"
            '{"sequence": "BBB", "ddg": -10.0}\n'
        )
        f.write_text(content)
        loaded = load_experiment_records(f)
        assert len(loaded) == 2

    def test_blank_lines_skipped(self, tmp_path):
        f = tmp_path / "log.jsonl"
        content = (
            '{"a": 1}\n'
            "\n"
            "   \n"
            '{"b": 2}\n'
        )
        f.write_text(content)
        loaded = load_experiment_records(f)
        assert len(loaded) == 2


# ===================================================================
# append_experiment_records tests
# ===================================================================

class TestAppendExperimentRecords:

    def test_append_creates_file(self, tmp_path):
        f = tmp_path / "new_dir" / "log.jsonl"
        records = [{"seq": "AAA"}]
        append_experiment_records(f, records)
        assert f.exists()
        loaded = load_experiment_records(f)
        assert len(loaded) == 1

    def test_append_to_existing(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"a": 1}])
        append_experiment_records(f, [{"b": 2}])
        loaded = load_experiment_records(f)
        assert len(loaded) == 2

    def test_append_empty_records(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [])
        assert not f.exists()


# ===================================================================
# extract_historical_sequences tests
# ===================================================================

class TestExtractHistoricalSequences:

    def test_extract(self, sample_experiment_records):
        seqs = extract_historical_sequences(sample_experiment_records)
        assert isinstance(seqs, set)
        # All 4 candidate records have sequences
        assert len(seqs) == 4

    def test_non_candidate_records_ignored(self):
        records = [
            {"record_type": "summary", "sequence": "SHOULD_IGNORE"},
            {"record_type": "candidate", "sequence": "KEEP_ME"},
        ]
        seqs = extract_historical_sequences(records)
        assert seqs == {"KEEP_ME"}

    def test_empty_records(self):
        assert extract_historical_sequences([]) == set()


# ===================================================================
# summarize_top_hits tests  (Medium priority #10)
# ===================================================================

class TestSummarizeTopHits:

    def test_basic_top_hits(self, sample_experiment_records):
        hits = summarize_top_hits(sample_experiment_records, top_n=10)
        # Only records with status=success AND ddg < 0 qualify
        assert len(hits) == 2  # cand001 (ddg=-5.0), cand002 (ddg=-12.3)
        # Sorted by ddg ascending
        assert hits[0]["ddg"] <= hits[1]["ddg"]

    def test_top_n_limit(self, sample_experiment_records):
        hits = summarize_top_hits(sample_experiment_records, top_n=1)
        assert len(hits) == 1
        assert hits[0]["ddg"] == -12.3

    def test_no_successful_candidates(self):
        records = [
            {"record_type": "candidate", "status": "failed", "ddg": 999.0, "sequence": "X"},
        ]
        assert summarize_top_hits(records) == []

    def test_positive_ddg_excluded(self):
        records = [
            {"record_type": "candidate", "status": "success", "ddg": 5.0, "sequence": "X"},
        ]
        assert summarize_top_hits(records) == []


# ===================================================================
# build_historical_candidates tests  (Medium priority #10)
# ===================================================================

class TestBuildHistoricalCandidates:

    def test_basic_ranking(self, sample_experiment_records):
        result = build_historical_candidates(sample_experiment_records)
        assert len(result) == 4
        # rank 1 should be the best successful candidate
        assert result[0]["rank"] == 1
        assert result[0]["ddG"] == -12.3  # best ddg

    def test_success_before_failure(self, sample_experiment_records):
        result = build_historical_candidates(sample_experiment_records)
        # Successful candidates should rank before failed ones
        results = [r["result"] for r in result]
        # Find first FAIL index
        first_fail = results.index("FAIL") if "FAIL" in results else len(results)
        # All before first_fail should be PASS
        for i in range(first_fail):
            assert results[i] == "PASS"

    def test_limit(self, sample_experiment_records):
        result = build_historical_candidates(sample_experiment_records, limit=2)
        assert len(result) == 2

    def test_empty_records(self):
        assert build_historical_candidates([]) == []

    def test_final_score_calculation(self):
        records = [
            {
                "record_type": "candidate",
                "status": "success",
                "ddg": -10.0,
                "total_score": -100.0,
                "clash_score": 1.0,
                "candidate_id": "c1",
                "sequence": "AAA",
            },
        ]
        result = build_historical_candidates(records)
        assert result[0]["finalScore"] == round(10.0, 3)

    def test_positive_ddg_gets_zero_final_score(self):
        records = [
            {
                "record_type": "candidate",
                "status": "success",
                "ddg": 5.0,
                "total_score": -50.0,
                "clash_score": 2.0,
                "candidate_id": "c1",
                "sequence": "BBB",
            },
        ]
        result = build_historical_candidates(records)
        assert result[0]["finalScore"] == 0.0


# ===================================================================
# High-water-mark 데이터 유실 가드 tests
# (2026-07-01: continuous 엔진 재시작 시 experiment_log.jsonl 이 외부 요인
#  — 예: git checkout/reset 이 runs/ 데이터 디렉토리를 덮어씀 — 으로 축소되어
#  544개 고유서열 유실된 사고에 대한 회귀 테스트)
# ===================================================================

class TestHighWaterMarkGuard:

    def test_append_creates_hwm_sidecar(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"a": 1}, {"b": 2}])
        hwm = tmp_path / "log.jsonl.hwm.json"
        assert hwm.exists()
        assert json.loads(hwm.read_text())["line_count"] == 2

    def test_hwm_monotonic_increase(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"a": 1}])
        append_experiment_records(f, [{"b": 2}, {"c": 3}])
        hwm = tmp_path / "log.jsonl.hwm.json"
        assert json.loads(hwm.read_text())["line_count"] == 3

    def test_no_shrinkage_detected_when_growing(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"a": 1}])
        append_experiment_records(f, [{"b": 2}])
        assert check_experiment_log_shrinkage(f) is None

    def test_shrinkage_detected_after_external_truncation(self, tmp_path):
        """재시작 시뮬레이션: 정상 append 후, 외부 요인(git checkout 등)으로
        파일이 이전 상태로 되돌아가면(=축소) 다음 로드 시점에 반드시 탐지되어야 한다.
        """
        f = tmp_path / "log.jsonl"
        # 1차 실행: 5건 누적
        append_experiment_records(f, [{"seq": f"S{i}"} for i in range(5)])
        assert check_experiment_log_shrinkage(f) is None  # 정상 갱신 확인용 아님(마지막 append가 이미 hwm 갱신)

        # 외부 요인으로 파일이 3건짜리 구버전으로 되돌아감 (git checkout 시뮬레이션)
        f.write_text("\n".join(json.dumps({"seq": f"S{i}"}) for i in range(3)) + "\n")

        shrinkage = check_experiment_log_shrinkage(f)
        assert shrinkage is not None
        assert shrinkage["prior_line_count"] == 5
        assert shrinkage["current_line_count"] == 3
        assert shrinkage["missing"] == 2

    def test_no_shrinkage_on_first_run_without_hwm(self, tmp_path):
        """HWM sidecar가 아직 없는 최초 실행 시에는 축소로 오판하지 않는다."""
        f = tmp_path / "log.jsonl"
        f.write_text('{"seq": "X"}\n')
        assert check_experiment_log_shrinkage(f) is None

    def test_guarded_load_returns_records_even_on_shrinkage(self, tmp_path):
        """축소가 감지되어도 파이프라인이 죽지 않고(fail-open) 현재 파일 기준으로
        레코드를 반환해야 한다 — 단, 인시던트 마커는 남긴다."""
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"seq": f"S{i}"} for i in range(5)])
        f.write_text("\n".join(json.dumps({"seq": f"S{i}"}) for i in range(3)) + "\n")

        records = load_experiment_records_guarded(f)
        assert len(records) == 3  # 현재 파일 기준으로 정상 반환

        incident = tmp_path / "log.jsonl.SHRINKAGE_DETECTED.json"
        assert incident.exists()
        payload = json.loads(incident.read_text())
        assert payload["missing"] == 2

    def test_guarded_load_no_incident_marker_when_healthy(self, tmp_path):
        f = tmp_path / "log.jsonl"
        append_experiment_records(f, [{"seq": "S0"}])
        load_experiment_records_guarded(f)
        append_experiment_records(f, [{"seq": "S1"}])
        records = load_experiment_records_guarded(f)
        assert len(records) == 2
        incident = tmp_path / "log.jsonl.SHRINKAGE_DETECTED.json"
        assert not incident.exists()

    def test_restart_simulation_preserves_prior_records(self, tmp_path):
        """핵심 회귀 테스트: '재시작 시뮬레이션(기존 로그 있는 상태에서 flush) 후
        기존 레코드가 모두 보존되는지' 검증.

        시나리오: run A가 100건을 기록하고 종료(프로세스1). 이후 run B(warm-start,
        새 프로세스)가 시작되어 prior_records를 로드하고, 자신의 신규 레코드를
        append 한다. 최종 파일에는 run A의 100건 + run B의 신규분이 모두
        빠짐없이 존재해야 한다 (append-only 계약 검증).
        """
        f = tmp_path / "experiment_log.jsonl"

        # --- run A (프로세스1): 100건 기록 후 "종료" ---
        run_a_records = [{"record_type": "candidate", "sequence": f"SEQ_A{i}", "run_id": "runA"} for i in range(100)]
        append_experiment_records(f, run_a_records)
        assert len(load_experiment_records(f)) == 100

        # --- run B (프로세스2, warm-start): prior_records 로드 ---
        prior = load_experiment_records_guarded(f)
        assert len(prior) == 100
        assert check_experiment_log_shrinkage(f) is None

        # run B가 신규 레코드 20건 flush (iteration partial flush 시뮬레이션)
        run_b_records = [{"record_type": "candidate", "sequence": f"SEQ_B{i}", "run_id": "runB"} for i in range(20)]
        append_experiment_records(f, run_b_records)

        # --- 검증: run A의 100건이 전부 그대로 보존되고, run B의 20건이 추가됨 ---
        final_records = load_experiment_records(f)
        assert len(final_records) == 120
        final_sequences = {r["sequence"] for r in final_records}
        for i in range(100):
            assert f"SEQ_A{i}" in final_sequences, f"run A 레코드 SEQ_A{i} 유실됨"
        for i in range(20):
            assert f"SEQ_B{i}" in final_sequences, f"run B 레코드 SEQ_B{i} 유실됨"
        assert check_experiment_log_shrinkage(f) is None
