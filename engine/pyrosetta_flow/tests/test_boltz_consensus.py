"""
test_boltz_consensus.py
========================
pyrosetta_flow/boltz_consensus.py 단위 테스트.

모의(mock) 데이터로 아티팩트 탐지·부호 일치·컨센서스 라벨·순위 차이 로직을 검증.
Boltz 실제 실행 없음 (subprocess 격리).
기존 테스트에 무영향.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.boltz_consensus import (
    BoltzResult,
    ConsensusRecord,
    ConsensusReport,
    LeaderboardEntry,
    _build_markdown,
    _compute_rank_diff,
    compare_ddg_boltz,
    detect_artifact,
    load_leaderboard_top_k,
    run_consensus_batch,
    save_report,
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

def _make_entry(
    seq: str = "AGCKNFFWKTFTSC",
    ddg: Optional[float] = -30.0,
    delta_margin: Optional[float] = 5.0,
    source: str = "silo_b",
    candidate_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> LeaderboardEntry:
    return LeaderboardEntry(
        sequence=seq,
        ddg=ddg,
        delta_margin=delta_margin,
        source=source,
        candidate_id=candidate_id,
        run_id=run_id,
    )


def _make_boltz_result(
    affinity: Optional[float] = -8.5,
    iptm: Optional[float] = 0.85,
    ptm: Optional[float] = 0.90,
    elapsed: float = 45.0,
    error: Optional[str] = None,
) -> BoltzResult:
    return BoltzResult(
        affinity_kcal_mol=affinity,
        ipTM=iptm,
        pTM=ptm,
        elapsed_s=elapsed,
        error=error,
    )


# ---------------------------------------------------------------------------
# detect_artifact 테스트
# ---------------------------------------------------------------------------

class TestDetectArtifact:
    def test_no_artifact_normal(self) -> None:
        """정상 후보: 아티팩트 없음."""
        is_art, reason = detect_artifact(ddg=-20.0, boltz_affinity=-8.0, ipTM=0.85)
        assert is_art is False
        assert reason == ""

    def test_artifact_polg_pattern(self) -> None:
        """poly-G 아티팩트: ddG 극단 강 + Boltz affinity 약함."""
        is_art, reason = detect_artifact(
            ddg=-40.0,           # < -30 임계값
            boltz_affinity=-2.0,  # > -5 (약함)
            ipTM=0.85,
        )
        assert is_art is True
        assert "poly-G/artifact" in reason
        assert "ddG=" in reason

    def test_artifact_low_iptm(self) -> None:
        """낮은 iPTM: 구조 신뢰도 부족."""
        is_art, reason = detect_artifact(
            ddg=-15.0,
            boltz_affinity=-7.0,
            ipTM=0.3,  # < 0.6 임계값
        )
        assert is_art is True
        assert "iPTM" in reason

    def test_artifact_both_rules(self) -> None:
        """두 규칙 동시 충족."""
        is_art, reason = detect_artifact(
            ddg=-50.0,
            boltz_affinity=-1.0,
            ipTM=0.2,
        )
        assert is_art is True
        assert "poly-G/artifact" in reason
        assert "iPTM" in reason

    def test_no_artifact_missing_ddg(self) -> None:
        """ddG 없음 → rule1 비활성."""
        is_art, reason = detect_artifact(ddg=None, boltz_affinity=-2.0, ipTM=0.85)
        assert is_art is False

    def test_no_artifact_missing_affinity(self) -> None:
        """Boltz affinity 없음 → rule1 비활성."""
        is_art, reason = detect_artifact(ddg=-50.0, boltz_affinity=None, ipTM=0.85)
        assert is_art is False

    def test_no_artifact_missing_iptm(self) -> None:
        """iPTM 없음 → rule2 비활성."""
        is_art, reason = detect_artifact(ddg=-15.0, boltz_affinity=-7.0, ipTM=None)
        assert is_art is False

    def test_custom_thresholds(self) -> None:
        """커스텀 임계값 적용: ddg=-20 < ddg_threshold=-15, affinity=-1.0 > affinity_weak=-3.0."""
        is_art, reason = detect_artifact(
            ddg=-20.0,
            boltz_affinity=-1.0,   # > affinity_weak(-3.0) → "약함"
            ipTM=0.85,
            ddg_threshold=-15.0,   # -20 < -15 → "극단 강함"
            affinity_weak=-3.0,    # -1.0 > -3.0 → "약함"
        )
        assert is_art is True
        assert "poly-G" in reason


# ---------------------------------------------------------------------------
# compare_ddg_boltz 테스트
# ---------------------------------------------------------------------------

class TestCompareDdgBoltz:
    def test_consistent_record(self) -> None:
        """ddG<0, Boltz affinity<0 → consistent."""
        entry = _make_entry(ddg=-25.0)
        boltz = _make_boltz_result(affinity=-9.0, iptm=0.88)
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.sign_agree is True
        assert rec.consensus_label == "consistent"
        assert rec.artifact_flag is False

    def test_discrepant_record(self) -> None:
        """ddG<0, Boltz affinity>0 → discrepant."""
        entry = _make_entry(ddg=-20.0)
        boltz = _make_boltz_result(affinity=2.0, iptm=0.85)
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.sign_agree is False
        assert rec.consensus_label == "discrepant"

    def test_artifact_record(self) -> None:
        """poly-G 패턴 → artifact 라벨."""
        entry = _make_entry(ddg=-45.0)
        boltz = _make_boltz_result(affinity=-1.0, iptm=0.88)
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.artifact_flag is True
        assert rec.consensus_label == "artifact"

    def test_missing_boltz_error(self) -> None:
        """Boltz 실패 → missing."""
        entry = _make_entry()
        boltz = _make_boltz_result(affinity=None, iptm=None, error="timeout")
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.consensus_label == "missing"
        assert rec.boltz_error == "timeout"

    def test_missing_boltz_no_affinity(self) -> None:
        """Boltz 성공(error=None), affinity=None이지만 iPTM 정상 → consistent.

        현 정책 (boltz_consensus.py compare_ddg_boltz 비교 전략):
          - affinity가 없어도 iPTM 기반으로 라벨을 결정할 수 있음.
          - Boltz error=None + iPTM 정상 + 아티팩트 없음 → consistent.
          - affinity=None은 "펩타이드는 affinity 지원 불가" 상황 — missing이 아닌 consistent.
          - missing은 error is not None OR iptm is None 일 때만.
        """
        entry = _make_entry()
        boltz = _make_boltz_result(affinity=None, iptm=0.80, error=None)
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.consensus_label == "consistent"
        assert rec.boltz_error is None
        assert rec.sign_agree is None  # affinity 없으면 부호 비교 불가

    def test_sign_agree_none_when_ddg_missing(self) -> None:
        """ddG 없으면 sign_agree=None."""
        entry = _make_entry(ddg=None)
        boltz = _make_boltz_result(affinity=-8.0)
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.sign_agree is None

    def test_record_has_disclaimer(self) -> None:
        """모든 레코드에 disclaimer 존재."""
        entry = _make_entry()
        boltz = _make_boltz_result()
        rec = compare_ddg_boltz(entry, boltz)
        assert "surrogate" in rec.disclaimer or "proxy" in rec.disclaimer

    def test_sequence_preserved(self) -> None:
        """서열이 레코드에 정확히 보존."""
        seq = "AGCKNFFWKTDTSC"
        entry = _make_entry(seq=seq)
        boltz = _make_boltz_result()
        rec = compare_ddg_boltz(entry, boltz)
        assert rec.sequence == seq


# ---------------------------------------------------------------------------
# _compute_rank_diff 테스트
# ---------------------------------------------------------------------------

class TestComputeRankDiff:
    def _make_records(self) -> list[ConsensusRecord]:
        """3개 레코드: ddg 순위와 affinity 순위 동일."""
        records = []
        data = [
            ("AAAA", -30.0, -10.0),  # rank 1 ddg, rank 1 aff → diff 0
            ("BBBB", -20.0, -8.0),   # rank 2 ddg, rank 2 aff → diff 0
            ("CCCC", -10.0, -6.0),   # rank 3 ddg, rank 3 aff → diff 0
        ]
        for seq, ddg, aff in data:
            entry = _make_entry(seq=seq, ddg=ddg)
            boltz = _make_boltz_result(affinity=aff)
            rec = compare_ddg_boltz(entry, boltz)
            records.append(rec)
        return records

    def test_rank_diff_zero_when_same_order(self) -> None:
        """ddG, Boltz affinity 순위 동일 → rank_diff=0."""
        records = self._make_records()
        _compute_rank_diff(records)
        for r in records:
            assert r.rank_diff == 0.0

    def test_rank_diff_nonzero_when_different_order(self) -> None:
        """순위 역전 → rank_diff > 0."""
        entry_a = _make_entry(seq="AAAA", ddg=-30.0)
        entry_b = _make_entry(seq="BBBB", ddg=-10.0)
        boltz_a = _make_boltz_result(affinity=-2.0)   # affinity 약함 → rank 2
        boltz_b = _make_boltz_result(affinity=-9.0)   # affinity 강함 → rank 1
        rec_a = compare_ddg_boltz(entry_a, boltz_a)
        rec_b = compare_ddg_boltz(entry_b, boltz_b)
        _compute_rank_diff([rec_a, rec_b])
        assert rec_a.rank_diff == 1.0
        assert rec_b.rank_diff == 1.0

    def test_rank_diff_none_when_missing(self) -> None:
        """Boltz 실패 후보는 rank_diff=None."""
        entry = _make_entry()
        boltz = _make_boltz_result(affinity=None, error="fail")
        rec = compare_ddg_boltz(entry, boltz)
        _compute_rank_diff([rec])
        assert rec.rank_diff is None


# ---------------------------------------------------------------------------
# load_leaderboard_top_k 테스트
# ---------------------------------------------------------------------------

class TestLoadLeaderboardTopK:
    def test_load_basic(self, tmp_path: Path) -> None:
        """기본 리더보드 로딩."""
        lb = {
            "entries": [
                {"sequence": "AAAA", "ddg": -30.0, "delta_margin": 5.0, "run_id": "r1"},
                {"sequence": "BBBB", "ddg": -20.0, "delta_margin": 3.0, "run_id": "r2"},
                {"sequence": "CCCC", "ddg": -10.0, "delta_margin": 1.0, "run_id": "r3"},
            ]
        }
        p = tmp_path / "lb.json"
        p.write_text(json.dumps(lb), encoding="utf-8")
        result = load_leaderboard_top_k(str(p), k=2, source_label="silo_b")
        assert len(result) == 2
        assert result[0].sequence == "AAAA"  # ddg 낮음 = 강한 결합 = 1위
        assert result[0].source == "silo_b"

    def test_load_missing_file(self) -> None:
        """파일 없음 → 빈 리스트 반환."""
        result = load_leaderboard_top_k("/nonexistent/path.json", k=5, source_label="silo_b")
        assert result == []

    def test_load_empty_entries(self, tmp_path: Path) -> None:
        """빈 entries → 빈 리스트."""
        p = tmp_path / "lb.json"
        p.write_text(json.dumps({"entries": []}), encoding="utf-8")
        result = load_leaderboard_top_k(str(p), k=5, source_label="silo_a")
        assert result == []

    def test_load_ddg_none_sorted_last(self, tmp_path: Path) -> None:
        """ddg=None 항목은 정렬 맨 뒤."""
        lb = {
            "entries": [
                {"sequence": "AAAA", "ddg": None},
                {"sequence": "BBBB", "ddg": -20.0},
            ]
        }
        p = tmp_path / "lb.json"
        p.write_text(json.dumps(lb), encoding="utf-8")
        result = load_leaderboard_top_k(str(p), k=2, source_label="silo_b")
        assert result[0].sequence == "BBBB"
        assert result[1].sequence == "AAAA"

    def test_load_silo_a_format(self, tmp_path: Path) -> None:
        """Silo A candidate_id 필드 파싱."""
        lb = {
            "entries": [
                {"candidate_id": "ca001", "sequence": "GGGGG", "ddg": -50.0, "delta_margin": None},
            ]
        }
        p = tmp_path / "silo_a.json"
        p.write_text(json.dumps(lb), encoding="utf-8")
        result = load_leaderboard_top_k(str(p), k=1, source_label="silo_a")
        assert len(result) == 1
        assert result[0].candidate_id == "ca001"
        assert result[0].source == "silo_a"


# ---------------------------------------------------------------------------
# ConsensusReport 테스트
# ---------------------------------------------------------------------------

class TestConsensusReport:
    def _make_report(self) -> ConsensusReport:
        records = [
            compare_ddg_boltz(_make_entry(ddg=-25.0), _make_boltz_result(affinity=-8.0, iptm=0.88)),
            compare_ddg_boltz(_make_entry(ddg=-45.0, seq="GGGGG"), _make_boltz_result(affinity=-1.0, iptm=0.85)),
            compare_ddg_boltz(_make_entry(ddg=-15.0, seq="AAAAA"), _make_boltz_result(affinity=None, error="fail")),
        ]
        report = ConsensusReport(
            generated_at="2026-06-19T00:00:00Z",
            silo_b_path="/path/b.json",
            silo_a_path="/path/a.json",
            k_top=10,
            cuda_device=2,
            boltz_env="boltz",
            records=records,
        )
        report.compute_summary()
        return report

    def test_summary_counts(self) -> None:
        """요약 카운트 정확성."""
        report = self._make_report()
        assert report.n_total == 3
        # GGGGG: artifact (ddg=-45, affinity=-1 → poly-G 의심)
        assert report.n_artifact >= 1
        # fail → missing
        assert report.n_missing == 1

    def test_save_report_creates_files(self, tmp_path: Path) -> None:
        """save_report가 파일을 생성."""
        report = self._make_report()
        saved = save_report(report, tmp_path)
        assert "all_json" in saved
        assert Path(saved["all_json"]).exists()
        assert "markdown" in saved
        assert Path(saved["markdown"]).exists()

    def test_save_report_json_parseable(self, tmp_path: Path) -> None:
        """저장된 JSON이 파싱 가능하고 disclaimer 포함."""
        report = self._make_report()
        saved = save_report(report, tmp_path)
        with open(saved["all_json"], encoding="utf-8") as f:
            data = json.load(f)
        assert "disclaimer" in data
        assert "candidates" in data
        assert len(data["candidates"]) == 3

    def test_markdown_contains_disclaimer(self, tmp_path: Path) -> None:
        """Markdown에 disclaimer 포함."""
        report = self._make_report()
        saved = save_report(report, tmp_path)
        md = Path(saved["markdown"]).read_text(encoding="utf-8")
        assert "surrogate" in md or "proxy" in md or "disclaimer" in md.lower()

    def test_markdown_contains_artifact_section(self, tmp_path: Path) -> None:
        """Markdown에 아티팩트 섹션 포함."""
        report = self._make_report()
        saved = save_report(report, tmp_path)
        md = Path(saved["markdown"]).read_text(encoding="utf-8")
        assert "아티팩트" in md


# ---------------------------------------------------------------------------
# run_consensus_batch 통합 모의 테스트
# ---------------------------------------------------------------------------

class TestRunConsensusBatch:
    def _make_leaderboard_file(self, tmp_path: Path, entries: list, name: str = "lb.json") -> str:
        p = tmp_path / name
        p.write_text(json.dumps({"entries": entries}), encoding="utf-8")
        return str(p)

    @patch("pyrosetta_flow.boltz_consensus.run_boltz_single")
    def test_batch_deduplication(self, mock_boltz: MagicMock, tmp_path: Path) -> None:
        """중복 서열이 한 번만 실행됨."""
        silo_b = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -30.0, "delta_margin": 5.0},
            {"sequence": "BBBB", "ddg": -20.0, "delta_margin": 3.0},
        ], "silo_b.json")
        # Silo A에 AAAA 중복
        silo_a = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -25.0, "delta_margin": 4.0},
        ], "silo_a.json")

        mock_boltz.return_value = _make_boltz_result()

        report = run_consensus_batch(
            silo_b_path=silo_b,
            silo_a_path=silo_a,
            work_dir=tmp_path / "work",
            k=10,
            cuda_device=2,
        )
        # 중복 제거 → 2개만 실행
        assert mock_boltz.call_count == 2
        assert report.n_total == 2

    @patch("pyrosetta_flow.boltz_consensus.run_boltz_single")
    def test_batch_all_missing(self, mock_boltz: MagicMock, tmp_path: Path) -> None:
        """모두 Boltz 실패 → n_missing = n_total."""
        silo_b = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -30.0},
        ])
        silo_a = self._make_leaderboard_file(tmp_path, [], name="silo_a.json")
        mock_boltz.return_value = _make_boltz_result(affinity=None, error="GPU 오류")

        report = run_consensus_batch(
            silo_b_path=silo_b,
            silo_a_path=silo_a,
            work_dir=tmp_path / "work",
            k=5,
            cuda_device=2,
        )
        assert report.n_missing == report.n_total
        assert report.n_total >= 1

    @patch("pyrosetta_flow.boltz_consensus.run_boltz_single")
    def test_batch_consistent_count(self, mock_boltz: MagicMock, tmp_path: Path) -> None:
        """정상 케이스: consistent 카운트 정확."""
        silo_b = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -25.0, "delta_margin": 5.0},
            {"sequence": "BBBB", "ddg": -20.0, "delta_margin": 3.0},
        ])
        silo_a = self._make_leaderboard_file(tmp_path, [], name="silo_a.json")
        mock_boltz.return_value = _make_boltz_result(affinity=-8.0, iptm=0.88)

        report = run_consensus_batch(
            silo_b_path=silo_b,
            silo_a_path=silo_a,
            work_dir=tmp_path / "work",
            k=5,
            cuda_device=2,
        )
        assert report.n_consistent == 2
        assert report.n_missing == 0

    @patch("pyrosetta_flow.boltz_consensus.run_boltz_single")
    def test_cuda_device_passed_correctly(self, mock_boltz: MagicMock, tmp_path: Path) -> None:
        """cuda_device=2가 run_boltz_single에 정확히 전달됨."""
        silo_b = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -20.0},
        ])
        silo_a = self._make_leaderboard_file(tmp_path, [], name="silo_a.json")
        mock_boltz.return_value = _make_boltz_result()

        run_consensus_batch(
            silo_b_path=silo_b,
            silo_a_path=silo_a,
            work_dir=tmp_path / "work",
            k=5,
            cuda_device=2,
        )
        call_kwargs = mock_boltz.call_args
        assert call_kwargs.kwargs.get("cuda_device") == 2

    @patch("pyrosetta_flow.boltz_consensus.run_boltz_single")
    def test_no_score_fabrication_on_failure(self, mock_boltz: MagicMock, tmp_path: Path) -> None:
        """Boltz 실패 후보에 임의 점수 지어내기 없음."""
        silo_b = self._make_leaderboard_file(tmp_path, [
            {"sequence": "AAAA", "ddg": -30.0},
        ])
        silo_a = self._make_leaderboard_file(tmp_path, [], name="silo_a.json")
        mock_boltz.return_value = BoltzResult(
            affinity_kcal_mol=None, ipTM=None, pTM=None, elapsed_s=1.0, error="CLI not found"
        )

        report = run_consensus_batch(
            silo_b_path=silo_b,
            silo_a_path=silo_a,
            work_dir=tmp_path / "work",
            k=5,
            cuda_device=2,
        )
        rec = report.records[0]
        assert rec.boltz_affinity is None     # 점수 지어내기 없음
        assert rec.ipTM is None
        assert rec.boltz_error is not None
        assert rec.consensus_label == "missing"
