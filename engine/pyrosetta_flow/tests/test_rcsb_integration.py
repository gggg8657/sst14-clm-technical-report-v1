"""RCSB PDB 서열 검색 + runner 통합 테스트.

- Unit tests: mock으로 네트워크 없이 실행
- Integration tests (mark: network): 실제 RCSB API 호출 (CI에서 skip 가능)
"""
from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.rcsb_sequence_search import (
    RCSBSequenceSearcher,
    SequenceHit,
    SequenceSearchParams,
    SequenceSearchResult,
    _build_request_body,
    _parse_hit,
    _parse_identifier,
    _parse_match_context,
    search_similar_peptides,
)
from pyrosetta_flow.runner import _rcsb_check_candidates


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sst14_sequence() -> str:
    return "AGCKNFFWKTFTSC"


@pytest.fixture
def mock_rcsb_response() -> Dict[str, Any]:
    """RCSB API 응답 mock 데이터 (SST-14 검색 결과)."""
    return {
        "query_id": "test-query-id-001",
        "total_count": 2,
        "result_set": [
            {
                "identifier": "7T10_5",
                "score": 0.95,
                "services": [
                    {
                        "nodes": [
                            {
                                "match_context": [
                                    {
                                        "sequence_identity": 1.0,
                                        "evalue": 0.0002255,
                                        "bitscore": 37,
                                        "alignment_length": 14,
                                        "mismatches": 0,
                                        "gaps_opened": 0,
                                        "query_beg": 1,
                                        "query_end": 14,
                                        "subject_beg": 1,
                                        "subject_end": 14,
                                        "query_aligned_seq": "AGCKNFFWKTFTSC",
                                        "subject_aligned_seq": "AGCKNFFWKTFTSC",
                                    }
                                ]
                            }
                        ]
                    }
                ],
            },
            {
                "identifier": "2MI1_1",
                "score": 0.90,
                "services": [
                    {
                        "nodes": [
                            {
                                "match_context": [
                                    {
                                        "sequence_identity": 1.0,
                                        "evalue": 0.0002255,
                                        "bitscore": 37,
                                        "alignment_length": 14,
                                        "mismatches": 0,
                                        "gaps_opened": 0,
                                        "query_beg": 1,
                                        "query_end": 14,
                                        "subject_beg": 1,
                                        "subject_end": 14,
                                        "query_aligned_seq": "AGCKNFFWKTFTSC",
                                        "subject_aligned_seq": "AGCKNFFWKTFTSC",
                                    }
                                ]
                            }
                        ]
                    }
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Unit Tests: rcsb_sequence_search 모듈
# ---------------------------------------------------------------------------

class TestParseIdentifier:
    def test_polymer_entity(self):
        pdb_id, entity_id = _parse_identifier("7T10_5", "polymer_entity")
        assert pdb_id == "7T10"
        assert entity_id == "5"

    def test_polymer_instance(self):
        pdb_id, chain_id = _parse_identifier("7T10.E", "polymer_instance")
        assert pdb_id == "7T10"
        assert chain_id == "E"

    def test_entry(self):
        pdb_id, extra = _parse_identifier("7T10", "entry")
        assert pdb_id == "7T10"
        assert extra == ""

    def test_complex_pdb_id_with_underscore(self):
        """PDB ID에 _가 포함된 경우 rsplit으로 마지막 _ 기준 분리."""
        pdb_id, entity_id = _parse_identifier("4HHB_1", "polymer_entity")
        assert pdb_id == "4HHB"
        assert entity_id == "1"


class TestParseMatchContext:
    def test_verbose_context(self, mock_rcsb_response):
        raw = mock_rcsb_response["result_set"][0]
        ctx = _parse_match_context(raw)
        assert ctx["sequence_identity"] == 1.0
        assert ctx["evalue"] == 0.0002255
        assert ctx["bitscore"] == 37
        assert ctx["alignment_length"] == 14

    def test_empty_services(self):
        assert _parse_match_context({"services": []}) == {}

    def test_no_services_key(self):
        assert _parse_match_context({}) == {}

    def test_empty_nodes(self):
        assert _parse_match_context({"services": [{"nodes": []}]}) == {}

    def test_empty_match_context(self):
        raw = {"services": [{"nodes": [{"match_context": []}]}]}
        assert _parse_match_context(raw) == {}


class TestParseHit:
    def test_parse_full_hit(self, mock_rcsb_response):
        raw = mock_rcsb_response["result_set"][0]
        hit = _parse_hit(raw, "polymer_entity")
        assert hit.pdb_id == "7T10"
        assert hit.entity_or_chain_id == "5"
        assert hit.score == 0.95
        assert hit.sequence_identity == 1.0
        assert hit.identity_percent == 100.0
        assert hit.evalue == 0.0002255
        assert hit.alignment_length == 14
        assert hit.query_aligned_seq == "AGCKNFFWKTFTSC"

    def test_parse_minimal_hit(self):
        raw = {"identifier": "XXXX_1", "score": 0.5}
        hit = _parse_hit(raw, "polymer_entity")
        assert hit.pdb_id == "XXXX"
        assert hit.sequence_identity is None
        assert hit.identity_percent is None


class TestBuildRequestBody:
    def test_default_params(self, sst14_sequence):
        params = SequenceSearchParams(sequence=sst14_sequence)
        body = _build_request_body(params, start=0, rows=25)
        assert body["query"]["service"] == "sequence"
        assert body["query"]["parameters"]["value"] == sst14_sequence
        assert body["query"]["parameters"]["identity_cutoff"] == 0.5
        assert body["return_type"] == "polymer_entity"
        assert body["request_options"]["paginate"]["start"] == 0
        assert body["request_options"]["paginate"]["rows"] == 25

    def test_custom_params(self):
        params = SequenceSearchParams(
            sequence="ACDE",
            identity_cutoff=0.8,
            evalue_cutoff=1.0,
            return_type="entry",
            verbosity="minimal",
        )
        body = _build_request_body(params, start=10, rows=50)
        assert body["query"]["parameters"]["identity_cutoff"] == 0.8
        assert body["query"]["parameters"]["evalue_cutoff"] == 1.0
        assert body["return_type"] == "entry"
        assert body["request_options"]["results_verbosity"] == "minimal"
        assert body["request_options"]["paginate"]["start"] == 10


class TestSearcherMocked:
    """RCSBSequenceSearcher를 mock HTTP로 테스트."""

    def test_search_returns_results(self, sst14_sequence, mock_rcsb_response):
        searcher = RCSBSequenceSearcher()
        with patch.object(searcher, "_post_query", return_value=mock_rcsb_response):
            result = searcher.search(SequenceSearchParams(sequence=sst14_sequence))

        assert result.total_count == 2
        assert len(result.hits) == 2
        assert result.hits[0].pdb_id == "7T10"
        assert result.hits[1].pdb_id == "2MI1"
        assert result.query_id == "test-query-id-001"

    def test_search_empty_result(self, sst14_sequence):
        searcher = RCSBSequenceSearcher()
        empty_resp = {"query_id": "empty", "total_count": 0, "result_set": []}
        with patch.object(searcher, "_post_query", return_value=empty_resp):
            result = searcher.search(SequenceSearchParams(sequence=sst14_sequence))

        assert result.total_count == 0
        assert len(result.hits) == 0

    def test_search_max_results_limits_rows_param(self, sst14_sequence, mock_rcsb_response):
        """max_results가 _post_query의 rows 파라미터를 제한하는지 확인."""
        searcher = RCSBSequenceSearcher()
        with patch.object(searcher, "_post_query", return_value=mock_rcsb_response) as mock_post:
            searcher.search(SequenceSearchParams(
                sequence=sst14_sequence,
                max_results=1,
                rows_per_page=25,
            ))
            # rows는 min(rows_per_page, remaining)이므로 1이어야 함
            _, kwargs = mock_post.call_args
            assert kwargs["rows"] == 1

    def test_iter_all_hits(self, sst14_sequence, mock_rcsb_response):
        searcher = RCSBSequenceSearcher()
        with patch.object(searcher, "_post_query", return_value=mock_rcsb_response):
            hits = list(searcher.iter_all_hits(SequenceSearchParams(sequence=sst14_sequence)))

        assert len(hits) == 2


# ---------------------------------------------------------------------------
# Unit Tests: runner._rcsb_check_candidates (mocked)
# ---------------------------------------------------------------------------

class TestRCSBCheckCandidates:
    """runner._rcsb_check_candidates 통합 테스트 (mock)."""

    def _make_search_result(self, hits: list[SequenceHit]) -> SequenceSearchResult:
        return SequenceSearchResult(
            query_id="mock",
            total_count=len(hits),
            hits=hits,
            params=SequenceSearchParams(sequence="MOCK"),
        )

    def test_returns_matches(self):
        mock_hit = SequenceHit(
            identifier="7T10_5",
            pdb_id="7T10",
            entity_or_chain_id="5",
            score=0.95,
            sequence_identity=1.0,
            evalue=0.0002,
            bitscore=37,
        )
        mock_result = self._make_search_result([mock_hit])

        with patch("pyrosetta_flow.runner.search_similar_peptides", return_value=mock_result):
            results = _rcsb_check_candidates({"cand_001": "AGCKNFFWKTFTSC"})

        assert "cand_001" in results
        assert results["cand_001"][0]["pdb_id"] == "7T10"
        assert results["cand_001"][0]["identity"] == 1.0

    def test_empty_sequences_skipped(self):
        results = _rcsb_check_candidates({"cand_001": "", "cand_002": "AC"})
        assert results == {}

    def test_no_hits_returns_empty(self):
        empty_result = self._make_search_result([])
        with patch("pyrosetta_flow.runner.search_similar_peptides", return_value=empty_result):
            results = _rcsb_check_candidates({"cand_001": "XYZXYZXYZ"})

        assert results == {}

    def test_network_error_graceful(self):
        with patch("pyrosetta_flow.runner.search_similar_peptides", side_effect=Exception("Network unreachable")):
            results = _rcsb_check_candidates({"cand_001": "AGCKNFFWKTFTSC"})

        assert results == {}

    def test_empty_input(self):
        assert _rcsb_check_candidates({}) == {}

    def test_multiple_candidates(self):
        def mock_search(sequence, **kwargs):
            if "FFWK" in sequence:
                return self._make_search_result([
                    SequenceHit(identifier="7T10_5", pdb_id="7T10", entity_or_chain_id="5",
                                score=0.9, sequence_identity=0.8, evalue=0.001, bitscore=30),
                ])
            return self._make_search_result([])

        seqs = {
            "cand_001": "AGCKNFFWKTFTSC",
            "cand_002": "AAAAAAAAAAAAAA",
        }
        with patch("pyrosetta_flow.runner.search_similar_peptides", side_effect=mock_search):
            results = _rcsb_check_candidates(seqs)

        assert "cand_001" in results
        assert "cand_002" not in results


# ---------------------------------------------------------------------------
# Integration Tests: 실제 RCSB API 호출 (네트워크 필요)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestRCSBLive:
    """실제 RCSB API 호출 테스트. `pytest -m network`로 실행."""

    def test_sst14_search(self, sst14_sequence):
        result = search_similar_peptides(
            sequence=sst14_sequence,
            identity_cutoff=0.9,
            max_results=10,
        )
        assert result.total_count > 0
        pdb_ids = {h.pdb_id for h in result.hits}
        assert "7T10" in pdb_ids, "7T10 (SST-14+SSTR2 gold standard) should be in results"

    def test_rcsb_check_candidates_live(self, sst14_sequence):
        results = _rcsb_check_candidates(
            {"native": sst14_sequence},
            identity_cutoff=0.9,
            max_results=3,
        )
        assert "native" in results
        assert any(h["pdb_id"] == "7T10" for h in results["native"])

    def test_novel_mutant_no_exact_match(self):
        """충분히 다른 변이체는 100% identity match가 없어야 함."""
        results = _rcsb_check_candidates(
            {"mutant": "AGCKNAARKTAASC"},
            identity_cutoff=0.95,
            max_results=5,
        )
        if "mutant" in results:
            for hit in results["mutant"]:
                assert hit["identity"] < 1.0, "Novel mutant should not have exact PDB match"
