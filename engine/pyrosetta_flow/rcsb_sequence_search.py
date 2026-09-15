"""RCSB PDB Sequence Similarity Search 모듈.

RCSB Search API v2 (https://search.rcsb.org/rcsbsearch/v2/query) 를 사용하여
펩타이드 서열 유사도 검색을 수행합니다.

내부적으로 MMseqs2 기반 BLAST-like 검색을 수행하며, 검색 결과로
PDB ID, Chain ID, 서열 동일성(%), e-value, bitscore 등을 반환합니다.

사용 예:
    from pyrosetta_flow.rcsb_sequence_search import RCSBSequenceSearcher, SequenceSearchParams

    searcher = RCSBSequenceSearcher()
    params = SequenceSearchParams(
        sequence="AGCKNFFWKTFTSC",
        identity_cutoff=0.5,
        evalue_cutoff=10.0,
    )
    results = searcher.search(params)
    for hit in results.hits:
        print(hit.identifier, hit.sequence_identity, hit.evalue)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------

# RCSB Search API v2 엔드포인트
RCSB_SEARCH_ENDPOINT = "https://search.rcsb.org/rcsbsearch/v2/query"

# 페이지네이션 최대 rows (API 제한: 10,000)
MAX_ROWS_PER_PAGE = 10_000

# 기본 요청 타임아웃 (초)
DEFAULT_TIMEOUT_SECONDS = 30

# 재시도 대기 시간 (초)
RETRY_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class SequenceSearchParams:
    """서열 유사도 검색 파라미터.

    Attributes:
        sequence: 검색 대상 아미노산 서열 (1문자 코드, 예: "AGCKNFFWKTFTSC")
        identity_cutoff: 최소 서열 동일성 임계값 (0.0~1.0, 기본 0.5 = 50%)
        evalue_cutoff: E-value 임계값 (클수록 관대한 검색, 기본 10.0)
        sequence_type: 서열 유형 ("protein" | "dna" | "rna")
        return_type: 반환 ID 단위
            - "polymer_entity" : PDB_ID_EntityID (예: 4HHB_1)  ← match_context 포함
            - "polymer_instance": PDB_ID.ChainID (예: 4HHB.A)
            - "entry"           : PDB ID만 (예: 4HHB)
        rows_per_page: 한 번에 가져올 결과 수 (기본 25, 최대 10,000)
        max_results: 최대 반환 결과 수 (None = 전체)
        verbosity: 응답 상세도 ("verbose" | "minimal" | "compact")
            verbose 일 때 match_context(identity/evalue/bitscore/alignment) 포함
    """
    sequence: str
    identity_cutoff: float = 0.5
    evalue_cutoff: float = 10.0
    sequence_type: str = "protein"
    return_type: str = "polymer_entity"
    rows_per_page: int = 25
    max_results: Optional[int] = None
    verbosity: str = "verbose"


@dataclass
class SequenceHit:
    """서열 검색 결과 한 건.

    Attributes:
        identifier: RCSB 식별자 (return_type에 따라 형식 상이)
            - polymer_entity  → "4HHB_1"    (PDB ID + '_' + entity ID)
            - polymer_instance→ "4HHB.A"    (PDB ID + '.' + chain ID)
            - entry           → "4HHB"
        pdb_id: PDB 4글자 코드
        entity_or_chain_id: entity ID 또는 chain ID 문자열 (파싱된 값)
        score: RCSB 정규화 점수 (0.0~1.0)
        sequence_identity: 서열 동일성 (0.0~1.0, verbose 모드에서만 제공)
        evalue: E-value (verbose 모드에서만 제공)
        bitscore: 비트스코어 (verbose 모드에서만 제공)
        alignment_length: 정렬 길이
        mismatches: 불일치 잔기 수
        gaps_opened: 갭 삽입 횟수
        query_beg: 쿼리 서열 정렬 시작 위치 (1-indexed)
        query_end: 쿼리 서열 정렬 종료 위치
        subject_beg: 히트 서열 정렬 시작 위치
        subject_end: 히트 서열 정렬 종료 위치
        query_aligned_seq: 정렬된 쿼리 서열 문자열
        subject_aligned_seq: 정렬된 히트 서열 문자열
    """
    identifier: str
    pdb_id: str
    entity_or_chain_id: str
    score: float = 0.0
    sequence_identity: Optional[float] = None
    evalue: Optional[float] = None
    bitscore: Optional[float] = None
    alignment_length: Optional[int] = None
    mismatches: Optional[int] = None
    gaps_opened: Optional[int] = None
    query_beg: Optional[int] = None
    query_end: Optional[int] = None
    subject_beg: Optional[int] = None
    subject_end: Optional[int] = None
    query_aligned_seq: Optional[str] = None
    subject_aligned_seq: Optional[str] = None

    @property
    def identity_percent(self) -> Optional[float]:
        """서열 동일성을 퍼센트(%)로 반환."""
        if self.sequence_identity is None:
            return None
        return self.sequence_identity * 100.0


@dataclass
class SequenceSearchResult:
    """서열 검색 전체 결과.

    Attributes:
        query_id: RCSB 서버 측 쿼리 UUID
        total_count: 조건에 부합하는 전체 히트 수 (페이지네이션과 무관)
        hits: 실제 반환된 SequenceHit 리스트
        params: 사용된 검색 파라미터
    """
    query_id: str
    total_count: int
    hits: list[SequenceHit]
    params: SequenceSearchParams


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------

def _build_request_body(
    params: SequenceSearchParams,
    start: int,
    rows: int,
) -> dict[str, Any]:
    """RCSB Search API 요청 바디 JSON을 생성합니다.

    Args:
        params: 검색 파라미터
        start: 페이지네이션 시작 오프셋 (0-indexed)
        rows: 이번 요청에서 가져올 결과 수

    Returns:
        dict: JSON 직렬화 가능한 요청 바디
    """
    return {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                # 서열 동일성 임계값 (0.0~1.0)
                "identity_cutoff": params.identity_cutoff,
                # E-value 임계값
                "evalue_cutoff": params.evalue_cutoff,
                # 서열 유형: "protein" | "dna" | "rna"
                "sequence_type": params.sequence_type,
                # 검색 쿼리 서열 (1문자 아미노산 코드)
                "value": params.sequence,
            },
        },
        # 반환 ID 단위: polymer_entity / polymer_instance / entry
        "return_type": params.return_type,
        "request_options": {
            # verbose: match_context(identity/evalue/bitscore/alignment) 포함
            "results_verbosity": params.verbosity,
            "paginate": {
                "start": start,
                "rows": rows,
            },
        },
    }


def _parse_identifier(identifier: str, return_type: str) -> tuple[str, str]:
    """RCSB 식별자에서 PDB ID와 entity/chain ID를 분리합니다.

    Args:
        identifier: RCSB 식별자 문자열
        return_type: 검색 시 사용한 return_type

    Returns:
        (pdb_id, entity_or_chain_id) 튜플

    Examples:
        >>> _parse_identifier("4HHB_1", "polymer_entity")
        ("4HHB", "1")
        >>> _parse_identifier("4HHB.A", "polymer_instance")
        ("4HHB", "A")
        >>> _parse_identifier("4HHB", "entry")
        ("4HHB", "")
    """
    if return_type == "polymer_entity" and "_" in identifier:
        # 형식: "PDBID_EntityID" (예: "4HHB_1", "7T10_5")
        pdb_id, entity_id = identifier.rsplit("_", 1)
        return pdb_id, entity_id
    elif return_type == "polymer_instance" and "." in identifier:
        # 형식: "PDBID.ChainID" (예: "4HHB.A", "2MI1.A")
        pdb_id, chain_id = identifier.rsplit(".", 1)
        return pdb_id, chain_id
    else:
        # 형식: "PDBID" (entry) 또는 파싱 불가
        return identifier, ""


def _parse_match_context(
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """result_set 항목에서 match_context 데이터를 추출합니다.

    verbose 모드에서 services[0].nodes[0].match_context[0] 경로에 위치합니다.

    Args:
        raw_result: result_set의 단일 항목 dict

    Returns:
        match_context dict (없으면 빈 dict)
    """
    try:
        services = raw_result.get("services", [])
        if not services:
            return {}
        nodes = services[0].get("nodes", [])
        if not nodes:
            return {}
        match_context_list = nodes[0].get("match_context", [])
        if not match_context_list:
            return {}
        return match_context_list[0]
    except (IndexError, KeyError, TypeError):
        return {}


def _parse_hit(raw_result: dict[str, Any], return_type: str) -> SequenceHit:
    """result_set 항목 하나를 SequenceHit 으로 파싱합니다.

    Args:
        raw_result: result_set의 단일 항목 dict
        return_type: 검색 시 사용한 return_type

    Returns:
        SequenceHit 인스턴스
    """
    identifier: str = raw_result.get("identifier", "")
    score: float = float(raw_result.get("score", 0.0))
    pdb_id, entity_or_chain_id = _parse_identifier(identifier, return_type)

    ctx = _parse_match_context(raw_result)

    return SequenceHit(
        identifier=identifier,
        pdb_id=pdb_id,
        entity_or_chain_id=entity_or_chain_id,
        score=score,
        sequence_identity=ctx.get("sequence_identity"),
        evalue=ctx.get("evalue"),
        bitscore=ctx.get("bitscore"),
        alignment_length=ctx.get("alignment_length"),
        mismatches=ctx.get("mismatches"),
        gaps_opened=ctx.get("gaps_opened"),
        query_beg=ctx.get("query_beg"),
        query_end=ctx.get("query_end"),
        subject_beg=ctx.get("subject_beg"),
        subject_end=ctx.get("subject_end"),
        query_aligned_seq=ctx.get("query_aligned_seq"),
        subject_aligned_seq=ctx.get("subject_aligned_seq"),
    )


# ---------------------------------------------------------------------------
# 메인 검색 클래스
# ---------------------------------------------------------------------------

class RCSBSequenceSearcher:
    """RCSB PDB 서열 유사도 검색기.

    RCSB Search API v2를 사용하여 MMseqs2 기반 BLAST-like 검색을 수행합니다.

    Args:
        timeout: HTTP 요청 타임아웃 (초, 기본 30)
        max_retries: 실패 시 재시도 횟수 (기본 3)
        session: 커스텀 requests.Session (없으면 내부 생성)

    Example:
        searcher = RCSBSequenceSearcher()
        results = searcher.search(SequenceSearchParams(
            sequence="AGCKNFFWKTFTSC",
            identity_cutoff=0.5,
        ))
        for hit in results.hits:
            print(f"{hit.pdb_id}  identity={hit.identity_percent:.1f}%  evalue={hit.evalue:.2e}")
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session or requests.Session()
        # User-Agent 설정 (API 에티켓)
        self._session.headers.update({
            "Content-Type": "application/json",
            "User-Agent": "PRST-N-FM/pyrosetta_flow rcsb_sequence_search.py",
        })

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def search(self, params: SequenceSearchParams) -> SequenceSearchResult:
        """서열 유사도 검색을 수행하고 전체 결과를 반환합니다.

        페이지네이션을 자동으로 처리하여 max_results 개수만큼 결과를 수집합니다.

        Args:
            params: 검색 파라미터

        Returns:
            SequenceSearchResult (total_count + hits 리스트)

        Raises:
            requests.HTTPError: API 응답이 4xx/5xx인 경우
            requests.Timeout: 타임아웃 초과
            ValueError: API 응답 파싱 실패
        """
        all_hits: list[SequenceHit] = []
        query_id: str = ""
        total_count: int = 0
        start: int = 0

        logger.info(
            "RCSB 서열 검색 시작: sequence=%s, identity_cutoff=%.2f, evalue_cutoff=%.1f",
            params.sequence,
            params.identity_cutoff,
            params.evalue_cutoff,
        )

        # 페이지네이션 루프
        while True:
            # 이번 페이지에서 요청할 rows 계산
            rows = params.rows_per_page
            if params.max_results is not None:
                remaining = params.max_results - len(all_hits)
                if remaining <= 0:
                    break
                rows = min(rows, remaining)

            # API 호출
            raw_response = self._post_query(params, start=start, rows=rows)

            # 첫 페이지에서 query_id, total_count 추출
            if start == 0:
                query_id = raw_response.get("query_id", "")
                total_count = int(raw_response.get("total_count", 0))
                logger.info("검색 완료: total_count=%d, query_id=%s", total_count, query_id)

            # result_set 파싱
            result_set: list[dict[str, Any]] = raw_response.get("result_set", [])
            if not result_set:
                break  # 결과 없음 또는 마지막 페이지

            for raw_hit in result_set:
                all_hits.append(_parse_hit(raw_hit, params.return_type))

            logger.debug(
                "페이지 수집: start=%d, 이번=%d, 누적=%d / 전체=%d",
                start, len(result_set), len(all_hits), total_count,
            )

            # 다음 페이지 여부 판단
            start += len(result_set)
            if start >= total_count:
                break  # 전체 수집 완료
            if params.max_results is not None and len(all_hits) >= params.max_results:
                break  # max_results 도달

        return SequenceSearchResult(
            query_id=query_id,
            total_count=total_count,
            hits=all_hits,
            params=params,
        )

    def iter_all_hits(
        self,
        params: SequenceSearchParams,
    ) -> Iterator[SequenceHit]:
        """페이지네이션 없이 전체 결과를 제너레이터로 스트리밍합니다.

        메모리 효율이 필요한 대량 결과 처리 시 사용합니다.

        Args:
            params: 검색 파라미터

        Yields:
            SequenceHit 인스턴스 (순서: RCSB 점수 내림차순)
        """
        start: int = 0
        total_count: Optional[int] = None

        # 페이지 크기를 최대로 설정하여 API 호출 횟수 최소화
        page_params = SequenceSearchParams(
            sequence=params.sequence,
            identity_cutoff=params.identity_cutoff,
            evalue_cutoff=params.evalue_cutoff,
            sequence_type=params.sequence_type,
            return_type=params.return_type,
            rows_per_page=min(params.rows_per_page, MAX_ROWS_PER_PAGE),
            verbosity=params.verbosity,
        )

        while True:
            raw_response = self._post_query(page_params, start=start, rows=page_params.rows_per_page)

            if total_count is None:
                total_count = int(raw_response.get("total_count", 0))

            result_set = raw_response.get("result_set", [])
            if not result_set:
                return

            for raw_hit in result_set:
                yield _parse_hit(raw_hit, params.return_type)

            start += len(result_set)
            if start >= total_count:
                return

    # ------------------------------------------------------------------
    # 내부 HTTP 요청
    # ------------------------------------------------------------------

    def _post_query(
        self,
        params: SequenceSearchParams,
        start: int,
        rows: int,
    ) -> dict[str, Any]:
        """단일 페이지 API 요청을 수행합니다 (재시도 로직 포함).

        Args:
            params: 검색 파라미터
            start: 페이지네이션 오프셋
            rows: 이번 요청 결과 수

        Returns:
            파싱된 JSON 응답 dict

        Raises:
            requests.HTTPError: 4xx/5xx 응답
            requests.Timeout: 타임아웃
            ValueError: JSON 파싱 실패
        """
        body = _build_request_body(params, start=start, rows=rows)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "API 요청 (시도 %d/%d): start=%d, rows=%d",
                    attempt + 1, self.max_retries, start, rows,
                )
                response = self._session.post(
                    RCSB_SEARCH_ENDPOINT,
                    data=json.dumps(body),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                # RCSB returns 204 No Content (or empty body) when no hits found
                if response.status_code == 204 or not response.text.strip():
                    return {"query_id": "", "total_count": 0, "result_set": []}
                return response.json()  # type: ignore[no-any-return]

            except requests.exceptions.Timeout as exc:
                logger.warning("타임아웃 (시도 %d/%d)", attempt + 1, self.max_retries)
                last_exc = exc
            except requests.exceptions.HTTPError as exc:
                # 4xx 클라이언트 오류는 재시도 불필요
                if response.status_code < 500:
                    raise
                logger.warning(
                    "서버 오류 %d (시도 %d/%d)",
                    response.status_code, attempt + 1, self.max_retries,
                )
                last_exc = exc
            except (requests.exceptions.RequestException, ValueError) as exc:
                logger.warning("요청 오류 (시도 %d/%d): %s", attempt + 1, self.max_retries, exc)
                last_exc = exc

            if attempt < self.max_retries - 1:
                wait_time = RETRY_BACKOFF_SECONDS * (2 ** attempt)
                logger.info("%.1f초 후 재시도...", wait_time)
                time.sleep(wait_time)

        raise last_exc or RuntimeError("알 수 없는 요청 오류")


# ---------------------------------------------------------------------------
# 편의 함수
# ---------------------------------------------------------------------------

def search_similar_peptides(
    sequence: str,
    identity_cutoff: float = 0.5,
    evalue_cutoff: float = 10.0,
    max_results: Optional[int] = 100,
    return_type: str = "polymer_entity",
) -> SequenceSearchResult:
    """SSTR2 결합 펩타이드 유사체 검색을 위한 편의 함수.

    Args:
        sequence: 쿼리 아미노산 서열 (예: "AGCKNFFWKTFTSC")
        identity_cutoff: 최소 서열 동일성 (0.0~1.0, 기본 0.5)
        evalue_cutoff: E-value 상한 (기본 10.0)
        max_results: 최대 반환 수 (None=전체)
        return_type: "polymer_entity" | "polymer_instance" | "entry"

    Returns:
        SequenceSearchResult

    Example:
        results = search_similar_peptides("AGCKNFFWKTFTSC", identity_cutoff=0.5)
        print(f"총 {results.total_count}개 히트 발견")
        for hit in results.hits:
            print(f"  {hit.pdb_id}: {hit.identity_percent:.1f}% identity, evalue={hit.evalue:.2e}")
    """
    searcher = RCSBSequenceSearcher()
    params = SequenceSearchParams(
        sequence=sequence,
        identity_cutoff=identity_cutoff,
        evalue_cutoff=evalue_cutoff,
        sequence_type="protein",
        return_type=return_type,
        rows_per_page=25,
        max_results=max_results,
        verbosity="verbose",
    )
    return searcher.search(params)


# ---------------------------------------------------------------------------
# CLI / 데모 실행
# ---------------------------------------------------------------------------

def _demo() -> None:
    """SST-14 서열(AGCKNFFWKTFTSC)로 SSTR2 결합 펩타이드 유사체 검색 데모."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # SST-14 native 서열
    query_seq = "AGCKNFFWKTFTSC"

    print("=" * 60)
    print(f"RCSB PDB 서열 유사도 검색 데모")
    print(f"쿼리 서열: {query_seq} ({len(query_seq)}aa)")
    print("=" * 60)

    # ── 검색 1: identity 50% 이상 ──────────────────────────────────
    print("\n[1] identity ≥ 50%, evalue ≤ 10.0 (polymer_entity)")
    results = search_similar_peptides(
        sequence=query_seq,
        identity_cutoff=0.5,
        evalue_cutoff=10.0,
        max_results=50,
        return_type="polymer_entity",
    )

    print(f"  query_id   : {results.query_id}")
    print(f"  total_count: {results.total_count} (반환: {len(results.hits)}개)")
    print()
    print(f"  {'식별자':<15} {'PDB':<6} {'Entity':<8} {'Identity%':>10} {'E-value':>12} {'Bitscore':>10} {'AlnLen':>8}")
    print("  " + "-" * 75)
    for hit in results.hits:
        identity_str = f"{hit.identity_percent:.1f}%" if hit.identity_percent is not None else "N/A"
        evalue_str   = f"{hit.evalue:.2e}"            if hit.evalue          is not None else "N/A"
        bitscore_str = f"{hit.bitscore:.1f}"          if hit.bitscore        is not None else "N/A"
        aln_str      = str(hit.alignment_length)       if hit.alignment_length is not None else "N/A"
        print(
            f"  {hit.identifier:<15} {hit.pdb_id:<6} {hit.entity_or_chain_id:<8} "
            f"{identity_str:>10} {evalue_str:>12} {bitscore_str:>10} {aln_str:>8}"
        )

    # ── 검색 2: chain ID 포함 (polymer_instance) ──────────────────
    print("\n[2] polymer_instance 반환 (chain ID 포함)")
    results2 = search_similar_peptides(
        sequence=query_seq,
        identity_cutoff=0.5,
        evalue_cutoff=10.0,
        max_results=10,
        return_type="polymer_instance",
    )
    print(f"  total_count: {results2.total_count}")
    for hit in results2.hits:
        print(f"  식별자: {hit.identifier}  (PDB={hit.pdb_id}, Chain={hit.entity_or_chain_id})")

    # ── 정렬 서열 출력 (상위 3개) ──────────────────────────────────
    print("\n[3] 상위 3개 히트 정렬 서열")
    for hit in results.hits[:3]:
        print(f"  [{hit.identifier}]")
        print(f"    Query  : {hit.query_aligned_seq}")
        print(f"    Subject: {hit.subject_aligned_seq}")
        print(f"    범위   : query {hit.query_beg}-{hit.query_end}, subject {hit.subject_beg}-{hit.subject_end}")
        print(f"    통계   : identity={hit.identity_percent:.1f}%, evalue={hit.evalue:.2e}, bitscore={hit.bitscore}")

    print("\n검색 완료.")


if __name__ == "__main__":
    _demo()
