"""pdb_store 모듈 단위 테스트.

SQLite in-memory DB를 사용하므로 외부 의존성 없이 실행 가능합니다.
Biopython이 없는 환경에서는 RMSD 관련 테스트가 skip됩니다.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="pdb_store tests require SQLAlchemy")
from sqlalchemy.orm import Session

from pyrosetta_flow.pdb_store import (
    PDBRecord,
    _sha256,
    batch_rmsd_vs_native,
    bulk_register,
    compute_rmsd,
    export_to_dataframe,
    get_engine,
    get_top_candidates,
    import_from_jsonl,
    import_from_manifest,
    query_candidates,
    register_candidate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    """in-memory SQLite 엔진을 반환하고 테스트 후 dispose합니다."""
    eng = get_engine(":memory:")
    yield eng
    eng.dispose()


@pytest.fixture()
def session(engine):
    """테스트용 세션. 각 테스트 후 롤백 없이 사용합니다 (in-memory이므로 격리됨)."""
    with Session(engine) as sess:
        yield sess


@pytest.fixture()
def base_meta() -> dict[str, Any]:
    """experiment_log.jsonl 스타일의 기본 메타데이터."""
    return {
        "record_type": "candidate",
        "status": "success",
        "run_id": "test_run_001",
        "iteration": 1,
        "sequence": "AGCKNFFWKTFTSC",
        "ddg": -31.12,
        "clash": 0,
        "selected": True,
        "plddt": 85.0,
        "dock_score": -4.88,
        "lddt": 0.75,
        "final_score": 31.12,
        "error_summary": "",
        "ts": "2026-03-13T00:00:00+00:00",
    }


@pytest.fixture()
def tmp_pdb_file(tmp_path: Path) -> Path:
    """최소 유효 PDB 파일을 생성합니다."""
    pdb = tmp_path / "test_peptide.pdb"
    pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
        "ATOM      3  N   GLY A   2       2.916   0.000   0.000  1.00  0.00\n"
        "ATOM      4  CA  GLY A   2       4.374   0.000   0.000  1.00  0.00\n"
        "END\n",
        encoding="utf-8",
    )
    return pdb


# ---------------------------------------------------------------------------
# get_engine / 테이블 생성
# ---------------------------------------------------------------------------

class TestGetEngine:
    def test_creates_tables(self, engine):
        """엔진 생성 시 pdb_records 테이블이 자동 생성되어야 합니다."""
        with Session(engine) as sess:
            count = sess.query(PDBRecord).count()
        assert count == 0

    def test_wal_mode(self, tmp_path: Path):
        """파일 DB에서 WAL 모드가 활성화되어야 합니다. in-memory는 memory 모드."""
        import sqlalchemy as sa
        db_path = tmp_path / "wal_test.db"
        eng = get_engine(str(db_path))
        with eng.connect() as conn:
            result = conn.execute(sa.text("PRAGMA journal_mode")).fetchone()
        assert result[0] == "wal"
        eng.dispose()

    def test_file_db(self, tmp_path: Path):
        """파일 DB가 정상적으로 생성되어야 합니다."""
        db_path = tmp_path / "test.db"
        eng = get_engine(str(db_path))
        assert db_path.exists()
        eng.dispose()


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_consistent_hash(self, tmp_pdb_file: Path):
        """동일 파일은 동일 해시를 반환해야 합니다."""
        h1 = _sha256(tmp_pdb_file)
        h2 = _sha256(tmp_pdb_file)
        assert h1 is not None
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_nonexistent_file_returns_none(self, tmp_path: Path):
        """없는 파일은 None을 반환해야 합니다."""
        assert _sha256(tmp_path / "ghost.pdb") is None


# ---------------------------------------------------------------------------
# register_candidate
# ---------------------------------------------------------------------------

class TestRegisterCandidate:
    def test_basic_insert(self, session, base_meta):
        """기본 후보 등록이 정상적으로 이루어져야 합니다."""
        rec = register_candidate(session, "iter01_cand001", None, base_meta)
        assert rec.id == "iter01_cand001"
        assert rec.sequence == "AGCKNFFWKTFTSC"
        assert rec.ddg == pytest.approx(-31.12)
        assert rec.clash_count == 0
        assert rec.pass_gates is True

    def test_field_mapping_clash(self, session, base_meta):
        """clash 필드가 clash_count로 올바르게 매핑되어야 합니다."""
        base_meta["clash"] = 5
        rec = register_candidate(session, "test_clash", None, base_meta)
        assert rec.clash_count == 5

    def test_field_mapping_plddt(self, session, base_meta):
        """plddt 필드가 plddt_mean으로 올바르게 매핑되어야 합니다."""
        base_meta["plddt"] = 92.5
        rec = register_candidate(session, "test_plddt", None, base_meta)
        assert rec.plddt_mean == pytest.approx(92.5)

    def test_failed_status_sets_pass_gates_false(self, session, base_meta):
        """status='failed'인 레코드는 pass_gates=False로 강제 설정되어야 합니다."""
        base_meta["status"] = "failed"
        base_meta["selected"] = True  # selected=True라도 failed면 False
        base_meta["error_summary"] = "segfault"
        rec = register_candidate(session, "test_failed", None, base_meta)
        assert rec.pass_gates is False
        assert rec.fail_reasons == "segfault"

    def test_upsert_updates_existing(self, session, base_meta):
        """동일 candidate_id로 두 번 등록하면 갱신되어야 합니다."""
        register_candidate(session, "iter01_cand001", None, base_meta)
        base_meta["ddg"] = -50.0
        rec = register_candidate(session, "iter01_cand001", None, base_meta)
        assert rec.ddg == pytest.approx(-50.0)
        # DB에 중복 없이 1건만 존재
        count = session.query(PDBRecord).filter_by(id="iter01_cand001").count()
        assert count == 1

    def test_pdb_hash_computed_when_file_exists(self, session, base_meta, tmp_pdb_file):
        """PDB 파일이 존재하면 pdb_hash가 계산되어야 합니다."""
        rec = register_candidate(
            session, "test_hash", str(tmp_pdb_file), base_meta
        )
        assert rec.pdb_hash is not None
        assert len(rec.pdb_hash) == 64

    def test_pdb_hash_none_when_file_missing(self, session, base_meta, tmp_path):
        """PDB 파일이 없으면 pdb_hash가 None이어야 합니다."""
        rec = register_candidate(
            session, "test_no_hash", str(tmp_path / "ghost.pdb"), base_meta
        )
        assert rec.pdb_hash is None

    def test_pharmacological_fields(self, session, base_meta):
        """약리학적 프로퍼티 필드가 올바르게 저장되어야 합니다."""
        base_meta["gravy"] = -0.543
        base_meta["boman_index"] = 2.1
        base_meta["instability_idx"] = 42.3
        base_meta["mol_weight"] = 1628.9
        base_meta["net_charge"] = 1.0
        rec = register_candidate(session, "test_pharma", None, base_meta)
        assert rec.gravy == pytest.approx(-0.543)
        assert rec.boman_index == pytest.approx(2.1)
        assert rec.mol_weight == pytest.approx(1628.9)

    def test_structural_rules(self, session, base_meta):
        """구조 규칙 Boolean 필드가 올바르게 저장되어야 합니다."""
        base_meta["fwkt_conserved"] = True
        base_meta["ss_bond_intact"] = True
        base_meta["k9_d122_salt"] = False
        rec = register_candidate(session, "test_struct", None, base_meta)
        assert rec.fwkt_conserved is True
        assert rec.ss_bond_intact is True
        assert rec.k9_d122_salt is False


# ---------------------------------------------------------------------------
# bulk_register
# ---------------------------------------------------------------------------

class TestBulkRegister:
    def test_bulk_insert_multiple(self, session, base_meta):
        """여러 레코드 일괄 등록이 정상적으로 이루어져야 합니다."""
        records = []
        for i in range(1, 6):
            meta = dict(base_meta)
            meta["candidate_id"] = f"iter01_cand{i:03d}"
            meta["ddg"] = -float(i * 10)
            records.append(meta)
        result = bulk_register(session, records)
        assert len(result) == 5
        assert session.query(PDBRecord).count() == 5

    def test_bulk_register_uses_id_key(self, session, base_meta):
        """candidate_id 대신 id 키도 허용해야 합니다."""
        base_meta_copy = dict(base_meta)
        base_meta_copy.pop("candidate_id", None)
        base_meta_copy["id"] = "iter_id_key_001"
        result = bulk_register(session, [base_meta_copy])
        assert result[0].id == "iter_id_key_001"

    def test_bulk_register_raises_without_id(self, session, base_meta):
        """candidate_id/id가 없으면 ValueError를 발생시켜야 합니다."""
        bad_meta = {"sequence": "AGCKNFFWKTFTSC", "ddg": -10.0}
        with pytest.raises(ValueError, match="candidate_id"):
            bulk_register(session, [bad_meta])


# ---------------------------------------------------------------------------
# query_candidates
# ---------------------------------------------------------------------------

class TestQueryCandidates:
    @pytest.fixture(autouse=True)
    def _populate(self, session, base_meta):
        """테스트 데이터를 사전에 삽입합니다."""
        records = [
            {**base_meta, "candidate_id": "iter01_cand001", "ddg": -40.0,
             "iteration": 1, "pass_gates": True, "selected": True},
            {**base_meta, "candidate_id": "iter01_cand002", "ddg": -20.0,
             "iteration": 1, "pass_gates": True, "selected": True},
            {**base_meta, "candidate_id": "iter02_cand001", "ddg": 5.0,
             "iteration": 2, "pass_gates": False, "selected": False,
             "status": "failed"},
            {**base_meta, "candidate_id": "iter02_cand002", "ddg": -15.0,
             "iteration": 2, "pass_gates": True, "selected": True,
             "run_id": "other_run"},
        ]
        bulk_register(session, records)

    def test_filter_by_run_id(self, session):
        """run_id 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, run_id="other_run")
        assert len(results) == 1
        assert results[0].id == "iter02_cand002"

    def test_filter_by_iteration(self, session):
        """iteration 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, iteration=1)
        assert len(results) == 2
        ids = {r.id for r in results}
        assert "iter01_cand001" in ids

    def test_filter_by_ddg_max(self, session):
        """ddg_max 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, ddg_max=-25.0)
        assert all(r.ddg <= -25.0 for r in results)

    def test_filter_by_ddg_min(self, session):
        """ddg_min 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, ddg_min=0.0)
        assert all(r.ddg >= 0.0 for r in results)

    def test_filter_by_pass_gates_true(self, session):
        """pass_gates=True 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, pass_gates=True)
        assert all(r.pass_gates is True for r in results)

    def test_filter_combined(self, session):
        """복합 필터가 정상적으로 동작해야 합니다."""
        results = query_candidates(session, iteration=1, ddg_max=-25.0)
        assert len(results) == 1
        assert results[0].id == "iter01_cand001"

    def test_no_filter_returns_all(self, session):
        """필터 없이 전체 레코드를 반환해야 합니다."""
        results = query_candidates(session)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# get_top_candidates
# ---------------------------------------------------------------------------

class TestGetTopCandidates:
    @pytest.fixture(autouse=True)
    def _populate(self, session, base_meta):
        records = [
            {**base_meta, "candidate_id": f"cand{i:03d}", "ddg": float(-i * 5)}
            for i in range(1, 11)
        ]
        bulk_register(session, records)

    def test_returns_n_records(self, session):
        """n개 레코드를 반환해야 합니다."""
        results = get_top_candidates(session, n=5)
        assert len(results) == 5

    def test_sorted_by_ddg_ascending(self, session):
        """ddg 오름차순 정렬이 정상적으로 동작해야 합니다."""
        results = get_top_candidates(session, n=3, sort_by="ddg", ascending=True)
        ddgs = [r.ddg for r in results]
        assert ddgs == sorted(ddgs)

    def test_sorted_descending(self, session):
        """내림차순 정렬이 정상적으로 동작해야 합니다."""
        results = get_top_candidates(session, n=3, sort_by="ddg", ascending=False)
        ddgs = [r.ddg for r in results]
        assert ddgs == sorted(ddgs, reverse=True)

    def test_invalid_sort_by_raises(self, session):
        """유효하지 않은 컬럼명은 ValueError를 발생시켜야 합니다."""
        with pytest.raises(ValueError, match="알 수 없는 정렬 컬럼"):
            get_top_candidates(session, sort_by="nonexistent_column")


# ---------------------------------------------------------------------------
# import_from_jsonl
# ---------------------------------------------------------------------------

class TestImportFromJsonl:
    def test_imports_candidate_records(self, session, tmp_path, base_meta):
        """jsonl에서 candidate 레코드만 import되어야 합니다."""
        jsonl = tmp_path / "experiment_log.jsonl"
        lines = []
        for i in range(1, 4):
            record = {
                "record_type": "candidate",
                "status": "success",
                "run_id": "test_run",
                "iteration": 1,
                "candidate_id": f"iter01_cand{i:03d}",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": float(-i * 10),
                "clash": 0,
                "selected": True,
                "plddt": 85.0,
                "dock_score": -5.0,
                "final_score": float(i * 10),
                "error_summary": "",
                "ts": "2026-03-13T00:00:00+00:00",
            }
            lines.append(json.dumps(record))
        # iteration_summary 타입은 무시되어야 함
        lines.append(json.dumps({
            "record_type": "iteration_summary",
            "run_id": "test_run",
            "iteration": 1,
        }))
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        records = import_from_jsonl(session, jsonl)
        assert len(records) == 3
        assert session.query(PDBRecord).count() == 3

    def test_dedup_keeps_last_occurrence(self, session, tmp_path):
        """동일 candidate_id가 여러 줄에 있으면 마지막 값으로 갱신되어야 합니다."""
        jsonl = tmp_path / "log.jsonl"
        lines = []
        for ddg in [-10.0, -20.0, -30.0]:
            lines.append(json.dumps({
                "record_type": "candidate",
                "status": "success",
                "run_id": "r",
                "iteration": 1,
                "candidate_id": "dup_cand",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": ddg,
                "clash": 0,
                "selected": True,
                "plddt": 85.0,
                "dock_score": -5.0,
                "final_score": abs(ddg),
                "error_summary": "",
                "ts": "2026-03-13T00:00:00+00:00",
            }))
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        records = import_from_jsonl(session, jsonl)
        # 중복 제거 후 1건만 등록
        assert len(records) == 1
        assert records[0].ddg == pytest.approx(-30.0)

    def test_skips_malformed_json(self, session, tmp_path):
        """잘못된 JSON 줄은 조용히 건너뛰어야 합니다."""
        jsonl = tmp_path / "bad.jsonl"
        jsonl.write_text(
            "not json at all\n"
            + json.dumps({
                "record_type": "candidate",
                "status": "success",
                "run_id": "r",
                "iteration": 1,
                "candidate_id": "good_cand",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": -5.0,
                "clash": 0,
                "selected": True,
                "plddt": 85.0,
                "dock_score": -5.0,
                "final_score": 5.0,
                "error_summary": "",
                "ts": "2026-03-13T00:00:00+00:00",
            })
            + "\n",
            encoding="utf-8",
        )
        records = import_from_jsonl(session, jsonl)
        assert len(records) == 1

    def test_failed_records_mapped_correctly(self, session, tmp_path):
        """status='failed' 레코드가 pass_gates=False로 저장되어야 합니다."""
        jsonl = tmp_path / "failed.jsonl"
        jsonl.write_text(
            json.dumps({
                "record_type": "candidate",
                "status": "failed",
                "run_id": "r",
                "iteration": 1,
                "candidate_id": "failed_cand",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": 999.0,
                "clash": 999,
                "selected": False,
                "plddt": 0.0,
                "dock_score": 0.0,
                "final_score": -999.0,
                "error_summary": "Segfault",
                "ts": "2026-03-13T00:00:00+00:00",
            }),
            encoding="utf-8",
        )
        records = import_from_jsonl(session, jsonl)
        assert records[0].pass_gates is False
        assert records[0].fail_reasons == "Segfault"


# ---------------------------------------------------------------------------
# import_from_manifest
# ---------------------------------------------------------------------------

class TestImportFromManifest:
    def test_imports_from_manifest(self, session, tmp_path):
        """iteration_manifest.json에서 후보가 올바르게 import되어야 합니다."""
        manifest = {
            "run_id": "manifest_run",
            "iteration": 3,
            "candidates": [
                {
                    "candidate_id": f"iter03_cand{i:03d}",
                    "sequence": "AGCKNFFWKTFTSC",
                    "pdb_path": None,
                    "ddg": float(-i * 5),
                    "clash_count": 0,
                    "pass_gates": True,
                    "final_score": float(i * 5),
                }
                for i in range(1, 4)
            ],
        }
        manifest_path = tmp_path / "iteration_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        records = import_from_manifest(session, manifest_path)
        assert len(records) == 3
        # run_id가 자동으로 채워져야 함
        assert all(r.run_id == "manifest_run" for r in records)
        # iteration이 자동으로 채워져야 함
        assert all(r.iteration == 3 for r in records)

    def test_manifest_candidate_run_id_override(self, session, tmp_path):
        """후보 레코드에 run_id가 이미 있으면 그것을 유지해야 합니다."""
        manifest = {
            "run_id": "global_run",
            "iteration": 1,
            "candidates": [
                {
                    "candidate_id": "iter01_override",
                    "run_id": "local_run",  # 개별 run_id 지정
                    "sequence": "AGCKNFFWKTFTSC",
                    "ddg": -10.0,
                },
            ],
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        records = import_from_manifest(session, manifest_path)
        # 개별 레코드의 run_id가 전역 run_id보다 우선
        assert records[0].run_id == "local_run"


# ---------------------------------------------------------------------------
# export_to_dataframe
# ---------------------------------------------------------------------------

class TestExportToDataframe:
    @pytest.fixture(autouse=True)
    def _populate(self, session, base_meta):
        records = [
            {**base_meta, "candidate_id": f"cand{i:03d}", "ddg": float(-i * 10)}
            for i in range(1, 6)
        ]
        bulk_register(session, records)

    def test_returns_dataframe(self, session):
        """pandas DataFrame이 반환되어야 합니다."""
        pytest.importorskip("pandas")
        df = export_to_dataframe(session)
        assert hasattr(df, "shape")
        assert df.shape[0] == 5

    def test_filtered_dataframe(self, session):
        """필터가 적용된 DataFrame이 반환되어야 합니다."""
        pytest.importorskip("pandas")
        df = export_to_dataframe(session, ddg_max=-30.0)
        assert (df["ddg"] <= -30.0).all()

    def test_empty_dataframe_when_no_match(self, session):
        """조건에 맞는 레코드가 없으면 빈 DataFrame이 반환되어야 합니다."""
        pytest.importorskip("pandas")
        df = export_to_dataframe(session, ddg_max=-999.0)
        assert df.shape[0] == 0

    def test_raises_without_pandas(self, session, monkeypatch):
        """pandas가 없으면 ImportError가 발생해야 합니다."""
        import pyrosetta_flow.pdb_store as pdb_store_mod
        monkeypatch.setattr(pdb_store_mod, "_PANDAS_OK", False)
        with pytest.raises(ImportError, match="pandas"):
            export_to_dataframe(session)


# ---------------------------------------------------------------------------
# compute_rmsd — Biopython 의존
# ---------------------------------------------------------------------------

class TestComputeRmsd:
    @pytest.fixture()
    def simple_pdb(self, tmp_path: Path):
        """CA 원자 4개를 가진 최소 PDB 파일을 생성합니다."""
        def _make_pdb(name: str, offset: float) -> Path:
            p = tmp_path / name
            lines = []
            for i, (x, y, z) in enumerate(
                [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
                 (3.0, 0.0, 0.0), (4.5, 0.0, 0.0)],
                start=1,
            ):
                lines.append(
                    f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                    f"{x + offset:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
            lines.append("END\n")
            p.write_text("".join(lines), encoding="utf-8")
            return p
        return _make_pdb

    def test_zero_rmsd_identical(self, simple_pdb):
        """동일한 두 PDB는 RMSD가 0에 가까워야 합니다."""
        pytest.importorskip("Bio")
        pdb_a = simple_pdb("a.pdb", 0.0)
        pdb_b = simple_pdb("b.pdb", 0.0)
        rmsd = compute_rmsd(pdb_a, pdb_b, chain_id="A")
        assert rmsd == pytest.approx(0.0, abs=1e-5)

    def test_nonzero_rmsd_different(self, simple_pdb, tmp_path):
        """다른 두 PDB는 RMSD가 0보다 커야 합니다.

        Note: Superimposer는 균일 translation/rotation을 제거하므로,
        비균일 좌표 변형(distortion)을 사용해야 RMSD > 0이 됩니다.
        """
        pytest.importorskip("Bio")
        pdb_a = simple_pdb("a.pdb", 0.0)
        # 비균일 변형: 각 원자에 다른 크기의 offset 적용
        p = tmp_path / "b_distorted.pdb"
        lines = []
        for i, (x, y, z) in enumerate(
            [(0.0, 0.0, 0.0), (1.5, 0.5, 0.0),
             (3.0, 0.0, 1.0), (4.5, 0.0, 0.0)],
            start=1,
        ):
            lines.append(
                f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
            )
        lines.append("END\n")
        p.write_text("".join(lines), encoding="utf-8")
        rmsd = compute_rmsd(pdb_a, p, chain_id="A")
        assert rmsd > 0.0

    def test_raises_without_biopython(self, simple_pdb, monkeypatch):
        """Biopython이 없으면 ImportError가 발생해야 합니다."""
        import pyrosetta_flow.pdb_store as pdb_store_mod
        monkeypatch.setattr(pdb_store_mod, "_BIOPYTHON_OK", False)
        pdb_a = simple_pdb("a.pdb", 0.0)
        pdb_b = simple_pdb("b.pdb", 0.0)
        with pytest.raises(ImportError, match="biopython"):
            compute_rmsd(pdb_a, pdb_b)


# ---------------------------------------------------------------------------
# batch_rmsd_vs_native
# ---------------------------------------------------------------------------

class TestBatchRmsdVsNative:
    def test_raises_without_biopython(self, session, tmp_path, monkeypatch):
        """Biopython이 없으면 ImportError가 발생해야 합니다."""
        import pyrosetta_flow.pdb_store as pdb_store_mod
        monkeypatch.setattr(pdb_store_mod, "_BIOPYTHON_OK", False)
        with pytest.raises(ImportError, match="biopython"):
            batch_rmsd_vs_native(session, tmp_path / "native.pdb")

    def test_handles_missing_pdb_gracefully(self, session, base_meta, tmp_path):
        """PDB 파일이 없는 레코드는 None으로 처리되어야 합니다."""
        pytest.importorskip("Bio")
        base_meta["pdb_path"] = str(tmp_path / "ghost.pdb")
        register_candidate(session, "ghost_cand", str(tmp_path / "ghost.pdb"), base_meta)

        native = tmp_path / "native.pdb"
        native.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
            "END\n",
            encoding="utf-8",
        )
        results = batch_rmsd_vs_native(session, native)
        assert results["ghost_cand"] is None


# ---------------------------------------------------------------------------
# 통합 — jsonl import → query → top
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        """jsonl import → 조회 → Top-N까지의 전체 흐름을 검증합니다."""
        # jsonl 파일 생성
        jsonl = tmp_path / "log.jsonl"
        lines = []
        for i in range(1, 11):
            ddg = float(-i * 5) if i % 3 != 0 else float(i * 10)  # 일부는 양수
            lines.append(json.dumps({
                "record_type": "candidate",
                "status": "success" if ddg < 0 else "failed",
                "run_id": "integration_run",
                "iteration": (i - 1) // 5 + 1,
                "candidate_id": f"int_cand{i:03d}",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": ddg,
                "clash": 0 if ddg < 0 else 999,
                "selected": ddg < 0,
                "plddt": 85.0,
                "dock_score": -5.0,
                "final_score": abs(ddg),
                "error_summary": "" if ddg < 0 else "Segfault",
                "ts": "2026-03-13T00:00:00+00:00",
            }))
        jsonl.write_text("\n".join(lines), encoding="utf-8")

        db_path = tmp_path / "integration.db"
        engine = get_engine(str(db_path))
        with Session(engine) as session:
            records = import_from_jsonl(session, jsonl)
            assert len(records) == 10

            # 전체 조회
            all_recs = query_candidates(session)
            assert len(all_recs) == 10

            # pass_gates=True만 조회
            passed = query_candidates(session, pass_gates=True)
            assert all(r.pass_gates is True for r in passed)

            # Top-3 (ddg 기준)
            top3 = get_top_candidates(session, n=3, sort_by="ddg")
            assert len(top3) == 3
            ddgs = [r.ddg for r in top3]
            assert ddgs == sorted(ddgs)

        engine.dispose()
