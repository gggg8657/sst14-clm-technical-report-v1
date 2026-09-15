"""PDB 메타데이터 관리 모듈 — SQLite + SQLAlchemy 기반.

experiment_log.jsonl, iteration_manifest.json으로부터 후보 레코드를 import하고
SHA-256 해시 무결성 검증, CA-RMSD 일괄 계산, Top-N 조회 등을 지원합니다.

사용 예:
    engine = get_engine("runs/pdb_store.db")
    with Session(engine) as session:
        register_candidate(session, "iter01_cand001", "path/to.pdb", meta)
        df = export_to_dataframe(session, pass_gates=True)
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]
    _PANDAS_OK = False

try:
    from Bio.PDB import PDBParser, Superimposer
    _BIOPYTHON_OK = True
except ImportError:  # pragma: no cover
    PDBParser = None  # type: ignore[assignment,misc]
    Superimposer = None  # type: ignore[assignment]
    _BIOPYTHON_OK = False

try:
    from sqlalchemy import (
        Boolean,
        Column,
        Integer,
        Float,
        Text,
        create_engine,
        event,
        text,
    )
    from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
    _SQLALCHEMY_OK = True
except ImportError:  # pragma: no cover
    Boolean = Column = Integer = Float = Text = None  # type: ignore[assignment]
    create_engine = event = text = None  # type: ignore[assignment]
    DeclarativeBase = object  # type: ignore[assignment,misc]
    Session = Any  # type: ignore[assignment]
    sessionmaker = None  # type: ignore[assignment]
    _SQLALCHEMY_OK = False


def _require_sqlalchemy() -> None:
    if not _SQLALCHEMY_OK:
        raise ImportError(
            "pdb_store database operations require SQLAlchemy: "
            "pip install sqlalchemy"
        )

# ---------------------------------------------------------------------------
# ORM 모델
# ---------------------------------------------------------------------------

if _SQLALCHEMY_OK:
    class _Base(DeclarativeBase):
        pass
else:
    class _Base:  # pragma: no cover
        metadata = None


if _SQLALCHEMY_OK:
    class PDBRecord(_Base):
        """PDB 후보 레코드 ORM 모델.

        experiment_log.jsonl의 ``candidate`` 레코드와 iteration_manifest.json의
        후보 항목 모두를 단일 테이블로 통합합니다.
        """

        __tablename__ = "pdb_records"

        # --- 기본 식별자 ---
        id = Column(Text, primary_key=True)          # candidate_id (e.g. "iter01_cand002")
        run_id = Column(Text)                        # 실험 실행 ID
        iteration = Column(Integer)                  # iteration 번호
        sequence = Column(Text)                      # 펩타이드 서열

        # --- 파일 경로 / 무결성 ---
        pdb_path = Column(Text)                      # PDB 파일 절대 경로
        pdb_hash = Column(Text)                      # SHA-256 해시
        receptor_pdb = Column(Text)                  # 수용체 PDB 경로

        # --- 도킹 결과 ---
        ddg = Column(Float)                           # ΔG (kcal/mol)
        clash_count = Column(Integer)                # 입체 충돌 수
        pass_gates = Column(Boolean)                 # QC 게이트 통과 여부
        fail_reasons = Column(Text)                  # 실패 사유 (nullable)
        final_score = Column(Float)                   # 최종 스코어
        plddt_mean = Column(Float)                    # pLDDT 평균
        dock_score = Column(Float)                    # 도킹 스코어

        # --- 약리학적 프로퍼티 (13가지, nullable) ---
        gravy = Column(Float)                         # GRAVY (Kyte-Doolittle)
        boman_index = Column(Float)                   # Boman Index
        instability_idx = Column(Float)               # Instability Index (Guruprasad)
        mol_weight = Column(Float)                    # 분자량 (Da)
        net_charge = Column(Float)                    # Net charge (pH 7.4)
        isoelectric_point = Column(Float)             # 등전점 (pI)
        aliphatic_index = Column(Float)               # Aliphatic Index (Ikai)
        hydrophobicity_ww = Column(Float)             # Wimley-White 소수성
        hydrophobicity_ei = Column(Float)             # Eisenberg 소수성
        boman_interaction = Column(Float)             # Boman interaction
        charge_density = Column(Float)                # 전하 밀도
        amphipathicity = Column(Float)                # 양친매성 지수
        hemolytic_score = Column(Float)               # 용혈 위험 스코어

        # --- 5가지 구조 규칙 (nullable Boolean) ---
        fwkt_conserved = Column(Boolean)             # FWKT pharmacophore 보존
        ss_bond_intact = Column(Boolean)             # Cys3-Cys14 이황화결합 유지
        k9_d122_salt = Column(Boolean)               # K9-D122 salt bridge
        phe_stacking = Column(Boolean)               # Phe6-Phe11 pi-stacking
        chelator_ready = Column(Boolean)             # N-term 킬레이터 준비 상태

        # --- 할루시네이션 검증 ---
        cross_validated = Column(Integer, default=0) # 0=pending, 1=verified, 2=flagged
        rmsd_vs_native = Column(Float)                # RMSD vs SST-14 native (Å)
        validation_note = Column(Text)               # 검증 메모

        # --- 메타 ---
        created_at = Column(Text, default=lambda: datetime.now(timezone.utc).isoformat())
        updated_at = Column(Text)

        def __repr__(self) -> str:  # pragma: no cover
            return (
                f"<PDBRecord id={self.id!r} seq={self.sequence!r} "
                f"ddg={self.ddg} pass={self.pass_gates}>"
            )
else:
    class PDBRecord:  # pragma: no cover
        """Placeholder when SQLAlchemy is unavailable."""


# ---------------------------------------------------------------------------
# 엔진 / 세션 팩토리
# ---------------------------------------------------------------------------

def get_engine(db_path: str | Path, echo: bool = False):
    """SQLite WAL 모드 엔진을 생성하고 테이블을 자동 생성합니다.

    Args:
        db_path: SQLite DB 파일 경로. ``:memory:`` 사용 가능.
        echo: True이면 SQL 로깅 활성화.

    Returns:
        SQLAlchemy Engine 인스턴스.
    """
    _require_sqlalchemy()
    path = str(db_path)
    url = f"sqlite:///{path}" if path != ":memory:" else "sqlite:///:memory:"
    engine = create_engine(url, echo=echo, future=True)

    # WAL 모드 활성화 (동시 읽기/쓰기 성능 향상)
    @event.listens_for(engine, "connect")
    def _set_wal(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    _Base.metadata.create_all(engine)
    return engine


def make_session(engine) -> Session:
    """세션 팩토리를 반환합니다."""
    _require_sqlalchemy()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _sha256(path: str | Path) -> Optional[str]:
    """파일의 SHA-256 해시를 계산합니다. 파일이 없으면 None 반환."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_PHARMA_FIELDS = (
    "gravy", "boman_index", "instability_idx", "mol_weight", "net_charge",
    "isoelectric_point", "aliphatic_index", "hydrophobicity_ww",
    "hydrophobicity_ei", "boman_interaction", "charge_density",
    "amphipathicity", "hemolytic_score",
)

_STRUCT_FIELDS = (
    "fwkt_conserved", "ss_bond_intact", "k9_d122_salt",
    "phe_stacking", "chelator_ready",
)

_VALIDATION_FIELDS = ("cross_validated", "rmsd_vs_native", "validation_note")


def parse_structural_rules_to_columns(
    structural_rules: dict,
) -> dict:
    """pharma.structural_rules dict → PDBRecord Boolean 컬럼 매핑.

    D2 패치 (2026-06-23): pharma_properties.check_structural_rules() 반환 형식에서
    5개 구조 규칙 Boolean을 PDBRecord 컬럼명으로 추출한다.

    check_structural_rules() 반환 형식::

        {
            "rules": {
                "fwkt_pharmacophore":  {"pass": bool, "detail": str},
                "cys3_cys14_disulfide": {"pass": bool, "detail": str},
                "k9_salt_bridge":      {"pass": bool, "detail": str},
                "phe6_phe11_stacking": {"pass": bool, "detail": str},
                "nterm_chelator":      {"pass": bool, "detail": str},
            },
            "all_pass": bool,
        }

    구현 정직성 주석:
        - 규칙 판정은 **서열 위치 기반** (잔기 종류 확인)이다.
        - PDB 3D 좌표 기반 실측 거리 계산(Asp122-Lys9 원자 거리 등)은 이 함수
          범위가 아니며, PyRosetta FlexPepDock 결과 PDB가 필요한 별도 구현 과제다.
        - 빈 dict 입력 시 모든 값이 None(nullable)으로 반환된다.

    Args:
        structural_rules: pharma dict의 ``"structural_rules"`` 키 값.

    Returns:
        PDBRecord 컬럼명 → Optional[bool] 매핑 dict:
            ``fwkt_conserved``, ``ss_bond_intact``, ``k9_d122_salt``,
            ``phe_stacking``, ``chelator_ready``
    """
    rules: dict = structural_rules.get("rules", {})
    return {
        "fwkt_conserved": rules.get("fwkt_pharmacophore", {}).get("pass"),
        "ss_bond_intact":  rules.get("cys3_cys14_disulfide", {}).get("pass"),
        "k9_d122_salt":    rules.get("k9_salt_bridge", {}).get("pass"),
        "phe_stacking":    rules.get("phe6_phe11_stacking", {}).get("pass"),
        "chelator_ready":  rules.get("nterm_chelator", {}).get("pass"),
    }

_ALL_OPTIONAL_FIELDS: tuple[str, ...] = (
    "receptor_pdb", "fail_reasons", "plddt_mean", "dock_score",
    *_PHARMA_FIELDS, *_STRUCT_FIELDS, *_VALIDATION_FIELDS,
)


def _build_record(
    candidate_id: str,
    pdb_path: Optional[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """meta dict로부터 PDBRecord 생성용 dict를 조립합니다.

    experiment_log.jsonl 필드명을 내부 컬럼명으로 매핑합니다:
        clash  → clash_count
        plddt  → plddt_mean
        error_summary / fail_reason → fail_reasons
        selected → pass_gates
    """
    now = _now_iso()

    # experiment_log.jsonl 필드 매핑
    clash_count_raw = meta.get("clash") if "clash" in meta else meta.get("clash_count")
    plddt_raw = meta.get("plddt") if "plddt" in meta else meta.get("plddt_mean")
    fail_raw = (
        meta.get("error_summary")
        or meta.get("fail_reason")
        or meta.get("fail_reasons")
        or ""
    )
    pass_gates_raw = meta.get("selected") if "selected" in meta else meta.get("pass_gates")

    # status가 "failed"이면 pass_gates=False로 강제
    if meta.get("status") == "failed":
        pass_gates_raw = False

    rec: dict[str, Any] = {
        "id": candidate_id,
        "run_id": meta.get("run_id"),
        "iteration": meta.get("iteration"),
        "sequence": meta.get("sequence"),
        "pdb_path": pdb_path,
        "pdb_hash": _sha256(pdb_path) if pdb_path else None,
        "ddg": meta.get("ddg"),
        "clash_count": int(clash_count_raw) if clash_count_raw is not None else None,
        "pass_gates": bool(pass_gates_raw) if pass_gates_raw is not None else None,
        "fail_reasons": fail_raw if fail_raw else None,
        "final_score": meta.get("final_score"),
        "plddt_mean": float(plddt_raw) if plddt_raw is not None else None,
        "dock_score": meta.get("dock_score"),
        "created_at": now,
        "updated_at": now,
    }

    # 선택적 필드 복사
    for field_name in _ALL_OPTIONAL_FIELDS:
        if field_name in meta:
            rec[field_name] = meta[field_name]

    return rec


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def register_candidate(
    session: Session,
    candidate_id: str,
    pdb_path: Optional[str],
    meta_dict: dict[str, Any],
) -> PDBRecord:
    """단일 후보를 DB에 등록합니다 (upsert 방식).

    동일 ``candidate_id``가 존재하면 갱신하고, 없으면 신규 삽입합니다.

    Args:
        session: SQLAlchemy 세션.
        candidate_id: 후보 ID (예: ``"iter01_cand002"``).
        pdb_path: PDB 파일 절대 경로. 파일이 없어도 등록 가능.
        meta_dict: 메타데이터 dict. experiment_log.jsonl 형식 호환.

    Returns:
        등록된 PDBRecord 인스턴스.
    """
    _require_sqlalchemy()
    rec_dict = _build_record(candidate_id, pdb_path, meta_dict)
    existing = session.get(PDBRecord, candidate_id)

    if existing is not None:
        # 갱신: None이 아닌 값만 덮어씀
        for k, v in rec_dict.items():
            if k == "created_at":
                continue  # 최초 등록 시각 유지
            if v is not None:
                setattr(existing, k, v)
        existing.updated_at = _now_iso()
        record = existing
    else:
        record = PDBRecord(**rec_dict)
        session.add(record)

    session.commit()
    return record


def bulk_register(
    session: Session,
    records: list[dict[str, Any]],
) -> list[PDBRecord]:
    """복수 후보를 일괄 등록합니다.

    각 레코드 dict에는 ``candidate_id`` 또는 ``id`` 키가 필요합니다.
    ``pdb_path`` 키가 있으면 해당 값을 사용하고, 없으면 None으로 처리합니다.

    Args:
        session: SQLAlchemy 세션.
        records: 메타데이터 dict 목록.

    Returns:
        등록된 PDBRecord 목록.
    """
    _require_sqlalchemy()
    result: list[PDBRecord] = []
    for meta in records:
        cid = meta.get("candidate_id") or meta.get("id")
        if not cid:
            raise ValueError(f"레코드에 candidate_id/id가 없습니다: {meta}")
        pdb_path = meta.get("pdb_path")
        result.append(register_candidate(session, cid, pdb_path, meta))
    return result


def query_candidates(
    session: Session,
    *,
    run_id: Optional[str] = None,
    iteration: Optional[int] = None,
    ddg_max: Optional[float] = None,
    ddg_min: Optional[float] = None,
    pass_gates: Optional[bool] = None,
    cross_validated: Optional[int] = None,
    sequence_contains: Optional[str] = None,
) -> list[PDBRecord]:
    """필터 조건으로 후보를 조회합니다.

    Args:
        session: SQLAlchemy 세션.
        run_id: 실험 run ID 필터.
        iteration: iteration 번호 필터.
        ddg_max: ΔG 상한값 (kcal/mol). 이 값 이하만 반환.
        ddg_min: ΔG 하한값 (kcal/mol). 이 값 이상만 반환.
        pass_gates: QC 게이트 통과 여부 필터.
        cross_validated: 검증 상태 필터 (0=pending, 1=verified, 2=flagged).
        sequence_contains: 서열에 포함된 부분 문자열 필터.

    Returns:
        조건을 만족하는 PDBRecord 목록.
    """
    _require_sqlalchemy()
    q = session.query(PDBRecord)

    if run_id is not None:
        q = q.filter(PDBRecord.run_id == run_id)
    if iteration is not None:
        q = q.filter(PDBRecord.iteration == iteration)
    if ddg_max is not None:
        q = q.filter(PDBRecord.ddg <= ddg_max)
    if ddg_min is not None:
        q = q.filter(PDBRecord.ddg >= ddg_min)
    if pass_gates is not None:
        q = q.filter(PDBRecord.pass_gates == pass_gates)
    if cross_validated is not None:
        q = q.filter(PDBRecord.cross_validated == cross_validated)
    if sequence_contains is not None:
        q = q.filter(PDBRecord.sequence.contains(sequence_contains))

    return q.all()


def get_top_candidates(
    session: Session,
    n: int = 10,
    sort_by: str = "ddg",
    ascending: bool = True,
) -> list[PDBRecord]:
    """Top N 후보를 정렬하여 반환합니다.

    Args:
        session: SQLAlchemy 세션.
        n: 반환할 후보 수.
        sort_by: 정렬 기준 컬럼명 (``"ddg"``, ``"final_score"``, ``"dock_score"`` 등).
        ascending: True이면 오름차순 (ddg의 경우 더 낮은 값이 더 좋음).

    Returns:
        정렬된 Top N PDBRecord 목록.
    """
    _require_sqlalchemy()
    col = getattr(PDBRecord, sort_by, None)
    if col is None:
        raise ValueError(f"알 수 없는 정렬 컬럼: {sort_by!r}")

    q = session.query(PDBRecord).filter(PDBRecord.ddg.isnot(None))
    if ascending:
        q = q.order_by(col.asc())
    else:
        q = q.order_by(col.desc())

    return q.limit(n).all()


# ---------------------------------------------------------------------------
# RMSD 계산
# ---------------------------------------------------------------------------

def compute_rmsd(
    pdb_a: str | Path,
    pdb_b: str | Path,
    chain_id: str = "A",
) -> float:
    """두 PDB 구조 간 CA-RMSD를 계산합니다.

    Biopython Superimposer를 사용합니다. Biopython이 설치되지 않은 경우
    ImportError를 발생시킵니다.

    Args:
        pdb_a: 기준 PDB 파일 경로.
        pdb_b: 비교 PDB 파일 경로.
        chain_id: CA 원자를 추출할 체인 ID.

    Returns:
        CA-RMSD 값 (Å).

    Raises:
        ImportError: Biopython이 설치되지 않은 경우.
        ValueError: CA 원자를 충분히 찾지 못한 경우.
    """
    if not _BIOPYTHON_OK:
        raise ImportError(
            "compute_rmsd 사용을 위해 biopython이 필요합니다: "
            "pip install biopython"
        )

    parser = PDBParser(QUIET=True)

    def _get_ca_atoms(path: str | Path, struct_id: str) -> list:
        structure = parser.get_structure(struct_id, str(path))
        atoms = []
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if "CA" in residue:
                            atoms.append(residue["CA"])
            break  # 첫 번째 모델만 사용
        return atoms

    atoms_a = _get_ca_atoms(pdb_a, "ref")
    atoms_b = _get_ca_atoms(pdb_b, "mob")

    n = min(len(atoms_a), len(atoms_b))
    if n < 1:
        raise ValueError(
            f"CA 원자를 찾지 못했습니다. chain_id={chain_id!r}, "
            f"파일A={pdb_a}, 파일B={pdb_b}"
        )

    sup = Superimposer()
    sup.set_atoms(atoms_a[:n], atoms_b[:n])
    return float(sup.rms)


def batch_rmsd_vs_native(
    session: Session,
    native_pdb: str | Path,
    chain_id: str = "A",
) -> dict[str, Optional[float]]:
    """모든 후보의 native 대비 CA-RMSD를 일괄 계산하고 DB를 갱신합니다.

    PDB 파일이 없거나 계산 실패 시 해당 후보는 건너뜁니다.

    Args:
        session: SQLAlchemy 세션.
        native_pdb: 기준(native) PDB 파일 경로.
        chain_id: CA 원자 추출 체인 ID.

    Returns:
        ``{candidate_id: rmsd_value}`` dict. 실패한 항목은 None.
    """
    _require_sqlalchemy()
    if not _BIOPYTHON_OK:
        raise ImportError(
            "batch_rmsd_vs_native 사용을 위해 biopython이 필요합니다: "
            "pip install biopython"
        )

    all_records = session.query(PDBRecord).filter(
        PDBRecord.pdb_path.isnot(None)
    ).all()

    results: dict[str, Optional[float]] = {}
    now = _now_iso()

    for record in all_records:
        try:
            rmsd = compute_rmsd(native_pdb, record.pdb_path, chain_id=chain_id)
            record.rmsd_vs_native = rmsd
            record.updated_at = now
            results[record.id] = rmsd
        except Exception as exc:  # noqa: BLE001
            results[record.id] = None
            note = f"RMSD 계산 실패: {exc}"
            record.validation_note = note

    session.commit()
    return results


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_from_manifest(
    session: Session,
    manifest_path: str | Path,
) -> list[PDBRecord]:
    """iteration_manifest.json으로부터 후보를 일괄 import합니다.

    Manifest 스키마 예상 구조::

        {
          "run_id": "...",
          "iteration": 1,
          "candidates": [
            {
              "candidate_id": "iter01_cand001",
              "sequence": "...",
              "pdb_path": "...",
              "ddg": -31.12,
              ...
            },
            ...
          ]
        }

    Args:
        session: SQLAlchemy 세션.
        manifest_path: iteration_manifest.json 파일 경로.

    Returns:
        등록된 PDBRecord 목록.
    """
    _require_sqlalchemy()
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    run_id = data.get("run_id", "")
    iteration = data.get("iteration")
    candidates: list[dict[str, Any]] = data.get("candidates", [])

    records: list[dict[str, Any]] = []
    for cand in candidates:
        meta = dict(cand)
        if run_id and "run_id" not in meta:
            meta["run_id"] = run_id
        if iteration is not None and "iteration" not in meta:
            meta["iteration"] = iteration
        records.append(meta)

    return bulk_register(session, records)


def import_from_jsonl(
    session: Session,
    jsonl_path: str | Path,
) -> list[PDBRecord]:
    """experiment_log.jsonl로부터 후보 레코드를 일괄 import합니다.

    ``record_type == "candidate"`` 인 줄만 처리합니다.
    동일 candidate_id가 여러 줄에 걸쳐 나타나는 경우
    (실험을 재실행한 케이스) 마지막 줄의 데이터로 upsert됩니다.

    Args:
        session: SQLAlchemy 세션.
        jsonl_path: experiment_log.jsonl 파일 경로.

    Returns:
        등록된 PDBRecord 목록 (중복 제거 후).
    """
    _require_sqlalchemy()
    jsonl_path = Path(jsonl_path)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()

    # candidate_id 별로 마지막 레코드만 유지 (재실행 중복 제거)
    seen: dict[str, dict[str, Any]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("record_type") != "candidate":
            continue
        cid = obj.get("candidate_id")
        if not cid:
            continue
        seen[cid] = obj

    records = list(seen.values())
    return bulk_register(session, records)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_to_dataframe(
    session: Session,
    **filters: Any,
):
    """필터 조건으로 조회한 결과를 pandas DataFrame으로 반환합니다.

    Args:
        session: SQLAlchemy 세션.
        **filters: ``query_candidates``와 동일한 키워드 인자.

    Returns:
        pandas DataFrame. pandas가 없으면 ImportError.

    Raises:
        ImportError: pandas가 설치되지 않은 경우.
    """
    _require_sqlalchemy()
    if not _PANDAS_OK:
        raise ImportError(
            "export_to_dataframe 사용을 위해 pandas가 필요합니다: "
            "pip install pandas"
        )

    records = query_candidates(session, **filters)
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        row: dict[str, Any] = {}
        for col in PDBRecord.__table__.columns:
            row[col.name] = getattr(rec, col.name)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_init(args: list[str]) -> None:
    """DB를 초기화합니다.

    사용법: python pdb_store.py init [db_path]
    """
    db_path = args[0] if args else "runs/pdb_store.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    print(f"DB 초기화 완료: {db_path}")
    print(f"  테이블: {list(_Base.metadata.tables.keys())}")
    engine.dispose()


def _cli_import_jsonl(args: list[str]) -> None:
    """experiment_log.jsonl을 import합니다.

    사용법: python pdb_store.py import-jsonl <jsonl_path> [db_path]
    """
    if not args:
        print("사용법: python pdb_store.py import-jsonl <jsonl_path> [db_path]")
        sys.exit(1)
    jsonl_path = args[0]
    db_path = args[1] if len(args) > 1 else "runs/pdb_store.db"

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    with Session(engine) as session:
        records = import_from_jsonl(session, jsonl_path)
    print(f"import 완료: {len(records)}개 레코드 → {db_path}")
    engine.dispose()


def _cli_import_manifest(args: list[str]) -> None:
    """iteration_manifest.json을 import합니다.

    사용법: python pdb_store.py import-manifest <manifest_path> [db_path]
    """
    if not args:
        print("사용법: python pdb_store.py import-manifest <manifest_path> [db_path]")
        sys.exit(1)
    manifest_path = args[0]
    db_path = args[1] if len(args) > 1 else "runs/pdb_store.db"

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    engine = get_engine(db_path)
    with Session(engine) as session:
        records = import_from_manifest(session, manifest_path)
    print(f"import 완료: {len(records)}개 레코드 → {db_path}")
    engine.dispose()


def _cli_stats(args: list[str]) -> None:
    """DB 통계를 출력합니다.

    사용법: python pdb_store.py stats [db_path]
    """
    db_path = args[0] if args else "runs/pdb_store.db"
    if not Path(db_path).exists():
        print(f"DB 파일이 없습니다: {db_path}")
        sys.exit(1)

    engine = get_engine(db_path)
    with Session(engine) as session:
        total = session.query(PDBRecord).count()
        passed = session.query(PDBRecord).filter(
            PDBRecord.pass_gates == True  # noqa: E712
        ).count()
        failed_status = session.query(PDBRecord).filter(
            PDBRecord.pass_gates == False  # noqa: E712
        ).count()
        pending = session.query(PDBRecord).filter(
            PDBRecord.cross_validated == 0
        ).count()
        verified = session.query(PDBRecord).filter(
            PDBRecord.cross_validated == 1
        ).count()
        flagged = session.query(PDBRecord).filter(
            PDBRecord.cross_validated == 2
        ).count()

        # Top-5 (ddg 기준)
        top5 = get_top_candidates(session, n=5, sort_by="ddg", ascending=True)

    print(f"=== PDB Store 통계: {db_path} ===")
    print(f"  전체 레코드  : {total}")
    print(f"  QC 통과      : {passed}")
    print(f"  QC 실패      : {failed_status}")
    print(f"  검증 대기    : {pending}")
    print(f"  검증 완료    : {verified}")
    print(f"  할루시네이션 : {flagged}")
    print()
    print("  [Top-5 by ΔG]")
    for i, rec in enumerate(top5, 1):
        print(
            f"  {i:2d}. {rec.id:<20s} "
            f"seq={rec.sequence!s:15s} "
            f"ddg={rec.ddg:>8.3f}  "
            f"clash={rec.clash_count}"
        )

    engine.dispose()


def _cli_top(args: list[str]) -> None:
    """Top N 후보를 출력합니다.

    사용법: python pdb_store.py top [n] [sort_by] [db_path]
    """
    n = int(args[0]) if len(args) > 0 else 10
    sort_by = args[1] if len(args) > 1 else "ddg"
    db_path = args[2] if len(args) > 2 else "runs/pdb_store.db"

    engine = get_engine(db_path)
    with Session(engine) as session:
        candidates = get_top_candidates(session, n=n, sort_by=sort_by)

    print(f"=== Top {n} ({sort_by} 기준) ===")
    for i, rec in enumerate(candidates, 1):
        print(
            f"  {i:2d}. {rec.id:<20s} "
            f"seq={rec.sequence!s:15s} "
            f"ddg={rec.ddg:>8.3f}  "
            f"pass={rec.pass_gates}"
        )
    engine.dispose()


_COMMANDS: dict[str, Any] = {
    "init": _cli_init,
    "import-jsonl": _cli_import_jsonl,
    "import-manifest": _cli_import_manifest,
    "stats": _cli_stats,
    "top": _cli_top,
}


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print("사용법: python pdb_store.py <command> [args...]")
        print("명령어:")
        print("  init [db_path]")
        print("  import-jsonl <jsonl_path> [db_path]")
        print("  import-manifest <manifest_path> [db_path]")
        print("  stats [db_path]")
        print("  top [n] [sort_by] [db_path]")
        sys.exit(0)

    cmd = argv[0]
    rest = argv[1:]

    handler = _COMMANDS.get(cmd)
    if handler is None:
        print(f"알 수 없는 명령어: {cmd!r}")
        print(f"사용 가능한 명령어: {list(_COMMANDS.keys())}")
        sys.exit(1)

    handler(rest)
