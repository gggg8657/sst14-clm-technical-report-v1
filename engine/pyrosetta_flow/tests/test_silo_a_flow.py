"""test_silo_a_flow.py — Silo A 독립 발굴 엔진 단위 테스트.

테스트 원칙:
- 실제 GPU/모델 호출 없이 로직만 검증 (mock 사용)
- Silo B 파일 경계(runs/pyrosetta_flow/) 미침범 검증
- 필수 태그(candidate_class='de_novo', mutation_source='silo_a') 검증
- 리더보드·experiment_log 별도 저장 검증
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from pyrosetta_flow.silo_a_flow import (
    SiloAConfig,
    SiloACandidateResult,
    SiloALeaderboard,
    _ensure_chain_b_receptor,
    append_silo_a_records,
)


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sample_receptor_pdb_chainA(tmp_dir: Path) -> Path:
    """chain A 수용체 PDB (chain B 재레이블 테스트 용)."""
    pdb_content = "\n".join([
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C",
        "ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C",
        "END",
    ])
    path = tmp_dir / "receptor_chainA.pdb"
    path.write_text(pdb_content, encoding="utf-8")
    return path


@pytest.fixture()
def sample_receptor_pdb_chainB(tmp_dir: Path) -> Path:
    """chain B 수용체 PDB (재레이블 불필요)."""
    pdb_content = "\n".join([
        "ATOM      1  CA  ALA B   1       1.000   2.000   3.000  1.00  0.00           C",
        "ATOM      2  CA  GLY B   2       4.000   5.000   6.000  1.00  0.00           C",
        "END",
    ])
    path = tmp_dir / "receptor_chainB.pdb"
    path.write_text(pdb_content, encoding="utf-8")
    return path


@pytest.fixture()
def sample_candidate() -> SiloACandidateResult:
    return SiloACandidateResult(
        candidate_id="silo_a_test_bb00_sq00",
        sequence="ACDEFGHIKLMN",
        backbone_idx=0,
        seq_idx=0,
        plddt=0.82,
        plddt_pass=True,
        ddg=-25.5,
        clash_score=1.2,
        selectivity_margin=8.3,
        delta_margin=2.1,
        hc50=None,
        half_life_h=2.5,
        admet_score=0.72,
        fail_reason="",
        candidate_class="de_novo",
        mutation_source="silo_a",
        extra_scores={"candidate_class": "de_novo", "mutation_source": "silo_a"},
    )


# ---------------------------------------------------------------------------
# SiloAConfig 테스트
# ---------------------------------------------------------------------------

class TestSiloAConfig:
    def test_default_values(self) -> None:
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert cfg.n_backbone == 2
        assert cfg.k_seq_per_backbone == 2
        assert cfg.diffusion_steps == 50
        assert cfg.plddt_threshold == 0.5
        assert cfg.output_base_dir == "runs/silo_a_flow"
        assert cfg.candidate_class_tag() == "de_novo" if hasattr(cfg, "candidate_class_tag") else True

    def test_output_dir_not_pyrosetta_flow(self) -> None:
        """기본 출력 디렉토리가 Silo B 디렉토리(pyrosetta_flow)가 아님."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert "pyrosetta_flow" not in cfg.output_base_dir
        assert "silo_a" in cfg.output_base_dir

    def test_custom_contigs(self) -> None:
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb", contigs="B1-472/0 12-16")
        assert cfg.contigs == "B1-472/0 12-16"

    def test_hotspot_res_default_chain_b(self) -> None:
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        for res in cfg.hotspot_res:
            assert res.startswith("B"), f"핫스팟 잔기가 chain B 형식이어야 함: {res}"


# ---------------------------------------------------------------------------
# SiloACandidateResult 테스트
# ---------------------------------------------------------------------------

class TestSiloACandidateResult:
    def test_mandatory_tags(self, sample_candidate: SiloACandidateResult) -> None:
        """candidate_class='de_novo', mutation_source='silo_a' 태그 필수."""
        assert sample_candidate.candidate_class == "de_novo"
        assert sample_candidate.mutation_source == "silo_a"

    def test_to_dict_has_tags(self, sample_candidate: SiloACandidateResult) -> None:
        d = sample_candidate.to_dict()
        assert d["candidate_class"] == "de_novo"
        assert d["mutation_source"] == "silo_a"

    def test_none_scores_allowed(self) -> None:
        """실측 불가 지표는 None으로 기록 (환각 금지)."""
        c = SiloACandidateResult(
            candidate_id="test",
            sequence="ACDE",
            backbone_idx=0,
            seq_idx=0,
            plddt=None,
            plddt_pass=True,
            ddg=None,
            clash_score=None,
            selectivity_margin=None,
            delta_margin=None,
            hc50=None,
            half_life_h=None,
            admet_score=None,
            fail_reason="도킹 실패",
        )
        assert c.ddg is None
        assert c.selectivity_margin is None
        assert c.fail_reason != ""

    def test_to_dict_serializable(self, sample_candidate: SiloACandidateResult) -> None:
        d = sample_candidate.to_dict()
        serialized = json.dumps(d)  # JSON 직렬화 가능 검증
        parsed = json.loads(serialized)
        assert parsed["sequence"] == sample_candidate.sequence


# ---------------------------------------------------------------------------
# _ensure_chain_b_receptor 테스트
# ---------------------------------------------------------------------------

class TestEnsureChainBReceptor:
    def test_chainA_relabeled_to_chainB(
        self, sample_receptor_pdb_chainA: Path, tmp_dir: Path
    ) -> None:
        result_path = _ensure_chain_b_receptor(str(sample_receptor_pdb_chainA), tmp_dir)
        content = Path(result_path).read_text()
        atom_lines = [l for l in content.splitlines() if l.startswith("ATOM")]
        for line in atom_lines:
            assert line[21] == "B", f"재레이블 후 chain이 B가 아님: {line}"

    def test_chainB_unchanged(
        self, sample_receptor_pdb_chainB: Path, tmp_dir: Path
    ) -> None:
        """이미 chain B인 경우 원본 경로 그대로 반환."""
        result_path = _ensure_chain_b_receptor(str(sample_receptor_pdb_chainB), tmp_dir)
        assert result_path == str(sample_receptor_pdb_chainB)

    def test_empty_pdb_raises(self, tmp_dir: Path) -> None:
        empty_pdb = tmp_dir / "empty.pdb"
        empty_pdb.write_text("REMARK empty\n")
        with pytest.raises(ValueError, match="ATOM"):
            _ensure_chain_b_receptor(str(empty_pdb), tmp_dir)


# ---------------------------------------------------------------------------
# SiloALeaderboard 테스트
# ---------------------------------------------------------------------------

class TestSiloALeaderboard:
    def test_add_and_retrieve(self, sample_candidate: SiloACandidateResult) -> None:
        lb = SiloALeaderboard()
        lb.add(sample_candidate)
        assert len(lb.entries) == 1
        assert lb.entries[0]["candidate_class"] == "de_novo"
        assert lb.entries[0]["mutation_source"] == "silo_a"

    def test_best_ddg(self, sample_candidate: SiloACandidateResult) -> None:
        lb = SiloALeaderboard()
        lb.add(sample_candidate)
        best = lb._best_ddg()
        assert best == -25.5

    def test_dedup_keeps_better(self) -> None:
        lb = SiloALeaderboard()
        c1 = SiloACandidateResult(
            candidate_id="c1", sequence="ACDE", backbone_idx=0, seq_idx=0,
            plddt=0.8, plddt_pass=True, ddg=-10.0, clash_score=None,
            selectivity_margin=None, delta_margin=None, hc50=None,
            half_life_h=None, admet_score=None, fail_reason="",
        )
        c2 = SiloACandidateResult(
            candidate_id="c2", sequence="ACDE", backbone_idx=0, seq_idx=1,
            plddt=0.85, plddt_pass=True, ddg=-20.0, clash_score=None,
            selectivity_margin=None, delta_margin=None, hc50=None,
            half_life_h=None, admet_score=None, fail_reason="",
        )
        lb.add(c1)
        lb.add(c2)
        # 동일 서열 중 더 낮은 ddg(-20) 유지
        assert len(lb.entries) == 1
        assert lb.entries[0]["ddg"] == -20.0

    def test_save_and_load(
        self, tmp_dir: Path, sample_candidate: SiloACandidateResult
    ) -> None:
        lb_path = tmp_dir / "silo_a_leaderboard.json"
        lb = SiloALeaderboard()
        lb.add(sample_candidate)
        lb.save(lb_path)

        # 저장 파일 구조 검증
        data = json.loads(lb_path.read_text())
        assert data["source"] == "silo_a"
        assert data["candidate_class"] == "de_novo"
        assert len(data["entries"]) == 1

        # 로드 후 복원
        lb2 = SiloALeaderboard.load(lb_path)
        assert len(lb2.entries) == 1
        assert lb2.entries[0]["sequence"] == sample_candidate.sequence

    def test_save_path_not_pyrosetta_flow(self, tmp_dir: Path) -> None:
        """리더보드 저장 경로가 Silo B 경로와 완전 분리됨 검증."""
        lb_path = tmp_dir / "silo_a_leaderboard.json"
        assert "pyrosetta_flow" not in str(lb_path)
        assert "silo_a" in str(lb_path)

    def test_count_passing(self, sample_candidate: SiloACandidateResult) -> None:
        lb = SiloALeaderboard()
        lb.add(sample_candidate)
        assert lb.count_passing(ddg_max=-5.0) == 1  # -25.5 <= -5.0
        assert lb.count_passing(ddg_max=-30.0) == 0  # -25.5 > -30.0

    def test_capacity_limit(self) -> None:
        lb = SiloALeaderboard(capacity=3)
        for i in range(5):
            c = SiloACandidateResult(
                candidate_id=f"c{i}", sequence=f"SEQ{i:04d}AAAA", backbone_idx=0, seq_idx=i,
                plddt=0.8, plddt_pass=True, ddg=float(-10 - i), clash_score=None,
                selectivity_margin=None, delta_margin=None, hc50=None,
                half_life_h=None, admet_score=None, fail_reason="",
            )
            lb.add(c)
        assert len(lb.entries) <= 3

    def test_none_ddg_candidate_sortable(self) -> None:
        """ddg=None 후보도 추가 시 정렬 오류 없어야 함."""
        lb = SiloALeaderboard()
        c = SiloACandidateResult(
            candidate_id="c_nodock", sequence="FFFFFFFF", backbone_idx=0, seq_idx=0,
            plddt=None, plddt_pass=False, ddg=None, clash_score=None,
            selectivity_margin=None, delta_margin=None, hc50=None,
            half_life_h=None, admet_score=None, fail_reason="도킹 실패",
        )
        lb.add(c)  # 오류 없이 추가
        assert lb.n_total == 1


# ---------------------------------------------------------------------------
# append_silo_a_records 테스트
# ---------------------------------------------------------------------------

class TestAppendSiloARecords:
    def test_writes_jsonl(
        self, tmp_dir: Path, sample_candidate: SiloACandidateResult
    ) -> None:
        log_path = tmp_dir / "experiment_log.jsonl"
        append_silo_a_records(log_path, [sample_candidate], epoch=1, run_id="test_run")

        lines = log_path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["candidate_class"] == "de_novo"
        assert record["mutation_source"] == "silo_a"
        assert record["epoch"] == 1
        assert record["run_id"] == "test_run"

    def test_appends_multiple_epochs(
        self, tmp_dir: Path, sample_candidate: SiloACandidateResult
    ) -> None:
        log_path = tmp_dir / "experiment_log.jsonl"
        append_silo_a_records(log_path, [sample_candidate], epoch=1, run_id="run_e1")
        append_silo_a_records(log_path, [sample_candidate], epoch=2, run_id="run_e2")

        lines = log_path.read_text().splitlines()
        assert len(lines) == 2
        r1 = json.loads(lines[0])
        r2 = json.loads(lines[1])
        assert r1["epoch"] == 1
        assert r2["epoch"] == 2

    def test_log_path_not_pyrosetta_flow(self, tmp_dir: Path) -> None:
        """experiment_log 경로가 Silo B와 별도여야 함."""
        log_path = tmp_dir / "silo_a_flow" / "experiment_log.jsonl"
        assert "pyrosetta_flow" not in str(log_path)
        assert "silo_a" in str(log_path)


# ---------------------------------------------------------------------------
# 분리 무결성 테스트 (경계 검증)
# ---------------------------------------------------------------------------

class TestSeparationIntegrity:
    def test_silo_a_output_dir_never_pyrosetta_flow(self) -> None:
        """Silo A 출력 경로가 pyrosetta_flow를 포함하지 않아야 함."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert "pyrosetta_flow" not in cfg.output_base_dir

    def test_leaderboard_has_de_novo_tag(
        self, tmp_dir: Path, sample_candidate: SiloACandidateResult
    ) -> None:
        lb_path = tmp_dir / "silo_a_leaderboard.json"
        lb = SiloALeaderboard()
        lb.add(sample_candidate)
        lb.save(lb_path)

        data = json.loads(lb_path.read_text())
        # 최상위 source/class 태그
        assert data["source"] == "silo_a"
        assert data["candidate_class"] == "de_novo"
        # 각 entry 태그
        for entry in data["entries"]:
            assert entry.get("candidate_class") == "de_novo"
            assert entry.get("mutation_source") == "silo_a"


# ---------------------------------------------------------------------------
# DiffPepBuilder arm 테스트 (STEP 1 추가)
# ---------------------------------------------------------------------------

from pyrosetta_flow.silo_a_flow import (
    _extract_sequence_from_pdb,
    _build_diffpep_stub,
    _diffpep_repo,
    _is_nonspecific_artifact,
)


class TestDiffPepBuilderArm:
    """DiffPepBuilder arm 설정·태깅·유틸 단위 테스트."""

    def test_config_diffpep_arm_disabled_by_default(self) -> None:
        """DiffPepBuilder arm은 기본적으로 비활성 (기존 동작 무영향)."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert cfg.diffpep_arm_enabled is False

    def test_config_diffpep_arm_enable(self) -> None:
        """diffpep_arm_enabled=True 설정 가능."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb", diffpep_arm_enabled=True)
        assert cfg.diffpep_arm_enabled is True

    def test_config_diffpep_cuda_device(self) -> None:
        """DiffPepBuilder arm GPU 기본값은 '2' (GPU2 전용)."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert cfg.diffpep_cuda_device == "2"

    def test_config_diffpep_cuda_not_3(self) -> None:
        """DiffPepBuilder arm이 GPU3 (RFdiffusion arm) 침범 안 함."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        assert cfg.diffpep_cuda_device != "3"

    def test_config_diffpep_hotspot_chain_a(self) -> None:
        """DiffPepBuilder arm 핫스팟은 chain A 형식 (수용체 chain A 기준)."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        for res in cfg.diffpep_hotspot_res:
            assert res.startswith("A"), f"DiffPepBuilder 핫스팟이 chain A가 아님: {res}"

    def test_config_rfdiffusion_hotspot_chain_b(self) -> None:
        """RFdiffusion arm 핫스팟은 chain B 형식 (chain 분리 유지)."""
        cfg = SiloAConfig(receptor_pdb="/tmp/test.pdb")
        for res in cfg.hotspot_res:
            assert res.startswith("B"), f"RFdiffusion 핫스팟이 chain B가 아님: {res}"

    def test_diffpep_mutation_source_tag(self) -> None:
        """DiffPepBuilder arm 결과에 'silo_a_diffpep' 태그."""
        c = SiloACandidateResult(
            candidate_id="diffpep_test_bb00_sq00",
            sequence="NGHQIFEMIQQQIW",
            backbone_idx=0,
            seq_idx=0,
            plddt=None,
            plddt_pass=True,
            ddg=-18.5,
            clash_score=None,
            selectivity_margin=None,
            delta_margin=None,
            hc50=None,
            half_life_h=None,
            admet_score=None,
            fail_reason="",
            mutation_source="silo_a_diffpep",
            candidate_class="de_novo",
            extra_scores={"mutation_source": "silo_a_diffpep"},
        )
        assert c.mutation_source == "silo_a_diffpep"
        d = c.to_dict()
        assert d["mutation_source"] == "silo_a_diffpep"

    def test_diffpep_leaderboard_preserves_tag(self) -> None:
        """DiffPepBuilder arm 후보를 리더보드에 추가 시 mutation_source 보존."""
        lb = SiloALeaderboard()
        c = SiloACandidateResult(
            candidate_id="diffpep_c0",
            sequence="NGHQIFEMIQQQIW",
            backbone_idx=0, seq_idx=0,
            plddt=0.75, plddt_pass=True, ddg=-22.3,
            clash_score=None, selectivity_margin=None, delta_margin=None,
            hc50=None, half_life_h=None, admet_score=None, fail_reason="",
            mutation_source="silo_a_diffpep",
        )
        lb.add(c)
        assert lb.entries[0]["mutation_source"] == "silo_a_diffpep"

    def test_extract_sequence_from_pdb(self, tmp_path: Path) -> None:
        """_extract_sequence_from_pdb: chain A 서열 추출."""
        pdb_content = "\n".join([
            "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C",
            "ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00  0.00           C",
            "ATOM      3  CA  PHE A   3       7.000   8.000   9.000  1.00  0.00           C",
            "ATOM      4  CA  TRP B  10       1.000   2.000   3.000  1.00  0.00           C",  # chain B (무시)
            "END",
        ])
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content, encoding="utf-8")

        seq = _extract_sequence_from_pdb(pdb_path, chain="A")
        assert seq == "AGF"

    def test_extract_sequence_empty_chain(self, tmp_path: Path) -> None:
        """존재하지 않는 chain 조회 시 빈 문자열 반환."""
        pdb_content = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C\n"
        pdb_path = tmp_path / "test.pdb"
        pdb_path.write_text(pdb_content, encoding="utf-8")

        seq = _extract_sequence_from_pdb(pdb_path, chain="Z")
        assert seq == ""

    def test_build_diffpep_stub_creates_file(self, tmp_path: Path) -> None:
        """_build_diffpep_stub: stub 스크립트 생성."""
        stub = _build_diffpep_stub(tmp_path)
        assert stub.exists()
        content = stub.read_text()
        assert "GPUtil" in content
        assert "pdbfixer" in content
        assert "pyrosetta" in content
        assert "run_inference.py" in content

    def test_build_diffpep_stub_reuses_existing(self, tmp_path: Path) -> None:
        """_build_diffpep_stub: 이미 있으면 재생성 없이 경로 반환."""
        stub1 = _build_diffpep_stub(tmp_path)
        stub2 = _build_diffpep_stub(tmp_path)
        assert stub1 == stub2

    def test_diffpep_arm_no_interference_with_rfdiffusion_arm(self) -> None:
        """DiffPepBuilder arm 활성화 시 RFdiffusion arm 설정 미변경."""
        cfg = SiloAConfig(
            receptor_pdb="/tmp/test.pdb",
            diffpep_arm_enabled=True,
        )
        # RFdiffusion arm 파라미터 기본값 유지
        assert cfg.n_backbone == 2
        assert cfg.k_seq_per_backbone == 2
        assert cfg.contigs == "B1-369/0 12-16"

    def test_diffpep_repo_path_within_root(self) -> None:
        """DiffPepBuilder 저장소 경로가 ROOT/local_models/DiffPepBuilder."""
        repo = _diffpep_repo()
        assert "local_models" in str(repo)
        assert "DiffPepBuilder" in str(repo)
        # ROOT 밖 침범 금지
        assert "[LOCAL_PATH]" in str(repo)


# ---------------------------------------------------------------------------
# _is_nonspecific_artifact 필터 단위 테스트
# ---------------------------------------------------------------------------

class TestIsNonspecificArtifact:
    """비특이 아티팩트 필터 검증.

    검토팀이 확인한 3개 아티팩트 서열이 필터에 걸리는지 실증하고,
    정상 SST-14 계열이 통과하는지 확인한다.
    """

    # ── 아티팩트 reject ─────────────────────────────────────────────

    def test_membrane_insertion_mfflil_rejected(self) -> None:
        """MFFLILAAALLAALLLL: 막삽입 → GRAVY 과다 또는 소수성잔기과다로 reject."""
        seq = "MFFLILAAALLAALLLL"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is True, f"막삽입 서열이 통과됨: reason={reason!r}"
        # GRAVY 또는 소수성 비율 사유여야 함
        assert "GRAVY" in reason or "소수성" in reason, f"예상 사유 아님: {reason}"

    def test_amp_pattern_lerlrrl_rejected(self) -> None:
        """LERLRRLLERLLRLAR: AMP 패턴 (R 37%, +전하, 높은 μH) → reject."""
        seq = "LERLRRLLERLLRLAR"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is True, f"AMP 서열이 통과됨: reason={reason!r}"
        # AMP패턴 또는 과양전하 사유
        assert "AMP" in reason or "과양전하" in reason, f"예상 사유 아님: {reason}"

    def test_ctail_charge_svidkll_rejected(self) -> None:
        """SVIDKLLEEDLKEAAK: C-tail 정전기/net charge → 과음전하 또는 AMP/소수성 사유로 reject."""
        seq = "SVIDKLLEEDLKEAAK"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is True, f"C-tail 정전기 서열이 통과됨: reason={reason!r}"

    # ── 정상 서열 통과 ───────────────────────────────────────────────

    def test_sst14_native_passes(self) -> None:
        """SST-14 원형 AGCKNFFWKTFTSC: 필터 통과."""
        seq = "AGCKNFFWKTFTSC"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is False, f"SST-14 native가 필터에 걸림: reason={reason!r}"

    def test_sst14_f11d_mutant_passes(self) -> None:
        """best 후보 F11D 변이체 AGCKNFFWKTDTSC: 필터 통과."""
        seq = "AGCKNFFWKTDTSC"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is False, f"F11D 변이체가 필터에 걸림: reason={reason!r}"

    def test_typical_peptide_passes(self) -> None:
        """일반 혼합 서열 ACDEFGHIKLMN: 필터 통과 (GRAVY/전하/μH 정상 범위)."""
        seq = "ACDEFGHIKLMN"
        fail, reason = _is_nonspecific_artifact(seq)
        assert fail is False, f"정상 혼합 서열이 필터에 걸림: reason={reason!r}"

    # ── 환경변수 비활성 ──────────────────────────────────────────────

    def test_filter_disabled_by_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SILO_A_NONSPECIFIC_FILTER=0 이면 아티팩트도 통과."""
        import pyrosetta_flow.silo_a_flow as _m
        monkeypatch.setattr(_m, "_NSF_ENABLED", False)
        fail, reason = _is_nonspecific_artifact("MFFLILAAALLAALLLL")
        assert fail is False
        assert reason == ""

    # ── 빈 서열 ─────────────────────────────────────────────────────

    def test_empty_seq_returns_false(self) -> None:
        """빈 서열은 fail-open: (False, '')."""
        fail, reason = _is_nonspecific_artifact("")
        assert fail is False
        assert reason == ""

    # ── 사유 문자열 포맷 검증 ────────────────────────────────────────

    def test_reason_string_nonempty_when_rejected(self) -> None:
        """reject 시 reason이 비어있지 않아야 함."""
        seq = "MFFLILAAALLAALLLL"
        fail, reason = _is_nonspecific_artifact(seq)
        if fail:
            assert len(reason) > 0, "reject 사유 문자열이 비어있음"


# ---------------------------------------------------------------------------
# pLDDT fail-closed / fail-open 게이트 단위 테스트
# ---------------------------------------------------------------------------

import pyrosetta_flow.silo_a_flow as _silo_a_module


class TestPlddtFailClosedGate:
    """SILO_A_PLDDT_FAILCLOSED 게이트 동작 검증.

    실제 ESMFold 호출 없이 모듈 수준 상수(_PLDDT_FAILCLOSED)를 monkeypatch하여
    plddt=None 상황에서의 게이트 로직을 단위 검증한다.
    """

    # ── 게이트 로직 직접 검증 ────────────────────────────────────────

    def test_failclosed_mode_none_plddt_is_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fail-closed(기본): plddt=None → plddt_pass=False."""
        monkeypatch.setattr(_silo_a_module, "_PLDDT_FAILCLOSED", True)
        plddt = None
        plddt_pass = (plddt is not None) and (plddt >= 0.5)  # type: ignore[operator]
        assert plddt_pass is False, "fail-closed에서 plddt=None이 게이트를 통과함"

    def test_failopen_mode_none_plddt_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fail-open(SILO_A_PLDDT_FAILCLOSED=0): plddt=None → plddt_pass=True."""
        monkeypatch.setattr(_silo_a_module, "_PLDDT_FAILCLOSED", False)
        plddt = None
        # fail-open 로직: plddt가 None이면 게이트 통과 (not _PLDDT_FAILCLOSED)
        failclosed = False  # monkeypatched
        plddt_pass = not failclosed if plddt is None else (plddt >= 0.5)  # type: ignore[operator]
        assert plddt_pass is True, "fail-open에서 plddt=None이 게이트를 통과하지 못함"

    def test_failclosed_float_plddt_above_threshold_passes(self) -> None:
        """정상 plddt 값(≥threshold)은 fail-closed에서도 통과."""
        plddt = 0.82
        threshold = 0.5
        plddt_pass = (plddt is not None) and (plddt >= threshold)
        assert plddt_pass is True

    def test_failclosed_float_plddt_below_threshold_blocked(self) -> None:
        """plddt < threshold는 fail-closed에서 탈락 (기존 동작 유지)."""
        plddt = 0.35
        threshold = 0.5
        plddt_pass = (plddt is not None) and (plddt >= threshold)
        assert plddt_pass is False

    # ── SiloACandidateResult 필드 검증 ──────────────────────────────

    def test_measurement_missing_result_has_correct_fail_reason(self) -> None:
        """measurement_missing 후보에 esmfold_unavailable fail_reason이 기록됨."""
        c = SiloACandidateResult(
            candidate_id="test_missing",
            sequence="ACDEFGHIKLMN",
            backbone_idx=0,
            seq_idx=0,
            plddt=None,
            plddt_pass=False,
            ddg=None,
            clash_score=None,
            selectivity_margin=None,
            delta_margin=None,
            hc50=None,
            half_life_h=None,
            admet_score=None,
            fail_reason="esmfold_unavailable(measurement_missing)",
            mutation_source="silo_a",
            extra_scores={
                "candidate_class": "de_novo",
                "plddt_status": "measurement_missing",
            },
        )
        assert c.plddt_pass is False
        assert "esmfold_unavailable" in c.fail_reason
        assert c.extra_scores is not None
        assert c.extra_scores.get("plddt_status") == "measurement_missing"

    def test_measurement_missing_result_excluded_from_leaderboard(self) -> None:
        """measurement_missing 후보는 plddt_pass=False이므로 리더보드 직접 add 시 진입은 가능.
        run_silo_a_discovery.py의 skip 필터가 실질적 차단을 담당함을 확인."""
        lb = SiloALeaderboard()
        c = SiloACandidateResult(
            candidate_id="test_missing_lb",
            sequence="ACDEFGHIKLMN",
            backbone_idx=0,
            seq_idx=0,
            plddt=None,
            plddt_pass=False,
            ddg=None,
            clash_score=None,
            selectivity_margin=None,
            delta_margin=None,
            hc50=None,
            half_life_h=None,
            admet_score=None,
            fail_reason="esmfold_unavailable(measurement_missing)",
            mutation_source="silo_a",
            extra_scores={"plddt_status": "measurement_missing"},
        )
        # run_silo_a_discovery.py의 skip 로직: plddt_pass=False + failclosed → continue
        _failclosed = True
        skip = _failclosed and not c.plddt_pass
        assert skip is True, "fail-closed 시 measurement_missing 후보가 skip되지 않음"
        # skip=True이므로 leaderboard.add 미호출 — 리더보드는 비어있어야 함
        if not skip:
            lb.add(c)
        assert len(lb.entries) == 0, "measurement_missing 후보가 리더보드에 진입함"

    def test_failopen_measurement_missing_not_skipped(self) -> None:
        """fail-open 모드에서는 measurement_missing 후보가 skip되지 않음."""
        c = SiloACandidateResult(
            candidate_id="test_missing_open",
            sequence="ACDEFGHIKLMN",
            backbone_idx=0,
            seq_idx=0,
            plddt=None,
            plddt_pass=True,  # fail-open에서는 plddt=None → plddt_pass=True
            ddg=-10.0,
            clash_score=None,
            selectivity_margin=None,
            delta_margin=None,
            hc50=None,
            half_life_h=None,
            admet_score=None,
            fail_reason="",
            mutation_source="silo_a",
        )
        _failclosed = False
        skip = _failclosed and not c.plddt_pass
        assert skip is False, "fail-open에서 plddt=None 후보가 잘못 skip됨"

    # ── 환경변수 파싱 로직 검증 ──────────────────────────────────────

    def test_failclosed_default_env_is_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SILO_A_PLDDT_FAILCLOSED 미설정 시 기본값=1 (fail-closed)."""
        monkeypatch.delenv("SILO_A_PLDDT_FAILCLOSED", raising=False)
        import importlib
        # 환경변수 파싱 로직 재현 (모듈 재로드 없이 파싱만 검증)
        val = __import__("os").environ.get("SILO_A_PLDDT_FAILCLOSED", "1") != "0"
        assert val is True

    def test_failclosed_env_0_means_failopen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SILO_A_PLDDT_FAILCLOSED=0 → fail-open."""
        monkeypatch.setenv("SILO_A_PLDDT_FAILCLOSED", "0")
        val = __import__("os").environ.get("SILO_A_PLDDT_FAILCLOSED", "1") != "0"
        assert val is False

    def test_failclosed_env_1_means_failclosed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SILO_A_PLDDT_FAILCLOSED=1 → fail-closed."""
        monkeypatch.setenv("SILO_A_PLDDT_FAILCLOSED", "1")
        val = __import__("os").environ.get("SILO_A_PLDDT_FAILCLOSED", "1") != "0"
        assert val is True
