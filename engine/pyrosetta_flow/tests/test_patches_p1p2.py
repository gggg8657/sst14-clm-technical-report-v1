"""P1/P2 1차 패치 단위 테스트 (2026-06-23).

C2 — NSGA-II 명칭 교정 (pareto_ranking.py docstring 정직성)
E1 — seed 재현성 (BayesianPeptideOptimizer seed 파라미터)
D2 — key-contact Boolean write (parse_structural_rules_to_columns)
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys

import pytest


# =========================================================================
# C2 — NSGA-II 명칭 교정 테스트
# 주: pymoo가 없는 환경에서도 동작하도록 소스 파일 텍스트 직접 검증
# =========================================================================

def _read_pareto_ranking_source() -> str:
    """pareto_ranking.py 소스 텍스트를 반환한다 (import 불필요)."""
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / "pareto_ranking.py"
    return src.read_text(encoding="utf-8")


class TestC2NsgaNaming:
    """pareto_ranking.py 명칭이 정직하게 교정되었는지 소스 텍스트로 확인.

    pymoo가 없는 환경에서도 import 없이 소스를 직접 읽어 검증한다.
    """

    def test_module_first_line_no_nsga2_headline(self):
        """모듈 docstring 첫 줄이 'NSGA-II Pareto Ranking'이 아니어야 한다."""
        src = _read_pareto_ranking_source()
        # 첫 번째 docstring 첫 줄 추출
        doc_match = re.search(r'"""(.*?)"""', src, re.DOTALL)
        assert doc_match, "모듈 docstring을 찾을 수 없음"
        first_line = doc_match.group(1).strip().splitlines()[0]
        assert "NSGA-II Pareto Ranking" not in first_line, (
            f"첫 줄에 'NSGA-II Pareto Ranking'이 남아 있음: {first_line!r}"
        )

    def test_module_docstring_has_honesty_comment(self):
        """모듈 docstring에 단발 정렬임을 명시하는 정직성 주석이 있어야 한다."""
        src = _read_pareto_ranking_source()
        assert "단발" in src or "single-shot" in src, (
            "단발(single-shot) 명시 없음"
        )
        assert "세대 루프" in src or "generation" in src, (
            "세대 루프 부재 설명 없음"
        )

    def test_pareto_rank_candidates_docstring_updated(self):
        """pareto_rank_candidates docstring이 교정된 표현을 포함해야 한다."""
        src = _read_pareto_ranking_source()
        # 업데이트된 표현 중 하나 이상 존재해야 함
        has_updated = (
            "NSGA-II 정렬 단계 활용" in src
            or "단발" in src
            or "single-shot" in src
        )
        assert has_updated, "pareto_rank_candidates docstring이 교정되지 않음"

    def test_no_nsga_ii_standalone_claim_in_source(self):
        """소스에 'NSGA-II'가 독립 알고리즘 명칭으로 단독 주장되지 않아야 한다.
        (활용/단계 맥락 없이 'NSGA-II'만 있으면 안 됨 — 경고 수준 확인)"""
        src = _read_pareto_ranking_source()
        # "NSGA-II" 줄이 있다면 같은 줄에 맥락 설명이 있어야 함
        nsga_lines = [ln for ln in src.splitlines() if "NSGA-II" in ln]
        for ln in nsga_lines:
            ln_stripped = ln.strip()
            # 주석, docstring, 또는 맥락 키워드 포함 여부
            has_context = (
                "정렬 단계" in ln_stripped
                or "단발" in ln_stripped
                or "활용" in ln_stripped
                or "fast non-dominated" in ln_stripped
                or "세대 루프" in ln_stripped
                or ln_stripped.startswith("#")
                or ln_stripped.startswith('"')
                or ln_stripped.startswith("'")
                or "Sort" in ln_stripped
            )
            assert has_context, (
                f"'NSGA-II' 단독 주장 줄 감지 (맥락 없음): {ln_stripped!r}"
            )

    def test_pareto_ranking_functional_with_pymoo(self):
        """pymoo가 있는 경우 기능 동작이 불변임을 확인."""
        pytest.importorskip("pymoo", reason="pymoo 없음 — 스킵")
        from pyrosetta_flow.pareto_ranking import (
            pareto_rank_candidates,
            select_from_pareto_front,
        )
        candidates = [
            {"ddG": -10.0, "stability": 0.8, "druggability": 0.7, "diversity": 0.5},
            {"ddG": -5.0,  "stability": 0.9, "druggability": 0.6, "diversity": 0.6},
            {"ddG": -8.0,  "stability": 0.75, "druggability": 0.65, "diversity": 0.55},
        ]
        ranked = pareto_rank_candidates(candidates)
        for c in ranked:
            assert "pareto_rank" in c
            assert "crowding_distance" in c
        best = min(ranked, key=lambda c: c["ddG"])
        assert best["pareto_rank"] == 0

        top1 = select_from_pareto_front(ranked, n=1)
        assert len(top1) == 1
        assert top1[0]["pareto_rank"] == 0


# =========================================================================
# E1 — seed 재현성 테스트
# =========================================================================

class TestE1SeedReproducibility:
    """BayesianPeptideOptimizer seed 파라미터 추가 및 재현성 확인."""

    def test_seed_parameter_exists(self):
        """BayesianPeptideOptimizer.__init__에 seed 파라미터가 있어야 한다."""
        from pyrosetta_flow.bayesian_optimizer import BayesianPeptideOptimizer
        sig = inspect.signature(BayesianPeptideOptimizer.__init__)
        assert "seed" in sig.parameters, (
            "__init__ 시그니처에 'seed' 파라미터 없음"
        )

    def test_seed_default_is_none(self):
        """seed 기본값은 None (하위호환 보존)."""
        from pyrosetta_flow.bayesian_optimizer import BayesianPeptideOptimizer
        sig = inspect.signature(BayesianPeptideOptimizer.__init__)
        default = sig.parameters["seed"].default
        assert default is None, f"seed 기본값이 None이 아님: {default!r}"

    def test_seed_stored_on_instance(self):
        """seed가 인스턴스 속성 _seed로 저장되어야 한다."""
        from pyrosetta_flow.bayesian_optimizer import (
            BayesianPeptideOptimizer,
            OneHotEmbedder,
        )
        embedder = OneHotEmbedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=embedder,
            objectives=["ddG"],
            seed=42,
        )
        assert bo._seed == 42

    def test_seed_none_does_not_crash(self):
        """seed=None 시 기존 동작(비결정적)이 유지되고 오류 없음."""
        from pyrosetta_flow.bayesian_optimizer import (
            BayesianPeptideOptimizer,
            OneHotEmbedder,
        )
        embedder = OneHotEmbedder(max_len=14)
        bo = BayesianPeptideOptimizer(
            embedder=embedder,
            objectives=["ddG"],
            seed=None,
        )
        assert bo._seed is None

    def test_fallback_gp_reproducible_with_seed(self):
        """동일 seed + 동일 입력 → fallback GP suggest 결과 일치.

        BoTorch 없는 환경에서만 동작하는 fallback GP 경로를 테스트한다.
        BoTorch 환경에서도 같은 seed로 두 번 생성 시 동일한 acquisition 순서를
        가져야 한다 (전역 seed 설정의 효과).
        """
        from pyrosetta_flow.bayesian_optimizer import (
            BayesianPeptideOptimizer,
            OneHotEmbedder,
        )
        import warnings

        candidates = [
            {"sequence": "AGCKNFFWKTFTSC", "ddG": -10.0},
            {"sequence": "AGCRNFFWKTFTSC", "ddG": -8.0},
            {"sequence": "AGCKNFFWATFTSC", "ddG": -9.0},
            {"sequence": "AGCKNFFWKTFTSC", "ddG": -11.0},  # 중복 서열 다른 ddG
        ]

        def _run(seed_val: int):
            embedder = OneHotEmbedder(max_len=14)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bo = BayesianPeptideOptimizer(
                    embedder=embedder,
                    objectives=["ddG"],
                    maximize=[False],
                    seed=seed_val,
                )
            bo.fit(candidates)
            return bo.suggest(n=3, reference_seq="AGCKNFFWKTFTSC")

        r1 = _run(42)
        r2 = _run(42)

        seqs1 = [s["sequence"] for s in r1]
        seqs2 = [s["sequence"] for s in r2]
        assert seqs1 == seqs2, (
            f"동일 seed 두 번 실행 시 서열 순서 불일치: {seqs1} vs {seqs2}"
        )

    def test_different_seeds_may_differ(self):
        """다른 seed 두 실행의 결과가 항상 같지는 않을 수 있다 (확정적 테스트 아님).
        단지 인스턴스 생성이 오류 없이 성공해야 한다."""
        from pyrosetta_flow.bayesian_optimizer import (
            BayesianPeptideOptimizer,
            OneHotEmbedder,
        )
        import warnings
        embedder = OneHotEmbedder(max_len=14)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bo1 = BayesianPeptideOptimizer(
                embedder=embedder, objectives=["ddG"], seed=1
            )
            bo2 = BayesianPeptideOptimizer(
                embedder=embedder, objectives=["ddG"], seed=99999
            )
        assert bo1._seed == 1
        assert bo2._seed == 99999


# =========================================================================
# D2 — key-contact Boolean write 테스트
# =========================================================================

class TestD2KeyContact:
    """parse_structural_rules_to_columns 헬퍼 함수 동작 검증."""

    def test_function_importable(self):
        """parse_structural_rules_to_columns이 pdb_store에서 import 가능해야 한다."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        assert callable(parse_structural_rules_to_columns)

    def test_full_input_all_true(self):
        """모든 규칙이 pass=True인 경우 모든 컬럼이 True."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        sr = {
            "rules": {
                "fwkt_pharmacophore":   {"pass": True,  "detail": "positions 7-10 = FWKT"},
                "cys3_cys14_disulfide": {"pass": True,  "detail": "pos3=C, pos14=C"},
                "k9_salt_bridge":       {"pass": True,  "detail": "position 9 = K"},
                "phe6_phe11_stacking":  {"pass": True,  "detail": "pos6=F, pos11=F"},
                "nterm_chelator":       {"pass": True,  "detail": "pos1=A"},
            },
            "all_pass": True,
        }
        cols = parse_structural_rules_to_columns(sr)
        assert cols["fwkt_conserved"] is True
        assert cols["ss_bond_intact"] is True
        assert cols["k9_d122_salt"] is True
        assert cols["phe_stacking"] is True
        assert cols["chelator_ready"] is True

    def test_partial_fail(self):
        """k9_salt_bridge pass=False → k9_d122_salt이 False."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        sr = {
            "rules": {
                "fwkt_pharmacophore":   {"pass": True},
                "cys3_cys14_disulfide": {"pass": True},
                "k9_salt_bridge":       {"pass": False, "detail": "position 9 = R"},
                "phe6_phe11_stacking":  {"pass": True},
                "nterm_chelator":       {"pass": True},
            },
            "all_pass": False,
        }
        cols = parse_structural_rules_to_columns(sr)
        assert cols["k9_d122_salt"] is False
        assert cols["fwkt_conserved"] is True

    def test_empty_dict_returns_none_for_all(self):
        """빈 dict 입력 시 모든 값이 None."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        cols = parse_structural_rules_to_columns({})
        assert cols["fwkt_conserved"] is None
        assert cols["ss_bond_intact"] is None
        assert cols["k9_d122_salt"] is None
        assert cols["phe_stacking"] is None
        assert cols["chelator_ready"] is None

    def test_missing_rules_key_returns_none(self):
        """'rules' 키가 없는 dict도 방어적으로 처리."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        cols = parse_structural_rules_to_columns({"all_pass": False})
        assert all(v is None for v in cols.values())

    def test_returns_all_five_columns(self):
        """반환 dict에 5개 컬럼 키가 모두 있어야 한다."""
        from pyrosetta_flow.pdb_store import parse_structural_rules_to_columns
        cols = parse_structural_rules_to_columns({})
        expected_keys = {"fwkt_conserved", "ss_bond_intact", "k9_d122_salt",
                         "phe_stacking", "chelator_ready"}
        assert set(cols.keys()) == expected_keys

    def test_compatible_with_pdb_store_register(self):
        """parse_structural_rules_to_columns 결과를 meta에 넣으면
        register_candidate가 PDBRecord Boolean 컬럼에 정상 저장해야 한다."""
        pytest.importorskip("sqlalchemy", reason="pdb_store persistence requires SQLAlchemy")
        from pyrosetta_flow.pdb_store import (
            parse_structural_rules_to_columns,
            register_candidate,
            get_engine,
            PDBRecord,
        )
        from sqlalchemy.orm import Session

        engine = get_engine(":memory:")
        with Session(engine) as session:
            sr = {
                "rules": {
                    "fwkt_pharmacophore":   {"pass": True},
                    "cys3_cys14_disulfide": {"pass": True},
                    "k9_salt_bridge":       {"pass": False},
                    "phe6_phe11_stacking":  {"pass": True},
                    "nterm_chelator":       {"pass": True},
                },
            }
            sr_cols = parse_structural_rules_to_columns(sr)
            meta = {
                "candidate_id": "test_d2_001",
                "sequence": "AGCKNFFWKTFTSC",
                "ddg": -10.0,
                "run_id": "test_run",
                "iteration": 1,
                **sr_cols,
            }
            rec = register_candidate(session, "test_d2_001", None, meta)
            assert rec.fwkt_conserved is True
            assert rec.ss_bond_intact is True
            assert rec.k9_d122_salt is False
            assert rec.phe_stacking is True
            assert rec.chelator_ready is True

    def test_field_names_match_struct_fields(self):
        """parse_structural_rules_to_columns 반환 키가 _STRUCT_FIELDS와 일치."""
        from pyrosetta_flow.pdb_store import (
            parse_structural_rules_to_columns,
            _STRUCT_FIELDS,
        )
        cols = parse_structural_rules_to_columns({})
        assert set(cols.keys()) == set(_STRUCT_FIELDS)
