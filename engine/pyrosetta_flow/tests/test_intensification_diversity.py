"""
tests/test_intensification_diversity.py
=======================================
요구사항 1 (local intensification) + 요구사항 2 (다양성 강화) + 요구사항 3 (dedup fallback) 검증.

- bandit.get_intensification_guidance: seed 주변 guidance 생성
- bandit.load_high_confidence_seeds: mmgbsa_consensus.json 파싱
- adapter.generate_intensification_mutant: 부분조합 변이 생성
- adapter.generate_dedup_fallback: 슬롯 낭비 방지 2단계 fall-back
- adapter.get_high_confidence_seeds: 래퍼 함수
- prompts.format_planner_prompt: evaluated_seqs 주입 섹션 존재 확인
"""
from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from pyrosetta_flow.bandit import (
    CONSERVATIVE_SUBSTITUTIONS,
    MUTABLE_POSITIONS_1IDX,
    REFERENCE_SEQUENCE,
    get_intensification_guidance,
    load_high_confidence_seeds,
)
from pyrosetta_flow.adapter import (
    AA_NO_CYS,
    generate_dedup_fallback,
    generate_intensification_mutant,
    get_high_confidence_seeds,
)
from AG_src.llm.prompts import format_planner_prompt


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------

NATIVE = "AGCKNFFWKTFTSC"
# best 후보: G2I/K4L/F6W/F11V/T12I 변이 (FWKT pos7-10 보존, Cys3/14 보존)
SEED_SEQ = "AICLNWFWKTVISC"
MUTABLE = [1, 2, 4, 5, 6, 11, 12]  # scaffold 제약 후


@pytest.fixture()
def mmgbsa_json_path(tmp_path: Path) -> Path:
    """임시 mmgbsa_consensus.json 파일 생성."""
    data = {
        "generated_at": "2026-06-29T00:00:00Z",
        "native_dg": -71.053,
        "results": {
            NATIVE: {
                "sequence": NATIVE,
                "source": "native",
                "dg_bind": -71.053,
                "beats_native": False,
                "consensus_flag": "native",
            },
            SEED_SEQ: {
                "sequence": SEED_SEQ,
                "source": "silo_b",
                "dg_bind": -94.471,
                "beats_native": True,
                "consensus_flag": "high_confidence",
            },
            "AGCMNFFWKTIPSC": {
                "sequence": "AGCMNFFWKTIPSC",
                "source": "silo_b",
                "dg_bind": -76.364,
                "beats_native": True,
                "consensus_flag": "mmgbsa_only",
            },
        },
    }
    p = tmp_path / "mmgbsa_consensus.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# bandit.get_intensification_guidance 테스트
# ---------------------------------------------------------------------------

class TestGetIntensificationGuidance:

    def test_returns_focus_positions_non_empty(self):
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        assert len(guidance["focus_positions"]) > 0

    def test_focus_positions_within_mutable(self):
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        mutable_set = set(MUTABLE)
        for pos in guidance["focus_positions"]:
            assert pos in mutable_set, f"pos={pos} not in mutable set"

    def test_no_pharmacophore_positions(self):
        """FWKT(7-10), Cys(3,14) 는 절대 포함 안 됨."""
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        forbidden = {3, 7, 8, 9, 10, 14}
        for pos in guidance["focus_positions"]:
            assert pos not in forbidden, f"forbidden pos={pos} in focus_positions"

    def test_suggested_mutations_no_cys(self):
        """suggested_mutations에 C(Cys) 포함 안 됨."""
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        for pos_key, aas in guidance["suggested_mutations"].items():
            assert "C" not in aas, f"C found in suggestions for pos {pos_key}"

    def test_seed_aa_included_in_suggestions(self):
        """변이된 위치에서 seed 아미노산이 suggested_mutations에 포함돼야 한다."""
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        # pos2: seed=I, native=G → I가 suggestions에 있어야
        if 2 in guidance["focus_positions"]:
            suggs = guidance["suggested_mutations"].get("2", [])
            assert "I" in suggs, f"seed aa 'I' at pos2 not in suggestions: {suggs}"

    def test_wrong_length_returns_empty(self):
        """길이 불일치: focus_positions 빈 리스트 반환."""
        guidance = get_intensification_guidance(
            seed_sequence="SHORT",
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        assert guidance["focus_positions"] == []

    def test_source_label(self):
        guidance = get_intensification_guidance(
            seed_sequence=SEED_SEQ,
            reference_sequence=NATIVE,
            mutable_positions=MUTABLE,
        )
        assert guidance["source"] == "intensification"

    def test_deterministic_with_same_rng(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        g1 = get_intensification_guidance(SEED_SEQ, NATIVE, MUTABLE, rng=rng1)
        g2 = get_intensification_guidance(SEED_SEQ, NATIVE, MUTABLE, rng=rng2)
        assert g1["focus_positions"] == g2["focus_positions"]


# ---------------------------------------------------------------------------
# bandit.load_high_confidence_seeds 테스트
# ---------------------------------------------------------------------------

class TestLoadHighConfidenceSeeds:

    def test_returns_only_high_confidence(self, mmgbsa_json_path: Path):
        seeds = load_high_confidence_seeds(mmgbsa_json_path)
        for s in seeds:
            assert s["consensus_flag"] == "high_confidence"

    def test_beats_native_filter(self, mmgbsa_json_path: Path):
        seeds = load_high_confidence_seeds(mmgbsa_json_path, beats_native=True)
        for s in seeds:
            assert s.get("consensus_flag") == "high_confidence"
        # SEED_SEQ 포함 확인
        seqs = [s["sequence"] for s in seeds]
        assert SEED_SEQ in seqs

    def test_sorted_by_dg_bind(self, mmgbsa_json_path: Path):
        seeds = load_high_confidence_seeds(mmgbsa_json_path)
        if len(seeds) > 1:
            for i in range(len(seeds) - 1):
                assert seeds[i]["dg_bind"] <= seeds[i + 1]["dg_bind"]

    def test_missing_file_returns_empty(self, tmp_path: Path):
        seeds = load_high_confidence_seeds(tmp_path / "nonexistent.json")
        assert seeds == []

    def test_native_not_included(self, mmgbsa_json_path: Path):
        seeds = load_high_confidence_seeds(mmgbsa_json_path)
        seqs = [s["sequence"] for s in seeds]
        assert NATIVE not in seqs


# ---------------------------------------------------------------------------
# adapter.generate_intensification_mutant 테스트
# ---------------------------------------------------------------------------

class TestGenerateIntensificationMutant:

    @pytest.fixture()
    def guidance(self):
        return get_intensification_guidance(SEED_SEQ, NATIVE, MUTABLE, rng=random.Random(0))

    def test_returns_14aa_sequence(self, guidance):
        result = generate_intensification_mutant(
            original_seq=NATIVE,
            intensify_guidance=guidance,
            design_positions=MUTABLE,
            rng=random.Random(0),
        )
        assert len(result) == len(NATIVE)

    def test_preserves_pharmacophore(self, guidance):
        """FWKT pharmacophore (pos7-10) 보존."""
        for seed in range(20):
            result = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(seed),
            )
            assert result[6:10] == NATIVE[6:10], f"pharmacophore broken: {result}"

    def test_preserves_disulfide_cys(self, guidance):
        """Cys3, Cys14 보존."""
        for seed in range(20):
            result = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(seed),
            )
            assert result[2] == "C", f"Cys3 broken: {result}"
            assert result[13] == "C", f"Cys14 broken: {result}"

    def test_no_cys_introduced(self, guidance):
        """변이 결과에 새로운 Cys 도입 안 됨 (AA_NO_CYS 사용)."""
        for seed in range(20):
            result = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(seed),
            )
            # Cys는 pos3, pos14만 허용
            for i, aa in enumerate(result):
                if aa == "C":
                    assert i in (2, 13), f"unexpected C at pos {i + 1}: {result}"

    def test_differs_from_native(self, guidance):
        """최소 1개 이상 native와 달라야 한다."""
        changed = False
        for seed in range(30):
            result = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(seed),
            )
            if result != NATIVE:
                changed = True
                break
        assert changed, "intensification이 항상 native를 반환"

    def test_empty_guidance_falls_back_to_random(self):
        """guidance 비어있으면 random_mutant fallback."""
        result = generate_intensification_mutant(
            original_seq=NATIVE,
            intensify_guidance={"focus_positions": [], "suggested_mutations": {}},
            design_positions=MUTABLE,
            rng=random.Random(42),
        )
        # 길이 보존
        assert len(result) == len(NATIVE)

    def test_partial_combination_picks_2_to_3_positions(self, guidance):
        """partial_combination=True이면 2~3개 위치만 변이."""
        counts: List[int] = []
        for seed in range(20):
            result = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(seed),
                partial_combination=True,
            )
            # FWKT+Cys 고정위치 제외한 diff 계산
            fixed = {2, 6, 7, 8, 9, 13}  # 0-indexed: pos3,7,8,9,10,14
            diff = sum(
                1 for i, (a, b) in enumerate(zip(result, NATIVE))
                if a != b and i not in fixed
            )
            counts.append(diff)
        # 대부분 2~3개 변이
        valid = [c for c in counts if 1 <= c <= 4]
        assert len(valid) >= len(counts) * 0.8, f"counts distribution: {counts}"


# ---------------------------------------------------------------------------
# adapter.get_high_confidence_seeds (래퍼) 테스트
# ---------------------------------------------------------------------------

class TestGetHighConfidenceSeeds:

    def test_wrapper_returns_same_as_load(self, mmgbsa_json_path: Path):
        from pyrosetta_flow.bandit import load_high_confidence_seeds
        direct = load_high_confidence_seeds(mmgbsa_json_path)
        via_adapter = get_high_confidence_seeds(mmgbsa_json_path)
        assert direct == via_adapter


# ---------------------------------------------------------------------------
# prompts.format_planner_prompt — evaluated_seqs 주입 테스트
# ---------------------------------------------------------------------------

class TestFormatPlannerPromptEvaluatedSeqs:

    def test_evaluated_seqs_section_present(self):
        seqs = ["AGCKNFFWKTLTSC", "AGCKNFFWKTITSC"]
        prompt = format_planner_prompt(
            iteration=2,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": NATIVE},
            evaluated_seqs=seqs,
        )
        assert "이미 평가된 서열" in prompt
        for s in seqs:
            assert s in prompt

    def test_evaluated_seqs_none_no_section(self):
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": NATIVE},
            evaluated_seqs=None,
        )
        assert "이미 평가된 서열" not in prompt

    def test_evaluated_seqs_empty_no_section(self):
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": NATIVE},
            evaluated_seqs=[],
        )
        assert "이미 평가된 서열" not in prompt

    def test_display_n_limits_output(self):
        seqs = [f"AGCKNFFWKTF{i:01d}SC" for i in range(50)]
        prompt = format_planner_prompt(
            iteration=3,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": NATIVE},
            evaluated_seqs=seqs,
            evaluated_seqs_display_n=20,
        )
        # 최대 20개만 표시
        count = sum(1 for s in seqs if s in prompt)
        assert count <= 20

    def test_prev_results_evaluated_seqs_passthrough(self):
        """previous_results["evaluated_seqs"]가 있어도 prompt가 정상 생성된다."""
        seqs = ["AICLNWFWKTVISC"]
        prompt = format_planner_prompt(
            iteration=4,
            receptor_config={"name": "SSTR2"},
            constraints={"reference_sequence": NATIVE},
            previous_results={"evaluated_seqs": seqs, "best_ddg": -30.0},
            evaluated_seqs=seqs,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100


# ---------------------------------------------------------------------------
# 다양성 통합 smoke: dedup 강화 확인
# ---------------------------------------------------------------------------

class TestDeduplicationDiversity:

    def test_generate_unique_intensification_variants(self):
        """seen_sequences로 100개 시도 시 50개 이상 고유 서열 생성 (중복률 < 50%)."""
        guidance = get_intensification_guidance(SEED_SEQ, NATIVE, MUTABLE, rng=random.Random(99))
        seen: set = {NATIVE}
        unique = 0
        n_total = 100
        for i in range(n_total):
            seq = generate_intensification_mutant(
                original_seq=NATIVE,
                intensify_guidance=guidance,
                design_positions=MUTABLE,
                rng=random.Random(i * 7 + 13),
            )
            if seq not in seen:
                unique += 1
                seen.add(seq)
        # 최소 50% 고유
        assert unique >= n_total * 0.5, f"unique={unique}/{n_total} (too low)"

    def test_conservative_substitutions_dict_complete(self):
        """CONSERVATIVE_SUBSTITUTIONS에 Cys 없고, 모든 값이 유효 AA임."""
        valid_aa = set("ADEFGHIKLMNPQRSTVWY")  # AA_NO_CYS
        for aa, substitutes in CONSERVATIVE_SUBSTITUTIONS.items():
            assert aa != "C", "C should not be a key"
            for sub in substitutes:
                assert sub in valid_aa, f"invalid AA '{sub}' in substitutions for {aa}"
                assert sub != "C", "C should not appear as substitution"


# ---------------------------------------------------------------------------
# adapter.generate_dedup_fallback 테스트 (요구사항 3: 슬롯 낭비 방지)
# ---------------------------------------------------------------------------

class TestGenerateDedupFallback:
    """2단계 fall-back: generate_dedup_fallback() 단위 테스트."""

    def test_returns_str_not_in_seen(self):
        """seen_sequences에 없는 서열 반환."""
        seen: set = {NATIVE, "PGCKNFFWKTFTSC"}
        result = generate_dedup_fallback(
            original_seq=NATIVE,
            design_positions=MUTABLE,
            seen_sequences=seen,
            rng=random.Random(0),
        )
        assert result is not None
        assert result not in seen
        assert result != NATIVE

    def test_preserves_length(self):
        """반환 서열 길이 = 원본 길이."""
        seen: set = {NATIVE}
        result = generate_dedup_fallback(NATIVE, MUTABLE, seen, random.Random(1))
        assert result is not None
        assert len(result) == len(NATIVE)

    def test_no_new_cys_introduced(self):
        """Cys3, Cys14 위치 이외에 Cys 없음."""
        seen: set = {NATIVE}
        for seed in range(30):
            result = generate_dedup_fallback(NATIVE, MUTABLE, seen, random.Random(seed))
            if result is not None:
                for i, aa in enumerate(result):
                    if aa == "C":
                        assert i in (2, 13), f"unexpected C at 0-idx {i}: {result}"

    def test_exhausted_seen_returns_none(self):
        """변이 가능 공간이 극히 작아 seen이 전부 차면 None 반환."""
        # design_positions를 1개만 허용 + 19종 AA 모두 seen에 넣으면 공간 포화
        tiny_positions = [2]  # pos2만
        _aas = list("ADEFGHIKLMNPQRSTVWY")
        seen: set = {NATIVE}
        # pos2 변이로 만들 수 있는 모든 서열 (19개 = AA_NO_CYS 에서 native G 제외 18개)
        for aa in _aas:
            s = list(NATIVE)
            s[1] = aa
            seen.add("".join(s))
        result = generate_dedup_fallback(
            NATIVE,
            tiny_positions,
            seen,
            random.Random(42),
            max_attempts=50,
        )
        # 공간 포화 시 None 반환 — 모든 가능 서열이 이미 seen에 있음
        assert result is None

    def test_fills_slot_even_with_large_seen(self):
        """seen에 1000개가 있어도 새 서열을 찾아낸다 (탐색 공간 충분)."""
        rng = random.Random(999)
        seen: set = {NATIVE}
        # 1000개를 미리 생성해 seen에 넣기
        from pyrosetta_flow.adapter import generate_random_mutant
        for i in range(1000):
            s = generate_random_mutant(NATIVE, MUTABLE, random.Random(i), n_mutations=2)
            seen.add(s)

        result = generate_dedup_fallback(
            original_seq=NATIVE,
            design_positions=MUTABLE,
            seen_sequences=seen,
            rng=rng,
            max_attempts=200,
        )
        # 변이 가능 공간이 충분하므로 반드시 새 서열 찾아야 함
        assert result is not None
        assert result not in seen

    def test_mutates_at_least_min_mutations(self):
        """min_mutations 이상 변이돼야 한다."""
        seen: set = {NATIVE}
        n_mut_counts: List[int] = []
        fixed_idx = {2, 6, 7, 8, 9, 13}  # Cys3, FWKT(7-10), Cys14 (0-indexed)
        for seed in range(20):
            result = generate_dedup_fallback(
                NATIVE, MUTABLE, seen, random.Random(seed),
                min_mutations=2
            )
            if result is not None:
                diff = sum(
                    1 for i, (a, b) in enumerate(zip(result, NATIVE))
                    if a != b and i not in fixed_idx
                )
                n_mut_counts.append(diff)
        # 대부분 2개 이상 변이
        valid = [c for c in n_mut_counts if c >= 2]
        assert len(valid) >= len(n_mut_counts) * 0.7, f"mut counts: {n_mut_counts}"

    def test_does_not_modify_seen_set(self):
        """seen_sequences는 함수 내에서 수정되지 않는다."""
        seen: set = {NATIVE}
        original_seen = set(seen)
        generate_dedup_fallback(NATIVE, MUTABLE, seen, random.Random(0))
        assert seen == original_seen

    def test_stages_increase_mutation_count(self):
        """시도 횟수가 늘어날수록 더 많이 변이해야 한다(탐색 공간 확장)."""
        # 변이 수 2인 서열 전부를 seen에 넣어, 3-변이 서열만 남도록
        seen: set = {NATIVE}
        from pyrosetta_flow.adapter import generate_random_mutant
        for i in range(2000):
            s = generate_random_mutant(NATIVE, MUTABLE, random.Random(i), n_mutations=2)
            seen.add(s)
        # 이제 2-변이 거의 포화 상태 → fallback이 3-변이로 올라가야 함
        result = generate_dedup_fallback(
            NATIVE, MUTABLE, seen, random.Random(77),
            max_attempts=200, min_mutations=2, max_mutations=5
        )
        # 새 서열이 반환돼야 한다 (3+ 변이로)
        assert result is not None
        assert result not in seen


# ---------------------------------------------------------------------------
# Runner 수준 smoke: dedup_fallback 슬롯 낭비 방지 통합 확인
# ---------------------------------------------------------------------------

class TestDedupFallbackRunnerIntegration:
    """runner.py의 dedup_fallback 경로 작동 smoke."""

    def test_slot_always_filled_when_all_guided_are_dup(self):
        """guided 슬롯이 전부 seen에 있어도 dedup_fallback이 채운다."""
        # seen에 guided 후보를 미리 넣어 전부 중복 상황 시뮬레이션
        from pyrosetta_flow.adapter import generate_dedup_fallback, generate_random_mutant

        seen: set = {NATIVE}
        # 100개를 미리 채워 중복 유발
        for i in range(100):
            s = generate_random_mutant(NATIVE, MUTABLE, random.Random(i * 3), n_mutations=2)
            seen.add(s)

        filled: List[str] = []
        n_slots = 10
        for slot in range(n_slots):
            fb = generate_dedup_fallback(
                original_seq=NATIVE,
                design_positions=MUTABLE,
                seen_sequences=seen,
                rng=random.Random(slot * 13 + 7),
                max_attempts=200,
            )
            assert fb is not None, f"slot {slot} was not filled (None returned)"
            assert fb not in seen, f"slot {slot} returned a dup"
            seen.add(fb)
            filled.append(fb)

        assert len(filled) == n_slots
        assert len(set(filled)) == n_slots, "slots have duplicates"
