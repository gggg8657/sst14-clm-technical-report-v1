"""test_planner_prompt.py (P2 — Planner prompt engineering mutations 강제 회귀)
M4 발견 버그 픽스 검증 — Planner prompt engineering.

M4 비교실험에서 발견:
  Qwen3.5-35B-A3B가 mutations 명시 prompt에도 원본 시퀀스를 그대로 반환
  (보수적 응답 — mutations=[]).

이 파일은 두 계층으로 검증한다:
  1. TestVariantGenerationPromptStructure — 프롬프트 구조 단위 테스트 (LLM 불필요)
  2. TestPlannerMutationPromptLive — 실 vLLM 호출 테스트 (vLLM 미구동 시 skip)

실행:
    pytest AG_src/tests/test_llm_benchmark.py -v
    pytest AG_src/tests/test_llm_benchmark.py -v -k live  # live 테스트만
"""

from __future__ import annotations

import os
import socket
import unittest
from typing import List, Optional

import pytest

from AG_src.llm.prompts import (
    VARIANT_DESIGN_SYSTEM_PROMPT,
    _SST14_FEW_SHOT_EXAMPLES,
    build_variant_generation_prompt,
    format_planner_prompt,
    get_system_prompt,
)
from AG_src.llm.provider import VLLMProvider

# ---------------------------------------------------------------------------
# 공통 상수
# ---------------------------------------------------------------------------

SST14_REF = "AGCKNFFWKTFTSC"
MUTABLE_POS: List[int] = [1, 2, 4, 5, 6, 11, 12, 13]
FIXED_POS_SET = {3, 7, 8, 9, 10, 14}  # 0-indexed → 1-indexed로 저장
N_MUTATIONS = 3
VLLM_HOST = os.environ.get("VLLM_HOST", "localhost")
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen3.5-35b-a3b")


def _is_vllm_available(host: str = VLLM_HOST, port: int = VLLM_PORT) -> bool:
    """vLLM 서버가 구동 중인지 TCP 소켓으로 확인한다."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _validate_variant_output(
    result: dict,
    expected_n_mutations: int,
    reference_sequence: str,
    mutable_positions: List[int],
) -> Optional[str]:
    """변이 생성 출력 유효성 검사.

    Returns:
        None: 유효 (검사 통과)
        str: 실패 사유 (검사 실패)
    """
    # 필수 키 확인
    for key in ("sequence", "mutations"):
        if key not in result:
            return f"필수 키 '{key}' 누락: {list(result.keys())}"

    seq = result["sequence"]
    mutations = result["mutations"]

    # 시퀀스 길이
    if len(seq) != len(reference_sequence):
        return (
            f"시퀀스 길이 불일치: expected {len(reference_sequence)}, got {len(seq)}"
        )

    # 이황화결합 Cys 보존 (pos 3, 14 — 0-indexed 2, 13)
    if seq[2] != "C":
        return f"C3 (이황화결합) 변이됨: got '{seq[2]}'"
    if seq[13] != "C":
        return f"C14 (이황화결합) 변이됨: got '{seq[13]}'"

    # FWKT pharmacophore 보존 (pos 7,8,9,10 — 0-indexed 6,7,8,9)
    pharmacophore = seq[6:10]
    if pharmacophore != "FWKT":
        return f"FWKT pharmacophore 변이됨: got '{pharmacophore}'"

    # mutations 배열 길이
    if len(mutations) != expected_n_mutations:
        return (
            f"mutations 개수 불일치: expected {expected_n_mutations}, got {len(mutations)}"
        )

    # 각 변이 유효성
    mutable_set = set(mutable_positions)
    for m in mutations:
        pos = m.get("pos")
        from_aa = m.get("from")
        to_aa = m.get("to")

        if pos is None or from_aa is None or to_aa is None:
            return f"변이 항목에 필수 키 누락: {m}"

        if pos not in mutable_set:
            return f"변이 위치 {pos}는 mutable_positions에 없음"

        if from_aa == to_aa:
            return f"pos={pos}: from_aa == to_aa = '{from_aa}' (no-op 변이)"

        if from_aa != reference_sequence[pos - 1]:
            return (
                f"pos={pos}: from_aa='{from_aa}' != ref='{reference_sequence[pos - 1]}'"
            )

    return None  # 유효


# ---------------------------------------------------------------------------
# 1. 프롬프트 구조 단위 테스트 (LLM 불필요)
# ---------------------------------------------------------------------------


class TestVariantGenerationPromptStructure(unittest.TestCase):
    """build_variant_generation_prompt가 올바른 구조를 생성하는지 검사."""

    def setUp(self) -> None:
        self.prompt = build_variant_generation_prompt(
            reference_sequence=SST14_REF,
            mutable_positions=MUTABLE_POS,
            n_mutations=N_MUTATIONS,
            variant_id="v03",
        )

    def test_prompt_not_empty(self) -> None:
        """프롬프트가 비어 있지 않다."""
        self.assertTrue(len(self.prompt) > 100, "프롬프트가 너무 짧음")

    def test_prompt_contains_must_directive(self) -> None:
        """프롬프트에 MUST 대문자 강제 지시가 포함되어야 한다."""
        self.assertIn("MUST", self.prompt)
        self.assertIn("EXACTLY", self.prompt)

    def test_prompt_contains_n_mutations_count(self) -> None:
        """프롬프트에 n_mutations 수가 명시되어야 한다."""
        self.assertIn(str(N_MUTATIONS), self.prompt)

    def test_prompt_contains_reference_sequence(self) -> None:
        """프롬프트에 참조 시퀀스가 포함되어야 한다."""
        self.assertIn(SST14_REF, self.prompt)

    def test_prompt_contains_mutable_positions(self) -> None:
        """프롬프트에 mutable_positions가 포함되어야 한다."""
        # 적어도 일부 mutable position이 언급되어야 함
        mutable_mentioned = any(
            str(pos) in self.prompt for pos in MUTABLE_POS
        )
        self.assertTrue(mutable_mentioned, "mutable positions가 프롬프트에 없음")

    def test_prompt_contains_few_shot_examples(self) -> None:
        """few-shot 예시가 최소 2개 포함되어야 한다."""
        # 검증된 few-shot 출력 시퀀스가 포함되어 있는지 확인
        for ex in _SST14_FEW_SHOT_EXAMPLES:
            self.assertIn(
                ex["output_seq"],
                self.prompt,
                f"few-shot 예시 시퀀스 '{ex['output_seq']}' 누락",
            )

    def test_prompt_contains_variant_id(self) -> None:
        """프롬프트에 요청한 variant_id가 포함되어야 한다."""
        self.assertIn("v03", self.prompt)

    def test_prompt_warns_original_return_is_failure(self) -> None:
        """원본 반환이 FAILURE임을 명시해야 한다."""
        prompt_lower = self.prompt.lower()
        self.assertIn("failure", prompt_lower)

    def test_few_shot_examples_are_valid(self) -> None:
        """_SST14_FEW_SHOT_EXAMPLES의 내용이 과학적으로 유효해야 한다."""
        for ex in _SST14_FEW_SHOT_EXAMPLES:
            with self.subTest(vid=ex["vid"]):
                import json
                mutations = json.loads(ex["mutations_json"])
                err = _validate_variant_output(
                    {
                        "variant_id": ex["vid"],
                        "sequence": ex["output_seq"],
                        "mutations": mutations,
                    },
                    expected_n_mutations=int(ex["n"]),
                    reference_sequence=ex["input_seq"],
                    mutable_positions=[1, 2, 4, 5, 6, 11, 12, 13],
                )
                self.assertIsNone(
                    err,
                    f"few-shot 예시 '{ex['vid']}' 유효성 실패: {err}",
                )


class TestVariantDesignSystemPrompt(unittest.TestCase):
    """VARIANT_DESIGN_SYSTEM_PROMPT 구조 검사."""

    def test_system_prompt_contains_must(self) -> None:
        """시스템 프롬프트에 MUST 지시가 있어야 한다."""
        self.assertIn("MUST", VARIANT_DESIGN_SYSTEM_PROMPT)

    def test_system_prompt_mentions_disulfide(self) -> None:
        """이황화결합(C3, C14) 보존 규칙이 명시되어야 한다."""
        self.assertIn("C3", VARIANT_DESIGN_SYSTEM_PROMPT)
        self.assertIn("C14", VARIANT_DESIGN_SYSTEM_PROMPT)

    def test_system_prompt_mentions_pharmacophore(self) -> None:
        """FWKT pharmacophore 보존 규칙이 명시되어야 한다."""
        self.assertIn("FWKT", VARIANT_DESIGN_SYSTEM_PROMPT)


class TestPlannerSystemPromptEnforcement(unittest.TestCase):
    """Planner 시스템 프롬프트 강화 검사 (pyrosetta_only 모드)."""

    def setUp(self) -> None:
        self.system_prompt = get_system_prompt("planner", planner_mode="pyrosetta_only")

    def test_system_prompt_has_must(self) -> None:
        """pyrosetta_only 시스템 프롬프트에 MUST가 포함되어야 한다."""
        self.assertIn("MUST", self.system_prompt)

    def test_system_prompt_has_failure_consequence(self) -> None:
        """빈 mutation_guidance가 FAILURE임을 명시해야 한다."""
        self.assertIn("FAILURE", self.system_prompt)

    def test_format_planner_prompt_pyrosetta_enforces_mutations(self) -> None:
        """format_planner_prompt pyrosetta_only가 강화된 mutation 지시를 포함해야 한다."""
        prompt = format_planner_prompt(
            iteration=1,
            receptor_config={"name": "SSTR2"},
            constraints={
                "reference_sequence": SST14_REF,
                "design_positions": MUTABLE_POS,
            },
            planner_mode="pyrosetta_only",
        )
        # 강화된 규칙이 포함되어 있는지 확인
        self.assertIn("MUST", prompt)
        self.assertIn("INVALID", prompt)


class TestVLLMProviderEnableThinking(unittest.TestCase):
    """VLLMProvider enable_thinking 파라미터 검사."""

    def test_default_enable_thinking_false(self) -> None:
        """기본값은 enable_thinking=False (M4 14.99x speedup + content null 회피).

        Planner agent 등 thinking이 도움되는 경우는 명시적으로 True 지정 (M3 override).
        """
        p = VLLMProvider(model="test-model", base_url="http://localhost:8000")
        self.assertFalse(p.enable_thinking)

    def test_enable_thinking_false_stored(self) -> None:
        """enable_thinking=False 설정이 저장되어야 한다."""
        p = VLLMProvider(
            model="test-model",
            base_url="http://localhost:8000",
            enable_thinking=False,
        )
        self.assertFalse(p.enable_thinking)

    def test_repr_includes_model(self) -> None:
        """repr에 모델명이 포함되어야 한다."""
        p = VLLMProvider(model="qwen3.5-35b-a3b", base_url="http://localhost:8000")
        self.assertIn("qwen3.5-35b-a3b", repr(p))


# ---------------------------------------------------------------------------
# 2. Live LLM 테스트 (vLLM 서버 미구동 시 skip)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _is_vllm_available(),
    f"vLLM 서버 미구동 ({VLLM_HOST}:{VLLM_PORT}) — live 테스트 skip",
)
@pytest.mark.integration
@pytest.mark.network
class TestPlannerMutationPromptLive(unittest.TestCase):
    """실 vLLM 호출로 Planner prompt mutation 강제 검증.

    M4 발견 버그 픽스: 강화된 프롬프트로 Qwen3.5-35B-A3B가 정확히
    N개 위치를 변이하는지 10회 반복 실행 → 성공률 ≥ 80% 목표.

    환경 변수:
        VLLM_HOST: vLLM 서버 호스트 (기본: localhost)
        VLLM_PORT: vLLM 서버 포트 (기본: 8000)
        VLLM_MODEL: 모델명 (기본: qwen3.5-35b-a3b)
        LLM_BENCH_REPEATS: 반복 횟수 (기본: 10)
        LLM_BENCH_MIN_SUCCESS: 최소 성공률 0-1 (기본: 0.8)
    """

    N_REPEATS: int = int(os.environ.get("LLM_BENCH_REPEATS", "10"))
    MIN_SUCCESS_RATE: float = float(os.environ.get("LLM_BENCH_MIN_SUCCESS", "0.8"))

    def _make_provider(self) -> VLLMProvider:
        # 본 통합 PR (PR #87 + #88): enable_thinking=False + 강화된 prompt 조합이 mutations 강제 효과 입증.
        # thinking=True 시 max_tokens 4096 부족 + reasoning만 차고 content 없음 → JSON parse 실패.
        return VLLMProvider(
            model=VLLM_MODEL,
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}",
            enable_thinking=False,
            timeout=90,
        )

    def test_planner_actually_mutates_three_positions(self) -> None:
        """강화된 프롬프트로 Qwen3.5가 정확히 3 positions를 mutate해야 한다.

        M4 발견 버그 픽스 검증.
        - 성공 기준: mutations 길이 == 3, 모든 변이 유효
        - 목표 성공률: ≥ 80% (10회 중 8회 이상)
        """
        provider = self._make_provider()
        prompt = build_variant_generation_prompt(
            reference_sequence=SST14_REF,
            mutable_positions=MUTABLE_POS,
            n_mutations=N_MUTATIONS,
        )

        successes = 0
        failures = []

        for trial in range(self.N_REPEATS):
            with self.subTest(trial=trial):
                result = provider.generate_json(
                    prompt,
                    system_prompt=VARIANT_DESIGN_SYSTEM_PROMPT,
                    max_tokens=1024,  # thinking 비활성 → 적은 토큰으로 충분
                )

                if result is None:
                    failures.append(f"trial {trial}: result is None")
                    continue

                err = _validate_variant_output(
                    result,
                    expected_n_mutations=N_MUTATIONS,
                    reference_sequence=SST14_REF,
                    mutable_positions=MUTABLE_POS,
                )
                if err is None:
                    successes += 1
                else:
                    failures.append(f"trial {trial}: {err} | result={result}")

        success_rate = successes / self.N_REPEATS
        fail_summary = "\n  ".join(failures[:5])  # 최대 5개만 출력
        self.assertGreaterEqual(
            success_rate,
            self.MIN_SUCCESS_RATE,
            f"성공률 {success_rate:.0%} < 목표 {self.MIN_SUCCESS_RATE:.0%} "
            f"({successes}/{self.N_REPEATS} 성공)\n"
            f"실패 사례:\n  {fail_summary}",
        )

    def test_variant_sequence_differs_from_reference(self) -> None:
        """생성된 시퀀스가 참조 시퀀스와 달라야 한다 (단일 호출)."""
        provider = self._make_provider()
        prompt = build_variant_generation_prompt(
            reference_sequence=SST14_REF,
            mutable_positions=MUTABLE_POS,
            n_mutations=N_MUTATIONS,
        )
        result = provider.generate_json(
            prompt,
            system_prompt=VARIANT_DESIGN_SYSTEM_PROMPT,
            max_tokens=1024,
        )

        self.assertIsNotNone(result, "LLM 응답이 None — vLLM 서버 응답 없음")
        seq = result.get("sequence", "")
        self.assertNotEqual(
            seq,
            SST14_REF,
            f"원본 시퀀스 그대로 반환됨: '{seq}' (M4 버그 재현)",
        )

    def test_cys_and_pharmacophore_preserved(self) -> None:
        """단일 호출에서 Cys 이황화결합 + FWKT pharmacophore가 보존되어야 한다."""
        provider = self._make_provider()
        prompt = build_variant_generation_prompt(
            reference_sequence=SST14_REF,
            mutable_positions=MUTABLE_POS,
            n_mutations=N_MUTATIONS,
        )
        result = provider.generate_json(
            prompt,
            system_prompt=VARIANT_DESIGN_SYSTEM_PROMPT,
            max_tokens=1024,
        )

        self.assertIsNotNone(result)
        seq = result.get("sequence", "")
        self.assertEqual(len(seq), 14, f"시퀀스 길이 오류: {len(seq)}")
        self.assertEqual(seq[2], "C", f"C3 변이됨: '{seq[2]}'")
        self.assertEqual(seq[13], "C", f"C14 변이됨: '{seq[13]}'")
        self.assertEqual(seq[6:10], "FWKT", f"FWKT 변이됨: '{seq[6:10]}'")
