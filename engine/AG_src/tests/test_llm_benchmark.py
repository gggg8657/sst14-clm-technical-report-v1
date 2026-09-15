"""test_llm_benchmark.py
=========================
M4 (2026-05-20): LLM 비교실험 인프라 — agent-level prompt 응답 품질 비교.

memory/project_agent_flow_benchmark.md 비교 축:
- Planner: 변이 전략 다양성, JSON parsing 성공률
- Critic: 실패 원인 분석 정확도
- Reporter: 문서 작성 품질
- 추론 속도 (latency)

현재 자원:
- vLLM Qwen3.5-35B-A3B (port 8000) ✅
- ollama qwen3:8b (멈춤 — GPU 0/1 좀비, task #3 deferred)

본 모듈: vLLM thinking on/off + 시나리오별 응답 비교 (인프라 + smoke test).
실 비교실험 (다중 모델, 5 iteration full pipeline)은 ollama 좀비 정리 후 별도 sprint.
"""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AG_src.llm.provider import VLLMProvider


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark scenarios — pipeline의 실 agent prompt 패턴 모방
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SCENARIO = (
    "당신은 SSTR2 펩타이드 디자이너입니다. SST-14 (AGCKNFFWKTFTSC)의 "
    "Cys3-Cys14 disulfide와 FWKT(7-10) pharmacophore는 보존하면서, "
    "mutable positions [1,2,4,5,6,11,12,13] 중 3개를 무작위로 mutate한 "
    "변이체 1개를 JSON으로 제안해주세요. 형식: "
    '{"variant_id": "v01", "sequence": "...", "mutations": [{"pos": ..., "from": ..., "to": ...}]}'
)

CRITIC_SCENARIO = (
    "이전 iteration ddG=-30.5 (baseline -48.4 대비 +17.9). "
    "Critic으로서 (1) 실패 원인 1줄, (2) 개선 제안 1줄을 JSON으로: "
    '{"failure_reason": "...", "improvement_suggestion": "..."}'
)

REPORTER_SCENARIO = (
    "다음 결과 JSON으로 요약: variants=[{seq: AGCKNFFWKTLTSC, ddg: -45.2}, "
    "{seq: AGCKNFFWKTYTSC, ddg: -48.8}]. "
    'Output: {"best_variant": "...", "best_ddg": ..., "summary": "..."}'
)


@pytest.mark.integration
@pytest.mark.network
class TestVLLMBenchmarkSmoke(unittest.TestCase):
    """vLLM smoke test — 4 시나리오 모두 정상 응답 + JSON parsing."""

    @classmethod
    def setUpClass(cls):
        cls.provider = VLLMProvider(
            model="qwen3-32b",
            base_url="http://localhost:8000",
            enable_thinking=False,
            timeout=60,
        )
        # 사전 health-check — vLLM 가동 안 됐으면 skip
        try:
            import urllib.request
            req = urllib.request.Request("http://localhost:8000/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"vLLM port 8000 unreachable: {e}")

    def test_planner_scenario_returns_json(self):
        start = time.time()
        result = self.provider.generate_json(
            PLANNER_SCENARIO, max_tokens=512,
        )
        elapsed = time.time() - start
        self.assertIsNotNone(result, "Planner JSON parsing failed")
        self.assertIn("sequence", result, f"sequence key missing: {result}")
        # SST-14 길이 14 유지
        self.assertEqual(len(result["sequence"]), 14,
                         f"sequence length not 14: {result['sequence']}")
        # Cys3 / Cys14 보존
        self.assertEqual(result["sequence"][2], "C", "Cys3 보존 위반")
        self.assertEqual(result["sequence"][13], "C", "Cys14 보존 위반")
        # FWKT (7-10, 1-indexed = 6-9 0-indexed) 보존
        self.assertEqual(result["sequence"][6:10], "FWKT", "FWKT pharmacophore 위반")
        print(f"\n  [planner] {elapsed:.2f}s, variant={result.get('variant_id')} "
              f"seq={result['sequence']}")

    def test_critic_scenario_returns_json(self):
        start = time.time()
        result = self.provider.generate_json(CRITIC_SCENARIO, max_tokens=300)
        elapsed = time.time() - start
        self.assertIsNotNone(result, "Critic JSON parsing failed")
        self.assertIn("failure_reason", result)
        self.assertIn("improvement_suggestion", result)
        print(f"\n  [critic] {elapsed:.2f}s, reason={result['failure_reason'][:60]!r}")

    def test_reporter_scenario_returns_json(self):
        start = time.time()
        result = self.provider.generate_json(REPORTER_SCENARIO, max_tokens=300)
        elapsed = time.time() - start
        self.assertIsNotNone(result, "Reporter JSON parsing failed")
        self.assertIn("best_variant", result)
        self.assertIn("best_ddg", result)
        print(f"\n  [reporter] {elapsed:.2f}s, best={result['best_variant']!r} "
              f"ddg={result['best_ddg']}")


@pytest.mark.integration
@pytest.mark.network
class TestVLLMThinkingABComparison(unittest.TestCase):
    """A/B 비교: enable_thinking=True vs False — 동일 prompt 응답 시간 + 품질."""

    @classmethod
    def setUpClass(cls):
        cls.p_no_think = VLLMProvider(
            model="qwen3-32b",
            base_url="http://localhost:8000",
            enable_thinking=False,
            timeout=120,
        )
        cls.p_with_think = VLLMProvider(
            model="qwen3-32b",
            base_url="http://localhost:8000",
            enable_thinking=True,
            timeout=120,
        )
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3):
                pass
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f"vLLM unreachable: {e}")

    def test_planner_thinking_vs_no_thinking(self):
        """동일 planner prompt — thinking on/off로 응답 시간 + 결과 분석."""
        # No-thinking
        t0 = time.time()
        no_think_result = self.p_no_think.generate_json(
            PLANNER_SCENARIO, max_tokens=512,
        )
        no_think_elapsed = time.time() - t0
        # With-thinking — token 더 필요 (reasoning 출력 후 content)
        t1 = time.time()
        with_think_raw = self.p_with_think.generate(
            PLANNER_SCENARIO, json_mode=False, max_tokens=2048,
        )
        with_think_elapsed = time.time() - t1

        self.assertIsNotNone(no_think_result)
        self.assertIsNotNone(with_think_raw)
        print(
            f"\n  [A/B planner]\n"
            f"    no_thinking:   {no_think_elapsed:6.2f}s, JSON: "
            f"{'✓' if no_think_result else '✗'}\n"
            f"    with_thinking: {with_think_elapsed:6.2f}s, raw len: "
            f"{len(with_think_raw) if with_think_raw else 0}\n"
            f"    speedup (no_thinking faster): "
            f"{with_think_elapsed/no_think_elapsed:.2f}x"
        )
