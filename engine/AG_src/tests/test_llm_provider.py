"""test_llm_provider.py
=========================
LLM Provider 단위 테스트 — Qwen3 thinking mode 호환성 + factory 분기.

2026-05-20 SOD: vLLM qwen3.5-35b-a3b 업그레이드에 따른 회귀 보장.
- Qwen3 thinking 모델은 content=null, reasoning에 thought process 분리
- chat_template_kwargs={"enable_thinking": False} 강제 → content만 받음
- content 빈값일 때 reasoning fallback
"""
from __future__ import annotations

import json
import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from AG_src.llm.provider import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_VLLM_URL,
    NoneProvider,
    OllamaProvider,
    VLLMProvider,
    create_provider,
    _extract_json_block,
    _strip_think_blocks,
)


class TestFactoryProviderSelection(unittest.TestCase):
    """create_provider() 분기 검증."""

    def test_none_config_returns_none_provider(self):
        p = create_provider(None)
        self.assertIsInstance(p, NoneProvider)
        self.assertIsNone(p.generate("test"))

    def test_ollama_provider_factory(self):
        cfg = {"llm": {"provider": "ollama", "model": "qwen3:32b",
                       "base_url": "http://localhost:11434"}}
        p = create_provider(cfg)
        self.assertIsInstance(p, OllamaProvider)
        self.assertEqual(p.model, "qwen3:32b")
        self.assertEqual(p.base_url, "http://localhost:11434")

    def test_vllm_provider_factory(self):
        cfg = {"llm": {"provider": "vllm", "model": "qwen3.5-35b-a3b",
                       "base_url": "http://localhost:8000",
                       "enable_thinking": False}}
        p = create_provider(cfg)
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "qwen3.5-35b-a3b")
        self.assertFalse(p.enable_thinking)

    def test_vllm_default_enable_thinking_false(self):
        """기본값: thinking 비활성 (pipeline은 final answer만 사용)."""
        self.assertFalse(DEFAULT_ENABLE_THINKING)
        cfg = {"llm": {"provider": "vllm", "model": "any"}}
        p = create_provider(cfg)
        self.assertFalse(p.enable_thinking)

    def test_unknown_provider_falls_back_to_none(self):
        cfg = {"llm": {"provider": "unknown_xyz"}}
        p = create_provider(cfg)
        self.assertIsInstance(p, NoneProvider)


class TestVLLMQwen3ThinkingCompat(unittest.TestCase):
    """vLLM Qwen3 thinking model 호환성 — content 추출 + reasoning fallback."""

    def test_payload_includes_chat_template_kwargs(self):
        """generate() payload에 chat_template_kwargs={enable_thinking: False} 포함."""
        p = VLLMProvider(model="qwen3.5-35b-a3b", enable_thinking=False)
        captured_payload = {}

        def fake_post(url, payload):
            captured_payload.update(payload)
            return "pong"

        with patch.object(p, "_post", side_effect=fake_post):
            p.generate("test")

        self.assertIn("chat_template_kwargs", captured_payload)
        self.assertEqual(
            captured_payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_payload_thinking_enabled_when_set(self):
        """enable_thinking=True 설정 시 payload에 반영."""
        p = VLLMProvider(model="any", enable_thinking=True)
        captured = {}
        with patch.object(p, "_post", side_effect=lambda u, pl: captured.update(pl)):
            p.generate("test")
        self.assertEqual(
            captured["chat_template_kwargs"]["enable_thinking"], True,
        )

    def test_content_null_falls_back_to_reasoning(self):
        """Qwen3 thinking model: content=null + reasoning 있으면 reasoning 반환."""
        p = VLLMProvider(model="any")
        fake_response = json.dumps({
            "choices": [{
                "message": {
                    "content": None,
                    "reasoning": "Thinking: ...the answer is pong.",
                },
                "finish_reason": "length",
            }],
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = p.generate("test")
        self.assertIn("pong", result)
        self.assertIn("Thinking", result)

    def test_content_present_returns_content_not_reasoning(self):
        """content가 있으면 reasoning 무시하고 content 반환."""
        p = VLLMProvider(model="any")
        fake_response = json.dumps({
            "choices": [{
                "message": {
                    "content": "pong",
                    "reasoning": "thought process",
                },
                "finish_reason": "stop",
            }],
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = p.generate("test")
        self.assertEqual(result, "pong")

    def test_send_chat_template_kwargs_default_true_backward_compat(self):
        """send_chat_template_kwargs 기본값 True — 기존 Qwen 동작과 100% 동일(회귀 없음)."""
        p = VLLMProvider(model="qwen3-32b")
        self.assertTrue(p.send_chat_template_kwargs)
        captured = {}
        with patch.object(p, "_post", side_effect=lambda u, pl: captured.update(pl)):
            p.generate("test")
        self.assertIn("chat_template_kwargs", captured)

    def test_send_chat_template_kwargs_false_omits_field(self):
        """send_chat_template_kwargs=False — payload에서 필드 자체가 빠진다.

        배경: 이질(heterogeneous) 모델 배선(Mistral 등) 실측 결과, chat_template_kwargs
        필드 자체가 있으면 Mistral 토크나이저가 HTTP 400("chat_template is not supported
        for Mistral tokenizers.")을 반환한다 — 필드를 아예 보내지 않아야 한다.
        """
        p = VLLMProvider(model="mistral-7b-instruct", send_chat_template_kwargs=False)
        captured = {"chat_template_kwargs": "sentinel"}  # 남아있으면 실패하도록 sentinel
        captured.clear()
        with patch.object(p, "_post", side_effect=lambda u, pl: captured.update(pl)):
            p.generate("test")
        self.assertNotIn("chat_template_kwargs", captured,
                          "send_chat_template_kwargs=False인데 필드가 payload에 포함됨")

    def test_content_alternative_reasoning_content_key(self):
        """일부 모델은 'reasoning_content' 키 사용 — fallback 호환."""
        p = VLLMProvider(model="any")
        fake_response = json.dumps({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": "alternative key thought",
                },
                "finish_reason": "stop",
            }],
        }).encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.read.return_value = fake_response
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = p.generate("test")
        self.assertEqual(result, "alternative key thought")


class TestJSONExtractionHelper(unittest.TestCase):
    """_extract_json_block 헬퍼 회귀."""

    def test_markdown_json_block(self):
        text = 'Here is the answer:\n```json\n{"a": 1, "b": "x"}\n```\n'
        result = _extract_json_block(text)
        self.assertEqual(result, {"a": 1, "b": "x"})

    def test_brace_extraction_fallback(self):
        text = 'preamble text {"key": "val"} trailing'
        result = _extract_json_block(text)
        self.assertEqual(result, {"key": "val"})

    def test_invalid_returns_none(self):
        result = _extract_json_block("no json here")
        self.assertIsNone(result)


class TestNoneProviderBehavior(unittest.TestCase):
    """NoneProvider 규칙 기반 폴백."""

    def test_always_returns_none(self):
        p = NoneProvider()
        self.assertIsNone(p.generate("any"))
        self.assertIsNone(p.generate_json("any"))


class TestDeepSeekR1ThinkBlockStrip(unittest.TestCase):
    """P4 (2026-05-20): DeepSeek-R1 <think>...</think> 블록 파싱 회귀.

    실 vLLM port 8001 응답 패턴:
        '...long reasoning...</think>\\n\\n{"answer":"pong"}'
    """

    def test_strip_closed_think_block(self):
        text = "<think>\nlong reasoning here\nmore reasoning\n</think>\n\n{\"answer\":\"pong\"}"
        result = _strip_think_blocks(text)
        self.assertEqual(result, '{"answer":"pong"}')

    def test_strip_unclosed_think_block(self):
        """max_tokens 초과로 </think> 미출력 시 — <think> 이후 모두 제거."""
        text = "<think>\nthis reasoning never closed because max_tokens"
        result = _strip_think_blocks(text)
        self.assertEqual(result, "")

    def test_strip_real_deepseek_pattern(self):
        """실 DeepSeek-R1 응답 패턴 (장문 reasoning + 최종 JSON)."""
        text = (
            "Okay, so I need to figure out how to respond. The user wrote, "
            '"Output ONLY this JSON". I should comply directly.\n</think>\n\n'
            '{"answer":"pong"}'
        )
        # 닫는 </think>만 있고 여는 <think>는 없는 경우 — 시작이 implicit
        # 본 패턴은 _strip_think_blocks가 처리 못 함 — _extract_json_block이 brace 추출
        result = _extract_json_block(text)
        self.assertEqual(result, {"answer": "pong"})

    def test_strip_preserves_non_think_text(self):
        """think 블록 없는 일반 텍스트는 그대로."""
        text = '{"answer":"pong"}'
        self.assertEqual(_strip_think_blocks(text), text)

    def test_generate_json_strips_think_before_parsing(self):
        """generate_json — <think> 통째 응답에서 JSON 추출."""
        p = VLLMProvider(model="deepseek-r1-distill-32b")
        with unittest.mock.patch.object(
            p, "generate",
            return_value='<think>analyzing...</think>\n{"answer":"pong"}',
        ):
            result = p.generate_json("test")
        self.assertEqual(result, {"answer": "pong"})


class TestAgentOverride(unittest.TestCase):
    """M3 (2026-05-20): agent별 LLM override 회귀."""

    BASE_CFG = {
        "llm": {
            "provider": "vllm",
            "model": "qwen3.5-35b-a3b",
            "base_url": "http://localhost:8000",
            "enable_thinking": False,
            "agents": {
                "planner": {
                    "provider": "ollama",
                    "model": "deepseek-r1:70b",
                    "base_url": "http://localhost:11434",
                },
                "critic": {
                    "provider": "vllm",
                    "model": "qwen3.5-35b-a3b",
                },
                # reporter는 override 없음 — 상위 llm 값 상속
            },
        },
    }

    def test_default_uses_base_llm_when_no_agent_name(self):
        p = create_provider(self.BASE_CFG)
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "qwen3.5-35b-a3b")

    def test_planner_override_to_ollama(self):
        p = create_provider(self.BASE_CFG, agent_name="planner")
        self.assertIsInstance(p, OllamaProvider)
        self.assertEqual(p.model, "deepseek-r1:70b")
        self.assertEqual(p.base_url, "http://localhost:11434")

    def test_critic_keeps_vllm_with_different_model(self):
        p = create_provider(self.BASE_CFG, agent_name="critic")
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "qwen3.5-35b-a3b")

    def test_reporter_inherits_base_llm(self):
        """reporter는 agents.reporter 부재 — 상위 llm 값 상속."""
        p = create_provider(self.BASE_CFG, agent_name="reporter")
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "qwen3.5-35b-a3b")

    def test_unknown_agent_inherits_base(self):
        """알 수 없는 agent_name — 상위 llm 값 그대로."""
        p = create_provider(self.BASE_CFG, agent_name="unknown_agent_xyz")
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "qwen3.5-35b-a3b")

    def test_no_agents_section_back_compat(self):
        """llm.agents 섹션이 없을 때도 정상 작동 (back-compat)."""
        cfg = {"llm": {"provider": "ollama", "model": "qwen3:32b",
                       "base_url": "http://localhost:11434"}}
        p = create_provider(cfg, agent_name="planner")
        self.assertIsInstance(p, OllamaProvider)
        self.assertEqual(p.model, "qwen3:32b")

    def test_agent_override_partial_key_inherits_rest(self):
        """override가 일부 키만 지정 — 나머지는 상위에서 상속."""
        cfg = {
            "llm": {
                "provider": "vllm",
                "model": "base-model",
                "base_url": "http://base:9999",
                "timeout": 999,
                "agents": {
                    "planner": {"model": "override-only"},  # model만 override
                },
            },
        }
        p = create_provider(cfg, agent_name="planner")
        self.assertIsInstance(p, VLLMProvider)
        self.assertEqual(p.model, "override-only")
        # provider, base_url, timeout은 상위에서 상속
        self.assertEqual(p.base_url, "http://base:9999")
        self.assertEqual(p.timeout, 999)


if __name__ == "__main__":
    unittest.main()
