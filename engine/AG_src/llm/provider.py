"""
provider.py
===========
LLM Provider abstraction layer.

Qwen3 8B 모델을 Ollama 또는 vLLM 백엔드로 호출하는 통합 인터페이스.
"none" 모드에서는 LLM을 호출하지 않고 None을 반환하여 기존 규칙 기반 로직이 동작한다.

Providers:
    - NoneProvider:   LLM 미사용 (규칙 기반 폴백)
    - OllamaProvider: Ollama REST API (http://localhost:11434)
    - VLLMProvider:   vLLM OpenAI-compatible API (http://localhost:8000)

Factory:
    create_provider(config) -> LLMProvider
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_VLLM_URL = "http://localhost:8000"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 60  # seconds
# 2026-06-09 (D4/F18): 일시적 전송 오류(URLError/timeout)에 대한 재시도. 에이전트 루프가
# iteration 당 LLM 을 여러 번 호출하므로, vLLM 의 순간적 장애가 조용히 rule-based 로
# degrade 되지 않도록 backoff 재시도한다. HTTP 4xx 등 영구 오류는 재시도하지 않는다.
DEFAULT_MAX_RETRIES = 2          # 총 시도 = 1 + DEFAULT_MAX_RETRIES
DEFAULT_RETRY_BACKOFF = 1.5      # seconds, 지수 백오프 베이스


def _http_post_json(
    url: str,
    payload: "Dict[str, Any]",
    headers: "Dict[str, str]",
    timeout: int,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> "Optional[Dict[str, Any]]":
    """POST JSON 후 응답 dict 반환. 일시적 오류(URLError/timeout/5xx)는 backoff 재시도.

    영구 오류(HTTP 4xx)는 즉시 None. 모든 시도 실패 시 None.
    """
    import time
    data = json.dumps(payload).encode("utf-8")
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # 4xx = 영구(요청 문제) → 재시도 안 함. 5xx = 일시 → 재시도.
            if 400 <= e.code < 500:
                logger.error("LLM HTTP %s (permanent, no retry): %s", e.code, url)
                return None
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
        except Exception as e:  # JSON 파싱 등 — 재시도 무의미하나 1회 한정 허용
            logger.error("LLM unexpected error: %s", e)
            return None
        if attempt < max_retries:
            sleep_s = DEFAULT_RETRY_BACKOFF * (2 ** attempt)
            logger.warning("LLM transient error (attempt %d/%d): %s — retry in %.1fs",
                           attempt + 1, max_retries + 1, last_err, sleep_s)
            time.sleep(sleep_s)
    logger.error("LLM request failed after %d attempts: %s", max_retries + 1, last_err)
    return None
# Qwen3 thinking 모델 (Qwen3.5-35B-A3B 등) 기본 동작: reasoning content만 출력하고 content=null.
# 본 파이프라인은 final answer만 사용하므로 thinking 비활성화 기본값.
DEFAULT_ENABLE_THINKING = False


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    """LLM 호출을 위한 추상 기본 클래스."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        """프롬프트에 대한 LLM 응답을 생성한다.

        Args:
            prompt: 사용자 프롬프트 (메인 질문/지시)
            system_prompt: 시스템 프롬프트 (역할 설정)
            json_mode: True이면 JSON 형식 응답을 강제
            temperature: 생성 온도 (0.0~1.0)
            max_tokens: 최대 생성 토큰 수

        Returns:
            LLM 응답 문자열. NoneProvider는 None 반환.
        """
        ...

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[Dict[str, Any]]:
        """JSON 모드로 생성하고 파싱된 dict를 반환한다.

        Returns:
            파싱된 JSON dict, 또는 파싱 실패 시 None.
        """
        raw = self.generate(
            prompt,
            system_prompt=system_prompt,
            json_mode=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if raw is None:
            return None
        # DeepSeek-R1 호환 (M4+ 후속): content에 <think>...</think> 블록 인라인 포함될 수 있음.
        # JSON 파싱 전 think 블록 제거.
        raw = _strip_think_blocks(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # JSON 블록 추출 시도 (```json ... ```)
            return _extract_json_block(raw)

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.provider_name}(model={self.model!r})"


# ---------------------------------------------------------------------------
# NoneProvider (rule-based fallback)
# ---------------------------------------------------------------------------

class NoneProvider(LLMProvider):
    """LLM을 호출하지 않는 더미 프로바이더.

    기존 규칙 기반 에이전트 로직이 그대로 동작하도록 None을 반환한다.
    """

    def __init__(self) -> None:
        super().__init__(model="none")

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        return None


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------

class OllamaProvider(LLMProvider):
    """Ollama REST API를 통한 LLM 호출.

    Ollama는 로컬에서 Qwen3 8B를 실행한다.
    - 엔드포인트: POST /api/chat
    - JSON 모드: format="json" 파라미터로 네이티브 지원
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        url = f"{self.base_url}/api/chat"
        return self._post(url, payload)

    def _post(self, url: str, payload: Dict[str, Any]) -> Optional[str]:
        """HTTP POST 요청을 보내고 응답을 반환한다 (일시적 오류 backoff 재시도)."""
        body = _http_post_json(
            url, payload, {"Content-Type": "application/json"}, self.timeout,
        )
        if body is None:
            return None
        return body.get("message", {}).get("content", "")


# ---------------------------------------------------------------------------
# VLLMProvider
# ---------------------------------------------------------------------------

class VLLMProvider(LLMProvider):
    """vLLM OpenAI-compatible API를 통한 LLM 호출.

    vLLM 서버는 OpenAI 호환 엔드포인트를 제공한다.
    - 엔드포인트: POST /v1/chat/completions
    - JSON 모드: response_format={"type": "json_object"}
    - enable_thinking=False: Qwen3/Qwen3.5 thinking 모드 비활성화 (보수적 응답 방지)
    - send_chat_template_kwargs: Qwen 계열 전용 필드 전송 여부. 실측 결과 Mistral 등
      일부 토크나이저는 이 필드 자체를 거부(HTTP 400 "chat_template is not supported
      for Mistral tokenizers.")하므로, 이질(heterogeneous) 모델 배선 시 False로 꺼야 한다.
      Qwen 기본 경로는 기존과 동일하게 True(하위호환, 회귀 없음).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_VLLM_URL,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        enable_thinking: bool = DEFAULT_ENABLE_THINKING,
        send_chat_template_kwargs: bool = True,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"  # vLLM default
        self.timeout = timeout
        # Qwen3/Qwen3.5 thinking 모드 설정.
        # False이면 chat_template_kwargs로 thinking 비활성화 — M4 보수적 응답 방지 + 14.99x speedup.
        self.enable_thinking = enable_thinking
        # 기본 True = 기존 Qwen 동작 완전 보존. Mistral 등 비-Qwen 이질 모델은 False로 생성할 것.
        self.send_chat_template_kwargs = send_chat_template_kwargs

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Optional[str]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.send_chat_template_kwargs:
            # Qwen3/Qwen3.5: chat_template_kwargs를 payload 최상위에 직접 포함
            # (OpenAI SDK extra_body와 달리 raw HTTP 요청은 top-level 필요).
            # 실측: Mistral 등 일부 토크나이저는 이 필드 자체로 HTTP 400을 반환하므로
            # "다른 모델은 무시" 가정은 성립하지 않는다 — send_chat_template_kwargs=False로 배제.
            # 참고: https://qwen.readthedocs.io/en/latest/inference/vllm.html
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/v1/chat/completions"
        return self._post(url, payload)

    def _post(self, url: str, payload: Dict[str, Any]) -> Optional[str]:
        """HTTP POST 요청을 보내고 응답을 반환한다.

        Qwen3/Qwen3.5 thinking 모드 처리:
        - thinking 활성 상태에서 max_tokens 부족 시 content=null, reasoning만 존재
        - content=null이면 None 반환 (토큰 초과 신호 — 호출자가 max_tokens 늘려야 함)
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        # 일시적 전송 오류는 backoff 재시도 (D4/F18)
        body = _http_post_json(url, payload, headers, self.timeout)
        if body is None:
            return None
        choices = body.get("choices", [])
        if not choices:
            return None
        msg = choices[0].get("message", {})
        content = msg.get("content")
        # Qwen3 thinking 모델 호환: content가 null/빈값이면 reasoning fallback.
        # PR #87 (M2) + PR #88 (P2) 통합: reasoning 있으면 사용, 없으면 명확한 경고.
        if not content:
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            if reasoning:
                logger.warning(
                    "vLLM content empty, falling back to reasoning (model=%s, finish=%s). "
                    "Consider enable_thinking=false to avoid this.",
                    self.model, choices[0].get("finish_reason"),
                )
                return reasoning
            logger.warning(
                "vLLM content=null AND no reasoning (token 초과 가능성). "
                "max_tokens 늘리거나 enable_thinking=False 권장. finish_reason=%s, usage=%s",
                choices[0].get("finish_reason"), body.get("usage"),
            )
            return None
        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_provider(
    config: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> LLMProvider:
    """설정에서 적절한 LLM 프로바이더를 생성한다.

    Config structure (pipeline_config.yaml):
        llm:
          provider: "ollama" | "vllm" | "none"
          model: "qwen3:8b"
          base_url: "http://localhost:11434"
          api_key: null
          timeout: 60
          temperature: 0.3
          max_tokens: 4096

          # M3 (2026-05-20): agent별 override — 키만 지정, 나머지는 상위 llm 값 상속
          # agents:
          #   planner: { provider: "ollama", model: "deepseek-r1:70b", ... }
          #   critic:  { provider: "vllm",   model: "qwen3.5-35b-a3b", ... }
          #   reporter:{ provider: "vllm",   model: "qwen3.5-35b-a3b", ... }

    Args:
        config: 파이프라인 설정 dict. None이면 NoneProvider 반환.
        model_override: 설정 파일의 모델명을 덮어쓸 모델명. None이면 config 값 사용.
        agent_name: agent 이름 (planner/critic/reporter 등). 지정 시
                    `llm.agents.<name>` 섹션이 상위 `llm` 값보다 우선.

    Returns:
        LLMProvider 인스턴스
    """
    if config is None:
        return NoneProvider()

    llm_cfg = config.get("llm", {})
    # M3: agent별 override 병합 — agent_name 지정 시 llm.agents.<name>이 상위 값 덮어씀
    if agent_name:
        agents_cfg = llm_cfg.get("agents", {}) or {}
        override = agents_cfg.get(agent_name)
        if isinstance(override, dict):
            llm_cfg = {**llm_cfg, **override}
            logger.info(
                "LLM agent override applied: agent=%s, provider=%s, model=%s",
                agent_name, llm_cfg.get("provider"), llm_cfg.get("model"),
            )
    provider_type = llm_cfg.get("provider", "none").lower()
    model = model_override or llm_cfg.get("model", DEFAULT_MODEL)
    timeout = llm_cfg.get("timeout", DEFAULT_TIMEOUT)

    if provider_type == "ollama":
        base_url = llm_cfg.get("base_url", DEFAULT_OLLAMA_URL)
        logger.info("LLM Provider: Ollama (%s @ %s)", model, base_url)
        return OllamaProvider(model=model, base_url=base_url, timeout=timeout)

    if provider_type == "vllm":
        base_url = llm_cfg.get("base_url", DEFAULT_VLLM_URL)
        api_key = llm_cfg.get("api_key")
        enable_thinking = llm_cfg.get("enable_thinking", DEFAULT_ENABLE_THINKING)
        # Mistral 등 chat_template_kwargs 미지원 모델용 — config로 끌 수 있게 노출
        # (기본 True: 기존 Qwen 동작 보존). 미지정 시 종전과 동일.
        send_chat_template_kwargs = llm_cfg.get("send_chat_template_kwargs", True)
        logger.info(
            "LLM Provider: vLLM (%s @ %s, enable_thinking=%s, send_chat_template_kwargs=%s)",
            model, base_url, enable_thinking, send_chat_template_kwargs,
        )
        return VLLMProvider(
            model=model, base_url=base_url, api_key=api_key, timeout=timeout,
            enable_thinking=enable_thinking,
            send_chat_template_kwargs=send_chat_template_kwargs,
        )

    logger.info("LLM Provider: None (rule-based mode)")
    return NoneProvider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_think_blocks(text: str) -> str:
    """DeepSeek-R1 등 reasoning 모델의 <think>...</think> 블록을 제거한다.

    DeepSeek-R1-Distill-Qwen-32B (vLLM port 8001)는 응답 content 내부에 인라인으로
    <think>...long reasoning...</think>\\n\\n<final answer> 형식 출력. JSON 파싱 전
    think 블록을 정규식으로 제거한다.

    - 정상 종료 (</think> 있음): 블록 통째로 제거
    - 미종료 (<think> 만 있고 </think> 없음, max_tokens 초과 시): <think> 이후 전체 제거
    """
    import re
    # <think>...</think> 통째로 제거 (re.DOTALL — 줄바꿈 포함)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 미종료 think 블록도 정리 (<think> 이후 모든 텍스트 — finish_reason=length 등)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    return cleaned.strip()


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """마크다운 코드 블록에서 JSON을 추출한다.

    DeepSeek-R1 등 reasoning 모델의 <think>...</think> 블록도 자동 제거.
    """
    import re
    # 1) <think> 블록 제거 후 시도
    text = _strip_think_blocks(text)
    # 2) markdown ```json ... ``` 블록
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # 3) 중괄호로 시작/끝 추출 시도
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start : brace_end + 1])
        except json.JSONDecodeError:
            pass
    return None
