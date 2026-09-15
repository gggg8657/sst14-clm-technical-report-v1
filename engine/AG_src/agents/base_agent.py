"""
base_agent.py
SSTR2 펩타이드 바인더 Co-Scientist 에이전트 베이스 클래스
Base class for all agents in the SSTR2 peptide binder Co-Scientist pipeline.

모든 에이전트가 공유하는 공통 인터페이스와 메시지 통신 구조를 정의한다.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from ..llm.provider import LLMProvider


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class MessageType(str, Enum):
    """에이전트 간 메시지 유형."""
    INFO = "info"           # 단순 정보 전달
    REQUEST = "request"     # 작업 요청
    DECISION = "decision"   # 의사 결정 전달
    ALERT = "alert"         # 긴급 알림 / 오류


@dataclass
class AgentMessage:
    """에이전트 간 통신 메시지 단위.

    Attributes:
        sender: 발신 에이전트 이름
        receiver: 수신 에이전트 이름
        content: 메시지 본문 (자유 형식 dict)
        timestamp: 생성 시각 (ISO-8601)
        message_type: 메시지 종류 (info/request/decision/alert)
    """
    sender: str
    receiver: str
    content: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message_type: MessageType = MessageType.INFO


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """Co-Scientist 파이프라인의 모든 에이전트가 상속하는 베이스 클래스.

    Subclasses must implement :meth:`execute`.

    Attributes:
        name: 에이전트 고유 이름 (식별자로 사용)
        role: 역할 설명 (한국어/영문 혼용 허용)
        description: 에이전트가 수행하는 작업 상세 설명
        llm_provider: 연결할 LLM 제공자 식별자 ('claude', 'gpt-4o', 'none' 등)
        _inbox: 수신된 메시지 큐
        _logger: 에이전트 전용 logger 인스턴스
    """

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        llm_provider: Union[str, "LLMProvider"] = "claude",
    ) -> None:
        self.name: str = name
        self.role: str = role
        self.description: str = description
        # Accept both string (legacy) and LLMProvider instance
        self.llm_provider: Union[str, "LLMProvider"] = llm_provider
        self._inbox: list[AgentMessage] = []
        self._logger: logging.Logger = logging.getLogger(f"co_scientist.{name}")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(f"[%(asctime)s][{name}] %(levelname)s: %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """에이전트 주요 작업을 수행한다.

        Args:
            context: 파이프라인 공유 컨텍스트 딕셔너리.
                     run_id, iteration, previous_results 등을 포함.

        Returns:
            에이전트 실행 결과 딕셔너리. 최소 'status' 키를 포함해야 함.
        """
        ...

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """에이전트 이름 prefix와 함께 메시지를 로깅한다.

        Args:
            message: 로그 메시지 본문
            level: 로그 레벨 ('debug', 'info', 'warning', 'error')
        """
        log_fn = getattr(self._logger, level, self._logger.info)
        log_fn(message)

    # ------------------------------------------------------------------
    # Inter-agent communication
    # ------------------------------------------------------------------

    def send_message(
        self,
        target_agent: "BaseAgent",
        content: dict[str, Any],
        message_type: MessageType = MessageType.INFO,
    ) -> AgentMessage:
        """다른 에이전트에게 메시지를 전송한다.

        실제 비동기 버스가 없는 경우 target_agent의 receive_message()를
        직접 호출하는 동기 방식으로 동작한다.

        Args:
            target_agent: 수신 에이전트 인스턴스
            content: 전달할 내용 (자유 형식 dict)
            message_type: 메시지 종류

        Returns:
            생성된 AgentMessage 인스턴스
        """
        msg = AgentMessage(
            sender=self.name,
            receiver=target_agent.name,
            content=content,
            message_type=message_type,
        )
        self.log(f"-> {target_agent.name} [{message_type.value}]: {list(content.keys())}")
        target_agent.receive_message(from_agent=self, message=msg)
        return msg

    def receive_message(self, from_agent: "BaseAgent", message: AgentMessage) -> None:
        """수신된 메시지를 inbox에 저장하고 로깅한다.

        Args:
            from_agent: 발신 에이전트 인스턴스
            message: 수신된 AgentMessage
        """
        self._inbox.append(message)
        self.log(f"<- {from_agent.name} [{message.message_type.value}]: {list(message.content.keys())}")

    def get_messages(
        self,
        from_agent_name: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ) -> list[AgentMessage]:
        """inbox에서 필터링된 메시지 목록을 반환한다.

        Args:
            from_agent_name: 특정 발신자 이름으로 필터 (None = 전체)
            message_type: 특정 메시지 유형으로 필터 (None = 전체)

        Returns:
            조건에 맞는 AgentMessage 리스트
        """
        msgs = self._inbox
        if from_agent_name:
            msgs = [m for m in msgs if m.sender == from_agent_name]
        if message_type:
            msgs = [m for m in msgs if m.message_type == message_type]
        return msgs

    def clear_inbox(self) -> None:
        """수신 메시지 큐를 비운다."""
        self._inbox.clear()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    @property
    def has_llm(self) -> bool:
        """LLM 프로바이더가 실제로 연결되어 있는지 여부."""
        if isinstance(self.llm_provider, str):
            return self.llm_provider not in ("none", "", None)
        # LLMProvider instance — check if it's NoneProvider
        return getattr(self.llm_provider, "provider_name", "NoneProvider") != "NoneProvider"

    def llm_generate(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
    ) -> Optional[str]:
        """LLM 프로바이더를 통해 텍스트를 생성한다.

        LLM이 연결되어 있지 않으면 None을 반환하여 규칙 기반 폴백이 동작한다.
        """
        if isinstance(self.llm_provider, str):
            return None
        return self.llm_provider.generate(
            prompt, system_prompt=system_prompt, json_mode=json_mode,
        )

    def llm_generate_json(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """LLM 프로바이더를 통해 JSON dict를 생성한다.

        Args:
            temperature: 생성 온도. None이면 provider 기본값(DEFAULT_TEMPERATURE=0.3) 사용.
                         Planner 다양성 제어 등 에이전트 국소 override에 활용한다.
        """
        if isinstance(self.llm_provider, str):
            return None
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return self.llm_provider.generate_json(
            prompt, system_prompt=system_prompt, **kwargs
        )

    def __repr__(self) -> str:
        llm_label = (
            self.llm_provider if isinstance(self.llm_provider, str)
            else self.llm_provider.provider_name
        )
        return f"{self.__class__.__name__}(name={self.name!r}, llm={llm_label!r})"
