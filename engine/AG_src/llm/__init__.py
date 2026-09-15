"""
AG_src.llm
==========
LLM Provider abstraction for the Co-Scientist pipeline.

Supports Ollama (local), vLLM (server), and "none" (rule-based fallback).
Default model: Qwen3 8B (native JSON output, 128K context).

Usage:
    from AG_src.llm import create_provider
    provider = create_provider(config)
    response = provider.generate(prompt, json_mode=True)
"""

from .provider import (
    LLMProvider,
    NoneProvider,
    OllamaProvider,
    VLLMProvider,
    create_provider,
)
from .prompts import (
    build_variant_generation_prompt,
    VARIANT_DESIGN_SYSTEM_PROMPT,
    get_system_prompt,
    format_planner_prompt,
)

__all__ = [
    "LLMProvider",
    "NoneProvider",
    "OllamaProvider",
    "VLLMProvider",
    "create_provider",
    # Prompt builders (M4 fix)
    "build_variant_generation_prompt",
    "VARIANT_DESIGN_SYSTEM_PROMPT",
    "get_system_prompt",
    "format_planner_prompt",
]
