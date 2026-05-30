"""Anthropic adapter for H-Mem."""

from __future__ import annotations

import os
from typing import Any

from anthropic import AsyncAnthropic

from hmem.config import HMemConfig
from hmem.llm.adapter import LLMAdapter, LLMResponse


class AnthropicAdapter(LLMAdapter):
    """Anthropic API backend."""

    def __init__(self, config: HMemConfig) -> None:
        super().__init__(config)
        api_key = config.llm_api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or config.llm_api_key"
            )
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = config.llm_model or "claude-3-haiku-20240307"

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        temp = (
            temperature
            if temperature is not None
            else (self.config.llm_temperature if self.config else 0.3)
        )
        max_tok = (
            max_tokens
            if max_tokens is not None
            else (self.config.llm_max_tokens if self.config else 1024)
        )

        system_msg = None
        user_messages = messages[:]
        if user_messages and user_messages[0].get("role") == "system":
            system_msg = user_messages[0].get("content", "")
            user_messages = user_messages[1:]

        response = await self._client.messages.create(
            model=self.model,
            system=system_msg,
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in user_messages
            ],
            max_tokens=max_tok,
            temperature=temp,
            timeout=self.config.llm_timeout if self.config else 30,
            **kwargs,
        )
        return LLMResponse(
            text=response.content[0].text if response.content else "",
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            model=response.model,
            finish_reason=response.stop_reason or "",
        )
