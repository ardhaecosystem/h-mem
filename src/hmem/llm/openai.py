"""OpenAI-compatible adapter (works for OpenAI, Azure, local endpoints)."""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

from hmem.config import HMemConfig
from hmem.llm.adapter import LLMAdapter, LLMResponse


class OpenAIAdapter(LLMAdapter):
    """OpenAI API backend."""

    def __init__(self, config: HMemConfig) -> None:
        super().__init__(config)
        api_key = config.llm_api_key or os.getenv("OPENAI_API_KEY")
        base_url = config.llm_base_url or config.openai_api_base
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY or config.llm_api_key"
            )
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = config.llm_model or "gpt-4o-mini"

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

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
            timeout=self.config.llm_timeout if self.config else 30,
            **kwargs,
        )
        choice = response.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            model=response.model,
            finish_reason=choice.finish_reason or "",
        )
