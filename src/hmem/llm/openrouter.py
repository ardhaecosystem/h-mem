"""OpenRouter adapter for H-Mem.

Uses httpx for async HTTP.
Defaults to gpt-4o-mini via OpenRouter (cheap, fast, good for benchmarks).
"""

from __future__ import annotations

from typing import Any

import httpx
import os
from hmem.config import HMemConfig
from hmem.llm.adapter import LLMAdapter, LLMResponse


class OpenRouterAdapter(LLMAdapter):
    """OpenRouter LLM backend."""

    def __init__(self, config: HMemConfig) -> None:
        super().__init__(config)
        self.api_base = config.openrouter_api_base
        self.api_key = config.llm_api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = config.llm_model or "openai/gpt-4o-mini"
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY or config.llm_api_key")

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else (self.config.llm_temperature if self.config else 0.3)
        max_tok = max_tokens if max_tokens is not None else (self.config.llm_max_tokens if self.config else 1024)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ardhaecosystem/h-mem",
            "X-Title": "H-Mem",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            **kwargs,
        }

        async with httpx.AsyncClient(timeout=self.config.llm_timeout if self.config else 30) as client:
            response = await client.post(f"{self.api_base}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})
        return LLMResponse(
            text=choice["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
        )
