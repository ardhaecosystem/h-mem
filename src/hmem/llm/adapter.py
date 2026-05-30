"""Abstract LLM adapter for H-Mem.

All LLM interactions go through this interface.
Concrete adapters implement the concrete API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from hmem.config import HMemConfig


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    finish_reason: str = ""


class LLMAdapter(ABC):
    """Base class for all LLM backends.

    Implementations must be async-chat compatible.
    Sync wrappers are provided for convenience.
    """

    def __init__(self, config: HMemConfig) -> None:
        self.config = config
        self._cache: LLMCache | None = None
        if config.cache_dir:
            self._cache = LLMCache(config.cache_dir)

    # ── Low-level ─────────────────────────────────

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send messages to the LLM and return the response."""
        ...

    async def chat_sync(self, *args: Any, **kwargs: Any) -> LLMResponse:
        return await self.chat(*args, **kwargs)

    # ── Specific prompts (default f-string prompts) ─

    PROMPT_EXTRACT_ENTITIES = """\nExtract all entities and their relationships from the following text.

Return strictly valid JSON with this structure:
{
  "entities": [
    {"name": "Alice", "type": "person"},
    {"name": "Acme Corp", "type": "organization"}
  ],
  "relations": [
    {"source": "Alice", "target": "Acme Corp", "type": "works_for"}
  ]
}

Entity types: person, organization, location, product, event, concept, date, other.
Only include important entities with real information. No generics.

TEXT:
{text}
"""

    PROMPT_CONSOLIDATE = """\nMerge the following two memory fragments into a single coherent summary.
Preserve key facts and remove redundancy. Keep the summary concise.

FRAGMENT 1:
{text1}

FRAGMENT 2:
{text2}

SUMMARY:
"""

    PROMPT_DECOMPOSE = """\nDecompose the following user question into sub-questions.
Each sub-question should address one aspect of the main query.
Return strictly valid JSON:
{
  "sub_queries": [
    {"text": "sub question 1", "scope": "SHORT|LONG|MIXED"}
  ]
}

QUESTION:
{query}
"""

    PROMPT_SYNTHESIZE = """\nGiven the following pieces of evidence, answer the question.
If the evidence is insufficient, say "I don't have enough information."

EVIDENCE:
{evidence}

QUESTION:
{query}

ANSWER:
"""

    PROMPT_MISSING_INFO = """\nThe following question has been partially answered.
Generate a focused follow-up query to retrieve the missing information.

ORIGINAL QUESTION: {original_query}
PARTIAL ANSWER: {partial_answer}

MISSING INFO QUERY:
"""

    # ── High-level wrappers ───────────────────────

    async def extract_entities_relations(
        self,
        text: str,
    ) -> dict[str, list[dict[str, str]]]:
        """Extract named entities and relations from text.

        Returns {"entities": [...], "relations": [...]}
        """
        prompt = self.PROMPT_EXTRACT_ENTITIES.format(text=text)
        resp = await self.chat([{"role": "user", "content": prompt}])
        return self._parse_json(resp.text)

    async def consolidate(self, texts: list[str]) -> str:
        """Merge two texts into a single summary."""
        if len(texts) != 2:
            return texts[0] if len(texts) == 1 else "\n".join(texts)
        prompt = self.PROMPT_CONSOLIDATE.format(text1=texts[0], text2=texts[1])
        resp = await self.chat([{"role": "user", "content": prompt}])
        return resp.text.strip()

    async def decompose_query(self, query: str) -> list[dict[str, str]]:
        """Decompose a query into sub-queries with predicted scopes."""
        prompt = self.PROMPT_DECOMPOSE.format(query=query)
        resp = await self.chat([{"role": "user", "content": prompt}])
        parsed = self._parse_json(resp.text)
        return parsed.get("sub_queries", [])

    async def synthesize(self, query: str, evidence: list[str]) -> str:
        """Generate final answer from evidence."""
        evidence_block = "\n\n".join(f"- {e}" for e in evidence)
        prompt = self.PROMPT_SYNTHESIZE.format(evidence=evidence_block, query=query)
        resp = await self.chat([{"role": "user", "content": prompt}])
        return resp.text.strip()

    async def generate_missing_query(self, original: str, partial: str) -> str:
        """Generate a follow-up query when evidence is insufficient."""
        prompt = self.PROMPT_MISSING_INFO.format(
            original_query=original, partial_answer=partial
        )
        resp = await self.chat([{"role": "user", "content": prompt}])
        return resp.text.strip()

    # ── Helpers ─────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Robust JSON extraction from LLM text."""
        import json

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        import re
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Failing everything, return raw text in a wrapper
        return {"raw": text}

    def _cache_key(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Deterministic cache key for LLM calls."""
        payload = json.dumps({"messages": messages, **kwargs}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class LLMCache:
    """Simple disk cache for LLM responses."""

    def __init__(self, cache_dir: Path) -> None:
        self.dir = cache_dir / "llm_cache"
        self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> LLMResponse | None:
        path = self.dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return LLMResponse(**data)
        return None

    def set(self, key: str, response: LLMResponse) -> None:
        path = self.dir / f"{key}.json"
        path.write_text(response.model_dump_json())
