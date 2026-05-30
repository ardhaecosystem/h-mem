"""Evidence re-ranking by semantic relevance to sub-query."""

from __future__ import annotations

import numpy as np

from hmem.types import Evidence
from hmem.utils.embeddings import SentenceEmbedder


class Reranker:
    """Rerank evidence by cosine similarity to the sub-query text.

    Simple but effective: encode sub-query, encode each evidence text,
    rank by dot product.
    """

    def __init__(self, embedder: SentenceEmbedder) -> None:
        self.embedder = embedder

    async def rerank(self, sub_query: str, evidence: list[Evidence], top_k: int) -> list[Evidence]:
        """Rerank evidence and return the top_k."""
        if not evidence:
            return []
        query_emb = await self.embedder.encode_async(sub_query)

        scored = []
        for ev in evidence:
            ev_emb = await self.embedder.encode_async(ev.text)
            sim = self._cosine(query_emb, ev_emb)
            # Blend retrieval score with semantic similarity
            blended = 0.5 * ev.score + 0.5 * sim
            scored.append((blended, ev))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ev for _, ev in scored[:top_k]]

        # Update score on returned items
        for i, ev in enumerate(top):
            ev.score = scored[i][0]

        return top

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
