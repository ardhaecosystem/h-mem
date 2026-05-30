"""Tree search: retrieve evidence from the temporal-semantic tree."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from hmem.config import HMemConfig
from hmem.core.tree import TemporalSemanticTree
from hmem.types import Evidence, TreeNode
from hmem.utils.embeddings import SentenceEmbedder


class TreeSearcher:
    """Search the temporal-semantic tree for evidence."""

    def __init__(self, tree: TemporalSemanticTree, embedder: SentenceEmbedder) -> None:
        self.tree = tree
        self.embedder = embedder

    async def search(
        self,
        sub_query: str,
        tree_level: int | None = None,
        time_filter: tuple[datetime, datetime] | None = None,
        top_k: int = 10,
    ) -> list[Evidence]:
        """Search tree nodes by semantic similarity at a specific level."""
        query_emb = await self.embedder.encode_async(sub_query)

        # If tree_level specified, search that level + one above for context
        candidates = []
        for level in [tree_level] if tree_level is not None else range(4):
            level_results = self.tree.search_level(
                level=level,
                time_filter=time_filter,
                semantic_query=query_emb,
                top_k=top_k,
            )
            for node, score in level_results:
                candidates.append((node, score, level))

        # Deduplicate and rerank
        seen = set()
        evidence = []
        for node, score, level in sorted(candidates, key=lambda x: x[1], reverse=True):
            if node.id in seen:
                continue
            seen.add(node.id)
            text = node.summary or node.text
            evidence.append(Evidence(
                text=text,
                source_type="tree",
                source_id=node.id,
                score=score,
                metadata={"level": level, "time_window": (node.time_window_start, node.time_window_end)},
            ))

        return evidence[:top_k]
