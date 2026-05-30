"""Main indexer: orchestrates tree + graph + embeddings for offline indexing."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from hmem.config import HMemConfig
from hmem.core.graph import KnowledgeGraph
from hmem.core.tree import TemporalSemanticTree
from hmem.indexing.graph_builder import GraphBuilder
from hmem.indexing.tree_builder import TreeBuilder
from hmem.llm.adapter import LLMAdapter
from hmem.types import MemoryFragment
from hmem.utils.embeddings import SentenceEmbedder


class Indexer:
    """Orchestrates the full offline indexing pipeline.

    Usage:
        indexer = Indexer(config, llm)
        indexer.index(fragment)
        indexer.index_batch(fragments)
        indexer.save("./index_data")
    """

    def __init__(
        self,
        config: HMemConfig,
        llm: LLMAdapter,
        embedder: SentenceEmbedder | None = None,
        tree_builder: TreeBuilder | None = None,
        graph_builder: GraphBuilder | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder or SentenceEmbedder(config)

        # Sub-builders
        self.tree_builder = tree_builder or TreeBuilder(
            config=config, llm=llm, embedder=self.embedder
        )
        self.graph_builder = graph_builder or GraphBuilder(
            config=config, llm=llm, embedder=self.embedder
        )

        self._fragments: list[MemoryFragment] = []
        self._indexed_count = 0

    # ── Public API ────────────────────────────

    async def index(self, fragment: MemoryFragment) -> None:
        """Index a single fragment into the hybrid structure."""
        self._fragments.append(fragment)

        # Parallel: tree + graph (both are independent writes)
        await asyncio.gather(
            self.tree_builder.add_fragment(fragment),
            self.graph_builder.add_fragment(fragment),
        )

        self._indexed_count += 1

        # Periodic consolidation on the tree
        if self.tree_builder.needs_consolidation():
            await self.tree_builder.consolidate()

    async def index_batch(
        self,
        fragments: Sequence[MemoryFragment],
        progress: bool = False,
    ) -> None:
        """Index a batch of fragments with optional progress bar."""
        iterator = fragments
        if progress:
            from tqdm import tqdm
            iterator = tqdm(list(fragments), desc="Indexing")

        for frag in iterator:
            await self.index(frag)

        # Final consolidation
        await self.tree_builder.consolidate()

    async def consolidate(self) -> None:
        """Force consolidation of the tree."""
        await self.tree_builder.consolidate()

    def get_tree(self) -> TemporalSemanticTree:
        return self.tree_builder.get_tree()

    def get_graph(self) -> KnowledgeGraph:
        return self.graph_builder.get_graph()

    def stats(self) -> dict[str, Any]:
        """Return current indexing stats."""
        tree = self.tree_builder.get_tree()
        graph = self.graph_builder.get_graph()
        return {
            "indexed_fragments": self._indexed_count,
            "tree_nodes": len(tree._nodes),
            "tree_leaves": len(tree._leaves),
            "graph_entities": len(graph._entities),
            "graph_relations": len(graph._relations),
            "salient_entities": len(graph.get_salient_entities()),
        }

    # ── Persistence ───────────────────────────

    def save(self, dir_path: str | Path) -> None:
        """Serialize the full index to disk."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Tree
        self.tree_builder.save(dir_path / "tree")

        # Graph
        self.graph_builder.save(str(dir_path / "graph.json"))

        # Stats
        import json
        (dir_path / "stats.json").write_text(
            json.dumps(self.stats(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    async def load_async(
        cls,
        dir_path: str | Path,
        config: HMemConfig,
        llm: LLMAdapter,
    ) -> "Indexer":
        """Restore an Indexer from disk."""
        dir_path = Path(dir_path)
        embedder = SentenceEmbedder(config)

        tree_builder = TreeBuilder.load(
            dir_path / "tree",
            config=config,
            llm=llm,
        )
        graph = GraphBuilder.load(str(dir_path / "graph.json"))
        graph_builder = GraphBuilder(
            config=config,
            llm=llm,
            embedder=embedder,
            graph=graph,
        )

        return cls(
            config=config,
            llm=llm,
            embedder=embedder,
            tree_builder=tree_builder,
            graph_builder=graph_builder,
        )

    # Backwards-compat alias
    load = load_async

    # ── Reset ─────────────────────────────────

    def reset(self) -> None:
        """Clear all indexed data."""
        self.tree_builder = TreeBuilder(
            config=self.config,
            llm=self.llm,
            embedder=self.embedder,
        )
        self.graph_builder = GraphBuilder(
            config=self.config,
            llm=self.llm,
            embedder=self.embedder,
        )
        self._fragments = []
        self._indexed_count = 0
