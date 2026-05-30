"""H-Mem core engine: orchestrates indexing and retrieval."""

from __future__ import annotations

from hmem.config import HMemConfig
from hmem.types import MemoryFragment, RetrievalResult


class HMemEngine:
    """Main H-Mem engine.  Wraps the full indexing + retrieval pipeline.

    This is a stub that will be fleshed out as we implement each module.
    """

    def __init__(self, config: HMemConfig) -> None:
        self.config = config
        # TODO: wire up Indexer, Retriever, LLM adapter, embedder, tree, graph
        self._indexer: object | None = None
        self._retriever: object | None = None
        self._llm: object | None = None
        self._cache: object | None = None

    def index(self, fragment: MemoryFragment) -> None:
        """Index a single memory fragment into the hybrid structure."""
        raise NotImplementedError("Indexing pipeline not yet implemented.")

    def index_batch(self, fragments: list[MemoryFragment]) -> None:
        """Index a batch of fragments."""
        for frag in fragments:
            self.index(frag)

    def query(self, question: str, **kwargs) -> RetrievalResult:
        """Answer a question using the hybrid memory structure."""
        raise NotImplementedError("Retrieval pipeline not yet implemented.")

    def consolidate(self) -> None:
        """Trigger tree consolidation."""
        raise NotImplementedError("Consolidation not yet implemented.")

    def reset(self) -> None:
        """Clear all indexed memory."""
        # TODO: wipe tree, graph, cache, vector store
        pass
