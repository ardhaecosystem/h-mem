"""H-Mem core engine: orchestrates indexing and retrieval."""

from __future__ import annotations

from hmem.config import HMemConfig
from hmem.core.graph import KnowledgeGraph
from hmem.core.tree import TemporalSemanticTree
from hmem.indexing.indexer import Indexer
from hmem.llm.adapter import LLMAdapter
from hmem.llm.openrouter import OpenRouterAdapter
from hmem.retrieval.engine import RetrievalEngine
from hmem.types import MemoryFragment, RetrievalResult
from hmem.utils.async_helpers import run_sync
from hmem.utils.embeddings import SentenceEmbedder


class HMemEngine:
    """Main H-Mem engine. Wraps the full indexing + retrieval pipeline."""

    def __init__(self, config: HMemConfig | None = None) -> None:
        self.config = config or HMemConfig()
        self.llm = self._init_llm()
        self.embedder = SentenceEmbedder(self.config)

        # Subsystems
        self.indexer = Indexer(
            config=self.config,
            llm=self.llm,
            embedder=self.embedder,
        )
        self.retrieval: RetrievalEngine | None = None  # wired after tree/graph ready

    # ── Indexing ────────────────────────────────

    def index(self, fragment: MemoryFragment) -> None:
        """Index a single memory fragment synchronously."""
        run_sync(self.indexer.index(fragment))

    def index_batch(self, fragments: list[MemoryFragment]) -> None:
        """Index a batch synchronously."""
        run_sync(self.indexer.index_batch(fragments))

    def consolidate(self) -> None:
        """Trigger tree consolidation."""
        run_sync(self.indexer.consolidate())

    # ── Query ───────────────────────────────────

    def query(self, question: str) -> RetrievalResult:
        """Answer a question synchronously."""
        if self.retrieval is None:
            self.retrieval = RetrievalEngine(
                config=self.config,
                llm=self.llm,
                tree=self.indexer.get_tree(),
                graph=self.indexer.get_graph(),
                embedder=self.embedder,
            )
        return run_sync(self.retrieval.query(question))

    # ── Properties ─────────────────────────────

    def get_tree(self) -> TemporalSemanticTree:
        return self.indexer.get_tree()

    def get_graph(self) -> KnowledgeGraph:
        return self.indexer.get_graph()

    @property
    def stats(self) -> dict:
        return self.indexer.stats()

    def reset(self) -> None:
        """Clear all indexed memory."""
        self.indexer.reset()
        self.retrieval = None

    # ── Persistence ───────────────────────────────

    def save(self, dir_path: str) -> None:
        self.indexer.save(dir_path)

    def load(self, dir_path: str) -> None:
        self.indexer = run_sync(Indexer.load_async(dir_path, self.config, self.llm))
        self.retrieval = None

    # ── Private ───────────────────────────────────

    def _init_llm(self) -> LLMAdapter:
        provider = self.config.llm_provider
        if provider == "openrouter":
            return OpenRouterAdapter(self.config)
        from hmem.llm.openai import OpenAIAdapter

        if provider == "openai":
            return OpenAIAdapter(self.config)
        from hmem.llm.anthropic import AnthropicAdapter

        if provider == "anthropic":
            return AnthropicAdapter(self.config)
        raise NotImplementedError(
            f"LLM provider '{provider}' not yet wired. Supported: openrouter, openai, anthropic"
        )
