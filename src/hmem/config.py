"""H-Mem configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class HMemConfig:
    """Configuration for H-Mem instance.

    All parameters match or extend the hyperparameters reported in the paper.
    """

    # ── LLM ────────────────────────────────────────────────
    llm_provider: Literal["openrouter", "openai", "anthropic", "local"] = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str | None = None
    llm_base_url: str | None = None          # For local endpoints
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.3
    llm_timeout: float = 30.0

    # ── Embeddings ─────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_device: str = "cpu"

    # ── Tree ───────────────────────────────────────────────
    tree_similarity_threshold: float = 0.75   # For consolidation
    tree_max_depth: int = 4                   # Levels: 0=leaf, 3=root
    tree_consolidation_batch_size: int = 10   # Fragments before consolidation
    tree_time_window_max_seconds: float = 86400.0  # 1 day

    # ── Graph ─────────────────────────────────────────────
    graph_entity_threshold: int = 2             # Min mentions for salience
    graph_max_hops: int = 3
    graph_overlap_edge_threshold: float = 0.8   # For prefix/suffix matching

    # ── Retrieval ─────────────────────────────────────────
    retrieval_top_k: int = 10                  # Evidence per sub-query
    retrieval_rerank_top_k: int = 5             # After semantic reranking
    retrieval_max_subqueries: int = 5           # Max decomposition count
    retrieval_max_rounds: int = 2              # Including follow-up

    # ── Paths ─────────────────────────────────────────────
    cache_dir: Path = field(default_factory=lambda: Path(".cache/hmem"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_level: str = "INFO"

    # ── API Backends ────────────────────────────────────────
    openrouter_api_base: str = "https://openrouter.ai/api/v1"
    openai_api_base: str = "https://api.openai.com/v1"
    anthropic_api_base: str = "https://api.anthropic.com"

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.data_dir = Path(self.data_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
