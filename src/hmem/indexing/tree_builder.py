"""Tree builder: incremental construction + consolidation of the temporal-semantic tree.

Wraps the TemporalSemanticTree from core/ and adds:
- Lazy embedding computation
- Periodic consolidation triggers
- State serialization
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hmem.config import HMemConfig
from hmem.core.tree import TemporalSemanticTree
from hmem.llm.adapter import LLMAdapter
from hmem.types import MemoryFragment, TreeNode
from hmem.utils.embeddings import SentenceEmbedder


class TreeBuilder:
    """Manages building and maintaining the temporal-semantic tree."""

    def __init__(
        self,
        config: HMemConfig,
        llm: LLMAdapter,
        embedder: SentenceEmbedder,
        tree: TemporalSemanticTree | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder

        # Initialize or restore tree
        self._tree = tree or TemporalSemanticTree(
            similarity_threshold=config.tree_similarity_threshold,
            max_depth=config.tree_max_depth,
            time_window_max_seconds=config.tree_time_window_max_seconds,
        )

        # Track batch + fragments not yet consolidated
        self._leaf_fragments: list[str] = []          # fragment IDs since last consolidate
        self._leaf_nodes: dict[str, str] = {}          # fragment_id -> leaf_node_id (temp)

    # ── Public ────────────────────────────────

    async def add_fragment(self, fragment: MemoryFragment) -> str:
        """Add a single memory fragment to the tree.

        Returns the leaf node ID.
        """
        # Compute embedding first (async)
        embedding = await self.embedder.encode_async(fragment.text)

        # Add to tree
        leaf_id = self._tree.add_leaf(fragment, embedding=embedding)

        # Track
        self._leaf_fragments.append(fragment.id)
        self._leaf_nodes[fragment.id] = leaf_id

        # Update tree node with proper text
        node = self._tree.get_node(leaf_id)
        if node:
            node.text = fragment.text  # ensure text is preserved
            self._tree.set_embedding(leaf_id, embedding)

        return leaf_id

    def needs_consolidation(self) -> bool:
        """Check if enough new fragments have accumulated to trigger consolidation."""
        return len(self._leaf_fragments) >= self.config.tree_consolidation_batch_size

    async def consolidate(self) -> None:
        """Run consolidation over all pending leaf fragments."""
        if not self._leaf_fragments:
            return

        # Consolidate bottom-up
        self._tree.consolidate(llm_consolidate=self._llm_consolidate)

        # Recompute embeddings for newly created internal nodes
        for node_id in list(self._tree._nodes.keys()):
            node = self._tree._nodes.get(node_id)
            if node is None or (hasattr(node, 'is_leaf') and node.is_leaf):
                continue
            if self._tree.get_embedding(node_id) is not None:
                continue  # already has embedding
            self._tree.set_embedding(node_id, await self.embedder.encode_async(node.text))

        # Clear batch tracking
        self._leaf_fragments = []
        self._leaf_nodes = {}

    def get_tree(self) -> TemporalSemanticTree:
        return self._tree

    # ── Persistence ───────────────────────────

    def save(self, path: Path) -> None:
        """Serialize tree + embeddings to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save nodes as JSON
        nodes_data = []
        for node_id, node in self._tree._nodes.items():
            d = node.model_dump(mode="json")
            emb = self._tree.get_embedding(node_id)
            if emb is not None:
                d["embedding"] = emb.tolist()
            nodes_data.append(d)

        (path / "nodes.json").write_text(
            json.dumps(nodes_data, indent=2, default=str),
            encoding="utf-8",
        )

        # Save adjacency
        meta = {
            "children": self._tree._children,
            "parent": self._tree._parent,
            "leaves": self._tree._leaves,
            "roots": self._tree._roots,
        }
        (path / "tree_meta.json").write_text(
            json.dumps(meta, indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, config: HMemConfig, llm: LLMAdapter) -> TreeBuilder:
        """Restore a TreeBuilder from disk."""
        path = Path(path)
        nodes_data = json.loads((path / "nodes.json").read_text(encoding="utf-8"))
        meta = json.loads((path / "tree_meta.json").read_text(encoding="utf-8"))

        tree = TemporalSemanticTree(
            similarity_threshold=config.tree_similarity_threshold,
            max_depth=config.tree_max_depth,
        )

        # Restore nodes
        for d in nodes_data:
            if "embedding" in d:
                emb = np.array(d.pop("embedding"))
            else:
                emb = None
            node = TreeNode(**d)
            tree._nodes[node.id] = node
            if emb is not None:
                tree.set_embedding(node.id, emb)

        # Restore structure
        tree._children = meta["children"]
        tree._parent = meta["parent"]
        tree._leaves = meta["leaves"]
        tree._roots = meta["roots"]

        builder = cls(config=config, llm=llm, embedder=SentenceEmbedder(config), tree=tree)
        return builder

    # ── Internal ──────────────────────────────

    async def _llm_consolidate(self, texts: list[str]) -> str:
        """LLM-based consolidation of two memory texts."""
        if not texts or len(texts) == 0:
            return ""
        if len(texts) == 1:
            return texts[0]
        return await self.llm.consolidate(texts)
