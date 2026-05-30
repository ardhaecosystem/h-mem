"""Temporal-semantic tree with incremental consolidation.

Implements the tree structure from the H-Mem paper:
- Leaf nodes = raw memory fragments (short-term)
- Internal nodes = consolidated summaries (long-term)
- Consolidation merges adjacent, semantically-similar siblings upward
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Callable

import networkx as nx
import numpy as np

from hmem.types import MemoryFragment, TreeNode

NodeID = str
SimilarityFn = Callable[[np.ndarray, np.ndarray], float]


class TemporalSemanticTree:
    """Hierarchical memory tree.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity for two nodes to be consolidated.
    max_depth : int
        Max tree depth (root at max_depth-1, leaves at 0).
    time_window_max : timedelta
        Max time span allowed for a single node.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_depth: int = 4,
        time_window_max: timedelta = timedelta(days=1),
    ):
        self.similarity_threshold = similarity_threshold
        self.max_depth = max_depth
        self.time_window_max = time_window_max

        # Internal storage
        self._nodes: dict[NodeID, TreeNode] = {}
        self._children: dict[NodeID, list[NodeID]] = {}  # parent -> [child_id...]
        self._parent: dict[NodeID, NodeID] = {}
        self._embeddings: dict[NodeID, np.ndarray] = {}
        self._leaves: list[NodeID] = []
        self._roots: list[NodeID] = []  # trees may be a forest initially

    # ── Public ────────────────────────────────────

    def add_leaf(self, fragment: MemoryFragment, embedding: np.ndarray | None = None) -> str:
        """Add a memory fragment as a new leaf node.

        Returns the leaf node ID.
        """
        node_id = str(uuid.uuid4())
        node = TreeNode(
            id=node_id,
            level=0,
            time_window_start=fragment.timestamp,
            time_window_end=fragment.timestamp,
            text=fragment.text,
            source_fragments=[fragment.id],
        )
        self._nodes[node_id] = node
        self._children[node_id] = []
        if embedding is not None:
            self._embeddings[node_id] = embedding
        self._leaves.append(node_id)
        self._roots.append(node_id)  # initially each leaf is its own root
        return node_id

    def consolidate(self, llm_consolidate: Callable[[list[str]], str]) -> None:
        """Run a single bottom-up consolidation pass.

        Merges adjacent siblings whose semantic similarity >= threshold.
        """
        for level in range(self.max_depth):
            self._consolidate_level(level, llm_consolidate)

    def get_node(self, node_id: NodeID) -> TreeNode | None:
        return self._nodes.get(node_id)

    def get_embedding(self, node_id: NodeID) -> np.ndarray | None:
        return self._embeddings.get(node_id)

    def set_embedding(self, node_id: NodeID, embedding: np.ndarray) -> None:
        self._embeddings[node_id] = embedding

    def search_level(
        self,
        level: int,
        time_filter: tuple[datetime, datetime] | None = None,
        semantic_query: np.ndarray | None = None,
        top_k: int = 10,
    ) -> list[tuple[TreeNode, float]]:
        """Search nodes at a specific tree level.

        If semantic_query is provided, ranks by cosine similarity.
        If time_filter is provided, only returns nodes in range.
        """
        candidates: list[TreeNode] = [
            n for n in self._nodes.values()
            if n.level == level
            and (time_filter is None
                 or (n.time_window_start <= time_filter[1]
                     and n.time_window_end >= time_filter[0]))
        ]
        if semantic_query is None:
            return [(n, 0.0) for n in candidates]

        scored = []
        for node in candidates:
            emb = self._embeddings.get(node.id)
            if emb is None:
                continue
            score = self._cosine_similarity(semantic_query, emb)
            scored.append((node, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def subtree(self, root_id: NodeID) -> list[TreeNode]:
        """Return all nodes in the subtree rooted at root_id."""
        result: list[TreeNode] = [self._nodes[root_id]]
        stack = self._children.get(root_id, [])[:]
        while stack:
            child_id = stack.pop()
            result.append(self._nodes[child_id])
            stack.extend(self._children.get(child_id, []))
        return result

    def all_nodes(self) -> list[TreeNode]:
        return list(self._nodes.values())

    def all_leaves(self) -> list[TreeNode]:
        return [self._nodes[nid] for nid in self._leaves]

    # ── Internal ──────────────────────────────────

    def _consolidate_level(
        self,
        level: int,
        llm_consolidate: Callable[[list[str]], str],
    ) -> None:
        """Try to merge adjacent sibling pairs at this level."""
        parents_at_level = self._parents_at_level(level)
        for parent_id in parents_at_level:
            child_ids = self._children.get(parent_id, [])
            if len(child_ids) < 2:
                continue

            # Try merging adjacent children
            i = 0
            while i < len(child_ids) - 1:
                left = self._nodes[child_ids[i]]
                right = self._nodes[child_ids[i + 1]]

                if self._can_merge(left, right):
                    merged = self._merge_nodes(left, right, llm_consolidate)
                    # Replace left and right with merged under same parent
                    new_children = child_ids[:i] + [merged.id] + child_ids[i + 2:]
                    self._children[parent_id] = new_children
                    child_ids = new_children
                    # Promote merged to next level
                    merged.level = level + 1
                    if level + 1 >= self.max_depth:
                        # This becomes a root candidate
                        if parent_id in self._roots:
                            self._roots.remove(parent_id)
                        self._roots.append(merged.id)
                    # i stays same to check next pair
                else:
                    i += 1

    def _can_merge(self, left: TreeNode, right: TreeNode) -> bool:
        """Check if two adjacent nodes can be consolidated."""
        # Time adjacency
        time_gap = right.time_window_start - left.time_window_end
        if time_gap > self.time_window_max:
            return False

        # Time overlap is OK (adjacent means contiguous or overlapping)
        # Semantic similarity
        left_emb = self._embeddings.get(left.id)
        right_emb = self._embeddings.get(right.id)
        if left_emb is None or right_emb is None:
            return False
        sim = self._cosine_similarity(left_emb, right_emb)
        return sim >= self.similarity_threshold

    def _merge_nodes(
        self,
        left: TreeNode,
        right: TreeNode,
        llm_consolidate: Callable[[list[str]], str],
    ) -> TreeNode:
        """Merge two nodes into a parent node."""
        merged_text = llm_consolidate([left.text, right.text])
        summary = merged_text if len(merged_text) > 0 else ""
        merged = TreeNode(
            id=str(uuid.uuid4()),
            level=max(left.level, right.level) + 1,
            time_window_start=min(left.time_window_start, right.time_window_start),
            time_window_end=max(left.time_window_end, right.time_window_end),
            text=merged_text,
            summary=summary,
            children=[left.id, right.id],
            source_fragments=left.source_fragments + right.source_fragments,
        )
        self._nodes[merged.id] = merged
        self._children[merged.id] = [left.id, right.id]
        self._parent[left.id] = merged.id
        self._parent[right.id] = merged.id

        # Update parent's parent if needed
        for parent_id, child_ids in self._children.items():
            if left.id in child_ids and right.id in child_ids:
                # Already handled by consolidate_level
                pass

        # If right/left were roots, remove them
        if left.id in self._roots:
            self._roots.remove(left.id)
        if right.id in self._roots:
            self._roots.remove(right.id)

        return merged

    def _parents_at_level(self, level: int) -> list[NodeID]:
        """Return all node IDs that have children at the given level."""
        results = []
        for node_id, node in self._nodes.items():
            if node.level < level:  # could be parent
                has_child = any(
                    self._nodes.get(cid, TreeNode(id="", level=-1, time_window_start=datetime.utcnow(), time_window_end=datetime.utcnow(), text="")).level == level
                    for cid in self._children.get(node_id, [])
                )
                if has_child:
                    results.append(node_id)
        # For root nodes at this level: they are their own parent
        if not results:
            results = [nid for nid in self._roots if self._nodes.get(nid, TreeNode(id="", level=-1)).level == level]
        return results

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
