"""Smoke test: end-to-end indexing and query without real LLM calls."""

import asyncio
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hmem.core.graph import KnowledgeGraph
from hmem.core.tree import TemporalSemanticTree
from hmem.types import (
    Entity, EntityType, MemoryFragment,
    Relation, ScopeType, SourceType,
)
from hmem.retrieval.tree_search import TreeSearcher
from hmem.retrieval.graph_search import GraphSearcher
from hmem.utils.embeddings import DummyEmbedder


def make_fragment(text: str, day_offset: int) -> MemoryFragment:
    ts = datetime(2024, 1, 1) + timedelta(days=day_offset)
    return MemoryFragment(
        id=f"frag-{day_offset}",
        text=text,
        timestamp=ts,
        source_type=SourceType.CONVERSATION,
        scope=ScopeType.MIXED,
        raw_data={"text": text},
    )


def build_dummy_tree(frags: list[MemoryFragment]) -> TemporalSemanticTree:
    tree = TemporalSemanticTree(similarity_threshold=0.5)
    embedder = DummyEmbedder(8)
    for frag in frags:
        tree.add_leaf(frag, embedding=embedder.embed(frag.text))
    return tree


def build_dummy_graph(frags: list[MemoryFragment]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for i, frag in enumerate(frags):
        entity = Entity(
            id=f"concept-{i}",
            name=f"Concept {i}",
            entity_type=EntityType.CONCEPT,
            profile=frag.text,
            first_seen=frag.timestamp,
            last_seen=frag.timestamp,
            mention_count=1,
            source_fragments=[frag.id],
        )
        graph.add_entity(entity)
        if i > 0:
            rel = Relation(
                id=f"rel-{i}",
                source=f"concept-{i-1}",
                target=f"concept-{i}",
                relation_type="followed_by",
                confidence=1.0,
                timestamp=frag.timestamp,
            )
            graph.add_relation(rel)
    return graph


class _Cfg:
    graph_max_hops: int = 2
    retrieval_rerank_top_k: int = 5


def test_tree_smoke():
    frags = [
        make_fragment("Alice met Bob at the conference.", 0),
        make_fragment("They discussed machine learning projects.", 1),
        make_fragment("Alice scheduled a follow-up call next week.", 2),
    ]
    tree = build_dummy_tree(frags)
    embedder = DummyEmbedder(dim=8)
    searcher = TreeSearcher(tree, embedder)

    results = asyncio.get_event_loop().run_until_complete(
        searcher.search("machine learning", top_k=2)
    )
    assert len(results) > 0, "Tree search should return evidence"
    print("Tree smoke OK")


def test_graph_smoke():
    frags = [
        make_fragment("Alice met Bob at the conference.", 0),
        make_fragment("They discussed machine learning projects.", 1),
        make_fragment("Alice scheduled a follow-up call next week.", 2),
    ]
    graph = build_dummy_graph(frags)
    searcher = GraphSearcher(graph, _Cfg())

    results = asyncio.get_event_loop().run_until_complete(
        searcher.search(["concept-0"], max_hops=2, top_k=5)
    )
    assert len(results) > 0, "Graph search should return evidence"
    print("Graph smoke OK")


def test_end_to_end():
    frags = [
        make_fragment("Alice met Bob at the conference in January.", 0),
        make_fragment("They discussed machine learning projects over lunch.", 1),
        make_fragment("Alice scheduled a follow-up call next week with Bob.", 2),
    ]
    tree = build_dummy_tree(frags)
    graph = build_dummy_graph(frags)
    embedder = DummyEmbedder(dim=8)

    tree_searcher = TreeSearcher(tree, embedder)
    graph_searcher = GraphSearcher(graph, _Cfg())

    # Query for machine learning in tree
    tree_ev = asyncio.get_event_loop().run_until_complete(
        tree_searcher.search("machine learning projects", top_k=3)
    )
    assert any("machine learning" in e.text.lower() for e in tree_ev), "Tree should find ML evidence"

    # Query for Alice in graph
    graph_ev = asyncio.get_event_loop().run_until_complete(
        graph_searcher.search(["concept-0"], max_hops=2, top_k=3)
    )
    assert len(graph_ev) > 0, "Graph should return evidence"

    print("End-to-end smoke passed.")


if __name__ == "__main__":
    test_tree_smoke()
    test_graph_smoke()
    test_end_to_end()
