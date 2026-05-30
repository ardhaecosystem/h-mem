"""Graph builder: extract entities/relations from fragments and update the knowledge graph."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from hmem.config import HMemConfig
from hmem.core.graph import KnowledgeGraph
from hmem.llm.adapter import LLMAdapter
from hmem.types import Entity, EntityType, MemoryFragment, Relation
from hmem.utils.embeddings import SentenceEmbedder


class GraphBuilder:
    """Manages extracting entities and relations from memory fragments
    and updating the incremental knowledge graph.
    """

    def __init__(
        self,
        config: HMemConfig,
        llm: LLMAdapter,
        embedder: SentenceEmbedder,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self._graph = graph or KnowledgeGraph(
            entity_threshold=config.graph_entity_threshold,
            overlap_threshold=config.graph_overlap_edge_threshold,
        )

    # ── Public ────────────────────────────────

    async def add_fragment(self, fragment: MemoryFragment) -> None:
        """Extract entities/relations from a fragment and update the graph."""
        # Extract via LLM
        raw = await self.llm.extract_entities_relations(fragment.text)

        for e_data in raw.get("entities", []):
            entity = self._make_entity(e_data, fragment.id)
            canonical_id = self._graph.add_entity(entity)

        for r_data in raw.get("relations", []):
            relation = self._make_relation(r_data)
            self._graph.add_relation(relation)

    def get_graph(self) -> KnowledgeGraph:
        return self._graph

    def set_graph(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    # ── Serialization ─────────────────────────

    def save(self, path: str) -> None:
        """Serialize graph to JSON."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self._graph.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> KnowledgeGraph:
        import json
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return KnowledgeGraph.from_dict(data)

    # ── Internal helpers ──────────────────────

    @staticmethod
    def _make_entity(data: dict[str, Any], fragment_id: str) -> Entity:
        """Convert raw LLM entity dict into typed Entity."""
        raw_type = (data.get("type") or "other").lower()
        try:
            entity_type = EntityType(raw_type)
        except ValueError:
            entity_type = EntityType.OTHER

        return Entity(
            id="",  # will be assigned by KnowledgeGraph.add_entity
            name=data.get("name", ""),
            entity_type=entity_type,
            mention_count=1,
            source_fragments=[fragment_id],
        )

    def _make_relation(self, data: dict[str, Any]) -> Relation:
        """Convert raw LLM relation dict into typed Relation."""
        return Relation(
            id="",
            source=data.get("source", ""),
            target=data.get("target", ""),
            relation_type=data.get("type", ""),
            evidence=[],
            confidence=1.0,
        )
