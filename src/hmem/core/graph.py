"""Knowledge graph for entity-centric memory retrieval."""

from __future__ import annotations

import uuid
from typing import Any

import networkx as nx

from hmem.types import Entity, Relation

NodeID = str  # entity ID


class KnowledgeGraph:
    """Entity-relationship graph backed by NetworkX.

    Provides:
    - Entity merging (exact match + type, prefix/suffix overlap)
    - Multi-hop traversal
    - Salient entity profiles
    - Incremental updates
    """

    def __init__(self, entity_threshold: int = 2, overlap_threshold: float = 0.8) -> None:
        self.entity_threshold = entity_threshold
        self.overlap_threshold = overlap_threshold

        # NX graph: nodes = Entity, edges with relation data
        self._graph: nx.Graph = nx.Graph()
        self._entities: dict[str, Entity] = {}  # id -> Entity
        self._relations: dict[str, Relation] = {}  # id -> Relation

    # ── Entity Management ─────────────────────────

    def add_entity(self, entity: Entity) -> str:
        """Add or merge an entity.  Returns the canonical entity ID."""
        # Try exact match + type compatibility
        for existing in self._entities.values():
            if self._is_same_entity(entity, existing):
                self._merge_entities(existing, entity)
                return existing.id

        # No merge: insert new
        self._graph.add_node(entity.id, entity=entity)
        self._entities[entity.id] = entity
        return entity.id

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Entity | None:
        for e in self._entities.values():
            if e.name.lower() == name.lower():
                return e
        return None

    def get_salient_entities(self) -> list[Entity]:
        """Return entities with mention_count >= threshold."""
        return [e for e in self._entities.values() if e.mention_count >= self.entity_threshold]

    # ── Relation Management ───────────────────────

    def add_relation(self, relation: Relation) -> None:
        """Add or update a relation between entities."""
        # Check if relation already exists (same source, target, type)
        for existing in self._relations.values():
            if (existing.source == relation.source
                and existing.target == relation.target
                and existing.relation_type == relation.relation_type):
                existing.evidence.extend(relation.evidence)
                existing.confidence = max(existing.confidence, relation.confidence)
                return

        self._relations[relation.id] = relation
        self._graph.add_edge(
            relation.source, relation.target,
            relation=relation,
        )

    def get_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations involving this entity."""
        rels = []
        for _, _, data in self._graph.edges(entity_id, data=True):
            rels.append(data["relation"])
        return rels

    def add_overlap_edge(self, entity_a: str, entity_b: str) -> None:
        """Create a weak 'overlap' edge for traversal recall."""
        if self._graph.has_edge(entity_a, entity_b):
            return
        overlap_rel = Relation(
            id=str(uuid.uuid4()),
            source=entity_a,
            target=entity_b,
            relation_type="overlap",
            confidence=0.5,
        )
        self._graph.add_edge(entity_a, entity_b, relation=overlap_rel)

    # ── Traversal ─────────────────────────────────

    def multi_hop_traversal(
        self,
        start_entity_id: str,
        max_hops: int = 3,
        relation_types: set[str] | None = None,
    ) -> list[tuple[str, int]]:
        """Return (entity_id, hop_distance) reachable within max_hops.

        Filters by relation_types if provided.  'overlap' is always included.
        """
        visited = {start_entity_id: 0}
        queue = [(start_entity_id, 0)]
        results: list[tuple[str, int]] = []

        while queue:
            current, hops = queue.pop(0)
            if hops >= max_hops:
                continue

            for neighbor in self._graph.neighbors(current):
                edge_data = self._graph.get_edge_data(current, neighbor)
                if not edge_data:
                    continue
                rel = edge_data.get("relation")
                if rel is None:
                    continue
                if relation_types and rel.relation_type not in relation_types and rel.relation_type != "overlap":
                    continue

                if neighbor not in visited:
                    visited[neighbor] = hops + 1
                    queue.append((neighbor, hops + 1))
                    results.append((neighbor, hops + 1))

        return results

    def get_subgraph(self, entity_ids: list[str], max_hops: int = 1) -> nx.Graph:
        """Return the ego-network around the given entities."""
        nodes = set(entity_ids)
        for eid in entity_ids:
            reachable = self.multi_hop_traversal(eid, max_hops)
            nodes.update(rid for rid, _ in reachable)
        return self._graph.subgraph(nodes).copy()

    # ── Serialization helpers ─────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [e.model_dump(mode="json") for e in self._entities.values()],
            "relations": [r.model_dump(mode="json") for r in self._relations.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGraph:
        kg = cls()
        for e_data in data.get("entities", []):
            e = Entity(**e_data)
            kg.add_entity(e)
        for r_data in data.get("relations", []):
            r = Relation(**r_data)
            kg.add_relation(r)
        return kg

    # ── Internal helpers ──────────────────────────

    def _is_same_entity(self, a: Entity, b: Entity) -> bool:
        """Check if two entities refer to the same real-world entity."""
        if a.entity_type != b.entity_type:
            return False
        if a.name.lower() == b.name.lower():
            return True
        # Check overlap
        a_names = set(a.name.lower().split())
        b_names = set(b.name.lower().split())
        if not a_names or not b_names:
            return False
        overlap = len(a_names & b_names) / min(len(a_names), len(b_names))
        return overlap >= self.overlap_threshold and a.entity_type == b.entity_type

    def _merge_entities(self, existing: Entity, incoming: Entity) -> None:
        """Merge incoming into existing."""
        existing.mention_count += incoming.mention_count
        existing.last_seen = max(existing.last_seen, incoming.last_seen)
        existing.source_fragments.extend(incoming.source_fragments)
        for eid in incoming.source_fragments:
            if eid not in existing.source_fragments:
                existing.source_fragments.append(eid)
        # Update graph node
        self._graph.nodes[existing.id]["entity"] = existing
