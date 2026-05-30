"""Graph search: multi-hop traversal for entity-centric evidence."""

from __future__ import annotations

from hmem.config import HMemConfig
from hmem.core.graph import KnowledgeGraph
from hmem.types import Evidence


class GraphSearcher:
    """Search the knowledge graph via multi-hop traversal."""

    def __init__(self, graph: KnowledgeGraph, config: HMemConfig) -> None:
        self.graph = graph
        self.config = config

    async def search(
        self,
        start_entity_ids: list[str],
        max_hops: int | None = None,
        top_k: int = 10,
    ) -> list[Evidence]:
        """Traverse the graph from starting entities and collect evidence.

        Returns evidence items from traversed entities and their relations.
        """
        hops = max_hops or self.config.graph_max_hops
        evidence = []
        seen_entities = set()

        for start_id in start_entity_ids:
            if start_id in seen_entities:
                continue
            reachable = self.graph.multi_hop_traversal(
                start_entity_id=start_id,
                max_hops=hops,
            )
            for entity_id, distance in reachable:
                seen_entities.add(entity_id)
                entity = self.graph.get_entity(entity_id)
                if entity is None:
                    continue
                # Entity text evidence
                text = f"{entity.name} ({entity.entity_type}): {entity.profile}" if entity.profile else entity.name
                evidence.append(Evidence(
                    text=text,
                    source_type="graph",
                    source_id=entity_id,
                    score=1.0 / (distance + 1),  # closer is better
                    metadata={"hop_distance": distance, "type": entity.entity_type},
                ))

                # Relation evidence
                for rel in self.graph.get_relations(entity_id):
                    other_id = rel.target if rel.source == entity_id else rel.source
                    other = self.graph.get_entity(other_id)
                    other_name = other.name if other else "unknown"
                    rel_text = f"{entity.name} {rel.relation_type} {other_name}"
                    evidence.append(Evidence(
                        text=rel_text,
                        source_type="graph",
                        source_id=rel.id,
                        score=1.0 / (distance + 1) * rel.confidence,
                        metadata={
                            "hop_distance": distance,
                            "relation_type": rel.relation_type,
                            "relation_id": rel.id,
                        },
                    ))

        # Deduplicate by source_id
        seen = set()
        deduped = []
        for ev in sorted(evidence, key=lambda e: e.score, reverse=True):
            if ev.source_id in seen:
                continue
            seen.add(ev.source_id)
            deduped.append(ev)

        return deduped[:top_k]
