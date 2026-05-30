"""Retrieval planner: decompose queries, predict scope, and generate workflows."""

from __future__ import annotations

import json

from hmem.config import HMemConfig
from hmem.llm.adapter import LLMAdapter
from hmem.types import Entity, RetrievalWorkflow, ScopeType, SubQuery


class RetrievalPlanner:
    """Plans the retrieval strategy for a given user query."""

    def __init__(self, config: HMemConfig, llm: LLMAdapter) -> None:
        self.config = config
        self.llm = llm

    async def plan(self, query: str, salient_entities: list[Entity]) -> list[SubQuery]:
        """Plan full retrieval for a query.

        Steps:
        1. Decompose query into sub-queries
        2. Predict scope for each
        3. Identify key entities
        4. Generate workflows
        """
        raw_sq = await self.llm.decompose_query(query)

        sub_queries: list[SubQuery] = []
        for item in raw_sq:
            sq = SubQuery(
                text=item.get("text", ""),
                predicted_scope=ScopeType(item.get("scope", "MIXED").upper()),
            )

            # Identify key entities from the sub-query
            for entity in salient_entities:
                if entity.name.lower() in sq.text.lower():
                    sq.key_entities.append(entity.id)

            sq.workflow = self._build_workflow(sq, salient_entities)
            sub_queries.append(sq)

        # Cap max sub-queries
        return sub_queries[: self.config.retrieval_max_subqueries]

    # ── Internal ──────────────────────────────

    def _build_workflow(
        self,
        sub_query: SubQuery,
        salient_entities: list[Entity],
    ) -> RetrievalWorkflow:
        """Build a concrete retrieval workflow for a sub-query."""
        workflow = RetrievalWorkflow(
            sub_query_id=sub_query.id,
            scope=sub_query.predicted_scope,
            search_tree=True,
            search_graph=bool(sub_query.key_entities),
            graph_start_entities=sub_query.key_entities,
            graph_hops=min(self.config.graph_max_hops, 3),
            top_k=self.config.retrieval_top_k,
            rerank=True,
            missing_info_check=True,
        )

        # Map scope to tree level
        if workflow.scope == ScopeType.SHORT:
            workflow.tree_level = 0  # leaf-level detail
        elif workflow.scope == ScopeType.LONG:
            workflow.tree_level = self.config.tree_max_depth - 1  # summary
        else:  # MIXED
            workflow.tree_level = 1  # intermediate

        # Temporal filtering is rare in initial pass; added during follow-up
        workflow.time_filter = None

        return workflow

    async def detect_missing_info(
        self,
        sub_query: SubQuery,
        evidence: list[str],
    ) -> str:
        """Detect if evidence is insufficient and generate a follow-up query."""
        partial_answer = "\n".join(evidence) if evidence else "No evidence found"
        return await self.llm.generate_missing_query(sub_query.text, partial_answer)
