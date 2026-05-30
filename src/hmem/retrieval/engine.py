"""Full retrieval engine: orchestrates planner + evidence retrieval + synthesis."""

from __future__ import annotations

from hmem.config import HMemConfig
from hmem.core.graph import KnowledgeGraph
from hmem.core.tree import TemporalSemanticTree
from hmem.llm.adapter import LLMAdapter
from hmem.retrieval.graph_search import GraphSearcher
from hmem.retrieval.planner import RetrievalPlanner
from hmem.retrieval.reranker import Reranker
from hmem.retrieval.synthesizer import AnswerSynthesizer
from hmem.retrieval.tree_search import TreeSearcher
from hmem.types import Evidence, RetrievalResult, SubQuery
from hmem.utils.embeddings import SentenceEmbedder


class RetrievalEngine:
    """Orchestrates the full online retrieval pipeline.

    Usage:
        engine = RetrievalEngine(config, llm, tree, graph)
        result = await engine.query("What did Alice say about the project?")
    """

    def __init__(
        self,
        config: HMemConfig,
        llm: LLMAdapter,
        tree: TemporalSemanticTree,
        graph: KnowledgeGraph,
        embedder: SentenceEmbedder | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.tree = tree
        self.graph = graph
        self.embedder = embedder or SentenceEmbedder(config)

        # Sub-components
        self.planner = RetrievalPlanner(config, llm)
        self.tree_search = TreeSearcher(tree, self.embedder)
        self.graph_search = GraphSearcher(graph, config)
        self.reranker = Reranker(self.embedder)
        self.synthesizer = AnswerSynthesizer(llm)

    async def query(self, question: str) -> RetrievalResult:
        """Answer a question using the hybrid memory structure.

        Pipeline:
        1. Plan decomposition + workflows
        2. Execute retrieval per workflow
        3. Detect missing info, follow-up
        4. Synthesize sub-answers
        5. Return final answer
        """
        salient = self.graph.get_salient_entities()
        sub_queries = await self.planner.plan(question, salient)

        all_evidence: list[Evidence] = []
        sub_answers: dict[str, str] = {}

        round_count = 0
        for sq in sub_queries:
            if round_count >= self.config.retrieval_max_rounds:
                break

            # Execute initial retrieval
            ev = await self._retrieve(sq)
            round_count += 1

            # Check missing info + follow-up
            if sq.workflow and sq.workflow.missing_info_check and round_count < self.config.retrieval_max_rounds:
                follow_up = await self.planner.detect_missing_info(sq, [e.text for e in ev])
                if follow_up and follow_up.lower() not in {"none", "no", "", "no missing info"}:
                    follow_sq = SubQuery(text=follow_up, workflow=sq.workflow)
                    ev2 = await self._retrieve(follow_sq)
                    ev.extend(ev2)
                    round_count += 1

            # Generate sub-answer
            sub_answer = await self.synthesizer.synthesize(sq.text, ev)
            sub_answers[sq.id] = sub_answer
            all_evidence.extend(ev)

        # Final synthesis across all sub-answers
        final = await self.synthesizer.synthesize_final(question, sub_answers, all_evidence)

        return RetrievalResult(
            query=question,
            sub_queries=sub_queries,
            evidence=all_evidence,
            sub_answers=sub_answers,
            final_answer=final,
        )

    # ── Internal ───────────────────────────────

    async def _retrieve(self, sq: SubQuery) -> list[Evidence]:
        """Execute a retrieval workflow for a single sub-query."""
        wf = sq.workflow
        if wf is None:
            return []

        all_evidence = []

        # 1. Tree search
        if wf.search_tree:
            tree_ev = await self.tree_search.search(
                sub_query=sq.text,
                tree_level=wf.tree_level,
                time_filter=wf.time_filter,
                top_k=wf.top_k,
            )
            all_evidence.extend(tree_ev)

        # 2. Graph search
        if wf.search_graph and wf.graph_start_entities:
            graph_ev = await self.graph_search.search(
                start_entity_ids=wf.graph_start_entities,
                max_hops=wf.graph_hops,
                top_k=wf.top_k,
            )
            all_evidence.extend(graph_ev)

        # 3. Rerank
        if wf.rerank:
            all_evidence = await self.reranker.rerank(sq.text, all_evidence, wf.top_k)

        return all_evidence[: wf.top_k]
