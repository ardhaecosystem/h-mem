"""Answer synthesizer: generate sub-answers and final answer from evidence."""

from __future__ import annotations

from hmem.llm.adapter import LLMAdapter
from hmem.types import Evidence


class AnswerSynthesizer:
    """Generates answers from retrieved evidence using the LLM.

    Handles two levels:
    1. Sub-answer: answer a single sub-query from its evidence
    2. Final answer: combine all sub-answers into a coherent response
    """

    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm

    async def synthesize(self, sub_query: str, evidence: list[Evidence]) -> str:
        """Generate answer for a sub-query from evidence snippets."""
        if not evidence:
            return "No relevant evidence found."
        texts = [ev.text for ev in evidence]
        return await self.llm.synthesize(sub_query, texts)

    async def synthesize_final(
        self,
        original_query: str,
        sub_answers: dict[str, str],
        evidence: list[Evidence],
    ) -> str:
        """Combine all sub-answers into a final coherent answer."""
        if not sub_answers:
            return "I don't have enough information to answer that."
        if len(sub_answers) == 1:
            return list(sub_answers.values())[0]

        # Build context for final synthesis
        context_parts = []
        for sub_id, answer in sub_answers.items():
            context_parts.append(f"Sub-answer: {answer}")

        evidence_texts = [ev.text for ev in evidence[:20]]  # cap evidence
        evidence_block = "\n".join(f"- {t}" for t in evidence_texts)

        combined = "\n\n".join(context_parts)
        prompt = (
            f"The user asked: {original_query}\n\n"
            f"We obtained the following sub-answers:\n{combined}\n\n"
            f"Relevant evidence:\n{evidence_block}\n\n"
            "Synthesize these into a single, clear, accurate final answer. "
            "If there is conflict or uncertainty, say so."
        )
        return await self.llm.synthesize(original_query, [combined])
