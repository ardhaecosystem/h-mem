"""Evaluation harness: run H-Mem over benchmark datasets.

Usage:
    python -m hmem.evaluation.harness --dataset locomo --llm openrouter/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hmem.config import HMemConfig
from hmem.engine import HMem
from hmem.evaluation.datasets import get_loader
from hmem.evaluation.metrics import compute_f1, compute_llm_judge_accuracy
from hmem.llm.openrouter import OpenRouterAdapter


class EvalHarness:
    """Run evaluation for a benchmark dataset."""

    def __init__(self, config: HMemConfig, llm_adapter: Any | None = None) -> None:
        self.config = config
        self.llm = llm_adapter or OpenRouterAdapter(config)
        self.results: list[dict[str, Any]] = []

    async def evaluate_dataset(
        self,
        dataset_name: str,
        data_dir: str = "data",
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate H-Mem on a benchmark dataset.

        Returns aggregate metrics.
        """
        loader = get_loader(dataset_name, data_dir)
        conversations = loader.load_conversations()
        qa_items = loader.load_qa()

        if limit:
            conversations = conversations[:limit]
            conv_ids = {c.id for c in conversations}
            qa_items = [q for q in qa_items if q.conversation_id in conv_ids]

        print(f"[Eval] {dataset_name}: {len(conversations)} conversations, {len(qa_items)} QA pairs")

        # Build index for each conversation
        conv_results: list[dict[str, Any]] = []
        for conv in conversations:
            frags = loader.to_memory_fragments(conv)
            hmem = HMem(self.config, llm=self.llm)
            await hmem.index_async(frags)

            # Query each QA
            conv_qa = [q for q in qa_items if q.conversation_id == conv.id]
            for qa in conv_qa:
                predicted = await hmem.query(qa.question)
                f1 = compute_f1(predicted, qa.answer)
                conv_results.append({
                    "conversation_id": conv.id,
                    "question": qa.question,
                    "predicted": predicted,
                    "ground_truth": qa.answer,
                    "f1": f1,
                    "question_type": qa.question_type,
                })

        # Aggregate
        avg_f1 = sum(r["f1"] for r in conv_results) / len(conv_results) if conv_results else 0.0

        self.results = conv_results
        return {
            "dataset": dataset_name,
            "num_conversations": len(conversations),
            "num_qa": len(qa_items),
            "avg_f1": avg_f1,
            # LLM-Judge requires running LLM over every prediction; skip for now
            "llm_judge_accuracy": None,
        }

    def save_results(self, output_dir: str | Path) -> None:
        """Save raw predictions + aggregate metrics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pred_path = output_dir / f"predictions_{timestamp}.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        print(f"[Eval] Predictions saved: {pred_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("hmem.evaluation.harness")
    p.add_argument("--dataset", required=True, choices=["locomo", "longmemevals", "realtalk"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--llm", default="openrouter/gpt-4o-mini")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", default="results")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    config = HMemConfig(llm_model=args.llm)
    if "/" in args.llm:
        provider, model = args.llm.split("/", 1)
        config.llm_provider = provider
        config.llm_model = model

    harness = EvalHarness(config)
    metrics = await harness.evaluate_dataset(args.dataset, args.data_dir, args.limit)
    print("\n[Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    harness.save_results(args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
