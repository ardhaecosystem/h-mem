"""Evaluation harness: run H-Mem over benchmark datasets.

Usage:
    python -m hmem.evaluation.harness --dataset locomo

Metrics reported: F1 (token overlap) and optional LLM-as-Judge accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hmem import HMem
from hmem.config import HMemConfig
from hmem.evaluation.datasets import get_loader
from hmem.evaluation.metrics import (
    compute_exact_match,
    compute_f1,
    compute_llm_judge_accuracy,
    llm_judge_prompt,
)
from hmem.llm.openrouter import OpenRouterAdapter
from hmem.utils.async_helpers import run_sync
from hmem.types import MemoryFragment


class EvalHarness:
    """Run evaluation for a benchmark dataset."""

    def __init__(self, config: HMemConfig, judge: Any | None = None) -> None:
        self.config = config
        self.engine = HMem(config)
        self.judge = judge  # optional LLM for judge
        self.results: list[dict[str, Any]] = []

    def evaluate_dataset(
        self,
        dataset_name: str,
        data_dir: str = "data",
        limit: int | None = None,
        use_llm_judge: bool = False,
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
        if not qa_items:
            return {"dataset": dataset_name, "error": "No QA items found"}

        # Build index for each conversation
        conv_results: list[dict[str, Any]] = []
        for conv in conversations:
            frags = loader.to_memory_fragments(conv)
            self.engine.index_batch(frags)

            # Query each QA for this conversation
            conv_qa = [q for q in qa_items if q.conversation_id == conv.id]
            for qa in conv_qa:
                predicted = self.engine.query(qa.question)
                pred_text = predicted.final_answer if hasattr(predicted, 'final_answer') else str(predicted)
                f1 = compute_f1(pred_text, qa.answer)
                exact = compute_exact_match(pred_text, qa.answer)
                conv_results.append({
                    "conversation_id": conv.id,
                    "question": qa.question,
                    "predicted": pred_text,
                    "ground_truth": qa.answer,
                    "f1": f1,
                    "exact_match": exact,
                    "question_type": qa.question_type,
                })

            # Reset engine between conversations (optional)
            self.engine.reset()

        # Aggregate
        f1_scores = [r["f1"] for r in conv_results]
        em_scores = [r["exact_match"] for r in conv_results]
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0.0

        # Optional LLM-as-Judge
        judge_accuracy = None
        if use_llm_judge and self.judge:
            predictions = [r["predicted"] for r in conv_results]
            ground_truths = [r["ground_truth"] for r in conv_results]
            questions = [r["question"] for r in conv_results]

            def judge_fn(pred: str, gt: str) -> bool:
                qi = next(i for i, r in enumerate(conv_results) if r["predicted"] == pred)
                prompt = llm_judge_prompt(questions[qi], pred, gt)
                resp = run_sync(self.judge.generate(prompt))
                return resp.strip().lower().startswith("yes")

            judge_accuracy = compute_llm_judge_accuracy(predictions, ground_truths, judge_fn)

        self.results = conv_results
        return {
            "dataset": dataset_name,
            "num_conversations": len(conversations),
            "num_qa": len(qa_items),
            "avg_f1": round(avg_f1, 4),
            "avg_exact_match": round(avg_em, 4),
            "llm_judge_accuracy": round(judge_accuracy, 4) if judge_accuracy is not None else None,
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
    p.add_argument("--use-llm-judge", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = HMemConfig(llm_model=args.llm)
    if "/" in args.llm:
        provider, model = args.llm.split("/", 1)
        config.llm_provider = provider
        config.llm_model = model

    judge = None
    if args.use_llm_judge:
        judge = OpenRouterAdapter(config)

    harness = EvalHarness(config, judge=judge)
    metrics = harness.evaluate_dataset(
        args.dataset, args.data_dir, args.limit, args.use_llm_judge
    )
    print("\n[Results]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    harness.save_results(args.output_dir)


if __name__ == "__main__":
    main()
