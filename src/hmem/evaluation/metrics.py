"""Evaluation metrics for benchmark QA.

Metrics: F1, Exact Match, LLM-as-Judge Accuracy.
"""

from __future__ import annotations

import re
from typing import Callable


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = " ".join(text.split())
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 overlap.

    Returns a score between 0.0 and 1.0.
    """
    pred_tokens = set(_normalize_text(prediction).split())
    gt_tokens = set(_normalize_text(ground_truth).split())

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact string match after normalization."""
    return _normalize_text(prediction) == _normalize_text(ground_truth)


def compute_llm_judge_accuracy(
    predictions: list[str],
    ground_truths: list[str],
    judge_fn: Callable[[str, str], bool],
) -> float:
    """LLM-as-a-judge accuracy.

    Args:
        predictions: Predicted answers.
        ground_truths: Ground truth answers.
        judge_fn: Function that returns True if prediction is correct.

    Returns:
        Fraction of correct answers (0.0–1.0).
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have same length")
    if not predictions:
        return 0.0

    correct = sum(
        1 for pred, gt in zip(predictions, ground_truths) if judge_fn(pred, gt)
    )
    return correct / len(predictions)


def llm_judge_prompt(question: str, prediction: str, ground_truth: str) -> str:
    """Return a prompt for LLM-as-judge evaluation."""
    return f"""You are an expert evaluator. Determine whether the predicted answer is correct based on the ground truth answer.

Question: {question}
Predicted Answer: {prediction}
Ground Truth Answer: {ground_truth}

Is the predicted answer correct? Answer with only "yes" or "no".
"""


__all__ = [
    "compute_f1",
    "compute_exact_match",
    "compute_llm_judge_accuracy",
    "llm_judge_prompt",
]
