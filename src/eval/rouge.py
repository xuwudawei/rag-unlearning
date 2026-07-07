"""ROUGE-L recall: overlap between the unlearned answer and the original answer.

Lower recall = the unlearned output diverges more from what the model used to say
= more effective forgetting (paper's Effectiveness criterion).
"""
from __future__ import annotations

from rouge_score import rouge_scorer

_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_recall(original_answer: str, unlearned_answer: str) -> float:
    if not original_answer.strip():
        return 0.0
    score = _scorer.score(original_answer, unlearned_answer)["rougeL"]
    return float(score.recall)


def mean_rouge_l_recall(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(rouge_l_recall(o, u) for o, u in pairs) / len(pairs)
