"""Min-K% membership inference (paper's residual-memorisation check).

Min-K% Prob (Shi et al.): score a text by the mean log-prob of its lowest-k% tokens.
Members (memorised) score higher than non-members. We threshold scores to report
TPR@1%FPR — closer to 1% means the target is indistinguishable from a non-member,
i.e. effectively forgotten.

Requires per-token log-probabilities from the backend. Closed chat APIs generally
do NOT expose logprobs for arbitrary *input* text, so this degrades gracefully:
if the client can't score tokens, `available` is False and callers should skip MIA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MiaResult:
    available: bool
    tpr_at_1pct_fpr: float | None
    note: str = ""


def min_k_score(token_logprobs: list[float], k_percent: float = 20.0) -> float:
    """Mean log-prob over the lowest-k% tokens. Higher => more likely a member."""
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty.")
    arr = np.sort(np.asarray(token_logprobs, dtype=float))
    k = max(1, int(len(arr) * k_percent / 100.0))
    return float(arr[:k].mean())


def tpr_at_fpr(member_scores: list[float], nonmember_scores: list[float],
               fpr_target: float = 0.01) -> float:
    """TPR when the threshold is set so non-members trigger at `fpr_target`."""
    if not member_scores or not nonmember_scores:
        raise ValueError("Both member and non-member scores are required.")
    nm = np.sort(np.asarray(nonmember_scores))[::-1]
    idx = min(len(nm) - 1, max(0, int(np.ceil(fpr_target * len(nm))) - 1))
    threshold = nm[idx]
    ms = np.asarray(member_scores)
    return float((ms >= threshold).mean() * 100.0)


def evaluate_mia(client, forget_texts, holdout_texts) -> MiaResult:
    """Run Min-K% MIA if the client exposes token scoring; else report unavailable."""
    scorer = getattr(client, "token_logprobs", None)
    if not callable(scorer):
        return MiaResult(
            available=False,
            tpr_at_1pct_fpr=None,
            note="Backend does not expose input-token logprobs; MIA skipped. "
                 "Use an open-weights backend (e.g. HF/Ollama with logits) for Min-K%.",
        )
    member = [min_k_score(scorer(t)) for t in forget_texts]
    nonmember = [min_k_score(scorer(t)) for t in holdout_texts]
    return MiaResult(available=True, tpr_at_1pct_fpr=tpr_at_fpr(member, nonmember))
