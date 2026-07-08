"""Minimal Intervention Optimization (MIO).

Given the current surface and the causal drivers, find the cheapest *legitimate*
action that pushes the claim rate below a target. This is the paper's constrained
optimization, inverted: instead of optimizing the knowledge base to suppress, we
optimize the public surface to correct — using only legitimate actions.

Legitimate actions only:
  - publish a corrective document on an OWNED property           (cheap)
  - request a correction/removal on a low-authority third party  (expensive)
High-authority third-party edits are treated as infeasible here (out of client control).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..rag.pipeline import RagPipeline
from ..rag.retriever import HybridRetriever
from .corpus import Source, to_kb
from .probe import measure_claim_rate

# Action cost model (relative effort/feasibility).
COST_PUBLISH_OWNED = 1
COST_CORRECT_LOW_AUTHORITY = 5
COST_CORRECT_HIGH_AUTHORITY = 20  # effectively infeasible for most clients


@dataclass(frozen=True)
class Intervention:
    label: str
    cost: int
    predicted_claim_rate: float
    feasible: bool


def _rate(client, cfg_retriever, judge_model, entity, claim, sources, variants) -> float:
    pipe = RagPipeline(HybridRetriever(to_kb(sources, entity), client, cfg_retriever), client)
    return measure_claim_rate(pipe, client, judge_model, entity, claim, variants).claim_rate


def evaluate(client, cfg_retriever, judge_model: str, entity: str, claim: str,
             sources: list[Source], correction: Source, variants: list[str],
             target_rate: float = 0.2) -> list[Intervention]:
    """Try candidate legitimate interventions; predict each one's effect."""
    candidates: list[Intervention] = []

    # 1) Publish a corrective document on an owned property.
    rate_add = _rate(client, cfg_retriever, judge_model, entity, claim,
                     sources + [correction], variants)
    candidates.append(Intervention(
        f"Publish corrective statement on owned property ({correction.sid})",
        COST_PUBLISH_OWNED, round(rate_add, 4), feasible=True))

    # 2) Request correction/removal of each low-authority driver (owned=False, low auth).
    for s in sources:
        if s.owned or s.authority >= 0.5:
            continue
        reduced = [x for x in sources if x.sid != s.sid]
        rate_rm = _rate(client, cfg_retriever, judge_model, entity, claim, reduced, variants)
        candidates.append(Intervention(
            f"Request correction/removal of low-authority source ({s.sid})",
            COST_CORRECT_LOW_AUTHORITY, round(rate_rm, 4), feasible=True))

    return candidates


def recommend(interventions: list[Intervention], target_rate: float = 0.2):
    """Cheapest feasible action that meets the target; else the most effective one."""
    meeting = [i for i in interventions if i.feasible and i.predicted_claim_rate <= target_rate]
    if meeting:
        return min(meeting, key=lambda i: i.cost)
    return min(interventions, key=lambda i: (i.predicted_claim_rate, i.cost)) if interventions else None
