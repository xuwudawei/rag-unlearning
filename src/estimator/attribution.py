"""Causal attribution: which sources drive the claim?

Leave-one-out (LOO) influence — the black-box causal method. Remove one source,
re-measure the claim rate; the drop is that source's causal contribution:

    influence(S) = claim_rate(full surface) - claim_rate(surface without S)

A large positive influence means S is a *driver* of the false claim. This is an
influence-function idea applied black-box to a grounded LLM via designed queries —
no model internals needed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..rag.pipeline import RagPipeline
from ..rag.retriever import HybridRetriever
from .corpus import Source, to_kb
from .probe import measure_claim_rate


@dataclass(frozen=True)
class SourceInfluence:
    sid: str
    kind: str
    authority: float
    owned: bool
    influence: float           # drop in claim_rate when this source is removed


def leave_one_out(client, cfg_retriever, judge_model: str, entity: str, claim: str,
                  sources: list[Source], variants: list[str],
                  baseline_rate: float) -> list[SourceInfluence]:
    out = []
    for s in sources:
        reduced = [x for x in sources if x.sid != s.sid]
        if not reduced:
            out.append(SourceInfluence(s.sid, s.kind, s.authority, s.owned, baseline_rate))
            continue
        pipe = RagPipeline(HybridRetriever(to_kb(reduced, entity), client, cfg_retriever), client)
        r = measure_claim_rate(pipe, client, judge_model, entity, claim, variants)
        out.append(SourceInfluence(
            sid=s.sid, kind=s.kind, authority=s.authority, owned=s.owned,
            influence=round(baseline_rate - r.claim_rate, 4)))
    return sorted(out, key=lambda x: x.influence, reverse=True)


def with_topk(cfg_retriever, k: int):
    """Return a retriever config that pulls k sources into context."""
    return dataclasses.replace(cfg_retriever, top_k=k)
