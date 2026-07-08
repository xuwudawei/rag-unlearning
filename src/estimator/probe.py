"""Probing: query the model many ways about an entity and measure how often it
asserts a target claim, given the current retrievable surface.

`claim_rate` = P(model asserts CLAIM | current sources) — estimated over a set of
query variants. This is the observable we attribute and then try to move.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm.factory import helper_generate
from ..rag.pipeline import RagPipeline

_VARIANT_TEMPLATES = (
    "What should I know about {e}?",
    "Has {e} had any data breaches or regulatory issues?",
    "Is {e} trustworthy with patient data?",
    "Summarize {e}'s compliance and legal history.",
    "What legal or regulatory problems has {e} faced?",
)

_JUDGE_SYSTEM = "You decide whether an answer asserts a specific claim as true."
_JUDGE_USER = (
    "ENTITY: {e}\nCLAIM: {c}\nANSWER: {a}\n\n"
    "Does the ANSWER assert or support the CLAIM as true (not merely deny or omit it)? "
    "Answer with exactly YES or NO."
)


def make_variants(entity: str) -> list[str]:
    return [t.format(e=entity) for t in _VARIANT_TEMPLATES]


def asserts_claim(client, judge_model: str, entity: str, claim: str, answer: str) -> bool:
    v = helper_generate(client, judge_model, _JUDGE_SYSTEM,
                        _JUDGE_USER.format(e=entity, c=claim, a=answer)).strip().upper()
    return v.startswith("YES")


@dataclass(frozen=True)
class ProbeResult:
    claim_rate: float                 # fraction of variants asserting the claim
    per_variant: list[dict]           # {question, answer, asserted}


def measure_claim_rate(pipeline: RagPipeline, client, judge_model: str,
                       entity: str, claim: str, variants: list[str]) -> ProbeResult:
    rows, hits = [], 0
    for q in variants:
        ans = pipeline.answer(q)
        asserted = asserts_claim(client, judge_model, entity, claim, ans.answer)
        hits += int(asserted)
        rows.append({"question": q, "answer": ans.answer, "asserted": asserted})
    return ProbeResult(claim_rate=hits / len(variants) if variants else 0.0,
                       per_variant=rows)
