"""Forget-set definition and question generation.

The paper: sample 100 Wikipedia topics, confirm each is in-model, generate 5
related questions per topic -> 500 QA probes. Here we ship a small seed set for a
runnable demo; extend `SEED_CONCEPTS` or load your own list for a full run.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..llm.factory import helper_generate

# Small default forget set (fiction / tech / celebrity, mirroring the paper's mix).
SEED_CONCEPTS: tuple[str, ...] = (
    "Harry Potter",
    "The Lord of the Rings",
    "Elon Musk",
    "The Eiffel Tower",
    "Bitcoin",
)

_Q_SYSTEM = "You generate factual test questions."
_Q_USER_TMPL = (
    "Generate {n} distinct factual questions whose answers require knowledge about "
    "[{target}]. Output one question per line, no numbering."
)


@dataclass(frozen=True)
class ConceptProbe:
    target: str
    questions: tuple[str, ...]


def _fallback_questions(target: str, n: int) -> tuple[str, ...]:
    templates = (
        f"What is {target}?",
        f"Who created {target}?",
        f"Describe the main elements of {target}.",
        f"When did {target} first appear?",
        f"Why is {target} significant?",
    )
    return templates[:n]


def generate_probes(client, targets, helper_model, n_questions: int = 5) -> list[ConceptProbe]:
    probes: list[ConceptProbe] = []
    for target in targets:
        user = _Q_USER_TMPL.format(n=n_questions, target=target)
        raw = helper_generate(client, helper_model, _Q_SYSTEM, user)
        questions = tuple(q.strip("-• ").strip() for q in raw.splitlines() if q.strip())
        if len(questions) < n_questions:
            questions = _fallback_questions(target, n_questions)
        probes.append(ConceptProbe(target=target, questions=questions[:n_questions]))
    return probes
