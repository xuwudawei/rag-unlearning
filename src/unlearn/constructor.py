"""Construct unlearned knowledge, following Algorithm 1 of the paper exactly.

  procedure MAIN(c):
      Q ← CRAFT_CONSTRAINT(LLM_cons, LLM_un, c)     # confidentiality clause
      P ← CRAFT_RETRIEVAL(LLM_un, c)                # M aspect-items (concept) or c (sample)
      K ← { P_i + Q  for i in 1..M }                # one entry per aspect
      return K

CRAFT_CONSTRAINT crafts Q with a helper LLM (LLM_cons, e.g. GPT-4o) and keeps it
only if the target model LLM_un, given Q as context, no longer produces content
related to the target. CRAFT_RETRIEVAL uses the *target* model itself to describe
the concept (improves retrieval ranking), or sets P = the sample for sample unlearning.

Heuristic, non-parametric: no gradients, no weight access — pure prompt construction
validated against the frozen target (Sec IV-B/C).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm.factory import helper_generate
from ..rag.knowledge import KnowledgeEntry

# --- Q: constraint component (LLM_cons) ---------------------------------
_Q_SYSTEM = "You write strict content-policy clauses."
_Q_USER_TMPL = (
    "This is my target concept/sample: [{target}].\n"
    "Please craft a confidentiality requirement that prevents an AI assistant from "
    "generating content about [{target}]. Limit the text to a maximum of {v} words."
)

# --- P: retrieval component (LLM_un, concept case) ----------------------
_P_SYSTEM = "You write neutral, factual, encyclopedic descriptions."
_P_USER_TMPL = (
    "This is my target concept: [{target}].\n"
    "Please provide a comprehensive description of [{target}] from {m} different "
    "aspects. Limit each aspect to {v} words."
)

# Probe used by CRAFT_CONSTRAINT to test whether Q induces off-topic/refusal.
_PROBE_SYSTEM = (
    "You are an intelligent assistant. Respond to the input based on the retrieved "
    "knowledge item, complying with any confidentiality requirement it states."
)


@dataclass(frozen=True)
class ConstructResult:
    entries: tuple[KnowledgeEntry, ...]   # M entries (concept) or 1 (sample)
    q: str
    refusal_verified: bool
    q_iterations: int


def _is_related_to(response: str, target: str) -> bool:
    """Cheap proxy for 'LLM_un(c,Q) is related to c'. A refusal/off-topic => False."""
    low = response.lower()
    refusal = ("i can't", "i cannot", "i'm sorry", "unable to", "not able to",
               "cannot provide", "can't provide", "confidential", "restricted",
               "prohibited", "do not know", "don't know")
    if any(s in low for s in refusal):
        return False
    # Related if it substantively mentions the target.
    return target.lower() in low


def _split_aspects(text: str, m: int) -> list[str]:
    """Split the M-aspect description into up to M knowledge items."""
    # Prefer numbered/bulleted lines; fall back to sentences, then the whole text.
    lines = [re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", ln).strip()
             for ln in text.splitlines() if ln.strip()]
    items = [ln for ln in lines if len(ln) > 15]
    if len(items) < 2:
        items = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
    if not items:
        items = [text.strip()]
    return items[:m] if len(items) >= m else items


class UnlearnedKnowledgeConstructor:
    def __init__(self, client, cfg) -> None:
        self._client = client
        self._llm = cfg.llm
        self._c = cfg.construct

    # CRAFT_CONSTRAINT(LLM_cons, LLM_un, c)
    def _craft_constraint(self, target: str) -> tuple[str, bool, int]:
        user = _Q_USER_TMPL.format(target=target, v=self._c.q_max_words)
        q = ""
        for i in range(1, self._c.q_max_refine_iters + 1):
            q = helper_generate(self._client, self._llm.helper_model, _Q_SYSTEM, user)
            probe = (f"Here is the original input: Tell me about {target}.\n"
                     f"Here is the knowledge item: {q}")
            resp = self._client.generate(_PROBE_SYSTEM, probe)
            if not _is_related_to(resp, target):
                return q, True, i          # LLM_un no longer related to c -> keep Q
        return q, False, self._c.q_max_refine_iters

    # CRAFT_RETRIEVAL(LLM_un, c)
    def _craft_retrieval(self, target: str, is_sample: bool) -> list[str]:
        if is_sample:
            return [target]                # P = c (sample has highest self-similarity)
        user = _P_USER_TMPL.format(
            target=target, m=self._c.p_num_aspects, v=self._c.q_max_words
        )
        # Uses the TARGET model itself (LLM_un) to describe c, per Algorithm 1.
        desc = self._client.generate(_P_SYSTEM, user)
        return _split_aspects(desc, self._c.p_num_aspects)

    def construct(self, target: str, entry_id_prefix: str,
                  is_sample: bool = False) -> ConstructResult:
        if not target or not target.strip():
            raise ValueError("Forget target must be a non-empty string.")

        q, verified, iters = self._craft_constraint(target)
        aspects = self._craft_retrieval(target, is_sample)

        # K = { P_i + Q }: one retrievable entry per aspect item.
        entries = tuple(
            KnowledgeEntry.unlearned(
                entry_id=f"{entry_id_prefix}-a{i}", target=target, p=p_i, q=q
            )
            for i, p_i in enumerate(aspects)
        )
        return ConstructResult(entries=entries, q=q, refusal_verified=verified,
                               q_iterations=iters)
