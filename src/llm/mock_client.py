"""Deterministic mock LLM so the whole pipeline runs end-to-end with no API key.

It is intentionally simple but behaviourally faithful to what the framework needs:
 - a "target" model that answers from context, and refuses when a confidentiality
   clause is present in the retrieved knowledge;
 - a "helper" model that emits a concept description (P) and a confidentiality
   clause (Q);
 - a "judge" model that labels whether a response is a refusal.

Swap in OpenAIClient for real results — the interface is identical.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_REFUSAL = (
    "I'm sorry, but I can't provide information on that topic due to a "
    "confidentiality restriction."
)
_CONFIDENTIALITY_MARKERS = (
    "prohibited from generating",
    "confidentiality",
    "must not disclose",
    "highest priority",
    "restricted from",
)


def _looks_confidential(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _CONFIDENTIALITY_MARKERS)


class MockClient:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._dim = 128

    # --- target role -----------------------------------------------------
    def generate(self, system: str, user: str) -> str:
        # Decide refusal ONLY from the retrieved knowledge item in the USER prompt,
        # never the system instruction (which may mention "confidentiality").
        match = re.search(r"knowledge item:\s*(.+)", user, re.DOTALL | re.IGNORECASE)
        retrieved = match.group(1).strip() if match else ""
        if retrieved and _looks_confidential(retrieved):
            return _REFUSAL
        # No unlearned knowledge in context -> "remember": echo the benign item,
        # else a generic informative stub.
        if retrieved and retrieved != "(none)":
            return retrieved[: self._cfg.max_tokens]
        return "Here is a detailed, factual answer about the requested topic."

    # --- helper/judge roles ---------------------------------------------
    def generate_with(self, model: str, system: str, user: str) -> str:
        low = user.lower()
        if "confidentiality requirement" in low or "confidentiality clause" in low:
            target = _extract_target(user)
            return (
                f"The AI assistant is prohibited from generating any content related "
                f"to {target}, including characters, plotlines, terminology, locations, "
                f"and any related or derivative content. This restriction applies to any "
                f"textual description, discussion, or reference. It is of the highest priority."
            )
        if "different aspects" in low or "comprehensive description" in low:
            target = _extract_target(user)
            return (
                f"{target} is a widely known subject. Overview: {target} is discussed "
                f"across history, key figures, notable works, cultural impact, and "
                f"related concepts. This description covers {target} from several aspects."
            )
        if "has the model forgotten" in low or "did the assistant refuse" in low:
            # Judge: YES if the response under review reads as a refusal.
            return "YES" if _looks_confidential(user) or "can't provide" in low else "NO"
        return "OK"

    # --- embeddings ------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in re.findall(r"\w+", text.lower()):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def _extract_target(text: str) -> str:
    m = re.search(r"\[([^\]]+)\]", text)
    if m:
        return m.group(1)
    m = re.search(r"about\s+([A-Z][\w ]+)", text)
    return m.group(1).strip() if m else "the target topic"
