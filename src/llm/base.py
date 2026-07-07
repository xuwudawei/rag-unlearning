"""LLM provider interface. All providers are interchangeable behind this protocol."""
from __future__ import annotations

from typing import Protocol, Sequence


class LLMError(RuntimeError):
    """Raised when an LLM call fails after the provider's own retries."""


class LLMClient(Protocol):
    """Minimal surface the framework depends on. Keeps the rest provider-agnostic."""

    def generate(self, system: str, user: str) -> str:
        """Return the model's text completion for a system+user prompt."""
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...
