"""OpenAI-backed LLM client (GPT-4o family + embeddings)."""
from __future__ import annotations

import time
from typing import Sequence

from .base import LLMError

_MAX_RETRIES = 4
_BACKOFF_SECONDS = 2.0


class OpenAIClient:
    def __init__(self, cfg) -> None:
        if not cfg.api_key:
            raise LLMError(
                "OPENAI_API_KEY is not set. Export it or add it to a .env file "
                "before using the openai provider."
            )
        # Imported lazily so other providers don't require the OpenAI SDK.
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.api_key)
        self._cfg = cfg

    def _chat(self, model: str, system: str, user: str) -> str:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise LLMError("OpenAI returned empty content.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001 - surfaced as LLMError below
                last_err = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        raise LLMError(f"OpenAI chat call failed after retries: {last_err}")

    def generate(self, system: str, user: str) -> str:
        return self._chat(self._cfg.target_model, system, user)

    def generate_with(self, model: str, system: str, user: str) -> str:
        """Explicit-model variant for helper/judge roles."""
        return self._chat(model, system, user)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            resp = self._client.embeddings.create(
                model=self._cfg.embed_model, input=list(texts)
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"OpenAI embedding call failed: {exc}") from exc
        return [item.embedding for item in resp.data]
