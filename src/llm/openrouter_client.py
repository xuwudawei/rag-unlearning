"""OpenRouter-backed LLM client — one key for every closed model the paper uses.

OpenRouter is OpenAI-API-compatible (base_url + model id), so a single key reaches
GPT-4o, GPT-4o-mini, GPT-4, and Gemini. This is how the reproduction runs the
closed-model targets AND the paper's GPT-4o auxiliary roles (judge, clause writer,
question generator) through one billed account. PaLM 2 is retired by Google and is
not available on any provider.

Model ids are OpenRouter-qualified, e.g. "openai/gpt-4o", "openai/gpt-4o-mini",
"openai/gpt-4", "google/gemini-2.5-flash".
"""
from __future__ import annotations

import time
from typing import Sequence

from .base import LLMError
from .embedders import build_embedder

_BASE_URL = "https://openrouter.ai/api/v1"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = 2.0
_TIMEOUT = 60.0


class OpenRouterClient:
    def __init__(self, cfg) -> None:
        if not cfg.api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set. Create a key at openrouter.ai/keys "
                "and export OPENROUTER_API_KEY before using the openrouter provider."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=cfg.api_key, base_url=_BASE_URL,
            timeout=_TIMEOUT, max_retries=0,
            default_headers={
                "HTTP-Referer": "https://github.com/xuwudawei/rag-unlearning",
                "X-Title": "RAG Unlearning Reproduction",
            },
        )
        self._cfg = cfg
        # Real semantic retrieval for the reproduction: set UNLEARN_EMBEDDER=st.
        self._embedder = build_embedder(cfg)

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
                if not resp.choices:
                    raise LLMError(f"OpenRouter returned no choices for '{model}'.")
                content = resp.choices[0].message.content
                if content is None:
                    raise LLMError(f"OpenRouter returned empty content for '{model}'.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        raise LLMError(f"OpenRouter chat call failed for '{model}' after retries: {last_err}")

    def generate(self, system: str, user: str) -> str:
        return self._chat(self._cfg.target_model, system, user)

    def generate_with(self, model: str, system: str, user: str) -> str:
        """Call a specific model id (e.g. the GPT-4o judge/clause-writer)."""
        return self._chat(model, system, user)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
