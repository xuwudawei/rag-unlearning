"""DeepSeek-backed LLM client.

DeepSeek's chat API is OpenAI-compatible (just a different base_url + model id), so
it drives generation for every LLM role (target LLM_un, helper LLM_cons, judge).
DeepSeek has NO embeddings endpoint, so the retriever's semantic half uses a
pluggable embedder (default: the torch-free numpy HashingEmbedder) — no second API
and no torch, which is what lets this provider deploy to serverless.

Min-K% MIA needs per-token logprobs on arbitrary *input* text, which no chat API
exposes; MIA is therefore unavailable on this backend (use the local `hf` backend).
"""
from __future__ import annotations

import time
from typing import Sequence

from .base import LLMError
from .embedders import build_embedder

_BASE_URL = "https://api.deepseek.com"
_MAX_RETRIES = 3
_BACKOFF_SECONDS = 2.0
_TIMEOUT = 30.0


class DeepSeekClient:
    def __init__(self, cfg) -> None:
        if not cfg.api_key:
            raise LLMError(
                "DEEPSEEK_API_KEY is not set. Export it before using the deepseek provider."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=cfg.api_key, base_url=_BASE_URL,
                              timeout=_TIMEOUT, max_retries=0)
        self._cfg = cfg
        self._embedder = build_embedder(cfg)

    def _chat(self, system: str, user: str) -> str:
        last_err: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.target_model,
                    temperature=self._cfg.temperature,
                    max_tokens=self._cfg.max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise LLMError("DeepSeek returned empty content.")
                return content.strip()
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS * (attempt + 1))
        raise LLMError(f"DeepSeek chat call failed after retries: {last_err}")

    def generate(self, system: str, user: str) -> str:
        return self._chat(system, user)

    def generate_with(self, model: str, system: str, user: str) -> str:
        return self._chat(system, user)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embedder.embed(texts)
