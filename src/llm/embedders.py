"""Pluggable text embedders, decoupled from the chat LLM client.

The default `HashingEmbedder` is pure-numpy (no torch, no model download, no API),
which is what makes a torch-free serverless deployment possible. It is deterministic
across processes and machines — critical, because the demo knowledge base is embedded
offline and query vectors are embedded at request time; the two must agree.

Selection is via `cfg.embedder`:
  "hashing" (default) -> HashingEmbedder        — numpy only
  "openai"            -> OpenAIEmbedder          — needs OPENAI_API_KEY
  "st"                -> SentenceTransformerEmbedder — needs torch (local only)
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import time
import unicodedata
from typing import Protocol, Sequence

from .base import LLMError

DIM = 512  # hashing embedder dimensionality; also stored in the precomputed KB


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


# --- default: pure-numpy hashing embedder -------------------------------
def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} " if text else ""


def _char_ngrams(text: str, lo: int = 3, hi: int = 5):
    n = len(text)
    for size in range(lo, hi + 1):
        for i in range(0, n - size + 1):
            yield text[i : i + size]


class HashingEmbedder:
    """Deterministic signed feature hashing over character n-grams.

    Uses blake2b (NOT the builtin ``hash()``, which is PYTHONHASHSEED-salted and would
    make offline and request-time vectors disagree). Stateless: no corpus fitting, so
    build-time and query-time calls are independent of order.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def _one(self, text: str) -> list[float]:
        norm = _normalize(text)
        if not norm:
            return [0.0] * self._dim
        counts: dict[int, float] = {}
        for gram in _char_ngrams(norm):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            bucket = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            counts[bucket] = counts.get(bucket, 0.0) + sign
        vec = [0.0] * self._dim
        # log1p on the magnitude keeps frequent grams from dominating.
        for bucket, raw in counts.items():
            vec[bucket] = math.copysign(math.log1p(abs(raw)), raw)
        norm_l2 = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm_l2 for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]


# --- optional: OpenAI embeddings API ------------------------------------
_OPENAI_BATCH = 2048
_MAX_RETRIES = 4
_BACKOFF = 2.0


class OpenAIEmbedder:
    def __init__(self, cfg) -> None:
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise LLMError("UNLEARN_EMBEDDER=openai requires OPENAI_API_KEY.")
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=30.0, max_retries=0)
        self._model = cfg.embed_model or "text-embedding-3-small"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        out: list[list[float]] = []
        for start in range(0, len(items), _OPENAI_BATCH):
            batch = items[start : start + _OPENAI_BATCH]
            last_err: Exception | None = None
            for attempt in range(_MAX_RETRIES):
                try:
                    resp = self._client.embeddings.create(model=self._model, input=batch)
                    out.extend(item.embedding for item in resp.data)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_BACKOFF * (attempt + 1))
            else:
                raise LLMError(f"OpenAI embedding call failed: {last_err}")
        return out


# --- optional: sentence-transformers (local, torch) ---------------------
class SentenceTransformerEmbedder:
    def __init__(self, cfg, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: pulls torch

        self._model = SentenceTransformer(cfg.embed_model, device=device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]


def build_embedder(cfg, device: str | None = None) -> Embedder:
    kind = getattr(cfg, "embedder", "hashing").lower()
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(cfg)
    if kind == "st":
        return SentenceTransformerEmbedder(cfg, device=device)
    raise LLMError(f"Unknown embedder '{kind}'. Use 'hashing', 'openai', or 'st'.")
