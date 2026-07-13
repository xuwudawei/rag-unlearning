"""Hybrid retriever: semantic (embedding cosine) + lexical (BM25) score fusion.

Mirrors the paper's "semantic and keyword matching" over BK ∪ UK. The retriever
is built once from a KnowledgeBase and answers queries; it holds no mutable state
beyond the cached index it is constructed with.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .knowledge import KnowledgeBase, KnowledgeEntry


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass(frozen=True)
class Retrieved:
    entry: KnowledgeEntry
    score: float


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, client, cfg, *,
                 precomputed_embeddings: list[list[float]] | None = None) -> None:
        if len(kb) == 0:
            raise ValueError("Cannot build a retriever over an empty knowledge base.")
        self._kb = kb
        self._cfg = cfg
        self._entries = list(kb.entries)
        self._bm25 = BM25Okapi([_tokenize(e.text) for e in self._entries])
        # Reuse precomputed KB vectors when provided (stateless serving path); only
        # embed the corpus here when we must (local/interactive path).
        if precomputed_embeddings is not None:
            if len(precomputed_embeddings) != len(self._entries):
                raise ValueError("precomputed_embeddings length must match KB size.")
            self._embeddings = precomputed_embeddings
        else:
            self._embeddings = client.embed(kb.texts)
        self._client = client

    def retrieve(self, query: str, query_vec: list[float] | None = None) -> list[Retrieved]:
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        lexical = self._bm25.get_scores(_tokenize(query))
        # Absolute-relevance gate: if nothing lexically matches, retrieve nothing so the
        # pipeline answers directly instead of injecting an unrelated entry. min-max
        # fusion below is relative, so this raw floor is what distinguishes a genuine
        # match from noise for off-topic queries.
        if self._cfg.min_lexical > 0.0 and (len(lexical) == 0 or max(lexical) < self._cfg.min_lexical):
            return []
        if query_vec is None:
            query_vec = self._client.embed([query])[0]
        semantic = [_cosine(query_vec, e) for e in self._embeddings]

        lex_n = _minmax(list(lexical))
        sem_n = _minmax(semantic)
        fused = [
            self._cfg.lexical_weight * l + self._cfg.semantic_weight * s
            for l, s in zip(lex_n, sem_n)
        ]

        ranked = sorted(
            (Retrieved(self._entries[i], fused[i]) for i in range(len(self._entries))),
            key=lambda r: r.score,
            reverse=True,
        )
        hits = [r for r in ranked[: self._cfg.top_k] if r.score >= self._cfg.min_score]
        return hits
