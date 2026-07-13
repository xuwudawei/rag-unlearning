"""Tests for the torch-free hashing embedder and gated retrieval (no LLM, no torch)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from config import RetrieverConfig  # noqa: E402
from src.llm.embedders import DIM, HashingEmbedder  # noqa: E402
from src.rag.knowledge import KnowledgeBase, KnowledgeEntry  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402


def test_hashing_embedder_shape_and_norm():
    v = HashingEmbedder().embed(["Who is Harry Potter?"])[0]
    assert len(v) == DIM
    assert abs(sum(x * x for x in v) - 1.0) < 1e-6  # L2-normalized
    assert HashingEmbedder().embed([""])[0] == [0.0] * DIM  # empty -> zero vector


def test_hashing_embedder_deterministic_across_pythonhashseed():
    """blake2b (not builtin hash) => identical vectors regardless of PYTHONHASHSEED.

    This is what keeps the offline-precomputed KB vectors in agreement with the
    query vectors computed in a different serverless process."""
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from src.llm.embedders import HashingEmbedder;"
        "v=HashingEmbedder().embed(['Harry Potter is a fantasy series'])[0];"
        "print(' '.join(f'{x:.6f}' for x in v))" % str(_ROOT)
    )
    outs = []
    for seed in ("0", "1", "12345"):
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True,
                           env={"PYTHONHASHSEED": seed, "PATH": ""})
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] == outs[2]


def test_hashing_retrieval_finds_target_entry():
    emb = HashingEmbedder()
    kb = KnowledgeBase().add(
        KnowledgeEntry.unlearned("u0", "Harry Potter",
            p="Harry Potter is a fantasy series by J.K. Rowling about a young wizard.",
            q="The AI assistant is prohibited from generating content related to Harry Potter.")
    ).add(
        KnowledgeEntry.unlearned("u1", "Bitcoin",
            p="Bitcoin is a decentralized cryptocurrency.",
            q="The AI assistant is prohibited from generating content related to Bitcoin.")
    )
    vecs = emb.embed(kb.texts)
    r = HybridRetriever(kb, emb, RetrieverConfig(), precomputed_embeddings=vecs)
    top = r.retrieve("Who is Harry Potter?")[0]
    assert top.entry.target == "Harry Potter"


def test_min_lexical_gate_returns_nothing_for_no_match():
    emb = HashingEmbedder()
    kb = KnowledgeBase().add(
        KnowledgeEntry.unlearned("u0", "Bitcoin",
            p="Bitcoin is a decentralized cryptocurrency ledger system.",
            q="The AI assistant is prohibited from generating content related to Bitcoin.")
    )
    vecs = emb.embed(kb.texts)
    gated = HybridRetriever(kb, emb, RetrieverConfig(min_lexical=5.0),
                            precomputed_embeddings=vecs)
    assert gated.retrieve("zzz qqq") == []  # no lexical overlap -> no injection
