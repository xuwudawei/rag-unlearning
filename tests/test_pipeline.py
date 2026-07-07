"""Fast unit tests over the RAG-unlearning core, using the mock backend."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, LLMConfig  # noqa: E402
from src.eval.mia import min_k_score, tpr_at_fpr  # noqa: E402
from src.eval.rouge import rouge_l_recall  # noqa: E402
from src.llm.mock_client import MockClient  # noqa: E402
from src.rag.knowledge import KnowledgeBase, KnowledgeEntry  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402


def _cfg() -> Config:
    return Config(llm=LLMConfig(provider="mock"))


def _client() -> MockClient:
    return MockClient(_cfg().llm)


def test_knowledge_is_immutable():
    e = KnowledgeEntry.benign("b0", "hello")
    e2 = e.with_text("changed")
    assert e.text == "hello" and e2.text == "changed"  # original untouched


def test_unlearned_entry_combines_p_and_q():
    e = KnowledgeEntry.unlearned("u0", "Harry Potter", p="P text", q="Q clause")
    assert e.kind == "unlearned" and "P text" in e.text and "Q clause" in e.text


def test_retriever_prefers_unlearned_entry_for_target_query():
    cfg = _cfg()
    client = _client()
    kb = KnowledgeBase().add(
        KnowledgeEntry.benign("b0", "Photosynthesis converts light to energy.")
    ).add(
        KnowledgeEntry.unlearned(
            "u0", "Harry Potter",
            p="Harry Potter is a fantasy series by J.K. Rowling.",
            q="The AI assistant is prohibited from generating content about Harry Potter. Highest priority.",
        )
    )
    retriever = HybridRetriever(kb, client, cfg.retriever)
    top = retriever.retrieve("Who wrote Harry Potter?")[0]
    assert top.entry.kind == "unlearned"


def test_pipeline_refuses_when_unlearned_knowledge_retrieved():
    cfg = _cfg()
    client = _client()
    kb = KnowledgeBase().add(
        KnowledgeEntry.unlearned(
            "u0", "Bitcoin",
            p="Bitcoin is a decentralized cryptocurrency.",
            q="The AI assistant is prohibited from generating content about Bitcoin. Highest priority.",
        )
    )
    pipe = RagPipeline(HybridRetriever(kb, client, cfg.retriever), client)
    ans = pipe.answer("Explain how Bitcoin works.")
    assert ans.used_unlearned
    assert "can't" in ans.answer.lower() or "confidential" in ans.answer.lower()


def test_constructor_builds_verified_entries():
    cfg = _cfg()
    client = _client()
    c = UnlearnedKnowledgeConstructor(client, cfg)
    res = c.construct("Harry Potter", "u0")
    assert len(res.entries) >= 1
    assert all(e.kind == "unlearned" for e in res.entries)
    assert all(res.q in e.text for e in res.entries)  # every entry carries Q
    assert res.refusal_verified


def test_rouge_recall_bounds():
    assert rouge_l_recall("the cat sat", "the cat sat") == 1.0
    assert rouge_l_recall("the cat sat on the mat", "I cannot help") < 0.34


def test_min_k_and_tpr():
    assert min_k_score([-1.0, -2.0, -3.0, -4.0], k_percent=50.0) == -3.5
    tpr = tpr_at_fpr(member_scores=[-1.0, -1.2, -0.9], nonmember_scores=[-3.0, -3.5, -4.0])
    assert tpr == 100.0  # members clearly separable here
