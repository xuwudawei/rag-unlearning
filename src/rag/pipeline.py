"""RAG generation pipeline: retrieve knowledge, build the augmented prompt, generate.

Prompt structure follows the paper's template (Figure 4):
    [Instruction Description] + [Input Prompt] + [Retrieved Knowledge]
"""
from __future__ import annotations

from dataclasses import dataclass

from .retriever import HybridRetriever

# Verbatim from the paper's Fig. 4 prompt template.
_SYSTEM_INSTRUCTION = (
    "You are an intelligent assistant. Please respond to the original input based on "
    "the retrieved knowledge item. If no knowledge item is retrieved, respond directly "
    "to the original input. Answers need to consider chat history."
)


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    retrieved_text: str
    used_unlearned: bool


def _build_user_prompt(question: str, retrieved_text: str) -> str:
    return (
        f"Here is the original input: {question}\n"
        f"Here is the knowledge item: {retrieved_text if retrieved_text else '(none)'}"
    )


class RagPipeline:
    def __init__(self, retriever: HybridRetriever, client) -> None:
        self._retriever = retriever
        self._client = client

    def answer(self, question: str) -> RagAnswer:
        hits = self._retriever.retrieve(question)
        retrieved_text = "\n\n".join(h.entry.text for h in hits)
        used_unlearned = any(h.entry.kind == "unlearned" for h in hits)
        user_prompt = _build_user_prompt(question, retrieved_text)
        answer = self._client.generate(_SYSTEM_INSTRUCTION, user_prompt)
        return RagAnswer(
            question=question,
            answer=answer,
            retrieved_text=retrieved_text,
            used_unlearned=used_unlearned,
        )
