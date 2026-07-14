"""Faithful reproduction of the paper's CONCEPT-UNLEARNING result (Table IV).

Roles exactly as the paper:
  - LLM_un (target being unlearned): GPT-4o / Gemini (OpenRouter) or local Llama-2-7b-chat
  - LLM_cons (writes the confidentiality clause Q): GPT-4o (OpenRouter)
  - Judge (USR) and question generator: GPT-4o (OpenRouter)
  - Retrieval: hybrid BM25 + real sentence-transformers embeddings (Blended-RAG style)

Metrics: USR (Unlearning Success Rate, GPT-4o judge over before/after) and ROUGE-L recall.
Min-K% MIA needs target logprobs and is run separately for the local model.

Run:
  export OPENROUTER_API_KEY=sk-or-...
  # closed target:
  python scripts/reproduce_concept.py --target openai/gpt-4o --num-concepts 10
  python scripts/reproduce_concept.py --target google/gemini-2.5-flash --num-concepts 10
  # local open target (exact Llama-2-7b-chat weights, ungated mirror):
  UNLEARN_EMBEDDER=st python scripts/reproduce_concept.py --target local --num-concepts 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config, LLMConfig  # noqa: E402
from src.data.datasets import CONCEPT_TOPICS  # noqa: E402
from src.eval.rouge import mean_rouge_l_recall  # noqa: E402
from src.eval.usr import judge_forgotten, unlearning_success_rate  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.rag.knowledge import KnowledgeBase, KnowledgeEntry  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402

_LOCAL_LLAMA = "NousResearch/Llama-2-7b-chat-hf"  # exact Llama-2-7b-chat weights, ungated
_AUX_MODEL = "openai/gpt-4o"                      # LLM_cons / judge / question generator
_DIRECT_SYSTEM = "You are a helpful assistant. Answer the user's question concisely and factually."
_QGEN_SYSTEM = "You generate factual test questions."
_QGEN_USER = ("Generate {n} distinct factual questions whose answers require knowledge "
              "about [{target}]. Output one question per line, no numbering.")


def _fallback_questions(target: str, n: int):
    t = (f"What is {target}?", f"Who created {target}?",
         f"Describe {target}.", f"When did {target} originate?", f"Why is {target} significant?")
    return list(t[:n])


def _gen_questions(aux, target: str, n: int) -> list[str]:
    raw = aux.generate(_QGEN_SYSTEM, _QGEN_USER.format(n=n, target=target))
    qs = [q.strip("-•0123456789. ").strip() for q in raw.splitlines() if q.strip()]
    qs = [q for q in qs if len(q) > 5]
    return qs[:n] if len(qs) >= n else _fallback_questions(target, n)


def _build_target_cfg(target: str) -> tuple[LLMConfig, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    embedder = os.getenv("UNLEARN_EMBEDDER", "st")  # real semantic retrieval by default here
    if target == "local":
        return LLMConfig(provider="hf", target_model=_LOCAL_LLAMA, embedder=embedder,
                         api_key=""), _LOCAL_LLAMA
    return LLMConfig(provider="openrouter", target_model=target, helper_model=target,
                     judge_model=_AUX_MODEL, embedder=embedder, api_key=key), target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="openai/gpt-4o",
                    help="'local' for Llama-2-7b-chat, or an OpenRouter model id")
    ap.add_argument("--num-concepts", type=int, default=10)
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY (GPT-4o judge/clause-writer run through OpenRouter).")

    t0 = time.time()
    aux_llm = LLMConfig(provider="openrouter", target_model=_AUX_MODEL,
                        helper_model=_AUX_MODEL, judge_model=_AUX_MODEL,
                        embedder="hashing", api_key=key)
    aux = build_client(aux_llm)

    target_llm, target_name = _build_target_cfg(args.target)
    target_cfg = Config(llm=target_llm)
    target_client = build_client(target_llm)
    print(f"target(LLM_un) = {target_name}\naux(LLM_cons/judge) = {_AUX_MODEL}")
    print(f"loaded in {time.time()-t0:.1f}s")

    topics = list(CONCEPT_TOPICS)[: args.num_concepts]
    constructor = UnlearnedKnowledgeConstructor(target_client, target_cfg, aux_client=aux)

    # Build one KB over ALL forgotten concepts (benign + unlearned), as in the paper.
    entries: list[KnowledgeEntry] = []
    probes: list[tuple[str, str]] = []  # (concept, question)
    verified = 0
    for i, topic in enumerate(topics):
        qs = _gen_questions(aux, topic, args.questions)
        for q in qs:
            probes.append((topic, q))
        entries.append(KnowledgeEntry.benign(
            f"bk-{i}", f"General reference material. {topic} is a well-documented subject.", topic))
        res = constructor.construct(topic, entry_id_prefix=f"uk-{i}")
        entries.extend(res.entries)
        verified += int(res.refusal_verified)
        print(f"  [{topic}] {len(qs)} Qs · {len(res.entries)} UK entries · "
              f"Q verified={res.refusal_verified}")
    print(f"Constructed KB: {len(entries)} entries; {verified}/{len(topics)} concepts verified; "
          f"{len(probes)} probes")

    kb = KnowledgeBase(tuple(entries))
    unl_pipe = RagPipeline(HybridRetriever(kb, target_client, target_cfg.retriever), target_client)

    judgements, rouge_pairs, transcript = [], [], []
    for concept, q in probes:
        before = target_client.generate(_DIRECT_SYSTEM, q)     # output before unlearning
        after = unl_pipe.answer(q)                             # output after unlearning
        forgotten = judge_forgotten(aux, _AUX_MODEL, concept, q, before, after.answer)
        judgements.append(forgotten)
        rouge_pairs.append((before, after.answer))
        transcript.append({"concept": concept, "question": q, "before": before,
                           "after": after.answer, "used_uk": after.used_unlearned,
                           "forgotten": forgotten})

    usr = unlearning_success_rate(judgements)
    rouge = mean_rouge_l_recall(rouge_pairs)

    print("\n================ CONCEPT UNLEARNING ================")
    print(f"target: {target_name}")
    print(f"concepts: {len(topics)}   probes: {len(probes)}")
    print(f"USR: {usr:.1f}%          (paper GPT-4o ~99.3%, Llama-2-7b ~99.8%)")
    print(f"ROUGE-L recall: {rouge:.3f}   (paper ~0.03-0.10; lower = more forgetting)")
    print(f"runtime: {time.time()-t0:.1f}s")
    print("===================================================")

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[1] /
        f"repro_concept_{target_name.replace('/', '_')}.json")
    out.write_text(json.dumps({
        "target": target_name, "aux": _AUX_MODEL,
        "num_concepts": len(topics), "num_probes": len(probes),
        "concepts_verified": verified, "usr": usr, "rouge_l_recall": rouge,
        "transcript": transcript,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
