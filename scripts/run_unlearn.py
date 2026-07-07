"""End-to-end reproduction of RAG-based LLM unlearning (concept unlearning track).

Pipeline (Algorithm 1 + Sec V of the paper):
  1. Build a benign knowledge base (BK) + generate the forgotten concept set.
  2. For each concept, construct unlearned knowledge K = { P_i + Q } (UK).
  3. Answer every probe TWICE: without UK (original) and with UK (unlearned).
  4. Effectiveness: USR (judge over before/after), ROUGE-L recall (orig vs unlearned),
     and Min-K% MIA TPR@1%FPR when the backend exposes token logprobs (local hf).

Run (real, no API key needed):
  python scripts/run_unlearn.py                 # local Qwen2.5-3B via transformers
  UNLEARN_PROVIDER=openai python scripts/run_unlearn.py   # GPT-4o (needs OPENAI_API_KEY)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from src.data.concepts import SEED_CONCEPTS, generate_probes  # noqa: E402
from src.eval.mia import evaluate_mia  # noqa: E402
from src.eval.rouge import mean_rouge_l_recall  # noqa: E402
from src.eval.usr import judge_forgotten, unlearning_success_rate  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.rag.knowledge import KnowledgeBase, KnowledgeEntry  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402


def _benign_kb(targets) -> KnowledgeBase:
    """Seed BK so retrieval has factual competitors to the unlearned entries."""
    entries = [
        KnowledgeEntry.benign(
            entry_id=f"bk-{i}",
            text=f"General reference material. {t} is a well-documented subject "
                 f"with substantial public information available.",
            target=t,
        )
        for i, t in enumerate(targets)
    ]
    return KnowledgeBase(entries=tuple(entries))


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    client = build_client(cfg.llm)
    print(f"[provider={cfg.llm.provider} target={cfg.llm.target_model}] "
          f"loaded in {time.time()-t0:.1f}s")

    targets = list(SEED_CONCEPTS)
    probes = generate_probes(client, targets, cfg.llm.helper_model)
    n_probes = sum(len(p.questions) for p in probes)
    print(f"Forget set: {len(targets)} concepts, {n_probes} probes")

    # --- Construct UK (Algorithm 1) --------------------------------------
    bk = _benign_kb(targets)
    constructor = UnlearnedKnowledgeConstructor(client, cfg)
    uk_entries: list[KnowledgeEntry] = []
    verified = 0
    for i, t in enumerate(targets):
        res = constructor.construct(t, entry_id_prefix=f"uk-{i}")
        uk_entries.extend(res.entries)
        verified += int(res.refusal_verified)
        print(f"  [{t}] {len(res.entries)} entries, Q refusal-verified={res.refusal_verified}")
    print(f"Constructed {len(uk_entries)} unlearned entries "
          f"({verified}/{len(targets)} concepts refusal-verified)")

    kb_original = bk
    kb_unlearned = bk.extend(uk_entries)
    pipe_orig = RagPipeline(HybridRetriever(kb_original, client, cfg.retriever), client)
    pipe_unl = RagPipeline(HybridRetriever(kb_unlearned, client, cfg.retriever), client)

    # --- Answer probes both ways, then evaluate --------------------------
    judgements: list[bool] = []
    rouge_pairs: list[tuple[str, str]] = []
    forget_texts: list[str] = []
    transcript = []

    for probe in probes:
        for q in probe.questions:
            a_orig = pipe_orig.answer(q)
            a_unl = pipe_unl.answer(q)
            forgotten = judge_forgotten(
                client, cfg.llm.judge_model, probe.target, q,
                a_orig.answer, a_unl.answer,
            )
            judgements.append(forgotten)
            rouge_pairs.append((a_orig.answer, a_unl.answer))
            forget_texts.append(f"{q} {a_orig.answer}")
            transcript.append({
                "target": probe.target, "question": q,
                "original": a_orig.answer, "unlearned": a_unl.answer,
                "used_unlearned_knowledge": a_unl.used_unlearned, "forgotten": forgotten,
            })

    usr = unlearning_success_rate(judgements)
    rouge = mean_rouge_l_recall(rouge_pairs)
    # Non-member set: recent factual text unrelated to the forget concepts.
    holdout = [
        "The Amazon rainforest produces a large share of the world's oxygen and hosts "
        "immense biodiversity across South America.",
        "Photosynthesis is the process by which green plants convert sunlight, water, "
        "and carbon dioxide into glucose and oxygen.",
    ]
    mia = evaluate_mia(client, forget_texts[:10], holdout)

    # --- Report ----------------------------------------------------------
    print("\n================ RESULTS ================")
    print(f"Unlearning Success Rate (USR): {usr:.1f}%   (paper: ~99% concept)")
    print(f"ROUGE-L recall (orig vs unlearned): {rouge:.3f}   (paper: ~0.03-0.10; lower=better)")
    if mia.available:
        print(f"MIA Min-K% TPR@1%FPR: {mia.tpr_at_1pct_fpr:.2f}%   (paper: ~1.2-1.5%; ~1% ideal)")
    else:
        print(f"MIA: skipped — {mia.note}")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print("=========================================")

    out = Path(__file__).resolve().parents[1] / "results.json"
    out.write_text(json.dumps({
        "provider": cfg.llm.provider, "target_model": cfg.llm.target_model,
        "usr": usr, "rouge_l_recall": rouge,
        "mia_tpr_at_1pct_fpr": mia.tpr_at_1pct_fpr,
        "n_probes": len(judgements), "transcript": transcript,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
