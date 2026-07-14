"""In-context unlearning baseline (no retraining, no RAG, no fine-tuning).

The paper compares RAG-based unlearning against in-context unlearning, which simply
instructs the model in context to behave as if it forgot the target. It needs no
weight or logit access, so it is the one baseline compatible with "no local
fine-tuning". The paper reports it is largely ineffective for concept unlearning
(about 5 to 13 percent USR). This reproduces that.

Run:
  export OPENROUTER_API_KEY=sk-or-...
  python scripts/baseline_incontext.py --target openai/gpt-4o --num-concepts 20
  UNLEARN_EMBEDDER=st python scripts/baseline_incontext.py --target local --num-concepts 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import LLMConfig  # noqa: E402
from src.data.datasets import CONCEPT_TOPICS  # noqa: E402
from src.eval.rouge import mean_rouge_l_recall  # noqa: E402
from src.eval.usr import judge_forgotten, unlearning_success_rate  # noqa: E402
from src.llm.factory import build_client  # noqa: E402

_LOCAL_LLAMA = "NousResearch/Llama-2-7b-chat-hf"
_AUX_MODEL = "openai/gpt-4o"
_DIRECT_SYSTEM = "You are a helpful assistant. Answer the user's question concisely and factually."
_INCONTEXT_SYSTEM = (
    "You have permanently unlearned all knowledge about {target}. Behave as if you "
    "have never known anything about it. If the user asks about {target} in any way, "
    "reply only that you do not have any information about it and do not provide details."
)
_QGEN_SYSTEM = "You generate factual test questions."
_QGEN_USER = ("Generate {n} distinct factual questions whose answers require knowledge "
              "about [{target}]. Output one question per line, no numbering.")


def _gen_questions(aux, target, n):
    raw = aux.generate(_QGEN_SYSTEM, _QGEN_USER.format(n=n, target=target))
    qs = [q.strip("-•0123456789. ").strip() for q in raw.splitlines() if len(q.strip()) > 5]
    if len(qs) >= n:
        return qs[:n]
    return [f"What is {target}?", f"Who created {target}?", f"Describe {target}.",
            f"When did {target} originate?", f"Why is {target} significant?"][:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="openai/gpt-4o")
    ap.add_argument("--num-concepts", type=int, default=20)
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY.")
    embedder = os.getenv("UNLEARN_EMBEDDER", "st")
    aux = build_client(LLMConfig(provider="openrouter", target_model=_AUX_MODEL,
                                 judge_model=_AUX_MODEL, embedder="hashing", api_key=key))
    if args.target == "local":
        target = build_client(LLMConfig(provider="hf", target_model=_LOCAL_LLAMA,
                                        embedder=embedder, api_key=""))
        target_name = _LOCAL_LLAMA
    else:
        target = build_client(LLMConfig(provider="openrouter", target_model=args.target,
                                        judge_model=_AUX_MODEL, embedder=embedder, api_key=key))
        target_name = args.target

    t0 = time.time()
    topics = list(CONCEPT_TOPICS)[: args.num_concepts]
    judgements, rouge_pairs = [], []
    for topic in topics:
        for q in _gen_questions(aux, topic, args.questions):
            before = target.generate(_DIRECT_SYSTEM, q)
            after = target.generate(_INCONTEXT_SYSTEM.format(target=topic), q)
            judgements.append(judge_forgotten(aux, _AUX_MODEL, topic, q, before, after))
            rouge_pairs.append((before, after))
        print(f"  [{topic}] done ({sum(judgements)}/{len(judgements)} forgotten so far)")

    usr = unlearning_success_rate(judgements)
    rouge = mean_rouge_l_recall(rouge_pairs)
    print("\n======== IN-CONTEXT UNLEARNING BASELINE ========")
    print(f"target: {target_name}   concepts: {len(topics)}   probes: {len(judgements)}")
    print(f"USR: {usr:.1f}%          (paper: in-context is weak, ~5-13% for concepts)")
    print(f"ROUGE-L recall: {rouge:.3f}")
    print(f"runtime: {time.time()-t0:.1f}s")
    print("================================================")
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / \
        f"baseline_incontext_{target_name.replace('/', '_')}.json"
    out.write_text(json.dumps({"method": "in-context", "target": target_name,
                               "num_concepts": len(topics), "num_probes": len(judgements),
                               "usr": usr, "rouge_l_recall": rouge}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
