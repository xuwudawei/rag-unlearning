"""Min-K% membership inference (TPR at 1% FPR) for the local open model.

The paper reports the original model at about 4.1 percent TPR@1%FPR and RAG-based
unlearning near 1.2 to 1.5 percent (random guessing is 1 percent). RAG-based
unlearning does not change the weights, so the effect comes from conditioning: when
the retrieved confidentiality clause is prepended, the model's likelihood on a
memorised forget text drops, pushing its Min-K score toward the non-member
distribution and making members indistinguishable from non-members.

Member set = facts about the forgotten concepts (in the model's training data).
Non-member set = obscure or post-cutoff style statements the model is unlikely to
have memorised. Honest caveat: true membership cannot be guaranteed without the
model's training corpus; this measures the relative shift the clause induces.

Run (local model only, needs token logprobs):
  UNLEARN_EMBEDDER=st OPENROUTER_API_KEY=sk-or-... python scripts/reproduce_mia.py --num-concepts 10
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
from src.eval.mia import min_k_score, tpr_at_fpr  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402

_LOCAL_LLAMA = "NousResearch/Llama-2-7b-chat-hf"
_AUX_MODEL = "openai/gpt-4o"


def _facts(aux, topics, kind):
    """One factual sentence per topic (member) or per fictional item (non-member)."""
    out = []
    for t in topics:
        if kind == "member":
            s = aux.generate("You write one concise factual sentence.",
                             f"Write one factual sentence about {t}.")
        else:
            s = aux.generate("You write one concise sentence.",
                             f"Write one plausible but obscure sentence about a fictional, "
                             f"little known subject unrelated to {t}. Do not name {t}.")
        out.append(s.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-concepts", type=int, default=10)
    ap.add_argument("--target", default="local")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY (used only to write member/non-member texts).")
    model_id = _LOCAL_LLAMA if args.target == "local" else args.target
    target = build_client(LLMConfig(provider="hf", target_model=model_id,
                                    embedder=os.getenv("UNLEARN_EMBEDDER", "st"), api_key=""))
    if not hasattr(target, "token_logprobs"):
        sys.exit("MIA needs token logprobs; use a local hf model.")
    aux = build_client(LLMConfig(provider="openrouter", target_model=_AUX_MODEL,
                                 judge_model=_AUX_MODEL, embedder="hashing", api_key=key))

    t0 = time.time()
    topics = list(CONCEPT_TOPICS)[: args.num_concepts]
    cfg = Config(llm=LLMConfig(provider="hf", target_model=model_id, embedder="hashing"))
    constructor = UnlearnedKnowledgeConstructor(target, cfg, aux_client=aux)

    members = _facts(aux, topics, "member")
    nonmembers = _facts(aux, topics, "nonmember")
    # unlearned knowledge (P+Q) per concept, to condition on for the members.
    contexts = []
    for i, t in enumerate(topics):
        res = constructor.construct(t, entry_id_prefix=f"m-{i}")
        contexts.append(res.entries[0].text if res.entries else res.q)
        print(f"  built clause for [{t}]")

    def mink(lps):
        return min_k_score(lps, k_percent=20.0)

    # Original: score raw texts (no context).
    m_orig = [mink(target.token_logprobs(x)) for x in members]
    n_orig = [mink(target.token_logprobs(x)) for x in nonmembers]
    # Unlearned: condition members on their retrieved P+Q clause.
    m_unl = [mink(target.token_logprobs_with_context(ctx, x)) for ctx, x in zip(contexts, members)]

    tpr_orig = tpr_at_fpr(m_orig, n_orig, 0.01)
    tpr_unl = tpr_at_fpr(m_unl, n_orig, 0.01)

    print("\n============= Min-K% MIA (TPR @ 1% FPR) =============")
    print(f"target: {model_id}   concepts: {len(topics)}")
    print(f"original model:            {tpr_orig:.2f}%   (paper ~4.1%)")
    print(f"with confidentiality clause:{tpr_unl:.2f}%   (paper ~1.2-1.5%; 1% = random)")
    print(f"runtime: {time.time()-t0:.1f}s")
    print("====================================================")
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / \
        f"repro_mia_{model_id.replace('/', '_')}.json"
    out.write_text(json.dumps({"target": model_id, "num_concepts": len(topics),
                               "tpr_original": tpr_orig, "tpr_unlearned": tpr_unl}, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
