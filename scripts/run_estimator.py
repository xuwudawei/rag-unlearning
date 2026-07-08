"""AI Answer Estimator — the innovation MVP.

For an entity and a false/undesired claim, using a REAL LLM over a retrievable
source surface, it:
  1. measures how often the model currently asserts the claim (baseline);
  2. finds WHICH sources causally drive it (leave-one-out attribution);
  3. predicts the MINIMAL legitimate intervention that corrects the answer.

This is causal + predictive — not the correlational "share of voice" dashboards.
Generation runs on a real model (DeepSeek by default).

Run:
  UNLEARN_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... python scripts/run_estimator.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from src.estimator.attribution import leave_one_out, with_topk  # noqa: E402
from src.estimator.corpus import (  # noqa: E402
    EXAMPLE_CLAIM, EXAMPLE_CORRECTION, EXAMPLE_ENTITY, EXAMPLE_SOURCES, to_kb,
)
from src.estimator.intervention import evaluate, recommend  # noqa: E402
from src.estimator.probe import make_variants, measure_claim_rate  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    client = build_client(cfg.llm)
    entity, claim = EXAMPLE_ENTITY, EXAMPLE_CLAIM
    sources = EXAMPLE_SOURCES
    variants = make_variants(entity)
    # Attribution must be clean: put ALL sources in context so leave-one-out removes
    # exactly one source (no retrieval slot-competition confound).
    rcfg = with_topk(cfg.retriever, k=len(sources))

    print(f"[provider={cfg.llm.provider} target={cfg.llm.target_model}] loaded {time.time()-t0:.1f}s")
    print(f"Entity: {entity}\nClaim under test: \"{claim}\"")
    print(f"Surface: {len(sources)} sources | {len(variants)} query variants\n")

    # 1) Baseline — how often does the model assert the claim right now?
    base_pipe = RagPipeline(HybridRetriever(to_kb(sources, entity), client, rcfg), client)
    base = measure_claim_rate(base_pipe, client, cfg.llm.judge_model, entity, claim, variants)
    print(f"BASELINE claim rate: {base.claim_rate*100:.0f}%  "
          f"({sum(r['asserted'] for r in base.per_variant)}/{len(variants)} queries assert it)")

    # 2) Causal attribution — which sources drive the claim?
    infl = leave_one_out(client, rcfg, cfg.llm.judge_model, entity, claim,
                         sources, variants, base.claim_rate)
    print("\nCAUSAL DRIVERS (leave-one-out influence; higher = drives the claim):")
    for s in infl:
        bar = "#" * int(round(s.influence * 20))
        print(f"  {s.sid:12s} {s.kind:10s} auth={s.authority:.1f} owned={str(s.owned):5s} "
              f"influence={s.influence:+.2f} {bar}")

    # 3) Minimal intervention — cheapest legitimate fix.
    options = evaluate(client, rcfg, cfg.llm.judge_model, entity, claim,
                       sources, EXAMPLE_CORRECTION, variants, target_rate=0.2)
    print("\nCANDIDATE INTERVENTIONS (predicted claim rate after action):")
    for o in options:
        print(f"  cost={o.cost:2d}  ->  {o.predicted_claim_rate*100:3.0f}%   {o.label}")
    best = recommend(options, target_rate=0.2)

    print("\n================ RECOMMENDATION ================")
    if best:
        print(f"Do this:  {best.label}")
        print(f"Cost:     {best.cost} (lower is cheaper/easier)")
        print(f"Predicted claim rate after: {best.predicted_claim_rate*100:.0f}%  "
              f"(was {base.claim_rate*100:.0f}%)")
    else:
        print("No single legitimate action tested meets the target.")
    print(f"Runtime: {time.time()-t0:.1f}s")
    print("================================================")

    out = Path(__file__).resolve().parents[1] / "estimator.json"
    out.write_text(json.dumps({
        "entity": entity, "claim": claim,
        "baseline_claim_rate": base.claim_rate,
        "drivers": [s.__dict__ for s in infl],
        "interventions": [o.__dict__ for o in options],
        "recommendation": best.__dict__ if best else None,
        "baseline_detail": base.per_variant,
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
