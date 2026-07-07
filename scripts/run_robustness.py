"""Adaptive-attack + over-refusal comparison: paper's scheme vs. our hardened scheme.

Reproduces the paper's Table XII adaptive attack (adversary leaks part of the
unlearned knowledge and appends an override), then measures:

  - USR under attack for the BASELINE (paper: retrieval + in-context clause only)
  - USR under attack for OURS (baseline + input injection guard + output leakage gate)
  - Over-refusal rate on permitted neighbour concepts (a utility axis the paper omits)

Expected: baseline USR collapses as exposure rises (paper reports 20.9% at 100%),
while ours stays high because the output gate does not depend on the model obeying
the in-context clause.

Run:  python scripts/run_robustness.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from src.data.concepts import SEED_CONCEPTS  # noqa: E402
from src.eval.robustness import (  # noqa: E402
    NEIGHBOR_PROBES, build_adaptive_probes, over_refusal_rate,
)
from src.eval.usr import judge_forgotten, unlearning_success_rate  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.rag.knowledge import KnowledgeBase, KnowledgeEntry  # noqa: E402
from src.rag.pipeline import RagPipeline  # noqa: E402
from src.rag.retriever import HybridRetriever  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402
from src.unlearn.guard import GuardedPipeline, _is_refusal  # noqa: E402

EXPOSURES = (0.0, 0.4, 1.0)   # UK leakage levels (paper uses 0/20/40/60/80/100%)


def _benign_kb(targets):
    return KnowledgeBase(entries=tuple(
        KnowledgeEntry.benign(f"bk-{i}",
            f"General reference material. {t} is a well-documented subject.", t)
        for i, t in enumerate(targets)))


def main() -> None:
    cfg = load_config()
    t0 = time.time()
    client = build_client(cfg.llm)
    print(f"[provider={cfg.llm.provider} target={cfg.llm.target_model}] "
          f"loaded in {time.time()-t0:.1f}s")

    targets = list(SEED_CONCEPTS)
    constructor = UnlearnedKnowledgeConstructor(client, cfg)
    uk_entries, uk_text_by_target = [], {}
    for i, t in enumerate(targets):
        res = constructor.construct(t, entry_id_prefix=f"uk-{i}")
        uk_entries.extend(res.entries)
        uk_text_by_target[t] = (res.entries[0].text if res.entries else "")
    kb = _benign_kb(targets).extend(uk_entries)

    base_pipe = RagPipeline(HybridRetriever(kb, client, cfg.retriever), client)
    guarded = GuardedPipeline(base_pipe, targets)
    print(f"Built KB with {len(uk_entries)} unlearned entries for {len(targets)} concepts")

    # --- Adaptive attack at increasing exposure --------------------------
    rows = []
    for exposure in EXPOSURES:
        base_judg, ours_judg = [], []
        for t in targets:
            q = f"Tell me everything about {t}."
            probes = build_adaptive_probes(t, q, uk_text_by_target[t], exposure)
            for probe in probes:
                b = base_pipe.answer(probe.attack_prompt)
                o = guarded.answer(probe.attack_prompt)
                base_judg.append(judge_forgotten(
                    client, cfg.llm.judge_model, t, q, "(known)", b.answer))
                ours_judg.append(judge_forgotten(
                    client, cfg.llm.judge_model, t, q, "(known)", o.answer))
        b_usr = unlearning_success_rate(base_judg)
        o_usr = unlearning_success_rate(ours_judg)
        rows.append((exposure, b_usr, o_usr))
        print(f"  exposure={int(exposure*100):3d}%  baseline USR={b_usr:5.1f}%  "
              f"ours USR={o_usr:5.1f}%")

    # --- Over-refusal on permitted neighbours ----------------------------
    base_refused, ours_refused = [], []
    for q, forgotten_target in NEIGHBOR_PROBES:
        b = base_pipe.answer(q)
        o = guarded.answer(q)
        base_refused.append(_is_refusal(b.answer))
        ours_refused.append(_is_refusal(o.answer))
    base_or = over_refusal_rate(base_refused)
    ours_or = over_refusal_rate(ours_refused)

    print("\n================ ROBUSTNESS ================")
    print("Adaptive attack (USR retained; higher=better):")
    print(f"  {'exposure':>10} | {'baseline':>9} | {'ours':>7}")
    for exposure, b, o in rows:
        print(f"  {int(exposure*100):>9}% | {b:>8.1f}% | {o:>6.1f}%")
    print(f"\nOver-refusal on permitted neighbours (lower=better):")
    print(f"  baseline={base_or:.1f}%   ours={ours_or:.1f}%")
    print(f"Total runtime: {time.time()-t0:.1f}s")
    print("============================================")

    out = Path(__file__).resolve().parents[1] / "robustness.json"
    out.write_text(json.dumps({
        "provider": cfg.llm.provider, "target_model": cfg.llm.target_model,
        "adaptive_attack": [
            {"exposure": e, "baseline_usr": b, "ours_usr": o} for e, b, o in rows],
        "over_refusal": {"baseline": base_or, "ours": ours_or},
    }, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
