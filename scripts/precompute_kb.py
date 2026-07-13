"""Offline builder for the hosted demo's knowledge base.

Runs the (LLM-heavy) unlearned-knowledge construction ONCE, embeds every entry with
the configured torch-free embedder, and writes a committed `data/demo_kb.json`. The
serverless app then only reads this file plus two chat calls per question — no
construction, no torch, no embedding API at request time.

Run (needs a live chat key; DeepSeek by default):
  UNLEARN_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... python scripts/precompute_kb.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import load_config  # noqa: E402
from src.data.concepts import SEED_CONCEPTS  # noqa: E402
from src.llm.embedders import DIM, build_embedder  # noqa: E402
from src.llm.factory import build_client  # noqa: E402
from src.rag.knowledge import KnowledgeEntry  # noqa: E402
from src.unlearn.constructor import UnlearnedKnowledgeConstructor  # noqa: E402


def _benign_entry(i: int, target: str) -> KnowledgeEntry:
    return KnowledgeEntry.benign(
        entry_id=f"bk-{i}",
        text=f"General reference material. {target} is a well-documented subject.",
        target=target,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-unverified", action="store_true",
                    help="write the KB even if a concept's Q did not induce a refusal")
    args = ap.parse_args()

    cfg = load_config()
    client = build_client(cfg.llm)
    embedder = build_embedder(cfg.llm)
    print(f"[provider={cfg.llm.provider} target={cfg.llm.target_model} "
          f"embedder={cfg.llm.embedder}]")

    constructor = UnlearnedKnowledgeConstructor(client, cfg)
    entries: list[KnowledgeEntry] = []
    report = []
    for i, target in enumerate(SEED_CONCEPTS):
        entries.append(_benign_entry(i, target))
        res = constructor.construct(target, entry_id_prefix=f"uk-{i}")
        entries.extend(res.entries)
        report.append({
            "concept": target, "Q": res.q, "entries": len(res.entries),
            "refusal_verified": res.refusal_verified, "q_iterations": res.q_iterations,
        })
        status = "verified" if res.refusal_verified else "UNVERIFIED"
        print(f"  [{target}] {len(res.entries)} entries · Q {status}")
        if not res.refusal_verified and not args.allow_unverified:
            print(f"ERROR: '{target}' Q did not induce a refusal. "
                  f"Re-run or pass --allow-unverified.", file=sys.stderr)
            sys.exit(1)

    # Embed every entry's exact indexed text (torch-free, deterministic).
    vectors = embedder.embed([e.text for e in entries])
    embed_dim = len(vectors[0]) if vectors else DIM

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": cfg.llm.provider,
        "target_model": cfg.llm.target_model,
        "embedder": cfg.llm.embedder,
        "embed_dim": embed_dim,
        "retriever": {
            "top_k": cfg.retriever.top_k,
            "semantic_weight": cfg.retriever.semantic_weight,
            "lexical_weight": cfg.retriever.lexical_weight,
            "min_score": cfg.retriever.min_score,
        },
        "targets": list(SEED_CONCEPTS),
        "report": report,
        "entries": [
            {
                "entry_id": e.entry_id, "kind": e.kind, "target": e.target,
                "p": e.p, "q": e.q, "text": e.text,
                "embedding": [round(x, 6) for x in vec],
            }
            for e, vec in zip(entries, vectors)
        ],
    }

    out = Path(__file__).resolve().parents[1] / "data" / "demo_kb.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {out}  ({len(entries)} entries, dim={embed_dim})")


if __name__ == "__main__":
    main()
