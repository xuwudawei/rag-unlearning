"""The retrievable 'source surface' about an entity.

Each Source is a document a grounded LLM could retrieve and use. `authority` and
`owned` drive the intervention cost model later (editing your own site is cheap;
correcting a high-authority third party is expensive).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..rag.knowledge import KnowledgeBase, KnowledgeEntry


@dataclass(frozen=True)
class Source:
    sid: str
    text: str
    authority: float   # 0..1, how much a grounded model trusts it
    owned: bool        # can the client edit/publish it cheaply?
    kind: str          # blog | aggregator | news | official


def to_kb(sources: list[Source], entity: str) -> KnowledgeBase:
    return KnowledgeBase(entries=tuple(
        KnowledgeEntry.benign(entry_id=s.sid, text=s.text, target=entity)
        for s in sources))


# --- A worked example: entity with a false/outdated claim circulating ----
EXAMPLE_ENTITY = "Acme Health"
EXAMPLE_CLAIM = "Acme Health was fined for a 2023 data breach"

EXAMPLE_SOURCES: list[Source] = [
    Source("blog1",
           "Rumor mill: Acme Health was reportedly fined after a 2023 data breach "
           "exposed patient records, according to unverified posts.",
           authority=0.2, owned=False, kind="blog"),
    Source("aggregator1",
           "Acme Health — company profile. Notes: allegedly fined over a 2023 data "
           "breach incident. Founded 2015. Healthcare software provider.",
           authority=0.3, owned=False, kind="aggregator"),
    Source("news1",
           "Acme Health launched a new telehealth platform in 2024, expanding its "
           "patient-scheduling products across three regions.",
           authority=0.6, owned=False, kind="news"),
    Source("official1",
           "Acme Health is a healthcare software provider founded in 2015, offering "
           "scheduling and telehealth tools to clinics.",
           authority=0.9, owned=True, kind="official"),
]

# A legitimate corrective document the client could publish on its own site.
EXAMPLE_CORRECTION = Source(
    "correction1",
    "Official statement: Acme Health has never been fined for a data breach. The "
    "2023 incident sometimes cited refers to a different, unrelated company. Acme "
    "Health maintains SOC 2 and HIPAA compliance with no regulatory penalties.",
    authority=0.9, owned=True, kind="official")
