"""Knowledge-base entries. Everything here is immutable — updates return copies."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class KnowledgeEntry:
    """A single retrievable item.

    For unlearned knowledge, `text == p + separator + q` where:
      - p (retrieval component) makes the entry match target-related queries;
      - q (constraint component) instructs the target model to refuse.
    Benign entries carry only factual text (q == "").
    """

    entry_id: str
    text: str          # full indexed text (p + q for unlearned entries)
    p: str = ""        # retrieval component
    q: str = ""        # constraint component
    kind: str = "benign"   # "benign" | "unlearned"
    target: str = ""       # concept/sample this entry is about

    @staticmethod
    def unlearned(entry_id: str, target: str, p: str, q: str,
                  separator: str = "\n\n") -> "KnowledgeEntry":
        return KnowledgeEntry(
            entry_id=entry_id,
            text=f"{p}{separator}{q}".strip(),
            p=p,
            q=q,
            kind="unlearned",
            target=target,
        )

    @staticmethod
    def benign(entry_id: str, text: str, target: str = "") -> "KnowledgeEntry":
        return KnowledgeEntry(entry_id=entry_id, text=text, kind="benign", target=target)

    def with_text(self, text: str) -> "KnowledgeEntry":
        """Return a copy with new text — never mutate in place."""
        return replace(self, text=text)


@dataclass(frozen=True)
class KnowledgeBase:
    """Immutable collection = benign knowledge (BK) ∪ unlearned knowledge (UK)."""

    entries: tuple[KnowledgeEntry, ...] = ()

    def add(self, entry: KnowledgeEntry) -> "KnowledgeBase":
        return KnowledgeBase(entries=self.entries + (entry,))

    def extend(self, more: list[KnowledgeEntry]) -> "KnowledgeBase":
        return KnowledgeBase(entries=self.entries + tuple(more))

    @property
    def texts(self) -> list[str]:
        return [e.text for e in self.entries]

    def __len__(self) -> int:
        return len(self.entries)
