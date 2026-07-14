"""Real datasets for the faithful reproduction (not a seed list).

- CONCEPT_TOPICS: 100 real Wikipedia entities across fiction / technology /
  celebrities / landmarks, matching the paper's concept-unlearning setup
  ("100 topics ... fiction, technology, and celebrities", 5 questions each = 500).
- load_tiny_nq(): the paper's "Tiny-nq" = 2,000 Natural-Questions QA pairs, fetched
  from the Hugging Face datasets-server over HTTP (no extra package needed).
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

# --- 100 real Wikipedia concept topics (fiction / tech / celebrities / landmarks) ---
CONCEPT_TOPICS: tuple[str, ...] = (
    # Fiction (30)
    "Harry Potter", "The Lord of the Rings", "Game of Thrones", "Star Wars",
    "Sherlock Holmes", "Batman", "Spider-Man", "The Great Gatsby", "Pride and Prejudice",
    "Moby-Dick", "Dracula", "Frankenstein", "The Hobbit", "The Chronicles of Narnia",
    "Percy Jackson", "The Hunger Games", "Twilight", "Dune", "The Witcher", "Naruto",
    "One Piece", "Pokemon", "The Legend of Zelda", "Super Mario", "Wonder Woman",
    "Iron Man", "The Matrix", "Jurassic Park", "Breaking Bad", "The Simpsons",
    # Technology (25)
    "Bitcoin", "Ethereum", "Blockchain", "Linux", "Python (programming language)",
    "JavaScript", "Docker (software)", "Kubernetes", "TensorFlow", "iPhone",
    "Android (operating system)", "Tesla Model S", "Quantum computing", "5G",
    "CRISPR", "Cloud computing", "Machine learning", "Artificial intelligence",
    "Electric vehicle", "Solar panel", "GPS", "Wi-Fi", "USB", "Bluetooth", "GPU",
    # Celebrities / people (30)
    "Elon Musk", "Bill Gates", "Steve Jobs", "Taylor Swift", "Cristiano Ronaldo",
    "Lionel Messi", "Barack Obama", "Albert Einstein", "Leonardo da Vinci",
    "Marie Curie", "Beyonce", "LeBron James", "Oprah Winfrey", "Mark Zuckerberg",
    "Jeff Bezos", "Serena Williams", "Stephen Hawking", "Nikola Tesla", "Walt Disney",
    "Michael Jackson", "Ada Lovelace", "Isaac Newton", "Charles Darwin", "Mahatma Gandhi",
    "Nelson Mandela", "William Shakespeare", "Vincent van Gogh", "Frida Kahlo",
    "Ludwig van Beethoven", "Wolfgang Amadeus Mozart",
    # Landmarks / places (15)
    "Eiffel Tower", "Great Wall of China", "Statue of Liberty", "Taj Mahal",
    "Colosseum", "Mount Everest", "Amazon rainforest", "Grand Canyon",
    "Big Ben", "Sydney Opera House", "Machu Picchu", "Stonehenge",
    "Golden Gate Bridge", "Niagara Falls", "Mount Fuji",
)

_ROWS_API = "https://datasets-server.huggingface.co/rows"


def load_tiny_nq(n: int = 2000, dataset: str = "google-research-datasets/nq_open",
                 config: str = "nq_open", split: str = "train") -> list[dict]:
    """Fetch n Natural-Questions QA pairs. Returns [{'question','answer'}].

    Uses the HF datasets-server rows API (HTTP, paginated ≤100/request). The paper's
    sample-unlearning track fine-tunes on these and tests prefix->suffix regurgitation.
    """
    out: list[dict] = []
    offset = 0
    while len(out) < n:
        length = min(100, n - len(out))
        url = (f"{_ROWS_API}?dataset={urllib.parse.quote(dataset)}"
               f"&config={config}&split={split}&offset={offset}&length={length}")
        req = urllib.request.Request(url, headers={"User-Agent": "rag-unlearning-repro"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
        rows = payload.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            ans = row.get("answer")
            answer = ans[0] if isinstance(ans, list) and ans else (ans or "")
            out.append({"question": row["question"], "answer": answer})
        offset += len(rows)
    if not out:
        raise RuntimeError(
            f"Could not fetch NQ rows from the datasets-server for '{dataset}'. "
            f"Check network or the dataset/config names."
        )
    return out[:n]
