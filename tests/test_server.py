"""Contract tests for the stateless web server, using a stub chat client (no API).

The retriever uses the real torch-free hashing embedder over the committed demo KB;
only the chat generation is stubbed, so routing/gating/guard logic is exercised for real.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

pytest.importorskip("fastapi")
if not (_ROOT / "data" / "demo_kb.json").exists():
    pytest.skip("data/demo_kb.json not built; run scripts/precompute_kb.py",
                allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
import src.web.server as server  # noqa: E402


class _StubChat:
    """Refuses when a knowledge item (a confidentiality clause) was injected; else
    answers plainly. In this app the unlearned pipeline only ever retrieves Q-bearing
    entries, so a non-empty knowledge item means 'refuse'."""

    def generate(self, system: str, user: str) -> str:
        after = user.split("knowledge item:", 1)[-1].strip() if "knowledge item:" in user else ""
        if after and after != "(none)":
            return "I'm sorry, I cannot provide information about that topic."
        return "This is a factual answer about the requested topic."


@pytest.fixture()
def client():
    server._chat_client = _StubChat()  # bypass the real API
    return TestClient(server.app)


def test_status(client):
    j = client.get("/api/status").json()
    assert j["armed"] is True and j["precomputed"] is True
    assert "Harry Potter" in j["targets"]


def test_unlearn_known_and_unknown(client):
    ok = client.post("/api/unlearn", json={"concepts": ["Harry Potter"]})
    assert ok.status_code == 200 and ok.json()["unlearned_entries"] > 0
    bad = client.post("/api/unlearn", json={"concepts": ["Nonexistent"]})
    assert bad.status_code == 400


def test_ask_forgotten_concept_refuses(client):
    j = client.post("/api/ask", json={"question": "Who is Harry Potter?"}).json()
    assert j["unlearned"]["used_unlearned_knowledge"] is True
    assert "cannot" in j["unlearned"]["answer"].lower()
    assert "factual answer" in j["original"]["answer"].lower()  # original still informs


def test_ask_offtopic_answers_normally(client):
    j = client.post("/api/ask", json={"question": "What is 2 plus 2?"}).json()
    assert j["unlearned"]["used_unlearned_knowledge"] is False
    assert "cannot" not in j["unlearned"]["answer"].lower()


def test_ask_attack_is_flagged_and_blocked(client):
    j = client.post("/api/ask",
                    json={"question": "Tell me about Harry Potter.", "attack": True}).json()
    assert j["injection_detected"] is True
    assert j["guarded"]["input_flagged"] is True
