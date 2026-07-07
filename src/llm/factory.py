"""Provider selection. Keeps construction of clients in one place."""
from __future__ import annotations

from .base import LLMError


def build_client(cfg):
    provider = cfg.provider.lower()
    if provider == "hf":
        from .hf_client import HFClient  # lazy: heavy torch import
        return HFClient(cfg)
    if provider == "deepseek":
        from .deepseek_client import DeepSeekClient
        return DeepSeekClient(cfg)
    if provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(cfg)
    if provider == "mock":
        from .mock_client import MockClient  # tests only, not a real backend
        return MockClient(cfg)
    raise LLMError(f"Unknown provider '{cfg.provider}'. Use 'hf', 'openai', or 'mock'.")


def helper_generate(client, model: str, system: str, user: str) -> str:
    """Call a specific helper/judge model, regardless of provider."""
    fn = getattr(client, "generate_with", None)
    if callable(fn):
        return fn(model, system, user)
    # Fallback: provider has no per-model override; use default generate.
    return client.generate(system, user)
