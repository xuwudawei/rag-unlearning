"""Provider selection. Keeps construction of clients in one place."""
from __future__ import annotations

from .base import LLMError


def build_client(cfg):
    provider = cfg.provider.lower()
    try:
        if provider == "hf":
            from .hf_client import HFClient  # lazy: heavy torch import
            return HFClient(cfg)
        if provider == "deepseek":
            from .deepseek_client import DeepSeekClient
            return DeepSeekClient(cfg)
        if provider == "openai":
            from .openai_client import OpenAIClient
            return OpenAIClient(cfg)
        if provider == "openrouter":
            from .openrouter_client import OpenRouterClient
            return OpenRouterClient(cfg)
    except ImportError as exc:
        raise LLMError(
            f"Provider '{provider}' needs the local research stack (torch/transformers). "
            f"Install it with `pip install -r requirements-dev.txt`, or use a torch-free "
            f"provider (deepseek/openai) with UNLEARN_EMBEDDER=hashing. Underlying: {exc}"
        ) from exc
    raise LLMError(f"Unknown provider '{cfg.provider}'. Use 'hf', 'deepseek', or 'openai'.")


def helper_generate(client, model: str, system: str, user: str) -> str:
    """Call a specific helper/judge model, regardless of provider."""
    fn = getattr(client, "generate_with", None)
    if callable(fn):
        return fn(model, system, user)
    # Fallback: provider has no per-model override; use default generate.
    return client.generate(system, user)
