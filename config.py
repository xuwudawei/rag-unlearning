"""Central configuration. No hardcoded magic values scattered across modules."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMConfig:
    """Target/helper/judge model settings. Frozen: config is immutable at runtime."""

    provider: str = os.getenv("UNLEARN_PROVIDER", "hf")  # "hf" | "deepseek" | "openai"
    # Local (hf) defaults: an ungated instruct model as LLM_un, matching the
    # paper's open-source track (they use Llama-2-7b-chat). Override via env.
    target_model: str = os.getenv("UNLEARN_TARGET_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    helper_model: str = os.getenv("UNLEARN_HELPER_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    judge_model: str = os.getenv("UNLEARN_JUDGE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    embed_model: str = os.getenv("UNLEARN_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    # Embedder implementation: "hashing" (numpy, torch-free, default) | "openai" | "st".
    embedder: str = os.getenv("UNLEARN_EMBEDDER", "hashing")
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    temperature: float = 0.0
    max_tokens: int = 384


@dataclass(frozen=True)
class RetrieverConfig:
    """Hybrid retriever: semantic (embedding) + lexical (BM25) fusion."""

    top_k: int = 1              # paper returns the single most relevant unlearned entry
    semantic_weight: float = 0.5
    lexical_weight: float = 0.5
    min_score: float = 0.0      # fused-score floor; below this nothing is injected
    min_lexical: float = 0.0    # absolute raw-BM25 floor: 0 disables; >0 gates injection
                                # to genuine lexical matches (used by the hosted demo)


@dataclass(frozen=True)
class ConstructConfig:
    """Unlearned-knowledge construction (k = P + Q)."""

    p_num_aspects: int = 5      # M: "describe [target] from M different aspects"
    q_max_words: int = 80       # V: word cap on the confidentiality clause
    q_max_refine_iters: int = 3 # iterate Q until the target model refuses


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    construct: ConstructConfig = field(default_factory=ConstructConfig)


_PROVIDER_DEFAULTS = {
    # provider -> (default target model, default embed model, api-key env var)
    "hf": ("Qwen/Qwen2.5-3B-Instruct", "sentence-transformers/all-MiniLM-L6-v2", None),
    "deepseek": ("deepseek-chat", "sentence-transformers/all-MiniLM-L6-v2", "DEEPSEEK_API_KEY"),
    "openai": ("gpt-4o", "text-embedding-3-small", "OPENAI_API_KEY"),
}


def load_config() -> Config:
    """Resolve provider-appropriate model ids and API key (env overrides win)."""
    provider = os.getenv("UNLEARN_PROVIDER", "hf").lower()
    target_default, embed_default, key_env = _PROVIDER_DEFAULTS.get(
        provider, _PROVIDER_DEFAULTS["hf"]
    )
    target = os.getenv("UNLEARN_TARGET_MODEL", target_default)
    llm = LLMConfig(
        provider=provider,
        target_model=target,
        helper_model=os.getenv("UNLEARN_HELPER_MODEL", target),
        judge_model=os.getenv("UNLEARN_JUDGE_MODEL", target),
        embed_model=os.getenv("UNLEARN_EMBED_MODEL", embed_default),
        embedder=os.getenv("UNLEARN_EMBEDDER", "hashing"),
        api_key=os.getenv(key_env, "") if key_env else "",
    )
    return Config(llm=llm)
