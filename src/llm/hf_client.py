"""Real local LLM backend (HuggingFace transformers) — no API key, no mock.

Serves as the target model LLM_un (and, in the closed-source-free setting, also the
helper LLM_cons and the judge). Runs on Apple Silicon MPS / CUDA / CPU. Exposes
per-token logprobs so the Min-K% membership-inference metric actually runs — the
paper's open-source track uses exactly this kind of white-box-scored open model.

Embeddings come from a sentence-transformers model for genuine semantic retrieval.
"""
from __future__ import annotations

from typing import Sequence

from .base import LLMError


def _pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class HFClient:
    def __init__(self, cfg) -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._cfg = cfg
        self._torch = torch
        self._device = _pick_device()
        model_id = cfg.target_model

        try:
            self._tok = AutoTokenizer.from_pretrained(model_id)
            dtype = torch.float16 if self._device != "cpu" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype
            ).to(self._device).eval()
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Failed to load HF model '{model_id}': {exc}") from exc

        # Lightweight, real semantic embedder for the hybrid retriever.
        self._embedder = SentenceTransformer(cfg.embed_model, device=self._device)

    # --- generation ------------------------------------------------------
    def _run_chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        inputs = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self._device)
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = dict(
            max_new_tokens=self._cfg.max_tokens,
            do_sample=self._cfg.temperature > 0,
            pad_token_id=self._tok.eos_token_id,
        )
        if self._cfg.temperature > 0:
            gen_kwargs["temperature"] = self._cfg.temperature
        with self._torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        text = self._tok.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()

    def generate(self, system: str, user: str) -> str:
        return self._run_chat(system, user)

    def generate_with(self, model: str, system: str, user: str) -> str:
        # Single local model plays every role in the key-free setting.
        return self._run_chat(system, user)

    # --- embeddings ------------------------------------------------------
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._embedder.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    # --- per-token logprobs (enables Min-K% MIA) -------------------------
    def token_logprobs(self, text: str) -> list[float]:
        """log p(token_i | token_<i) for each token in `text` under the model."""
        if not text.strip():
            raise ValueError("Cannot score empty text.")
        ids = self._tok(text, return_tensors="pt").input_ids.to(self._device)
        if ids.shape[1] < 2:
            raise ValueError("Text too short to score.")
        with self._torch.no_grad():
            logits = self._model(ids).logits
        log_probs = self._torch.log_softmax(logits[0, :-1], dim=-1)
        targets = ids[0, 1:]
        picked = log_probs[range(targets.shape[0]), targets]
        return picked.float().cpu().tolist()
