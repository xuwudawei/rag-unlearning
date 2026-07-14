"""Real local LLM backend (HuggingFace transformers) — no API key required.

Serves as the target model LLM_un (and, in the closed-source-free setting, also the
helper LLM_cons and the judge). Runs on Apple Silicon MPS / CUDA / CPU. Exposes
per-token logprobs so the Min-K% membership-inference metric actually runs — the
paper's open-source track uses exactly this kind of white-box-scored open model.

Embeddings come from a sentence-transformers model for genuine semantic retrieval.
"""
from __future__ import annotations

from typing import Sequence

from .base import LLMError
from .embedders import build_embedder


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

        # Pluggable embedder (default hashing; set UNLEARN_EMBEDDER=st for MiniLM).
        self._embedder = build_embedder(cfg, device=self._device)

    # --- generation ------------------------------------------------------
    def _run_chat(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Some model mirrors (e.g. NousResearch/Llama-2-*) ship no chat_template;
        # fall back to the Llama-2 [INST] format so generation still works faithfully.
        if getattr(self._tok, "chat_template", None):
            inputs = self._tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(self._device)
        else:
            prompt = (f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
                      if system else f"<s>[INST] {user} [/INST]")
            inputs = self._tok(prompt, return_tensors="pt").to(self._device)
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
        return self._embedder.embed(texts)

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

    def token_logprobs_with_context(self, context: str, text: str) -> list[float]:
        """log p(text tokens | context): logprobs for text's tokens given a prefix.

        The RAG-unlearning MIA effect: prepending the retrieved confidentiality clause
        lowers the model's likelihood on a memorised forget text, pushing its Min-K%
        score toward the non-member distribution."""
        ctx_ids = self._tok(context, return_tensors="pt").input_ids.to(self._device)
        full_ids = self._tok(context + "\n" + text, return_tensors="pt").input_ids.to(self._device)
        n_ctx = ctx_ids.shape[1]
        if full_ids.shape[1] <= n_ctx + 1:
            raise ValueError("Text adds too few tokens after context to score.")
        with self._torch.no_grad():
            logits = self._model(full_ids).logits
        log_probs = self._torch.log_softmax(logits[0, :-1], dim=-1)
        targets = full_ids[0, 1:]
        picked = log_probs[range(targets.shape[0]), targets]
        return picked[n_ctx - 1:].float().cpu().tolist()  # text-token positions only
