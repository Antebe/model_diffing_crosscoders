"""
autointerp_local.py — local Gemma-2-9B-it backend for the autointerp pipeline.

Defines ``LocalGemmaClient`` with the same async ``call(system, user) -> str``
surface as ``autointerp.AsyncClaude``, and ``LocalAutoInterpPipeline`` which
swaps the Anthropic-API client for the local one. Everything else
(``TopKFinder``, ``TextStore``, ``AsyncExplainer``, ``AsyncDetectionScorer``,
``ResultsStore``, prompts) is reused from ``autointerp``.

Why mock async on a synchronous local backend: the existing pipeline's
explainer / scorer call ``await self.claude.call(...)``. To avoid forking
the call sites, ``LocalGemmaClient.call`` runs the synchronous Gemma
generation inside ``asyncio.to_thread``. Concurrency is gated by an
``asyncio.Semaphore`` of size 1 by default — local single-GPU generation
is the bottleneck and parallel calls would just compete for VRAM.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from autointerp import (
    AsyncDetectionScorer,
    AsyncExplainer,
    AutoInterpPipeline,
    ResultsStore,
    TextStore,
    TopKFinder,
)

log = logging.getLogger("autointerp_local")


# ──────────────────────────────────────────────────────────────────────────────
# Local Gemma client
# ──────────────────────────────────────────────────────────────────────────────

class LocalGemmaClient:
    """Mimics ``autointerp.AsyncClaude``: async ``.call(system, user) -> str``.

    Loads the model once on init; subsequent ``.call`` invocations run the
    generation in a background thread via ``asyncio.to_thread`` so that the
    surrounding async pipeline keeps its existing structure.

    Args:
        model_id:        HuggingFace repo id, default ``google/gemma-2-9b-it``.
        device:          Torch device.
        max_new_tokens:  Generation cap per call.
        temperature:     Sampling temperature; ``<= 0`` ⇒ greedy.
        max_concurrent:  Semaphore size; default 1 (single GPU == single job).
    """

    def __init__(
        self,
        model_id: str = "google/gemma-2-9b-it",
        device: str = "cuda:0",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        max_concurrent: int = 1,
    ):
        log.info(f"[local-gemma] loading {model_id} on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._sem = asyncio.Semaphore(max_concurrent)
        self._gen_lock = threading.Lock()      # serialise GPU access from threads
        self.n_calls = 0
        self.n_errors = 0
        log.info("[local-gemma] ready.")

    # — sync core —
    @torch.no_grad()
    def _generate_sync(self, system: str, user: str) -> str:
        """Run one Gemma chat-template completion."""
        # gemma-2 chat template uses <start_of_turn>user / <start_of_turn>model.
        # We fold the system prompt into the user turn (Gemma has no system role).
        merged_user = f"{system.strip()}\n\n{user.strip()}"
        messages = [{"role": "user", "content": merged_user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096,
        ).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if self.temperature > 0:
            gen_kwargs["temperature"] = float(self.temperature)
        with self._gen_lock:
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    # — async surface compatible with AsyncClaude —
    async def call(self, system: str, user: str) -> str:
        async with self._sem:
            try:
                text = await asyncio.to_thread(self._generate_sync, system, user)
                self.n_calls += 1
                return text
            except Exception as e:                          # noqa: BLE001
                self.n_errors += 1
                log.error(f"[local-gemma] generate failed: {e}")
                return ""


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline subclass — swaps the client
# ──────────────────────────────────────────────────────────────────────────────

class LocalAutoInterpPipeline(AutoInterpPipeline):
    """``AutoInterpPipeline`` with ``LocalGemmaClient`` instead of ``AsyncClaude``.

    Mirrors the parent's ``__init__`` signature minus the Claude-specific
    knobs (model name → ``gemma_model_id``).
    """

    def __init__(
        self,
        feat_cache_dir: str,
        source_cache_dir: str,
        results_dir: str,
        gemma_model_id: str = "google/gemma-2-9b-it",
        gemma_device: str = "cuda:0",
        top_k: int = 10,
        n_random_for_detection: int = 10,
        max_concurrent: int = 1,
        dead_feature_threshold: int = 0,
        interpretability_threshold: float = 0.8,
        feature_subset: Optional[list[int]] = None,
    ):
        # Replicate parent init body but inject LocalGemmaClient.
        self.feat_cache_dir = feat_cache_dir
        self.source_cache_dir = source_cache_dir
        self.top_k = top_k
        self.n_random = n_random_for_detection

        self.claude = LocalGemmaClient(
            model_id=gemma_model_id,
            device=gemma_device,
            max_concurrent=max_concurrent,
        )
        self.explainer = AsyncExplainer(self.claude, top_k=top_k)
        self.scorer = AsyncDetectionScorer(
            self.claude, n_match=top_k, n_nomatch=n_random_for_detection,
        )
        self.store = ResultsStore(results_dir, interpretability_threshold)

        self.feature_subset = feature_subset
        self.dead_threshold = dead_feature_threshold

        self.topk_results: dict = {}
        self.text_store: TextStore | None = None
        self._all_global_indices: list[int] = []


# Re-export the helper classes so callers can pull everything from one module.
__all__ = [
    "LocalGemmaClient",
    "LocalAutoInterpPipeline",
    "TopKFinder",
    "TextStore",
    "ResultsStore",
]
