"""
autointerp.py
─────────────
Core library for DFC feature automated interpretability.

Pipeline:
  1. TopKFinder     — stream feature shards, build per-feature top-K activation index
  2. TextStore      — map (shard_idx, row) back to source text
  3. AsyncExplainer — call Claude to generate feature explanations
  4. DetectionScorer— call Claude to classify examples against explanation (detection score)
  5. ResultsStore   — structured JSON output with per-feature resume support

Designed to match the paper's methodology (Section B.4.1):
  - Top-10 max-activating examples per feature
  - Detection score: LLM classifies 10 max-activating + 10 random examples
  - Interpretability threshold: detection_score >= 0.8
  - Harmful content flagging inline with explanation
"""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

log = logging.getLogger("autointerp")


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOPK FINDER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActivationExample:
    """One activating example for a feature."""
    global_idx: int          # absolute row index across all shards
    shard_idx: int           # which shard file
    row_in_shard: int        # row within that shard
    activation: float        # feature activation value


class TopKFinder:
    """
    Streams feature shards and maintains a min-heap of size K
    per feature — finds the global top-K activating examples
    without holding all shards in memory simultaneously.

    Usage:
        finder = TopKFinder(feat_cache_dir, k=10)
        finder.run()
        topk = finder.results   # dict[feature_idx -> list[ActivationExample]]
    """

    def __init__(self, feat_cache_dir: str, k: int = 10):
        self.cache_dir = Path(feat_cache_dir)
        self.k = k

        meta = json.loads((self.cache_dir / "meta.json").read_text())
        self.shards      = meta["shards"]
        self.dict_size   = meta["dict_size"]
        self.n_a         = meta["n_a"]
        self.n_b         = meta["n_b"]
        self.n_shared    = meta["n_shared"]
        self.a_end       = self.n_a
        self.b_end       = self.n_a + self.n_b

        # min-heap per feature: list of (activation, global_idx, shard_idx, row)
        # Stored as (neg_activation, ...) so heapq (min-heap) gives us a max-heap
        self._heaps: list[list] = [[] for _ in range(self.dict_size)]
        self.results: dict[int, list[ActivationExample]] = {}

    def partition_of(self, feat_idx: int) -> str:
        if feat_idx < self.a_end:
            return "a_excl"
        if feat_idx < self.b_end:
            return "b_excl"
        return "shared"

    def run(self, progress: bool = True) -> None:
        global_offset = 0
        total_shards = len(self.shards)

        for shard_num, fname in enumerate(tqdm(self.shards, desc="Processing shards", disable=not progress)):
            shard_path = self.cache_dir / fname
            log.info(f"  TopK: shard {shard_num + 1}/{total_shards}  ({fname})")

            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            # shard: (shard_size, dict_size) float16
            shard_f32 = shard.float()                        # work in float32
            shard_size = shard_f32.shape[0]

            # Transpose → (dict_size, shard_size) so we iterate features
            # For large dict_size (131K) this is a big transpose; do it once
            feat_acts = shard_f32.T                          # (dict_size, shard_size)

            for feat_idx in range(self.dict_size):
                acts = feat_acts[feat_idx]                   # (shard_size,)
                active_mask = acts > 0
                if not active_mask.any():
                    continue

                active_indices = active_mask.nonzero(as_tuple=True)[0]
                heap = self._heaps[feat_idx]

                for row in active_indices.tolist():
                    val = acts[row].item()
                    global_idx = global_offset + row
                    entry = (val, global_idx, shard_num, row)

                    if len(heap) < self.k:
                        heapq.heappush(heap, entry)
                    elif val > heap[0][0]:
                        heapq.heapreplace(heap, entry)

            global_offset += shard_size
            del shard, shard_f32, feat_acts

        # Convert heaps → sorted lists (highest activation first)
        log.info("  TopK: sorting results...")
        for feat_idx, heap in enumerate(self._heaps):
            if heap:
                sorted_entries = sorted(heap, reverse=True)
                self.results[feat_idx] = [
                    ActivationExample(
                        global_idx=e[1],
                        shard_idx=e[2],
                        row_in_shard=e[3],
                        activation=e[0],
                    )
                    for e in sorted_entries
                ]

        active_feats = sum(1 for h in self._heaps if h)
        log.info(f"  TopK: done. {active_feats:,} / {self.dict_size:,} features active.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TEXT STORE
# ══════════════════════════════════════════════════════════════════════════════

class TextStore:
    """
    Maps (shard_idx, row_in_shard) → source text.

    Expects the LLM activation cache to have a parallel text store:
      source_cache/
        texts/
          shard_00000.jsonl    ← one JSON object per line: {"text": "..."}
          shard_00001.jsonl
          ...

    Falls back to a flat texts.jsonl at the cache root.
    If neither exists, returns a placeholder string so the rest of the
    pipeline can still run (explanations will be low quality but the
    pipeline won't crash).
    """

    def __init__(self, source_cache_dir: str):
        self.cache_dir = Path(source_cache_dir)
        self._shard_cache: dict[int, list[str]] = {}

    def _load_shard(self, shard_idx: int) -> list[str]:
        if shard_idx in self._shard_cache:
            return self._shard_cache[shard_idx]

        fname = f"shard_{shard_idx:05d}.jsonl"

        # Try texts/ subdirectory first
        candidates = [
            self.cache_dir / "texts" / fname,
            self.cache_dir / fname,
        ]

        for path in candidates:
            if path.exists():
                texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line.strip()]
                self._shard_cache[shard_idx] = texts
                return texts

        log.warning(f"  TextStore: no text shard found for shard {shard_idx}; using placeholders.")
        placeholder = [f"[text unavailable — shard {shard_idx}]"] * 100_000
        self._shard_cache[shard_idx] = placeholder
        return placeholder

    def get(self, shard_idx: int, row_in_shard: int) -> str:
        texts = self._load_shard(shard_idx)
        if row_in_shard < len(texts):
            return texts[row_in_shard]
        return f"[row {row_in_shard} out of range in shard {shard_idx}]"

    def get_batch(self, examples: list[ActivationExample]) -> list[str]:
        return [self.get(e.shard_idx, e.row_in_shard) for e in examples]


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

EXPLAIN_SYSTEM = """\
You are a meticulous AI researcher analyzing patterns in neural network features.
You will be shown text examples that strongly activate a specific feature in a \
language model. Your task is to identify what concept, pattern, or topic the \
feature represents.

Guidelines:
- Focus on patterns COMMON across examples, not one-off details.
- Be concise: 1–2 sentences maximum.
- Describe the feature in terms of linguistic or semantic content.
- Do NOT mention activation strengths or technical details.
- After your explanation, assess whether this feature relates to harmful content.

Your response MUST end with exactly these three lines:
[EXPLANATION]: <your concise explanation>
[HARMFUL]: YES or NO
[HARMFUL_REASON]: <1–2 sentence rationale>"""


EXPLAIN_USER_TEMPLATE = """\
The following {n} text samples most strongly activate this feature \
(activation strength in parentheses, highest first):

{examples}

What does this feature represent?"""


DETECT_SYSTEM = """\
You are evaluating how well a feature explanation describes text examples.
You will be given:
  1. A feature explanation
  2. A list of text samples (some match the explanation, some are random)

For each sample, reply with MATCH or NO_MATCH on a separate line, in order.
Nothing else — just {n} lines of MATCH or NO_MATCH."""


DETECT_USER_TEMPLATE = """\
Feature explanation: {explanation}

Text samples (evaluate each in order):
{examples}"""


FLAG_CATEGORIES = [
    "discrimination, bias, or unfairness",
    "racism, sexism, or other forms of prejudice",
    "censorship or information filtering",
    "misalignment with human values",
    "deception or manipulation",
    "jailbreaking or security vulnerabilities",
    "medical misinformation or harmful advice",
    "any other concerning behavioral patterns",
]


# ══════════════════════════════════════════════════════════════════════════════
# 4. ASYNC CLAUDE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class AsyncClaude:
    """
    Thin async wrapper around the Anthropic API with:
      - Configurable concurrency limit (semaphore)
      - Exponential backoff on rate-limit / server errors
      - Per-call timeout
    """

    def __init__(
        self,
        model: str = "claude-opus-4-20250514",
        max_tokens: int = 512,
        max_concurrent: int = 20,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        try:
            import anthropic
            api_key = ""
            self._client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("pip install anthropic")

        self.model        = model
        self.max_tokens   = max_tokens
        self._sem         = asyncio.Semaphore(max_concurrent)
        self.max_retries  = max_retries
        self.base_delay   = base_delay
        self.n_calls      = 0
        self.n_errors     = 0

    async def call(self, system: str, user: str) -> str:
        delay = self.base_delay
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    self.n_calls += 1
                    return resp.content[0].text
            except Exception as e:
                self.n_errors += 1
                if attempt == self.max_retries - 1:
                    log.error(f"  Claude API failed after {self.max_retries} attempts: {e}")
                    return ""
                log.warning(f"  API error (attempt {attempt + 1}): {e}  — retry in {delay:.1f}s")
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60.0)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 5. EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExplanationResult:
    explanation: str
    harmful: bool
    harmful_reason: str
    raw_response: str


def _parse_explanation(raw: str) -> ExplanationResult:
    """Extract structured fields from Claude's explanation response."""
    explanation  = ""
    harmful      = False
    reason       = ""

    for line in raw.splitlines():
        if line.startswith("[EXPLANATION]:"):
            explanation = line[len("[EXPLANATION]:"):].strip()
        elif line.startswith("[HARMFUL]:"):
            val = line[len("[HARMFUL]:"):].strip().upper()
            harmful = val == "YES"
        elif line.startswith("[HARMFUL_REASON]:"):
            reason = line[len("[HARMFUL_REASON]:"):].strip()

    return ExplanationResult(
        explanation=explanation or raw[:200],
        harmful=harmful,
        harmful_reason=reason,
        raw_response=raw,
    )


class AsyncExplainer:
    def __init__(self, claude: AsyncClaude, top_k: int = 15, max_text_chars: int = 400):
        self.claude        = claude
        self.top_k         = top_k
        self.max_text_chars = max_text_chars

    def _format_examples(self, texts: list[str], activations: list[float]) -> str:
        lines = []
        for i, (txt, act) in enumerate(zip(texts, activations), 1):
            truncated = txt[:self.max_text_chars].replace("\n", " ").strip()
            lines.append(f"  {i}. (act={act:.3f})  {truncated}")
        return "\n".join(lines)

    async def explain(
        self,
        feat_idx: int,
        texts: list[str],
        activations: list[float],
    ) -> ExplanationResult:
        example_str = self._format_examples(texts, activations)
        user_msg = EXPLAIN_USER_TEMPLATE.format(
            n=len(texts),
            examples=example_str,
        )
        raw = await self.claude.call(EXPLAIN_SYSTEM, user_msg)
        return _parse_explanation(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DETECTION SCORER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionResult:
    score: float           # fraction correct (0.0 – 1.0)
    n_correct: int
    n_total: int
    raw_response: str


def _parse_detection(raw: str, n_match: int, n_nomatch: int) -> DetectionResult:
    """
    Parse MATCH / NO_MATCH lines from Claude's detection response.
    Labels: first n_match items are true positives, rest are true negatives.
    """
    lines = [l.strip().upper() for l in raw.splitlines() if l.strip() in ("MATCH", "NO_MATCH")]
    n_total = n_match + n_nomatch

    if len(lines) != n_total:
        log.debug(f"  Detection parse: expected {n_total} lines, got {len(lines)}")
        # Graceful degradation: score 0.5 (random chance)
        return DetectionResult(score=0.5, n_correct=0, n_total=n_total, raw_response=raw)

    true_labels = ["MATCH"] * n_match + ["NO_MATCH"] * n_nomatch
    n_correct = sum(1 for pred, true in zip(lines, true_labels) if pred == true)

    return DetectionResult(
        score=round(n_correct / n_total, 4),
        n_correct=n_correct,
        n_total=n_total,
        raw_response=raw,
    )


class AsyncDetectionScorer:
    """
    Presents the explanation + (n_max activating + n_random) texts to Claude
    and asks it to classify which match the explanation.
    Paper uses 10 + 10 = 20 examples; detection score = accuracy.
    """

    def __init__(
        self,
        claude: AsyncClaude,
        n_match: int = 10,
        n_nomatch: int = 10,
        max_text_chars: int = 300,
    ):
        self.claude        = claude
        self.n_match       = n_match
        self.n_nomatch     = n_nomatch
        self.max_text_chars = max_text_chars

    async def score(
        self,
        explanation: str,
        match_texts: list[str],
        random_texts: list[str],
    ) -> DetectionResult:
        # Interleave randomly so position doesn't give it away
        labeled = (
            [(t, True)  for t in match_texts[:self.n_match]] +
            [(t, False) for t in random_texts[:self.n_nomatch]]
        )
        random.shuffle(labeled)
        shuffled_texts, true_labels = zip(*labeled) if labeled else ([], [])

        n_match_actual  = sum(1 for _, is_match in labeled if is_match)
        n_nomatch_actual = len(labeled) - n_match_actual

        examples_str = "\n".join(
            f"  {i + 1}. {t[:self.max_text_chars].replace(chr(10), ' ').strip()}"
            for i, (t, _) in enumerate(labeled)
        )
        system = DETECT_SYSTEM.format(n=len(labeled))
        user   = DETECT_USER_TEMPLATE.format(
            explanation=explanation,
            examples=examples_str,
        )
        raw = await self.claude.call(system, user)

        # Re-derive correct answers from shuffle order
        true_order = ["MATCH" if is_match else "NO_MATCH" for _, is_match in labeled]
        lines = [l.strip().upper() for l in raw.splitlines() if l.strip() in ("MATCH", "NO_MATCH")]
        n_total = len(labeled)

        if len(lines) != n_total:
            return DetectionResult(score=0.5, n_correct=0, n_total=n_total, raw_response=raw)

        n_correct = sum(1 for pred, true in zip(lines, true_order) if pred == true)
        return DetectionResult(
            score=round(n_correct / n_total, 4),
            n_correct=n_correct,
            n_total=n_total,
            raw_response=raw,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS STORE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureRecord:
    feat_idx: int
    partition: str                    # "a_excl" | "b_excl" | "shared"
    is_dead: bool

    # TopK examples (populated after TopKFinder)
    top_activations: list[float]      = field(default_factory=list)
    top_texts: list[str]              = field(default_factory=list)

    # Explanation (populated after AsyncExplainer)
    explanation: str                  = ""
    harmful: bool                     = False
    harmful_reason: str               = ""

    # Detection score (populated after AsyncDetectionScorer)
    detection_score: float            = -1.0
    detection_n_correct: int          = -1
    detection_n_total: int            = -1

    # Meta
    interpretable: bool               = False   # detection_score >= threshold
    error: str                        = ""


class ResultsStore:
    """
    Persists one JSON file per feature under results_dir/.
    Supports resume: skips features whose file already exists.
    """

    def __init__(self, results_dir: str, interpretability_threshold: float = 0.8):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = interpretability_threshold

    def path_for(self, feat_idx: int) -> Path:
        # Bucket into subdirectories of 1000 to avoid giant flat directories
        bucket = (feat_idx // 1000) * 1000
        subdir = self.results_dir / f"{bucket:07d}"
        subdir.mkdir(exist_ok=True)
        return subdir / f"feat_{feat_idx:07d}.json"

    def exists(self, feat_idx: int) -> bool:
        return self.path_for(feat_idx).exists()

    def save(self, rec: FeatureRecord) -> None:
        rec.interpretable = rec.detection_score >= self.threshold
        data = asdict(rec)
        self.path_for(rec.feat_idx).write_text(json.dumps(data, indent=2))

    def load(self, feat_idx: int) -> Optional[FeatureRecord]:
        p = self.path_for(feat_idx)
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return FeatureRecord(**data)

    def summary(self) -> dict:
        """Aggregate stats over all saved records."""
        records = list(self.results_dir.rglob("feat_*.json"))
        total        = len(records)
        dead         = 0
        interpretable = 0
        harmful       = 0
        by_partition: dict[str, dict] = {
            "a_excl": {"total": 0, "interpretable": 0, "harmful": 0},
            "b_excl": {"total": 0, "interpretable": 0, "harmful": 0},
            "shared": {"total": 0, "interpretable": 0, "harmful": 0},
        }

        for p in records:
            try:
                d = json.loads(p.read_text())
                part = d.get("partition", "shared")
                is_dead = d.get("is_dead", False)
                interp  = d.get("interpretable", False)
                harm    = d.get("harmful", False)

                if is_dead:
                    dead += 1
                if interp:
                    interpretable += 1
                if harm:
                    harmful += 1

                if part in by_partition:
                    by_partition[part]["total"] += 1
                    if interp:
                        by_partition[part]["interpretable"] += 1
                    if harm:
                        by_partition[part]["harmful"] += 1

            except Exception:
                pass

        return {
            "total_processed": total,
            "dead":            dead,
            "interpretable":   interpretable,
            "harmful":         harmful,
            "pct_interpretable": round(100 * interpretable / max(total, 1), 1),
            "pct_harmful":       round(100 * harmful / max(total, 1), 1),
            "by_partition":    by_partition,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 8. PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class AutoInterpPipeline:
    """
    Orchestrates the full autointerp flow for one feature cache.

    Steps per feature:
      1. Look up top-K activating texts (from TopKFinder results)
      2. Call AsyncExplainer  → explanation + harmful flag
      3. Call AsyncDetectionScorer → detection_score
      4. Save FeatureRecord to ResultsStore

    Features are processed in concurrent async tasks bounded by
    claude.max_concurrent (default 20).
    """

    def __init__(
        self,
        feat_cache_dir: str,
        source_cache_dir: str,
        results_dir: str,
        claude_model: str = "claude-opus-4-20250514",
        top_k: int = 10,
        n_random_for_detection: int = 10,
        max_concurrent: int = 1,  # Sequential processing
        dead_feature_threshold: int = 0,    # 0 = any activation counts
        interpretability_threshold: float = 0.8,
        feature_subset: Optional[list[int]] = None,  # None = all features
    ):
        self.feat_cache_dir  = feat_cache_dir
        self.source_cache_dir = source_cache_dir
        self.top_k           = top_k
        self.n_random        = n_random_for_detection

        self.claude   = AsyncClaude(model=claude_model, max_concurrent=max_concurrent)
        self.explainer = AsyncExplainer(self.claude, top_k=top_k)
        self.scorer    = AsyncDetectionScorer(self.claude, n_match=top_k, n_nomatch=n_random_for_detection)
        self.store     = ResultsStore(results_dir, interpretability_threshold)

        self.feature_subset = feature_subset
        self.dead_threshold = dead_feature_threshold

        # Will be populated after run_topk()
        self.topk_results: dict[int, list[ActivationExample]] = {}
        self.text_store: Optional[TextStore] = None
        self._all_global_indices: list[int] = []   # for random sampling

    # ── Stage 1: TopK ─────────────────────────────────────────────────────────

    def run_topk(self) -> None:
        log.info("[Stage 1] Finding top-K activating examples per feature...")
        finder = TopKFinder(self.feat_cache_dir, k=self.top_k)
        finder.run()
        self.topk_results = finder.results

        meta = json.loads((Path(self.feat_cache_dir) / "meta.json").read_text())
        self._n_a   = meta["n_a"]
        self._n_b   = meta["n_b"]
        self._dict_size = meta["dict_size"]
        self._a_end = self._n_a
        self._b_end = self._n_a + self._n_b

        # Build global index list for random sampling during detection scoring
        self._all_global_indices = list(range(meta["total_samples"]))

        log.info(f"  {len(self.topk_results):,} features have at least one activation.")

    # ── Stage 2: Text Store ────────────────────────────────────────────────────

    def load_text_store(self) -> None:
        log.info("[Stage 2] Initialising text store...")
        # Text files are saved alongside features in the feature cache directory
        self.text_store = TextStore(self.feat_cache_dir)
        log.info("  TextStore ready.")

    # ── Stage 3: Explain + Score (async) ──────────────────────────────────────

    def _partition_of(self, feat_idx: int) -> str:
        if feat_idx < self._a_end:
            return "a_excl"
        if feat_idx < self._b_end:
            return "b_excl"
        return "shared"

    def _is_dead(self, feat_idx: int) -> bool:
        examples = self.topk_results.get(feat_idx, [])
        return len(examples) == 0

    def _random_texts(self, n: int, exclude_indices: set[int]) -> list[str]:
        """Sample n random texts that are NOT in the top-K set."""
        pool = [i for i in random.sample(
            self._all_global_indices,
            min(n * 10, len(self._all_global_indices))
        ) if i not in exclude_indices]
        chosen = pool[:n]
        # Convert global_idx → (shard_idx, row) using the shard list
        # We need meta to do this; store shard_sizes during run_topk
        # Simplified: use global_idx directly via text_store if it supports it
        texts = []
        for gidx in chosen:
            shard_idx, row = self._global_to_shard(gidx)
            texts.append(self.text_store.get(shard_idx, row))
        return texts

    def _global_to_shard(self, global_idx: int) -> tuple[int, int]:
        """Convert a flat global index to (shard_idx, row_in_shard)."""
        # Uses self._shard_sizes populated in run_topk
        offset = 0
        for shard_idx, size in enumerate(self._shard_sizes):
            if global_idx < offset + size:
                return shard_idx, global_idx - offset
            offset += size
        # Fallback: last shard
        return len(self._shard_sizes) - 1, global_idx - offset

    def _load_shard_sizes(self) -> None:
        """Pre-compute shard sizes (needed for global_idx → shard mapping)."""
        meta = json.loads((Path(self.feat_cache_dir) / "meta.json").read_text())
        self._shard_sizes = []
        for fname in meta["shards"]:
            shard = torch.load(
                Path(self.feat_cache_dir) / fname,
                map_location="cpu",
                weights_only=True,
            )
            self._shard_sizes.append(shard.shape[0])
            del shard

    async def _process_feature(self, feat_idx: int) -> None:
        """Full pipeline for one feature: explain → score → save."""
        if self.store.exists(feat_idx):
            return

        rec = FeatureRecord(
            feat_idx=feat_idx,
            partition=self._partition_of(feat_idx),
            is_dead=self._is_dead(feat_idx),
        )

        if rec.is_dead:
            self.store.save(rec)
            return

        examples = self.topk_results[feat_idx]
        texts    = self.text_store.get_batch(examples)
        activations = [e.activation for e in examples]

        rec.top_texts       = texts
        rec.top_activations = activations

        try:
            # ── Explanation ──────────────────────────────────────────────
            expl = await self.explainer.explain(feat_idx, texts, activations)
            rec.explanation   = expl.explanation
            rec.harmful       = expl.harmful
            rec.harmful_reason = expl.harmful_reason

            # ── Detection Score ──────────────────────────────────────────
            if rec.explanation:
                exclude = {e.global_idx for e in examples}
                rand_texts = self._random_texts(self.n_random, exclude)

                det = await self.scorer.score(
                    explanation=rec.explanation,
                    match_texts=texts,
                    random_texts=rand_texts,
                )
                rec.detection_score   = det.score
                rec.detection_n_correct = det.n_correct
                rec.detection_n_total   = det.n_total

        except Exception as e:
            rec.error = str(e)
            log.error(f"  Feature {feat_idx}: error — {e}")

        self.store.save(rec)

    async def _run_async(self, feature_ids: list[int]) -> None:
        total = len(feature_ids)
        t_start = time.time()

        # Process features sequentially with progress bar
        with tqdm(total=total, desc="Processing features", unit="feat") as pbar:
            for done, feat_id in enumerate(feature_ids):
                await self._process_feature(feat_id)
                pbar.update(1)
                
                # Print feature description every 10 features
                if feat_id % 10 == 0:
                    try:
                        rec = self.store.load(feat_id)
                        if rec and rec.explanation:
                            partition = self._partition_of(feat_id)
                            model_info = {
                                "a_excl": "Model A exclusive",
                                "b_excl": "Model B exclusive", 
                                "shared": "Shared between models"
                            }.get(partition, partition)
                            log.info(f"  Feature {feat_id} ({model_info}): {rec.explanation}")
                    except Exception:
                        pass
                
                # Log progress every 100 features
                if (done + 1) % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = (done + 1) / elapsed if elapsed > 0 else 0
                    eta = (total - done - 1) / rate if rate > 0 else 0
                    log.info(
                        f"  [Stage 3] {done + 1:,}/{total:,}  "
                        f"({100*(done + 1)/total:.1f}%)  "
                        f"{rate:.1f} feat/s  "
                        f"ETA {eta/60:.1f}m  "
                        f"API calls={self.claude.n_calls:,}  errors={self.claude.n_errors}"
                    )

    def run_explain_and_score(self) -> None:
        log.info("[Stage 3] Running explanation + detection scoring...")

        feature_ids = sorted(
            self.feature_subset
            if self.feature_subset is not None
            else range(self._dict_size)
        )
        # Skip already-done
        pending = [fi for fi in feature_ids if not self.store.exists(fi)]
        log.info(f"  {len(pending):,} features to process "
                 f"({len(feature_ids) - len(pending):,} already done).")

        if not pending:
            return

        asyncio.run(self._run_async(pending))

    # ── Full pipeline ──────────────────────────────────────────────────────────

    def run(self) -> dict:
        log.info("=" * 60)
        log.info("  DFC AutoInterp Pipeline")
        log.info("=" * 60)

        self.run_topk()
        log.info("  Loading shard sizes for index mapping...")
        self._load_shard_sizes()
        self.load_text_store()
        self.run_explain_and_score()

        log.info("[Stage 4] Computing summary...")
        summary = self.store.summary()
        log.info(json.dumps(summary, indent=2))

        summary_path = Path(self.store.results_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info(f"  Summary written to {summary_path}")

        log.info("=" * 60)
        log.info("  AutoInterp complete.")
        log.info("=" * 60)
        return summary