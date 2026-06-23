"""Similarity scoring — the 'semantic overlap proxy' used across the audit.

Two interchangeable methods:
- 'lexical'   : offline bag-of-words cosine with sublinear TF weighting. Free,
                deterministic, no network. The default.
- 'embedding' : cosine over Gemini text embeddings (injected via embed_fn).

All scores are in [0, 1]. They measure relatedness, NOT causal use by the model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Sequence

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in",
    "on", "for", "with", "as", "by", "at", "from", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its", "i",
    "you", "he", "she", "they", "we", "my", "your", "their", "our", "what",
    "which", "who", "whom", "how", "when", "where", "why", "do", "does", "did",
    "not", "no", "can", "will", "would", "should", "could", "about", "into",
    "over", "than", "so", "such", "up", "out", "down", "more", "most", "some",
    "any", "all", "there", "here", "also", "very", "just", "best",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _tf_vector(text: str) -> dict[str, float]:
    counts = Counter(_tokens(text))
    # Sublinear TF dampens repetition so a few hot words don't dominate.
    return {term: 1.0 + math.log(c) for term, c in counts.items()}


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb)) if na and nb else 0.0


def _cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SimilarityEngine:
    """Scores text pairs. Embedding mode requires an injected embed_fn."""

    def __init__(
        self,
        method: str = "lexical",
        embed_fn: Callable[[Sequence[str]], list[list[float]]] | None = None,
    ) -> None:
        self.method = "embedding" if (method or "").startswith("embedding") or "embed" in (method or "") else "lexical"
        if self.method == "embedding" and embed_fn is None:
            # No embed function available -> fall back gracefully to lexical.
            self.method = "lexical"
        self._embed_fn = embed_fn
        self._embed_cache: dict[str, np.ndarray] = {}

    # -- embedding helpers -------------------------------------------------- #
    def _embed(self, texts: Sequence[str]) -> list[np.ndarray]:
        missing = [t for t in texts if t and t not in self._embed_cache]
        if missing:
            vecs = self._embed_fn(missing)  # type: ignore[misc]
            for t, v in zip(missing, vecs):
                self._embed_cache[t] = np.asarray(v, dtype=float)
        return [self._embed_cache.get(t, np.zeros(1)) for t in texts]

    # -- public API --------------------------------------------------------- #
    def score(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        if self.method == "embedding":
            va, vb = self._embed([a, b])
            return round(_cosine_dense(va, vb), 4)
        return round(_cosine_sparse(_tf_vector(a), _tf_vector(b)), 4)

    def score_many(self, target: str, texts: Sequence[str]) -> list[float]:
        """Score one target against many texts (used for chunk relevance)."""
        if not target or not texts:
            return [0.0] * len(texts)
        if self.method == "embedding":
            vt = self._embed([target])[0]
            vs = self._embed(list(texts))
            return [round(_cosine_dense(vt, v), 4) for v in vs]
        tv = _tf_vector(target)
        return [round(_cosine_sparse(tv, _tf_vector(t)), 4) for t in texts]


def summarize_scores(scores: Sequence[float]) -> dict[str, float]:
    """max / mean / mean-of-top-3 — handy aggregates for chunk scores."""
    if not scores:
        return {"max": 0.0, "mean": 0.0, "mean_top3": 0.0}
    arr = sorted(scores, reverse=True)
    top3 = arr[:3]
    return {
        "max": round(max(arr), 4),
        "mean": round(sum(arr) / len(arr), 4),
        "mean_top3": round(sum(top3) / len(top3), 4),
    }
