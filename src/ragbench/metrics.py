"""Standard information-retrieval metrics, implemented from scratch and unit-tested."""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of relevant documents found in the top-k."""
    if not relevant:
        return 0.0
    hits = sum(1 for d in ranked[:k] if d in relevant)
    return hits / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document within the top-k."""
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain with graded relevance."""
    dcg = sum(
        (2 ** relevant.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], start=1)
    )
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0
