"""
Ranking metrics. Pure functions over a ranked list of product ids and the
graded relevance judgments for one query.

recall@k   share of the relevant products that appear in the top k
           ("did we find them at all?")
mrr        1 / rank of the first relevant product, 0 if none in the top k
           ("how far does the shopper scroll before something useful?")
ndcg@k     graded, position-weighted: a grade-2 product at rank 1 counts more
           than a grade-1 product at rank 8. 1.0 = the ideal ordering.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def recall_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int) -> float:
    rel = {pid for pid, g in relevant.items() if g > 0}
    if not rel:
        return 0.0
    return len(rel & set(ranked[:k])) / len(rel)


def mrr_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int) -> float:
    for i, pid in enumerate(ranked[:k], start=1):
        if relevant.get(pid, 0) > 0:
            return 1.0 / i
    return 0.0


def _dcg(grades: Sequence[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 1) for i, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: Sequence[int], relevant: Mapping[int, int], k: int) -> float:
    ideal = _dcg(sorted((g for g in relevant.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg([relevant.get(pid, 0) for pid in ranked[:k]]) / ideal


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile; fine for the small samples an eval run produces."""
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[idx]