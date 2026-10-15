"""
Reciprocal Rank Fusion (Cormack, Clarke & Büttcher, 2009).

Each method contributes 1 / (k + rank) for every product it returned:

    score(p) = sum over methods m of  weight_m / (k + rank_m(p))

RRF uses ranks, not raw scores, which is why it works here: BM25 scores
(unbounded, higher is better) and cosine distances (0..2, lower is better)
can't be compared directly, but "3rd place" means the same thing in both.
A product found by both methods beats one found by only one, unless the
single method ranked it far higher. k = 60 damps the gap between rank 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass
class FusedHit:
    product_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)    # method -> 1-based rank


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[int]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusedHit]:
    """
    ranked_lists: method name -> product ids, best first. Duplicates within a
    list keep their first (best) position.
    Ties are broken by the best single rank, then product id, so the output is
    deterministic.
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    weights = weights or {}

    fused: dict[int, FusedHit] = {}
    for method, ids in ranked_lists.items():
        w = weights.get(method, 1.0)
        seen: set[int] = set()
        rank = 0
        for pid in ids:
            if pid in seen:
                continue
            seen.add(pid)
            rank += 1
            hit = fused.setdefault(pid, FusedHit(pid, 0.0))
            hit.score += w / (k + rank)
            hit.ranks[method] = rank

    ordered = sorted(fused.values(), key=lambda h: (-h.score, min(h.ranks.values()), h.product_id))
    return ordered[:limit] if limit is not None else ordered