"""
Turn a QueryPlan into a concrete search: validated filters, merged with the
filters the shopper set in the UI, plus fusion weights for the intent.

Rules (each is visible in the debug info):
  1. Values the catalog doesn't have are dropped (vocabulary.clean).
  2. A filter the shopper set explicitly always wins over an inferred one.
  3. "adult" audience excludes kids items, unless a gender filter is already set.
  4. Intent picks the dense weight: lower for named products, higher for needs.
  5. Inferred subcategories are not applied unless qu_subcategory_filters is on
     (a third of products have none, so the filter hides good results).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from app.core.config import get_settings
from app.query.plan import QueryPlan
from app.query.vocabulary import Vocabulary
from app.schemas.query_models import SearchFilters


@dataclass
class ResolvedSearch:
    text: str
    filters: SearchFilters                       # explicit + inferred
    explicit: SearchFilters                      # what the shopper set; the relaxation fallback uses this
    weights: dict[str, float]
    inferred: dict = field(default_factory=dict)  # filter fields that came from the plan
    dropped: dict[str, list[str]] = field(default_factory=dict)

    @property
    def has_inferred_filters(self) -> bool:
        return bool(self.inferred)


def _price(v: float | None) -> Decimal | None:
    return None if v is None or v < 0 else Decimal(str(round(v, 2)))


def resolve(plan: QueryPlan, explicit: SearchFilters, vocab: Vocabulary, raw_text: str) -> ResolvedSearch:
    s = get_settings()
    cats, subs, brands, dropped = vocab.clean(categories=plan.categories,
                                              subcategories=plan.subcategories, brands=plan.brands)

    inferred: dict = {}
    if not explicit.categories and cats:
        inferred["categories"] = cats
    if s.qu_subcategory_filters and not explicit.subcategories and subs:
        inferred["subcategories"] = subs
    if not explicit.brands and brands:
        inferred["brands"] = brands
    if not explicit.genders and plan.genders:
        inferred["genders"] = list(dict.fromkeys(plan.genders))

    lo, hi = _price(plan.min_price), _price(plan.max_price)
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo                          # "between 200 and 100": take the intent, not the order
    if explicit.min_price is None and lo is not None and (explicit.max_price is None or lo <= explicit.max_price):
        inferred["min_price"] = lo
    if explicit.max_price is None and hi is not None and (explicit.min_price is None or hi >= explicit.min_price):
        inferred["max_price"] = hi

    if plan.on_sale_only and not explicit.on_sale_only:
        inferred["on_sale_only"] = True

    genders_set = explicit.genders or inferred.get("genders")
    if plan.audience == "adult" and not genders_set and not explicit.exclude_genders:
        inferred["exclude_genders"] = ["kids"]
    elif plan.audience == "kids" and not genders_set:
        inferred["genders"] = ["kids"]

    filters = explicit.model_copy(update=inferred)
    # Re-validate the merged filters (price order, list cleaning).
    filters = SearchFilters.model_validate(filters.model_dump())

    dense = s.qu_dense_weight_specific if plan.intent == "specific" else s.qu_dense_weight_explore
    text = " ".join(plan.search_text.split()) or raw_text
    return ResolvedSearch(
        text=text[:500],
        filters=filters,
        explicit=explicit,
        weights={"bm25": s.rrf_weight_bm25, "dense": dense},
        inferred={k: (list(v) if isinstance(v, list) else v) for k, v in inferred.items()},
        dropped=dropped,
    )
