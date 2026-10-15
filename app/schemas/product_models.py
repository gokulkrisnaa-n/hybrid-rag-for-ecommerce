"""
Product shapes the API returns.

ProductCard is what a search result needs: enough to render a card and to
filter or compare. The full detail (description bullets, variants) gets its
own model when generation needs it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.query_models import SearchMode


class ProductCard(BaseModel):
    product_id: int
    title: str
    brand: Optional[str] = None
    collaborator: Optional[str] = None
    model: Optional[str] = None
    colorway: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    gender: Optional[str] = None
    price_min: Decimal
    price_max: Decimal
    compare_at_price: Optional[Decimal] = None
    on_sale: bool
    in_stock: bool
    sizes_available: list[str] = Field(default_factory=list)
    url: str
    image_url: Optional[str] = None


# Columns selected for a ProductCard, in one place so SQL and model stay in sync.
PRODUCT_CARD_COLUMNS = list(ProductCard.model_fields)


class RetrievedProduct(BaseModel):
    """A product plus how it was found. The ranks are what evals and debugging look at."""

    product: ProductCard
    score: float = Field(description="RRF score in hybrid mode; BM25 score or cosine distance otherwise")
    bm25_rank: Optional[int] = None       # 1-based; None if BM25 didn't return it
    dense_rank: Optional[int] = None      # 1-based; None if vector search didn't return it


class SearchResult(BaseModel):
    mode: SearchMode
    results: list[RetrievedProduct]
    timings_ms: dict[str, float] = Field(default_factory=dict)