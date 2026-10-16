"""
What a search asks for: free text plus structured filters.

For now these are built by hand (CLI, tests, evals). Later the query-
understanding step will turn a chat message like "waterproof jacket under
$200 for women" into exactly this shape.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

Gender = Literal["mens", "womens", "unisex", "kids"]
SearchMode = Literal["hybrid", "bm25", "dense"]


class SearchFilters(BaseModel):
    """Every filter is optional; an empty filter set means 'everything active and in stock'."""

    model_config = ConfigDict(extra="forbid")

    min_price: Optional[Decimal] = Field(default=None, ge=0)
    max_price: Optional[Decimal] = Field(default=None, ge=0)
    genders: list[Gender] = Field(default_factory=list)        # e.g. ["womens", "unisex"]
    # Leave these out. Unlike `genders`, products with no known gender stay in,
    # so "adult products" (exclude kids) doesn't drop the ~4% that are unlabeled.
    exclude_genders: list[Gender] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)        # e.g. ["footwear"]
    subcategories: list[str] = Field(default_factory=list)     # e.g. ["low top sneakers"]
    brands: list[str] = Field(default_factory=list)            # matches brand OR collaborator
    sizes: list[str] = Field(default_factory=list)             # any of these sizes in stock
    in_stock_only: bool = True
    on_sale_only: bool = False

    @field_validator("categories", "subcategories", "brands", "sizes", mode="before")
    @classmethod
    def _clean_list(cls, v: object, info: ValidationInfo) -> object:
        """
        Strip, drop blanks and dedupe. Categories are stored lowercase, so they
        are lowercased BEFORE deduping ("Footwear" and "footwear" are one value).
        Sizes keep their case ("M" vs "m" could differ in the data).
        """
        if not isinstance(v, list):
            return v
        lower = info.field_name in ("categories", "subcategories")
        out: dict[str, None] = {}
        for item in v:
            s = str(item).strip()
            if lower:
                s = s.lower()
            if s:
                out.setdefault(s, None)
        return list(out)

    @model_validator(mode="after")
    def _price_range(self) -> "SearchFilters":
        if self.min_price is not None and self.max_price is not None and self.min_price > self.max_price:
            raise ValueError("min_price is greater than max_price")
        return self


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=50)

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("search text is empty")
        return v