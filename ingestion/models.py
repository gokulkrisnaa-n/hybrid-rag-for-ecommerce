"""
Pydantic models for the Kith product ingestion pipeline.

RawProductRecord: the product exactly as products.json returned it.
CleanProduct + Variant: the cleaned, typed record that feeds Postgres.

Derived fields (price range, stock, sale flag, sizes, search text, hash) are
computed from the source fields, so they can never drift out of sync.
"""

from __future__ import annotations

import hashlib, re
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    computed_field,
    field_validator,
    model_validator,
)

# Pyndantic model for Fetcher Output
class RawProductRecord(BaseModel):
    """One product as scraped. `raw` is stored untouched and never edited."""

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(pattern=r"^\d+$", description="Shopify product ID")
    scraped_at: datetime
    source_url: HttpUrl
    raw: dict[str, Any] = Field(description="Full product object from baseurl/products.json")

    @model_validator(mode="after")
    def _id_matches_raw(self) -> "RawProductRecord":
        raw_id = self.raw.get("id")
        if raw_id is not None and str(raw_id) != self.product_id:
            raise ValueError(f"product_id {self.product_id} != raw id {raw_id}")
        return self


Gender = Literal["mens", "womens", "unisex", "kids"]

# A tag like <li> or </p>, or an entity like &amp; / &#39;
_HTML_RE = re.compile(r"</?[a-zA-Z][^>]*>|&(?:[a-zA-Z]+|#\d+|#x[0-9a-fA-F]+);")

# Pydantic model for each purchasable variant in every product
class Variant(BaseModel):
    """One purchasable size/color of a product. Becomes a row in `variants`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    variant_id: str = Field(pattern=r"^\d+$")
    size: Optional[str] = None
    sku: Optional[str] = None
    price: Decimal = Field(ge=0, decimal_places=2)
    compare_at_price: Optional[Decimal] = Field(default=None, ge=0, decimal_places=2)
    available: bool

    @field_validator("sku", "size", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        return None if isinstance(v, str) and not v.strip() else v

    @model_validator(mode="after")
    def _normalize_compare_at(self) -> "Variant":
        # Shopify sometimes sends compare_at_price equal to (or below) price.
        # That is not a discount, so treat it as absent.
        if self.compare_at_price is not None and self.compare_at_price <= self.price:
            self.compare_at_price = None
        return self

    @computed_field
    @property
    def on_sale(self) -> bool:
        return self.compare_at_price is not None


# Pydantic model for clean and parsed product
class CleanProduct(BaseModel):
    """The cleaned product. Becomes a row in `products` (variants go to their own table)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # Identity and links
    product_id: str = Field(pattern=r"^\d+$")
    handle: str = Field(min_length=1)
    url: HttpUrl
    image_url: Optional[HttpUrl] = None

    # Parsed attributes (None when the parser could not find them; never guess)
    title: str = Field(min_length=1)
    brand: Optional[str] = None
    collaborator: Optional[str] = None
    collaborator_note: Optional[str] = None
    model: Optional[str] = None
    colorway: Optional[str] = None
    style_code: Optional[str] = None
    product_type: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    gender: Optional[Gender] = None
    materials: list[str] = Field(default_factory=list)

    # Description, kept as bullets rather than one flattened string
    description_bullets: list[str] = Field(default_factory=list)

    # Tags that survived cleaning, plus facts lifted out of tags
    clean_tags: list[str] = Field(default_factory=list)
    purchase_limit: Optional[int] = Field(default=None, ge=1)

    # Per-size detail
    variants: list[Variant] = Field(min_length=1)

    # Timestamps
    shopify_created_at: Optional[datetime] = None
    shopify_updated_at: datetime
    shopify_published_at: Optional[datetime] = None
    scraped_at: datetime

    # --- normalizers -------------------------------------------------------

    @field_validator("clean_tags", "materials", mode="before")
    @classmethod
    def _lower_dedupe(cls, v: Any) -> Any:
        """Lowercase, strip, drop empties, dedupe while keeping order."""
        if not isinstance(v, list):
            return v
        seen: dict[str, None] = {}
        for item in v:
            s = str(item).strip().lower()
            if s:
                seen.setdefault(s, None)
        return list(seen)

    @field_validator("description_bullets", mode="before")
    @classmethod
    def _clean_bullets(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        return [str(b).strip() for b in v if str(b).strip()]

    @field_validator("description_bullets", "title", mode="after")
    @classmethod
    def _no_html(cls, v: Any) -> Any:
        """The clean layer must be HTML-free; parse body_html with cleaning.parse_description first."""
        items = v if isinstance(v, list) else [v]
        for item in items:
            if _HTML_RE.search(item):
                raise ValueError(f"HTML left in clean text: {item[:60]!r}")
        return v

    @model_validator(mode="after")
    def _unique_variant_ids(self) -> "CleanProduct":
        ids = [v.variant_id for v in self.variants]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate variant_id in variants")
        return self

    # --- commerce fields, rolled up from variants --------------------------

    @computed_field
    @property
    def price_min(self) -> Decimal:
        return min(v.price for v in self.variants)

    @computed_field
    @property
    def price_max(self) -> Decimal:
        return max(v.price for v in self.variants)

    @computed_field
    @property
    def compare_at_price(self) -> Optional[Decimal]:
        """Highest original price among discounted variants, if any."""
        prices = [v.compare_at_price for v in self.variants if v.compare_at_price is not None]
        return max(prices) if prices else None

    @computed_field
    @property
    def on_sale(self) -> bool:
        return any(v.on_sale for v in self.variants)

    @computed_field
    @property
    def in_stock(self) -> bool:
        return any(v.available for v in self.variants)

    @computed_field
    @property
    def sizes_available(self) -> list[str]:
        return [v.size for v in self.variants if v.available and v.size]

    # --- search document ---------------------------------------------------

    @computed_field
    @property
    def search_text(self) -> str:
        """
        The one string both the BM25 index and the embedding use.
        Prices, sizes and stock are deliberately excluded: they are SQL filters.
        """
        head = self.title
        if self.colorway and self.colorway not in self.title:
            head += f", {self.colorway}"
        lines = [head + "."]

        if self.brand:
            lines.append(f"Brand: {self.brand}.")
        if self.collaborator:
            collab = self.collaborator
            if self.collaborator_note:
                collab += f" {self.collaborator_note}"
            lines.append(f"Collaboration: {collab}.")

        audience = {"mens": "Men's", "womens": "Women's", "kids": "Kids'", "unisex": "Unisex"}
        kind = [p for p in (self.category, self.subcategory) if p]
        if self.gender or kind:
            prefix = audience.get(self.gender, "") if self.gender else ""
            lines.append(f"{prefix} {', '.join(kind)}.".strip())

        if self.materials:
            lines.append(f"Material: {', '.join(self.materials)}.")
        if self.style_code:
            lines.append(f"Style code: {self.style_code}.")

        extra_tags = [
            t for t in self.clean_tags
            if t not in {self.category, self.subcategory, self.gender}
        ]
        if extra_tags:
            lines.append(f"Tags: {', '.join(extra_tags)}.")

        text = " ".join(lines)
        if self.description_bullets:
            text += "\n" + "\n".join(f"- {b}" for b in self.description_bullets)
        return text

    @computed_field
    @property
    def search_text_hash(self) -> str:
        """Re-embed only when this changes."""
        return hashlib.sha256(self.search_text.encode("utf-8")).hexdigest()

    # --- helpers for the DB layer ------------------------------------------

    def product_row(self) -> dict[str, Any]:
        """Everything for the `products` table (variants excluded)."""
        return self.model_dump(exclude={"variants"})

    def variant_rows(self) -> list[dict[str, Any]]:
        """Rows for the `variants` table, each carrying the product_id foreign key."""
        return [{"product_id": self.product_id, **v.model_dump()} for v in self.variants]
