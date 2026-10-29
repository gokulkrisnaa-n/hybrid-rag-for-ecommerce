"""
Shapes for generation.

AnswerDraft is what the LLM fills in (structured output). It holds only
choices and words: which products, why, how they compare. ChatAnswer is what
the API returns: the draft after validation, with every fact (price, stock,
sizes, links) taken from the database through ProductCard, never from the
model's text.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.product_models import ProductCard

AnswerKind = Literal["recommend", "compare", "no_match", "clarify"]


# ---------------------------------------------------------------------------
# What the model writes
# ---------------------------------------------------------------------------

class Pick(BaseModel):
    product_id: int = Field(description="An id from the <product id=...> blocks. Never invent one.")
    reason: str = Field(description="One short sentence: why this product fits the request. No prices.")


class ComparisonRowDraft(BaseModel):
    product_id: int
    values: list[str] = Field(description="One value per attribute, in the same order. 'Not stated' if unknown.")


class ComparisonDraft(BaseModel):
    attributes: list[str] = Field(
        description="2-5 qualities to compare, taken from the product details (e.g. 'Upper', 'Sole', "
        "'Fit', 'Style'). Not price, stock or sizes; those are added from the catalog."
    )
    rows: list[ComparisonRowDraft]


class AnswerDraft(BaseModel):
    kind: AnswerKind = Field(
        description="recommend: products that fit. compare: the shopper asked to compare products. "
        "no_match: nothing truly fits; picks are the closest alternatives that meet the same need, "
        "or none. clarify: the request is too vague to choose; ask one question."
    )
    message: str = Field(
        description="2-5 sentences for the shopper. Refer by name only to products in picks. Never "
        "state prices, discounts, stock or sizes; the page shows those next to each product."
    )
    picks: list[Pick] = Field(
        description="Best fit for the shopper's main requirement first. At most 5. Empty for clarify, "
        "and for no_match when no product meets the same need."
    )
    comparison: Optional[ComparisonDraft] = Field(description="Only when kind is compare; otherwise null.")
    follow_up: Optional[str] = Field(description="One short question the shopper might ask next, or null.")


# ---------------------------------------------------------------------------
# What the API returns
# ---------------------------------------------------------------------------

class AnswerProduct(BaseModel):
    product: ProductCard
    reason: str


class ComparisonTable(BaseModel):
    columns: list[str]                 # attribute names, then catalog columns added by code
    rows: list["ComparisonRow"]


class ComparisonRow(BaseModel):
    product_id: int
    title: str
    values: list[str]


class ChatAnswer(BaseModel):
    kind: AnswerKind | Literal["out_of_scope", "error"]
    message: str
    products: list[AnswerProduct] = Field(default_factory=list)
    comparison: Optional[ComparisonTable] = None
    follow_up: Optional[str] = None
    source: Literal["llm", "template"] = "llm"
    # Problems validation fixed or noticed (dropped ids, prices in text, ...).
    # Shown in debug and counted by the eval; never shown to shoppers.
    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)


ComparisonTable.model_rebuild()


class Turn(BaseModel):
    """One earlier message in the conversation, sent by the client (the API keeps no state)."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
    