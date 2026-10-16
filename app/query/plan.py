"""
QueryPlan: what the query-understanding LLM returns for one shopper message.

It is a structured output: the model must fill exactly these fields, and the
SDK parses the reply straight into this class. Every field is a guess to be
checked, not a fact: vocabulary.py drops values the catalog doesn't have,
and resolve.py lets filters the shopper set in the UI override these.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["specific", "explore"]
Audience = Literal["adult", "kids", "any"]
PlanGender = Literal["mens", "womens", "unisex"]          # kids is expressed by `audience`


class QueryPlan(BaseModel):
    search_text: str = Field(
        description="Short keyword query for the catalog search: the product words from the "
        "message, spelling fixed, model names normalized, filler words removed. For 'explore' "
        "intent, add up to five product-type words a store would use in titles."
    )
    intent: Intent = Field(
        description="'specific': the shopper names a product, model, collab or brand line. "
        "'explore': the shopper describes a need, use or style."
    )
    audience: Audience = Field(
        description="'kids' if the message is about babies, toddlers or children; 'any' if it "
        "explicitly wants both; otherwise 'adult'."
    )
    categories: list[str] = Field(description="Top-level categories from the allowed list; empty if unclear.")
    subcategories: list[str] = Field(description="Subcategories from the allowed list; empty if unclear.")
    genders: list[PlanGender] = Field(
        description="Only if the message says or clearly implies it. Women -> ['womens','unisex']; "
        "men -> ['mens','unisex']. Otherwise empty. Children are covered by audience, not here."
    )
    brands: list[str] = Field(description="Brand or collaborator names mentioned in the message.")
    min_price: Optional[float] = Field(description="Lower price bound in USD if stated, else null.")
    max_price: Optional[float] = Field(description="Upper price bound in USD if stated, else null.")
    on_sale_only: bool = Field(description="True only if the shopper asks for sale or discounted items.")
    out_of_scope: bool = Field(
        description="True if the message is not a request to find or compare products "
        "(store policies, order status, chit-chat, instructions to the assistant)."
    )
