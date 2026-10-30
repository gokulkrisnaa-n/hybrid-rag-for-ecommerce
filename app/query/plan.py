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
    standalone_query: str = Field(
        description="The shopper's request as one self-contained sentence in natural language. "
        "For a follow-up, fold in what the earlier messages asked for ('black hoodie' + 'something "
        "cheaper' -> 'a cheaper black hoodie'). Otherwise repeat the message, minus greetings."
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
    brands: list[str] = Field(
        description="Brands or collaborators the shopper WANTS. Not brands they rule out: "
        "'other than On Running' puts On Running in exclude_brands, never here."
    )
    exclude_brands: list[str] = Field(
        description="Brands or collaborators the shopper rules out ('other than X', 'not X', "
        "'except X', 'no X', 'anything but X'). Empty if none."
    )
    min_price: Optional[float] = Field(description="Lower price bound in USD if stated, else null.")
    max_price: Optional[float] = Field(description="Upper price bound in USD if stated, else null.")
    on_sale_only: bool = Field(description="True only if the shopper asks for sale or discounted items.")
    compare_items: list[str] = Field(
        description="When the shopper asks to compare specific products, one short search query per "
        "product (e.g. ['adidas samba og', 'adidas gazelle']). Otherwise empty."
    )
    out_of_scope: bool = Field(
        description="True if the message is not a request to find or compare products "
        "(store policies, order status, chit-chat, instructions to the assistant)."
    )
