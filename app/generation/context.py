"""
The product facts the model sees, one block per retrieved product.

Only what the answer needs, kept short: a long context costs tokens and
latency and gives the model more to get wrong. Product text comes from the
store, not from us, so it sits inside <product> blocks the prompt marks as
data: an instruction hidden in a description is just text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

MAX_BULLETS = 6
MAX_BULLET_CHARS = 200
MAX_SIZES = 15

_DETAILS_SQL = """
SELECT product_id, colorway, model, materials, description_bullets, style_code
FROM clean.products
WHERE product_id = ANY(%(ids)s)
"""


@dataclass
class ProductDetails:
    colorway: str | None = None
    model: str | None = None
    style_code: str | None = None
    materials: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


async def load_details(pool: AsyncConnectionPool, ids: list[int]) -> dict[int, ProductDetails]:
    if not ids:
        return {}
    async with pool.connection() as conn:
        rows = await (await conn.execute(_DETAILS_SQL, {"ids": ids})).fetchall()
    return {
        r["product_id"]: ProductDetails(
            colorway=r["colorway"], model=r["model"], style_code=r["style_code"],
            materials=list(r["materials"] or []), bullets=list(r["description_bullets"] or []),
        )
        for r in rows
    }


def _money(v: Decimal) -> str:
    return f"${v:,.0f}" if v == v.to_integral_value() else f"${v:,.2f}"


def _clean(text: str, limit: int) -> str:
    text = " ".join(str(text).split()).replace("<", "(").replace(">", ")")   # no tags inside a block
    return text if len(text) <= limit else text[: limit - 1] + "…"


def product_block(card, details: ProductDetails | None) -> str:
    """One <product> block. `card` is a ProductCard (prices and stock come from the database)."""
    d = details or ProductDetails()
    price = _money(card.price_min) if card.price_min == card.price_max else \
        f"{_money(card.price_min)}-{_money(card.price_max)}"
    if card.on_sale and card.compare_at_price:
        price += f" (on sale, was {_money(card.compare_at_price)})"
    kind = " / ".join(x for x in (card.category, card.subcategory) if x) or "unknown"
    sizes = ", ".join(card.sizes_available[:MAX_SIZES]) + (" …" if len(card.sizes_available) > MAX_SIZES else "")

    lines = [
        f'<product id="{card.product_id}">',
        f"Title: {_clean(card.title, 200)}",
        "Brand: " + " x ".join(x for x in (card.brand, card.collaborator) if x) if card.brand or card.collaborator else None,
        f"Type: {kind} | For: {card.gender or 'unknown'}",
        f"Colorway: {_clean(d.colorway, 120)}" if d.colorway else None,
        f"Materials: {_clean(', '.join(d.materials), 200)}" if d.materials else None,
        f"Price: {price} | In stock: {'yes' if card.in_stock else 'no'}"
        + (f" | Sizes in stock: {sizes}" if card.sizes_available else ""),
    ]
    bullets = [_clean(b, MAX_BULLET_CHARS) for b in d.bullets[:MAX_BULLETS] if b and b.strip()]
    if bullets:
        lines.append("Details:")
        lines += [f"- {b}" for b in bullets]
    lines.append("</product>")
    return "\n".join(x for x in lines if x)


def build_context(cards: list, details: dict[int, ProductDetails]) -> str:
    return "\n\n".join(product_block(c, details.get(c.product_id)) for c in cards)
