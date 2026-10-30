"""
Title parsing (brand, collaborator, model, colorway) and variant price cleanup.

    pytest tests/test_parser_titles.py -q
"""

from __future__ import annotations

import pytest

from ingestion.parser import parse_title


@pytest.mark.parametrize("title, vendor, collab, model, colorway", [
    ("adidas x CLOT Samba Espadrille - Core Black / Cloud White", "adidas", "CLOT", "Samba Espadrille",
     "Core Black / Cloud White"),
    ("Kith for ASICS GEL-KAYANO 12.1 - Peat / Cream", "Kith", "ASICS", "GEL-KAYANO 12.1", "Peat / Cream"),
    # Multi-word partners: 'New Era' was cut to 'New' before it was a known partner.
    ("Timberland x New Era Heritage Premium 6-Inch Houston Astros Waterproof Boot - Wheat", "Timberland",
     "New Era", "Heritage Premium 6-Inch Houston Astros Waterproof Boot", "Wheat"),
    ("Kith Williams III Hoodie - Black", "Kith", None, "Williams III Hoodie", "Black"),
])
def test_parse_title(title, vendor, collab, model, colorway):
    t = parse_title(title, vendor, [])
    assert (t.collaborator, t.model, t.colorway) == (collab, model, colorway)


# --- variant prices -----------------------------------------------------------

from decimal import Decimal  # noqa: E402

from ingestion.models import Variant  # noqa: E402


@pytest.mark.parametrize("price, compare_at, exp_price, exp_compare, on_sale", [
    ("13.00", "130.00", "130.00", None, False),     # placeholder: a tenth of the real price
    ("100.00", "130.00", "100.00", "130.00", True),  # a real discount is kept
    ("130.00", "130.00", "130.00", None, False),     # compare-at equal to price is no discount
    ("0.00", "0.00", "0.00", None, False),           # free item: not a placeholder
])
def test_variant_price_cleanup(price, compare_at, exp_price, exp_compare, on_sale):
    v = Variant(variant_id="1", price=price, compare_at_price=compare_at, available=True)
    assert v.price == Decimal(exp_price)
    assert v.compare_at_price == (Decimal(exp_compare) if exp_compare else None)
    assert v.on_sale is on_sale
