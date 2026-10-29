"""
Title parsing: brand, collaborator, model and colorway from Shopify titles.

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
