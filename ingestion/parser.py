"""
Parser for the Kith product ingestion pipeline.

Reads raw products from the database queue, cleans and parses the raw
Shopify product JSON, validates each result against CleanProduct, and
writes it to clean.products / clean.variants (see run_parse).

    raw.products (queue)  ->  parse_record()  ->  CleanProduct  ->  clean.*

What it does per product:
  - description: body_html -> clean bullets + facts (Style / Color / Material)
  - title: brand, collaborator, model and colorway
  - tags: drops merchandising noise, lifts out gender, category, subcategory,
          purchase limit and the collaborator note ("by Edison Chen")
  - variants: size, SKU, price, compare-at price, availability
  - image, timestamps

A product that fails validation is reported, not silently dropped, so you can
see which parsing rules need work.
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import psycopg
from bs4 import BeautifulSoup
from pydantic import ValidationError

from ingestion import db
from ingestion.models import CleanProduct, RawProductRecord, Variant

log = logging.getLogger("parser")


# ===========================================================================
# Vocabularies
# Tags are matched against these as an ALLOWLIST: a tag only counts if it
# maps to a known gender, category or product type. Promo codes, internal
# labels and typos are ignored automatically. Grow these lists from a
# tag-frequency count over the real catalog.
# ===========================================================================

GENDER_WORDS = {
    "mens": "mens", "men": "mens", "men's": "mens", "male": "mens",
    "womens": "womens", "women": "womens", "women's": "womens", "wmns": "womens", "female": "womens",
    "kids": "kids", "kid": "kids", "youth": "kids", "children": "kids", "baby": "kids",
    "grade school": "kids", "toddler": "kids", "infant": "kids",
    "unisex": "unisex",
}
# Tags like "women's footwear" or "m-accessories": a gender prefix + a known term.
# The prefix only counts when the rest is a known term ("m spring acc 26" is ignored).
GENDER_PREFIXES = {
    "mens": "mens", "men's": "mens", "men": "mens", "m": "mens",
    "womens": "womens", "women's": "womens", "women": "womens", "wmns": "womens", "w": "womens",
    "kids": "kids",
}

# Spelling -> category
CATEGORY_ALIASES = {
    "footwear": "footwear", "shoes": "footwear",
    "apparel": "apparel", "clothing": "apparel", "outerwear": "apparel",
    "tops": "apparel", "bottoms": "apparel",
    "accessories": "accessories", "accessory": "accessories",
    "accesory": "accessories", "acessories": "accessories", "acessory": "accessories",
    "headwear": "headwear", "bags": "bags", "eyewear": "eyewear",
    "jewelry": "jewelry", "home": "home",
}
# Narrower categories that also count as their parent when checking consistency.
CATEGORY_PARENT = {"headwear": "accessories", "bags": "accessories",
                   "eyewear": "accessories", "jewelry": "accessories"}

# Product type (subcategory) -> the category it belongs to
SUBCATEGORY_FAMILY = {
    "sneakers": "footwear", "low top sneakers": "footwear", "mid top sneakers": "footwear",
    "high top sneakers": "footwear", "running shoes": "footwear", "sandals": "footwear",
    "slides": "footwear", "boots": "footwear", "loafers": "footwear", "clogs": "footwear",
    "mules": "footwear",
    "t-shirts": "apparel", "long sleeve t-shirts": "apparel", "hoodies": "apparel",
    "sweatshirts": "apparel", "sweaters": "apparel", "cardigans": "apparel", "shirts": "apparel",
    "polos": "apparel", "jackets": "apparel", "coats": "apparel", "vests": "apparel",
    "pants": "apparel", "shorts": "apparel", "jeans": "apparel", "sweatpants": "apparel",
    "joggers": "apparel",
    "hats": "headwear", "caps": "headwear", "beanies": "headwear",
    "socks": "accessories", "backpacks": "bags", "totes": "bags",
}

# Spelling -> subcategory. Each canonical name and its singular are added below.
SUBCATEGORY_ALIASES = {
    "tees": "t-shirts", "tee": "t-shirts", "t shirts": "t-shirts", "t shirt": "t-shirts",
    "tshirts": "t-shirts", "tshirt": "t-shirts",
    "long sleeve tees": "long sleeve t-shirts", "long sleeve tee": "long sleeve t-shirts",
    "long sleeves": "long sleeve t-shirts",
    "crewnecks": "sweatshirts", "crewneck": "sweatshirts",
    "low tops": "low top sneakers", "high tops": "high top sneakers", "mid tops": "mid top sneakers",
    "crew socks": "socks", "running": "running shoes",
}
for _sub in SUBCATEGORY_FAMILY:
    SUBCATEGORY_ALIASES.setdefault(_sub, _sub)
    if _sub.endswith("s") and " " not in _sub:
        SUBCATEGORY_ALIASES.setdefault(_sub[:-1], _sub)      # "hoodie" -> "hoodies"
    SUBCATEGORY_ALIASES.setdefault(_sub.replace("-", " "), _sub)

# Nouns in a product title -> subcategory. The LAST one in the title wins,
# because that's the product noun: "Hat Trick Vintage Tee" is a tee, not a hat.
TITLE_NOUNS = {
    "tee": "t-shirts", "tees": "t-shirts", "t-shirt": "t-shirts",
    "hoodie": "hoodies", "sweatshirt": "sweatshirts", "crewneck": "sweatshirts",
    "sweater": "sweaters", "cardigan": "cardigans", "shirt": "shirts", "polo": "polos",
    "jacket": "jackets", "coat": "coats", "parka": "coats", "vest": "vests",
    "pant": "pants", "pants": "pants", "short": "shorts", "shorts": "shorts",
    "jean": "jeans", "jeans": "jeans", "sweatpant": "sweatpants", "sweatpants": "sweatpants",
    "jogger": "joggers", "joggers": "joggers",
    "hat": "hats", "cap": "caps", "beanie": "beanies",
    "sock": "socks", "socks": "socks", "backpack": "backpacks", "tote": "totes",
    "sneaker": "sneakers", "sneakers": "sneakers", "sandal": "sandals", "sandals": "sandals",
    "slide": "slides", "slides": "slides", "boot": "boots", "boots": "boots",
    "loafer": "loafers", "loafers": "loafers", "clog": "clogs", "clogs": "clogs",
    "mule": "mules", "mules": "mules",
}

# Collaboration partners that aren't vendors in the catalog, for titles like
# "Kith for the New York Rangers ...". Vendors are added automatically.
# Extend from: SELECT title FROM clean.products WHERE title ILIKE '% for %' AND collaborator IS NULL;
KNOWN_PARTNERS = {
    "ASICS", "Nike", "adidas", "New Balance", "Converse", "Vans", "Clarks", "Salomon",
    "Birkenstock", "BMW", "Columbia",
    "New York Rangers", "New York Knicks", "New York Yankees", "New York Mets",
    "New York Giants", "New York Jets", "Brooklyn Nets", "New York Liberty",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# Description lines like "Color: Core Black" become facts, not bullets.
FACT_KEYS = {"style", "color", "colour", "material", "materials", "fit", "sku"}


# ===========================================================================
# Small helpers
# ===========================================================================

_WS_RE = re.compile(r"\s+")
_MARKS_RE = re.compile("[\u2122\u2120\u00ae\u00a9]")


def _norm(text: Any) -> str:
    """Unicode-normalize, decode HTML entities, collapse whitespace."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = _MARKS_RE.sub("", text)                # drop ™ ® ©, which NFKC would turn into "TM"
    text = unicodedata.normalize("NFKC", text)    # &nbsp; -> space, full-width chars -> normal
    return _WS_RE.sub(" ", text).strip()


def _or_none(text: Any) -> Optional[str]:
    t = _norm(text)
    return t or None


def _same(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a) and bool(b) and a.casefold() == b.casefold()


# ===========================================================================
# Description: body_html -> bullets + facts
# ===========================================================================

_BLOCK_TAGS = ["p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr"]
_FACT_RE = re.compile(r"^\s*([A-Za-z ]{2,20}):\s*(.+?)\s*$")


def parse_description(body_html: Optional[str]) -> tuple[list[str], dict[str, str]]:
    """
    Returns (bullets, facts).

    bullets: one HTML-free string per feature line. <li>, <p>, <br> and other
             block elements each start a new line, so list structure survives.
    facts:   lowercase key -> value for known keys,
             e.g. {"color": "Core Black / Cloud White", "material": "Leather / Suede"}.
    """
    if not body_html or not str(body_html).strip():
        return [], {}

    soup = BeautifulSoup(body_html, "html.parser")

    for tag in soup(["script", "style", "iframe", "img", "video", "noscript"]):
        tag.decompose()

    for tag in soup.find_all(_BLOCK_TAGS):
        if tag.name == "br":
            tag.replace_with("\n")
        else:
            tag.append("\n")

    bullets: list[str] = []
    facts: dict[str, str] = {}

    for raw_line in soup.get_text().split("\n"):
        line = _norm(raw_line).strip(" -•*·")
        if not line:
            continue

        m = _FACT_RE.match(line)
        if m and m.group(1).strip().lower() in FACT_KEYS:
            key = m.group(1).strip().lower()
            key = {"colour": "color", "materials": "material"}.get(key, key)
            facts.setdefault(key, m.group(2))
            continue

        if line not in bullets:          # descriptions sometimes repeat lines
            bullets.append(line)

    return bullets, facts


def split_materials(value: Optional[str]) -> list[str]:
    """'Leather / Suede' -> ['leather', 'suede']; '100% Cotton' -> ['cotton']."""
    if not value:
        return []
    parts = re.split(r"\s*(?:/|,|&|\+|\band\b)\s*", value, flags=re.IGNORECASE)
    out = []
    for p in parts:
        p = re.sub(r"\d+\s*%", "", p).strip().lower()   # drop percentages
        if p:
            out.append(p)
    return out


# ===========================================================================
# Title: collaborator, model, colorway
# ===========================================================================

_COLLAB_SPLIT = re.compile(r"\s+[x×]\s+", re.IGNORECASE)
_FOR_SPLIT = re.compile(r"\s+for\s+(?:the\s+)?", re.IGNORECASE)


@dataclass
class TitleParts:
    collaborator: Optional[str] = None
    model: Optional[str] = None
    colorway: Optional[str] = None


def _known_prefix(text: str, names: Iterable[str]) -> Optional[str]:
    """The longest name that `text` starts with (whole words only), in its catalog spelling."""
    lower = text.casefold()
    for name in sorted(names, key=len, reverse=True):
        n = name.casefold()
        if lower.startswith(n) and (len(lower) == len(n) or not lower[len(n)].isalnum()):
            return name
    return None


def _strip_lead(text: str, lead: Optional[str]) -> str:
    if lead and text.casefold().startswith(lead.casefold()):
        return text[len(lead):].strip()
    return text.strip()


def parse_title(title: str, vendor: Optional[str], known_brands: Iterable[str]) -> TitleParts:
    """
    'adidas x CLOT Samba Espadrille - Core Black / Cloud White'
        -> collaborator='CLOT', model='Samba Espadrille', colorway='Core Black / Cloud White'
    'Kith for ASICS GEL-KAYANO 12.1 - Peat / Cream'
        -> collaborator='ASICS', model='GEL-KAYANO 12.1', colorway='Peat / Cream'
    'Kith for the New York Rangers On Ice Vintage Tee - White'
        -> collaborator='New York Rangers', model='On Ice Vintage Tee'
    'Kith Williams III Hoodie - Black'
        -> collaborator=None, model='Williams III Hoodie', colorway='Black'
    """
    parts = TitleParts()

    base, sep, colorway = title.rpartition(" - ")
    if sep:
        parts.colorway = _or_none(colorway)
    else:
        base = title
    base = base.strip()

    brands = set(known_brands)
    if vendor:
        brands.add(vendor)
    partners = brands | KNOWN_PARTNERS

    # "A x B Model": a two-brand collaboration
    pieces = _COLLAB_SPLIT.split(base, maxsplit=1)
    if len(pieces) == 2:
        left, right = pieces[0].strip(), pieces[1].strip()
        right_brand = _known_prefix(right, partners) or right.partition(" ")[0]
        remainder = _strip_lead(right, right_brand)

        if _same(left, vendor):
            parts.collaborator = right_brand
        elif _same(right_brand, vendor):
            parts.collaborator = left
        else:
            # Neither side is the vendor (e.g. a Kith-sold 'Nike x Sacai' item):
            # the right-hand brand is the partner.
            parts.collaborator = right_brand
        parts.model = _or_none(remainder)
        return parts

    # "Kith for ASICS Model" / "Kith for the New York Rangers Model":
    # only when the text before "for" is the vendor or a known brand.
    pieces = _FOR_SPLIT.split(base, maxsplit=1)
    if len(pieces) == 2 and (_same(pieces[0], vendor) or _known_prefix(pieces[0], brands) == pieces[0]):
        rest = pieces[1].strip()
        partner = _known_prefix(rest, partners)
        if partner:
            parts.collaborator = partner
            parts.model = _or_none(_strip_lead(rest, partner))
        else:
            # Unknown partner: don't guess where its name ends. Add it to KNOWN_PARTNERS.
            parts.model = _or_none(rest)
        return parts

    parts.model = _or_none(_strip_lead(base, vendor) if vendor and base.casefold().startswith(vendor.casefold() + " ") else base)
    return parts


def subcategory_from_title(text: Optional[str]) -> Optional[str]:
    """The last product noun in the title, e.g. 'Hat Trick Vintage Tee' -> 't-shirts'."""
    if not text:
        return None
    words = re.findall(r"[a-z]+(?:-[a-z]+)?", text.casefold())
    for i in range(len(words) - 1, -1, -1):
        sub = TITLE_NOUNS.get(words[i])
        if sub:
            if sub == "t-shirts" and "long sleeve" in " ".join(words):
                return "long sleeve t-shirts"
            return sub
    return None


# ===========================================================================
# Tags: allowlist signals
# ===========================================================================

_BY_PERSON = re.compile(r"\bby\s+((?:[A-Z][\w.'-]*\s?){1,4})")
_LIMIT_RE = re.compile(r"limit-quantity-(\w+)", re.IGNORECASE)


@dataclass
class TagSignals:
    """What the tags say. Nothing here is final; parse_record() reconciles it with the title."""
    genders: set[str] = field(default_factory=set)
    categories: list[str] = field(default_factory=list)
    subcategories: list[str] = field(default_factory=list)
    purchase_limit: Optional[int] = None
    collaborator_note: Optional[str] = None


def _as_tag_list(tags: Any) -> list[str]:
    """products.json returns a list; some Shopify endpoints return 'a, b, c'."""
    if isinstance(tags, str):
        tags = tags.split(",")
    return [_norm(t) for t in (tags or []) if _norm(t)]


def _tag_key(tag: str) -> str:
    """'High-Top-Sneakers' -> 'high top sneakers'."""
    return _WS_RE.sub(" ", re.sub(r"[-_]+", " ", tag.casefold())).strip()


def _known_term(key: str) -> tuple[Optional[str], Optional[str]]:
    """(category, subcategory) for a normalized tag, or (None, None) if it's not in the vocabulary."""
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key], None
    if key in SUBCATEGORY_ALIASES:
        return None, SUBCATEGORY_ALIASES[key]
    return None, None


def _add(lst: list[str], value: Optional[str]) -> None:
    if value and value not in lst:
        lst.append(value)


def parse_tags(tags: Any, *, title: str) -> TagSignals:
    out = TagSignals()
    title_base = (title.rpartition(" - ")[0] or title).casefold()

    for tag in _as_tag_list(tags):
        low = tag.casefold()

        # Purchase limit: a real fact, but not search text.
        if m := _LIMIT_RE.search(low):
            n = m.group(1)
            out.purchase_limit = int(n) if n.isdigit() else NUMBER_WORDS.get(n)
            continue

        # Long descriptive tag repeating the title, e.g.
        # 'adidas x CLOT Samba Espadrille by Edison Chen - Core Black / Cloud White'.
        # Keep only the 'by <person>' part.
        if len(tag) > 40 or low.startswith(title_base):
            if (m := _BY_PERSON.search(tag)) and not out.collaborator_note:
                out.collaborator_note = "by " + m.group(1).strip()
            continue

        key = _tag_key(tag)

        if key in GENDER_WORDS:
            out.genders.add(GENDER_WORDS[key])
            continue

        cat, sub = _known_term(key)
        if not (cat or sub):
            # "women's footwear", "m accessories": gender prefix + known term
            prefix, _, rest = key.partition(" ")
            if prefix in GENDER_PREFIXES and rest:
                cat, sub = _known_term(rest)
                if cat or sub:
                    out.genders.add(GENDER_PREFIXES[prefix])

        _add(out.categories, cat)
        _add(out.subcategories, sub)
        # Anything else (promo codes, internal labels, typos) is ignored.

    return out


def _compatible(sub: str, category: Optional[str]) -> bool:
    """Does this subcategory belong under this category? ('caps' fits 'headwear' and 'accessories')."""
    if category is None:
        return True
    family = SUBCATEGORY_FAMILY.get(sub)
    return family == category or CATEGORY_PARENT.get(family) == category


def resolve_gender(signals: set[str]) -> Optional[str]:
    """Several signals are common: a shoe tagged both 'mens' and 'wmns' is unisex."""
    if "kids" in signals:
        return "kids"
    if "unisex" in signals or {"mens", "womens"} <= signals:
        return "unisex"
    if len(signals) == 1:
        return next(iter(signals))
    return None


# Only unambiguous words count in titles: "baby" (as in "Baby Blue") or "kid"
# (as in "Kid Cudi") would mislabel products.
_TITLE_GENDER_WORDS = {"kids", "mens", "men's", "womens", "women's", "women", "wmns",
                       "unisex", "toddler", "infant", "youth"}


def _gender_from_text(text: str) -> set[str]:
    """Gender words in the vendor or title (without colorway): 'Kith Kids', "Women's ...", 'Wmns ...'."""
    words = set(re.findall(r"[a-z']+", text.casefold()))
    return {GENDER_WORDS[w] for w in words & _TITLE_GENDER_WORDS}


# ===========================================================================
# Variants
# ===========================================================================

def _size_option_index(options: Any) -> Optional[int]:
    """Which of option1/option2/option3 holds the size, from the product's options list."""
    for i, opt in enumerate(options or [], start=1):
        name = _norm(opt.get("name") if isinstance(opt, dict) else opt).casefold()
        if "size" in name:
            pos = opt.get("position", i) if isinstance(opt, dict) else i
            return int(pos)
    return None


def parse_variants(raw_variants: Any, options: Any) -> list[Variant]:
    size_idx = _size_option_index(options)
    variants = []
    for v in raw_variants or []:
        size = None
        if size_idx:
            size = _or_none(v.get(f"option{size_idx}"))
        if size and size.casefold() == "default title":
            size = None
        variants.append(
            Variant(
                variant_id=str(v.get("id", "")),
                size=size,
                sku=_or_none(v.get("sku")),
                price=v.get("price"),
                compare_at_price=v.get("compare_at_price") or None,
                available=bool(v.get("available", False)),
            )
        )
    return variants


# ===========================================================================
# Image
# ===========================================================================

def first_image_url(raw: dict[str, Any]) -> Optional[str]:
    images = raw.get("images") or []
    src = images[0].get("src") if images and isinstance(images[0], dict) else None
    if not src:
        return None
    return "https:" + src if src.startswith("//") else src


# ===========================================================================
# One product
# ===========================================================================

def parse_record(record: RawProductRecord, known_brands: Iterable[str] = ()) -> CleanProduct:
    """
    Parse one raw record into a validated CleanProduct.
    Raises pydantic.ValidationError (or ValueError) if the product can't be cleaned.
    """
    raw = record.raw
    title = _norm(raw.get("title"))
    vendor = _or_none(raw.get("vendor"))
    handle = _norm(raw.get("handle"))
    product_type = _or_none(raw.get("product_type"))

    bullets, facts = parse_description(raw.get("body_html"))
    style_code = facts.get("style")

    t = parse_title(title, vendor, known_brands)
    tg = parse_tags(raw.get("tags"), title=title)

    # Category: Shopify's product_type first (set by the merchandiser), then the
    # title's product noun, then the tags.
    type_key = _tag_key(product_type) if product_type else ""
    type_cat, type_sub = _known_term(type_key)
    title_sub = subcategory_from_title(t.model or title)
    category = (
        type_cat
        or (SUBCATEGORY_FAMILY.get(type_sub) if type_sub else None)
        or (SUBCATEGORY_FAMILY.get(title_sub) if title_sub else None)
        or next(iter(tg.categories), None)
        or next((SUBCATEGORY_FAMILY[s_] for s_ in tg.subcategories), None)
    )

    # Subcategory: title noun, then product_type, then the most specific
    # compatible tag. Tags that contradict the category are dropped
    # (Kith tags some socks 'sweatshirt').
    tag_subs = [s_ for s_ in tg.subcategories if _compatible(s_, category)]
    candidates = [c for c in (title_sub, type_sub) if c and _compatible(c, category)]
    candidates += sorted(tag_subs, key=lambda s_: -len(s_.split()))
    subcategory = candidates[0] if candidates else None

    title_base = title.rpartition(" - ")[0] or title           # colorway excluded
    gender = resolve_gender(tg.genders | _gender_from_text(f"{vendor or ''} {title_base}"))

    clean_tags = [x for x in [category, gender, subcategory, *tag_subs] if x]

    return CleanProduct(
        product_id=record.product_id,
        handle=handle,
        url=str(record.source_url),
        image_url=first_image_url(raw),
        title=title,
        brand=vendor,
        collaborator=t.collaborator,
        collaborator_note=tg.collaborator_note if t.collaborator else None,
        model=t.model,
        colorway=t.colorway or facts.get("color"),
        style_code=style_code,
        product_type=product_type,
        category=category,
        subcategory=subcategory,
        gender=gender,
        materials=split_materials(facts.get("material")),
        description_bullets=bullets,
        clean_tags=clean_tags,
        purchase_limit=tg.purchase_limit,
        variants=parse_variants(raw.get("variants"), raw.get("options")),
        shopify_created_at=raw.get("created_at"),
        shopify_updated_at=raw.get("updated_at"),
        shopify_published_at=raw.get("published_at"),
        scraped_at=record.scraped_at,
    )


# ===========================================================================
# Whole catalog
# ===========================================================================

@dataclass
class ParseFailure:
    product_id: str
    title: str
    error: str


@dataclass
class ParseResult:
    products: list[CleanProduct]
    failures: list[ParseFailure]

    def summary(self) -> str:
        total = len(self.products) + len(self.failures)
        lines = [f"Parsed {len(self.products)}/{total} products ({len(self.failures)} failed)."]
        if self.products:
            def coverage(attr: str) -> str:
                n = sum(1 for p in self.products if getattr(p, attr))
                return f"{attr}={100 * n // len(self.products)}%"
            lines.append(
                "Field coverage: "
                + ", ".join(coverage(a) for a in (
                    "collaborator", "model", "colorway", "category", "subcategory",
                    "gender", "materials", "style_code", "description_bullets",
                ))
            )
        for f in self.failures[:10]:
            lines.append(f"  FAILED {f.product_id} ({f.title[:50]}): {f.error}")
        return "\n".join(lines)


def _short_error(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:3]
        )
    return str(e)


def parse_catalog(
    records: list[RawProductRecord], known_brands: Optional[set[str]] = None
) -> ParseResult:
    """
    Parse every record in memory. Failures are collected, not raised, so one
    odd product never stops the whole run.
    """
    # Every vendor in the catalog doubles as a brand vocabulary for collab parsing
    # ('Kith x New Balance ...' -> 'New Balance', not just 'New').
    if known_brands is None:
        known_brands = {_norm(r.raw.get("vendor")) for r in records if r.raw.get("vendor")}

    products, failures = [], []
    for record in records:
        try:
            products.append(parse_record(record, known_brands))
        except (ValidationError, ValueError, TypeError) as e:
            failures.append(
                ParseFailure(record.product_id, _norm(record.raw.get("title")), _short_error(e))
            )
    return ParseResult(products, failures)


# ===========================================================================
# The parse step: raw.products queue -> clean.*
# ===========================================================================

def run_parse(conn, source: str = "https://kith.com", batch_size: int = 500) -> ParseResult:
    """
    Drain the parse queue (raw products whose current payload hasn't been
    parsed), writing clean rows or failures, then sync soft-deletes against
    the latest COMPLETE fetch run.

    Each product is saved in its own transaction, so one bad product (even one
    the database rejects) is recorded as a failure and never blocks the rest.
    """
    brands = {_norm(v) for v in db.known_vendors(conn)}
    result = ParseResult(products=[], failures=[])
    processed: set[tuple[str, str]] = set()

    while True:
        queue = db.fetch_parse_queue(conn, limit=batch_size)
        fresh = [q for q in queue if (q.record.product_id, q.payload_hash) not in processed]
        if not fresh:
            if queue:   # rows that keep coming back would loop forever; stop and say so
                log.warning("%d products stayed in the queue after processing; stopping.", len(queue))
            break

        for item in fresh:
            processed.add((item.record.product_id, item.payload_hash))
            title = _norm(item.record.raw.get("title"))
            try:
                product = parse_record(item.record, brands)
                db.save_clean_product(conn, product, item.payload_hash)
                result.products.append(product)
            except (ValidationError, ValueError, TypeError, psycopg.Error) as e:
                if isinstance(e, psycopg.Error):
                    msg = getattr(e.diag, "message_primary", None) or str(e)
                    error = f"database rejected: {msg}"
                else:
                    error = _short_error(e)
                db.save_parse_failure(conn, item.record.product_id, item.payload_hash, title, error)
                result.failures.append(ParseFailure(item.record.product_id, title, error))

        log.info("Batch done: %d parsed, %d failed so far.", len(result.products), len(result.failures))

    sync = db.sync_active_flags(conn, source)
    if sync.run_id is None:
        log.info("No complete fetch run yet for %s; soft-delete step skipped.", source)
    else:
        log.info(
            "Synced with complete run %d: %d deactivated, %d reactivated.",
            sync.run_id, sync.deactivated, sync.reactivated,
        )
    return result


if __name__ == "__main__":
    import argparse
    import logging

    ap = argparse.ArgumentParser(description="Parse queued raw products into clean.*")
    ap.add_argument("--source", default="https://kith.com")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--requeue-failed", action="store_true",
                    help="retry products that failed before (after fixing a parser bug)")
    ap.add_argument("--requeue-all", action="store_true",
                    help="re-parse every product (after changing parsing rules)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with db.connect() as conn:
        if args.requeue_all:
            log.info("Requeued %d products.", db.requeue_all(conn))
        elif args.requeue_failed:
            log.info("Requeued %d failed products.", db.requeue_failed(conn))

        result = run_parse(conn, args.source, args.batch_size)
        print(result.summary())

