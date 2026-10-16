"""
The catalog's real filter values: categories, subcategories and brands.

The LLM is shown the categories and subcategories so it picks from them, and
every value it returns is checked against these sets afterwards. A filter
value the catalog doesn't have would silently return nothing, so it is
dropped instead (and reported in the debug info).

Loaded once per process and refreshed after `ttl_seconds`, so a catalog
refresh is picked up without a restart.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from psycopg_pool import AsyncConnectionPool

_SQL_CATEGORIES = """
SELECT category, subcategory, count(*) AS n
FROM clean.products
WHERE is_active AND category IS NOT NULL
GROUP BY category, subcategory
ORDER BY category, n DESC
"""

_SQL_BRANDS = """
SELECT DISTINCT lower(name) AS name FROM (
    SELECT brand AS name FROM clean.products WHERE is_active
    UNION
    SELECT collaborator FROM clean.products WHERE is_active
) b WHERE name IS NOT NULL
"""


@dataclass
class Vocabulary:
    categories: dict[str, list[str]] = field(default_factory=dict)   # category -> subcategories, most common first
    brands: set[str] = field(default_factory=set)                     # lowercase

    @property
    def subcategories(self) -> set[str]:
        return {s for subs in self.categories.values() for s in subs}

    @property
    def version(self) -> str:
        """Changes when the category tree changes, so cached plans built on an old one are not reused."""
        text = "|".join(f"{c}:{','.join(subs)}" for c, subs in sorted(self.categories.items()))
        return hashlib.sha256(text.encode()).hexdigest()[:12]

    def prompt_block(self) -> str:
        lines = [f"- {c}: {', '.join(subs) if subs else '(no subcategories)'}" for c, subs in self.categories.items()]
        return "\n".join(lines)

    def clean(self, *, categories: list[str], subcategories: list[str], brands: list[str]
              ) -> tuple[list[str], list[str], list[str], dict[str, list[str]]]:
        """Keep only known values (case-insensitive). Returns the kept lists and what was dropped."""
        def keep(values: list[str], allowed: set[str]) -> tuple[list[str], list[str]]:
            kept, dropped = [], []
            for v in values:
                k = v.strip().lower()
                if not k:
                    continue
                (kept if k in allowed and k not in kept else dropped).append(k)
            return kept, [d for d in dropped if d not in kept]

        cats, dc = keep(categories, set(self.categories))
        subs, ds = keep(subcategories, self.subcategories)
        brs, db = keep(brands, self.brands)
        dropped = {k: v for k, v in (("categories", dc), ("subcategories", ds), ("brands", db)) if v}
        return cats, subs, brs, dropped


_cache: tuple[float, Vocabulary] | None = None


async def load_vocabulary(pool: AsyncConnectionPool, ttl_seconds: float = 3600) -> Vocabulary:
    global _cache
    now = time.monotonic()
    if _cache is not None and now - _cache[0] < ttl_seconds:
        return _cache[1]

    vocab = Vocabulary()
    async with pool.connection() as conn:
        for r in await (await conn.execute(_SQL_CATEGORIES)).fetchall():
            subs = vocab.categories.setdefault(r["category"].lower(), [])
            if r["subcategory"]:
                subs.append(r["subcategory"].lower())
        vocab.brands = {r["name"] for r in await (await conn.execute(_SQL_BRANDS)).fetchall()}
    _cache = (now, vocab)
    return vocab


def clear_vocabulary_cache() -> None:
    global _cache
    _cache = None
