"""
SearchFilters -> SQL WHERE clause, shared by BM25 and vector search.

Both methods rank exactly the same candidate set, so fusion never mixes a
product one method was allowed to return with one the other filtered out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from psycopg import sql

from app.schemas.query_models import SearchFilters


@dataclass
class WhereClause:
    clause: sql.Composable
    params: dict[str, Any] = field(default_factory=dict)


def build_where(f: SearchFilters) -> WhereClause:
    """All values are passed as parameters, never formatted into the SQL text."""
    conds: list[sql.Composable] = [sql.SQL("is_active")]
    params: dict[str, Any] = {}

    if f.in_stock_only:
        conds.append(sql.SQL("in_stock"))
    if f.on_sale_only:
        conds.append(sql.SQL("on_sale"))

    # "Under $200" means at least one variant costs <= 200, i.e. the cheapest one.
    if f.max_price is not None:
        conds.append(sql.SQL("price_min <= %(f_max_price)s"))
        params["f_max_price"] = f.max_price
    # "Over $100" means at least one variant costs >= 100, i.e. the dearest one.
    if f.min_price is not None:
        conds.append(sql.SQL("price_max >= %(f_min_price)s"))
        params["f_min_price"] = f.min_price

    if f.genders:
        conds.append(sql.SQL("gender = ANY(%(f_genders)s)"))
        params["f_genders"] = list(f.genders)
    if f.categories:
        conds.append(sql.SQL("category = ANY(%(f_categories)s)"))
        params["f_categories"] = f.categories
    if f.subcategories:
        conds.append(sql.SQL("subcategory = ANY(%(f_subcategories)s)"))
        params["f_subcategories"] = f.subcategories
    if f.brands:
        # "CLOT" should find adidas x CLOT, so match the collaborator too.
        conds.append(sql.SQL("(lower(brand) = ANY(%(f_brands)s) OR lower(collaborator) = ANY(%(f_brands)s))"))
        params["f_brands"] = [b.lower() for b in f.brands]
    if f.sizes:
        conds.append(sql.SQL("sizes_available && %(f_sizes)s::text[]"))
        params["f_sizes"] = f.sizes

    return WhereClause(sql.SQL(" AND ").join(conds), params)