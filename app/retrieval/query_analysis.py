"""
Cheap, rule-based looks at the query text before searching. No LLM involved.

    is_code_query("aaki4323")             -> True   (send to BM25 only)
    content_terms("a hoodie for my mom")  -> ["hoodie", "my", "mom"]
    fuzzy_distance("hoddie")              -> 2

The LLM query-understanding step (Week 3) will do much more; these rules are
the baseline it has to beat, and the fallback when it is unavailable.
"""

from __future__ import annotations

import re

# Tantivy's (and Lucene's) English stopword list, the same words the BM25
# index drops when built with stopwords_language=english. Keep the two in
# sync: a word dropped by the index but not here would look like a term with
# no matches.
STOPWORDS = frozenset("""
a an and are as at be but by for if in into is it no not of on or such
that the their then there these they this to was will with
""".split())

_WORD = re.compile(r"[a-z0-9]+")

# One token, at least 6 characters, with both letters and digits, optionally
# joined by - . or /. Matches aaki4323, ai1203a899-105, ch6101-32-ss26.
# Short model names (990v6, 1906r) stay with hybrid search.
_CODE = re.compile(r"^(?=[^\s]*[a-z])(?=[^\s]*\d)[a-z0-9][a-z0-9\-./]{5,}$")


def is_code_query(text: str) -> bool:
    """A style code or SKU: embeddings know nothing about these, so BM25 should answer alone."""
    return bool(_CODE.match(text.strip().lower()))


def content_terms(text: str) -> list[str]:
    """Lowercased words, without stopwords, one-letter tokens or duplicates, in query order."""
    seen: set[str] = set()
    out = []
    for w in _WORD.findall(text.lower()):
        if len(w) < 2 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def fuzzy_distance(term: str) -> int:
    """
    Edit distance allowed when a term matches nothing. Short words get no
    fuzziness ("tee" within 2 edits matches half the catalog); swapping two
    letters (nupste -> nuptse) costs 2.
    """
    if len(term) >= 6:
        return 2
    if len(term) >= 4:
        return 1
    return 0
