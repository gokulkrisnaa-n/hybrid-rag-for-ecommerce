"""
Write the answer: one LLM call for the draft, then validation in code.

The model chooses and explains; code makes sure what reaches the shopper is
true to the catalog:
  - picks must be products that were retrieved (anything else is dropped);
  - a "compare" needs at least two valid products, or it becomes "recommend";
  - comparison price / sale / size columns are filled from ProductCard;
  - a dollar amount, size or stock statement in the text is flagged (the
    prompt forbids them: the page shows those facts from the database).
If the draft is unusable, the shopper still gets the top results with a fixed
message, never an invented product.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable

from pydantic_core import from_json

from app.core import tracing
from app.core.config import get_settings
from app.core.llm import call_llm, get_openai_client, llm_slot, trace_usage, usage_counts
from app.core.retry import backoff_delay, is_retryable
from app.generation.prompt import INSTRUCTIONS, PROMPT_VERSION, build_input
from app.generation.schemas import (
    AnswerDraft,
    AnswerProduct,
    ChatAnswer,
    ComparisonRow,
    ComparisonTable,
)
from app.schemas.product_models import ProductCard

_PRICE_RE = re.compile(r"(\$\s?\d|\d+(\.\d+)?\s?(usd|dollars?)\b|\b\d{1,3}\s?% off\b)", re.IGNORECASE)
# Sizes and stock change daily; the page shows them next to each product. "oversized",
# "true to size" style wording is fine, so only size *values* and stock claims match.
_STOCK_RE = re.compile(
    r"(\bsizes?\s+(?:\d|[xsml]{1,3}\b|range|start|run|available|in stock)"
    r"|\b\d{1,2}\s?[-–]\s?\d{1,2}\s?months?\b|\bin stock\b|\bsold out\b|\bout of stock\b)",
    re.IGNORECASE,
)
CATALOG_COLUMNS = ("Price", "On sale", "Sizes in stock")
MAX_TABLE_SIZES = 12


@dataclass
class Draft:
    answer: AnswerDraft
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    first_token_ms: float | None = None      # streamed answers: when the first words were sent


async def draft_answer(message: str, history: list[tuple[str, str]], context: str,
                       applied_filters: dict) -> Draft:
    """Raises on timeout, API error or unparsable output; the caller falls back to a template."""
    s = get_settings()
    trace_input, metadata = _trace_input(message, history, context, applied_filters)
    with tracing.observe("generate-answer", as_type="generation", model=s.generation_model, input=trace_input,
                         metadata=metadata) as gen:
        draft = await _call_model(message, history, context, applied_filters)
        gen.update(output=draft.answer.model_dump(mode="json"),
                   usage_details=trace_usage(draft.input_tokens, draft.cached_tokens, draft.output_tokens))
    return draft


async def stream_answer(message: str, history: list[tuple[str, str]], context: str, applied_filters: dict,
                        on_delta: Callable[[str], Awaitable[None]]) -> Draft:
    """
    Like draft_answer, but calls on_delta(text) with each new piece of the answer's
    `message` field as the model writes it (read from the partial JSON), so a shopper
    sees words before the whole answer exists. Returns the complete draft, which the
    caller validates as usual.

    Retries (full jitter) only before the first piece is sent: after that the shopper
    has seen text, so a failure is raised and the caller falls back.
    """
    s = get_settings()
    trace_input, metadata = _trace_input(message, history, context, applied_filters)
    with tracing.observe("generate-answer", as_type="generation", model=s.generation_model, input=trace_input,
                         metadata={**metadata, "streamed": True}) as gen:
        client = get_openai_client().with_options(timeout=s.generation_timeout_seconds, max_retries=0)
        request = _request(message, history, context, applied_filters)
        attempts = 1 + s.generation_max_retries
        start = time.perf_counter()
        sent, first_ms, final = "", None, None
        async with asyncio.timeout(s.generation_timeout_seconds):      # the whole stream, slot wait included
            async with llm_slot():
                for attempt in range(attempts):
                    buffer = ""
                    try:
                        async with client.responses.stream(**request) as stream:
                            async for event in stream:
                                if getattr(event, "type", "") != "response.output_text.delta":
                                    continue
                                buffer += event.delta
                                text = partial_message(buffer)
                                if text and len(text) > len(sent) and text.startswith(sent):
                                    if not sent:
                                        first_ms = (time.perf_counter() - start) * 1000
                                        gen.update(completion_start_time=datetime.now(timezone.utc))
                                    await on_delta(text[len(sent):])
                                    sent = text
                            final = await stream.get_final_response()
                        break
                    except Exception as e:
                        if sent or attempt == attempts - 1 or not is_retryable(e):
                            raise
                        await asyncio.sleep(backoff_delay(attempt, s.llm_retry_base_seconds, s.llm_retry_cap_seconds))
        parsed = getattr(final, "output_parsed", None)
        if parsed is None:
            raise ValueError("model returned no parsable answer (refusal or incomplete output)")
        input_tokens, cached_tokens, output_tokens = usage_counts(final.usage)
        draft = Draft(parsed, (time.perf_counter() - start) * 1000, input_tokens, output_tokens, cached_tokens,
                      first_token_ms=first_ms)
        gen.update(output=parsed.model_dump(mode="json"),
                   usage_details=trace_usage(input_tokens, cached_tokens, output_tokens))
    return draft


def partial_message(buffer: str) -> str | None:
    """The `message` field of a structured answer still being written:
    '{"kind":"recommend","message":"Here are tw' -> 'Here are tw'. None until it starts.
    An escape sequence cut in half is left out until the rest arrives."""
    try:
        data = from_json(buffer, allow_partial="trailing-strings")
    except ValueError:
        return None
    msg = data.get("message") if isinstance(data, dict) else None
    return msg if isinstance(msg, str) else None


_PRODUCT_TITLE = re.compile(r'<product id="(\d+)">\nTitle: ([^\n]*)')


def _trace_input(message: str, history: list[tuple[str, str]], context: str,
                 applied_filters: dict) -> tuple[list[dict], dict]:
    """The conversation as role-labelled messages; what the model chose from goes in metadata
    (the full product blocks only when asked: they are long)."""
    s = get_settings()
    messages = [{"role": r, "content": tracing.mask_pii(t)} for r, t in history]
    messages.append({"role": "user", "content": tracing.mask_pii(message)})
    products = context if s.tracing_capture_context else \
        [{"id": int(i), "title": t} for i, t in _PRODUCT_TITLE.findall(context)]
    return messages, {"prompt": PROMPT_VERSION, "reasoning_effort": s.generation_reasoning_effort,
                      "applied_filters": applied_filters, "products": products}


def _request(message: str, history: list[tuple[str, str]], context: str, applied_filters: dict) -> dict:
    """The Responses API arguments, shared by the plain and the streaming call."""
    s = get_settings()
    return {
        **({"reasoning": {"effort": s.generation_reasoning_effort}} if s.generation_reasoning_effort else {}),
        "model": s.generation_model,
        "instructions": INSTRUCTIONS,
        "input": build_input(message, history, context, applied_filters),
        "text_format": AnswerDraft,
        "store": False,
        "prompt_cache_key": f"kith-gen-{PROMPT_VERSION}",
    }


async def _call_model(message: str, history: list[tuple[str, str]], context: str, applied_filters: dict) -> Draft:
    s = get_settings()
    client = get_openai_client().with_options(timeout=s.generation_timeout_seconds, max_retries=0)
    request = _request(message, history, context, applied_filters)
    start = time.perf_counter()
    response = await call_llm(lambda: client.responses.parse(**request),
                              budget_seconds=s.generation_timeout_seconds,
                              attempts=1 + s.generation_max_retries)
    latency = (time.perf_counter() - start) * 1000
    if response.output_parsed is None:
        raise ValueError("model returned no parsable answer (refusal or incomplete output)")
    input_tokens, cached_tokens, output_tokens = usage_counts(response.usage)
    return Draft(response.output_parsed, latency, input_tokens, output_tokens, cached_tokens)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _price_text(card: ProductCard) -> str:
    lo, hi = card.price_min, card.price_max
    return f"${lo:,.2f}" if lo == hi else f"${lo:,.2f}-${hi:,.2f}"


def _sizes_text(card: ProductCard) -> str:
    """Sizes for the comparison table. A long list is cut, and says so: a shopper must never
    read a cut list as complete (half sizes from 3.5 once hid every size above 9)."""
    if not (card.in_stock and card.sizes_available):
        return "Sold out"
    shown = card.sizes_available[:MAX_TABLE_SIZES]
    more = len(card.sizes_available) - len(shown)
    return ", ".join(shown) + (f" … (+{more} more)" if more > 0 else "")


def _catalog_values(card: ProductCard) -> list[str]:
    return [_price_text(card), "Yes" if card.on_sale else "No", _sizes_text(card)]


def has_price_text(text: str) -> bool:
    return bool(_PRICE_RE.search(text or ""))


def has_stock_text(text: str) -> bool:
    return bool(_STOCK_RE.search(text or ""))


def validate(draft: AnswerDraft, cards: list[ProductCard], max_picks: int) -> ChatAnswer:
    by_id = {c.product_id: c for c in cards}
    warnings: list[str] = []

    picks: list[AnswerProduct] = []
    for p in draft.picks:
        if p.product_id not in by_id:
            warnings.append(f"dropped unknown product_id {p.product_id}")
        elif any(x.product.product_id == p.product_id for x in picks):
            warnings.append(f"dropped duplicate product_id {p.product_id}")
        elif len(picks) >= max_picks:
            warnings.append(f"dropped pick beyond the first {max_picks}")
        else:
            picks.append(AnswerProduct(product=by_id[p.product_id], reason=" ".join(p.reason.split())))

    kind = draft.kind
    comparison = None
    if kind == "compare":
        rows = []
        n_attr = len(draft.comparison.attributes) if draft.comparison else 0
        picked = {x.product.product_id for x in picks}
        for r in (draft.comparison.rows if draft.comparison else []):
            if r.product_id in picked and all(r.product_id != x.product_id for x in rows):
                values = (list(r.values) + ["Not stated"] * n_attr)[:n_attr]
                card = by_id[r.product_id]
                rows.append(ComparisonRow(product_id=card.product_id, title=card.title,
                                          values=values + _catalog_values(card)))
        if len(rows) >= 2:
            comparison = ComparisonTable(columns=list(draft.comparison.attributes) + list(CATALOG_COLUMNS), rows=rows)
        else:
            warnings.append("compare needs at least two valid products; answered as recommend")
            kind = "recommend"
    elif draft.comparison is not None:
        warnings.append(f"comparison ignored for kind {kind}")

    if kind in ("recommend", "compare") and not picks:
        warnings.append("no valid picks; using the top results")
        return fallback_answer(cards, warnings=warnings)

    texts = [draft.message] + [p.reason for p in picks]
    if any(has_price_text(t) for t in texts):
        warnings.append("price or discount written in the text")
    if any(has_stock_text(t) for t in texts):
        warnings.append("size or stock written in the text")

    return ChatAnswer(
        kind=kind,
        message=" ".join(draft.message.split()),
        products=picks if kind != "clarify" else [],
        comparison=comparison,
        follow_up=(draft.follow_up or None),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Fixed answers (no LLM)
# ---------------------------------------------------------------------------

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with finding and comparing products at Kith. For orders, returns, shipping "
    "or other store questions, please check Kith's help center or contact their customer service."
)
NO_RESULTS_MESSAGE = (
    "I couldn't find any products matching that. Try different words, or remove a filter such as "
    "price, size or category."
)
FALLBACK_MESSAGE = "Here are the closest matches I found."


def out_of_scope_answer() -> ChatAnswer:
    return ChatAnswer(kind="out_of_scope", message=OUT_OF_SCOPE_MESSAGE, source="template")


def no_results_answer(filters: dict) -> ChatAnswer:
    msg = NO_RESULTS_MESSAGE
    if filters:
        msg += " Filters applied: " + "; ".join(f"{k} {v}" for k, v in filters.items()) + "."
    return ChatAnswer(kind="no_match", message=msg, source="template")


def fallback_answer(cards: list[ProductCard], warnings: list[str] | None = None, n: int = 3) -> ChatAnswer:
    return ChatAnswer(
        kind="error" if not cards else "recommend",
        message=FALLBACK_MESSAGE if cards else NO_RESULTS_MESSAGE,
        products=[AnswerProduct(product=c, reason="") for c in cards[:n]],
        source="template",
        warnings=list(warnings or []),
    )
