"""
Try the assistant from the terminal.

    python -m app.chat "comfortable shoes for walking all day"
    python -m app.chat "compare the samba og and the gazelle"
    python -m app.chat "cheaper ones?" --history "user: black hoodie"
    python -m app.chat "warm jacket" --max-price 300 --gender womens unisex --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.chat.service import chat
from app.core.db import create_pool
from app.generation.schemas import ChatAnswer, Turn
from app.retrieval.hybrid import add_filter_arguments, filters_from_args


def _turn(text: str) -> Turn:
    role, sep, content = text.partition(":")
    if not sep or role.strip() not in ("user", "assistant"):
        raise argparse.ArgumentTypeError("history items look like 'user: black hoodie' or 'assistant: ...'")
    return Turn(role=role.strip(), content=content.strip())


def print_answer(a: ChatAnswer) -> None:
    print(f"\n[{a.kind}{' · template' if a.source == 'template' else ''}]  {a.message}\n")
    for i, p in enumerate(a.products, start=1):
        c = p.product
        price = f"${c.price_min:,.0f}" + ("*" if c.on_sale else "") + ("" if c.in_stock else " sold out")
        print(f"  {i}. {c.title[:80]}  ({price})")
        if p.reason:
            print(f"     {p.reason}")
    if a.comparison:
        print("\n  " + " | ".join(["Product"] + a.comparison.columns))
        for r in a.comparison.rows:
            print("  " + " | ".join([r.title[:40]] + r.values))
    if a.follow_up:
        print(f"\n  Next: {a.follow_up}")
    if a.warnings:
        print("\n  warnings: " + "; ".join(a.warnings))
    print("\n  " + ", ".join(f"{k} {v:.0f} ms" for k, v in a.timings_ms.items()))


async def _main() -> None:
    ap = argparse.ArgumentParser(description="Ask the shopping assistant.")
    ap.add_argument("message")
    ap.add_argument("--history", nargs="+", default=[], type=_turn, help="earlier turns, oldest first")
    ap.add_argument("--debug", action="store_true", help="also print the plan, filters and draft")
    add_filter_arguments(ap)
    args = ap.parse_args()

    async with create_pool(max_size=4) as pool:
        answer = await chat(pool, args.message, args.history, filters_from_args(args))
    print_answer(answer)
    if args.debug:
        print(json.dumps(answer.debug, indent=2, default=str))


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(_main())
