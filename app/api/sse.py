"""
Server-Sent Events for streamed answers.

    async def run(emit):              # does the work, reporting progress
        await emit("status", "searching")
        ...
        return {"kind": ...}          # the final payload, sent as the "answer" event

    return StreamingResponse(sse_stream(run, request_id), media_type="text/event-stream")

The work runs in its own task, feeding a queue the response reads from. The task
owns all its context (the trace, the model's concurrency slot, timeouts), so none
of it lives inside the response generator. If the client disconnects, the response
generator is closed and the task cancelled: the model stops, the slot is freed.

Events, in order:  meta -> status* -> delta* -> answer | error
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable

import psycopg
from psycopg_pool import PoolTimeout

log = logging.getLogger("api.error")

Emit = Callable[[str, Any], Awaitable[None]]


def format_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def error_payload(exc: BaseException, request_id: str) -> dict:
    """The API's error format, for failures after the stream has started (status 200 is sent)."""
    if isinstance(exc, (PoolTimeout, psycopg.OperationalError)):
        code, message = "database_unavailable", "The product database is temporarily unavailable. Please retry shortly."
    else:
        code, message = "internal_error", "Something went wrong. Please try again."
    return {"error": {"code": code, "message": message, "request_id": request_id}}


async def sse_stream(run: Callable[[Emit], Awaitable[Any]], request_id: str) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

    async def emit(event: str, data: Any) -> None:
        queue.put_nowait((event, data))

    async def produce() -> None:
        try:
            queue.put_nowait(("answer", await run(emit)))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - reported to the client as an error event
            log.exception("stream failed request_id=%s", request_id)
            queue.put_nowait(("error", error_payload(e, request_id)))
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(produce())
    try:
        yield format_event("meta", {"request_id": request_id})
        while (item := await queue.get()) is not None:
            event, data = item
            if event == "status":
                data = {"stage": data}
            elif event == "delta":
                data = {"text": data}
            yield format_event(event, data)
    finally:
        if not task.done():
            task.cancel()         # the client went away: stop the work
