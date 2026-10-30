"""
Request context: an id for every request, timing, one access-log line, and a
last line of defence for unexpected errors.

- X-Request-ID: taken from the client if it sends a sane one, else generated.
  Returned on every response and written in every log line, so a shopper's
  report ("it broke at 3:02") can be matched to the logs (and later to traces).
- Unexpected exceptions become a plain 500 in the API's error format, with no
  stack trace for the client; the full trace goes to the log with the id.

Plain ASGI rather than BaseHTTPMiddleware, so it also works with streaming
responses later.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

access_log = logging.getLogger("api.access")
error_log = logging.getLogger("api.error")

_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")     # nothing that could forge a log line


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = dict(scope.get("headers") or []).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 500, "sent": False}

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"], status["sent"] = message["status"], True
                elapsed = (time.perf_counter() - started) * 1000
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode()))
                headers.append((b"server-timing", f"total;dur={elapsed:.1f}".encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            error_log.exception("unhandled error request_id=%s path=%s", request_id, scope.get("path"))
            if status["sent"]:
                raise                        # the response already started; nothing clean left to send
            body = json.dumps({"error": {"code": "internal_error",
                                         "message": "Something went wrong. Please try again.",
                                         "request_id": request_id}}).encode()
            await send_with_id({"type": "http.response.start", "status": 500,
                                "headers": [(b"content-type", b"application/json"),
                                            (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
        finally:
            client = scope.get("client")
            access_log.info(json.dumps({
                "request_id": request_id,
                "method": scope.get("method"),
                "path": scope.get("path"),
                "status": status["code"],
                "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                "client": client[0] if client else None,
            }))
