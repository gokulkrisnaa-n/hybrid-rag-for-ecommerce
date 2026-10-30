"""
One error format for every failure the API reports:

    {"error": {"code": "invalid_request", "message": "...", "request_id": "...", "details": [...]}}

`code` is stable and meant for programs; `message` is for people. No stack
traces or internal details reach the client: unexpected errors are logged
with the request id and returned as a plain 500 (see middleware.py).
"""

from __future__ import annotations

from typing import Any, Optional

import psycopg
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import PoolTimeout
from starlette.exceptions import HTTPException as StarletteHTTPException

_STATUS_CODES = {400: "bad_request", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
                 413: "payload_too_large", 415: "unsupported_media_type", 429: "rate_limited"}


class ApiError(Exception):
    """Raise anywhere in a request to return this status and error body."""

    def __init__(self, status: int, code: str, message: str, headers: Optional[dict[str, str]] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def error_response(request: Request, status: int, code: str, message: str,
                   details: Optional[list[dict[str, Any]]] = None,
                   headers: Optional[dict[str, str]] = None) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message, "request_id": request_id_of(request)}
    if details:
        body["details"] = details
    return JSONResponse({"error": body}, status_code=status, headers=headers)


def _field(loc: tuple) -> str:
    # ("body", "filters", "max_price") -> "filters.max_price". Only the first element says where
    # in the request it is; a field may itself be called "query".
    parts = [str(p) for p in loc]
    if parts and parts[0] in ("body", "query", "path", "header"):
        parts = parts[1:]
    return ".".join(parts) or "body"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(request, exc.status, exc.code, exc.message, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"field": _field(tuple(e.get("loc", ()))), "message": e.get("msg", "invalid")}
                   for e in exc.errors()]
        return error_response(request, 422, "invalid_request", "The request body is not valid.", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        return error_response(request, exc.status_code, code, str(exc.detail), headers=getattr(exc, "headers", None))

    @app.exception_handler(PoolTimeout)
    @app.exception_handler(psycopg.OperationalError)
    async def _database(request: Request, exc: Exception) -> JSONResponse:
        # Pool exhausted or database down: temporary, so 503 and a hint to retry.
        return error_response(request, 503, "database_unavailable",
                              "The product database is temporarily unavailable. Please retry shortly.",
                              headers={"Retry-After": "5"})
