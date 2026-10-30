"""
Request and response bodies of the HTTP API.

They reuse the service models (SearchFilters, Turn, RetrievedProduct,
ChatAnswer), so the API can't drift from what the services do. Money is
serialized as strings ("140.00"), the safe JSON form for decimals.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.generation.schemas import ChatAnswer, Turn
from app.schemas.product_models import RetrievedProduct
from app.schemas.query_models import SearchFilters, SearchMode

MAX_MESSAGE_CHARS = 500
MAX_HISTORY_TURNS = 20      # accepted; the chat service uses only the last chat_history_turns


def _clean_text(v: str) -> str:
    v = " ".join(v.split())
    if not v:
        raise ValueError("must not be blank")
    return v


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, examples=["black hoodie under $150"])
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=10, ge=1, le=50)
    mode: SearchMode = "hybrid"
    understand: Optional[bool] = Field(
        default=None,
        description="Run query understanding (an LLM call) before searching: spelling, inferred "
        "filters, intent. Defaults to on for hybrid mode; only available in hybrid mode.",
    )
    debug: bool = Field(default=False, description="Include the query plan and retrieval details.")

    @field_validator("query")
    @classmethod
    def _clean_query(cls, v: str) -> str:
        return _clean_text(v)

    @model_validator(mode="after")
    def _understand_needs_hybrid(self) -> "SearchRequest":
        if self.understand is None:
            self.understand = self.mode == "hybrid"
        elif self.understand and self.mode != "hybrid":
            raise ValueError("understand is only available with mode 'hybrid'")
        return self


class SearchResponse(BaseModel):
    request_id: str
    mode: SearchMode
    results: list[RetrievedProduct]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    debug: Optional[dict[str, Any]] = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS, examples=["comfortable shoes for walking"])
    history: list[Turn] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS,
        description="Earlier turns, oldest first. The API keeps no conversation state.",
    )
    filters: SearchFilters = Field(default_factory=SearchFilters)
    debug: bool = Field(default=False, description="Include the query plan, timings and token counts.")

    @field_validator("message")
    @classmethod
    def _clean_message(cls, v: str) -> str:
        return _clean_text(v)


class ChatResponse(ChatAnswer):
    request_id: str
    debug: Optional[dict[str, Any]] = None


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: Optional[list[dict[str, str]]] = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    checks: dict[str, str] = Field(default_factory=dict)


# Shown in the OpenAPI docs for every endpoint.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    403: {"model": ErrorResponse, "description": "Debug output requested but disabled on this server"},
    422: {"model": ErrorResponse, "description": "Invalid request body"},
    500: {"model": ErrorResponse, "description": "Unexpected error"},
    503: {"model": ErrorResponse, "description": "Database unavailable"},
}
