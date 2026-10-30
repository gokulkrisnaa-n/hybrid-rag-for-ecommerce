"""
POST /v1/feedback: a shopper's thumbs up / down on one chat answer.

Stored as the user-thumbs score (BOOLEAN) on that answer's trace, found by its
request_id. Voting again on the same answer replaces the score. The Langfuse
secret key stays on the server: the browser only ever calls this endpoint.

Anyone who knows a request_id can rate that answer; for a public demo that's
acceptable (ids are random and only returned to the client that asked).
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.errors import ApiError
from app.api.schemas import ERROR_RESPONSES, ErrorResponse, FeedbackRequest, FeedbackResponse
from app.core import tracing

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=202,
    responses={**ERROR_RESPONSES, 503: {"model": ErrorResponse, "description": "Tracing is not configured"}},
)
async def feedback(body: FeedbackRequest) -> FeedbackResponse:
    if not tracing.record_feedback(body.request_id, thumbs_up=body.rating == 1, comment=body.comment):
        raise ApiError(503, "tracing_disabled", "Feedback can't be recorded: tracing is not configured on this server.")
    return FeedbackResponse(request_id=body.request_id)
