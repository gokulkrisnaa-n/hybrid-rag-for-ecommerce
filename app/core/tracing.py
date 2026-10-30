"""
Tracing: one Langfuse trace per API request, one observation per step.

    with request_trace("chat", request_id=rid, input={...}, session_id=sid, tags=[...]) as root:
        with observe("generate-answer", as_type="generation", model=m) as gen:
            ...
            gen.update(output=..., usage_details={"input": 1900, "output": 200})
        root.update(output=...)
        root.score("answer-fallback", 0, data_type="BOOLEAN")

The rest of the app only uses this module, so the vendor can change (or move
to plain OpenTelemetry) in one place. Rules:

- Off unless LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (and
  tracing_enabled): every helper then returns a no-op, so tests, evals and
  the CLIs behave exactly as before.
- Tracing never breaks a request. Errors inside tracing are logged and
  dropped; the SDK sends data from a background thread.
- The trace id is the request id (both are 32 lowercase hex characters), so a
  shopper's error report, the access log and the trace all share one id.
- Shopper text is masked (emails, phone numbers, card numbers) before it is
  sent. Traces carry product ids and titles, not the product blocks, unless
  tracing_capture_context is on.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.core.config import get_settings

log = logging.getLogger("tracing")

_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_client: Any = None            # Langfuse client, created on first use; False = unavailable


# ---------------------------------------------------------------------------
# PII masking (shopper text only: product ids are long numbers too)
# ---------------------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?!\w)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def mask_pii(text: Optional[str]) -> Optional[str]:
    """'mail me at a@b.com or 555-123-4567' -> 'mail me at [email] or [phone]'.
    Card numbers only when they pass the Luhn check, so order numbers and sizes survive."""
    if not text:
        return text
    text = _EMAIL.sub("[email]", text)
    text = _CARD.sub(lambda m: "[card]" if _luhn_ok(re.sub(r"\D", "", m.group())) else m.group(), text)
    return _PHONE.sub("[phone]", text)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_client() -> Any:
    """The Langfuse client, or None when tracing is off or failed to start."""
    global _client
    if _client is None:
        s = get_settings()
        if not (s.tracing_enabled and s.langfuse_public_key and s.langfuse_secret_key):
            _client = False
        else:
            try:
                from langfuse import Langfuse      # imported only when tracing is on
                _client = Langfuse(
                    public_key=s.langfuse_public_key,
                    secret_key=s.langfuse_secret_key.get_secret_value(),
                    base_url=s.langfuse_base_url,
                    environment=s.tracing_environment,
                    sample_rate=s.tracing_sample_rate,
                )
                log.info("tracing to %s (environment %s)", s.langfuse_base_url, s.tracing_environment)
            except Exception as e:  # noqa: BLE001 - no tracing is better than no service
                log.warning("tracing disabled: %s: %s", type(e).__name__, e)
                _client = False
    return _client or None


def tracing_active() -> bool:
    return _get_client() is not None


def reset_client() -> None:
    """Forget the client (tests, or after settings change)."""
    global _client
    _client = None


def trace_id_for(request_id: str) -> str:
    """Langfuse trace ids are 32 lowercase hex characters. Ours already are, unless the
    client sent its own X-Request-ID; then derive a stable one from it."""
    if _HEX32.match(request_id):
        return request_id
    client = _get_client()
    if client is not None:
        return client.create_trace_id(seed=request_id)
    import hashlib
    return hashlib.sha256(request_id.encode()).hexdigest()[:32]


def flush() -> None:
    client = _get_client()
    if client is not None:
        try:
            client.flush()
        except Exception as e:  # noqa: BLE001
            log.warning("tracing flush failed: %s", e)


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class Observation:
    """A step in the trace. Every method swallows tracing errors."""

    def __init__(self, raw: Any = None):
        self._raw = raw

    @property
    def active(self) -> bool:
        return self._raw is not None

    def update(self, **kw: Any) -> None:
        if self._raw is None:
            return
        try:
            self._raw.update(**{k: v for k, v in kw.items() if v is not None})
        except Exception as e:  # noqa: BLE001
            log.warning("tracing update failed: %s", e)

    def score(self, name: str, value: float | str, *, data_type: str, comment: Optional[str] = None) -> None:
        """A score on the whole trace (e.g. answer-fallback)."""
        if self._raw is None:
            return
        try:
            self._raw.score_trace(name=name, value=value, data_type=data_type, comment=comment)
        except Exception as e:  # noqa: BLE001
            log.warning("tracing score failed: %s", e)


NOOP = Observation()


@contextmanager
def observe(name: str, *, as_type: str = "span", input: Any = None, metadata: Any = None,
            model: Optional[str] = None, trace_context: Optional[dict] = None) -> Iterator[Observation]:
    """A child step of whatever observation is current. An exception inside marks the
    step ERROR (with the message) and is re-raised unchanged."""
    client = _get_client()
    if client is None:
        yield NOOP
        return
    try:
        kwargs: dict[str, Any] = {"name": name, "as_type": as_type, "input": input, "metadata": metadata}
        if model:
            kwargs["model"] = model
        if trace_context:
            kwargs["trace_context"] = trace_context
        cm = client.start_as_current_observation(**kwargs)
        raw = cm.__enter__()
    except Exception as e:  # noqa: BLE001
        log.warning("tracing start failed for %s: %s", name, e)
        yield NOOP
        return
    obs = Observation(raw)
    try:
        yield obs
    except BaseException as e:
        obs.update(level="ERROR", status_message=f"{type(e).__name__}: {e}"[:500])
        raise
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            log.warning("tracing end failed for %s: %s", name, e)


@contextmanager
def request_trace(name: str, *, request_id: str, input: Any = None, session_id: Optional[str] = None,
                  tags: Optional[list[str]] = None, metadata: Optional[dict[str, str]] = None) -> Iterator[Observation]:
    """The root of a request's trace. The root's input/output become the trace's."""
    client = _get_client()
    if client is None:
        yield NOOP
        return
    try:
        from langfuse import propagate_attributes
        attrs = propagate_attributes(trace_name=name, session_id=session_id, tags=tags or None,
                                     metadata={k: str(v)[:200] for k, v in (metadata or {}).items()} or None)
        attrs.__enter__()
    except Exception as e:  # noqa: BLE001
        log.warning("tracing attributes failed: %s", e)
        attrs = None
    try:
        with observe(name, input=input, trace_context={"trace_id": trace_id_for(request_id)}) as root:
            yield root
    finally:
        if attrs is not None:
            try:
                attrs.__exit__(None, None, None)
            except Exception as e:  # noqa: BLE001
                log.warning("tracing attributes exit failed: %s", e)


def record_feedback(request_id: str, *, thumbs_up: bool, comment: Optional[str] = None) -> bool:
    """A shopper's thumbs up / down on one answer, as the user-thumbs score on that request's
    trace. One score per answer: changing the vote replaces it. False if tracing is off."""
    client = _get_client()
    if client is None:
        return False
    trace_id = trace_id_for(request_id)
    try:
        client.create_score(name="user-thumbs", value=1 if thumbs_up else 0, data_type="BOOLEAN",
                            trace_id=trace_id, score_id=f"{trace_id}-user-thumbs",
                            comment=mask_pii(comment) if comment else None)
    except Exception as e:  # noqa: BLE001
        log.warning("feedback score failed: %s", e)
        return False
    return True
