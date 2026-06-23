"""Exponential-backoff retry for transient API errors.

Retries: 429, 500, 502, 503, 504, and timeout/unavailable/rate-limit hints.
Never retries: 400, 401, 403, 404, 422 (bad request / invalid key / bad actor id),
so we don't waste calls or mask real configuration errors.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from . import config

_RETRYABLE_CODES = {429, 500, 502, 503, 504}
_NONRETRYABLE_CODES = {400, 401, 403, 404, 422}

_RETRYABLE_HINTS = (
    "resource_exhausted", "unavailable", "deadline_exceeded", "timeout",
    "timed out", "temporarily", "try again", "rate limit", "too many requests",
    "overloaded",
)
_NONRETRYABLE_HINTS = (
    "invalid_argument", "permission_denied", "unauthenticated", "invalid api key",
    "api key not valid", "no such actor", "actor not found", "forbidden",
    "bad request",
)


def _status_code(exc: Exception) -> int | None:
    for attr in ("code", "status_code", "status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    m = re.search(r"\b([45]\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


def is_retryable(exc: Exception) -> bool:
    code = _status_code(exc)
    if code in _NONRETRYABLE_CODES:
        return False
    if code in _RETRYABLE_CODES:
        return True
    msg = str(exc).lower()
    if any(h in msg for h in _NONRETRYABLE_HINTS):
        return False
    if any(h in msg for h in _RETRYABLE_HINTS):
        return True
    return False  # unknown errors are NOT retried (surface them)


def with_retry(
    fn: Callable,
    *,
    retries: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Call fn(); retry transient failures with exponential backoff."""
    retries = config.RETRY_COUNT if retries is None else retries
    base = config.RETRY_BASE_DELAY if base_delay is None else base_delay
    cap = config.RETRY_MAX_DELAY if max_delay is None else max_delay
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentional broad catch for retry
            if attempt >= retries or not is_retryable(exc):
                raise
            sleep(min(cap, base * (2 ** attempt)))
            attempt += 1
