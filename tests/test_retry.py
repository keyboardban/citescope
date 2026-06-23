"""Retry/backoff: retries transient errors, never retries client errors."""

from __future__ import annotations

import pytest

from src import retry


class _CodeErr(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(str(code))


def test_is_retryable_codes_and_hints():
    assert retry.is_retryable(_CodeErr(429)) is True
    assert retry.is_retryable(_CodeErr(503)) is True
    assert retry.is_retryable(_CodeErr(400)) is False
    assert retry.is_retryable(_CodeErr(401)) is False
    assert retry.is_retryable(Exception("503 Service Unavailable")) is True
    assert retry.is_retryable(Exception("429 RESOURCE_EXHAUSTED")) is True
    assert retry.is_retryable(Exception("401 API key not valid")) is False
    assert retry.is_retryable(Exception("totally unknown error")) is False


def test_with_retry_retries_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _CodeErr(503)
        return "ok"

    out = retry.with_retry(fn, retries=5, base_delay=0, sleep=lambda s: None)
    assert out == "ok" and calls["n"] == 3


def test_with_retry_does_not_retry_client_error():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _CodeErr(400)

    with pytest.raises(_CodeErr):
        retry.with_retry(fn, retries=5, base_delay=0, sleep=lambda s: None)
    assert calls["n"] == 1


def test_with_retry_gives_up_after_max():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _CodeErr(429)

    with pytest.raises(_CodeErr):
        retry.with_retry(fn, retries=2, base_delay=0, sleep=lambda s: None)
    assert calls["n"] == 3  # initial attempt + 2 retries
