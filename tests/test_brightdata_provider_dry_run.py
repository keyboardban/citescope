from __future__ import annotations

import gzip
import pytest
import requests

from src.econometrics_eda_v2.brightdata_config import BrightDataSettings
from src.econometrics_eda_v2.scrape_providers import brightdata_provider
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import (
    build_brightdata_request_payload,
    build_brightdata_request_params,
    download_brightdata_snapshot,
    scrape_url_brightdata,
    trigger_brightdata_crawler_async,
    wait_for_brightdata_snapshot,
)


def _settings(api_key: str = "") -> BrightDataSettings:
    return BrightDataSettings(
        api_key=api_key,
        provider_mode="browser_api",
        endpoint="https://api.brightdata.com/request",
        zone="web_unlocker1",
        render_js=True,
        country="us",
        timeout_seconds=60,
        max_retries=2,
    )


def test_dry_run_brightdata_request_does_not_call_api_and_masks_key():
    config = _settings(api_key="do-not-print")
    result = scrape_url_brightdata("https://example.com", "browser_api", config, live=False)
    rendered = repr(result)
    assert "do-not-print" not in rendered
    assert result["planned_request"]["headers"]["Authorization"] == "Bearer ***"
    assert result["normalized_result"]["success"] is False
    assert result["normalized_result"]["error"] == "dry_run_no_api_call"


def test_live_without_api_key_fails_before_request_fn_is_called():
    called = False

    def request_fn(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("request_fn should not be called")

    with pytest.raises(RuntimeError, match="Live Bright Data execution requires BRIGHTDATA_API_KEY"):
        scrape_url_brightdata("https://example.com", "browser_api", _settings(), live=True, request_fn=request_fn)
    assert called is False


def test_build_payload_includes_render_and_country():
    payload = build_brightdata_request_payload("https://example.com", "unlocker_api", _settings())
    assert payload["url"] == "https://example.com"
    assert payload["zone"] == "web_unlocker1"
    assert payload["format"] == "raw"
    assert payload["country"] == "us"
    assert payload["render"] == "true"


def test_live_response_with_brightdata_proxy_error_header_is_not_success():
    class Response:
        status_code = 200
        text = ""
        headers = {
            "x-brd-status-code": "401",
            "x-brd-err-code": "client_10050",
            "x-brd-err-msg": "request IP is blacklisted in this zone",
        }

        def json(self):
            return {"text": ""}

    result = scrape_url_brightdata(
        "https://example.com", "browser_api", _settings(api_key="secret"), live=True,
        request_fn=lambda *_args, **_kwargs: Response(),
    )
    normalized = result["normalized_result"]
    assert normalized["success"] is False
    assert normalized["status_code"] == 401
    assert normalized["error"] == "request IP is blacklisted in this zone"


def test_crawler_api_uses_dataset_request_shape_and_normalizes_markdown_response():
    config = BrightDataSettings(
        api_key="secret",
        provider_mode="crawler_api",
        endpoint="https://api.brightdata.com/datasets/v3/scrape",
        zone="",
        render_js=True,
        country="us",
        timeout_seconds=60,
        max_retries=2,
        dataset_id="gd_example",
    )
    assert build_brightdata_request_payload("https://example.com/a", "crawler_api", config) == {
        "input": [{"url": "https://example.com/a"}]
    }
    assert build_brightdata_request_payload("https://example.com/a?utm_source=test&keep=1", "crawler_api", config) == {
        "input": [{"url": "https://example.com/a?keep=1"}]
    }
    assert build_brightdata_request_params("crawler_api", config) == {
        "dataset_id": "gd_example",
        "notify": "false",
        "include_errors": "true",
        "format": "json",
    }

    calls = []

    class Response:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return [{
                "url": "https://example.com/a",
                "page_title": "Example article",
                "markdown": "# Example article\\n\\nUseful crawler content " * 30,
                "page_html": "<html><body><main>Useful crawler content</main></body></html>",
            }]

    def request_fn(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return Response()

    result = scrape_url_brightdata("https://example.com/a", "crawler_api", config, live=True, request_fn=request_fn)
    normalized = result["normalized_result"]
    assert normalized["success"] is True
    assert normalized["provider_mode"] == "brightdata_crawler_api"
    assert normalized["title"] == "Example article"
    assert calls[0][0] == config.endpoint
    assert calls[0][1]["params"]["dataset_id"] == "gd_example"
    assert calls[0][1]["json"]["input"][0]["url"] == "https://example.com/a"


def test_async_crawler_trigger_wait_and_download_use_snapshot_lifecycle(monkeypatch):
    config = BrightDataSettings(
        api_key="secret", provider_mode="crawler_api", endpoint="https://api.brightdata.com/datasets/v3/trigger",
        zone="", render_js=True, country="us", timeout_seconds=60, max_retries=2,
        dataset_id="gd_example", crawler_async=True, crawler_poll_seconds=1, crawler_wait_seconds=5,
    )
    calls = []

    class Response:
        status_code = 200
        headers = {}
        text = ""

        def __init__(self, body): self.body = body
        def json(self): return self.body

    def post(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return Response({"snapshot_id": "s_example"})

    triggered = trigger_brightdata_crawler_async(["https://example.com/a?utm_source=test"], config, request_fn=post)
    assert triggered["snapshot_id"] == "s_example"
    assert calls[0][1]["json"] == [{"url": "https://example.com/a"}]

    assert wait_for_brightdata_snapshot("s_example", config, request_fn=lambda *_args, **_kwargs: Response({"status": "ready"}))["status"] == "ready"
    rows = download_brightdata_snapshot("s_example", config, request_fn=lambda *_args, **_kwargs: Response([{"url": "https://example.com/a", "markdown": "content"}]))
    assert rows == [{"url": "https://example.com/a", "markdown": "content"}]
    wrapped_rows = download_brightdata_snapshot("s_example", config, request_fn=lambda *_args, **_kwargs: Response({"data": rows}))
    assert wrapped_rows == rows

    class CompressedResponse:
        status_code = 200
        content = gzip.compress(b'{"url":"https://example.com/a","markdown":"content"}\n')

    assert download_brightdata_snapshot("s_example", config, request_fn=lambda *_args, **_kwargs: CompressedResponse()) == rows

    class BuildingResponse:
        status_code = 202

    build_attempts = []
    def compression_building(*_args, **_kwargs):
        build_attempts.append(1)
        return BuildingResponse() if len(build_attempts) == 1 else CompressedResponse()

    assert download_brightdata_snapshot("s_example", config, request_fn=compression_building) == rows
    assert len(build_attempts) == 2

    attempts = []
    monkeypatch.setattr(brightdata_provider.time, "sleep", lambda _seconds: None)

    def flaky_download(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.exceptions.ChunkedEncodingError("connection interrupted")
        return Response(rows)

    assert download_brightdata_snapshot("s_example", config, request_fn=flaky_download) == rows
    assert len(attempts) == 2
