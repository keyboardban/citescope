from __future__ import annotations

from src.econometrics_eda_v2.brightdata_scraper import (
    BrightDataConfig,
    check_brightdata_config,
    normalize_brightdata_response,
    scrape_url_brightdata,
)


def test_check_brightdata_config_reports_missing_required_env():
    status = check_brightdata_config({})
    assert status["ok"] is False
    assert status["missing"] == ["BRIGHTDATA_API_KEY", "BRIGHTDATA_ENDPOINT"]


def test_normalize_brightdata_response_accepts_common_body_fields():
    out = normalize_brightdata_response(
        {
            "url": "https://example.com/final",
            "statusCode": 200,
            "title": "Example",
            "description": "Meta",
            "content": "Useful body text " * 40,
        },
        "https://example.com",
    )
    assert out["provider"] == "brightdata"
    assert out["requested_url"] == "https://example.com"
    assert out["final_url"] == "https://example.com/final"
    assert out["status_code"] == 200
    assert out["text"] == ("Useful body text " * 40).strip()
    assert out["success"] is True


def test_scrape_url_brightdata_uses_configurable_endpoint_and_payload():
    calls = []

    class Response:
        status_code = 200
        text = ""

        def json(self):
            return {"text": "Body " * 40, "title": "Title"}

    def request_fn(endpoint, headers, json, timeout):
        calls.append((endpoint, headers, json, timeout))
        return Response()

    config = BrightDataConfig(
        api_key="secret",
        endpoint="https://bright.example/scrape",
        zone="z1",
        country="us",
        render_js=True,
    )
    out = scrape_url_brightdata("https://example.com", config, request_fn=request_fn)
    assert out["success"] is True
    endpoint, headers, payload, timeout = calls[0]
    assert endpoint == "https://bright.example/scrape"
    assert headers["Authorization"] == "Bearer secret"
    assert payload["url"] == "https://example.com"
    assert payload["zone"] == "z1"
    assert payload["country"] == "us"
    assert payload["render_js"] is True
    assert timeout == 90
