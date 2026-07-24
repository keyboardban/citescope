from __future__ import annotations

from src.econometrics_eda_v2.brightdata_response_parser import extract_body_like_content


def test_extracts_nested_result_html_body():
    raw = {
        "status_code": 200,
        "result": {
            "page": {
                "html": "<html><head><title>Example</title></head><body><h1>Guide</h1><p>"
                + "Useful paragraph about research methods. " * 30
                + "</p></body></html>",
            }
        },
    }
    out = extract_body_like_content(raw)
    assert out["parse_success"] is True
    assert out["body_field_selected"] == "$.result.page.html"
    assert out["body_field_kind"] == "html"
    assert "Useful paragraph" in out["page_text"]


def test_extracts_root_html_string():
    raw = "<!doctype html><html><body><main>" + ("Static article content " * 35) + "</main></body></html>"
    out = extract_body_like_content(raw)
    assert out["parse_success"] is True
    assert out["body_field_selected"] == "$"
    assert out["body_field_kind"] == "html"


def test_detects_verification_page_as_not_successful():
    raw = {"data": {"html": "<html><title>Just a moment</title><body>Checking your browser before accessing this site. Cloudflare CAPTCHA.</body></html>"}}
    out = extract_body_like_content(raw)
    assert out["parse_success"] is False
    assert out["parse_error_category"] == "blocked_or_verification_page"
    assert out["content_quality_flag"] == "blocked_or_error_page"


def test_ordinary_content_using_blocked_word_is_not_verification_page():
    raw = {"text": "Sensitive skin symptoms can be triggered when repair pathways are blocked by irritation. " * 40}
    out = extract_body_like_content(raw)
    assert out["parse_success"] is True
    assert out["content_quality_flag"] == "ok"


def test_metadata_only_response_has_clear_category():
    raw = {
        "url": "https://example.com",
        "source_url": "https://www.reddit.com/r/example/comments/1234567890/very_long_slug_that_is_not_page_body/?utm_source=chatgpt.com",
        "status_code": 200,
        "title": "Example",
    }
    out = extract_body_like_content(raw)
    assert out["parse_success"] is False
    assert out["parse_error"] == "No body field detected"
    assert out["parse_error_category"] == "metadata_only_response"
