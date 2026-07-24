from __future__ import annotations

from src.econometrics_eda_v2.brightdata_response_parser import (
    build_parser_before_after,
    parse_brightdata_cache_payload,
)

import pandas as pd


def test_parse_payload_prefers_preserved_raw_response_body():
    payload = {
        "benchmark_id": "bd_001",
        "source_url": "https://example.com",
        "requested_url": "https://example.com",
        "provider": "brightdata",
        "provider_mode": "brightdata_browser_api",
        "provider_status": "failed",
        "error": "No body-like field found",
        "raw_response": {
            "status_code": 200,
            "data": {
                "content": "Recovered article body " * 80,
                "title": "Recovered",
            },
        },
    }
    parsed = parse_brightdata_cache_payload(payload)
    assert parsed["parse_success"] is True
    assert parsed["scrape_success"] is True
    assert parsed["raw_response_present"] is True
    assert parsed["word_count"] >= 200
    assert parsed["parse_error"] == ""


def test_parse_payload_marks_http_502_as_error_even_with_no_body():
    payload = {
        "benchmark_id": "bd_502",
        "source_url": "https://example.com",
        "requested_url": "https://example.com",
        "provider_status": "failed",
        "status_code": 502,
        "error": "HTTP 502",
        "raw_response": {"status_code": 502, "message": "Bad gateway"},
    }
    parsed = parse_brightdata_cache_payload(payload)
    assert parsed["parse_success"] is False
    assert parsed["scrape_success"] is False
    assert parsed["content_quality_flag"] == "blocked_or_error_page"
    assert parsed["parse_error_category"] == "unsupported_response_shape"


def test_before_after_marks_parser_fixed_rows():
    before = pd.DataFrame([{"benchmark_id": "bd_001", "parse_success": False, "word_count": 0, "parse_error": "No body-like field found"}])
    after = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_001",
                "source_url": "https://example.com",
                "parse_success": True,
                "scrape_success": True,
                "word_count": 160,
                "parse_error": "",
                "parse_error_category": "",
                "response_shape": "nested_response_wrapper",
                "body_field_selected": "$.data.content",
            }
        ]
    )
    out = build_parser_before_after(before, after)
    assert bool(out.iloc[0]["fixed_by_parser"]) is True
