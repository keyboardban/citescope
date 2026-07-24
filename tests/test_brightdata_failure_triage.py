from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.brightdata_response_parser import build_failure_triage, retry_queue_frames


def test_triage_routes_missing_original_raw_response_to_payload_fix():
    rows = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_001",
                "source_url": "https://example.com",
                "normalized_url": "https://example.com",
                "domain": "example.com",
                "provider_mode": "brightdata_unlocker_api",
                "live_attempted": True,
                "status_code": 200,
                "scrape_success": False,
                "parse_success": False,
                "scraped_body_available": False,
                "word_count": 0,
                "content_quality_flag": "parse_failed",
                "parse_error": "No body-like field found",
                "parse_error_category": "metadata_only_response",
                "response_shape": "metadata_only_wrapper",
                "raw_response_present": False,
                "body_field_selected": "",
            }
        ]
    )
    triage = build_failure_triage(rows)
    assert triage.iloc[0]["failure_bucket"] == "raw_response_not_preserved"
    assert triage.iloc[0]["retry_queue"] == "payload_fix"
    assert len(retry_queue_frames(triage)["payload_fix"]) == 1


def test_triage_routes_browser_verification_to_unlocker_queue():
    rows = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_002",
                "source_url": "https://example.com",
                "normalized_url": "https://example.com",
                "domain": "example.com",
                "provider_mode": "brightdata_browser_api",
                "live_attempted": True,
                "status_code": 403,
                "scrape_success": False,
                "parse_success": False,
                "scraped_body_available": True,
                "word_count": 8,
                "content_quality_flag": "blocked_or_error_page",
                "parse_error": "Blocked or verification page",
                "parse_error_category": "blocked_or_verification_page",
                "response_shape": "body_fields_at_root",
                "raw_response_present": True,
                "body_field_selected": "$.html",
            }
        ]
    )
    triage = build_failure_triage(rows)
    assert triage.iloc[0]["failure_bucket"] == "blocked_or_verification"
    assert triage.iloc[0]["retry_queue"] == "unlocker_api"
    assert len(retry_queue_frames(triage)["unlocker_api"]) == 1


def test_triage_routes_request_validation_400_to_payload_fix():
    rows = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_400",
                "source_url": "https://example.com",
                "normalized_url": "https://example.com",
                "domain": "example.com",
                "provider_mode": "brightdata_browser_api",
                "live_attempted": True,
                "status_code": 400,
                "scrape_success": False,
                "parse_success": False,
                "scraped_body_available": False,
                "word_count": 0,
                "content_quality_flag": "blocked_or_error_page",
                "parse_error": "Request validation failed",
                "parse_error_category": "request_validation_failed",
                "response_shape": "body_fields_at_root",
                "raw_response_present": False,
                "body_field_selected": "",
            }
        ]
    )
    triage = build_failure_triage(rows)
    assert triage.iloc[0]["failure_bucket"] == "request_or_provider_http_error"
    assert triage.iloc[0]["retry_queue"] == "payload_fix"
