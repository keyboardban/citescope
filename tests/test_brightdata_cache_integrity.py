from __future__ import annotations

import json

from src.econometrics_eda_v2.brightdata_response_parser import (
    build_cache_integrity_audit,
    cache_integrity_row,
    parse_brightdata_cache_payload,
)


def test_raw_cache_with_raw_response_is_valid(tmp_path):
    path = tmp_path / "bd_001.raw.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "bd_001",
                "requested_url": "https://example.com",
                "provider": "brightdata",
                "provider_mode": "brightdata_browser_api",
                "raw_response": {"text": "Useful body text " * 40},
            }
        ),
        "utf-8",
    )
    row = cache_integrity_row(path)
    assert row["has_raw_response"] is True
    assert row["has_body_like_field"] is True
    assert row["can_reparse"] is True
    assert row["cache_status"] == "valid_raw_response_available"


def test_normalized_failed_result_without_raw_response_is_unrecoverable(tmp_path):
    path = tmp_path / "bd_002.json"
    path.write_text(
        json.dumps(
            {
                "benchmark_id": "bd_002",
                "requested_url": "https://example.com",
                "provider": "brightdata",
                "provider_status": "failed",
                "success": False,
                "error": "No body-like field found",
                "text": "",
                "html": "",
            }
        ),
        "utf-8",
    )
    row = cache_integrity_row(path)
    assert row["has_raw_response"] is False
    assert row["looks_like_normalized_failed_result"] is True
    assert row["can_reparse"] is False
    assert row["cache_status"] == "normalized_failure_only_unrecoverable"


def test_parser_reports_raw_response_missing_not_no_body_like_field():
    parsed = parse_brightdata_cache_payload(
        {
            "benchmark_id": "bd_003",
            "requested_url": "https://example.com",
            "provider": "brightdata",
            "provider_status": "failed",
            "success": False,
            "error": "No body-like field found",
        }
    )
    assert parsed["parse_success"] is False
    assert parsed["can_reparse"] is False
    assert parsed["parse_error_category"] == "raw_response_missing_from_cache"
    assert parsed["parse_error"] == "Old cache contains normalized failure only; cannot reparse"


def test_integrity_audit_marks_dry_run_only(tmp_path):
    (tmp_path / "bd_004.json").write_text(
        json.dumps(
            {
                "benchmark_id": "bd_004",
                "provider_status": "planned_dry_run",
                "error": "dry_run_no_api_call",
                "planned_request": {"payload": {"url": "https://example.com"}},
            }
        ),
        "utf-8",
    )
    audit = build_cache_integrity_audit(tmp_path)
    assert len(audit) == 1
    assert audit.iloc[0]["cache_status"] == "dry_run_only"
