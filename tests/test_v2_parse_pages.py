from __future__ import annotations

import json

from src.econometrics_eda_v2.parse_pages import parse_raw_cache


def test_parse_html_text_markdown_counts(tmp_path):
    path = tmp_path / "s1.json"
    path.write_text(
        json.dumps(
            {
                "scrape_id": "s1",
                "requested_url": "https://example.com",
                "final_url": "https://example.com",
                "provider_status": "success",
                "html": "<h1>Hello</h1><p>one two three</p><table></table><a href='/x'>x</a>",
            }
        ),
        "utf-8",
    )
    row = parse_raw_cache(path)
    assert row["scraped_body_available"] is True
    assert row["word_count"] >= 4
    assert row["heading_count"] == 1
    assert row["table_count"] == 1


def test_scraped_body_available_false_without_body_fields(tmp_path):
    path = tmp_path / "s2.json"
    path.write_text(json.dumps({"scrape_id": "s2", "requested_url": "https://example.com", "provider_status": "success", "title": "Only metadata"}), "utf-8")
    row = parse_raw_cache(path)
    assert row["scraped_body_available"] is False
    assert row["parse_success"] is False
