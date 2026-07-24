from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.diagnostics import export_econometrics_rows
from src.econometrics_eda_v2.url_features import build_source_url_features


def test_final_export_keeps_url_seed_scraped_and_final_page_type():
    source_rows = pd.DataFrame(
        [
            {
                "run_id": "run",
                "record_id": "r1",
                "prompt_id": "p1",
                "source_row_id": "s1",
                "normalized_url": "https://example.com/pricing",
                "source_url": "https://example.com/pricing",
                "source_domain": "example.com",
                "cited": 1,
                "intent": "info",
                "topic": "health",
                "language": "en",
                "country": "TH",
                "expected_source_types": "official_brand",
                "source_title": "Pricing",
                "source_description": "Package",
                "source_snippet": "",
                "source_position": 1,
                "observed_rank": 1,
                "cited_label": 1,
                "source_group": "cited",
                "source_origin": "citations",
            },
            {
                "run_id": "run",
                "record_id": "r1",
                "prompt_id": "p1",
                "source_row_id": "s2",
                "normalized_url": "https://example.com/about",
                "source_url": "https://example.com/about",
                "source_domain": "example.com",
                "cited": 0,
                "intent": "info",
                "topic": "health",
                "language": "en",
                "country": "TH",
                "expected_source_types": "official_brand",
                "source_title": "About",
                "source_description": "",
                "source_snippet": "",
                "source_position": 2,
                "observed_rank": 2,
                "cited_label": 0,
                "source_group": "more_only",
                "source_origin": "search_sources_more",
            },
        ]
    )
    url_features, _ = build_source_url_features(source_rows)
    page_parse = pd.DataFrame(
        [
            {
                "normalized_url": "https://example.com/pricing",
                "requested_normalized_url": "https://example.com/pricing",
                "final_normalized_url": "https://example.com/pricing",
                "scrape_success": True,
                "parse_success": True,
                "scraped_body_available": True,
                "word_count": 100,
                "heading_count": 1,
                "table_count": 0,
                "link_count": 2,
                "page_text": "pricing package FAQ contact",
            }
        ]
    )
    page_features = pd.DataFrame(
        [
            {
                "normalized_url": "https://example.com/pricing",
                "requested_normalized_url": "https://example.com/pricing",
                "final_normalized_url": "https://example.com/pricing",
                "content_feature_available": True,
                "content_feature_missing_reason": "",
                "page_type_scraped_enriched": "price_package_page",
                "has_faq": 1,
                "has_price_or_package": 1,
                "has_contact_info": 1,
            }
        ]
    )
    df, summary = export_econometrics_rows(source_rows, url_features, page_parse, page_features)
    assert df["cited"].nunique() == 2
    assert df["page_type_url_seed"].notna().mean() == 1
    assert df["page_type_final"].notna().mean() == 1
    assert set(df["page_type_final_source"]) == {"scraped_content", "url_seed"}
    assert summary["rows_with_page_type_final"] == 2
