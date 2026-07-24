from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.feature_extraction import classify_page_type, extract_page_features


def test_feature_detection_flags():
    parse = pd.DataFrame(
        [
            {
                "scrape_id": "s1",
                "normalized_url": "https://hospital.test/package",
                "requested_normalized_url": "https://hospital.test/package",
                "final_normalized_url": "https://hospital.test/package",
                "requested_url": "https://hospital.test/package",
                "final_url": "https://hospital.test/package",
                "domain": "hospital.test",
                "scrape_success": True,
                "scraped_body_available": True,
                "table_count": 1,
                "page_title": "FAQ price package contact",
                "meta_description": "",
                "page_text": "FAQ price package contact us phone test@example.com appointment references updated",
            }
        ]
    )
    df, summary = extract_page_features(parse)
    row = df.iloc[0]
    assert row["has_faq"] == 1
    assert row["has_price_or_package"] == 1
    assert row["has_contact_info"] == 1
    assert row["page_type_scraped_enriched"] in {"faq_page", "price_package_page", "contact_page"}
    assert summary["content_feature_available"] == 1


def test_page_type_classifier_uses_page_evidence_only():
    row_a = {
        "final_url": "https://example.com/pricing",
        "page_title": "Package price",
        "meta_description": "",
        "page_text": "The service package price is listed here.",
        "cited": 1,
        "answer_text": "answer",
        "intent": "ignored",
    }
    row_b = dict(row_a, cited=0, answer_text="different", intent="other")
    assert classify_page_type(row_a) == classify_page_type(row_b)
