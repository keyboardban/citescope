from __future__ import annotations

import json

import pandas as pd

from src import econometrics_qa, storage


def test_snapshot_path_uses_source_url_hash(tmp_path):
    source_url = "https://example.com/page?utm_source=chatgpt.com"
    expected = tmp_path / "crawler_api" / f"{econometrics_qa.snapshot_key(source_url)}.json"
    expected.parent.mkdir(parents=True)
    expected.write_text(json.dumps({"normalized_url": "https://example.com/page"}), "utf-8")

    snapshot, path = econometrics_qa.load_snapshot(source_url, snapshot_root=tmp_path)

    assert path == expected
    assert snapshot["normalized_url"] == "https://example.com/page"


def test_classify_frame_policy():
    assert econometrics_qa.classify_frame_policy({"X-Frame-Options": "DENY"})[0] == "blocked"
    assert econometrics_qa.classify_frame_policy(
        {"Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'"}
    )[0] == "blocked"
    assert econometrics_qa.classify_frame_policy(
        {"Content-Security-Policy": "frame-ancestors *"}
    )[0] == "allowed"
    assert econometrics_qa.classify_frame_policy({})[0] == "unknown"


def test_rule_v2_taxonomy_is_additive_and_preserves_historical_labels():
    evidence = pd.DataFrame(
        [
            {
                "source_url": "https://example.com/search?q=condo",
                "normalized_url": "https://example.com/search?q=condo",
                "page_type_general": "listing_page",
                "page_type_url_seed_general": "listing_page",
                "content_quality_flag": "ok",
                "content_strength": "strong",
            }
        ]
    )

    result = econometrics_qa.add_general_taxonomy_v2(evidence)

    assert result.loc[0, "page_type_general"] == "listing_page"
    assert result.loc[0, "page_type_general_rule_v2"] == "search_results_page"
    assert result.loc[0, "page_type_family_general_rule_v2"] == "search_or_results"
    assert result.loc[0, "general_taxonomy_rule_version"] == "general_page_taxonomy_v2"


def test_gemini_taxonomy_is_additive_and_joined_by_normalized_url():
    evidence = pd.DataFrame(
        [{"normalized_url": "https://example.com/page", "page_type_general_rule_v2": "unknown"}]
    )
    gemini = pd.DataFrame(
        [
            {
                "normalized_url": "https://example.com/page",
                "llm_page_type_general": "product_page",
                "llm_page_type_family_general": "commercial_product_or_service",
                "llm_site_type_general": "official_company_or_brand",
                "llm_confidence": "medium",
                "llm_evidence": "URL and page body describe one project.",
                "result_valid": True,
            }
        ]
    )

    result = econometrics_qa.add_gemini_taxonomy(evidence, gemini)

    assert result.loc[0, "page_type_general_rule_v2"] == "unknown"
    assert result.loc[0, "llm_page_type_general"] == "product_page"
    assert result.loc[0, "llm_site_type_general"] == "official_company_or_brand"


def test_taxonomy_comparison_helpers_do_not_call_agreement_accuracy():
    evidence = pd.DataFrame(
        [
            {
                "page_type_general_rule_v2": "unknown",
                "llm_page_type_general": "product_page",
                "llm_confidence": "medium",
                "result_valid": True,
                "markdown_available": True,
            },
            {
                "page_type_general_rule_v2": "listing_page",
                "llm_page_type_general": "listing_page",
                "llm_confidence": "high",
                "result_valid": True,
                "markdown_available": False,
            },
            {
                "page_type_general_rule_v2": "blog_article",
                "llm_page_type_general": "guide_article",
                "llm_confidence": "high",
                "result_valid": True,
                "markdown_available": True,
            },
        ]
    )

    summary = econometrics_qa.taxonomy_comparison_summary(evidence)
    matrix = econometrics_qa.taxonomy_confusion_table(
        evidence,
        "page_type_general_rule_v2",
        "llm_page_type_general",
    )

    assert summary["rule_unknown_resolved"] == 1
    assert summary["known_exact_agreement_rate"] == 0.5
    assert summary["high_medium_confidence_disagreements"] == 1
    assert matrix["unique_urls"].sum() == 3


def test_econometrics_review_round_trip():
    storage.save_econometrics_review(
        {
            "normalized_url": "https://example.com/page",
            "source_url": "https://example.com/page?utm_source=chatgpt.com",
            "snapshot_key": "abc123",
            "review_status": "needs_rescrape",
            "scrape_completeness": "incomplete",
            "live_page_changed": True,
            "taxonomy_suggestion": "listing_page",
            "notes": "Main body is missing.",
        }
    )

    review = storage.get_econometrics_review("https://example.com/page")

    assert review["review_status"] == "needs_rescrape"
    assert review["live_page_changed"] is True
    assert len(storage.list_econometrics_reviews()) == 1
