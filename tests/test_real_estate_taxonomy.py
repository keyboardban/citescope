from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.general_page_taxonomy import classify_general_page_type
from src.econometrics_eda_v2.real_estate_taxonomy import (
    classify_real_estate_page_type,
    classify_source_type_real_estate,
    finalise_real_estate_page_type,
)


def _row(url: str, title: str = "", text: str = "", quality: str = "ok") -> dict:
    return {
        "source_url": url,
        "source_root_domain": url.split("/")[2].removeprefix("www."),
        "source_title": title,
        "page_title": title,
        "page_text": text,
        "content_quality_flag": quality,
    }


def test_real_estate_source_taxonomy_prefers_known_domains():
    assert classify_source_type_real_estate("https://www.ddproperty.com/en/property-for-rent/7") == "property_portal"
    assert classify_source_type_real_estate("https://www.superagent.co/en/condo/28-chidlom") == "broker_agency"
    assert classify_source_type_real_estate("https://scopethonglor.com/") == "project_official"
    assert classify_source_type_real_estate("https://www.reddit.com/r/Bangkok/comments/1") == "social_forum"
    assert classify_source_type_real_estate("https://www.youtube.com/watch?v=abc") == "video_platform"
    assert classify_source_type_real_estate("https://example.go.th/regulation.pdf") == "pdf_document"


def test_listing_signal_beats_location_signal_for_portal_page():
    row = _row(
        "https://www.ddproperty.com/en/condo-for-rent/at-q-langsuan-222",
        "Q Langsuan Condo Rent",
    )
    result = classify_real_estate_page_type(row, source_type="property_portal")
    assert result.detail == "rental_listing_page"
    assert result.confidence in {"high", "medium"}


def test_bad_scrape_cannot_overwrite_a_useful_seed():
    row = _row(
        "https://www.ddproperty.com/en/condo-for-sale/at-q-langsuan-222",
        "Q Langsuan Condo Sale",
    )
    seed = classify_real_estate_page_type(row, source_type="property_portal")
    weak = classify_real_estate_page_type({**row, "page_text": ""}, source_type="property_portal", include_content=True)
    final, source = finalise_real_estate_page_type(seed, weak, "parse_failed")
    assert final.detail == "resale_listing_page"
    assert source in {"domain_rule", "url_seed"}


def test_taxonomy_does_not_use_citation_outcome():
    row = _row("https://www.superagent.co/en/blog/buying-condo-checklist", "Buying Condo Checklist")
    cited = classify_real_estate_page_type({**row, "cited": 1, "answer_text": "ignore me"}, source_type="broker_agency")
    more_only = classify_real_estate_page_type({**row, "cited": 0, "answer_text": "different answer"}, source_type="broker_agency")
    assert cited == more_only


def test_usable_structured_data_supports_page_function_classification():
    general = classify_general_page_type(
        {"source_url": "https://example.com/post/1", "structured_data_types": "Article"},
        include_content=True,
    )
    assert general.detail == "blog_article"

    real_estate = classify_real_estate_page_type(
        {
            **_row("https://www.ddproperty.com/property/123", "Condo details"),
            "structured_data_types": "Product; Offer",
        },
        source_type="property_portal",
        include_content=True,
    )
    assert real_estate.detail == "project_listing_page"


def test_scope_taxonomy_runner_writes_required_outputs(tmp_path: Path):
    module_path = Path(__file__).parents[1] / "scripts" / "v2_apply_scope_real_estate_taxonomy.py"
    spec = importlib.util.spec_from_file_location("scope_taxonomy_runner", module_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    sources = pd.DataFrame(
        [
            {
                "normalized_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "source_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "source_root_domain": "ddproperty.com",
                "source_title": "Q Langsuan Condo Rent",
                "source_type_url": "unknown",
                "cited": 1,
                "source_position": 1,
                "answer_text": "answer should not reach EDA output",
            }
        ]
    )
    audit = pd.DataFrame(
        [
            {
                "normalized_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "source_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "source_root_domain": "ddproperty.com",
                "scraped_body_available": True,
                "scrape_success": True,
                "parse_success": True,
                "word_count": 500,
                "content_quality_flag": "ok",
                "page_title": "Q Langsuan Condo Rent",
                "page_text_excerpt": "A condo for rent near BTS Langsuan.",
                "page_type_final": "article_health_info",
                "page_type_family": "information_content",
            }
        ]
    )
    parse = pd.DataFrame(
        [
            {
                "requested_normalized_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "normalized_url": "https://www.ddproperty.com/en/condo-for-rent/q-langsuan",
                "page_title": "Q Langsuan Condo Rent",
                "meta_description": "Rental property listing",
                "page_text": "Condo for rent. Rent 80,000 THB per month.",
                "heading_count": 1,
                "table_count": 1,
            }
        ]
    )
    source_path = tmp_path / "sources.csv"
    audit_path = tmp_path / "audit.csv"
    parse_path = tmp_path / "parse.csv"
    sources.to_csv(source_path, index=False)
    audit.to_csv(audit_path, index=False)
    parse.to_csv(parse_path, index=False)
    result = runner.run(source_path, audit_path, parse_path, tmp_path / "features.csv", tmp_path)

    assert result["validation_passed"]
    for name in [
        "scope_taxonomy_error_audit.csv",
        "scope_condo_sources_with_real_estate_taxonomy.csv",
        "scope_real_estate_taxonomy_summary.csv",
        "scope_real_estate_taxonomy_review_sample.csv",
        "scope_real_estate_taxonomy_validation.json",
        "scope_condo_eda_ready_with_real_estate_taxonomy.csv",
    ]:
        assert (tmp_path / name).exists()
    eda = pd.read_csv(tmp_path / "scope_condo_eda_ready_with_real_estate_taxonomy.csv")
    assert "page_type_family_real_estate" in eda
    assert "source_position" not in eda
    assert "answer_text" not in eda
