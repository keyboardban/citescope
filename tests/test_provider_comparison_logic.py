from __future__ import annotations

import pandas as pd

from scripts.v2_compare_brightdata_vs_apify import _strategy_recommendation
from src.econometrics_eda_v2.provider_benchmark import build_page_type_comparison, compare_providers_with_quality


def test_brightdata_not_better_solely_because_word_count_is_longer():
    benchmark = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_001",
                "source_url": "https://example.com",
                "normalized_url": "https://example.com",
                "source_root_domain": "example.com",
                "reason_selected": "dynamic_js_likely",
            }
        ]
    )
    quality = pd.DataFrame(
        [
            {
                "normalized_url": "https://example.com",
                "scrape_success": True,
                "parse_success": True,
                "scraped_body_available": True,
                "word_count": 500,
                "heading_count": 2,
                "table_count": 0,
                "content_quality_flag": "ok",
                "page_text_excerpt": "Useful article content",
            }
        ]
    )
    final = pd.DataFrame([{"normalized_url": "https://example.com", "page_type_final": "article_health_info", "page_type_family": "information_content"}])
    bright = pd.DataFrame(
        [
            {
                "benchmark_id": "bd_001",
                "source_url": "https://example.com",
                "requested_url": "https://example.com",
                "final_url": "https://example.com",
                "normalized_url": "https://example.com",
                "provider": "brightdata",
                "scrape_success": True,
                "parse_success": True,
                "scraped_body_available": True,
                "word_count": 900,
                "heading_count": 1,
                "table_count": 0,
                "content_quality_flag": "ok",
                "page_text_excerpt": "Menu privacy terms cookie login subscribe footer repeated",
                "page_text": "Menu privacy terms cookie login subscribe footer " * 120,
            }
        ]
    )
    results, _ = compare_providers_with_quality(benchmark, bright, quality, final)
    assert bool(results.iloc[0]["brightdata_better_text"]) is False
    assert results.iloc[0]["recommended_provider_for_url"] == "apify"


def test_strategy_does_not_default_to_brightdata_primary():
    results = pd.DataFrame(
        [
            {
                "brightdata_scrape_success": pd.NA,
                "recommended_provider_for_url": "not_evaluated",
            }
        ]
    )
    page_types = pd.DataFrame([{"brightdata_resolved_unknown": False}])
    strategy = _strategy_recommendation(results, page_types)
    primary = strategy[strategy["strategy"].eq("brightdata_primary_for_all")].iloc[0]
    assert bool(primary["recommended"]) is False


def test_existing_apify_pipeline_mode_names_remain_available():
    from src.econometrics_eda_v2.scrape_providers.apify_provider import crawler_type_to_provider_mode

    assert crawler_type_to_provider_mode("cheerio") == "apify_cheerio"
    assert crawler_type_to_provider_mode("playwright:adaptive") == "apify_playwright_adaptive"
    assert crawler_type_to_provider_mode("playwright:firefox") == "apify_playwright_firefox"
