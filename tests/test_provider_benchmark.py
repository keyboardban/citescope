from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.brightdata_benchmark import (
    benchmark_family,
    mix_benchmark_order,
)
from src.econometrics_eda_v2.provider_benchmark import (
    build_page_type_comparison,
    build_strategy_recommendations,
    compare_providers_with_quality,
    select_provider_benchmark_urls,
)


def _rows(n: int = 40):
    final = []
    quality = []
    for i in range(n):
        nurl = f"https://example{i}.com/page"
        if i < 12:
            page_type = "unknown"
            family = "unknown"
            flag = "parse_failed"
            wc = 0
            scrape = False
            parse = False
            body = False
        elif i < 24:
            page_type = "unknown"
            family = "unknown"
            flag = "ok"
            wc = 900
            scrape = True
            parse = True
            body = True
        elif i < 34:
            page_type = "product_marketplace_page"
            family = "commercial_price_package"
            flag = "ok"
            wc = 450
            scrape = True
            parse = True
            body = True
        else:
            page_type = "article_health_info"
            family = "information_content"
            flag = "ok"
            wc = 700
            scrape = True
            parse = True
            body = True
        final.append(
            {
                "normalized_url": nurl,
                "source_url": nurl + "?utm_source=chatgpt.com",
                "source_root_domain": f"example{i}.com",
                "cited": 1 if i >= 34 else 0,
                "source_position": i + 1,
                "page_type_final": page_type,
                "page_type_family": family,
            }
        )
        quality.append(
            {
                "normalized_url": nurl,
                "scrape_success": scrape,
                "parse_success": parse,
                "scraped_body_available": body,
                "word_count": wc,
                "heading_count": 1 if body else 0,
                "table_count": 0,
                "content_quality_flag": flag,
                "page_text_excerpt": "body text" if body else "",
            }
        )
    return pd.DataFrame(final), pd.DataFrame(quality)


def test_select_provider_benchmark_urls_builds_requested_reason_buckets():
    final, quality = _rows()
    out = select_provider_benchmark_urls(final, quality)
    assert len(out) == 40
    assert out["normalized_url"].is_unique
    counts = out["reason_selected"].value_counts().to_dict()
    assert counts["unknown_low_word_count"] == 10
    assert counts["unknown_quality_ok"] == 10
    assert counts["parse_failed_no_body_or_dynamic"] >= 2
    assert counts["ecommerce_marketplace_product"] == 5
    assert counts["high_impact_cited"] == 5


def test_compare_providers_flags_brightdata_parse_fix_and_strategy():
    final, quality = _rows(5)
    benchmark = select_provider_benchmark_urls(final, quality, total=5).head(1)
    bright = pd.DataFrame(
        [
            {
                "benchmark_id": benchmark.iloc[0]["benchmark_id"],
                "source_url": benchmark.iloc[0]["source_url"],
                "requested_url": benchmark.iloc[0]["normalized_url"],
                "final_url": benchmark.iloc[0]["normalized_url"],
                "normalized_url": benchmark.iloc[0]["normalized_url"],
                "provider": "brightdata",
                "scrape_success": True,
                "parse_success": True,
                "scraped_body_available": True,
                "page_title": "Article",
                "meta_description": "",
                "page_text": "Article " + ("health information " * 80),
                "word_count": 161,
                "heading_count": 1,
                "table_count": 0,
                "content_quality_flag": "ok",
                "page_text_excerpt": "Article health information",
            }
        ]
    )
    results, summary = compare_providers_with_quality(benchmark, bright, quality, final)
    assert bool(results.iloc[0]["brightdata_fixed_parse_failure"]) is True
    assert results.iloc[0]["recommended_provider_for_url"] == "brightdata"
    assert summary.set_index("provider").loc["brightdata", "attempted"] == 1
    page_types = build_page_type_comparison(results, bright)
    strategy = build_strategy_recommendations(results, page_types)
    assert "apify_then_brightdata_fallback" in set(strategy["strategy"])


def _mixed_input():
    specs = [
        # (benchmark_id, domain, mode, reason, page_type, family, expected_family)
        ("bd_bench_001", "reddit.com", "unlocker_api", "dynamic_js_likely", "forum_review_page", "user_generated", "reddit_or_blocked"),
        ("bd_bench_002", "reddit.com", "unlocker_api", "dynamic_js_likely", "forum_review_page", "user_generated", "reddit_or_blocked"),
        ("bd_bench_003", "byrdie.com", "unlocker_api", "blocked_or_captcha", "unknown", "unknown", "reddit_or_blocked"),
        ("bd_bench_004", "jmir.org", "browser_api", "parse_failed_empty_or_no_body", "unknown", "unknown", "article_institutional"),
        ("bd_bench_005", "moph.go.th", "browser_api", "very_short_or_boilerplate", "news_announcement_page", "news_or_update", "article_institutional"),
        ("bd_bench_006", "alibaba.com", "browser_api", "parse_failed_empty_or_no_body", "price_package_page", "commercial_price_package", "ecommerce_product"),
        ("bd_bench_007", "bigc.co.th", "browser_api", "parse_failed_empty_or_no_body", "unknown", "unknown", "ecommerce_product"),
        ("bd_bench_008", "example.com", "browser_api", "parse_failed_empty_or_no_body", "unknown", "unknown", "parse_failed_other"),
    ]
    cols = ["benchmark_id", "source_root_domain", "recommended_brightdata_mode", "reason_selected", "current_page_type_final", "current_page_type_family"]
    df = pd.DataFrame([dict(zip(cols, s[:6])) for s in specs])
    df["source_url"] = df["benchmark_id"] + "-url"
    df["normalized_url"] = df["benchmark_id"] + "-url"
    expected = {s[0]: s[6] for s in specs}
    return df, expected


def test_benchmark_family_assigns_expected_buckets():
    df, expected = _mixed_input()
    for _, row in df.iterrows():
        assert benchmark_family(row) == expected[row["benchmark_id"]]


def test_mix_benchmark_order_preserves_ids_and_spreads_first_rows():
    df, _ = _mixed_input()
    mixed = mix_benchmark_order(df)
    # No row lost, no id renumbered, every original column preserved.
    assert set(mixed["benchmark_id"]) == set(df["benchmark_id"])
    assert len(mixed) == len(df)
    assert set(df.columns).issubset(set(mixed.columns))
    # First 4 rows must not be dominated by one family (round-robin spread).
    assert mixed["family"].head(4).nunique() >= 3
    # Leading family follows the configured cycle head.
    assert mixed["family"].iloc[0] == "article_institutional"


def test_mix_benchmark_order_handles_empty():
    empty = pd.DataFrame(columns=["benchmark_id", "source_root_domain", "recommended_brightdata_mode"])
    mixed = mix_benchmark_order(empty)
    assert len(mixed) == 0
    assert "family" in mixed.columns
