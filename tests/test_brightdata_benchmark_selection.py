from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.brightdata_benchmark import select_brightdata_benchmark_urls


def _audit() -> pd.DataFrame:
    rows = []
    cases = [
        ("dynamic_js_likely", 5, 100, "unknown", "https://dyn.example/a"),
        ("parse_failed", 0, 0, "unknown", "https://parse.example/a"),
        ("empty_text", 0, 0, "unknown", "https://empty.example/a"),
        ("very_short_text", 12, 80, "unknown", "https://short.example/a"),
        ("boilerplate_only", 50, 200, "unknown", "https://boiler.example/a"),
        ("blocked_or_error_page", 3, 40, "unknown", "https://blocked.example/a"),
        ("ok", 500, 3000, "unknown", "https://ok.example/a"),
        ("parse_failed", 0, 0, "unknown", "https://pdf.example/file.pdf"),
    ]
    for i, (flag, wc, chars, page_type, url) in enumerate(cases):
        rows.append(
            {
                "source_url": url,
                "normalized_url": url,
                "source_root_domain": url.split("/")[2],
                "cited_rows_n": 1 if i < 6 else 0,
                "source_rows_n": 1,
                "scrape_success": flag not in {"parse_failed", "empty_text"},
                "parse_success": flag not in {"parse_failed", "empty_text"},
                "scraped_body_available": flag not in {"parse_failed", "empty_text"},
                "word_count": wc,
                "text_char_count": chars,
                "heading_count": 0,
                "table_count": 0,
                "page_title": "Access denied" if flag == "blocked_or_error_page" else "",
                "page_text_excerpt": "",
                "page_type_url_seed": "unknown",
                "page_type_scraped_enriched": "unknown",
                "page_type_final": page_type,
                "page_type_final_source": "scraped_content",
                "content_quality_flag": flag,
            }
        )
    return pd.DataFrame(rows)


def test_benchmark_input_has_no_duplicate_urls_and_pdf_not_sent_to_brightdata():
    out = select_brightdata_benchmark_urls(_audit(), max_urls=20)
    assert out["normalized_url"].is_unique
    assert not out["normalized_url"].str.endswith(".pdf").any()


def test_dynamic_and_blocked_modes_are_recommended_correctly():
    out = select_brightdata_benchmark_urls(_audit(), max_urls=20)
    by_flag = out.set_index("current_content_quality_flag")
    assert by_flag.loc["dynamic_js_likely", "recommended_brightdata_mode"] == "browser_api"
    assert by_flag.loc["blocked_or_error_page", "recommended_brightdata_mode"] == "unlocker_api"
