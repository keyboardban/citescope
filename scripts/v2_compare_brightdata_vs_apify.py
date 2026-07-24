#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv
from src.econometrics_eda_v2.provider_benchmark import build_page_type_comparison, compare_providers_with_quality


def _read_if_exists(path: str | Path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return __import__("pandas").DataFrame()
    return read_csv(p)


def _strategy_recommendation(results, page_types):
    import pandas as pd

    attempted = int(results["brightdata_scrape_success"].notna().sum()) if len(results) else 0
    wins = int(results["recommended_provider_for_url"].eq("brightdata").sum()) if len(results) else 0
    resolved = int(page_types["brightdata_resolved_unknown"].sum()) if len(page_types) else 0
    strong = attempted > 0 and wins / max(attempted, 1) >= 0.7 and resolved >= 5
    any_browser_win = wins > 0
    any_unlocker_win = wins > 0
    rows = [
        ("keep_apify_only", attempted == 0 or wins == 0, "No live Bright Data wins are available yet." if attempted == 0 else "Apify remains preferred unless Bright Data clearly wins."),
        ("apify_then_playwright_fallback", True, "Use rendered Apify first for JS-heavy/static extraction misses before switching providers."),
        ("apify_then_brightdata_browser_fallback", bool(any_browser_win), "Use only if Browser API benchmark fixes parse/short/dynamic pages."),
        ("apify_then_brightdata_unlocker_fallback", bool(any_unlocker_win), "Use only if Unlocker benchmark fixes blocked/captcha pages."),
        ("brightdata_for_problematic_domains_only", bool(wins >= 3), "Use only for domains with repeated Bright Data wins."),
        ("brightdata_primary_for_all", bool(strong), "Do not recommend unless Bright Data strongly outperforms Apify across the benchmark."),
    ]
    return pd.DataFrame(rows, columns=["strategy", "recommended", "reason"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--brightdata-parse", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_parse_rows.csv"))
    ap.add_argument("--scrape-audit", default=str(OUTPUT_DIR / "tables" / "scrape_quality_audit.csv"))
    ap.add_argument("--apify-audit", default=None, help="Alias for --scrape-audit.")
    ap.add_argument("--final-rows", default="data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "tables"), help="Directory for provider comparison outputs.")
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = read_csv(args.benchmark_input)
    bright = _read_if_exists(args.brightdata_parse)
    quality = read_csv(args.apify_audit or args.scrape_audit)
    final_rows = read_csv(args.final_rows)
    results, summary = compare_providers_with_quality(benchmark, bright, quality, final_rows)
    page_types = build_page_type_comparison(results, bright)
    strategy = _strategy_recommendation(results, page_types)
    write_csv(out_dir / "provider_scrape_benchmark_results.csv", results)
    write_csv(out_dir / "provider_scrape_benchmark_summary.csv", summary)
    write_csv(out_dir / "provider_page_type_comparison.csv", page_types)
    write_csv(out_dir / "provider_strategy_recommendation.csv", strategy)
    print(f"Bright Data vs Apify comparison complete: rows={len(results)} brightdata_attempted={int(results['brightdata_scrape_success'].notna().sum()) if len(results) else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
