#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv
from src.econometrics_eda_v2.provider_benchmark import (
    build_domain_strategy,
    build_page_type_comparison,
    build_strategy_recommendations,
    compare_providers,
    compare_providers_with_quality,
)


def _read_if_exists(path: str | Path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return __import__("pandas").DataFrame()
    return read_csv(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-input", default=str(OUTPUT_DIR / "tables" / "provider_benchmark_input_urls.csv"))
    ap.add_argument("--brightdata-parse", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_parse_rows.csv"))
    ap.add_argument("--apify-parse", default="data/econometrics_v2/scrape_cache/parsed/page_parse_rows.csv")
    ap.add_argument("--apify-quality", default=str(OUTPUT_DIR / "tables" / "scrape_quality_audit.csv"))
    ap.add_argument("--final-rows", default="data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "tables" / "provider_scrape_benchmark_results.csv"))
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    benchmark = read_csv(args.benchmark_input)
    bright = _read_if_exists(args.brightdata_parse)
    final_rows = read_csv(args.final_rows)
    if Path(args.apify_quality).exists():
        results, summary = compare_providers_with_quality(benchmark, bright, read_csv(args.apify_quality), final_rows)
    else:
        results, summary = compare_providers(benchmark, bright, read_csv(args.apify_parse), final_rows)
    page_types = build_page_type_comparison(results, bright)
    strategy = build_strategy_recommendations(results, page_types)
    domain = build_domain_strategy(results)
    output = Path(args.output)
    write_csv(output, results)
    write_csv(OUTPUT_DIR / "tables" / "provider_scrape_benchmark_summary.csv", summary)
    write_csv(OUTPUT_DIR / "tables" / "provider_page_type_comparison.csv", page_types)
    write_csv(OUTPUT_DIR / "tables" / "provider_strategy_recommendation.csv", strategy)
    write_csv(OUTPUT_DIR / "tables" / "provider_strategy_by_domain.csv", domain)
    print(
        "Provider comparison complete: "
        f"rows={len(results)} brightdata_attempted={int(results['brightdata_scrape_success'].notna().sum()) if len(results) else 0}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
