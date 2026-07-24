#!/usr/bin/env python3
"""Reorder the Bright Data benchmark input into a family-mixed smoke queue.

Keeps every original ``benchmark_id`` (so the raw-cache mapping and any already
successful cached scrape stay valid) and only changes row order, round-robin
across coarse page families. Running the benchmark on the mixed file with
``--max-urls 10`` then yields a spread (article/institutional, Reddit/blocked,
parse-failed, ecommerce) instead of a Reddit-heavy first batch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_benchmark import mix_benchmark_order
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls_mixed.csv"))
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    mixed = mix_benchmark_order(read_csv(args.input))
    write_csv(args.out, mixed)
    print(f"Mixed Bright Data benchmark written: rows={len(mixed)} output={args.out}")
    print("Family order (round-robin):")
    print(mixed["family"].value_counts().to_string() if len(mixed) else "none")
    head = min(10, len(mixed))
    print(f"\nFirst {head} (what --max-urls {head} would attempt):")
    cols = [c for c in ["mixed_rank", "benchmark_id", "family", "recommended_brightdata_mode", "source_root_domain"] if c in mixed.columns]
    print(mixed.head(head)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
