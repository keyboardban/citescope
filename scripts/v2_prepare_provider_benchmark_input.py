#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs
from src.econometrics_eda_v2.provider_benchmark import write_provider_benchmark_input


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-rows", default="data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    ap.add_argument("--quality", default="outputs/econometrics_eda_v2/tables/scrape_quality_audit.csv")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "tables" / "provider_benchmark_input_urls.csv"))
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    df = write_provider_benchmark_input(args.final_rows, args.quality, args.output)
    print(f"Provider benchmark input written: rows={len(df)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
