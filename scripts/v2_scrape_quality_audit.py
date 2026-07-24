#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv
from src.econometrics_eda_v2.scrape_quality_audit import write_scrape_quality_outputs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-rows", default="data/econometrics_v2/exports/source_rows_raw.csv")
    ap.add_argument("--queue", default="data/econometrics_v2/scrape_queue/scrape_queue.csv")
    ap.add_argument("--parsed", default="data/econometrics_v2/scrape_cache/parsed/page_parse_rows.csv")
    ap.add_argument("--page-features", default="data/econometrics_v2/exports/page_features.csv")
    ap.add_argument("--final-rows", default="data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    summary = write_scrape_quality_outputs(
        read_csv(args.source_rows),
        read_csv(args.queue),
        read_csv(args.parsed),
        read_csv(args.page_features),
        read_csv(args.final_rows),
        args.output_dir,
    )
    print(
        "Scrape quality audit complete: "
        f"urls={summary['scrape_quality_rows']} "
        f"unknown_sample={summary['unknown_sample_rows']} "
        f"crawler_plan_rows={summary['crawler_plan_rows']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
