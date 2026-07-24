#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, read_csv, write_csv, write_json
from src.econometrics_eda_v2.scrape_queue import build_scrape_queue


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--force-rescrape", action="store_true")
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    df, summary = build_scrape_queue(read_csv(args.input), force_rescrape=args.force_rescrape)
    write_csv(args.output, df)
    write_json(AUDIT_DIR / "scrape_queue_summary.json", summary)
    print(f"Scrape queue built: urls={summary['rows']} should_scrape={summary['should_scrape']} cached={summary['cached_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
