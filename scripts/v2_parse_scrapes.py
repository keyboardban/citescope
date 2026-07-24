#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, write_csv, write_json
from src.econometrics_eda_v2.parse_pages import parse_scrape_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    df, summary = parse_scrape_dir(args.input_dir)
    write_csv(args.output, df)
    write_json(AUDIT_DIR / "page_parse_summary.json", summary)
    print(f"Parsed scrapes: rows={summary['rows']} body={summary['rows_with_scraped_body']} parse_success={summary['parse_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
