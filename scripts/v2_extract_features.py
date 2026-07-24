#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.feature_extraction import extract_page_features
from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, read_csv, write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--source-rows", default=None)
    ap.add_argument("--page-parse", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    page_parse_path = args.page_parse or args.input
    if not page_parse_path:
        ap.error("--input or --page-parse is required")
    source_rows = read_csv(args.source_rows) if args.source_rows else None
    df, summary = extract_page_features(read_csv(page_parse_path), source_rows)
    write_csv(args.output, df)
    write_json(AUDIT_DIR / "page_feature_summary.json", summary)
    print(f"Extracted page features: rows={summary['rows']} coverage={summary['content_feature_coverage']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
