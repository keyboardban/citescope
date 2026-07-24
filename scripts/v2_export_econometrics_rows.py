#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.diagnostics import export_econometrics_rows
from src.econometrics_eda_v2.io import ensure_v2_dirs, read_csv, write_csv, write_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-rows", required=True)
    ap.add_argument("--source-url-features", default=None)
    ap.add_argument("--page-parse", required=True)
    ap.add_argument("--page-features", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    source_url_features = read_csv(args.source_url_features) if args.source_url_features else None
    df, summary = export_econometrics_rows(
        read_csv(args.source_rows),
        source_url_features,
        read_csv(args.page_parse),
        read_csv(args.page_features),
    )
    write_csv(args.output, df)
    summary_path = Path(args.output).with_suffix(".summary.json")
    write_json(summary_path, summary)
    print(f"Econometrics rows exported: rows={summary['rows_exported']} cited_rate={summary['cited_rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
