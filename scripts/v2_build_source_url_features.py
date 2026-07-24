#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, read_csv, write_csv, write_json
from src.econometrics_eda_v2.url_features import build_source_url_features


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-rows", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    df, summary = build_source_url_features(read_csv(args.source_rows))
    write_csv(args.output, df)
    write_json(AUDIT_DIR / "source_url_features_summary.json", summary)
    print(
        f"Source URL features built: rows={summary['rows']} "
        f"page_type_url_seed_coverage={summary['page_type_url_seed_coverage']:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
