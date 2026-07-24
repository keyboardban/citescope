#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, write_csv, write_json
from src.econometrics_eda_v2.normalize_sources import build_source_rows_from_files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-json", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    df, summary = build_source_rows_from_files(args.ai_json, args.manifest)
    write_csv(args.output, df)
    write_json(AUDIT_DIR / "source_rows_summary.json", summary)
    print(f"Source rows built: rows={summary['rows']} cited={summary['cited_count']} more_only={summary['more_only_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
