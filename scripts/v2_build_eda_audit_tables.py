#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.eda_audit import build_eda_audit_tables
from src.econometrics_eda_v2.io import ensure_v2_dirs, read_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    result = build_eda_audit_tables(read_csv(args.rows), args.output_dir)
    print(f"EDA audit tables built: {result['tables_created']} tables -> {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
