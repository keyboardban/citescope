#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.diagnostics import write_eda_outputs
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--disable-lightgbm", action="store_true")
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    result = write_eda_outputs(read_csv(args.input), args.output_dir, enable_lightgbm=not args.disable_lightgbm)
    meta = result["metadata"]
    print(f"EDA complete: rows={meta['rows']} plots={meta['plot_count']} warnings={len(meta['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
