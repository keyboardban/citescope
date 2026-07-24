#!/usr/bin/env python3
"""Independently recompute and audit SCOPE condo EDA-ready metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.metric_recheck import run_metric_recheck


DEFAULT_INPUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/tables/scope_condo_eda_ready_post_scrape.csv"
FALLBACK_INPUT = Path.home() / "Downloads/scope_condo_eda_ready_post_scrape.csv"
DEFAULT_OUTPUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/tables/metric_recheck"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    input_path = args.input if args.input.exists() else FALLBACK_INPUT
    if not input_path.exists():
        parser.error(f"EDA-ready CSV not found at {args.input} or {FALLBACK_INPUT}")
    summary = run_metric_recheck(input_path, args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
