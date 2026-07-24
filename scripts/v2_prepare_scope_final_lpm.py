#!/usr/bin/env python3
"""Create conservative, leakage-free final-LPM preparation outputs for SCOPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.final_lpm_prep import run_final_lpm_prep


DEFAULT_INPUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/tables/scope_condo_eda_ready_post_scrape.csv"
DEFAULT_OUTPUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/tables/final_lpm_prep"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"EDA-ready CSV not found: {args.input}")
    print(json.dumps(run_final_lpm_prep(args.input, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
