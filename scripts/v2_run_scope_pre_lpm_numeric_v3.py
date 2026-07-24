#!/usr/bin/env python3
"""Generate detailed scatter-style pre-LPM numeric shape diagnostics for SCOPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.pre_lpm_numeric_shape_v3 import run_numeric_shape_diagnostics_v3

BASE = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpm", type=Path, default=BASE / "tables/final_lpm_prep/scope_condo_lpm_ready.csv")
    parser.add_argument("--eda", type=Path, default=BASE / "tables/scope_condo_eda_ready_post_scrape.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "tables/pre_lpm_eda_v3")
    parser.add_argument("--figure-dir", type=Path, default=BASE / "figures/pre_lpm_eda_v3")
    args = parser.parse_args(argv)
    if not args.lpm.exists():
        parser.error(f"LPM-ready CSV not found: {args.lpm}")
    print(json.dumps(run_numeric_shape_diagnostics_v3(args.lpm, args.eda, args.output_dir, args.figure_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
