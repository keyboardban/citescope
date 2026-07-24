#!/usr/bin/env python3
"""Generate descriptive SCOPE condo pre-LPM diagnostics; no model is fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.pre_lpm_diagnostics import run_pre_lpm_diagnostics

BASE = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded"
DEFAULT_LPM = BASE / "tables/final_lpm_prep/scope_condo_lpm_ready.csv"
DEFAULT_EDA = BASE / "tables/scope_condo_eda_ready_post_scrape.csv"
DEFAULT_TABLES = BASE / "tables/pre_lpm_eda"
DEFAULT_FIGURES = BASE / "figures/pre_lpm_eda"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpm", type=Path, default=DEFAULT_LPM)
    parser.add_argument("--eda", type=Path, default=DEFAULT_EDA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args(argv)
    input_path = args.lpm if args.lpm.exists() else args.eda
    if not input_path.exists():
        parser.error(f"Neither LPM-ready nor EDA-ready CSV exists: {args.lpm}; {args.eda}")
    summary = run_pre_lpm_diagnostics(input_path, args.eda if args.eda.exists() else None, args.output_dir, args.figure_dir)
    summary["input_path"] = str(input_path)
    summary["input_mode"] = "lpm_ready" if input_path == args.lpm else "eda_ready_fallback"
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
