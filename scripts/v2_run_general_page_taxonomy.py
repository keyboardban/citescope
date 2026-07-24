#!/usr/bin/env python3
"""Generate the additive cross-domain general page taxonomy outputs."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.econometrics_eda_v2.general_page_taxonomy_pipeline import run_general_page_taxonomy
BASE = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded"
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lpm", type=Path, default=BASE / "tables/final_lpm_prep/scope_condo_lpm_ready.csv")
    parser.add_argument("--eda", type=Path, default=BASE / "tables/scope_condo_eda_ready_post_scrape.csv")
    parser.add_argument("--table-dir", type=Path, default=BASE / "tables/general_page_taxonomy")
    parser.add_argument("--figure-dir", type=Path, default=BASE / "figures/general_page_taxonomy")
    args = parser.parse_args()
    print(json.dumps(run_general_page_taxonomy(args.lpm, args.eda if args.eda.exists() else None, args.table_dir, args.figure_dir), indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
