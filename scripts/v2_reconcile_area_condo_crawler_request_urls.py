#!/usr/bin/env python3
"""Repair request-URL bookkeeping from cached async Crawler results without live calls."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.scrape_providers.brightdata_provider import prepare_brightdata_url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    detail_path = args.output_dir / "brightdata_content_pilot_url_results.csv"
    detail = pd.read_csv(detail_path, low_memory=False)
    if not detail["final_provider_mode"].fillna("").eq("brightdata_crawler_api").all():
        raise RuntimeError("Refusing to rewrite a mixed-provider pilot table.")
    detail["final_request_url"] = detail["source_url"].map(prepare_brightdata_url)
    detail["tracking_parameters_removed"] = detail["source_url"].str.strip().ne(detail["final_request_url"])
    detail.to_csv(detail_path, index=False)
    columns = ["source_url", "final_request_url", "tracking_parameters_removed", "final_provider_mode", "scrape_success", "content_strength", "scrape_error"]
    detail[columns].to_csv(args.output_dir / "tracking_parameter_request_audit.csv", index=False)
    print({"rows": len(detail), "tracking_parameters_removed": int(detail["tracking_parameters_removed"].sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
