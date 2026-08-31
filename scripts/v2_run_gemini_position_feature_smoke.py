#!/usr/bin/env python3
"""Run a small Gemini semantic position-feature smoke test and frontend export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.gemini_position_features import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    FRONTEND_DATA_DIR,
    default_document_features_path,
    run_gemini_position_smoke,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position-urls",
        type=Path,
        default=ROOT / "outputs/position_feature_eda_final_20260731/data/url_position_features.csv",
    )
    parser.add_argument("--document-features", type=Path, default=default_document_features_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frontend-data-dir", type=Path, default=FRONTEND_DATA_DIR)
    parser.add_argument("--max-urls", type=int, default=12)
    parser.add_argument(
        "--all-urls",
        action="store_true",
        help="Run every position-feature-measurable URL with durable per-chunk checkpoints.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--execute-live-gemini", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.all_urls and (args.max_urls < 1 or args.max_urls > 50):
        parser.error("--max-urls must be between 1 and 50 for the smoke runner")
    result = run_gemini_position_smoke(
        position_urls_path=args.position_urls,
        document_features_path=args.document_features,
        output_dir=args.output_dir,
        frontend_data_dir=args.frontend_data_dir,
        max_urls=None if args.all_urls else args.max_urls,
        model=args.model,
        execute_live=args.execute_live_gemini,
        force=args.force,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
