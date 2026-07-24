#!/usr/bin/env python3
"""Build lightweight, leakage-safe frontend artifacts for manual feature QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.econometrics_eda_v2.feature_distribution_support import build_support_artifacts
from src.econometrics_eda_v2.manual_feature_validation import (
    build_artifacts,
    load_validated_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/econometrics_redesign_v2_20260722/frontend",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    manifest = build_artifacts(REPO_ROOT, output_dir)
    review_rows, review_content, _ = load_validated_artifacts(output_dir)
    review_metadata = review_content[
        [
            "normalized_url",
            "source_url",
            "url_title",
            "url_description",
            "authoritative_content_source",
        ]
    ].drop_duplicates(["normalized_url", "source_url"])
    review_rows = review_rows.merge(
        review_metadata,
        on=["normalized_url", "source_url"],
        how="left",
        validate="many_to_one",
    )
    support_manifest = build_support_artifacts(
        review_rows,
        output_dir.parent / "tables",
        output_dir / "feature_distribution_support_manifest.json",
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "manual_validation_files": manifest["files"],
                "distribution_support_files": support_manifest["files"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
