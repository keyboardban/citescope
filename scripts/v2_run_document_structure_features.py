#!/usr/bin/env python3
"""Extract HTML-first document-structure features from normalized crawler snapshots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.document_structure_features import run_document_structure_layer
from src.econometrics_eda_v2.paths import topic_output_dir


def main() -> int:
    base = topic_output_dir()
    package = base / "content_econometrics_ai_package"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=package)
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=base / "tables/area_condo_brightdata_content_pilot/normalized",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = run_document_structure_layer(args.package_dir, args.snapshot_root, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
