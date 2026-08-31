#!/usr/bin/env python3
"""Run the final pre-model position-feature EDA pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.position_feature_eda import (
    DEFAULT_OUTPUT_DIR,
    run_position_feature_eda,
)
from src.econometrics_eda_v2.paths import topic_output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=topic_output_dir() / "content_econometrics_ai_package",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_position_feature_eda(args.package_dir, args.output_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
