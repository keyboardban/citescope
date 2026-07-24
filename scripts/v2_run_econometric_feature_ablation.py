#!/usr/bin/env python3
"""Generate read-only feature-ablation artifacts for the QA frontend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.econometric_feature_ablation import run_feature_ablation
from src.econometrics_eda_v2.paths import topic_output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=topic_output_dir() / "content_econometrics_ai_package",
    )
    args = parser.parse_args()
    print(run_feature_ablation(args.package_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
