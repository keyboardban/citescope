#!/usr/bin/env python3
"""Write the versioned Core-General feature specification to repo and package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import econometrics_qa as qa  # noqa: E402
from src.econometrics_eda_v2.core_general_feature_registry import (  # noqa: E402
    build_core_general_feature_registry,
    write_core_general_feature_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=qa.default_package_dir())
    args = parser.parse_args()
    canonical = ROOT / "config/core_general_content_feature_dictionary.csv"
    registry = build_core_general_feature_registry(canonical)
    package_copy = write_core_general_feature_registry(
        args.package_dir / "tables/core_general_content_feature_dictionary.csv",
        registry,
    )
    print(canonical)
    print(package_copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
