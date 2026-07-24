#!/usr/bin/env python3
"""Validate the offline cross-model comparison artifact contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.model_comparison import validate_model_comparison_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    manifest = validate_model_comparison_artifacts(args.artifact_dir, verify_hashes=not args.skip_hashes)
    print(f"Valid {manifest['contract_version']} artifact set generated {manifest['generated_at']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
