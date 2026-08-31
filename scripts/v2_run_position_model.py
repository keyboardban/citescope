#!/usr/bin/env python3
"""Run the separate position-focused M0-M6 econometric analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.econometrics_eda_v2.position_model import DEFAULT_OUTPUT_DIR, run_position_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO / DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_position_model(REPO, args.output_dir)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

