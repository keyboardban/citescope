#!/usr/bin/env python3
"""Build validated, lightweight artifacts for the Streamlit econometrics frontend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.econometrics_frontend import build_frontend_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = build_frontend_artifacts(args.package_dir, args.output_dir)
    print(f"Validated econometrics frontend artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
