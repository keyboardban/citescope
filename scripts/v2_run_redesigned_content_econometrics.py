#!/usr/bin/env python3
"""Run the governed five-layer econometrics redesign offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.econometrics_eda_v2.redesigned_pipeline_v2 import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs/econometrics_redesign_v2_20260722"))
    args = parser.parse_args()
    repo = REPO_ROOT
    result = run(repo, args.output_root.resolve())
    print(json.dumps({"status": result["status"], "output_root": str(args.output_root.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
