#!/usr/bin/env python3
"""Run post-estimation diagnostics for the updated governed D0-FE4 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.result_diagnostics import run_diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/econometrics_redesign_v3_20260727_faq_deduplicated",
    )
    parser.add_argument("--diagnostic-dir", type=Path)
    args = parser.parse_args()
    result = run_diagnostics(ROOT, args.output_root, args.diagnostic_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
