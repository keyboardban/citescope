#!/usr/bin/env python3
"""Run focused verified-HTML-table EDA and stability diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.verified_table_diagnostics import (  # noqa: E402
    run_verified_table_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/econometrics_redesign_v3_20260727_faq_deduplicated",
    )
    parser.add_argument("--diagnostic-dir", type=Path)
    args = parser.parse_args()
    result = run_verified_table_diagnostics(
        ROOT,
        args.output_root,
        args.diagnostic_dir,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
