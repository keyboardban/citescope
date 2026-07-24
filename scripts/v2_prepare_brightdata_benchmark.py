#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_benchmark import select_brightdata_benchmark_urls
from src.econometrics_eda_v2.brightdata_config import check_brightdata_config
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape-audit", default=str(OUTPUT_DIR / "tables" / "scrape_quality_audit.csv"))
    ap.add_argument("--out", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--max-urls", type=int, default=40)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    cfg = check_brightdata_config(live=False)
    df = select_brightdata_benchmark_urls(read_csv(args.scrape_audit), max_urls=args.max_urls)
    write_csv(args.out, df)
    print(
        "Bright Data benchmark prepared: "
        f"rows={len(df)} output={args.out} "
        f"api_key_present={cfg['api_key_present']} provider_mode={cfg['provider_mode']} render_js={cfg['render_js']}"
    )
    print("Reason breakdown:")
    print(df["reason_selected"].value_counts().to_string() if len(df) else "none")
    print("Recommended Bright Data modes:")
    print(df["recommended_brightdata_mode"].value_counts().to_string() if len(df) else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
