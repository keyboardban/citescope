#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_response_parser import (
    audit_raw_response_shape,
    build_cache_integrity_audit,
    build_failure_triage,
    build_parser_before_after,
    build_unrecoverable_cache_retry_queue,
    retry_queue_frames,
    retry_queue_path,
)
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv
from src.econometrics_eda_v2.provider_benchmark import parse_brightdata_raw_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data/econometrics_v2/scrape_cache/brightdata_benchmark/raw")
    ap.add_argument("--output", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_parse_rows.csv"))
    ap.add_argument("--benchmark-input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--cache-integrity-output", default=str(OUTPUT_DIR / "tables" / "brightdata_cache_integrity_audit.csv"))
    ap.add_argument("--reparse-output", default=str(OUTPUT_DIR / "tables" / "brightdata_reparse_results.csv"))
    ap.add_argument("--unrecoverable-retry-output", default=str(OUTPUT_DIR / "tables" / "brightdata_unrecoverable_cache_retry_queue.csv"))
    ap.add_argument("--shape-audit-output", default=str(OUTPUT_DIR / "tables" / "brightdata_raw_response_shape_audit.csv"))
    ap.add_argument("--before-after-output", default=str(OUTPUT_DIR / "tables" / "brightdata_parser_before_after.csv"))
    ap.add_argument("--failure-triage-output", default=str(OUTPUT_DIR / "tables" / "brightdata_failure_triage.csv"))
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    output = Path(args.output)
    before = read_csv(output) if output.exists() and output.stat().st_size > 0 else pd.DataFrame()
    benchmark_input = read_csv(args.benchmark_input) if Path(args.benchmark_input).exists() else pd.DataFrame()
    integrity = build_cache_integrity_audit(args.input_dir)
    df = parse_brightdata_raw_dir(args.input_dir)
    shape = pd.DataFrame([audit_raw_response_shape(path) for path in sorted(Path(args.input_dir).glob("*.json"))])
    before_after = build_parser_before_after(before, df)
    triage = build_failure_triage(df)
    unrecoverable = build_unrecoverable_cache_retry_queue(integrity, benchmark_input)
    write_csv(output, df)
    write_csv(args.cache_integrity_output, integrity)
    write_csv(args.reparse_output, df)
    write_csv(args.unrecoverable_retry_output, unrecoverable)
    write_csv(args.shape_audit_output, shape)
    write_csv(args.before_after_output, before_after)
    write_csv(args.failure_triage_output, triage)
    for queue, frame in retry_queue_frames(triage).items():
        write_csv(retry_queue_path(queue), frame)
    parse_success = int(df["parse_success"].sum()) if len(df) and "parse_success" in df else 0
    print(
        "Bright Data benchmark parse complete: "
        f"rows={len(df)} parse_success={parse_success} output={output}"
    )
    print(f"Shape audit: {args.shape_audit_output}")
    print(f"Cache integrity audit: {args.cache_integrity_output}")
    print(f"Unrecoverable retry queue: {args.unrecoverable_retry_output}")
    print(f"Failure triage: {args.failure_triage_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
