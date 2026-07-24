#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_config import (
    ALLOWED_BRIGHTDATA_PROVIDER_MODES,
    check_brightdata_config,
    load_brightdata_settings,
)
from src.econometrics_eda_v2.brightdata_response_parser import sanitize_brightdata_value
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv, write_json
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import build_brightdata_request_payload, prepare_brightdata_url


def _validation_report(live: bool) -> dict[str, Any]:
    status = check_brightdata_config(live=live)
    settings = load_brightdata_settings()
    errors = []
    if settings.provider_mode not in ALLOWED_BRIGHTDATA_PROVIDER_MODES:
        errors.append(f"Invalid provider mode: {settings.provider_mode}")
    if not settings.endpoint:
        errors.append("BRIGHTDATA_ENDPOINT is empty.")
    if live and not settings.api_key:
        errors.append("BRIGHTDATA_API_KEY is missing.")
    if live and not settings.zone:
        errors.append("BRIGHTDATA_ZONE is missing.")
    return {
        "live_ready": bool(status["live_ready"] and not errors),
        "missing_env_vars": status["missing_env_vars"],
        "provider_modes_available": sorted(ALLOWED_BRIGHTDATA_PROVIDER_MODES),
        "configured_provider_mode": settings.provider_mode,
        "endpoint_configured": bool(settings.endpoint),
        "endpoint": settings.endpoint,
        "api_key_present_masked": "present" if settings.api_key else "missing",
        "zone_present": bool(settings.zone),
        "validation_errors": errors,
    }


def _payload_debug(url: str) -> pd.DataFrame:
    settings = load_brightdata_settings()
    rows = []
    for mode in sorted(ALLOWED_BRIGHTDATA_PROVIDER_MODES):
        payload = build_brightdata_request_payload(url, mode, settings)
        errors = []
        if mode not in ALLOWED_BRIGHTDATA_PROVIDER_MODES:
            errors.append("invalid_provider_mode")
        if not payload.get("url") or not str(payload.get("url")).startswith(("http://", "https://")):
            errors.append("invalid_or_missing_url")
        if not payload.get("zone"):
            errors.append("missing_zone")
        if not payload.get("format"):
            errors.append("missing_format")
        if mode == "browser_api" and payload.get("data_format") != "markdown":
            errors.append("browser_api_expected_markdown_data_format")
        if mode == "unlocker_api" and "data_format" in payload:
            errors.append("unlocker_api_should_not_force_markdown")
        rows.append(
            {
                "provider_mode": mode,
                "endpoint": settings.endpoint,
                "request_payload_sanitized": sanitize_brightdata_value(payload),
                "validation_status": "ok" if not errors else "invalid",
                "provider_error_message": "",
                "recommended_fix": "" if not errors else "; ".join(errors),
            }
        )
    return pd.DataFrame(rows)


def _first_matching(df: pd.DataFrame, mask: pd.Series, bucket: str, seen: set[str]) -> dict[str, Any] | None:
    sub = df[mask].sort_values(["cited_rows_n", "source_rows_n"], ascending=[False, False])
    for _, row in sub.iterrows():
        bid = str(row.get("benchmark_id") or "")
        if bid and bid not in seen:
            seen.add(bid)
            out = row.to_dict()
            out["smoke_bucket"] = bucket
            return out
    return None


def select_smoke_urls(benchmark: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    rows = []
    flag = benchmark.get("current_content_quality_flag", pd.Series([""] * len(benchmark))).fillna("").astype(str)
    scrape_success = benchmark.get("current_scrape_success", pd.Series([False] * len(benchmark))).map(lambda v: str(v).casefold() in {"true", "1", "yes"})
    parse_success = benchmark.get("current_parse_success", pd.Series([False] * len(benchmark))).map(lambda v: str(v).casefold() in {"true", "1", "yes"})
    wc = pd.to_numeric(benchmark.get("current_word_count", pd.Series([0] * len(benchmark))), errors="coerce").fillna(0)
    specs = [
        ("apify_succeeded", scrape_success & parse_success),
        ("apify_dynamic_js_likely", flag.eq("dynamic_js_likely")),
        ("apify_parse_failed", flag.eq("parse_failed") | (~parse_success)),
        ("apify_blocked_or_error_page", flag.eq("blocked_or_error_page")),
        ("normal_static_content", flag.eq("ok") & (wc >= 300)),
    ]
    for bucket, mask in specs:
        row = _first_matching(benchmark, mask, bucket, seen)
        if row:
            rows.append(row)
    if len(rows) < 3:
        for _, row in benchmark.iterrows():
            bid = str(row.get("benchmark_id") or "")
            if bid not in seen:
                seen.add(bid)
                out = row.to_dict()
                out["smoke_bucket"] = "fill"
                rows.append(out)
            if len(rows) >= 5:
                break
    return pd.DataFrame(rows).head(5)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "tables"))
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark = read_csv(args.benchmark_input)
    example_url = prepare_brightdata_url(str(benchmark.iloc[0]["normalized_url"])) if len(benchmark) else "https://example.com"
    report = _validation_report(live=args.live)
    write_json(out_dir / "brightdata_config_validation_report.json", report)
    write_csv(out_dir / "brightdata_payload_validation_debug.csv", _payload_debug(example_url))
    smoke = select_smoke_urls(benchmark)
    write_csv(out_dir / "brightdata_smoke_test_urls.csv", smoke)
    print(f"Bright Data config live_ready={report['live_ready']} missing={report['missing_env_vars']}")
    print(f"Config report: {out_dir / 'brightdata_config_validation_report.json'}")
    print(f"Payload debug: {out_dir / 'brightdata_payload_validation_debug.csv'}")
    print(f"Smoke URLs: {out_dir / 'brightdata_smoke_test_urls.csv'} rows={len(smoke)}")
    return 0 if report["live_ready"] or not args.live else 2


if __name__ == "__main__":
    raise SystemExit(main())
