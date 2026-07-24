#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.brightdata_config import check_brightdata_config, load_brightdata_settings
from src.econometrics_eda_v2.brightdata_response_parser import (
    brightdata_raw_cache_payload,
    parse_brightdata_cache_payload,
    sanitize_brightdata_value,
    write_brightdata_parsed_cache,
    write_brightdata_raw_cache,
)
from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv, write_json
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import scrape_url_brightdata


SUMMARY_PATH = OUTPUT_DIR / "tables" / "brightdata_benchmark_scrape_summary.csv"


def _bool_text(value: str) -> bool:
    low = str(value).strip().casefold()
    if low in {"1", "true", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def _cache_paths(raw_dir: Path, parsed_dir: Path, dry_run_dir: Path, benchmark_id: str) -> tuple[Path, Path, Path]:
    return (
        raw_dir / f"{benchmark_id}.raw.json",
        parsed_dir / f"{benchmark_id}.parsed.json",
        dry_run_dir / f"{benchmark_id}.dry_run.json",
    )


def _summary_row(row: pd.Series, provider_mode: str, raw_path: Path, parsed_path: Path, attempted: bool, parsed: dict, error: str = "") -> dict:
    return {
        "benchmark_id": row["benchmark_id"],
        "source_url": row["source_url"],
        "provider": "brightdata",
        "provider_mode": provider_mode,
        "attempted": bool(attempted),
        "success": bool(parsed.get("parse_success")),
        "status_code": parsed.get("status_code", ""),
        "error": error or parsed.get("parse_error", ""),
        "raw_cache_path": str(raw_path),
        "parsed_cache_path": str(parsed_path),
        "fetched_at": parsed.get("fetched_at", ""),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--raw-cache-dir", default="data/econometrics_v2/scrape_cache/brightdata_benchmark/raw")
    ap.add_argument("--parsed-cache-dir", default="data/econometrics_v2/scrape_cache/brightdata_benchmark/parsed")
    ap.add_argument("--dry-run-dir", default="data/econometrics_v2/scrape_cache/brightdata_benchmark/dry_run")
    ap.add_argument("--summary-output", default=str(SUMMARY_PATH))
    ap.add_argument("--output-dir", dest="output_dir_alias", default=None, help="Alias for --raw-cache-dir.")
    ap.add_argument("--max-urls", type=int, default=None)
    ap.add_argument("--force-mode", choices=["browser_api", "unlocker_api", "crawler_api"], default=None,
                    help="Override recommended_brightdata_mode for every row (e.g. run the whole benchmark in browser_api).")
    ap.add_argument("--dry-run", action="store_true", help="Dry-run only. This is also the default unless --execute-live-brightdata is passed.")
    ap.add_argument("--execute-live-brightdata", action="store_true", help="Explicitly allow live Bright Data HTTP calls.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--save-raw-response", type=_bool_text, default=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    live = bool(args.execute_live_brightdata)
    raw_dir = Path(args.output_dir_alias or args.raw_cache_dir)
    parsed_dir = Path(args.parsed_cache_dir)
    dry_run_dir = Path(args.dry_run_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    dry_run_dir.mkdir(parents=True, exist_ok=True)
    cfg_status = check_brightdata_config(live=live)
    settings = load_brightdata_settings()
    print(
        "Bright Data config: "
        f"api_key_present={cfg_status['api_key_present']} "
        f"provider_mode={cfg_status['provider_mode']} "
        f"endpoint={cfg_status['endpoint']} "
        f"render_js={cfg_status['render_js']} "
        f"missing={','.join(cfg_status['missing_env_vars']) or 'none'} "
        f"live={live}"
    )
    if live and cfg_status["missing_env_vars"]:
        missing = ", ".join(cfg_status["missing_env_vars"])
        if "BRIGHTDATA_API_KEY" in cfg_status["missing_env_vars"]:
            print("Live Bright Data execution requires BRIGHTDATA_API_KEY.", file=sys.stderr)
        else:
            print(f"Live Bright Data execution missing required env vars: {missing}.", file=sys.stderr)
        return 2
    if live and not args.save_raw_response:
        print("Live Bright Data benchmark requires --save-raw-response true.", file=sys.stderr)
        return 2

    input_df = read_csv(args.input)
    if args.max_urls is not None:
        input_df = input_df.head(args.max_urls)
    rows = []
    for _, row in input_df.iterrows():
        bid = str(row["benchmark_id"])
        mode = args.force_mode or str(row.get("recommended_brightdata_mode") or settings.provider_mode)
        requested_url = str(row["normalized_url"])
        raw_path, parsed_path, dry_run_path = _cache_paths(raw_dir, parsed_dir, dry_run_dir, bid)
        if raw_path.exists() and parsed_path.exists() and not args.force:
            parsed = json.loads(parsed_path.read_text("utf-8"))
            rows.append(_summary_row(row, mode, raw_path, parsed_path, False, parsed))
            continue
        if raw_path.exists() and not args.force and not parsed_path.exists():
            raw_payload = json.loads(raw_path.read_text("utf-8"))
            parsed = parse_brightdata_cache_payload(raw_payload, path=raw_path)
            write_brightdata_parsed_cache(parsed_path, parsed, force=False)
            rows.append(_summary_row(row, mode, raw_path, parsed_path, False, parsed))
            continue
        if mode == "pdf_parser_needed":
            parsed = {
                "benchmark_id": bid,
                "requested_url": requested_url,
                "provider": "brightdata",
                "provider_mode": "pdf_parser_needed",
                "final_url": requested_url,
                "parse_success": False,
                "scraped_body_available": False,
                "page_text": "",
                "word_count": 0,
                "text_char_count": 0,
                "content_quality_flag": "parse_failed",
                "parse_error": "pdf_parser_needed_not_brightdata",
                "parse_error_category": "pdf_parser_needed",
            }
            write_brightdata_parsed_cache(parsed_path, parsed, force=args.force)
            rows.append(_summary_row(row, mode, raw_path, parsed_path, False, parsed, "pdf_parser_needed_not_brightdata"))
            continue
        if not live:
            result = scrape_url_brightdata(requested_url, mode, settings, live=False)
            dry_payload = {
                "benchmark_id": bid,
                "requested_url": requested_url,
                "provider": "brightdata",
                "provider_mode": mode,
                "dry_run": True,
                "planned_request": sanitize_brightdata_value(result.get("planned_request") or {}),
            }
            if args.force or not dry_run_path.exists():
                write_json(dry_run_path, dry_payload)
            parsed = result.get("normalized_result") or {}
            rows.append(_summary_row(row, mode, raw_path, parsed_path, False, parsed, "dry_run_no_api_call"))
            print(f"DRY RUN {bid}: mode={mode} planned_request={dry_run_path}")
            continue

        result = scrape_url_brightdata(requested_url, mode, settings, live=True, raw_response_path="")
        normalized = dict(result.get("normalized_result") or {})
        raw_payload = brightdata_raw_cache_payload(
            benchmark_id=bid,
            requested_url=requested_url,
            provider_mode=str(normalized.get("provider_mode") or f"brightdata_{mode}"),
            fetched_at=str(normalized.get("fetched_at") or ""),
            status_code=normalized.get("status_code"),
            request_payload=result.get("request_payload") or {},
            request_params=result.get("request_params") or {},
            response_headers=result.get("response_headers") or {},
            raw_response=result.get("raw_response"),
            error_if_request_failed=str(normalized.get("error") or ""),
        )
        write_brightdata_raw_cache(raw_path, raw_payload, force=args.force)
        saved_raw = json.loads(raw_path.read_text("utf-8"))
        if "raw_response" not in saved_raw:
            raise RuntimeError("Bright Data live call did not persist raw_response; benchmark invalid.")
        parsed = parse_brightdata_cache_payload(raw_payload, path=raw_path)
        write_brightdata_parsed_cache(parsed_path, parsed, force=args.force)
        rows.append(_summary_row(row, mode, raw_path, parsed_path, True, parsed))
        print(f"LIVE {bid}: mode={mode} parse_success={bool(parsed.get('parse_success'))} raw={raw_path} parsed={parsed_path}")

    summary_path = Path(args.summary_output)
    write_csv(summary_path, pd.DataFrame(rows))
    print(
        "Bright Data benchmark scrape preparation complete: "
        f"rows={len(rows)} live_attempted={sum(bool(r['attempted']) for r in rows)} summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
