#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.apify_scraper import DEFAULT_ACTOR_ID, scrape_queue_with_apify
from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, read_csv, write_json


def _bool_arg(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--actor-id", default=DEFAULT_ACTOR_ID)
    ap.add_argument("--provider", default="apify")
    ap.add_argument("--max-urls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-rescrape", action="store_true")
    ap.add_argument("--cache", type=_bool_arg, default=True)
    args = ap.parse_args(argv)
    if args.provider != "apify":
        print("Only --provider apify is supported in v2.", file=sys.stderr)
        return 2
    ensure_v2_dirs()
    summary = scrape_queue_with_apify(
        read_csv(args.queue),
        args.output_dir,
        actor_id=args.actor_id,
        max_urls=args.max_urls,
        dry_run=args.dry_run,
        force_rescrape=args.force_rescrape,
    )
    write_json(AUDIT_DIR / "apify_scrape_run_summary.json", summary)
    print(f"Apify scrape {'dry-run ' if args.dry_run else ''}complete: attempted={summary['urls_attempted']} success={summary['urls_success']} failed={summary['urls_failed']} cached={summary['urls_cached_skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
