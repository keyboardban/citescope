#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import AUDIT_DIR, ensure_v2_dirs, write_json
from src.econometrics_eda_v2.normalize_sources import audit_ai_json, audit_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-json", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    ai = audit_ai_json(args.ai_json)
    manifest = audit_manifest(args.manifest)
    apify = {"APIFY_TOKEN_exists": bool(os.environ.get("APIFY_TOKEN")), "token_value_printed": False}
    summary = {"ai_json": ai, "manifest": manifest, "apify": apify}
    write_json(AUDIT_DIR / "input_audit_summary.json", summary)
    if not ai.get("exists"):
        print(f"AI JSON missing: {args.ai_json}", file=sys.stderr)
        return 2
    if not manifest.get("exists"):
        print(f"Manifest missing: {args.manifest}", file=sys.stderr)
        return 2
    if not ai.get("cited_can_be_constructed"):
        print("Cannot construct cited outcome. Input must include surfaced sources and cited-source indicators.", file=sys.stderr)
        return 2
    print(f"Input audit complete: records={ai.get('n_records')} surfaced_sources={ai.get('n_surfaced_sources')} apify_token={apify['APIFY_TOKEN_exists']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
