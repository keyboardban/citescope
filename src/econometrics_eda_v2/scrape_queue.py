from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.url_utils import domain as url_domain
from src.url_utils import normalize_url
from src.econometrics_eda_v2.io import RAW_CACHE_DIR
from src.econometrics_eda_v2.normalize_sources import stable_hash


def cache_is_success(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text("utf-8"))
    except Exception:
        return False
    status = str(data.get("provider_status") or "").lower()
    return status in {"success", "ok", "completed"} and not data.get("error")


def valid_normalized_url(url: str) -> bool:
    if not url:
        return False
    try:
        p = urlparse(url)
    except Exception:
        return False
    return p.scheme in {"http", "https"} and bool(p.netloc)


def build_scrape_queue(source_rows: pd.DataFrame, raw_dir: str | Path = RAW_CACHE_DIR, force_rescrape: bool = False) -> tuple[pd.DataFrame, dict]:
    raw_dir = Path(raw_dir)
    work = source_rows.copy()
    work["normalized_url"] = work["normalized_url"].fillna("").map(lambda x: normalize_url(str(x)) if str(x).strip() else "")
    valid_source_urls = {
        u for u in work["normalized_url"].dropna().astype(str)
        if u and valid_normalized_url(u)
    }
    rows = []
    for nurl, g in work[work["normalized_url"] != ""].groupby("normalized_url", dropna=False):
        scrape_id = "s_" + stable_hash(nurl, n=20)
        cache_path = raw_dir / f"{scrape_id}.json"
        is_valid = valid_normalized_url(nurl)
        cached = cache_is_success(cache_path)
        rows.append(
            {
                "scrape_id": scrape_id,
                "normalized_url": nurl,
                "source_url_example": g["source_url"].dropna().astype(str).iloc[0] if "source_url" in g else nurl,
                "domain": url_domain(nurl),
                "n_source_rows": int(len(g)),
                "n_cited_rows": int(pd.to_numeric(g["cited"], errors="coerce").fillna(0).sum()),
                "n_more_only_rows": int((pd.to_numeric(g["cited"], errors="coerce").fillna(0) == 0).sum()),
                "first_seen_answer_id": str(g["answer_id"].iloc[0]) if "answer_id" in g else "",
                "cache_path": str(cache_path),
                "scrape_status": "cached_success" if cached else ("invalid_url" if not is_valid else "pending"),
                "should_scrape": bool(is_valid and (force_rescrape or not cached)),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "scrape_id", "normalized_url", "source_url_example", "domain", "n_source_rows",
                "n_cited_rows", "n_more_only_rows", "first_seen_answer_id", "cache_path",
                "scrape_status", "should_scrape",
            ]
        )
    summary = {
        "rows": int(len(df)),
        "source_unique_valid_urls": int(len(valid_source_urls)),
        "queue_unique_urls": int(df["normalized_url"].nunique(dropna=True)) if len(df) else 0,
        "missing_source_urls_from_queue": int(len(valid_source_urls - set(df["normalized_url"].dropna().astype(str)))) if len(df) else int(len(valid_source_urls)),
        "should_scrape": int(df["should_scrape"].sum()) if len(df) else 0,
        "cached_success": int((df["scrape_status"] == "cached_success").sum()) if len(df) else 0,
        "invalid_urls": int((df["scrape_status"] == "invalid_url").sum()) if len(df) else 0,
        "unique_domains": int(df["domain"].nunique(dropna=True)) if len(df) else 0,
    }
    if summary["source_unique_valid_urls"] > summary["queue_unique_urls"]:
        raise ValueError(
            "Scrape queue stale or incomplete: source rows have "
            f"{summary['source_unique_valid_urls']} unique URLs but queue has {summary['queue_unique_urls']}."
        )
    return df, summary
