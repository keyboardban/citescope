from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.econometrics_eda_v2.io import utc_now_iso, write_json
from src.econometrics_eda_v2.scrape_queue import cache_is_success

DEFAULT_ACTOR_ID = "apify/website-content-crawler"


def _run_get(run: Any, key: str) -> Any:
    if isinstance(run, dict):
        return run.get(key)
    return getattr(run, key, None)


def _item_url(item: dict[str, Any]) -> str:
    for key in ("url", "requestedUrl", "loadedUrl", "finalUrl", "pageUrl"):
        if item.get(key):
            return str(item[key])
    return ""


def raw_cache_payload(
    *,
    scrape_id: str,
    requested_url: str,
    actor_id: str,
    provider_status: str,
    raw_item: dict[str, Any] | None = None,
    run_id: str | None = None,
    dataset_item_id: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    raw_item = raw_item or {}
    return {
        "scrape_id": scrape_id,
        "requested_url": requested_url,
        "final_url": raw_item.get("url") or raw_item.get("loadedUrl") or raw_item.get("finalUrl") or requested_url,
        "fetched_at": utc_now_iso(),
        "provider": "apify",
        "actor_id": actor_id,
        "run_id": run_id,
        "dataset_item_id": dataset_item_id,
        "status_code": raw_item.get("statusCode") or raw_item.get("status_code"),
        "provider_status": provider_status,
        "html": raw_item.get("html"),
        "markdown": raw_item.get("markdown"),
        "text": raw_item.get("text") or raw_item.get("content"),
        "title": raw_item.get("title") or raw_item.get("pageTitle"),
        "meta_description": raw_item.get("metaDescription") or raw_item.get("description"),
        "error": error,
        "raw_item": raw_item,
    }


def scrape_queue_with_apify(
    queue: pd.DataFrame,
    output_dir: str | Path,
    *,
    actor_id: str = DEFAULT_ACTOR_ID,
    max_urls: int | None = None,
    dry_run: bool = False,
    force_rescrape: bool = False,
    token: str | None = None,
    client_cls: Any | None = None,
    batch_mode: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_now_iso()
    warnings: list[str] = []
    token = token or os.environ.get("APIFY_TOKEN")
    if not token and not dry_run:
        raise RuntimeError("APIFY_TOKEN is required for real scraping. Use --dry-run to inspect the queue without scraping.")

    q = queue.copy()
    q = q[q.get("should_scrape", False).astype(bool)] if "should_scrape" in q.columns else q
    if max_urls is not None:
        q = q.head(max_urls)

    cached_skipped = 0
    pending = []
    for row in q.to_dict("records"):
        cache_path = output_dir / f"{row['scrape_id']}.json"
        if cache_is_success(cache_path) and not force_rescrape:
            cached_skipped += 1
            continue
        pending.append(row)

    summary = {
        "urls_total": int(len(queue)),
        "urls_attempted": 0,
        "urls_success": 0,
        "urls_failed": 0,
        "urls_cached_skipped": int(cached_skipped),
        "provider": "apify",
        "actor_id": actor_id,
        "started_at": started_at,
        "finished_at": None,
        "warnings": warnings,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        summary["urls_attempted"] = int(len(pending))
        summary["finished_at"] = utc_now_iso()
        return summary
    if not pending:
        summary["finished_at"] = utc_now_iso()
        return summary

    if client_cls is None:
        try:
            from apify_client import ApifyClient
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The Apify Python client is not installed in this virtualenv. "
                "Run `.venv/bin/python -m pip install apify-client` or "
                "`.venv/bin/python -m pip install -r requirements.txt`, then rerun the scrape command."
            ) from exc

        client_cls = ApifyClient
    client = client_cls(token)

    def _save(row: dict[str, Any], item: dict[str, Any] | None, status: str, run_id: str | None = None, error: str | None = None) -> None:
        payload = raw_cache_payload(
            scrape_id=row["scrape_id"],
            requested_url=row["normalized_url"],
            actor_id=actor_id,
            provider_status=status,
            raw_item=item or {},
            run_id=run_id,
            dataset_item_id=str((item or {}).get("id") or (item or {}).get("#id") or "") or None,
            error=error,
        )
        write_json(output_dir / f"{row['scrape_id']}.json", payload)

    try:
        if batch_mode and len(pending) > 1:
            run_input = {
                "startUrls": [{"url": r["normalized_url"]} for r in pending],
                "maxCrawlPages": len(pending),
                "maxCrawlDepth": 0,
                "maxResults": len(pending),
                "crawlerType": "cheerio",
                "saveHtml": False,
                "saveMarkdown": True,
                "readableTextCharThreshold": 100,
            }
            run = client.actor(actor_id).call(run_input=run_input)
            run_id = _run_get(run, "id") or _run_get(run, "default_dataset_id")
            dataset_id = _run_get(run, "default_dataset_id") or _run_get(run, "defaultDatasetId")
            items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
            by_url = {_item_url(item): item for item in items if isinstance(item, dict)}
            for row in pending:
                summary["urls_attempted"] += 1
                item = by_url.get(row["normalized_url"]) or by_url.get(row.get("source_url_example", ""))
                if item is None and len(pending) == len(items):
                    item = items[summary["urls_attempted"] - 1]
                if item:
                    _save(row, item, "success", str(run_id) if run_id else None)
                    summary["urls_success"] += 1
                else:
                    _save(row, {}, "failed", str(run_id) if run_id else None, "No matching Apify dataset item")
                    summary["urls_failed"] += 1
            summary["finished_at"] = utc_now_iso()
            return summary
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Batch Apify scrape failed; falling back to single URL mode: {exc}")

    for row in pending:
        summary["urls_attempted"] += 1
        try:
            run = client.actor(actor_id).call(
                run_input={
                    "startUrls": [{"url": row["normalized_url"]}],
                    "maxCrawlPages": 1,
                    "maxCrawlDepth": 0,
                    "maxResults": 1,
                    "crawlerType": "cheerio",
                    "saveHtml": False,
                    "saveMarkdown": True,
                    "readableTextCharThreshold": 100,
                }
            )
            run_id = _run_get(run, "id")
            dataset_id = _run_get(run, "default_dataset_id") or _run_get(run, "defaultDatasetId")
            items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
            item = items[0] if items else {}
            if item:
                _save(row, item, "success", str(run_id) if run_id else None)
                summary["urls_success"] += 1
            else:
                _save(row, {}, "failed", str(run_id) if run_id else None, "No Apify dataset item returned")
                summary["urls_failed"] += 1
        except Exception as exc:  # noqa: BLE001
            _save(row, {}, "failed", None, str(exc))
            summary["urls_failed"] += 1

    summary["finished_at"] = utc_now_iso()
    return summary
