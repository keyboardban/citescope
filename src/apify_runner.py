"""Apify wrappers: SERP reconstruction + website content scraping.

Both functions run an actor synchronously, then normalise the dataset output
into a stable shape (actor output keys vary across versions). Run metadata
(actor run id, dataset id, timestamp) is returned for the audit trail.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .ids import now_iso


def _lazy_client(token: str):
    from apify_client import ApifyClient
    return ApifyClient(token)


def build_client(token: str):
    return _lazy_client(token)


def _g(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return default


def _run_field(run: Any, attr: str, alias: str | None = None) -> Any:
    """Read a field from an Apify actor run result.

    apify-client v3 returns a pydantic ``Run`` object (snake_case attributes:
    ``id``, ``status``, ``default_dataset_id``); older versions returned a dict
    (camelCase keys). Handle both.
    """
    if run is None:
        return None
    if isinstance(run, dict):
        val = run.get(alias) if alias else None
        return val if val is not None else run.get(attr)
    return getattr(run, attr, None)


# --------------------------------------------------------------------------- #
# SERP reconstruction
# --------------------------------------------------------------------------- #
def run_serp(
    client,
    queries: Sequence[str],
    top_k: int = 20,
    country: str = "us",
    language: str = "en",
    actor: str = "apify/google-search-scraper",
) -> dict:
    queries = [q for q in queries if q and q.strip()]
    if not queries:
        return {"candidates": [], "items": [], "run_id": None, "dataset_id": None,
                "actor": actor, "ts": now_iso(), "error": "no queries provided"}

    pages_per_query = max(1, math.ceil(top_k / 10))
    run_input = {
        "queries": "\n".join(queries),
        "maxPagesPerQuery": pages_per_query,
        "resultsPerPage": min(max(top_k, 10), 100),
        "countryCode": country,
        "languageCode": language,
        "searchLanguage": language,
        "saveHtml": False,
        "mobileResults": False,
    }
    try:
        run = client.actor(actor).call(run_input=run_input)
    except Exception as exc:
        return {"candidates": [], "items": [], "run_id": None, "dataset_id": None,
                "actor": actor, "ts": now_iso(), "error": f"{type(exc).__name__}: {exc}"}

    dataset_id = _run_field(run, "default_dataset_id", "defaultDatasetId")
    items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
    candidates = normalize_serp(items, top_k=top_k)
    return {
        "candidates": candidates,
        "items": items,
        "run_id": _run_field(run, "id"),
        "dataset_id": dataset_id,
        "status": _run_field(run, "status"),
        "actor": actor,
        "ts": now_iso(),
        "error": None if candidates else "no organic results returned",
    }


def normalize_serp(items: list[dict], top_k: int = 20) -> list[dict]:
    """Flatten raw SERP pages into ranked candidate rows (trimmed to top_k)."""
    out: list[dict] = []
    for item in items:
        sq = _g(item, "searchQuery", "search_query", default={}) or {}
        query = _g(sq, "term", "query", default=_g(item, "query", default="")) or ""
        organic = _g(item, "organicResults", "organic_results", "results", default=[]) or []
        for idx, r in enumerate(organic, start=1):
            rank = int(_g(r, "position", "rank", default=idx) or idx)
            if rank > top_k:
                continue
            url = _g(r, "url", "link", default="")
            if not url:
                continue
            out.append(
                {
                    "query": query,
                    "rank": rank,
                    "url": url,
                    "title": _g(r, "title", default="") or "",
                    "snippet": _g(r, "description", "snippet", "desc", default="") or "",
                    "displayed_url": _g(r, "displayedUrl", "displayUrl", default="") or "",
                    "result_type": _g(r, "type", default="organic") or "organic",
                }
            )
    return out


# --------------------------------------------------------------------------- #
# Website content scraping
# --------------------------------------------------------------------------- #
def run_scrape(
    client,
    urls: Sequence[str],
    crawler_type: str = "cheerio",
    actor: str = "apify/website-content-crawler",
) -> dict:
    urls = [u for u in dict.fromkeys(urls) if u]  # dedupe, preserve order
    if not urls:
        return {"pages": [], "items": [], "run_id": None, "dataset_id": None,
                "actor": actor, "ts": now_iso(), "error": "no urls provided"}

    run_input = {
        "startUrls": [{"url": u} for u in urls],
        "crawlerType": crawler_type,
        "maxCrawlDepth": 0,           # only the given pages, no link-following
        "maxCrawlPages": len(urls),
        "maxResults": len(urls),
        "saveMarkdown": True,
        "saveHtml": False,
        "readableTextCharThreshold": 100,
    }
    try:
        run = client.actor(actor).call(run_input=run_input)
    except Exception as exc:
        return {"pages": [], "items": [], "run_id": None, "dataset_id": None,
                "actor": actor, "ts": now_iso(), "error": f"{type(exc).__name__}: {exc}"}

    dataset_id = _run_field(run, "default_dataset_id", "defaultDatasetId")
    items = list(client.dataset(dataset_id).iterate_items()) if dataset_id else []
    pages = normalize_pages(items)
    return {
        "pages": pages,
        "items": items,
        "run_id": _run_field(run, "id"),
        "dataset_id": dataset_id,
        "status": _run_field(run, "status"),
        "actor": actor,
        "ts": now_iso(),
        "error": None,
    }


def normalize_pages(items: list[dict]) -> list[dict]:
    """Normalise crawler output into page records (handles output variants)."""
    out: list[dict] = []
    for item in items:
        crawl = _g(item, "crawl", default={}) or {}
        meta = _g(item, "metadata", default={}) or {}
        url = _g(item, "url", default="") or _g(crawl, "loadedUrl", default="")
        loaded = _g(crawl, "loadedUrl", default=None) or _g(item, "loadedUrl", default=url)
        text = _g(item, "text", default="") or ""
        markdown = _g(item, "markdown", default="") or text
        canonical = _g(meta, "canonicalUrl", "canonical", default=None)
        published = _g(
            meta, "publishedTime", "datePublished", "published",
            "articlePublishedTime", "modifiedTime", default=None,
        )
        http_status = _g(crawl, "httpStatusCode", "statusCode", default=None)
        err = _g(item, "errorMessages", default=None) or _g(crawl, "errorMessage", default=None)
        ok = bool(text or markdown) and not err

        out.append(
            {
                "url": url,
                "final_url": loaded,
                "canonical_url": canonical,
                "title": _g(meta, "title", default=_g(item, "title", default="")) or "",
                "description": _g(meta, "description", default="") or "",
                "author": _g(meta, "author", default=None),
                "language": _g(meta, "languageCode", "language", default=None),
                "published_date": published,
                "text": text,
                "markdown": markdown,
                "http_status": http_status,
                "status": "success" if ok else "failed",
                "error": (str(err) if err else None),
            }
        )
    return out
