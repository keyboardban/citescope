from __future__ import annotations

import gzip
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit

import requests

from src.econometrics_eda_v2.brightdata_response_parser import extract_body_like_content
from src.econometrics_eda_v2.brightdata_config import BrightDataSettings
from src.econometrics_eda_v2.io import utc_now_iso, write_json
from src.econometrics_eda_v2.scrape_providers.base import NormalizedScrapeResult
from src.econometrics_eda_v2.scrape_providers.provider_quality import (
    clean_html_text,
    count_heading_table_link_image,
    count_words,
    normalized_url_for,
)
from src.econometrics_eda_v2.scrape_providers.provider_types import BRIGHTDATA_MODE_TO_PROVIDER_MODE
from src.url_utils import strip_tracking_params


def _provider_mode(mode: str) -> str:
    return BRIGHTDATA_MODE_TO_PROVIDER_MODE.get(str(mode or "browser_api"), BRIGHTDATA_MODE_TO_PROVIDER_MODE["browser_api"])


def prepare_brightdata_url(url: str) -> str:
    raw = strip_tracking_params(str(url or "").strip())
    if not raw:
        return raw
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw
    try:
        host = parts.hostname.encode("idna").decode("ascii") if parts.hostname else ""
    except UnicodeError:
        host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    if parts.username:
        auth = quote(parts.username, safe="")
        if parts.password:
            auth += f":{quote(parts.password, safe='')}"
        host = f"{auth}@{host}"
    path = quote(parts.path or "", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="=&?/%:+,;@!$'()*-._~")
    fragment = quote(parts.fragment, safe="=&?/%:+,;@!$'()*-._~")
    return urlunsplit((parts.scheme, host, path, query, fragment))


def build_brightdata_request_payload(url: str, mode: str, config: BrightDataSettings) -> dict[str, Any]:
    clean_url = prepare_brightdata_url(url)
    if mode == "crawler_api":
        return {"input": [{"url": clean_url}]}
    payload = {
        "zone": config.zone,
        "url": clean_url,
        "format": "raw",
        "country": config.country,
        "render": "true" if bool(config.render_js) else "false",
    }
    if mode == "browser_api":
        payload["data_format"] = "markdown"
    return payload


def build_brightdata_request_params(mode: str, config: BrightDataSettings) -> dict[str, str]:
    if mode != "crawler_api":
        return {}
    return {
        "dataset_id": config.dataset_id,
        "notify": "false",
        "include_errors": "true",
        "format": "json",
    }


def build_brightdata_crawler_async_payload(urls: list[str]) -> list[dict[str, str]]:
    return [{"url": prepare_brightdata_url(url)} for url in urls if str(url or "").strip()]


def trigger_brightdata_crawler_async(
    urls: list[str],
    config: BrightDataSettings,
    *,
    request_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not config.api_key:
        raise RuntimeError("Live Bright Data execution requires BRIGHTDATA_API_KEY.")
    if not config.dataset_id:
        raise RuntimeError("Live Bright Data Crawler API execution requires BRIGHTDATA_CRAWLER_DATASET_ID.")
    payload = build_brightdata_crawler_async_payload(urls)
    if not payload:
        raise ValueError("Crawler API requires at least one URL.")
    request_fn = request_fn or requests.post
    response = request_fn(
        config.endpoint,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        params=build_brightdata_request_params("crawler_api", config),
        json=payload,
        timeout=config.timeout_seconds,
    )
    try:
        raw = response.json()
    except ValueError:
        raw = {"message": response.text}
    snapshot_id = str(raw.get("snapshot_id") or "") if isinstance(raw, dict) else ""
    if not (200 <= int(getattr(response, "status_code", 0) or 0) < 300) or not snapshot_id:
        message = raw.get("error") or raw.get("message") if isinstance(raw, dict) else ""
        raise RuntimeError(str(message or f"Crawler API trigger failed (HTTP {getattr(response, 'status_code', 'unknown')})."))
    return {"snapshot_id": snapshot_id, "request_payload": payload, "request_params": build_brightdata_request_params("crawler_api", config), "response_headers": dict(getattr(response, "headers", {}) or {}), "raw_response": raw}


def wait_for_brightdata_snapshot(
    snapshot_id: str,
    config: BrightDataSettings,
    *,
    request_fn: Callable[..., Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    request_fn = request_fn or requests.get
    headers = {"Authorization": f"Bearer {config.api_key}"}
    deadline = time.monotonic() + max(1, config.crawler_wait_seconds)
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = request_fn(f"https://api.brightdata.com/datasets/v3/progress/{snapshot_id}", headers=headers, timeout=config.timeout_seconds)
        try:
            last = response.json()
        except ValueError:
            last = {"message": response.text}
        status = str(last.get("status") or "").casefold()
        if progress_callback is not None:
            progress_callback(last)
        if status == "ready":
            return last
        if status in {"failed", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"Crawler snapshot {snapshot_id} finished with status {status}: {last}")
        time.sleep(max(1, config.crawler_poll_seconds))
    raise TimeoutError(f"Crawler snapshot {snapshot_id} was not ready after {config.crawler_wait_seconds} seconds.")


def download_brightdata_snapshot(snapshot_id: str, config: BrightDataSettings, *, request_fn: Callable[..., Any] | None = None) -> list[dict[str, Any]]:
    request_fn = request_fn or requests.get
    network_attempts = max(1, int(config.max_retries) + 1)
    preparation_attempts = max(3, min(12, int(config.crawler_wait_seconds) // max(1, int(config.crawler_poll_seconds))))
    network_failures = 0
    raw: Any = None
    for preparation_attempt in range(1, preparation_attempts + 1):
        try:
            response = request_fn(
                f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                headers={"Authorization": f"Bearer {config.api_key}"},
                params={"format": "ndjson", "compress": "true"},
                timeout=config.timeout_seconds,
            )
            status_code = int(getattr(response, "status_code", 0) or 0)
            if status_code == 202:
                if preparation_attempt == preparation_attempts:
                    raise RuntimeError("Crawler snapshot compression was still being prepared after the wait limit.")
                time.sleep(min(max(1, int(config.crawler_poll_seconds)), 30))
                continue
            if not (200 <= status_code < 300):
                raise RuntimeError(f"Crawler snapshot download failed (HTTP {getattr(response, 'status_code', 'unknown')}).")
            content = getattr(response, "content", None)
            if isinstance(content, bytes):
                try:
                    decoded = gzip.decompress(content) if content.startswith(b"\x1f\x8b") else content
                    raw = [json.loads(line) for line in decoded.splitlines() if line.strip()]
                except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                    raise RuntimeError("Crawler snapshot was not valid compressed NDJSON.") from exc
            else:
                try:
                    raw = response.json()
                except ValueError as exc:
                    raise RuntimeError("Crawler snapshot was not valid JSON.") from exc
            break
        except requests.exceptions.RequestException as exc:
            network_failures += 1
            if network_failures == network_attempts:
                raise RuntimeError(
                    f"Crawler snapshot download failed after {network_attempts} network attempts. "
                    "The saved snapshot can be resumed without re-scraping."
                ) from exc
            time.sleep(min(5 * 2 ** (network_failures - 1), 30))
    if isinstance(raw, list):
        return [row for row in raw if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("data", "results", "items", "records"):
            wrapped = raw.get(key)
            if isinstance(wrapped, list):
                return [row for row in wrapped if isinstance(row, dict)]
        message = raw.get("error") or raw.get("message") or raw.get("status") or "no message"
        raise RuntimeError(
            "Crawler snapshot response contained no page-record list "
            f"(keys={sorted(raw)[:12]}, message={message!s})."
        )
    raise RuntimeError(f"Crawler snapshot response had unsupported type {type(raw).__name__}.")


def dry_run_brightdata_request(url: str, mode: str, config: BrightDataSettings) -> dict[str, Any]:
    missing = []
    if not config.api_key:
        missing.append("BRIGHTDATA_API_KEY")
    if mode == "crawler_api" and not config.dataset_id:
        missing.append("BRIGHTDATA_CRAWLER_DATASET_ID")
    if mode != "crawler_api" and config.endpoint.rstrip("/") == "https://api.brightdata.com/request" and not config.zone:
        missing.append("BRIGHTDATA_ZONE")
    return {
        "provider": "brightdata",
        "provider_mode": _provider_mode(mode),
        "live": False,
        "endpoint": config.endpoint,
        "params": build_brightdata_request_params(mode, config),
        "payload": build_brightdata_request_payload(url, mode, config),
        "headers": {"Authorization": "Bearer ***", "Content-Type": "application/json"},
        "config_status": {
            "config_available": True,
            "live_ready": bool(config.api_key) and not missing,
            "missing_env_vars": missing,
            **config.masked(),
        },
    }


def _first_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
        return {}
    return raw if isinstance(raw, dict) else {}


def _first_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def normalize_brightdata_response(
    raw_response: Any,
    requested_url: str,
    *,
    mode: str = "browser_api",
    raw_response_path: str = "",
) -> NormalizedScrapeResult:
    raw = _first_dict(raw_response)
    nested = _first_dict(raw.get("data") or raw.get("result") or raw.get("response") or {})
    merged = {**raw, **nested}
    extracted = extract_body_like_content(raw_response)
    html = extracted["html"]
    markdown = extracted["markdown"]
    page_text = extracted["page_text"]
    counts = count_heading_table_link_image(html=html, markdown=markdown)
    title = str(merged.get("title") or merged.get("page_title") or merged.get("pageTitle") or "")
    if not title and html:
        import re

        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
        if m:
            title = clean_html_text(m.group(1))
    error = str(merged.get("error") or "")
    status_code = merged.get("status_code") or merged.get("statusCode") or merged.get("http_status")
    if extracted["parse_error_category"] in {"blocked_or_verification_page", "request_validation_failed", "wrong_provider_mode_or_payload", "unsupported_response_shape"}:
        error = error or extracted["parse_error"]
    success = bool(page_text.strip()) and not error and bool(extracted["parse_success"])
    final_url = str(merged.get("final_url") or merged.get("finalUrl") or merged.get("url") or requested_url)
    wc = count_words(page_text)
    return NormalizedScrapeResult(
        provider="brightdata",
        provider_mode=_provider_mode(mode),
        requested_url=requested_url,
        final_url=final_url,
        normalized_url=normalized_url_for(final_url, requested_url),
        status_code=int(status_code) if str(status_code or "").isdigit() else None,
        success=success,
        error="" if success else (error or extracted["parse_error"] or "No body-like field found"),
        fetched_at=utc_now_iso(),
        html=html,
        markdown=markdown,
        text=page_text,
        title=title,
        meta_description=str(merged.get("meta_description") or merged.get("metaDescription") or merged.get("description") or ""),
        raw_response_path=raw_response_path,
        text_char_count=len(page_text),
        word_count=wc,
        heading_count=counts["heading_count"],
        table_count=counts["table_count"],
        link_count=counts["link_count"],
        image_count=counts["image_count"],
        content_quality_flag=extracted["content_quality_flag"],
    )


def scrape_url_brightdata(
    url: str,
    mode: str,
    config: BrightDataSettings,
    *,
    live: bool = False,
    raw_response_path: str | Path = "",
    request_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if not live:
        return {
            "planned_request": dry_run_brightdata_request(url, mode, config),
            "normalized_result": NormalizedScrapeResult(
                provider="brightdata",
                provider_mode=_provider_mode(mode),
                requested_url=url,
                final_url=url,
                normalized_url=normalized_url_for(url, url),
                status_code=None,
                success=False,
                error="dry_run_no_api_call",
                fetched_at=utc_now_iso(),
                html="",
                markdown="",
                text="",
                title="",
                meta_description="",
                raw_response_path=str(raw_response_path),
                text_char_count=0,
                word_count=0,
                heading_count=0,
                table_count=0,
                link_count=0,
                image_count=0,
                content_quality_flag="parse_failed",
            ).to_dict(),
        }
    if not config.api_key:
        raise RuntimeError("Live Bright Data execution requires BRIGHTDATA_API_KEY.")
    if mode == "crawler_api" and not config.dataset_id:
        raise RuntimeError("Live Bright Data Crawler API execution requires BRIGHTDATA_CRAWLER_DATASET_ID.")
    if mode != "crawler_api" and config.endpoint.rstrip("/") == "https://api.brightdata.com/request" and not config.zone:
        raise RuntimeError("Live Bright Data execution missing required env vars: BRIGHTDATA_ZONE.")
    request_fn = request_fn or requests.post
    payload = build_brightdata_request_payload(url, mode, config)
    params = build_brightdata_request_params(mode, config)
    requested_url = str(payload["input"][0]["url"] if mode == "crawler_api" else payload["url"])
    try:
        request_kwargs: dict[str, Any] = {
            "headers": {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            "json": payload,
            "timeout": config.timeout_seconds,
        }
        if params:
            request_kwargs["params"] = params
        response = request_fn(config.endpoint, **request_kwargs)
        try:
            raw = response.json()
        except ValueError:
            raw = {"text": response.text, "status_code": getattr(response, "status_code", None)}
    except requests.RequestException as exc:
        raw = {
            "error": type(exc).__name__,
            "message": str(exc),
            "status_code": None,
            "request_failed": True,
        }
        result = normalize_brightdata_response(raw, requested_url, mode=mode, raw_response_path=str(raw_response_path))
        result.success = False
        result.error = str(exc)
        return {
            "raw_response": raw,
            "request_payload": payload,
            "response_headers": {},
            "normalized_result": result.to_dict(),
        }
    result = normalize_brightdata_response(raw, requested_url, mode=mode, raw_response_path=str(raw_response_path))
    response_headers = dict(getattr(response, "headers", {}) or {})
    brightdata_status = str(response_headers.get("x-brd-status-code") or "")
    brightdata_error = str(response_headers.get("x-brd-err-msg") or response_headers.get("x-brd-error") or "")
    # Bright Data can send an outer HTTP 200 while exposing a proxy failure in x-brd-* headers.
    if brightdata_error or (brightdata_status and not brightdata_status.startswith("2")):
        result.success = False
        result.status_code = int(brightdata_status) if brightdata_status.isdigit() else result.status_code
        result.error = brightdata_error or f"Bright Data proxy status {brightdata_status}"
    if getattr(response, "status_code", 0) and not (200 <= int(response.status_code) < 300):
        result.success = False
        result.error = result.error or f"HTTP {response.status_code}"
        result.status_code = int(response.status_code)
    if raw_response_path:
        write_json(raw_response_path, raw)
    return {
        "raw_response": raw,
        "request_payload": payload,
        "request_params": params,
        "response_headers": response_headers,
        "normalized_result": result.to_dict(),
    }
