from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import requests

from src.econometrics_eda_v2.brightdata_response_parser import extract_body_like_content
from src.econometrics_eda_v2.io import utc_now_iso


@dataclass(frozen=True)
class BrightDataConfig:
    api_key: str
    endpoint: str
    zone: str = ""
    scraper_id: str = ""
    dataset_id: str = ""
    country: str = ""
    render_js: bool | None = None


def _bool_env(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


def check_brightdata_config(env: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ if env is None else env
    missing: list[str] = []
    api_key = env.get("BRIGHTDATA_API_KEY", "")
    endpoint = env.get("BRIGHTDATA_ENDPOINT", "")
    if not api_key:
        missing.append("BRIGHTDATA_API_KEY")
    if not endpoint:
        missing.append("BRIGHTDATA_ENDPOINT")
    config = BrightDataConfig(
        api_key=api_key,
        endpoint=endpoint,
        zone=env.get("BRIGHTDATA_ZONE", ""),
        scraper_id=env.get("BRIGHTDATA_SCRAPER_ID", ""),
        dataset_id=env.get("BRIGHTDATA_DATASET_ID", ""),
        country=env.get("BRIGHTDATA_COUNTRY", ""),
        render_js=_bool_env(env.get("BRIGHTDATA_RENDER_JS")),
    )
    return {
        "ok": not missing,
        "missing": missing,
        "config": config,
        "api_key_present": bool(api_key),
        "endpoint_present": bool(endpoint),
        "optional_present": {
            "BRIGHTDATA_ZONE": bool(config.zone),
            "BRIGHTDATA_SCRAPER_ID": bool(config.scraper_id),
            "BRIGHTDATA_DATASET_ID": bool(config.dataset_id),
            "BRIGHTDATA_COUNTRY": bool(config.country),
            "BRIGHTDATA_RENDER_JS": config.render_js is not None,
        },
    }


def _first_text(raw: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _first_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
        return {}
    return raw if isinstance(raw, dict) else {}


def normalize_brightdata_response(raw_response: Any, requested_url: str) -> dict[str, Any]:
    raw = _first_dict(raw_response)
    nested = _first_dict(raw.get("data") or raw.get("result") or raw.get("response") or {})
    merged = {**raw, **nested}
    extracted = extract_body_like_content(raw_response)
    html = extracted["html"]
    markdown = extracted["markdown"]
    text = extracted["page_text"]
    status_code = merged.get("status_code") or merged.get("statusCode") or merged.get("http_status") or merged.get("httpStatus")
    provider_status = str(merged.get("provider_status") or merged.get("status") or ("success" if html or markdown or text else "unknown"))
    error = merged.get("error") or merged.get("message") if str(provider_status).lower() in {"error", "failed", "fail"} else merged.get("error")
    if extracted["parse_error_category"] in {"blocked_or_verification_page", "request_validation_failed", "wrong_provider_mode_or_payload", "unsupported_response_shape"}:
        error = error or extracted["parse_error"]
    success = not error and str(provider_status).lower() in {"success", "ok", "completed", "done", "unknown"} and bool(text) and bool(extracted["parse_success"])
    return {
        "provider": "brightdata",
        "requested_url": requested_url,
        "final_url": merged.get("final_url") or merged.get("finalUrl") or merged.get("url") or requested_url,
        "status_code": status_code,
        "provider_status": "success" if success else provider_status,
        "fetched_at": utc_now_iso(),
        "title": merged.get("title") or merged.get("page_title") or merged.get("pageTitle") or "",
        "meta_description": merged.get("meta_description") or merged.get("metaDescription") or merged.get("description") or "",
        "html": html,
        "markdown": markdown,
        "text": text,
        "raw_response": raw_response,
        "error": "" if success else str(error or "No body-like field found"),
        "success": bool(success),
    }


def build_brightdata_payload(url: str, config: BrightDataConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {"url": url}
    if config.zone:
        payload["zone"] = config.zone
    if config.scraper_id:
        payload["scraper_id"] = config.scraper_id
    if config.dataset_id:
        payload["dataset_id"] = config.dataset_id
    if config.country:
        payload["country"] = config.country
    if config.render_js is not None:
        payload["render_js"] = config.render_js
    return payload


def scrape_url_brightdata(
    url: str,
    config: BrightDataConfig,
    *,
    request_fn: Callable[..., Any] | None = None,
    timeout: int = 90,
) -> dict[str, Any]:
    request_fn = request_fn or requests.post
    try:
        response = request_fn(
            config.endpoint,
            headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
            json=build_brightdata_payload(url, config),
            timeout=timeout,
        )
        try:
            raw = response.json()
        except ValueError:
            raw = {"text": response.text}
        normalized = normalize_brightdata_response(raw, url)
        normalized["status_code"] = normalized.get("status_code") or getattr(response, "status_code", None)
        if getattr(response, "status_code", 0) and not (200 <= int(response.status_code) < 300):
            normalized["success"] = False
            normalized["provider_status"] = "failed"
            normalized["error"] = normalized.get("error") or f"HTTP {response.status_code}"
        return normalized
    except Exception as exc:  # noqa: BLE001
        return {
            "provider": "brightdata",
            "requested_url": url,
            "final_url": url,
            "status_code": None,
            "provider_status": "failed",
            "fetched_at": utc_now_iso(),
            "title": "",
            "meta_description": "",
            "html": "",
            "markdown": "",
            "text": "",
            "raw_response": {},
            "error": str(exc),
            "success": False,
        }


def scrape_urls_brightdata(
    urls: list[str],
    config: BrightDataConfig,
    *,
    max_urls: int | None = None,
    request_fn: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    selected = urls[:max_urls] if max_urls is not None else urls
    return [scrape_url_brightdata(url, config, request_fn=request_fn) for url in selected]
