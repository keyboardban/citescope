from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd

from src.econometrics_eda_v2.io import QUEUE_DIR
from src.econometrics_eda_v2.io import write_json
from src.econometrics_eda_v2.parse_pages import detect_language
from src.econometrics_eda_v2.scrape_providers.provider_quality import (
    body_text_from_parts,
    clean_html_text,
    count_heading_table_link_image,
    count_words,
    infer_content_quality_flag,
    normalized_url_for,
)
from src.econometrics_eda_v2.scrape_quality_audit import _excerpt
from src.url_utils import domain as url_domain
from src.url_utils import normalize_url


BODY_KEYS = {
    "body",
    "content",
    "html",
    "markdown",
    "md",
    "text",
    "page_text",
    "pagetext",
    "main_text",
    "maintext",
    "maincontent",
    "main_content",
    "pagecontent",
    "page_content",
    "body_html",
    "content_html",
    "page_html",
    "html_content",
    "rendered_html",
    "raw_html",
    "browser_html",
    "article",
    "articlebody",
    "article_body",
}

METADATA_KEYS = {
    "url",
    "source_url",
    "final_url",
    "finalurl",
    "requested_url",
    "requested_normalized_url",
    "final_normalized_url",
    "normalized_url",
    "raw_response_path",
    "raw_cache_path",
    "status",
    "status_code",
    "statuscode",
    "http_status",
    "httpstatus",
    "provider_status",
    "error",
    "message",
    "title",
    "page_title",
    "pagetitle",
    "description",
    "meta_description",
    "metadescription",
    "zone",
    "country",
    "endpoint",
    "provider",
    "provider_mode",
    "reason_selected",
    "benchmark_id",
    "scrape_id",
    "fetched_at",
    "planned_request",
}

BLOCKED_PATTERNS = re.compile(
    r"(?is)\b("
    r"captcha|recaptcha|hcaptcha|human verification|verify you are human|"
    r"checking your browser|just a moment|please wait|cloudflare|access denied|"
    r"forbidden|request blocked|access blocked|you have been blocked|bot detection|"
    r"robot|enable javascript|rate limit|too many requests"
    r")\b|(?:\b403\b|\b429\b)"
)

HTML_RE = re.compile(r"(?is)<!doctype html|<html\b|<body\b|<main\b|<article\b|<section\b|<div\b")
HTML_TAG_RE = re.compile(r"(?is)<[a-z][^>]{0,120}>")


@dataclass(frozen=True)
class BodyCandidate:
    path: str
    key: str
    kind: str
    text: str
    char_count: int
    score: float


def _last_key(path: str) -> str:
    key = path.split(".")[-1].strip("[]0123456789")
    return re.sub(r"[^a-z0-9_]", "", key.casefold())


def _kind_for(path: str, text: str) -> str:
    key = _last_key(path)
    low_path = path.casefold()
    if "markdown" in low_path or key == "md":
        return "markdown"
    if "html" in low_path or HTML_RE.search(text[:2500]) or HTML_TAG_RE.search(text[:1000]):
        return "html"
    return "text"


def _iter_strings(value: Any, path: str = "$", depth: int = 0) -> list[tuple[str, str]]:
    if depth > 9:
        return []
    if isinstance(value, str):
        return [(path, value)]
    if isinstance(value, dict):
        rows: list[tuple[str, str]] = []
        for key, item in value.items():
            rows.extend(_iter_strings(item, f"{path}.{key}", depth + 1))
        return rows
    if isinstance(value, list):
        rows = []
        for idx, item in enumerate(value[:50]):
            rows.extend(_iter_strings(item, f"{path}[{idx}]", depth + 1))
        return rows
    return []


def _candidate_score(path: str, text: str) -> float:
    stripped = text.strip()
    key = _last_key(path)
    if not stripped:
        return -1000
    if key in METADATA_KEYS and key not in BODY_KEYS:
        return -50
    kind = _kind_for(path, stripped)
    score = min(len(stripped), 50000) / 1000
    if key in BODY_KEYS:
        score += 50
    if kind == "html":
        score += 20
    if kind == "markdown":
        score += 12
    if count_words(clean_html_text(stripped) if kind == "html" else stripped) >= 20:
        score += 10
    if BLOCKED_PATTERNS.search(stripped[:4000]):
        score -= 8
    if len(stripped) < 30 and key not in BODY_KEYS:
        score -= 20
    return score


def _body_candidates(raw_response: Any) -> list[BodyCandidate]:
    rows = []
    for path, text in _iter_strings(raw_response):
        stripped = text.strip()
        key = _last_key(path)
        kind = _kind_for(path, stripped)
        body_named = key in BODY_KEYS or any(part in path.casefold() for part in (".data.", ".result.", ".response."))
        body_shaped = bool(HTML_RE.search(stripped[:2500]) or len(stripped) >= 100)
        if path == "$" or body_named or body_shaped:
            score = _candidate_score(path, stripped)
            if score > -10:
                rows.append(
                    BodyCandidate(
                        path=path,
                        key=key,
                        kind=kind,
                        text=stripped,
                        char_count=len(stripped),
                        score=score,
                    )
                )
    return sorted(rows, key=lambda c: (c.score, c.char_count), reverse=True)


def _largest_string(raw_response: Any) -> tuple[str, int]:
    strings = [(path, len(text.strip())) for path, text in _iter_strings(raw_response) if text.strip()]
    if not strings:
        return "", 0
    return max(strings, key=lambda item: item[1])


def _first_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
    return {}


def _value_from_paths(raw: Any, names: tuple[str, ...]) -> Any:
    for path, text in _iter_strings(raw):
        if _last_key(path) in names and text.strip():
            return text.strip()
    mapping = _first_mapping(raw)
    for name in names:
        if mapping.get(name):
            return mapping.get(name)
    return ""


def _status_code(raw: Any) -> int | None:
    names = {"status_code", "statuscode", "http_status", "httpstatus"}
    for path, text in _iter_strings(raw):
        if _last_key(path) in names and str(text).strip().isdigit():
            return int(str(text).strip())
    mapping = _first_mapping(raw)
    for key in ("status_code", "statusCode", "http_status", "httpStatus"):
        value = mapping.get(key)
        if str(value or "").isdigit():
            return int(value)
    return None


def _response_shape(raw: Any) -> str:
    if isinstance(raw, str):
        return "root_string_html" if HTML_RE.search(raw[:2500]) else "root_string"
    if isinstance(raw, list):
        return "root_list"
    if not isinstance(raw, dict):
        return type(raw).__name__
    keys = {str(k).casefold() for k in raw}
    if "raw_response" in keys:
        return "normalized_wrapper_with_raw_response"
    if keys & {"html", "markdown", "text", "content", "body"}:
        return "body_fields_at_root"
    if keys & {"data", "result", "response"}:
        return "nested_response_wrapper"
    if keys <= METADATA_KEYS | {"fetched_at", "planned_request", "benchmark_id", "source_url", "reason_selected", "success"}:
        return "metadata_only_wrapper"
    return "dict_other"


def extract_body_like_content(raw_response: Any) -> dict[str, Any]:
    candidates = _body_candidates(raw_response)
    selected = candidates[0] if candidates else None
    largest_path, largest_len = _largest_string(raw_response)
    html = markdown = text_raw = ""
    if selected:
        if selected.kind == "html":
            html = selected.text
        elif selected.kind == "markdown":
            markdown = selected.text
        else:
            text_raw = selected.text
    page_text = body_text_from_parts(html=html, markdown=markdown, text=text_raw)
    visible_text = clean_html_text(selected.text) if selected and selected.kind == "html" else page_text
    blocked = bool(BLOCKED_PATTERNS.search(" ".join([str(_value_from_paths(raw_response, ("title", "page_title", "pagetitle"))), visible_text[:5000]])))
    body_available = bool(page_text.strip())
    status = _status_code(raw_response)
    status_error = status is not None and status >= 400
    raw_empty = raw_response in (None, "", {}, [])
    provider_error = str(_value_from_paths(raw_response, ("error", "message")) or "")
    provider_error_low = provider_error.casefold()
    if "request validation failed" in provider_error_low:
        category = "request_validation_failed"
        parse_error = provider_error or "Request validation failed"
        quality = "blocked_or_error_page"
        success = False
    elif raw_empty:
        category = "empty_response"
        parse_error = "Empty Bright Data raw_response"
        quality = "empty_text"
        success = False
    elif blocked:
        category = "blocked_or_verification_page"
        parse_error = "Blocked or verification page"
        quality = "blocked_or_error_page"
        success = False
    elif status_error:
        category = "wrong_provider_mode_or_payload" if status == 400 else "blocked_or_verification_page" if status in {403, 429} else "unsupported_response_shape"
        parse_error = f"HTTP {status}"
        quality = "blocked_or_error_page"
        success = False
    elif not body_available:
        shape = _response_shape(raw_response)
        category = "metadata_only_response" if "metadata_only" in shape else "no_body_field_detected"
        parse_error = "No body field detected"
        quality = "parse_failed"
        success = False
    elif count_words(page_text) < 20:
        category = "body_too_short"
        parse_error = "Body text too short"
        quality = "very_short_text"
        success = False
    else:
        category = "ok"
        parse_error = ""
        wc = count_words(page_text)
        quality = infer_content_quality_flag(success=True, title=str(_value_from_paths(raw_response, ("title", "page_title", "pagetitle"))), text=page_text, word_count=wc)
        success = True
    return {
        "html": html,
        "markdown": markdown,
        "text_raw": text_raw,
        "page_text": page_text,
        "body_field_selected": selected.path if selected else "",
        "body_field_kind": selected.kind if selected else "",
        "body_field_char_count": selected.char_count if selected else 0,
        "body_field_candidate_count": len(candidates),
        "body_field_candidates": "; ".join(f"{c.path}:{c.kind}:{c.char_count}" for c in candidates[:8]),
        "largest_string_field_path": largest_path,
        "largest_string_length": largest_len,
        "scraped_body_available": body_available,
        "body_available_but_not_main_content": bool(body_available and (blocked or count_words(page_text) < 20)),
        "parse_success": bool(success),
        "parse_error": parse_error,
        "parse_error_category": category,
        "content_quality_flag": quality,
        "blocked_or_verification_detected": blocked,
        "status_code_detected": status,
    }


def _target_raw_response(cache_payload: Any) -> tuple[Any, bool]:
    if isinstance(cache_payload, dict) and "raw_response" in cache_payload:
        return cache_payload.get("raw_response"), True
    return None, False


def _raw_top_level_keys(raw_response: Any) -> list[str]:
    return list(raw_response.keys()) if isinstance(raw_response, dict) else []


def sanitize_brightdata_value(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if str(key).strip().casefold() in {"authorization", "api_key", "apikey", "token", "password", "secret"}:
                out[key] = "***"
            else:
                out[key] = sanitize_brightdata_value(item)
        return out
    if isinstance(value, list):
        return [sanitize_brightdata_value(item) for item in value]
    return value


def brightdata_raw_cache_payload(
    *,
    benchmark_id: str,
    requested_url: str,
    provider_mode: str,
    fetched_at: str,
    status_code: Any,
    request_payload: dict[str, Any] | None,
    request_params: dict[str, Any] | None = None,
    response_headers: dict[str, Any] | None = None,
    raw_response: Any = None,
    error_if_request_failed: str = "",
    provider: str = "brightdata",
) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "requested_url": requested_url,
        "provider": provider,
        "provider_mode": provider_mode,
        "fetched_at": fetched_at,
        "status_code": status_code,
        "request_payload_sanitized": sanitize_brightdata_value(request_payload or {}),
        "request_params_sanitized": sanitize_brightdata_value(request_params or {}),
        "response_headers_sanitized": sanitize_brightdata_value(response_headers or {}),
        "raw_response": raw_response,
        "raw_response_type": type(raw_response).__name__,
        "raw_response_top_level_keys": _raw_top_level_keys(raw_response),
        "error_if_request_failed": error_if_request_failed or "",
    }


def write_brightdata_raw_cache(path: str | Path, payload: dict[str, Any], *, dry_run: bool = False, force: bool = False) -> None:
    if not dry_run and "raw_response" not in payload:
        raise ValueError("Refusing to save normalized result as raw cache: raw_response missing.")
    p = Path(path)
    if p.exists() and not force:
        return
    write_json(p, payload)


def brightdata_parsed_cache_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_id": parsed.get("benchmark_id"),
        "requested_url": parsed.get("requested_url"),
        "provider": parsed.get("provider") or "brightdata",
        "provider_mode": parsed.get("provider_mode"),
        "final_url": parsed.get("final_url"),
        "parse_success": parsed.get("parse_success"),
        "scraped_body_available": parsed.get("scraped_body_available"),
        "html": parsed.get("html", ""),
        "markdown": parsed.get("markdown", ""),
        "text": parsed.get("page_text", ""),
        "title": parsed.get("page_title", ""),
        "meta_description": parsed.get("meta_description", ""),
        "word_count": parsed.get("word_count", 0),
        "text_char_count": parsed.get("text_char_count", 0),
        "content_quality_flag": parsed.get("content_quality_flag", ""),
        "body_field_selected": parsed.get("body_field_selected", ""),
        "body_field_candidates": parsed.get("body_field_candidates", ""),
        "largest_string_field_path": parsed.get("largest_string_field_path", ""),
        "largest_string_length": parsed.get("largest_string_length", 0),
        "parse_error": parsed.get("parse_error", ""),
        "parse_error_category": parsed.get("parse_error_category", ""),
    }


def write_brightdata_parsed_cache(path: str | Path, parsed: dict[str, Any], *, force: bool = False) -> None:
    p = Path(path)
    if p.exists() and not force:
        return
    write_json(p, brightdata_parsed_cache_payload(parsed))


def _metadata(payload: dict[str, Any], raw: Any) -> dict[str, Any]:
    requested_url = str(payload.get("requested_url") or payload.get("source_url") or _value_from_paths(raw, ("requested_url", "url")) or "")
    final_url = str(payload.get("final_url") or _value_from_paths(raw, ("final_url", "finalurl", "url")) or requested_url)
    return {
        "requested_url": requested_url,
        "final_url": final_url,
        "normalized_url": normalize_url(str(payload.get("normalized_url") or final_url or requested_url or "")),
        "domain": url_domain(final_url or requested_url),
        "status_code": payload.get("status_code") or _status_code(raw),
        "page_title": payload.get("title") or payload.get("page_title") or _value_from_paths(raw, ("title", "page_title", "pagetitle")) or "",
        "meta_description": payload.get("meta_description") or _value_from_paths(raw, ("meta_description", "metadescription", "description")) or "",
        "error": payload.get("error") or _value_from_paths(raw, ("error", "message")) or "",
    }


def parse_brightdata_cache_payload(payload: dict[str, Any], *, path: str | Path = "") -> dict[str, Any]:
    raw, has_raw_response = _target_raw_response(payload)
    if not has_raw_response:
        meta = _metadata(payload, {})
        return {
            "scrape_id": payload.get("scrape_id") or payload.get("benchmark_id") or Path(path).stem.removesuffix(".raw"),
            "benchmark_id": payload.get("benchmark_id") or Path(path).stem.removesuffix(".raw"),
            "source_url": payload.get("source_url") or meta["requested_url"],
            "requested_url": meta["requested_url"],
            "final_url": meta["final_url"] or meta["requested_url"],
            "requested_normalized_url": normalize_url(meta["requested_url"]),
            "final_normalized_url": normalize_url(meta["final_url"] or meta["requested_url"]),
            "normalized_url": normalized_url_for(meta["final_url"], meta["requested_url"]),
            "domain": meta["domain"],
            "provider": payload.get("provider") or "brightdata",
            "provider_mode": payload.get("provider_mode") or "",
            "live_attempted": str(payload.get("provider_status") or "").casefold() not in {"planned_dry_run", "dry_run", ""},
            "scrape_success": False,
            "parse_success": False,
            "scraped_body_available": False,
            "can_reparse": False,
            "html": "",
            "markdown": "",
            "html_available": False,
            "markdown_available": False,
            "text_available": False,
            "page_title": meta["page_title"],
            "meta_description": meta["meta_description"],
            "page_text": "",
            "text_char_count": 0,
            "word_count": 0,
            "heading_count": 0,
            "table_count": 0,
            "link_count": 0,
            "image_count": 0,
            "language_detected": "",
            "status_code": meta["status_code"],
            "parse_error": "Old cache contains normalized failure only; cannot reparse",
            "parse_error_category": "raw_response_missing_from_cache",
            "content_quality_flag": "parse_failed",
            "page_text_excerpt": "",
            "raw_response_present": False,
            "response_shape": "raw_response_missing_from_cache",
            "body_field_selected": "",
            "body_field_kind": "",
            "body_field_char_count": 0,
            "body_field_candidate_count": 0,
            "body_field_candidates": "",
            "largest_string_field_path": "",
            "largest_string_length": 0,
            "body_available_but_not_main_content": False,
            "blocked_or_verification_detected": False,
        }
    extracted = extract_body_like_content(raw)
    meta = _metadata(payload, raw)
    html = extracted["html"]
    markdown = extracted["markdown"]
    page_text = extracted["page_text"]
    counts = count_heading_table_link_image(html=html, markdown=markdown)
    provider_status = str(payload.get("provider_status") or payload.get("status") or "").casefold()
    status_code = meta["status_code"]
    error = str(meta["error"] or extracted["parse_error"] or "")
    body_available = bool(page_text.strip())
    http_ok = status_code is None or not str(status_code).isdigit() or int(status_code) < 400
    scrape_success = bool(body_available and http_ok and not extracted["blocked_or_verification_detected"])
    if provider_status in {"success", "ok", "completed"} and body_available and http_ok:
        scrape_success = True
    if provider_status in {"failed", "fail", "error"} and not body_available:
        scrape_success = False
    parse_success = bool(body_available and scrape_success and not error.lower().startswith("http "))
    word_count = count_words(page_text)
    quality = extracted["content_quality_flag"]
    if not has_raw_response and not body_available and str(payload.get("provider_status") or "").casefold() == "planned_dry_run":
        quality = "parse_failed"
    return {
        "scrape_id": payload.get("scrape_id") or payload.get("benchmark_id") or Path(path).stem,
        "benchmark_id": payload.get("benchmark_id") or Path(path).stem,
        "source_url": payload.get("source_url") or meta["requested_url"],
        "requested_url": meta["requested_url"],
        "final_url": meta["final_url"],
        "requested_normalized_url": normalize_url(meta["requested_url"]),
        "final_normalized_url": normalize_url(meta["final_url"]),
        "normalized_url": normalized_url_for(meta["final_url"], meta["requested_url"]),
        "domain": meta["domain"],
        "provider": payload.get("provider") or "brightdata",
        "provider_mode": payload.get("provider_mode") or "",
        "live_attempted": str(payload.get("provider_status") or "").casefold() not in {"planned_dry_run", "dry_run", ""},
        "scrape_success": bool(scrape_success),
        "parse_success": bool(parse_success),
        "scraped_body_available": bool(body_available),
        "can_reparse": True,
        "html": html,
        "markdown": markdown,
        "html_available": bool(html.strip()),
        "markdown_available": bool(markdown.strip()),
        "text_available": bool(page_text.strip()),
        "page_title": meta["page_title"],
        "meta_description": meta["meta_description"],
        "page_text": page_text,
        "text_char_count": int(len(page_text)),
        "word_count": int(word_count),
        "heading_count": int(counts["heading_count"]),
        "table_count": int(counts["table_count"]),
        "link_count": int(counts["link_count"]),
        "image_count": int(counts["image_count"]),
        "language_detected": detect_language(page_text),
        "status_code": status_code,
        "parse_error": "" if parse_success else (error or extracted["parse_error"] or "No body field detected"),
        "parse_error_category": extracted["parse_error_category"] or ("ok" if parse_success else "parse_failed"),
        "content_quality_flag": quality,
        "page_text_excerpt": _excerpt(page_text),
        "raw_response_present": bool(has_raw_response),
        "response_shape": _response_shape(raw),
        "body_field_selected": extracted["body_field_selected"],
        "body_field_kind": extracted["body_field_kind"],
        "body_field_char_count": extracted["body_field_char_count"],
        "body_field_candidate_count": extracted["body_field_candidate_count"],
        "body_field_candidates": extracted["body_field_candidates"],
        "largest_string_field_path": extracted["largest_string_field_path"],
        "largest_string_length": extracted["largest_string_length"],
        "body_available_but_not_main_content": extracted["body_available_but_not_main_content"],
        "blocked_or_verification_detected": extracted["blocked_or_verification_detected"],
    }


def parse_brightdata_raw_cache(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text("utf-8"))
    if not isinstance(payload, dict):
        payload = {"raw_response": payload, "benchmark_id": p.stem}
    return parse_brightdata_cache_payload(payload, path=p)


def _looks_like_normalized_failed_result(payload: Any) -> bool:
    if not isinstance(payload, dict) or "raw_response" in payload:
        return False
    keys = {str(k).casefold() for k in payload}
    normalized_keys = {"success", "error", "html", "markdown", "text", "word_count", "content_quality_flag"}
    failed = payload.get("success") is False or str(payload.get("provider_status") or "").casefold() in {"failed", "fail", "error"}
    return bool(failed and keys & normalized_keys)


def _is_dry_run_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        str(payload.get("provider_status") or "").casefold() in {"planned_dry_run", "dry_run"}
        or str(payload.get("error") or "").casefold() == "dry_run_no_api_call"
    )


def _has_body_like_field(payload: Any) -> bool:
    raw, has_raw = _target_raw_response(payload)
    if not has_raw:
        return False
    return bool(_body_candidates(raw))


def cache_integrity_row(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    benchmark_id = p.name
    if benchmark_id.endswith(".raw.json"):
        benchmark_id = benchmark_id[:-9]
    elif benchmark_id.endswith(".json"):
        benchmark_id = benchmark_id[:-5]
    if not p.exists():
        return {
            "benchmark_id": benchmark_id,
            "raw_cache_path": str(p),
            "has_raw_response": False,
            "looks_like_normalized_failed_result": False,
            "has_body_like_field": False,
            "can_reparse": False,
            "cache_status": "missing_file",
        }
    try:
        payload = json.loads(p.read_text("utf-8"))
    except Exception:  # noqa: BLE001
        return {
            "benchmark_id": benchmark_id,
            "raw_cache_path": str(p),
            "has_raw_response": False,
            "looks_like_normalized_failed_result": False,
            "has_body_like_field": False,
            "can_reparse": False,
            "cache_status": "malformed_json",
        }
    has_raw = isinstance(payload, dict) and "raw_response" in payload
    has_body = _has_body_like_field(payload)
    dry_run = _is_dry_run_payload(payload)
    normalized_failed = _looks_like_normalized_failed_result(payload)
    if has_raw:
        status = "valid_raw_response_available"
        can_reparse = True
    elif dry_run:
        status = "dry_run_only"
        can_reparse = False
    else:
        status = "normalized_failure_only_unrecoverable"
        can_reparse = False
    return {
        "benchmark_id": payload.get("benchmark_id") if isinstance(payload, dict) and payload.get("benchmark_id") else benchmark_id,
        "raw_cache_path": str(p),
        "has_raw_response": bool(has_raw),
        "looks_like_normalized_failed_result": bool(normalized_failed),
        "has_body_like_field": bool(has_body),
        "can_reparse": bool(can_reparse),
        "cache_status": status,
    }


def build_cache_integrity_audit(input_dir: str | Path) -> pd.DataFrame:
    root = Path(input_dir)
    files = sorted(root.glob("*.raw.json")) + sorted(p for p in root.glob("*.json") if not p.name.endswith(".raw.json"))
    rows = [cache_integrity_row(path) for path in files]
    return pd.DataFrame(
        rows,
        columns=[
            "benchmark_id",
            "raw_cache_path",
            "has_raw_response",
            "looks_like_normalized_failed_result",
            "has_body_like_field",
            "can_reparse",
            "cache_status",
        ],
    )


def build_unrecoverable_cache_retry_queue(integrity: pd.DataFrame, benchmark_input: pd.DataFrame | None = None) -> pd.DataFrame:
    if integrity.empty:
        return pd.DataFrame(columns=["benchmark_id", "source_url", "provider_mode", "retry_reason", "previous_error", "recommended_action"])
    sub = integrity[integrity["cache_status"].eq("normalized_failure_only_unrecoverable")].copy()
    if benchmark_input is not None and not benchmark_input.empty and "benchmark_id" in benchmark_input.columns:
        keep = [c for c in ["benchmark_id", "source_url", "normalized_url", "recommended_brightdata_mode"] if c in benchmark_input.columns]
        sub = sub.merge(benchmark_input[keep].drop_duplicates("benchmark_id"), on="benchmark_id", how="left")
    rows = []
    for _, row in sub.iterrows():
        previous_error = ""
        try:
            payload = json.loads(Path(str(row.get("raw_cache_path"))).read_text("utf-8"))
            if isinstance(payload, dict):
                previous_error = str(payload.get("error") or "")
        except Exception:  # noqa: BLE001
            previous_error = ""
        rows.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "source_url": row.get("source_url") or row.get("normalized_url") or "",
                "provider_mode": row.get("recommended_brightdata_mode") or "",
                "retry_reason": "raw_response_missing_from_old_cache",
                "previous_error": previous_error,
                "recommended_action": "rerun_live_after_cache_fix",
            }
        )
    return pd.DataFrame(rows, columns=["benchmark_id", "source_url", "provider_mode", "retry_reason", "previous_error", "recommended_action"])


def audit_raw_response_shape(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    try:
        payload = json.loads(p.read_text("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "benchmark_id": p.stem,
            "raw_cache_path": str(p),
            "read_error": str(exc),
            "response_shape": "unreadable_json",
        }
    raw, has_raw = _target_raw_response(payload if isinstance(payload, dict) else {"raw_response": payload})
    candidates = _body_candidates(raw)
    top_keys = list(payload.keys()) if isinstance(payload, dict) else []
    raw_keys = list(raw.keys()) if isinstance(raw, dict) else []
    return {
        "benchmark_id": payload.get("benchmark_id") if isinstance(payload, dict) else p.stem,
        "source_url": payload.get("source_url", "") if isinstance(payload, dict) else "",
        "provider_mode": payload.get("provider_mode", "") if isinstance(payload, dict) else "",
        "provider_status": payload.get("provider_status", "") if isinstance(payload, dict) else "",
        "status_code": payload.get("status_code", "") if isinstance(payload, dict) else "",
        "error": payload.get("error", "") if isinstance(payload, dict) else "",
        "raw_cache_path": str(p),
        "top_level_type": type(payload).__name__,
        "top_level_key_count": len(top_keys),
        "top_level_keys": "|".join(map(str, top_keys[:50])),
        "raw_response_present": bool(has_raw),
        "raw_response_type": type(raw).__name__,
        "raw_response_key_count": len(raw_keys),
        "raw_response_keys": "|".join(map(str, raw_keys[:50])),
        "response_shape": _response_shape(raw),
        "body_candidate_count": len(candidates),
        "selected_body_field": candidates[0].path if candidates else "",
        "selected_body_kind": candidates[0].kind if candidates else "",
        "selected_body_char_count": candidates[0].char_count if candidates else 0,
        "candidate_fields": "; ".join(f"{c.path}:{c.kind}:{c.char_count}" for c in candidates[:8]),
    }


def build_parser_before_after(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    if before.empty:
        before = pd.DataFrame(columns=["benchmark_id"])
    b = before.drop_duplicates("benchmark_id").add_prefix("before_")
    a = after.drop_duplicates("benchmark_id").add_prefix("after_")
    joined = a.merge(b, left_on="after_benchmark_id", right_on="before_benchmark_id", how="left")
    rows = []
    for _, row in joined.iterrows():
        rows.append(
            {
                "benchmark_id": row.get("after_benchmark_id"),
                "source_url": row.get("after_source_url"),
                "before_parse_success": row.get("before_parse_success"),
                "after_parse_success": row.get("after_parse_success"),
                "before_scrape_success": row.get("before_scrape_success"),
                "after_scrape_success": row.get("after_scrape_success"),
                "before_word_count": row.get("before_word_count"),
                "after_word_count": row.get("after_word_count"),
                "before_error": row.get("before_parse_error") or row.get("before_error"),
                "after_error": row.get("after_parse_error"),
                "after_parse_error_category": row.get("after_parse_error_category"),
                "after_response_shape": row.get("after_response_shape"),
                "after_body_field_selected": row.get("after_body_field_selected"),
                "fixed_by_parser": bool(not _truthy(row.get("before_parse_success")) and _truthy(row.get("after_parse_success"))),
            }
        )
    return pd.DataFrame(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def build_failure_triage(parse_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in parse_rows.iterrows():
        mode = str(row.get("provider_mode") or "")
        category = str(row.get("parse_error_category") or "")
        status = row.get("status_code")
        flag = str(row.get("content_quality_flag") or "")
        word_count = int(pd.to_numeric(pd.Series([row.get("word_count")]), errors="coerce").fillna(0).iloc[0])
        if _truthy(row.get("parse_success")):
            bucket = "parsed_ok"
            cause = "Usable body text extracted."
            action = "no_retry_needed"
            queue = ""
        elif category in {"raw_response_missing_from_cache", "metadata_only_response", "no_body_like_field"} and not _truthy(row.get("raw_response_present")):
            bucket = "raw_response_not_preserved"
            cause = "Cache contains normalized metadata only; original Bright Data response was overwritten."
            action = "rerun_after_raw_response_cache_fix"
            queue = "payload_fix"
        elif category in {"request_validation_failed", "wrong_provider_mode_or_payload", "unsupported_response_shape"} and not str(status).startswith(("403", "429")):
            bucket = "request_or_provider_http_error"
            cause = "Bright Data returned an HTTP error such as request validation failure."
            action = "fix_payload_or_retry_same_mode"
            queue = "payload_fix"
        elif category == "blocked_or_verification_page" or flag == "blocked_or_error_page" or str(status).startswith(("403", "429")):
            bucket = "blocked_or_verification"
            cause = "Returned a block, captcha, rate limit, or verification page."
            action = "try_unlocker_api" if "browser" in mode else "try_browser_api_if_unlocker_is_metadata_only"
            queue = "unlocker_api" if "browser" in mode else "browser_api"
        elif str(status).startswith("5"):
            bucket = "provider_http_error"
            cause = "Bright Data returned a 5xx response."
            action = "retry_same_mode_later"
            queue = "payload_fix"
        elif word_count < 100 and _truthy(row.get("scraped_body_available")):
            bucket = "short_or_boilerplate"
            cause = "A body was extracted but it is too short or boilerplate-like."
            action = "try_browser_api_rendered_markdown" if "unlocker" in mode else "try_unlocker_api"
            queue = "browser_api" if "unlocker" in mode else "unlocker_api"
        else:
            bucket = "parser_or_payload_shape"
            cause = "No supported body-like field was found in the available response shape."
            action = "inspect_payload_shape_then_retry"
            queue = "payload_fix"
        rows.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "source_url": row.get("source_url"),
                "normalized_url": row.get("normalized_url"),
                "domain": row.get("domain"),
                "provider_mode": mode,
                "live_attempted": row.get("live_attempted"),
                "status_code": status,
                "scrape_success": row.get("scrape_success"),
                "parse_success": row.get("parse_success"),
                "scraped_body_available": row.get("scraped_body_available"),
                "word_count": word_count,
                "content_quality_flag": flag,
                "parse_error": row.get("parse_error"),
                "parse_error_category": category,
                "response_shape": row.get("response_shape"),
                "raw_response_present": row.get("raw_response_present"),
                "body_field_selected": row.get("body_field_selected"),
                "failure_bucket": bucket,
                "likely_root_cause": cause,
                "recommended_next_action": action,
                "retry_queue": queue,
            }
        )
    return pd.DataFrame(rows)


def retry_queue_frames(triage: pd.DataFrame) -> dict[str, pd.DataFrame]:
    columns = ["benchmark_id", "source_url", "normalized_url", "domain", "provider_mode", "recommended_mode", "failure_bucket", "recommended_next_action"]
    queues: dict[str, pd.DataFrame] = {}
    for queue, mode in (("browser_api", "browser_api"), ("unlocker_api", "unlocker_api"), ("payload_fix", "")):
        sub = triage[triage["retry_queue"].eq(queue)].copy() if len(triage) else pd.DataFrame(columns=triage.columns)
        sub["recommended_mode"] = mode or sub.get("provider_mode", "")
        queues[queue] = sub[[c for c in columns if c in sub.columns]]
    return queues


def retry_queue_path(queue: str) -> Path:
    if queue == "browser_api":
        return QUEUE_DIR / "brightdata_retry_browser_api_queue.csv"
    if queue == "unlocker_api":
        return QUEUE_DIR / "brightdata_retry_unlocker_api_queue.csv"
    return QUEUE_DIR / "brightdata_retry_payload_fix_queue.csv"
