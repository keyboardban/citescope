from __future__ import annotations

import glob
import json
import re
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd

from src import apify_runner
from src.url_utils import domain as url_domain
from src.url_utils import normalize_url


def clean_html_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html or "")
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"[\w\u0E00-\u0E7F]+", text or "", flags=re.UNICODE))


def detect_language(text: str) -> str:
    if not text:
        return ""
    thai = len(re.findall(r"[\u0E00-\u0E7F]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if thai > latin:
        return "th"
    if latin:
        return "en"
    return "unknown"


def parse_raw_cache(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text("utf-8"))
    norm_page = {}
    if isinstance(data.get("raw_item"), dict) and data.get("raw_item"):
        try:
            pages = apify_runner.normalize_pages([data["raw_item"]])
            norm_page = pages[0] if pages else {}
        except Exception:
            norm_page = {}
    html = data.get("html") or ""
    markdown = data.get("markdown") or norm_page.get("markdown") or ""
    text = data.get("text") or norm_page.get("text") or ""
    page_text = text.strip() or re.sub(r"(?m)^#+\s*", "", markdown).strip() or clean_html_text(html)
    body_available = bool(page_text.strip())
    heading_count = 0
    heading_count += len(re.findall(r"(?m)^#{1,6}\s+", markdown or ""))
    heading_count += len(re.findall(r"(?is)<h[1-6][^>]*>", html or ""))
    table_count = len(re.findall(r"(?is)<table\b", html or "")) + len(re.findall(r"(?m)^\s*\|.+\|\s*$", markdown or "")) // 2
    link_count = len(re.findall(r"(?is)<a\b", html or "")) + len(re.findall(r"\[[^\]]+\]\([^)]+\)", markdown or ""))
    image_count = len(re.findall(r"(?is)<img\b", html or "")) + len(re.findall(r"!\[[^\]]*\]\([^)]+\)", markdown or ""))
    requested_url = data.get("requested_url") or norm_page.get("url") or ""
    final_url = data.get("final_url") or norm_page.get("final_url") or requested_url
    error = data.get("error") or ""
    provider_status = str(data.get("provider_status") or "").lower()
    scrape_success = provider_status in {"success", "ok", "completed"} and not error
    return {
        "scrape_id": data.get("scrape_id") or p.stem,
        "requested_url": requested_url,
        "final_url": final_url,
        "requested_normalized_url": normalize_url(requested_url),
        "final_normalized_url": normalize_url(final_url),
        "normalized_url": normalize_url(final_url or requested_url or ""),
        "domain": url_domain(final_url or requested_url or ""),
        "scrape_success": bool(scrape_success),
        "parse_success": bool(body_available and not error),
        "scraped_body_available": bool(body_available),
        "html_available": bool(html),
        "markdown_available": bool(markdown),
        "text_available": bool(text),
        "page_title": data.get("title") or norm_page.get("title") or "",
        "meta_description": data.get("meta_description") or norm_page.get("description") or "",
        "page_text": page_text,
        "text_char_count": int(len(page_text)),
        "word_count": int(count_words(page_text)),
        "heading_count": int(heading_count),
        "table_count": int(table_count),
        "link_count": int(link_count),
        "image_count": int(image_count),
        "language_detected": detect_language(page_text),
        "parse_error": "" if body_available else (error or "No body-like field found"),
    }


def parse_scrape_dir(input_dir: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(glob.glob(str(Path(input_dir) / "*.json")))
    rows = []
    for f in files:
        try:
            rows.append(parse_raw_cache(f))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "scrape_id": Path(f).stem,
                    "requested_url": "",
                    "final_url": "",
                    "requested_normalized_url": "",
                    "final_normalized_url": "",
                    "normalized_url": "",
                    "domain": "",
                    "scrape_success": False,
                    "parse_success": False,
                    "scraped_body_available": False,
                    "html_available": False,
                    "markdown_available": False,
                    "text_available": False,
                    "page_title": "",
                    "meta_description": "",
                    "page_text": "",
                    "text_char_count": 0,
                    "word_count": 0,
                    "heading_count": 0,
                    "table_count": 0,
                    "link_count": 0,
                    "image_count": 0,
                    "language_detected": "",
                    "parse_error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "raw_files": int(len(files)),
        "rows": int(len(df)),
        "scrape_success": int(df["scrape_success"].sum()) if len(df) else 0,
        "parse_success": int(df["parse_success"].sum()) if len(df) else 0,
        "rows_with_scraped_body": int(df["scraped_body_available"].sum()) if len(df) else 0,
        "rows_with_word_count": int((df["word_count"] > 0).sum()) if len(df) else 0,
    }
    return df, summary
