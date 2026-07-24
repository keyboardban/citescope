from __future__ import annotations

import re
from html import unescape
from typing import Any

from src.url_utils import normalize_url


QUALITY_SCORE = {
    "no_raw_cache": 0,
    "parse_failed": 1,
    "empty_text": 1,
    "blocked_or_error_page": 2,
    "dynamic_js_likely": 2,
    "very_short_text": 3,
    "boilerplate_only": 3,
    "nav_footer_only": 3,
    "ok": 5,
}


def clean_html_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html or "")
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def count_words(text: str) -> int:
    return len(re.findall(r"[\w\u0E00-\u0E7F]+", text or "", flags=re.UNICODE))


def count_heading_table_link_image(html: str = "", markdown: str = "") -> dict[str, int]:
    return {
        "heading_count": int(len(re.findall(r"(?m)^#{1,6}\s+", markdown or "")) + len(re.findall(r"(?is)<h[1-6][^>]*>", html or ""))),
        "table_count": int(len(re.findall(r"(?is)<table\b", html or "")) + len(re.findall(r"(?m)^\s*\|.+\|\s*$", markdown or "")) // 2),
        "link_count": int(len(re.findall(r"(?is)<a\b", html or "")) + len(re.findall(r"\[[^\]]+\]\([^)]+\)", markdown or ""))),
        "image_count": int(len(re.findall(r"(?is)<img\b", html or "")) + len(re.findall(r"!\[[^\]]*\]\([^)]+\)", markdown or ""))),
    }


def infer_content_quality_flag(
    *,
    success: bool,
    error: str = "",
    title: str = "",
    text: str = "",
    word_count: int | None = None,
    text_char_count: int | None = None,
) -> str:
    if not success and not text:
        return "parse_failed"
    low = " ".join([title or "", (text or "")[:800], error or ""]).casefold()
    wc = count_words(text) if word_count is None else int(word_count)
    chars = len(text or "") if text_char_count is None else int(text_char_count)
    if chars == 0 or not str(text or "").strip():
        return "empty_text"
    if re.search(r"\b(?:403|404)\b|not found|access denied|forbidden|nginx|error page", low):
        return "blocked_or_error_page"
    if re.search(r"please wait|verification|captcha|enable javascript|just a moment|checking your browser|cloudflare", low):
        return "dynamic_js_likely"
    if wc < 20:
        return "very_short_text"
    nav_terms = len(re.findall(r"\b(menu|home|privacy|terms|cookie|login|subscribe|follow us|copyright)\b|หน้าแรก|เมนู|เข้าสู่ระบบ", low))
    content_terms = len(re.findall(r"health|hospital|doctor|service|treatment|product|article|โรค|แพทย์|รักษา|บริการ|สุขภาพ|ผลิตภัณฑ์", low))
    if wc < 80 and nav_terms >= 3 and content_terms == 0:
        return "nav_footer_only"
    if wc < 120 and re.search(r"cookie|privacy|terms|copyright|all rights reserved", low) and content_terms == 0:
        return "boilerplate_only"
    return "ok"


def quality_score(flag: Any) -> int:
    return QUALITY_SCORE.get(str(flag or "").strip(), 0)


def body_text_from_parts(html: str = "", markdown: str = "", text: str = "") -> str:
    cleaned = (text or "").strip()
    if cleaned:
        return cleaned
    if markdown and markdown.strip():
        return re.sub(r"(?m)^#+\s*", "", markdown).strip()
    return clean_html_text(html)


def normalized_url_for(final_url: str, requested_url: str) -> str:
    return normalize_url(final_url or requested_url or "")
