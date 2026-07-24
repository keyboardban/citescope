from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.econometrics_eda_v2.page_type_classifier import classify_page_type_family


BLOCKED_PAT = re.compile(r"human verification|please wait|checking your browser|access denied|just a moment|cloudflare|captcha|recaptcha|robot|forbidden|\b403\b|\b404\b", re.I)
PDF_PAT = re.compile(r"\.pdf(?:$|[?#])", re.I)


def _num(series: pd.Series, default: float = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().casefold() in {"1", "true", "yes", "y"}


def _text(row: pd.Series, col: str) -> str:
    value = row.get(col)
    return "" if pd.isna(value) else str(value)


def is_pdf_or_binary(row: pd.Series) -> bool:
    return bool(PDF_PAT.search(" ".join([_text(row, "source_url"), _text(row, "normalized_url")])))


def recommended_brightdata_mode(row: pd.Series) -> str:
    flag = _text(row, "content_quality_flag")
    haystack = " ".join([_text(row, "page_title"), _text(row, "page_text_excerpt")])
    if is_pdf_or_binary(row):
        return "pdf_parser_needed"
    if flag == "blocked_or_error_page" or BLOCKED_PAT.search(haystack):
        return "unlocker_api"
    if flag in {"dynamic_js_likely", "very_short_text", "boilerplate_only", "nav_footer_only", "parse_failed", "empty_text"}:
        return "browser_api"
    if not _bool(row.get("scraped_body_available")) or not _bool(row.get("parse_success")):
        return "browser_api"
    return "browser_api"


def select_brightdata_benchmark_urls(scrape_audit: pd.DataFrame, max_urls: int = 40) -> pd.DataFrame:
    work = scrape_audit.copy()
    work["word_count_num"] = _num(work.get("word_count", pd.Series([0] * len(work))), 0)
    work["text_char_count_num"] = _num(work.get("text_char_count", pd.Series([0] * len(work))), 0)
    work["cited_rows_num"] = _num(work.get("cited_rows_n", pd.Series([0] * len(work))), 0)
    work["source_rows_num"] = _num(work.get("source_rows_n", pd.Series([0] * len(work))), 0)
    work["page_type_family_derived"] = work.get("page_type_final", pd.Series(["unknown"] * len(work))).fillna("unknown").map(classify_page_type_family)
    work["title_excerpt"] = work.get("page_title", pd.Series([""] * len(work))).fillna("").astype(str) + " " + work.get("page_text_excerpt", pd.Series([""] * len(work))).fillna("").astype(str)
    non_pdf = work[~work.apply(is_pdf_or_binary, axis=1)].copy()
    picks: list[tuple[str, pd.Series]] = []
    seen: set[str] = set()

    def add(frame: pd.DataFrame, reason: str, limit: int) -> None:
        count = 0
        for _, row in frame.iterrows():
            nurl = str(row.get("normalized_url") or "")
            if not nurl or nurl in seen:
                continue
            picks.append((reason, row))
            seen.add(nurl)
            count += 1
            if count >= limit or len(picks) >= max_urls:
                break

    dyn = non_pdf[non_pdf["content_quality_flag"].fillna("").eq("dynamic_js_likely")]
    add(dyn.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "dynamic_js_likely", 10)
    parse = non_pdf[
        non_pdf["content_quality_flag"].fillna("").isin({"parse_failed", "empty_text"})
        | ~non_pdf.get("scraped_body_available", pd.Series([False] * len(non_pdf))).map(_bool)
        | ~non_pdf.get("parse_success", pd.Series([False] * len(non_pdf))).map(_bool)
    ]
    add(parse.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "parse_failed_empty_or_no_body", 10)
    short = non_pdf[non_pdf["content_quality_flag"].fillna("").isin({"very_short_text", "boilerplate_only", "nav_footer_only"}) | ((non_pdf["word_count_num"] > 0) & (non_pdf["word_count_num"] < 100))]
    add(short.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "very_short_or_boilerplate", 10)
    blocked = non_pdf[non_pdf["content_quality_flag"].fillna("").eq("blocked_or_error_page") | non_pdf["title_excerpt"].str.contains(BLOCKED_PAT, na=False)]
    add(blocked.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "blocked_or_captcha", 5)
    poor_flags = {"dynamic_js_likely", "parse_failed", "empty_text", "very_short_text", "boilerplate_only", "nav_footer_only", "blocked_or_error_page"}
    high = non_pdf[(non_pdf["cited_rows_num"] > 0) & non_pdf["content_quality_flag"].fillna("").isin(poor_flags)]
    add(high.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "high_impact_cited_poor_scrape", 5)
    if len(picks) < max_urls:
        fill = non_pdf[non_pdf["content_quality_flag"].fillna("").isin(poor_flags)]
        add(fill.sort_values(["cited_rows_num", "word_count_num"], ascending=[False, True]), "poor_scrape_fill", max_urls - len(picks))

    rows = []
    for i, (reason, row) in enumerate(picks[:max_urls], start=1):
        rows.append(
            {
                "benchmark_id": f"bd_bench_{i:03d}",
                "source_url": row.get("source_url"),
                "normalized_url": row.get("normalized_url"),
                "source_root_domain": row.get("source_root_domain"),
                "cited_rows_n": int(row.get("cited_rows_num", 0)),
                "source_rows_n": int(row.get("source_rows_num", 0)),
                "current_provider": "apify_cheerio",
                "current_scrape_success": row.get("scrape_success"),
                "current_parse_success": row.get("parse_success"),
                "current_word_count": int(row.get("word_count_num", 0)),
                "current_text_char_count": int(row.get("text_char_count_num", 0)),
                "current_content_quality_flag": row.get("content_quality_flag"),
                "current_page_type_final": row.get("page_type_final"),
                "current_page_type_family": row.get("page_type_family_derived"),
                "reason_selected": reason,
                "recommended_brightdata_mode": recommended_brightdata_mode(row),
            }
        )
    return pd.DataFrame(rows)


# Coarse families used only to spread a small smoke sample across page kinds.
# Order = the round-robin cycle; leading with an easy-win family gives a clear
# smoke signal, then the harder blocked/parse/commerce cases.
BENCHMARK_FAMILY_CYCLE = (
    "article_institutional",
    "reddit_or_blocked",
    "parse_failed_other",
    "ecommerce_product",
)
_COMMERCE_HINT_DOMAINS = {
    "alibaba.com", "bigc.co.th", "cetaphil.co.th", "central.co.th", "acer.com",
    "shopee.co.th", "lazada.co.th",
}
_COMMERCE_PAGE_TYPES = {"product_marketplace_page", "price_package_page", "third_party_platform_page"}
_ARTICLE_PAGE_TYPES = {"article_health_info", "news_announcement_page", "doctor_profile"}
_ARTICLE_FAMILIES = {"information_content", "news_or_update", "hospital_service_info"}
_INSTITUTIONAL_TLDS = (".gov", ".go.th", ".edu", ".org", ".org.uk", ".ac.th")


def benchmark_family(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    domain = str(get("source_root_domain") or "").lower()
    mode = str(get("recommended_brightdata_mode") or "")
    reason = str(get("reason_selected") or "")
    page_type = str(get("current_page_type_final") or "")
    family = str(get("current_page_type_family") or "")
    if domain == "reddit.com" or mode == "unlocker_api" or reason == "blocked_or_captcha":
        return "reddit_or_blocked"
    if domain in _COMMERCE_HINT_DOMAINS or page_type in _COMMERCE_PAGE_TYPES or family == "commercial_price_package":
        return "ecommerce_product"
    if family in _ARTICLE_FAMILIES or page_type in _ARTICLE_PAGE_TYPES or domain.endswith(_INSTITUTIONAL_TLDS):
        return "article_institutional"
    return "parse_failed_other"


def mix_benchmark_order(df: pd.DataFrame, cycle: tuple[str, ...] = BENCHMARK_FAMILY_CYCLE) -> pd.DataFrame:
    """Reorder benchmark rows by round-robin over coarse families.

    Preserves every original column and each row's original ``benchmark_id`` so
    the raw-cache mapping (and any already-successful cached scrape) stays valid;
    only row order changes. ``--max-urls N`` on the result yields a family-mixed
    smoke sample instead of a Reddit/blocked-heavy run.
    """
    if df.empty:
        out = df.copy()
        out["family"] = pd.Series(dtype=str)
        out["mixed_rank"] = pd.Series(dtype=int)
        return out
    work = df.copy().reset_index(drop=True)
    work["family"] = work.apply(benchmark_family, axis=1)
    buckets: dict[str, list[int]] = {fam: [] for fam in cycle}
    for idx, fam in work["family"].items():
        buckets.setdefault(fam, []).append(idx)
    order = [fam for fam in cycle] + [fam for fam in buckets if fam not in cycle]
    interleaved: list[int] = []
    while any(buckets[fam] for fam in order):
        for fam in order:
            if buckets[fam]:
                interleaved.append(buckets[fam].pop(0))
    out = work.loc[interleaved].reset_index(drop=True)
    out["mixed_rank"] = range(1, len(out) + 1)
    return out
