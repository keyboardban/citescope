#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import OUTPUT_DIR, ensure_v2_dirs, read_csv, write_csv
from src.econometrics_eda_v2.page_type_classifier import classify_page_type_details
from src.source_type import classify as classify_source_type


UNKNOWN = "unknown"
BAD_QUALITY_FLAGS = {
    "",
    "parse_failed",
    "empty_text",
    "very_short_text",
    "boilerplate_only",
    "nav_footer_only",
    "dynamic_js_likely",
    "blocked_or_error_page",
}
BAD_PARSE_CATEGORIES = {
    "blocked_or_verification_page",
    "metadata_only_response",
    "raw_response_missing_from_cache",
}


def _read_optional(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return read_csv(p)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = 0) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _status_2xx(value: Any) -> bool:
    n = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return bool(pd.notna(n) and 200 <= int(n) < 300)


def _quality_score(flag: str) -> int:
    return {
        "ok": 5,
        "nav_footer_only": 2,
        "boilerplate_only": 2,
        "very_short_text": 2,
        "dynamic_js_likely": 1,
        "blocked_or_error_page": 1,
        "parse_failed": 0,
        "empty_text": 0,
    }.get(str(flag or "").strip(), 0)


def _usable(flag: Any, word_count: Any, parse_category: Any = "") -> bool:
    flag_s = str(flag or "").strip()
    cat = str(parse_category or "").strip()
    return bool(flag_s == "ok" and _num(word_count) >= 300 and cat not in BAD_PARSE_CATEGORIES)


def _winner_from_bools(ap: bool | None, bd: bool | None) -> str:
    if ap is None or bd is None:
        return UNKNOWN
    if ap and bd:
        return "tie"
    if ap:
        return "apify"
    if bd:
        return "brightdata"
    return "neither"


def _word_count_winner(row: pd.Series) -> str:
    ap_usable = _usable(row.get("apify_content_quality_flag"), row.get("apify_word_count"))
    bd_usable = _usable(row.get("brightdata_content_quality_flag"), row.get("brightdata_word_count"), row.get("brightdata_parse_error_category"))
    if not ap_usable and not bd_usable:
        return "neither"
    if ap_usable and not bd_usable:
        return "apify"
    if bd_usable and not ap_usable:
        return "brightdata"
    ap_wc = _num(row.get("apify_word_count"))
    bd_wc = _num(row.get("brightdata_word_count"))
    ap_chars = _num(row.get("apify_text_char_count"))
    bd_chars = _num(row.get("brightdata_text_char_count"))
    if bd_wc >= ap_wc * 1.5 and bd_chars >= ap_chars * 1.2:
        return "brightdata"
    if ap_wc >= bd_wc * 1.5 and ap_chars >= bd_chars * 1.2:
        return "apify"
    return "tie"


def _page_type_winner(row: pd.Series) -> str:
    ap_ok = str(row.get("apify_page_type_final") or "") not in {"", UNKNOWN}
    bd_ok = str(row.get("brightdata_page_type_final") or "") not in {"", UNKNOWN}
    ap_usable = _usable(row.get("apify_content_quality_flag"), row.get("apify_word_count"))
    bd_usable = _usable(row.get("brightdata_content_quality_flag"), row.get("brightdata_word_count"), row.get("brightdata_parse_error_category"))
    if ap_ok and ap_usable and not bd_ok:
        return "apify"
    if bd_ok and bd_usable and not ap_ok:
        return "brightdata"
    if ap_ok and bd_ok:
        return "tie"
    return "neither"


def _request_success(provider: str, row: pd.Series) -> bool | None:
    if provider == "apify":
        raw = _bool(row.get("apify_raw_cache_exists"))
        req = _bool(row.get("apify_request_success"))
        status = row.get("apify_status_code")
    else:
        raw = _bool(row.get("brightdata_raw_cache_exists"))
        req = _bool(row.get("brightdata_request_success"))
        status = row.get("brightdata_status_code")
        cat = str(row.get("brightdata_parse_error_category") or "")
        err = str(row.get("brightdata_error") or "").casefold()
        if cat == "raw_response_missing_from_cache":
            return None
        if "request validation failed" in err:
            return False
    return bool(raw and (req or _status_2xx(status)))


def _comparison(row: pd.Series) -> dict[str, str]:
    ap_req = _request_success("apify", row)
    bd_req = _request_success("brightdata", row)
    request_winner = _winner_from_bools(ap_req, bd_req)
    ap_body = _bool(row.get("apify_scraped_body_available"))
    bd_body = _bool(row.get("brightdata_scraped_body_available"))
    body_winner = _winner_from_bools(ap_body, bd_body)
    if body_winner == "tie":
        wc_winner = _word_count_winner(row)
        if wc_winner in {"apify", "brightdata"}:
            body_winner = wc_winner
    ap_usable = _usable(row.get("apify_content_quality_flag"), row.get("apify_word_count"))
    bd_usable = _usable(row.get("brightdata_content_quality_flag"), row.get("brightdata_word_count"), row.get("brightdata_parse_error_category"))
    if ap_usable and bd_usable:
        content_winner = _word_count_winner(row)
        if content_winner == "neither":
            content_winner = "tie"
    else:
        content_winner = _winner_from_bools(ap_usable, bd_usable)
    word_winner = _word_count_winner(row)
    page_type_winner = _page_type_winner(row)

    bd_cat = str(row.get("brightdata_parse_error_category") or "")
    bd_err = str(row.get("brightdata_error") or "")
    domain = str(row.get("source_root_domain") or "")
    ap_flag = str(row.get("apify_content_quality_flag") or "")
    bd_flag = str(row.get("brightdata_content_quality_flag") or "")
    if bd_cat == "raw_response_missing_from_cache":
        overall = "unknown"
        rec = "retry_brightdata_after_payload_fix"
        reason = "Bright Data old cache is missing raw_response; rerun live after cache fix before judging provider quality."
    elif "request validation failed" in bd_err.casefold():
        overall = "unknown" if not ap_usable else "apify"
        rec = "retry_brightdata_after_payload_fix"
        reason = "Bright Data request validation failed; fix payload/cache path before treating it as a scrape-quality loss."
    elif content_winner in {"apify", "brightdata"}:
        overall = content_winner
        rec = "use_apify" if overall == "apify" else "use_brightdata"
        reason = f"{overall} has usable content quality while the other provider does not."
    elif content_winner == "tie":
        overall = "tie"
        rec = "use_apify_primary_brightdata_fallback"
        reason = "Both providers have usable content; keep Apify primary unless Bright Data shows domain-level advantage."
    elif body_winner in {"apify", "brightdata"}:
        overall = "neither"
        if body_winner == "apify" and ap_flag in {"dynamic_js_likely", "very_short_text", "blocked_or_error_page", "parse_failed"}:
            rec = "retry_with_playwright"
        elif body_winner == "brightdata" and bd_cat in {"body_too_short", "no_body_field_detected", "metadata_only_response"}:
            rec = "skip_or_manual_review"
        else:
            rec = "skip_or_manual_review"
        reason = f"{body_winner} has body/request evidence, but not usable content quality; not awarding an overall provider win."
    elif request_winner in {"apify", "brightdata"}:
        overall = "neither"
        rec = "retry_with_playwright" if request_winner == "apify" and ap_flag in {"dynamic_js_likely", "very_short_text", "blocked_or_error_page", "parse_failed"} else "skip_or_manual_review"
        reason = f"{request_winner} has request-level success, but usable body/content is not established."
    elif "reddit" in domain or ap_flag in {"dynamic_js_likely", "blocked_or_error_page"} or bd_flag in {"blocked_or_error_page", "dynamic_js_likely"}:
        overall = "neither"
        rec = "use_serper_metadata_fallback"
        reason = "Both provider paths look blocked, verification-heavy, or metadata-only; preserve metadata or manual review."
    else:
        overall = "neither"
        rec = "skip_or_manual_review"
        reason = "Neither provider has usable request/body/content evidence in the current cache."
    return {
        "request_winner": request_winner,
        "body_availability_winner": body_winner,
        "word_count_winner": word_winner,
        "content_quality_winner": content_winner,
        "page_type_winner": page_type_winner,
        "overall_provider_winner": overall,
        "overall_provider_recommendation": rec,
        "comparison_reason": reason,
    }


def _brightdata_page_types(bright: pd.DataFrame) -> pd.DataFrame:
    if bright.empty:
        return pd.DataFrame(columns=["benchmark_id", "brightdata_page_type_final", "brightdata_page_type_family"])
    rows = []
    for _, row in bright.iterrows():
        if not _text(row.get("page_text")).strip() or str(row.get("parse_error_category") or "") == "raw_response_missing_from_cache":
            rows.append({"benchmark_id": row.get("benchmark_id"), "brightdata_page_type_final": UNKNOWN, "brightdata_page_type_family": UNKNOWN})
            continue
        source_type, _ = classify_source_type(str(row.get("source_url") or row.get("final_url") or ""))
        detail = classify_page_type_details(
            {
                "final_url": row.get("final_url"),
                "requested_url": row.get("requested_url"),
                "page_title": row.get("page_title"),
                "meta_description": row.get("meta_description"),
                "page_text": row.get("page_text"),
                "table_count": row.get("table_count"),
                "source_type_url": source_type,
            },
            source_type,
        )
        rows.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "brightdata_page_type_final": detail.page_type,
                "brightdata_page_type_family": detail.family,
            }
        )
    return pd.DataFrame(rows)


def build_head_to_head(
    benchmark: pd.DataFrame,
    apify: pd.DataFrame,
    bright_summary: pd.DataFrame,
    bright_parse: pd.DataFrame,
    bright_cache: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = [
        "benchmark_id", "source_url", "normalized_url", "source_root_domain", "cited_rows_n",
        "source_rows_n", "reason_selected", "recommended_brightdata_mode",
    ]
    out = benchmark[[c for c in base_cols if c in benchmark.columns]].drop_duplicates("benchmark_id").copy()

    ap = apify.rename(
        columns={
            "in_scrape_queue": "apify_in_queue",
            "raw_cache_exists": "apify_raw_cache_exists",
            "scrape_success": "apify_request_success",
            "parse_success": "apify_parse_success",
            "scraped_body_available": "apify_scraped_body_available",
            "content_quality_flag": "apify_content_quality_flag",
            "word_count": "apify_word_count",
            "text_char_count": "apify_text_char_count",
            "heading_count": "apify_heading_count",
            "table_count": "apify_table_count",
            "page_title": "apify_title",
            "page_text_excerpt": "apify_excerpt",
            "page_type_final": "apify_page_type_final",
        }
    )
    ap_cols = [
        "normalized_url", "apify_in_queue", "apify_raw_cache_exists", "apify_request_success",
        "apify_parse_success", "apify_scraped_body_available", "apify_content_quality_flag",
        "apify_word_count", "apify_text_char_count", "apify_heading_count", "apify_table_count",
        "apify_title", "apify_excerpt", "apify_page_type_final",
    ]
    if "page_type_final" in apify.columns:
        # scrape_quality_audit lacks page_type_family; benchmark input carries the current family.
        pass
    out = out.merge(ap[[c for c in ap_cols if c in ap.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")
    out["apify_provider_mode"] = benchmark.get("current_provider", "apify").values if "current_provider" in benchmark.columns else "apify"
    out["apify_status_code"] = ""
    out["apify_error"] = ""
    out["apify_link_count"] = ""
    out["apify_page_type_family"] = benchmark.get("current_page_type_family", UNKNOWN).values if "current_page_type_family" in benchmark.columns else UNKNOWN

    bs = bright_summary.rename(
        columns={
            "provider_mode": "brightdata_provider_mode_summary",
            "success": "brightdata_request_success_summary",
            "status_code": "brightdata_status_code_summary",
            "error": "brightdata_error_summary",
        }
    )
    bs_cols = [
        "benchmark_id", "brightdata_provider_mode_summary", "attempted", "brightdata_request_success_summary",
        "brightdata_status_code_summary", "brightdata_error_summary", "raw_cache_path",
    ]
    out = out.merge(bs[[c for c in bs_cols if c in bs.columns]].drop_duplicates("benchmark_id"), on="benchmark_id", how="left")
    bp = bright_parse.rename(
        columns={
            "provider_mode": "brightdata_provider_mode_parse",
            "scrape_success": "brightdata_request_success_parse",
            "status_code": "brightdata_status_code_parse",
            "parse_success": "brightdata_parse_success",
            "scraped_body_available": "brightdata_scraped_body_available",
            "parse_error": "brightdata_error_parse",
            "parse_error_category": "brightdata_parse_error_category",
            "content_quality_flag": "brightdata_content_quality_flag",
            "word_count": "brightdata_word_count",
            "text_char_count": "brightdata_text_char_count",
            "heading_count": "brightdata_heading_count",
            "table_count": "brightdata_table_count",
            "link_count": "brightdata_link_count",
            "page_title": "brightdata_title",
            "page_text_excerpt": "brightdata_excerpt",
            "body_field_selected": "brightdata_body_field_selected",
        }
    )
    bp_cols = [
        "benchmark_id", "brightdata_provider_mode_parse", "brightdata_request_success_parse",
        "brightdata_status_code_parse", "brightdata_parse_success", "brightdata_scraped_body_available",
        "brightdata_error_parse", "brightdata_parse_error_category", "brightdata_content_quality_flag",
        "brightdata_word_count", "brightdata_text_char_count", "brightdata_heading_count",
        "brightdata_table_count", "brightdata_link_count", "brightdata_title", "brightdata_excerpt",
        "brightdata_body_field_selected",
    ]
    out = out.merge(bp[[c for c in bp_cols if c in bp.columns]].drop_duplicates("benchmark_id"), on="benchmark_id", how="left")
    out = out.merge(_brightdata_page_types(bright_parse), on="benchmark_id", how="left")

    bc = bright_cache.rename(
        columns={
            "has_raw_response": "brightdata_has_raw_response",
            "can_reparse": "brightdata_can_reparse",
            "cache_status": "brightdata_cache_status",
            "raw_cache_path": "brightdata_raw_cache_path",
        }
    )
    bc_cols = ["benchmark_id", "brightdata_raw_cache_path", "brightdata_has_raw_response", "brightdata_can_reparse", "brightdata_cache_status"]
    out = out.merge(bc[[c for c in bc_cols if c in bc.columns]].drop_duplicates("benchmark_id"), on="benchmark_id", how="left")

    out["brightdata_provider_mode"] = out["brightdata_provider_mode_summary"].fillna(out.get("brightdata_provider_mode_parse")).fillna(out.get("recommended_brightdata_mode")).fillna(UNKNOWN)
    out["brightdata_raw_cache_exists"] = out.get("brightdata_raw_cache_path", pd.Series([""] * len(out))).fillna("").astype(str).map(lambda p: bool(p.strip()) and Path(p).exists())
    out["brightdata_has_raw_response"] = out.get("brightdata_has_raw_response", pd.Series([pd.NA] * len(out))).map(_bool)
    out["brightdata_request_success"] = out.get("brightdata_request_success_parse", pd.Series([pd.NA] * len(out))).combine_first(out.get("brightdata_request_success_summary", pd.Series([pd.NA] * len(out))))
    out["brightdata_status_code"] = out.get("brightdata_status_code_parse", pd.Series([pd.NA] * len(out))).combine_first(out.get("brightdata_status_code_summary", pd.Series([pd.NA] * len(out))))
    out["brightdata_error"] = out.get("brightdata_error_parse", pd.Series([""] * len(out))).fillna("").where(lambda s: s.astype(str).str.len() > 0, out.get("brightdata_error_summary", pd.Series([""] * len(out))).fillna(""))
    out["brightdata_request_success"] = out.apply(
        lambda r: bool(
            _bool(r.get("brightdata_raw_cache_exists"))
            and (
                _status_2xx(r.get("brightdata_status_code"))
                or _bool(r.get("brightdata_request_success"))
            )
            and "request validation failed" not in str(r.get("brightdata_error") or "").casefold()
            and str(r.get("brightdata_parse_error_category") or "") != "raw_response_missing_from_cache"
        ),
        axis=1,
    )
    out["apify_page_type_family"] = out["apify_page_type_family"].fillna(UNKNOWN)
    out["brightdata_page_type_final"] = out["brightdata_page_type_final"].fillna(UNKNOWN)
    out["brightdata_page_type_family"] = out["brightdata_page_type_family"].fillna(UNKNOWN)

    defaults = {
        "apify_in_queue": False,
        "apify_raw_cache_exists": False,
        "apify_request_success": False,
        "apify_parse_success": False,
        "apify_scraped_body_available": False,
        "apify_content_quality_flag": "",
        "apify_word_count": 0,
        "apify_text_char_count": 0,
        "apify_heading_count": 0,
        "apify_table_count": 0,
        "brightdata_parse_success": False,
        "brightdata_scraped_body_available": False,
        "brightdata_parse_error_category": "",
        "brightdata_content_quality_flag": "",
        "brightdata_word_count": 0,
        "brightdata_text_char_count": 0,
        "brightdata_heading_count": 0,
        "brightdata_table_count": 0,
        "brightdata_link_count": 0,
        "brightdata_title": "",
        "brightdata_excerpt": "",
        "brightdata_body_field_selected": "",
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
        out[col] = out[col].fillna(default)

    comparisons = out.apply(_comparison, axis=1, result_type="expand")
    out = pd.concat([out, comparisons], axis=1)
    columns = [
        "benchmark_id", "source_url", "normalized_url", "source_root_domain", "cited_rows_n",
        "source_rows_n", "reason_selected",
        "apify_provider_mode", "apify_in_queue", "apify_raw_cache_exists", "apify_request_success",
        "apify_status_code", "apify_parse_success", "apify_scraped_body_available", "apify_error",
        "apify_content_quality_flag", "apify_word_count", "apify_text_char_count", "apify_heading_count",
        "apify_table_count", "apify_link_count", "apify_title", "apify_excerpt", "apify_page_type_final",
        "apify_page_type_family",
        "brightdata_provider_mode", "brightdata_raw_cache_exists", "brightdata_has_raw_response",
        "brightdata_request_success", "brightdata_status_code", "brightdata_parse_success",
        "brightdata_scraped_body_available", "brightdata_error", "brightdata_parse_error_category",
        "brightdata_content_quality_flag", "brightdata_word_count", "brightdata_text_char_count",
        "brightdata_heading_count", "brightdata_table_count", "brightdata_link_count", "brightdata_title",
        "brightdata_excerpt", "brightdata_body_field_selected", "brightdata_page_type_final",
        "brightdata_page_type_family",
        "request_winner", "body_availability_winner", "word_count_winner", "content_quality_winner",
        "page_type_winner", "overall_provider_winner", "overall_provider_recommendation",
        "comparison_reason",
    ]
    return out[[c for c in columns if c in out.columns]]


def build_summary(head: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("total_urls_compared", len(head)),
        ("apify_request_success_n", int(head["apify_request_success"].map(_bool).sum())),
        ("brightdata_request_success_n", int(head["brightdata_request_success"].map(_bool).sum())),
        ("apify_body_available_n", int(head["apify_scraped_body_available"].map(_bool).sum())),
        ("brightdata_body_available_n", int(head["brightdata_scraped_body_available"].map(_bool).sum())),
        ("apify_usable_content_n", int(head.apply(lambda r: _usable(r.get("apify_content_quality_flag"), r.get("apify_word_count")), axis=1).sum())),
        ("brightdata_usable_content_n", int(head.apply(lambda r: _usable(r.get("brightdata_content_quality_flag"), r.get("brightdata_word_count"), r.get("brightdata_parse_error_category")), axis=1).sum())),
        ("apify_overall_wins_n", int(head["overall_provider_winner"].eq("apify").sum())),
        ("brightdata_overall_wins_n", int(head["overall_provider_winner"].eq("brightdata").sum())),
        ("ties_n", int(head["overall_provider_winner"].eq("tie").sum())),
        ("neither_n", int(head["overall_provider_winner"].eq("neither").sum())),
        ("unknown_n", int(head["overall_provider_winner"].eq("unknown").sum())),
        ("brightdata_request_validation_failed_n", int(head["brightdata_error"].fillna("").str.contains("Request validation failed", case=False).sum())),
        ("brightdata_no_body_like_field_n", int(head["brightdata_error"].fillna("").str.contains("No body-like field found", case=False).sum())),
        ("brightdata_raw_response_missing_n", int(head["brightdata_parse_error_category"].eq("raw_response_missing_from_cache").sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_domain(head: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, group in head.groupby("source_root_domain", dropna=False):
        ap_wins = int(group["overall_provider_winner"].eq("apify").sum())
        bd_wins = int(group["overall_provider_winner"].eq("brightdata").sum())
        ties = int(group["overall_provider_winner"].eq("tie").sum())
        neither = int(group["overall_provider_winner"].eq("neither").sum())
        unknown = int(group["overall_provider_winner"].eq("unknown").sum())
        ap_rate = float(group.apply(lambda r: _usable(r.get("apify_content_quality_flag"), r.get("apify_word_count")), axis=1).mean())
        bd_rate = float(group.apply(lambda r: _usable(r.get("brightdata_content_quality_flag"), r.get("brightdata_word_count"), r.get("brightdata_parse_error_category")), axis=1).mean())
        if unknown == len(group) and group["brightdata_parse_error_category"].eq("raw_response_missing_from_cache").any():
            rec = "retry_brightdata_after_payload_fix"
            reason = "Bright Data cache missing raw_response; cannot judge provider for this domain yet."
        elif ap_wins > bd_wins:
            rec = "apify"
            reason = "Apify has more URL-level usable-content wins."
        elif bd_wins > ap_wins:
            rec = "brightdata"
            reason = "Bright Data has more URL-level usable-content wins."
        elif neither == len(group):
            rec = "metadata_or_manual_review"
            reason = "Neither provider produced usable cached content."
        else:
            rec = "apify_primary_with_fallback"
            reason = "No clear provider advantage in current cached benchmark."
        rows.append(
            {
                "source_root_domain": domain,
                "urls_compared": int(len(group)),
                "apify_wins_n": ap_wins,
                "brightdata_wins_n": bd_wins,
                "ties_n": ties,
                "neither_n": neither,
                "unknown_n": unknown,
                "apify_usable_content_rate": ap_rate,
                "brightdata_usable_content_rate": bd_rate,
                "recommended_provider_for_domain": rec,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["urls_compared", "apify_wins_n", "brightdata_wins_n"], ascending=[False, False, False])


def _examples(frame: pd.DataFrame, mask: pd.Series, col: str, limit: int = 5) -> str:
    vals = frame.loc[mask, col].dropna().astype(str)
    return " | ".join(vals.drop_duplicates().head(limit))


def build_rules(head: pd.DataFrame) -> pd.DataFrame:
    conditions = [
        ("static_ok", head["apify_content_quality_flag"].eq("ok") & (pd.to_numeric(head["apify_word_count"], errors="coerce").fillna(0) >= 300), "apify", "Apify already has usable content; no provider switch needed."),
        ("dynamic_js_likely", head["apify_content_quality_flag"].isin(["dynamic_js_likely", "very_short_text"]), "apify_playwright_fallback", "Use rendered crawling before judging provider replacement."),
        ("blocked_or_verification", head["apify_content_quality_flag"].eq("blocked_or_error_page") | head["brightdata_content_quality_flag"].eq("blocked_or_error_page"), "metadata_or_manual_review", "Blocked or verification pages need rendered/fallback/manual handling."),
        ("request_validation_failed", head["brightdata_error"].fillna("").str.contains("Request validation failed", case=False), "brightdata_payload_fix_then_retry", "Fix Bright Data request payload before interpreting result."),
        ("reddit_or_social_verification", head["source_root_domain"].fillna("").str.contains("reddit|facebook|instagram|tiktok|x.com|twitter", case=False), "serper_metadata_or_manual_review", "Social/UGC pages are frequently verification-heavy."),
        ("apify_ok_brightdata_failed", head["overall_provider_winner"].eq("apify") & ~head["brightdata_parse_success"].map(_bool), "apify", "Apify has usable content while Bright Data lacks parsed usable body."),
        ("brightdata_ok_apify_failed", head["overall_provider_winner"].eq("brightdata"), "brightdata", "Bright Data has usable content where Apify does not."),
        ("both_failed_metadata_only", head["overall_provider_winner"].isin(["neither", "unknown"]) & head["brightdata_parse_error_category"].eq("raw_response_missing_from_cache"), "retry_brightdata_after_payload_fix", "Bright Data old cache is unrecoverable; rerun live with raw preservation."),
    ]
    rows = []
    for condition, mask, provider, reason in conditions:
        rows.append(
            {
                "condition": condition,
                "recommended_provider": provider,
                "reason": reason,
                "example_domains": _examples(head, mask, "source_root_domain"),
                "example_urls": _examples(head, mask, "source_url", 3),
            }
        )
    return pd.DataFrame(rows)


def build_review_sample(head: pd.DataFrame) -> pd.DataFrame:
    work = head.copy()
    work["_failed_first"] = work["overall_provider_winner"].isin(["neither", "unknown"]).astype(int)
    work["_disagree"] = (
        work["content_quality_winner"].isin(["apify", "brightdata"])
        | work["body_availability_winner"].isin(["apify", "brightdata"])
        | work["request_winner"].isin(["apify", "brightdata"])
    ).astype(int)
    work = work.sort_values(["cited_rows_n", "source_rows_n", "_failed_first", "_disagree"], ascending=[False, False, False, False])
    cols = [
        "source_url", "source_root_domain", "cited_rows_n", "apify_content_quality_flag",
        "brightdata_content_quality_flag", "apify_word_count", "brightdata_word_count",
        "apify_excerpt", "brightdata_excerpt", "overall_provider_winner",
        "overall_provider_recommendation", "comparison_reason",
    ]
    return work[[c for c in cols if c in work.columns]].head(100)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apify-audit", default=str(OUTPUT_DIR / "tables" / "scrape_quality_audit.csv"))
    ap.add_argument("--brightdata-summary", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_scrape_summary.csv"))
    ap.add_argument("--brightdata-parse", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_parse_rows.csv"))
    ap.add_argument("--brightdata-cache-audit", default=str(OUTPUT_DIR / "tables" / "brightdata_cache_integrity_audit.csv"))
    ap.add_argument("--benchmark-input", default=str(OUTPUT_DIR / "tables" / "brightdata_benchmark_input_urls.csv"))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR / "tables"))
    ap.add_argument("--output-suffix", default="", help="Optional suffix before .csv, e.g. _v2.")
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.benchmark_input).exists():
        raise FileNotFoundError(f"Benchmark input not found: {args.benchmark_input}")
    benchmark = read_csv(args.benchmark_input)
    apify_audit = _read_optional(args.apify_audit)
    bright_summary = _read_optional(args.brightdata_summary)
    bright_parse = _read_optional(args.brightdata_parse)
    bright_cache = _read_optional(args.brightdata_cache_audit)
    head = build_head_to_head(benchmark, apify_audit, bright_summary, bright_parse, bright_cache)
    summary = build_summary(head)
    domain = build_domain(head)
    rules = build_rules(head)
    sample = build_review_sample(head)
    suffix = args.output_suffix
    files = {
        "benchmark": out_dir / f"provider_head_to_head_benchmark{suffix}.csv",
        "summary": out_dir / f"provider_head_to_head_summary{suffix}.csv",
        "domain": out_dir / f"provider_domain_level_benchmark{suffix}.csv",
        "rules": out_dir / f"provider_recommended_fallback_rules{suffix}.csv",
        "sample": out_dir / f"provider_head_to_head_review_sample{suffix}.csv",
    }
    write_csv(files["benchmark"], head)
    write_csv(files["summary"], summary)
    write_csv(files["domain"], domain)
    write_csv(files["rules"], rules)
    write_csv(files["sample"], sample)
    metrics = summary.set_index("metric")["value"].to_dict()
    print("Provider head-to-head benchmark complete")
    print(f"Total URLs compared: {metrics.get('total_urls_compared', 0)}")
    print(f"Apify request success: {metrics.get('apify_request_success_n', 0)}")
    print(f"Bright Data request success: {metrics.get('brightdata_request_success_n', 0)}")
    print(f"Apify usable content: {metrics.get('apify_usable_content_n', 0)}")
    print(f"Bright Data usable content: {metrics.get('brightdata_usable_content_n', 0)}")
    print(f"Apify overall wins: {metrics.get('apify_overall_wins_n', 0)}")
    print(f"Bright Data overall wins: {metrics.get('brightdata_overall_wins_n', 0)}")
    print(f"Ties: {metrics.get('ties_n', 0)}")
    print(f"Neither: {metrics.get('neither_n', 0)}")
    print("Files generated:")
    for path in files.values():
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
