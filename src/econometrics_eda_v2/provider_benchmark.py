from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.econometrics_eda_v2.brightdata_response_parser import parse_brightdata_raw_cache
from src.econometrics_eda_v2.io import read_csv, write_csv
from src.econometrics_eda_v2.page_type_classifier import classify_page_type_details
from src.econometrics_eda_v2.scrape_quality_audit import _excerpt, content_quality_flag
from src.source_type import classify as classify_source_type
from src.url_utils import normalize_url

QUALITY_SCORE = {
    "no_raw_cache": 0,
    "parse_failed": 1,
    "blocked_or_error_page": 2,
    "dynamic_js_likely": 2,
    "empty_text": 2,
    "very_short_text": 3,
    "boilerplate_only": 3,
    "nav_footer_only": 3,
    "ok": 5,
}


def _num(series: pd.Series, default: float = 0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().casefold() in {"1", "true", "yes", "y"}


def _col(df: pd.DataFrame, name: str, default: Any = "") -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _one_per_url(final_rows: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    rows = final_rows.copy()
    rows["_cited_num"] = _num(rows.get("cited", pd.Series([0] * len(rows))), 0)
    rows["_source_position_num"] = _num(rows.get("source_position", pd.Series([999] * len(rows))), 999)
    rows = rows.sort_values(["_cited_num", "_source_position_num"], ascending=[False, True])
    one = rows.drop_duplicates("normalized_url").copy()
    keep_quality = [
        "normalized_url",
        "scrape_success",
        "parse_success",
        "scraped_body_available",
        "word_count",
        "heading_count",
        "table_count",
        "content_quality_flag",
        "page_text_excerpt",
    ]
    q = quality[[c for c in keep_quality if c in quality.columns]].drop_duplicates("normalized_url")
    return one.merge(q, on="normalized_url", how="left", suffixes=("", "_quality"))


def select_provider_benchmark_urls(final_rows: pd.DataFrame, quality: pd.DataFrame, total: int = 40) -> pd.DataFrame:
    work = _one_per_url(final_rows, quality)
    work["current_word_count"] = _num(work.get("word_count_quality", work.get("word_count", pd.Series([0] * len(work)))), 0)
    work["current_content_quality_flag"] = work.get("content_quality_flag", pd.Series([""] * len(work))).fillna("")
    work["_cited_num"] = _num(work.get("cited", pd.Series([0] * len(work))), 0)
    work["_source_position_num"] = _num(work.get("source_position", pd.Series([999] * len(work))), 999)
    picks: list[tuple[str, pd.Series]] = []
    seen: set[str] = set()

    def add(frame: pd.DataFrame, reason: str, limit: int) -> None:
        nonlocal picks
        for _, row in frame.iterrows():
            nurl = str(row.get("normalized_url") or "")
            if not nurl or nurl in seen:
                continue
            seen.add(nurl)
            picks.append((reason, row))
            if sum(1 for r, _ in picks if r == reason) >= limit:
                break

    unknown = work[_col(work, "page_type_final").fillna("").astype(str).eq("unknown")]
    add(unknown[unknown["current_word_count"] < 100].sort_values(["_cited_num", "current_word_count"], ascending=[False, True]), "unknown_low_word_count", 10)
    add(unknown[unknown["current_content_quality_flag"].eq("ok")].sort_values(["_cited_num", "current_word_count"], ascending=[False, False]), "unknown_quality_ok", 10)
    bad_flags = {"parse_failed", "no_scraped_body", "dynamic_js_likely", "very_short_text", "boilerplate_only", "blocked_or_error_page"}
    bad = work[work["current_content_quality_flag"].isin(bad_flags) | ~_col(work, "scraped_body_available", False).map(_bool)]
    add(bad.sort_values(["_cited_num", "current_word_count"], ascending=[False, True]), "parse_failed_no_body_or_dynamic", 10)
    commercial = work[
        _col(work, "page_type_final").fillna("").astype(str).isin({"product_marketplace_page", "price_package_page"})
        | _col(work, "page_type_family").fillna("").astype(str).eq("commercial_price_package")
    ]
    add(commercial.sort_values(["_cited_num", "_source_position_num"], ascending=[False, True]), "ecommerce_marketplace_product", 5)
    cited = work[work["_cited_num"].eq(1)]
    add(cited.sort_values("_source_position_num"), "high_impact_cited", 5)
    if len(picks) < total:
        add(work.sort_values(["_cited_num", "_source_position_num"], ascending=[False, True]), "benchmark_fill", total - len(picks))

    out_rows = []
    for idx, (reason, row) in enumerate(picks[:total], start=1):
        out_rows.append(
            {
                "benchmark_id": f"bd_bench_{idx:03d}",
                "source_url": row.get("source_url"),
                "normalized_url": row.get("normalized_url"),
                "source_root_domain": row.get("source_root_domain") or row.get("source_domain_host"),
                "cited": int(row.get("_cited_num", 0)),
                "current_scrape_success": row.get("scrape_success"),
                "current_parse_success": row.get("parse_success"),
                "current_word_count": int(row.get("current_word_count", 0)),
                "current_content_quality_flag": row.get("current_content_quality_flag"),
                "current_page_type_final": row.get("page_type_final"),
                "current_page_type_family": row.get("page_type_family"),
                "reason_selected": reason,
            }
        )
    return pd.DataFrame(out_rows)


def parse_brightdata_raw_dir(input_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(input_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text("utf-8"))
            parsed = parse_brightdata_raw_cache(path)
            parsed["benchmark_id"] = data.get("benchmark_id") or path.stem
            parsed["source_url"] = data.get("source_url") or data.get("requested_url") or parsed.get("requested_url")
            parsed["provider"] = data.get("provider") or "brightdata"
            parsed["live_attempted"] = str(data.get("provider_status") or "").lower() not in {"planned_dry_run", "dry_run", ""}
            parsed["content_quality_flag"] = parsed.get("content_quality_flag") or content_quality_flag({**parsed, "raw_cache_exists": True})
            parsed["page_text_excerpt"] = _excerpt(parsed.get("page_text"))
            rows.append(parsed)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "benchmark_id": path.stem,
                    "source_url": "",
                    "requested_url": "",
                    "final_url": "",
                    "normalized_url": "",
                    "provider": "brightdata",
                    "live_attempted": False,
                    "scrape_success": False,
                    "parse_success": False,
                    "scraped_body_available": False,
                    "can_reparse": False,
                    "html": "",
                    "markdown": "",
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
                    "parse_error": str(exc),
                    "parse_error_category": "parser_exception",
                    "content_quality_flag": "parse_failed",
                    "page_text_excerpt": "",
                    "raw_response_present": False,
                    "response_shape": "parser_exception",
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
            )
    columns = [
        "benchmark_id", "source_url", "requested_url", "final_url", "normalized_url", "provider", "live_attempted",
        "provider_mode", "scrape_success", "parse_success", "scraped_body_available", "can_reparse",
        "html", "markdown", "html_available",
        "markdown_available", "text_available", "page_title", "meta_description", "page_text",
        "text_char_count", "word_count", "heading_count", "table_count", "link_count",
        "image_count", "language_detected", "status_code", "parse_error", "parse_error_category",
        "content_quality_flag", "page_text_excerpt", "raw_response_present", "response_shape",
        "body_field_selected", "body_field_kind", "body_field_char_count", "body_field_candidate_count",
        "body_field_candidates", "largest_string_field_path", "largest_string_length",
        "body_available_but_not_main_content", "blocked_or_verification_detected",
    ]
    return pd.DataFrame(rows, columns=columns)


def _quality_score(flag: Any) -> int:
    return QUALITY_SCORE.get(str(flag or "").strip(), 0)


def _is_short_or_bad(flag: Any) -> bool:
    return str(flag or "") in {"very_short_text", "boilerplate_only", "nav_footer_only", "dynamic_js_likely", "blocked_or_error_page", "parse_failed", "no_raw_cache", "empty_text"}


def _provider_summary(provider: str, df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    attempted = len(df) if provider == "apify" else int(df[f"{prefix}_scrape_success"].notna().sum()) if f"{prefix}_scrape_success" in df else 0
    if attempted == 0:
        return {
            "provider": provider,
            "attempted": 0,
            "scrape_success_rate": 0.0,
            "parse_success_rate": 0.0,
            "scraped_body_available_rate": 0.0,
            "median_word_count": 0.0,
            "pct_word_count_lt_100": 0.0,
            "pct_word_count_lt_300": 0.0,
            "pct_content_quality_ok": 0.0,
            "pct_boilerplate_or_short": 0.0,
            "pct_dynamic_or_blocked": 0.0,
        }
    sub = df[df[f"{prefix}_scrape_success"].notna()] if provider != "apify" else df
    wc = _num(sub[f"{prefix}_word_count"], 0)
    flag = sub[f"{prefix}_content_quality_flag"].fillna("")
    return {
        "provider": provider,
        "attempted": int(len(sub)),
        "scrape_success_rate": float(sub[f"{prefix}_scrape_success"].map(_bool).mean()),
        "parse_success_rate": float(sub[f"{prefix}_parse_success"].map(_bool).mean()),
        "scraped_body_available_rate": float(sub[f"{prefix}_scraped_body_available"].map(_bool).mean()),
        "median_word_count": float(wc.median()),
        "pct_word_count_lt_100": float((wc < 100).mean()),
        "pct_word_count_lt_300": float((wc < 300).mean()),
        "pct_content_quality_ok": float(flag.eq("ok").mean()),
        "pct_boilerplate_or_short": float(flag.isin({"very_short_text", "boilerplate_only", "nav_footer_only"}).mean()),
        "pct_dynamic_or_blocked": float(flag.isin({"dynamic_js_likely", "blocked_or_error_page"}).mean()),
    }


def compare_providers(benchmark: pd.DataFrame, bright: pd.DataFrame, apify_parse: pd.DataFrame, final_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ap_cols = [
        "normalized_url", "scrape_success", "parse_success", "scraped_body_available", "word_count",
        "heading_count", "table_count", "page_text", "page_text_excerpt", "content_quality_flag",
    ]
    ap = apify_parse[[c for c in ap_cols if c in apify_parse.columns]].drop_duplicates("normalized_url").copy()
    if "content_quality_flag" not in ap.columns:
        ap["content_quality_flag"] = ap.apply(lambda r: content_quality_flag({**r.to_dict(), "raw_cache_exists": True}), axis=1)
    if "page_text_excerpt" not in ap.columns and "page_text" in ap.columns:
        ap["page_text_excerpt"] = ap["page_text"].map(_excerpt)
    final = final_rows[
        [c for c in ["normalized_url", "page_type_final", "page_type_family"] if c in final_rows.columns]
    ].drop_duplicates("normalized_url")
    ap = ap.merge(final, on="normalized_url", how="left")
    if "benchmark_id" not in bright.columns:
        bright = pd.DataFrame(columns=["benchmark_id"])
    bright_one = bright.drop_duplicates("benchmark_id").copy() if len(bright) else pd.DataFrame(columns=bright.columns)
    out = benchmark.merge(ap, on="normalized_url", how="left", suffixes=("", "_apify"))
    out = out.merge(bright_one, on="benchmark_id", how="left", suffixes=("", "_brightdata"))

    records = []
    for _, row in out.iterrows():
        ap_wc = int(pd.to_numeric(pd.Series([row.get("word_count")]), errors="coerce").fillna(0).iloc[0])
        bd_wc = int(pd.to_numeric(pd.Series([row.get("word_count_brightdata")]), errors="coerce").fillna(0).iloc[0])
        ap_flag = row.get("content_quality_flag") or row.get("current_content_quality_flag") or ""
        bd_flag = row.get("content_quality_flag_brightdata") or ""
        bd_attempted = _bool(row.get("live_attempted_brightdata")) if "live_attempted_brightdata" in row.index else not pd.isna(row.get("scrape_success_brightdata"))
        bd_success = _bool(row.get("scrape_success_brightdata")) if bd_attempted else False
        ap_success = _bool(row.get("scrape_success"))
        quality_better = _quality_score(bd_flag) > _quality_score(ap_flag)
        fixed_parse_failure = (not _bool(row.get("parse_success"))) and _bool(row.get("parse_success_brightdata"))
        fixed_short = ap_flag in {"very_short_text", "boilerplate_only", "dynamic_js_likely"} and bd_flag == "ok"
        better_text = bool(
            bd_success
            and (
                fixed_parse_failure
                or fixed_short
                or (ap_wc > 0 and bd_wc >= ap_wc * 1.5 and quality_better)
                or (ap_wc == 0 and bd_wc >= 100 and bd_flag == "ok")
            )
        )
        if not bd_attempted:
            provider = "not_evaluated"
            reason = "Bright Data was not attempted for this URL."
        elif better_text:
            provider = "brightdata"
            reason = "Bright Data improved parse/text quality under benchmark rules."
        elif bd_success and ap_flag != "ok" and bd_flag == "ok":
            provider = "brightdata"
            reason = "Bright Data returned ok content where Apify quality was weak."
        elif not bd_success:
            provider = "apify"
            reason = "Bright Data failed or returned no usable body."
        else:
            provider = "apify"
            reason = "Apify remains adequate or Bright Data did not clearly improve quality."
        records.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "source_url": row.get("source_url"),
                "source_root_domain": row.get("source_root_domain"),
                "cited": row.get("cited"),
                "reason_selected": row.get("reason_selected"),
                "apify_scrape_success": row.get("scrape_success"),
                "apify_parse_success": row.get("parse_success"),
                "apify_scraped_body_available": row.get("scraped_body_available"),
                "apify_word_count": ap_wc,
                "apify_heading_count": row.get("heading_count"),
                "apify_table_count": row.get("table_count"),
                "apify_content_quality_flag": ap_flag,
                "apify_page_text_excerpt": row.get("page_text_excerpt"),
                "apify_page_type_final": row.get("page_type_final"),
                "apify_page_type_family": row.get("page_type_family"),
                "brightdata_scrape_success": row.get("scrape_success_brightdata") if bd_attempted else pd.NA,
                "brightdata_parse_success": row.get("parse_success_brightdata") if bd_attempted else pd.NA,
                "brightdata_scraped_body_available": row.get("scraped_body_available_brightdata") if bd_attempted else pd.NA,
                "brightdata_word_count": bd_wc if bd_attempted else pd.NA,
                "brightdata_heading_count": row.get("heading_count_brightdata") if bd_attempted else pd.NA,
                "brightdata_table_count": row.get("table_count_brightdata") if bd_attempted else pd.NA,
                "brightdata_content_quality_flag": bd_flag if bd_attempted else pd.NA,
                "brightdata_page_text_excerpt": row.get("page_text_excerpt_brightdata") if bd_attempted else pd.NA,
                "word_count_delta": bd_wc - ap_wc if bd_attempted else pd.NA,
                "brightdata_better_text": better_text,
                "brightdata_fixed_parse_failure": fixed_parse_failure,
                "brightdata_fixed_short_text": fixed_short,
                "brightdata_quality_better": quality_better if bd_attempted else False,
                "recommended_provider_for_url": provider,
                "recommendation_reason": reason,
            }
        )
    result = pd.DataFrame(records)
    summary = pd.DataFrame(
        [
            _provider_summary("apify", result, "apify"),
            _provider_summary("brightdata", result, "brightdata"),
        ]
    )
    return result, summary


def compare_providers_with_quality(
    benchmark: pd.DataFrame,
    bright: pd.DataFrame,
    apify_quality: pd.DataFrame,
    final_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ap = apify_quality.rename(
        columns={
            "scrape_success": "apify_scrape_success",
            "parse_success": "apify_parse_success",
            "scraped_body_available": "apify_scraped_body_available",
            "word_count": "apify_word_count",
            "heading_count": "apify_heading_count",
            "table_count": "apify_table_count",
            "content_quality_flag": "apify_content_quality_flag",
            "page_text_excerpt": "apify_page_text_excerpt",
        }
    )
    ap_cols = [
        "normalized_url", "apify_scrape_success", "apify_parse_success",
        "apify_scraped_body_available", "apify_word_count", "apify_heading_count",
        "apify_table_count", "apify_content_quality_flag", "apify_page_text_excerpt",
    ]
    ap = ap[[c for c in ap_cols if c in ap.columns]].drop_duplicates("normalized_url")
    final = final_rows[
        [c for c in ["normalized_url", "page_type_final", "page_type_family"] if c in final_rows.columns]
    ].drop_duplicates("normalized_url").rename(
        columns={"page_type_final": "apify_page_type_final", "page_type_family": "apify_page_type_family"}
    )
    base = benchmark.merge(ap, on="normalized_url", how="left").merge(final, on="normalized_url", how="left")
    if "benchmark_id" not in bright.columns:
        bright = pd.DataFrame(columns=["benchmark_id"])
    bright_one = bright.drop_duplicates("benchmark_id").add_prefix("brightdata_") if len(bright) else pd.DataFrame(columns=["brightdata_benchmark_id"])
    out = base.merge(bright_one, left_on="benchmark_id", right_on="brightdata_benchmark_id", how="left")
    rows = []
    for _, row in out.iterrows():
        ap_wc = int(pd.to_numeric(pd.Series([row.get("apify_word_count")]), errors="coerce").fillna(0).iloc[0])
        bd_wc = int(pd.to_numeric(pd.Series([row.get("brightdata_word_count")]), errors="coerce").fillna(0).iloc[0])
        ap_flag = row.get("apify_content_quality_flag") or row.get("current_content_quality_flag") or ""
        bd_flag = row.get("brightdata_content_quality_flag") or ""
        bd_attempted = _bool(row.get("brightdata_live_attempted")) if "brightdata_live_attempted" in row.index else not pd.isna(row.get("brightdata_scrape_success"))
        bd_success = _bool(row.get("brightdata_scrape_success")) if bd_attempted else False
        quality_better = _quality_score(bd_flag) > _quality_score(ap_flag)
        fixed_parse_failure = (not _bool(row.get("apify_parse_success"))) and _bool(row.get("brightdata_parse_success"))
        fixed_short = ap_flag in {"very_short_text", "boilerplate_only", "dynamic_js_likely"} and bd_flag == "ok"
        better_text = bool(
            bd_success
            and (
                fixed_parse_failure
                or fixed_short
                or (ap_wc > 0 and bd_wc >= ap_wc * 1.5 and quality_better)
                or (ap_wc == 0 and bd_wc >= 100 and bd_flag == "ok")
            )
        )
        if not bd_attempted:
            provider = "not_evaluated"
            reason = "Bright Data was not attempted for this URL."
        elif better_text:
            provider = "brightdata"
            reason = "Bright Data improved parse/text quality under benchmark rules."
        elif bd_success and ap_flag != "ok" and bd_flag == "ok":
            provider = "brightdata"
            reason = "Bright Data returned ok content where Apify quality was weak."
        elif not bd_success:
            provider = "apify"
            reason = "Bright Data failed or returned no usable body."
        else:
            provider = "apify"
            reason = "Apify remains adequate or Bright Data did not clearly improve quality."
        rows.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "source_url": row.get("source_url"),
                "source_root_domain": row.get("source_root_domain"),
                "cited": row.get("cited"),
                "reason_selected": row.get("reason_selected"),
                "apify_scrape_success": row.get("apify_scrape_success"),
                "apify_parse_success": row.get("apify_parse_success"),
                "apify_scraped_body_available": row.get("apify_scraped_body_available"),
                "apify_word_count": ap_wc,
                "apify_heading_count": row.get("apify_heading_count"),
                "apify_table_count": row.get("apify_table_count"),
                "apify_content_quality_flag": ap_flag,
                "apify_page_text_excerpt": row.get("apify_page_text_excerpt"),
                "apify_page_type_final": row.get("apify_page_type_final"),
                "apify_page_type_family": row.get("apify_page_type_family"),
                "brightdata_scrape_success": row.get("brightdata_scrape_success") if bd_attempted else pd.NA,
                "brightdata_parse_success": row.get("brightdata_parse_success") if bd_attempted else pd.NA,
                "brightdata_scraped_body_available": row.get("brightdata_scraped_body_available") if bd_attempted else pd.NA,
                "brightdata_word_count": bd_wc if bd_attempted else pd.NA,
                "brightdata_heading_count": row.get("brightdata_heading_count") if bd_attempted else pd.NA,
                "brightdata_table_count": row.get("brightdata_table_count") if bd_attempted else pd.NA,
                "brightdata_content_quality_flag": bd_flag if bd_attempted else pd.NA,
                "brightdata_page_text_excerpt": row.get("brightdata_page_text_excerpt") if bd_attempted else pd.NA,
                "word_count_delta": bd_wc - ap_wc if bd_attempted else pd.NA,
                "brightdata_better_text": better_text,
                "brightdata_fixed_parse_failure": fixed_parse_failure,
                "brightdata_fixed_short_text": fixed_short,
                "brightdata_quality_better": quality_better if bd_attempted else False,
                "recommended_provider_for_url": provider,
                "recommendation_reason": reason,
            }
        )
    result = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            _provider_summary("apify", result, "apify"),
            _provider_summary("brightdata", result, "brightdata"),
        ]
    )
    return result, summary


def build_page_type_comparison(results: pd.DataFrame, bright: pd.DataFrame) -> pd.DataFrame:
    bright_one = bright.drop_duplicates("benchmark_id").set_index("benchmark_id") if len(bright) else pd.DataFrame()
    rows = []
    for _, row in results.iterrows():
        bid = row.get("benchmark_id")
        brow = bright_one.loc[bid].to_dict() if len(bright_one) and bid in bright_one.index else {}
        if brow and str(brow.get("page_text") or "").strip():
            stype, _ = classify_source_type(str(row.get("source_url") or brow.get("final_url") or ""))
            pt = classify_page_type_details(
                {
                    "final_url": brow.get("final_url"),
                    "requested_url": brow.get("requested_url"),
                    "page_title": brow.get("page_title"),
                    "meta_description": brow.get("meta_description"),
                    "page_text": brow.get("page_text"),
                    "table_count": brow.get("table_count"),
                    "source_type_url": stype,
                },
                stype,
            )
        else:
            pt = None
        bd_type = pt.page_type if pt else "unknown"
        bd_family = pt.family if pt else "unknown"
        bd_conf = pt.confidence if pt else "unknown"
        bd_evidence = pt.evidence if pt else "not_attempted_or_no_text"
        ap_unknown = str(row.get("apify_page_type_final") or "") == "unknown"
        bd_unknown = bd_type == "unknown"
        resolved = ap_unknown and not bd_unknown and bd_conf in {"medium", "high"}
        better_text = _bool(row.get("brightdata_better_text"))
        if resolved:
            notes = "Bright Data text resolved an Apify unknown with medium/high confidence."
        elif ap_unknown and bd_unknown and better_text:
            notes = "Bright Data text improved, but page type remains unknown; likely classifier/taxonomy gap."
        elif ap_unknown and bd_unknown:
            notes = "Unknown remains; likely scrape quality or unsupported page evidence."
        else:
            notes = "No Apify unknown resolution."
        rows.append(
            {
                "benchmark_id": bid,
                "source_url": row.get("source_url"),
                "reason_selected": row.get("reason_selected"),
                "apify_page_type_final": row.get("apify_page_type_final"),
                "apify_page_type_family": row.get("apify_page_type_family"),
                "brightdata_page_type_final": bd_type,
                "brightdata_page_type_family": bd_family,
                "brightdata_page_type_confidence": bd_conf,
                "brightdata_page_type_evidence": bd_evidence,
                "apify_unknown": ap_unknown,
                "brightdata_unknown": bd_unknown,
                "brightdata_resolved_unknown": resolved,
                "page_type_changed": str(row.get("apify_page_type_final") or "") != bd_type,
                "likely_improvement": bool(resolved or (_bool(row.get("brightdata_quality_better")) and not bd_unknown)),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def build_strategy_recommendations(results: pd.DataFrame, page_types: pd.DataFrame) -> pd.DataFrame:
    attempted = int(results["brightdata_scrape_success"].notna().sum()) if "brightdata_scrape_success" in results else 0
    wins = int(results["recommended_provider_for_url"].eq("brightdata").sum()) if len(results) else 0
    resolved = int(page_types["brightdata_resolved_unknown"].sum()) if len(page_types) else 0
    strong = attempted > 0 and (wins / attempted >= 0.5 or resolved >= 5)
    any_win = attempted > 0 and wins > 0
    rows = [
        ("keep_apify_only", "Continue using current Apify scrape cache for the full dataset.", attempted == 0 or not any_win, "Recommended until Bright Data benchmark produces successful improvements." if attempted == 0 else "Apify remains sufficient for most benchmark URLs."),
        ("apify_cheerio_then_apify_playwright_fallback", "Keep Apify primary and test Apify Playwright for parse_failed/dynamic/short pages.", attempted == 0 or not strong, "Lowest-change fallback path; still useful before adding a second provider."),
        ("apify_then_brightdata_fallback", "Use Bright Data only after Apify parse/quality failure.", bool(any_win), "Use if benchmark wins concentrate on parse_failed, dynamic, or very short pages."),
        ("brightdata_primary_for_problematic_domains", "Use Bright Data first for domains where benchmark win rate is high.", bool(strong), "Only reasonable if domain-level wins are repeated."),
        ("brightdata_primary_for_all", "Replace Apify with Bright Data for all source URLs.", False, "Not recommended unless a larger benchmark strongly outperforms Apify on quality, coverage, and cost."),
        ("serper_metadata_fallback_only", "Use search metadata only when scraping fails.", attempted == 0, "Metadata can preserve descriptive features but cannot replace page-content features."),
    ]
    return pd.DataFrame(rows, columns=["strategy", "description", "recommended", "reason"])


def build_domain_strategy(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame(columns=["source_root_domain", "n_benchmark_urls", "apify_ok_rate", "brightdata_ok_rate", "brightdata_win_rate", "recommended_provider", "reason"])
    rows = []
    for domain, group in results.groupby("source_root_domain", dropna=False):
        ap_ok = group["apify_content_quality_flag"].fillna("").eq("ok").mean()
        bd_attempted = group[group["brightdata_scrape_success"].notna()]
        bd_ok = bd_attempted["brightdata_content_quality_flag"].fillna("").eq("ok").mean() if len(bd_attempted) else 0.0
        win_rate = group["recommended_provider_for_url"].eq("brightdata").mean()
        if len(bd_attempted) == 0:
            provider = "not_evaluated"
            reason = "Bright Data not attempted yet."
        elif win_rate >= 0.5 and bd_ok > ap_ok:
            provider = "brightdata_fallback_for_domain"
            reason = "Bright Data wins on at least half of benchmark URLs for this domain."
        else:
            provider = "apify"
            reason = "No clear Bright Data domain advantage in current benchmark."
        rows.append(
            {
                "source_root_domain": domain,
                "n_benchmark_urls": int(len(group)),
                "apify_ok_rate": float(ap_ok),
                "brightdata_ok_rate": float(bd_ok),
                "brightdata_win_rate": float(win_rate),
                "recommended_provider": provider,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_provider_benchmark_input(final_rows_path: str | Path, quality_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    df = select_provider_benchmark_urls(read_csv(final_rows_path), read_csv(quality_path))
    write_csv(output_path, df)
    return df
