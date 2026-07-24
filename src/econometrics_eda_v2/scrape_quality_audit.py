from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.io import OUTPUT_DIR, RAW_CACHE_DIR


def _series(df: pd.DataFrame, col: str, default=None) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _excerpt(text: object, n: int = 280) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(text) else str(text)).strip()[:n]


def content_quality_flag(row: pd.Series) -> str:
    if not bool(row.get("raw_cache_exists", False)):
        return "no_raw_cache"
    if not bool(row.get("parse_success", False)):
        return "parse_failed"
    text = str(row.get("page_text") or "")
    title = str(row.get("page_title") or "")
    low = " ".join([title, text[:800]]).casefold()
    word_count = pd.to_numeric(pd.Series([row.get("word_count")]), errors="coerce").fillna(0).iloc[0]
    text_chars = pd.to_numeric(pd.Series([row.get("text_char_count")]), errors="coerce").fillna(0).iloc[0]
    if text_chars == 0 or not text.strip():
        return "empty_text"
    if re.search(r"\b(?:403|404)\b|not found|access denied|forbidden|nginx|error page", low):
        return "blocked_or_error_page"
    if re.search(r"please wait|verification|captcha|enable javascript|just a moment|checking your browser", low):
        return "dynamic_js_likely"
    if word_count < 20:
        return "very_short_text"
    nav_terms = len(re.findall(r"\b(menu|home|privacy|terms|cookie|login|subscribe|follow us|copyright)\b|หน้าแรก|เมนู|เข้าสู่ระบบ", low))
    content_terms = len(re.findall(r"health|hospital|doctor|service|treatment|product|article|โรค|แพทย์|รักษา|บริการ|สุขภาพ|ผลิตภัณฑ์", low))
    if word_count < 80 and nav_terms >= 3 and content_terms == 0:
        return "nav_footer_only"
    if word_count < 120 and re.search(r"cookie|privacy|terms|copyright|all rights reserved", low) and content_terms == 0:
        return "boilerplate_only"
    return "ok"


def suspected_root_cause(row: pd.Series) -> str:
    quality = str(row.get("content_quality_flag") or "")
    if quality != "ok":
        return quality
    if not bool(row.get("scraped_body_available", False)):
        return "no_scraped_body"
    reason = str(row.get("unknown_reason") or row.get("page_type_unknown_reason") or "")
    if reason:
        return reason
    if str(row.get("page_type_url_seed") or "") == "unknown" and str(row.get("page_type_scraped_enriched") or "") == "unknown":
        return "classifier_rules_or_weak_page_evidence"
    return "needs_manual_review"


def build_scrape_quality_audit(
    source_rows: pd.DataFrame,
    queue: pd.DataFrame,
    parsed: pd.DataFrame,
    page_features: pd.DataFrame,
    final_rows: pd.DataFrame,
    raw_dir: str | Path = RAW_CACHE_DIR,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    source = source_rows.copy()
    queue = queue.copy()
    parsed = parsed.copy()
    page_features = page_features.copy()
    final_rows = final_rows.copy()

    source_urls = source.groupby("normalized_url", dropna=False).agg(
        source_url=("source_url", "first"),
        source_root_domain=("source_domain", "first"),
        source_rows_n=("source_row_id", "count"),
        cited_rows_n=("cited", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    ).reset_index()
    if "source_root_domain" in final_rows.columns:
        roots = final_rows.groupby("normalized_url", dropna=False)["source_root_domain"].first()
        source_urls["source_root_domain"] = source_urls["normalized_url"].map(roots).fillna(source_urls["source_root_domain"])

    q_cols = ["normalized_url", "scrape_id", "cache_path", "scrape_status", "should_scrape"]
    merged = source_urls.merge(queue[[c for c in q_cols if c in queue.columns]], on="normalized_url", how="left")
    if "cache_path" in merged.columns:
        merged["raw_cache_exists"] = merged["cache_path"].fillna("").map(lambda p: Path(str(p)).exists() if str(p).strip() else False)
    else:
        merged["raw_cache_exists"] = merged["normalized_url"].map(lambda u: any(raw_dir.glob(f"*{str(u)}*"))).fillna(False)

    parse_cols = [
        "normalized_url", "requested_normalized_url", "final_normalized_url", "scrape_success",
        "parse_success", "scraped_body_available", "text_char_count", "word_count",
        "heading_count", "table_count", "page_title", "meta_description", "page_text",
        "parse_error",
    ]
    merged = merged.merge(parsed[[c for c in parse_cols if c in parsed.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")

    feat_cols = ["normalized_url", "content_feature_available", "page_type_scraped_enriched", "page_type_scraped_confidence"]
    merged = merged.merge(page_features[[c for c in feat_cols if c in page_features.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")

    final_cols = [
        "normalized_url", "page_type_url_seed", "page_type_final", "page_type_final_source",
        "unknown_reason", "page_type_unknown_reason",
    ]
    final_one = final_rows[[c for c in final_cols if c in final_rows.columns]].drop_duplicates("normalized_url")
    merged = merged.merge(final_one, on="normalized_url", how="left")
    joined_counts = final_rows.groupby("normalized_url", dropna=False).size()
    merged["joined_final_rows"] = merged["normalized_url"].map(joined_counts).fillna(0).astype(int)
    merged["in_scrape_queue"] = merged["scrape_id"].notna()
    merged["in_parsed_pages"] = merged["scrape_success"].notna()
    merged["in_page_features"] = merged["content_feature_available"].notna()
    merged["page_text_excerpt"] = merged["page_text"].map(_excerpt)
    merged["content_quality_flag"] = merged.apply(content_quality_flag, axis=1)
    merged["source_root_domain"] = merged["source_root_domain"].fillna("")
    return merged[
        [
            "normalized_url", "source_url", "source_root_domain", "source_rows_n", "cited_rows_n",
            "in_scrape_queue", "raw_cache_exists", "in_parsed_pages", "in_page_features", "joined_final_rows",
            "scrape_success", "parse_success", "scraped_body_available", "text_char_count", "word_count",
            "heading_count", "table_count", "page_title", "meta_description", "page_text_excerpt",
            "page_type_url_seed", "page_type_scraped_enriched", "page_type_final", "page_type_final_source",
            "content_quality_flag",
        ]
    ]


def build_unknown_scrape_review_sample(final_rows: pd.DataFrame, quality: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    unknown = final_rows[final_rows.get("page_type_final", pd.Series(index=final_rows.index, dtype=object)).fillna("").astype(str).eq("unknown")].copy()
    if unknown.empty:
        return pd.DataFrame()
    q_cols = ["normalized_url", "content_quality_flag", "page_text_excerpt"]
    merged = unknown.merge(quality[[c for c in q_cols if c in quality.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")
    merged["suspected_root_cause"] = merged.apply(suspected_root_cause, axis=1)
    merged["_priority"] = pd.to_numeric(merged.get("cited", 0), errors="coerce").fillna(0) * 1000 + pd.to_numeric(merged.get("source_position", 999), errors="coerce").fillna(999).rsub(999)
    out = merged.sort_values(["_priority", "word_count"], ascending=[False, True]).drop_duplicates("normalized_url").head(n)
    return pd.DataFrame(
        {
            "source_url": out.get("source_url"),
            "domain": out.get("source_root_domain", out.get("source_domain_host")),
            "title": out.get("page_title", out.get("source_title")),
            "word_count": out.get("word_count"),
            "page_text_excerpt": out.get("page_text_excerpt", pd.Series([""] * len(out), index=out.index)),
            "page_type_url_seed": out.get("page_type_url_seed"),
            "page_type_scraped_enriched": out.get("page_type_scraped_enriched"),
            "page_type_final": out.get("page_type_final"),
            "unknown_reason": out.get("unknown_reason", out.get("page_type_unknown_reason")),
            "suspected_root_cause": out.get("suspected_root_cause"),
        }
    )


def build_crawler_type_comparison_plan(final_rows: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    work = final_rows.merge(
        quality[["normalized_url", "content_quality_flag"]].drop_duplicates("normalized_url"),
        on="normalized_url",
        how="left",
    )
    work["word_count_num"] = pd.to_numeric(work.get("word_count"), errors="coerce").fillna(0)
    picks = []

    def add(frame: pd.DataFrame, bucket: str, limit: int) -> None:
        for _, r in frame.drop_duplicates("normalized_url").head(limit).iterrows():
            picks.append((bucket, r))

    add(work[work["page_type_final"].fillna("").astype(str).eq("unknown") & (work["word_count_num"] < 100)].sort_values("word_count_num"), "unknown_low_word_count", 10)
    add(work[work["page_type_family"].fillna("").astype(str).eq("commercial_price_package")].sort_values("source_position"), "commercial_product_marketplace", 10)
    add(work[pd.to_numeric(work.get("cited"), errors="coerce").fillna(0).eq(1)].sort_values("source_position"), "high_impact_cited", 10)

    seen = set()
    rows = []
    for bucket, r in picks:
        nurl = str(r.get("normalized_url") or "")
        if nurl in seen:
            continue
        seen.add(nurl)
        for crawler in ["cheerio", "playwright:adaptive", "playwright:firefox"]:
            rows.append(
                {
                    "selection_bucket": bucket,
                    "crawler_type": crawler,
                    "source_url": r.get("source_url"),
                    "normalized_url": nurl,
                    "current_word_count": r.get("word_count"),
                    "current_content_quality_flag": r.get("content_quality_flag"),
                    "current_page_type_final": r.get("page_type_final"),
                    "planned_metrics": "scrape_success; word_count; heading_count; page_text_excerpt quality; page_type result",
                    "dry_run_command": (
                        ".venv/bin/python scripts/v2_scrape_urls_apify.py "
                        "--queue <single-url-queue.csv> "
                        "--output-dir data/econometrics_v2/scrape_cache/raw_crawler_compare/"
                        f"{crawler.replace(':', '_')} --provider apify --cache true "
                        f"--crawler-type {crawler}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_scrape_quality_outputs(
    source_rows: pd.DataFrame,
    queue: pd.DataFrame,
    parsed: pd.DataFrame,
    page_features: pd.DataFrame,
    final_rows: pd.DataFrame,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    tables = Path(output_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    quality = build_scrape_quality_audit(source_rows, queue, parsed, page_features, final_rows)
    unknown = build_unknown_scrape_review_sample(final_rows, quality)
    plan = build_crawler_type_comparison_plan(final_rows, quality)
    quality.to_csv(tables / "scrape_quality_audit.csv", index=False)
    unknown.to_csv(tables / "unknown_scrape_review_sample.csv", index=False)
    plan.to_csv(tables / "crawler_type_comparison_plan.csv", index=False)
    return {
        "scrape_quality_rows": int(len(quality)),
        "unknown_sample_rows": int(len(unknown)),
        "crawler_plan_rows": int(len(plan)),
    }
