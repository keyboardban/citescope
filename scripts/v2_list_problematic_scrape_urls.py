#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.io import ensure_v2_dirs, read_csv, write_csv, write_json


HIGH_FLAGS = {
    "dynamic_js_likely",
    "parse_failed",
    "blocked_or_error_page",
    "boilerplate_only",
    "nav_footer_only",
    "empty_text",
}
BAD_UNKNOWN_FLAGS = HIGH_FLAGS | {"very_short_text"}
TITLE_BLOCK_PAT = re.compile(
    r"human verification|please wait|checking your browser|access denied|just a moment|cloudflare|captcha|recaptcha|robot|forbidden|\b403\b|\b404\b",
    flags=re.I,
)
JS_PAT = re.compile(r"enable javascript|javascript|single page app|please wait|loading|checking your browser", flags=re.I)
REDIRECT_PAT = re.compile(r"pageredirect|redirect|tracking|/jump|/out|[?&]url=|[?&]u=", flags=re.I)
PDF_PAT = re.compile(r"\.pdf(?:$|[?#])", flags=re.I)
DYNAMIC_DOMAIN_PAT = re.compile(
    r"(?:facebook|instagram|tiktok|youtube|reddit|line\.me|apps\.apple|play\.google|shopee|lazada|amazon|bigc|watsons|jdcentral|shopify|seller\.tiktok|lemon8|cosrx|alibaba)",
    flags=re.I,
)
ECOM_SOCIAL_PAT = re.compile(
    r"(?:shopee|lazada|amazon|bigc|watsons|jdcentral|shopify|seller\.tiktok|tiktok|facebook|instagram|youtube|line\.me|lemon8|alibaba)",
    flags=re.I,
)


def _num(row: pd.Series, key: str, default: float = 0) -> float:
    return float(pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").fillna(default).iloc[0])


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if pd.isna(v):
        return False
    return str(v).strip().casefold() in {"1", "true", "yes", "y"}


def _text(row: pd.Series, key: str) -> str:
    v = row.get(key)
    return "" if pd.isna(v) else str(v)


def classify_issue(row: pd.Series) -> str:
    flag = _text(row, "content_quality_flag")
    title_excerpt = " ".join([_text(row, "page_title"), _text(row, "page_text_excerpt")])
    url = " ".join([_text(row, "source_url"), _text(row, "normalized_url")])
    word_count = _num(row, "word_count")
    if PDF_PAT.search(url) and flag in HIGH_FLAGS | {"very_short_text"}:
        return "pdf_or_binary"
    if TITLE_BLOCK_PAT.search(title_excerpt):
        return "blocked_or_captcha"
    if flag == "blocked_or_error_page":
        return "blocked_or_captcha"
    if flag == "parse_failed" or (not _bool(row.get("parse_success")) and not _bool(row.get("scraped_body_available")) and word_count == 0):
        if REDIRECT_PAT.search(url):
            return "redirect_or_tracking_url"
        return "parse_failed"
    if flag == "dynamic_js_likely" or JS_PAT.search(title_excerpt) or (DYNAMIC_DOMAIN_PAT.search(url) and word_count < 100):
        return "js_heavy_likely"
    if flag == "boilerplate_only":
        return "boilerplate_only"
    if flag == "nav_footer_only":
        return "nav_footer_only"
    if word_count > 0 and word_count < 100:
        return "very_short_text"
    if flag == "ok" and word_count >= 300 and _text(row, "page_type_final") == "unknown":
        return "classifier_issue_not_scraper"
    if flag == "ok" and word_count < 300:
        return "low_content_but_probably_valid"
    return "unknown_problem"


def is_problematic(row: pd.Series) -> bool:
    flag = _text(row, "content_quality_flag")
    word_count = _num(row, "word_count")
    text_chars = _num(row, "text_char_count")
    page_type = _text(row, "page_type_final")
    url_seed = _text(row, "page_type_url_seed")
    scraped_type = _text(row, "page_type_scraped_enriched")
    title = _text(row, "page_title")
    high_or_medium = (
        flag in HIGH_FLAGS
        or flag == "very_short_text"
        or word_count < 100
        or text_chars < 500
        or not _bool(row.get("scraped_body_available"))
        or not _bool(row.get("parse_success"))
        or not _bool(row.get("scrape_success"))
    )
    suspicious = (
        (page_type == "unknown" and word_count < 300)
        or (page_type == "unknown" and flag != "ok")
        or (url_seed not in {"", "unknown"} and scraped_type == "unknown")
        or (url_seed not in {"", "unknown"} and page_type == "unknown")
        or bool(TITLE_BLOCK_PAT.search(title))
        or (flag == "ok" and word_count >= 300 and page_type == "unknown")
    )
    return bool(high_or_medium or suspicious)


def recommend_next_crawler(row: pd.Series, issue: str) -> str:
    url = " ".join([_text(row, "source_url"), _text(row, "normalized_url"), _text(row, "source_root_domain")])
    word_count = _num(row, "word_count")
    flag = _text(row, "content_quality_flag")
    if issue == "pdf_or_binary":
        return "pdf_parser"
    if issue == "classifier_issue_not_scraper":
        return "keep_cheerio"
    if issue in {"blocked_or_captcha"}:
        return "playwright:firefox"
    if issue == "redirect_or_tracking_url" and word_count == 0:
        return "playwright:adaptive"
    if issue == "js_heavy_likely":
        return "playwright:adaptive"
    if issue == "parse_failed":
        return "playwright:adaptive"
    if issue in {"very_short_text", "boilerplate_only", "nav_footer_only"}:
        return "playwright:adaptive" if ECOM_SOCIAL_PAT.search(url) or flag != "ok" else "manual_review"
    if flag == "ok" and word_count >= 300:
        return "keep_cheerio"
    if word_count < 100:
        return "playwright:adaptive"
    return "manual_review"


def recommended_action(row: pd.Series, issue: str, crawler: str) -> str:
    if issue == "classifier_issue_not_scraper":
        return "likely_classifier_taxonomy_issue"
    if crawler == "playwright:firefox":
        return "retry_with_playwright_firefox"
    if crawler == "playwright:adaptive":
        return "retry_with_playwright_adaptive"
    if crawler == "serper_metadata_only":
        return "use_serper_metadata_fallback"
    if crawler == "keep_cheerio":
        return "keep_cheerio"
    return "manual_review"


def priority_bucket(row: pd.Series, issue: str) -> str:
    cited = _num(row, "cited_rows_n") > 0
    flag = _text(row, "content_quality_flag")
    word_count = _num(row, "word_count")
    page_type = _text(row, "page_type_final")
    if cited and issue in {"blocked_or_captcha", "js_heavy_likely"}:
        return "P0_blocked_or_dynamic_cited"
    if issue in {"parse_failed", "redirect_or_tracking_url"} or flag in {"empty_text"} or word_count == 0:
        return "P1_parse_failed_or_empty"
    if page_type == "unknown" and word_count < 300:
        return "P2_unknown_low_word_count"
    if issue in {"very_short_text", "boilerplate_only", "nav_footer_only"}:
        return "P3_very_short_or_boilerplate"
    if issue == "classifier_issue_not_scraper":
        return "P4_unknown_but_content_ok_classifier_issue"
    return "P5_low_priority_manual_review"


def reason_for(row: pd.Series, issue: str, crawler: str) -> str:
    bits = [
        f"issue={issue}",
        f"flag={_text(row, 'content_quality_flag') or 'missing'}",
        f"word_count={int(_num(row, 'word_count'))}",
        f"page_type={_text(row, 'page_type_final') or 'missing'}",
    ]
    if _num(row, "cited_rows_n") > 0:
        bits.append(f"cited_rows_n={int(_num(row, 'cited_rows_n'))}")
    bits.append(f"next={crawler}")
    return "; ".join(bits)


def add_problem_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["suspected_scrape_issue_type"] = work.apply(classify_issue, axis=1)
    work["recommended_next_crawler"] = work.apply(lambda r: recommend_next_crawler(r, r["suspected_scrape_issue_type"]), axis=1)
    work["recommended_action"] = work.apply(lambda r: recommended_action(r, r["suspected_scrape_issue_type"], r["recommended_next_crawler"]), axis=1)
    work["priority_bucket"] = work.apply(lambda r: priority_bucket(r, r["suspected_scrape_issue_type"]), axis=1)
    work["reason"] = work.apply(lambda r: reason_for(r, r["suspected_scrape_issue_type"], r["recommended_next_crawler"]), axis=1)
    return work


def build_problem_urls(scrape_audit: pd.DataFrame) -> pd.DataFrame:
    work = add_problem_columns(scrape_audit)
    problem = work[work.apply(is_problematic, axis=1)].copy()
    bucket_order = {
        "P0_blocked_or_dynamic_cited": 0,
        "P1_parse_failed_or_empty": 1,
        "P2_unknown_low_word_count": 2,
        "P3_very_short_or_boilerplate": 3,
        "P4_unknown_but_content_ok_classifier_issue": 4,
        "P5_low_priority_manual_review": 5,
    }
    issue_order = {
        "blocked_or_captcha": 0,
        "js_heavy_likely": 1,
        "parse_failed": 2,
        "redirect_or_tracking_url": 3,
        "boilerplate_only": 4,
        "nav_footer_only": 5,
        "very_short_text": 6,
        "classifier_issue_not_scraper": 7,
    }
    problem["_bucket_order"] = problem["priority_bucket"].map(bucket_order).fillna(99)
    problem["_issue_order"] = problem["suspected_scrape_issue_type"].map(issue_order).fillna(99)
    problem["_cited_sort"] = pd.to_numeric(problem.get("cited_rows_n"), errors="coerce").fillna(0)
    problem["_source_rows_sort"] = pd.to_numeric(problem.get("source_rows_n"), errors="coerce").fillna(0)
    problem["_word_sort"] = pd.to_numeric(problem.get("word_count"), errors="coerce").fillna(0)
    problem["_unknown_sort"] = problem["page_type_final"].fillna("").astype(str).eq("unknown").astype(int)
    problem = problem.sort_values(
        ["_bucket_order", "_cited_sort", "_issue_order", "_word_sort", "_unknown_sort", "_source_rows_sort"],
        ascending=[True, False, True, True, False, False],
    ).reset_index(drop=True)
    problem.insert(0, "priority_rank", range(1, len(problem) + 1))
    cols = [
        "priority_rank", "priority_bucket", "source_url", "normalized_url", "source_root_domain",
        "cited_rows_n", "source_rows_n", "word_count", "text_char_count", "heading_count",
        "table_count", "content_quality_flag", "scrape_success", "parse_success",
        "scraped_body_available", "page_title", "page_text_excerpt", "page_type_url_seed",
        "page_type_scraped_enriched", "page_type_final", "page_type_final_source",
        "suspected_scrape_issue_type", "recommended_next_crawler", "recommended_action", "reason",
    ]
    return problem[[c for c in cols if c in problem.columns]]


def build_domain_summary(all_audit: pd.DataFrame, problem_urls: pd.DataFrame) -> pd.DataFrame:
    all_counts = all_audit.groupby("source_root_domain", dropna=False).size().rename("total_urls")
    if problem_urls.empty:
        return pd.DataFrame(columns=["source_root_domain", "total_urls", "problematic_urls", "problematic_rate"])
    p = problem_urls.copy()
    grouped = p.groupby("source_root_domain", dropna=False)
    out = grouped.size().rename("problematic_urls").to_frame()
    out = out.join(all_counts, how="left")
    out["problematic_rate"] = out["problematic_urls"] / out["total_urls"].replace(0, pd.NA)
    for col, mask in {
        "dynamic_js_likely_n": p["suspected_scrape_issue_type"].eq("js_heavy_likely"),
        "parse_failed_n": p["suspected_scrape_issue_type"].isin({"parse_failed", "redirect_or_tracking_url"}),
        "very_short_text_n": p["suspected_scrape_issue_type"].eq("very_short_text"),
        "blocked_or_captcha_n": p["suspected_scrape_issue_type"].eq("blocked_or_captcha"),
        "boilerplate_only_n": p["suspected_scrape_issue_type"].isin({"boilerplate_only", "nav_footer_only"}),
        "unknown_with_bad_content_n": p["page_type_final"].fillna("").eq("unknown") & p["content_quality_flag"].fillna("").ne("ok"),
        "unknown_with_ok_content_n": p["page_type_final"].fillna("").eq("unknown") & p["content_quality_flag"].fillna("").eq("ok"),
    }.items():
        out[col] = p[mask].groupby("source_root_domain", dropna=False).size()
    out = out.fillna(0).reset_index()

    def action(row: pd.Series) -> str:
        if row["blocked_or_captcha_n"] >= 1 or row["dynamic_js_likely_n"] >= 2:
            return "retry_with_playwright_firefox"
        domain = str(row["source_root_domain"])
        if row["parse_failed_n"] >= 1 and ECOM_SOCIAL_PAT.search(domain):
            return "retry_with_playwright_adaptive"
        if row["parse_failed_n"] + row["very_short_text_n"] >= 2:
            return "retry_with_playwright_adaptive"
        if row["unknown_with_ok_content_n"] > 0 and row["unknown_with_bad_content_n"] == 0:
            return "likely_classifier_taxonomy_issue"
        if str(row["source_root_domain"]).lower().endswith(".pdf"):
            return "manual_review"
        return "manual_review"

    out["recommended_action"] = out.apply(action, axis=1)
    first_cols = [
        "source_root_domain", "total_urls", "problematic_urls", "problematic_rate",
        "dynamic_js_likely_n", "parse_failed_n", "very_short_text_n", "blocked_or_captcha_n",
        "boilerplate_only_n", "unknown_with_bad_content_n", "unknown_with_ok_content_n", "recommended_action",
    ]
    return out[first_cols].sort_values(["problematic_urls", "problematic_rate"], ascending=[False, False])


def build_retry_queues(problem_urls: pd.DataFrame, top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retry = problem_urls[problem_urls["recommended_next_crawler"].isin({"playwright:adaptive", "playwright:firefox"})].head(top_n).copy()
    queue = pd.DataFrame(
        {
            "source_url": retry["source_url"],
            "normalized_url": retry["normalized_url"],
            "source_root_domain": retry["source_root_domain"],
            "retry_crawler_type": retry["recommended_next_crawler"],
            "retry_reason": retry["reason"],
            "current_content_quality_flag": retry["content_quality_flag"],
            "current_word_count": retry["word_count"],
            "current_page_type_final": retry["page_type_final"],
            "cited_rows_n": retry["cited_rows_n"],
        }
    )
    adaptive = queue[queue["retry_crawler_type"].eq("playwright:adaptive")].copy()
    firefox = queue[queue["retry_crawler_type"].eq("playwright:firefox")].copy()
    return queue, adaptive, firefox


def add_plan_picks(picks: list[pd.Series], frame: pd.DataFrame, n: int) -> None:
    seen = {str(r.get("normalized_url")) for r in picks}
    for _, row in frame.iterrows():
        nurl = str(row.get("normalized_url") or "")
        if not nurl or nurl in seen:
            continue
        picks.append(row)
        seen.add(nurl)
        if len(picks) >= n:
            return


def build_crawler_plan(problem_urls: pd.DataFrame) -> pd.DataFrame:
    picks: list[pd.Series] = []
    dyn_block = problem_urls[problem_urls["suspected_scrape_issue_type"].isin({"js_heavy_likely", "blocked_or_captcha"})]
    parse_empty = problem_urls[problem_urls["suspected_scrape_issue_type"].isin({"parse_failed", "redirect_or_tracking_url"})]
    ecommerce = problem_urls[
        problem_urls["normalized_url"].fillna("").str.contains(ECOM_SOCIAL_PAT)
        | problem_urls["page_type_final"].fillna("").isin({"product_marketplace_page", "price_package_page"})
    ]
    cited = problem_urls[pd.to_numeric(problem_urls["cited_rows_n"], errors="coerce").fillna(0).gt(0)]
    selected: list[tuple[str, pd.Series]] = []

    def select_bucket(frame: pd.DataFrame, bucket: str, limit: int) -> None:
        seen = {str(r.get("normalized_url")) for _, r in selected}
        count = 0
        for _, row in frame.iterrows():
            nurl = str(row.get("normalized_url") or "")
            if nurl in seen:
                continue
            selected.append((bucket, row))
            seen.add(nurl)
            count += 1
            if count >= limit:
                break

    select_bucket(dyn_block, "dynamic_js_likely_or_blocked", 10)
    select_bucket(parse_empty, "parse_failed_or_empty", 10)
    select_bucket(ecommerce, "ecommerce_social_product", 5)
    select_bucket(cited, "high_impact_cited", 5)
    if len(selected) < 30:
        select_bucket(problem_urls, "problematic_fill", 30 - len(selected))
    rows = []
    for i, (bucket, row) in enumerate(selected[:30], start=1):
        for crawler in ["playwright:adaptive", "playwright:firefox"]:
            rows.append(
                {
                    "benchmark_id": f"pw_retry_{i:03d}",
                    "selection_bucket": bucket,
                    "source_url": row.get("source_url"),
                    "normalized_url": row.get("normalized_url"),
                    "source_root_domain": row.get("source_root_domain"),
                    "current_word_count": row.get("word_count"),
                    "current_content_quality_flag": row.get("content_quality_flag"),
                    "current_page_type_final": row.get("page_type_final"),
                    "retry_crawler_type": crawler,
                    "planned_metrics": "scrape_success; parse_success; word_count; heading_count; page_text_excerpt quality; page_type result",
                    "dry_run_command": (
                        ".venv/bin/python scripts/v2_scrape_urls_apify.py "
                        "--queue data/econometrics_v2/scrape_queue/<single-url-queue.csv> "
                        f"--output-dir data/econometrics_v2/scrape_cache/raw_playwright_retry/{crawler.replace(':', '_')} "
                        f"--provider apify --cache true --crawler-type {crawler} --dry-run"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_summary(all_audit: pd.DataFrame, problem_urls: pd.DataFrame, domains: pd.DataFrame) -> dict[str, Any]:
    total = int(len(all_audit))
    prob = int(len(problem_urls))
    issues = problem_urls["suspected_scrape_issue_type"] if prob else pd.Series(dtype=object)
    flags = problem_urls["content_quality_flag"] if prob else pd.Series(dtype=object)
    return {
        "total_urls": total,
        "problematic_urls": prob,
        "problematic_rate": prob / total if total else 0,
        "dynamic_js_likely_count": int(issues.eq("js_heavy_likely").sum()),
        "parse_failed_count": int(issues.isin({"parse_failed", "redirect_or_tracking_url"}).sum()),
        "very_short_text_count": int(issues.eq("very_short_text").sum()),
        "blocked_or_error_page_count": int((flags.eq("blocked_or_error_page") | issues.eq("blocked_or_captcha")).sum()),
        "boilerplate_only_count": int(issues.isin({"boilerplate_only", "nav_footer_only"}).sum()),
        "unknown_with_bad_content_count": int((problem_urls["page_type_final"].fillna("").eq("unknown") & problem_urls["content_quality_flag"].fillna("").ne("ok")).sum()) if prob else 0,
        "unknown_with_ok_content_count": int((problem_urls["page_type_final"].fillna("").eq("unknown") & problem_urls["content_quality_flag"].fillna("").eq("ok")).sum()) if prob else 0,
        "urls_recommended_for_playwright_adaptive": int(problem_urls["recommended_next_crawler"].eq("playwright:adaptive").sum()) if prob else 0,
        "urls_recommended_for_playwright_firefox": int(problem_urls["recommended_next_crawler"].eq("playwright:firefox").sum()) if prob else 0,
        "urls_recommended_for_serper_metadata": int(problem_urls["recommended_next_crawler"].eq("serper_metadata_only").sum()) if prob else 0,
        "urls_likely_classifier_issue_not_scraper": int(issues.eq("classifier_issue_not_scraper").sum()),
        "top_problematic_domains": domains.head(15).to_dict("records") if len(domains) else [],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrape-audit", default="outputs/econometrics_eda_v2/tables/scrape_quality_audit.csv")
    ap.add_argument("--unknown-sample", default="outputs/econometrics_eda_v2/tables/unknown_scrape_review_sample.csv")
    ap.add_argument("--out-dir", default="outputs/econometrics_eda_v2/tables")
    ap.add_argument("--retry-queue-dir", default="data/econometrics_v2/scrape_queue")
    ap.add_argument("--top-n-retry", type=int, default=50)
    args = ap.parse_args(argv)
    ensure_v2_dirs()
    audit_path = Path(args.scrape_audit)
    if not audit_path.exists():
        for root in [Path("outputs"), Path("data/econometrics_v2"), Path("/mnt/data")]:
            matches = sorted(root.glob("**/scrape_quality_audit.csv")) if root.exists() else []
            if matches:
                audit_path = matches[0]
                break
    if not audit_path.exists():
        raise FileNotFoundError(f"scrape_quality_audit.csv not found: {args.scrape_audit}")

    out_dir = Path(args.out_dir)
    retry_dir = Path(args.retry_queue_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retry_dir.mkdir(parents=True, exist_ok=True)
    audit = read_csv(audit_path)
    problem_urls = build_problem_urls(audit)
    domains = build_domain_summary(audit, problem_urls)
    retry, adaptive, firefox = build_retry_queues(problem_urls, args.top_n_retry)
    plan = build_crawler_plan(problem_urls)
    summary = build_summary(audit, problem_urls, domains)

    write_csv(out_dir / "problematic_scrape_urls_prioritized.csv", problem_urls)
    write_csv(out_dir / "problematic_scrape_domains.csv", domains)
    write_csv(out_dir / "crawler_type_comparison_plan_updated.csv", plan)
    write_csv(retry_dir / "playwright_retry_queue.csv", retry)
    write_csv(retry_dir / "playwright_adaptive_retry_queue.csv", adaptive)
    write_csv(retry_dir / "playwright_firefox_retry_queue.csv", firefox)
    write_json(out_dir / "problematic_scrape_summary.json", summary)

    print("Top 30 problematic URLs:")
    cols = ["priority_rank", "priority_bucket", "source_root_domain", "word_count", "content_quality_flag", "suspected_scrape_issue_type", "recommended_next_crawler", "source_url"]
    print(problem_urls[cols].head(30).to_string(index=False))
    print(
        "Problematic scrape URL audit complete: "
        f"total={summary['total_urls']} problematic={summary['problematic_urls']} "
        f"adaptive={summary['urls_recommended_for_playwright_adaptive']} "
        f"firefox={summary['urls_recommended_for_playwright_firefox']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
