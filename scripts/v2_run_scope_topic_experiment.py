#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional runtime convenience
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(ROOT / ".env")

from src import brightdata
from src.econometrics_eda_v2.apify_scraper import DEFAULT_ACTOR_ID, scrape_queue_with_apify
from src.econometrics_eda_v2.feature_extraction import extract_page_features
from src.econometrics_eda_v2.normalize_sources import build_source_rows_from_files, stable_hash
from src.econometrics_eda_v2.page_type_classifier import PAGE_TYPE_FAMILY
from src.econometrics_eda_v2.parse_pages import parse_scrape_dir
from src.econometrics_eda_v2.scrape_quality_audit import _excerpt, content_quality_flag
from src.econometrics_eda_v2.url_features import build_source_url_features
from src.source_type import classify
from src.url_utils import root_domain
from scripts.v2_apply_scope_real_estate_taxonomy import run as run_scope_real_estate_taxonomy
from scripts.v2_run_scope_post_scrape_eda import run as run_scope_post_scrape_eda


TOPIC_NAME = "scope_condo_nonbranded"
BASE = Path("data/econometrics_v2/topics") / TOPIC_NAME
RAW_INPUT = BASE / "raw_inputs"
PROCESSED = BASE / "processed"
QUEUE_DIR = BASE / "scrape_queue"
APIFY_CACHE = BASE / "scrape_cache" / "apify"
BRIGHTDATA_CACHE = BASE / "scrape_cache" / "brightdata"
OUT = Path("outputs/econometrics_eda_v2/topic_sensitivity") / TOPIC_NAME
TABLES = OUT / "tables"
FIGURES = OUT / "figures"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), ensure_ascii=False, indent=2, default=str), "utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _ensure_dirs() -> None:
    for path in [RAW_INPUT, PROCESSED, QUEUE_DIR, APIFY_CACHE, BRIGHTDATA_CACHE, BRIGHTDATA_CACHE / "raw", BRIGHTDATA_CACHE / "parsed", TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def _copy_inputs(ai_json: Path, manifest: Path, brightdata_input: Path) -> dict[str, str]:
    targets = {
        "ai_json": RAW_INPUT / "scope_condo_ai_search.json",
        "manifest": RAW_INPUT / "scope_condo_manifest.csv",
        "brightdata_input": RAW_INPUT / "scope_condo_brightdata_input.csv",
    }
    shutil.copy2(ai_json, targets["ai_json"])
    shutil.copy2(manifest, targets["manifest"])
    shutil.copy2(brightdata_input, targets["brightdata_input"])
    return {k: str(v) for k, v in targets.items()}


def _col(df: pd.DataFrame, names: list[str], default: Any = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _bool_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.casefold().isin({"true", "1", "yes", "y"})


def validate_inputs(ai_path: Path, manifest_path: Path, brightdata_input_path: Path) -> dict[str, Any]:
    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    brightdata_input_df = pd.read_csv(brightdata_input_path, low_memory=False)
    manifest_obj = brightdata.parse_manifest(manifest_path.read_text("utf-8"), manifest_path.name)
    run = brightdata.parse_run(ai_path.read_text("utf-8"), ai_path.name)
    brightdata.apply_manifest(run, manifest_obj)
    prompt_id = _col(manifest_df, ["prompt_id", "id"])
    brand = _col(manifest_df, ["brand", "client_brand", "db_brand_slug"], "")
    topic = _col(manifest_df, ["topic"], "")
    nonbranded = _col(manifest_df, ["prompt_is_nonbranded"], "")
    language = _col(manifest_df, ["language", "prompt_language"], "")
    required = ["prompt_id", "topic", "brand", "db_brand_slug", "language", "intent", "prompt_hash"]
    return {
        "ai_json_path": str(ai_path),
        "manifest_path": str(manifest_path),
        "brightdata_input_path": str(brightdata_input_path),
        "manifest_rows": int(len(manifest_df)),
        "manifest_columns": list(manifest_df.columns),
        "brightdata_input_rows": int(len(brightdata_input_df)),
        "brightdata_input_columns": list(brightdata_input_df.columns),
        "prompt_id_exists": any(c in manifest_df.columns for c in ["prompt_id", "id"]),
        "prompt_id_unique": bool(prompt_id.astype(str).is_unique) if len(manifest_df) else False,
        "prompt_ids_n": int(prompt_id.replace("", pd.NA).nunique(dropna=True)) if len(manifest_df) else 0,
        "ai_records": int(run.get("n_records", 0)),
        "ai_sources": int(run.get("n_sources", 0)),
        "ai_cited": int(run.get("n_cited", 0)),
        "ai_more_only": int(run.get("n_more_only", 0)),
        "manifest_match": run.get("manifest", {}),
        "ai_search_output_can_match_manifest": bool((run.get("manifest") or {}).get("matched", 0) > 0),
        "brand_values": sorted(brand.dropna().astype(str).str.lower().unique().tolist())[:20],
        "brand_is_scope_like": bool(brand.dropna().astype(str).str.lower().str.contains("scope").any()),
        "prompt_is_nonbranded_all_true": bool(_bool_series(nonbranded).all()) if "prompt_is_nonbranded" in manifest_df.columns else None,
        "topic_values": sorted(topic.dropna().astype(str).unique().tolist())[:20],
        "topic_scope_condo_like": bool(topic.dropna().astype(str).str.lower().str.contains("scope|condo|condominium|bangkok").any()),
        "language_values": sorted(language.dropna().astype(str).unique().tolist())[:20],
        "required_fields_present": {field: field in manifest_df.columns for field in required},
        "warnings": list(run.get("warnings", [])) + list(manifest_obj.get("warnings", [])),
    }


def add_manifest_extras(source_rows: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    work = source_rows.copy()
    keep_cols = [c for c in manifest_df.columns if c not in {"prompt"}]
    if "prompt_id" in manifest_df.columns and "prompt_id" in work.columns:
        work = work.merge(manifest_df[keep_cols].drop_duplicates("prompt_id"), on="prompt_id", how="left", suffixes=("", "_manifest"))
    work["topic_name"] = TOPIC_NAME
    work["brand"] = _col(work, ["brand", "brand_manifest", "client_brand", "db_brand_slug"], "scope").replace("", "scope")
    work["db_brand_slug"] = _col(work, ["db_brand_slug", "db_brand_slug_manifest"], "scope").replace("", "scope")
    work["keyword"] = _col(work, ["keyword", "prompt_text", "prompt"], "")
    if "prompt_is_nonbranded" not in work.columns:
        work["prompt_is_nonbranded"] = True
    return work


def enrich_source_rows(source_rows: pd.DataFrame, manifest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = add_manifest_extras(source_rows, manifest_path)
    url_features, _summary = build_source_url_features(source_rows)
    enriched = source_rows.merge(url_features.drop_duplicates("source_row_id"), on=["source_row_id", "normalized_url", "source_url"], how="left")
    return source_rows, enriched


def _domain_category(domain: str, source_type: str = "") -> str:
    d = str(domain or "").casefold()
    if source_type == "government" or d.endswith(".go.th") or ".go." in d:
        return "government"
    if source_type in {"forum", "social", "video"} or any(x in d for x in ["reddit", "pantip", "facebook", "youtube", "tiktok", "instagram", "x.com"]):
        return "social_forum"
    if any(x in d for x in ["ddproperty", "fazwaz", "hipflat", "propertyhub", "livinginsider", "dotproperty", "condothai", "renthub", "baania"]):
        return "real_estate_listing"
    if any(x in d for x in ["scope", "scasset", "sansiri", "ananda", "apthai", "landandhouses", "lh.co.th", "developer"]):
        return "developer_official"
    if source_type in {"news", "blog", "review"} or any(x in d for x in ["review", "blog", "thinkofliving", "condonewb", "livingpop", "timeout", "timeout"]):
        return "review_media"
    if source_type == "ecommerce":
        return "marketplace_listing"
    return "other"


def source_mix_audit(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["domain_category"] = work.apply(lambda r: _domain_category(r.get("source_root_domain"), r.get("source_type_url")), axis=1)
    cited = pd.to_numeric(work["cited"], errors="coerce").fillna(0)
    metrics: list[dict[str, Any]] = []

    def add(metric: str, value: Any, detail: str = "") -> None:
        metrics.append({"metric": metric, "value": value, "detail": detail})

    add("total_rows", int(len(work)))
    add("unique_urls", int(work["normalized_url"].replace("", pd.NA).nunique(dropna=True)))
    add("cited_rows", int(cited.sum()))
    add("more_only_rows", int((cited == 0).sum()))
    add("cited_rate", float(cited.mean()) if len(cited) else 0.0)
    add("top_domains", json.dumps(work["source_root_domain"].value_counts().head(20).to_dict(), ensure_ascii=False))
    add("source_type_distribution", json.dumps(work["source_type_url"].value_counts(dropna=False).to_dict(), ensure_ascii=False))
    add("expected_source_types_distribution", json.dumps(work["expected_source_types"].fillna("").value_counts().head(20).to_dict(), ensure_ascii=False))
    for category in ["developer_official", "real_estate_listing", "review_media", "marketplace_listing", "government", "social_forum"]:
        add(f"{category}_rate", float(work["domain_category"].eq(category).mean()) if len(work) else 0.0)
    return pd.DataFrame(metrics)


def build_scrape_queue(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for nurl, group in rows[rows["normalized_url"].fillna("").astype(str).str.strip() != ""].groupby("normalized_url", dropna=False):
        cited_n = int(pd.to_numeric(group["cited"], errors="coerce").fillna(0).sum())
        reason = "high_impact_cited" if cited_n else "topic_unique_url"
        domains = group["source_root_domain"].dropna().astype(str)
        domains = domains[domains.str.strip() != ""] if len(domains) else domains
        scrape_id = "s_" + stable_hash(nurl, n=20)
        grouped.append(
            {
                "scrape_id": scrape_id,
                "source_url": group["source_url"].dropna().astype(str).iloc[0],
                "source_url_example": group["source_url"].dropna().astype(str).iloc[0],
                "normalized_url": nurl,
                "source_root_domain": domains.iloc[0] if len(domains) else root_domain(nurl),
                "topic_name": TOPIC_NAME,
                "cited_rows_n": cited_n,
                "source_rows_n": int(len(group)),
                "reason_selected": reason,
                "cache_path": str(APIFY_CACHE / f"{scrape_id}.json"),
                "scrape_status": "pending",
                "should_scrape": True,
            }
        )
    return pd.DataFrame(grouped).sort_values(["cited_rows_n", "source_rows_n"], ascending=[False, False])


def _final_page_type(seed: Any, scraped: Any, scraped_conf: Any) -> tuple[str, str, str]:
    seed_s = "" if pd.isna(seed) else str(seed)
    scraped_s = "" if pd.isna(scraped) else str(scraped)
    conf = "" if pd.isna(scraped_conf) else str(scraped_conf).casefold()
    if scraped_s and scraped_s != "unknown" and conf in {"medium", "high"}:
        return scraped_s, PAGE_TYPE_FAMILY.get(scraped_s, "unknown"), "scraped_content"
    final = seed_s or "unknown"
    return final, PAGE_TYPE_FAMILY.get(final, "unknown"), "url_seed"


def build_scrape_quality_audit(queue: pd.DataFrame, parsed: pd.DataFrame, features: pd.DataFrame, url_features: pd.DataFrame) -> pd.DataFrame:
    p = parsed.copy()
    if "normalized_url" not in p.columns:
        p["normalized_url"] = pd.Series(dtype=object)
    if "requested_normalized_url" in p.columns:
        requested_key = p["requested_normalized_url"].fillna("").astype(str).str.strip()
        p.loc[requested_key != "", "normalized_url"] = requested_key[requested_key != ""]
    if "page_text" not in p.columns:
        p["page_text"] = ""
    p["content_quality_flag"] = p.apply(lambda r: content_quality_flag({**r.to_dict(), "raw_cache_exists": True}), axis=1)
    p["page_text_excerpt"] = p["page_text"].map(_excerpt) if "page_text" in p.columns else ""
    features = features.copy()
    if "normalized_url" not in features.columns:
        features["normalized_url"] = pd.Series(dtype=object)
    url_features = url_features.copy()
    if "normalized_url" not in url_features.columns:
        url_features["normalized_url"] = pd.Series(dtype=object)
    fcols = ["normalized_url", "page_type_scraped_enriched", "page_type_scraped_confidence", "page_type_family_scraped", "content_feature_available"]
    ucols = ["normalized_url", "page_type_url_seed", "source_type_url", "official_source", "institutional_official"]
    p_one = p.drop_duplicates("normalized_url") if "normalized_url" in p.columns else p
    out = queue.merge(p_one, on="normalized_url", how="left", suffixes=("", "_parse"))
    out = out.merge(features[[c for c in fcols if c in features.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")
    out = out.merge(url_features[[c for c in ucols if c in url_features.columns]].drop_duplicates("normalized_url"), on="normalized_url", how="left")
    finals = out.apply(lambda r: _final_page_type(r.get("page_type_url_seed"), r.get("page_type_scraped_enriched"), r.get("page_type_scraped_confidence")), axis=1)
    out["page_type_final"] = [x[0] for x in finals]
    out["page_type_family"] = [x[1] for x in finals]
    out["page_type_final_source"] = [x[2] for x in finals]
    keep = [
        "source_url", "normalized_url", "source_root_domain", "scrape_success", "parse_success",
        "scraped_body_available", "word_count", "text_char_count", "heading_count", "table_count",
        "link_count", "content_quality_flag", "page_title", "page_text_excerpt", "page_type_url_seed",
        "page_type_scraped_enriched", "page_type_final", "page_type_final_source", "page_type_family",
        "source_type_url", "official_source", "institutional_official", "cited_rows_n", "source_rows_n",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    out["content_quality_flag"] = out["content_quality_flag"].astype(object)
    no_parse = out["parse_success"].isna()
    out.loc[no_parse, "content_quality_flag"] = "no_raw_cache"
    for col in ["scrape_success", "parse_success", "scraped_body_available"]:
        out[col] = out[col].fillna(False)
    return out[keep]


def topic_metrics(rows: pd.DataFrame, audit: pd.DataFrame, label: str) -> dict[str, Any]:
    cited = pd.to_numeric(rows.get("cited", pd.Series(dtype=float)), errors="coerce").fillna(0) if len(rows) else pd.Series(dtype=float)
    government = rows.get("source_type_url", pd.Series(dtype=object)).eq("government") if len(rows) else pd.Series(dtype=bool)
    official = rows.get("official_source", pd.Series(dtype=object)).fillna(False).astype(bool) if len(rows) else pd.Series(dtype=bool)
    commercial = rows.get("source_type_url", pd.Series(dtype=object)).isin(["ecommerce", "review", "blog"]) | rows.get("page_type_url_seed", pd.Series(dtype=object)).isin(["product_marketplace_page", "price_package_page"])
    usable = audit["content_quality_flag"].eq("ok") & (pd.to_numeric(audit["word_count"], errors="coerce").fillna(0) >= 300) if len(audit) else pd.Series(dtype=bool)
    return {
        "topic": label,
        "total_rows": int(len(rows)),
        "unique_urls": int(rows["normalized_url"].replace("", pd.NA).nunique(dropna=True)) if len(rows) else 0,
        "cited_rate": float(cited.mean()) if len(cited) else 0.0,
        "government_source_rate": float(government.mean()) if len(government) else 0.0,
        "official_source_rate": float(official.mean()) if len(official) else 0.0,
        "real_estate_commercial_source_rate": float(commercial.mean()) if len(commercial) else 0.0,
        "scraped_body_available_rate": float(audit["scraped_body_available"].fillna(False).astype(bool).mean()) if len(audit) else 0.0,
        "parse_success_rate": float(audit["parse_success"].fillna(False).astype(bool).mean()) if len(audit) else 0.0,
        "usable_content_rate": float(usable.mean()) if len(usable) else 0.0,
        "median_word_count": float(pd.to_numeric(audit["word_count"], errors="coerce").fillna(0).median()) if len(audit) else 0.0,
        "dynamic_js_likely_rate": float(audit["content_quality_flag"].eq("dynamic_js_likely").mean()) if len(audit) else 0.0,
        "parse_failed_rate": float(audit["content_quality_flag"].eq("parse_failed").mean()) if len(audit) else 0.0,
        "blocked_or_error_page_rate": float(audit["content_quality_flag"].eq("blocked_or_error_page").mean()) if len(audit) else 0.0,
        "page_type_unknown_rate": float(audit["page_type_final"].fillna("unknown").eq("unknown").mean()) if len(audit) else 0.0,
        "top_10_domains": json.dumps(rows["source_root_domain"].value_counts().head(10).to_dict(), ensure_ascii=False) if len(rows) else "{}",
    }


def build_problematic_brightdata_queue(audit: pd.DataFrame) -> pd.DataFrame:
    flags = {"dynamic_js_likely", "parse_failed", "empty_text", "very_short_text", "blocked_or_error_page"}
    work = audit.copy()
    work["_bad"] = work["content_quality_flag"].isin(flags)
    work["_cited"] = pd.to_numeric(work["cited_rows_n"], errors="coerce").fillna(0)
    sub = work[work["_bad"] | ((work["_cited"] > 0) & ~work["content_quality_flag"].eq("ok"))].sort_values(["_cited", "source_rows_n"], ascending=[False, False])
    cols = ["source_url", "normalized_url", "source_root_domain", "cited_rows_n", "source_rows_n", "content_quality_flag", "word_count", "page_type_final"]
    return sub[[c for c in cols if c in sub.columns]].drop_duplicates("normalized_url").head(40)


def write_report(validation: dict[str, Any], rows: pd.DataFrame, audit: pd.DataFrame, comparison: pd.DataFrame, bd_queue: pd.DataFrame) -> None:
    top_domains = rows["source_root_domain"].value_counts().head(10)
    source_types = rows["source_type_url"].value_counts().head(10)
    page_types = audit["page_type_final"].value_counts(dropna=False).head(10)
    problematic = audit[~audit["content_quality_flag"].eq("ok")]["source_root_domain"].value_counts().head(10)
    scope = comparison[comparison["topic"].eq("scope_condo_nonbranded")].iloc[0].to_dict()
    siriraj = comparison[comparison["topic"].eq("siriraj_existing")].iloc[0].to_dict() if comparison["topic"].eq("siriraj_existing").any() else {}
    scrape_summary_path = PROCESSED / "apify_scrape_run_summary.json"
    scrape_summary = json.loads(scrape_summary_path.read_text("utf-8")) if scrape_summary_path.exists() else {}
    scrape_success_rate = (
        float(scrape_summary.get("urls_success", 0)) / float(scrape_summary.get("urls_total", 1))
        if scrape_summary.get("urls_total") else 0.0
    )
    lines = [
        "# SCOPE Condo Topic Sensitivity Diagnosis",
        "",
        f"Prompts loaded: {validation.get('manifest_rows')}",
        f"Source rows parsed: {len(rows)}",
        f"Unique URLs: {rows['normalized_url'].nunique()}",
        f"Cited rows: {int(pd.to_numeric(rows['cited'], errors='coerce').fillna(0).sum())}",
        f"More-only rows: {int((pd.to_numeric(rows['cited'], errors='coerce').fillna(0) == 0).sum())}",
        "",
        "## Main Findings",
        f"- SCOPE government source rate: {scope.get('government_source_rate', 0):.1%}",
        f"- Siriraj government source rate: {siriraj.get('government_source_rate', 0):.1%}" if siriraj else "- Siriraj comparison unavailable.",
        f"- SCOPE Apify scrape success rate: {scrape_success_rate:.1%}",
        f"- SCOPE parsed body available rate: {scope.get('parse_success_rate', 0):.1%}",
        f"- SCOPE Apify usable content rate: {scope.get('usable_content_rate', 0):.1%}",
        f"- Siriraj usable content rate: {siriraj.get('usable_content_rate', 0):.1%}" if siriraj else "- Siriraj usable content unavailable.",
        f"- SCOPE page-type unknown rate: {scope.get('page_type_unknown_rate', 0):.1%}",
        f"- Apify run status: {scrape_summary.get('actor_status', 'not recorded')} ({scrape_summary.get('urls_success', 0)} success / {scrape_summary.get('urls_total', 0)} URLs).",
        "",
        "## Top Domains",
        top_domains.to_string(),
        "",
        "## Source Types",
        source_types.to_string(),
        "",
        "## Page Types",
        page_types.to_string(),
        "",
        "Note: page-type labels are reused from the original CiteScope classifier vocabulary; labels such as article_health_info mean article-like content and should be renamed before final condo-facing reporting.",
        "",
        "## Problematic Domains",
        problematic.to_string(),
        "",
        "## Bright Data Fallback",
        f"Problematic Bright Data benchmark queue rows prepared: {len(bd_queue)}. Bright Data was not run live by this script.",
        "",
        "## Recommendation",
        "Use Apify first for the SCOPE topic, then benchmark Bright Data only on the prepared problematic subset if more body text is needed. Treat all findings as observable scrape/source-mix patterns, not claims about hidden retrieval or rejection.",
    ]
    (OUT / "topic_diagnosis_report.md").write_text("\n".join(lines), "utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_dirs()
    copied = _copy_inputs(Path(args.ai_json), Path(args.manifest), Path(args.brightdata_input))
    ai_path = Path(copied["ai_json"])
    manifest_path = Path(copied["manifest"])
    validation = validate_inputs(ai_path, manifest_path, Path(copied["brightdata_input"]))
    write_json(TABLES / "input_validation_report.json", validation)

    source_rows, source_summary = build_source_rows_from_files(ai_path, manifest_path)
    source_rows, enriched = enrich_source_rows(source_rows, manifest_path)
    write_csv(TABLES / "scope_condo_sources_raw.csv", source_rows)
    write_csv(TABLES / "scope_condo_sources_flattened.csv", enriched)
    write_csv(TABLES / "scope_condo_sources_with_manifest.csv", enriched)
    write_json(PROCESSED / "source_rows_summary.json", source_summary)

    mix = source_mix_audit(enriched)
    write_csv(TABLES / "scope_condo_source_mix_audit.csv", mix)
    queue = build_scrape_queue(enriched)
    queue_path = QUEUE_DIR / "apify_scope_condo_scrape_queue.csv"
    write_csv(queue_path, queue)

    if not args.skip_apify:
        summary = scrape_queue_with_apify(
            queue,
            APIFY_CACHE,
            actor_id=args.actor_id,
            max_urls=args.max_urls,
            dry_run=args.dry_run,
            force_rescrape=args.force_rescrape,
        )
        write_json(PROCESSED / "apify_scrape_run_summary.json", summary)
    parsed, parse_summary = parse_scrape_dir(APIFY_CACHE)
    parsed_path = PROCESSED / "apify_page_parse_rows.csv"
    write_csv(parsed_path, parsed)
    write_json(PROCESSED / "apify_page_parse_summary.json", parse_summary)
    features, feature_summary = extract_page_features(parsed, enriched)
    features_path = PROCESSED / "apify_page_features.csv"
    write_csv(features_path, features)
    write_json(PROCESSED / "apify_page_feature_summary.json", feature_summary)
    url_features, _ = build_source_url_features(enriched)
    audit = build_scrape_quality_audit(queue, parsed, features, url_features)
    write_csv(TABLES / "scope_condo_scrape_quality_audit.csv", audit)

    bd_queue = build_problematic_brightdata_queue(audit)
    write_csv(QUEUE_DIR / "brightdata_scope_condo_benchmark_queue.csv", bd_queue)

    siriraj_audit_path = Path("outputs/econometrics_eda_v2/tables/scrape_quality_audit.csv")
    siriraj_rows_path = Path("data/econometrics_v2/exports/econometrics_row_level_sources.csv")
    comparison_rows = [topic_metrics(enriched, audit, TOPIC_NAME)]
    if siriraj_audit_path.exists() and siriraj_rows_path.exists():
        siriraj_audit = pd.read_csv(siriraj_audit_path, low_memory=False)
        siriraj_rows = pd.read_csv(siriraj_rows_path, low_memory=False)
        comparison_rows.append(topic_metrics(siriraj_rows, siriraj_audit, "siriraj_existing"))
    comparison = pd.DataFrame(comparison_rows)
    write_csv(Path("outputs/econometrics_eda_v2/topic_sensitivity/topic_comparison_scope_vs_siriraj.csv"), comparison)
    write_report(validation, enriched, audit, comparison, bd_queue)
    taxonomy_summary = run_scope_real_estate_taxonomy()
    post_scrape_eda_summary = run_scope_post_scrape_eda()
    return {
        "validation": validation,
        "source_rows": int(len(enriched)),
        "unique_urls": int(queue["normalized_url"].nunique()),
        "cited_rows": int(pd.to_numeric(enriched["cited"], errors="coerce").fillna(0).sum()),
        "more_only_rows": int((pd.to_numeric(enriched["cited"], errors="coerce").fillna(0) == 0).sum()),
        "top_domains": enriched["source_root_domain"].value_counts().head(10).to_dict(),
        "source_type_distribution": enriched["source_type_url"].value_counts().to_dict(),
        "apify_scrape_success_rate": float(audit["scrape_success"].fillna(False).astype(bool).mean()) if len(audit) else 0.0,
        "apify_usable_content_rate": float((audit["content_quality_flag"].eq("ok") & (pd.to_numeric(audit["word_count"], errors="coerce").fillna(0) >= 300)).mean()) if len(audit) else 0.0,
        "page_type_unknown_rate": float(audit["page_type_final"].fillna("unknown").eq("unknown").mean()) if len(audit) else 0.0,
        "brightdata_benchmark_run": False,
        "real_estate_taxonomy": taxonomy_summary,
        "post_scrape_eda": post_scrape_eda_summary,
        "outputs_dir": str(OUT),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-json", default="/Users/pmootr/Downloads/sd_mreoachd2kf31omlko.json")
    ap.add_argument("--manifest", default="/Users/pmootr/Downloads/scope_condo_nonbranded_prompt_manifest_100.csv")
    ap.add_argument("--brightdata-input", default="/Users/pmootr/Downloads/scope_condo_nonbranded_brightdata_input_100.csv")
    ap.add_argument("--actor-id", default=DEFAULT_ACTOR_ID)
    ap.add_argument("--max-urls", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-apify", action="store_true")
    ap.add_argument("--force-rescrape", action="store_true")
    args = ap.parse_args(argv)
    result = run(args)
    write_json(TABLES / "scope_condo_experiment_run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
