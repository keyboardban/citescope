from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.io import OUTPUT_DIR, PLOTS_DIR, QUEUE_DIR, TABLES_DIR, utc_now_iso, write_json
from src.econometrics_eda_v2.leakage import DIAGNOSTIC_ONLY, LEAKAGE_EXCLUSIONS
from src.econometrics_eda_v2.lpm_prep import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTENT_BINARY_FEATURES,
    NUMERIC_FEATURES,
    build_lpm_readiness,
)
from src.econometrics_eda_v2.page_type_classifier import page_type_family
from src.econometrics_eda_v2.url_features import source_domain_raw_looks_like_label
from src.url_utils import root_domain

FEATURE_GROUPS = {
    "identity/raw columns": [
        "run_id", "record_id", "prompt_id", "source_row_id", "normalized_url", "source_url",
        "source_title", "source_description", "source_snippet", "answer_text", "page_text",
    ],
    "scrape status": ["scrape_success", "parse_success", "scraped_body_available", "content_feature_available", "content_feature_missing_reason"],
    "content flags": [c for c in BINARY_FEATURES if c.startswith("has_")],
    "page structure": ["word_count", "heading_count", "table_count", "link_count"],
    "prompt relevance": ["title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity", "max_chunk_prompt_similarity", "relevance_score_prompt_only"],
    "safe predictor candidates": ["domain_plot_label", "source_root_domain", "source_type_url", "page_type_url_seed", "page_type_scraped_enriched", "page_type_final", "page_type_family", "page_type_final_source", "page_type_missing_reason", "intent", "topic", "language", "country", "url_length", "url_path_depth", "https_flag", "url_has_query_params", "domain_seen_count", "domain_seen_count_loo", "log1p_domain_seen_count"],
    "diagnostic-only": sorted(DIAGNOSTIC_ONLY | {"in_scrape_queue", "has_raw_apify_cache"}),
    "leakage excluded": sorted(LEAKAGE_EXCLUSIONS),
}

IDENTITY_RAW_COLUMNS = set(FEATURE_GROUPS["identity/raw columns"])


def _url_host(url: str) -> str:
    if not str(url or "").strip():
        return ""
    try:
        p = urlparse(str(url) if re.match(r"^[a-z][a-z0-9+.-]*://", str(url), flags=re.I) else "https://" + str(url))
        return (p.hostname or "").lower()
    except Exception:
        return ""


def text_similarity(a: str, b: str) -> float:
    toks_a = set(re.findall(r"[\w\u0E00-\u0E7F]+", str(a or "").casefold(), flags=re.U))
    toks_b = set(re.findall(r"[\w\u0E00-\u0E7F]+", str(b or "").casefold(), flags=re.U))
    if not toks_a or not toks_b:
        return math.nan
    return len(toks_a & toks_b) / len(toks_a | toks_b)


def max_chunk_similarity(prompt: str, page_text: str, chunk_words: int = 120) -> float:
    words = re.findall(r"[\w\u0E00-\u0E7F]+", str(page_text or ""), flags=re.U)
    if not words:
        return math.nan
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    scores = [text_similarity(prompt, c) for c in chunks[:80]]
    scores = [s for s in scores if not pd.isna(s)]
    return max(scores) if scores else math.nan


def _dedup_by_url(df: pd.DataFrame, url_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    frames = []
    for col in url_cols:
        if col in df.columns:
            tmp = df.copy()
            tmp["join_normalized_url"] = tmp[col]
            frames.append(tmp[tmp["join_normalized_url"].notna() & (tmp["join_normalized_url"].astype(str) != "")])
    if not frames:
        return df.assign(join_normalized_url=df.get("normalized_url", ""))
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates("join_normalized_url", keep="first")


def export_econometrics_rows(
    source_rows: pd.DataFrame,
    source_url_features: pd.DataFrame | None,
    page_parse: pd.DataFrame,
    page_features: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    df = source_rows.copy()
    def col(name: str, default="") -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)
    if source_url_features is None:
        source_url_features = pd.DataFrame()
    url_feature_cols = [
        "source_row_id", "source_domain_ai_label", "source_domain_host", "source_root_domain",
        "domain_plot_label", "domain_raw_looks_like_label", "source_type_url", "institutional_official", "official_source",
        "page_type_url_seed", "page_type_url_seed_source", "page_type_url_seed_confidence",
        "page_type_url_seed_evidence", "url_length", "url_path_depth", "https_flag",
        "url_has_query_params", "domain_seen_count", "domain_seen_count_loo",
        "log1p_domain_seen_count",
    ]
    if not source_url_features.empty:
        df = df.merge(source_url_features[[c for c in url_feature_cols if c in source_url_features.columns]], on="source_row_id", how="left")

    queue_path = QUEUE_DIR / "scrape_queue.csv"
    queue_urls: set[str] = set()
    cache_paths: dict[str, str] = {}
    if queue_path.exists():
        q = pd.read_csv(queue_path, low_memory=False)
        queue_urls = set(q["normalized_url"].dropna().astype(str)) if "normalized_url" in q.columns else set()
        if {"normalized_url", "cache_path"}.issubset(q.columns):
            cache_paths = dict(zip(q["normalized_url"].astype(str), q["cache_path"].astype(str)))
    df["in_scrape_queue"] = df["normalized_url"].astype(str).isin(queue_urls) if queue_urls else False
    df["has_raw_apify_cache"] = df["normalized_url"].astype(str).map(lambda u: Path(cache_paths.get(u, "")).exists() if cache_paths else False)

    parse_cols = [
        "join_normalized_url", "scrape_success", "parse_success", "scraped_body_available",
        "word_count", "heading_count", "table_count", "link_count", "page_text",
        "page_title", "meta_description",
        "requested_normalized_url", "final_normalized_url",
    ]
    feat_cols = [
        "join_normalized_url", "has_faq", "has_price_or_package", "has_contact_info", "has_table",
        "has_bullets", "has_author", "has_reviewer", "has_schema", "has_phone_number",
        "has_email", "has_address", "has_opening_hours", "has_booking_or_appointment",
        "has_step_by_step", "has_medical_disclaimer", "has_references", "has_updated_date",
        "page_type_scraped_enriched", "page_type_scraped_confidence", "content_feature_available",
        "content_feature_missing_reason", "page_prompt_similarity", "max_chunk_prompt_similarity",
        "page_type_evidence", "page_type_score_map", "page_type_unknown_reason",
        "page_type_family_scraped", "h1_or_top_heading", "currency_count", "price_keyword_count",
    ]
    if not page_parse.empty:
        parse_join = _dedup_by_url(page_parse, ["requested_normalized_url", "final_normalized_url", "normalized_url"])
        df = df.merge(parse_join[[c for c in parse_cols if c in parse_join.columns]], left_on="normalized_url", right_on="join_normalized_url", how="left")
        df = df.drop(columns=[c for c in ["join_normalized_url"] if c in df.columns])
    else:
        for c in parse_cols:
            if c != "join_normalized_url":
                df[c] = np.nan
    if not page_features.empty:
        feat_join = _dedup_by_url(page_features, ["requested_normalized_url", "final_normalized_url", "normalized_url"])
        df = df.merge(feat_join[[c for c in feat_cols if c in feat_join.columns]], left_on="normalized_url", right_on="join_normalized_url", how="left")
        df = df.drop(columns=[c for c in ["join_normalized_url"] if c in df.columns])
    else:
        for c in feat_cols:
            if c != "join_normalized_url":
                df[c] = np.nan

    df["intent_plot_label"] = df.get("intent_plot_label", df.get("intent", "")).fillna("").replace("", "missing_intent")
    df["log1p_source_position"] = np.log1p(pd.to_numeric(df.get("source_position"), errors="coerce"))
    df["title_prompt_similarity"] = [text_similarity(p, t) for p, t in zip(col("prompt_text"), col("source_title"))]
    df["description_prompt_similarity"] = [
        text_similarity(p, " ".join([str(d or ""), str(s or "")]))
        for p, d, s in zip(col("prompt_text"), col("source_description"), col("source_snippet"))
    ]
    if "page_prompt_similarity" not in df.columns or df["page_prompt_similarity"].isna().all():
        df["page_prompt_similarity"] = [text_similarity(p, t) for p, t in zip(col("prompt_text"), col("page_text"))]
    if "max_chunk_prompt_similarity" not in df.columns or df["max_chunk_prompt_similarity"].isna().all():
        df["max_chunk_prompt_similarity"] = [max_chunk_similarity(p, t) for p, t in zip(col("prompt_text"), col("page_text"))]
    body_available = df.get("scraped_body_available", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    for sim_col in ["page_prompt_similarity", "max_chunk_prompt_similarity"]:
        if sim_col in df.columns:
            df.loc[~body_available, sim_col] = np.nan
    sim_any = df[["page_prompt_similarity", "max_chunk_prompt_similarity"]].notna().any(axis=1)
    df["page_similarity_missing_reason"] = np.where(
        body_available & ~sim_any,
        "similarity_unavailable",
        np.where(~body_available, "no_scraped_body", ""),
    )
    sim_cols = ["title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity", "max_chunk_prompt_similarity"]
    df["relevance_score_prompt_only"] = df[sim_cols].max(axis=1, skipna=True)
    if "domain_seen_count" not in df.columns or df["domain_seen_count"].isna().all():
        domain_counts = df.groupby("source_domain")["source_domain"].transform("size")
        df["domain_seen_count"] = domain_counts.astype(float)
        df["domain_seen_count_loo"] = (domain_counts - 1).astype(float)
        df["log1p_domain_seen_count"] = np.log1p(df["domain_seen_count"].clip(lower=0))
    if "url_length" not in df.columns:
        df["url_length"] = df["normalized_url"].fillna("").astype(str).str.len()
    if "url_path_depth" not in df.columns:
        df["url_path_depth"] = df["normalized_url"].fillna("").map(lambda u: len([p for p in urlparse(str(u)).path.split("/") if p]))
    if "https_flag" not in df.columns:
        df["https_flag"] = df["normalized_url"].fillna("").map(lambda u: int(str(u).startswith("https://")))
    if "url_has_query_params" not in df.columns:
        df["url_has_query_params"] = df["normalized_url"].fillna("").map(lambda u: int(bool(urlparse(str(u)).query)))

    if "source_domain_ai_label" not in df.columns:
        df["source_domain_ai_label"] = df.get("source_domain", pd.Series([""] * len(df), index=df.index))
    if "source_domain_host" not in df.columns:
        df["source_domain_host"] = df["source_url"].fillna(df["normalized_url"]).map(_url_host)
    if "source_root_domain" not in df.columns:
        df["source_root_domain"] = df["source_url"].fillna(df["normalized_url"]).map(lambda u: root_domain(str(u)) if str(u).strip() else "")
    if "domain_plot_label" not in df.columns:
        df["domain_plot_label"] = df["source_root_domain"].where(df["source_root_domain"].fillna("").astype(str).str.strip() != "", "(missing_url)")
    if "domain_raw_looks_like_label" not in df.columns:
        df["domain_raw_looks_like_label"] = df["source_domain_ai_label"].map(source_domain_raw_looks_like_label)

    scraped_pt = df.get("page_type_scraped_enriched", pd.Series([np.nan] * len(df), index=df.index))
    url_pt = df.get("page_type_url_seed", pd.Series([np.nan] * len(df), index=df.index))
    if "page_type_scraped_confidence" in df.columns:
        scraped_conf = df["page_type_scraped_confidence"].fillna("unknown").astype(str).str.lower()
    else:
        scraped_conf = pd.Series(
            np.where(
                scraped_pt.notna() & (scraped_pt.astype(str).str.strip() != "") & (scraped_pt.astype(str).str.lower() != "unknown"),
                "medium",
                "unknown",
            ),
            index=df.index,
        )
    use_scraped = (
        scraped_pt.notna()
        & (scraped_pt.astype(str).str.strip() != "")
        & (scraped_pt.astype(str).str.lower() != "unknown")
        & scraped_conf.isin(["medium", "high"])
    )
    df["page_type_final"] = scraped_pt.where(use_scraped, url_pt)
    df["page_type_confidence"] = df.get("page_type_scraped_confidence", pd.Series([np.nan] * len(df), index=df.index)).where(
        use_scraped,
        df.get("page_type_url_seed_confidence", pd.Series(["unknown"] * len(df), index=df.index)),
    )
    df["page_type_evidence"] = df.get("page_type_evidence", pd.Series([np.nan] * len(df), index=df.index)).where(
        use_scraped,
        df.get("page_type_url_seed_evidence", pd.Series([np.nan] * len(df), index=df.index)),
    )
    url_present = df["normalized_url"].fillna("").astype(str).str.strip() != ""
    final_present = df["page_type_final"].notna() & (df["page_type_final"].astype(str).str.strip() != "")
    df["page_type_final_source"] = np.where(use_scraped, "scraped_content", np.where(url_present & final_present, "url_seed", np.where(~url_present, "missing_no_url", "missing_no_page_type")))
    df["page_type_missing_reason"] = np.where(final_present, "", np.where(~url_present, "missing_no_url", "missing_no_page_type"))
    df["page_type_family"] = df["page_type_final"].fillna("unknown").map(page_type_family)
    if "page_type_unknown_reason" not in df.columns:
        df["page_type_unknown_reason"] = ""
    df["page_type_unknown_reason"] = np.where(
        df["page_type_final"].fillna("").astype(str).str.lower().eq("unknown") & (df["page_type_unknown_reason"].fillna("").astype(str).str.strip() == ""),
        np.where(df["scraped_body_available"].fillna(False).astype(bool), "no_url_path_signal", "no_scraped_body"),
        df["page_type_unknown_reason"].fillna(""),
    )
    df["unknown_reason"] = df["page_type_unknown_reason"]

    required = [
        "run_id", "record_id", "prompt_id", "source_row_id", "normalized_url", "source_url", "source_domain",
        "source_domain_ai_label", "source_domain_host", "source_root_domain", "domain_plot_label", "domain_raw_looks_like_label",
        "cited", "intent", "intent_plot_label", "topic", "language", "prompt_language", "country", "expected_source_types",
        "source_title", "source_description", "source_snippet", "source_position", "observed_rank", "log1p_source_position",
        "source_type_url", "institutional_official", "official_source", "page_type_url_seed",
        "page_type_scraped_enriched", "page_type_final", "page_type_family", "page_type_final_source",
        "page_type_confidence", "page_type_evidence", "page_type_score_map", "page_type_unknown_reason", "unknown_reason", "page_type_missing_reason",
        "in_scrape_queue", "has_raw_apify_cache", "scrape_success", "parse_success", "scraped_body_available", "content_feature_available",
        "content_feature_missing_reason", "page_similarity_missing_reason", "word_count", "heading_count", "table_count", "link_count",
        "page_title", "meta_description", "h1_or_top_heading", "currency_count", "price_keyword_count",
        "has_faq", "has_price_or_package", "has_contact_info", "has_table", "has_bullets",
        "has_author", "has_reviewer", "has_schema", "has_phone_number", "has_email", "has_address",
        "has_opening_hours", "has_booking_or_appointment", "has_step_by_step", "has_medical_disclaimer",
        "has_references", "has_updated_date", "page_type",
        "title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity",
        "max_chunk_prompt_similarity", "relevance_score_prompt_only", "domain_seen_count",
        "domain_seen_count_loo", "log1p_domain_seen_count", "url_length", "url_path_depth",
        "https_flag", "url_has_query_params", "cited_label", "is_more_only", "source_group", "source_origin",
    ]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan
    final = df[required + [c for c in df.columns if c not in required and c not in {"answer_text", "page_text"}]]
    cited = pd.to_numeric(final["cited"], errors="coerce").fillna(0)
    warnings = []
    if "source_origin" in final.columns:
        warnings.append("source_origin retained only as diagnostic metadata; exclude from safe predictors.")
    summary = {
        "rows_exported": int(len(final)),
        "cited_count": int(cited.sum()),
        "more_only_count": int((cited == 0).sum()),
        "cited_rate": float(cited.mean()) if len(cited) else 0.0,
        "unique_prompts": int(final["prompt_id"].replace("", pd.NA).nunique(dropna=True) or final["record_id"].nunique(dropna=True)),
        "unique_records": int(final["record_id"].nunique(dropna=True)),
        "unique_urls": int(final["normalized_url"].replace("", pd.NA).nunique(dropna=True)),
        "scrape_queue_unique_urls": int(len(queue_urls)),
        "rows_with_intent": int((final["intent_plot_label"] != "missing_intent").sum()),
        "rows_missing_intent": int((final["intent_plot_label"] == "missing_intent").sum()),
        "intent_missing_rate": float((final["intent_plot_label"] == "missing_intent").mean()) if len(final) else 0.0,
        "rows_with_scraped_body": int(final["scraped_body_available"].fillna(False).astype(bool).sum()) if len(final) else 0,
        "rows_with_word_count": int((pd.to_numeric(final["word_count"], errors="coerce") > 0).sum()) if len(final) else 0,
        "rows_with_page_type_url_seed": int(final["page_type_url_seed"].notna().sum()) if len(final) else 0,
        "rows_with_page_type_scraped_enriched": int(final["page_type_scraped_enriched"].notna().sum()) if len(final) else 0,
        "rows_with_page_type_final": int(final["page_type_final"].notna().sum()) if len(final) else 0,
        "rows_with_content_feature_available": int(final["content_feature_available"].fillna(False).astype(bool).sum()) if len(final) else 0,
        "scrape_join_rate": float(final["scrape_success"].notna().mean()) if len(final) else 0.0,
        "page_type_final_distribution": final["page_type_final"].value_counts(dropna=False).to_dict() if len(final) else {},
        "page_type_final_source_distribution": final["page_type_final_source"].value_counts(dropna=False).to_dict() if len(final) else {},
        "warnings": warnings,
    }
    return final, summary


def feature_group(feature: str) -> str:
    if feature in LEAKAGE_EXCLUSIONS:
        return "leakage excluded"
    if feature in DIAGNOSTIC_ONLY:
        return "diagnostic-only"
    if feature in IDENTITY_RAW_COLUMNS:
        return "identity/raw columns"
    for group, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return group
    if feature in BINARY_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        return "safe predictor candidates"
    return "identity/raw columns"


def feature_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        non_null = int(df[c].notna().sum())
        group = feature_group(c)
        rows.append(
            {
                "feature": c,
                "feature_group": group,
                "show_in_main_coverage": group not in {"identity/raw columns", "leakage excluded"},
                "non_null_count": non_null,
                "coverage": float(non_null / len(df)) if len(df) else 0.0,
                "missing_rate": float(df[c].isna().mean()) if len(df) else 0.0,
                "n_unique": int(df[c].dropna().nunique()) if len(df) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_group", "coverage"], ascending=[True, True])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def page_type_final_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = len(df)
    url_valid = df.get("normalized_url", pd.Series([""] * rows, index=df.index)).fillna("").astype(str).str.strip() != ""
    src_dist = df.get("page_type_final_source", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
    return pd.DataFrame(
        [
            {
                "rows": int(rows),
                "valid_url_rows": int(url_valid.sum()) if rows else 0,
                "rows_with_page_type_url_seed": int(df.get("page_type_url_seed", pd.Series(index=df.index, dtype=object)).notna().sum()),
                "rows_with_page_type_scraped_enriched": int(df.get("page_type_scraped_enriched", pd.Series(index=df.index, dtype=object)).notna().sum()),
                "rows_with_page_type_final": int(df.get("page_type_final", pd.Series(index=df.index, dtype=object)).notna().sum()),
                "page_type_url_seed_coverage": float(df.get("page_type_url_seed", pd.Series(index=df.index, dtype=object)).notna().mean()) if rows else 0.0,
                "page_type_url_seed_coverage_valid_url_rows": float(df.loc[url_valid, "page_type_url_seed"].notna().mean()) if rows and "page_type_url_seed" in df.columns and url_valid.any() else 0.0,
                "page_type_scraped_enriched_coverage": float(df.get("page_type_scraped_enriched", pd.Series(index=df.index, dtype=object)).notna().mean()) if rows else 0.0,
                "page_type_final_coverage": float(df.get("page_type_final", pd.Series(index=df.index, dtype=object)).notna().mean()) if rows else 0.0,
                "page_type_final_source_distribution": json.dumps(src_dist, sort_keys=True, ensure_ascii=False),
            }
        ]
    )


def page_type_funnel(df: pd.DataFrame) -> pd.DataFrame:
    rows = len(df)
    valid_url = df.get("normalized_url", pd.Series([""] * rows, index=df.index)).fillna("").astype(str).str.strip() != ""
    stages = [
        ("all_rows", pd.Series([True] * rows, index=df.index)),
        ("valid_url", valid_url),
        ("url_seed_measured", df.get("page_type_url_seed", pd.Series(index=df.index, dtype=object)).notna()),
        ("scraped_enriched_measured", df.get("page_type_scraped_enriched", pd.Series(index=df.index, dtype=object)).notna()),
        ("final_page_type_measured", df.get("page_type_final", pd.Series(index=df.index, dtype=object)).notna()),
    ]
    return pd.DataFrame(
        [{"stage": name, "rows": int(mask.sum()), "coverage": float(mask.mean()) if rows else 0.0} for name, mask in stages]
    )


def page_similarity_availability_audit(df: pd.DataFrame) -> pd.DataFrame:
    body = df.get("scraped_body_available", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    rows = []
    for feature in ["page_prompt_similarity", "max_chunk_prompt_similarity"]:
        if feature not in df.columns:
            continue
        non_null = df[feature].notna()
        rows.append(
            {
                "feature": feature,
                "non_null_count": int(non_null.sum()),
                "coverage": float(non_null.mean()) if len(df) else 0.0,
                "rows_with_scraped_body": int(body.sum()),
                "rows_without_scraped_body_but_similarity_non_null": int((~body & non_null).sum()),
            }
        )
    return pd.DataFrame(rows)


def domain_field_audit(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_url", "source_domain", "source_domain_ai_label", "source_domain_host",
        "source_root_domain", "domain_plot_label", "domain_raw_looks_like_label", "cited",
    ]
    out = pd.DataFrame({c: df[c] if c in df.columns else pd.Series([np.nan] * len(df), index=df.index) for c in cols})
    return out.rename(columns={"source_domain": "source_domain_raw"})


def domain_field_summary(df: pd.DataFrame) -> pd.DataFrame:
    raw_label = df.get("domain_raw_looks_like_label", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    root = df.get("source_root_domain", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    rows = [
        {"metric": "raw_domains_that_look_like_labels", "value": int(raw_label.sum())},
        {"metric": "url_derived_domains", "value": int((root.str.strip() != "").sum())},
    ]
    for label, count in df.loc[raw_label, "source_domain_ai_label"].value_counts().head(20).items() if "source_domain_ai_label" in df.columns else []:
        rows.append({"metric": "top_raw_label", "label": label, "value": int(count)})
    for label, count in root.value_counts().head(20).items():
        rows.append({"metric": "top_root_domain", "label": label, "value": int(count)})
    return pd.DataFrame(rows)


def price_package_page_audit(df: pd.DataFrame) -> pd.DataFrame:
    strong = (
        df.get("page_type_evidence", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.contains("strong_price_signal|pricing_table|multiple_independent", case=False, regex=True)
    )
    out = pd.DataFrame(
        {
            "source_url": df.get("source_url", pd.Series([np.nan] * len(df), index=df.index)),
            "title": df.get("page_title", df.get("source_title", pd.Series([np.nan] * len(df), index=df.index))),
            "h1_or_top_heading": df.get("h1_or_top_heading", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_url_seed": df.get("page_type_url_seed", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_scraped_enriched": df.get("page_type_scraped_enriched", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_final": df.get("page_type_final", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_family": df.get("page_type_family", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_confidence": df.get("page_type_confidence", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_evidence": df.get("page_type_evidence", pd.Series([np.nan] * len(df), index=df.index)),
            "has_price_or_package": df.get("has_price_or_package", pd.Series([np.nan] * len(df), index=df.index)),
            "currency_count": df.get("currency_count", pd.Series([np.nan] * len(df), index=df.index)),
            "price_keyword_count": df.get("price_keyword_count", pd.Series([np.nan] * len(df), index=df.index)),
            "table_count": df.get("table_count", pd.Series([np.nan] * len(df), index=df.index)),
            "cited": df.get("cited", pd.Series([np.nan] * len(df), index=df.index)),
        }
    )
    out["suspicious_price_package_page"] = (
        out["page_type_final"].fillna("").astype(str).eq("price_package_page")
        & ~strong
        & (pd.to_numeric(out["currency_count"], errors="coerce").fillna(0) == 0)
    )
    out["suspicious_price_package_flag"] = out["suspicious_price_package_page"]
    return out[
        out["page_type_final"].fillna("").astype(str).eq("price_package_page")
        | out["has_price_or_package"].fillna(0).astype(float).eq(1)
        | out["suspicious_price_package_page"]
    ].copy()


def page_type_unknown_audit(df: pd.DataFrame) -> pd.DataFrame:
    mask = df.get("page_type_final", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower().eq("unknown")
    text = df.get("page_text", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    out = pd.DataFrame(
        {
            "source_url": df.get("source_url", pd.Series([np.nan] * len(df), index=df.index)),
            "source_domain_host": df.get("source_domain_host", pd.Series([np.nan] * len(df), index=df.index)),
            "source_root_domain": df.get("source_root_domain", pd.Series([np.nan] * len(df), index=df.index)),
            "source_type_url": df.get("source_type_url", pd.Series([np.nan] * len(df), index=df.index)),
            "title": df.get("page_title", df.get("source_title", pd.Series([np.nan] * len(df), index=df.index))),
            "meta_description": df.get("meta_description", df.get("source_description", pd.Series([np.nan] * len(df), index=df.index))),
            "h1_or_top_heading": df.get("h1_or_top_heading", pd.Series([np.nan] * len(df), index=df.index)),
            "page_text_excerpt": text.str.slice(0, 280),
            "word_count": df.get("word_count", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_url_seed": df.get("page_type_url_seed", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_scraped_enriched": df.get("page_type_scraped_enriched", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_final": df.get("page_type_final", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_family": df.get("page_type_family", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_confidence": df.get("page_type_confidence", pd.Series([np.nan] * len(df), index=df.index)),
            "page_type_evidence": df.get("page_type_evidence", pd.Series([np.nan] * len(df), index=df.index)),
            "unknown_reason": df.get("unknown_reason", df.get("page_type_unknown_reason", pd.Series([np.nan] * len(df), index=df.index))),
            "cited": df.get("cited", pd.Series([np.nan] * len(df), index=df.index)),
        }
    )
    return out[mask].copy()


def page_type_manual_review_sample(df: pd.DataFrame) -> pd.DataFrame:
    text = df.get("page_text", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    base = pd.DataFrame(
        {
            "source_url": df.get("source_url", pd.Series([np.nan] * len(df), index=df.index)),
            "title": df.get("page_title", df.get("source_title", pd.Series([np.nan] * len(df), index=df.index))),
            "h1_or_top_heading": df.get("h1_or_top_heading", pd.Series([np.nan] * len(df), index=df.index)),
            "page_text_excerpt": text.str.slice(0, 280),
            "predicted_page_type": df.get("page_type_final", pd.Series([np.nan] * len(df), index=df.index)),
            "predicted_page_type_family": df.get("page_type_family", pd.Series([np.nan] * len(df), index=df.index)),
            "confidence": df.get("page_type_confidence", pd.Series([np.nan] * len(df), index=df.index)),
            "evidence": df.get("page_type_evidence", pd.Series([np.nan] * len(df), index=df.index)),
            "human_label_page_type": "",
            "human_label_family": "",
            "reviewer_notes": "",
        },
        index=df.index,
    )
    parts = []
    work = df.copy()
    work["_priority_n"] = work.groupby("normalized_url", dropna=False)["normalized_url"].transform("size") if "normalized_url" in work.columns else 1
    unknown_idx = work[work.get("page_type_final", "").astype(str).eq("unknown")].sort_values("_priority_n", ascending=False).head(50).index
    parts.append(base.loc[unknown_idx])
    price_idx = work[work.get("page_type_final", "").astype(str).eq("price_package_page")].head(50).index
    parts.append(base.loc[price_idx])
    if "page_type_family" in work.columns:
        for _, g in work.groupby("page_type_family", dropna=False):
            parts.append(base.loc[g.head(10).index])
    if not parts:
        return base.head(0)
    out = pd.concat(parts).drop_duplicates("source_url", keep="first")
    return out.head(250)


def outcome_has_two_classes(df: pd.DataFrame) -> bool:
    return pd.to_numeric(df["cited"], errors="coerce").nunique(dropna=True) >= 2


def _skip(plot_name: str, feature: str, reason: str, n_available=0, n_unique=0, cited_classes=0, min_group_size=np.nan) -> dict:
    return {
        "plot_name": plot_name,
        "feature": feature,
        "reason": reason,
        "n_available": int(n_available) if pd.notna(n_available) else 0,
        "n_unique": int(n_unique) if pd.notna(n_unique) else 0,
        "cited_classes": int(cited_classes) if pd.notna(cited_classes) else 0,
        "min_group_size": min_group_size,
    }


def binary_cited_rates(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, list[dict]]:
    features = features or [c for c in BINARY_FEATURES if c in df.columns]
    y_full = pd.to_numeric(df["cited"], errors="coerce")
    cited_classes = int(y_full.nunique(dropna=True))
    content_available = df.get("content_feature_available", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool)
    rows = []
    skips = []
    for f in features:
        scraped_subset = f in CONTENT_BINARY_FEATURES
        base_mask = content_available if scraped_subset else pd.Series([True] * len(df), index=df.index)
        y = y_full[base_mask]
        x = pd.to_numeric(df.loc[base_mask, f], errors="coerce")
        n_available = int(x.notna().sum())
        n_unique = int(x.dropna().nunique())
        n0 = int((x == 0).sum())
        n1 = int((x == 1).sum())
        min_group = min(n0, n1)
        rate0 = float(y[x == 0].mean()) if n0 else np.nan
        rate1 = float(y[x == 1].mean()) if n1 else np.nan
        diff = rate1 - rate0 if pd.notna(rate0) and pd.notna(rate1) else np.nan
        s0 = int(y[x == 0].sum()) if n0 else 0
        s1 = int(y[x == 1].sum()) if n1 else 0
        r0_low, r0_high = wilson_interval(s0, n0)
        r1_low, r1_high = wilson_interval(s1, n1)
        diff_low = (r1_low - r0_high) * 100 if n0 and n1 else np.nan
        diff_high = (r1_high - r0_low) * 100 if n0 and n1 else np.nan
        reason = ""
        eligible = True
        if cited_classes < 2:
            eligible = False
            reason = "outcome has one class"
        elif n_unique < 2 or n0 == 0 or n1 == 0:
            eligible = False
            reason = "binary feature does not have both 0 and 1"
        elif n0 < 10 or n1 < 10:
            eligible = False
            reason = "one binary group has n < 10"
        if not eligible:
            skips.append(_skip("binary_feature_forest_diff_pp", f, reason, n_available, n_unique, cited_classes, min_group))
        rows.append(
            {
                "feature": f,
                "scraped_subset_only": bool(scraped_subset),
                "n_total_rows": int(len(df)),
                "content_feature_available_denominator": int(content_available.sum()) if scraped_subset else np.nan,
                "coverage": float(n_available / len(df)) if len(df) else 0.0,
                "n_available": n_available,
                "n0": n0,
                "n1": n1,
                "cited_rate_0": rate0,
                "cited_rate_0_ci_low": r0_low,
                "cited_rate_0_ci_high": r0_high,
                "cited_rate_1": rate1,
                "cited_rate_1_ci_low": r1_low,
                "cited_rate_1_ci_high": r1_high,
                "diff_pp": diff * 100 if pd.notna(diff) else np.nan,
                "ci_low": max(-100.0, diff_low) if pd.notna(diff_low) else np.nan,
                "ci_high": min(100.0, diff_high) if pd.notna(diff_high) else np.nan,
                "min_group_size": min_group,
                "sparse_flag": min_group < 20,
                "plot_eligible": eligible,
            }
        )
    return pd.DataFrame(rows), skips


def numeric_binned_summary(df: pd.DataFrame, features: list[str] | None = None, bins: int = 6, min_bin_n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    features = features or [c for c in NUMERIC_FEATURES if c in df.columns]
    y = pd.to_numeric(df["cited"], errors="coerce")
    cited_classes = int(y.nunique(dropna=True))
    rows = []
    recs = []
    skips = []
    for f in features:
        x = pd.to_numeric(df[f], errors="coerce")
        sub = pd.DataFrame({"x": x, "y": y}).dropna()
        n_available = int(len(sub))
        n_unique = int(sub["x"].nunique()) if len(sub) else 0
        if cited_classes < 2:
            skips.append(_skip("numeric_binned", f, "outcome has one class", n_available, n_unique, cited_classes, np.nan))
            continue
        if n_unique < 3:
            skips.append(_skip("numeric_binned", f, "too few unique values", n_available, n_unique, cited_classes, np.nan))
            continue
        if sub.empty:
            skips.append(_skip("numeric_binned", f, "no non-null numeric values", n_available, n_unique, cited_classes, np.nan))
            continue
        method = "qcut"
        q = min(bins, n_unique)
        try:
            sub["bin"] = pd.qcut(sub["x"], q=q, duplicates="drop")
        except Exception:
            method = "cut_after_qcut_failed"
            sub["bin"] = pd.cut(sub["x"], bins=q, duplicates="drop")
        non_empty_bins = int(sub["bin"].nunique(dropna=True))
        if non_empty_bins < 3 and 3 <= n_unique <= 12:
            method = "unique_value_bins"
            sub["bin"] = sub["x"].map(lambda v: f"{v:g}" if pd.notna(v) else np.nan)
            non_empty_bins = int(sub["bin"].nunique(dropna=True))
        elif non_empty_bins < 3:
            skips.append(_skip("numeric_binned", f, "qcut collapsed due tied values; fewer than 3 non-empty bins", n_available, n_unique, cited_classes, np.nan))
            continue
        bin_counts = sub.groupby("bin", observed=True).size()
        eligible_bins = int((bin_counts >= min_bin_n).sum())
        min_count = int(bin_counts.min()) if len(bin_counts) else 0
        if eligible_bins < 3:
            skips.append(_skip("numeric_binned", f, f"fewer than 3 eligible bins after min_n={min_bin_n}", n_available, n_unique, cited_classes, min_count))
            continue
        for b, g in sub.groupby("bin", observed=True):
            rows.append({"feature": f, "bin": str(b), "x_mean": float(g["x"].mean()), "n": int(len(g)), "cited_rate": float(g["y"].mean())})
        recs.append(
            {
                "feature": f,
                "n_available": n_available,
                "n_unique": n_unique,
                "binning_method": method,
                "non_empty_bins": non_empty_bins,
                "eligible_bins_n_ge_20": eligible_bins,
                "min_bin_size": min_count,
                "skew_hint": "use log1p scale" if f in {"word_count", "link_count", "domain_seen_count", "domain_seen_count_loo"} else "linear/quantile bins",
                "numeric_shape_recommendation": "inspect binned cited-rate shape before choosing LPM transform",
                "plot_eligible": True,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(recs), skips


def categorical_cited_rates(df: pd.DataFrame, features: list[str] | None = None, top_n: int = 15) -> tuple[pd.DataFrame, list[dict]]:
    features = features or [c for c in CATEGORICAL_FEATURES if c in df.columns]
    y = pd.to_numeric(df["cited"], errors="coerce")
    cited_classes = int(y.nunique(dropna=True))
    rows = []
    skips = []
    for f in features:
        raw = df[f]
        missing_n = int(raw.isna().sum())
        s = raw.dropna().astype(str)
        if f == "page_type_final":
            s = s[s.str.strip() != ""]
        vc = s.value_counts()
        n_unique = int(vc.size)
        if cited_classes < 2:
            skips.append(_skip("categorical_cited_rate", f, "outcome has one class", int(s.notna().sum()), n_unique, cited_classes, int(vc.min()) if len(vc) else 0))
            continue
        if n_unique < 2:
            skips.append(_skip("categorical_cited_rate", f, "categorical feature has fewer than 2 categories", int(s.notna().sum()), n_unique, cited_classes, int(vc.min()) if len(vc) else 0))
            continue
        top = set(vc.head(top_n).index)
        s2 = s.where(s.isin(top), "Other")
        for cat, n in s2.value_counts().items():
            mask = s2 == cat
            aligned_y = y.loc[s2.index]
            successes = int(aligned_y[mask].sum())
            rate = float(aligned_y[mask].mean())
            ci_low, ci_high = wilson_interval(successes, int(n))
            rows.append({"feature": f, "category": cat, "n": int(n), "missing_n": missing_n, "cited_rate": rate, "ci_low": ci_low, "ci_high": ci_high, "sparse_flag": int(n) < 20})
    return pd.DataFrame(rows), skips


def correlation_and_vif(df: pd.DataFrame, features: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocked = LEAKAGE_EXCLUSIONS | DIAGNOSTIC_ONLY
    features = features or [c for c in NUMERIC_FEATURES if c in df.columns and c not in blocked]
    num = df[features].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    corr = num.corr() if not num.empty else pd.DataFrame()
    pairs = []
    for i, a in enumerate(corr.columns):
        for b in corr.columns[i + 1:]:
            val = corr.loc[a, b]
            if pd.notna(val) and abs(val) >= 0.75:
                pairs.append({"feature_a": a, "feature_b": b, "correlation": float(val)})
    high = pd.DataFrame(pairs)
    vif_rows = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        mat = num.replace([np.inf, -np.inf], np.nan).dropna()
        mat = mat.loc[:, mat.nunique() > 1]
        if len(mat) >= 5 and mat.shape[1] >= 2:
            arr = mat.to_numpy(dtype=float)
            for idx, col in enumerate(mat.columns):
                vif_rows.append({"feature": col, "vif": float(variance_inflation_factor(arr, idx))})
    except Exception as exc:  # noqa: BLE001
        vif_rows.append({"feature": "__vif_error__", "vif": np.nan, "note": str(exc)})
    return corr, high, pd.DataFrame(vif_rows)


def write_eda_outputs(df: pd.DataFrame, output_dir: str | Path = OUTPUT_DIR, enable_lightgbm: bool = True) -> dict:
    from src.econometrics_eda_v2 import plotting

    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    plots = output_dir / "plots"
    if plots.exists():
        shutil.rmtree(plots)
    if tables.exists():
        shutil.rmtree(tables)
    tables.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    warnings = []
    df = df.copy()
    df["cited"] = pd.to_numeric(df["cited"], errors="coerce").fillna(0).astype(int)
    if "page_type_final" not in df.columns and "page_type" in df.columns:
        df["page_type_final"] = df["page_type"]
        df["page_type_final_source"] = "legacy_page_type_debug"
        df["page_type_missing_reason"] = np.where(df["page_type_final"].notna(), "", "missing_no_page_type")

    two_class = outcome_has_two_classes(df)
    if not two_class:
        warnings.append("outcome_has_single_class_no_more_only_sources")
        warnings.append("Cannot evaluate feature association with cited because outcome has only one class.")
    availability = feature_availability(df)
    page_type_audit = page_type_final_audit(df)
    page_type_funnel_df = page_type_funnel(df)
    page_similarity_audit = page_similarity_availability_audit(df)
    domain_audit = domain_field_audit(df)
    domain_summary = domain_field_summary(df)
    price_audit = price_package_page_audit(df)
    unknown_audit = page_type_unknown_audit(df)
    manual_review = page_type_manual_review_sample(df)
    binary, binary_skips = binary_cited_rates(df)
    numeric, numeric_recs, numeric_skips = numeric_binned_summary(df)
    categorical, categorical_skips = categorical_cited_rates(df)
    corr, high_corr, vif = correlation_and_vif(df)
    readiness = build_lpm_readiness(df, vif)
    skip_reasons = pd.DataFrame(binary_skips + numeric_skips + categorical_skips)
    if skip_reasons.empty:
        skip_reasons = pd.DataFrame(columns=["plot_name", "feature", "reason", "n_available", "n_unique", "cited_classes", "min_group_size"])

    availability.to_csv(tables / "feature_inventory_all_columns.csv", index=False)
    main_availability = availability[availability["show_in_main_coverage"].astype(bool)].copy()
    main_availability.to_csv(tables / "feature_availability_summary.csv", index=False)
    main_availability.groupby("feature_group", as_index=False).agg(
        features=("feature", "count"),
        mean_coverage=("coverage", "mean"),
        min_coverage=("coverage", "min"),
    ).to_csv(tables / "feature_coverage_by_group.csv", index=False)
    page_type_audit.to_csv(tables / "page_type_final_audit.csv", index=False)
    page_type_funnel_df.to_csv(tables / "page_type_coverage_funnel.csv", index=False)
    page_similarity_audit.to_csv(tables / "page_similarity_availability_audit.csv", index=False)
    domain_audit.to_csv(tables / "domain_field_audit.csv", index=False)
    domain_summary.to_csv(tables / "domain_field_summary.csv", index=False)
    price_audit.to_csv(tables / "price_package_page_audit.csv", index=False)
    unknown_audit.to_csv(tables / "page_type_unknown_audit.csv", index=False)
    manual_review.to_csv(tables / "page_type_manual_review_sample.csv", index=False)
    binary.to_csv(tables / "binary_feature_cited_rate_summary.csv", index=False)
    numeric.to_csv(tables / "numeric_binned_scatter_summary.csv", index=False)
    numeric_recs.to_csv(tables / "numeric_shape_recommendations.csv", index=False)
    categorical.to_csv(tables / "categorical_cited_rate_summary.csv", index=False)
    corr.to_csv(tables / "correlation_matrix.csv")
    high_corr.to_csv(tables / "high_correlation_pairs.csv", index=False)
    vif.to_csv(tables / "vif_summary.csv", index=False)
    pd.DataFrame(
        [{"recommendation": "Review high_correlation_pairs.csv and prefer one variable from each highly correlated feature family before LPM."}]
    ).to_csv(tables / "correlation_vif_recommendations.csv", index=False)
    skip_reasons.to_csv(tables / "plot_skip_reasons.csv", index=False)
    readiness.to_csv(tables / "lpm_feature_readiness.csv", index=False)

    plot_paths = []
    plot_paths += plotting.plot_outcome_balance(df, plots)
    plot_paths += plotting.plot_feature_availability(main_availability, plots)
    plot_paths += plotting.plot_page_type_funnel(page_type_funnel_df, plots)
    if two_class:
        plot_paths += plotting.plot_binary_rates(binary, plots)
        plot_paths += plotting.plot_numeric_binned(numeric, plots)
        plot_paths += plotting.plot_categorical_rates(categorical, plots)
    else:
        warnings.append("association_plots_skipped_single_class_outcome")
    plot_paths += plotting.plot_intent_page_type(df, plots)
    plot_paths += plotting.plot_correlation(corr, plots)

    lgbm_paths = []
    if enable_lightgbm and two_class:
        try:
            lgbm_paths = plotting.plot_lightgbm_discovery(df, plots)
            plot_paths += lgbm_paths
            warnings.append("lightgbm_safe_model_excludes_source_position")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LightGBM discovery skipped: {exc}")
    elif enable_lightgbm and not two_class:
        warnings.append("lightgbm_skipped_single_class_outcome")

    if binary.empty:
        warnings.append("No binary feature cited-rate rows generated.")
    if numeric.empty:
        warnings.append("No numeric binned scatter rows generated.")
    if categorical.empty:
        warnings.append("No categorical cited-rate rows generated.")
    pt_cov = float(page_type_audit["page_type_final_coverage"].iloc[0]) if not page_type_audit.empty else 0.0
    pt_url_cov = float(page_type_audit["page_type_url_seed_coverage_valid_url_rows"].iloc[0]) if not page_type_audit.empty else 0.0
    if pt_cov < 0.95 or pt_url_cov < 0.95:
        warnings.append("page_type_coverage_below_threshold")
    content_cov = float(df.get("content_feature_available", pd.Series([False] * len(df), index=df.index)).fillna(False).astype(bool).mean()) if len(df) else 0.0
    if content_cov < 0.95:
        warnings.append("content_feature_coverage_below_threshold")
    if "content_feature_available" in df.columns and two_class:
        content_bool = df["content_feature_available"].fillna(False).astype(bool)
        rates = content_bool.groupby(df["cited"]).mean()
        if {0, 1}.issubset(set(rates.index)) and abs(float(rates.loc[1] - rates.loc[0])) >= 0.05:
            warnings.append("content_feature_availability_selected_by_cited")
    if not page_similarity_audit.empty and int(page_similarity_audit["rows_without_scraped_body_but_similarity_non_null"].max()) > 0:
        warnings.append("page_similarity_computed_without_scraped_body")
    if not high_corr.empty:
        warnings.append("high_vif_or_high_correlation_pairs_present")
    if not vif.empty and "vif" in vif.columns and (pd.to_numeric(vif["vif"], errors="coerce") >= 10).any():
        warnings.append("high_vif_or_high_correlation_pairs_present")
    if not readiness.empty and ((readiness["n_unique"] < 2) & (~readiness["recommended_for_lpm"].astype(bool))).any():
        warnings.append("constant_controls_not_recommended_for_lpm")
    warnings.append("diagnostic_only_features_excluded_from_safe_predictors")
    warnings = list(dict.fromkeys(warnings))
    warnings_df = pd.DataFrame({"warning": warnings})
    warnings_df.to_csv(output_dir / "eda_warnings.csv", index=False)
    meta = {
        "created_at": utc_now_iso(),
        "rows": int(len(df)),
        "cited_count": int(df["cited"].sum()),
        "more_only_count": int((df["cited"] == 0).sum()),
        "cited_rate": float(df["cited"].mean()) if len(df) else 0.0,
        "plot_count": len(plot_paths),
        "binary_plot_count": len([p for p in plot_paths if "binary_feature" in Path(p).name]),
        "numeric_plot_count": len([p for p in plot_paths if Path(p).name.startswith("05_numeric_binned_")]),
        "categorical_plot_count": len([p for p in plot_paths if Path(p).name.startswith("06_categorical_")]),
        "lightgbm_plot_count": len(lgbm_paths),
        "warnings": warnings,
        "wording": "Descriptive associations among surfaced sources only; more-only means surfaced but not cited.",
    }
    write_json(output_dir / "run_metadata.json", meta)
    return {"metadata": meta, "plots": plot_paths, "warnings": warnings, "readiness": readiness}
