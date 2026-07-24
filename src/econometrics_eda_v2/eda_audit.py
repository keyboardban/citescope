from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.io import write_csv

SAFE_PREDICTORS = [
    "source_type_url", "page_type_final", "page_type_final_source",
    "title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity",
    "max_chunk_prompt_similarity", "domain_seen_count_loo", "log1p_domain_seen_count",
    "url_length", "url_path_depth", "has_faq", "has_price_or_package",
    "has_contact_info", "has_table", "has_booking_or_appointment", "word_count",
    "heading_count", "intent", "topic", "country", "language",
]
DIAGNOSTIC_ONLY = [
    "source_position", "observed_rank", "log1p_source_position", "scrape_success",
    "parse_success", "content_feature_available", "page_type_final_source",
]
LEAKAGE_EXCLUDED = [
    "cited_label", "is_more_only", "source_group", "source_origin",
    "page_answer_similarity", "max_chunk_answer_similarity",
    "answer_like_text_in_first_500_chars",
]


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    return df[col].fillna(False).astype(bool)


def build_eda_audit_tables(rows: pd.DataFrame, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y = pd.to_numeric(rows["cited"], errors="coerce").fillna(0)
    source_audit = pd.DataFrame(
        [
            {
                "rows": int(len(rows)),
                "cited_count": int((y == 1).sum()),
                "more_only_count": int((y == 0).sum()),
                "cited_rate": float(y.mean()) if len(y) else 0.0,
                "unique_prompts": int(rows.get("prompt_id", pd.Series(dtype=object)).nunique(dropna=True)),
                "unique_records": int(rows.get("record_id", pd.Series(dtype=object)).nunique(dropna=True)),
                "unique_urls": int(rows.get("normalized_url", pd.Series(dtype=object)).nunique(dropna=True)),
            }
        ]
    )
    write_csv(out / "source_outcome_audit.csv", source_audit)

    stages = {
        "source_rows": pd.Series([True] * len(rows), index=rows.index),
        "valid_url": rows.get("normalized_url", pd.Series([""] * len(rows))).fillna("").astype(str) != "",
        "in_scrape_queue": _bool_series(rows, "in_scrape_queue"),
        "has_raw_cache": _bool_series(rows, "has_raw_apify_cache"),
        "scrape_success": _bool_series(rows, "scrape_success"),
        "parse_success": _bool_series(rows, "parse_success"),
        "scraped_body_available": _bool_series(rows, "scraped_body_available"),
        "content_feature_available": _bool_series(rows, "content_feature_available"),
        "page_type_scraped_enriched_available": rows.get("page_type_scraped_enriched", pd.Series([None] * len(rows))).notna(),
        "page_type_final_available": rows.get("page_type_final", pd.Series([None] * len(rows))).notna(),
    }
    funnel_rows = []
    for stage, mask in stages.items():
        for cited_value, label in [(1, "cited"), (0, "more_only")]:
            m = mask & (y == cited_value)
            funnel_rows.append({"stage": stage, "cited_status": label, "rows": int(m.sum())})
    write_csv(out / "scrape_feature_funnel_by_cited.csv", pd.DataFrame(funnel_rows))

    if "page_type_final" in rows.columns:
        g = rows.groupby("page_type_final", dropna=False)
        pt = g.agg(n=("cited", "size"), cited_count=("cited", "sum")).reset_index()
        pt["cited_rate"] = pt["cited_count"] / pt["n"]
        if "page_type_final_source" in rows.columns:
            seed = rows["page_type_final_source"].eq("url_seed")
            scraped = rows["page_type_final_source"].eq("scraped_content")
            pt["source_url_seed_n"] = pt["page_type_final"].map(rows[seed]["page_type_final"].value_counts()).fillna(0).astype(int)
            pt["scraped_content_n"] = pt["page_type_final"].map(rows[scraped]["page_type_final"].value_counts()).fillna(0).astype(int)
            pt["source_share_url_seed"] = pt["source_url_seed_n"] / pt["n"]
            pt["source_share_scraped"] = pt["scraped_content_n"] / pt["n"]
        write_csv(out / "page_type_source_audit.csv", pt)

    missing = pd.DataFrame(
        [
            {
                "field": c,
                "non_null_count": int(rows[c].notna().sum()),
                "coverage": float(rows[c].notna().mean()) if len(rows) else 0.0,
                "missing_rate": float(rows[c].isna().mean()) if len(rows) else 0.0,
            }
            for c in ["page_type_url_seed", "page_type_scraped_enriched", "page_type_final", "content_feature_available"]
            if c in rows.columns
        ]
    )
    write_csv(out / "page_type_missingness_audit.csv", missing)

    feature_rows = []
    for c in rows.columns:
        if c in SAFE_PREDICTORS:
            group, safe, reason = "safe_predictor", True, "pre-answer observable feature/control"
        elif c in DIAGNOSTIC_ONLY:
            group, safe, reason = "diagnostic_only", False, "diagnostic/selection/status feature"
        elif c in LEAKAGE_EXCLUDED:
            group, safe, reason = "leakage_excluded", False, "outcome/post-output/leakage risk"
        else:
            group, safe, reason = "other", False, "not in curated safe predictor list"
        feature_rows.append(
            {
                "feature": c,
                "group": group,
                "non_null_count": int(rows[c].notna().sum()),
                "coverage": float(rows[c].notna().mean()) if len(rows) else 0.0,
                "missing_rate": float(rows[c].isna().mean()) if len(rows) else 0.0,
                "safe_for_lpm": safe,
                "reason": reason,
            }
        )
    write_csv(out / "feature_availability_by_group.csv", pd.DataFrame(feature_rows))
    write_csv(out / "safe_predictor_list.csv", pd.DataFrame({"feature": SAFE_PREDICTORS}))
    write_csv(out / "diagnostic_only_feature_list.csv", pd.DataFrame({"feature": DIAGNOSTIC_ONLY}))
    write_csv(out / "leakage_excluded_feature_list.csv", pd.DataFrame({"feature": LEAKAGE_EXCLUDED}))
    return {"tables_created": 8, "output_dir": str(out)}
