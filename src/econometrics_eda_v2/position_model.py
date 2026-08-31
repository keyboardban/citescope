"""Separate position-focused citation econometrics pipeline.

This module deliberately does not modify or reuse the governed D0-FE4 model
registry. It builds a new M0-M6 position model from frozen HTML-position and
Gemini block-classification artifacts. All estimates are observational adjusted
associations among surfaced webpages with measurable content.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


POSITION_MODEL_VERSION = "position_model_v3_20260804_taxonomy_6"
DEFAULT_OUTPUT_DIR = "outputs/position_model_v1"
PRIMARY_FEATURES = (
    "direct_answer_placement",
    "table_placement",
    "question_heading_placement",
    "z_numeric_evidence_total_density",
)
PLACEMENT_FEATURES = (
    "direct_answer_placement",
    "table_placement",
    "question_heading_placement",
    "external_source_placement",
)
REFERENCE_LEVELS = {
    "direct_answer_placement": "no_direct_answer",
    "table_placement": "no_table",
    "question_heading_placement": "no_question_heading",
    "external_source_placement": "no_external_source",
}
FEATURE_LABELS = {
    "direct_answer_placement": "Direct-answer placement",
    "table_placement": "Table placement",
    "question_heading_placement": "Question-heading placement",
    "z_numeric_evidence_total_density": "Total numeric-evidence density",
    "numeric_evidence_early_share": "Numeric-evidence early share",
    "external_source_placement": "External-source placement",
}
PAGE_TYPE_MODEL_6_MAP = {
    "informational_content": "blog_guide_or_editorial",
    "news_or_press": "blog_guide_or_editorial",
    "directory_or_listing": "directory_or_listing",
    "commercial_product_or_service": "commercial_product_or_service",
    "comparison_or_review": "comparison_or_review",
    "landing_or_brand_page": "landing_contact_or_support",
    "contact_or_location": "landing_contact_or_support",
    "support_or_help": "landing_contact_or_support",
    "document_or_media": "other_page_function",
    "social_or_user_generated": "other_page_function",
    "search_or_results": "other_page_function",
    "unknown": "other_page_function",
    "rare_other": "other_page_function",
}
SOURCE_TYPE_MODEL_6_MAP = {
    "official_company_or_brand": "official_company_or_brand",
    "marketplace_or_platform": "marketplace_or_directory_platform",
    "directory_or_listing_platform": "marketplace_or_directory_platform",
    "blog_or_content_site": "blog_or_news_publisher",
    "news_media": "blog_or_news_publisher",
    "review_platform": "review_or_community_platform",
    "social_or_forum": "review_or_community_platform",
    "government": "government_or_public_institution",
    "rare_other": "other_or_unknown",
    "unknown": "other_or_unknown",
}
STATUS_LEVELS = (
    "success_present",
    "success_absent",
    "scrape_failure",
    "main_content_failure",
    "parser_failure",
    "ineligible",
    "ambiguous",
)
SE_METHODS = (
    "HC3",
    "cluster_domain",
    "cluster_prompt",
    "two_way_cluster_domain_prompt",
)


@dataclass(frozen=True)
class PositionPaths:
    selected_rows: Path
    legacy_position_rows: Path
    gemini_pages: Path
    gemini_evidence: Path


def source_paths(repo: Path) -> PositionPaths:
    return PositionPaths(
        selected_rows=repo / (
            "outputs/econometrics_redesign_v4_20260803_gemini_semantic_features/"
            "data/selected_feature_rows.csv"
        ),
        legacy_position_rows=repo / (
            "outputs/position_feature_eda_final_20260731/data/"
            "scope_condo_eda_ready_with_position_features.parquet"
        ),
        gemini_pages=repo / (
            "outputs/position_feature_eda_final_20260731/llm_semantic_smoke/tables/"
            "gemini_position_smoke_pages.csv"
        ),
        gemini_evidence=repo / (
            "outputs/position_feature_eda_final_20260731/llm_semantic_smoke/tables/"
            "gemini_position_detection_evidence.csv"
        ),
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _sample_zscore(series: pd.Series) -> pd.Series:
    """Standardize measured values with the row-sample mean and sample SD."""
    values = _numeric(series)
    standard_deviation = values.std(ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (values - values.mean()) / standard_deviation


def _collapse_page_type_for_model(series: pd.Series) -> pd.Series:
    detailed = series.fillna("unknown").astype(str)
    return detailed.map(PAGE_TYPE_MODEL_6_MAP).fillna("other_page_function").astype(str)


def _collapse_source_type_for_model(series: pd.Series) -> pd.Series:
    detailed = series.fillna("unknown").astype(str)
    return detailed.map(SOURCE_TYPE_MODEL_6_MAP).fillna("other_or_unknown").astype(str)


def _domain_source_type_assignments(rows: pd.DataFrame) -> pd.DataFrame:
    """Choose one collapsed source class per domain using unique-URL support."""
    url_labels = rows[
        ["source_root_domain", "normalized_url", "source_type_row_collapsed"]
    ].drop_duplicates(["source_root_domain", "normalized_url"], keep="last")
    counts = url_labels.groupby(
        ["source_root_domain", "source_type_row_collapsed"], observed=True
    ).agg(unique_urls=("normalized_url", "nunique")).reset_index()
    counts["domain_unique_urls"] = counts.groupby("source_root_domain")[
        "unique_urls"
    ].transform("sum")
    counts["domain_class_count"] = counts.groupby("source_root_domain")[
        "source_type_row_collapsed"
    ].transform("nunique")
    counts["top_unique_urls"] = counts.groupby("source_root_domain")[
        "unique_urls"
    ].transform("max")
    counts["is_top"] = counts["unique_urls"].eq(counts["top_unique_urls"])
    counts["top_class_tie"] = counts.groupby("source_root_domain")["is_top"].transform("sum").gt(1)
    counts["candidate_entry"] = counts.apply(
        lambda row: f"{row.source_type_row_collapsed}:{int(row.unique_urls)}", axis=1
    )
    summaries = counts.groupby("source_root_domain", observed=True)[
        "candidate_entry"
    ].agg(lambda values: "; ".join(sorted(values))).rename("candidate_summary")
    top = counts[counts["is_top"]].sort_values(
        ["source_root_domain", "source_type_row_collapsed"], kind="stable"
    ).drop_duplicates("source_root_domain", keep="first")
    top["source_type_model_6"] = top["source_type_row_collapsed"].where(
        ~top["top_class_tie"], "other_or_unknown"
    )
    top["dominant_url_share"] = top["top_unique_urls"] / top["domain_unique_urls"]
    top["low_confidence_below_60pct"] = top["dominant_url_share"].lt(.60)
    top = top.merge(summaries, on="source_root_domain", how="left")
    return top[[
        "source_root_domain", "source_type_model_6", "source_type_row_collapsed",
        "dominant_url_share", "top_class_tie", "low_confidence_below_60pct",
        "domain_unique_urls", "domain_class_count", "candidate_summary",
    ]].rename(columns={"source_type_row_collapsed": "dominant_source_type_before_tie_rule"})


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _map_gemini_status(row: pd.Series) -> str:
    gemini = str(row.get("gemini_status") or "").strip().casefold()
    block = str(row.get("block_extraction_status") or "").strip().casefold()
    if gemini == "success" and block == "measured":
        return "success_absent"
    if block == "no_html":
        return "scrape_failure"
    if block in {"main_content_parse_failed", "no_eligible_blocks"}:
        return "main_content_failure"
    if gemini == "partial_failure":
        return "parser_failure"
    if gemini in {"dry_run", "unmeasured_no_blocks", ""}:
        return "ineligible"
    return "ambiguous"


def _map_html_status(value: Any) -> str:
    status = str(value or "").strip().casefold()
    if status == "measured":
        return "success_absent"
    if status in {"no_raw_snapshot", "no_html", "scrape_failure"}:
        return "scrape_failure"
    if status in {"main_content_parse_failed", "no_main_content"}:
        return "main_content_failure"
    if "parse" in status:
        return "parser_failure"
    if not status or status in {"unmeasured", "not_available"}:
        return "ineligible"
    return "ambiguous"


def _placement(
    presence: pd.Series,
    ratio: pd.Series,
    base_status: pd.Series,
    *,
    no_label: str,
    early_label: str,
    late_label: str,
) -> tuple[pd.Series, pd.Series]:
    present = _numeric(presence)
    position = _numeric(ratio)
    status = base_status.astype("string").copy()
    category = pd.Series(pd.NA, index=presence.index, dtype="string")
    measured = status.eq("success_absent")
    absent = measured & present.eq(0)
    valid_present = measured & present.eq(1) & position.between(0, 1, inclusive="both")
    invalid_present = measured & present.eq(1) & ~position.between(0, 1, inclusive="both")
    category.loc[absent] = no_label
    category.loc[valid_present & position.lt(0.5)] = early_label
    category.loc[valid_present & position.ge(0.5)] = late_label
    status.loc[valid_present] = "success_present"
    status.loc[invalid_present] = "ambiguous"
    return category, status


def _gemini_url_features(pages: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    pages = pages.drop_duplicates("normalized_url", keep="last").copy()
    evidence = evidence.copy()
    evidence["position_ratio"] = _numeric(evidence["position_ratio"])
    evidence["start_token"] = _numeric(evidence["start_token"])

    grouped: list[pd.DataFrame] = []
    for feature, prefix, allowed_tags in (
        ("direct_answer", "direct_answer", None),
        ("question_heading", "question_heading", {"h2", "h3"}),
        ("numeric_evidence", "numeric_evidence", None),
    ):
        subset = evidence[evidence["feature"].eq(feature)].copy()
        if allowed_tags is not None:
            subset = subset[subset["tag"].astype(str).str.casefold().isin(allowed_tags)]
        if subset.empty:
            continue
        subset = subset.sort_values(["normalized_url", "start_token"], kind="stable")
        summary = subset.groupby("normalized_url", observed=True).agg(
            **{
                f"{prefix}_validated_count": ("block_id", "nunique"),
                f"{prefix}_validated_start_token": ("start_token", "min"),
                f"{prefix}_validated_position_ratio": ("position_ratio", "min"),
                f"{prefix}_validated_evidence": ("evidence_text", "first"),
            }
        ).reset_index()
        grouped.append(summary)
    for summary in grouped:
        pages = pages.merge(summary, on="normalized_url", how="left", validate="one_to_one")

    base_status = pages.apply(_map_gemini_status, axis=1).astype("string")
    success = base_status.eq("success_absent")
    for prefix in ("direct_answer", "question_heading", "numeric_evidence"):
        count_col = f"{prefix}_validated_count"
        pages[count_col] = _numeric(pages.get(count_col, pd.Series(index=pages.index))).where(success)
        pages.loc[success, count_col] = pages.loc[success, count_col].fillna(0)

    direct_presence = pages["direct_answer_validated_count"].gt(0).where(success)
    pages["direct_answer_placement"], pages["direct_answer_extraction_status"] = _placement(
        direct_presence,
        pages["direct_answer_validated_position_ratio"],
        base_status,
        no_label="no_direct_answer",
        early_label="direct_answer_early",
        late_label="direct_answer_late",
    )
    pages["direct_answer_start_token"] = _numeric(pages["direct_answer_validated_start_token"])
    pages["direct_answer_position_ratio"] = _numeric(
        pages["direct_answer_validated_position_ratio"]
    )

    question_presence = pages["question_heading_validated_count"].gt(0).where(success)
    pages["question_heading_placement"], pages["question_heading_extraction_status"] = _placement(
        question_presence,
        pages["question_heading_validated_position_ratio"],
        base_status,
        no_label="no_question_heading",
        early_label="question_heading_early",
        late_label="question_heading_late",
    )
    pages["question_heading_count"] = pages["question_heading_validated_count"]
    pages["first_question_heading_start_token"] = _numeric(
        pages["question_heading_validated_start_token"]
    )
    pages["first_question_heading_position_ratio"] = _numeric(
        pages["question_heading_validated_position_ratio"]
    )

    numeric = evidence[evidence["feature"].eq("numeric_evidence")].copy()
    numeric["early"] = numeric["position_ratio"].lt(0.5)
    numeric = numeric.sort_values(["normalized_url", "start_token"], kind="stable")
    numeric_summary = numeric.groupby("normalized_url", observed=True).agg(
        numeric_evidence_total_count=("block_id", "nunique"),
        numeric_evidence_early_count=("early", "sum"),
        numeric_evidence_first_evidence=("evidence_text", "first"),
    ).reset_index()
    pages = pages.drop(
        columns=[
            column
            for column in ("numeric_evidence_total_count", "numeric_evidence_early_count")
            if column in pages
        ]
    ).merge(numeric_summary, on="normalized_url", how="left", validate="one_to_one")
    for column in ("numeric_evidence_total_count", "numeric_evidence_early_count"):
        pages[column] = _numeric(pages[column]).where(success)
        pages.loc[success, column] = pages.loc[success, column].fillna(0)
    tokens = _numeric(pages["total_main_content_tokens"])
    pages["numeric_evidence_total_density"] = (
        pages["numeric_evidence_total_count"] / tokens.clip(lower=1) * 1000
    ).where(success & tokens.gt(0))
    pages["numeric_evidence_early_share"] = (
        pages["numeric_evidence_early_count"] / pages["numeric_evidence_total_count"]
    ).where(success & pages["numeric_evidence_total_count"].gt(0))
    pages["numeric_evidence_extraction_status"] = base_status
    pages.loc[success & pages["numeric_evidence_total_count"].gt(0), "numeric_evidence_extraction_status"] = "success_present"
    return pages


def build_position_model_dataset(repo: Path) -> pd.DataFrame:
    """Build the row-level position dataset without changing prior artifacts."""
    paths = source_paths(repo)
    for path in vars(paths).values():
        if not Path(path).exists():
            raise FileNotFoundError(path)

    base = pd.read_csv(paths.selected_rows, low_memory=False)
    legacy = pd.read_parquet(paths.legacy_position_rows)
    gemini_pages = pd.read_csv(paths.gemini_pages, low_memory=False)
    evidence = pd.read_csv(paths.gemini_evidence, low_memory=False)

    context_columns = [
        "normalized_url", "source_url", "intent", "position_extraction_status",
        "position_measurement_source", "position_features_available", "total_main_content_token_count",
        "has_table", "table_count", "table_start_token_index", "first_table_position_ratio",
        "table_evidence",
        "has_external_sources", "outbound_citation_count", "external_citation_start_token_index",
        "first_external_citation_position_ratio", "external_link_domains", "external_citation_evidence",
    ]
    context_columns = [column for column in context_columns if column in legacy]
    context = legacy[context_columns].drop_duplicates("normalized_url", keep="last")
    gemini = _gemini_url_features(gemini_pages, evidence)
    gemini_columns = [
        "normalized_url", "block_extraction_status", "block_extraction_method",
        "total_main_content_tokens", "selected_blocks", "blocks_truncated", "gemini_status",
        "gemini_error", "direct_answer_placement", "direct_answer_start_token",
        "direct_answer_position_ratio", "direct_answer_extraction_status", "direct_answer_validated_evidence",
        "question_heading_placement", "question_heading_count",
        "first_question_heading_start_token", "first_question_heading_position_ratio",
        "question_heading_extraction_status", "question_heading_validated_evidence", "numeric_evidence_early_count",
        "numeric_evidence_total_count", "numeric_evidence_total_density",
        "numeric_evidence_early_share",
        "numeric_evidence_extraction_status", "numeric_evidence_first_evidence",
    ]
    rows = base.merge(context, on="normalized_url", how="left", validate="many_to_one")
    rows = rows.merge(gemini[gemini_columns], on="normalized_url", how="left", validate="many_to_one")

    rows["page_type_detailed"] = (
        rows["page_type_family_gemini_v1_collapsed"].fillna("unknown").astype(str)
    )
    rows["page_type_model_6"] = _collapse_page_type_for_model(rows["page_type_detailed"])
    rows["page_type"] = rows["page_type_model_6"]
    rows["source_type_detailed"] = (
        rows["source_type_general_gemini_v1_collapsed"].fillna("unknown").astype(str)
    )
    rows["source_type_row_collapsed"] = _collapse_source_type_for_model(
        rows["source_type_detailed"]
    )
    source_assignments = _domain_source_type_assignments(rows)
    rows = rows.merge(
        source_assignments, on="source_root_domain", how="left", validate="many_to_one"
    )
    rows["source_type"] = rows["source_type_model_6"]
    rows["intent"] = rows.get("intent", pd.Series("unknown", index=rows.index)).fillna("unknown").astype(str)
    rows["word_count"] = np.power(2.0, _numeric(rows["log2_word_count_plus1"])) - 1
    rows["log_word_count"] = np.log1p(rows["word_count"])

    html_status = rows["position_extraction_status"].map(_map_html_status).astype("string")
    verified_table = _numeric(rows["has_verified_html_table"])
    legacy_table = _numeric(rows.get("has_table", pd.Series(index=rows.index)))
    table_presence = verified_table.where(verified_table.notna(), legacy_table)
    rows["table_placement"], rows["table_extraction_status"] = _placement(
        table_presence,
        _numeric(rows.get("first_table_position_ratio", pd.Series(index=rows.index))),
        html_status,
        no_label="no_table",
        early_label="table_early",
        late_label="table_late",
    )
    rows["first_table_start_token"] = _numeric(
        rows.get("table_start_token_index", pd.Series(index=rows.index))
    )

    external_presence = _numeric(rows.get("has_external_sources", pd.Series(index=rows.index)))
    rows["external_source_placement"], rows["external_source_extraction_status"] = _placement(
        external_presence,
        _numeric(rows.get("first_external_citation_position_ratio", pd.Series(index=rows.index))),
        html_status,
        no_label="no_external_source",
        early_label="external_source_early",
        late_label="external_source_late",
    )
    rows["external_source_start_token"] = _numeric(
        rows.get("external_citation_start_token_index", pd.Series(index=rows.index))
    )
    rows["external_source_position_ratio"] = _numeric(
        rows.get("first_external_citation_position_ratio", pd.Series(index=rows.index))
    )

    density = _numeric(rows["numeric_evidence_total_density"])
    rows["z_numeric_evidence_total_density"] = _sample_zscore(density)
    p99 = density.quantile(0.99)
    rows["numeric_evidence_total_density_winsorized_p99"] = density.clip(upper=p99)
    win = rows["numeric_evidence_total_density_winsorized_p99"]
    rows["z_numeric_evidence_total_density_winsorized_p99"] = _sample_zscore(win)

    for feature, ratio in (
        ("direct_answer_placement", "direct_answer_position_ratio"),
        ("table_placement", "first_table_position_ratio"),
        ("question_heading_placement", "first_question_heading_position_ratio"),
        ("external_source_placement", "external_source_position_ratio"),
    ):
        no_feature = rows[feature].eq(REFERENCE_LEVELS[feature])
        rows.loc[no_feature, ratio] = np.nan
        invalid = _numeric(rows[ratio]).dropna().loc[lambda value: ~value.between(0, 1)]
        if not invalid.empty:
            raise ValueError(f"{ratio} contains values outside [0, 1]")

    rows["position_model_version"] = POSITION_MODEL_VERSION
    rows["joint_position_features_measured"] = rows[
        [
            "direct_answer_placement", "table_placement", "question_heading_placement",
            "z_numeric_evidence_total_density",
        ]
    ].notna().all(axis=1)
    rows["high_quality_extraction"] = (
        rows["content_strength"].astype(str).str.casefold().eq("strong")
        & rows["direct_answer_extraction_status"].isin(["success_present", "success_absent"])
        & rows["table_extraction_status"].isin(["success_present", "success_absent"])
        & rows["question_heading_extraction_status"].isin(["success_present", "success_absent"])
        & rows["numeric_evidence_extraction_status"].isin(["success_present", "success_absent"])
        & _numeric(rows["total_main_content_tokens"]).ge(100)
    )
    return rows


def _concentration(values: pd.Series) -> dict[str, float | int]:
    counts = values.dropna().astype(str).value_counts()
    total = counts.sum()
    if total == 0:
        return {
            "groups_represented": 0, "top_group_share": np.nan, "top_five_group_share": np.nan,
            "hhi": np.nan, "effective_groups": np.nan, "median_observations_per_group": np.nan,
            "singleton_observation_share": np.nan,
        }
    shares = counts / total
    hhi = float(np.square(shares).sum())
    return {
        "groups_represented": int(len(counts)),
        "top_group_share": float(shares.iloc[0]),
        "top_five_group_share": float(shares.iloc[:5].sum()),
        "hhi": hhi,
        "effective_groups": float(1 / hhi) if hhi else np.nan,
        "median_observations_per_group": float(counts.median()),
        "singleton_observation_share": float(counts[counts.eq(1)].sum() / total),
    }


def feature_coverage(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for feature in PLACEMENT_FEATURES:
        eligible = rows[rows[feature].notna()].copy()
        for category, group in eligible.groupby(feature, observed=True):
            n = len(group)
            cited = int(_numeric(group["cited"]).sum())
            low, high = _wilson(cited, n)
            output.append({
                "feature": feature, "feature_label": FEATURE_LABELS[feature],
                "category": str(category), "eligible_observations": len(eligible),
                "n_rows": n, "category_share": n / len(eligible) if len(eligible) else np.nan,
                "cited_rows": cited, "non_cited_rows": n - cited,
                "citation_rate": cited / n if n else np.nan,
                "ci_lower": low, "ci_upper": high,
                "n_domains": group["source_root_domain"].nunique(),
                "n_prompts": group["prompt_id"].nunique(),
                "sparse_n_lt_20": n < 20, "sparse_cited_lt_5": cited < 5,
                "sparse_non_cited_lt_5": n - cited < 5,
                "share_lt_5pct": n / len(eligible) < 0.05 if len(eligible) else True,
            })

    for raw_feature, model_feature in (
        ("numeric_evidence_total_density", "z_numeric_evidence_total_density"),
        ("numeric_evidence_early_share", "numeric_evidence_early_share"),
    ):
        measured = rows[rows[raw_feature].notna()].copy()
        if measured.empty:
            continue
        quantiles = measured[raw_feature].quantile([0, .25, .5, .75, .9, .95, .99, 1])
        for percentile, value in quantiles.items():
            output.append({
                "feature": model_feature,
                "feature_label": FEATURE_LABELS[model_feature],
                "category": f"percentile_{percentile:g}", "eligible_observations": len(measured),
                "n_rows": np.nan, "category_share": np.nan, "cited_rows": np.nan,
                "non_cited_rows": np.nan, "citation_rate": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n_domains": measured["source_root_domain"].nunique(),
                "n_prompts": measured["prompt_id"].nunique(), "numeric_value": value,
            })
    return pd.DataFrame(output)


def page_type_model_6_audit(rows: pd.DataFrame) -> pd.DataFrame:
    audit = rows.groupby(
        ["page_type_detailed", "page_type_model_6"], observed=True, dropna=False
    ).agg(
        n_rows=("cited", "size"), cited_rows=("cited", "sum"),
        unique_urls=("normalized_url", "nunique"),
        unique_domains=("source_root_domain", "nunique"),
    ).reset_index()
    audit["citation_rate"] = audit["cited_rows"] / audit["n_rows"]
    audit["mapping_version"] = POSITION_MODEL_VERSION
    return audit.sort_values(["page_type_model_6", "n_rows"], ascending=[True, False])


def source_type_model_6_audit(rows: pd.DataFrame) -> pd.DataFrame:
    audit = rows.groupby(
        ["source_type_detailed", "source_type_row_collapsed"], observed=True, dropna=False
    ).agg(
        n_rows=("cited", "size"), cited_rows=("cited", "sum"),
        unique_urls=("normalized_url", "nunique"),
        unique_domains=("source_root_domain", "nunique"),
    ).reset_index()
    audit["citation_rate"] = audit["cited_rows"] / audit["n_rows"]
    audit["domain_consensus_rule"] = "unique-URL modal class; exact top-class ties become other_or_unknown"
    audit["mapping_version"] = POSITION_MODEL_VERSION
    return audit.sort_values(["source_type_row_collapsed", "n_rows"], ascending=[True, False])


def source_type_domain_audit(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_root_domain", "source_type_model_6",
        "dominant_source_type_before_tie_rule", "dominant_url_share",
        "top_class_tie", "low_confidence_below_60pct", "domain_unique_urls",
        "domain_class_count", "candidate_summary",
    ]
    return rows[columns].drop_duplicates("source_root_domain").sort_values(
        ["low_confidence_below_60pct", "domain_unique_urls"], ascending=[False, False]
    )


def concentration_tables(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    selections: list[tuple[str, str, pd.DataFrame]] = []
    for feature in PLACEMENT_FEATURES:
        for category, group in rows[rows[feature].notna()].groupby(feature, observed=True):
            selections.append((feature, str(category), group))
    for raw_feature, analysis_feature in (
        ("numeric_evidence_total_density", "z_numeric_evidence_total_density"),
        ("numeric_evidence_early_share", "numeric_evidence_early_share"),
    ):
        values = _numeric(rows[raw_feature])
        cutoff = values.quantile(.9)
        selections.append((analysis_feature, "top_10_percent", rows[values.ge(cutoff)]))

    for feature, category, group in selections:
        for dimension, target, collector in (
            ("domain", "source_root_domain", domain_rows),
            ("prompt", "prompt_id", prompt_rows),
        ):
            metrics = _concentration(group[target])
            counts = group[target].astype(str).value_counts()
            max_share = metrics["top_group_share"]
            top_five = metrics["top_five_group_share"]
            effective = metrics["effective_groups"]
            raw_groups = metrics["groups_represented"]
            flags = []
            if pd.notna(max_share) and max_share > .25:
                flags.append("top_group_gt_25pct")
            if pd.notna(top_five) and top_five > .60:
                flags.append("top_five_gt_60pct")
            if raw_groups and pd.notna(effective) and effective < max(5, raw_groups * .20):
                flags.append("low_effective_group_count")
            collector.append({
                "feature": feature, "category": category, "dimension": dimension,
                "n_rows": len(group), **metrics,
                "concentration_flag": ";".join(flags) if flags else "not_flagged",
                "top_contributors": "; ".join(f"{name}:{count}" for name, count in counts.head(10).items()),
            })
    return pd.DataFrame(domain_rows), pd.DataFrame(prompt_rows)


def within_domain_variation(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for feature in PLACEMENT_FEATURES:
        eligible = rows[rows[feature].notna()].copy()
        group = eligible.groupby("source_root_domain", observed=True)
        sizes = group.size()
        placement_nunique = group[feature].nunique()
        outcome_nunique = group["cited"].nunique()
        presence = eligible[feature].ne(REFERENCE_LEVELS[feature]).astype(float)
        eligible = eligible.assign(_presence=presence)
        presence_nunique = eligible.groupby("source_root_domain")["_presence"].nunique()
        informative = (
            set(sizes[sizes.ge(2)].index.astype(str))
            & set(placement_nunique[placement_nunique.gt(1)].index.astype(str))
            & set(outcome_nunique[outcome_nunique.gt(1)].index.astype(str))
        )
        within = eligible["_presence"] - eligible.groupby("source_root_domain")["_presence"].transform("mean")
        between = eligible.groupby("source_root_domain")["_presence"].mean()
        domains_with_variation = int(placement_nunique.gt(1).sum())
        if domains_with_variation >= 30 and len(informative) >= 20:
            readiness = "Ready"
        elif domains_with_variation >= 15 and len(informative) >= 10:
            readiness = "Usable with caution"
        elif domains_with_variation > 0:
            readiness = "Weak within-domain variation"
        else:
            readiness = "Not suitable"
        output.append({
            "feature": feature, "eligible_rows": len(eligible),
            "domains_total": eligible["source_root_domain"].nunique(),
            "domains_with_at_least_two_pages": int(sizes.ge(2).sum()),
            "domains_with_presence_variation": int(presence_nunique.gt(1).sum()),
            "domains_with_early_late_variation": int(
                group[feature].apply(lambda values: values.astype(str).str.endswith(("early", "late")).nunique()).gt(1).sum()
            ),
            "domains_with_outcome_variation": int(outcome_nunique.gt(1).sum()),
            "domains_with_placement_and_outcome_variation": len(informative),
            "observations_in_informative_domains": int(
                eligible["source_root_domain"].isin(informative).sum()
            ),
            "within_domain_standard_deviation": float(within.std(ddof=0)),
            "between_domain_standard_deviation": float(between.std(ddof=0)),
            "fixed_effect_readiness": readiness,
            "readiness_rule": "Ready >=30 varying and >=20 informative; caution >=15 and >=10; otherwise weak/not suitable",
        })
    return pd.DataFrame(output)


def feature_audit(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    status_columns = {
        "direct_answer_placement": "direct_answer_extraction_status",
        "table_placement": "table_extraction_status",
        "question_heading_placement": "question_heading_extraction_status",
        "z_numeric_evidence_total_density": "numeric_evidence_extraction_status",
        "numeric_evidence_early_share": "numeric_evidence_extraction_status",
        "external_source_placement": "external_source_extraction_status",
    }
    for feature, status_column in status_columns.items():
        status = rows[status_column].fillna("ineligible").astype(str)
        counts = status.value_counts()
        measured = status.isin(["success_present", "success_absent"])
        output.append({
            "feature": feature, "n_rows": len(rows), "measured_rows": int(measured.sum()),
            "measured_rate": float(measured.mean()), "missing_or_unmeasured_rows": int((~measured).sum()),
            "missing_or_unmeasured_rate": float((~measured).mean()),
            **{f"status_{level}": int(counts.get(level, 0)) for level in STATUS_LEVELS},
            "manual_validation_precision": np.nan,
            "manual_validation_recall": np.nan,
            "manual_validation_status": (
                "Gemini semantic detections manually spot-checked; formal labeled precision/recall unavailable"
                if feature != "table_placement"
                else "HTML table presence previously audited; placement-specific labeled precision/recall unavailable"
            ),
        })
    return pd.DataFrame(output)


def sample_flow(rows: pd.DataFrame) -> pd.DataFrame:
    stages = [
        ("all_selected_surfaced_rows", pd.Series(True, index=rows.index)),
        ("controls_complete", rows[["cited", "prompt_id", "source_root_domain", "log_word_count", "page_type_model_6", "source_type_model_6"]].notna().all(axis=1)),
        ("gemini_semantic_measured", rows["direct_answer_placement"].notna() & rows["question_heading_placement"].notna()),
        ("table_placement_measured", rows["table_placement"].notna()),
        ("numeric_total_density_measured", rows["z_numeric_evidence_total_density"].notna()),
        ("numeric_early_share_measured", rows["numeric_evidence_early_share"].notna()),
        ("M5_joint_complete_case", rows["joint_position_features_measured"]),
        ("M5_high_quality_extraction", rows["joint_position_features_measured"] & rows["high_quality_extraction"]),
    ]
    output = []
    previous = len(rows)
    for order, (stage, mask) in enumerate(stages):
        group = rows[mask].copy()
        output.append({
            "stage_order": order, "stage": stage, "n_rows": len(group),
            "rows_lost_from_previous": previous - len(group),
            "cited_rows": int(_numeric(group["cited"]).sum()),
            "citation_rate": float(_numeric(group["cited"]).mean()) if len(group) else np.nan,
            "n_urls": group["normalized_url"].nunique(),
            "n_domains": group["source_root_domain"].nunique(),
            "n_prompts": group["prompt_id"].nunique(),
        })
        previous = len(group)
    return pd.DataFrame(output)


def manual_validation_examples(rows: pd.DataFrame) -> pd.DataFrame:
    """Return a small, purposive stored-evidence QA set reviewed for this run."""
    labels = [
        ("direct_answer_placement", "direct_answer_early", "https://9asset.com/article/ซื้อขายคอนโดต้องทำอย่างไร-คู่มือโอนกรรมสิทธิ์และกฎหมายที่ควรรู้-ครอบคลุมทั้งคนไทยและต่างชาติ-478", "pass", "Stored block directly answers the nearby definition question."),
        ("direct_answer_placement", "direct_answer_late", "https://amazingproperties.org/blogs/chidlom-bangkok-neighborhood-guide", "pass_with_context", "Stored block answers the nearby why-live-in-Chidlom question; context is required."),
        ("direct_answer_placement", "no_direct_answer", "https://108siam.com/en/condo-for-rent/thailand/bangkok/pathum-wan/lumphini/near-lumphini-park", "pass", "Stored main-content review found listing facts and a later question heading, but no concise direct-answer block."),
        ("direct_answer_placement", "unmeasured", "https://adamdecorcenter.com/vive-2", "pass", "Main-content extraction failed and the page is correctly unmeasured rather than absent."),
        ("table_placement", "table_early", "https://apthai.com/th/blog/homestory/high-rise-condo-vs-low-rise-condo", "pass", "Stored evidence is a substantive High Rise versus Low Rise comparison table."),
        ("table_placement", "table_late", "https://amazingproperties.org/search/bangkok/penthouse-for-rent/phrom-phong", "pass", "Stored evidence is a late rent/space table with multiple rows and columns."),
        ("table_placement", "no_table", "https://108siam.com/en/condo-for-rent/thailand/bangkok/pathum-wan/lumphini/near-lumphini-park", "pass", "Stored Markdown contains listing blocks but no table structure."),
        ("table_placement", "unmeasured", "https://adamdecorcenter.com/vive-2", "pass", "Verified table presence and position disagree under failed extraction, so the page is correctly ambiguous/unmeasured."),
        ("question_heading_placement", "question_heading_early", "https://9asset.com/article/ซื้อขายคอนโดต้องทำอย่างไร-คู่มือโอนกรรมสิทธิ์และกฎหมายที่ควรรู้-ครอบคลุมทั้งคนไทยและต่างชาติ-478", "pass", "Stored H2/H3 evidence is an explicit Thai question."),
        ("question_heading_placement", "question_heading_late", "https://108siam.com/en/condo-for-rent/thailand/bangkok/pathum-wan/lumphini/near-lumphini-park", "pass", "Stored H2/H3 evidence explicitly asks about average rental rates."),
        ("question_heading_placement", "no_question_heading", "https://1stopbangkok.net/sukhumvit-road", "pass", "Stored Markdown headings are declarative; body questions are not promoted to headings."),
        ("question_heading_placement", "unmeasured", "https://adamdecorcenter.com/vive-2", "pass", "Main-content failure is correctly unmeasured rather than no question heading."),
        ("numeric_evidence_total_density", "positive", "https://9asset.com/en/condo-project/hi-sukhumvit-bangchak-station-484133", "pass_with_caution", "Validated numeric evidence is normalized by all main-content tokens; repeated listing-template quantities remain a caveat."),
        ("numeric_evidence_total_density", "zero", "https://home.co.th/condo/lumpini-ville-chaengwatthana-pakkret-station-11924", "pass", "No validated numeric evidence is present, so total density is correctly zero."),
        ("numeric_evidence_total_density", "unmeasured", "https://adamdecorcenter.com/vive-2", "pass", "Main-content failure is correctly unmeasured rather than zero density."),
        ("numeric_evidence_early_share", "partial", "https://9asset.com/en/condo-project/hi-sukhumvit-bangchak-station-484133", "pass_with_caution", "One of three validated numeric-evidence blocks is in the first half."),
        ("numeric_evidence_early_share", "none_early", "https://108siam.com/en/condo-for-rent/thailand/bangkok/pathum-wan/lumphini/near-lumphini-park", "pass", "Validated numeric evidence exists, but all detected blocks begin in the second half."),
        ("numeric_evidence_early_share", "undefined_no_numeric_evidence", "https://home.co.th/condo/lumpini-ville-chaengwatthana-pakkret-station-11924", "pass", "The page has no validated numeric evidence, so early share is NaN rather than zero."),
        ("numeric_evidence_early_share", "unmeasured", "https://adamdecorcenter.com/vive-2", "pass", "Main-content failure leaves early share unmeasured."),
    ]
    evidence_columns = {
        "direct_answer_placement": "direct_answer_validated_evidence",
        "table_placement": "table_evidence",
        "question_heading_placement": "question_heading_validated_evidence",
        "numeric_evidence_total_density": "numeric_evidence_first_evidence",
        "numeric_evidence_early_share": "numeric_evidence_first_evidence",
    }
    status_columns = {
        "direct_answer_placement": "direct_answer_extraction_status",
        "table_placement": "table_extraction_status",
        "question_heading_placement": "question_heading_extraction_status",
        "numeric_evidence_total_density": "numeric_evidence_extraction_status",
        "numeric_evidence_early_share": "numeric_evidence_extraction_status",
    }
    urls = rows.drop_duplicates("normalized_url").set_index("normalized_url")
    output: list[dict[str, Any]] = []
    for feature, expected_state, url, result, note in labels:
        if url not in urls.index:
            output.append({
                "feature": feature, "expected_state": expected_state, "normalized_url": url,
                "manual_review_result": "unavailable_url_not_in_sample", "manual_review_note": note,
            })
            continue
        row = urls.loc[url]
        value = row.get(feature)
        if feature == "numeric_evidence_total_density":
            actual_state = "unmeasured" if pd.isna(value) else "positive" if float(value) > 0 else "zero"
        elif feature == "numeric_evidence_early_share":
            total = row.get("numeric_evidence_total_count")
            status = row.get("numeric_evidence_extraction_status")
            if pd.isna(value):
                actual_state = (
                    "undefined_no_numeric_evidence"
                    if status == "success_absent" and total == 0
                    else "unmeasured"
                )
            elif float(value) == 0:
                actual_state = "none_early"
            elif float(value) == 1:
                actual_state = "all_early"
            else:
                actual_state = "partial"
        else:
            actual_state = "unmeasured" if pd.isna(value) else str(value)
        output.append({
            "feature": feature, "expected_state": expected_state, "actual_state": actual_state,
            "normalized_url": url, "source_root_domain": row.get("source_root_domain"),
            "extraction_status": row.get(status_columns[feature]),
            "stored_evidence": row.get(evidence_columns[feature]),
            "manual_review_result": result if actual_state == expected_state else "fail_state_mismatch",
            "manual_review_note": note,
            "validation_design": "purposive_positive_negative_unmeasured_examples",
            "eligible_for_precision_recall": False,
        })
    return pd.DataFrame(output)


def cluster_support(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for dimension, column in (("domain", "source_root_domain"), ("prompt", "prompt_id")):
        counts = rows[column].astype(str).value_counts()
        output.append({
            "dimension": dimension, "n_clusters": len(counts),
            "median_observations": float(counts.median()), "minimum_observations": int(counts.min()),
            "maximum_observations": int(counts.max()), "singleton_cluster_share": float(counts.eq(1).mean()),
            "singleton_observation_share": float(counts[counts.eq(1)].sum() / counts.sum()),
        })
    return pd.DataFrame(output)


def select_main_se_method(support: pd.DataFrame) -> tuple[str, str]:
    indexed = support.set_index("dimension")
    adequate = all(
        indexed.loc[name, "n_clusters"] >= 30
        and indexed.loc[name, "singleton_observation_share"] <= .50
        for name in ("domain", "prompt")
    )
    if adequate:
        return (
            "two_way_cluster_domain_prompt",
            "Both dimensions have at least 30 clusters and singleton observations do not exceed 50%; finite-sample cluster correction applied.",
        )
    domain_clusters = indexed.loc["domain", "n_clusters"]
    prompt_clusters = indexed.loc["prompt", "n_clusters"]
    if domain_clusters >= prompt_clusters and domain_clusters >= 30:
        return "cluster_domain", "Prompt clustering support was weak; domain-clustered inference selected."
    if prompt_clusters >= 30:
        return "cluster_prompt", "Domain clustering support was weak; prompt-clustered inference selected."
    return "HC3", "Both cluster dimensions had limited support; HC3 selected and cluster inference flagged."


def _used_rows(result: Any, data: pd.DataFrame) -> pd.DataFrame:
    return data.loc[list(result.model.data.row_labels)].copy()


def _covariance(result: Any, data: pd.DataFrame, method: str) -> np.ndarray:
    used = _used_rows(result, data)
    if method == "HC3":
        return np.asarray(result.get_robustcov_results(cov_type="HC3").cov_params())
    if method == "cluster_domain":
        return np.asarray(result.get_robustcov_results(
            cov_type="cluster", groups=pd.factorize(used["source_root_domain"])[0], use_correction=True,
        ).cov_params())
    if method == "cluster_prompt":
        return np.asarray(result.get_robustcov_results(
            cov_type="cluster", groups=pd.factorize(used["prompt_id"])[0], use_correction=True,
        ).cov_params())
    if method == "two_way_cluster_domain_prompt":
        domain = pd.factorize(used["source_root_domain"])[0]
        prompt = pd.factorize(used["prompt_id"])[0]
        return np.asarray(cov_cluster_2groups(result, domain, prompt, use_correction=True)[0])
    raise ValueError(method)


def _fit_model(
    model_id: str,
    formula: str,
    data: pd.DataFrame,
    primary_se: str,
) -> tuple[pd.DataFrame, Any, pd.DataFrame]:
    model = smf.ols(formula, data=data, missing="drop")
    result = model.fit()
    used = _used_rows(result, data)
    rank = int(np.linalg.matrix_rank(result.model.exog))
    if rank != result.model.exog.shape[1]:
        raise ValueError(f"{model_id} model matrix is rank deficient: {rank}/{result.model.exog.shape[1]}")
    rows: list[dict[str, Any]] = []
    params = np.asarray(result.params)
    names = result.model.exog_names
    for method in SE_METHODS:
        try:
            covariance = _covariance(result, data, method)
            variance = np.diag(covariance)
            standard_error = np.sqrt(np.clip(variance, 0, None))
            standard_error[variance < -1e-10] = np.nan
            statistic = np.divide(
                params, standard_error, out=np.full_like(params, np.nan),
                where=np.isfinite(standard_error) & (standard_error > 0),
            )
            pvalues = 2 * scipy_stats.norm.sf(np.abs(statistic))
        except Exception as exc:
            warnings.warn(f"{model_id} {method} covariance failed: {exc}")
            standard_error = np.full_like(params, np.nan)
            pvalues = np.full_like(params, np.nan)
        for index, term in enumerate(names):
            se = standard_error[index]
            rows.append({
                "model_id": model_id, "formula": formula, "term": term,
                "estimate": params[index], "estimate_pp": params[index] * 100,
                "standard_error": se, "ci_lower": params[index] - 1.959963984540054 * se,
                "ci_upper": params[index] + 1.959963984540054 * se,
                "ci_lower_pp": (params[index] - 1.959963984540054 * se) * 100,
                "ci_upper_pp": (params[index] + 1.959963984540054 * se) * 100,
                "ci_width_pp": 2 * 1.959963984540054 * se * 100,
                "p_value": pvalues[index], "se_method": method,
                "is_primary_inference": method == primary_se,
                "n_obs": len(used), "n_cited": int(_numeric(used["cited"]).sum()),
                "n_domains": used["source_root_domain"].nunique(),
                "n_prompts": used["prompt_id"].nunique(),
                "domain_clusters": used["source_root_domain"].nunique(),
                "prompt_clusters": used["prompt_id"].nunique(),
                "adjusted_r_squared": result.rsquared_adj,
                "fixed_effects": "prompt_id", "model_matrix_rank": rank,
                "model_matrix_columns": result.model.exog.shape[1],
            })
    predictions = result.predict(used)
    predicted = pd.DataFrame({
        "model_id": model_id, "n_obs": [len(predictions)],
        "minimum_predicted_probability": [float(predictions.min())],
        "maximum_predicted_probability": [float(predictions.max())],
        "n_below_zero": [int((predictions < 0).sum())],
        "share_below_zero": [float((predictions < 0).mean())],
        "n_above_one": [int((predictions > 1).sum())],
        "share_above_one": [float((predictions > 1).mean())],
    })
    return pd.DataFrame(rows), result, predicted


def model_formulas(include_external: bool = False) -> dict[str, str]:
    controls = (
        "log_word_count + C(page_type_model_6, Treatment(reference='blog_guide_or_editorial')) + "
        "C(source_type_model_6, Treatment(reference='official_company_or_brand')) + C(prompt_id)"
    )
    direct = "C(direct_answer_placement, Treatment(reference='no_direct_answer'))"
    table = "C(table_placement, Treatment(reference='no_table'))"
    question = "C(question_heading_placement, Treatment(reference='no_question_heading'))"
    numeric = "z_numeric_evidence_total_density"
    formulas = {
        "M0": f"cited ~ {controls}",
        "M1": f"cited ~ {direct} + {controls}",
        "M2": f"cited ~ {table} + {controls}",
        "M3": f"cited ~ {question} + {controls}",
        "M4": f"cited ~ {numeric} + {controls}",
        "M5": f"cited ~ {direct} + {table} + {question} + {numeric} + {controls}",
    }
    if include_external:
        external = "C(external_source_placement, Treatment(reference='no_external_source'))"
        formulas["M6"] = formulas["M5"] + f" + {external}"
    return formulas


def _model_samples(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    controls = [
        "cited", "prompt_id", "source_root_domain", "log_word_count",
        "page_type_model_6", "source_type_model_6",
    ]
    complete_controls = rows[controls].notna().all(axis=1)
    return {
        "M0": rows[complete_controls].copy(),
        "M1": rows[complete_controls & rows["direct_answer_placement"].notna()].copy(),
        "M2": rows[complete_controls & rows["table_placement"].notna()].copy(),
        "M3": rows[complete_controls & rows["question_heading_placement"].notna()].copy(),
        "M4": rows[complete_controls & rows["z_numeric_evidence_total_density"].notna()].copy(),
        "M5": rows[complete_controls & rows["joint_position_features_measured"]].copy(),
        "M6": rows[
            complete_controls & rows["joint_position_features_measured"]
            & rows["external_source_placement"].notna()
        ].copy(),
    }


def external_source_eligibility(
    coverage: pd.DataFrame,
    domain_concentration: pd.DataFrame,
) -> tuple[bool, str]:
    categories = coverage[coverage["feature"].eq("external_source_placement")]
    concentration = domain_concentration[
        domain_concentration["feature"].eq("external_source_placement")
    ]
    gates = {
        "all_cells_n_ge_20": bool(len(categories) and categories["n_rows"].ge(20).all()),
        "all_cells_share_ge_5pct": bool(len(categories) and categories["category_share"].ge(.05).all()),
        "no_top_domain_gt_25pct": bool(len(concentration) and concentration["top_group_share"].le(.25).all()),
        "formal_manual_validation_available": False,
    }
    eligible = all(gates.values())
    reason = "; ".join(f"{name}={'pass' if value else 'fail'}" for name, value in gates.items())
    return eligible, reason


def multicollinearity_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rhs = (
        "1 + C(direct_answer_placement, Treatment(reference='no_direct_answer')) + "
        "C(table_placement, Treatment(reference='no_table')) + "
        "C(question_heading_placement, Treatment(reference='no_question_heading')) + "
        "z_numeric_evidence_total_density + log_word_count + "
        "C(page_type_model_6, Treatment(reference='blog_guide_or_editorial')) + "
        "C(source_type_model_6, Treatment(reference='official_company_or_brand'))"
    )
    design = dmatrix(rhs, data=data, return_type="dataframe")
    nonconstant = design.drop(columns=["Intercept"], errors="ignore")
    nonconstant = nonconstant.loc[:, nonconstant.nunique(dropna=True).gt(1)]
    standardized = (nonconstant - nonconstant.mean()) / nonconstant.std(ddof=0)
    condition = float(np.linalg.cond(standardized.to_numpy()))
    rows: list[dict[str, Any]] = []
    for column in nonconstant.columns:
        index = design.columns.get_loc(column)
        vif = float(variance_inflation_factor(design.to_numpy(), index))
        rows.append({
            "row_type": "vif", "variable": column, "related_variable": "",
            "association": np.nan, "vif": vif, "condition_number": condition,
            "warning": "serious_vif_gt_10" if vif > 10 else "caution_vif_gt_5" if vif > 5 else "not_flagged",
        })
    correlations = nonconstant.corr()
    for left_index, left in enumerate(correlations.columns):
        for right in correlations.columns[left_index + 1:]:
            value = correlations.loc[left, right]
            rows.append({
                "row_type": "pairwise_predictor_association", "variable": left,
                "related_variable": right, "association": value, "vif": np.nan,
                "condition_number": condition,
                "warning": (
                    "high_abs_association_ge_0.70" if abs(value) >= .70
                    else "moderate_abs_association_ge_0.30" if abs(value) >= .30
                    else "below_review_threshold"
                ),
            })
    return pd.DataFrame(rows)


def _focal_term(term: str) -> bool:
    tokens = (
        "direct_answer_placement", "table_placement", "question_heading_placement",
        "numeric_evidence_total_density", "external_source_placement",
    )
    return any(token in term for token in tokens)


def _leave_one_group_out(
    result: Any,
    used: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    names = result.model.exog_names
    keep_indices = [
        index for index, name in enumerate(names)
        if name != "Intercept" and not name.startswith("C(prompt_id)")
    ]
    x = pd.DataFrame(result.model.exog[:, keep_indices], columns=[names[index] for index in keep_indices], index=used.index)
    y = _numeric(used["cited"])
    focal = [column for column in x if _focal_term(column)]
    full_estimates = result.params.reindex(focal)
    output: list[dict[str, Any]] = []
    for removed, indices in used.groupby(group_column, observed=True).groups.items():
        mask = ~used.index.isin(indices)
        subset = used.loc[mask]
        if subset.empty:
            continue
        xs = x.loc[mask]
        ys = y.loc[mask]
        means_x = xs.groupby(subset["prompt_id"]).transform("mean")
        means_y = ys.groupby(subset["prompt_id"]).transform("mean")
        x_within = xs - means_x
        y_within = ys - means_y
        estimates, *_ = np.linalg.lstsq(x_within.to_numpy(), y_within.to_numpy(), rcond=None)
        mapping = dict(zip(xs.columns, estimates))
        for term in focal:
            estimate = mapping.get(term, np.nan)
            output.append({
                "influence_dimension": group_column, "removed_group": str(removed),
                "removed_rows": len(indices), "term": term, "estimate": estimate,
                "estimate_pp": estimate * 100, "full_M5_estimate": full_estimates.get(term),
                "change_pp": (estimate - full_estimates.get(term)) * 100,
                "sign_changed": bool(
                    np.isfinite(estimate) and np.isfinite(full_estimates.get(term))
                    and np.sign(estimate) != np.sign(full_estimates.get(term))
                ),
            })
    return pd.DataFrame(output)


def _fit_absorbed_domain_prompt_fe(
    data: pd.DataFrame,
    primary_se: str,
) -> pd.DataFrame:
    """Fit M5 slopes after absorbing prompt and domain effects by projection."""
    rhs = (
        "1 + C(direct_answer_placement, Treatment(reference='no_direct_answer')) + "
        "C(table_placement, Treatment(reference='no_table')) + "
        "C(question_heading_placement, Treatment(reference='no_question_heading')) + "
        "z_numeric_evidence_total_density + log_word_count"
    )
    design = dmatrix(rhs, data=data, return_type="dataframe").drop(columns="Intercept")
    working = data.loc[design.index].copy().reset_index(drop=True)
    design = design.reset_index(drop=True)
    matrix = np.column_stack([_numeric(working["cited"]).to_numpy(), design.to_numpy()])
    prompt_codes = pd.factorize(working["prompt_id"])[0]
    domain_codes = pd.factorize(working["source_root_domain"])[0]

    for _ in range(500):
        previous = matrix.copy()
        matrix -= pd.DataFrame(matrix).groupby(prompt_codes).transform("mean").to_numpy()
        matrix -= pd.DataFrame(matrix).groupby(domain_codes).transform("mean").to_numpy()
        if np.max(np.abs(matrix - previous)) < 1e-10:
            break
    y = matrix[:, 0]
    x = matrix[:, 1:]
    varying = np.std(x, axis=0) > 1e-10
    x = x[:, varying]
    names = design.columns[varying].tolist()
    result = sm.OLS(y, x).fit()
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError(f"absorbed R3 model matrix is rank deficient: {rank}/{x.shape[1]}")

    rows: list[dict[str, Any]] = []
    for method in SE_METHODS:
        if method == "HC3":
            covariance = np.asarray(result.get_robustcov_results(cov_type="HC3").cov_params())
        elif method == "cluster_domain":
            covariance = np.asarray(result.get_robustcov_results(
                cov_type="cluster", groups=domain_codes, use_correction=True,
            ).cov_params())
        elif method == "cluster_prompt":
            covariance = np.asarray(result.get_robustcov_results(
                cov_type="cluster", groups=prompt_codes, use_correction=True,
            ).cov_params())
        else:
            covariance = np.asarray(
                cov_cluster_2groups(result, domain_codes, prompt_codes, use_correction=True)[0]
            )
        variance = np.diag(covariance)
        standard_error = np.sqrt(np.clip(variance, 0, None))
        standard_error[variance < -1e-10] = np.nan
        statistic = np.divide(
            np.asarray(result.params), standard_error,
            out=np.full_like(np.asarray(result.params), np.nan),
            where=np.isfinite(standard_error) & (standard_error > 0),
        )
        pvalues = 2 * scipy_stats.norm.sf(np.abs(statistic))
        for index, term in enumerate(names):
            estimate = result.params[index]
            se = standard_error[index]
            rows.append({
                "model_id": "R3_domain_fixed_effects", "term": term,
                "estimate": estimate, "estimate_pp": estimate * 100,
                "standard_error": se, "ci_lower_pp": (estimate - 1.959963984540054 * se) * 100,
                "ci_upper_pp": (estimate + 1.959963984540054 * se) * 100,
                "p_value": pvalues[index], "se_method": method,
                "is_primary_inference": method == primary_se,
                "n_obs": len(working), "n_cited": int(_numeric(working["cited"]).sum()),
                "n_domains": working["source_root_domain"].nunique(),
                "n_prompts": working["prompt_id"].nunique(),
                "fixed_effects": "prompt_id + source_root_domain (absorbed)",
                "model_matrix_rank": rank, "model_matrix_columns": x.shape[1],
                "robustness_type": "domain_fixed_effects",
                "notes": (
                    "Page/source taxonomy controls omitted because domain FE absorb or nearly absorb "
                    "domain-level taxonomy variation."
                ),
            })
    return pd.DataFrame(rows)


def _robustness(
    rows: pd.DataFrame,
    m5_data: pd.DataFrame,
    primary_se: str,
    within: pd.DataFrame,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []

    def append_fit(
        model_id: str,
        formula: str,
        sample: pd.DataFrame,
        robustness_type: str,
    ) -> None:
        try:
            table, _, _ = _fit_model(model_id, formula, sample, primary_se)
            table["robustness_type"] = robustness_type
            results.append(table)
        except Exception as exc:
            results.append(pd.DataFrame([{
                "model_id": model_id, "term": "model_status", "estimate": np.nan,
                "robustness_type": robustness_type, "n_obs": len(sample),
                "notes": f"not estimable: {type(exc).__name__}: {exc}",
            }]))

    controls = (
        "log_word_count + C(page_type_model_6, Treatment(reference='blog_guide_or_editorial')) + "
        "C(source_type_model_6, Treatment(reference='official_company_or_brand')) + C(prompt_id)"
    )
    for feature, ratio in (
        ("direct_answer", "direct_answer_position_ratio"),
        ("table", "first_table_position_ratio"),
        ("question_heading", "first_question_heading_position_ratio"),
    ):
        sample = rows[_numeric(rows[ratio]).notna()].copy()
        if len(sample) >= 50 and sample["cited"].nunique() == 2:
            append_fit(
                f"R1_{feature}_continuous_position",
                f"cited ~ {ratio} + {controls}", sample,
                "continuous_position_conditional_on_feature_presence",
            )

        placement = f"{feature}_placement" if feature != "question_heading" else "question_heading_placement"
        ratio_values = _numeric(rows[ratio])
        quartile = pd.Series("No feature", index=rows.index, dtype="string")
        quartile.loc[ratio_values.between(0, .25, inclusive="left")] = "Q1"
        quartile.loc[ratio_values.between(.25, .5, inclusive="left")] = "Q2"
        quartile.loc[ratio_values.between(.5, .75, inclusive="left")] = "Q3"
        quartile.loc[ratio_values.between(.75, 1, inclusive="both")] = "Q4"
        quartile.loc[rows[placement].isna()] = pd.NA
        column = f"{feature}_position_quartile_model"
        sample = rows.assign(**{column: quartile}).dropna(subset=[column]).copy()
        cell = sample.groupby(column)["cited"].agg(["size", "sum"])
        if len(cell) == 5 and cell["size"].ge(20).all() and cell["sum"].ge(5).all() and (cell["size"] - cell["sum"]).ge(5).all():
            append_fit(
                f"R2_{feature}_quartile_position",
                f"cited ~ C({column}, Treatment(reference='No feature')) + {controls}",
                sample,
                "Q1_Q4_placement",
            )

    domain_ready = within.set_index("feature")["fixed_effect_readiness"]
    if all(domain_ready.get(feature) in {"Ready", "Usable with caution"} for feature in PLACEMENT_FEATURES[:3]):
        counts = m5_data.groupby("source_root_domain")["normalized_url"].nunique()
        supported = set(counts[counts.ge(2)].index)
        sample = m5_data[m5_data["source_root_domain"].isin(supported)].copy()
        try:
            results.append(_fit_absorbed_domain_prompt_fe(sample, primary_se))
        except Exception as exc:
            results.append(pd.DataFrame([{
                "model_id": "R3_domain_fixed_effects", "term": "model_status",
                "estimate": np.nan, "robustness_type": "domain_fixed_effects",
                "n_obs": len(sample), "notes": f"not estimable: {type(exc).__name__}: {exc}",
            }]))

    high_quality = m5_data[m5_data["high_quality_extraction"]].copy()
    if len(high_quality) >= 100:
        append_fit(
            "R5_high_quality_extraction", model_formulas()["M5"], high_quality,
            "high_quality_extraction_sample",
        )

    winsorized = m5_data.copy()
    formula = model_formulas()["M5"].replace(
        "z_numeric_evidence_total_density", "z_numeric_evidence_total_density_winsorized_p99"
    )
    append_fit(
        "R5B_numeric_density_winsorized_p99", formula, winsorized,
        "explicit_p99_winsorization",
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            logistic = smf.logit(model_formulas()["M5"], data=m5_data, missing="drop").fit(
                method="lbfgs", maxiter=200, disp=0,
            )
            marginal = logistic.get_margeff(at="overall").summary_frame()
        ame = marginal.reset_index().rename(columns={
            "index": "term", "dy/dx": "estimate", "Std. Err.": "standard_error",
            "Pr(>|z|)": "p_value", "Conf. Int. Low": "ci_lower", "Cont. Int. Hi.": "ci_upper",
        })
        ame["model_id"] = "R4_logit_average_marginal_effects"
        ame["estimate_pp"] = ame["estimate"] * 100
        ame["ci_lower_pp"] = ame.get("ci_lower", np.nan) * 100
        ame["ci_upper_pp"] = ame.get("ci_upper", np.nan) * 100
        ame["se_method"] = "delta_method_nonclustered"
        ame["robustness_type"] = "logit_average_marginal_effects"
        ame["n_obs"] = len(m5_data)
        results.append(ame)
    except Exception as exc:
        results.append(pd.DataFrame([{
            "model_id": "R4_logit_average_marginal_effects", "term": "model_status",
            "estimate": np.nan, "robustness_type": "logit_average_marginal_effects",
            "notes": f"not estimable: {type(exc).__name__}: {exc}", "n_obs": len(m5_data),
        }]))
    return pd.concat(results, ignore_index=True, sort=False) if results else pd.DataFrame()


def _ci_explanation(row: pd.Series) -> str:
    reasons: list[str] = []
    n = row.get("category_sample_size")
    cited = row.get("category_cited_count")
    non_cited = row.get("category_non_cited_count")
    if pd.notna(n) and n < 20:
        reasons.append(f"the category has only {int(n)} rows")
    if pd.notna(row.get("category_share")) and row["category_share"] < .05:
        reasons.append(f"the category represents only {row['category_share']:.1%} of eligible rows")
    if pd.notna(cited) and cited < 5:
        reasons.append(f"only {int(cited)} cited outcomes")
    if pd.notna(non_cited) and non_cited < 5:
        reasons.append(f"only {int(non_cited)} non-cited outcomes")
    if pd.notna(row.get("maximum_domain_share")) and row["maximum_domain_share"] > .25:
        reasons.append(f"{row['maximum_domain_share']:.0%} of rows come from one domain")
    if pd.notna(row.get("maximum_prompt_share")) and row["maximum_prompt_share"] > .25:
        reasons.append(f"{row['maximum_prompt_share']:.0%} of rows come from one prompt")
    if pd.notna(row.get("vif")) and row["vif"] > 5:
        reasons.append(f"VIF is {row['vif']:.1f}")
    if pd.notna(row.get("two_way_clustered_standard_error")) and pd.notna(row.get("hc3_standard_error")) and row["two_way_clustered_standard_error"] > row["hc3_standard_error"] * 1.5:
        reasons.append("two-way clustered uncertainty is much larger than HC3")
    if pd.notna(row.get("leave_one_domain_out_sign_changes")) and row["leave_one_domain_out_sign_changes"] > 0:
        reasons.append("the sign changes in leave-one-domain-out analysis")
    if not reasons:
        reasons.append("cell support, cluster support, concentration, and VIF are not individually severe")
    width = row.get("ci_width_pp")
    label = "wide" if pd.notna(width) and width >= 20 else "comparatively narrow"
    return f"This interval is {label} because " + "; ".join(reasons) + "."


def ci_diagnostics(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    domain: pd.DataFrame,
    prompt: pd.DataFrame,
    multicollinearity: pd.DataFrame,
    influence: pd.DataFrame,
    primary_se: str,
) -> pd.DataFrame:
    focal = results[results["term"].map(_focal_term)].copy()
    primary = focal[focal["se_method"].eq(primary_se)].copy()
    se_wide = focal.pivot_table(index=["model_id", "term"], columns="se_method", values="standard_error", aggfunc="first").reset_index()
    output = primary.merge(se_wide, on=["model_id", "term"], how="left", suffixes=("", "_se"))
    output = output.rename(columns={
        "HC3": "hc3_standard_error", "cluster_domain": "domain_clustered_standard_error",
        "cluster_prompt": "prompt_clustered_standard_error",
        "two_way_cluster_domain_prompt": "two_way_clustered_standard_error",
    })
    category_lookup: dict[str, str] = {}
    for term in output["term"]:
        match = re.search(r"\[T\.([^\]]+)\]", term)
        category_lookup[term] = match.group(1) if match else "continuous"
    output["category"] = output["term"].map(category_lookup)
    output["feature"] = output["term"].map(
        lambda term: next((name for name in [*PLACEMENT_FEATURES, "z_numeric_evidence_total_density"] if name in term), "")
    )
    cov_lookup = coverage.set_index(["feature", "category"])
    dom_lookup = domain.set_index(["feature", "category"])
    prompt_lookup = prompt.set_index(["feature", "category"])
    vif_rows = multicollinearity[multicollinearity["row_type"].eq("vif")]
    for index, row in output.iterrows():
        key = (row["feature"], row["category"])
        if key in cov_lookup.index:
            source = cov_lookup.loc[key]
            output.loc[index, "category_sample_size"] = source["n_rows"]
            output.loc[index, "category_share"] = source["category_share"]
            output.loc[index, "category_cited_count"] = source["cited_rows"]
            output.loc[index, "category_non_cited_count"] = source["non_cited_rows"]
            output.loc[index, "contributing_domains"] = source["n_domains"]
            output.loc[index, "contributing_prompts"] = source["n_prompts"]
        if key in dom_lookup.index:
            output.loc[index, "maximum_domain_share"] = dom_lookup.loc[key, "top_group_share"]
        if key in prompt_lookup.index:
            output.loc[index, "maximum_prompt_share"] = prompt_lookup.loc[key, "top_group_share"]
        matches = vif_rows[vif_rows["variable"].astype(str).map(lambda value: row["feature"] in value and (row["category"] == "continuous" or row["category"] in value))]
        if not matches.empty:
            output.loc[index, "vif"] = matches["vif"].max()
        influenced = influence[(influence["influence_dimension"].eq("source_root_domain")) & influence["term"].eq(row["term"])]
        if not influenced.empty:
            output.loc[index, "leave_one_domain_out_min_pp"] = influenced["estimate_pp"].min()
            output.loc[index, "leave_one_domain_out_max_pp"] = influenced["estimate_pp"].max()
            output.loc[index, "leave_one_domain_out_sign_changes"] = influenced["sign_changed"].sum()
            largest = influenced.loc[influenced["change_pp"].abs().idxmax()]
            output.loc[index, "largest_change_domain"] = largest["removed_group"]
            output.loc[index, "largest_change_pp"] = largest["change_pp"]
    output["grounded_ci_explanation"] = output.apply(_ci_explanation, axis=1)
    return output.sort_values("ci_width_pp", ascending=False)


def _write_findings(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    domain: pd.DataFrame,
    prompt: pd.DataFrame,
    within: pd.DataFrame,
    ci: pd.DataFrame,
    results: pd.DataFrame,
    robustness: pd.DataFrame,
    influence: pd.DataFrame,
    primary_se: str,
    external_reason: str,
) -> None:
    supported = coverage[
        coverage["n_rows"].notna()
        & ~coverage[["sparse_n_lt_20", "sparse_cited_lt_5", "sparse_non_cited_lt_5"]].fillna(False).any(axis=1)
    ]["feature"].drop_duplicates().tolist()
    imbalanced = coverage[coverage.get("share_lt_5pct", False).fillna(False)][["feature", "category"]]
    concentrated = domain[domain["concentration_flag"].ne("not_flagged")][["feature", "category", "concentration_flag"]]
    prompt_concentrated = prompt[prompt["concentration_flag"].ne("not_flagged")][["feature", "category", "concentration_flag"]]
    primary = results[results["is_primary_inference"] & results["term"].map(_focal_term)]
    m5 = primary[primary["model_id"].eq("M5")].copy()
    separate = primary[primary["model_id"].isin(["M1", "M2", "M3", "M4"])].set_index("term")
    joint_stability: list[str] = []
    inference_stability: list[str] = []
    influential: list[str] = []
    reportable: list[str] = []
    inconclusive: list[str] = []
    for row in m5.itertuples():
        separate_estimate = separate.loc[row.term, "estimate_pp"] if row.term in separate.index else np.nan
        same_sign = pd.notna(separate_estimate) and np.sign(separate_estimate) == np.sign(row.estimate_pp)
        joint_stability.append(f"{row.term}: {'same sign' if same_sign else 'sign changed'} from separate model to M5")
        variants = results[results["model_id"].eq("M5") & results["term"].eq(row.term)]
        excludes = ((variants["ci_lower_pp"] > 0) | (variants["ci_upper_pp"] < 0)).sum()
        inference_stability.append(f"{row.term}: CI excludes zero under {int(excludes)}/{len(variants)} SE methods")
        lodo = influence[
            influence["influence_dimension"].eq("source_root_domain")
            & influence["term"].eq(row.term)
        ]
        sign_changes = int(lodo["sign_changed"].sum()) if len(lodo) else 0
        max_change = float(lodo["change_pp"].abs().max()) if len(lodo) else np.nan
        influential.append(f"{row.term}: {sign_changes} LODO sign changes; maximum absolute change {max_change:.2f} pp")
        primary_excludes = row.ci_lower_pp > 0 or row.ci_upper_pp < 0
        if primary_excludes and same_sign and sign_changes == 0:
            reportable.append(row.term)
        else:
            inconclusive.append(row.term)

    logit = robustness[robustness["model_id"].eq("R4_logit_average_marginal_effects")]
    logit = logit[logit["term"].map(_focal_term)] if not logit.empty else logit
    logit_summary = [
        f"{row.term}: AME {row.estimate_pp:.2f} pp"
        for row in logit.itertuples()
        if pd.notna(getattr(row, "estimate_pp", np.nan))
    ]
    domain_fe = robustness[
        robustness["model_id"].eq("R3_domain_fixed_effects")
        & robustness.get("se_method", pd.Series(index=robustness.index)).eq(primary_se)
        & robustness["term"].map(_focal_term)
    ]
    domain_fe_summary = [
        f"{row.term}: {row.estimate_pp:.2f} pp (95% CI {row.ci_lower_pp:.2f}, {row.ci_upper_pp:.2f})"
        for row in domain_fe.itertuples()
    ]
    lines = [
        "POSITION MODEL FINDINGS",
        "=======================",
        "",
        "Research estimand: P(cited = 1 | source surfaced in this audit, measurable content).",
        "All estimates are adjusted observational associations, not causal effects.",
        "",
        f"Rows: {len(rows):,}; URLs: {rows.normalized_url.nunique():,}; domains: {rows.source_root_domain.nunique():,}; prompts: {rows.prompt_id.nunique():,}.",
        f"Primary inference: {primary_se}.",
        f"External-source M6 gate: {external_reason}.",
        f"Page-type control: page_type_model_6 ({rows.page_type_model_6.nunique()} classes); detailed Gemini labels preserved.",
        f"Source-type control: source_type_model_6 ({rows.source_type_model_6.nunique()} classes); detailed Gemini labels preserved.",
        f"Source domain-consensus audit: {rows.loc[rows.low_confidence_below_60pct, 'source_root_domain'].nunique()} low-confidence domains; {rows.loc[rows.top_class_tie, 'source_root_domain'].nunique()} tied domains assigned other_or_unknown.",
        "",
        f"1. Adequate category support: {', '.join(supported) if supported else 'none across every category'}.",
        "2. Imbalanced cells: " + (imbalanced.to_dict("records").__str__() if len(imbalanced) else "none flagged below 5%"),
        "3. Domain concentration flags: " + (concentrated.to_dict("records").__str__() if len(concentrated) else "none"),
        "4. Prompt concentration flags: " + (prompt_concentrated.to_dict("records").__str__() if len(prompt_concentrated) else "none"),
        "5. Adequate within-domain variation: " + within.set_index("feature")["fixed_effect_readiness"].to_dict().__str__(),
        "",
        "Primary M5 estimates (percentage points):",
    ]
    for row in primary[primary["model_id"].eq("M5")].itertuples():
        lines.append(
            f"- {row.term}: {row.estimate_pp:.2f} pp (95% CI {row.ci_lower_pp:.2f}, {row.ci_upper_pp:.2f}; p={row.p_value:.4g})."
        )
    lines.extend(["", "6. Coefficients with the widest confidence intervals and 7. evidence-based reasons:"])
    for row in ci.head(10).itertuples():
        lines.append(f"- {row.model_id} {row.term}: {row.grounded_ci_explanation}")
    lines.extend([
        "",
        "8. Influential-domain sensitivity:",
        *[f"- {item}" for item in influential],
        "",
        "9. Separate-model to M5 stability:",
        *[f"- {item}" for item in joint_stability],
        "",
        "10. Alternative-standard-error stability:",
        *[f"- {item}" for item in inference_stability],
        "",
        "11. Logistic-regression AME cross-check:",
        *([f"- {item}" for item in logit_summary] or ["- Logistic AMEs were not estimable."]),
        "",
        "12. Domain fixed effects feasibility:",
        "- Feasible after absorbing prompt and domain effects on supported multi-URL domains; page/source taxonomy controls are omitted because they are absorbed or nearly absorbed.",
        *[f"- {item}" for item in domain_fe_summary],
        "",
        "13. Findings sufficiently stable to report as adjusted associations (still non-causal):",
        *([f"- {item}" for item in reportable] or ["- None meet every primary stability screen."]),
        "",
        "14. Inconclusive findings:",
        *([f"- {item}" for item in inconclusive] or ["- None."]),
        "",
        "Interpretation boundary:",
        "A null or statistically insignificant result is retained. Moving a feature on a page is not proven to change citation probability.",
        "Domain/template, page-function, prompt, extraction, and selection confounding remain possible.",
        "The external-source detector is descriptive only because formal placement-specific manual validation is unavailable.",
    ])
    (output_dir / "POSITION_MODEL_FINDINGS.txt").write_text("\n".join(lines), encoding="utf-8")


def run_position_model(repo: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the complete isolated position analysis and write versioned artifacts."""
    repo = repo.resolve()
    output_dir = (output_dir or repo / DEFAULT_OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_position_model_dataset(repo)
    page_type_audit = page_type_model_6_audit(rows)
    source_type_audit = source_type_model_6_audit(rows)
    source_domain_audit = source_type_domain_audit(rows)
    coverage = feature_coverage(rows)
    domain, prompt = concentration_tables(rows)
    within = within_domain_variation(rows)
    audit = feature_audit(rows)
    manual_qa = manual_validation_examples(rows)
    flow = sample_flow(rows)
    samples = _model_samples(rows)
    support = cluster_support(samples["M5"])
    primary_se, se_reason = select_main_se_method(support)
    external_eligible, external_reason = external_source_eligibility(coverage, domain)

    formulas = model_formulas(include_external=external_eligible)
    result_frames: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    fitted: dict[str, tuple[Any, pd.DataFrame]] = {}
    for model_id, formula in formulas.items():
        table, result, predicted = _fit_model(model_id, formula, samples[model_id], primary_se)
        result_frames.append(table)
        predictions.append(predicted)
        fitted[model_id] = (result, _used_rows(result, samples[model_id]))
    results = pd.concat(result_frames, ignore_index=True)

    exploratory_family = results[
        results["model_id"].isin(["M1", "M2", "M3", "M4"])
        & results["is_primary_inference"] & results["term"].map(_focal_term)
    ].copy()
    results["bh_q_value"] = np.nan
    if not exploratory_family.empty:
        valid = exploratory_family["p_value"].notna()
        qvalues = multipletests(exploratory_family.loc[valid, "p_value"], method="fdr_bh")[1]
        results.loc[exploratory_family.loc[valid].index, "bh_q_value"] = qvalues

    multicollinearity = multicollinearity_diagnostics(samples["M5"])
    m5_result, m5_used = fitted["M5"]
    influence = pd.concat([
        _leave_one_group_out(m5_result, m5_used, "source_root_domain"),
        _leave_one_group_out(m5_result, m5_used, "prompt_id"),
    ], ignore_index=True)
    robustness = _robustness(rows, samples["M5"], primary_se, within)
    ci = ci_diagnostics(
        results, coverage, domain, prompt, multicollinearity, influence, primary_se,
    )

    model_wide = results[results["is_primary_inference"]].pivot_table(
        index="term", columns="model_id", values="estimate_pp", aggfunc="first",
    ).reset_index()

    rows.to_csv(output_dir / "position_model_dataset.csv", index=False)
    rows.to_parquet(output_dir / "position_model_dataset.parquet", index=False)
    audit.to_csv(output_dir / "position_model_feature_audit.csv", index=False)
    manual_qa.to_csv(output_dir / "position_model_manual_validation_examples.csv", index=False)
    coverage.to_csv(output_dir / "position_model_feature_coverage.csv", index=False)
    page_type_audit.to_csv(output_dir / "position_model_page_type_6_mapping.csv", index=False)
    source_type_audit.to_csv(output_dir / "position_model_source_type_6_mapping.csv", index=False)
    source_domain_audit.to_csv(output_dir / "position_model_source_type_domain_audit.csv", index=False)
    domain.to_csv(output_dir / "position_model_domain_concentration.csv", index=False)
    prompt.to_csv(output_dir / "position_model_prompt_concentration.csv", index=False)
    within.to_csv(output_dir / "position_model_within_domain_variation.csv", index=False)
    multicollinearity.to_csv(output_dir / "position_model_multicollinearity.csv", index=False)
    ci.to_csv(output_dir / "position_model_ci_diagnostics.csv", index=False)
    influence.to_csv(output_dir / "position_model_influence_diagnostics.csv", index=False)
    results.to_csv(output_dir / "position_model_results_long.csv", index=False)
    model_wide.to_csv(output_dir / "position_model_results_wide.csv", index=False)
    robustness.to_csv(output_dir / "position_model_robustness_results.csv", index=False)
    flow.to_csv(output_dir / "position_model_sample_flow.csv", index=False)
    support.to_csv(output_dir / "position_model_cluster_support.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "position_model_predicted_probability_diagnostics.csv", index=False
    )

    _write_findings(
        output_dir, rows, coverage, domain, prompt, within, ci, results,
        robustness, influence,
        primary_se, external_reason,
    )
    manifest = {
        "version": POSITION_MODEL_VERSION,
        "status": "complete",
        "research_question": (
            "Among surfaced webpages with measurable content, how is the presence and placement "
            "of selected content features associated with AI citation probability?"
        ),
        "causal_interpretation": False,
        "rows": len(rows), "urls": rows["normalized_url"].nunique(),
        "domains": rows["source_root_domain"].nunique(), "prompts": rows["prompt_id"].nunique(),
        "citation_rate": float(rows["cited"].mean()),
        "M5_rows": len(samples["M5"]), "M5_domains": samples["M5"]["source_root_domain"].nunique(),
        "M5_prompts": samples["M5"]["prompt_id"].nunique(),
        "primary_model": "M5", "primary_se_method": primary_se,
        "primary_se_reason": se_reason,
        "external_source_in_M6": external_eligible,
        "external_source_eligibility_reason": external_reason,
        "manual_validation_examples": len(manual_qa),
        "manual_validation_pass_or_caution": int(
            manual_qa["manual_review_result"].isin(["pass", "pass_with_context", "pass_with_caution"]).sum()
        ),
        "manual_validation_precision_recall_available": False,
        "numeric_evidence_primary_feature": "z_numeric_evidence_total_density",
        "numeric_evidence_total_density_mean": float(
            _numeric(rows["numeric_evidence_total_density"]).mean()
        ),
        "numeric_evidence_total_density_sample_sd": float(
            _numeric(rows["numeric_evidence_total_density"]).std(ddof=1)
        ),
        "numeric_evidence_position_extension": "numeric_evidence_early_share",
        "page_type_control": "page_type_model_6",
        "page_type_control_classes": sorted(rows["page_type_model_6"].unique().tolist()),
        "page_type_detailed_classes_preserved": int(rows["page_type_detailed"].nunique()),
        "source_type_control": "source_type_model_6",
        "source_type_control_classes": sorted(rows["source_type_model_6"].unique().tolist()),
        "source_type_detailed_classes_preserved": int(rows["source_type_detailed"].nunique()),
        "source_type_low_confidence_domains": int(
            source_domain_audit["low_confidence_below_60pct"].sum()
        ),
        "source_type_tied_domains": int(source_domain_audit["top_class_tie"].sum()),
        "old_D0_FE4_outputs_modified": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
