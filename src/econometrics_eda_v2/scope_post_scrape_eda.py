"""Post-scrape EDA helpers for the SCOPE condo citation study.

The functions in this module are deliberately observational.  They never use
answer text, source position, observed rank, or answer-derived similarity to
build taxonomy fields or the exploratory linear probability models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


POST_SCRAPE_FAMILIES = (
    "project_page",
    "listing_page",
    "developer_page",
    "aggregator_page",
    "blog_guide_news",
    "review_article",
    "directory_contact",
    "social_video_forum",
    "official_or_corporate",
    "unknown",
)

CONTENT_STRENGTH = ("strong", "medium", "weak", "failed")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold().isin({"true", "1", "1.0", "yes", "y"})


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, pd.Series(0, index=df.index)), errors="coerce").fillna(0.0)


def _parse_status_by_url(parse: pd.DataFrame) -> pd.DataFrame:
    """Return parser provenance keyed to the requested, rather than redirected, URL."""
    if parse.empty:
        return pd.DataFrame(columns=["normalized_url", "html_available", "markdown_available", "text_available"])
    work = parse.copy()
    if "requested_normalized_url" in work:
        requested = work["requested_normalized_url"].fillna("").astype(str).str.strip()
        work.loc[requested.ne(""), "normalized_url"] = requested[requested.ne("")]
    wanted = ["normalized_url", "html_available", "markdown_available", "text_available"]
    for col in wanted:
        if col not in work:
            work[col] = False if col != "normalized_url" else ""
    return work[wanted].drop_duplicates("normalized_url")


def _broad_page_family(row: pd.Series) -> tuple[str, str]:
    """Map the detailed real-estate label into the compact EDA taxonomy."""
    detail = _clean(row.get("page_type_detail_real_estate"))
    source_type = _clean(row.get("source_type_real_estate"))
    if detail in {"condo_project_page", "amenities_or_facilities_page", "floor_plan_page", "promotion_page"}:
        return "project_page", f"detail={detail}"
    if detail in {"resale_listing_page", "rental_listing_page", "broker_property_page"}:
        return "listing_page", f"detail={detail}"
    if detail == "developer_brand_page":
        return "developer_page", "detail=developer_brand_page"
    if detail == "project_listing_page":
        return "aggregator_page", "detail=project_listing_page"
    if detail in {"buying_guide", "investment_guide", "price_market_report", "neighborhood_guide", "location_transport_page", "news_press_release"}:
        return "blog_guide_news", f"detail={detail}"
    if detail in {"condo_review_page", "comparison_article"}:
        return "review_article", f"detail={detail}"
    if detail in {"contact_sales_page", "promotion_page"}:
        return "directory_contact", f"detail={detail}"
    if detail in {"forum_discussion", "video_page"}:
        return "social_video_forum", f"detail={detail}"
    if detail == "pdf_brochure":
        return "official_or_corporate", "detail=pdf_brochure"

    # Keep unknown as a valid outcome, but make an explicit domain-level family
    # when the detail is unavailable and the source itself is unambiguous.
    if source_type == "developer_official":
        return "developer_page", "domain=developer_official"
    if source_type == "project_official":
        return "project_page", "domain=project_official"
    if source_type in {"property_portal", "listing_marketplace"}:
        return "aggregator_page", f"domain={source_type}"
    if source_type == "broker_agency":
        return "listing_page", "domain=broker_agency"
    if source_type in {"real_estate_media", "condo_review_site"}:
        return "review_article", f"domain={source_type}"
    if source_type in {"investment_content", "neighborhood_guide_site", "news_media"}:
        return "blog_guide_news", f"domain={source_type}"
    if source_type in {"social_forum", "video_platform"}:
        return "social_video_forum", f"domain={source_type}"
    if source_type in {"government_or_regulatory", "pdf_document"}:
        return "official_or_corporate", f"domain={source_type}"
    return "unknown", "no_confident_detail_or_domain_signal"


def _domain_family(source_type: str) -> str:
    mapping = {
        "developer_official": "developer_or_project",
        "project_official": "developer_or_project",
        "property_portal": "property_aggregator",
        "listing_marketplace": "property_aggregator",
        "broker_agency": "broker_agency",
        "real_estate_media": "editorial_real_estate",
        "condo_review_site": "editorial_real_estate",
        "investment_content": "editorial_real_estate",
        "neighborhood_guide_site": "editorial_real_estate",
        "news_media": "news_media",
        "social_forum": "social_video_forum",
        "video_platform": "social_video_forum",
        "government_or_regulatory": "official_corporate",
        "pdf_document": "official_corporate",
        "map_or_transport_reference": "official_corporate",
    }
    return mapping.get(_clean(source_type), "other_or_unknown")


def prepare_post_scrape_eda(eda_ready: pd.DataFrame, parse: pd.DataFrame) -> pd.DataFrame:
    """Attach transparent scrape diagnostics and EDA-family taxonomy fields."""
    df = eda_ready.copy()
    parse_status = _parse_status_by_url(parse)
    if not parse_status.empty:
        df = df.merge(parse_status, on="normalized_url", how="left", suffixes=("", "_parse"))
        for col in ["html_available", "markdown_available", "text_available"]:
            parse_col = f"{col}_parse"
            if parse_col in df:
                existing = df[col] if col in df else pd.Series(False, index=df.index)
                df[col] = existing.fillna(df[parse_col])
                df = df.drop(columns=[parse_col])
    for col in ["html_available", "markdown_available", "text_available"]:
        if col not in df:
            df[col] = False
        df[col] = _as_bool(df[col]).astype(int)

    df["is_cited"] = _numeric(df, "cited").eq(1).astype(int)
    df["content_chars"] = _numeric(df, "text_char_count").astype(int)
    # The parser stores its extracted main text in text_char_count; retain the
    # separate name so downstream EDA does not imply an HTML-total measure.
    df["main_text_chars"] = df["content_chars"]
    parse_success = _as_bool(df.get("parse_success", pd.Series(False, index=df.index)))
    body_available = _as_bool(df.get("scraped_body_available", pd.Series(False, index=df.index)))
    quality = df.get("content_quality_flag", pd.Series("", index=df.index)).fillna("").astype(str)
    word_count = _numeric(df, "word_count")
    df["scraped_ok"] = (parse_success & body_available & quality.eq("ok")).astype(int)

    strength = pd.Series("failed", index=df.index, dtype=object)
    has_body = parse_success & body_available
    strength.loc[has_body] = "weak"
    strength.loc[has_body & quality.eq("ok") & word_count.ge(100)] = "medium"
    strength.loc[has_body & quality.eq("ok") & word_count.ge(300)] = "strong"
    df["content_strength"] = pd.Categorical(strength, categories=CONTENT_STRENGTH, ordered=True).astype(str)

    scrape_success = _as_bool(df.get("scrape_success", pd.Series(False, index=df.index)))
    error = pd.Series("none", index=df.index, dtype=object)
    error.loc[~scrape_success] = "scrape_failed"
    error.loc[scrape_success & ~parse_success] = "parse_failed"
    bad_quality = quality.ne("ok") & quality.ne("")
    error.loc[parse_success & bad_quality] = quality.loc[parse_success & bad_quality]
    df["scrape_error_type"] = error

    extraction = pd.Series("not_available", index=df.index, dtype=object)
    extraction.loc[parse_success & _as_bool(df["markdown_available"]) & _as_bool(df["text_available"])] = "markdown_and_text"
    extraction.loc[parse_success & _as_bool(df["markdown_available"]) & ~_as_bool(df["text_available"])] = "markdown_only"
    extraction.loc[parse_success & ~_as_bool(df["markdown_available"]) & _as_bool(df["text_available"])] = "text_only"
    extraction.loc[parse_success & ~_as_bool(df["markdown_available"]) & ~_as_bool(df["text_available"])] = "parsed_without_extractable_text"
    df["extraction_method"] = extraction

    extracted_features = _as_bool(df.get("content_feature_available", pd.Series(False, index=df.index)))
    df["content_features_extracted"] = extracted_features.astype(int)
    df["content_features_available"] = (extracted_features & df["content_strength"].isin(["strong", "medium"])).astype(int)

    if "page_type_family_real_estate" in df:
        df["page_type_family_real_estate_taxonomy_v1"] = df["page_type_family_real_estate"]
    families = df.apply(_broad_page_family, axis=1)
    df["page_type_family_real_estate"] = [item[0] for item in families]
    df["page_type_family_real_estate_reason"] = [item[1] for item in families]
    df["page_type_available"] = df["page_type_family_real_estate"].ne("unknown").astype(int)
    df["domain_family"] = df.get("source_type_real_estate", pd.Series("", index=df.index)).map(_domain_family)
    return df


def scrape_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    for label, group in df.groupby(["content_strength", "scrape_error_type", "extraction_method"], dropna=False, observed=True):
        rows.append(
            {
                "content_strength": label[0],
                "scrape_error_type": label[1],
                "extraction_method": label[2],
                "n_rows": int(len(group)),
                "share_rows": float(len(group) / total) if total else np.nan,
                "n_unique_urls": int(group["normalized_url"].nunique()),
                "cited_rate": float(group["is_cited"].mean()) if len(group) else np.nan,
                "median_main_text_chars": float(group["main_text_chars"].median()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["content_strength", "scrape_error_type", "extraction_method"])


def cited_vs_more_only_comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status, group in df.groupby("is_cited", dropna=False):
        rows.append(
            {
                "citation_status": "cited" if int(status) == 1 else "more_only",
                "n_rows": int(len(group)),
                "n_unique_urls": int(group["normalized_url"].nunique()),
                "scraped_ok_rate": float(group["scraped_ok"].mean()),
                "strong_content_rate": float(group["content_strength"].eq("strong").mean()),
                "content_features_available_rate": float(group["content_features_available"].mean()),
                "page_type_available_rate": float(group["page_type_available"].mean()),
                "unknown_page_type_rate": float(group["page_type_family_real_estate"].eq("unknown").mean()),
                "median_main_text_chars": float(group["main_text_chars"].median()),
            }
        )
    return pd.DataFrame(rows)


def page_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("page_type_family_real_estate", dropna=False, observed=True)
        .agg(
            n_rows=("is_cited", "size"),
            n_unique_urls=("normalized_url", "nunique"),
            cited_n=("is_cited", "sum"),
            cited_rate=("is_cited", "mean"),
            scraped_ok_rate=("scraped_ok", "mean"),
            strong_content_rate=("content_strength", lambda s: float(s.eq("strong").mean())),
        )
        .reset_index()
    )
    out["share_rows"] = out["n_rows"] / len(df) if len(df) else np.nan
    return out.sort_values("n_rows", ascending=False)


def availability_by(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    work = df.copy()
    if group_column not in work:
        work[group_column] = "missing"
    out = (
        work.groupby(group_column, dropna=False, observed=True)
        .agg(
            n_rows=("is_cited", "size"),
            n_unique_urls=("normalized_url", "nunique"),
            cited_rate=("is_cited", "mean"),
            scraped_ok_rate=("scraped_ok", "mean"),
            strong_content_rate=("content_strength", lambda s: float(s.eq("strong").mean())),
            medium_or_strong_content_rate=("content_strength", lambda s: float(s.isin(["medium", "strong"]).mean())),
            content_features_available_rate=("content_features_available", "mean"),
            page_type_available_rate=("page_type_available", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("n_rows", ascending=False)


def unknown_page_type_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    unknown = df[df["page_type_family_real_estate"].eq("unknown")].copy()
    if unknown.empty:
        return pd.DataFrame(columns=["diagnostic_dimension", "value", "n_rows", "share_unknown", "cited_rate"])
    rows = []
    for column in ["content_strength", "scrape_error_type", "domain_family", "source_type_real_estate"]:
        for value, group in unknown.groupby(column, dropna=False, observed=True):
            rows.append(
                {
                    "diagnostic_dimension": column,
                    "value": value,
                    "n_rows": int(len(group)),
                    "share_unknown": float(len(group) / len(unknown)),
                    "cited_rate": float(group["is_cited"].mean()),
                    "median_main_text_chars": float(group["main_text_chars"].median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["diagnostic_dimension", "n_rows"], ascending=[True, False])


def _ordered_categories(series: pd.Series, reference: str) -> list[str]:
    values = sorted(set(series.fillna(reference).astype(str)))
    return [reference] + [value for value in values if value != reference]


def _add_categorical_design(
    design: pd.DataFrame,
    series: pd.Series,
    prefix: str,
    reference: str,
) -> pd.DataFrame:
    values = series.fillna(reference).astype(str)
    categories = _ordered_categories(values, reference)
    cat = pd.Categorical(values, categories=categories)
    dummies = pd.get_dummies(cat, prefix=prefix, dtype=float)
    dummies.index = design.index
    reference_column = f"{prefix}_{reference}"
    if reference_column in dummies:
        dummies = dummies.drop(columns=[reference_column])
    # A one-level factor is intentionally omitted in restricted sensitivities.
    dummies = dummies.loc[:, dummies.nunique(dropna=False) > 1]
    return pd.concat([design, dummies], axis=1)


def run_exploratory_lpm(df: pd.DataFrame, scenario: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a robust, descriptive LPM with prompt fixed effects when estimable."""
    try:
        import statsmodels.api as sm
    except ModuleNotFoundError:
        return pd.DataFrame(), {"scenario": scenario, "model_status": "statsmodels_unavailable"}

    model_df = df.copy()
    required = ["is_cited", "page_type_family_real_estate", "scraped_ok", "content_strength", "domain_family"]
    for col in required:
        if col not in model_df:
            return pd.DataFrame(), {"scenario": scenario, "model_status": f"missing_{col}"}
    y = pd.to_numeric(model_df["is_cited"], errors="coerce")
    usable = y.notna()
    model_df = model_df.loc[usable].copy()
    y = y.loc[usable].astype(float)
    if len(model_df) < 30 or y.nunique() < 2:
        return pd.DataFrame(), {"scenario": scenario, "model_status": "insufficient_rows_or_outcome_variation", "n_rows": int(len(model_df))}

    design = pd.DataFrame(index=model_df.index)
    design["scraped_ok"] = pd.to_numeric(model_df["scraped_ok"], errors="coerce").fillna(0.0)
    design = _add_categorical_design(design, model_df["page_type_family_real_estate"], "page_type", "unknown")
    design = _add_categorical_design(design, model_df["content_strength"], "content_strength", "failed")
    design = _add_categorical_design(design, model_df["domain_family"], "domain_family", "other_or_unknown")
    prompt_fe_included = "prompt_id" in model_df and model_df["prompt_id"].nunique(dropna=True) > 1
    if prompt_fe_included:
        design = _add_categorical_design(design, model_df["prompt_id"], "prompt_fe", sorted(model_df["prompt_id"].dropna().astype(str).unique())[0])
    design = design.loc[:, design.nunique(dropna=False) > 1]
    design = sm.add_constant(design.astype(float), has_constant="add")
    try:
        model = sm.OLS(y, design).fit(cov_type="HC3")
    except Exception as exc:  # pragma: no cover - depends on data rank/pathologies
        return pd.DataFrame(), {"scenario": scenario, "model_status": f"fit_failed:{type(exc).__name__}", "n_rows": int(len(model_df))}

    terms = pd.DataFrame(
        {
            "scenario": scenario,
            "term": model.params.index,
            "coefficient": model.params.values,
            "robust_std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_95_low": model.conf_int().iloc[:, 0].values,
            "ci_95_high": model.conf_int().iloc[:, 1].values,
        }
    )
    terms["term_group"] = np.select(
        [
            terms["term"].eq("const"),
            terms["term"].eq("scraped_ok"),
            terms["term"].str.startswith("page_type_"),
            terms["term"].str.startswith("content_strength_"),
            terms["term"].str.startswith("domain_family_"),
            terms["term"].str.startswith("prompt_fe_"),
        ],
        ["intercept", "scrape_status", "page_type_family", "content_strength", "domain_family", "prompt_fixed_effect"],
        default="other",
    )
    summary = {
        "scenario": scenario,
        "model_status": "ok",
        "n_rows": int(model.nobs),
        "cited_rate": float(y.mean()),
        "r_squared": float(model.rsquared),
        "adjusted_r_squared": float(model.rsquared_adj),
        "n_parameters": int(len(model.params)),
        "prompt_fixed_effects_included": bool(prompt_fe_included),
        "topic_fixed_effects_included": False,
        "topic_fixed_effects_note": "single SCOPE topic; topic FE is not estimable",
        "page_type_reference": "unknown",
        "content_strength_reference": "failed",
        "domain_family_reference": "other_or_unknown",
    }
    return terms, summary


def sensitivity_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all_urls": df.copy(),
        "scraped_ok_only": df[df["scraped_ok"].eq(1)].copy(),
        "strong_content_only": df[df["content_strength"].eq("strong")].copy(),
        "exclude_unknown_page_types": df[df["page_type_family_real_estate"].ne("unknown")].copy(),
        "unknown_own_category": df.copy(),
    }


def sensitivity_descriptive_summary(subsets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for scenario, data in subsets.items():
        rows.append(
            {
                "scenario": scenario,
                "n_rows": int(len(data)),
                "n_unique_urls": int(data["normalized_url"].nunique()),
                "cited_rate": float(data["is_cited"].mean()) if len(data) else np.nan,
                "scraped_ok_rate": float(data["scraped_ok"].mean()) if len(data) else np.nan,
                "strong_content_rate": float(data["content_strength"].eq("strong").mean()) if len(data) else np.nan,
                "unknown_page_type_rate": float(data["page_type_family_real_estate"].eq("unknown").mean()) if len(data) else np.nan,
                "content_features_available_rate": float(data["content_features_available"].mean()) if len(data) else np.nan,
            }
        )
    return pd.DataFrame(rows)
