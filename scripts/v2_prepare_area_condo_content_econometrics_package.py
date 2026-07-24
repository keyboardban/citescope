#!/usr/bin/env python3
"""Prepare an AI-ready package for Area Condo content econometrics planning."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded"
DEFAULT_MASTER = BASE / "tables/area_condo_final_pre_lpm_master/area_condo_lpm_ready_final_pre_lpm_master.csv"
DEFAULT_URLS = BASE / "tables/area_condo_lpm_prep/area_condo_url_taxonomy.csv"
DEFAULT_PROMPTS = BASE / "tables/area_condo_lpm_prep/prompt_manifest_join_audit.csv"
DEFAULT_OUTPUT = BASE / "content_econometrics_ai_package"

LEAKAGE_CHECKS = (
    ("answer_derived_similarity", ("answer_similarity", "page_answer_similarity", "max_chunk_answer_similarity")),
    ("page_answer_similarity", ("page_answer_similarity",)),
    ("max_chunk_answer_similarity", ("max_chunk_answer_similarity",)),
    ("answer_overlap", ("answer_overlap",)),
    ("answer_like_text", ("answer_like_text",)),
    ("brand_appeared_in_answer", ("brand_appeared_in_answer",)),
    ("cited_label_as_predictor", ("cited_label",)),
    ("source_group", ("source_group",)),
    ("source_origin", ("source_origin",)),
    ("source_position_or_observed_rank", ("source_position", "observed_rank")),
    ("domain_citation_rate_or_proxy", ("domain_citation_rate", "citation_rate", "cited_rate")),
)

TAXONOMY_COLLAPSE_FEATURES = (
    "page_type_url_seed_general",
    "page_type_family_general",
    "site_type_general",
)

SPARSE_AUDIT_FEATURES = (
    "page_type_url_seed_general",
    "page_type_family_general",
    "site_type_general",
    "heading_count_group",
    "link_count_group",
    "word_count_group",
    "content_strength",
    "intent",
)

OUTLIER_FEATURES = (
    "word_count",
    "content_chars",
    "heading_count",
    "link_count",
    "table_count",
    "heading_density_per_1000_words",
    "link_density_per_1000_words",
    "table_density_per_1000_words",
)


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.casefold().isin({"1", "1.0", "true", "yes", "y"})


def _prepare_rows(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["cited"] = _bool(df["cited"]).astype(int)
    for column in (
        "scrape_success",
        "content_feature_available",
        "has_table",
        "has_headings",
        "has_links",
        "has_substantial_text",
        "has_multiple_tables",
        "taxonomy_confidence_high_or_medium",
        "page_type_general_confidence_high_or_medium",
    ):
        if column in df:
            df[column] = _bool(df[column])
    for column in ("word_count", "content_chars", "heading_count", "table_count", "link_count"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["log2_word_count_plus1"] = np.log2(df["word_count"].clip(lower=0) + 1)
    df["log2_content_chars_plus1"] = np.log2(df["content_chars"].clip(lower=0) + 1)
    df["heading_density_per_1000_words"] = (
        1000 * df["heading_count"] / df["word_count"].replace(0, np.nan)
    )
    df["link_density_per_1000_words"] = (
        1000 * df["link_count"] / df["word_count"].replace(0, np.nan)
    )
    df["table_density_per_1000_words"] = (
        1000 * df["table_count"] / df["word_count"].replace(0, np.nan)
    )
    content = df[_bool(df["content_feature_available"])]
    word_p99 = float(content["word_count"].quantile(0.99))
    link_p99 = float(content["link_count"].quantile(0.99))
    df["word_count_top_1pct"] = df["word_count"].gt(word_p99)
    df["link_count_top_1pct"] = df["link_count"].gt(link_p99)
    df["log2_word_count_plus1_winsorized_p99"] = np.log2(df["word_count"].clip(upper=word_p99) + 1)
    df["low_link_count"] = df["link_count"].lt(9)
    prompt_stats = df.groupby("prompt_id")["cited"].agg(prompt_source_rows="size", prompt_cited_rows="sum")
    prompt_stats["prompt_has_outcome_variation"] = (
        prompt_stats.prompt_cited_rows.gt(0)
        & prompt_stats.prompt_cited_rows.lt(prompt_stats.prompt_source_rows)
    )
    domain_stats = df.groupby("source_root_domain").agg(
        domain_source_rows=("cited", "size"),
        domain_unique_urls=("normalized_url", "nunique"),
    )
    url_stats = df.groupby("normalized_url").agg(
        url_source_rows=("cited", "size"),
        url_unique_prompts=("prompt_id", "nunique"),
        url_cited_rows=("cited", "sum"),
        url_cited_rate=("cited", "mean"),
    )
    df = df.merge(prompt_stats, left_on="prompt_id", right_index=True, how="left", validate="many_to_one")
    df = df.merge(domain_stats, left_on="source_root_domain", right_index=True, how="left", validate="many_to_one")
    df = df.merge(url_stats, left_on="normalized_url", right_index=True, how="left", validate="many_to_one")
    return df


def _category_counts(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = df[feature].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
    work = df.assign(_category=values)
    summary = work.groupby("_category", dropna=False).agg(
        n_rows=("cited", "size"),
        cited_rows=("cited", "sum"),
        unique_prompts=("prompt_id", "nunique"),
    ).reset_index().rename(columns={"_category": "category"})
    summary["more_only_rows"] = summary.n_rows - summary.cited_rows
    summary["cited_rate"] = summary.cited_rows / summary.n_rows
    summary["sparse_flag"] = summary.n_rows.lt(20)
    summary["unstable_flag"] = summary.cited_rows.lt(5) | summary.more_only_rows.lt(5)
    summary["perfect_prediction_flag"] = summary.cited_rate.isin([0.0, 1.0])
    return summary


def _apply_taxonomy_collapses(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    content = output[_bool(output["content_feature_available"])]
    for feature in TAXONOMY_COLLAPSE_FEATURES:
        summary = _category_counts(content, feature)
        collapse = set(
            summary.loc[
                summary.category.ne("unknown")
                & (summary.sparse_flag | summary.unstable_flag | summary.perfect_prediction_flag),
                "category",
            ]
        )
        values = output[feature].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
        output[f"{feature}_collapsed"] = values.where(~values.isin(collapse), "rare_other")
    return output


def _sparse_category_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    taxonomy = set(TAXONOMY_COLLAPSE_FEATURES)
    for feature in SPARSE_AUDIT_FEATURES:
        summary = _category_counts(df, feature)
        for record in summary.to_dict("records"):
            category = str(record["category"])
            if category == "unknown":
                action = "keep_unknown_as_own_category"
            elif feature in taxonomy and (
                record["sparse_flag"] or record["unstable_flag"] or record["perfect_prediction_flag"]
            ):
                action = "collapse_to_rare_other"
            elif feature in {"heading_count_group", "link_count_group", "word_count_group"} and (
                record["sparse_flag"] or record["unstable_flag"] or record["perfect_prediction_flag"]
            ):
                action = "merge_adjacent_bin_or_use_sensitivity_only"
            elif feature == "content_strength":
                action = "keep_as_extraction_quality_control_and_run_strong_only_sensitivity"
            elif feature == "intent":
                action = "descriptive_or_limited_interactions_only_prompt_FE_absorbs_main_effect"
            else:
                action = "keep"
            rows.append({"feature": feature, **record, "recommended_action": action})
    return pd.DataFrame(rows).sort_values(["feature", "n_rows", "category"], ascending=[True, False, True])


def _outlier_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in OUTLIER_FEATURES:
        raw = pd.to_numeric(df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = raw.dropna()
        p99 = float(values.quantile(0.99))
        above = values.gt(p99)
        rows.append(
            {
                "feature": feature,
                "n_rows": len(df),
                "n_available": int(values.size),
                "n_missing_or_nonfinite": int(raw.isna().sum()),
                "minimum": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "p99_threshold": p99,
                "maximum": float(values.max()),
                "n_above_p99": int(above.sum()),
                "share_above_p99": float(above.mean()),
                "max_to_p99_ratio": float(values.max() / p99) if p99 > 0 else np.nan,
                "recommended_sensitivity": (
                    "remove values above p99 and compare focal coefficients"
                    if feature in {"word_count", "link_count"}
                    else "inspect p99 tail; use transformed/grouped form rather than raw count"
                ),
            }
        )
    return pd.DataFrame(rows)


def _model_columns(df: pd.DataFrame) -> list[str]:
    requested = [
        "record_id",
        "prompt_id",
        "intent",
        "area_tag",
        "expansion_group",
        "normalized_url",
        "source_url",
        "source_root_domain",
        "cited",
        "scrape_success",
        "content_feature_available",
        "content_strength",
        "content_quality_flag",
        "word_count",
        "content_chars",
        "heading_count",
        "table_count",
        "link_count",
        "log2_word_count_plus1",
        "log2_word_count_plus1_winsorized_p99",
        "log2_content_chars_plus1",
        "heading_density_per_1000_words",
        "link_density_per_1000_words",
        "table_density_per_1000_words",
        "has_table",
        "has_headings",
        "has_links",
        "has_substantial_text",
        "has_multiple_tables",
        "low_link_count",
        "word_count_top_1pct",
        "link_count_top_1pct",
        "heading_count_group",
        "link_count_group",
        "word_count_group",
        "site_type_general",
        "site_type_general_collapsed",
        "page_type_url_seed_general",
        "page_type_url_seed_general_collapsed",
        "page_type_family_general",
        "page_type_family_general_collapsed",
        "page_type_general_common",
        "page_type_general_confidence",
        "page_type_general_confidence_high_or_medium",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "taxonomy_confidence_high_or_medium",
        "prompt_source_rows",
        "domain_source_rows",
        "domain_unique_urls",
        "url_source_rows",
        "url_unique_prompts",
    ]
    return [column for column in requested if column in df]


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        ("cited", "outcome", "binary", "1 if the surfaced source was cited; 0 if more-only", "all surfaced rows", "Dependent variable, not a predictor", "Observed citation among surfaced sources, not web-wide citation probability"),
        ("log2_word_count_plus1", "focal content feature", "continuous log2", "Page text length", "content_feature_available=true", "Coefficient is the percentage-point association for approximately doubling word count", "Do not include with log2_content_chars_plus1 in the same preferred model"),
        ("has_table", "focal content feature", "binary", "Any extracted HTML table", "content_feature_available=true", "Coefficient is the adjusted cited-rate difference for pages with a table", "Use instead of raw table_count because table_count is zero-inflated"),
        ("heading_count_group", "focal content feature", "categorical", "0-1, 2-6, 7-12, 13+ headings", "content_feature_available=true", "Each coefficient is a percentage-point difference from the selected reference group", "No monotonic effect should be assumed"),
        ("link_count_group", "focal content feature", "categorical", "0-3, 4-8, 9+ extracted links", "content_feature_available=true", "Each coefficient is relative to the selected reference group", "May proxy site template or navigation rather than writing quality"),
        ("low_link_count", "link-count sensitivity feature", "binary", "1 when extracted link_count is below 9", "content_feature_available=true", "Optional binary sensitivity contrast for the highly imbalanced three-level link group", "The low-link group is small and its coefficient should be interpreted cautiously"),
        ("log2_word_count_plus1_winsorized_p99", "outlier sensitivity feature", "continuous log2", "Page length after capping word count at the measurable-sample 99th percentile", "content_feature_available=true", "Compare with the preferred log2 word-count coefficient to assess tail influence", "Sensitivity form only; report the p99 threshold"),
        ("heading_density_per_1000_words", "alternative focal feature", "continuous", "Headings relative to page length", "content_feature_available=true and word_count>0", "Association for one additional heading per 1,000 words", "Prefer over raw heading count only after checking nonlinear shape"),
        ("link_density_per_1000_words", "alternative focal feature", "continuous", "Links relative to page length", "content_feature_available=true and word_count>0", "Association for one additional link per 1,000 words", "Can remain template-driven"),
        ("content_strength", "measurement-quality control", "categorical", "Strong, medium, weak, failed extraction strength", "all surfaced rows", "Controls extraction quality; not a writing-style effect", "Strong-only sample is the more transparent sensitivity check"),
        ("content_quality_flag", "measurement-quality diagnostic", "categorical", "Parse failure, short text, blocked page, boilerplate, or okay", "all surfaced rows", "Use for missingness/quality diagnostics or sensitivity controls", "It partly reflects scraper behavior, not the page's intended writing"),
        ("content_feature_available", "selection indicator", "binary", "Content metrics can be measured", "all surfaced rows", "Use in availability analysis; do not treat unavailable content features as zero", "Missingness is plausibly non-random"),
        ("prompt_id", "fixed effect", "categorical ID", "Exact prompt identity", "all surfaced rows", "Absorbs all characteristics shared by sources within the same prompt", "Intent and area main effects cannot be estimated alongside prompt fixed effects"),
        ("source_root_domain", "fixed effect or cluster", "categorical ID", "Root domain", "content models", "Domain fixed effects compare URLs within a domain; clustering allows arbitrary domain correlation", "URL fixed effects cannot identify URL-level content features"),
        ("normalized_url", "cluster ID", "categorical ID", "Canonical page URL", "content models", "Cluster by URL because content features repeat across prompts", "Do not include URL fixed effects in a content-feature model"),
        ("site_type_general", "confounding sensitivity control", "categorical", "Broad website class", "all surfaced rows", "Tests whether content associations persist after broad source-class adjustment", "Not a focal client recommendation"),
        ("site_type_general_collapsed", "collapsed confounding sensitivity control", "categorical", "Broad website class with sparse, unstable, or separating levels combined as rare_other", "content_feature_available=true", "Preferred site-type sensitivity form if included", "Unknown remains its own category"),
        ("page_type_url_seed_general", "rule-v2 robustness comparison", "categorical", "Page function inferred without scraped body text", "all surfaced rows", "Retained for comparison with Gemini taxonomy", "Not the preferred taxonomy classification"),
        ("page_type_url_seed_general_collapsed", "rule-v2 robustness comparison only", "categorical", "URL-seed page function with sparse levels combined as rare_other", "content_feature_available=true", "Use in M4R, not M4", "Unknown remains its own category"),
        ("page_type_general_gemini_v1", "descriptive Gemini detailed taxonomy", "categorical", "Detailed page function classified by Gemini 3.1 Flash Lite", "all surfaced rows after URL join", "Use for descriptive review; prefer family level in LPM", "May use scraped body content and can over-control writing features"),
        ("page_type_family_gemini_v1", "Gemini taxonomy sensitivity control", "categorical", "Broad Gemini-classified page-function family", "all surfaced rows after URL join", "Source for collapsed M4 page-family control", "Secondary to M1/M2 because classification may use scraped content"),
        ("page_type_family_gemini_v1_collapsed", "preferred page-function sensitivity control", "categorical", "Gemini page-function family with low-support levels collapsed", "content_feature_available=true", "Use jointly with source_type_general_gemini_v1_collapsed in M4", "Unknown remains explicit; rare collapse uses support only, not citation outcome"),
        ("source_type_general_gemini_v1", "Gemini source/site taxonomy sensitivity control", "categorical", "Website or publisher role classified by Gemini", "all surfaced rows after URL join", "Source for collapsed M4 source/site-type control", "Publisher role remains observational and may correlate with domain authority"),
        ("source_type_general_gemini_v1_collapsed", "preferred source/site-type sensitivity control", "categorical", "Gemini source/site type with low-support levels collapsed", "content_feature_available=true", "Use jointly with page_type_family_gemini_v1_collapsed in M4", "Not a focal writing feature"),
        ("taxonomy_confidence_gemini_v1", "diagnostic taxonomy quality", "categorical", "Gemini classification confidence", "all joined rows", "Use for QA and confidence-restricted sensitivity only", "Model-reported confidence is not calibrated probability"),
        ("page_type_family_general", "taxonomy sensitivity control", "categorical", "Broad final page-function family", "all surfaced rows", "Tests robustness to broad page-function adjustment", "Final taxonomy may use scraped content and can over-control the focal construct"),
        ("page_type_family_general_collapsed", "collapsed taxonomy sensitivity control", "categorical", "Broad final page-function family with sparse, unstable, or separating levels combined as rare_other", "content_feature_available=true", "Use only as a secondary sensitivity alternative", "Final taxonomy may use scraped content"),
        ("page_type_general_common", "descriptive taxonomy", "categorical", "FAQ, contact, landing, listing, and other familiar page functions", "all surfaced rows", "Useful for sample description and stratified checks", "Not the primary content-effect result"),
        ("intent", "descriptive stratum or interaction", "categorical", "Prompt intent group", "all surfaced rows", "Use only for descriptive tables or pre-specified feature interactions", "Main effect is absorbed by prompt fixed effects"),
        ("word_count", "diagnostic raw measure", "continuous count", "Extracted words", "content_feature_available=true", "Use for distributions and thresholds", "Heavy-tailed; do not use raw in the preferred linear specification"),
        ("heading_count", "diagnostic raw measure", "continuous count", "Extracted headings", "content_feature_available=true", "Use for shape checks", "Observed relationship is not clearly linear"),
        ("table_count", "diagnostic raw measure", "continuous count", "Extracted tables", "content_feature_available=true", "Use for descriptive diagnostics", "Nearly redundant with has_table and strongly zero-inflated"),
        ("link_count", "diagnostic raw measure", "continuous count", "Extracted links", "content_feature_available=true", "Use for distributions and nonlinear diagnostics", "Can be dominated by site templates"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "variable_name",
            "econometric_role",
            "variable_form",
            "construct",
            "analysis_sample",
            "lpm_use_and_interpretation",
            "main_caveat",
        ],
    )


def _proposed_features() -> pd.DataFrame:
    rows = [
        ("sentence_count", "structure", "count", "Number of sentence-like units", "page text only", "diagnostic then log/threshold", "Thai sentence segmentation requires validation"),
        ("paragraph_count", "structure", "count", "Number of content paragraphs", "HTML/markdown blocks", "log2 or threshold", "Extraction must preserve paragraph boundaries"),
        ("avg_sentence_length", "clarity", "continuous", "Average tokens per sentence", "page text only", "spline or bins", "Language-specific segmentation"),
        ("avg_paragraph_words", "clarity", "continuous", "Average words per paragraph", "paragraph blocks", "spline or bins", "Template text can distort it"),
        ("heading_density", "structure", "continuous", "Headings per 1,000 words", "headings and text", "continuous plus nonlinear check", "May proxy page type"),
        ("list_item_count", "structure", "count", "Bulleted and numbered list items", "HTML/markdown structure", "binary plus log2 count", "Needs list-preserving extraction"),
        ("list_item_density", "structure", "continuous", "List items per 1,000 words", "HTML/markdown structure", "continuous", "Highly correlated with list_item_count"),
        ("question_heading_count", "answerability", "count", "Headings ending in a question or FAQ-like wording", "heading text", "binary plus count group", "Avoid using answer text"),
        ("faq_schema_flag", "answerability", "binary", "FAQPage structured data", "structured data", "binary", "Schema can be present without visible useful answers"),
        ("opening_summary_flag", "answerability", "binary", "Opening section gives a concise summary before detail", "first content block", "binary", "Requires transparent deterministic rule"),
        ("definition_pattern_flag", "answerability", "binary", "Page begins with an explicit definition/explanation", "first content block", "binary", "Language-specific phrasing"),
        ("numeric_token_density", "factual specificity", "continuous", "Numbers per 1,000 tokens", "page text only", "continuous or quartiles", "Can be inflated by navigation or IDs"),
        ("currency_price_mention_count", "factual specificity", "count", "THB/baht/price mentions", "page text only", "binary plus log2 count", "Strongly related to pricing/listing page function"),
        ("area_measurement_count", "factual specificity", "count", "sqm, square metre, rai, unit-size mentions", "page text only", "binary plus log2 count", "Condo-domain-specific"),
        ("bed_bath_unit_count", "factual specificity", "count", "Bedroom, bathroom, and unit-layout facts", "page text only", "binary plus count group", "Condo-domain-specific"),
        ("transit_distance_count", "factual specificity", "count", "BTS/MRT/distance/travel-time facts", "page text only", "binary plus count group", "May be confounded by area/location intent"),
        ("amenity_term_coverage", "topical completeness", "count or share", "Coverage of common condo amenities", "page text only", "standardized score", "Dictionary must be frozen before outcome analysis"),
        ("developer_project_entity_count", "factual specificity", "count", "Named developers and projects", "NER/dictionary", "log2 count", "Entity extraction quality varies by language"),
        ("location_entity_count", "factual specificity", "count", "Named neighborhoods, roads, stations, landmarks", "NER/dictionary", "log2 count", "Prompt relevance can confound association"),
        ("internal_link_ratio", "navigation", "continuous", "Internal links divided by all links", "resolved hrefs", "continuous", "Requires domain-aware link extraction"),
        ("external_citation_count", "evidence", "count", "Outbound links to external evidence", "resolved hrefs", "binary plus log2 count", "Do not infer endorsement from a link"),
        ("title_length_tokens", "metadata", "continuous", "Page-title length", "title", "spline or bins", "Metadata rather than body writing"),
        ("meta_description_length", "metadata", "continuous", "Meta-description length", "meta description", "spline or bins", "Often templated"),
        ("language_match_prompt", "relevance control", "binary", "Page language matches prompt language", "prompt and page text", "control", "Language detection must handle mixed Thai/English"),
        ("prompt_page_lexical_similarity", "relevance control", "continuous", "Prompt-page overlap computed without answer text", "prompt and page text", "control/sensitivity", "Pre-outcome and leakage-safe, but not a writing-style effect"),
        ("prompt_page_embedding_similarity", "relevance control", "continuous", "Semantic prompt-page relevance without answer text", "prompt and page text", "control/sensitivity", "Embedding model/version must be fixed and cached"),
        ("boilerplate_ratio", "measurement quality", "continuous", "Likely navigation/footer text share", "HTML blocks", "quality control", "Requires block-level extraction"),
        ("duplicate_content_cluster", "measurement quality", "categorical ID", "Near-duplicate page-content cluster", "content hashes/embeddings", "cluster or sensitivity", "Avoid including as a focal coefficient"),
        ("published_or_updated_age", "freshness", "continuous", "Page age at audit date", "structured data/page metadata", "spline or bins", "Dates are often missing or unreliable"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "feature_name",
            "feature_family",
            "raw_form",
            "construct",
            "source_evidence",
            "recommended_model_form",
            "risk_or_validation_need",
        ],
    )


def _risk_register() -> pd.DataFrame:
    rows = [
        ("Conditioning on surfaced sources", "selection/collider", "The sample contains cited and more-only sources surfaced by ChatGPT, not all eligible web pages.", "Define estimand as P(cited | surfaced in this audit); never call more-only rejected.", "Cannot identify web-wide citation probability."),
        ("Non-random scrape availability", "selection/missingness", "Blocked, dynamic, and failed pages have unmeasured writing features.", "Report availability by outcome/domain; strong-only sensitivity; estimate pre-content availability weights.", "IPW only addresses observed predictors of missingness."),
        ("Domain authority and publisher reputation", "omitted variable", "High-authority domains may both write differently and be cited more.", "Domain fixed-effects robustness on domains with multiple URLs; prompt-URL or prompt-domain clustered SE.", "Within-domain estimates may not generalize to a new publisher website."),
        ("Prompt-page relevance", "omitted variable", "Relevant pages may have both different wording and higher citation probability.", "Prompt fixed effects plus prompt-page similarity computed without answer text.", "Similarity is a relevance proxy, not proof of use."),
        ("Page function", "confounding/over-control", "FAQ, listing, and review pages have different structures and citation rates.", "Primary content model without final taxonomy; URL-seed page function as sensitivity.", "Final scraped-enriched page type partly uses focal content and may over-control."),
        ("Repeated URLs across prompts", "dependent observations", "The same URL-level content appears in multiple prompt rows.", "Two-way cluster by prompt_id and normalized_url.", "Effective content sample is closer to 2,600 URLs than 5,264 rows."),
        ("Shared domain templates", "dependent observations", "Pages from one domain share layout and writing templates.", "Domain-clustered SE alternative and domain FE robustness.", "Few-URL domains provide weak within-domain identification."),
        ("Temporal mismatch", "measurement/endogeneity", "Scraped content may differ from the page version visible when citation was observed.", "Record scrape/audit dates; use archived content if available; flag mutable pages.", "Current data cannot fully recover historical page state."),
        ("Extraction measurement error", "measurement error", "Counts can reflect crawler rendering, boilerplate, or truncation.", "Use normalized full-text cache, quality flags, strong-only analysis, and manual validation.", "Classical error attenuates estimates; non-classical error can bias direction."),
        ("Multicollinearity", "model instability", "Word/character length and table indicators are highly correlated.", "Use log2 word count rather than both word and character length; has_table instead of table_count.", "Joint coefficients answer conditional questions and may differ from one-feature models."),
        ("Functional-form misspecification", "model specification", "Raw counts are skewed and effects may be nonlinear.", "Use log2 transforms, threshold groups, splines, and predicted contrasts.", "A single LPM coefficient can hide thresholds or reversals."),
        ("Multiple testing", "false discovery", "A rich writing-feature set creates many hypothesis tests.", "Pre-register primary features; report Benjamini-Hochberg FDR for exploratory features.", "FDR control does not repair confounding."),
        ("LPM probability bounds", "model specification", "Linear predictions can fall outside 0-1.", "Report out-of-bound share and compare logit average marginal effects.", "Logit robustness does not make associations causal."),
        ("Intent heterogeneity", "effect heterogeneity", "A feature may matter differently for price, location, or recommendation prompts.", "Pre-specify a small number of feature-by-intent interactions and show cell support.", "Prompt FE absorbs intent main effects; sparse interactions can overfit."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "risk",
            "risk_type",
            "why_it_matters",
            "planned_mitigation",
            "residual_limitation",
        ],
    )


def _model_ladder() -> pd.DataFrame:
    rows = [
        ("M0", "Descriptive", "content_feature_available=true", "Cited-rate plots, Wilson intervals, sample support, sparse-category audit, and outlier audit", "None", "Describe shape, support, missingness, imbalance, and tails before modeling", "No adjusted interpretation"),
        ("M1", "One-feature screening LPM", "content_feature_available=true", "cited ~ focal_feature + C(prompt_id)", "Two-way cluster: prompt_id and normalized_url", "Estimate prompt-adjusted association for each feature separately", "Preferred first coefficient table"),
        ("M2", "Joint structural LPM", "content_feature_available=true", "cited ~ log2_word_count_plus1 + has_table + C(heading_count_group) + C(link_count_group) + C(content_strength) + C(prompt_id)", "Two-way cluster: prompt_id and normalized_url", "Estimate conditional associations among current structural features", "Do not include redundant raw forms; link_count_group is highly imbalanced and low-link coefficients require caution; content_strength is extraction quality, not writing quality"),
        ("M3", "Domain-adjusted LPM", "content available; source_root_domain has >=2 unique URLs", "M2 + C(source_root_domain)", "Two-way cluster: prompt_id and normalized_url", "Reduce confounding from domain authority and templates", "Identified within domains; may not generalize to a new publisher website"),
        ("M4", "Gemini taxonomy sensitivity", "content_feature_available=true", "M2 + C(page_type_family_gemini_v1_collapsed) + C(source_type_general_gemini_v1_collapsed)", "Two-way cluster: prompt_id and normalized_url", "Check whether content coefficients survive Gemini page-function and source/site-type adjustment", "Sensitivity only because Gemini may use scraped body content; keep unknown and collapse rare levels to rare_other; retain rule-v2 as M4R comparison"),
        ("M5", "Strong-content sensitivity", "content_strength == 'strong'", "Estimate M2, M3, and M4 on strong-content rows only", "Two-way cluster: prompt_id and normalized_url", "Reduce extraction measurement error", "content_strength is extraction quality, not writing quality"),
        ("M6", "Availability and missingness sensitivity", "all surfaced rows for availability model; measurable rows for weighted content model", "Estimate P(content_feature_available) from pre-content fields, then repeat M2 with stabilized inverse-probability weights", "Two-way cluster: prompt_id and normalized_url", "Address observed non-random scrape availability", "Not valid for unobserved missingness drivers; never impute unavailable content as zero"),
        ("M7", "Logit AME cross-check", "same sample and covariates as M2/M3", "Logit analogue of M2 and M3; report average marginal effects", "Cluster-robust if supported", "Check LPM functional form and probability bounds", "Average marginal effects remain conditional associations"),
        ("M8", "Prompt-page relevance sensitivity", "content_feature_available=true", "M2 + prompt-page lexical or embedding similarity computed from prompt and page only", "Two-way cluster: prompt_id and normalized_url", "Reduce omitted prompt-page relevance", "Must exclude answer text, answer similarity, and citation outcome"),
        ("M9", "Limited intent interactions", "only intent-feature cells with adequate cited and more-only support", "M2 plus a small pre-specified focal_feature x intent interaction set", "Two-way cluster: prompt_id and normalized_url", "Assess descriptive heterogeneity by user need", "Prompt FE absorbs intent main effects; apply FDR and avoid sparse cells"),
        ("M10", "Outlier and winsorized sensitivity", "content_feature_available=true", "Repeat M2 after removing word_count > p99; removing link_count > p99; and replacing log2_word_count_plus1 with log2_word_count_plus1_winsorized_p99", "Two-way cluster: prompt_id and normalized_url", "Assess whether extreme page size or link tails drive focal coefficients", "Compare coefficient direction, magnitude, confidence interval, rows, URLs, and prompts with M2"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "model_name",
            "sample",
            "formula_or_method",
            "inference",
            "purpose",
            "interpretation_boundary",
        ],
    )


def _descriptives(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    content = df[df["content_feature_available"]].copy()
    for feature in (
        "word_count",
        "content_chars",
        "heading_count",
        "table_count",
        "link_count",
        "log2_word_count_plus1",
        "heading_density_per_1000_words",
        "link_density_per_1000_words",
    ):
        values = pd.to_numeric(content[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "level": "<continuous>",
                "n_rows": int(values.notna().sum()),
                "mean_or_share": float(values.mean()),
                "median": float(values.median()),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "cited_rate": float(content.loc[values.notna(), "cited"].mean()),
                "analysis_subset": "content_feature_available=true",
            }
        )
    for feature in ("has_table", "has_headings", "has_links", "has_substantial_text", "has_multiple_tables"):
        values = _bool(content[feature])
        for level in (False, True):
            group = content.loc[values.eq(level)]
            rows.append(
                {
                    "feature": feature,
                    "level": str(level).lower(),
                    "n_rows": len(group),
                    "mean_or_share": len(group) / len(content),
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "cited_rate": float(group.cited.mean()) if len(group) else np.nan,
                    "analysis_subset": "content_feature_available=true",
                }
            )
    for feature in ("heading_count_group", "link_count_group", "content_strength", "content_quality_flag"):
        source = content if feature in {"heading_count_group", "link_count_group"} else df
        for level, group in source.groupby(feature, dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "level": str(level),
                    "n_rows": len(group),
                    "mean_or_share": len(group) / len(source),
                    "median": np.nan,
                    "p25": np.nan,
                    "p75": np.nan,
                    "cited_rate": float(group.cited.mean()),
                    "analysis_subset": "content_feature_available=true" if source is content else "all surfaced rows",
                }
            )
    return pd.DataFrame(rows)


def _support_table(df: pd.DataFrame, full_audit_prompts: int) -> pd.DataFrame:
    content = df[df.content_feature_available]
    prompts = df.groupby("prompt_id").cited.agg(["size", "sum"])
    domains = content.groupby("source_root_domain").agg(rows=("cited", "size"), urls=("normalized_url", "nunique"))
    urls = content.groupby("normalized_url").agg(rows=("cited", "size"), cited_rate=("cited", "mean"))
    rows = [
        ("full_audit_prompt_manifest_rows", full_audit_prompts, "full prompt-design denominator"),
        ("all_surfaced_rows", len(df), "source appearances"),
        ("all_unique_urls", df.normalized_url.nunique(), "URL clusters"),
        ("all_prompts_with_sources", df.prompt_id.nunique(), "prompt fixed-effect groups"),
        ("all_unique_domains", df.source_root_domain.nunique(), "source-domain groups"),
        ("all_cited_rows", int(df.cited.sum()), "observed cited source appearances"),
        ("all_cited_rate", float(df.cited.mean()), "observed citation share among surfaced sources"),
        ("content_available_rows", len(content), "preferred content-analysis sample before sensitivity restrictions"),
        ("content_available_unique_urls", content.normalized_url.nunique(), "independent content units/clusters"),
        ("content_available_prompts", content.prompt_id.nunique(), "prompt fixed-effect groups in measurable-content sample"),
        ("content_available_domains", content.source_root_domain.nunique(), "domain groups"),
        ("content_available_cited_rows", int(content.cited.sum()), "cited appearances in measurable-content sample"),
        ("content_available_cited_rate", float(content.cited.mean()), "citation share in measurable-content sample"),
        ("prompts_with_both_outcomes", int(((prompts["sum"] > 0) & (prompts["sum"] < prompts["size"])).sum()), "prompts contributing direct within-prompt outcome variation"),
        ("urls_repeated_across_rows", int((urls.rows >= 2).sum()), "URL clusters with repeated source appearances"),
        ("rows_from_repeated_urls", int(urls.loc[urls.rows >= 2, "rows"].sum()), "rows requiring URL-clustered inference"),
        ("urls_with_citation_variation", int(((urls.cited_rate > 0) & (urls.cited_rate < 1)).sum()), "same URL cited for some prompts but not others"),
        ("domains_with_at_least_2_urls", int((domains.urls >= 2).sum()), "domains supporting within-domain feature comparisons"),
        ("rows_in_domains_with_at_least_2_urls", int(domains.loc[domains.urls >= 2, "rows"].sum()), "domain-FE robustness support"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "econometric_relevance"])


def _sample_count_recheck(all_rows: pd.DataFrame, measurable: pd.DataFrame, full_audit_prompts: int) -> pd.DataFrame:
    records = []
    for sample, frame in (("all_surfaced_rows", all_rows), ("measurable_content_lpm", measurable)):
        records.append(
            {
                "sample": sample,
                "n_rows": len(frame),
                "unique_normalized_url": frame.normalized_url.nunique(),
                "unique_prompt_id": frame.prompt_id.nunique(),
                "unique_source_root_domain": frame.source_root_domain.nunique(),
                "cited_rows": int(frame.cited.sum()),
                "cited_rate": float(frame.cited.mean()),
                "full_audit_prompt_manifest_count": full_audit_prompts,
                "prompt_count_caveat": (
                    f"Full audit = {full_audit_prompts} prompts; measurable-content LPM sample = "
                    f"{measurable.prompt_id.nunique()} prompts."
                ),
            }
        )
    return pd.DataFrame(records)


def _link_imbalance_check(measurable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(measurable)
    for category, group in measurable.groupby("link_count_group", dropna=False):
        rows.append(
            {
                "link_count_group": str(category),
                "n_rows": len(group),
                "row_share": len(group) / total,
                "cited_rows": int(group.cited.sum()),
                "more_only_rows": int((1 - group.cited).sum()),
                "cited_rate": float(group.cited.mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    dominant_share = float(result.row_share.max())
    result["dominant_group_share"] = dominant_share
    result["highly_imbalanced_flag"] = dominant_share >= 0.90
    result["warning"] = np.where(
        result.highly_imbalanced_flag,
        "link_count_group is highly imbalanced; coefficients for low-link groups should be interpreted cautiously.",
        "",
    )
    return result


def _content_strength_sensitivity(measurable: pd.DataFrame) -> pd.DataFrame:
    formula = (
        "cited ~ log2_word_count_plus1 + has_table + C(heading_count_group) + "
        "C(link_count_group) + C(content_strength) + C(prompt_id)"
    )
    rows = []
    for sample_name, frame, model_note in (
        (
            "measurable_content_with_quality_control",
            measurable,
            "Estimate preferred model with C(content_strength).",
        ),
        (
            "strong_content_only",
            measurable[measurable.content_strength.eq("strong")],
            "Repeat preferred model after restricting to content_strength == 'strong' and omit C(content_strength).",
        ),
    ):
        rows.append(
            {
                "sample": sample_name,
                "n_rows": len(frame),
                "unique_urls": frame.normalized_url.nunique(),
                "unique_prompts": frame.prompt_id.nunique(),
                "unique_domains": frame.source_root_domain.nunique(),
                "cited_rows": int(frame.cited.sum()),
                "cited_rate": float(frame.cited.mean()),
                "formula_or_action": formula if sample_name.startswith("measurable") else formula.replace(" + C(content_strength)", ""),
                "interpretation": model_note,
                "boundary": "content_strength is extraction-quality control, not writing quality.",
            }
        )
    return pd.DataFrame(rows)


def _formula_has_token(formulas: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", formulas.casefold()))


def _leakage_guardrail(
    all_rows: pd.DataFrame,
    measurable: pd.DataFrame,
    ladder: pd.DataFrame,
) -> pd.DataFrame:
    formula_text = "\n".join(
        ladder.loc[ladder.model_id.ne("M0"), "formula_or_method"].fillna("").astype(str)
    ).casefold()
    rows = []
    for check_name, tokens in LEAKAGE_CHECKS:
        all_matches = sorted(
            {
                column
                for column in all_rows.columns
                for token in tokens
                if token in column.casefold()
            }
        )
        measurable_matches = sorted(
            {
                column
                for column in measurable.columns
                for token in tokens
                if token in column.casefold()
            }
        )
        formula_matches = sorted(token for token in tokens if _formula_has_token(formula_text, token))
        passed = not all_matches and not measurable_matches and not formula_matches
        rows.append(
            {
                "guardrail_check": check_name,
                "forbidden_tokens": "; ".join(tokens),
                "all_surfaced_dataset_matches": "; ".join(all_matches) or "none",
                "measurable_model_dataset_matches": "; ".join(measurable_matches) or "none",
                "candidate_formula_matches": "; ".join(formula_matches) or "none",
                "status": "pass" if passed else "fail",
                "required_action": "none" if passed else "remove from model dataset and candidate formulas",
            }
        )
    return pd.DataFrame(rows)


def _compact_url_evidence(urls: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    exposure = rows.groupby("normalized_url").agg(
        source_appearances=("cited", "size"),
        cited_appearances=("cited", "sum"),
        cited_rate=("cited", "mean"),
        unique_prompts=("prompt_id", "nunique"),
    ).reset_index()
    wanted = [
        "normalized_url",
        "source_url",
        "source_root_domain",
        "page_title",
        "meta_description",
        "page_text_excerpt",
        "page_text",
        "structured_data_types",
        "scrape_success",
        "content_strength",
        "content_quality_flag",
        "scrape_error",
        "content_chars",
        "word_count",
        "heading_count",
        "table_count",
        "link_count",
        "site_type_general",
        "page_type_url_seed_general",
        "page_type_family_general",
        "page_type_general",
        "page_type_general_confidence",
        "page_type_general_reason",
        "source_type_real_estate",
        "page_type_family_real_estate",
    ]
    compact = urls[[column for column in wanted if column in urls]].copy()
    if "page_text" in compact:
        compact["page_text_preview_3000_chars"] = compact.pop("page_text").fillna("").astype(str).str.slice(0, 3000)
    return compact.merge(exposure, on="normalized_url", how="left", validate="one_to_one")


def _write_docs(output: Path, metrics: dict[str, object]) -> None:
    docs = output / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    scope = f"""# 01 Research scope and estimand

## Main question

Among source pages surfaced in the 500-prompt area-condo / SCOPE-relevant nonbranded audit, which observable writing and page-content features are associated with a higher probability of being cited?

## Estimand

The target is `P(cited = 1 | source was surfaced in this audit)`. This is not the probability that an arbitrary webpage on the open web will be cited. A more-only row is surfaced but not cited; it is not evidence that the page was rejected from a hidden retrieval set.

## Unit and current sample

- {metrics['all_rows']:,} surfaced source appearances
- {metrics['all_urls']:,} unique URLs
- {metrics['content_rows']:,} appearances with measurable content
- {metrics['content_urls']:,} unique URLs with measurable content
- {metrics['content_prompts']:,} prompts in the measurable-content LPM sample
- {metrics['content_domains']:,} domains in the measurable-content LPM sample
- {metrics['content_cited_rows']:,} cited rows in the measurable-content LPM sample
- {metrics['content_cited_rate']:.2%} measurable-sample cited rate

**Full audit = {metrics['full_audit_prompts']:,} prompts; measurable-content LPM sample = {metrics['content_prompts']:,} prompts.** Two manifest prompts produced no source rows and therefore cannot enter a source-row LPM.

The same URL can appear under multiple prompts. Writing features are therefore URL-level variables repeated across rows. Standard errors must account for prompt and URL clustering.

## Content-centric scope

Page type and source type are not the focal business result. They are retained only for:

1. sample description;
2. confounding sensitivity;
3. checking whether a content association is actually a page-function or domain-composition pattern.

The practical interpretation should focus on content features that a publisher can plausibly change: length, structure, headings, lists, tables, factual specificity, answerability, and relevance. All results remain observational associations among surfaced sources.
"""
    (docs / "01_research_scope_and_estimand.md").write_text(scope, encoding="utf-8")

    identification = """# 02 Identification, confounding, and endogeneity plan

## Preferred design

1. Restrict writing-feature models to `content_feature_available = true`.
2. Estimate one-feature LPMs with exact prompt fixed effects.
3. Estimate a joint structural-feature LPM.
4. Use two-way clustered standard errors by `prompt_id` and `normalized_url`.
5. Add domain fixed effects as a robustness model on domains with at least two URLs.
6. Add URL-seed page function as a sensitivity control, not as the primary specification.
7. Repeat on strong-content pages only.
8. Compare with logit average marginal effects.
9. Repeat after removing values above the 99th percentile for word count and link count, and with p99-winsorized log2 word count.

## Why prompt fixed effects?

They compare surfaced sources within the same exact question, absorbing prompt wording, area, intent, and all other prompt-level constants.

## Why URL clustering rather than URL fixed effects?

Content features are properties of a URL. URL fixed effects would absorb those features and make their coefficients unidentified. URL clustering retains the coefficients while allowing repeated observations of the same content to be correlated.

## Why taxonomy is not primary

The final page taxonomy can use scraped body evidence. Using it as a primary control while estimating body-content effects can over-control or create a mechanical relationship. Prefer `page_type_url_seed_general` for taxonomy sensitivity because it is classified without scraped body text.

## Main remaining limitations

- Sources are selected into the surfaced-source panel.
- Scrape availability is non-random.
- Domain authority, SEO strength, backlinks, and historical reputation are incompletely observed.
- The page may have changed between the citation audit and the later scrape.
- `link_count_group` is highly imbalanced; coefficients for low-link groups should be interpreted cautiously.
- Sparse or separating taxonomy categories must be collapsed to `rare_other`, while `unknown` stays explicit.
- Associations do not identify the causal effect of editing a page.
"""
    (docs / "02_identification_and_endogeneity_plan.md").write_text(identification, encoding="utf-8")

    roadmap = """# 03 Writing-feature engineering roadmap

## Stage A: current structural features

Use the current measurable features first:

- `log2_word_count_plus1`
- `has_table`
- `heading_count_group`
- `link_count_group`
- heading/link/table density

This gives a transparent baseline but does not fully answer how prose is written.

## Stage B: deterministic writing features

Extract sentence and paragraph length, list density, question headings, FAQ schema, opening-summary patterns, numeric density, price facts, unit-size facts, location/transit facts, amenities, entities, and external evidence links. Freeze dictionaries and regex rules before looking at model coefficients.

## Stage C: prompt relevance controls

Compute lexical and embedding similarity between the prompt and page text. These are allowed because they use the prompt and page only, not the generated answer or citation outcome. Treat them as relevance controls, not proof that the page was used.

## Stage D: validation

Manually label a stratified sample for each new feature. Report precision/recall or agreement before econometric use. Validate Thai and mixed Thai-English pages separately.

## Stage E: model reporting

For each focal feature report:

1. unadjusted cited-rate contrast;
2. prompt-FE coefficient;
3. joint-model coefficient;
4. domain-FE robustness;
5. page-function sensitivity;
6. strong-content sensitivity;
7. logit average marginal effect;
8. predicted probability contrast at actionable values.

Use Benjamini-Hochberg FDR for the expanded exploratory feature set.

## Stage F: mandatory tail sensitivity

Run the preferred model normally, with `word_count > p99` removed, with `link_count > p99` removed, and with `log2_word_count_plus1_winsorized_p99`. Compare focal coefficient direction, magnitude, confidence intervals, and sample support.
"""
    (docs / "03_writing_feature_engineering_roadmap.md").write_text(roadmap, encoding="utf-8")

    guide = """# 04 How features enter the LPM

## Outcome

`cited` is binary. In an LPM, a coefficient of `0.05` means a five-percentage-point conditional difference in cited probability, not a five-percent relative increase.

## Continuous content features

Use `log2_word_count_plus1` instead of raw word count. Its coefficient is approximately the percentage-point association with doubling page length. For nonlinear variables, also report bins, splines, and predicted contrasts.

## Binary features

For `has_table`, the coefficient compares pages with versus without a table, conditional on controls. It is not necessarily the causal effect of adding a table.

## Categorical features

`heading_count_group` and `link_count_group` enter as dummy variables. Every coefficient is relative to an explicitly reported reference category. However, `link_count_group` is highly imbalanced; coefficients for low-link groups should be interpreted cautiously. `low_link_count` is available as an optional binary sensitivity form.

## Fixed effects

`C(prompt_id)` absorbs all prompt-level constants. `C(source_root_domain)` in the robustness model compares different URLs from the same domain.

## Taxonomy

- `site_type_general`: type of website/source.
- `page_type_family_general`: broad page-function family.
- `page_type_general_common`: detailed familiar function such as FAQ, listing, contact, or landing page.
- `page_type_family_gemini_v1_collapsed`: preferred page-function sensitivity control.
- `source_type_general_gemini_v1_collapsed`: preferred source/site-type sensitivity control.
- `page_type_url_seed_general_collapsed`: older rule-v2 comparison only.

Taxonomy should not be the focal client recommendation in this content workstream.

## Probability summaries

Rank features using comparable outputs:

- binary: adjusted probability difference;
- continuous: change from P25 to P75 and effect of doubling;
- categorical: contrast against the reference group;
- standardized exploratory ranking: percentage points per one standard deviation.

Always show confidence intervals and coefficient stability across model specifications.
"""
    (docs / "04_lpm_feature_usage_guide.md").write_text(guide, encoding="utf-8")

    notebook = """# 05 Specification for the next notebook

Recommended notebook: `09_area_condo_content_feature_econometrics.ipynb`

## Sections

1. Load and validate the package.
2. Define the surfaced-source estimand.
3. Audit content availability and strong-content selection.
4. Show current structural-feature distributions.
5. Run one-feature prompt-FE LPMs.
6. Run the joint structural-feature LPM.
7. Run domain-FE robustness on domains with at least two URLs.
8. Run Gemini page-family and source/site-type sensitivity; retain rule-v2 as M4R comparison.
9. Run strong-content and IPW sensitivity.
10. Run logit average marginal effects.
11. Compare coefficient stability in one forest table.
12. Check LPM predictions outside 0-1.
13. Apply FDR to exploratory writing features.
14. Produce a client-readable table translating estimates into percentage-point contrasts.
15. Run M10 word-count and link-count outlier/winsorized sensitivity and compare focal coefficients.

## Mandatory boundaries

- No answer text or answer similarity.
- No source position or observed rank in the main model.
- No causal language.
- Do not impute unavailable content to zero.
- Cluster repeated content observations.
- Keep page/source taxonomy secondary.
- Treat `content_strength` as extraction quality, not writing quality.
- Keep `unknown` explicit and use collapsed taxonomy variables in LPM sensitivity models.
"""
    (docs / "05_next_notebook_spec.md").write_text(notebook, encoding="utf-8")

    prompt = """# AI handoff prompt

You are analyzing an observational area-condo / SCOPE-relevant nonbranded AI-search citation audit.

The outcome is whether a surfaced source appearance was cited. The comparison group is more-only: surfaced but not cited. Do not describe more-only sources as rejected, and do not infer the system's hidden retrieval process.

Primary research question: which observable webpage writing/content features are associated with citation probability among surfaced sources?

Use:

- `data/content_lpm_all_surfaced_rows.csv` for availability and selection diagnostics.
- `data/content_lpm_measurable_rows.csv` for content-feature models.
- `data/url_content_evidence_compact.csv` for manual examples and URL-level text evidence.
- `tables/current_lpm_feature_dictionary.csv` for exact variable interpretation.
- `tables/model_specification_ladder.csv` for the required model sequence.
- `tables/confounder_endogeneity_risk_register.csv` for limitations.

Main inference:

- prompt fixed effects;
- two-way clustering by prompt and normalized URL;
- domain fixed effects as robustness;
- page/source taxonomy as sensitivity controls, not the focal result;
- no answer-derived variables, rank, position, source origin, or outcome-derived predictors.
- sparse taxonomy levels collapsed to `rare_other`, while `unknown` remains explicit;
- strong-content-only and outlier/winsorized sensitivity analyses are mandatory.

For every feature report coefficient units, confidence intervals, sample size, URL clusters, prompt clusters, reference category, and stability across specifications. Use percentage-point language for LPM coefficients. Treat results as conditional associations, not causal effects.
"""
    (docs / "06_ai_handoff_prompt.md").write_text(prompt, encoding="utf-8")

    packet = f"""# AREA CONDO CONTENT ECONOMETRICS: AI MASTER PACKET

## One-paragraph summary

This package prepares a separate content-focused econometric analysis for an **area-condo / SCOPE-relevant nonbranded audit**. The outcome is `cited` among sources already surfaced in a 500-prompt ChatGPT audit. The full surfaced-source table contains {metrics['all_rows']:,} rows, {metrics['all_urls']:,} URLs, {metrics['all_prompts']:,} prompts with sources, {metrics['all_domains']:,} domains, and {metrics['all_cited_rows']:,} cited rows ({metrics['all_cited_rate']:.2%}). The measurable-content model sample contains {metrics['content_rows']:,} rows, {metrics['content_urls']:,} URLs, {metrics['content_prompts']:,} prompts, {metrics['content_domains']:,} domains, and {metrics['content_cited_rows']:,} cited rows ({metrics['content_cited_rate']:.2%}).

**Full audit = {metrics['full_audit_prompts']:,} prompts; measurable-content LPM sample = {metrics['content_prompts']:,} prompts.** All prompt rows use `expansion_group = natural_nonbranded`; the prompt design is explicitly nonbranded.

## Correct estimand

`P(cited = 1 | source surfaced in this audit)`

This is not web-wide citation likelihood and not a causal effect of rewriting a page.

## Preferred current model

`cited ~ log2_word_count_plus1 + has_table + C(heading_count_group) + C(link_count_group) + C(content_strength) + C(prompt_id)`

Use two-way clustered standard errors by `prompt_id` and `normalized_url`.

## Strong robustness model

Repeat the preferred model with `C(source_root_domain)` on domains with at least two URLs. Then add `C(page_type_family_gemini_v1_collapsed) + C(source_type_general_gemini_v1_collapsed)` in a separate Gemini taxonomy sensitivity model. Retain rule-v2 as M4R only.

## Why page type is secondary

Gemini taxonomy is preferred after manual QA, but it can use scraped content and absorb the same writing variation being studied. Therefore taxonomy remains secondary to M1/M2.

Use `page_type_family_gemini_v1_collapsed` and `source_type_general_gemini_v1_collapsed` in M4. Low-support levels are combined as `rare_other` without using the citation outcome; `unknown` remains explicit.

## Current focal features

- `log2_word_count_plus1`: approximate effect of doubling page length.
- `has_table`: adjusted difference for table presence.
- `heading_count_group`: nonlinear heading structure.
- `link_count_group`: nonlinear link structure.
- density forms: headings, links, and tables per 1,000 words.

The list above documents the frozen area-condo baseline only. In the future
Core-General layer, broad `has_table` is a deprecated compatibility field. Use
`has_any_data_table` only after semantic/layout QA, or `has_verified_html_table` as
a broader provenance-specific sensitivity when semantic classification is not ready.
Table presence, function, content signals, and page-level aggregates remain separate.
Pricing tables are Commerce-General; real-estate unit tables are paused vertical
extensions. No table feature is approved for model v1 until its recorded QA gate passes.
`has_verified_html_table` is the single canonical verified-presence field;
`has_any_verified_table` and `has_table` are deprecated aliases. Table-level features
must be aggregated explicitly before entering a page-level dataset. No-table,
measured-unknown, mixed, and extraction-unavailable categorical states remain distinct.

`link_count_group` is highly imbalanced; coefficients for low-link groups should be interpreted cautiously. Use `low_link_count = link_count < 9` only as a clearly labeled sensitivity alternative.

## Required next feature layer

Add deterministic features for sentence/paragraph length, list structure, question headings, FAQ schema, opening summaries, factual/numeric density, prices, unit sizes, transit/location facts, amenities, entities, and external evidence links. Add prompt-page similarity using prompt and page only; never use answer similarity.

## Main risks

1. Selection into surfaced sources.
2. Non-random scrape availability.
3. Domain authority and site-template confounding.
4. Prompt-page relevance.
5. Page function and potential over-control.
6. Repeated URLs across prompts.
7. Temporal mismatch between citation audit and scrape.
8. Extraction measurement error.
9. Multiple testing and nonlinear relationships.
10. Sparse/separating taxonomy levels.
11. Extreme word-count, link-count, and density tails.

## Mandatory sensitivities before interpretation

- Estimate M2 with `C(content_strength)`.
- Repeat M2-M4 on `content_strength == "strong"`; `content_strength` is extraction-quality control, not writing quality.
- Repeat M2 after removing observations with `word_count` above p99.
- Repeat M2 after removing observations with `link_count` above p99.
- Repeat M2 with `log2_word_count_plus1_winsorized_p99`.
- Compare focal coefficient direction, magnitude, confidence interval, row count, URL clusters, and prompt clusters.

## Minimum reporting table

For every focal feature show:

1. unadjusted contrast;
2. prompt-FE coefficient;
3. joint-model coefficient;
4. domain-FE robustness;
5. Gemini page-family and source/site-type sensitivity;
6. strong-content sensitivity;
7. logit average marginal effect;
8. actionable predicted probability contrast;
9. 95% confidence interval and cluster counts.

## Files

- `data/content_lpm_all_surfaced_rows.csv`: all rows, including failed/weak content.
- `data/content_lpm_measurable_rows.csv`: content-analysis sample.
- `data/url_content_evidence_compact.csv`: URL examples with title, excerpt, preview, taxonomy, and exposure.
- `data/prompt_reference.csv`: prompt text and intent.
- `tables/current_lpm_feature_dictionary.csv`: current variables and coefficient interpretation.
- `tables/proposed_writing_feature_dictionary.csv`: next extraction layer.
- `tables/model_specification_ladder.csv`: model sequence.
- `tables/confounder_endogeneity_risk_register.csv`: identification limitations.
- `tables/content_lpm_sample_count_recheck.csv`: exact all-row and measurable-sample counts.
- `tables/content_lpm_leakage_guardrail_check.csv`: model-data and formula leakage verification.
- `tables/content_lpm_sparse_category_audit.csv`: category support and collapse actions.
- `tables/content_lpm_outlier_audit.csv`: p99 thresholds and tail diagnostics.
- `docs/`: detailed scope, identification, feature engineering, LPM guide, and notebook specification.

If only one context file can be uploaded, use this Markdown packet. For actual estimation, also upload `data/content_lpm_measurable_rows.csv`.

## Final readiness

`{metrics['readiness_status']}`
"""
    (output / "AI_ANALYSIS_MASTER_PACKET.md").write_text(packet, encoding="utf-8")

    readme = """# Content Econometrics AI Package

Start with `AI_ANALYSIS_MASTER_PACKET.md`.

This package is deliberately separate from the page/source taxonomy EDA. Its focal question is how measurable writing and content structure are associated with citation among surfaced sources in an area-condo / SCOPE-relevant nonbranded audit. Taxonomy remains available only for confounding and sensitivity analysis.

The package is generated by:

```bash
.venv/bin/python scripts/v2_prepare_area_condo_content_econometrics_package.py
```

No econometric model is fit by the package builder. It creates a reproducible analysis handoff and prevents answer-derived, rank, position, and outcome-derived leakage variables from entering the model files.
"""
    (output / "README_FIRST.md").write_text(readme, encoding="utf-8")


def _manifest(output: Path) -> pd.DataFrame:
    purposes = {
        "README_FIRST.md": "Starting instructions",
        "AI_ANALYSIS_MASTER_PACKET.md": "Single-file AI context and analysis rules",
        "data/content_lpm_all_surfaced_rows.csv": "All source appearances for availability and selection analysis",
        "data/content_lpm_measurable_rows.csv": "Primary row-level content econometrics dataset",
        "data/url_content_evidence_compact.csv": "URL-level title, excerpt, text preview, taxonomy, and exposure evidence",
        "data/prompt_reference.csv": "Prompt text and intent reference",
        "tables/current_lpm_feature_dictionary.csv": "Exact current feature roles and coefficient interpretation",
        "tables/proposed_writing_feature_dictionary.csv": "Planned richer writing/content features",
        "tables/model_specification_ladder.csv": "Ordered descriptive and econometric models",
        "tables/confounder_endogeneity_risk_register.csv": "Bias risks, mitigations, and residual caveats",
        "tables/baseline_feature_descriptive_summary.csv": "Current feature distributions and cited rates",
        "tables/model_estimability_support.csv": "Cluster counts and variation supporting inference",
        "tables/leakage_audit.csv": "Forbidden-variable check for model datasets",
        "tables/content_lpm_sample_count_recheck.csv": "Exact all-row and measurable-content sample counts",
        "tables/content_lpm_leakage_guardrail_check.csv": "Leakage checks across model datasets and candidate formulas",
        "tables/content_lpm_sparse_category_audit.csv": "Sparse, unstable, and separating category audit",
        "tables/content_lpm_outlier_audit.csv": "P99 thresholds and extreme-value sensitivity requirements",
        "tables/content_lpm_link_count_imbalance_check.csv": "Link-count group support and imbalance warning",
        "tables/content_lpm_content_strength_sensitivity.csv": "Extraction-quality control and strong-content sample plan",
    }
    rows = []
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file() and not any(part.startswith(".") for part in item.relative_to(output).parts)
    ):
        relative = str(path.relative_to(output))
        n_rows = pd.NA
        n_columns = pd.NA
        if path.suffix == ".csv":
            frame = pd.read_csv(path, low_memory=False)
            n_rows, n_columns = len(frame), len(frame.columns)
        rows.append(
            {
                "relative_path": relative,
                "file_size_mb": path.stat().st_size / 1024 / 1024,
                "n_rows": n_rows,
                "n_columns": n_columns,
                "purpose": purposes.get(relative, "Supporting documentation"),
            }
        )
    return pd.DataFrame(rows)


def build(master_path: Path, url_path: Path, prompt_path: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    for metadata_file in output.rglob(".DS_Store"):
        metadata_file.unlink(missing_ok=True)
    (output / "data").mkdir(exist_ok=True)
    (output / "tables").mkdir(exist_ok=True)
    rows = _apply_taxonomy_collapses(_prepare_rows(pd.read_csv(master_path, low_memory=False)))
    urls = pd.read_csv(url_path, low_memory=False)
    prompts = pd.read_csv(prompt_path, low_memory=False)

    columns = _model_columns(rows)
    all_rows = rows[columns].copy()
    measurable = all_rows[_bool(all_rows["content_feature_available"])].copy()
    all_rows.to_csv(output / "data/content_lpm_all_surfaced_rows.csv", index=False)
    measurable.to_csv(output / "data/content_lpm_measurable_rows.csv", index=False)
    _compact_url_evidence(urls, rows).to_csv(output / "data/url_content_evidence_compact.csv", index=False)
    prompts.to_csv(output / "data/prompt_reference.csv", index=False)

    ladder = _model_ladder()
    _feature_dictionary().to_csv(output / "tables/current_lpm_feature_dictionary.csv", index=False)
    _proposed_features().to_csv(output / "tables/proposed_writing_feature_dictionary.csv", index=False)
    _risk_register().to_csv(output / "tables/confounder_endogeneity_risk_register.csv", index=False)
    ladder.to_csv(output / "tables/model_specification_ladder.csv", index=False)
    _descriptives(rows).to_csv(output / "tables/baseline_feature_descriptive_summary.csv", index=False)
    support = _support_table(rows, len(prompts))
    support.to_csv(output / "tables/model_estimability_support.csv", index=False)
    _sample_count_recheck(all_rows, measurable, len(prompts)).to_csv(
        output / "tables/content_lpm_sample_count_recheck.csv", index=False
    )
    sparse = _sparse_category_audit(measurable)
    sparse.to_csv(output / "tables/content_lpm_sparse_category_audit.csv", index=False)
    _outlier_audit(measurable).to_csv(output / "tables/content_lpm_outlier_audit.csv", index=False)
    _link_imbalance_check(measurable).to_csv(
        output / "tables/content_lpm_link_count_imbalance_check.csv", index=False
    )
    _content_strength_sensitivity(measurable).to_csv(
        output / "tables/content_lpm_content_strength_sensitivity.csv", index=False
    )
    leakage = _leakage_guardrail(all_rows, measurable, ladder)
    leakage.to_csv(output / "tables/content_lpm_leakage_guardrail_check.csv", index=False)
    leakage.to_csv(output / "tables/leakage_audit.csv", index=False)
    if leakage.status.eq("fail").any():
        raise RuntimeError("Forbidden leakage variables are present in the content econometrics package.")

    support_map = dict(zip(support.metric, support.value))
    metrics = {
        "full_audit_prompts": int(support_map["full_audit_prompt_manifest_rows"]),
        "all_rows": int(support_map["all_surfaced_rows"]),
        "all_urls": int(support_map["all_unique_urls"]),
        "all_prompts": int(support_map["all_prompts_with_sources"]),
        "all_domains": int(support_map["all_unique_domains"]),
        "all_cited_rows": int(support_map["all_cited_rows"]),
        "all_cited_rate": float(support_map["all_cited_rate"]),
        "content_rows": int(support_map["content_available_rows"]),
        "content_urls": int(support_map["content_available_unique_urls"]),
        "content_prompts": int(support_map["content_available_prompts"]),
        "content_domains": int(support_map["content_available_domains"]),
        "content_cited_rows": int(support_map["content_available_cited_rows"]),
        "content_cited_rate": float(support_map["content_available_cited_rate"]),
    }
    expected_models = ["M0", "M1", "M2", "M3", "M4", "M4R", *[f"M{index}" for index in range(5, 11)]]
    checks_pass = (
        not leakage.status.eq("fail").any()
        and ladder.model_id.tolist() == expected_models
        and set(all_rows.expansion_group.dropna().astype(str)) == {"natural_nonbranded"}
        and all(
            column in measurable.columns
            for column in (
                "page_type_url_seed_general_collapsed",
                "page_type_family_general_collapsed",
                "site_type_general_collapsed",
                "low_link_count",
                "log2_word_count_plus1_winsorized_p99",
            )
        )
        and len(prompts) == 500
        and len(measurable) == 5264
    )
    metrics["readiness_status"] = (
        "ready_for_09_area_condo_content_feature_econometrics"
        if checks_pass
        else "not_ready_for_09_area_condo_content_feature_econometrics"
    )
    _write_docs(output, metrics)
    validation = {
        **metrics,
        "content_availability_rate": metrics["content_rows"] / metrics["all_rows"],
        "package_files": 0,
        "leakage_failures": int(leakage.status.eq("fail").sum()),
        "sparse_category_rows": int(sparse.sparse_flag.sum()),
        "perfect_prediction_category_rows": int(sparse.perfect_prediction_flag.sum()),
        "readiness_status": metrics["readiness_status"],
        "output": str(output),
    }
    validation_path = output / "package_validation.json"
    manifest_path = output / "FILE_MANIFEST.csv"
    manifest_path.unlink(missing_ok=True)
    validation["package_files"] = int(
        len(
            [
                item
                for item in output.rglob("*")
                if item.is_file() and not any(part.startswith(".") for part in item.relative_to(output).parts)
            ]
        )
        + 1
    )
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    manifest = _manifest(output)
    manifest = pd.concat(
        [
            manifest,
            pd.DataFrame(
                [
                    {
                        "relative_path": "FILE_MANIFEST.csv",
                        "file_size_mb": 0.0,
                        "n_rows": len(manifest) + 1,
                        "n_columns": len(manifest.columns),
                        "purpose": "Inventory of every package artifact",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    manifest.to_csv(manifest_path, index=False)
    manifest.loc[manifest.relative_path.eq("FILE_MANIFEST.csv"), "file_size_mb"] = (
        manifest_path.stat().st_size / 1024 / 1024
    )
    manifest.to_csv(manifest_path, index=False)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--url-taxonomy", type=Path, default=DEFAULT_URLS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(build(args.master, args.url_taxonomy, args.prompts, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
