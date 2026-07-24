"""Build and validate lightweight, read-only econometrics frontend artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import econometrics_qa as qa
from src.econometrics_eda_v2 import model_comparison


CONTRACT_VERSION = "econometrics_frontend_v2"
ARTIFACT_DIRNAME = "econometrics_frontend"
FORBIDDEN_PREDICTOR_TOKENS = (
    "answer_similarity", "page_answer_similarity", "max_chunk_answer_similarity",
    "answer_overlap", "answer_like_text", "brand_appeared_in_answer", "cited_label",
    "source_group", "source_origin", "source_position", "observed_rank",
    "domain_citation_rate", "citation_rate_proxy",
)


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    kind: str
    group: str
    layer: str
    interpretation_unit: str
    reference_group: str = ""
    warning: str = ""


FEATURE_SPECS = (
    FeatureSpec("log2_word_count_plus1", "Page length", "continuous", "page_length", "core_general", "approximate page-length doubling"),
    FeatureSpec("has_table", "Detected table (legacy proxy)", "binary", "table_structure", "core_general", "present versus absent", warning="Legacy proxy; may be excerpt- and domain/template-sensitive."),
    FeatureSpec("heading_count_group", "Heading structure", "categorical", "heading_structure", "core_general", "category versus 0-1 headings", "0-1"),
    FeatureSpec("link_count_group", "Link structure", "categorical", "link_structure", "core_general", "category versus 9+ links", "9+", warning="Low-link categories have limited support and the feature is highly imbalanced."),
    FeatureSpec("content_strength", "Content extraction strength", "categorical", "extraction_quality", "core_general", "category versus strong extraction", "strong", warning="Extraction quality control, not writing quality."),
    FeatureSpec("factual_numeric_density_score", "Factual and numeric density", "continuous", "factual_density", "core_general", "one score-unit change"),
    FeatureSpec("external_evidence_score", "External evidence structure", "continuous", "external_evidence", "core_general", "one score-unit change", warning="Only a one-feature sensitivity estimate is currently available."),
    FeatureSpec("writing_structure_score", "Writing structure", "continuous", "writing_structure", "core_general", "one score-unit change", warning="Composite diagnostic; primitives should be checked before model use."),
    FeatureSpec("prompt_page_relevance_score", "Prompt-page relevance", "continuous", "prompt_relevance", "core_general", "one score-unit change", warning="Leakage-safe prompt-page relevance; no answer text is used."),
)
SPEC_BY_NAME = {spec.name: spec for spec in FEATURE_SPECS}

REQUIRED_SCHEMAS: dict[str, tuple[str, ...]] = {
    "core_general_feature_scorecard.csv": (
        "feature_name", "human_label", "feature_layer", "feature_group", "feature_type",
        "measurement_status", "qa_status", "approved_for_model", "support_rows",
        "support_prompts", "support_urls", "support_domains", "missing_rate",
        "raw_association_pp", "g1_estimate_pp", "g2_estimate_pp", "domain_fe_estimate_pp",
        "g2_ci_lower_pp", "g2_ci_upper_pp", "robustness_status", "evidence_quality_label",
        "interpretation_summary", "primary_warning", "dataset_version", "feature_registry_version",
    ),
    "feature_cited_rate_summary.csv": (
        "feature_name", "feature_level", "feature_state", "bin_order", "n_rows", "n_cited",
        "n_more_only", "cited_rate", "ci_lower", "ci_upper", "n_prompts", "n_urls",
        "n_domains", "support_flag", "source_artifact",
    ),
    "feature_model_estimates.csv": (
        "feature_name", "model_id", "source_model_id", "model_role", "term", "term_label",
        "estimate", "estimate_pp", "standard_error", "ci_lower", "ci_upper", "ci_lower_pp",
        "ci_upper_pp", "p_value", "adjusted_p_value", "interpretation_unit", "reference_group",
        "n_rows", "n_cited", "n_prompts", "n_urls", "n_domains", "prompt_clusters",
        "url_clusters", "se_method", "fixed_effects", "sample_restriction", "model_status",
        "model_warning", "model_version", "formula", "source_artifact", "model_change",
    ),
    "feature_probability_contrasts.csv": (
        "feature_name", "contrast_name", "condition_a", "condition_b", "probability_a",
        "probability_b", "difference_pp", "ci_lower_pp", "ci_upper_pp", "model_id", "notes",
    ),
    "feature_subgroup_statistics.csv": (
        "feature_name", "subgroup_dimension", "subgroup_name", "feature_state", "n_rows",
        "n_cited", "n_more_only", "cited_rate", "ci_lower", "ci_upper", "n_prompts",
        "n_urls", "support_flag", "panel_type",
    ),
    "feature_related_associations.csv": (
        "feature_name", "related_feature", "association_measure", "association", "direction",
        "magnitude", "pairwise_n", "missing_rate", "related_feature_layer", "same_model",
        "possible_interpretation",
    ),
    "feature_multicollinearity_diagnostics.csv": (
        "feature_name", "strongest_related_feature", "strongest_association", "vif",
        "condition_number", "g1_estimate_pp", "g2_estimate_pp", "coefficient_change_pp",
        "standard_error_change", "sign_change", "risk_classification", "explanation",
    ),
    "feature_confounding_diagnostics.csv": (
        "feature_name", "risk_dimension", "comparison_model", "baseline_estimate_pp",
        "comparison_estimate_pp", "absolute_change_pp", "relative_change", "sign_flip",
        "baseline_n", "comparison_n", "sample_change", "over_control_risk", "classification",
        "explanation",
    ),
    "feature_evidence_quality.csv": (
        "feature_name", "dimension", "status", "supporting_statistic", "explanation", "limitation",
    ),
    "feature_sample_audit.csv": (
        "feature_name", "stage_order", "stage", "n_rows", "rows_lost", "cited_rate",
        "n_prompts", "n_urls", "n_domains", "repeated_url_frequency",
    ),
    "feature_example_pages.csv": (
        "feature_name", "feature_value", "feature_state", "cited", "prompt_id", "prompt_text",
        "intent", "title", "normalized_url", "source_root_domain", "page_type",
        "page_type_family", "source_type", "content_strength", "extraction_scope", "language",
        "relevant_excerpt", "evidence_highlight", "audit_timestamp", "scrape_timestamp",
        "example_group", "example_quality_flag",
    ),
    "feature_comparable_pairs.csv": (
        "feature_name", "pair_id", "prompt_id", "distance_score", "match_quality",
        "exact_match_fields", "unmatched_differences", "cited_title", "cited_url",
        "cited_domain", "cited_feature_value", "uncited_title", "uncited_url", "uncited_domain",
        "uncited_feature_value", "cited_page_type_family", "uncited_page_type_family",
        "cited_source_type", "uncited_source_type", "cited_content_strength",
        "uncited_content_strength", "cited_word_count", "uncited_word_count",
        "cited_relevance", "uncited_relevance", "warning",
    ),
}


@dataclass
class FrontendArtifacts:
    root: Path
    manifest: dict[str, Any]
    overview: dict[str, Any]
    tables: dict[str, pd.DataFrame]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return center - half, center + half


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _binary_series(values: pd.Series) -> pd.Series:
    mapped = values.map(lambda value: math.nan if pd.isna(value) else int(_bool_value(value)))
    return pd.to_numeric(mapped, errors="coerce")


def _registry_metadata(package: Path) -> dict[str, dict[str, Any]]:
    registry = pd.read_csv(package / "tables/core_general_content_feature_dictionary.csv", low_memory=False)
    registry = registry[registry.get("registry_record_type", "canonical").astype(str).eq("canonical")]
    return {str(row["feature_name"]): row.to_dict() for _, row in registry.iterrows()}


def _attach_context(bundle: qa.QABundle) -> pd.DataFrame:
    rows = bundle.writing_rows.copy()
    context_columns = [
        "normalized_url", "llm_page_type_general", "llm_page_type_family_general",
        "llm_site_type_general", "page_type_general_rule_v2", "page_type_family_general_rule_v2",
        "site_type_general_rule_v2", "scrape_success",
    ]
    context = bundle.url_evidence[[column for column in context_columns if column in bundle.url_evidence]].copy()
    rows = rows.merge(context, on="normalized_url", how="left", validate="many_to_one", suffixes=("", "_url"))
    rows["page_type"] = rows.get("llm_page_type_general", pd.Series(index=rows.index, dtype=object)).fillna(rows.get("page_type_general_rule_v2")).fillna(rows.get("page_type_url_seed_general")).fillna("unknown")
    rows["page_type_family"] = rows.get("llm_page_type_family_general", pd.Series(index=rows.index, dtype=object)).fillna(rows.get("page_type_family_general_rule_v2")).fillna(rows.get("page_type_family_general")).fillna("unknown")
    rows["source_type"] = rows.get("llm_site_type_general", pd.Series(index=rows.index, dtype=object)).fillna(rows.get("site_type_general_rule_v2")).fillna(rows.get("site_type_general")).fillna("unknown")
    rows["extraction_scope"] = rows.get("feature_extraction_text_scope", pd.Series("unknown", index=rows.index)).fillna("unknown")
    rows["language"] = rows.get("prompt_language", pd.Series("not_available", index=rows.index)).fillna("not_available")
    return rows


def _feature_state(rows: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    values = rows[spec.name]
    if spec.kind == "binary":
        return _binary_series(values).map({0.0: "absent", 1.0: "present"})
    if spec.kind == "categorical":
        return values.astype("string").fillna("unknown")
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median()
    return pd.Series(np.where(numeric.isna(), None, np.where(numeric >= median, "high", "low")), index=rows.index, dtype="object")


def _rate_rows(rows: pd.DataFrame, spec: FeatureSpec) -> list[dict[str, Any]]:
    data = rows.dropna(subset=[spec.name]).copy()
    if data.empty:
        return []
    if spec.kind == "continuous":
        numeric = pd.to_numeric(data[spec.name], errors="coerce")
        data = data[numeric.notna()].copy()
        numeric = numeric[numeric.notna()]
        try:
            bins = pd.qcut(numeric, 4, duplicates="drop")
        except ValueError:
            bins = pd.cut(numeric, bins=min(4, max(1, numeric.nunique())), duplicates="drop")
        data["_level"] = bins.astype(str)
        ordered = list(dict.fromkeys(data["_level"].tolist()))
        order = {value: index for index, value in enumerate(ordered)}
        data["_state"] = data["_level"]
    else:
        data["_level"] = _binary_series(data[spec.name]).map({0: "absent", 1: "present"}) if spec.kind == "binary" else data[spec.name].astype(str)
        data["_state"] = data["_level"]
        order = {value: index for index, value in enumerate(data["_level"].drop_duplicates().tolist())}
    output = []
    for level, group in data.groupby("_level", sort=False, observed=True):
        cited = int(pd.to_numeric(group["cited"], errors="coerce").fillna(0).sum())
        total = len(group)
        low, high = _wilson(cited, total)
        output.append({
            "feature_name": spec.name, "feature_level": str(level), "feature_state": str(group["_state"].iloc[0]),
            "bin_order": order[str(level)], "n_rows": total, "n_cited": cited, "n_more_only": total - cited,
            "cited_rate": cited / total, "ci_lower": low, "ci_upper": high,
            "n_prompts": int(group["prompt_id"].nunique()), "n_urls": int(group["normalized_url"].nunique()),
            "n_domains": int(group["source_root_domain"].nunique()),
            "support_flag": "supported" if total >= 20 and cited >= 5 and total - cited >= 5 else "low_support",
            "source_artifact": "data/content_lpm_measurable_rows_with_writing_factual_features.csv",
        })
    return output


def _feature_from_term(term: str) -> str | None:
    term = str(term)
    for categorical in ("heading_count_group", "link_count_group", "content_strength"):
        if f"C({categorical}" in term:
            return categorical
    for name in SPEC_BY_NAME:
        if term == name or term.startswith(name + "_winsorized"):
            return name
    return None


def _term_label(term: str, feature: str) -> str:
    match = re.search(r"\[T\.(.+)\]$", term)
    if match:
        return match.group(1)
    return SPEC_BY_NAME[feature].interpretation_unit


MODEL_SOURCES = (
    ("G1", "headline", "tables/09_content_feature_econometrics/M1_one_feature_prompt_fe_results.csv", None, "One feature plus prompt fixed effects"),
    ("G2", "headline", "tables/09_content_feature_econometrics/M2_preferred_joint_lpm_results.csv", None, "Adds the joint structural content controls"),
    ("G3", "robustness", "tables/09_content_feature_econometrics/M3_domain_fe_results.csv", None, "Uses within-domain comparisons where supported"),
    ("G4A", "sensitivity", "tables/09_content_feature_econometrics/M4R_rule_v2_taxonomy_robustness_results.csv", None, "Adds metadata-only Rule-v2 taxonomy"),
    ("G4B", "sensitivity", "tables/09_content_feature_econometrics/M4_gemini_taxonomy_sensitivity_results.csv", None, "Adds Gemini content-informed taxonomy"),
    ("G5A", "sensitivity", "tables/09_content_feature_econometrics/M5_strong_content_sensitivity_results.csv", "M5_M2_strong", "Restricts to strong extraction quality"),
    ("G7", "sensitivity", "tables/09_content_feature_econometrics/M7_logit_ame_crosscheck_results.csv", None, "Logit average marginal-effect cross-check"),
    ("G8", "sensitivity", "tables/09_content_feature_econometrics/M10_outlier_winsorized_sensitivity_results.csv", None, "Removes or winsorizes upper-tail observations"),
    ("G1", "headline", "tables/11_writing_factual_density_econometrics/F_one_feature_prompt_fe_results.csv", None, "One feature plus prompt fixed effects"),
    ("G2", "headline", "tables/11_writing_factual_density_econometrics/W_joint_writing_factual_results.csv", "W3_structural_plus_writing_factual", "Adds joint structural and writing/factual controls"),
    ("G3", "robustness", "tables/11_writing_factual_density_econometrics/D_domain_fe_writing_factual_results.csv", "D_W3_domain_fe", "Uses within-domain comparisons where supported"),
    ("G4A", "sensitivity", "tables/11_writing_factual_density_econometrics/P_page_function_sensitivity_results.csv", "P_W3_rule_v2_taxonomy", "Adds metadata-only Rule-v2 taxonomy"),
    ("G4B", "sensitivity", "tables/11_writing_factual_density_econometrics/P_page_function_sensitivity_results.csv", "P_W3_gemini_taxonomy", "Adds Gemini content-informed taxonomy"),
    ("G5A", "sensitivity", "tables/11_writing_factual_density_econometrics/S_text_scope_content_strength_sensitivity.csv", "S_strong_content_W3", "Restricts to strong extraction quality"),
    ("G5B", "diagnostic", "tables/11_writing_factual_density_econometrics/S_text_scope_content_strength_sensitivity.csv", "S_full_text_W3", "Restricts to full-text-equivalent extraction"),
    ("G5C", "diagnostic", "tables/11_writing_factual_density_econometrics/S_text_scope_content_strength_sensitivity.csv", "S_excerpt_only_W3", "Restricts to excerpt-only extraction"),
    ("G8", "sensitivity", "tables/11_writing_factual_density_econometrics/O_outlier_sensitivity_writing_factual_results.csv", "_W3", "Removes or winsorizes upper-tail observations"),
)


def _model_estimates(package: Path, rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    structural = {"log2_word_count_plus1", "has_table", "heading_count_group", "link_count_group", "content_strength"}
    for model_id, role, relative, selector, change in MODEL_SOURCES:
        path = package / relative
        if not path.exists():
            continue
        table = pd.read_csv(path, low_memory=False)
        if selector:
            if selector.startswith("_"):
                table = table[table["model_id"].astype(str).str.endswith(selector)]
            else:
                table = table[table["model_id"].astype(str).eq(selector)]
        for _, row in table.iterrows():
            feature = _feature_from_term(str(row.get("term", "")))
            if feature is None:
                continue
            is_notebook_11 = "11_writing" in relative
            if is_notebook_11 and feature in structural:
                continue
            if not is_notebook_11 and feature not in structural:
                continue
            ci_low = pd.to_numeric(pd.Series([row.get("conf_low")]), errors="coerce").iloc[0]
            ci_high = pd.to_numeric(pd.Series([row.get("conf_high")]), errors="coerce").iloc[0]
            valid = pd.notna(ci_low) and pd.notna(ci_high) and np.isfinite(ci_low) and np.isfinite(ci_high)
            formula = str(row.get("formula", ""))
            fixed = []
            if "C(prompt_id)" in formula:
                fixed.append("prompt_id")
            if "C(source_root_domain)" in formula:
                fixed.append("source_root_domain")
            source_model = str(row.get("model_id", model_id))
            n_rows = int(pd.to_numeric(pd.Series([row.get("n_obs")]), errors="coerce").fillna(0).iloc[0])
            output.append({
                "feature_name": feature, "model_id": model_id, "source_model_id": source_model,
                "model_role": role, "term": row.get("term"), "term_label": _term_label(str(row.get("term")), feature),
                "estimate": row.get("estimate"), "estimate_pp": row.get("estimate_pp"),
                "standard_error": row.get("std_error"), "ci_lower": row.get("conf_low"), "ci_upper": row.get("conf_high"),
                "ci_lower_pp": row.get("conf_low_pp"), "ci_upper_pp": row.get("conf_high_pp"),
                "p_value": row.get("p_value"), "adjusted_p_value": math.nan,
                "interpretation_unit": SPEC_BY_NAME[feature].interpretation_unit,
                "reference_group": SPEC_BY_NAME[feature].reference_group,
                "n_rows": n_rows, "n_cited": int(rows["cited"].sum()) if n_rows == len(rows) else math.nan,
                "n_prompts": row.get("n_prompts"), "n_urls": row.get("n_urls"), "n_domains": row.get("n_domains"),
                "prompt_clusters": row.get("n_prompts") if "cluster_prompt" in str(row.get("cov_type", "")) or "two_way" in str(row.get("cov_type", "")) else math.nan,
                "url_clusters": row.get("n_urls") if "normalized_url" in str(row.get("cov_type", "")) or "two_way" in str(row.get("cov_type", "")) else math.nan,
                "se_method": row.get("cov_type", "unknown"), "fixed_effects": ";".join(fixed) or "none",
                "sample_restriction": str(row.get("notes", "")),
                "model_status": "available" if valid else "invalid_covariance",
                "model_warning": str(row.get("warning", "")) if valid else "Confidence interval is non-finite; estimate is diagnostic only.",
                "model_version": source_model, "formula": formula, "source_artifact": relative,
                "model_change": change,
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_model_estimates.csv"])


def _preferred_estimate(estimates: pd.DataFrame, feature: str, model: str) -> pd.Series | None:
    subset = estimates[(estimates["feature_name"] == feature) & (estimates["model_id"] == model)].copy()
    if subset.empty:
        return None
    order = pd.Categorical(subset["se_method"], ["two_way_cluster_prompt_url", "cluster_normalized_url", "cluster_prompt_id", "HC3", "logit_standard_ame"], ordered=True)
    subset = subset.assign(_order=order).sort_values(["model_status", "_order"], ascending=[True, True])
    return subset.iloc[0]


def _association(x: pd.Series, y: pd.Series, x_kind: str, y_kind: str) -> tuple[str, float, int, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)
    missing = 1 - n / max(1, len(x))
    if n < 5:
        return "not_assessable", math.nan, n, missing
    if x_kind in {"continuous", "binary"} and y_kind in {"continuous", "binary"}:
        values = data[["x", "y"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(values) < 5 or values.nunique().min() < 2:
            return "not_assessable", math.nan, len(values), missing
        method = "phi" if x_kind == y_kind == "binary" else "point_biserial" if "binary" in {x_kind, y_kind} else "spearman"
        corr = values.corr(method="spearman" if method == "spearman" else "pearson").iloc[0, 1]
        return method, float(corr), len(values), missing
    if x_kind == "categorical" and y_kind == "categorical":
        table = pd.crosstab(data["x"], data["y"])
        observed = table.to_numpy(dtype=float)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
        chi2 = np.divide((observed - expected) ** 2, expected, out=np.zeros_like(expected), where=expected > 0).sum()
        denom = observed.sum() * max(1, min(observed.shape[0] - 1, observed.shape[1] - 1))
        return "cramers_v", float(math.sqrt(chi2 / denom)), n, missing
    categorical, numeric = (data["x"], data["y"]) if x_kind == "categorical" else (data["y"], data["x"])
    numeric = pd.to_numeric(numeric, errors="coerce")
    valid = numeric.notna()
    categorical, numeric = categorical[valid], numeric[valid]
    grand = numeric.mean()
    denominator = ((numeric - grand) ** 2).sum()
    numerator = sum(len(group) * (group.mean() - grand) ** 2 for _, group in numeric.groupby(categorical, observed=True))
    eta = math.sqrt(numerator / denominator) if denominator > 0 else math.nan
    return "correlation_ratio_eta", float(eta), len(numeric), missing


def _related_associations(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for spec in FEATURE_SPECS:
        if spec.name not in rows:
            continue
        for other in FEATURE_SPECS:
            if other.name == spec.name or other.name not in rows:
                continue
            method, value, n, missing = _association(rows[spec.name], rows[other.name], spec.kind, other.kind)
            absolute = abs(value) if pd.notna(value) else math.nan
            output.append({
                "feature_name": spec.name, "related_feature": other.name, "association_measure": method,
                "association": value, "direction": "positive" if pd.notna(value) and value > 0 else "negative" if pd.notna(value) and value < 0 else "unsigned_or_uncertain",
                "magnitude": "strong" if pd.notna(absolute) and absolute >= .5 else "moderate" if pd.notna(absolute) and absolute >= .25 else "weak" if pd.notna(absolute) else "not_assessable",
                "pairwise_n": n, "missing_rate": missing, "related_feature_layer": other.layer,
                "same_model": other.name in {"log2_word_count_plus1", "has_table", "heading_count_group", "link_count_group", "content_strength", "factual_numeric_density_score", "prompt_page_relevance_score"},
                "possible_interpretation": "Measured features co-occur; this is not causal influence.",
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_related_associations.csv"])


def _vif_and_condition(rows: pd.DataFrame) -> tuple[dict[str, float], float]:
    names = [spec.name for spec in FEATURE_SPECS if spec.kind != "categorical" and spec.name in rows]
    matrix = rows[names].apply(pd.to_numeric, errors="coerce").dropna()
    if len(matrix) < 20:
        return {}, math.nan
    std = matrix.std(ddof=0).replace(0, np.nan)
    z = ((matrix - matrix.mean()) / std).dropna(axis=1).to_numpy()
    kept = std.dropna().index.tolist()
    condition = float(np.linalg.cond(z)) if z.size else math.nan
    output: dict[str, float] = {}
    for index, name in enumerate(kept):
        y = z[:, index]
        x = np.delete(z, index, axis=1)
        x = np.column_stack([np.ones(len(x)), x])
        fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        total = ((y - y.mean()) ** 2).sum()
        residual = ((y - fitted) ** 2).sum()
        r2 = 1 - residual / total if total else 1
        output[name] = float(1 / max(1e-9, 1 - r2))
    return output, condition


def _multicollinearity(estimates: pd.DataFrame, associations: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    vifs, condition = _vif_and_condition(rows)
    output = []
    for spec in FEATURE_SPECS:
        related = associations[associations["feature_name"].eq(spec.name)].copy()
        related["_abs"] = pd.to_numeric(related["association"], errors="coerce").abs()
        strongest = related.sort_values("_abs", ascending=False).head(1)
        g1, g2 = _preferred_estimate(estimates, spec.name, "G1"), _preferred_estimate(estimates, spec.name, "G2")
        g1_value = float(g1["estimate_pp"]) if g1 is not None else math.nan
        g2_value = float(g2["estimate_pp"]) if g2 is not None else math.nan
        vif = vifs.get(spec.name, math.nan)
        risk = "not_assessable_for_this_feature" if spec.kind == "categorical" else "high_multicollinearity_concern" if pd.notna(vif) and vif >= 10 else "moderate_overlap_with_other_predictors" if pd.notna(vif) and vif >= 5 else "low_apparent_multicollinearity_risk"
        output.append({
            "feature_name": spec.name,
            "strongest_related_feature": strongest.iloc[0]["related_feature"] if not strongest.empty else "not_available",
            "strongest_association": strongest.iloc[0]["association"] if not strongest.empty else math.nan,
            "vif": vif, "condition_number": condition, "g1_estimate_pp": g1_value, "g2_estimate_pp": g2_value,
            "coefficient_change_pp": g2_value - g1_value if pd.notna(g1_value) and pd.notna(g2_value) else math.nan,
            "standard_error_change": float(g2["standard_error"] - g1["standard_error"]) if g1 is not None and g2 is not None else math.nan,
            "sign_change": bool(np.sign(g1_value) != np.sign(g2_value)) if pd.notna(g1_value) and pd.notna(g2_value) else False,
            "risk_classification": risk,
            "explanation": "Multicollinearity affects precision and separation of joint associations; it does not invalidate a measured feature.",
        })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_multicollinearity_diagnostics.csv"])


def _confounding(estimates: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("domain_or_template", "G3", "Domain fixed effects may absorb publisher/template differences.", False),
        ("metadata_taxonomy", "G4A", "Metadata-only taxonomy may proxy page function.", True),
        ("content_informed_taxonomy", "G4B", "Content-informed taxonomy can overlap with page-body features.", True),
        ("extraction_quality", "G5A", "Strong-content restriction tests extraction sensitivity.", False),
        ("outliers", "G8", "Tail restrictions test outlier sensitivity.", False),
    )
    output = []
    for spec in FEATURE_SPECS:
        baseline = _preferred_estimate(estimates, spec.name, "G2")
        for risk, model_id, explanation, over_control in comparisons:
            comparison = _preferred_estimate(estimates, spec.name, model_id)
            base = float(baseline["estimate_pp"]) if baseline is not None else math.nan
            comp = float(comparison["estimate_pp"]) if comparison is not None else math.nan
            change = comp - base if pd.notna(base) and pd.notna(comp) else math.nan
            relative = abs(change / base) if pd.notna(change) and base != 0 else math.nan
            sign_flip = bool(np.sign(base) != np.sign(comp)) if pd.notna(base) and pd.notna(comp) else False
            classification = "not_available" if comparison is None or baseline is None else "substantial_sensitivity" if sign_flip or relative >= .6 else "moderate_sensitivity" if relative >= .3 else "limited_observed_sensitivity"
            output.append({
                "feature_name": spec.name, "risk_dimension": risk, "comparison_model": model_id,
                "baseline_estimate_pp": base, "comparison_estimate_pp": comp, "absolute_change_pp": change,
                "relative_change": relative, "sign_flip": sign_flip,
                "baseline_n": baseline["n_rows"] if baseline is not None else math.nan,
                "comparison_n": comparison["n_rows"] if comparison is not None else math.nan,
                "sample_change": (comparison["n_rows"] - baseline["n_rows"]) if baseline is not None and comparison is not None else math.nan,
                "over_control_risk": over_control, "classification": classification, "explanation": explanation,
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_confounding_diagnostics.csv"])


def _robustness(feature: str, estimates: pd.DataFrame, approved: bool, support: int) -> str:
    if not approved:
        return "not_model_ready"
    if support < 100:
        return "insufficient_support"
    g2, g3, g5, g8 = (_preferred_estimate(estimates, feature, model) for model in ("G2", "G3", "G5A", "G8"))
    if g2 is None:
        return "suggestive" if _preferred_estimate(estimates, feature, "G1") is not None else "insufficient_support"
    base = float(g2["estimate_pp"])
    low, high = float(g2["ci_lower_pp"]), float(g2["ci_upper_pp"])
    for candidate, label in ((g3, "domain_template_confounded"), (g5, "extraction_sensitive"), (g8, "outlier_sensitive")):
        if candidate is not None:
            value = float(candidate["estimate_pp"])
            if np.sign(value) != np.sign(base) or (base and abs(value / base) < .4):
                return label
    if low <= 0 <= high:
        return "null_or_imprecise"
    return "stable_association"


def _raw_difference(rate_table: pd.DataFrame, spec: FeatureSpec) -> float:
    part = rate_table[rate_table["feature_name"].eq(spec.name)].sort_values("bin_order")
    if len(part) < 2:
        return math.nan
    return float((part.iloc[-1]["cited_rate"] - part.iloc[0]["cited_rate"]) * 100)


def _scorecard(package: Path, rows: pd.DataFrame, rates: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    metadata = _registry_metadata(package)
    output = []
    for spec in FEATURE_SPECS:
        if spec.name not in rows:
            continue
        registry = metadata.get(spec.name, {})
        measured = rows[spec.name].notna()
        support_rows = int(measured.sum())
        approved = _bool_value(registry.get("approved_for_model_v1", False))
        g1, g2, g3 = (_preferred_estimate(estimates, spec.name, model) for model in ("G1", "G2", "G3"))
        robustness = _robustness(spec.name, estimates, approved, support_rows)
        g2_value = float(g2["estimate_pp"]) if g2 is not None else math.nan
        raw = _raw_difference(rates, spec)
        raw_direction = "higher" if raw > 0 else "lower" if raw < 0 else "similar"
        adjusted_direction = "positive" if g2_value > 0 else "negative" if g2_value < 0 else "not available"
        domain_text = " A domain-FE estimate is unavailable." if g3 is None else f" The domain-FE estimate was {float(g3['estimate_pp']):+.1f} pp."
        summary = (
            f"Among surfaced sources, the highest displayed level had a {raw_direction} unadjusted cited rate "
            f"({raw:+.1f} pp versus the lowest displayed level)."
            if pd.notna(raw)
            else "The available values do not support a two-level unadjusted cited-rate contrast."
        )
        if g2 is not None:
            summary += f" The joint prompt-FE association was {adjusted_direction} ({g2_value:+.1f} pp; 95% CI {float(g2['ci_lower_pp']):+.1f} to {float(g2['ci_upper_pp']):+.1f})." + domain_text
        else:
            summary += " A joint-model estimate is not available for this feature."
        warning = spec.warning or str(registry.get("model_entry_blocker", "") or "")
        output.append({
            "feature_name": spec.name, "human_label": spec.label, "feature_layer": spec.layer,
            "feature_group": spec.group, "feature_type": spec.kind,
            "measurement_status": registry.get("current_implementation_status", "implemented_from_existing_artifact"),
            "qa_status": registry.get("qa_status", "not_started"), "approved_for_model": approved,
            "support_rows": support_rows, "support_prompts": int(rows.loc[measured, "prompt_id"].nunique()),
            "support_urls": int(rows.loc[measured, "normalized_url"].nunique()),
            "support_domains": int(rows.loc[measured, "source_root_domain"].nunique()),
            "missing_rate": float(1 - measured.mean()), "raw_association_pp": raw,
            "g1_estimate_pp": g1["estimate_pp"] if g1 is not None else math.nan,
            "g2_estimate_pp": g2_value, "domain_fe_estimate_pp": g3["estimate_pp"] if g3 is not None else math.nan,
            "g2_ci_lower_pp": g2["ci_lower_pp"] if g2 is not None else math.nan,
            "g2_ci_upper_pp": g2["ci_upper_pp"] if g2 is not None else math.nan,
            "robustness_status": robustness,
            "evidence_quality_label": "insufficient measurement quality" if not approved else "stronger descriptive evidence" if robustness == "stable_association" else "suggestive but confounded" if robustness.endswith("confounded") else "exploratory and unstable",
            "interpretation_summary": summary, "primary_warning": warning,
            "dataset_version": "area_condo_content_lpm_measurable_rows_v1",
            "feature_registry_version": registry.get("registry_version", "core_general_registry_unknown"),
        })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["core_general_feature_scorecard.csv"])


def _subgroups(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    dimensions = [column for column in ("intent", "page_type", "page_type_family", "source_type", "content_strength", "extraction_scope", "language") if column in rows]
    for spec in FEATURE_SPECS:
        if spec.name not in rows:
            continue
        data = rows.dropna(subset=[spec.name]).copy()
        data["_state"] = _feature_state(data, spec)
        for dimension in dimensions:
            for (subgroup, state), group in data.groupby([dimension, "_state"], observed=True):
                total = len(group)
                cited = int(group["cited"].sum())
                low, high = _wilson(cited, total)
                output.append({
                    "feature_name": spec.name, "subgroup_dimension": dimension, "subgroup_name": str(subgroup),
                    "feature_state": str(state), "n_rows": total, "n_cited": cited, "n_more_only": total - cited,
                    "cited_rate": cited / total, "ci_lower": low, "ci_upper": high,
                    "n_prompts": int(group["prompt_id"].nunique()), "n_urls": int(group["normalized_url"].nunique()),
                    "support_flag": "supported" if total >= 20 and cited >= 5 and total - cited >= 5 else "low_support",
                    "panel_type": "descriptive_subgroup_comparison",
                })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_subgroup_statistics.csv"])


def _examples(rows: pd.DataFrame, generated_at: str) -> pd.DataFrame:
    output = []
    for spec in FEATURE_SPECS:
        if spec.name not in rows:
            continue
        data = rows.dropna(subset=[spec.name]).copy()
        data["_state"] = _feature_state(data, spec)
        if spec.kind == "categorical":
            data["_presence"] = np.where(data[spec.name].astype(str).eq(spec.reference_group), "absent", "present")
        else:
            data["_presence"] = data["_state"].map({"low": "absent", "high": "present"}).fillna(data["_state"])
        data["_citation"] = data["cited"].map({1: "cited", 0: "not_cited"})
        data["_group"] = "feature_" + data["_presence"].astype(str) + "_and_" + data["_citation"]
        data = data.sort_values(["_group", "source_root_domain", "normalized_url", "prompt_id"])
        for _, row in data.groupby("_group", observed=True).head(8).iterrows():
            value = row[spec.name]
            output.append({
                "feature_name": spec.name, "feature_value": value, "feature_state": row["_state"],
                "cited": int(row["cited"]), "prompt_id": row["prompt_id"], "prompt_text": row.get("prompt_text", ""),
                "intent": row.get("intent", "unknown"), "title": row.get("url_title", ""),
                "normalized_url": row["normalized_url"], "source_root_domain": row["source_root_domain"],
                "page_type": row.get("page_type", "unknown"), "page_type_family": row.get("page_type_family", "unknown"),
                "source_type": row.get("source_type", "unknown"), "content_strength": row.get("content_strength", "unknown"),
                "extraction_scope": row.get("extraction_scope", "unknown"), "language": row.get("language", "not_available"),
                "relevant_excerpt": str(row.get("page_text_excerpt", ""))[:900],
                "evidence_highlight": f"{spec.label}: {value}", "audit_timestamp": generated_at,
                "scrape_timestamp": "not_available", "example_group": row["_group"],
                "example_quality_flag": "supported_example" if row.get("content_strength") == "strong" else "check_extraction_quality",
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_example_pages.csv"])


def _pairs(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    distance_fields = ("page_type_family", "source_type", "intent", "content_strength")
    word_scale = max(1.0, pd.to_numeric(rows.get("log2_word_count_plus1"), errors="coerce").std())
    relevance_scale = max(0.01, pd.to_numeric(rows.get("prompt_page_relevance_score"), errors="coerce").std())
    for spec in FEATURE_SPECS:
        if spec.name not in rows:
            continue
        data = rows.dropna(subset=[spec.name]).copy()
        data["_state"] = _feature_state(data, spec)
        candidates = []
        for prompt_id, group in data.groupby("prompt_id", sort=True):
            cited = group[group["cited"].eq(1)].sort_values(["normalized_url", "record_id"])
            uncited = group[group["cited"].eq(0)].sort_values(["normalized_url", "record_id"])
            for _, left in cited.iterrows():
                for _, right in uncited.iterrows():
                    if str(left["_state"]) == str(right["_state"]):
                        continue
                    mismatches = [field for field in distance_fields if str(left.get(field, "")) != str(right.get(field, ""))]
                    exact = [field for field in distance_fields if field not in mismatches]
                    word_distance = abs(float(left.get("log2_word_count_plus1", 0)) - float(right.get("log2_word_count_plus1", 0))) / word_scale
                    relevance_distance = abs(float(left.get("prompt_page_relevance_score", 0)) - float(right.get("prompt_page_relevance_score", 0))) / relevance_scale
                    same_domain = str(left["source_root_domain"]) == str(right["source_root_domain"])
                    distance = len(mismatches) + .35 * word_distance + .35 * relevance_distance - (.5 if same_domain else 0)
                    candidates.append((distance, str(prompt_id), str(left["record_id"]), str(right["record_id"]), left, right, exact, mismatches))
        used_left: set[str] = set()
        used_right: set[str] = set()
        kept = []
        for candidate in sorted(candidates, key=lambda item: item[:4]):
            _, _, left_id, right_id, *_ = candidate
            if left_id in used_left or right_id in used_right:
                continue
            kept.append(candidate)
            used_left.add(left_id)
            used_right.add(right_id)
            if len(kept) >= 200:
                break
        for index, (distance, prompt_id, _, _, left, right, exact, mismatches) in enumerate(kept, start=1):
            quality = "strong" if distance <= 1 else "moderate" if distance <= 2.5 else "weak"
            output.append({
                "feature_name": spec.name, "pair_id": f"{spec.name}_{index:04d}", "prompt_id": prompt_id,
                "distance_score": distance, "match_quality": quality, "exact_match_fields": ";".join(exact),
                "unmatched_differences": ";".join(mismatches), "cited_title": left.get("url_title", ""),
                "cited_url": left["normalized_url"], "cited_domain": left["source_root_domain"], "cited_feature_value": left[spec.name],
                "uncited_title": right.get("url_title", ""), "uncited_url": right["normalized_url"],
                "uncited_domain": right["source_root_domain"], "uncited_feature_value": right[spec.name],
                "cited_page_type_family": left.get("page_type_family", "unknown"), "uncited_page_type_family": right.get("page_type_family", "unknown"),
                "cited_source_type": left.get("source_type", "unknown"), "uncited_source_type": right.get("source_type", "unknown"),
                "cited_content_strength": left.get("content_strength", "unknown"), "uncited_content_strength": right.get("content_strength", "unknown"),
                "cited_word_count": left.get("word_count"), "uncited_word_count": right.get("word_count"),
                "cited_relevance": left.get("prompt_page_relevance_score"), "uncited_relevance": right.get("prompt_page_relevance_score"),
                "warning": "These pages are observationally similar on displayed variables; unobserved differences remain.",
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_comparable_pairs.csv"])


def _sample_audit(bundle: qa.QABundle, rows: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    output = []
    all_rows = bundle.all_rows.copy()
    all_rows["cited"] = pd.to_numeric(all_rows["cited"], errors="coerce").fillna(0).astype(int)
    for spec in FEATURE_SPECS:
        feature_rows = rows[rows[spec.name].notna()] if spec.name in rows else rows.iloc[0:0]
        model = _preferred_estimate(estimates, spec.name, "G2")
        if model is None:
            model = _preferred_estimate(estimates, spec.name, "G1")
        model_n = int(model["n_rows"]) if model is not None else 0
        stages = [
            ("surfaced", all_rows),
            ("scrape_available", all_rows[all_rows.get("scrape_success", False).fillna(False).astype(bool)] if "scrape_success" in all_rows else all_rows.iloc[0:0]),
            ("content_measurable", all_rows[all_rows.get("content_feature_available", False).fillna(False).astype(bool)] if "content_feature_available" in all_rows else bundle.measurable_rows),
            ("feature_measurable", feature_rows),
        ]
        previous = len(all_rows)
        for order, (stage, frame) in enumerate(stages, start=1):
            cited = pd.to_numeric(frame.get("cited", pd.Series(dtype=float)), errors="coerce").fillna(0)
            output.append({
                "feature_name": spec.name, "stage_order": order, "stage": stage, "n_rows": len(frame),
                "rows_lost": previous - len(frame), "cited_rate": float(cited.mean()) if len(frame) else math.nan,
                "n_prompts": int(frame["prompt_id"].nunique()) if "prompt_id" in frame else 0,
                "n_urls": int(frame["normalized_url"].nunique()) if "normalized_url" in frame else 0,
                "n_domains": int(frame["source_root_domain"].nunique()) if "source_root_domain" in frame else 0,
                "repeated_url_frequency": len(frame) / max(1, frame["normalized_url"].nunique()) if "normalized_url" in frame else math.nan,
            })
            previous = len(frame)
        output.append({
            "feature_name": spec.name, "stage_order": 5, "stage": "model_sample", "n_rows": model_n,
            "rows_lost": max(0, len(feature_rows) - model_n), "cited_rate": math.nan,
            "n_prompts": model["n_prompts"] if model is not None else 0, "n_urls": model["n_urls"] if model is not None else 0,
            "n_domains": model["n_domains"] if model is not None else 0, "repeated_url_frequency": math.nan,
        })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_sample_audit.csv"])


def _probability_contrasts(package: Path) -> pd.DataFrame:
    path = package / "tables/09_content_feature_econometrics/09_actionable_predicted_probability_contrasts.csv"
    output = []
    if path.exists():
        table = pd.read_csv(path)
        for _, row in table.iterrows():
            feature = _feature_from_term(str(row.get("term", "")))
            if feature is None:
                continue
            output.append({
                "feature_name": feature, "contrast_name": row.get("contrast_name"),
                "condition_a": row.get("baseline_condition"), "condition_b": row.get("comparison_condition"),
                "probability_a": row.get("predicted_probability_baseline"), "probability_b": row.get("predicted_probability_comparison"),
                "difference_pp": row.get("difference_pp"), "ci_lower_pp": math.nan, "ci_upper_pp": math.nan,
                "model_id": row.get("model_id"), "notes": row.get("notes"),
            })
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_probability_contrasts.csv"])


def _evidence_quality(scorecard: pd.DataFrame, confounding: pd.DataFrame, multicollinearity: pd.DataFrame) -> pd.DataFrame:
    output = []
    for _, score in scorecard.iterrows():
        feature = score["feature_name"]
        risks = confounding[confounding["feature_name"].eq(feature)]
        multi = multicollinearity[multicollinearity["feature_name"].eq(feature)].iloc[0]
        dimensions = [
            ("measurement_quality", "ready" if score["approved_for_model"] else "pending_qa", score["qa_status"], "Registry approval is required before headline use."),
            ("sample_support", "strong" if score["support_rows"] >= 1000 else "moderate" if score["support_rows"] >= 100 else "low", f"{int(score['support_rows'])} rows", "Support does not remove selection bias."),
            ("estimate_precision", "precise" if pd.notna(score["g2_ci_lower_pp"]) and score["g2_ci_upper_pp"] - score["g2_ci_lower_pp"] <= 10 else "wide_or_unavailable", f"G2 interval width: {score['g2_ci_upper_pp'] - score['g2_ci_lower_pp']:.1f} pp" if pd.notna(score["g2_ci_lower_pp"]) else "G2 unavailable", "Precision depends on covariance choice and repeated observations."),
            ("model_stability", score["robustness_status"], score["evidence_quality_label"], "Stability is assessed across predefined available models."),
            ("domain_template_sensitivity", str(risks.loc[risks["risk_dimension"].eq("domain_or_template"), "classification"].iloc[0]) if (risks["risk_dimension"] == "domain_or_template").any() else "not_available", "G2 versus G3", "Domain FE can be imprecise and changes the comparison sample."),
            ("extraction_sensitivity", str(risks.loc[risks["risk_dimension"].eq("extraction_quality"), "classification"].iloc[0]) if (risks["risk_dimension"] == "extraction_quality").any() else "not_available", "G2 versus G5A", "Content strength is extraction quality, not writing quality."),
            ("functional_form_stability", "not_fully_assessed", "Available bins and outlier checks", "No general nonlinear specification is available for every feature."),
            ("omitted_variable_risk", "substantial_observed_risk" if score["robustness_status"] == "domain_template_confounded" else "moderate_observed_risk", "Observed sensitivity panels", "Authority, freshness, retrieval signals, and non-surfaced sources remain unmeasured."),
            ("cross_industry_portability", "core_general" if score["feature_layer"] == "core_general" else "conditional", score["feature_layer"], "Portability still requires validation in other audits."),
            ("model_readiness", "approved" if score["approved_for_model"] else "not_model_ready", score["measurement_status"], "Approval is registry- and QA-based, not significance-based."),
            ("multicollinearity", multi["risk_classification"], f"VIF: {multi['vif']:.2f}" if pd.notna(multi["vif"]) else "VIF not assessable", "Overlap affects precision and separation of joint associations."),
        ]
        for dimension, status, statistic, limitation in dimensions:
            output.append({"feature_name": feature, "dimension": dimension, "status": status, "supporting_statistic": statistic, "explanation": "Status is derived from versioned measurement, support, and model diagnostics.", "limitation": limitation})
    return pd.DataFrame(output, columns=REQUIRED_SCHEMAS["feature_evidence_quality.csv"])


def build_frontend_artifacts(package_dir: str | Path | None = None, output_dir: str | Path | None = None) -> Path:
    package = Path(package_dir) if package_dir else qa.default_package_dir()
    bundle = qa.load_bundle(package)
    rows = _attach_context(bundle)
    rows["cited"] = pd.to_numeric(rows["cited"], errors="coerce").fillna(0).astype(int)
    generated_at = datetime.now(UTC).isoformat()
    output = Path(output_dir) if output_dir else package / "tables" / ARTIFACT_DIRNAME
    output.mkdir(parents=True, exist_ok=True)

    rates = pd.DataFrame([item for spec in FEATURE_SPECS for item in _rate_rows(rows, spec)], columns=REQUIRED_SCHEMAS["feature_cited_rate_summary.csv"])
    estimates = _model_estimates(package, rows)
    associations = _related_associations(rows)
    multicollinearity = _multicollinearity(estimates, associations, rows)
    confounding = _confounding(estimates)
    scorecard = _scorecard(package, rows, rates, estimates)
    tables = {
        "core_general_feature_scorecard.csv": scorecard,
        "feature_cited_rate_summary.csv": rates,
        "feature_model_estimates.csv": estimates,
        "feature_probability_contrasts.csv": _probability_contrasts(package),
        "feature_subgroup_statistics.csv": _subgroups(rows),
        "feature_related_associations.csv": associations,
        "feature_multicollinearity_diagnostics.csv": multicollinearity,
        "feature_confounding_diagnostics.csv": confounding,
        "feature_evidence_quality.csv": _evidence_quality(scorecard, confounding, multicollinearity),
        "feature_sample_audit.csv": _sample_audit(bundle, rows, estimates),
        "feature_example_pages.csv": _examples(rows, generated_at),
        "feature_comparable_pairs.csv": _pairs(rows),
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)

    threshold_path = Path(__file__).resolve().parents[2] / "config" / "model_comparison_thresholds.yaml"
    comparison_tables = model_comparison.build_model_comparison_artifacts(
        package=package,
        output=output,
        estimates=estimates,
        rates=rates,
        rows=rows,
        scorecard=scorecard,
        feature_specs=SPEC_BY_NAME,
        threshold_path=threshold_path,
        generated_at=generated_at,
    )

    summary = qa.bundle_summary(bundle)
    overview = {
        "contract_version": CONTRACT_VERSION, "generated_at": generated_at,
        "scope_statement": "Results describe associations among sources already surfaced in this audit. They are not causal effects or web-wide citation probabilities.",
        "surfaced_source_rows": summary["surfaced_rows"], "measurable_content_rows": summary["measurable_rows"],
        "cited_rows": summary["cited_rows"], "overall_cited_rate": summary["cited_rate"],
        "prompts": summary["full_audit_prompts"], "measurable_prompts": summary["measurable_prompts"],
        "unique_urls": summary["unique_urls"], "domains": int(rows["source_root_domain"].nunique()),
        "model_ready_features": int(scorecard["approved_for_model"].sum()),
        "features_blocked_pending_qa": int((~scorecard["approved_for_model"]).sum()),
        "supported_features": scorecard["feature_name"].tolist(),
        "dataset_version": "area_condo_content_lpm_measurable_rows_v1",
        "feature_registry_version": ";".join(sorted(scorecard["feature_registry_version"].dropna().astype(str).unique())),
    }
    overview_path = output / "econometrics_overview_summary.json"
    overview_path.write_text(json.dumps(overview, indent=2, ensure_ascii=True), "utf-8")

    artifacts = {}
    for path in sorted(output.glob("*.csv")):
        frame = tables[path.name]
        artifacts[path.name] = {"sha256": _sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    for filename, frame in comparison_tables.items():
        path = output / filename
        artifacts[filename] = {"sha256": _sha256(path), "rows": len(frame), "columns": list(frame.columns)}
    for filename in ("model_comparison_thresholds.yaml", "econometrics_model_comparison_manifest.json"):
        path = output / filename
        artifacts[filename] = {"sha256": _sha256(path), "rows": 1, "columns": []}
    artifacts[overview_path.name] = {"sha256": _sha256(overview_path), "rows": 1, "columns": sorted(overview)}
    manifest = {
        "contract_version": CONTRACT_VERSION, "generated_at": generated_at,
        "package_dir": str(package.resolve()), "artifact_dir": str(output.resolve()),
        "artifacts": artifacts,
        "source_artifacts": sorted(set(estimates["source_artifact"].dropna().astype(str).tolist()) | {
            "data/content_lpm_all_surfaced_rows.csv", "data/content_lpm_measurable_rows_with_writing_factual_features.csv",
            "data/url_content_evidence_compact.csv", "tables/core_general_content_feature_dictionary.csv",
        }),
        "guardrails": {"models_fit_in_frontend": False, "raw_brightdata_loaded_in_frontend": False, "causal_interpretation_allowed": False},
        "model_comparison_contract": model_comparison.COMPARISON_CONTRACT_VERSION,
    }
    (output / "econometrics_frontend_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), "utf-8")
    validate_frontend_artifacts(output, verify_hashes=True)
    return output


def validate_frontend_artifacts(root: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "econometrics_frontend_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Frontend manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"Unsupported frontend contract: {manifest.get('contract_version')}")
    overview_path = root / "econometrics_overview_summary.json"
    if not overview_path.exists():
        raise FileNotFoundError(f"Required frontend artifact not found: {overview_path}")
    overview_entry = manifest.get("artifacts", {}).get(overview_path.name)
    if not overview_entry:
        raise ValueError("econometrics_overview_summary.json is missing from the frontend manifest")
    if verify_hashes and overview_entry.get("sha256") != _sha256(overview_path):
        raise ValueError("econometrics_overview_summary.json hash does not match the manifest")
    for filename, required in REQUIRED_SCHEMAS.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required frontend artifact not found: {path}")
        table = pd.read_csv(path, low_memory=False)
        missing = [column for column in required if column not in table]
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        manifest_entry = manifest.get("artifacts", {}).get(filename)
        if not manifest_entry:
            raise ValueError(f"{filename} is missing from the frontend manifest")
        if int(manifest_entry.get("rows", -1)) != len(table):
            raise ValueError(f"{filename} row count does not match the manifest")
        if verify_hashes and manifest_entry.get("sha256") != _sha256(path):
            raise ValueError(f"{filename} hash does not match the manifest")
        if filename == "core_general_feature_scorecard.csv":
            names = table["feature_name"].fillna("").astype(str).str.casefold()
            forbidden = [name for name in names if any(token in name for token in FORBIDDEN_PREDICTOR_TOKENS)]
            if forbidden:
                raise ValueError(f"Frontend scorecard contains leakage-risk features: {sorted(set(forbidden))}")
        if filename == "feature_model_estimates.csv":
            terms = table["term"].fillna("").astype(str)
            if terms.str.startswith(("Intercept", "C(prompt_id)", "C(source_root_domain)")).any():
                raise ValueError("Frontend model estimates expose fixed-effect or intercept coefficients")
            forbidden = [term for term in terms.str.casefold() if any(token in term for token in FORBIDDEN_PREDICTOR_TOKENS)]
            if forbidden:
                raise ValueError("Frontend model estimates contain leakage-risk predictors")
    model_comparison.validate_model_comparison_artifacts(root, verify_hashes=verify_hashes)
    return manifest


def load_frontend_artifacts(root: str | Path, verify_hashes: bool = True) -> FrontendArtifacts:
    root = Path(root)
    manifest = validate_frontend_artifacts(root, verify_hashes=verify_hashes)
    overview = json.loads((root / "econometrics_overview_summary.json").read_text("utf-8"))
    tables = {filename: pd.read_csv(root / filename, low_memory=False) for filename in REQUIRED_SCHEMAS}
    tables.update({filename: pd.read_parquet(root / filename) for filename in model_comparison.COMPARISON_SCHEMAS})
    return FrontendArtifacts(root=root, manifest=manifest, overview=overview, tables=tables)
