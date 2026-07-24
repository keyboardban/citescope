"""Build validated, offline cross-model comparison artifacts for the frontend."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


COMPARISON_CONTRACT_VERSION = "econometrics_model_comparison_v1"
MODEL_ORDER = ("G0", "G1", "G2", "G2R", "G3", "G4A", "G4B", "G5A", "G5B", "G5C", "G6", "G7", "G8", "G9")
COVARIANCE_ONLY_MESSAGE = "Clustering changes uncertainty assumptions; it does not add regression controls or remove confounding."

MODEL_REGISTRY = {
    "G0": ("descriptive", "Unadjusted cited-rate or unadjusted LPM contrast", "none", "none", "descriptive"),
    "G1": ("headline", "One-feature within-prompt association", "prompt_id", "focal feature only", "LPM"),
    "G2": ("headline", "Joint core-content association", "prompt_id", "joint content features", "LPM"),
    "G2R": ("sensitivity", "Prompt-page relevance sensitivity", "prompt_id", "G2 plus measured relevance", "LPM"),
    "G3": ("robustness", "Within-domain robustness", "prompt_id;source_root_domain", "joint content features", "LPM"),
    "G4A": ("sensitivity", "Metadata-only Rule-v2 taxonomy sensitivity", "prompt_id", "G2 plus metadata taxonomy", "LPM"),
    "G4B": ("sensitivity", "Gemini content-informed taxonomy sensitivity", "prompt_id", "G2 plus Gemini taxonomy", "LPM"),
    "G5A": ("sensitivity", "Strong-extraction-content sensitivity", "prompt_id", "joint content features", "LPM"),
    "G5B": ("diagnostic", "Full-text-equivalent measurement diagnostic", "prompt_id", "joint content features", "LPM"),
    "G5C": ("diagnostic", "Excerpt-only measurement diagnostic", "prompt_id", "joint content features", "LPM"),
    "G6": ("missingness_audit", "Content availability and missingness audit", "none", "none", "descriptive"),
    "G7": ("cross_check", "Logit average-marginal-effect cross-check", "none", "joint content plus intent and area", "logit_AME"),
    "G8": ("sensitivity", "Predefined outlier and winsorization sensitivity", "prompt_id", "joint content features", "LPM"),
    "G9": ("interaction", "Supported intent-interaction analysis", "prompt_id", "joint content features plus intent interaction", "LPM"),
}

PAIR_SPECS = (
    ("G0", "G1", "raw_to_prompt_fe"),
    ("G1", "G2", "one_feature_to_joint"),
    ("G2", "G2R", "relevance_sensitivity"),
    ("G2", "G3", "domain_fe_sensitivity"),
    ("G2", "G4A", "metadata_taxonomy_sensitivity"),
    ("G2", "G4B", "gemini_taxonomy_sensitivity"),
    ("G4A", "G4B", "taxonomy_method_comparison"),
    ("G2", "G5A", "strong_content_sensitivity"),
    ("G5B", "G5C", "text_scope_sensitivity"),
    ("G2", "G7", "functional_form_sensitivity"),
    ("G2", "G8", "outlier_sensitivity"),
)

COMPARISON_SCHEMAS: dict[str, tuple[str, ...]] = {
    "model_metadata_registry.parquet": (
        "model_id", "model_role", "purpose", "fixed_effects", "controls", "functional_form",
        "artifact_status", "source_model_ids", "source_artifacts", "warning",
    ),
    "feature_model_estimates_harmonized.parquet": (
        "feature_name", "feature_label", "model_id", "source_model_id", "model_role", "term",
        "term_label", "contrast_key", "estimate", "original_estimate_pp", "estimate_pp",
        "standard_error", "standard_error_pp", "ci_lower_pp", "ci_upper_pp", "ci_width_pp",
        "p_value", "adjusted_p_value", "interpretation_unit", "original_interpretation_unit",
        "unit_multiplier", "reference_group", "n_rows", "n_cited", "n_prompts", "n_urls",
        "n_domains", "prompt_clusters", "url_clusters", "se_method", "fixed_effects", "controls",
        "sample_restriction", "functional_form", "focal_feature_definition", "taxonomy_version",
        "extraction_scope", "dataset_version", "model_version", "model_status", "model_warning",
        "formula", "source_artifact", "is_preferred_covariance",
    ),
    "feature_model_comparisons.parquet": (
        "feature_name", "feature_label", "term_label", "contrast_key", "baseline_model_id",
        "comparison_model_id", "baseline_source_model_id", "comparison_source_model_id",
        "comparison_type", "baseline_estimate_pp", "comparison_estimate_pp", "estimate_change_pp",
        "absolute_magnitude_change_pp", "relative_magnitude_change", "relative_change_status",
        "baseline_ci_lower_pp", "baseline_ci_upper_pp", "comparison_ci_lower_pp",
        "comparison_ci_upper_pp", "baseline_ci_width_pp", "comparison_ci_width_pp",
        "ci_width_change_pp", "standard_error_change", "sign_changed",
        "baseline_ci_includes_zero", "comparison_ci_includes_zero", "ci_zero_status_changed",
        "baseline_n_rows", "comparison_n_rows", "rows_change", "rows_change_percent",
        "baseline_n_prompts", "comparison_n_prompts", "prompts_changed", "baseline_n_urls",
        "comparison_n_urls", "urls_changed", "baseline_n_domains", "comparison_n_domains",
        "domains_changed", "prompt_clusters_changed", "url_clusters_changed", "baseline_se_method",
        "comparison_se_method", "same_sample", "same_controls", "same_fixed_effects",
        "same_functional_form", "same_interpretation_unit", "directly_comparable",
        "comparability_status", "comparability_warning", "diagnostic_labels", "explanation",
        "explanation_template_id", "dataset_version", "model_version", "artifact_generated_at",
    ),
    "feature_model_transition_labels.parquet": (
        "feature_name", "term_label", "baseline_model_id", "comparison_model_id",
        "comparison_source_model_id", "diagnostic_labels", "point_estimate_status",
        "inference_status", "support_status", "comparability_status", "explanation",
    ),
    "feature_covariance_comparisons.parquet": (
        "feature_name", "term_label", "model_id", "source_model_id", "reference_se_method",
        "comparison_se_method", "estimate_pp", "comparison_estimate_pp", "estimate_equal",
        "reference_standard_error_pp", "comparison_standard_error_pp", "standard_error_ratio",
        "reference_ci_width_pp", "comparison_ci_width_pp", "ci_width_ratio",
        "reference_ci_includes_zero", "comparison_ci_includes_zero", "zero_inclusion_changed",
        "prompt_clusters", "url_clusters", "finite_variance", "covariance_warning",
        "point_estimate_status", "inference_status", "fallback_status", "explanation",
    ),
    "feature_intent_interaction_contrasts.parquet": (
        "feature_name", "model_id", "source_model_id", "intent", "estimate_type", "estimate_pp",
        "ci_lower_pp", "ci_upper_pp", "p_value", "se_method", "n_rows", "n_prompts", "n_urls",
        "n_domains", "interaction_supported", "formal_contrast_available", "warning",
    ),
    "feature_model_comparability.parquet": (
        "feature_name", "term_label", "baseline_model_id", "comparison_model_id",
        "comparison_source_model_id", "same_feature_definition", "same_interpretation_unit",
        "same_outcome", "same_sample", "same_controls", "same_fixed_effects",
        "same_functional_form", "comparability_status", "comparability_warning",
    ),
    "feature_model_comparison_summary.parquet": (
        "feature_name", "feature_label", "available_model_aliases", "missing_model_aliases",
        "largest_estimate_transition", "largest_estimate_change_pp", "largest_estimate_term",
        "largest_uncertainty_transition", "largest_ci_width_change_pp", "largest_sample_loss_transition",
        "largest_rows_lost", "first_sign_flip_transition", "most_consequential_transition",
        "primary_stability_label", "main_uncertainty_issue", "main_confounding_concern",
        "n_comparisons", "n_directly_comparable", "n_partially_comparable", "n_not_comparable",
        "narrative", "interpretation_boundary",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_thresholds(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text("utf-8"))
    required = {
        "minimum_baseline_magnitude_pp", "stable_magnitude_absolute_threshold_pp",
        "attenuation_absolute_threshold_pp", "attenuation_relative_threshold",
        "amplification_absolute_threshold_pp", "amplification_relative_threshold",
        "sign_flip_minimum_magnitude_pp", "large_sample_change_percent",
        "large_ci_width_change_percent", "low_support_rows", "low_support_prompts",
        "low_support_urls", "low_support_domains", "preferred_covariance_order",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Model comparison threshold config is missing: {missing}")
    return config


def _number(value: Any) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])


def _finite(value: Any) -> bool:
    numeric = _number(value)
    return pd.notna(numeric) and np.isfinite(numeric)


def _zero_in_interval(low: Any, high: Any) -> bool | None:
    if not (_finite(low) and _finite(high)):
        return None
    return _number(low) <= 0 <= _number(high)


def _extract_controls(formula: str, feature: str) -> str:
    if "~" not in formula:
        return "none"
    controls = []
    for token in formula.split("~", 1)[1].split("+"):
        token = token.strip()
        if not token or feature in token or token in {"C(prompt_id)", "C(source_root_domain)"}:
            continue
        controls.append(token)
    return "; ".join(controls) or "none"


def _taxonomy_version(model_id: str, formula: str) -> str:
    if model_id == "G4A":
        return "rule_v2_metadata_url_title_meta"
    if model_id == "G4B":
        return "gemini_3_1_flash_lite_taxonomy_v1_content_informed"
    return "none"


def _extraction_scope(model_id: str, source_model: str) -> str:
    if model_id == "G5A":
        return "strong_content_only"
    if model_id == "G5B":
        return "full_text_equivalent_only"
    if model_id == "G5C":
        return "excerpt_only"
    if model_id == "G8":
        return source_model
    return "measurable_content"


def _contrast_key(row: pd.Series, feature_kind: str) -> str:
    if feature_kind == "categorical":
        return str(row.get("term_label", "unknown"))
    if feature_kind == "binary":
        return "present_vs_absent"
    return "continuous_effect"


def _preferred_flags(table: pd.DataFrame, covariance_order: list[str]) -> pd.Series:
    flags = pd.Series(False, index=table.index)
    keys = ["feature_name", "model_id", "source_model_id", "term_label"]
    for _, group in table.groupby(keys, dropna=False, sort=False):
        if str(group.iloc[0]["model_id"]) == "G0":
            flags.loc[group.index[0]] = True
            continue
        valid = group[group["model_status"].eq("available")].copy()
        if valid.empty:
            continue
        rank = {name: index for index, name in enumerate(covariance_order)}
        valid["_rank"] = valid["se_method"].map(rank).fillna(len(rank))
        flags.loc[valid.sort_values("_rank").index[0]] = True
    return flags


def _g0_rows(rates: pd.DataFrame, scorecard: pd.DataFrame, feature_specs: dict[str, Any], generated_at: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    score_labels = dict(scorecard[["feature_name", "human_label"]].itertuples(index=False, name=None))
    for feature, spec in feature_specs.items():
        part = rates[rates["feature_name"].eq(feature)].sort_values("bin_order").copy()
        if len(part) < 2:
            continue
        if spec.kind == "binary":
            reference = part[part["feature_level"].eq("absent")]
            comparisons = part[part["feature_level"].eq("present")]
            unit = spec.interpretation_unit
        elif spec.kind == "categorical":
            reference = part[part["feature_level"].astype(str).eq(str(spec.reference_group))]
            comparisons = part[~part["feature_level"].astype(str).eq(str(spec.reference_group))]
            unit = spec.interpretation_unit
        else:
            reference, comparisons = part.head(1), part.tail(1)
            unit = "highest observed quartile versus lowest observed quartile"
        if reference.empty:
            continue
        ref = reference.iloc[0]
        for _, comp in comparisons.iterrows():
            p0, p1 = float(ref["cited_rate"]), float(comp["cited_rate"])
            n0, n1 = int(ref["n_rows"]), int(comp["n_rows"])
            se = math.sqrt(p0 * (1 - p0) / max(1, n0) + p1 * (1 - p1) / max(1, n1))
            estimate = (p1 - p0) * 100
            term_label = str(comp["feature_level"]) if spec.kind == "categorical" else unit
            contrast = str(comp["feature_level"]) if spec.kind == "categorical" else "present_vs_absent" if spec.kind == "binary" else "continuous_summary"
            output.append({
                "feature_name": feature, "feature_label": score_labels.get(feature, feature), "model_id": "G0",
                "source_model_id": "G0_descriptive", "model_role": "descriptive", "term": contrast,
                "term_label": term_label, "contrast_key": contrast, "estimate": estimate / 100,
                "original_estimate_pp": estimate, "estimate_pp": estimate, "standard_error": se,
                "standard_error_pp": se * 100, "ci_lower_pp": estimate - 1.959963984540054 * se * 100,
                "ci_upper_pp": estimate + 1.959963984540054 * se * 100,
                "ci_width_pp": 2 * 1.959963984540054 * se * 100, "p_value": math.nan,
                "adjusted_p_value": math.nan, "interpretation_unit": unit,
                "original_interpretation_unit": unit, "unit_multiplier": 1.0,
                "reference_group": str(ref["feature_level"]), "n_rows": n0 + n1,
                "n_cited": int(ref["n_cited"] + comp["n_cited"]),
                "n_prompts": max(int(ref["n_prompts"]), int(comp["n_prompts"])),
                "n_urls": max(int(ref["n_urls"]), int(comp["n_urls"])),
                "n_domains": max(int(ref["n_domains"]), int(comp["n_domains"])),
                "prompt_clusters": math.nan, "url_clusters": math.nan, "se_method": "independent_rate_difference",
                "fixed_effects": "none", "controls": "none", "sample_restriction": "measurable feature rows",
                "functional_form": "descriptive_rate_difference", "focal_feature_definition": feature,
                "taxonomy_version": "none", "extraction_scope": "measurable_content",
                "dataset_version": "area_condo_content_lpm_measurable_rows_v1", "model_version": "G0_v1",
                "model_status": "available", "model_warning": "Unadjusted descriptive contrast; repeated rows are not modeled.",
                "formula": "cited rate difference", "source_artifact": "feature_cited_rate_summary.csv",
                "is_preferred_covariance": True,
            })
    return output


def harmonize_model_estimates(
    estimates: pd.DataFrame,
    rates: pd.DataFrame,
    rows: pd.DataFrame,
    scorecard: pd.DataFrame,
    feature_specs: dict[str, Any],
    thresholds: dict[str, Any],
    generated_at: str,
) -> pd.DataFrame:
    labels = dict(scorecard[["feature_name", "human_label"]].itertuples(index=False, name=None))
    output = []
    for _, source in estimates.iterrows():
        feature = str(source["feature_name"])
        spec = feature_specs[feature]
        multiplier = 1.0
        interpretation = str(source.get("interpretation_unit", spec.interpretation_unit))
        if spec.kind == "continuous" and feature != "log2_word_count_plus1":
            numeric = pd.to_numeric(rows[feature], errors="coerce")
            multiplier = float(numeric.quantile(.75) - numeric.quantile(.25))
            if not np.isfinite(multiplier) or multiplier <= 0:
                multiplier = 1.0
            interpretation = "p25-to-p75 increase"
        original_pp = _number(source.get("estimate_pp"))
        estimate_pp = original_pp * multiplier
        se = _number(source.get("standard_error"))
        ci_low = _number(source.get("ci_lower_pp")) * multiplier
        ci_high = _number(source.get("ci_upper_pp")) * multiplier
        model_id = str(source["model_id"])
        source_model = str(source["source_model_id"])
        formula = str(source.get("formula", ""))
        output.append({
            "feature_name": feature, "feature_label": labels.get(feature, feature), "model_id": model_id,
            "source_model_id": source_model, "model_role": source.get("model_role"), "term": source.get("term"),
            "term_label": interpretation if spec.kind == "continuous" else source.get("term_label"),
            "contrast_key": _contrast_key(source, spec.kind),
            "estimate": source.get("estimate"), "original_estimate_pp": original_pp, "estimate_pp": estimate_pp,
            "standard_error": se, "standard_error_pp": se * 100 * multiplier,
            "ci_lower_pp": ci_low, "ci_upper_pp": ci_high,
            "ci_width_pp": ci_high - ci_low if _finite(ci_low) and _finite(ci_high) else math.nan,
            "p_value": source.get("p_value"), "adjusted_p_value": source.get("adjusted_p_value"),
            "interpretation_unit": interpretation, "original_interpretation_unit": source.get("interpretation_unit"),
            "unit_multiplier": multiplier, "reference_group": source.get("reference_group"),
            "n_rows": source.get("n_rows"), "n_cited": source.get("n_cited"), "n_prompts": source.get("n_prompts"),
            "n_urls": source.get("n_urls"), "n_domains": source.get("n_domains"),
            "prompt_clusters": source.get("prompt_clusters"), "url_clusters": source.get("url_clusters"),
            "se_method": source.get("se_method"), "fixed_effects": source.get("fixed_effects"),
            "controls": _extract_controls(formula, feature), "sample_restriction": source.get("sample_restriction"),
            "functional_form": "logit_AME" if model_id == "G7" else "LPM",
            "focal_feature_definition": f"{feature}::winsorized" if "winsorized" in str(source.get("term", "")).casefold() else feature,
            "taxonomy_version": _taxonomy_version(model_id, formula),
            "extraction_scope": _extraction_scope(model_id, source_model),
            "dataset_version": "area_condo_content_lpm_measurable_rows_v1", "model_version": source.get("model_version"),
            "model_status": source.get("model_status"), "model_warning": source.get("model_warning"),
            "formula": formula, "source_artifact": source.get("source_artifact"), "is_preferred_covariance": False,
        })
    output.extend(_g0_rows(rates, scorecard, feature_specs, generated_at))
    result = pd.DataFrame(output, columns=COMPARISON_SCHEMAS["feature_model_estimates_harmonized.parquet"])
    result["is_preferred_covariance"] = _preferred_flags(result, list(thresholds["preferred_covariance_order"]))
    return result


def _comparability(base: pd.Series, comp: pd.Series) -> tuple[str, str, dict[str, bool]]:
    flags = {
        "same_feature_definition": str(base["focal_feature_definition"]) == str(comp["focal_feature_definition"]),
        "same_interpretation_unit": str(base["interpretation_unit"]) == str(comp["interpretation_unit"]),
        "same_outcome": True,
        "same_sample": _number(base["n_rows"]) == _number(comp["n_rows"]),
        "same_controls": str(base["controls"]) == str(comp["controls"]),
        "same_fixed_effects": str(base["fixed_effects"]) == str(comp["fixed_effects"]),
        "same_functional_form": str(base["functional_form"]) == str(comp["functional_form"]),
    }
    if not flags["same_feature_definition"] or not flags["same_interpretation_unit"]:
        return "not_directly_comparable", "Feature definition or interpretation unit differs.", flags
    if str(base["model_id"]) == "G0" and str(base["contrast_key"]) == "continuous_summary":
        return "not_directly_comparable", "The descriptive top-versus-bottom contrast is not the same unit as the regression coefficient.", flags
    if str(comp["model_id"]) == "G7":
        return "not_directly_comparable", "G7 is a logit AME with intent/area controls rather than the G2 prompt fixed effects.", flags
    if all(flags.values()):
        return "directly_comparable", "Same feature unit, sample, controls, fixed effects, and functional form.", flags
    return "partially_comparable", "Feature units are compatible, but the specification or sample changes.", flags


def _labels(base: pd.Series, comp: pd.Series, comparison_type: str, thresholds: dict[str, Any]) -> tuple[list[str], str, str]:
    b, c = _number(base["estimate_pp"]), _number(comp["estimate_pp"])
    change_mag = abs(c) - abs(b)
    relative = change_mag / abs(b) if abs(b) >= thresholds["minimum_baseline_magnitude_pp"] else math.nan
    labels: list[str] = []
    direction_changed = np.sign(b) != np.sign(c)
    meaningful_flip = direction_changed and min(abs(b), abs(c)) >= thresholds["sign_flip_minimum_magnitude_pp"]
    if meaningful_flip:
        labels.append("sign_flip")
        point = "sign_flip"
    elif direction_changed:
        labels.append("direction_change_below_threshold")
        if abs(change_mag) <= thresholds["stable_magnitude_absolute_threshold_pp"]:
            labels.append("stable_magnitude")
            point = "stable_point_estimate"
        elif change_mag <= -thresholds["attenuation_absolute_threshold_pp"] and _finite(relative) and relative <= -thresholds["attenuation_relative_threshold"]:
            labels.append("substantial_attenuation")
            point = "attenuated_point_estimate"
        elif change_mag >= thresholds["amplification_absolute_threshold_pp"] and _finite(relative) and relative >= thresholds["amplification_relative_threshold"]:
            labels.append("substantial_amplification")
            point = "amplified_point_estimate"
        else:
            point = "moderate_point_estimate_change"
    elif abs(change_mag) <= thresholds["stable_magnitude_absolute_threshold_pp"]:
        labels.extend(["stable_direction", "stable_magnitude"])
        point = "stable_point_estimate"
    elif change_mag <= -thresholds["attenuation_absolute_threshold_pp"] and _finite(relative) and relative <= -thresholds["attenuation_relative_threshold"]:
        labels.extend(["stable_direction", "substantial_attenuation"])
        point = "attenuated_point_estimate"
    elif change_mag >= thresholds["amplification_absolute_threshold_pp"] and _finite(relative) and relative >= thresholds["amplification_relative_threshold"]:
        labels.extend(["stable_direction", "substantial_amplification"])
        point = "amplified_point_estimate"
    else:
        labels.append("stable_direction")
        point = "moderate_point_estimate_change"

    base_width, comp_width = _number(base["ci_width_pp"]), _number(comp["ci_width_pp"])
    width_ratio = (comp_width - base_width) / base_width if _finite(base_width) and base_width > 0 and _finite(comp_width) else math.nan
    zero_changed = _zero_in_interval(base["ci_lower_pp"], base["ci_upper_pp"]) != _zero_in_interval(comp["ci_lower_pp"], comp["ci_upper_pp"])
    if _finite(width_ratio) and width_ratio >= thresholds["large_ci_width_change_percent"]:
        labels.append("wider_uncertainty")
    elif _finite(width_ratio) and width_ratio <= -thresholds["large_ci_width_change_percent"]:
        labels.append("narrower_uncertainty")
    inference = "inference_sensitive" if zero_changed or "wider_uncertainty" in labels else "inference_stable"

    rows_pct = (_number(comp["n_rows"]) - _number(base["n_rows"])) / _number(base["n_rows"]) if _number(base["n_rows"]) else math.nan
    if _finite(rows_pct) and abs(rows_pct) >= thresholds["large_sample_change_percent"]:
        labels.append("sample_sensitive")
    transition_label = {
        "raw_to_prompt_fe": "prompt_composition_sensitive",
        "relevance_sensitivity": "relevance_sensitive",
        "domain_fe_sensitivity": "domain_template_confounded",
        "metadata_taxonomy_sensitivity": "metadata_taxonomy_sensitive",
        "gemini_taxonomy_sensitivity": "gemini_taxonomy_sensitive",
        "strong_content_sensitivity": "extraction_sensitive",
        "text_scope_sensitivity": "text_scope_sensitive",
        "functional_form_sensitivity": "functional_form_sensitive",
        "outlier_sensitivity": "outlier_sensitive",
    }.get(comparison_type)
    if transition_label and (meaningful_flip or abs(change_mag) >= thresholds["attenuation_absolute_threshold_pp"]):
        labels.append(transition_label)
    if comparison_type == "gemini_taxonomy_sensitivity" and "substantial_attenuation" in labels:
        labels.append("possible_taxonomy_overcontrol")
    return list(dict.fromkeys(labels)), point, inference


def _explanation(base: pd.Series, comp: pd.Series, comparison_type: str, labels: list[str]) -> str:
    change = _number(comp["estimate_pp"]) - _number(base["estimate_pp"])
    rows = int(_number(comp["n_rows"]) - _number(base["n_rows"]))
    opening = f"The estimate changed from {_number(base['estimate_pp']):+.1f} pp to {_number(comp['estimate_pp']):+.1f} pp ({change:+.1f} pp)."
    sample = f" The comparison sample changed by {rows:+,} rows."
    context = {
        "raw_to_prompt_fe": " Prompt fixed effects were introduced, so the change is evidence of prompt-composition sensitivity, not removal of all confounding.",
        "one_feature_to_joint": " Joint content controls were introduced; change may reflect observed confounding, predictor overlap, suppression, or multicollinearity, not causal mediation.",
        "relevance_sensitivity": " Measured prompt-page relevance is incomplete and does not solve surfaced-source selection.",
        "domain_fe_sensitivity": " Domain fixed effects and the narrower domain-supported sample were introduced; the two changes cannot be separated here.",
        "metadata_taxonomy_sensitivity": " Metadata-only page-function controls were introduced.",
        "gemini_taxonomy_sensitivity": " Content-informed taxonomy controls were introduced; attenuation can reflect improved adjustment or possible over-control.",
        "taxonomy_method_comparison": " Metadata-only and content-informed taxonomy controls differ; the content-informed version may overlap with scraped content structure.",
        "strong_content_sensitivity": " Strong-content restriction changes measurement reliability and sample composition.",
        "text_scope_sensitivity": " Full-text-equivalent and excerpt-only measurements use different selected samples.",
        "functional_form_sensitivity": " LPM and logit AME functional forms and control sets differ, so this is a cross-check rather than a replacement.",
        "outlier_sensitivity": " A predefined tail treatment was applied; all available treatments remain displayed.",
    }.get(comparison_type, "")
    return opening + sample + context


def build_model_comparisons(estimates: pd.DataFrame, thresholds: dict[str, Any], generated_at: str) -> pd.DataFrame:
    preferred = estimates[estimates["is_preferred_covariance"] & estimates["model_status"].eq("available")].copy()
    output: list[dict[str, Any]] = []
    for baseline_id, comparison_id, comparison_type in PAIR_SPECS:
        bases = preferred[preferred["model_id"].eq(baseline_id)]
        comps = preferred[preferred["model_id"].eq(comparison_id)]
        if bases.empty or comps.empty:
            continue
        for _, base in bases.iterrows():
            candidates = comps[comps["feature_name"].eq(base["feature_name"])]
            exact = candidates[candidates["contrast_key"].eq(base["contrast_key"])]
            if not exact.empty:
                candidates = exact
            elif str(base["model_id"]) != "G0" or str(base["contrast_key"]) != "continuous_summary":
                continue
            for _, comp in candidates.iterrows():
                status, warning, flags = _comparability(base, comp)
                labels, point_status, inference_status = _labels(base, comp, comparison_type, thresholds)
                if status == "not_directly_comparable":
                    labels.append("not_directly_comparable")
                b, c = _number(base["estimate_pp"]), _number(comp["estimate_pp"])
                change = c - b
                magnitude_change = abs(c) - abs(b)
                stable_relative = abs(b) >= thresholds["minimum_baseline_magnitude_pp"] and flags["same_interpretation_unit"]
                relative = magnitude_change / abs(b) if stable_relative else math.nan
                rows_change = _number(comp["n_rows"]) - _number(base["n_rows"])
                rows_pct = rows_change / _number(base["n_rows"]) if _number(base["n_rows"]) else math.nan
                output.append({
                    "feature_name": base["feature_name"], "feature_label": base["feature_label"],
                    "term_label": comp["term_label"], "contrast_key": comp["contrast_key"],
                    "baseline_model_id": baseline_id, "comparison_model_id": comparison_id,
                    "baseline_source_model_id": base["source_model_id"], "comparison_source_model_id": comp["source_model_id"],
                    "comparison_type": comparison_type, "baseline_estimate_pp": b, "comparison_estimate_pp": c,
                    "estimate_change_pp": change, "absolute_magnitude_change_pp": magnitude_change,
                    "relative_magnitude_change": relative, "relative_change_status": "available" if _finite(relative) else "relative_change_not_stable",
                    "baseline_ci_lower_pp": base["ci_lower_pp"], "baseline_ci_upper_pp": base["ci_upper_pp"],
                    "comparison_ci_lower_pp": comp["ci_lower_pp"], "comparison_ci_upper_pp": comp["ci_upper_pp"],
                    "baseline_ci_width_pp": base["ci_width_pp"], "comparison_ci_width_pp": comp["ci_width_pp"],
                    "ci_width_change_pp": _number(comp["ci_width_pp"]) - _number(base["ci_width_pp"]),
                    "standard_error_change": _number(comp["standard_error_pp"]) - _number(base["standard_error_pp"]),
                    "sign_changed": "sign_flip" in labels, "baseline_ci_includes_zero": _zero_in_interval(base["ci_lower_pp"], base["ci_upper_pp"]),
                    "comparison_ci_includes_zero": _zero_in_interval(comp["ci_lower_pp"], comp["ci_upper_pp"]),
                    "ci_zero_status_changed": _zero_in_interval(base["ci_lower_pp"], base["ci_upper_pp"]) != _zero_in_interval(comp["ci_lower_pp"], comp["ci_upper_pp"]),
                    "baseline_n_rows": base["n_rows"], "comparison_n_rows": comp["n_rows"], "rows_change": rows_change,
                    "rows_change_percent": rows_pct, "baseline_n_prompts": base["n_prompts"], "comparison_n_prompts": comp["n_prompts"],
                    "prompts_changed": _number(comp["n_prompts"]) - _number(base["n_prompts"]),
                    "baseline_n_urls": base["n_urls"], "comparison_n_urls": comp["n_urls"],
                    "urls_changed": _number(comp["n_urls"]) - _number(base["n_urls"]),
                    "baseline_n_domains": base["n_domains"], "comparison_n_domains": comp["n_domains"],
                    "domains_changed": _number(comp["n_domains"]) - _number(base["n_domains"]),
                    "prompt_clusters_changed": _number(comp["prompt_clusters"]) - _number(base["prompt_clusters"]) if _finite(comp["prompt_clusters"]) and _finite(base["prompt_clusters"]) else math.nan,
                    "url_clusters_changed": _number(comp["url_clusters"]) - _number(base["url_clusters"]) if _finite(comp["url_clusters"]) and _finite(base["url_clusters"]) else math.nan,
                    "baseline_se_method": base["se_method"], "comparison_se_method": comp["se_method"],
                    "same_sample": flags["same_sample"], "same_controls": flags["same_controls"],
                    "same_fixed_effects": flags["same_fixed_effects"], "same_functional_form": flags["same_functional_form"],
                    "same_interpretation_unit": flags["same_interpretation_unit"], "directly_comparable": status == "directly_comparable",
                    "comparability_status": status, "comparability_warning": warning,
                    "diagnostic_labels": ";".join(dict.fromkeys(labels)),
                    "explanation": _explanation(base, comp, comparison_type, labels),
                    "explanation_template_id": comparison_type, "dataset_version": base["dataset_version"],
                    "model_version": f"{base['model_version']}__{comp['model_version']}", "artifact_generated_at": generated_at,
                    "_point_status": point_status, "_inference_status": inference_status,
                    "_same_feature_definition": flags["same_feature_definition"], "_same_outcome": flags["same_outcome"],
                })
    columns = list(COMPARISON_SCHEMAS["feature_model_comparisons.parquet"])
    return pd.DataFrame(output).reindex(columns=columns + ["_point_status", "_inference_status", "_same_feature_definition", "_same_outcome"])


def build_covariance_comparisons(estimates: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    output = []
    group_cols = ["feature_name", "model_id", "source_model_id", "term_label"]
    for _, group in estimates[estimates["model_id"].ne("G0")].groupby(group_cols, dropna=False):
        if group["se_method"].nunique() < 2:
            continue
        hc3 = group[group["se_method"].eq("HC3")]
        valid_hc3 = hc3[
            hc3["model_status"].eq("available")
            & np.isfinite(pd.to_numeric(hc3["standard_error_pp"], errors="coerce"))
            & np.isfinite(pd.to_numeric(hc3["ci_lower_pp"], errors="coerce"))
            & np.isfinite(pd.to_numeric(hc3["ci_upper_pp"], errors="coerce"))
        ]
        preferred = group[group["is_preferred_covariance"] & group["model_status"].eq("available")]
        if not valid_hc3.empty:
            reference = valid_hc3.iloc[0]
        elif not preferred.empty:
            reference = preferred.iloc[0]
        else:
            continue
        for _, comp in group.iterrows():
            if comp.name == reference.name:
                continue
            ref_width, comp_width = _number(reference["ci_width_pp"]), _number(comp["ci_width_pp"])
            ref_se, comp_se = _number(reference["standard_error_pp"]), _number(comp["standard_error_pp"])
            finite = all(_finite(value) for value in (comp_se, comp["ci_lower_pp"], comp["ci_upper_pp"])) and comp_se > 0
            equal = abs(_number(reference["estimate_pp"]) - _number(comp["estimate_pp"])) <= thresholds["estimate_equality_tolerance_pp"]
            zero_changed = _zero_in_interval(reference["ci_lower_pp"], reference["ci_upper_pp"]) != _zero_in_interval(comp["ci_lower_pp"], comp["ci_upper_pp"])
            ratio = comp_se / ref_se if _finite(ref_se) and ref_se > 0 and _finite(comp_se) else math.nan
            inference = "inference_sensitive" if zero_changed or (_finite(ratio) and ratio >= thresholds["covariance_standard_error_ratio_threshold"]) else "inference_stable"
            warning = str(comp.get("model_warning", "") or "")
            if not finite:
                warning = (warning + "; covariance unavailable because focal variance or interval is non-finite").strip("; ")
            output.append({
                "feature_name": comp["feature_name"], "term_label": comp["term_label"], "model_id": comp["model_id"],
                "source_model_id": comp["source_model_id"], "reference_se_method": reference["se_method"],
                "comparison_se_method": comp["se_method"], "estimate_pp": reference["estimate_pp"],
                "comparison_estimate_pp": comp["estimate_pp"], "estimate_equal": equal,
                "reference_standard_error_pp": ref_se, "comparison_standard_error_pp": comp_se,
                "standard_error_ratio": ratio, "reference_ci_width_pp": ref_width,
                "comparison_ci_width_pp": comp_width, "ci_width_ratio": comp_width / ref_width if _finite(ref_width) and ref_width > 0 and _finite(comp_width) else math.nan,
                "reference_ci_includes_zero": _zero_in_interval(reference["ci_lower_pp"], reference["ci_upper_pp"]),
                "comparison_ci_includes_zero": _zero_in_interval(comp["ci_lower_pp"], comp["ci_upper_pp"]),
                "zero_inclusion_changed": zero_changed, "prompt_clusters": comp["prompt_clusters"], "url_clusters": comp["url_clusters"],
                "finite_variance": finite, "covariance_warning": warning, "point_estimate_status": "stable_point_estimate" if equal else "unexpected_estimate_change",
                "inference_status": inference, "fallback_status": "preferred" if bool(comp["is_preferred_covariance"]) else "available_not_preferred" if finite else "unavailable",
                "explanation": COVARIANCE_ONLY_MESSAGE,
            })
    return pd.DataFrame(output, columns=COMPARISON_SCHEMAS["feature_covariance_comparisons.parquet"])


def build_intent_interactions(package: Path) -> pd.DataFrame:
    result_path = package / "tables/09_content_feature_econometrics/M9_intent_interaction_sensitivity_results.csv"
    support_path = package / "tables/09_content_feature_econometrics/M9_intent_interaction_cell_support.csv"
    if not result_path.exists():
        return pd.DataFrame(columns=COMPARISON_SCHEMAS["feature_intent_interaction_contrasts.parquet"])
    results = pd.read_csv(result_path, low_memory=False)
    support = pd.read_csv(support_path, low_memory=False) if support_path.exists() else pd.DataFrame()
    output = []
    patterns = {
        "log2_word_count_plus1": re.compile(r"C\(intent\)\[(.+?)\]:log2_word_count_plus1$"),
        "has_table": re.compile(r"C\(intent\)\[(.+?)\]:has_table$"),
    }
    for _, row in results.iterrows():
        for feature, pattern in patterns.items():
            match = pattern.search(str(row.get("term", "")))
            if not match:
                continue
            intent = match.group(1)
            support_feature = "word_count_median_group" if feature == "log2_word_count_plus1" else "has_table"
            cells = support[(support.get("interaction_feature") == support_feature) & (support.get("intent") == intent)] if not support.empty else pd.DataFrame()
            supported = bool(not cells.empty and cells["interaction_supported"].fillna(False).all())
            output.append({
                "feature_name": feature, "model_id": "G9", "source_model_id": row.get("model_id"), "intent": intent,
                "estimate_type": "subgroup_specific_slope", "estimate_pp": row.get("estimate_pp"),
                "ci_lower_pp": row.get("conf_low_pp"), "ci_upper_pp": row.get("conf_high_pp"), "p_value": row.get("p_value"),
                "se_method": row.get("cov_type"), "n_rows": row.get("n_obs"), "n_prompts": row.get("n_prompts"),
                "n_urls": row.get("n_urls"), "n_domains": row.get("n_domains"), "interaction_supported": supported,
                "formal_contrast_available": False,
                "warning": "Subgroup-specific slope from a formal interaction model; pairwise intent-difference covariance was not exported, so formal between-intent contrasts are unavailable.",
            })
    return pd.DataFrame(output, columns=COMPARISON_SCHEMAS["feature_intent_interaction_contrasts.parquet"])


def _transition_tables(comparisons: pd.DataFrame, thresholds: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = []
    comparability = []
    for _, row in comparisons.iterrows():
        support = "insufficient_support" if any(_number(row[name]) < thresholds[limit] for name, limit in (
            ("comparison_n_rows", "low_support_rows"), ("comparison_n_prompts", "low_support_prompts"),
            ("comparison_n_urls", "low_support_urls"), ("comparison_n_domains", "low_support_domains"),
        )) else "supported"
        diagnostic = str(row["diagnostic_labels"])
        point = row.get("_point_status", "not_assessable")
        inference = row.get("_inference_status", "not_assessable")
        labels.append({
            "feature_name": row["feature_name"], "term_label": row["term_label"], "baseline_model_id": row["baseline_model_id"],
            "comparison_model_id": row["comparison_model_id"], "comparison_source_model_id": row["comparison_source_model_id"],
            "diagnostic_labels": diagnostic, "point_estimate_status": point, "inference_status": inference,
            "support_status": support, "comparability_status": row["comparability_status"], "explanation": row["explanation"],
        })
        comparability.append({
            "feature_name": row["feature_name"], "term_label": row["term_label"], "baseline_model_id": row["baseline_model_id"],
            "comparison_model_id": row["comparison_model_id"], "comparison_source_model_id": row["comparison_source_model_id"],
            "same_feature_definition": row.get("_same_feature_definition", True), "same_interpretation_unit": row["same_interpretation_unit"],
            "same_outcome": row.get("_same_outcome", True), "same_sample": row["same_sample"], "same_controls": row["same_controls"],
            "same_fixed_effects": row["same_fixed_effects"], "same_functional_form": row["same_functional_form"],
            "comparability_status": row["comparability_status"], "comparability_warning": row["comparability_warning"],
        })
    return (
        pd.DataFrame(labels, columns=COMPARISON_SCHEMAS["feature_model_transition_labels.parquet"]),
        pd.DataFrame(comparability, columns=COMPARISON_SCHEMAS["feature_model_comparability.parquet"]),
    )


def _summary(estimates: pd.DataFrame, comparisons: pd.DataFrame, feature_specs: dict[str, Any]) -> pd.DataFrame:
    output = []
    for feature, spec in feature_specs.items():
        available = estimates[(estimates["feature_name"].eq(feature)) & estimates["model_status"].eq("available")]["model_id"].drop_duplicates().tolist()
        part = comparisons[comparisons["feature_name"].eq(feature)].copy()
        comparable = part[part["comparability_status"].ne("not_directly_comparable")]
        largest = comparable.loc[comparable["estimate_change_pp"].abs().idxmax()] if not comparable.empty else None
        uncertainty = part.loc[part["ci_width_change_pp"].idxmax()] if not part.empty else None
        loss = part.loc[part["rows_change"].idxmin()] if not part.empty else None
        flips = part[part["sign_changed"]]
        first_flip = flips.iloc[0] if not flips.empty else None
        consequential = largest if largest is not None else uncertainty
        preferred = estimates[
            estimates["feature_name"].eq(feature)
            & estimates["is_preferred_covariance"]
            & estimates["model_status"].eq("available")
        ]
        g1_terms = preferred[preferred["model_id"].eq("G1")]["term_label"].dropna().astype(str).drop_duplicates().tolist()
        g2_terms = set(preferred[preferred["model_id"].eq("G2")]["term_label"].dropna().astype(str))
        aligned_term = next((term for term in g1_terms if term in g2_terms), g1_terms[0] if g1_terms else "not_available")

        def aligned_text(model_id: str) -> str:
            rows = preferred[
                preferred["model_id"].eq(model_id)
                & preferred["term_label"].astype(str).eq(aligned_term)
            ]
            return "unavailable" if rows.empty else f"{_number(rows.iloc[0]['estimate_pp']):+.1f} pp"

        raw_text, g1_text, g2_text = (aligned_text(model_id) for model_id in ("G0", "G1", "G2"))
        if consequential is not None:
            transition = f"{consequential['baseline_model_id']} to {consequential['comparison_model_id']}"
            largest_sentence = (
                f"The largest compatible displayed change was {transition} for `{consequential['term_label']}` "
                f"({_number(consequential['estimate_change_pp']):+.1f} pp)."
            )
            classification = str(consequential["diagnostic_labels"]).split(";")[-1]
        else:
            transition, largest_sentence, classification = "not_available", "No predefined comparable transition is available.", "insufficient_support"
        narrative = (
            f"For the representative aligned regression contrast `{aligned_term}`, the matching G0 raw contrast was {raw_text}, "
            f"the G1 within-prompt estimate was {g1_text}, and the G2 joint estimate was {g2_text}. "
            f"{largest_sentence} This pattern is a robustness diagnostic among surfaced sources, not evidence that the specification change identifies a causal pathway."
        )
        output.append({
            "feature_name": feature, "feature_label": spec.label, "available_model_aliases": ";".join(model for model in MODEL_ORDER if model in available),
            "missing_model_aliases": ";".join(model for model in MODEL_ORDER if model not in available),
            "largest_estimate_transition": transition, "largest_estimate_change_pp": largest["estimate_change_pp"] if largest is not None else math.nan,
            "largest_estimate_term": largest["term_label"] if largest is not None else "not_available",
            "largest_uncertainty_transition": f"{uncertainty['baseline_model_id']} to {uncertainty['comparison_model_id']}" if uncertainty is not None else "not_available",
            "largest_ci_width_change_pp": uncertainty["ci_width_change_pp"] if uncertainty is not None else math.nan,
            "largest_sample_loss_transition": f"{loss['baseline_model_id']} to {loss['comparison_model_id']}" if loss is not None else "not_available",
            "largest_rows_lost": loss["rows_change"] if loss is not None else math.nan,
            "first_sign_flip_transition": f"{first_flip['baseline_model_id']} to {first_flip['comparison_model_id']}" if first_flip is not None else "none",
            "most_consequential_transition": transition, "primary_stability_label": classification,
            "main_uncertainty_issue": "covariance and interval sensitivity must be read separately from the point estimate",
            "main_confounding_concern": "prompt composition, correlated content features, domain/template differences, and hidden retrieval factors remain possible",
            "n_comparisons": len(part), "n_directly_comparable": int(part["comparability_status"].eq("directly_comparable").sum()),
            "n_partially_comparable": int(part["comparability_status"].eq("partially_comparable").sum()),
            "n_not_comparable": int(part["comparability_status"].eq("not_directly_comparable").sum()),
            "narrative": narrative,
            "interpretation_boundary": "Conditional association among surfaced sources; not causal and not a web-wide citation probability.",
        })
    return pd.DataFrame(output, columns=COMPARISON_SCHEMAS["feature_model_comparison_summary.parquet"])


def _model_metadata(estimates: pd.DataFrame, package: Path) -> pd.DataFrame:
    rows = []
    for model_id in MODEL_ORDER:
        part = estimates[estimates["model_id"].eq(model_id)]
        role, purpose, fixed, controls, functional = MODEL_REGISTRY[model_id]
        status = "available" if not part.empty else "descriptive_audit_only" if model_id == "G6" and (package / "tables/09_content_feature_econometrics/M6_measurable_selection_audit.csv").exists() else "subgroup_slopes_available_no_pairwise_contrasts" if model_id == "G9" and (package / "tables/09_content_feature_econometrics/M9_intent_interaction_sensitivity_results.csv").exists() else "not_available"
        warning = ""
        if model_id == "G2R":
            warning = "No otherwise-identical G2-plus-relevance artifact exists; Notebook 09 M8 was skipped."
        elif model_id == "G7":
            warning = "Uses intent/area controls rather than full prompt fixed effects."
        elif model_id == "G9":
            warning = "Only supported subgroup slopes are available; pairwise intent contrasts were not exported."
        rows.append({
            "model_id": model_id, "model_role": role, "purpose": purpose, "fixed_effects": fixed,
            "controls": controls, "functional_form": functional, "artifact_status": status,
            "source_model_ids": ";".join(sorted(part["source_model_id"].dropna().astype(str).unique())),
            "source_artifacts": ";".join(sorted(part["source_artifact"].dropna().astype(str).unique())), "warning": warning,
        })
    return pd.DataFrame(rows, columns=COMPARISON_SCHEMAS["model_metadata_registry.parquet"])


def build_model_comparison_artifacts(
    package: Path,
    output: Path,
    estimates: pd.DataFrame,
    rates: pd.DataFrame,
    rows: pd.DataFrame,
    scorecard: pd.DataFrame,
    feature_specs: dict[str, Any],
    threshold_path: Path,
    generated_at: str | None = None,
) -> dict[str, pd.DataFrame]:
    generated_at = generated_at or datetime.now(UTC).isoformat()
    thresholds = load_thresholds(threshold_path)
    harmonized = harmonize_model_estimates(estimates, rates, rows, scorecard, feature_specs, thresholds, generated_at)
    comparisons_internal = build_model_comparisons(harmonized, thresholds, generated_at)
    transition_labels, comparability = _transition_tables(comparisons_internal, thresholds)
    comparisons = comparisons_internal[list(COMPARISON_SCHEMAS["feature_model_comparisons.parquet"])]
    tables = {
        "model_metadata_registry.parquet": _model_metadata(harmonized, package),
        "feature_model_estimates_harmonized.parquet": harmonized,
        "feature_model_comparisons.parquet": comparisons,
        "feature_model_transition_labels.parquet": transition_labels,
        "feature_covariance_comparisons.parquet": build_covariance_comparisons(harmonized, thresholds),
        "feature_intent_interaction_contrasts.parquet": build_intent_interactions(package),
        "feature_model_comparability.parquet": comparability,
        "feature_model_comparison_summary.parquet": _summary(harmonized, comparisons, feature_specs),
    }
    for filename, table in tables.items():
        table.to_parquet(output / filename, index=False)
    shutil.copyfile(threshold_path, output / "model_comparison_thresholds.yaml")
    manifest = {
        "contract_version": COMPARISON_CONTRACT_VERSION,
        "generated_at": generated_at,
        "dataset_version": "area_condo_content_lpm_measurable_rows_v1",
        "model_aliases": list(MODEL_ORDER),
        "comparison_pairs": [f"{left}->{right}" for left, right, _ in PAIR_SPECS],
        "artifacts": {
            filename: {"sha256": _sha256(output / filename), "rows": len(table), "columns": list(table.columns)}
            for filename, table in tables.items()
        },
        "thresholds": {"path": "model_comparison_thresholds.yaml", "sha256": _sha256(output / "model_comparison_thresholds.yaml")},
        "guardrails": {
            "models_fit_in_streamlit": False,
            "frozen_notebook_outputs_overwritten": False,
            "labels_based_only_on_p_values": False,
            "causal_interpretation_allowed": False,
            "missing_models_fabricated": False,
        },
        "known_unavailable": {
            "G2R": "No otherwise-identical relevance-only sensitivity artifact.",
            "G6": "Descriptive missingness audit has no coefficient.",
            "G9_formal_pairwise_contrasts": "Subgroup slopes exist, but pairwise contrast covariance was not exported.",
        },
    }
    (output / "econometrics_model_comparison_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), "utf-8")
    return tables


def validate_model_comparison_artifacts(root: str | Path, verify_hashes: bool = True) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "econometrics_model_comparison_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model comparison manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("contract_version") != COMPARISON_CONTRACT_VERSION:
        raise ValueError(f"Unsupported model comparison contract: {manifest.get('contract_version')}")
    for filename, required in COMPARISON_SCHEMAS.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required model comparison artifact not found: {path}")
        table = pd.read_parquet(path)
        missing = sorted(set(required) - set(table.columns))
        if missing:
            raise ValueError(f"{filename} is missing columns: {missing}")
        entry = manifest.get("artifacts", {}).get(filename)
        if not entry or int(entry.get("rows", -1)) != len(table):
            raise ValueError(f"{filename} row count does not match the manifest")
        if verify_hashes and entry.get("sha256") != _sha256(path):
            raise ValueError(f"{filename} hash does not match the manifest")
    comparisons = pd.read_parquet(root / "feature_model_comparisons.parquet")
    threshold_path = root / "model_comparison_thresholds.yaml"
    if not threshold_path.exists() or (verify_hashes and manifest["thresholds"]["sha256"] != _sha256(threshold_path)):
        raise ValueError("Model comparison threshold artifact is missing or does not match its manifest hash")
    thresholds = load_thresholds(threshold_path)
    unstable = comparisons[comparisons["relative_change_status"].eq("relative_change_not_stable")]
    if unstable["relative_magnitude_change"].notna().any():
        raise ValueError("Near-zero or incompatible comparisons retain a relative magnitude change")
    tiny_flips = comparisons[
        comparisons["sign_changed"]
        & (comparisons[["baseline_estimate_pp", "comparison_estimate_pp"]].abs().min(axis=1) < thresholds["sign_flip_minimum_magnitude_pp"])
    ]
    if not tiny_flips.empty:
        raise ValueError("Negligible sign changes were classified as sign flips")
    direction_changed = np.sign(comparisons["baseline_estimate_pp"]) != np.sign(comparisons["comparison_estimate_pp"])
    contradictory_direction = comparisons[
        direction_changed
        & comparisons["diagnostic_labels"].fillna("").str.contains(r"(?:^|;)stable_direction(?:;|$)", regex=True)
    ]
    if not contradictory_direction.empty:
        raise ValueError("A direction-changing transition was classified as stable direction")
    if comparisons["diagnostic_labels"].fillna("").str.fullmatch(r"p[_-]?value.*", case=False).any():
        raise ValueError("A transition label is based solely on p-value wording")
    estimates = pd.read_parquet(root / "feature_model_estimates_harmonized.parquet")
    forbidden = ("answer_similarity", "answer_overlap", "source_position", "observed_rank", "domain_citation_rate")
    formulas = estimates["formula"].fillna("").str.casefold()
    if any(formulas.str.contains(token, regex=False).any() for token in forbidden):
        raise ValueError("Comparison estimates contain a forbidden leakage predictor")
    return manifest
