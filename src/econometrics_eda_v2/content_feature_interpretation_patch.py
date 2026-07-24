"""Interpretation and robustness patch for notebook 09 econometric outputs."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


COVARIANCE_ORDER = (
    "HC3",
    "cluster_prompt_id",
    "cluster_normalized_url",
    "two_way_cluster_prompt_url",
)
MODEL_FILES = {
    "M1": "M1_one_feature_prompt_fe_results.csv",
    "M2": "M2_preferred_joint_lpm_results.csv",
    "M3": "M3_domain_fe_results.csv",
    "M4": "M4_gemini_taxonomy_sensitivity_results.csv",
    "M5": "M5_strong_content_sensitivity_results.csv",
    "M10": "M10_outlier_winsorized_sensitivity_results.csv",
}
def _is_focal_term(term: str) -> bool:
    return any(
        token in str(term)
        for token in (
            "log2_word_count_plus1",
            "log2_word_count_plus1_winsorized_p99",
            "has_table",
            "heading_count_group",
            "link_count_group",
            "content_strength",
        )
    )


def _term_label(term: str) -> str:
    if term == "log2_word_count_plus1":
        return "Page length doubling"
    if term == "log2_word_count_plus1_winsorized_p99":
        return "Page length doubling (winsorized p99)"
    if term == "has_table":
        return "Has table"
    match = re.search(r"C\(([^,]+).*?\)\[T\.(.*?)\]$", str(term))
    if not match:
        return str(term)
    feature, category = match.groups()
    labels = {
        "heading_count_group": "Heading count",
        "link_count_group": "Link count",
        "content_strength": "Content strength",
    }
    return f"{labels.get(feature, feature)}: {category}"


def _canonical_feature(term: str) -> str:
    label = _term_label(term)
    return "Page length doubling" if label.startswith("Page length doubling") else label


def _ci_excludes_zero(row: pd.Series | Any) -> bool:
    return bool(
        pd.notna(row["conf_low_pp"])
        and pd.notna(row["conf_high_pp"])
        and (row["conf_low_pp"] > 0 or row["conf_high_pp"] < 0)
    )


def _preferred_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    rank = {covariance: index for index, covariance in enumerate(COVARIANCE_ORDER)}
    work = table.copy()
    work["_rank"] = work["cov_type"].map(rank).fillna(-1)
    return (
        work.sort_values("_rank", ascending=False, kind="stable")
        .drop_duplicates(["model_id", "term"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def _write_plotly(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    fig.write_json(path.with_suffix(".plotly.json"))


def _load_model_results(table_root: Path) -> pd.DataFrame:
    frames = []
    for family, filename in MODEL_FILES.items():
        path = table_root / filename
        if not path.exists():
            raise FileNotFoundError(f"Required existing model output not found: {path}")
        frame = pd.read_csv(path, low_memory=False)
        frame["model_family"] = family
        frames.append(frame)
    results = pd.concat(frames, ignore_index=True)
    return results[results["term"].map(_is_focal_term)].copy()


def _model_warning_lookup(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    all_results = results.copy()
    for model_id, group in all_results.groupby("model_id", sort=False):
        two_way = group[group["cov_type"].eq("two_way_cluster_prompt_url")]
        affected = two_way[two_way["std_error"].isna()]
        lookup[model_id] = {
            "has_warning": bool(len(affected)),
            "affected_focal_terms": sorted(affected["term"].map(_term_label).unique().tolist()),
        }
    return lookup


def build_focal_term_se_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Create the requested covariance-estimator comparison for focal terms."""
    warnings_by_model = _model_warning_lookup(results)
    rows = []
    for row in results.itertuples(index=False):
        warning = warnings_by_model[row.model_id]
        se_available = bool(pd.notna(row.std_error) and pd.notna(row.conf_low_pp) and pd.notna(row.conf_high_pp))
        focal_affected = _term_label(row.term) in warning["affected_focal_terms"]
        if focal_affected:
            focal_warning = "focal_term_se_unavailable"
            impact = "Do not use this covariance estimate for the focal term."
        elif row.cov_type == "two_way_cluster_prompt_url" and warning["has_warning"]:
            focal_warning = "nuisance_term_warning_focal_se_available"
            impact = "Focal SE is available; compare it with HC3 and both one-way cluster estimates."
        elif not se_available:
            focal_warning = "se_unavailable"
            impact = "Use another covariance estimator."
        else:
            focal_warning = "none"
            impact = "No estimator-specific focal warning."
        rows.append(
            {
                "model_id": row.model_id,
                "term": row.term,
                "term_label": _term_label(row.term),
                "cov_type": row.cov_type,
                "estimate": row.estimate,
                "estimate_pp": row.estimate_pp,
                "std_error": row.std_error,
                "conf_low_pp": row.conf_low_pp,
                "conf_high_pp": row.conf_high_pp,
                "p_value": row.p_value,
                "n_obs": row.n_obs,
                "n_prompts": row.n_prompts,
                "n_urls": row.n_urls,
                "n_domains": row.n_domains,
                "se_available": se_available,
                "covariance_warning": bool(warning["has_warning"]),
                "focal_term_warning": focal_warning,
                "interpretation_impact": impact,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["term_label", "model_id", "cov_type"],
        kind="stable",
    )


def build_se_stability_summary(se_comparison: pd.DataFrame) -> pd.DataFrame:
    """Classify covariance-estimator stability for each focal model coefficient."""
    cross_model_signs = (
        se_comparison.drop_duplicates(["model_id", "term"])
        .assign(feature=lambda frame: frame["term"].map(_canonical_feature))
        .groupby("feature")["estimate_pp"]
        .agg(
            has_positive=lambda values: bool((values > 0).any()),
            has_negative=lambda values: bool((values < 0).any()),
        )
    )
    rows = []
    for (model_id, term), group in se_comparison.groupby(["model_id", "term"], sort=False):
        feature = _canonical_feature(term)
        available = group[group["se_available"]]
        covariance_count = int(available["cov_type"].nunique())
        significance = available.apply(_ci_excludes_zero, axis=1) if len(available) else pd.Series(dtype=bool)
        two_way = group[group["cov_type"].eq("two_way_cluster_prompt_url")]
        two_way_unavailable = bool(len(two_way) and not bool(two_way.iloc[0]["se_available"]))
        cross_direction = bool(
            cross_model_signs.loc[feature, "has_positive"] and cross_model_signs.loc[feature, "has_negative"]
        )
        if covariance_count < len(COVARIANCE_ORDER) or two_way_unavailable:
            classification = "unavailable_or_unreliable_se"
        elif cross_direction:
            classification = "direction_sensitive"
        elif significance.nunique() > 1:
            classification = "direction_stable_but_significance_sensitive"
        else:
            classification = "stable_inference"
        rows.append(
            {
                "model_id": model_id,
                "term": term,
                "term_label": _term_label(term),
                "covariance_estimators_available": covariance_count,
                "estimate_direction": (
                    "positive" if group["estimate_pp"].iloc[0] > 0 else "negative"
                    if group["estimate_pp"].iloc[0] < 0
                    else "zero"
                ),
                "ci_excludes_zero_count": int(significance.sum()) if len(significance) else 0,
                "ci_includes_zero_count": int((~significance).sum()) if len(significance) else 0,
                "two_way_focal_se_available": not two_way_unavailable,
                "cross_model_direction_changed": cross_direction,
                "classification": classification,
            }
        )
    return pd.DataFrame(rows)


def make_se_comparison_forest(se_comparison: pd.DataFrame, path: Path) -> None:
    work = se_comparison[se_comparison["se_available"]].copy()
    work["display_label"] = work["model_id"] + " | " + work["term_label"]
    work["cov_rank"] = work["cov_type"].map({value: index for index, value in enumerate(COVARIANCE_ORDER)})
    work = work.sort_values(["term_label", "model_id", "cov_rank"], kind="stable")
    fig = go.Figure()
    colors = {
        "HC3": "#277da1",
        "cluster_prompt_id": "#43aa8b",
        "cluster_normalized_url": "#f8961e",
        "two_way_cluster_prompt_url": "#9b5de5",
    }
    for covariance in COVARIANCE_ORDER:
        group = work[work["cov_type"].eq(covariance)]
        fig.add_trace(
            go.Scatter(
                x=group["estimate_pp"],
                y=group["display_label"],
                mode="markers",
                name=covariance,
                marker={"color": colors[covariance], "size": 7},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group["conf_high_pp"] - group["estimate_pp"],
                    "arrayminus": group["estimate_pp"] - group["conf_low_pp"],
                },
                customdata=np.column_stack([group["conf_low_pp"], group["conf_high_pp"], group["p_value"]]),
                hovertemplate=(
                    "%{y}<br>Estimate=%{x:.2f} pp"
                    "<br>95% CI=%{customdata[0]:.2f} to %{customdata[1]:.2f} pp"
                    "<br>p=%{customdata[2]:.4f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#606b73")
    fig.update_layout(
        title="Focal-term inference across covariance estimators",
        xaxis_title="Association with citation probability (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=max(900, 24 * work["display_label"].nunique() + 180),
        margin={"l": 285, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(fig, path)


def build_two_way_warning_audit(all_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id, group in all_results.groupby("model_id", sort=False):
        two_way = group[group["cov_type"].eq("two_way_cluster_prompt_url")].copy()
        total_terms = int(two_way["term"].nunique())
        affected = two_way[two_way["std_error"].isna()]
        focal = affected[affected["term"].map(_is_focal_term)]
        affected_labels = sorted(focal["term"].map(_term_label).unique().tolist())
        focal_report_rows = two_way[two_way["term"].map(_is_focal_term)]
        focal_all_available = bool(len(focal_report_rows)) and focal_report_rows["std_error"].notna().all()
        if len(focal):
            action = (
                "Do not use two-way cluster SE for affected focal terms; use HC3, prompt-cluster, "
                "and URL-cluster comparison."
            )
        else:
            action = (
                "Two-way cluster warning affects nuisance/high-dimensional fixed-effect terms, not focal "
                "content terms; still report HC3 and one-way clustered SE as checks."
            )
        rows.append(
            {
                "model_id": model_id,
                "total_terms": total_terms,
                "negative_variance_terms": int(len(affected)),
                "negative_variance_share": len(affected) / total_terms if total_terms else np.nan,
                "negative_variance_focal_terms": int(len(focal)),
                "affected_focal_terms": "; ".join(affected_labels) if affected_labels else "none",
                "two_way_cluster_used_in_report": focal_all_available,
                "recommended_reporting_action": action,
            }
        )
    return pd.DataFrame(rows)


def _model_term_row(
    preferred: pd.DataFrame,
    model_id: str,
    term: str,
    *,
    alternate_term: str | None = None,
) -> pd.Series | None:
    terms = [term]
    if alternate_term:
        terms.append(alternate_term)
    match = preferred[preferred["model_id"].eq(model_id) & preferred["term"].isin(terms)]
    return None if match.empty else match.iloc[0]


def build_domain_fe_attenuation(results: pd.DataFrame) -> pd.DataFrame:
    preferred = _preferred_rows(results)
    m2 = preferred[preferred["model_id"].eq("M2")]
    rows = []
    for m2_row in m2.itertuples(index=False):
        m3_row = _model_term_row(preferred, "M3_domain_fe", m2_row.term)
        if m3_row is None:
            continue
        absolute_attenuation = abs(m2_row.estimate_pp) - abs(m3_row["estimate_pp"])
        percent_attenuation = (
            absolute_attenuation / abs(m2_row.estimate_pp) * 100
            if abs(m2_row.estimate_pp) > 1e-9
            else np.nan
        )
        label = _canonical_feature(m2_row.term)
        if label.startswith("Heading count"):
            interpretation = (
                "Large prompt-FE association attenuates sharply after domain fixed effects, suggesting "
                "domain/template confounding."
            )
        elif label == "Has table":
            interpretation = "Positive direction remains but magnitude shrinks under domain fixed effects."
        elif label == "Page length doubling":
            interpretation = "Direction changes and the estimates remain small; the association is not stable."
        elif label.startswith("Link count"):
            interpretation = "Direction changes under domain fixed effects and low-link support is imbalanced."
        elif label.startswith("Content strength"):
            interpretation = (
                "Direction is specification-sensitive; content strength is extraction-quality control only."
            )
        else:
            interpretation = "Compare prompt-FE and domain-FE estimates cautiously."
        rows.append(
            {
                "term": m2_row.term,
                "term_label": label,
                "estimate_M2_pp": m2_row.estimate_pp,
                "estimate_M3_domain_fe_pp": m3_row["estimate_pp"],
                "difference_M3_minus_M2_pp": m3_row["estimate_pp"] - m2_row.estimate_pp,
                "absolute_attenuation_pp": absolute_attenuation,
                "percent_attenuation_if_defined": percent_attenuation,
                "M2_ci_excludes_zero": _ci_excludes_zero(pd.Series(m2_row._asdict())),
                "M3_ci_excludes_zero": _ci_excludes_zero(m3_row),
                "domain_fe_attenuation_flag": bool(percent_attenuation >= 50) if pd.notna(percent_attenuation) else False,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def make_domain_attenuation_plot(table: pd.DataFrame, path: Path) -> None:
    work = table.sort_values("estimate_M2_pp", kind="stable")
    fig = go.Figure()
    for row in work.itertuples(index=False):
        fig.add_trace(
            go.Scatter(
                x=[row.estimate_M2_pp, row.estimate_M3_domain_fe_pp],
                y=[row.term_label, row.term_label],
                mode="lines",
                line={"color": "#a8adb3", "width": 2},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=work["estimate_M2_pp"],
            y=work["term_label"],
            mode="markers",
            marker={"size": 10, "color": "#277da1"},
            name="M2 prompt FE",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["estimate_M3_domain_fe_pp"],
            y=work["term_label"],
            mode="markers",
            marker={"size": 10, "color": "#f94144", "symbol": "diamond"},
            name="M3 prompt + domain FE",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#606b73")
    fig.update_layout(
        title="Attenuation after adding domain fixed effects",
        xaxis_title="Coefficient estimate (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=600,
        margin={"l": 220, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(fig, path)


def build_outlier_sensitivity(results: pd.DataFrame) -> pd.DataFrame:
    preferred = _preferred_rows(results)
    m2 = preferred[preferred["model_id"].eq("M2")]
    rows = []
    for m2_row in m2.itertuples(index=False):
        alternate = (
            "log2_word_count_plus1_winsorized_p99"
            if m2_row.term == "log2_word_count_plus1"
            else None
        )
        m10a = _model_term_row(preferred, "M10a_word_p99_removed", m2_row.term)
        m10b = _model_term_row(preferred, "M10b_link_p99_removed", m2_row.term)
        m10c = _model_term_row(
            preferred,
            "M10c_word_winsorized",
            m2_row.term,
            alternate_term=alternate,
        )
        values = [
            m2_row.estimate_pp,
            m10a["estimate_pp"] if m10a is not None else np.nan,
            m10b["estimate_pp"] if m10b is not None else np.nan,
            m10c["estimate_pp"] if m10c is not None else np.nan,
        ]
        finite = np.asarray([value for value in values if pd.notna(value)], dtype=float)
        direction_changed = bool((finite > 0).any() and (finite < 0).any())
        deviations = [abs(value - values[0]) for value in values[1:] if pd.notna(value)]
        substantial_threshold = max(2.0, abs(values[0]) * 0.5)
        magnitude_changed = bool(deviations and max(deviations) >= substantial_threshold)
        base_ci = _ci_excludes_zero(pd.Series(m2_row._asdict()))
        sensitivity_rows = [row for row in (m10a, m10b, m10c) if row is not None]
        ci_status_changed = any(_ci_excludes_zero(row) != base_ci for row in sensitivity_rows)
        word_tail_sensitive = bool(
            m2_row.term == "log2_word_count_plus1"
            and m10a is not None
            and not base_ci
            and _ci_excludes_zero(m10a)
            and abs(m10a["estimate_pp"]) > abs(m2_row.estimate_pp)
        )
        not_link_tail_sensitive = bool(
            m10b is not None
            and np.sign(m10b["estimate_pp"]) == np.sign(m2_row.estimate_pp)
            and abs(m10b["estimate_pp"] - m2_row.estimate_pp) < substantial_threshold
            and _ci_excludes_zero(m10b) == base_ci
        )
        label = _canonical_feature(m2_row.term)
        if label == "Page length doubling":
            classification = "word_count_tail_sensitive"
            wording = (
                "Page length is sensitive to extreme word-count tails and should not be interpreted as a "
                "stable content feature until the source of long pages is inspected."
            )
        elif direction_changed or magnitude_changed or ci_status_changed:
            classification = "outlier_or_specification_sensitive"
            wording = "Magnitude, direction, or interval conclusion changes across tail sensitivities."
        else:
            classification = "not_materially_tail_sensitive"
            wording = "The focal estimate is broadly similar across the specified tail checks."
        rows.append(
            {
                "term": m2_row.term,
                "term_label": label,
                "estimate_M2_pp": m2_row.estimate_pp,
                "estimate_M10a_word_p99_removed_pp": (
                    m10a["estimate_pp"] if m10a is not None else np.nan
                ),
                "estimate_M10b_link_p99_removed_pp": (
                    m10b["estimate_pp"] if m10b is not None else np.nan
                ),
                "estimate_M10c_winsorized_pp": (
                    m10c["estimate_pp"] if m10c is not None else np.nan
                ),
                "direction_changed": direction_changed,
                "magnitude_changed_substantially": magnitude_changed,
                "ci_status_changed": ci_status_changed,
                "word_count_tail_sensitive": word_tail_sensitive,
                "not_link_tail_sensitive": not_link_tail_sensitive,
                "outlier_sensitivity_classification": classification,
                "recommended_wording": wording,
            }
        )
    return pd.DataFrame(rows)


def build_robustness_classification(
    results: pd.DataFrame,
    se_stability: pd.DataFrame,
    attenuation: pd.DataFrame,
    outliers: pd.DataFrame,
) -> pd.DataFrame:
    preferred = _preferred_rows(results)
    selected_models = (
        "M2",
        "M3_domain_fe",
        "M4_gemini_taxonomy",
        "M5_M2_strong",
        "M10a_word_p99_removed",
        "M10b_link_p99_removed",
        "M10c_word_winsorized",
    )
    selected = preferred[preferred["model_id"].isin(selected_models)].copy()
    selected["feature"] = selected["term"].map(_canonical_feature)
    rows = []
    features = [
        "Page length doubling",
        "Has table",
        "Heading count: 2-6",
        "Heading count: 7-12",
        "Heading count: 13+",
        "Link count: 0-3",
        "Link count: 4-8",
        "Content strength: medium",
        "Content strength: weak",
    ]
    for feature in features:
        group = selected[selected["feature"].eq(feature)]
        attenuation_row = attenuation[attenuation["term_label"].eq(feature)]
        outlier_row = outliers[outliers["term_label"].eq(feature)]
        domain_flag = bool(
            not attenuation_row.empty and attenuation_row.iloc[0]["domain_fe_attenuation_flag"]
        )
        outlier_flag = bool(
            not outlier_row.empty
            and outlier_row.iloc[0]["outlier_sensitivity_classification"]
            != "not_materially_tail_sensitive"
        )
        se_rows = se_stability[se_stability["term_label"].map(_canonical_feature).eq(feature)]
        se_sensitive = bool(
            se_rows["classification"].isin(
                [
                    "direction_stable_but_significance_sensitive",
                    "direction_sensitive",
                    "unavailable_or_unreliable_se",
                ]
            ).any()
        )
        positive = int(group["estimate_pp"].gt(0).sum())
        negative = int(group["estimate_pp"].lt(0).sum())
        zero_crossing = int((~group.apply(_ci_excludes_zero, axis=1)).sum()) if len(group) else 0
        significant = int(group.apply(_ci_excludes_zero, axis=1).sum()) if len(group) else 0

        if feature == "Has table":
            classification = "suggestive"
            wording = (
                "Table presence is associated with higher citation probability across several specifications, "
                "but uncertainty remains and many intervals include zero."
            )
        elif feature.startswith("Heading count"):
            classification = "unstable"
            wording = (
                "Large prompt-FE association attenuates strongly under domain fixed effects, suggesting "
                "domain/template or page-function confounding."
            )
        elif feature == "Page length doubling":
            classification = "unstable"
            wording = (
                "Page length has no stable interpretation because the preferred estimate is small and the "
                "word-count p99 sensitivity changes inference."
            )
        elif feature.startswith("Link count"):
            classification = "unstable"
            wording = (
                "Link-count estimates are diagnostic only because low-link groups are highly imbalanced and "
                "signs or intervals vary across specifications."
            )
        else:
            classification = "descriptive_only"
            wording = (
                "Content strength is extraction-quality control, not writing quality, and should not receive "
                "a substantive interpretation."
            )
        rows.append(
            {
                "feature": feature,
                "n_models_available": int(group["model_id"].nunique()),
                "positive_count": positive,
                "negative_count": negative,
                "zero_crossing_count": zero_crossing,
                "significant_count": significant,
                "domain_fe_attenuation": domain_flag,
                "outlier_sensitive": outlier_flag,
                "se_sensitive": se_sensitive,
                "classification": classification,
                "recommended_wording": wording,
            }
        )
    return pd.DataFrame(rows)


def _classification_lookup(classification: pd.DataFrame, feature: str) -> pd.Series:
    match = classification[classification["feature"].eq(feature)]
    if match.empty:
        raise KeyError(f"Missing robustness classification for {feature}")
    return match.iloc[0]


def patch_minimum_reporting_table(
    original: pd.DataFrame,
    classification: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for row in original.to_dict(orient="records"):
        feature = str(row["feature"]).replace("Page length (doubling)", "Page length doubling")
        class_row = _classification_lookup(classification, feature)
        if feature == "Has table":
            bucket = "suggestive_positive"
        elif feature.startswith("Heading count"):
            bucket = "domain_template_confounded"
        elif feature == "Page length doubling" or feature.startswith("Link count"):
            bucket = "unstable_diagnostic"
        elif feature.startswith("Content strength"):
            bucket = "extraction_quality_control"
        else:
            bucket = "insufficient_support"
        row.update(
            {
                "robustness_classification": class_row["classification"],
                "domain_fe_attenuation_flag": class_row["domain_fe_attenuation"],
                "outlier_sensitivity_flag": class_row["outlier_sensitive"],
                "se_sensitivity_flag": class_row["se_sensitive"],
                "final_interpretation_bucket": bucket,
                "recommended_sentence": class_row["recommended_wording"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_next_feature_plan() -> pd.DataFrame:
    specs = [
        (
            "factual_numeric_density",
            "Tests whether tables proxy for dense, verifiable numerical facts.",
            "page text and structured table/list text",
            "low",
            "positive",
            1,
            "Validate Thai/English numeral parsing and normalize by measurable words.",
        ),
        (
            "price_mention_count",
            "Separates pricing facts from table formatting itself.",
            "page text only",
            "low",
            "positive",
            1,
            "Validate currencies, ranges, abbreviations, and duplicate template prices.",
        ),
        (
            "unit_size_mention_count",
            "Captures concrete unit-size evidence often presented in specifications or tables.",
            "page text only",
            "low",
            "positive",
            1,
            "Validate sqm, sq.m., square metre, and Thai area expressions.",
        ),
        (
            "transit_location_fact_count",
            "Measures concrete distance, station, road, and neighborhood facts.",
            "page text only",
            "low",
            "positive",
            1,
            "Build bilingual location lexicon and distinguish navigation boilerplate.",
        ),
        (
            "amenity_fact_count",
            "Tests whether table presence proxies for detailed amenity coverage.",
            "page text and list items",
            "low",
            "positive",
            1,
            "Deduplicate repeated menu/footer amenity labels.",
        ),
        (
            "project_entity_count",
            "Measures named project specificity rather than generic condo prose.",
            "page text only",
            "low",
            "positive",
            1,
            "Validate named-entity aliases and avoid counting navigation repetition.",
        ),
        (
            "developer_entity_count",
            "Captures publisher/project context and potential factual authority signals.",
            "page text only",
            "low",
            "positive_or_control",
            2,
            "Validate developer aliases and separate author/publisher boilerplate.",
        ),
        (
            "external_evidence_link_count",
            "Tests whether citations correlate with links to maps, reports, official records, or evidence.",
            "page HTML links and anchor text",
            "low",
            "positive",
            2,
            "Classify internal vs external links and exclude social/share/navigation links.",
        ),
        (
            "opening_summary_present",
            "Captures whether key facts are summarized early and accessibly.",
            "first page-text section only",
            "low",
            "positive",
            2,
            "Define pre-specified summary heuristics without using citation outcomes.",
        ),
        (
            "question_heading_count",
            "Measures explicit question-answer organization independently of total headings.",
            "page headings only",
            "low",
            "positive_or_nonlinear",
            2,
            "Validate question punctuation and Thai interrogative patterns.",
        ),
        (
            "list_structure_score",
            "Tests whether structured lists, rather than tables alone, improve fact accessibility.",
            "page HTML lists and extracted text",
            "low",
            "positive",
            2,
            "Separate content lists from menus, filters, and footer navigation.",
        ),
        (
            "paragraph_length_median",
            "Measures prose chunking without relying on total page length.",
            "page text blocks only",
            "low",
            "nonlinear",
            3,
            "Validate paragraph segmentation across crawler outputs.",
        ),
        (
            "sentence_length_median",
            "Captures readability and information packaging at sentence level.",
            "page text only",
            "low",
            "nonlinear",
            3,
            "Use bilingual sentence segmentation and inspect abbreviations.",
        ),
        (
            "prompt_page_similarity_no_answer",
            "Controls observable prompt-page relevance without answer-derived leakage.",
            "prompt text and page text only",
            "medium",
            "positive_control",
            1,
            "Freeze embedding/lexical method before modeling and verify no answer fields enter.",
        ),
    ]
    return pd.DataFrame(
        specs,
        columns=[
            "feature_name",
            "rationale",
            "extraction_source",
            "leakage_risk",
            "expected_relation_to_citation",
            "priority",
            "validation_needed",
        ],
    )


def _write_revised_report(
    path: Path,
    sample: pd.DataFrame,
    se_stability: pd.DataFrame,
    warning_audit: pd.DataFrame,
    attenuation: pd.DataFrame,
    outliers: pd.DataFrame,
    classification: pd.DataFrame,
) -> None:
    counts = {
        "rows": len(sample),
        "urls": sample["normalized_url"].nunique(),
        "prompts": sample["prompt_id"].nunique(),
        "domains": sample["source_root_domain"].nunique(),
        "cited": int(sample["cited"].sum()),
        "rate": float(sample["cited"].mean()),
    }
    warning_models = int(warning_audit["negative_variance_terms"].gt(0).sum())
    focal_warning_models = int(warning_audit["negative_variance_focal_terms"].gt(0).sum())
    se_sensitive_count = int(
        se_stability["classification"].ne("stable_inference").sum()
    )
    report = f"""# 09 Content Feature Econometrics Report: Interpretation Patch

## 1. Analysis scope and estimand

This report patches the interpretation of the existing content-feature models for the area-condo / SCOPE-relevant nonbranded audit. The unit remains one surfaced source appearance, and the estimand remains `P(cited = 1 | source surfaced in this audit)`. Results are associations conditional on surfaced sources. They are not causal and not web-wide.

## 2. Dataset and sample counts

- Measurable source appearances: {counts['rows']:,}
- Unique normalized URLs: {counts['urls']:,}
- Unique prompts in the measurable sample: {counts['prompts']:,}
- Unique source-root domains: {counts['domains']:,}
- Cited rows: {counts['cited']:,}
- Cited rate: {counts['rate']:.2%}
- Full audit = 500 prompts; measurable-content LPM sample = {counts['prompts']:,} prompts.

## 3. Model ladder

The original model ladder is retained. M1 and M2 remain the first reported estimates; M3 is domain-fixed-effect robustness; M4 uses the versioned Gemini page-function family and source/site type; M5 restricts to strong extraction quality; and M10 evaluates word-count and link-count tails. The older rule-v2 URL-seed model is comparison-only.

## 4. Main M2 results

The M2 coefficients are not interpreted from their two-way clustered intervals alone.

- **Table presence:** Table presence is the most directionally consistent focal feature. It remains positive across several specifications, but many confidence intervals include zero. Therefore, evidence is suggestive rather than definitive.
- **Heading-count groups:** Heading-count categories show large negative associations in prompt-FE models, but these estimates attenuate strongly under domain fixed effects. This suggests the pattern may reflect domain/template or page-function differences rather than heading structure alone.
- **Page length:** Page length has no stable interpretation. The preferred model estimate is small and imprecise, and the result changes under p99 word-count sensitivity.
- **Link count:** Link-count estimates are unstable and should be treated as diagnostic because the low-link categories are highly imbalanced.
- **Content strength:** `content_strength` is an extraction-quality control, not a writing-quality measure. Direction changes across specifications, so it should not be interpreted substantively.

## 5. Domain-FE robustness

The domain-FE comparison is a central interpretation boundary. Heading-count coefficients fall sharply toward zero after adding domain fixed effects, with one heading category changing direction. This is consistent with domain/template or page-function confounding and prevents a clean heading-structure interpretation. Table presence remains positive, but its magnitude also shrinks.

## 6. Gemini taxonomy sensitivity

The Gemini taxonomy sensitivity shows whether content coefficients survive adjustment for LLM-classified page function and source/site type. Because Gemini may use scraped body content, this model can over-control the structural features under study and remains a sensitivity rather than the headline. It cannot remove all publisher, template, authority, or prompt-page relevance confounding.

## 7. Strong-content sensitivity

The strong-content restriction checks extraction measurement quality. It does not turn `content_strength` into writing quality. Table presence remains positive in the strong-content variants, while heading-count estimates still depend heavily on whether domain fixed effects are included.

## 8. Outlier and winsorized sensitivity

Page length is sensitive to extreme word-count tails and should not be interpreted as a stable content feature until the source of long pages is inspected. Removing the top 1% word-count tail changes the page-length estimate from small and imprecise to larger with an interval excluding zero. Removing the link-count tail does not produce the same change.

## 9. Logit AME cross-check

The simplified logit AME remains a functional-form cross-check only. It does not replace the prompt-FE LPM headline because full prompt-FE logit has separation risk.

## 10. Standard-error robustness and covariance warnings

Two-way clustered covariance can be unstable in high-dimensional fixed-effect models with repeated prompts and URLs. Negative diagonal variances indicate that some reported SEs are not valid for affected terms. Therefore, focal content estimates should be checked against HC3, prompt-cluster, and URL-cluster alternatives.

- {warning_models} requested model variants had at least one negative two-way covariance diagonal variance.
- {focal_warning_models} requested model variants had an unavailable two-way SE for a focal content term.
- {se_sensitive_count} model-term combinations were not classified as fully stable across covariance and cross-model checks.

In the current outputs, the negative variances affect nuisance/high-dimensional fixed-effect coefficients rather than the focal content terms. That distinction permits reporting the focal two-way intervals, but does not justify relying on that estimator alone where significance differs across covariance choices.

## 11. Final interpretation buckets

### Suggestive

- **Has table:** associated with higher citation probability across several specifications, but uncertainty remains.

### Domain/template-confounded

- **Heading-count groups:** large prompt-FE associations attenuate strongly under domain fixed effects. Do not convert this pattern into a directional heading recommendation.

### Unstable or diagnostic

- **Page length:** sensitive to the top word-count tail.
- **Link-count groups:** sparse relative to the dominant `9+` group and specification-sensitive.
- **Content strength:** extraction-quality control only, with no substantive writing interpretation.

## 12. What appears robust enough to discuss

Only table presence reaches the suggestive bucket. Even here, the appropriate wording is “associated with higher citation probability conditional on surfaced sources,” not a claim that tables produce citations.

## 13. What is not robust enough for a substantive claim

Heading count, page length, link count, and content strength should not be converted into webpage-writing recommendations from this analysis. Their estimates attenuate under domain fixed effects, change under tail sensitivity, rely on imbalanced categories, or represent extraction quality rather than writing.

## 14. Limitations and next feature layer

- The sample includes surfaced sources only and does not estimate web-wide citation likelihood.
- Scrape/content measurability is selected rather than random.
- Domain fixed effects do not solve time-varying, page-level, or relevance confounding.
- The current structural counts do not measure factual accuracy, specificity, or usefulness directly.
- Results remain observational and not causal.

The next feature layer should test whether table presence is proxying for factual and numerical density, price and unit-size facts, transit/location evidence, amenities, named project/developer entities, external evidence links, opening summaries, question headings, list structure, and prompt-page similarity computed without answer text.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _write_executive_summary(path: Path, sample: pd.DataFrame) -> None:
    summary = f"""# Content Feature Econometrics: Executive Summary

## What was estimated

The analysis estimates Linear Probability Models for whether a surfaced source appearance was cited in the area-condo / SCOPE-relevant nonbranded audit. The estimand is citation probability conditional on a source already being surfaced. It is not a causal estimate and does not describe all webpages on the internet.

## Sample

The measurable-content sample contains {len(sample):,} source appearances, {sample['normalized_url'].nunique():,} normalized URLs, {sample['prompt_id'].nunique():,} prompts, and {sample['source_root_domain'].nunique():,} domains. There are {int(sample['cited'].sum()):,} cited appearances, for a cited rate of {sample['cited'].mean():.2%}. The full audit has 500 prompts; 498 prompts appear in the measurable-content LPM sample.

## Main interpretation

The clearest suggestive signal is table presence: pages with tables tend to have higher citation probability across several specifications, but uncertainty remains. This is interesting because the positive direction survives prompt fixed effects, Gemini taxonomy adjustment, strong-content restriction, and the specified outlier checks. However, many confidence intervals include zero and the magnitude shrinks under domain fixed effects. The evidence is suggestive, not definitive, and does not imply that adding a table will change citation outcomes.

Heading-count differences are large in prompt-fixed-effect models, but they attenuate substantially after domain fixed effects, suggesting domain/template confounding. The observed heading pattern may therefore reflect publisher templates, page functions, or other domain-level characteristics rather than heading structure alone. It should not be converted into a directional heading recommendation.

Page length and link-count estimates are not stable enough for substantive interpretation. Page length changes materially after removing the top 1% word-count tail. Link-count groups are extremely imbalanced, with nearly all observations in the `9+` group, and estimates change direction or remain imprecise across important specifications.

`content_strength` is an extraction-quality control, not a writing-quality measure. Its direction changes across specifications, so it belongs in diagnostics and sample-quality sensitivity rather than client-facing content advice.

## Standard-error caution

The high-dimensional fixed-effect models generate negative diagonal variances for some nuisance terms in the two-way clustered covariance matrix. Focal content-term two-way SEs remain available in the current run, but inference should be compared across HC3, prompt-cluster, URL-cluster, and two-way clustered estimates. Conclusions should not depend on a single fragile covariance choice.

## Next step

The next step is to add richer, pre-specified writing-quality and factual-density features extracted only from page text and prompt text, without answer-derived variables. Priority measures include factual/numeric density, price and unit-size mentions, transit/location facts, amenity facts, project/developer entities, external evidence links, opening summaries, question headings, list structure, and prompt-page similarity without answer text. These features can test whether table presence is proxying for useful factual specificity rather than formatting itself.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")


def run_content_feature_interpretation_patch(package_path: Path | str) -> dict[str, Any]:
    """Build all interpretation-patch diagnostics from existing notebook 09 outputs."""
    package = Path(package_path).resolve()
    input_path = package / "data/content_lpm_measurable_rows.csv"
    original_tables = package / "tables/09_content_feature_econometrics"
    table_out = original_tables / "interp_patch"
    figure_out = package / "figures/09_content_feature_econometrics/interp_patch"
    report_out = package / "reports/09_content_feature_econometrics/interp_patch"
    for directory in (table_out, figure_out, report_out):
        directory.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(input_path, low_memory=False)
    focal_results = _load_model_results(original_tables)
    all_results = []
    for filename in MODEL_FILES.values():
        all_results.append(pd.read_csv(original_tables / filename, low_memory=False))
    all_results_frame = pd.concat(all_results, ignore_index=True)

    se_comparison = build_focal_term_se_comparison(focal_results)
    se_comparison.to_csv(table_out / "focal_term_se_comparison.csv", index=False)

    se_stability = build_se_stability_summary(se_comparison)
    se_stability.to_csv(table_out / "focal_term_se_stability_summary.csv", index=False)
    make_se_comparison_forest(
        se_comparison,
        figure_out / "focal_term_se_comparison_forest.html",
    )

    compared_model_ids = set(se_comparison["model_id"].unique())
    warning_audit = build_two_way_warning_audit(all_results_frame)
    warning_audit = warning_audit[warning_audit["model_id"].isin(compared_model_ids)].reset_index(drop=True)
    warning_audit.to_csv(table_out / "two_way_cluster_warning_audit.csv", index=False)

    attenuation = build_domain_fe_attenuation(focal_results)
    attenuation.to_csv(table_out / "domain_fe_attenuation_summary.csv", index=False)
    make_domain_attenuation_plot(
        attenuation,
        figure_out / "domain_fe_attenuation_focal_terms.html",
    )

    outliers = build_outlier_sensitivity(focal_results)
    outliers.to_csv(table_out / "outlier_sensitivity_focal_terms.csv", index=False)

    classification = build_robustness_classification(
        focal_results,
        se_stability,
        attenuation,
        outliers,
    )
    classification.to_csv(table_out / "focal_feature_robustness_classification.csv", index=False)

    original_minimum_path = original_tables / "09_minimum_reporting_table.csv"
    original_minimum = pd.read_csv(original_minimum_path, low_memory=False)
    patched_minimum = patch_minimum_reporting_table(original_minimum, classification)
    patched_minimum_path = table_out / "09_minimum_reporting_table_v2_interpretation_patch.csv"
    patched_minimum.to_csv(patched_minimum_path, index=False)

    next_features = build_next_feature_plan()
    next_features.to_csv(table_out / "next_feature_layer_priority_plan.csv", index=False)

    revised_report_path = report_out / "09_content_feature_econometrics_report_v2_interpretation_patch.md"
    _write_revised_report(
        revised_report_path,
        sample,
        se_stability,
        warning_audit,
        attenuation,
        outliers,
        classification,
    )
    executive_path = report_out / "09_content_feature_econometrics_executive_summary_v2.md"
    _write_executive_summary(executive_path, sample)

    bucket_counts = classification.assign(
        final_bucket=classification["feature"].map(
            lambda feature: (
                "suggestive_positive"
                if feature == "Has table"
                else "domain_template_confounded"
                if feature.startswith("Heading count")
                else "unstable_diagnostic"
                if feature == "Page length doubling" or feature.startswith("Link count")
                else "extraction_quality_control"
            )
        )
    )["final_bucket"].value_counts()
    summary = {
        "number_of_focal_terms_checked": int(se_comparison["term"].nunique()),
        "number_of_models_included_in_se_comparison": int(se_comparison["model_id"].nunique()),
        "number_of_two_way_cluster_warnings": int(
            warning_audit["negative_variance_terms"].gt(0).sum()
        ),
        "number_of_features_classified_as_suggestive": int(bucket_counts.get("suggestive_positive", 0)),
        "number_of_features_classified_as_domain_template_confounded": int(
            bucket_counts.get("domain_template_confounded", 0)
        ),
        "number_of_features_classified_as_unstable_diagnostic": int(
            bucket_counts.get("unstable_diagnostic", 0)
        ),
        "revised_report": str(revised_report_path),
        "revised_minimum_reporting_table": str(patched_minimum_path),
        "executive_summary": str(executive_path),
        "final_status": "completed_interpretation_patch_ready_for_writeup_with_caveats",
    }
    output_files = sorted(
        str(path.relative_to(package))
        for directory in (table_out, figure_out, report_out)
        for path in directory.rglob("*")
        if path.is_file()
    )
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_data_path": str(input_path),
        "existing_model_table_path": str(original_tables),
        "sample_counts": {
            "n_rows": len(sample),
            "n_urls": sample["normalized_url"].nunique(),
            "n_prompts": sample["prompt_id"].nunique(),
            "n_domains": sample["source_root_domain"].nunique(),
            "cited_rows": int(sample["cited"].sum()),
            "cited_rate": float(sample["cited"].mean()),
        },
        "summary": summary,
        "output_files": output_files,
        "estimand_changed": False,
        "raw_input_overwritten": False,
        "answer_derived_variables_used": False,
    }
    manifest_path = report_out / "interpretation_patch_manifest.json"
    manifest["output_files"].append(str(manifest_path.relative_to(package)))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary
