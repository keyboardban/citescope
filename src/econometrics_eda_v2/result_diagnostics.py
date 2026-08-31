"""Post-estimation diagnostics for the governed D0-FE4 econometric run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.econometrics_eda_v2.content_feature_econometrics import run_model_and_save
from src.econometrics_eda_v2.redesigned_pipeline_v2 import (
    CONTROL,
    FOCAL,
    PAGE_TYPE,
    SOURCE_TYPE,
    build_model_ready,
    formulas,
)
from src.econometrics_eda_v2.writing_structure_v3 import WRITING_STRUCTURE_COMPONENTS


PREFERRED_COVARIANCE = "two_way_cluster_prompt_url"
MODEL_FILES = {
    "FE1": "FE1_one_feature_prompt_fe_results.csv",
    "FE2": "FE2_joint_core_prompt_fe_results.csv",
    "FE3": "FE3_domain_fe_robustness_results.csv",
    "FE4": "FE4_taxonomy_sensitivity_results.csv",
}


def _read_model_tables(output_root: Path) -> pd.DataFrame:
    frames = []
    for layer, filename in MODEL_FILES.items():
        frame = pd.read_csv(output_root / "tables" / filename, low_memory=False)
        frame["analysis_layer"] = layer
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _focal_rows(models: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_layer", "model_id", "term", "cov_type", "estimate", "estimate_pp",
        "std_error", "conf_low_pp", "conf_high_pp", "p_value", "n_obs", "n_prompts",
        "n_urls", "n_domains", "r_squared", "notes",
    ]
    return models[models["term"].isin(FOCAL)][columns].copy()


def _variation_block(
    data: pd.DataFrame,
    feature: str,
    group: str,
    *,
    eligible_mask: pd.Series | None = None,
) -> dict[str, Any]:
    frame = data.loc[eligible_mask].copy() if eligible_mask is not None else data.copy()
    values = pd.to_numeric(frame[feature], errors="coerce")
    working = pd.DataFrame({group: frame[group].astype(str), "value": values}, index=frame.index)
    unique = working.groupby(group, observed=True)["value"].nunique(dropna=True)
    varying_groups = set(unique[unique.gt(1)].index)
    rows_mask = working[group].isin(varying_groups)
    return {
        "feature_name": feature,
        "grouping": group,
        "sample_rows": len(frame),
        "total_groups": int(working[group].nunique()),
        "groups_with_usable_variation": len(varying_groups),
        "groups_with_usable_variation_rate": (
            len(varying_groups) / working[group].nunique() if working[group].nunique() else np.nan
        ),
        "rows_in_varying_groups": int(rows_mask.sum()),
        "rows_in_varying_groups_rate": float(rows_mask.mean()) if len(rows_mask) else np.nan,
        "unique_urls_in_varying_groups": int(frame.loc[rows_mask, "normalized_url"].nunique()),
    }


def variation_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    domain_url_counts = data.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = set(domain_url_counts[domain_url_counts.ge(2)].index.astype(str))
    supported_mask = data["source_root_domain"].astype(str).isin(supported_domains)
    rows = []
    for feature in [*FOCAL, *WRITING_STRUCTURE_COMPONENTS]:
        rows.append(_variation_block(data, feature, "prompt_id"))
        rows.append(_variation_block(data, feature, "source_root_domain"))
        supported = _variation_block(
            data,
            feature,
            "source_root_domain",
            eligible_mask=supported_mask,
        )
        supported["grouping"] = "source_root_domain_FE3_sample"
        rows.append(supported)

        prompt_varies = (
            data.groupby("prompt_id", observed=True)[feature].nunique(dropna=True).gt(1)
        )
        domain_varies = (
            data.loc[supported_mask]
            .groupby("source_root_domain", observed=True)[feature]
            .nunique(dropna=True)
            .gt(1)
        )
        intersection = (
            data["prompt_id"].astype(str).isin(set(prompt_varies[prompt_varies].index.astype(str)))
            & data["source_root_domain"].astype(str).isin(
                set(domain_varies[domain_varies].index.astype(str))
            )
            & supported_mask
        )
        rows.append(
            {
                "feature_name": feature,
                "grouping": "prompt_and_domain_variation_intersection",
                "sample_rows": int(supported_mask.sum()),
                "total_groups": np.nan,
                "groups_with_usable_variation": np.nan,
                "groups_with_usable_variation_rate": np.nan,
                "rows_in_varying_groups": int(intersection.sum()),
                "rows_in_varying_groups_rate": (
                    float(intersection.sum() / supported_mask.sum())
                    if supported_mask.sum()
                    else np.nan
                ),
                "unique_urls_in_varying_groups": int(
                    data.loc[intersection, "normalized_url"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def coefficient_transitions(focal: pd.DataFrame) -> pd.DataFrame:
    preferred = focal[focal["cov_type"].eq(PREFERRED_COVARIANCE)].copy()
    wide = preferred.pivot(index="term", columns="analysis_layer", values="estimate_pp")
    rows = []
    for feature in FOCAL:
        values = wide.loc[feature]
        rows.append(
            {
                "feature_name": feature,
                "FE1_estimate_pp": values.get("FE1"),
                "FE2_estimate_pp": values.get("FE2"),
                "FE3_estimate_pp": values.get("FE3"),
                "FE4_estimate_pp": values.get("FE4"),
                "FE1_to_FE2_change_pp": values.get("FE2") - values.get("FE1"),
                "FE2_to_FE3_total_change_pp": values.get("FE3") - values.get("FE2"),
                "FE2_to_FE4_change_pp": values.get("FE4") - values.get("FE2"),
            }
        )
    return pd.DataFrame(rows)


def _component_descriptives(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in WRITING_STRUCTURE_COMPONENTS:
        values = pd.to_numeric(data[feature], errors="coerce")
        for state, label in ((0, "not_detected"), (1, "detected")):
            subset = data.loc[values.eq(state)]
            rows.append(
                {
                    "feature_name": feature,
                    "state": state,
                    "state_label": label,
                    "n_rows": len(subset),
                    "row_share": len(subset) / len(data),
                    "cited_rows": int(subset["cited"].sum()),
                    "cited_rate": float(subset["cited"].mean()) if len(subset) else np.nan,
                    "unique_prompts": int(subset["prompt_id"].nunique()),
                    "unique_urls": int(subset["normalized_url"].nunique()),
                }
            )
        rows.append(
            {
                "feature_name": feature,
                "state": np.nan,
                "state_label": "unmeasured",
                "n_rows": int(values.isna().sum()),
                "row_share": float(values.isna().mean()),
                "cited_rows": int(data.loc[values.isna(), "cited"].sum()),
                "cited_rate": (
                    float(data.loc[values.isna(), "cited"].mean())
                    if values.isna().any()
                    else np.nan
                ),
                "unique_prompts": int(data.loc[values.isna(), "prompt_id"].nunique()),
                "unique_urls": int(data.loc[values.isna(), "normalized_url"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _component_overlap(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    components = data[list(WRITING_STRUCTURE_COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    correlation = components.corr().rename_axis("feature_name").reset_index()
    faq = pd.to_numeric(data["has_faq_pattern"], errors="coerce")
    qa = pd.to_numeric(data["has_question_answer_structure"], errors="coerce")
    contingency = (
        pd.DataFrame({"has_faq_pattern": faq, "has_question_answer_structure": qa})
        .value_counts(dropna=False)
        .rename("n_rows")
        .reset_index()
    )
    contingency["row_share"] = contingency["n_rows"] / len(data)
    return correlation, contingency


def _run_component_fe1(
    data: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for feature in WRITING_STRUCTURE_COMPONENTS:
        result = run_model_and_save(
            f"cited ~ {feature} + C(prompt_id)",
            data,
            f"FE1_component_{feature}",
            output_dir / f".{feature}_working.csv",
        )
        focal = result.table[result.table["term"].eq(feature)].copy()
        focal.insert(0, "analysis_layer", "FE1_component_diagnostic")
        rows.append(focal)
        (output_dir / f".{feature}_working.csv").unlink(missing_ok=True)
    return pd.concat(rows, ignore_index=True)


def _multiple_testing(
    focal: pd.DataFrame,
    component_results: pd.DataFrame,
) -> pd.DataFrame:
    families = []
    primary = focal[
        focal["analysis_layer"].eq("FE2")
        & focal["cov_type"].eq(PREFERRED_COVARIANCE)
    ].copy()
    primary["hypothesis_family"] = "headline_FE2_four_core_features"
    primary["family_role"] = "pre_registered_headline"
    families.append(primary)

    exploratory = component_results[
        component_results["cov_type"].eq(PREFERRED_COVARIANCE)
    ].copy()
    exploratory["hypothesis_family"] = "exploratory_FE1_five_writing_components"
    exploratory["family_role"] = "expanded_exploratory"
    families.append(exploratory)

    output = pd.concat(families, ignore_index=True)
    adjusted = []
    for _, group in output.groupby("hypothesis_family", sort=False):
        reject, p_adjusted, _, _ = multipletests(
            group["p_value"].to_numpy(dtype=float),
            alpha=0.05,
            method="fdr_bh",
        )
        block = group.copy()
        block["p_value_bh_fdr"] = p_adjusted
        block["reject_at_fdr_0_05"] = reject
        block["family_size"] = len(block)
        block["correction_method"] = "Benjamini-Hochberg FDR"
        adjusted.append(block)
    return pd.concat(adjusted, ignore_index=True)


def _group_concentration(
    data: pd.DataFrame,
    group: str,
    feature: str,
    *,
    kind: str,
) -> pd.DataFrame:
    numeric = pd.to_numeric(data[feature], errors="coerce")
    cutoff = float(numeric.quantile(0.99))
    working = data[[group, "normalized_url", "cited"]].copy()
    working["value"] = numeric
    working["top_1pct"] = numeric.ge(cutoff)
    rows = []
    total_detected = int(numeric.eq(1).sum()) if kind == "binary" else 0
    for level, subset in working.groupby(group, dropna=False, observed=True):
        values = subset["value"]
        row = {
            "grouping": group,
            "group_value": str(level),
            "feature_name": feature,
            "n_rows": len(subset),
            "unique_urls": int(subset["normalized_url"].nunique()),
            "cited_rate": float(subset["cited"].mean()),
            "mean_value": float(values.mean()),
            "median_value": float(values.median()),
            "p90_value": float(values.quantile(0.90)),
            "p99_value": float(values.quantile(0.99)),
            "top_1pct_rows": int(subset["top_1pct"].sum()),
        }
        if kind == "binary":
            detected = int(values.eq(1).sum())
            row.update(
                {
                    "detected_rows": detected,
                    "detected_rate": float(values.eq(1).mean()),
                    "share_of_all_detected_rows": (
                        detected / total_detected if total_detected else np.nan
                    ),
                }
            )
        rows.append(row)
    output = pd.DataFrame(rows)
    sort_column = "detected_rows" if kind == "binary" else "top_1pct_rows"
    return output.sort_values([sort_column, "n_rows"], ascending=False)


def concentration_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = [PAGE_TYPE, SOURCE_TYPE, "source_root_domain"]
    factual = pd.concat(
        [
            _group_concentration(
                data,
                group,
                "factual_numeric_density_score",
                kind="continuous",
            )
            for group in groups
        ],
        ignore_index=True,
    )
    tables = pd.concat(
        [
            _group_concentration(
                data,
                group,
                "has_verified_html_table",
                kind="binary",
            )
            for group in groups
        ],
        ignore_index=True,
    )
    return factual, tables


def extraction_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for strength, subset in data.groupby(CONTROL, dropna=False, observed=True):
        rows.append(
            {
                "content_strength": str(strength),
                "n_rows": len(subset),
                "row_share": len(subset) / len(data),
                "unique_urls": int(subset["normalized_url"].nunique()),
                "cited_rate": float(subset["cited"].mean()),
                "html_available_rate": float(
                    pd.to_numeric(subset["html_available"], errors="coerce").mean()
                ),
                **{
                    f"mean_{feature}": float(
                        pd.to_numeric(subset[feature], errors="coerce").mean()
                    )
                    for feature in FOCAL
                },
            }
        )
    missing = []
    for feature in [
        *FOCAL,
        *WRITING_STRUCTURE_COMPONENTS,
        "html_available",
        "document_features_measurable",
        "content_feature_available",
        "text_feature_available",
    ]:
        values = data[feature] if feature in data else pd.Series(pd.NA, index=data.index)
        missing.append(
            {
                "feature_name": feature,
                "n_rows": len(data),
                "nonmissing_rows": int(values.notna().sum()),
                "missing_rows": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(missing)


def _restricted_sample_diagnostic(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    domain_url_counts = data.groupby("source_root_domain")["normalized_url"].nunique()
    supported = set(domain_url_counts[domain_url_counts.ge(2)].index)
    restricted = data[data["source_root_domain"].isin(supported)].copy()
    formula = formulas(data)["FE2"]
    run = run_model_and_save(
        formula,
        restricted,
        "FE2_on_FE3_sample_diagnostic",
        output_dir / ".FE2_on_FE3_sample_working.csv",
    )
    (output_dir / ".FE2_on_FE3_sample_working.csv").unlink(missing_ok=True)
    focal = run.table[run.table["term"].isin(FOCAL)].copy()
    return focal, restricted


def _sample_decomposition(
    focal: pd.DataFrame,
    restricted_results: pd.DataFrame,
) -> pd.DataFrame:
    full = focal[
        focal["analysis_layer"].eq("FE2")
        & focal["cov_type"].eq(PREFERRED_COVARIANCE)
    ].set_index("term")
    restricted = restricted_results[
        restricted_results["cov_type"].eq(PREFERRED_COVARIANCE)
    ].set_index("term")
    fe3 = focal[
        focal["analysis_layer"].eq("FE3")
        & focal["cov_type"].eq(PREFERRED_COVARIANCE)
    ].set_index("term")
    rows = []
    for feature in FOCAL:
        full_estimate = float(full.loc[feature, "estimate_pp"])
        restricted_estimate = float(restricted.loc[feature, "estimate_pp"])
        fe3_estimate = float(fe3.loc[feature, "estimate_pp"])
        rows.append(
            {
                "feature_name": feature,
                "FE2_full_estimate_pp": full_estimate,
                "FE2_restricted_to_FE3_sample_estimate_pp": restricted_estimate,
                "FE3_domain_FE_estimate_pp": fe3_estimate,
                "sample_composition_change_pp": restricted_estimate - full_estimate,
                "domain_FE_change_on_common_sample_pp": fe3_estimate - restricted_estimate,
                "total_FE2_to_FE3_change_pp": fe3_estimate - full_estimate,
            }
        )
    return pd.DataFrame(rows)


def influence_diagnostics(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    formula = formulas(data)["FE2"]
    prepared = data.dropna(
        subset=["cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL, CONTROL]
    ).copy()
    result = smf.ols(formula, data=prepared, missing="drop").fit()
    used = prepared.loc[list(result.model.data.row_labels)].copy()
    influence = result.get_influence()
    leverage = np.asarray(influence.hat_matrix_diag, dtype=float)
    # External studentization refits the high-dimensional prompt-FE design once
    # per row. Some leave-one-out designs become singular, so use the stable
    # full-fit internal studentization for screening.
    residual = np.asarray(result.resid, dtype=float)
    n_obs = len(used)
    n_parameters = len(result.params)
    leverage_denominator = np.clip(1 - leverage, np.finfo(float).eps, None)
    cooks = (
        residual**2
        / (n_parameters * float(result.mse_resid))
        * leverage
        / leverage_denominator**2
    )
    studentized = residual / np.sqrt(
        float(result.mse_resid)
        * leverage_denominator
    )
    cook_threshold = 4 / n_obs
    leverage_threshold = 2 * n_parameters / n_obs
    rows = used[
        [
            "prompt_id", "normalized_url", "source_root_domain", "cited", CONTROL,
            PAGE_TYPE, SOURCE_TYPE, *FOCAL,
        ]
    ].copy()
    rows["cooks_distance"] = cooks
    rows["leverage"] = leverage
    rows["studentized_residual"] = studentized
    rows["cook_flag_4_over_n"] = rows["cooks_distance"].gt(cook_threshold)
    rows["high_leverage_flag"] = rows["leverage"].gt(leverage_threshold)
    rows["absolute_studentized_residual"] = rows["studentized_residual"].abs()
    top_rows = rows.sort_values("cooks_distance", ascending=False).head(100)

    names = list(result.model.exog_names)
    focal_influence = []
    dfbeta_threshold = 2 / np.sqrt(n_obs)
    normalized_cov = np.asarray(result.normalized_cov_params, dtype=float)
    exog = np.asarray(result.model.exog, dtype=float)
    deletion_delta = (
        (exog @ normalized_cov) * (residual / leverage_denominator)[:, None]
    )
    beta_scale = np.sqrt(
        np.clip(float(result.mse_resid) * np.diag(normalized_cov), 0, None)
    )
    for feature in FOCAL:
        index = names.index(feature)
        values = np.abs(
            np.divide(
                deletion_delta[:, index],
                beta_scale[index],
                out=np.full(n_obs, np.nan),
                where=beta_scale[index] > 0,
            )
        )
        focal_influence.append(
            {
                "feature_name": feature,
                "max_abs_dfbeta": float(np.nanmax(values)),
                "p99_abs_dfbeta": float(np.nanquantile(values, 0.99)),
                "dfbeta_threshold_2_over_sqrt_n": dfbeta_threshold,
                "rows_above_dfbeta_threshold": int(np.sum(values > dfbeta_threshold)),
                "cook_threshold_4_over_n": cook_threshold,
                "rows_above_cook_threshold": int(np.sum(cooks > cook_threshold)),
                "leverage_threshold_2p_over_n": leverage_threshold,
                "rows_above_leverage_threshold": int(
                    np.sum(leverage > leverage_threshold)
                ),
            }
        )

    sensitivity_rows = []
    exclusions = {
        "remove_top_1pct_factual_numeric_density": pd.to_numeric(
            prepared["factual_numeric_density_score"], errors="coerce"
        ).le(prepared["factual_numeric_density_score"].quantile(0.99)),
        "remove_top_1pct_log2_word_count": pd.to_numeric(
            prepared["log2_word_count_plus1"], errors="coerce"
        ).le(prepared["log2_word_count_plus1"].quantile(0.99)),
    }
    cook_cutoff_99 = float(np.nanquantile(cooks, 0.99))
    exclusions["remove_top_1pct_cooks_distance"] = pd.Series(
        cooks <= cook_cutoff_99,
        index=used.index,
    ).reindex(prepared.index, fill_value=False)
    for label, mask in exclusions.items():
        subset = prepared.loc[mask].copy()
        run = run_model_and_save(
            formula,
            subset,
            f"FE2_{label}_diagnostic",
            output_dir / f".{label}_working.csv",
        )
        selected = run.table[run.table["term"].isin(FOCAL)].copy()
        selected.insert(0, "diagnostic_sample", label)
        selected["rows_removed"] = len(prepared) - len(subset)
        sensitivity_rows.append(selected)
        (output_dir / f".{label}_working.csv").unlink(missing_ok=True)
    return (
        top_rows,
        pd.DataFrame(focal_influence),
        pd.concat(sensitivity_rows, ignore_index=True),
    )


def _write_report(
    output_dir: Path,
    focal: pd.DataFrame,
    transitions: pd.DataFrame,
    sample_decomposition: pd.DataFrame,
    variation: pd.DataFrame,
    multiple_testing: pd.DataFrame,
    component_d0: pd.DataFrame,
    component_results: pd.DataFrame,
    component_correlation: pd.DataFrame,
    component_contingency: pd.DataFrame,
    factual_concentration: pd.DataFrame,
    table_concentration: pd.DataFrame,
    extraction_summary: pd.DataFrame,
    extraction_missingness: pd.DataFrame,
    influence_summary: pd.DataFrame,
    outlier_sensitivity: pd.DataFrame,
    focal_correlation: pd.DataFrame,
    focal_vif: pd.DataFrame,
) -> None:
    preferred = focal[focal["cov_type"].eq(PREFERRED_COVARIANCE)].copy()
    fe2 = preferred[preferred["analysis_layer"].eq("FE2")].set_index("term")
    primary_fdr = multiple_testing[
        multiple_testing["hypothesis_family"].eq("headline_FE2_four_core_features")
    ][["term", "p_value", "p_value_bh_fdr", "reject_at_fdr_0_05"]]
    component_fdr = multiple_testing[
        multiple_testing["hypothesis_family"].eq(
        "exploratory_FE1_five_writing_components"
        )
    ][["term", "estimate_pp", "conf_low_pp", "conf_high_pp", "p_value", "p_value_bh_fdr"]]
    covariance = focal.pivot_table(
        index=["analysis_layer", "term"],
        columns="cov_type",
        values="p_value",
        aggfunc="first",
    ).reset_index()
    covariance = covariance.rename(
        columns={
            "HC3": "p_HC3",
            "cluster_prompt_id": "p_prompt_cluster",
            "cluster_normalized_url": "p_URL_cluster",
            PREFERRED_COVARIANCE: "p_two_way",
        }
    )
    covariance = covariance[
        [
            "analysis_layer", "term", "p_HC3", "p_prompt_cluster",
            "p_URL_cluster", "p_two_way",
        ]
    ]
    prompt_variation = variation[
        variation["grouping"].eq("prompt_id") & variation["feature_name"].isin(FOCAL)
    ][
        [
            "feature_name", "groups_with_usable_variation",
            "rows_in_varying_groups", "unique_urls_in_varying_groups",
        ]
    ]
    domain_variation = variation[
        variation["grouping"].eq("source_root_domain_FE3_sample")
        & variation["feature_name"].isin(FOCAL)
    ][
        [
            "feature_name", "groups_with_usable_variation",
            "rows_in_varying_groups", "unique_urls_in_varying_groups",
        ]
    ]
    missing_total = int(
        extraction_missingness.loc[
            extraction_missingness["feature_name"].isin(FOCAL), "missing_rows"
        ].sum()
    )
    faq_overlap = component_contingency.to_dict("records")
    component_prevalence = component_d0[component_d0["state_label"].eq("detected")][
        ["feature_name", "n_rows", "row_share", "cited_rate", "unique_prompts"]
    ]
    component_prompt_variation = variation[
        variation["grouping"].eq("prompt_id")
        & variation["feature_name"].isin(WRITING_STRUCTURE_COMPONENTS)
    ][
        [
            "feature_name", "groups_with_usable_variation",
            "rows_in_varying_groups", "unique_urls_in_varying_groups",
        ]
    ]
    component_corr = component_correlation[
        ["feature_name", "has_faq_pattern"]
    ]
    focal_corr = focal_correlation[
        ["feature_name", *FOCAL]
    ]
    vif = focal_vif[focal_vif["term"].isin(FOCAL)][
        ["term", "vif", "condition_number"]
    ]
    outlier = outlier_sensitivity[
        outlier_sensitivity["cov_type"].eq(PREFERRED_COVARIANCE)
        & outlier_sensitivity["term"].isin(FOCAL)
    ][["diagnostic_sample", "term", "estimate_pp", "conf_low_pp", "conf_high_pp", "rows_removed"]]
    factual_page = factual_concentration[
        factual_concentration["grouping"].eq(PAGE_TYPE)
    ].sort_values("top_1pct_rows", ascending=False).head(5)[
        ["group_value", "n_rows", "mean_value", "top_1pct_rows"]
    ]
    factual_domain = factual_concentration[
        factual_concentration["grouping"].eq("source_root_domain")
    ].sort_values("top_1pct_rows", ascending=False).head(5)[
        ["group_value", "n_rows", "mean_value", "top_1pct_rows"]
    ]
    table_domain = table_concentration[
        table_concentration["grouping"].eq("source_root_domain")
    ].sort_values("detected_rows", ascending=False).head(5)[
        [
            "group_value", "n_rows", "detected_rows", "detected_rate",
            "share_of_all_detected_rows",
        ]
    ]
    strength = extraction_summary[
        [
            "content_strength", "n_rows", "row_share", "cited_rate",
            "html_available_rate",
        ]
    ]

    def markdown(frame: pd.DataFrame) -> str:
        clean = frame.copy()
        for column in clean.select_dtypes(include=[float]).columns:
            clean[column] = clean[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
        clean = clean.fillna("").astype(str)
        headers = [str(column) for column in clean.columns]
        rows = [headers, ["---"] * len(headers), *clean.values.tolist()]
        return "\n".join(
            "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
            for row in rows
        )

    report = f"""# Updated D0-FE4 Econometric Diagnostic Report

## Inference

The frontend displays `{PREFERRED_COVARIANCE}` inference. The model is fit once per specification, then the same coefficients are paired with HC3, prompt-clustered, URL-clustered, and two-way prompt/URL-clustered covariance estimates. Covariance choice changes uncertainty, not coefficients. The two-way estimator is retained because source appearances are dependent within both prompts and repeated URLs; it was not selected from p-values.

The two-way covariance produced finite standard errors for every focal coefficient, although some nuisance fixed-effect diagonals were negative and are explicitly left unavailable.

P-value comparison (coefficients are identical across columns):

{markdown(covariance)}

HC3 is consistently the least conservative estimator in FE2. URL clustering increases uncertainty more than prompt clustering, and two-way clustering is widest for all four FE2 focal terms. This pattern supports retaining the pre-specified two-way estimator; it is not a reason to choose whichever column crosses 0.05.

## Headline FE2 results

{markdown(fe2.reset_index()[["term", "estimate_pp", "conf_low_pp", "conf_high_pp", "p_value"]])}

The coefficients are percentage-point associations per one-unit feature increase, conditional on the FE2 controls and Prompt Fixed Effects. They are not causal effects.

## Coefficient movement

{markdown(transitions)}

FE1-to-FE2 movement is modest relative to uncertainty, consistent with the low focal-feature correlations and VIF values. FE2-to-FE3 movement combines a sample restriction and Domain Fixed Effects; the decomposition below separates them.

{markdown(sample_decomposition)}

For page length, sample restriction makes the estimate more negative by 0.44 pp, then Domain Fixed Effects move it 1.07 pp toward zero. For verified tables, both the restricted sample (-1.57 pp) and Domain Fixed Effects (-1.35 pp) contribute to attenuation. Factual density changes almost entirely because of Domain Fixed Effects, while the writing-score sign reversal is also almost entirely a within-domain adjustment rather than sample composition.

FE4 is a separate branch from FE2. Its larger table coefficient is consistent with suppression by taxonomy composition, while its weaker factual-density coefficient indicates that Gemini taxonomy absorbs some of the same page-composition signal. Because that taxonomy was inferred partly from scraped page content, FE4 may be an over-control sensitivity and is not preferred over FE2.

## Identifying variation

Prompt-FE support:

{markdown(prompt_variation)}

Within-domain support on the FE3 sample:

{markdown(domain_variation)}

Verified tables have the weakest FE3 identifying support: only 50 supported domains vary internally, although those domains contain 2,408 rows. The writing score varies within 136 supported domains. These constraints explain why FE3 is less precise and why template-level signals attenuate.

## Dependence among features

Core-feature correlations:

{markdown(focal_corr)}

Core-feature VIF:

{markdown(vif)}

All focal VIF values are below 1.55. The reported raw condition number is 140.39, but that scale-sensitive value includes the intercept and content-strength dummies; together with low focal correlations and VIF, it does not indicate severe focal multicollinearity.

## Multiple testing

The project-specified Benjamini-Hochberg FDR correction is applied separately to:

1. Four pre-registered headline FE2 focal hypotheses.
2. Five expanded exploratory writing-component FE1 hypotheses.

Robustness specifications are repeated views of the same focal hypotheses and are not counted as independent hypothesis-family members.

Headline family:

{markdown(primary_fdr)}

Exploratory component family:

{markdown(component_fdr)}

## Writing-score decomposition

Component prevalence:

{markdown(component_prevalence)}

Prompt-FE identifying support:

{markdown(component_prompt_variation)}

FAQ/Q&A joint states:

```json
{json.dumps(faq_overlap, indent=2, default=str)}
```

Selected component correlations:

{markdown(component_corr)}

`has_question_answer_structure` remains identical to `has_faq_pattern` in the
source evidence (560 joint-positive and 4,704 joint-negative rows), but it is
excluded from `writing_structure_score_v3` and from the active component
regressions. Ordered lists are detected in only 347 rows and vary within 181
prompts, so their interval is wide. The five active components are evaluated
as one exploratory BH-FDR family.

## Concentration

Factual-density top tail by Gemini page type:

{markdown(factual_page)}

Factual-density top tail by domain:

{markdown(factual_domain)}

Verified-table concentration by domain:

{markdown(table_domain)}

The 73 top-tail factual-density rows are concentrated in directory/listing and commercial/review structures; `bkkcondos.com` and `propertyhub.in.th` alone account for 39. Verified tables are also template-concentrated: the largest domain contributes 13.35% of all detections, and several domains detect tables on essentially every row. These patterns support the domain/template-confounding interpretation.

## Extraction and influence

Total missing focal-feature cells in the estimation sample: `{missing_total}`. All FE1/FE2/FE4 rows are complete for the four focal features. FE2 already adjusts for `content_strength`; this is extraction quality, not writing quality.

{markdown(strength)}

This is a selected measurable-content sample: HTML availability is 100% by construction and 93.62% of rows are classified strong. Therefore missingness within this estimation table does not drive the fitted results, but this audit cannot establish that extraction selection is ignorable relative to all surfaced sources.

Influence summary:

{markdown(influence_summary)}

Two-way-clustered outlier diagnostics:

{markdown(outlier)}

Factual density is stable after removing its top 1% (-1.14 pp) and after removing the top 1% by Cook's distance (-1.25 pp). Page length is not tail-stable: removing the longest 1% changes its estimate from -0.82 pp to -3.26 pp. Verified tables become somewhat larger after the long-page and Cook exclusions. These are diagnostics, not additional model layers or a basis for selecting a preferred estimate.

Outlier-exclusion estimates are diagnostic only and do not create another model layer.

## Pattern diagnosis

- **Page length:** negative but imprecise throughout; the common-sample decomposition determines how much attenuation is sample restriction versus Domain Fixed Effects. The evidence does not establish a meaningful length association.
- **Verified HTML table:** positive but specification-sensitive. Attenuation under FE3 is consistent with domain/template concentration or weak within-domain variation. Enlargement under FE4 is consistent with suppression by taxonomy composition, but FE4 may over-control because Gemini taxonomy used page content.
- **Factual/numeric density:** the most stable negative association. It survives the four registered specifications, but concentration and outlier tables must be consulted before interpreting it. The score may proxy for listings, pricing templates, repeated numeric clutter, or page composition rather than useful factuality.
- **Writing structure v2:** imprecise and sign-changing under FE3. The most plausible explanations are template/domain confounding, equal-weight component aggregation, FAQ/Q&A overlap, and cancellation across components. It does not demonstrate that writing structure is irrelevant.

## Next required analysis

Before interpreting the writing composite, remove or explicitly justify the exact FAQ/Q&A duplication and re-estimate the governed D0-FE4 sequence. Before interpreting factual density, manually inspect top-tail pages to distinguish useful evidence from listing/pricing clutter and duplicated extraction. Page length requires a pre-specified functional-form or tail treatment because its estimate is sensitive to the longest 1%. Do not convert these observational associations among surfaced sources into rewriting recommendations or causal claims.
"""
    (output_dir / "updated_D0_FE4_diagnostic_report.md").write_text(
        report,
        encoding="utf-8",
    )


def run_diagnostics(
    repo: Path,
    output_root: Path,
    diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output_root = Path(output_root).resolve()
    diagnostic_dir = Path(
        diagnostic_dir or output_root / "diagnostics" / "post_estimation_20260724"
    ).resolve()
    diagnostic_dir.mkdir(parents=True, exist_ok=True)

    data, _ = build_model_ready(repo)
    complete = data.dropna(
        subset=["cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL, CONTROL]
    ).copy()
    models = _read_model_tables(output_root)
    focal = _focal_rows(models)
    covariance_path = diagnostic_dir / "focal_covariance_estimator_comparison.csv"
    focal.to_csv(covariance_path, index=False)

    transitions = coefficient_transitions(focal)
    transitions.to_csv(diagnostic_dir / "focal_coefficient_transitions.csv", index=False)
    variation = variation_diagnostics(complete)
    variation.to_csv(diagnostic_dir / "focal_and_component_variation_support.csv", index=False)

    component_d0 = _component_descriptives(complete)
    component_d0.to_csv(diagnostic_dir / "writing_component_D0.csv", index=False)
    component_corr, faq_contingency = _component_overlap(complete)
    component_corr.to_csv(diagnostic_dir / "writing_component_correlation.csv", index=False)
    faq_contingency.to_csv(diagnostic_dir / "faq_qa_overlap_contingency.csv", index=False)
    component_results = _run_component_fe1(complete, diagnostic_dir)
    component_results.to_csv(
        diagnostic_dir / "writing_component_FE1_diagnostics.csv",
        index=False,
    )

    multiple_testing = _multiple_testing(focal, component_results)
    multiple_testing.to_csv(
        diagnostic_dir / "multiple_testing_bh_fdr.csv",
        index=False,
    )

    factual_concentration, table_concentration = concentration_diagnostics(complete)
    factual_concentration.to_csv(
        diagnostic_dir / "factual_density_concentration.csv",
        index=False,
    )
    table_concentration.to_csv(
        diagnostic_dir / "verified_table_concentration.csv",
        index=False,
    )

    extraction_summary, extraction_missingness = extraction_diagnostics(complete)
    extraction_summary.to_csv(
        diagnostic_dir / "extraction_strength_diagnostics.csv",
        index=False,
    )
    extraction_missingness.to_csv(
        diagnostic_dir / "estimation_sample_missingness.csv",
        index=False,
    )

    restricted_results, restricted = _restricted_sample_diagnostic(
        complete,
        diagnostic_dir,
    )
    restricted_results.to_csv(
        diagnostic_dir / "FE2_on_FE3_sample_diagnostic.csv",
        index=False,
    )
    sample_decomposition = _sample_decomposition(focal, restricted_results)
    sample_decomposition.to_csv(
        diagnostic_dir / "FE2_FE3_sample_domain_decomposition.csv",
        index=False,
    )

    top_influence, influence_summary, outlier_sensitivity = influence_diagnostics(
        complete,
        diagnostic_dir,
    )
    top_influence.to_csv(diagnostic_dir / "FE2_top_influence_rows.csv", index=False)
    influence_summary.to_csv(
        diagnostic_dir / "FE2_influence_summary.csv",
        index=False,
    )
    outlier_sensitivity.to_csv(
        diagnostic_dir / "FE2_outlier_sensitivity_diagnostics.csv",
        index=False,
    )

    existing_vif = pd.read_csv(
        output_root / "tables" / "selected_feature_vif_condition_number.csv",
        low_memory=False,
    )
    existing_corr = pd.read_csv(
        output_root / "tables" / "selected_feature_correlation_matrix.csv",
        low_memory=False,
    )
    existing_vif.to_csv(diagnostic_dir / "focal_vif_condition_number.csv", index=False)
    existing_corr.to_csv(diagnostic_dir / "focal_correlation_matrix.csv", index=False)

    _write_report(
        diagnostic_dir,
        focal,
        transitions,
        sample_decomposition,
        variation,
        multiple_testing,
        component_d0,
        component_results,
        component_corr,
        faq_contingency,
        factual_concentration,
        table_concentration,
        extraction_summary,
        extraction_missingness,
        influence_summary,
        outlier_sensitivity,
        existing_corr,
        existing_vif,
    )
    manifest = {
        "status": "completed",
        "output_dir": str(diagnostic_dir),
        "preferred_covariance": PREFERRED_COVARIANCE,
        "headline_hypothesis_family": list(FOCAL),
        "exploratory_component_family": list(WRITING_STRUCTURE_COMPONENTS),
        "n_rows": len(complete),
        "n_prompts": int(complete["prompt_id"].nunique()),
        "n_urls": int(complete["normalized_url"].nunique()),
        "n_domains": int(complete["source_root_domain"].nunique()),
        "FE3_restricted_rows": len(restricted),
    }
    (diagnostic_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
