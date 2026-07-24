"""Content-feature econometrics for the area-condo nonbranded citation audit."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as scipy_stats
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from .gemini_taxonomy_features import (
    GEMINI_PAGE_FAMILY,
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    GEMINI_TAXONOMY_VERSION,
    attach_gemini_taxonomy,
)


REQUIRED_COLUMNS = (
    "cited",
    "prompt_id",
    "normalized_url",
    "log2_word_count_plus1",
    "has_table",
    "heading_count_group",
    "link_count_group",
    "content_strength",
)
SENSITIVITY_COLUMNS = (
    "source_root_domain",
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    "page_type_url_seed_general_collapsed",
    "low_link_count",
    "log2_word_count_plus1_winsorized_p99",
)
FORBIDDEN_FORMULA_TOKENS = (
    "answer_similarity",
    "page_answer_similarity",
    "max_chunk_answer_similarity",
    "answer_overlap",
    "answer_like_text",
    "brand_appeared_in_answer",
    "cited_label",
    "source_group",
    "source_origin",
    "source_position",
    "observed_rank",
    "domain_citation_rate",
    "domain_citation_rate_loo",
)

HEADING = "C(heading_count_group, Treatment(reference='0-1'))"
LINK = "C(link_count_group, Treatment(reference='9+'))"
STRENGTH = "C(content_strength, Treatment(reference='strong'))"
GEMINI_PAGE_FAMILY_TERM = (
    f"C({GEMINI_PAGE_FAMILY_COLLAPSED}, Treatment(reference='informational_content'))"
)
GEMINI_SOURCE_TYPE_TERM = (
    f"C({GEMINI_SOURCE_TYPE_COLLAPSED}, Treatment(reference='official_company_or_brand'))"
)
GEMINI_TAXONOMY_TERMS = f"{GEMINI_PAGE_FAMILY_TERM} + {GEMINI_SOURCE_TYPE_TERM}"
RULE_V2_PAGE_SEED = "C(page_type_url_seed_general_collapsed, Treatment(reference='unknown'))"
PROMPT_FE = "C(prompt_id)"
DOMAIN_FE = "C(source_root_domain)"
M2_FORMULA = (
    f"cited ~ log2_word_count_plus1 + has_table + {HEADING} + {LINK} "
    f"+ {STRENGTH} + {PROMPT_FE}"
)

PREFERRED_COVARIANCE_ORDER = (
    "two_way_cluster_prompt_url",
    "cluster_prompt_id",
    "cluster_normalized_url",
    "HC3",
    "nonrobust",
)


@dataclass
class ModelRun:
    """One fitted base model plus its reported covariance variants."""

    model_id: str
    formula: str
    data: pd.DataFrame
    base_result: Any
    table: pd.DataFrame
    warnings: list[str]


def _clean_text(series: pd.Series, fallback: str = "unknown") -> pd.Series:
    return series.fillna(fallback).astype(str).str.strip().replace("", fallback)


def _prepare_model_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().reset_index(drop=True)
    data["cited"] = pd.to_numeric(data["cited"], errors="coerce")
    for column in ("has_table", "low_link_count", "word_count_top_1pct", "link_count_top_1pct"):
        if column in data:
            data[column] = (
                data[column]
                .replace(
                    {
                        True: 1,
                        False: 0,
                        "True": 1,
                        "False": 0,
                        "true": 1,
                        "false": 0,
                    }
                )
                .pipe(pd.to_numeric, errors="coerce")
            )
    for column in (
        "word_count",
        "link_count",
        "log2_word_count_plus1",
        "log2_word_count_plus1_winsorized_p99",
    ):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in (
        "prompt_id",
        "normalized_url",
        "source_root_domain",
        "heading_count_group",
        "link_count_group",
        "content_strength",
        GEMINI_PAGE_FAMILY,
        GEMINI_PAGE_FAMILY_COLLAPSED,
        GEMINI_SOURCE_TYPE,
        GEMINI_SOURCE_TYPE_COLLAPSED,
        "page_type_url_seed_general_collapsed",
        "intent",
        "area_tag",
    ):
        if column in data:
            data[column] = _clean_text(data[column])
    return data


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _formula_rhs(formula: str) -> str:
    return formula.split("~", 1)[1] if "~" in formula else formula


def forbidden_formula_matches(formula: str) -> list[str]:
    """Return forbidden predictor tokens found on the right-hand side."""
    rhs = _formula_rhs(formula).casefold()
    return sorted(token for token in FORBIDDEN_FORMULA_TOKENS if token in rhs)


def _assert_formula_safe(formula: str) -> None:
    matches = forbidden_formula_matches(formula)
    if matches:
        raise ValueError(f"Leakage guardrail failed for formula; forbidden tokens: {', '.join(matches)}")


def _row_counts(data: pd.DataFrame) -> dict[str, int]:
    return {
        "n_obs": int(len(data)),
        "n_prompts": int(data["prompt_id"].nunique()) if "prompt_id" in data else 0,
        "n_urls": int(data["normalized_url"].nunique()) if "normalized_url" in data else 0,
        "n_domains": int(data["source_root_domain"].nunique()) if "source_root_domain" in data else 0,
    }


def _used_rows(base_result: Any, data: pd.DataFrame) -> pd.DataFrame:
    labels = list(base_result.model.data.row_labels)
    return data.loc[labels].copy()


def _covariance_payload(base_result: Any, data: pd.DataFrame, cov_type: str) -> dict[str, Any]:
    used = _used_rows(base_result, data)
    names = list(base_result.model.exog_names)
    params = np.asarray(base_result.params, dtype=float)
    notes = ""
    if cov_type == "nonrobust":
        covariance = np.asarray(base_result.cov_params(), dtype=float)
    elif cov_type == "HC3":
        covariance = np.asarray(base_result.get_robustcov_results(cov_type="HC3").cov_params(), dtype=float)
    elif cov_type == "cluster_prompt_id":
        if used["prompt_id"].nunique() < 2:
            raise ValueError("Prompt-cluster covariance needs at least two prompt clusters.")
        robust = base_result.get_robustcov_results(
            cov_type="cluster",
            groups=pd.factorize(used["prompt_id"])[0],
            use_correction=True,
        )
        covariance = np.asarray(robust.cov_params(), dtype=float)
    elif cov_type == "cluster_normalized_url":
        if used["normalized_url"].nunique() < 2:
            raise ValueError("URL-cluster covariance needs at least two URL clusters.")
        robust = base_result.get_robustcov_results(
            cov_type="cluster",
            groups=pd.factorize(used["normalized_url"])[0],
            use_correction=True,
        )
        covariance = np.asarray(robust.cov_params(), dtype=float)
    elif cov_type == "two_way_cluster_prompt_url":
        prompt_codes = pd.factorize(used["prompt_id"])[0]
        url_codes = pd.factorize(used["normalized_url"])[0]
        if len(np.unique(prompt_codes)) < 2 or len(np.unique(url_codes)) < 2:
            raise ValueError("Two-way covariance needs at least two clusters in both dimensions.")
        covariance = np.asarray(
            cov_cluster_2groups(base_result, prompt_codes, url_codes, use_correction=True)[0],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported covariance type: {cov_type}")

    variances = np.diag(covariance)
    negative_variance_count = int(np.sum(variances < -1e-10))
    if negative_variance_count:
        notes = (
            f"cluster covariance had negative diagonal values for {negative_variance_count} terms; "
            "their SEs are unavailable"
        )
    standard_errors = np.sqrt(np.clip(variances, 0, None))
    standard_errors[variances < -1e-10] = np.nan
    statistics = np.divide(
        params,
        standard_errors,
        out=np.full_like(params, np.nan),
        where=np.isfinite(standard_errors) & (standard_errors > 0),
    )
    p_values = 2 * scipy_stats.norm.sf(np.abs(statistics))
    conf_low = params - 1.959963984540054 * standard_errors
    conf_high = params + 1.959963984540054 * standard_errors
    return {
        "names": names,
        "params": params,
        "std_error": standard_errors,
        "p_value": p_values,
        "conf_low": conf_low,
        "conf_high": conf_high,
        "cov_type": cov_type,
        "notes": notes,
    }


def tidy_lpm_result(
    result: Any,
    model_name: str,
    *,
    formula: str | None = None,
    data: pd.DataFrame | None = None,
    cov_type: str = "nonrobust",
    notes: str = "",
) -> pd.DataFrame:
    """Convert a fitted statsmodels result to the shared coefficient schema."""
    if data is None:
        data = result.model.data.frame
    formula = formula or getattr(result.model, "formula", "")
    payload = _covariance_payload(result, data, cov_type)
    counts = _row_counts(_used_rows(result, data))
    combined_notes = "; ".join(part for part in (notes, payload["notes"]) if part)
    rank = int(getattr(result.model, "rank", 0) or np.linalg.matrix_rank(result.model.exog))
    columns = int(result.model.exog.shape[1])
    if rank < columns:
        rank_note = f"rank_deficient: rank={rank}, columns={columns}"
        combined_notes = "; ".join(part for part in (combined_notes, rank_note) if part)
    rows = []
    for index, term in enumerate(payload["names"]):
        rows.append(
            {
                "model_id": model_name,
                "formula": formula,
                "term": term,
                "estimate": payload["params"][index],
                "estimate_pp": payload["params"][index] * 100,
                "std_error": payload["std_error"][index],
                "conf_low": payload["conf_low"][index],
                "conf_high": payload["conf_high"][index],
                "conf_low_pp": payload["conf_low"][index] * 100,
                "conf_high_pp": payload["conf_high"][index] * 100,
                "p_value": payload["p_value"][index],
                **counts,
                "r_squared": float(getattr(result, "rsquared", np.nan)),
                "cov_type": payload["cov_type"],
                "notes": combined_notes,
            }
        )
    return pd.DataFrame(rows)


def add_cluster_counts(table: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """Attach the model sample and cluster counts to an existing table."""
    out = table.copy()
    for key, value in _row_counts(data).items():
        out[key] = value
    return out


def _fit_ols_with_qr_fallback(model: Any) -> tuple[Any, str]:
    try:
        return model.fit(), ""
    except np.linalg.LinAlgError:
        q_matrix, r_matrix = np.linalg.qr(model.wexog, mode="reduced")
        diagonal = np.abs(np.diag(r_matrix))
        tolerance = max(r_matrix.shape) * np.finfo(float).eps * diagonal.max()
        if np.any(diagonal <= tolerance):
            raise
        model.pinv_wexog = np.linalg.solve(r_matrix, q_matrix.T)
        model.normalized_cov_params = model.pinv_wexog @ model.pinv_wexog.T
        model.rank = r_matrix.shape[1]
        model.wexog_singular_values = diagonal
        return model.fit(method="pinv"), "default SVD fit failed; recovered with full-rank QR"


def fit_lpm(formula: str, data: pd.DataFrame, model_name: str, cov_type: str = "HC3") -> pd.DataFrame:
    """Fit one LPM and return one covariance-specific tidy table."""
    _assert_formula_safe(formula)
    prepared = _prepare_model_data(data)
    model = smf.ols(formula, data=prepared, missing="drop")
    base, fit_note = _fit_ols_with_qr_fallback(model)
    return tidy_lpm_result(
        base,
        model_name,
        formula=formula,
        data=prepared,
        cov_type=cov_type,
        notes=fit_note,
    )


def run_model_and_save(
    formula: str,
    data: pd.DataFrame,
    model_name: str,
    output_path: Path,
    *,
    cov_types: Iterable[str] = (
        "HC3",
        "cluster_prompt_id",
        "cluster_normalized_url",
        "two_way_cluster_prompt_url",
    ),
    notes: str = "",
) -> ModelRun:
    """Fit once, calculate requested covariance variants, and save incrementally."""
    _assert_formula_safe(formula)
    prepared = _prepare_model_data(data)
    model = smf.ols(formula, data=prepared, missing="drop")
    base, fit_note = _fit_ols_with_qr_fallback(model)
    tables: list[pd.DataFrame] = []
    model_warnings: list[str] = []
    if fit_note:
        model_warnings.append(f"{model_name}: {fit_note}.")
    for cov_type in cov_types:
        try:
            tables.append(
                tidy_lpm_result(
                    base,
                    model_name,
                    formula=formula,
                    data=prepared,
                    cov_type=cov_type,
                    notes="; ".join(part for part in (notes, fit_note) if part),
                )
            )
        except Exception as exc:
            message = f"{model_name}: {cov_type} unavailable ({type(exc).__name__}: {exc})"
            model_warnings.append(message)
    if not tables:
        raise RuntimeError(f"No covariance result could be produced for {model_name}.")
    table = pd.concat(tables, ignore_index=True)
    affected = table[
        table["notes"].str.contains("negative diagonal", na=False)
        & table["std_error"].isna()
    ]
    for cov_type, group in affected.groupby("cov_type", sort=False):
        focal_affected = int(group["term"].map(_is_focal_term).sum())
        total_terms = int(table[table["cov_type"].eq(cov_type)]["term"].nunique())
        model_warnings.append(
            f"{model_name}: {cov_type} covariance yielded negative diagonal variances for "
            f"{len(group)}/{total_terms} terms ({focal_affected} focal content terms); those SEs are "
            "unavailable, while HC3 and one-way clustered variants remain available."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return ModelRun(model_name, formula, prepared, base, table, model_warnings)


def _preferred_covariance_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    rank = {name: index for index, name in enumerate(PREFERRED_COVARIANCE_ORDER)}
    work = table.copy()
    work["_cov_rank"] = work["cov_type"].map(rank).fillna(len(rank))
    keys = [column for column in ("model_id", "term") if column in work]
    return (
        work.sort_values("_cov_rank", kind="stable")
        .drop_duplicates(keys, keep="first")
        .drop(columns="_cov_rank")
        .reset_index(drop=True)
    )


def _is_focal_term(term: str) -> bool:
    return term != "Intercept" and any(
        token in term
        for token in (
            "log2_word_count_plus1",
            "has_table",
            "heading_count_group",
            "link_count_group",
            "content_strength",
        )
    )


def _pretty_term(term: str) -> str:
    replacements = {
        "log2_word_count_plus1_winsorized_p99": "Page length (doubling, winsorized)",
        "log2_word_count_plus1": "Page length (doubling)",
        "has_table": "Has table",
        "low_link_count": "Low link count (<9)",
    }
    if term in replacements:
        return replacements[term]
    match = re.search(r"C\(([^,]+).*?\)\[T\.(.*?)\]$", term)
    if match:
        feature, category = match.groups()
        labels = {
            "heading_count_group": "Heading count",
            "link_count_group": "Link count",
            "content_strength": "Content strength",
            GEMINI_PAGE_FAMILY_COLLAPSED: "Gemini page-function family",
            GEMINI_SOURCE_TYPE_COLLAPSED: "Gemini source/site type",
            "page_type_url_seed_general_collapsed": "URL-seed page type",
        }
        return f"{labels.get(feature, feature)}: {category}"
    return term


def _write_plotly(fig: go.Figure, html_path: Path) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    fig.write_json(html_path.with_suffix(".plotly.json"))


def make_coefficient_forest(
    table: pd.DataFrame,
    path: Path,
    *,
    title: str,
    preferred_only: bool = True,
) -> None:
    """Write a Plotly coefficient forest for focal content terms."""
    work = _preferred_covariance_rows(table) if preferred_only else table.copy()
    work = work[work["term"].map(_is_focal_term)].copy()
    if work.empty:
        fig = go.Figure().add_annotation(text="No focal coefficients available", showarrow=False)
        _write_plotly(fig, path)
        return
    work["label"] = work["term"].map(_pretty_term)
    if work["model_id"].nunique() > 1:
        work["label"] = work["model_id"].astype(str) + " | " + work["label"]
    work = work.sort_values(["model_id", "estimate_pp"], kind="stable")
    fig = go.Figure()
    for cov_type, group in work.groupby("cov_type", sort=False):
        fig.add_trace(
            go.Scatter(
                x=group["estimate_pp"],
                y=group["label"],
                mode="markers",
                name=cov_type,
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": group["conf_high_pp"] - group["estimate_pp"],
                    "arrayminus": group["estimate_pp"] - group["conf_low_pp"],
                },
                customdata=np.column_stack(
                    [
                        group["conf_low_pp"],
                        group["conf_high_pp"],
                        group["p_value"],
                        group["n_obs"],
                        group["n_prompts"],
                        group["n_urls"],
                    ]
                ),
                hovertemplate=(
                    "%{y}<br>Estimate: %{x:.2f} pp"
                    "<br>95% CI: %{customdata[0]:.2f} to %{customdata[1]:.2f} pp"
                    "<br>p=%{customdata[2]:.4f}<br>n=%{customdata[3]:.0f}"
                    "<br>prompts=%{customdata[4]:.0f}; URLs=%{customdata[5]:.0f}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line_dash="dash", line_color="#606b73")
    fig.update_layout(
        title=title,
        xaxis_title="Association with citation probability (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=max(480, 28 * len(work) + 150),
        margin={"l": 250, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(fig, path)


def make_prediction_contrast(
    base_result: Any,
    data: pd.DataFrame,
    *,
    contrast_name: str,
    baseline_condition: str,
    comparison_condition: str,
    baseline_data: pd.DataFrame,
    comparison_data: pd.DataFrame,
    model_id: str = "M2",
    term: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Calculate an observed-covariate model-implied probability contrast."""
    baseline = float(np.mean(base_result.predict(baseline_data)))
    comparison = float(np.mean(base_result.predict(comparison_data)))
    return {
        "term": term,
        "contrast_name": contrast_name,
        "baseline_condition": baseline_condition,
        "comparison_condition": comparison_condition,
        "predicted_probability_baseline": baseline,
        "predicted_probability_comparison": comparison,
        "difference_pp": (comparison - baseline) * 100,
        "model_id": model_id,
        "notes": notes,
    }


def _categorical_cited_rate(data: pd.DataFrame, feature: str, order: list[str] | None = None) -> pd.DataFrame:
    work = data[[feature, "cited"]].copy()
    work[feature] = _clean_text(work[feature])
    rows = []
    for category, group in work.groupby(feature, dropna=False):
        n_rows = int(len(group))
        cited_rows = int(group["cited"].sum())
        low, high = _wilson_interval(cited_rows, n_rows)
        rows.append(
            {
                "feature": feature,
                "category": category,
                "n_rows": n_rows,
                "cited_rows": cited_rows,
                "more_only_rows": n_rows - cited_rows,
                "cited_rate": cited_rows / n_rows,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "category_share": n_rows / len(work),
            }
        )
    out = pd.DataFrame(rows)
    if order:
        rank = {value: index for index, value in enumerate(order)}
        out["_rank"] = out["category"].map(rank).fillna(len(rank))
        out = out.sort_values(["_rank", "category"], kind="stable").drop(columns="_rank")
    else:
        out = out.sort_values(["n_rows", "category"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)


def _build_m0(data: pd.DataFrame, table_dir: Path, interactive_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_cited = int(data["cited"].sum())
    overall_low, overall_high = _wilson_interval(overall_cited, len(data))
    summaries = [
        pd.DataFrame(
            [
                {
                    "feature": "overall",
                    "category": "all_measurable_rows",
                    "n_rows": len(data),
                    "cited_rows": overall_cited,
                    "more_only_rows": len(data) - overall_cited,
                    "cited_rate": data["cited"].mean(),
                    "wilson_ci_low": overall_low,
                    "wilson_ci_high": overall_high,
                    "category_share": 1.0,
                }
            ]
        )
    ]
    definitions = [
        ("content_strength", ["weak", "medium", "strong"]),
        ("has_table", ["0", "1"]),
        ("heading_count_group", ["0-1", "2-6", "7-12", "13+"]),
        ("link_count_group", ["0-3", "4-8", "9+"]),
    ]
    plotting_tables: dict[str, pd.DataFrame] = {}
    for feature, order in definitions:
        plot_data = data.copy()
        plot_data[feature] = plot_data[feature].astype(str)
        table = _categorical_cited_rate(plot_data, feature, order)
        summaries.append(table)
        plotting_tables[feature] = table

    if "low_link_count" in data:
        low_link = data.copy()
        low_link["low_link_count"] = low_link["low_link_count"].astype("Int64").astype(str)
        summaries.append(_categorical_cited_rate(low_link, "low_link_count", ["0", "1"]))

    word_bins = pd.qcut(
        data["word_count"],
        q=4,
        labels=["Q1 shortest", "Q2", "Q3", "Q4 longest"],
        duplicates="drop",
    )
    word_data = data.assign(word_count_quartile=word_bins.astype(str))
    word_table = _categorical_cited_rate(
        word_data,
        "word_count_quartile",
        ["Q1 shortest", "Q2", "Q3", "Q4 longest"],
    )
    summaries.append(word_table)
    plotting_tables["word_count_quartile"] = word_table

    for taxonomy_feature in (GEMINI_PAGE_FAMILY_COLLAPSED, GEMINI_SOURCE_TYPE_COLLAPSED):
        if taxonomy_feature in data:
            summaries.append(_categorical_cited_rate(data, taxonomy_feature))
    if "source_root_domain" in data:
        top_domains = data["source_root_domain"].value_counts().head(20).index
        summaries.append(
            _categorical_cited_rate(
                data[data["source_root_domain"].isin(top_domains)],
                "source_root_domain",
            )
        )
    m0 = pd.concat(summaries, ignore_index=True)
    m0.to_csv(table_dir / "M0_descriptive_cited_rate_summary.csv", index=False)

    distribution_rows: list[dict[str, Any]] = []
    for feature in (
        "word_count",
        "log2_word_count_plus1",
        "heading_count",
        "link_count",
        "table_count",
    ):
        if feature not in data:
            continue
        numeric = pd.to_numeric(data[feature], errors="coerce")
        distribution_rows.append(
            {
                "summary_type": "numeric",
                "feature": feature,
                "category": "",
                "n_rows": int(numeric.notna().sum()),
                "share": numeric.notna().mean(),
                "mean": numeric.mean(),
                "std": numeric.std(),
                "minimum": numeric.min(),
                "p25": numeric.quantile(0.25),
                "median": numeric.median(),
                "p75": numeric.quantile(0.75),
                "p95": numeric.quantile(0.95),
                "p99": numeric.quantile(0.99),
                "maximum": numeric.max(),
            }
        )
    for feature in (
        "content_strength",
        "heading_count_group",
        "link_count_group",
        GEMINI_PAGE_FAMILY_COLLAPSED,
        GEMINI_SOURCE_TYPE_COLLAPSED,
    ):
        if feature not in data:
            continue
        counts = data[feature].value_counts(dropna=False)
        for category, n_rows in counts.items():
            distribution_rows.append(
                {
                    "summary_type": "categorical",
                    "feature": feature,
                    "category": category,
                    "n_rows": int(n_rows),
                    "share": n_rows / len(data),
                }
            )
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(table_dir / "M0_feature_distribution_summary.csv", index=False)

    for feature, filename, title in (
        ("has_table", "M0_cited_rate_by_has_table.html", "Unadjusted cited rate by table presence"),
        (
            "heading_count_group",
            "M0_cited_rate_by_heading_count_group.html",
            "Unadjusted cited rate by heading-count group",
        ),
        (
            "link_count_group",
            "M0_cited_rate_by_link_count_group.html",
            "Unadjusted cited rate by link-count group",
        ),
    ):
        plot = plotting_tables[feature].copy()
        plot["cited_rate_pct"] = plot["cited_rate"] * 100
        plot["ci_low_pct"] = plot["wilson_ci_low"] * 100
        plot["ci_high_pct"] = plot["wilson_ci_high"] * 100
        fig = go.Figure(
            go.Bar(
                x=plot["category"],
                y=plot["cited_rate_pct"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": plot["ci_high_pct"] - plot["cited_rate_pct"],
                    "arrayminus": plot["cited_rate_pct"] - plot["ci_low_pct"],
                },
                text=[f"n={value:,}" for value in plot["n_rows"]],
                textposition="outside",
                marker_color="#277da1",
                hovertemplate="%{x}<br>Cited rate=%{y:.1f}%<br>%{text}<extra></extra>",
            )
        )
        fig.add_hline(y=data["cited"].mean() * 100, line_dash="dash", line_color="#6c757d")
        fig.update_layout(
            title=title,
            xaxis_title="",
            yaxis_title="Cited rate (%)",
            template="plotly_white",
            height=480,
        )
        _write_plotly(fig, interactive_dir / filename)

    histogram = px.histogram(
        data,
        x="log2_word_count_plus1",
        color=data["cited"].map({0: "More-only", 1: "Cited"}),
        barmode="overlay",
        opacity=0.62,
        nbins=45,
        marginal="box",
        labels={"color": "Observed status", "log2_word_count_plus1": "log2(word count + 1)"},
        title="Page-length distribution among measurable surfaced sources",
        template="plotly_white",
    )
    histogram.update_layout(height=540)
    _write_plotly(histogram, interactive_dir / "M0_word_count_distribution.html")

    word_plot = word_table.copy()
    word_plot["cited_rate_pct"] = word_plot["cited_rate"] * 100
    word_plot["ci_low_pct"] = word_plot["wilson_ci_low"] * 100
    word_plot["ci_high_pct"] = word_plot["wilson_ci_high"] * 100
    fig = go.Figure(
        go.Scatter(
            x=word_plot["category"],
            y=word_plot["cited_rate_pct"],
            mode="lines+markers+text",
            text=[f"n={value:,}" for value in word_plot["n_rows"]],
            textposition="top center",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": word_plot["ci_high_pct"] - word_plot["cited_rate_pct"],
                "arrayminus": word_plot["cited_rate_pct"] - word_plot["ci_low_pct"],
            },
            line_color="#43aa8b",
        )
    )
    fig.add_hline(y=data["cited"].mean() * 100, line_dash="dash", line_color="#6c757d")
    fig.update_layout(
        title="Unadjusted cited rate by page-length quartile",
        xaxis_title="Word-count quartile",
        yaxis_title="Cited rate (%)",
        template="plotly_white",
        height=480,
    )
    _write_plotly(fig, interactive_dir / "M0_word_length_binned_cited_rate.html")
    return m0, distribution


def _runtime_guardrail(
    data: pd.DataFrame,
    formulas: dict[str, str],
    package_guardrail_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if package_guardrail_path.exists():
        package_guardrail = pd.read_csv(package_guardrail_path)
        for row in package_guardrail.itertuples(index=False):
            rows.append(
                {
                    "check_type": "package_guardrail",
                    "item": getattr(row, "guardrail_check", "package_check"),
                    "matches": getattr(row, "candidate_formula_matches", "none"),
                    "status": getattr(row, "status", "unknown"),
                    "details": "Loaded from package guardrail.",
                }
            )
    lower_columns = {column.casefold(): column for column in data.columns}
    for token in FORBIDDEN_FORMULA_TOKENS:
        matches = sorted(column for lower, column in lower_columns.items() if token in lower)
        rows.append(
            {
                "check_type": "dataset_column_scan",
                "item": token,
                "matches": "; ".join(matches) if matches else "none",
                "status": "review_only" if matches else "pass",
                "details": "A dataset column may exist, but it is forbidden from focal/headline formulas.",
            }
        )
    for model_id, formula in formulas.items():
        matches = forbidden_formula_matches(formula)
        rows.append(
            {
                "check_type": "formula_scan",
                "item": model_id,
                "matches": "; ".join(matches) if matches else "none",
                "status": "fail" if matches else "pass",
                "details": formula,
            }
        )
    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    if out.query("check_type == 'formula_scan' and status == 'fail'").shape[0]:
        raise ValueError("Runtime formula leakage check failed.")
    return out


def _selection_audit(all_rows: pd.DataFrame, measurable: pd.DataFrame) -> pd.DataFrame:
    all_rows = _prepare_model_data(all_rows)
    measurable = _prepare_model_data(measurable)
    rows: list[dict[str, Any]] = []

    def add_row(dimension: str, category: str, left: pd.DataFrame, right: pd.DataFrame) -> None:
        rows.append(
            {
                "dimension": dimension,
                "category": category,
                "all_surfaced_rows": len(left),
                "measurable_rows": len(right),
                "row_retention_rate": len(right) / len(left) if len(left) else np.nan,
                "all_unique_urls": left["normalized_url"].nunique(),
                "measurable_unique_urls": right["normalized_url"].nunique(),
                "all_unique_prompts": left["prompt_id"].nunique(),
                "measurable_unique_prompts": right["prompt_id"].nunique(),
                "all_unique_domains": left["source_root_domain"].nunique(),
                "measurable_unique_domains": right["source_root_domain"].nunique(),
                "all_cited_rate": left["cited"].mean() if len(left) else np.nan,
                "measurable_cited_rate": right["cited"].mean() if len(right) else np.nan,
                "cited_rate_difference_pp": (
                    (right["cited"].mean() - left["cited"].mean()) * 100 if len(left) and len(right) else np.nan
                ),
            }
        )

    add_row("overall", "all", all_rows, measurable)
    for cited in (0, 1):
        add_row(
            "cited_status",
            str(cited),
            all_rows[all_rows["cited"].eq(cited)],
            measurable[measurable["cited"].eq(cited)],
        )
    for dimension in ("intent", GEMINI_PAGE_FAMILY_COLLAPSED, GEMINI_SOURCE_TYPE_COLLAPSED):
        if dimension not in all_rows or dimension not in measurable:
            continue
        categories = all_rows[dimension].value_counts().head(25).index
        for category in categories:
            add_row(
                dimension,
                str(category),
                all_rows[all_rows[dimension].eq(category)],
                measurable[measurable[dimension].eq(category)],
            )
    top_domains = all_rows["source_root_domain"].value_counts().head(25).index
    for domain in top_domains:
        add_row(
            "source_root_domain_top25",
            str(domain),
            all_rows[all_rows["source_root_domain"].eq(domain)],
            measurable[measurable["source_root_domain"].eq(domain)],
        )
    return pd.DataFrame(rows)


def _fit_logit_ame(data: pd.DataFrame, formula: str, model_id: str, notes: str) -> pd.DataFrame:
    _assert_formula_safe(formula)
    prepared = _prepare_model_data(data)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = smf.logit(formula, data=prepared, missing="drop").fit(
                method="lbfgs",
                maxiter=300,
                disp=False,
            )
            marginal = result.get_margeff(at="overall", method="dydx", dummy=True)
        summary = marginal.summary_frame()
        columns = list(summary.columns)
        rows = []
        for term, row in summary.iterrows():
            estimate = float(row.iloc[0])
            std_error = float(row.iloc[1])
            statistic = float(row.iloc[2])
            p_value = float(row.iloc[3])
            conf_low = float(row.iloc[4])
            conf_high = float(row.iloc[5])
            rows.append(
                {
                    "model_id": model_id,
                    "formula": formula,
                    "term": term,
                    "estimate": estimate,
                    "estimate_pp": estimate * 100,
                    "std_error": std_error,
                    "z_statistic": statistic,
                    "p_value": p_value,
                    "conf_low": conf_low,
                    "conf_high": conf_high,
                    "conf_low_pp": conf_low * 100,
                    "conf_high_pp": conf_high * 100,
                    **_row_counts(prepared),
                    "cov_type": "logit_standard_ame",
                    "converged": bool(result.mle_retvals.get("converged", False)),
                    "status": "completed",
                    "notes": "; ".join(
                        [
                            notes,
                            f"summary_frame_columns={columns}",
                            *(str(item.message) for item in caught),
                        ]
                    ),
                }
            )
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "model_id": model_id,
                    "formula": formula,
                    "term": "",
                    "estimate": np.nan,
                    "estimate_pp": np.nan,
                    "std_error": np.nan,
                    "z_statistic": np.nan,
                    "p_value": np.nan,
                    "conf_low": np.nan,
                    "conf_high": np.nan,
                    "conf_low_pp": np.nan,
                    "conf_high_pp": np.nan,
                    **_row_counts(prepared),
                    "cov_type": "logit_standard_ame",
                    "converged": False,
                    "status": "failed",
                    "notes": f"{notes}; {type(exc).__name__}: {exc}",
                }
            ]
        )


def _intent_support(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, bool]]:
    work = data.copy()
    work["word_count_median_group"] = np.where(
        work["word_count"].ge(work["word_count"].median()), "at_or_above_median", "below_median"
    )
    rows = []
    supported: dict[str, bool] = {}
    for feature in ("word_count_median_group", "has_table", "heading_count_group"):
        feature_rows = []
        for (intent, category), group in work.groupby(["intent", feature], dropna=False):
            cited_rows = int(group["cited"].sum())
            more_only_rows = int(len(group) - cited_rows)
            stable = len(group) >= 20 and cited_rows >= 5 and more_only_rows >= 5
            feature_rows.append(stable)
            rows.append(
                {
                    "interaction_feature": feature,
                    "intent": intent,
                    "feature_category": category,
                    "n_rows": len(group),
                    "cited_rows": cited_rows,
                    "more_only_rows": more_only_rows,
                    "cited_rate": group["cited"].mean(),
                    "sparse_flag": len(group) < 20,
                    "unstable_flag": cited_rows < 5 or more_only_rows < 5,
                    "interaction_supported": stable,
                }
            )
        supported[feature] = bool(feature_rows) and all(feature_rows)
    return pd.DataFrame(rows), supported


def _actionable_contrasts(model: ModelRun) -> pd.DataFrame:
    data = model.data.copy()
    rows: list[dict[str, Any]] = []

    baseline = data.copy()
    comparison = data.copy()
    comparison["log2_word_count_plus1"] = comparison["log2_word_count_plus1"] + 1
    rows.append(
        make_prediction_contrast(
            model.base_result,
            data,
            contrast_name="Doubling page length",
            baseline_condition="Observed log2(word_count + 1)",
            comparison_condition="Observed log2(word_count + 1) + 1",
            baseline_data=baseline,
            comparison_data=comparison,
            term="log2_word_count_plus1",
            notes="Model-implied observed-covariate contrast; not causal.",
        )
    )

    baseline = data.assign(has_table=0)
    comparison = data.assign(has_table=1)
    rows.append(
        make_prediction_contrast(
            model.base_result,
            data,
            contrast_name="Table presence",
            baseline_condition="has_table = 0",
            comparison_condition="has_table = 1",
            baseline_data=baseline,
            comparison_data=comparison,
            term="has_table",
            notes="Model-implied observed-covariate contrast; not causal.",
        )
    )

    for feature, reference, categories, term_prefix, label in (
        (
            "heading_count_group",
            "0-1",
            ["2-6", "7-12", "13+"],
            HEADING,
            "Heading-count group",
        ),
        ("link_count_group", "9+", ["0-3", "4-8"], LINK, "Link-count group"),
        (
            "content_strength",
            "strong",
            ["medium", "weak"],
            STRENGTH,
            "Extraction-quality group",
        ),
    ):
        for category in categories:
            baseline = data.copy()
            comparison = data.copy()
            baseline[feature] = reference
            comparison[feature] = category
            term = f"{term_prefix}[T.{category}]"
            rows.append(
                make_prediction_contrast(
                    model.base_result,
                    data,
                    contrast_name=f"{label}: {category} vs {reference}",
                    baseline_condition=f"{feature} = {reference}",
                    comparison_condition=f"{feature} = {category}",
                    baseline_data=baseline,
                    comparison_data=comparison,
                    term=term,
                    notes=(
                        "Model-implied observed-covariate contrast; not causal. "
                        "Content strength measures extraction quality, not writing quality."
                        if feature == "content_strength"
                        else "Model-implied observed-covariate contrast; not causal."
                    ),
                )
            )
    return pd.DataFrame(rows)


def _unadjusted_focal_results(data: pd.DataFrame) -> pd.DataFrame:
    formulas = {
        "U_word_count": "cited ~ log2_word_count_plus1",
        "U_has_table": "cited ~ has_table",
        "U_heading_group": f"cited ~ {HEADING}",
        "U_link_group": f"cited ~ {LINK}",
        "U_content_strength": f"cited ~ {STRENGTH}",
    }
    rows = []
    for model_id, formula in formulas.items():
        rows.append(fit_lpm(formula, data, model_id, cov_type="HC3"))
    return pd.concat(rows, ignore_index=True)


def _first_term(table: pd.DataFrame, term: str, model_prefix: str | None = None) -> pd.Series | None:
    work = table[table["term"].eq(term)]
    if model_prefix is not None:
        work = work[work["model_id"].astype(str).str.startswith(model_prefix)]
    if work.empty:
        return None
    return _preferred_covariance_rows(work).iloc[0]


def _minimum_reporting_table(
    unadjusted: pd.DataFrame,
    m1: pd.DataFrame,
    m2: pd.DataFrame,
    m3: pd.DataFrame,
    m4: pd.DataFrame,
    m5: pd.DataFrame,
    m7: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    headline = _preferred_covariance_rows(m2)
    terms = headline[headline["term"].map(_is_focal_term)]["term"].tolist()
    rows = []
    for term in terms:
        row: dict[str, Any] = {
            "feature": _pretty_term(term),
            "term": term,
        }
        stages = {
            "unadjusted": _first_term(unadjusted, term),
            "M1_prompt_fe": _first_term(m1, term),
            "M2_joint": _first_term(m2, term),
            "M3_domain_fe": _first_term(m3, term),
            "M4_gemini_taxonomy": _first_term(m4, term),
            "M5_strong_content": _first_term(m5, term, "M5_M2"),
        }
        for prefix, result in stages.items():
            row[f"{prefix}_estimate_pp"] = result["estimate_pp"] if result is not None else np.nan
            row[f"{prefix}_conf_low_pp"] = result["conf_low_pp"] if result is not None else np.nan
            row[f"{prefix}_conf_high_pp"] = result["conf_high_pp"] if result is not None else np.nan
            row[f"{prefix}_p_value"] = result["p_value"] if result is not None else np.nan
        logit = m7[m7["term"].eq(term)] if not m7.empty and "term" in m7 else pd.DataFrame()
        if logit.empty and "log2_word_count_plus1" in term:
            logit = m7[m7["term"].str.contains("log2_word_count_plus1", na=False)]
        row["M7_logit_ame_pp"] = logit.iloc[0]["estimate_pp"] if not logit.empty else np.nan
        row["M7_logit_conf_low_pp"] = logit.iloc[0]["conf_low_pp"] if not logit.empty else np.nan
        row["M7_logit_conf_high_pp"] = logit.iloc[0]["conf_high_pp"] if not logit.empty else np.nan
        contrast = contrasts[contrasts["term"].eq(term)]
        row["actionable_contrast_pp"] = contrast.iloc[0]["difference_pp"] if not contrast.empty else np.nan
        m2_row = stages["M2_joint"]
        if m2_row is not None:
            for count in ("n_obs", "n_prompts", "n_urls", "n_domains", "cov_type"):
                row[f"M2_{count}"] = m2_row[count]
        row["interpretation_note"] = (
            "Extraction-quality control, not writing quality."
            if "content_strength" in term
            else (
                "Link-count groups are highly imbalanced; interpret low-link comparisons cautiously."
                if "link_count_group" in term
                else "Conditional association among surfaced sources; not causal and not web-wide."
            )
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _robustness_comparison(
    tables: list[pd.DataFrame],
    logit: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for table in tables:
        preferred = _preferred_covariance_rows(table)
        preferred = preferred[preferred["term"].map(_is_focal_term)].copy()
        if preferred.empty:
            continue
        preferred["feature"] = preferred["term"].map(_pretty_term)
        preferred["interpretation_note"] = preferred["term"].map(
            lambda term: (
                "Extraction-quality control, not writing quality."
                if "content_strength" in term
                else (
                    "Highly imbalanced link groups; low-link estimates may be unstable."
                    if "link_count_group" in term
                    else "Conditional association among surfaced sources."
                )
            )
        )
        rows.append(
            preferred[
                [
                    "feature",
                    "model_id",
                    "term",
                    "estimate_pp",
                    "conf_low_pp",
                    "conf_high_pp",
                    "p_value",
                    "n_obs",
                    "n_prompts",
                    "n_urls",
                    "n_domains",
                    "cov_type",
                    "interpretation_note",
                ]
            ]
        )
    if not logit.empty and logit["status"].eq("completed").any():
        focal = logit[logit["term"].map(_is_focal_term)].copy()
        focal["feature"] = focal["term"].map(_pretty_term)
        focal["interpretation_note"] = "Logit average marginal effect cross-check; not the headline model."
        rows.append(
            focal[
                [
                    "feature",
                    "model_id",
                    "term",
                    "estimate_pp",
                    "conf_low_pp",
                    "conf_high_pp",
                    "p_value",
                    "n_obs",
                    "n_prompts",
                    "n_urls",
                    "n_domains",
                    "cov_type",
                    "interpretation_note",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _coefficient_bullets(table: pd.DataFrame, limit: int | None = 12) -> str:
    preferred = _preferred_covariance_rows(table)
    preferred = preferred[preferred["term"].map(_is_focal_term)]
    if limit is not None:
        preferred = preferred.head(limit)
    include_model = preferred["model_id"].nunique() > 1
    bullets = []
    for row in preferred.itertuples(index=False):
        direction = "higher" if row.estimate_pp >= 0 else "lower"
        precision = "interval excludes zero" if row.conf_low_pp > 0 or row.conf_high_pp < 0 else "interval includes zero"
        model_label = f"`{row.model_id}`: " if include_model else ""
        bullets.append(
            f"- {model_label}`{_pretty_term(row.term)}` was associated with {direction} citation probability "
            f"({row.estimate_pp:.2f} pp; 95% CI {row.conf_low_pp:.2f} to {row.conf_high_pp:.2f}; "
            f"{row.cov_type}; {precision})."
        )
    return "\n".join(bullets) if bullets else "- No focal coefficient was available."


def _stability_summary(robustness: pd.DataFrame) -> tuple[str, str]:
    if robustness.empty:
        return "- No cross-model stability assessment was available.", "- All focal terms require review."
    selected = robustness[
        robustness["model_id"].astype(str).str.startswith(("M2", "M3", "M4", "M5_M2", "M10"))
    ]
    stable = []
    unstable = []
    for feature, group in selected.groupby("feature"):
        estimates = group["estimate_pp"].dropna()
        if len(estimates) < 3:
            unstable.append(f"- `{feature}` had too few comparable model estimates.")
            continue
        same_direction = (estimates.ge(0).all() or estimates.le(0).all())
        if same_direction:
            stable.append(
                f"- `{feature}` kept the same direction across {len(estimates)} reported robustness estimates."
            )
        else:
            unstable.append(
                f"- `{feature}` changed direction across specifications and should not receive a stable interpretation."
            )
    return (
        "\n".join(stable) if stable else "- No focal association met the directional-stability rule.",
        "\n".join(unstable) if unstable else "- No focal term changed direction under the compared specifications.",
    )


def _model_sample_lines(table: pd.DataFrame) -> str:
    if table.empty:
        return "- No model sample was available."
    lines = []
    for model_id, group in table.groupby("model_id", sort=False):
        row = group.iloc[0]
        lines.append(
            f"- `{model_id}`: {int(row['n_obs']):,} rows, {int(row['n_urls']):,} URLs, "
            f"{int(row['n_prompts']):,} prompts, and {int(row['n_domains']):,} domains."
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    data: pd.DataFrame,
    m2: pd.DataFrame,
    m3: pd.DataFrame,
    m4: pd.DataFrame,
    m5: pd.DataFrame,
    m7: pd.DataFrame,
    m10: pd.DataFrame,
    robustness: pd.DataFrame,
    contrasts: pd.DataFrame,
    domain_support: dict[str, int],
    warnings_list: list[str],
) -> None:
    stable, unstable = _stability_summary(robustness)
    m7_status = (
        "completed using the simplified intent and area controls because prompt fixed-effect logit had "
        "separation risk"
        if not m7.empty and m7["status"].eq("completed").any()
        else "not completed; see the M7 result notes"
    )
    contrast_lines = "\n".join(
        f"- `{row.contrast_name}`: {row.difference_pp:.2f} pp."
        for row in contrasts.itertuples(index=False)
    )
    warning_lines = "\n".join(f"- {message}" for message in warnings_list) or "- No additional runtime warning."
    report = f"""# 09 Content Feature Econometrics Report

## 1. Analysis scope and estimand

This analysis covers the area-condo / SCOPE-relevant nonbranded audit. The unit is one surfaced source appearance, and the estimand is `P(cited = 1 | source surfaced in this audit)`. Results are conditional associations among surfaced sources. They are not causal, not web-wide, and do not imply that rewriting a page will change citation outcomes.

## 2. Dataset and sample counts

- Measurable rows: {len(data):,}
- Unique normalized URLs: {data['normalized_url'].nunique():,}
- Unique prompts: {data['prompt_id'].nunique():,}
- Unique source-root domains: {data['source_root_domain'].nunique():,}
- Cited rows: {int(data['cited'].sum()):,}
- Cited rate: {data['cited'].mean():.2%}
- Full audit = 500 prompts; measurable-content LPM sample = {data['prompt_id'].nunique():,} prompts.

## 3. Model ladder

M0 describes unadjusted rates. M1 estimates one-feature prompt-fixed-effect LPMs. M2 is the preferred joint structural-content LPM. M3 adds domain fixed effects. M4 adds the collapsed Gemini page-function family and source/site type. M4R retains the older rule-based URL-seed label as a robustness comparison only. M5 restricts to strong extraction quality. M6 audits measurable-content selection. M7 provides a logit AME cross-check. M8 searches for leakage-safe prompt-page relevance. M9 allows only supported intent interactions. M10 checks extreme tails and winsorization.

## 4. Main M2 results

{_coefficient_bullets(m2)}

These estimates describe features associated with higher/lower citation probability conditional on surfaced sources. `content_strength` is extraction-quality control, not writing quality. `link_count_group` is highly imbalanced; coefficients for low-link groups should be interpreted cautiously.

## 5. Domain-FE robustness

M3 retained {domain_support['rows']:,} rows from {domain_support['domains']:,} domains with at least two unique URLs. Single-URL domains were filtered. Domain fixed effects absorb stable publisher and template differences, so M3 is a robustness check rather than the headline.

{_coefficient_bullets(m3)}

## 6. Gemini taxonomy sensitivity

M4 uses `{GEMINI_PAGE_FAMILY_COLLAPSED}` and `{GEMINI_SOURCE_TYPE_COLLAPSED}` from the versioned `{GEMINI_TAXONOMY_VERSION}` classification. The labels were produced without answer text or citation labels. Because Gemini can use scraped page content, M4 is a taxonomy-adjusted sensitivity model rather than the headline content model. M1/M2 remain the primary results.

{_coefficient_bullets(m4)}

## 7. Strong-content sensitivity

M5 tests the models on `content_strength == "strong"`. This restriction addresses extraction measurement quality; it is not a writing-quality score.

{_model_sample_lines(m5)}

{_coefficient_bullets(m5, limit=40)}

## 8. Outlier and winsorized sensitivity

M10 removes the top 1% word-count tail, removes the top 1% link-count tail, and replaces page length with its p99-winsorized transform.

{_model_sample_lines(m10)}

{_coefficient_bullets(m10, limit=40)}

## 9. Logit AME cross-check

The logit cross-check was {m7_status}. It does not replace the LPM headline.

## 10. Predicted probability contrasts

These are model-implied contrasts, not causal effects:

{contrast_lines}

## 11. What appears stable

{stable}

## 12. What is unstable or sensitive

{unstable}

## 13. Limitations

- The analysis is conditional on sources already being surfaced and does not represent all webpages on the internet.
- Scrape and content measurability are selected, not random. Failed or unavailable content is not feature absence.
- Prompt fixed effects control prompt-level differences but do not remove all page, publisher, or relevance confounding.
- Domain fixed effects identify within-domain differences and may not generalize to a new publisher.
- Repeated URLs motivate URL clustering; repeated prompts motivate prompt clustering.
- Sparse and imbalanced categories can produce unstable estimates.
- Associations are not causal and should not be described as evidence that an AI system prefers a feature.

## 14. Recommended next feature layer

Build pre-specified, leakage-safe writing measures from page text only: sentence-length distribution, section-level specificity, named-entity density, numerical-detail density, list structure, question-answer formatting, and prompt-page relevance computed from prompt text and page text without answer text. Validate extraction reliability before adding these features jointly.

## Runtime warnings

{warning_lines}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_content_feature_econometrics(package_path: Path | str) -> dict[str, Any]:
    """Run notebook 09's complete content-feature econometric workflow."""
    package = Path(package_path).resolve()
    data_path = package / "data/content_lpm_measurable_rows.csv"
    all_rows_path = package / "data/content_lpm_all_surfaced_rows.csv"
    table_dir = package / "tables/09_content_feature_econometrics"
    figure_dir = package / "figures/09_content_feature_econometrics"
    interactive_dir = figure_dir / "interactive"
    report_dir = package / "reports/09_content_feature_econometrics"
    for directory in (table_dir, figure_dir, interactive_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    warnings_list: list[str] = []
    fitted_model_ids: list[str] = []
    formulas: dict[str, str] = {
        "M1a": f"cited ~ log2_word_count_plus1 + {PROMPT_FE}",
        "M1b": f"cited ~ has_table + {PROMPT_FE}",
        "M1c": f"cited ~ {HEADING} + {PROMPT_FE}",
        "M1d": f"cited ~ {LINK} + {PROMPT_FE}",
        "M1e": f"cited ~ {STRENGTH} + {PROMPT_FE}",
        "M2": M2_FORMULA,
        "M3": f"{M2_FORMULA} + {DOMAIN_FE}",
        "M4": f"{M2_FORMULA} + {GEMINI_TAXONOMY_TERMS}",
        "M4R_rule_v2": f"{M2_FORMULA} + {RULE_V2_PAGE_SEED}",
        "M5_M2_strong": M2_FORMULA,
        "M5_M3_strong": f"{M2_FORMULA} + {DOMAIN_FE}",
        "M5_M4_strong": f"{M2_FORMULA} + {GEMINI_TAXONOMY_TERMS}",
        "M10a_word_p99_removed": M2_FORMULA,
        "M10b_link_p99_removed": M2_FORMULA,
        "M10c_word_winsorized": M2_FORMULA.replace(
            "log2_word_count_plus1", "log2_word_count_plus1_winsorized_p99"
        ),
    }

    if not data_path.exists():
        raise FileNotFoundError(f"Measurable-content model input not found: {data_path}")
    data, taxonomy_mappings, taxonomy_audit = attach_gemini_taxonomy(
        pd.read_csv(data_path, low_memory=False),
        package,
    )
    data = _prepare_model_data(data)
    pd.DataFrame([taxonomy_audit]).to_csv(
        table_dir / "09_gemini_taxonomy_join_audit.csv",
        index=False,
    )

    expected = {
        "n_rows": 5264,
        "unique_normalized_url": 2600,
        "unique_prompt_id": 498,
        "unique_source_root_domain": 541,
        "cited_rows": 1708,
    }
    actual = {
        "n_rows": len(data),
        "unique_normalized_url": data["normalized_url"].nunique() if "normalized_url" in data else 0,
        "unique_prompt_id": data["prompt_id"].nunique() if "prompt_id" in data else 0,
        "unique_source_root_domain": (
            data["source_root_domain"].nunique() if "source_root_domain" in data else 0
        ),
        "cited_rows": int(data["cited"].sum()) if "cited" in data else 0,
        "cited_rate": float(data["cited"].mean()) if "cited" in data else np.nan,
    }
    readiness_rows = []
    for metric, value in actual.items():
        expected_value = expected.get(metric, 1708 / 5264 if metric == "cited_rate" else np.nan)
        matches = (
            np.isclose(value, expected_value, rtol=0, atol=0.0005)
            if metric == "cited_rate"
            else value == expected_value
        )
        readiness_rows.append(
            {
                "section": "sample_count",
                "metric": metric,
                "value": value,
                "expected_value": expected_value,
                "status": "match" if matches else "differs_but_continue",
            }
        )
    for feature in ("expansion_group", "content_strength"):
        if feature not in data:
            continue
        for category, n_rows in data[feature].value_counts(dropna=False).items():
            readiness_rows.append(
                {
                    "section": f"{feature}_distribution",
                    "metric": str(category),
                    "value": int(n_rows),
                    "expected_value": np.nan,
                    "status": "observed",
                }
            )
    readiness = pd.DataFrame(readiness_rows)
    readiness.to_csv(table_dir / "09_dataset_readiness_summary.csv", index=False)

    required_rows = []
    for column in (*REQUIRED_COLUMNS, *SENSITIVITY_COLUMNS):
        required_rows.append(
            {
                "column": column,
                "requirement_level": "required" if column in REQUIRED_COLUMNS else "sensitivity_if_available",
                "present": column in data,
                "nonmissing_rows": int(data[column].notna().sum()) if column in data else 0,
                "status": (
                    "pass"
                    if column in data
                    else ("fail" if column in REQUIRED_COLUMNS else "unavailable_optional")
                ),
            }
        )
    cited_binary = "cited" in data and set(data["cited"].dropna().unique()).issubset({0, 1})
    required_rows.append(
        {
            "column": "cited_binary_validation",
            "requirement_level": "required",
            "present": "cited" in data,
            "nonmissing_rows": int(data["cited"].notna().sum()) if "cited" in data else 0,
            "status": "pass" if cited_binary else "fail",
        }
    )
    required_check = pd.DataFrame(required_rows)
    required_check.to_csv(table_dir / "09_required_column_check.csv", index=False)
    if required_check.query("requirement_level == 'required' and status == 'fail'").shape[0]:
        status = "failed_missing_required_columns"
        manifest = {
            "input_data_path": str(data_path),
            "package_path": str(package),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "row_counts": actual,
            "warnings": ["Required-column validation failed."],
            "final_notebook_status": status,
        }
        (report_dir / "09_model_run_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        raise ValueError("Required-column validation failed.")

    runtime_guardrail = _runtime_guardrail(
        data,
        formulas,
        package / "tables/content_lpm_leakage_guardrail_check.csv",
        table_dir / "09_leakage_guardrail_runtime_check.csv",
    )

    m0, distribution = _build_m0(data, table_dir, interactive_dir)

    m1_runs: list[ModelRun] = []
    if "low_link_count" in data:
        formulas["M1f"] = f"cited ~ low_link_count + {PROMPT_FE}"
    m1_path = table_dir / "M1_one_feature_prompt_fe_results.csv"
    for model_id in ("M1a", "M1b", "M1c", "M1d", "M1e", "M1f"):
        if model_id not in formulas:
            continue
        temporary = table_dir / f".{model_id}_working.csv"
        run = run_model_and_save(formulas[model_id], data, model_id, temporary)
        m1_runs.append(run)
        fitted_model_ids.append(model_id)
        warnings_list.extend(run.warnings)
        pd.concat([item.table for item in m1_runs], ignore_index=True).to_csv(m1_path, index=False)
        temporary.unlink(missing_ok=True)
    m1 = pd.concat([run.table for run in m1_runs], ignore_index=True)
    make_coefficient_forest(
        m1,
        interactive_dir / "M1_focal_feature_forest.html",
        title="M1 prompt-adjusted one-feature associations",
    )

    m2_run = run_model_and_save(
        formulas["M2"],
        data,
        "M2",
        table_dir / "M2_preferred_joint_lpm_results.csv",
        notes=(
            "Preferred joint LPM; content_strength is extraction quality, not writing quality; "
            "link_count_group is highly imbalanced."
        ),
    )
    fitted_model_ids.append("M2")
    warnings_list.extend(m2_run.warnings)
    m2 = m2_run.table
    make_coefficient_forest(
        m2,
        interactive_dir / "M2_preferred_joint_lpm_forest.html",
        title="M2 preferred joint structural-content LPM",
    )

    domain_url_counts = data.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = domain_url_counts[domain_url_counts.ge(2)].index
    m3_data = data[data["source_root_domain"].isin(supported_domains)].copy().reset_index(drop=True)
    domain_support = {"rows": len(m3_data), "domains": m3_data["source_root_domain"].nunique()}
    m3_run = run_model_and_save(
        formulas["M3"],
        m3_data,
        "M3_domain_fe",
        table_dir / "M3_domain_fe_results.csv",
        notes=(
            f"Filtered to domains with >=2 unique URLs; rows={len(m3_data)}; "
            f"domains={m3_data['source_root_domain'].nunique()}; single-URL domains excluded."
        ),
    )
    fitted_model_ids.append("M3_domain_fe")
    warnings_list.extend(m3_run.warnings)
    m3 = m3_run.table

    m4_run = run_model_and_save(
        formulas["M4"],
        data,
        "M4_gemini_taxonomy",
        table_dir / "M4_gemini_taxonomy_sensitivity_results.csv",
        notes=(
            f"Gemini taxonomy sensitivity using {GEMINI_PAGE_FAMILY_COLLAPSED} and "
            f"{GEMINI_SOURCE_TYPE_COLLAPSED}; taxonomy version={GEMINI_TAXONOMY_VERSION}."
        ),
    )
    fitted_model_ids.append("M4_gemini_taxonomy")
    warnings_list.extend(m4_run.warnings)
    m4 = m4_run.table

    m4r_run = run_model_and_save(
        formulas["M4R_rule_v2"],
        data,
        "M4R_rule_v2_url_seed",
        table_dir / "M4R_rule_v2_taxonomy_robustness_results.csv",
        notes="Legacy rule-v2 URL-seed page type retained only as a robustness comparison.",
    )
    fitted_model_ids.append("M4R_rule_v2_url_seed")
    warnings_list.extend(m4r_run.warnings)

    strong = data[data["content_strength"].eq("strong")].copy().reset_index(drop=True)
    strong_domain_counts = strong.groupby("source_root_domain")["normalized_url"].nunique()
    strong_supported_domains = strong_domain_counts[strong_domain_counts.ge(2)].index
    strong_domain = strong[strong["source_root_domain"].isin(strong_supported_domains)].copy()
    m5_runs: list[ModelRun] = []
    for model_id, formula, sample, note in (
        (
            "M5_M2_strong",
            formulas["M5_M2_strong"],
            strong,
            "Strong-content M2; content_strength is extraction-quality control.",
        ),
        (
            "M5_M3_strong",
            formulas["M5_M3_strong"],
            strong_domain,
            "Strong-content M3; domains must have >=2 unique URLs in strong sample.",
        ),
        (
            "M5_M4_strong",
            formulas["M5_M4_strong"],
            strong,
            "Strong-content M4 with Gemini page family and source/site type.",
        ),
    ):
        temporary = table_dir / f".{model_id}_working.csv"
        run = run_model_and_save(formula, sample, model_id, temporary, notes=note)
        m5_runs.append(run)
        fitted_model_ids.append(model_id)
        warnings_list.extend(run.warnings)
        pd.concat([item.table for item in m5_runs], ignore_index=True).to_csv(
            table_dir / "M5_strong_content_sensitivity_results.csv",
            index=False,
        )
        temporary.unlink(missing_ok=True)
    m5 = pd.concat([run.table for run in m5_runs], ignore_index=True)

    if all_rows_path.exists():
        all_rows, _, all_rows_taxonomy_audit = attach_gemini_taxonomy(
            pd.read_csv(all_rows_path, low_memory=False),
            package,
            collapse_mappings=taxonomy_mappings,
        )
        pd.DataFrame([all_rows_taxonomy_audit]).to_csv(
            table_dir / "09_gemini_taxonomy_all_rows_join_audit.csv",
            index=False,
        )
        m6 = _selection_audit(all_rows, data)
    else:
        sample_recheck = package / "tables/content_lpm_sample_count_recheck.csv"
        if sample_recheck.exists():
            recheck = pd.read_csv(sample_recheck)
            m6 = recheck.assign(
                dimension="sample_count_recheck",
                category=recheck["sample"],
                note="All-surfaced file unavailable; copied package sample-count recheck.",
            )
        else:
            m6 = pd.DataFrame(
                [
                    {
                        "dimension": "selection_note",
                        "category": "measurable_only",
                        "note": "No all-surfaced comparison file was available.",
                    }
                ]
            )
    m6.to_csv(table_dir / "M6_measurable_selection_audit.csv", index=False)

    prompt_outcomes = data.groupby("prompt_id")["cited"].agg(["min", "max"])
    nonvarying_prompts = int(prompt_outcomes["min"].eq(prompt_outcomes["max"]).sum())
    full_logit_note = (
        f"Full prompt-FE logit was not attempted because {nonvarying_prompts} prompt groups have no "
        "within-prompt outcome variation, creating separation risk."
    )
    simplified_controls = []
    for candidate in ("intent", "area_tag"):
        if candidate in data and data[candidate].nunique() > 1:
            simplified_controls.append(f"C({candidate})")
    simplified_logit_formula = (
        f"cited ~ log2_word_count_plus1 + has_table + {HEADING} + {LINK} + {STRENGTH}"
        + (f" + {' + '.join(simplified_controls)}" if simplified_controls else "")
    )
    formulas["M7_simplified_logit_ame"] = simplified_logit_formula
    m7 = _fit_logit_ame(
        data,
        simplified_logit_formula,
        "M7_simplified_logit_ame",
        f"{full_logit_note} Simplified controls: {', '.join(simplified_controls) or 'none'}.",
    )
    m7.to_csv(table_dir / "M7_logit_ame_crosscheck_results.csv", index=False)
    if m7["status"].eq("completed").any():
        fitted_model_ids.append("M7_simplified_logit_ame")
    else:
        warnings_list.append(str(m7.iloc[0]["notes"]))

    relevance_candidates = []
    for column in data.columns:
        lower = column.casefold()
        leakage_match = any(token in lower for token in FORBIDDEN_FORMULA_TOKENS) or "answer" in lower
        relevance_match = "prompt" in lower and any(token in lower for token in ("similarity", "relevance", "overlap"))
        if relevance_match and not leakage_match and pd.api.types.is_numeric_dtype(data[column]):
            relevance_candidates.append(column)
    if relevance_candidates:
        candidate = relevance_candidates[0]
        formula = f"{M2_FORMULA} + {candidate}"
        formulas["M8_prompt_page_relevance"] = formula
        m8_run = run_model_and_save(
            formula,
            data,
            "M8_prompt_page_relevance",
            table_dir / "M8_prompt_page_relevance_sensitivity_results.csv",
            notes=f"Leakage-safe prompt-page candidate selected: {candidate}",
        )
        m8 = m8_run.table
        fitted_model_ids.append("M8_prompt_page_relevance")
        warnings_list.extend(m8_run.warnings)
    else:
        m8 = pd.DataFrame(
            [
                {
                    "model_id": "M8",
                    "status": "skipped",
                    "candidate_feature": "",
                    "notes": "No leakage-safe prompt-page relevance feature found.",
                }
            ]
        )
        m8.to_csv(table_dir / "M8_prompt_page_relevance_sensitivity_results.csv", index=False)

    if "intent" in data:
        m9_support, supported = _intent_support(data)
    else:
        m9_support = pd.DataFrame(
            [{"interaction_feature": "", "intent": "", "interaction_supported": False}]
        )
        supported = {}
    m9_support.to_csv(table_dir / "M9_intent_interaction_cell_support.csv", index=False)
    m9_runs: list[ModelRun] = []
    if supported.get("word_count_median_group"):
        formula = (
            f"cited ~ C(intent):log2_word_count_plus1 + has_table + {HEADING} + {LINK} "
            f"+ {STRENGTH} + {PROMPT_FE}"
        )
        formulas["M9_word_count_by_intent"] = formula
        run = run_model_and_save(
            formula,
            data,
            "M9_word_count_by_intent",
            table_dir / ".M9_word_working.csv",
            notes="Limited interaction sensitivity; not a headline result.",
        )
        m9_runs.append(run)
        fitted_model_ids.append("M9_word_count_by_intent")
        warnings_list.extend(run.warnings)
        (table_dir / ".M9_word_working.csv").unlink(missing_ok=True)
    if supported.get("has_table"):
        formula = (
            f"cited ~ log2_word_count_plus1 + C(intent):has_table + {HEADING} + {LINK} "
            f"+ {STRENGTH} + {PROMPT_FE}"
        )
        formulas["M9_has_table_by_intent"] = formula
        run = run_model_and_save(
            formula,
            data,
            "M9_has_table_by_intent",
            table_dir / ".M9_table_working.csv",
            notes="Limited interaction sensitivity; not a headline result.",
        )
        m9_runs.append(run)
        fitted_model_ids.append("M9_has_table_by_intent")
        warnings_list.extend(run.warnings)
        (table_dir / ".M9_table_working.csv").unlink(missing_ok=True)
    if m9_runs:
        m9 = pd.concat([run.table for run in m9_runs], ignore_index=True)
    else:
        m9 = pd.DataFrame(
            [
                {
                    "model_id": "M9",
                    "status": "skipped_sparse_cells",
                    "term": "",
                    "notes": "Intent interactions were skipped because at least one required cell was unstable.",
                }
            ]
        )
    m9.to_csv(table_dir / "M9_intent_interaction_sensitivity_results.csv", index=False)

    outlier_path = package / "tables/content_lpm_outlier_audit.csv"
    outlier = pd.read_csv(outlier_path) if outlier_path.exists() else pd.DataFrame()

    def p99_for(feature: str, fallback: float) -> float:
        match = outlier[outlier["feature"].eq(feature)] if not outlier.empty else pd.DataFrame()
        return float(match.iloc[0]["p99_threshold"]) if not match.empty else fallback

    word_p99 = p99_for("word_count", float(data["word_count"].quantile(0.99)))
    link_p99 = p99_for("link_count", float(data["link_count"].quantile(0.99)))
    m10_samples = (
        (
            "M10a_word_p99_removed",
            formulas["M10a_word_p99_removed"],
            data[data["word_count"].le(word_p99)].copy(),
            f"Excluded word_count > package p99 ({word_p99:.4f}).",
        ),
        (
            "M10b_link_p99_removed",
            formulas["M10b_link_p99_removed"],
            data[data["link_count"].le(link_p99)].copy(),
            f"Excluded link_count > package p99 ({link_p99:.4f}).",
        ),
        (
            "M10c_word_winsorized",
            formulas["M10c_word_winsorized"],
            data,
            "Replaced page length with log2_word_count_plus1_winsorized_p99.",
        ),
    )
    m10_runs: list[ModelRun] = []
    for model_id, formula, sample, note in m10_samples:
        temporary = table_dir / f".{model_id}_working.csv"
        run = run_model_and_save(formula, sample, model_id, temporary, notes=note)
        m10_runs.append(run)
        fitted_model_ids.append(model_id)
        warnings_list.extend(run.warnings)
        pd.concat([item.table for item in m10_runs], ignore_index=True).to_csv(
            table_dir / "M10_outlier_winsorized_sensitivity_results.csv",
            index=False,
        )
        temporary.unlink(missing_ok=True)
    m10 = pd.concat([run.table for run in m10_runs], ignore_index=True)

    runtime_guardrail = _runtime_guardrail(
        data,
        formulas,
        package / "tables/content_lpm_leakage_guardrail_check.csv",
        table_dir / "09_leakage_guardrail_runtime_check.csv",
    )

    unadjusted = _unadjusted_focal_results(data)
    robustness = _robustness_comparison([m1, m2, m3, m4, m5, m10], m7)
    robustness.to_csv(table_dir / "09_focal_feature_robustness_comparison.csv", index=False)
    make_coefficient_forest(
        robustness.rename(columns={"feature": "existing_feature_label"}),
        interactive_dir / "09_focal_feature_robustness_forest.html",
        title="Focal content-feature robustness across specifications",
        preferred_only=False,
    )

    contrasts = _actionable_contrasts(m2_run)
    contrasts.to_csv(table_dir / "09_actionable_predicted_probability_contrasts.csv", index=False)

    minimum = _minimum_reporting_table(unadjusted, m1, m2, m3, m4, m5, m7, contrasts)
    minimum.to_csv(table_dir / "09_minimum_reporting_table.csv", index=False)

    report_path = report_dir / "09_content_feature_econometrics_report.md"
    _write_report(
        report_path,
        data,
        m2,
        m3,
        m4,
        m5,
        m7,
        m10,
        robustness,
        contrasts,
        domain_support,
        warnings_list,
    )

    key_models_complete = all(
        key in fitted_model_ids
        for key in (
            "M2",
            "M3_domain_fe",
            "M4_gemini_taxonomy",
            "M5_M2_strong",
            "M10a_word_p99_removed",
            "M10b_link_p99_removed",
            "M10c_word_winsorized",
        )
    )
    leakage_passed = not runtime_guardrail["status"].eq("fail").any()
    final_status = (
        "completed_ready_for_interpretation_with_caveats"
        if key_models_complete and leakage_passed
        else "completed_with_warnings"
    )

    output_files = sorted(
        str(path.relative_to(package))
        for directory in (table_dir, figure_dir, report_dir)
        for path in directory.rglob("*")
        if path.is_file()
    )
    manifest_relative_path = "reports/09_content_feature_econometrics/09_model_run_manifest.json"
    if manifest_relative_path not in output_files:
        output_files.append(manifest_relative_path)
        output_files.sort()
    manifest = {
        "input_data_path": str(data_path),
        "package_path": str(package),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_counts": actual,
        "model_formulas": formulas,
        "cov_type_used": list(PREFERRED_COVARIANCE_ORDER),
        "clustering_variables": ["prompt_id", "normalized_url"],
        "number_of_fitted_models": len(fitted_model_ids),
        "fitted_model_ids": fitted_model_ids,
        "output_files_created": output_files,
        "warnings": warnings_list,
        "leakage_check_passed": leakage_passed,
        "model_completion": {
            "M2": "M2" in fitted_model_ids,
            "M3": "M3_domain_fe" in fitted_model_ids,
            "M4": "M4_gemini_taxonomy" in fitted_model_ids,
            "M5": "M5_M2_strong" in fitted_model_ids,
            "M7": "M7_simplified_logit_ame" in fitted_model_ids,
            "M10": all(model_id in fitted_model_ids for model_id, *_ in m10_samples),
        },
        "final_notebook_status": final_status,
    }
    manifest_path = report_dir / "09_model_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "input_dataset_path": str(data_path),
        "rows": actual["n_rows"],
        "unique_urls": actual["unique_normalized_url"],
        "unique_prompts": actual["unique_prompt_id"],
        "unique_domains": actual["unique_source_root_domain"],
        "cited_rows": actual["cited_rows"],
        "cited_rate": actual["cited_rate"],
        "number_of_fitted_models": len(fitted_model_ids),
        "leakage_check_passed": leakage_passed,
        "M2_completed": "M2" in fitted_model_ids,
        "M3_completed": "M3_domain_fe" in fitted_model_ids,
        "M4_completed": "M4_gemini_taxonomy" in fitted_model_ids,
        "M5_completed": "M5_M2_strong" in fitted_model_ids,
        "M7_completed": "M7_simplified_logit_ame" in fitted_model_ids,
        "M10_completed": all(model_id in fitted_model_ids for model_id, *_ in m10_samples),
        "minimum_reporting_table": str(table_dir / "09_minimum_reporting_table.csv"),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "final_status": final_status,
    }
