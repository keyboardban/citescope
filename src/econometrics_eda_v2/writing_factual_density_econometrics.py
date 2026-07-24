"""Notebook 11 writing/factual-density econometrics for surfaced sources."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .gemini_taxonomy_features import (
    GEMINI_PAGE_FAMILY,
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    GEMINI_TAXONOMY_VERSION,
    attach_gemini_taxonomy,
)


HEADING = "C(heading_count_group, Treatment(reference='0-1'))"
LINK = "C(link_count_group, Treatment(reference='9+'))"
STRENGTH = "C(content_strength, Treatment(reference='strong'))"
PROMPT_FE = "C(prompt_id)"
DOMAIN_FE = "C(source_root_domain)"
GEMINI_PAGE_FAMILY_TERM = (
    f"C({GEMINI_PAGE_FAMILY_COLLAPSED}, Treatment(reference='informational_content'))"
)
GEMINI_SOURCE_TYPE_TERM = (
    f"C({GEMINI_SOURCE_TYPE_COLLAPSED}, Treatment(reference='official_company_or_brand'))"
)
GEMINI_TAXONOMY_TERMS = f"{GEMINI_PAGE_FAMILY_TERM} + {GEMINI_SOURCE_TYPE_TERM}"
RULE_V2_PAGE_SEED = "C(page_type_url_seed_general_collapsed, Treatment(reference='unknown'))"

MAIN_FEATURES = (
    "factual_numeric_density_score",
    "price_unit_detail_score",
    "location_transit_specificity_score",
    "prompt_page_relevance_score",
)
SENSITIVITY_FEATURES = (
    "amenity_project_detail_score",
    "external_evidence_score",
)
DIAGNOSTIC_FEATURES = ("writing_structure_score",)
FOCAL_FEATURES = ("has_table", *MAIN_FEATURES, *SENSITIVITY_FEATURES)
DESCRIPTIVE_FEATURES = (*MAIN_FEATURES, *SENSITIVITY_FEATURES, *DIAGNOSTIC_FEATURES)
OUTLIER_FEATURES = (
    "factual_numeric_density_score",
    "price_unit_detail_score",
    "location_transit_specificity_score",
    "amenity_project_detail_score",
    "prompt_page_relevance_score",
    "word_count",
    "link_count",
)
REQUIRED_FEATURES = (
    "cited",
    "has_table",
    "log2_word_count_plus1",
    "heading_count_group",
    "link_count_group",
    "content_strength",
    *MAIN_FEATURES,
    *SENSITIVITY_FEATURES,
    *DIAGNOSTIC_FEATURES,
    "feature_extraction_text_scope",
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    "source_root_domain",
    "prompt_id",
    "normalized_url",
)
COVARIANCE_TYPES = (
    "HC3",
    "cluster_prompt_id",
    "cluster_normalized_url",
    "two_way_cluster_prompt_url",
)
COVARIANCE_ORDER = {
    "two_way_cluster_prompt_url": 0,
    "cluster_prompt_id": 1,
    "cluster_normalized_url": 2,
    "HC3": 3,
}
FORBIDDEN_NOTEBOOK11_TOKENS = (
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
    "page_type_family_general",
    "page_type_general_common",
    "page_type_final",
)

STRUCTURAL_RHS = (
    f"log2_word_count_plus1 + has_table + {HEADING} + {LINK} + {STRENGTH}"
)
MAIN_RHS = " + ".join(MAIN_FEATURES)
SENSITIVITY_RHS = " + ".join(SENSITIVITY_FEATURES)

B0_FORMULA = f"cited ~ {STRUCTURAL_RHS} + {PROMPT_FE}"
W1_FORMULA = f"cited ~ {MAIN_RHS} + {PROMPT_FE}"
W2_FORMULA = f"cited ~ {MAIN_RHS} + {SENSITIVITY_RHS} + {PROMPT_FE}"
W3_FORMULA = f"cited ~ {STRUCTURAL_RHS} + {MAIN_RHS} + {PROMPT_FE}"

F_FORMULAS = {
    "F1_factual_numeric_density": f"cited ~ {MAIN_FEATURES[0]} + {PROMPT_FE}",
    "F2_price_unit_detail": f"cited ~ {MAIN_FEATURES[1]} + {PROMPT_FE}",
    "F3_location_transit": f"cited ~ {MAIN_FEATURES[2]} + {PROMPT_FE}",
    "F4_prompt_page_relevance": f"cited ~ {MAIN_FEATURES[3]} + {PROMPT_FE}",
    "F5_amenity_project_sensitivity": f"cited ~ {SENSITIVITY_FEATURES[0]} + {PROMPT_FE}",
    "F6_external_evidence_sensitivity": f"cited ~ {SENSITIVITY_FEATURES[1]} + {PROMPT_FE}",
    "F7_writing_structure_diagnostic": f"cited ~ {DIAGNOSTIC_FEATURES[0]} + {PROMPT_FE}",
}
W_FORMULAS = {
    "W1_main_writing_factual": W1_FORMULA,
    "W2_expanded_writing_factual": W2_FORMULA,
    "W3_structural_plus_writing_factual": W3_FORMULA,
}
T_FORMULAS = {
    "T0_has_table_prompt_fe": f"cited ~ has_table + {PROMPT_FE}",
    "T1_has_table_structural": B0_FORMULA,
    "T2_table_plus_factual_detail": (
        f"cited ~ has_table + {' + '.join(MAIN_FEATURES[:3])} + {PROMPT_FE}"
    ),
    "T3_table_plus_detail_relevance": f"cited ~ has_table + {MAIN_RHS} + {PROMPT_FE}",
    "T4_table_structural_detail_relevance": W3_FORMULA,
}


def _plotly_modules() -> tuple[Any, Any, Any]:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return px, go, make_subplots


def _write_plotly(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    figure.write_json(path.with_suffix(".plotly.json"))


def _prepare_data(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().reset_index(drop=True)
    numeric = {
        "cited",
        "has_table",
        "word_count",
        "link_count",
        "log2_word_count_plus1",
        *DESCRIPTIVE_FEATURES,
    }
    for column in numeric.intersection(data.columns):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for column in (
        "prompt_id",
        "normalized_url",
        "source_root_domain",
        "heading_count_group",
        "link_count_group",
        "content_strength",
        "feature_extraction_text_scope",
        GEMINI_PAGE_FAMILY,
        GEMINI_PAGE_FAMILY_COLLAPSED,
        GEMINI_SOURCE_TYPE,
        GEMINI_SOURCE_TYPE_COLLAPSED,
        "page_type_url_seed_general_collapsed",
    ):
        if column in data:
            data[column] = data[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
    return data


def _normalise_term(term: str) -> str:
    return str(term).replace("_winsorized_p99", "")


def _preferred_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table.copy()
    work = table[table["std_error"].notna()].copy()
    work["_rank"] = work["cov_type"].map(COVARIANCE_ORDER).fillna(99)
    unreliable_two_way = (
        work["cov_type"].eq("two_way_cluster_prompt_url")
        & work["warning"].astype(str).str.contains("negative diagonal", case=False, na=False)
    )
    work.loc[unreliable_two_way, "_rank"] += 50
    return (
        work.sort_values("_rank", kind="stable")
        .drop_duplicates(["model_id", "term"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def fit_lpm(
    formula: str,
    data: pd.DataFrame,
    model_id: str,
    cov_type: str = "HC3",
) -> pd.DataFrame:
    """Fit one LPM using notebook 09's shared estimator."""
    from src.econometrics_eda_v2.content_feature_econometrics import fit_lpm as shared_fit_lpm

    return tidy_results(shared_fit_lpm(formula, data, model_id, cov_type=cov_type))


def tidy_results(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize coefficient outputs to notebook 11's reporting schema."""
    out = table.copy()
    if "warning" not in out:
        out["warning"] = out.get("notes", "")
    if "model_status" not in out:
        out["model_status"] = "completed"
    return out


def focal_terms_only(
    table: pd.DataFrame,
    features: Iterable[str] = FOCAL_FEATURES,
) -> pd.DataFrame:
    """Return focal continuous/table terms, including winsorized aliases."""
    allowed = set(features)
    work = table.copy()
    return work[work["term"].map(_normalise_term).isin(allowed)].copy()


def compare_covariance_types(table: pd.DataFrame) -> pd.DataFrame:
    """Classify covariance sensitivity for each model and focal term."""
    rows: list[dict[str, Any]] = []
    focal = focal_terms_only(table)
    for (model_id, term), group in focal.groupby(["model_id", "term"], sort=False):
        available = group[group["std_error"].notna()].copy()
        signs = set(np.sign(available["estimate_pp"].dropna()))
        excludes_zero = (
            (available["conf_low_pp"] > 0) | (available["conf_high_pp"] < 0)
        )
        two_way = group[group["cov_type"].eq("two_way_cluster_prompt_url")]
        unreliable_two_way = (
            two_way.empty
            or two_way["std_error"].isna().any()
            or two_way["warning"].astype(str).str.contains("negative diagonal", case=False).any()
        )
        if len(signs) > 1:
            classification = "direction_sensitive"
        elif unreliable_two_way:
            classification = "unreliable_two_way"
        elif len(set(excludes_zero.astype(bool))) > 1:
            classification = "significance_sensitive"
        else:
            classification = "stable"
        for row in group.itertuples(index=False):
            values = row._asdict()
            values["se_robustness_classification"] = classification
            values["available_covariance_types"] = int(available["cov_type"].nunique())
            rows.append(values)
    return pd.DataFrame(rows)


def save_model_result(
    formula: str,
    data: pd.DataFrame,
    model_id: str,
    path: Path,
    *,
    cov_types: Iterable[str] = COVARIANCE_TYPES,
    notes: str = "",
) -> Any:
    """Run and save a model through notebook 09's shared model runner."""
    from src.econometrics_eda_v2.content_feature_econometrics import run_model_and_save

    return run_model_and_save(
        formula,
        data,
        model_id,
        path,
        cov_types=cov_types,
        notes=notes,
    )


def make_forest_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    """Write an interactive forest plot for notebook 11 focal features."""
    _, go, _ = _plotly_modules()
    work = _preferred_rows(focal_terms_only(table))
    if work.empty:
        figure = go.Figure().add_annotation(text="No focal coefficients available", showarrow=False)
        _write_plotly(figure, path)
        return
    labels = {
        "has_table": "Has table",
        "factual_numeric_density_score": "Factual/numeric density",
        "price_unit_detail_score": "Price/unit detail",
        "location_transit_specificity_score": "Location/transit specificity",
        "prompt_page_relevance_score": "Prompt-page relevance",
        "amenity_project_detail_score": "Amenity/project detail",
        "external_evidence_score": "External evidence",
    }
    work["label"] = (
        work["model_id"].astype(str)
        + " | "
        + work["term"].map(_normalise_term).map(labels).fillna(work["term"])
    )
    figure = go.Figure(
        go.Scatter(
            x=work["estimate_pp"],
            y=work["label"],
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": work["conf_high_pp"] - work["estimate_pp"],
                "arrayminus": work["estimate_pp"] - work["conf_low_pp"],
            },
            marker={"color": "#277da1", "size": 9},
            customdata=np.column_stack(
                [work["conf_low_pp"], work["conf_high_pp"], work["p_value"], work["cov_type"]]
            ),
            hovertemplate=(
                "%{y}<br>Estimate=%{x:.2f} pp"
                "<br>95% CI=%{customdata[0]:.2f} to %{customdata[1]:.2f} pp"
                "<br>p=%{customdata[2]:.4f}<br>%{customdata[3]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#606b73")
    figure.update_layout(
        title=title,
        xaxis_title="Conditional association (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=max(500, 30 * len(work) + 160),
        margin={"l": 310, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(figure, path)


def classify_has_table_path(path_table: pd.DataFrame) -> tuple[str, str]:
    """Classify the descriptive coefficient path and its precision."""
    ordered = path_table.sort_values("path_order", kind="stable")
    estimates = ordered["estimate_pp"].dropna().to_numpy()
    if len(estimates) < 2:
        return "stable", "imprecise"
    if np.any(np.sign(estimates) != np.sign(estimates[0])):
        coefficient_pattern = "sign_reversal"
    elif abs(estimates[-1] - estimates[0]) < 1:
        coefficient_pattern = "stable"
    elif abs(estimates[-1]) < abs(estimates[0]):
        coefficient_pattern = "attenuated"
    else:
        coefficient_pattern = "amplified"
    includes_zero = ~(
        (ordered["conf_low_pp"] > 0) | (ordered["conf_high_pp"] < 0)
    )
    precision = "imprecise" if includes_zero.mean() > 0.5 else "mixed_or_precise"
    return coefficient_pattern, precision


def make_has_table_path_plot(path_table: pd.DataFrame, path: Path) -> None:
    """Write the T0-T4 descriptive has-table coefficient path."""
    _, go, _ = _plotly_modules()
    work = path_table.sort_values("path_order", kind="stable")
    figure = go.Figure(
        go.Scatter(
            x=work["model_id"],
            y=work["estimate_pp"],
            mode="lines+markers+text",
            text=[f"{value:.2f} pp" for value in work["estimate_pp"]],
            textposition="top center",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": work["conf_high_pp"] - work["estimate_pp"],
                "arrayminus": work["estimate_pp"] - work["conf_low_pp"],
            },
            marker={"color": "#43aa8b", "size": 10},
        )
    )
    figure.add_hline(y=0, line_dash="dash", line_color="#606b73")
    figure.update_layout(
        title="Descriptive has-table coefficient path",
        xaxis_title="Proxy/attenuation ladder",
        yaxis_title="Has-table coefficient (percentage points)",
        template="plotly_white",
        height=520,
    )
    _write_plotly(figure, path)


def _alias_table(
    table: pd.DataFrame,
    model_id: str,
    formula: str,
    note: str,
) -> pd.DataFrame:
    out = table.copy()
    out["model_id"] = model_id
    out["formula"] = formula
    out["warning"] = out["warning"].fillna("").astype(str).map(
        lambda value: "; ".join(part for part in (value, note) if part)
    )
    return out


def _fit_group(
    formulas: dict[str, str],
    data: pd.DataFrame,
    output_path: Path,
    *,
    notes: str,
    cov_types: Iterable[str] = COVARIANCE_TYPES,
) -> tuple[pd.DataFrame, list[str]]:
    tables: list[pd.DataFrame] = []
    warnings_list: list[str] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for model_id, formula in formulas.items():
        temporary = output_path.parent / f".{model_id}_working.csv"
        run = save_model_result(
            formula,
            data,
            model_id,
            temporary,
            cov_types=cov_types,
            notes=notes,
        )
        tables.append(tidy_results(run.table))
        warnings_list.extend(run.warnings)
        temporary.unlink(missing_ok=True)
        pd.concat(tables, ignore_index=True).to_csv(output_path, index=False)
    return pd.concat(tables, ignore_index=True), warnings_list


def _eta_squared(categories: pd.Series, values: pd.Series) -> float:
    work = pd.DataFrame({"category": categories, "value": values}).dropna()
    if work.empty or work["value"].nunique() <= 1:
        return np.nan
    overall = work["value"].mean()
    denominator = ((work["value"] - overall) ** 2).sum()
    if denominator <= 0:
        return np.nan
    numerator = sum(
        len(group) * (group["value"].mean() - overall) ** 2
        for _, group in work.groupby("category", dropna=False)
    )
    return float(numerator / denominator)


def _feature_descriptives(
    data: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for feature in DESCRIPTIVE_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        available = values.dropna()
        summary_rows.append(
            {
                "row_type": "overall",
                "feature": feature,
                "quartile": "",
                "n": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "mean": available.mean(),
                "median": available.median(),
                "p10": available.quantile(0.10),
                "p90": available.quantile(0.90),
                "max": available.max(),
                "cited_rows": np.nan,
                "cited_rate": np.nan,
            }
        )
        try:
            bins = pd.qcut(values, q=4, duplicates="drop")
        except ValueError:
            bins = pd.Series(pd.NA, index=data.index, dtype="object")
        work = pd.DataFrame({"bin": bins, "cited": data["cited"]}).dropna(subset=["bin"])
        for order, (category, group) in enumerate(work.groupby("bin", observed=True, sort=True), start=1):
            quartile = f"Q{order}"
            row = {
                "row_type": "quartile",
                "feature": feature,
                "quartile": quartile,
                "interval": str(category),
                "n": int(len(group)),
                "missing": 0,
                "mean": np.nan,
                "median": np.nan,
                "p10": np.nan,
                "p90": np.nan,
                "max": np.nan,
                "cited_rows": int(group["cited"].sum()),
                "cited_rate": float(group["cited"].mean()),
                "quartile_order": order,
            }
            summary_rows.append(row)
            quartile_rows.append(row)
        correlation_rows.append(
            {
                "feature": feature,
                "correlation_with_has_table": values.corr(data["has_table"]),
                "correlation_with_word_count": values.corr(
                    pd.to_numeric(data["word_count"], errors="coerce")
                ),
                "page_type_eta_squared": _eta_squared(
                    data[GEMINI_PAGE_FAMILY_COLLAPSED],
                    values,
                ),
                "page_type_measure": "eta_squared",
                "notes": (
                    "Eta-squared summarizes between-page-function variation; it is not a linear correlation."
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    correlations = pd.DataFrame(correlation_rows)
    summary.to_csv(table_dir / "11_feature_distribution_summary.csv", index=False)
    correlations.to_csv(table_dir / "11_feature_correlation_with_has_table.csv", index=False)

    px, _, _ = _plotly_modules()
    long = data[list(DESCRIPTIVE_FEATURES)].melt(var_name="feature", value_name="value")
    figure = px.histogram(
        long,
        x="value",
        facet_col="feature",
        facet_col_wrap=2,
        nbins=35,
        color_discrete_sequence=["#277da1"],
        template="plotly_white",
        title="Writing/factual feature distributions",
    )
    figure.update_yaxes(matches=None)
    figure.update_xaxes(matches=None)
    figure.update_layout(height=1100, showlegend=False)
    _write_plotly(figure, figure_dir / "11_feature_distributions.html")

    quartiles = pd.DataFrame(quartile_rows)
    quartiles["cited_rate_pct"] = quartiles["cited_rate"] * 100
    figure = px.line(
        quartiles,
        x="quartile",
        y="cited_rate_pct",
        color="feature",
        markers=True,
        category_orders={"quartile": ["Q1", "Q2", "Q3", "Q4"]},
        labels={"cited_rate_pct": "Cited rate (%)", "quartile": "Observed feature quartile"},
        template="plotly_white",
        title="Unadjusted cited rate by feature quartile",
    )
    figure.update_layout(height=580)
    _write_plotly(figure, figure_dir / "11_cited_rate_by_feature_quartile.html")

    box_long = data[["has_table", *MAIN_FEATURES, *SENSITIVITY_FEATURES]].melt(
        id_vars="has_table",
        var_name="feature",
        value_name="value",
    )
    box_long["has_table"] = box_long["has_table"].map({0: "No detected table", 1: "Detected table"})
    figure = px.box(
        box_long,
        x="has_table",
        y="value",
        facet_col="feature",
        facet_col_wrap=2,
        points=False,
        color="has_table",
        template="plotly_white",
        title="Writing/factual feature distributions by detected table presence",
    )
    figure.update_yaxes(matches=None)
    figure.update_layout(height=950, showlegend=False)
    _write_plotly(figure, figure_dir / "11_has_table_feature_boxplots.html")
    return summary, correlations


def _readiness_outputs(data: pd.DataFrame, table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {
        "n_rows": 5264,
        "unique_normalized_url": 2600,
        "unique_prompt_id": 498,
        "unique_source_root_domain": 541,
        "cited_rows": 1708,
        "cited_rate": 1708 / 5264,
    }
    actual = {
        "n_rows": len(data),
        "unique_normalized_url": data["normalized_url"].nunique(),
        "unique_prompt_id": data["prompt_id"].nunique(),
        "unique_source_root_domain": data["source_root_domain"].nunique(),
        "cited_rows": int(data["cited"].sum()),
        "cited_rate": float(data["cited"].mean()),
    }
    rows = []
    for metric, value in actual.items():
        expected_value = expected[metric]
        matches = (
            np.isclose(value, expected_value, atol=0.0005, rtol=0)
            if metric == "cited_rate"
            else value == expected_value
        )
        rows.append(
            {
                "section": "sample_count",
                "metric": metric,
                "category": "",
                "value": value,
                "expected_value": expected_value,
                "status": "match" if matches else "differs_but_continue",
            }
        )
    for scope, group in data.groupby("feature_extraction_text_scope", dropna=False):
        rows.append(
            {
                "section": "feature_extraction_text_scope_rows",
                "metric": "row_count",
                "category": str(scope),
                "value": len(group),
                "expected_value": np.nan,
                "status": "observed",
            }
        )
        rows.append(
            {
                "section": "feature_extraction_text_scope_urls",
                "metric": "unique_urls",
                "category": str(scope),
                "value": group["normalized_url"].nunique(),
                "expected_value": np.nan,
                "status": "observed",
            }
        )
    for feature in (*MAIN_FEATURES, *SENSITIVITY_FEATURES, *DIAGNOSTIC_FEATURES):
        rows.append(
            {
                "section": "new_feature_missingness",
                "metric": feature,
                "category": "",
                "value": float(data[feature].isna().mean()) if feature in data else 1.0,
                "expected_value": 0.0,
                "status": "observed" if feature in data else "missing",
            }
        )
    readiness = pd.DataFrame(rows)
    readiness.to_csv(table_dir / "11_dataset_readiness_summary.csv", index=False)

    required_rows = []
    for feature in REQUIRED_FEATURES:
        present = feature in data
        required_rows.append(
            {
                "feature": feature,
                "present": present,
                "nonmissing_rows": int(data[feature].notna().sum()) if present else 0,
                "unique_values": int(data[feature].nunique(dropna=True)) if present else 0,
                "status": "pass" if present and data[feature].notna().any() else "fail",
            }
        )
    cited_binary = "cited" in data and set(data["cited"].dropna().unique()).issubset({0, 1})
    required_rows.append(
        {
            "feature": "cited_binary_validation",
            "present": "cited" in data,
            "nonmissing_rows": int(data["cited"].notna().sum()) if "cited" in data else 0,
            "unique_values": int(data["cited"].nunique(dropna=True)) if "cited" in data else 0,
            "status": "pass" if cited_binary else "fail",
        }
    )
    required = pd.DataFrame(required_rows)
    required.to_csv(table_dir / "11_required_feature_check.csv", index=False)
    return readiness, required


def formula_guardrail_matches(formula: str) -> list[str]:
    """Return forbidden notebook 11 tokens found on the formula RHS."""
    rhs = formula.split("~", 1)[-1].casefold()
    return sorted(
        token
        for token in FORBIDDEN_NOTEBOOK11_TOKENS
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", rhs)
    )


def build_leakage_scope_guardrail(
    data: pd.DataFrame,
    formulas: dict[str, str],
    package: Path,
) -> pd.DataFrame:
    """Build the hard leakage and extraction-scope checks."""
    rows: list[dict[str, Any]] = []
    prior_path = (
        package
        / "tables/10_writing_factual_density_features/writing_factual_feature_leakage_check.csv"
    )
    prior_pass = False
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        prior_pass = not prior["status"].eq("fail").any()
    rows.append(
        {
            "check": "notebook10_feature_construction_guardrail",
            "matches": "none" if prior_pass else "missing_or_failed_prior_guardrail",
            "status": "pass" if prior_pass else "fail",
            "details": str(prior_path),
        }
    )
    for model_id, formula in formulas.items():
        matches = formula_guardrail_matches(formula)
        rows.append(
            {
                "check": f"formula_scan:{model_id}",
                "matches": "; ".join(matches) if matches else "none",
                "status": "fail" if matches else "pass",
                "details": formula,
            }
        )
    main_formula = formulas["W3_structural_plus_writing_factual"]
    rows.extend(
        [
            {
                "check": "no_final_enriched_page_type_in_headline",
                "matches": "none" if "page_type_family_general" not in main_formula else "page_type_family_general",
                "status": "pass" if "page_type_family_general" not in main_formula else "fail",
                "details": main_formula,
            },
            {
                "check": "prompt_page_relevance_uses_prompt_and_page_only",
                "matches": "none",
                "status": "pass" if prior_pass else "fail",
                "details": "Validated by notebook 10 deterministic feature dictionary and leakage check.",
            },
            {
                "check": "feature_extraction_text_scope_documented",
                "matches": "feature_extraction_text_scope",
                "status": "pass" if "feature_extraction_text_scope" in data else "fail",
                "details": (
                    "Excerpt-derived zero means not observed in captured text, not proof of full-page absence."
                ),
            },
            {
                "check": "outcome_not_used_to_construct_features",
                "matches": "none",
                "status": "pass" if prior_pass else "fail",
                "details": "Notebook 10 extraction did not receive cited as a feature-construction input.",
            },
        ]
    )
    return pd.DataFrame(rows)


def _outlier_audit_and_winsorize(data: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    rows = []
    for feature in OUTLIER_FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        p99 = values.quantile(0.99)
        rows.append(
            {
                "feature": feature,
                "n": int(values.notna().sum()),
                "p95": values.quantile(0.95),
                "p99": p99,
                "max": values.max(),
                "rows_above_p99": int(values.gt(p99).sum()),
                "maximum_to_p99_ratio": values.max() / p99 if pd.notna(p99) and p99 != 0 else np.nan,
                "sensitivity_action": (
                    "remove_top_1pct_and_winsorize"
                    if feature in MAIN_FEATURES[:3]
                    else "audit_only"
                ),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(table_dir / "11_outlier_distribution_audit.csv", index=False)
    return audit


def _subset_supported(data: pd.DataFrame) -> bool:
    return (
        len(data) >= 200
        and data["prompt_id"].nunique() >= 30
        and data["normalized_url"].nunique() >= 100
        and data["cited"].sum() >= 20
        and data["cited"].eq(0).sum() >= 20
    )


def _has_table_path(table: pd.DataFrame) -> pd.DataFrame:
    preferred = _preferred_rows(table)
    path = preferred[preferred["term"].eq("has_table")].copy()
    order = {model_id: index for index, model_id in enumerate(T_FORMULAS)}
    path["path_order"] = path["model_id"].map(order)
    path = path.sort_values("path_order", kind="stable")
    baseline = path.iloc[0]["estimate_pp"]
    path["coefficient_change_from_T0_pp"] = path["estimate_pp"] - baseline
    path["absolute_change_from_T0_pp"] = path["coefficient_change_from_T0_pp"].abs()
    coefficient_pattern, precision = classify_has_table_path(path)
    path["proxy_path_pattern"] = coefficient_pattern
    path["precision_pattern"] = precision
    path["interpretation"] = "descriptive coefficient path; suggestive of omitted structure, not mediation"
    return path


def _se_forest(table: pd.DataFrame, path: Path) -> None:
    _, go, _ = _plotly_modules()
    work = focal_terms_only(table)
    labels = {
        feature: feature.replace("_score", "").replace("_", " ").title()
        for feature in FOCAL_FEATURES
    }
    work["label"] = (
        work["model_id"].astype(str)
        + " | "
        + work["term"].map(_normalise_term).map(labels).fillna(work["term"])
    )
    figure = go.Figure()
    for cov_type, group in work.groupby("cov_type", sort=False):
        available = group[group["std_error"].notna()]
        figure.add_trace(
            go.Scatter(
                x=available["estimate_pp"],
                y=available["label"],
                mode="markers",
                name=cov_type,
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": available["conf_high_pp"] - available["estimate_pp"],
                    "arrayminus": available["estimate_pp"] - available["conf_low_pp"],
                },
            )
        )
    figure.add_vline(x=0, line_dash="dash", line_color="#606b73")
    figure.update_layout(
        title="Focal-term covariance estimator comparison",
        xaxis_title="Conditional association (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=max(600, len(work["label"].unique()) * 30 + 160),
        margin={"l": 350, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(figure, path)


def _first_result(
    table: pd.DataFrame,
    feature: str,
    model_id: str | None = None,
) -> pd.Series | None:
    if table.empty:
        return None
    work = table[table["term"].map(_normalise_term).eq(feature)]
    if model_id is not None:
        work = work[work["model_id"].eq(model_id)]
    if work.empty:
        return None
    return _preferred_rows(work).iloc[0]


def _format_result(result: pd.Series | None) -> str:
    if result is None:
        return "not_estimated"
    return (
        f"{result['estimate_pp']:.2f} pp "
        f"[{result['conf_low_pp']:.2f}, {result['conf_high_pp']:.2f}], "
        f"{result['cov_type']}"
    )


def _result_range(table: pd.DataFrame, feature: str) -> str:
    preferred = _preferred_rows(table)
    values = preferred[
        preferred["term"].map(_normalise_term).eq(feature)
    ]["estimate_pp"].dropna()
    if values.empty:
        return "not_estimated"
    return f"{values.min():.2f} to {values.max():.2f} pp across {len(values)} specifications"


def _robustness_classification(
    f_table: pd.DataFrame,
    w_table: pd.DataFrame,
    t_table: pd.DataFrame,
    d_table: pd.DataFrame,
    p_table: pd.DataFrame,
    s_table: pd.DataFrame,
    o_table: pd.DataFrame,
    se_table: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    source_tables = {
        "one_feature": f_table,
        "joint": w_table,
        "table_proxy": t_table,
        "domain_fe": d_table,
        "page_function": p_table,
        "text_scope": s_table,
        "outlier": o_table,
    }
    for feature in FOCAL_FEATURES:
        evidence_parts = []
        for stage, table in source_tables.items():
            work = _preferred_rows(table)
            work = work[work["term"].map(_normalise_term).eq(feature)].copy()
            if not work.empty:
                work["evidence_stage"] = stage
                evidence_parts.append(work)
        evidence = pd.concat(evidence_parts, ignore_index=True) if evidence_parts else pd.DataFrame()
        estimates = evidence["estimate_pp"].dropna() if not evidence.empty else pd.Series(dtype=float)
        signs = set(np.sign(estimates[estimates.ne(0)]))
        significant_share = (
            float(
                (
                    (evidence["conf_low_pp"] > 0)
                    | (evidence["conf_high_pp"] < 0)
                ).mean()
            )
            if not evidence.empty
            else 0.0
        )

        base_model = (
            "W3_structural_plus_writing_factual"
            if feature in ("has_table", *MAIN_FEATURES)
            else "W2_expanded_writing_factual"
        )
        base = _first_result(w_table, feature, base_model)
        domain = _first_result(d_table, feature, "D_W3_domain_fe")
        full_text = _first_result(s_table, feature, "S_full_text_W3")
        excerpt = _first_result(s_table, feature, "S_excerpt_only_W3")
        outlier_values = _preferred_rows(o_table)
        outlier_values = outlier_values[
            outlier_values["term"].map(_normalise_term).eq(feature)
        ]["estimate_pp"].dropna()
        se_classes = set(
            se_table.loc[
                se_table["term"].map(_normalise_term).eq(feature),
                "se_robustness_classification",
            ].dropna()
        )

        domain_confounded = False
        if base is not None and domain is not None and abs(base["estimate_pp"]) > 0.5:
            domain_confounded = (
                np.sign(base["estimate_pp"]) != np.sign(domain["estimate_pp"])
                or abs(domain["estimate_pp"]) < 0.5 * abs(base["estimate_pp"])
            )
        excerpt_sensitive = (
            full_text is not None
            and excerpt is not None
            and np.sign(full_text["estimate_pp"]) != np.sign(excerpt["estimate_pp"])
        )
        outlier_sensitive = (
            base is not None
            and not outlier_values.empty
            and (
                any(np.sign(value) != np.sign(base["estimate_pp"]) for value in outlier_values)
                or (
                    outlier_values.max() - outlier_values.min()
                    > max(2.0, abs(base["estimate_pp"]))
                )
            )
        )

        if feature == "external_evidence_score":
            classification = "needs_extraction_fix"
        elif "direction_sensitive" in se_classes or len(signs) > 1 and len(estimates) >= 4:
            classification = "unstable_diagnostic"
        elif domain_confounded:
            classification = "domain_template_confounded"
        elif excerpt_sensitive:
            classification = "excerpt_sensitive"
        elif outlier_sensitive:
            classification = "outlier_sensitive"
        elif len(estimates) >= 5 and len(signs) <= 1 and significant_share >= 0.5:
            classification = "stable"
        else:
            classification = "suggestive"
        rows.append(
            {
                "feature": feature,
                "robustness_classification": classification,
                "evidence_estimates_n": len(estimates),
                "minimum_estimate_pp": estimates.min() if len(estimates) else np.nan,
                "maximum_estimate_pp": estimates.max() if len(estimates) else np.nan,
                "direction_set": ";".join(map(str, sorted(signs))) if signs else "none",
                "share_intervals_excluding_zero": significant_share,
                "domain_template_flag": domain_confounded,
                "excerpt_sensitivity_flag": excerpt_sensitive,
                "outlier_sensitivity_flag": outlier_sensitive,
                "se_classifications": ";".join(sorted(se_classes)) if se_classes else "unavailable",
                "notes": (
                    "Excerpt-based features remain measurement-limited; classification summarizes stability, "
                    "not causal credibility."
                ),
            }
        )
    return pd.DataFrame(rows)


def _recommended_wording(feature: str, classification: str) -> str:
    label = feature.replace("_score", "").replace("_", " ")
    if classification == "needs_extraction_fix":
        return f"{label} is not ready for substantive interpretation because captured links/text are incomplete."
    if classification == "domain_template_confounded":
        return f"The {label} association weakens within domains, consistent with publisher/template confounding."
    if classification == "excerpt_sensitive":
        return f"The {label} association differs by captured-text scope and should be described as excerpt-sensitive."
    if classification == "outlier_sensitive":
        return f"The {label} association is sensitive to extreme observations."
    if classification == "stable":
        return f"{label} is associated with citation probability conditional on surfaced sources across the tested specifications."
    if classification == "unstable_diagnostic":
        return f"{label} changes materially across specifications and remains diagnostic only."
    return f"The {label} result is suggestive but uncertain and remains conditional on surfaced sources."


def _minimum_reporting_table(
    u_table: pd.DataFrame,
    f_table: pd.DataFrame,
    w_table: pd.DataFrame,
    t_table: pd.DataFrame,
    d_table: pd.DataFrame,
    p_table: pd.DataFrame,
    s_table: pd.DataFrame,
    o_table: pd.DataFrame,
    se_table: pd.DataFrame,
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    f_models = dict(zip((*MAIN_FEATURES, *SENSITIVITY_FEATURES, *DIAGNOSTIC_FEATURES), F_FORMULAS))
    rows = []
    for feature in FOCAL_FEATURES:
        classification = robustness.loc[
            robustness["feature"].eq(feature), "robustness_classification"
        ].iloc[0]
        joint_model = (
            "W3_structural_plus_writing_factual"
            if feature == "has_table"
            else (
                "W1_main_writing_factual"
                if feature in MAIN_FEATURES
                else "W2_expanded_writing_factual"
            )
        )
        prompt_result = (
            _first_result(f_table, feature, f_models.get(feature))
            if feature != "has_table"
            else None
        )
        if feature == "has_table":
            prompt_result = _first_result(t_table, feature, "T0_has_table_prompt_fe")
        se_classes = se_table.loc[
            se_table["term"].map(_normalise_term).eq(feature),
            "se_robustness_classification",
        ].dropna()
        rows.append(
            {
                "feature": feature,
                "unadjusted_contrast": _format_result(_first_result(u_table, feature)),
                "prompt_fe_coefficient": _format_result(prompt_result),
                "joint_writing_factual_coefficient": _format_result(
                    _first_result(w_table, feature, joint_model)
                ),
                "structural_control_coefficient": _format_result(
                    _first_result(w_table, feature, "W3_structural_plus_writing_factual")
                ),
                "domain_fe_robustness": _format_result(
                    _first_result(d_table, feature, "D_W3_domain_fe")
                ),
                "page_function_sensitivity": _format_result(
                    _first_result(p_table, feature, "P_W3_gemini_taxonomy")
                ),
                "strong_content_sensitivity": _format_result(
                    _first_result(s_table, feature, "S_strong_content_W3")
                ),
                "full_text_sensitivity": _format_result(
                    _first_result(s_table, feature, "S_full_text_W3")
                ),
                "excerpt_only_sensitivity": _format_result(
                    _first_result(s_table, feature, "S_excerpt_only_W3")
                ),
                "outlier_sensitivity": _result_range(o_table, feature),
                "covariance_robustness": ";".join(sorted(set(se_classes))) if len(se_classes) else "unavailable",
                "final_interpretation_bucket": classification,
                "recommended_wording": _recommended_wording(feature, classification),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_bullets(table: pd.DataFrame, features: Iterable[str]) -> str:
    preferred = _preferred_rows(focal_terms_only(table, features))
    lines = []
    for row in preferred.itertuples(index=False):
        direction = "higher" if row.estimate_pp >= 0 else "lower"
        precision = (
            "interval excludes zero"
            if row.conf_low_pp > 0 or row.conf_high_pp < 0
            else "interval includes zero"
        )
        lines.append(
            f"- `{row.model_id}` / `{_normalise_term(row.term)}`: associated with {direction} "
            f"citation probability ({row.estimate_pp:.2f} pp; 95% CI {row.conf_low_pp:.2f} to "
            f"{row.conf_high_pp:.2f}; {row.cov_type}; {precision})."
        )
    return "\n".join(lines) or "- No focal estimate was available."


def _classification_lines(robustness: pd.DataFrame) -> str:
    return "\n".join(
        f"- `{row.feature}`: `{row.robustness_classification}`."
        for row in robustness.itertuples(index=False)
    )


def _write_reports(
    report_dir: Path,
    data: pd.DataFrame,
    f_table: pd.DataFrame,
    w_table: pd.DataFrame,
    path_table: pd.DataFrame,
    d_table: pd.DataFrame,
    p_table: pd.DataFrame,
    s_table: pd.DataFrame,
    o_table: pd.DataFrame,
    robustness: pd.DataFrame,
    warnings_list: list[str],
) -> tuple[Path, Path]:
    path_pattern = path_table["proxy_path_pattern"].iloc[0]
    precision_pattern = path_table["precision_pattern"].iloc[0]
    scope_counts = data["feature_extraction_text_scope"].value_counts()
    warning_lines = "\n".join(f"- {warning}" for warning in warnings_list) or "- No runtime warning."
    report = f"""# 11 Writing and Factual-Density Econometrics Report

## 1. Scope and estimand

The unit is one surfaced source appearance. The estimand is `P(cited = 1 | source surfaced in this audit)`. These models estimate conditional associations among surfaced sources, not causal effects of changing page content. The results are conditional on surfaced sources, not causal, and not web-wide.

## 2. Dataset and excerpt limitation

- Rows: {len(data):,}
- Unique normalized URLs: {data['normalized_url'].nunique():,}
- Unique prompts: {data['prompt_id'].nunique():,}
- Unique source-root domains: {data['source_root_domain'].nunique():,}
- Cited rows: {int(data['cited'].sum()):,}
- Cited rate: {data['cited'].mean():.2%}
- Full-text-equivalent row appearances: {int(scope_counts.get('full_text', 0)):,}
- Excerpt-only row appearances: {int(scope_counts.get('excerpt_only', 0)):,}

Most measures are excerpt-based features. A zero means a pattern was not observed in captured text; it is not proof of absence from the complete webpage.

## 3. Feature groups tested

The main candidates are factual/numeric density, price/unit detail, location/transit specificity, and prompt-page relevance. Amenity/project detail and external evidence are sensitivity candidates. Writing structure remains diagnostic because crawler-normalized previews often flatten paragraphs and lists.

## 4. Main one-feature screening

{_coefficient_bullets(f_table, (*MAIN_FEATURES, *SENSITIVITY_FEATURES, *DIAGNOSTIC_FEATURES))}

These estimates are screening associations only. Their signs cannot be translated into content-editing advice.

## 5. Joint writing/factual model

{_coefficient_bullets(w_table, FOCAL_FEATURES)}

The joint models assess direction, size, uncertainty, and specification stability. They do not identify a content intervention.

## 6. Has-table proxy ladder

The T0-T4 coefficient path is classified as `{path_pattern}` with `{precision_pattern}` inference. Table presence may proxy structured factual detail, but the current feature layer does not fully explain the table signal.

This is a proxy/attenuation pattern and descriptive coefficient path, suggestive of omitted structure, not mediation.

## 7. Domain-FE and Gemini taxonomy sensitivity

Domain fixed effects:

{_coefficient_bullets(d_table, FOCAL_FEATURES)}

Gemini page-family and source/site-type sensitivity (with rule-v2 comparison retained in the table):

{_coefficient_bullets(p_table, FOCAL_FEATURES)}

Attenuation under domain fixed effects is consistent with publisher, template, or stable domain differences. The taxonomy sensitivity uses `{GEMINI_PAGE_FAMILY_COLLAPSED}` and `{GEMINI_SOURCE_TYPE_COLLAPSED}` from `{GEMINI_TAXONOMY_VERSION}`. Since Gemini may use scraped page content, this remains a sensitivity specification rather than the headline writing model. The older rule-v2 URL-seed label is retained only as a robustness comparison.

## 8. Text-scope and strong-content sensitivity

{_coefficient_bullets(s_table, FOCAL_FEATURES)}

`content_strength` is extraction quality, not writing quality. Full-text and excerpt-only subsets test measurement sensitivity; they do not convert excerpt zeros into confirmed absence.

## 9. Outlier and SE robustness

{_coefficient_bullets(o_table, FOCAL_FEATURES)}

HC3, prompt-clustered, URL-clustered, and feasible two-way clustered uncertainty estimates are retained. When two-way covariance produces negative diagonal variance, that standard error is not used alone.

## 10. Robustness classification

{_classification_lines(robustness)}

## 11. What can be discussed

- Which captured writing/factual signals are associated with higher or lower citation probability among already surfaced sources.
- Whether those associations retain direction under prompt fixed effects, domain fixed effects, page-function sensitivity, text-scope restrictions, outlier checks, and alternative covariance estimators.
- Whether the descriptive `has_table` coefficient path is attenuated, amplified, stable, or sign-reversing after richer controls.

## 12. What cannot be claimed

- The models do not reveal an AI system's internal retrieval or ranking mechanism.
- This analysis does not imply rewriting will change citation outcomes.
- A captured-text zero is not proof of absence from the complete webpage.
- Coefficients do not establish that a content feature is preferred by an AI system.
- The surfaced-source estimand does not generalize to all webpages.

## 13. Recommendations for next extraction/final analysis

- Retrieve reliable full body text and preserve paragraph, list, table, and outbound-link structure for a larger share of URLs.
- Re-run the same pre-specified feature definitions on full text before treating excerpt-sensitive measures as substantive.
- Keep prompt fixed effects and URL clustering in the final write-up; retain domain and Gemini taxonomy controls as robustness checks.
- Describe robust results as conditional associations and keep unstable or extraction-limited features diagnostic.

## Runtime warnings

{warning_lines}
"""
    executive = f"""# 11 Writing and Factual-Density Econometrics Executive Summary

## Scope

Notebook 11 analyzes {len(data):,} surfaced source appearances from {data['prompt_id'].nunique():,} prompts and {data['normalized_url'].nunique():,} URLs. The estimand is citation probability conditional on a source already being surfaced. These models estimate conditional associations among surfaced sources, not causal effects of changing page content, and the findings are not web-wide.

## Main result

The descriptive `has_table` path is `{path_pattern}` and its uncertainty is `{precision_pattern}` across the T0-T4 ladder. Table presence may proxy structured factual detail, but the current feature layer does not fully explain the table signal. The pattern is not evidence of mediation and does not support a content-editing guarantee.

## New writing and factual features

{_classification_lines(robustness)}

The main candidates remain factual/numeric density, price/unit detail, location/transit specificity, and prompt-page relevance. Amenity/project detail is sensitivity-only. External evidence needs improved extraction because compact previews do not reliably preserve outbound links. Directions and intervals should be read from the minimum reporting table rather than converted into claims about what content an AI system favors.

## Extraction limitation

There are {int(scope_counts.get('excerpt_only', 0)):,} excerpt-only row appearances and {int(scope_counts.get('full_text', 0)):,} full-text-equivalent appearances. Most measures are therefore excerpt-based features. A zero is not proof of absence from a complete page, and scope-sensitive results should remain provisional.

## What to improve next

1. Capture full page text with preserved HTML structure, especially tables, headings, lists, and external links.
2. Recompute the same deterministic features without outcome tuning.
3. Preserve prompt fixed effects and repeated-URL clustering.
4. Report domain and Gemini page-family/source-type specifications as robustness checks; use rule-v2 only as a comparison.
5. State that results are conditional on surfaced sources, not causal, and do not imply rewriting will change citation outcomes.
"""
    report_path = report_dir / "11_writing_factual_density_econometrics_report.md"
    executive_path = report_dir / "11_writing_factual_density_econometrics_executive_summary.md"
    report_path.write_text(report, encoding="utf-8")
    executive_path.write_text(executive, encoding="utf-8")
    return report_path, executive_path


def _failure_manifest(
    report_dir: Path,
    input_path: Path,
    data: pd.DataFrame,
    formulas: dict[str, str],
    status: str,
    warnings_list: list[str],
) -> Path:
    path = report_dir / "11_writing_factual_density_econometrics_manifest.json"
    payload = {
        "input_file": str(input_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": len(data),
        "url_count": data["normalized_url"].nunique() if "normalized_url" in data else 0,
        "prompt_count": data["prompt_id"].nunique() if "prompt_id" in data else 0,
        "model_formulas": formulas,
        "covariance_estimators": list(COVARIANCE_TYPES),
        "output_files": [],
        "warnings": warnings_list,
        "final_status": status,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_writing_factual_density_econometrics(package_path: Path | str) -> dict[str, Any]:
    """Run notebook 11's complete writing/factual-density model ladder."""
    package = Path(package_path).resolve()
    input_path = package / "data/content_lpm_measurable_rows_with_writing_factual_features.csv"
    table_dir = package / "tables/11_writing_factual_density_econometrics"
    figure_dir = package / "figures/11_writing_factual_density_econometrics"
    report_dir = package / "reports/11_writing_factual_density_econometrics"
    for directory in (table_dir, figure_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Notebook 11 input not found: {input_path}")

    data, _, taxonomy_audit = attach_gemini_taxonomy(
        pd.read_csv(input_path, low_memory=False),
        package,
    )
    data = _prepare_data(data)
    pd.DataFrame([taxonomy_audit]).to_csv(
        table_dir / "11_gemini_taxonomy_join_audit.csv",
        index=False,
    )
    warnings_list: list[str] = []
    _, required = _readiness_outputs(data, table_dir)
    all_formulas = {
        "B0_notebook09_baseline": B0_FORMULA,
        **F_FORMULAS,
        **W_FORMULAS,
        **T_FORMULAS,
        "D_W3_domain_fe": f"{W3_FORMULA} + {DOMAIN_FE}",
        "P_W3_gemini_taxonomy": f"{W3_FORMULA} + {GEMINI_TAXONOMY_TERMS}",
        "P_W3_rule_v2_taxonomy": f"{W3_FORMULA} + {RULE_V2_PAGE_SEED}",
    }
    if required["status"].eq("fail").any():
        _failure_manifest(
            report_dir,
            input_path,
            data,
            all_formulas,
            "failed_missing_required_features",
            ["Required-feature validation failed."],
        )
        raise ValueError("Notebook 11 required-feature validation failed.")

    guardrail = build_leakage_scope_guardrail(data, all_formulas, package)
    guardrail.to_csv(table_dir / "11_leakage_and_scope_guardrail.csv", index=False)
    if guardrail["status"].eq("fail").any():
        _failure_manifest(
            report_dir,
            input_path,
            data,
            all_formulas,
            "failed_leakage_detected",
            ["Leakage or feature-scope guardrail failed."],
        )
        raise ValueError("Notebook 11 leakage guardrail failed.")

    _feature_descriptives(data, table_dir, figure_dir)
    _outlier_audit_and_winsorize(data, table_dir)

    b0_run = save_model_result(
        B0_FORMULA,
        data,
        "B0_notebook09_baseline",
        table_dir / "B0_notebook09_baseline_replication.csv",
        notes="Notebook 09 baseline replication on the notebook 11 dataset.",
    )
    b0 = tidy_results(b0_run.table)
    warnings_list.extend(b0_run.warnings)
    prior_path = package / "tables/09_content_feature_econometrics/M2_preferred_joint_lpm_results.csv"
    if prior_path.exists():
        prior = tidy_results(pd.read_csv(prior_path, low_memory=False))
        current_has_table = _first_result(b0, "has_table")
        prior_has_table = _first_result(prior, "has_table", "M2")
        if current_has_table is not None and prior_has_table is not None:
            difference = current_has_table["estimate_pp"] - prior_has_table["estimate_pp"]
            b0["notebook09_has_table_estimate_pp"] = prior_has_table["estimate_pp"]
            b0["replication_difference_pp"] = difference
            b0["replication_status"] = "close" if abs(difference) < 0.25 else "review"
    b0.to_csv(table_dir / "B0_notebook09_baseline_replication.csv", index=False)

    u_formulas = {f"U_{feature}": f"cited ~ {feature}" for feature in FOCAL_FEATURES}
    u_table, u_warnings = _fit_group(
        u_formulas,
        data,
        table_dir / "U_unadjusted_feature_results.csv",
        notes="Unadjusted descriptive LPM for the minimum reporting table.",
        cov_types=("HC3",),
    )
    warnings_list.extend(u_warnings)

    f_table, f_warnings = _fit_group(
        F_FORMULAS,
        data,
        table_dir / "F_one_feature_prompt_fe_results.csv",
        notes="One-feature prompt-FE screening association; not a final claim.",
    )
    warnings_list.extend(f_warnings)
    make_forest_plot(
        f_table,
        figure_dir / "F_one_feature_prompt_fe_forest.html",
        "One-feature writing/factual screening with prompt fixed effects",
    )

    w_table, w_warnings = _fit_group(
        W_FORMULAS,
        data,
        table_dir / "W_joint_writing_factual_results.csv",
        notes="Joint writing/factual LPM; conditional association among surfaced sources.",
    )
    warnings_list.extend(w_warnings)
    make_forest_plot(
        w_table,
        figure_dir / "W_joint_writing_factual_forest.html",
        "Joint writing/factual feature models",
    )

    t0_table, t0_warnings = _fit_group(
        {"T0_has_table_prompt_fe": T_FORMULAS["T0_has_table_prompt_fe"]},
        data,
        table_dir / ".T0_working.csv",
        notes="Descriptive has-table coefficient path.",
    )
    t2_t3_table, t23_warnings = _fit_group(
        {
            "T2_table_plus_factual_detail": T_FORMULAS["T2_table_plus_factual_detail"],
            "T3_table_plus_detail_relevance": T_FORMULAS["T3_table_plus_detail_relevance"],
        },
        data,
        table_dir / ".T2_T3_working.csv",
        notes="Descriptive proxy/attenuation ladder; not mediation.",
    )
    (table_dir / ".T0_working.csv").unlink(missing_ok=True)
    (table_dir / ".T2_T3_working.csv").unlink(missing_ok=True)
    t1 = _alias_table(
        b0,
        "T1_has_table_structural",
        T_FORMULAS["T1_has_table_structural"],
        "Reused the identical B0 equation.",
    )
    w3 = w_table[w_table["model_id"].eq("W3_structural_plus_writing_factual")]
    t4 = _alias_table(
        w3,
        "T4_table_structural_detail_relevance",
        T_FORMULAS["T4_table_structural_detail_relevance"],
        "Reused the identical W3 equation.",
    )
    t_table = pd.concat([t0_table, t1, t2_t3_table, t4], ignore_index=True)
    t_table.to_csv(table_dir / "T_has_table_proxy_ladder.csv", index=False)
    warnings_list.extend([*t0_warnings, *t23_warnings])
    path_table = _has_table_path(t_table)
    path_table.to_csv(table_dir / "T_has_table_coefficient_path_summary.csv", index=False)
    make_has_table_path_plot(
        path_table,
        figure_dir / "T_has_table_coefficient_path.html",
    )

    domain_url_counts = data.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = domain_url_counts[domain_url_counts.ge(2)].index
    domain_data = data[data["source_root_domain"].isin(supported_domains)].copy()
    d_base, d_warnings = _fit_group(
        {"D_W3_domain_fe": f"{W3_FORMULA} + {DOMAIN_FE}"},
        domain_data,
        table_dir / ".D_working.csv",
        notes=(
            f"Domain FE robustness; retained {len(domain_data)} rows from "
            f"{domain_data['source_root_domain'].nunique()} domains with >=2 unique URLs."
        ),
    )
    (table_dir / ".D_working.csv").unlink(missing_ok=True)
    d_alias = _alias_table(
        d_base,
        "D_T4_domain_fe",
        f"{T_FORMULAS['T4_table_structural_detail_relevance']} + {DOMAIN_FE}",
        "Reused the identical domain-FE W3 equation.",
    )
    d_table = pd.concat([d_base, d_alias], ignore_index=True)
    d_table.to_csv(table_dir / "D_domain_fe_writing_factual_results.csv", index=False)
    warnings_list.extend(d_warnings)

    p_base, p_warnings = _fit_group(
        {"P_W3_gemini_taxonomy": f"{W3_FORMULA} + {GEMINI_TAXONOMY_TERMS}"},
        data,
        table_dir / ".P_working.csv",
        notes=(
            f"Gemini taxonomy sensitivity using {GEMINI_PAGE_FAMILY_COLLAPSED} and "
            f"{GEMINI_SOURCE_TYPE_COLLAPSED}; taxonomy version={GEMINI_TAXONOMY_VERSION}."
        ),
    )
    (table_dir / ".P_working.csv").unlink(missing_ok=True)
    p_alias = _alias_table(
        p_base,
        "P_T4_gemini_taxonomy",
        f"{T_FORMULAS['T4_table_structural_detail_relevance']} + {GEMINI_TAXONOMY_TERMS}",
        "Reused the identical Gemini-taxonomy W3 equation.",
    )
    p_rule, p_rule_warnings = _fit_group(
        {"P_W3_rule_v2_taxonomy": f"{W3_FORMULA} + {RULE_V2_PAGE_SEED}"},
        data,
        table_dir / ".P_rule_working.csv",
        notes="Legacy rule-v2 URL-seed taxonomy retained only as a robustness comparison.",
    )
    (table_dir / ".P_rule_working.csv").unlink(missing_ok=True)
    p_table = pd.concat([p_base, p_alias, p_rule], ignore_index=True)
    p_table.to_csv(table_dir / "P_page_function_sensitivity_results.csv", index=False)
    warnings_list.extend([*p_warnings, *p_rule_warnings])

    sensitivity_tables: list[pd.DataFrame] = []
    sensitivity_support_rows = []
    for label, subset in (
        ("strong_content", data[data["content_strength"].eq("strong")].copy()),
        ("full_text", data[data["feature_extraction_text_scope"].eq("full_text")].copy()),
        ("excerpt_only", data[data["feature_extraction_text_scope"].eq("excerpt_only")].copy()),
    ):
        supported = _subset_supported(subset)
        sensitivity_support_rows.append(
            {
                "sensitivity": label,
                "n_rows": len(subset),
                "n_prompts": subset["prompt_id"].nunique(),
                "n_urls": subset["normalized_url"].nunique(),
                "cited_rows": int(subset["cited"].sum()),
                "more_only_rows": int(subset["cited"].eq(0).sum()),
                "supported": supported,
                "status": "run" if supported else "insufficient_support",
            }
        )
        if not supported:
            warnings_list.append(f"{label} sensitivity skipped for insufficient support.")
            continue
        model_id = f"S_{label}_W3"
        fitted, fitted_warnings = _fit_group(
            {model_id: W3_FORMULA},
            subset,
            table_dir / f".{model_id}_working.csv",
            notes=f"Text-scope/content-strength sensitivity: {label}.",
        )
        (table_dir / f".{model_id}_working.csv").unlink(missing_ok=True)
        sensitivity_tables.append(fitted)
        sensitivity_tables.append(
            _alias_table(
                fitted,
                f"S_{label}_T4",
                T_FORMULAS["T4_table_structural_detail_relevance"],
                "Reused the identical W3 equation for the T4 sensitivity path.",
            )
        )
        warnings_list.extend(fitted_warnings)
    s_table = pd.concat(sensitivity_tables, ignore_index=True) if sensitivity_tables else pd.DataFrame()
    s_table.to_csv(table_dir / "S_text_scope_content_strength_sensitivity.csv", index=False)
    pd.DataFrame(sensitivity_support_rows).to_csv(
        table_dir / "S_text_scope_content_strength_support.csv",
        index=False,
    )

    outlier_tables: list[pd.DataFrame] = []
    for feature in MAIN_FEATURES[:3]:
        threshold = data[feature].quantile(0.99)
        subset = data[data[feature].le(threshold)].copy()
        model_id = f"O_remove_top1pct_{feature}_W3"
        fitted, fitted_warnings = _fit_group(
            {model_id: W3_FORMULA},
            subset,
            table_dir / f".{model_id}_working.csv",
            notes=f"Removed rows above the p99 of {feature}.",
        )
        (table_dir / f".{model_id}_working.csv").unlink(missing_ok=True)
        outlier_tables.append(fitted)
        outlier_tables.append(
            _alias_table(
                fitted,
                model_id.replace("_W3", "_T4"),
                T_FORMULAS["T4_table_structural_detail_relevance"],
                "Reused the identical W3 equation for the T4 outlier path.",
            )
        )
        warnings_list.extend(fitted_warnings)
    winsorized = data.copy()
    winsor_mapping = {}
    for feature in MAIN_FEATURES:
        target = f"{feature}_winsorized_p99"
        winsorized[target] = winsorized[feature].clip(upper=winsorized[feature].quantile(0.99))
        winsor_mapping[feature] = target
    winsor_formula = W3_FORMULA
    for source, target in winsor_mapping.items():
        winsor_formula = winsor_formula.replace(source, target)
    winsor_fitted, winsor_warnings = _fit_group(
        {"O_winsorized_main_scores_W3": winsor_formula},
        winsorized,
        table_dir / ".O_winsorized_working.csv",
        notes="Main writing/factual scores winsorized at p99.",
    )
    (table_dir / ".O_winsorized_working.csv").unlink(missing_ok=True)
    outlier_tables.append(winsor_fitted)
    outlier_tables.append(
        _alias_table(
            winsor_fitted,
            "O_winsorized_main_scores_T4",
            winsor_formula,
            "Reused the identical W3 equation for the T4 outlier path.",
        )
    )
    warnings_list.extend(winsor_warnings)
    o_table = pd.concat(outlier_tables, ignore_index=True)
    o_table.to_csv(table_dir / "O_outlier_sensitivity_writing_factual_results.csv", index=False)

    se_source = pd.concat([w3, t4, d_table, p_table], ignore_index=True)
    se_table = compare_covariance_types(se_source)
    se_table.to_csv(table_dir / "SE_focal_term_covariance_comparison.csv", index=False)
    _se_forest(se_table, figure_dir / "SE_focal_term_covariance_forest.html")

    robustness = _robustness_classification(
        f_table,
        w_table,
        t_table,
        d_table,
        p_table,
        s_table,
        o_table,
        se_table,
    )
    robustness.to_csv(
        table_dir / "11_writing_factual_robustness_classification.csv",
        index=False,
    )
    minimum = _minimum_reporting_table(
        u_table,
        f_table,
        w_table,
        t_table,
        d_table,
        p_table,
        s_table,
        o_table,
        se_table,
        robustness,
    )
    minimum.to_csv(table_dir / "11_minimum_reporting_table.csv", index=False)

    report_path, executive_path = _write_reports(
        report_dir,
        data,
        f_table,
        w_table,
        path_table,
        d_table,
        p_table,
        s_table,
        o_table,
        robustness,
        warnings_list,
    )

    excerpt_share = data["feature_extraction_text_scope"].eq("excerpt_only").mean()
    final_status = (
        "completed_with_excerpt_limitations"
        if excerpt_share > 0.5
        else "completed_ready_for_final_content_writeup_with_caveats"
    )
    output_files = sorted(
        str(path.relative_to(package))
        for directory in (table_dir, figure_dir, report_dir)
        for path in directory.rglob("*")
        if path.is_file()
    )
    manifest_path = report_dir / "11_writing_factual_density_econometrics_manifest.json"
    manifest = {
        "input_file": str(input_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_count": len(data),
        "url_count": data["normalized_url"].nunique(),
        "prompt_count": data["prompt_id"].nunique(),
        "domain_count": data["source_root_domain"].nunique(),
        "cited_rows": int(data["cited"].sum()),
        "cited_rate": float(data["cited"].mean()),
        "feature_extraction_text_scope": data[
            "feature_extraction_text_scope"
        ].value_counts().to_dict(),
        "model_formulas": {
            **all_formulas,
            "O_winsorized_main_scores_W3": winsor_formula,
        },
        "covariance_estimators": list(COVARIANCE_TYPES),
        "output_files": sorted(
            {*output_files, str(manifest_path.relative_to(package))}
        ),
        "warnings": warnings_list,
        "has_table_proxy_path_pattern": path_table["proxy_path_pattern"].iloc[0],
        "has_table_precision_pattern": path_table["precision_pattern"].iloc[0],
        "final_status": final_status,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "input_file": str(input_path),
        "rows": len(data),
        "unique_urls": data["normalized_url"].nunique(),
        "unique_prompts": data["prompt_id"].nunique(),
        "unique_domains": data["source_root_domain"].nunique(),
        "cited_rows": int(data["cited"].sum()),
        "cited_rate": float(data["cited"].mean()),
        "excerpt_only_rows": int(data["feature_extraction_text_scope"].eq("excerpt_only").sum()),
        "full_text_rows": int(data["feature_extraction_text_scope"].eq("full_text").sum()),
        "has_table_proxy_path_pattern": path_table["proxy_path_pattern"].iloc[0],
        "has_table_precision_pattern": path_table["precision_pattern"].iloc[0],
        "leakage_check_passed": not guardrail["status"].eq("fail").any(),
        "report": str(report_path),
        "executive_summary": str(executive_path),
        "minimum_reporting_table": str(table_dir / "11_minimum_reporting_table.csv"),
        "manifest": str(manifest_path),
        "final_status": final_status,
    }
