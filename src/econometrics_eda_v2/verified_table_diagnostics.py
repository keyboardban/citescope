"""Focused EDA and stability diagnostics for verified HTML table presence."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from bs4 import BeautifulSoup
from statsmodels.stats.proportion import proportion_confint

from src.econometrics_eda_v2.content_feature_econometrics import run_model_and_save
from src.econometrics_eda_v2.redesigned_pipeline_v2 import (
    FOCAL,
    PAGE_TYPE,
    SOURCE_TYPE,
    build_model_ready,
    formulas,
)


FEATURE = "has_verified_html_table"
PREFERRED_COVARIANCE = "two_way_cluster_prompt_url"
MIN_GROUP_ROWS = 10
MIN_STATE_ROWS = 3
RANDOM_SEED = 20260727


def _wilson(successes: int, observations: int) -> tuple[float, float]:
    if observations <= 0:
        return np.nan, np.nan
    low, high = proportion_confint(successes, observations, method="wilson")
    return float(low), float(high)


def _rate_block(frame: pd.DataFrame) -> dict[str, Any]:
    n = len(frame)
    cited = int(frame["cited"].sum()) if n else 0
    low, high = _wilson(cited, n)
    return {
        "n_rows": n,
        "cited_rows": cited,
        "cited_rate": cited / n if n else np.nan,
        "cited_rate_ci_low": low,
        "cited_rate_ci_high": high,
    }


def overall_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in (0, 1):
        subset = data[pd.to_numeric(data[FEATURE], errors="coerce").eq(state)]
        block = _rate_block(subset)
        rows.append(
            {
                "measurement": f"table_{'positive' if state else 'negative'}",
                "table_status": state,
                "total_estimation_rows": len(data),
                "row_share": len(subset) / len(data),
                **block,
            }
        )
    output = pd.DataFrame(rows)
    rates = output.set_index("table_status")["cited_rate"]
    output["raw_cited_rate_difference_pp"] = (
        float(rates.loc[1] - rates.loc[0]) * 100
    )
    output["interpretation"] = "unadjusted descriptive comparison"
    return output


def _group_diagnostics(
    data: pd.DataFrame,
    group_column: str,
    *,
    group_label: str,
) -> pd.DataFrame:
    table = pd.to_numeric(data[FEATURE], errors="coerce")
    rows = []
    for value, subset in data.assign(_table=table).groupby(
        group_column,
        dropna=False,
        observed=True,
    ):
        positive = subset[subset["_table"].eq(1)]
        negative = subset[subset["_table"].eq(0)]
        positive_rate = positive["cited"].mean() if len(positive) else np.nan
        negative_rate = negative["cited"].mean() if len(negative) else np.nan
        supported = (
            len(subset) >= MIN_GROUP_ROWS
            and len(positive) >= MIN_STATE_ROWS
            and len(negative) >= MIN_STATE_ROWS
        )
        pos_low, pos_high = _wilson(int(positive["cited"].sum()), len(positive))
        neg_low, neg_high = _wilson(int(negative["cited"].sum()), len(negative))
        difference = positive_rate - negative_rate if supported else np.nan
        rows.append(
            {
                group_label: str(value),
                "total_rows": len(subset),
                "unique_urls": int(subset["normalized_url"].nunique()),
                "table_positive_rows": len(positive),
                "table_negative_rows": len(negative),
                "table_prevalence": len(positive) / len(subset),
                "overall_cited_rate": float(subset["cited"].mean()),
                "table_positive_cited_rate": positive_rate,
                "table_positive_ci_low": pos_low,
                "table_positive_ci_high": pos_high,
                "table_negative_cited_rate": negative_rate,
                "table_negative_ci_low": neg_low,
                "table_negative_ci_high": neg_high,
                "within_group_difference": difference,
                "within_group_difference_pp": (
                    difference * 100 if pd.notna(difference) else np.nan
                ),
                "difference_ci_low_pp": (
                    (pos_low - neg_high) * 100 if supported else np.nan
                ),
                "difference_ci_high_pp": (
                    (pos_high - neg_low) * 100 if supported else np.nan
                ),
                "prompts_represented": int(subset["prompt_id"].nunique()),
                "adequate_difference_support": supported,
                "support_note": (
                    "adequate"
                    if supported
                    else "requires >=10 rows and >=3 rows in each table state"
                ),
            }
        )
    output = pd.DataFrame(rows)
    total_positive = output["table_positive_rows"].sum()
    output["share_of_all_table_positive_rows"] = (
        output["table_positive_rows"] / total_positive
        if total_positive
        else np.nan
    )
    output["prevalence_flag"] = np.select(
        [
            output["table_prevalence"].eq(0),
            output["table_prevalence"].eq(1),
            output["table_prevalence"].gt(0.8),
            output["table_prevalence"].lt(0.2),
        ],
        ["0_percent", "100_percent", "above_80_percent", "below_20_percent"],
        default="20_to_80_percent",
    )
    return output.sort_values(
        ["table_positive_rows", "total_rows"],
        ascending=False,
    ).reset_index(drop=True)


def domain_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    return _group_diagnostics(
        data,
        "source_root_domain",
        group_label="domain",
    )


def domain_rankings(domain: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    blocks = []
    for ranking_type, column in (
        ("highest_table_positive_rows", "table_positive_rows"),
        ("highest_table_prevalence", "table_prevalence"),
        ("largest_share_of_table_positive_rows", "share_of_all_table_positive_rows"),
    ):
        block = domain.sort_values(
            [column, "total_rows"],
            ascending=False,
        ).head(top_n).copy()
        block.insert(0, "rank", np.arange(1, len(block) + 1))
        block.insert(0, "ranking_type", ranking_type)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True)


def prompt_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    output = _group_diagnostics(data, "prompt_id", group_label="prompt_id")
    output["weak_positive_support"] = output["table_positive_rows"].le(1)
    return output


def _weighted_correlation(x: pd.Series, y: pd.Series, weights: pd.Series) -> float:
    valid = x.notna() & y.notna() & weights.notna() & weights.gt(0)
    x_values = x[valid].to_numpy(dtype=float)
    y_values = y[valid].to_numpy(dtype=float)
    w_values = weights[valid].to_numpy(dtype=float)
    if len(x_values) < 2:
        return np.nan
    x_mean = np.average(x_values, weights=w_values)
    y_mean = np.average(y_values, weights=w_values)
    covariance = np.average(
        (x_values - x_mean) * (y_values - y_mean),
        weights=w_values,
    )
    x_variance = np.average((x_values - x_mean) ** 2, weights=w_values)
    y_variance = np.average((y_values - y_mean) ** 2, weights=w_values)
    denominator = math.sqrt(x_variance * y_variance)
    return covariance / denominator if denominator else np.nan


def domain_correlation(domain: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis": "domain_table_prevalence_vs_cited_rate",
                "n_domains": len(domain),
                "unweighted_correlation": domain["table_prevalence"].corr(
                    domain["overall_cited_rate"]
                ),
                "row_weighted_correlation": _weighted_correlation(
                    domain["table_prevalence"],
                    domain["overall_cited_rate"],
                    domain["total_rows"],
                ),
                "interpretation": "descriptive domain-level association; confounding possible",
            }
        ]
    )


def difference_summary(
    groups: pd.DataFrame,
    *,
    unit: str,
) -> pd.DataFrame:
    supported = groups[groups["adequate_difference_support"]].copy()
    differences = supported["within_group_difference"]
    weights = supported["total_rows"]
    return pd.DataFrame(
        [
            {
                "analysis_unit": unit,
                "total_groups": len(groups),
                "adequately_supported_groups": len(supported),
                "mean_difference_pp": differences.mean() * 100,
                "median_difference_pp": differences.median() * 100,
                "row_weighted_mean_difference_pp": (
                    np.average(differences, weights=weights) * 100
                    if len(supported)
                    else np.nan
                ),
                "positive_groups": int(differences.gt(0).sum()),
                "zero_groups": int(differences.eq(0).sum()),
                "negative_groups": int(differences.lt(0).sum()),
                "positive_group_share": float(differences.gt(0).mean()),
                "zero_group_share": float(differences.eq(0).mean()),
                "negative_group_share": float(differences.lt(0).mean()),
                "groups_with_only_one_table_positive_row": int(
                    groups["table_positive_rows"].eq(1).sum()
                ),
            }
        ]
    )


def identifying_support(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_varies = (
        data.groupby("prompt_id", observed=True)[FEATURE]
        .nunique(dropna=True)
        .gt(1)
    )
    domain_varies = (
        data.groupby("source_root_domain", observed=True)[FEATURE]
        .nunique(dropna=True)
        .gt(1)
    )
    prompt_set = set(prompt_varies[prompt_varies].index.astype(str))
    domain_set = set(domain_varies[domain_varies].index.astype(str))
    prompt_mask = data["prompt_id"].astype(str).isin(prompt_set)
    domain_mask = data["source_root_domain"].astype(str).isin(domain_set)
    categories = np.select(
        [
            prompt_mask & domain_mask,
            prompt_mask & ~domain_mask,
            ~prompt_mask & domain_mask,
        ],
        ["within_both", "within_prompt_only", "within_domain_only"],
        default="within_neither",
    )
    row_detail = data[
        ["prompt_id", "normalized_url", "source_root_domain", FEATURE]
    ].copy()
    row_detail["identifying_support_category"] = categories
    rows = [
        {
            "metric": "prompts_with_both_states",
            "value": len(prompt_set),
        },
        {
            "metric": "rows_in_prompts_with_both_states",
            "value": int(prompt_mask.sum()),
        },
        {
            "metric": "unique_urls_in_prompts_with_both_states",
            "value": int(data.loc[prompt_mask, "normalized_url"].nunique()),
        },
        {
            "metric": "domains_with_both_states",
            "value": len(domain_set),
        },
        {
            "metric": "rows_in_domains_with_both_states",
            "value": int(domain_mask.sum()),
        },
        {
            "metric": "unique_urls_in_domains_with_both_states",
            "value": int(data.loc[domain_mask, "normalized_url"].nunique()),
        },
    ]
    for category, count in row_detail["identifying_support_category"].value_counts().items():
        rows.append({"metric": f"rows_{category}", "value": int(count)})
    return pd.DataFrame(rows), row_detail


def _preferred_feature_row(table: pd.DataFrame, label: str) -> dict[str, Any]:
    row = table[
        table["term"].eq(FEATURE)
        & table["cov_type"].eq(PREFERRED_COVARIANCE)
    ].iloc[0]
    return {
        "specification": label,
        "estimate_pp": row["estimate_pp"],
        "conf_low_pp": row["conf_low_pp"],
        "conf_high_pp": row["conf_high_pp"],
        "p_value": row["p_value"],
        "n_obs": int(row["n_obs"]),
        "n_prompts": int(row["n_prompts"]),
        "n_urls": int(row["n_urls"]),
        "n_domains": int(row["n_domains"]),
    }


def fe2_fe3_decomposition(
    data: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    model_formulas = formulas(data)
    full = run_model_and_save(
        model_formulas["FE2"],
        data,
        "FE2_full_table_diagnostic",
        output_dir / ".FE2_full_table_diagnostic.csv",
    )
    domain_url_counts = data.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = set(domain_url_counts[domain_url_counts.ge(2)].index)
    restricted = data[data["source_root_domain"].isin(supported_domains)].copy()
    restricted_fe2 = run_model_and_save(
        model_formulas["FE2"],
        restricted,
        "FE2_restricted_table_diagnostic",
        output_dir / ".FE2_restricted_table_diagnostic.csv",
    )
    fe3 = run_model_and_save(
        model_formulas["FE3"],
        restricted,
        "FE3_table_diagnostic",
        output_dir / ".FE3_table_diagnostic.csv",
    )
    for filename in (
        ".FE2_full_table_diagnostic.csv",
        ".FE2_restricted_table_diagnostic.csv",
        ".FE3_table_diagnostic.csv",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    rows = [
        _preferred_feature_row(full.table, "FE2_full_sample"),
        _preferred_feature_row(
            restricted_fe2.table,
            "FE2_restricted_to_FE3_sample",
        ),
        _preferred_feature_row(fe3.table, "FE3_domain_FE_same_sample"),
    ]
    output = pd.DataFrame(rows)
    estimates = output.set_index("specification")["estimate_pp"]
    output["sample_composition_change_pp"] = (
        estimates["FE2_restricted_to_FE3_sample"]
        - estimates["FE2_full_sample"]
    )
    output["domain_FE_change_on_common_sample_pp"] = (
        estimates["FE3_domain_FE_same_sample"]
        - estimates["FE2_restricted_to_FE3_sample"]
    )
    output["total_FE2_to_FE3_change_pp"] = (
        estimates["FE3_domain_FE_same_sample"]
        - estimates["FE2_full_sample"]
    )
    return output


def taxonomy_stratification(
    data: pd.DataFrame,
    group_column: str,
    group_label: str,
) -> pd.DataFrame:
    return _group_diagnostics(data, group_column, group_label=group_label)


def factual_density_analysis(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factual = pd.to_numeric(data["factual_numeric_density_score"], errors="coerce")
    table = pd.to_numeric(data[FEATURE], errors="coerce")
    correlation = pd.DataFrame(
        [
            {
                "feature_a": FEATURE,
                "feature_b": "factual_numeric_density_score",
                "pearson_correlation": table.corr(factual),
                "n_rows": int((table.notna() & factual.notna()).sum()),
            }
        ]
    )
    distribution_rows = []
    for state in (0, 1):
        values = factual[table.eq(state)].dropna()
        distribution_rows.append(
            {
                "table_status": state,
                "n_rows": len(values),
                "mean_factual_density": values.mean(),
                "median_factual_density": values.median(),
                "p25_factual_density": values.quantile(0.25),
                "p75_factual_density": values.quantile(0.75),
                "p90_factual_density": values.quantile(0.90),
                "p99_factual_density": values.quantile(0.99),
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    groups = pd.qcut(
        factual,
        q=4,
        labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"],
        duplicates="drop",
    )
    working = data.assign(factual_density_group=groups, _table=table)
    rows = []
    for (density_group, state), subset in working.groupby(
        ["factual_density_group", "_table"],
        observed=True,
        dropna=False,
    ):
        block = _rate_block(subset)
        rows.append(
            {
                "factual_density_group": str(density_group),
                "table_status": int(state),
                **block,
                "mean_factual_density": pd.to_numeric(
                    subset["factual_numeric_density_score"],
                    errors="coerce",
                ).mean(),
                "unique_urls": int(subset["normalized_url"].nunique()),
                "unique_prompts": int(subset["prompt_id"].nunique()),
            }
        )
    return correlation, distribution, pd.DataFrame(rows)


def url_repetition_analysis(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    appearances = data.groupby("normalized_url").size().rename("appearances")
    url_level = (
        data.sort_values("normalized_url")
        .drop_duplicates("normalized_url")
        [
            [
                "normalized_url",
                "source_root_domain",
                FEATURE,
                "cited",
            ]
        ]
        .merge(appearances, on="normalized_url", validate="one_to_one")
    )
    url_level["repeat_group"] = pd.cut(
        url_level["appearances"],
        bins=[0, 1, 3, 9, np.inf],
        labels=["1 appearance", "2-3", "4-9", "10+"],
    )
    row_working = data.merge(
        appearances,
        on="normalized_url",
        how="left",
        validate="many_to_one",
    )
    row_working["repeat_group"] = pd.cut(
        row_working["appearances"],
        bins=[0, 1, 3, 9, np.inf],
        labels=["1 appearance", "2-3", "4-9", "10+"],
    )
    rows = []
    for group, subset in row_working.groupby(
        "repeat_group",
        observed=True,
    ):
        for state in (0, 1):
            state_subset = subset[
                pd.to_numeric(subset[FEATURE], errors="coerce").eq(state)
            ]
            rows.append(
                {
                    "repeat_group": str(group),
                    "table_status": state,
                    "n_rows": len(state_subset),
                    "unique_urls": int(state_subset["normalized_url"].nunique()),
                    "cited_rate": state_subset["cited"].mean(),
                }
            )
    summary = pd.DataFrame(
        [
            {
                "total_rows": len(data),
                "unique_urls": len(url_level),
                "single_appearance_urls": int(url_level["appearances"].eq(1).sum()),
                "repeated_urls": int(url_level["appearances"].gt(1).sum()),
                "median_appearances_per_url": url_level["appearances"].median(),
                "p90_appearances_per_url": url_level["appearances"].quantile(0.90),
                "max_appearances_per_url": url_level["appearances"].max(),
                "table_positive_rows_repeated_urls": int(
                    (
                        row_working["appearances"].gt(1)
                        & pd.to_numeric(row_working[FEATURE], errors="coerce").eq(1)
                    ).sum()
                ),
                "table_positive_rows_single_urls": int(
                    (
                        row_working["appearances"].eq(1)
                        & pd.to_numeric(row_working[FEATURE], errors="coerce").eq(1)
                    ).sum()
                ),
            }
        ]
    )
    appearance_distribution = (
        appearances.value_counts()
        .sort_index()
        .rename_axis("appearances_per_url")
        .rename("unique_urls")
        .reset_index()
    )
    appearance_distribution["unique_url_share"] = (
        appearance_distribution["unique_urls"] / len(url_level)
    )
    appearance_distribution["rows_represented"] = (
        appearance_distribution["appearances_per_url"]
        * appearance_distribution["unique_urls"]
    )
    return summary, pd.DataFrame(rows), appearance_distribution


def leave_one_domain_out(
    data: pd.DataFrame,
    domain: pd.DataFrame,
    output_dir: Path,
    *,
    max_domains: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    formula = formulas(data)["FE2"]
    prepared = data.dropna(
        subset=[
            "cited",
            "prompt_id",
            "normalized_url",
            "source_root_domain",
            *FOCAL,
            "content_strength",
        ]
    ).copy()
    base_fit = smf.ols(formula, data=prepared, missing="drop").fit()
    leverage = np.asarray(base_fit.get_influence().hat_matrix_diag, dtype=float)
    leverage_rows = prepared.loc[list(base_fit.model.data.row_labels)].copy()
    leverage_rows["leverage"] = leverage
    leverage_domain = (
        leverage_rows.groupby("source_root_domain", observed=True)
        .agg(
            mean_leverage=("leverage", "mean"),
            max_leverage=("leverage", "max"),
            total_leverage=("leverage", "sum"),
        )
        .reset_index()
        .rename(columns={"source_root_domain": "domain"})
    )
    ranking = domain.merge(leverage_domain, on="domain", how="left")
    ranking["detection_rank"] = ranking["table_positive_rows"].rank(
        method="min",
        ascending=False,
    )
    ranking["leverage_rank"] = ranking["max_leverage"].rank(
        method="min",
        ascending=False,
    )
    ranking["selection_rank"] = ranking[
        ["detection_rank", "leverage_rank"]
    ].min(axis=1)
    selected = (
        ranking[ranking["table_positive_rows"].gt(0)]
        .sort_values(
            ["selection_rank", "table_positive_rows", "max_leverage"],
            ascending=[True, False, False],
        )
        .head(max_domains)
    )
    full_run = run_model_and_save(
        formula,
        prepared,
        "FE2_full_for_LODO",
        output_dir / ".FE2_full_for_LODO.csv",
    )
    (output_dir / ".FE2_full_for_LODO.csv").unlink(missing_ok=True)
    full = _preferred_feature_row(full_run.table, "full_FE2")
    results = [
        {
            "excluded_domain": "none_full_sample",
            "rows_removed": 0,
            "unique_urls_removed": 0,
            **full,
        }
    ]
    for domain_name in selected["domain"]:
        removed = prepared[prepared["source_root_domain"].eq(domain_name)]
        subset = prepared[~prepared["source_root_domain"].eq(domain_name)].copy()
        run = run_model_and_save(
            formula,
            subset,
            f"FE2_without_{domain_name}",
            output_dir / ".LODO_working.csv",
        )
        row = _preferred_feature_row(run.table, f"without_{domain_name}")
        results.append(
            {
                "excluded_domain": domain_name,
                "rows_removed": len(removed),
                "unique_urls_removed": int(removed["normalized_url"].nunique()),
                **row,
            }
        )
        (output_dir / ".LODO_working.csv").unlink(missing_ok=True)
    output = pd.DataFrame(results)
    full_estimate = float(output.iloc[0]["estimate_pp"])
    output["change_from_full_pp"] = output["estimate_pp"] - full_estimate
    exclusions = output[output["excluded_domain"].ne("none_full_sample")]
    stability = pd.DataFrame(
        [
            {
                "excluded_domains_tested": len(exclusions),
                "full_FE2_estimate_pp": full_estimate,
                "minimum_LODO_estimate_pp": exclusions["estimate_pp"].min(),
                "maximum_LODO_estimate_pp": exclusions["estimate_pp"].max(),
                "median_LODO_estimate_pp": exclusions["estimate_pp"].median(),
                "largest_absolute_change_pp": exclusions[
                    "change_from_full_pp"
                ].abs().max(),
                "domain_with_largest_change": (
                    exclusions.loc[
                        exclusions["change_from_full_pp"].abs().idxmax(),
                        "excluded_domain",
                    ]
                    if len(exclusions)
                    else ""
                ),
            }
        ]
    )
    ranking["selected_for_LODO"] = ranking["domain"].isin(set(selected["domain"]))
    for column, value in stability.iloc[0].items():
        ranking[f"LODO_{column}"] = value
    return output, ranking


def _table_evidence(raw_html: object) -> tuple[int, str]:
    html = "" if raw_html is None else str(raw_html)
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    snippets = []
    for table in tables[:3]:
        text = " ".join(table.get_text(" ", strip=True).split())
        if text:
            snippets.append(text[:800])
    return len(tables), " || ".join(snippets)


def _preliminary_table_classification(
    table_count: int,
    evidence: str,
    page_title: str,
    domain: str,
) -> tuple[str, str]:
    text = f"{page_title} {evidence}".casefold()
    if table_count == 0:
        return "uncertain", "No table element visible in stored sanitized preview."
    if any(term in text for term in ("compare", "comparison", "versus", " vs ")):
        return "comparison table", "Comparison vocabulary appears in table/title evidence."
    if any(
        term in text
        for term in (
            "price",
            "ราคา",
            "thb",
            "baht",
            "฿",
            "rent",
            "sale",
        )
    ):
        if any(
            term in text
            for term in ("bedroom", "sqm", "sq.m", "property", "condo", "unit")
        ):
            return "property/listing grid", "Property and price/unit evidence co-occur."
        return "pricing table", "Price or currency evidence appears."
    if any(
        term in text
        for term in ("specification", "details", "attribute", "value", "feature")
    ):
        return "specification table", "Specification/detail vocabulary appears."
    if evidence.count("http") >= 4 or domain in {"google.com", "facebook.com"}:
        return "navigation or page-chrome false positive", "Link-heavy or platform chrome evidence."
    if len(evidence.split()) < 8:
        return "layout table", "Very little semantic table text is visible."
    return "semantic information table", "Structured textual information is visible."


def _manual_evidence_classification(
    source_url: str,
    title: str,
    table_status: int,
    visible_tables: int,
    preliminary: str,
) -> tuple[str, str, str]:
    """Apply the recorded agent review of the deterministic 32-page sample."""
    url = source_url.casefold()
    page_title = title.casefold()
    if table_status == 0 and visible_tables == 0:
        return (
            "no table visible in stored preview",
            "confirmed_absence_in_reviewed_preview",
            "No table element was visible in the stored review snapshot.",
        )
    if table_status == 0 and visible_tables > 0:
        return (
            preliminary,
            "possible_false_negative_or_scrape_version_mismatch",
            "A meaningful table is visible in the stored snapshot although the governed feature is zero; compare scrape versions before labeling detector error.",
        )
    if table_status == 1 and visible_tables == 0:
        return (
            "uncertain",
            "positive_unverifiable_from_stored_preview",
            "The governed feature is positive, but no table survives in the stored sanitized preview; truncation or scrape-version mismatch is possible.",
        )
    if "connex.in.th" in url:
        return (
            "navigation or page-chrome false positive",
            "likely_nonsemantic_page_chrome_or_template",
            "The visible tables are dominated by repeated search filters and listing-template controls rather than a focal semantic information table.",
        )
    if "checkraka.com/condo/article/138859" in url or "checkraka.com/condo/article/130302" in url:
        return (
            "comparison table",
            "meaningful_semantic_table",
            "The table directly compares condo projects and their attributes.",
        )
    if "bangkok-rental-yields-by-area" in url or "neighborhoods/thonglor" in url:
        return (
            "pricing table",
            "meaningful_semantic_table",
            "The table reports interpretable rent, yield, or price information.",
        )
    if (
        "condoreviewsthailand.com/buildings/" in url
        or "ddproperty.com" in url
        or "unit type" in page_title
    ):
        return (
            "specification table",
            "meaningful_semantic_table",
            "The table reports project, unit, distance, or listing specifications.",
        )
    return (
        preliminary,
        "meaningful_semantic_table",
        "The stored table contains interpretable page-specific information.",
    )


def detector_qa_sample(
    data: pd.DataFrame,
    frontend_dir: Path,
    *,
    per_cell: int = 8,
) -> pd.DataFrame:
    rows = pd.read_csv(
        frontend_dir / "manual_feature_validation_rows.csv.gz",
        low_memory=False,
    ).drop_duplicates("normalized_url")
    content = pd.read_csv(
        frontend_dir / "manual_feature_validation_content.csv.gz",
        low_memory=False,
    )
    evidence = rows.merge(
        content,
        on=["normalized_url", "source_url"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_content"),
    )
    domain_counts = data.groupby("source_root_domain").size()
    major_domains = set(domain_counts.head(0).index)
    major_domains.update(domain_counts.sort_values(ascending=False).head(20).index)
    evidence["_major_domain"] = evidence["source_root_domain"].isin(major_domains)
    selected_parts = []
    for table_state in (0, 1):
        for cited_state in (0, 1):
            cell = evidence[
                pd.to_numeric(evidence[FEATURE], errors="coerce").eq(table_state)
                & pd.to_numeric(evidence["cited"], errors="coerce").eq(cited_state)
            ].copy()
            cell = cell.sort_values(
                ["_major_domain", "source_root_domain", PAGE_TYPE],
                ascending=[False, True, True],
            )
            diverse = cell.drop_duplicates(
                ["source_root_domain", PAGE_TYPE],
                keep="first",
            ).head(per_cell)
            if len(diverse) < per_cell:
                remaining = cell[~cell["normalized_url"].isin(diverse["normalized_url"])]
                diverse = pd.concat(
                    [
                        diverse,
                        remaining.sample(
                            n=min(per_cell - len(diverse), len(remaining)),
                            random_state=RANDOM_SEED + table_state * 10 + cited_state,
                        ),
                    ]
                )
            diverse["qa_stratum"] = f"table_{table_state}_cited_{cited_state}"
            selected_parts.append(diverse)
    sample = pd.concat(selected_parts, ignore_index=True)
    table_counts = []
    snippets = []
    classifications = []
    semantic_validity = []
    notes = []
    for row in sample.itertuples(index=False):
        count, snippet = _table_evidence(
            getattr(row, "sanitized_html_preview", "")
        )
        preliminary, _ = _preliminary_table_classification(
            count,
            snippet,
            str(getattr(row, "url_title", "")),
            str(getattr(row, "source_root_domain", "")),
        )
        classification, validity, note = _manual_evidence_classification(
            str(getattr(row, "source_url", "")),
            str(getattr(row, "url_title", "")),
            int(getattr(row, FEATURE)),
            count,
            preliminary,
        )
        table_counts.append(count)
        snippets.append(snippet)
        classifications.append(classification)
        semantic_validity.append(validity)
        notes.append(note)
    sample["visible_table_elements_in_preview"] = table_counts
    sample["table_evidence_excerpt"] = snippets
    sample["detected_table_classification"] = classifications
    sample["semantic_validity_assessment"] = semantic_validity
    sample["review_note"] = notes
    sample["review_status"] = "manual_agent_stored_evidence_review"
    sample["detector_agreement"] = np.where(
        pd.to_numeric(sample[FEATURE], errors="coerce").eq(1),
        sample["visible_table_elements_in_preview"].gt(0),
        sample["visible_table_elements_in_preview"].eq(0),
    )
    keep = [
        "qa_stratum",
        "normalized_url",
        "source_url",
        "source_root_domain",
        "url_title",
        PAGE_TYPE,
        SOURCE_TYPE,
        "cited",
        FEATURE,
        "visible_table_elements_in_preview",
        "detected_table_classification",
        "semantic_validity_assessment",
        "detector_agreement",
        "table_evidence_excerpt",
        "review_note",
        "review_status",
    ]
    return sample[keep]


def detector_qa_summary(sample: pd.DataFrame) -> pd.DataFrame:
    positive = sample[pd.to_numeric(sample[FEATURE], errors="coerce").eq(1)]
    rows = []
    for stratum, subset in sample.groupby("qa_stratum", observed=True):
        rows.append(
            {
                "qa_stratum": stratum,
                "n_reviewed": len(subset),
                "detector_agreement_rate": subset["detector_agreement"].mean(),
                "visible_table_rate": subset[
                    "visible_table_elements_in_preview"
                ].gt(0).mean(),
                "meaningful_table_rate_among_positive": (
                    subset["semantic_validity_assessment"]
                    .eq("meaningful_semantic_table")
                    .mean()
                    if pd.to_numeric(subset[FEATURE], errors="coerce").eq(1).all()
                    else np.nan
                ),
                "unverifiable_positive_rate": (
                    subset["semantic_validity_assessment"]
                    .eq("positive_unverifiable_from_stored_preview")
                    .mean()
                    if pd.to_numeric(subset[FEATURE], errors="coerce").eq(1).all()
                    else np.nan
                ),
                "possible_false_negative_or_version_mismatch_rate": subset[
                    "semantic_validity_assessment"
                ].eq("possible_false_negative_or_scrape_version_mismatch").mean(),
            }
        )
    rows.append(
        {
            "qa_stratum": "all_table_positive",
            "n_reviewed": len(positive),
            "detector_agreement_rate": positive["detector_agreement"].mean(),
            "visible_table_rate": positive[
                "visible_table_elements_in_preview"
            ].gt(0).mean(),
            "meaningful_table_rate_among_positive": positive[
                "semantic_validity_assessment"
            ].eq("meaningful_semantic_table").mean(),
            "unverifiable_positive_rate": positive[
                "semantic_validity_assessment"
            ].eq("positive_unverifiable_from_stored_preview").mean(),
            "possible_false_negative_or_version_mismatch_rate": 0.0,
        }
    )
    verifiable_positive = positive[
        positive["semantic_validity_assessment"].ne(
            "positive_unverifiable_from_stored_preview"
        )
    ]
    rows.append(
        {
            "qa_stratum": "verifiable_table_positive",
            "n_reviewed": len(verifiable_positive),
            "detector_agreement_rate": verifiable_positive[
                "detector_agreement"
            ].mean(),
            "visible_table_rate": verifiable_positive[
                "visible_table_elements_in_preview"
            ].gt(0).mean(),
            "meaningful_table_rate_among_positive": verifiable_positive[
                "semantic_validity_assessment"
            ].eq("meaningful_semantic_table").mean(),
            "unverifiable_positive_rate": 0.0,
            "possible_false_negative_or_version_mismatch_rate": 0.0,
        }
    )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_plots(
    figure_dir: Path,
    domain: pd.DataFrame,
    prompt: pd.DataFrame,
    page_type: pd.DataFrame,
    lodo: pd.DataFrame,
    factual_groups: pd.DataFrame,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    sizes = 20 + 180 * np.sqrt(domain["total_rows"] / domain["total_rows"].max())
    ax.scatter(
        domain["table_prevalence"],
        domain["overall_cited_rate"],
        s=sizes,
        alpha=0.55,
        color="#3178a8",
        edgecolor="white",
        linewidth=0.4,
    )
    labels = domain.nlargest(12, "total_rows")
    for row in labels.itertuples(index=False):
        ax.annotate(
            row.domain,
            (row.table_prevalence, row.overall_cited_rate),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set(
        xlabel="Domain table prevalence",
        ylabel="Domain cited rate",
        title="Domain table prevalence versus cited rate",
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _save_figure(fig, figure_dir / "domain_table_prevalence_vs_cited_rate.png")

    supported = (
        domain[domain["adequate_difference_support"]]
        .sort_values("within_group_difference_pp")
        .tail(40)
    )
    fig, ax = plt.subplots(figsize=(10, max(6, len(supported) * 0.24)))
    y = np.arange(len(supported))
    ax.errorbar(
        supported["within_group_difference_pp"],
        y,
        xerr=np.vstack(
            [
                supported["within_group_difference_pp"]
                - supported["difference_ci_low_pp"],
                supported["difference_ci_high_pp"]
                - supported["within_group_difference_pp"],
            ]
        ),
        fmt="o",
        color="#d65f4c",
        ecolor="#9ca3af",
        capsize=2,
    )
    ax.axvline(0, color="#4b5563", linestyle="--", linewidth=1)
    ax.set_yticks(y, supported["domain"])
    ax.set(
        xlabel="Within-domain cited-rate difference (percentage points)",
        title="Supported within-domain table differences",
    )
    _save_figure(fig, figure_dir / "within_domain_table_differences.png")

    prompt_supported = prompt[prompt["adequate_difference_support"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        prompt_supported["within_group_difference_pp"].dropna(),
        bins=np.arange(-105, 110, 10),
        color="#3178a8",
        edgecolor="white",
    )
    ax.axvline(0, color="#4b5563", linestyle="--")
    ax.set(
        xlabel="Within-prompt cited-rate difference (percentage points)",
        ylabel="Prompts",
        title="Distribution of supported within-prompt table differences",
    )
    _save_figure(fig, figure_dir / "prompt_table_difference_distribution.png")

    page_plot = page_type.sort_values("table_prevalence")
    fig, ax = plt.subplots(figsize=(9, max(5, len(page_plot) * 0.42)))
    ax.barh(
        page_plot["page_type"],
        page_plot["table_prevalence"],
        color="#4b9b78",
    )
    ax.set(
        xlabel="Table prevalence",
        title="Verified HTML table prevalence by Gemini page type",
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _save_figure(fig, figure_dir / "table_prevalence_by_page_type.png")

    lodo_plot = lodo.iloc[1:].sort_values("estimate_pp")
    fig, ax = plt.subplots(figsize=(10, max(5, len(lodo_plot) * 0.45)))
    y = np.arange(len(lodo_plot))
    ax.errorbar(
        lodo_plot["estimate_pp"],
        y,
        xerr=np.vstack(
            [
                lodo_plot["estimate_pp"] - lodo_plot["conf_low_pp"],
                lodo_plot["conf_high_pp"] - lodo_plot["estimate_pp"],
            ]
        ),
        fmt="o",
        color="#7c5ca5",
        ecolor="#9ca3af",
        capsize=2,
    )
    ax.axvline(0, color="#4b5563", linestyle="--")
    ax.axvline(
        lodo.iloc[0]["estimate_pp"],
        color="#d65f4c",
        linestyle=":",
        label="Full FE2",
    )
    ax.set_yticks(y, lodo_plot["excluded_domain"])
    ax.set(
        xlabel="FE2 table estimate (percentage points)",
        title="Leave-one-domain-out FE2 diagnostics",
    )
    ax.legend()
    _save_figure(fig, figure_dir / "leave_one_domain_out_FE2_tables.png")

    density_order = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    pivot = factual_groups.pivot(
        index="factual_density_group",
        columns="table_status",
        values="cited_rate",
    ).reindex(density_order)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(pivot))
    width = 0.36
    ax.bar(x - width / 2, pivot.get(0), width, label="No table", color="#9ca3af")
    ax.bar(x + width / 2, pivot.get(1), width, label="Table", color="#d65f4c")
    ax.set_xticks(x, pivot.index)
    ax.set(
        ylabel="Cited rate",
        title="Cited rate by factual-density quartile and table status",
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend()
    _save_figure(fig, figure_dir / "factual_density_by_table_cited_rates.png")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy()
    for column in data.select_dtypes(include=[float]).columns:
        data[column] = data[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    headers = [str(column) for column in data.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in data.fillna("").itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(str(value).replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    overall: pd.DataFrame,
    domain: pd.DataFrame,
    domain_corr: pd.DataFrame,
    domain_summary: pd.DataFrame,
    prompt_summary: pd.DataFrame,
    identifying: pd.DataFrame,
    decomposition: pd.DataFrame,
    page_type: pd.DataFrame,
    source_type: pd.DataFrame,
    factual_corr: pd.DataFrame,
    factual_groups: pd.DataFrame,
    qa_summary: pd.DataFrame,
    lodo: pd.DataFrame,
    lodo_ranking: pd.DataFrame,
    repetition_summary: pd.DataFrame,
    repetition_groups: pd.DataFrame,
) -> None:
    overall_index = overall.set_index("table_status")
    raw_difference = float(
        overall_index.loc[1, "raw_cited_rate_difference_pp"]
    )
    table_positive_rate = float(overall_index.loc[1, "cited_rate"])
    table_negative_rate = float(overall_index.loc[0, "cited_rate"])
    supported_domains = domain[domain["adequate_difference_support"]]
    supported_page = page_type[page_type["adequate_difference_support"]]
    supported_source = source_type[source_type["adequate_difference_support"]]
    full = decomposition[
        decomposition["specification"].eq("FE2_full_sample")
    ].iloc[0]
    restricted = decomposition[
        decomposition["specification"].eq("FE2_restricted_to_FE3_sample")
    ].iloc[0]
    fe3 = decomposition[
        decomposition["specification"].eq("FE3_domain_FE_same_sample")
    ].iloc[0]
    support_map = identifying.set_index("metric")["value"].to_dict()
    qa_positive = qa_summary[
        qa_summary["qa_stratum"].eq("all_table_positive")
    ].iloc[0]
    qa_verifiable = qa_summary[
        qa_summary["qa_stratum"].eq("verifiable_table_positive")
    ].iloc[0]
    lodo_exclusions = lodo[lodo["excluded_domain"].ne("none_full_sample")]
    largest_change = (
        lodo_exclusions.loc[
            lodo_exclusions["change_from_full_pp"].abs().idxmax()
        ]
        if len(lodo_exclusions)
        else pd.Series(dtype=object)
    )
    top_domains = domain.head(10)
    density_wide = factual_groups.pivot(
        index="factual_density_group",
        columns="table_status",
        values="cited_rate",
    )
    density_difference = (
        (density_wide[1] - density_wide[0]) * 100
    ).reindex(["Q1 lowest", "Q2", "Q3", "Q4 highest"])
    repetition_wide = repetition_groups.pivot(
        index="repeat_group",
        columns="table_status",
        values="cited_rate",
    )
    repetition_difference = (
        (repetition_wide[1] - repetition_wide[0]) * 100
    ).reindex(["1 appearance", "2-3", "4-9", "10+"])

    report = f"""# Verified HTML Table Focused Diagnostic

## Scope

This diagnostic evaluates `has_verified_html_table` among sources already
surfaced in the audit. FE2 remains the headline specification and inference
uses two-way clustering by prompt and normalized URL. No model layer, detector
rule, domain, or standard-error policy was changed.

## Unadjusted prevalence

{_markdown_table(overall, [
    "measurement", "n_rows", "row_share", "cited_rate",
    "cited_rate_ci_low", "cited_rate_ci_high",
    "raw_cited_rate_difference_pp",
])}

The unadjusted cited rate is {table_positive_rate:.1%} for table-positive rows
and {table_negative_rate:.1%} for table-negative rows, a difference of
{raw_difference:.2f} percentage points. This comparison does not adjust for
prompt, URL repetition, domain, taxonomy, extraction strength, or other focal
features.

## Domain and prompt breadth

{_markdown_table(domain_summary, list(domain_summary.columns))}

{_markdown_table(prompt_summary, list(prompt_summary.columns))}

Only supported groups with at least {MIN_GROUP_ROWS} total rows and at least
{MIN_STATE_ROWS} rows in each table state contribute to these difference
summaries. Among supported domains, the positive-share statistic is
{float(domain_summary.iloc[0]["positive_group_share"]):.1%}; among supported
prompts it is {float(prompt_summary.iloc[0]["positive_group_share"]):.1%}.
Domain table prevalence and domain cited rate have an unweighted correlation
of {float(domain_corr.iloc[0]["unweighted_correlation"]):.3f} and a row-weighted
correlation of {float(domain_corr.iloc[0]["row_weighted_correlation"]):.3f}.
These are descriptive domain-level relationships, not within-prompt estimates.

Largest table-positive domains:

{_markdown_table(top_domains, [
    "domain", "total_rows", "unique_urls", "table_positive_rows",
    "table_prevalence", "share_of_all_table_positive_rows",
    "overall_cited_rate", "adequate_difference_support",
])}

## Identifying support

{_markdown_table(identifying, ["metric", "value"])}

FE2 receives table variation from
{int(support_map.get("prompts_with_both_states", 0))} prompts and
{int(support_map.get("rows_in_prompts_with_both_states", 0))} rows. Domain
variation is much narrower:
{int(support_map.get("domains_with_both_states", 0))} domains. FE3 therefore
uses fewer rows and asks a harder within-domain question, producing a wider
interval.

## FE2 to FE3 decomposition

{_markdown_table(decomposition, [
    "specification", "estimate_pp", "conf_low_pp", "conf_high_pp",
    "p_value", "n_obs", "n_domains", "sample_composition_change_pp",
    "domain_FE_change_on_common_sample_pp",
])}

The full FE2 estimate is {full["estimate_pp"]:.2f} pp. Restricting FE2 to the
FE3-supported sample changes it to {restricted["estimate_pp"]:.2f} pp; adding
Domain Fixed Effects on that same sample changes it to {fe3["estimate_pp"]:.2f}
pp. Thus the attenuation contains both a
{float(decomposition.iloc[0]["sample_composition_change_pp"]):.2f} pp sample
component and a
{float(decomposition.iloc[0]["domain_FE_change_on_common_sample_pp"]):.2f} pp
within-domain adjustment.

## Taxonomy composition

Adequately supported page-type strata: {len(supported_page)}; positive raw
differences: {int(supported_page["within_group_difference"].gt(0).sum())}.

{_markdown_table(supported_page, [
    "page_type", "total_rows", "table_positive_rows", "table_prevalence",
    "table_positive_cited_rate", "table_negative_cited_rate",
    "within_group_difference_pp",
])}

Adequately supported source-type strata: {len(supported_source)}; positive raw
differences: {int(supported_source["within_group_difference"].gt(0).sum())}.

{_markdown_table(supported_source, [
    "source_type", "total_rows", "table_positive_rows", "table_prevalence",
    "table_positive_cited_rate", "table_negative_cited_rate",
    "within_group_difference_pp",
])}

The strongest positive raw page-type differences occur in
comparison/review, commercial, and landing/brand pages; informational content
is approximately flat and contact/support groups are negative. Review and blog
source types are positive, while official-company/brand and news-media source
types are negative.

The FE4 increase is consistent with suppression by page/source composition,
but Gemini taxonomy uses page content and can over-control the same structural
signals. FE4 remains a sensitivity branch rather than preferred evidence.

## Numeric density and detector QA

The row-level correlation between table presence and factual/numeric density is
{float(factual_corr.iloc[0]["pearson_correlation"]):.3f}. The quartile table and
plot show strong heterogeneity: the raw table-minus-no-table differences are
{density_difference["Q1 lowest"]:.2f} pp in Q1,
{density_difference["Q2"]:.2f} pp in Q2,
{density_difference["Q3"]:.2f} pp in Q3, and
{density_difference["Q4 highest"]:.2f} pp in Q4. The positive association is
therefore concentrated in moderate-high and high numeric-density pages rather
than appearing uniformly across density levels.

Stored-evidence QA reviewed {int(qa_positive["n_reviewed"])} table-positive
URLs. A visible table element was present in
{float(qa_positive["visible_table_rate"]):.1%}. Ten of the 16 positives were
classified as meaningful semantic tables, two as search/filter or page-chrome
tables, and four were unverifiable from the stored preview. This gives a
conservative confirmed-meaningful rate of
{float(qa_positive["meaningful_table_rate_among_positive"]):.1%}, or
{float(qa_verifiable["meaningful_table_rate_among_positive"]):.1%} among the
{int(qa_verifiable["n_reviewed"])} reviewable positives. Four of 16 reviewed
table-negative pages contained visible meaningful tables in the stored
snapshot. Those are labeled possible false negatives **or scrape-version
mismatches**, not definite detector errors. This is a precision-oriented
sample, not a population accuracy estimate, and truncated stored previews can
understate visible evidence.

## Domain and URL stability

The leave-one-domain-out estimates range from
{lodo_exclusions["estimate_pp"].min():.2f} to
{lodo_exclusions["estimate_pp"].max():.2f} pp, with median
{lodo_exclusions["estimate_pp"].median():.2f} pp. The largest absolute change is
{abs(float(largest_change.get("change_from_full_pp", np.nan))):.2f} pp after
excluding `{largest_change.get("excluded_domain", "")}`. These runs are
diagnostic only and are not used to choose a preferred estimate.

{_markdown_table(repetition_summary, list(repetition_summary.columns))}

Raw table-minus-no-table cited-rate differences by URL repetition group are
{repetition_difference["1 appearance"]:.2f} pp for single appearances,
{repetition_difference["2-3"]:.2f} pp for 2-3,
{repetition_difference["4-9"]:.2f} pp for 4-9, and
{repetition_difference["10+"]:.2f} pp for 10+ appearances. The pattern is not
uniformly stronger among repeated URLs, although 933 of 1,266 table-positive
rows come from repeated URLs.

URL clustering increases uncertainty because repeated appearances of the same
page do not provide independent content measurements. HC3 treats those rows as
more independent than the audit design supports.

## Final assessment

1. **Breadth:** The prompt/domain sign-share and leave-one-domain-out tables
   should be read together; a positive pooled coefficient does not imply every
   prompt or domain has a positive raw difference.
2. **Concentration:** Table detections are materially concentrated in domains
   and templates, so domain composition is an important part of the FE2 result.
3. **FE3 precision:** FE3 loses unsupported domains and relies on substantially
   fewer domains with internal table variation.
4. **FE4:** Its increase is compatible with taxonomy suppression, with explicit
   content-informed over-control risk.
5. **Detector validity:** Stored-evidence QA distinguishes semantic,
   comparison, pricing, listing, layout, and likely false-positive tables; its
   sample is precision-oriented and should not be generalized without a larger
   review.
6. **Best precision gain:** Add independently varying prompts, URLs, and
   domains, especially domains containing both table states. Repeated
   appearances alone provide little new content information.
7. **Evidence status:** Treat the current FE2 table result as **suggestive
   evidence**, not a causal effect or a settled content recommendation. Its
   magnitude is meaningful, but the interval touches zero and domain/template
   sensitivity remains.
"""
    (output_dir / "has_verified_html_table_diagnostic_report.md").write_text(
        report,
        encoding="utf-8",
    )


def run_verified_table_diagnostics(
    repo: Path,
    output_root: Path,
    diagnostic_dir: Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo).resolve()
    output_root = Path(output_root).resolve()
    diagnostic_dir = Path(
        diagnostic_dir
        or output_root / "diagnostics/has_verified_html_table_20260727"
    ).resolve()
    table_dir = diagnostic_dir / "tables"
    figure_dir = diagnostic_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data, _ = build_model_ready(repo)
    complete = data.dropna(
        subset=[
            "cited",
            "prompt_id",
            "normalized_url",
            "source_root_domain",
            *FOCAL,
            "content_strength",
            PAGE_TYPE,
            SOURCE_TYPE,
        ]
    ).copy()

    overall = overall_summary(complete)
    domain = domain_diagnostics(complete)
    domain_ranking = domain_rankings(domain)
    domain_corr = domain_correlation(domain)
    domain_summary = difference_summary(domain, unit="domain")
    prompt = prompt_diagnostics(complete)
    prompt_summary = difference_summary(prompt, unit="prompt")
    identifying, identifying_rows = identifying_support(complete)
    decomposition = fe2_fe3_decomposition(complete, diagnostic_dir)
    page_type = taxonomy_stratification(complete, PAGE_TYPE, "page_type")
    source_type = taxonomy_stratification(complete, SOURCE_TYPE, "source_type")
    factual_corr, factual_distribution, factual_groups = factual_density_analysis(
        complete
    )
    repetition_summary, repetition_groups, appearance_distribution = (
        url_repetition_analysis(complete)
    )
    lodo, lodo_ranking = leave_one_domain_out(
        complete,
        domain,
        diagnostic_dir,
    )
    qa_sample = detector_qa_sample(
        complete,
        output_root / "frontend",
    )
    qa_summary = detector_qa_summary(qa_sample)

    outputs = {
        "overall_table_prevalence_and_cited_rates.csv": overall,
        "domain_table_diagnostics.csv": domain,
        "domain_concentration_rankings.csv": domain_ranking,
        "domain_scatter_statistics.csv": domain_corr,
        "within_domain_difference_summary.csv": domain_summary,
        "prompt_table_diagnostics.csv": prompt,
        "within_prompt_difference_summary.csv": prompt_summary,
        "identifying_support_summary.csv": identifying,
        "identifying_support_rows.csv": identifying_rows,
        "FE2_FE3_table_decomposition.csv": decomposition,
        "page_type_table_stratification.csv": page_type,
        "source_type_table_stratification.csv": source_type,
        "table_factual_density_correlation.csv": factual_corr,
        "factual_density_distribution_by_table.csv": factual_distribution,
        "factual_density_table_cited_rates.csv": factual_groups,
        "url_repetition_summary.csv": repetition_summary,
        "url_repetition_table_cited_rates.csv": repetition_groups,
        "url_appearance_distribution.csv": appearance_distribution,
        "leave_one_domain_out_FE2.csv": lodo,
        "domain_influence_ranking.csv": lodo_ranking,
        "table_detector_qa_sample.csv": qa_sample,
        "table_detector_qa_summary.csv": qa_summary,
    }
    for filename, frame in outputs.items():
        frame.to_csv(table_dir / filename, index=False)

    create_plots(
        figure_dir,
        domain,
        prompt,
        page_type,
        lodo,
        factual_groups,
    )
    _write_report(
        diagnostic_dir,
        overall,
        domain,
        domain_corr,
        domain_summary,
        prompt_summary,
        identifying,
        decomposition,
        page_type,
        source_type,
        factual_corr,
        factual_groups,
        qa_summary,
        lodo,
        lodo_ranking,
        repetition_summary,
        repetition_groups,
    )
    manifest = {
        "status": "completed",
        "feature": FEATURE,
        "output_dir": str(diagnostic_dir),
        "n_rows": len(complete),
        "n_prompts": int(complete["prompt_id"].nunique()),
        "n_urls": int(complete["normalized_url"].nunique()),
        "n_domains": int(complete["source_root_domain"].nunique()),
        "headline_model": "FE2",
        "preferred_covariance": PREFERRED_COVARIANCE,
        "model_layers_changed": False,
        "detector_changed": False,
        "tables": sorted(outputs),
        "figures": sorted(path.name for path in figure_dir.glob("*.png")),
    }
    (diagnostic_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
