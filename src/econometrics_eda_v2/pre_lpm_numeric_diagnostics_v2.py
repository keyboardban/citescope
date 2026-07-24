"""Ordered, zero-aware numeric diagnostics for the SCOPE pre-LPM notebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.econometrics_eda_v2.pre_lpm_diagnostics import (
    _bool,
    _category,
    _numeric,
    citation_rate_by_category,
    enrich_lpm_diagnostics,
    wilson_interval,
)


MISSING = "<missing>"


def _eligible(df: pd.DataFrame) -> pd.Series:
    return _bool(df, "content_feature_available").eq(1) | _bool(df, "scraped_body_available").eq(1)


def _rate_row(feature: str, label: str, order: int, values: pd.Series, cited: pd.Series, overall: float, method: str, low: float, high: float) -> dict[str, Any]:
    n = len(values)
    cited_n = int(cited.sum())
    ci_low, ci_high = wilson_interval(cited_n, n)
    rate = cited_n / n if n else np.nan
    return {
        "feature": feature,
        "bin_label": label,
        "bin_order": order,
        "bin_low": low,
        "bin_high": high,
        "n_rows": n,
        "cited_rows": cited_n,
        "cited_rate": rate,
        "overall_cited_rate": overall,
        "difference_from_overall": rate - overall if n else np.nan,
        "wilson_ci_low": ci_low,
        "wilson_ci_high": ci_high,
        "difference_ci_low": ci_low - overall if n else np.nan,
        "difference_ci_high": ci_high - overall if n else np.nan,
        "sparse_flag": "very_sparse_n_lt_20_or_cited_lt_5" if n < 20 or cited_n < 5 else ("sparse_n_lt_30" if n < 30 else ""),
        "binning_method": method,
        "eligible_definition": "content_feature_available == 1 OR scraped_body_available == 1",
    }


def quantile_bins(df: pd.DataFrame, feature: str, q: int = 4) -> tuple[pd.DataFrame, str]:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    work = pd.DataFrame({"value": values.loc[mask], "cited": _bool(df, "cited").loc[mask]})
    overall = float(_bool(df, "cited").mean())
    if len(work) < 4 or work["value"].nunique() < 2:
        return pd.DataFrame(), "insufficient_variation"
    bins = pd.qcut(work["value"], q=min(q, work["value"].nunique()), duplicates="drop")
    if bins.cat.categories.size <= 1:
        return pd.DataFrame(), "qcut_collapsed_to_one_bin"
    work["bin"] = bins
    rows = []
    for order, (_, group) in enumerate(work.groupby("bin", observed=True), start=1):
        actual_low = float(group["value"].min())
        actual_high = float(group["value"].max())
        if float(actual_low).is_integer() and float(actual_high).is_integer():
            label = f"{int(actual_low):,}" if actual_low == actual_high else f"{int(actual_low):,}–{int(actual_high):,}"
        else:
            label = f"{actual_low:.2f}–{actual_high:.2f}"
        rows.append(_rate_row(feature, label, order, group["value"], group["cited"], overall, "qcut_readable_observed_range", actual_low, actual_high))
    return pd.DataFrame(rows), "qcut_readable_observed_range"


def threshold_bins(df: pd.DataFrame, feature: str, definitions: list[tuple[str, float, float]]) -> pd.DataFrame:
    values = _numeric(df, feature)
    cited = _bool(df, "cited")
    eligible = _eligible(df) & values.notna()
    overall = float(cited.mean())
    rows = []
    for order, (label, low, high) in enumerate(definitions, start=1):
        mask = eligible & values.ge(low) & (values.le(high) if np.isfinite(high) else True)
        rows.append(_rate_row(feature, label, order, values.loc[mask], cited.loc[mask], overall, "manual_threshold", low, high))
    return pd.DataFrame(rows)


def _ordered_rate_plot(table: pd.DataFrame, path: Path, title: str, difference: bool = False) -> None:
    ordered = table.sort_values(["bin_order", "bin_low"], kind="stable")
    y = ordered["difference_from_overall"].to_numpy() if difference else ordered["cited_rate"].to_numpy()
    low = ordered["difference_ci_low"].to_numpy() if difference else ordered["wilson_ci_low"].to_numpy()
    high = ordered["difference_ci_high"].to_numpy() if difference else ordered["wilson_ci_high"].to_numpy()
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.plot(x, y, color="#3f7f8e", linewidth=1.6, marker="o", markersize=6)
    ax.errorbar(x, y, yerr=[y - low, high - y], fmt="none", ecolor="#24333d", capsize=3)
    ax.axhline(0 if difference else float(ordered["overall_cited_rate"].iloc[0]), color="#8a4f20" if difference else "#5d6670", linestyle="--", linewidth=1.1)
    ax.set_xticks(x, ordered["bin_label"], rotation=0)
    ax.set_ylabel("Difference from overall cited rate (percentage points)" if difference else "Cited rate (Wilson 95% CI)")
    ax.set_title(title)
    if difference:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value * 100:+.0f} pp")
    else:
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for xi, yi, row in zip(x, y, ordered.itertuples(index=False)):
        offset = 0.018 if not difference else (0.018 if yi >= 0 else -0.028)
        ax.text(xi, yi + offset, f"n={row.n_rows}", ha="center", va="bottom" if offset > 0 else "top", fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    fig.text(0.5, 0.01, "Numeric content features are conditional on content availability or scrape success.", ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _categorical_difference_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    ordered = table.sort_values("difference_from_overall", kind="stable")
    y = np.arange(len(ordered))
    diff = ordered["difference_from_overall"].to_numpy()
    low = ordered["wilson_ci_low"].to_numpy() - ordered["overall_cited_rate"].to_numpy()
    high = ordered["wilson_ci_high"].to_numpy() - ordered["overall_cited_rate"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.43 * len(ordered) + 1.2)))
    ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="o", color="#3f7f8e", ecolor="#24333d", capsize=3)
    ax.axvline(0, color="#8a4f20", linestyle="--", linewidth=1.1)
    ax.set_yticks(y, ordered["category"])
    ax.xaxis.set_major_formatter(lambda value, _: f"{value * 100:+.0f} pp")
    ax.set_xlabel("Cited-rate difference from overall (Wilson 95% CI)")
    ax.set_title(title)
    for yi, row in enumerate(ordered.itertuples(index=False)):
        ax.text(min(0.97, row.difference_from_overall + 0.012), yi, f"n={row.n_rows}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _distribution_plot(df: pd.DataFrame, feature: str, path: Path) -> None:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    groups = [values.loc[mask & _bool(df, "cited").eq(0)].to_numpy(), values.loc[mask & _bool(df, "cited").eq(1)].to_numpy()]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    box = ax.boxplot(groups, tick_labels=["Surfaced / more-only", "Cited"], patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#6da8b6")
    ax.set_title(f"Distribution of {feature} by citation status")
    ax.set_ylabel(feature)
    ax.grid(axis="y", alpha=0.22)
    fig.text(0.5, 0.01, "Diagnostic only; distributions are conditional on content availability or scrape success.", ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _shape_summary(feature: str, table: pd.DataFrame, method: str) -> dict[str, Any]:
    rates = table.sort_values("bin_order")["cited_rate"].dropna().tolist() if not table.empty else []
    pattern = "not_estimable" if len(rates) < 2 else ("roughly_increasing" if rates == sorted(rates) else ("roughly_decreasing" if rates == sorted(rates, reverse=True) else "non_monotonic"))
    interpretation = "Numeric content features are conditional on content availability / scrape success."
    recommendation = "diagnostic_or_sensitivity"
    if feature == "heading_count":
        interpretation = "heading_count does not show a clear monotonic or linear association with cited rate. Treat it as diagnostic or as a possible control/sensitivity variable, not as a main focal predictor."
    elif feature == "table_count":
        interpretation = "table_count is zero-inflated or duplicate-heavy, so quantile binning collapsed. Use has_table or table_count_group instead of raw table_count for EDA and LPM sensitivity."
        recommendation = "use_has_table_or_table_count_group"
    elif feature == "word_count":
        recommendation = "use_log1p_or_binned_sensitivity_only"
    return {"feature": feature, "binning_method": method, "n_bins": int(len(table)), "shape_pattern": pattern, "recommended_use": recommendation, "interpretation": interpretation}


def run_numeric_diagnostics_v2(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    df = enrich_lpm_diagnostics(lpm, eda)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    qcut_tables = {}
    shape_rows = []
    for feature in ("word_count", "text_char_count"):
        table, method = quantile_bins(df, feature)
        qcut_tables[feature] = table
        shape_rows.append(_shape_summary(feature, table, method))
    threshold_specs = {
        "heading_count": [("0–1 headings", 0, 1), ("2–6 headings", 2, 6), ("7–12 headings", 7, 12), ("13+ headings", 13, np.inf)],
        "table_count": [("0 tables", 0, 0), ("1 table", 1, 1), ("2+ tables", 2, np.inf)],
        "link_count": [("0–3 links", 0, 3), ("4–8 links", 4, 8), ("9+ links", 9, np.inf)],
    }
    threshold_tables = {feature: threshold_bins(df, feature, bins) for feature, bins in threshold_specs.items()}
    for feature, table in threshold_tables.items():
        shape_rows.append(_shape_summary(feature, table, "manual_threshold"))
    numeric_table = pd.concat(list(qcut_tables.values()), ignore_index=True)
    threshold_table = pd.concat(list(threshold_tables.values()), ignore_index=True)
    numeric_table.to_csv(output_dir / "numeric_feature_bin_diagnostics_v2.csv", index=False)
    threshold_table.to_csv(output_dir / "numeric_feature_threshold_diagnostics_v2.csv", index=False)
    shape = pd.DataFrame(shape_rows)
    shape.to_csv(output_dir / "numeric_feature_shape_summary_v2.csv", index=False)
    all_numeric = pd.concat([numeric_table, threshold_table], ignore_index=True)
    for feature, filename in (("word_count", "cited_rate_by_word_count_ordered.png"), ("heading_count", "cited_rate_by_heading_count_ordered.png"), ("table_count", "cited_rate_by_table_count_threshold.png"), ("link_count", "cited_rate_by_link_count_ordered.png")):
        table = all_numeric[all_numeric["feature"].eq(feature)]
        _ordered_rate_plot(table, figure_dir / filename, f"Cited rate by {feature} bin")
        _ordered_rate_plot(table, figure_dir / f"diff_from_overall_{feature}.png", f"Difference from overall cited rate by {feature} bin", difference=True)
    categorical = {feature: citation_rate_by_category(df, feature) for feature in ("page_type_family_real_estate", "source_type_real_estate", "content_quality_flag")}
    for table in categorical.values():
        table["difference_ci_low"] = table["wilson_ci_low"] - table["overall_cited_rate"]
        table["difference_ci_high"] = table["wilson_ci_high"] - table["overall_cited_rate"]
        table["bin_label"] = table["category"]
    for feature, filename, title in (
        ("page_type_family_real_estate", "diff_from_overall_page_type_family.png", "Difference from overall cited rate by page type family"),
        ("source_type_real_estate", "diff_from_overall_source_type.png", "Difference from overall cited rate by source type"),
        ("content_quality_flag", "diff_from_overall_content_quality.png", "Difference from overall cited rate by content quality"),
    ):
        _categorical_difference_plot(categorical[feature], figure_dir / filename, title)
    difference = pd.concat([*categorical.values(), all_numeric], ignore_index=True, sort=False)
    difference.to_csv(output_dir / "difference_from_overall_by_feature_v2.csv", index=False)
    for feature in ("log1p_word_count", "log1p_heading_count", "log1p_link_count"):
        if feature not in df:
            base = feature.removeprefix("log1p_")
            df[feature] = np.log1p(_numeric(df, base).clip(lower=0))
        _distribution_plot(df, feature, figure_dir / f"distribution_{feature}_by_cited.png")
    (output_dir / "numeric_feature_interpretation_v2.md").write_text(
        "# Numeric Feature Interpretation\n\n"
        "Negative lower bounds in old pandas `qcut` interval labels were display artifacts used to include zero, not negative measurements or cited rates. V2 labels use observed value ranges or explicit thresholds.\n\n"
        + "\n\n".join(f"## {row['feature']}\n{row['interpretation']}" for row in shape_rows)
        + "\n\nAll numeric content-feature diagnostics are conditional on content availability / scrape success and are descriptive only. No final LPM is fit here.\n",
        encoding="utf-8",
    )
    return {"rows": int(len(df)), "unique_urls": int(df["normalized_url"].nunique()), "output_dir": str(output_dir), "figure_dir": str(figure_dir), "numeric_rows": int(len(all_numeric))}
