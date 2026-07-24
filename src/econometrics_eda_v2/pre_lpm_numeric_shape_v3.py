"""Scatter-style, descriptive numeric-shape diagnostics for SCOPE pre-LPM EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.econometrics_eda_v2.pre_lpm_diagnostics import _bool, _numeric, enrich_lpm_diagnostics, wilson_interval


COUNT_FEATURES = ("heading_count", "table_count", "link_count")
ROLLING_FEATURES = ("word_count", "log1p_word_count", "text_char_count", "log1p_text_char_count", "heading_count", "link_count")
WINDOW = 50


def _eligible(df: pd.DataFrame) -> pd.Series:
    return _bool(df, "content_feature_available").eq(1) | _bool(df, "scraped_body_available").eq(1)


def _add_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in ("word_count", "text_char_count", "heading_count", "link_count"):
        out[f"log1p_{base}"] = np.log1p(_numeric(out, base).clip(lower=0))
    return out


def exact_value_rates(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = _numeric(df, feature)
    cited = _bool(df, "cited")
    mask = _eligible(df) & values.notna()
    overall = float(cited.mean())
    rows = []
    for value, group_index in values.loc[mask].groupby(values.loc[mask]).groups.items():
        outcomes = cited.loc[group_index]
        n = len(outcomes)
        cited_rows = int(outcomes.sum())
        more_only = int(n - cited_rows)
        low, high = wilson_interval(cited_rows, n)
        flags = []
        if n < 20:
            flags.append("sparse_n_lt_20")
        if cited_rows < 5:
            flags.append("unstable_cited_lt_5")
        if more_only < 5:
            flags.append("unstable_more_only_lt_5")
        rows.append({"feature_name": feature, "feature_value": float(value), "n_rows": n, "cited_rows": cited_rows, "cited_rate": cited_rows / n, "more_only_rows": more_only, "overall_cited_rate": overall, "diff_from_overall_pp": (cited_rows / n - overall) * 100, "sparse_flag": ";".join(flags), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows).sort_values("feature_value", kind="stable")


def _bubble_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    size = 20 + 10 * np.sqrt(table["n_rows"].to_numpy())
    rate = table["cited_rate"].to_numpy()
    ax.scatter(table["feature_value"], rate, s=size, color="#3f7f8e", alpha=0.60, edgecolor="#264653", linewidth=0.45)
    ax.errorbar(table["feature_value"], rate, yerr=[rate - table["ci_low"], table["ci_high"] - rate], fmt="none", ecolor="#24333d", alpha=0.48, capsize=2)
    ax.axhline(float(table["overall_cited_rate"].iloc[0]), color="#5d6670", linestyle="--", linewidth=1.1)
    ax.set_title(title)
    ax.set_xlabel("Exact feature value")
    ax.set_ylabel("Cited rate (Wilson 95% CI)")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    labels = table[(table["n_rows"] >= 30) | table["feature_value"].isin([0, 1])]
    for row in labels.itertuples(index=False):
        ax.annotate(f"{int(row.feature_value)} (n={row.n_rows})", (row.feature_value, row.cited_rate), xytext=(3, 4), textcoords="offset points", fontsize=7)
    ax.grid(axis="y", alpha=0.22)
    fig.text(0.5, 0.01, "Bubble area is proportional to row count. Sparse exact values are descriptive only.", ha="center", fontsize=8, color="#4b5563")
    fig.subplots_adjust(bottom=0.14, top=0.91, hspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _jitter_plot(df: pd.DataFrame, feature: str, path: Path) -> None:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    rng = np.random.default_rng(20260713)
    y = _bool(df, "cited").loc[mask].to_numpy(dtype=float) + rng.uniform(-0.075, 0.075, int(mask.sum()))
    x = values.loc[mask].to_numpy()
    fig, (ax_hist, ax) = plt.subplots(2, 1, figsize=(9.5, 5.7), sharex=True, gridspec_kw={"height_ratios": [1, 4], "hspace": 0.05})
    ax_hist.hist(x, bins=min(40, max(10, int(np.sqrt(len(x))))), color="#8db9c4", alpha=0.8)
    ax.scatter(x, y, s=12, color="#3f7f8e", alpha=0.25, linewidths=0)
    ax.set_yticks([0, 1], ["Surfaced / more-only", "Cited"])
    ax.set_xlabel(feature)
    ax.set_title(f"Row-level jitter by cited status: {feature}")
    ax.grid(axis="x", alpha=0.22)
    ax_hist.set_ylabel("Rows")
    ax_hist.grid(axis="y", alpha=0.16)
    fig.text(0.5, 0.01, "Binary y-values are vertically jittered to show concentration; this is not a cited-rate plot.", ha="center", fontsize=8, color="#4b5563")
    fig.subplots_adjust(bottom=0.14, top=0.91, hspace=0.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def rolling_cited_rates(df: pd.DataFrame, feature: str, window: int = WINDOW) -> pd.DataFrame:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    work = pd.DataFrame({"value": values.loc[mask], "cited": _bool(df, "cited").loc[mask]}).sort_values("value", kind="stable").reset_index(drop=True)
    rows = []
    if len(work) < window:
        return pd.DataFrame(columns=["feature_name", "window_start", "window_end", "x_mid", "cited_rate", "n_rows", "overall_cited_rate"])
    overall = float(_bool(df, "cited").mean())
    for start in range(0, len(work) - window + 1):
        group = work.iloc[start:start + window]
        rows.append({"feature_name": feature, "window_start": start, "window_end": start + window - 1, "x_mid": float(group["value"].median()), "cited_rate": float(group["cited"].mean()), "n_rows": window, "overall_cited_rate": overall})
    return pd.DataFrame(rows)


def _rolling_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.1))
    ax.plot(table["x_mid"], table["cited_rate"], color="#3f7f8e", alpha=0.88, linewidth=1.45)
    ax.axhline(float(table["overall_cited_rate"].iloc[0]), color="#5d6670", linestyle="--", linewidth=1.1)
    ax.set_xlabel("Feature value (rolling-window median)")
    ax.set_ylabel("Rolling cited rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.text(0.5, 0.01, f"Each point summarizes a sorted rolling window of {WINDOW} rows; overlapping windows are not independent.", ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _distribution_plot(df: pd.DataFrame, feature: str, path: Path) -> None:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    cited = _bool(df, "cited")
    non_cited = values.loc[mask & cited.eq(0)]
    cited_values = values.loc[mask & cited.eq(1)]
    bins = min(35, max(10, int(np.sqrt(mask.sum()))))
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    ax.hist(non_cited, bins=bins, density=True, alpha=0.48, color="#7d8790", label="Surfaced / more-only")
    ax.hist(cited_values, bins=bins, density=True, alpha=0.48, color="#3f7f8e", label="Cited")
    ax.set_title(f"Distribution of {feature} by cited status")
    ax.set_xlabel(feature)
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.text(0.5, 0.01, "Diagnostic only; distribution overlap indicates limited standalone separation.", ha="center", fontsize=8, color="#4b5563")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _tail_group(values: pd.Series, feature: str) -> pd.Series:
    cap = 4 if feature == "table_count" else 21
    return values.fillna(-1).map(lambda value: f"{cap}+" if value >= cap else str(int(value)))


def _heatmap(df: pd.DataFrame, feature: str, path: Path) -> None:
    values = _numeric(df, feature)
    mask = _eligible(df) & values.notna()
    labels = _tail_group(values.loc[mask], feature)
    status = _bool(df, "cited").loc[mask]
    order = [str(i) for i in range(0, 4 if feature == "table_count" else 21)] + (["4+"] if feature == "table_count" else ["21+"])
    matrix = pd.crosstab(status, labels).reindex(index=[0, 1], columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.46 * len(order)), 3.2))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_yticks([0, 1], ["Surfaced / more-only", "Cited"])
    ax.set_xticks(range(len(order)), order, rotation=0)
    ax.set_xlabel(f"{feature} exact value / grouped tail")
    ax.set_title(f"Row concentration by {feature} and cited status")
    for yi in range(matrix.shape[0]):
        for xi in range(matrix.shape[1]):
            ax.text(xi, yi, str(int(matrix.iloc[yi, xi])), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Row count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _shape_recommendations(exact_tables: dict[str, pd.DataFrame], rolling_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    guidance = {
        "heading_count": ("diagnostic_only", "Count values concentrate at low levels and the rolling/exact patterns are noisy; retain only as diagnostic or binned sensitivity."),
        "table_count": ("threshold", "Zero-inflated count with sparse one-table value; use has_table or 0/1/2+ threshold, not raw count."),
        "word_count": ("spline_or_bins", "Skewed continuous measure; inspect log scale or flexible bins/spline rather than raw linear count."),
        "link_count": ("diagnostic_only", "Count concentration and noisy exact values make a standalone linear specification fragile."),
    }
    for feature, (form, reason) in guidance.items():
        exact = exact_tables.get(feature, pd.DataFrame())
        rolling = rolling_tables.get(feature, pd.DataFrame())
        concentration = "not_available"
        if not exact.empty:
            top = exact.sort_values("n_rows", ascending=False).iloc[0]
            concentration = f"Most rows at value {int(top.feature_value)} (n={int(top.n_rows)}; {top.n_rows / exact.n_rows.sum():.1%} of eligible rows)."
        if feature == "word_count" and not rolling.empty:
            first, last = rolling.iloc[0].cited_rate, rolling.iloc[-1].cited_rate
            shape = "decreasing_or_non_linear" if last < first - 0.04 else "flat_or_noisy"
        elif feature == "heading_count":
            shape = "no_clear_monotonic_pattern"
        elif feature == "table_count":
            shape = "zero_inflated"
        else:
            shape = "noisy_exact_value_pattern"
        rows.append({"feature_name": feature, "exact_value_plot_useful": feature in exact_tables, "jitter_plot_useful": True, "rolling_curve_useful": feature in rolling_tables, "distribution_plot_useful": True, "observed_shape": shape, "concentration_summary": concentration, "recommended_lpm_form": form, "reason": reason})
    return pd.DataFrame(rows)


def run_numeric_shape_diagnostics_v3(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    df = _add_logs(enrich_lpm_diagnostics(lpm, eda))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    exact = {feature: exact_value_rates(df, feature) for feature in COUNT_FEATURES if feature in df}
    for feature, table in exact.items():
        table.to_csv(output_dir / f"exact_value_cited_rate_{feature}.csv", index=False)
        _bubble_plot(table, figure_dir / f"bubble_exact_cited_rate_{feature}.png", f"Exact-value cited rate: {feature}")
    for feature in ("heading_count", "log1p_word_count", "log1p_text_char_count", "link_count", "table_count"):
        if feature in df:
            _jitter_plot(df, feature, figure_dir / f"jitter_row_level_{feature}_by_cited.png")
    rolling = {feature: rolling_cited_rates(df, feature) for feature in ROLLING_FEATURES if feature in df}
    rolling_table = pd.concat(rolling.values(), ignore_index=True)
    rolling_table.to_csv(output_dir / "rolling_cited_rate_numeric_features.csv", index=False)
    for feature in ("log1p_word_count", "heading_count", "link_count"):
        _rolling_plot(rolling[feature], figure_dir / f"rolling_cited_rate_{feature}.png", f"Rolling cited rate: {feature}")
    for feature in ("log1p_word_count", "heading_count", "link_count", "table_count"):
        _distribution_plot(df, feature, figure_dir / f"distribution_{feature}_by_cited.png")
    for feature in COUNT_FEATURES:
        _heatmap(df, feature, figure_dir / f"heatmap_{feature}_by_cited.png")
    recommendations = _shape_recommendations(exact, rolling)
    recommendations.to_csv(output_dir / "numeric_feature_visual_shape_recommendations_v3.csv", index=False)
    (output_dir / "numeric_shape_interpretation_v3.md").write_text(
        "# Detailed Numeric Shape Diagnostics\n\n"
        "1. Row-level jitter shows raw binary observations and concentration; its y-axis is not a cited rate.\n"
        "2. Exact-value bubbles show cited rate by exact count and use bubble size for concentration. Sparse exact counts are not interpretable on their own.\n"
        "3. Continuous features such as word count are not grouped by exact value because tiny exact-value cells can manufacture misleading 0% or 100% rates.\n"
        f"4. Rolling cited-rate curves use sorted, overlapping windows of {WINDOW} rows to inspect rough shape; they are exploratory and not independent estimates.\n"
        "5. These plots inform functional-form choices only. They are descriptive, conditional on content availability/scrape success, and do not support causal claims or hidden-retrieval claims.\n",
        encoding="utf-8",
    )
    return {"rows": int(len(df)), "unique_urls": int(df["normalized_url"].nunique()), "output_dir": str(output_dir), "figure_dir": str(figure_dir), "rolling_rows": int(len(rolling_table))}
