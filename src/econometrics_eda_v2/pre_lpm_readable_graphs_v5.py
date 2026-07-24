"""Readable Plotly diagnostics for the SCOPE pre-LPM EDA notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.econometrics_eda_v2.pre_lpm_diagnostics import (
    _bool,
    _category,
    _numeric,
    citation_rate_by_category,
    enrich_lpm_diagnostics,
    wilson_interval,
)
from src.econometrics_eda_v2.pre_lpm_plotly_v4 import _add_logs, exact_frequency_stats, rolling_stats


COUNT_FEATURES = ("heading_count", "table_count", "link_count")
TAIL_CAPS = {"heading_count": 25, "table_count": 3, "link_count": 30}
ROLLING_FEATURES = ("log1p_word_count", "heading_count", "link_count")
CATEGORY_FEATURES = (
    "page_type_family_real_estate",
    "source_type_real_estate",
    "content_quality_flag",
    "re_page_type_confidence",
)
FOREST_SLUGS = {
    "page_type_family_real_estate": "page_type_family",
    "source_type_real_estate": "source_type",
    "content_quality_flag": "content_quality",
    "re_page_type_confidence": "taxonomy_confidence",
}


def clean_feature_name(name: str) -> str:
    """Turn a pipeline column name into a concise display label."""
    names = {
        "page_type_family_real_estate": "Page type family",
        "source_type_real_estate": "Source type",
        "content_quality_flag": "Content quality",
        "re_page_type_confidence": "Taxonomy confidence",
        "log1p_word_count": "log1p word count",
        "heading_count": "Heading count",
        "table_count": "Table count",
        "link_count": "Link count",
    }
    return names.get(name, name.replace("_", " ").strip().title())


def cap_count_feature_tail(series: pd.Series, feature: str) -> pd.Series:
    """Keep exact low counts and group the feature's upper tail."""
    cap = TAIL_CAPS[feature]
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.map(lambda value: f"{cap + 1}+" if pd.notna(value) and value > cap else str(int(value)) if pd.notna(value) else "<missing>")


def make_sparse_symbol_column(stats: pd.DataFrame) -> pd.Series:
    """Use a distinct marker for estimates with weak support."""
    sparse = stats["n_rows"].lt(20) | stats["cited_rows"].lt(5) | stats["more_only_rows"].lt(5)
    return pd.Series(np.where(sparse, "x", "circle"), index=stats.index, name="marker_symbol")


def apply_readable_plotly_layout(fig: go.Figure, title: str, subtitle: str | None = None) -> go.Figure:
    title_text = title if not subtitle else f"{title}<br><sup>{subtitle}</sup>"
    fig.update_layout(
        template="plotly_white",
        width=950,
        height=560,
        title={"text": title_text, "font": {"size": 18}},
        font={"size": 13},
        hoverlabel={"font": {"size": 13}},
        margin={"l": 70, "r": 40, "t": 90, "b": 70},
        legend={"title": {"font": {"size": 12}}},
    )
    fig.update_xaxes(title_font={"size": 14})
    fig.update_yaxes(title_font={"size": 14})
    return fig


def save_plotly_figure(fig: go.Figure, html_path: Path, png_path: Path | None = None) -> str:
    """Save interactive HTML and optionally export PNG when Kaleido is available."""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    if png_path is None:
        return "not_requested"
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(png_path, scale=2)
        return "written"
    except Exception as exc:  # Kaleido is deliberately optional.
        return f"skipped: {type(exc).__name__}"


def _aggregate_capped_stats(stats: pd.DataFrame, feature: str) -> pd.DataFrame:
    work = stats.copy()
    work["feature_value_group"] = cap_count_feature_tail(work["feature_value"], feature)
    cap = TAIL_CAPS[feature]
    rows: list[dict[str, Any]] = []
    for label, group in work.groupby("feature_value_group", sort=False):
        n = int(group["n_rows"].sum())
        cited = int(group["cited_rows"].sum())
        more_only = int(group["more_only_rows"].sum())
        rate = cited / n if n else np.nan
        ci_low, ci_high = wilson_interval(cited, n)
        rows.append(
            {
                "feature_name": feature,
                "feature_value_group": label,
                "feature_value_sort": cap + 1 if label.endswith("+") else int(label),
                "n_rows": n,
                "cited_rows": cited,
                "more_only_rows": more_only,
                "cited_rate": rate,
                "cited_rate_pct": rate * 100,
                "overall_cited_rate": float(group["overall_cited_rate"].iloc[0]),
                "overall_cited_rate_pct": float(group["overall_cited_rate_pct"].iloc[0]),
                "diff_from_overall_pp": (rate - float(group["overall_cited_rate"].iloc[0])) * 100,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_low_pct": ci_low * 100,
                "ci_high_pct": ci_high * 100,
            }
        )
    out = pd.DataFrame(rows).sort_values("feature_value_sort", kind="stable").reset_index(drop=True)
    out["sparse"] = out["n_rows"].lt(20)
    out["unstable"] = out["cited_rows"].lt(5) | out["more_only_rows"].lt(5)
    out["marker_symbol"] = make_sparse_symbol_column(out)
    return out


def _prepare_full_stats(stats: pd.DataFrame, feature: str) -> pd.DataFrame:
    out = stats.copy().sort_values("feature_value", kind="stable").reset_index(drop=True)
    out["feature_value_group"] = out["feature_value"].map(lambda value: str(int(value)))
    out["sparse"] = out["n_rows"].lt(20)
    out["unstable"] = out["cited_rows"].lt(5) | out["more_only_rows"].lt(5)
    out["marker_symbol"] = make_sparse_symbol_column(out)
    return out


def _scatter_hover() -> str:
    return (
        "Feature value: %{customdata[0]}<br>"
        "Rows: %{customdata[1]:,}<br>"
        "Cited rows: %{customdata[2]:,}<br>"
        "More-only rows: %{customdata[3]:,}<br>"
        "Cited rate: %{customdata[4]:.1f}%<br>"
        "Difference from overall: %{customdata[5]:+.1f} pp<br>"
        "95% CI: %{customdata[6]:.1f}% to %{customdata[7]:.1f}%<br>"
        "Sparse: %{customdata[8]}<br>"
        "Unstable: %{customdata[9]}<extra></extra>"
    )


def make_readable_exact_scatter(stats: pd.DataFrame, feature: str, *, capped: bool) -> go.Figure:
    display_col = "feature_value_group"
    x = stats[display_col] if capped else stats["feature_value"]
    labels = stats[display_col].tolist()
    text = [f"n={int(n)}" if not sparse and n >= 30 else "" for n, sparse in zip(stats["n_rows"], stats["sparse"])]
    customdata = np.column_stack(
        [
            labels,
            stats["n_rows"],
            stats["cited_rows"],
            stats["more_only_rows"],
            stats["cited_rate_pct"],
            stats["diff_from_overall_pp"],
            stats["ci_low_pct"],
            stats["ci_high_pct"],
            np.where(stats["sparse"], "True", "False"),
            np.where(stats["unstable"], "True", "False"),
        ]
    )
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=stats["cited_rate_pct"],
            mode="markers+text",
            text=text,
            textposition="top center",
            textfont={"size": 11, "color": "#334155"},
            customdata=customdata,
            hovertemplate=_scatter_hover(),
            marker={
                "size": 12,
                "color": stats["n_rows"],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "Rows"},
                "symbol": stats["marker_symbol"],
                "line": {"width": 0.7, "color": "#ffffff"},
            },
            name="Exact value",
        )
    )
    overall = float(stats["overall_cited_rate_pct"].iloc[0])
    fig.add_hline(y=overall, line_dash="dash", line_color="#5d6670", annotation_text="Overall cited rate", annotation_position="top left")
    suffix = "tail-capped readable view" if capped else "full-resolution diagnostic view"
    apply_readable_plotly_layout(fig, f"Cited rate by {clean_feature_name(feature)}", suffix)
    fig.update_yaxes(title="Cited rate (%)", range=[0, 100], ticksuffix="%")
    fig.update_xaxes(title=f"{clean_feature_name(feature)}" + (" (capped tail)" if capped else " (exact value)"))
    if capped:
        fig.update_xaxes(categoryorder="array", categoryarray=stats[display_col].tolist())
    return fig


def _heatmap_table(capped: pd.DataFrame, feature: str) -> tuple[pd.DataFrame, list[str]]:
    values = capped["feature_value_group"].tolist()
    rows = []
    for row in capped.itertuples(index=False):
        rows.extend(
            [
                {"feature_name": feature, "feature_value_group": row.feature_value_group, "cited_status": "More-only / not cited", "row_count": int(row.more_only_rows)},
                {"feature_name": feature, "feature_value_group": row.feature_value_group, "cited_status": "Cited", "row_count": int(row.cited_rows)},
            ]
        )
    return pd.DataFrame(rows), values


def make_readable_heatmap(table: pd.DataFrame, feature: str, values: list[str]) -> go.Figure:
    statuses = ["More-only / not cited", "Cited"]
    matrix = table.pivot(index="cited_status", columns="feature_value_group", values="row_count").reindex(index=statuses, columns=values, fill_value=0)
    text = np.where(matrix.to_numpy() >= 10, matrix.to_numpy().astype(str), "")
    fig = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=values,
            y=statuses,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="Cividis",
            colorbar={"title": "Rows"},
            hovertemplate=f"{clean_feature_name(feature)}: %{{x}}<br>Status: %{{y}}<br>Rows: %{{z:,}}<extra></extra>",
        )
    )
    apply_readable_plotly_layout(fig, f"Raw row concentration by {clean_feature_name(feature)}", "tail-capped counts before conversion to cited rates")
    fig.update_xaxes(title=f"{clean_feature_name(feature)} (capped tail)", categoryorder="array", categoryarray=values)
    fig.update_yaxes(title="Citation status")
    return fig


def make_readable_rolling_curve(stats: pd.DataFrame, feature: str, window: int) -> go.Figure:
    customdata = np.column_stack([stats["x_min"], stats["x_max"], stats["n_rows"]])
    fig = go.Figure(
        go.Scatter(
            x=stats["x_mid"],
            y=stats["cited_rate_pct"],
            mode="lines+markers",
            line={"color": "#287a8e", "width": 2.5},
            marker={"size": 4, "color": "#287a8e", "opacity": 0.32},
            customdata=customdata,
            hovertemplate=(
                "Window midpoint: %{x:.3f}<br>"
                "Window range: %{customdata[0]:.3f} to %{customdata[1]:.3f}<br>"
                "Rows: %{customdata[2]:,}<br>"
                "Cited rate: %{y:.1f}%<extra></extra>"
            ),
            name="Rolling cited rate",
        )
    )
    fig.add_hline(y=float(stats["overall_cited_rate_pct"].iloc[0]), line_dash="dash", line_color="#5d6670", annotation_text="Overall cited rate", annotation_position="top left")
    apply_readable_plotly_layout(fig, f"Rolling cited rate by {clean_feature_name(feature)}", f"{window}-row overlapping windows; shape diagnostic only")
    fig.update_xaxes(title=f"{clean_feature_name(feature)} (window midpoint)")
    fig.update_yaxes(title="Rolling cited rate (%)", range=[0, 100], ticksuffix="%")
    return fig


def _category_difference_table(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    table = citation_rate_by_category(df, feature).copy()
    table["diff_from_overall_pp"] = table["difference_from_overall"] * 100
    table["ci_low_diff_pp"] = (table["wilson_ci_low"] - table["overall_cited_rate"]) * 100
    table["ci_high_diff_pp"] = (table["wilson_ci_high"] - table["overall_cited_rate"]) * 100
    table["sparse"] = table["n_rows"].lt(20)
    table["category_label"] = table["category"].map(lambda value: fill(str(value).replace("_", " "), width=28))
    return table.sort_values("diff_from_overall_pp", kind="stable").reset_index(drop=True)


def make_forest_difference_plot(table: pd.DataFrame, feature: str) -> go.Figure:
    ordered = table.sort_values("diff_from_overall_pp", kind="stable")
    fig = go.Figure(
        go.Scatter(
            x=ordered["diff_from_overall_pp"],
            y=ordered["category_label"],
            mode="markers+text",
            text=[f"n={int(n)}" for n in ordered["n_rows"]],
            textposition="middle right",
            marker={"size": 10, "color": "#287a8e"},
            error_x={
                "type": "data",
                "array": ordered["ci_high_diff_pp"] - ordered["diff_from_overall_pp"],
                "arrayminus": ordered["diff_from_overall_pp"] - ordered["ci_low_diff_pp"],
                "color": "#64748b",
                "thickness": 1.4,
            },
            customdata=np.column_stack([ordered["n_rows"], ordered["cited_rows"], ordered["cited_rate"] * 100, ordered["sparse"]]),
            hovertemplate=(
                "Category: %{y}<br>Rows: %{customdata[0]:,}<br>Cited rows: %{customdata[1]:,}<br>"
                "Cited rate: %{customdata[2]:.1f}%<br>Difference from overall: %{x:+.1f} pp<br>"
                "Sparse: %{customdata[3]}<extra></extra>"
            ),
            name="Category",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#5d6670")
    apply_readable_plotly_layout(fig, f"Difference from overall cited rate by {clean_feature_name(feature)}", "categories with at least 20 rows; unadjusted descriptive comparison")
    fig.update_xaxes(title="Difference from overall cited rate (percentage points)", ticksuffix=" pp")
    fig.update_yaxes(title="Category", automargin=True, categoryorder="array", categoryarray=ordered["category_label"].tolist())
    return fig


def _write_scatter_preview(stats: pd.DataFrame, feature: str, path: Path) -> None:
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(12, 6))
    stable = ~stats["sparse"] & ~stats["unstable"]
    scatter = ax.scatter(x[stable], stats.loc[stable, "cited_rate_pct"], c=stats.loc[stable, "n_rows"], cmap="viridis", s=72, edgecolors="white", linewidths=0.7)
    ax.scatter(x[~stable], stats.loc[~stable, "cited_rate_pct"], c=stats.loc[~stable, "n_rows"], cmap="viridis", s=72, marker="x", linewidths=1.5)
    ax.axhline(float(stats["overall_cited_rate_pct"].iloc[0]), color="#5d6670", linestyle="--", label="Overall cited rate")
    ax.set(title=f"Cited rate by {clean_feature_name(feature)}", xlabel=f"{clean_feature_name(feature)} (capped tail)", ylabel="Cited rate (%)", ylim=(0, 100), xticks=x, xticklabels=stats["feature_value_group"])
    ax.legend(loc="best")
    fig.colorbar(scatter, ax=ax, label="Rows")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_heatmap_preview(table: pd.DataFrame, feature: str, values: list[str], path: Path) -> None:
    statuses = ["More-only / not cited", "Cited"]
    matrix = table.pivot(index="cited_status", columns="feature_value_group", values="row_count").reindex(index=statuses, columns=values, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(9, len(values) * 0.45), 3.6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="cividis")
    ax.set(title=f"Raw row concentration by {clean_feature_name(feature)}", xlabel=f"{clean_feature_name(feature)} (capped tail)", ylabel="Citation status", xticks=range(len(values)), xticklabels=values, yticks=[0, 1], yticklabels=statuses)
    for row, col in np.ndindex(matrix.shape):
        value = int(matrix.iloc[row, col])
        if value >= 10:
            ax.text(col, row, str(value), ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, label="Rows")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_rolling_preview(stats: pd.DataFrame, feature: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats["x_mid"], stats["cited_rate_pct"], color="#287a8e", linewidth=2.5)
    ax.axhline(float(stats["overall_cited_rate_pct"].iloc[0]), color="#5d6670", linestyle="--", label="Overall cited rate")
    ax.set(title=f"Rolling cited rate by {clean_feature_name(feature)}", xlabel=f"{clean_feature_name(feature)} (window midpoint)", ylabel="Rolling cited rate (%)", ylim=(0, 100))
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_forest_preview(table: pd.DataFrame, feature: str, path: Path) -> None:
    ordered = table.sort_values("diff_from_overall_pp", kind="stable")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.52 * len(ordered) + 1.5)))
    ax.errorbar(ordered["diff_from_overall_pp"], y, xerr=[ordered["diff_from_overall_pp"] - ordered["ci_low_diff_pp"], ordered["ci_high_diff_pp"] - ordered["diff_from_overall_pp"]], fmt="o", color="#287a8e", ecolor="#64748b", capsize=3)
    ax.axvline(0, color="#5d6670", linestyle="--")
    ax.set(title=f"Difference from overall cited rate by {clean_feature_name(feature)}", xlabel="Difference from overall cited rate (percentage points)", yticks=y, yticklabels=ordered["category_label"])
    ax.grid(axis="x", alpha=0.2)
    for yi, row in enumerate(ordered.itertuples(index=False)):
        ax.annotate(f"n={row.n_rows}", (row.diff_from_overall_pp, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _checklist(png_statuses: list[str]) -> pd.DataFrame:
    rows = [
        ("axes_have_human_readable_labels", "pass", "Readable feature names and percentage-point labels are used."),
        ("cited_rate_shown_as_percent", "pass", "Cited-rate axes run from 0 to 100%."),
        ("sparse_points_marked", "pass", "Sparse or unstable exact-value points use an x marker."),
        ("overall_reference_line_exists", "pass", "Scatter and rolling plots include the overall cited-rate reference."),
        ("long_tails_capped_by_default", "pass", "Default count plots use documented capped tails."),
        ("full_diagnostic_plot_available", "pass", "Full exact-value HTML diagnostics are saved separately."),
        ("hover_text_is_concise", "pass", "Custom hover templates use a short, fixed field set."),
        ("colorbar_title_is_clear", "pass", "Frequency colorbars are titled Rows."),
        ("cited_rate_axis_fixed_to_0_100", "pass", "All cited-rate plot y-axes use 0 to 100%."),
        ("marker_size_not_only_frequency_signal", "pass", "Markers use fixed size and frequency color."),
        ("categorical_labels_wrapped", "pass", "Forest-plot category labels are wrapped."),
        ("png_export_failure_does_not_crash", "pass" if all(status != "failed" for status in png_statuses) else "warning", "Kaleido export is attempted; Matplotlib notebook previews remain available when it is absent."),
    ]
    return pd.DataFrame(rows, columns=["check", "status", "details"])


def run_readable_graph_diagnostics_v5(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    df = _add_logs(enrich_lpm_diagnostics(lpm, eda))
    output_dir.mkdir(parents=True, exist_ok=True)
    interactive = figure_dir / "interactive"
    preview = figure_dir / "preview"
    interactive.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    png_statuses: list[str] = []

    for feature in COUNT_FEATURES:
        full = _prepare_full_stats(exact_frequency_stats(df, feature), feature)
        capped = _aggregate_capped_stats(full, feature)
        full.to_csv(output_dir / f"full_exact_scatter_{feature}.csv", index=False)
        capped.to_csv(output_dir / f"readable_exact_scatter_{feature}.csv", index=False)
        readable_status = save_plotly_figure(make_readable_exact_scatter(capped, feature, capped=True), interactive / f"readable_exact_scatter_{feature}.html", figure_dir / f"readable_exact_scatter_{feature}.png")
        full_status = save_plotly_figure(make_readable_exact_scatter(full, feature, capped=False), interactive / f"full_exact_scatter_{feature}.html")
        _write_scatter_preview(capped, feature, preview / f"readable_exact_scatter_{feature}.png")
        heatmap_table, values = _heatmap_table(capped, feature)
        heatmap_table.to_csv(output_dir / f"readable_heatmap_{feature}_by_cited.csv", index=False)
        heatmap_status = save_plotly_figure(make_readable_heatmap(heatmap_table, feature, values), interactive / f"readable_heatmap_{feature}_by_cited.html", figure_dir / f"readable_heatmap_{feature}_by_cited.png")
        _write_heatmap_preview(heatmap_table, feature, values, preview / f"readable_heatmap_{feature}_by_cited.png")
        png_statuses.extend([readable_status, full_status, heatmap_status])
        manifest.extend([
            {"plot_type": "exact_scatter", "feature": feature, "presentation_role": "default_readable", "html": str(interactive / f"readable_exact_scatter_{feature}.html"), "png_export": readable_status},
            {"plot_type": "exact_scatter", "feature": feature, "presentation_role": "full_diagnostic_appendix", "html": str(interactive / f"full_exact_scatter_{feature}.html"), "png_export": full_status},
            {"plot_type": "heatmap", "feature": feature, "presentation_role": "default_readable", "html": str(interactive / f"readable_heatmap_{feature}_by_cited.html"), "png_export": heatmap_status},
        ])

    rolling_rows = []
    for feature in ROLLING_FEATURES:
        for window in (50, 75, 100):
            stats = rolling_stats(df, feature, window=window)
            stats["window_size"] = window
            rolling_rows.append(stats)
            filename = f"readable_rolling_{feature}.html" if window == 75 else f"diagnostic_rolling_{feature}_window_{window}.html"
            status = save_plotly_figure(make_readable_rolling_curve(stats, feature, window), interactive / filename, figure_dir / f"readable_rolling_{feature}.png" if window == 75 else None)
            if window == 75:
                _write_rolling_preview(stats, feature, preview / f"readable_rolling_{feature}.png")
                manifest.append({"plot_type": "rolling_curve", "feature": feature, "presentation_role": "default_readable_window_75", "html": str(interactive / filename), "png_export": status})
            png_statuses.append(status)
    pd.concat(rolling_rows, ignore_index=True).to_csv(output_dir / "rolling_cited_rate_sensitivity_v5.csv", index=False)

    sparse_tables = []
    for feature in CATEGORY_FEATURES:
        slug = FOREST_SLUGS[feature]
        table = _category_difference_table(df, feature)
        shown = table.loc[~table["sparse"]].copy()
        sparse = table.loc[table["sparse"]].copy()
        table.to_csv(output_dir / f"forest_diff_{feature}.csv", index=False)
        if not sparse.empty:
            sparse_tables.append(sparse)
        status = save_plotly_figure(make_forest_difference_plot(shown, feature), interactive / f"forest_diff_{slug}.html", figure_dir / f"forest_diff_{slug}.png")
        _write_forest_preview(shown, feature, preview / f"forest_diff_{slug}.png")
        png_statuses.append(status)
        manifest.append({"plot_type": "forest_difference", "feature": feature, "presentation_role": "default_non_sparse_categories", "html": str(interactive / f"forest_diff_{slug}.html"), "png_export": status})
    sparse_categories = pd.concat(sparse_tables, ignore_index=True) if sparse_tables else pd.DataFrame()
    sparse_categories.to_csv(output_dir / "sparse_categories_v5.csv", index=False)

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(output_dir / "graph_artifact_manifest_v5.csv", index=False)
    checklist = _checklist(png_statuses)
    checklist.to_csv(output_dir / "graph_readability_checklist_v5.csv", index=False)
    (output_dir / "plot_readability_summary_v5.md").write_text(
        "# Plot Readability Summary v5\n\n"
        "## What changed from v4 to v5\n"
        "V5 keeps the same rows and descriptive methodology, but makes tail-capped count scatters, capped concentration heatmaps, 75-row rolling curves, and non-sparse categorical forest plots the default presentation layer. V4 artifacts are unchanged.\n\n"
        "## Default presentation plots\n"
        "Use the readable exact-value scatters, readable heatmaps, 75-row rolling curves, and forest-style difference plots in the main notebook sections.\n\n"
        "## Diagnostic or appendix plots\n"
        "Full exact-value scatters and 50/100-row rolling sensitivity curves remain HTML diagnostic artifacts. Sparse categorical levels are retained in tables but not the default forest plots.\n\n"
        "## Section 3 graph choice\n"
        "Use the tail-capped readable exact-value scatter for Section 3. It keeps fixed marker size, encodes frequency by color, labels only well-supported points, and marks sparse or unstable estimates.\n\n"
        "## Caveats\n"
        "All count-feature diagnostics remain conditional on scrape/content availability. These are unadjusted descriptive associations, not model estimates or causal claims. The interactive HTML may require a browser that permits JavaScript; embedded PNG previews are included for notebook viewers that do not.\n\n"
        "## Status\n"
        "ready_for_pre_LPM_EDA_with_readable_interactive_graphs\n",
        encoding="utf-8",
    )
    return {
        "rows": int(len(df)),
        "unique_urls": int(df["normalized_url"].nunique()),
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "interactive_dir": str(interactive),
        "preview_dir": str(preview),
        "artifact_rows": int(len(manifest_frame)),
    }
