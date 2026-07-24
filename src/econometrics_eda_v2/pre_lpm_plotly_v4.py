"""Interactive Plotly numeric diagnostics for SCOPE pre-LPM EDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.econometrics_eda_v2.pre_lpm_diagnostics import _bool, _numeric, enrich_lpm_diagnostics, wilson_interval


COUNT_FEATURES = ("heading_count", "table_count", "link_count")
TAIL_CAPS = {"heading_count": 25, "table_count": 5, "link_count": 30}
HEATMAP_CAPS = {"heading_count": 25, "table_count": 4, "link_count": 30}
WINDOW = 50


def _eligible(df: pd.DataFrame) -> pd.Series:
    return _bool(df, "content_feature_available").eq(1) | _bool(df, "scraped_body_available").eq(1)


def _add_logs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for base in ("word_count", "text_char_count", "heading_count", "link_count"):
        out[f"log1p_{base}"] = np.log1p(_numeric(out, base).clip(lower=0))
    return out


def exact_frequency_stats(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    values = _numeric(df, feature)
    cited = _bool(df, "cited")
    mask = _eligible(df) & values.notna()
    overall = float(cited.mean())
    rows = []
    for value, indices in values.loc[mask].groupby(values.loc[mask]).groups.items():
        outcome = cited.loc[indices]
        n = len(outcome)
        cited_rows = int(outcome.sum())
        more_only = int(n - cited_rows)
        rate = cited_rows / n
        low, high = wilson_interval(cited_rows, n)
        sparse = n < 20
        unstable = cited_rows < 5 or more_only < 5
        rows.append({
            "feature_name": feature,
            "feature_value": float(value),
            "n_rows": n,
            "cited_rows": cited_rows,
            "more_only_rows": more_only,
            "cited_rate": rate,
            "cited_rate_pct": rate * 100,
            "overall_cited_rate": overall,
            "overall_cited_rate_pct": overall * 100,
            "diff_from_overall_pp": (rate - overall) * 100,
            "ci_low": low,
            "ci_high": high,
            "ci_low_pct": low * 100,
            "ci_high_pct": high * 100,
            "sparse_flag": "sparse_n_lt_20" if sparse else "",
            "unstable_flag": "unstable_cited_or_more_only_lt_5" if unstable else "",
        })
    return pd.DataFrame(rows).sort_values("feature_value", kind="stable")


def capped_frequency_stats(stats: pd.DataFrame, feature: str) -> pd.DataFrame:
    cap = TAIL_CAPS[feature]
    work = stats.copy()
    work["feature_value_group"] = work["feature_value"].map(lambda value: f"{cap + 1}+" if value > cap else str(int(value)))
    rows = []
    for label, group in work.groupby("feature_value_group", sort=False):
        n = int(group["n_rows"].sum())
        cited = int(group["cited_rows"].sum())
        more_only = int(group["more_only_rows"].sum())
        rate = cited / n
        low, high = wilson_interval(cited, n)
        overall = float(group["overall_cited_rate"].iloc[0])
        tail = label.endswith("+")
        rows.append({
            "feature_name": feature,
            "feature_value_group": label,
            "feature_value_sort": cap + 1 if tail else int(label),
            "n_rows": n,
            "cited_rows": cited,
            "more_only_rows": more_only,
            "cited_rate": rate,
            "cited_rate_pct": rate * 100,
            "overall_cited_rate": overall,
            "overall_cited_rate_pct": overall * 100,
            "diff_from_overall_pp": (rate - overall) * 100,
            "ci_low": low,
            "ci_high": high,
            "ci_low_pct": low * 100,
            "ci_high_pct": high * 100,
            "sparse_flag": "sparse_n_lt_20" if n < 20 else "",
            "unstable_flag": "unstable_cited_or_more_only_lt_5" if cited < 5 or more_only < 5 else "",
            "tail_grouped": tail,
        })
    return pd.DataFrame(rows).sort_values("feature_value_sort", kind="stable")


def _write_png_if_possible(fig: go.Figure, path: Path) -> str:
    try:
        fig.write_image(path, scale=2)
        return "written"
    except Exception as exc:  # Kaleido is optional.
        return f"skipped: {type(exc).__name__}"


def write_exact_frequency_preview(stats: pd.DataFrame, feature: str, path: Path) -> None:
    """Write a non-JavaScript preview of the interactive scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        stats["feature_value"],
        stats["cited_rate_pct"],
        c=stats["n_rows"],
        cmap="viridis",
        s=64,
        edgecolors="white",
        linewidths=0.6,
        alpha=0.9,
    )
    ax.axhline(float(stats["overall_cited_rate_pct"].iloc[0]), color="#5d6670", linestyle="--", linewidth=1.25, label="Overall cited rate")
    ax.set(title=f"Exact cited rate by {feature}", xlabel=feature, ylabel="Cited rate (%)")
    ax.legend(loc="best")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Rows")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_concentration_preview(table: pd.DataFrame, feature: str, path: Path) -> None:
    """Write a non-JavaScript preview of the Plotly count heatmap."""
    groups = table["feature_value_group"].drop_duplicates().tolist()
    matrix = (
        table.pivot(index="cited", columns="feature_value_group", values="row_count")
        .reindex(index=[0, 1], columns=groups, fill_value=0)
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(max(10, len(groups) * 0.6), 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set(
        title=f"Raw row concentration by exact {feature} and cited status",
        xlabel=f"{feature} exact value / grouped tail",
        ylabel="Cited status",
        xticks=range(len(groups)),
        xticklabels=groups,
        yticks=[0, 1],
        yticklabels=["0: surfaced / more-only", "1: cited"],
    )
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Rows")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_rolling_preview(stats: pd.DataFrame, feature: str, path: Path) -> None:
    """Write a non-JavaScript preview of the Plotly rolling curve."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stats["x_mid"], stats["cited_rate_pct"], color="#3f7f8e", linewidth=2)
    ax.axhline(float(stats["overall_cited_rate_pct"].iloc[0]), color="#5d6670", linestyle="--", linewidth=1.25, label="Overall cited rate")
    ax.set(title=f"Rolling cited rate: {feature}", xlabel=f"{feature} (rolling-window median)", ylabel="Rolling cited rate (%)")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_exact_frequency_scatter(
    stats: pd.DataFrame,
    feature: str,
    *,
    show_ci: bool = False,
    categorical: bool = False,
) -> go.Figure:
    x = "feature_value_group" if categorical else "feature_value"
    labels = {x: f"{feature} (grouped)" if categorical else feature, "cited_rate_pct": "Cited rate (%)", "n_rows": "Rows"}
    fig = px.scatter(
        stats,
        x=x,
        y="cited_rate_pct",
        color="n_rows",
        color_continuous_scale="Viridis",
        hover_data={
            x: True, "n_rows": True, "cited_rows": True, "more_only_rows": True,
            "cited_rate_pct": ":.1f", "diff_from_overall_pp": ":.1f",
            "ci_low_pct": ":.1f", "ci_high_pct": ":.1f", "sparse_flag": True, "unstable_flag": True,
        },
        title=f"Exact cited rate by {feature} — color shows row frequency" + (" (tail-capped groups)" if categorical else ""),
        labels=labels,
    )
    fig.update_traces(marker={"size": 12, "opacity": 0.88, "line": {"width": 0.5, "color": "white"}})
    if show_ci:
        fig.update_traces(error_y={"type": "data", "array": stats["ci_high_pct"] - stats["cited_rate_pct"], "arrayminus": stats["cited_rate_pct"] - stats["ci_low_pct"], "visible": True})
    fig.add_hline(y=float(stats["overall_cited_rate_pct"].iloc[0]), line_dash="dash", line_color="#5d6670", annotation_text="Overall cited rate", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=520, width=900, coloraxis_colorbar={"title": "Rows"}, margin={"l": 70, "r": 80, "t": 70, "b": 70})
    if categorical:
        fig.update_xaxes(categoryorder="array", categoryarray=stats[x].tolist())
    return fig


def plot_exact_frequency_scatter(stats: pd.DataFrame, feature: str, html_path: Path, png_path: Path | None = None, show_ci: bool = False, categorical: bool = False) -> tuple[go.Figure, str]:
    fig = make_exact_frequency_scatter(
        stats,
        feature,
        show_ci=show_ci,
        categorical=categorical,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    png_status = _write_png_if_possible(fig, png_path) if png_path else "not_requested"
    return fig, png_status


def concentration_heatmap(df: pd.DataFrame, feature: str) -> tuple[go.Figure, pd.DataFrame]:
    values = _numeric(df, feature)
    cited = _bool(df, "cited")
    mask = _eligible(df) & values.notna()
    cap = HEATMAP_CAPS[feature]
    group = values.loc[mask].map(lambda value: f"{cap + 1}+" if value > cap else str(int(value)))
    order = [str(i) for i in range(cap + 1)] + [f"{cap + 1}+"]
    work = pd.DataFrame({"feature_value": group, "cited": cited.loc[mask]})
    counts = pd.crosstab(work["cited"], work["feature_value"]).reindex(index=[0, 1], columns=order, fill_value=0)
    total_by_value = counts.sum(axis=0).replace(0, np.nan)
    share = counts.divide(total_by_value, axis=1).fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=counts.to_numpy(), x=order, y=["0: surfaced / more-only", "1: cited"], colorscale="Blues",
        customdata=share.to_numpy(),
        hovertemplate=f"{feature}=%{{x}}<br>status=%{{y}}<br>rows=%{{z}}<br>share within value=%{{customdata:.1%}}<extra></extra>",
        colorbar={"title": "Rows"},
    ))
    fig.update_layout(template="plotly_white", height=340, width=max(850, len(order) * 35), title=f"Raw row concentration by exact {feature} and cited status", xaxis_title=f"{feature} exact value / grouped tail", yaxis_title="Cited status")
    table = counts.reset_index(names="cited").melt(id_vars="cited", var_name="feature_value_group", value_name="row_count")
    table["feature_name"] = feature
    table["share_of_feature_value"] = table.apply(lambda row: float(share.loc[row.cited, row.feature_value_group]), axis=1)
    return fig, table


def rolling_stats(df: pd.DataFrame, feature: str, window: int = WINDOW) -> pd.DataFrame:
    values = _numeric(df, feature)
    cited = _bool(df, "cited")
    mask = _eligible(df) & values.notna()
    work = pd.DataFrame({"value": values.loc[mask], "cited": cited.loc[mask]}).sort_values("value", kind="stable").reset_index(drop=True)
    overall = float(cited.mean())
    rows = []
    for start in range(0, len(work) - window + 1):
        group = work.iloc[start:start + window]
        rows.append({"feature_name": feature, "window_start": start, "window_end": start + window - 1, "x_mid": float(group.value.median()), "x_min": float(group.value.min()), "x_max": float(group.value.max()), "n_rows": len(group), "cited_rate": float(group.cited.mean()), "cited_rate_pct": float(group.cited.mean() * 100), "overall_cited_rate": overall, "overall_cited_rate_pct": overall * 100})
    return pd.DataFrame(rows)


def make_rolling_curve(stats: pd.DataFrame, feature: str) -> go.Figure:
    fig = px.line(stats, x="x_mid", y="cited_rate_pct", hover_data={"x_mid": ":.3f", "x_min": ":.3f", "x_max": ":.3f", "n_rows": True, "cited_rate_pct": ":.1f"}, title=f"Rolling cited rate: {feature}", labels={"x_mid": f"{feature} (rolling-window median)", "cited_rate_pct": "Rolling cited rate (%)"})
    fig.update_traces(line={"color": "#3f7f8e", "width": 2})
    fig.add_hline(y=float(stats.overall_cited_rate_pct.iloc[0]), line_dash="dash", line_color="#5d6670", annotation_text="Overall cited rate", annotation_position="top left")
    fig.update_layout(template="plotly_white", height=500, width=900, margin={"l": 70, "r": 35, "t": 70, "b": 70})
    return fig


def plot_rolling_curve(stats: pd.DataFrame, feature: str, html_path: Path) -> go.Figure:
    fig = make_rolling_curve(stats, feature)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True)
    return fig


def run_plotly_diagnostics_v4(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    df = _add_logs(enrich_lpm_diagnostics(lpm, eda))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    interactive = figure_dir / "interactive"
    interactive.mkdir(parents=True, exist_ok=True)
    preview = figure_dir / "preview"
    preview.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for feature in COUNT_FEATURES:
        stats = exact_frequency_stats(df, feature)
        stats.to_csv(output_dir / f"exact_value_frequency_scatter_{feature}.csv", index=False)
        capped = capped_frequency_stats(stats, feature)
        capped.to_csv(output_dir / f"exact_value_frequency_scatter_{feature}_capped.csv", index=False)
        clean, png_status = plot_exact_frequency_scatter(stats, feature, interactive / f"plotly_exact_frequency_scatter_{feature}_clean.html", figure_dir / f"plotly_exact_frequency_scatter_{feature}.png")
        write_exact_frequency_preview(stats, feature, preview / f"plotly_exact_frequency_scatter_{feature}_clean.png")
        clean.write_html(interactive / f"plotly_exact_frequency_scatter_{feature}.html", include_plotlyjs="cdn", full_html=True)
        plot_exact_frequency_scatter(stats, feature, interactive / f"plotly_exact_frequency_scatter_{feature}_ci.html", show_ci=True)
        plot_exact_frequency_scatter(stats, feature, interactive / f"plotly_exact_frequency_scatter_{feature}_full.html")
        plot_exact_frequency_scatter(capped, feature, interactive / f"plotly_exact_frequency_scatter_{feature}_capped.html", categorical=True)
        heatmap, heatmap_table = concentration_heatmap(df, feature)
        heatmap.write_html(interactive / f"plotly_count_heatmap_{feature}_by_cited.html", include_plotlyjs="cdn", full_html=True)
        heatmap_table.to_csv(output_dir / f"plotly_count_heatmap_{feature}_by_cited.csv", index=False)
        write_concentration_preview(heatmap_table, feature, preview / f"plotly_count_heatmap_{feature}_by_cited.png")
        summary_rows.append({"artifact_type": "exact_frequency_scatter", "feature_name": feature, "full_exact_values": len(stats), "capped_groups": len(capped), "clean_html": str(interactive / f"plotly_exact_frequency_scatter_{feature}_clean.html"), "ci_html": str(interactive / f"plotly_exact_frequency_scatter_{feature}_ci.html"), "png_export": png_status})
    rolling_tables = []
    for feature in ("log1p_word_count", "heading_count", "link_count"):
        stats = rolling_stats(df, feature)
        rolling_tables.append(stats)
        plot_rolling_curve(stats, feature, interactive / f"plotly_rolling_cited_rate_{feature}.html")
        write_rolling_preview(stats, feature, preview / f"plotly_rolling_cited_rate_{feature}.png")
        summary_rows.append({"artifact_type": "rolling_curve", "feature_name": feature, "full_exact_values": np.nan, "capped_groups": np.nan, "clean_html": str(interactive / f"plotly_rolling_cited_rate_{feature}.html"), "ci_html": "", "png_export": "not_requested"})
    pd.concat(rolling_tables, ignore_index=True).to_csv(output_dir / "plotly_rolling_cited_rate_numeric_features.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "plotly_numeric_diagnostics_summary_v4.csv", index=False)
    (output_dir / "plotly_exact_frequency_scatter_summary_v4.md").write_text(
        "# Plotly Exact-Value Frequency Scatter Summary\n\n"
        "Section 3 replaces bubble-size frequency encoding with fixed-size Plotly markers colored by row frequency. Hover reveals exact counts, cited rate, difference from the overall rate, Wilson interval, and sparse/unstable flags.\n\n"
        "- **Clean scatter**: primary readability view, with fixed marker size and frequency color.\n"
        "- **CI scatter**: diagnostic uncertainty view; it can be visually dense for sparse tails.\n"
        "- **Heatmap**: raw concentration by count value and cited status, before conversion to cited rates.\n"
        "- **Tail-capped scatter**: groups the long tail and states that its x-axis is categorical.\n\n"
        "A darker point has more supporting observations, but the plots remain descriptive. Treat `n_rows < 20` as sparse and any count with fewer than five cited or more-only rows as unstable. HTML is preferred for exploration; static PNG is attempted only when the optional Kaleido exporter is available.\n",
        encoding="utf-8",
    )
    return {"rows": int(len(df)), "unique_urls": int(df.normalized_url.nunique()), "output_dir": str(output_dir), "figure_dir": str(figure_dir), "interactive_dir": str(interactive), "preview_dir": str(preview), "summary_rows": int(len(summary))}
