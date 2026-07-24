"""Reusable Plotly charts for versioned econometrics frontend artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .theme import COLORS


MODEL_ORDER = ["G1", "G2", "G2R", "G3", "G4A", "G4B", "G5A", "G5B", "G5C", "G7", "G8", "G9"]


def _layout(fig: go.Figure, *, height: int = 420, x_title: str = "", y_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=66, b=44),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=COLORS["text"]),
        legend_title_text="",
        xaxis_title=x_title,
        yaxis_title=y_title,
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="#eef0f6", zeroline=False)
    fig.update_yaxes(gridcolor="#eef0f6", zeroline=False)
    return fig


def cited_rate_plot(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.sort_values("bin_order").copy()
    d["label"] = d["feature_level"].astype(str)
    d["support"] = d.apply(lambda row: f"n={int(row.n_rows):,}; cited={int(row.n_cited):,}; prompts={int(row.n_prompts):,}; URLs={int(row.n_urls):,}", axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["label"], y=d["cited_rate"], mode="markers+lines+text",
        text=d["n_rows"].map(lambda value: f"n={int(value):,}"), textposition="top center",
        marker=dict(size=11, color=COLORS["primary"], symbol="circle"),
        line=dict(color=COLORS["primary"], width=2),
        error_y=dict(type="data", symmetric=False, array=d["ci_upper"] - d["cited_rate"], arrayminus=d["cited_rate"] - d["ci_lower"], color=COLORS["text"], thickness=1.4),
        customdata=np.stack([d["support"], d["ci_lower"], d["ci_upper"]], axis=-1),
        hovertemplate="%{x}<br>Cited rate %{y:.1%}<br>95% CI %{customdata[1]:.1%} to %{customdata[2]:.1%}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    fig.update_layout(title=dict(text="Unadjusted descriptive association"))
    return _layout(fig, height=430, x_title=human_label, y_title="Cited rate")


def distribution_plot(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.sort_values("bin_order").copy()
    d["citation_status"] = "Cited"
    cited = d[["feature_level", "n_cited"]].rename(columns={"n_cited": "rows"}).assign(citation_status="Cited")
    uncited = d[["feature_level", "n_more_only"]].rename(columns={"n_more_only": "rows"}).assign(citation_status="Not cited")
    stacked = pd.concat([cited, uncited], ignore_index=True)
    fig = px.bar(stacked, x="feature_level", y="rows", color="citation_status", barmode="group",
                 color_discrete_map={"Cited": COLORS["cited"], "Not cited": COLORS["noncited"]}, text_auto=True)
    fig.update_layout(title=dict(text="Feature support distribution"))
    return _layout(fig, height=390, x_title=human_label, y_title="Surfaced-source rows")


def model_forest(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.copy()
    d["model_rank"] = d["model_id"].map({value: index for index, value in enumerate(MODEL_ORDER)}).fillna(99)
    d = d.sort_values(["model_rank", "source_model_id", "term_label"])
    d["display"] = d["model_id"] + " | " + d["term_label"].astype(str)
    d["support"] = d.apply(lambda row: f"n={int(row.n_rows):,}; prompts={int(row.n_prompts):,}; URLs={int(row.n_urls):,}; domains={int(row.n_domains):,}", axis=1)
    finite = np.isfinite(pd.to_numeric(d["ci_lower_pp"], errors="coerce")) & np.isfinite(pd.to_numeric(d["ci_upper_pp"], errors="coerce"))
    d = d[finite]
    fig = go.Figure(go.Scatter(
        x=d["estimate_pp"], y=d["display"], mode="markers",
        marker=dict(size=10, color=d["model_role"].map({"headline": COLORS["primary"], "robustness": "#0891b2", "sensitivity": "#d97706", "diagnostic": COLORS["noncited"]}).fillna(COLORS["primary"])),
        error_x=dict(type="data", symmetric=False, array=d["ci_upper_pp"] - d["estimate_pp"], arrayminus=d["estimate_pp"] - d["ci_lower_pp"], thickness=1.4),
        customdata=np.stack([d["source_model_id"], d["model_role"], d["se_method"], d["support"], d["ci_lower_pp"], d["ci_upper_pp"], d["model_change"]], axis=-1),
        hovertemplate="%{y}<br>%{x:+.2f} pp (95% CI %{customdata[4]:+.2f} to %{customdata[5]:+.2f})<br>%{customdata[3]}<br>SE: %{customdata[2]}<br>%{customdata[6]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Model-adjusted association"))
    return _layout(fig, height=max(420, 42 * len(d) + 120), x_title="Association (percentage points)", y_title="")


def coefficient_path(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.copy()
    rank = {value: index for index, value in enumerate(MODEL_ORDER)}
    d["rank"] = d["model_id"].map(rank).fillna(99)
    d = d.sort_values(["term_label", "rank"])
    fig = px.line(d, x="model_id", y="estimate_pp", color="term_label", markers=True,
                  category_orders={"model_id": MODEL_ORDER}, hover_data=["ci_lower_pp", "ci_upper_pp", "n_rows", "se_method"])
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Coefficient path"))
    return _layout(fig, height=410, x_title="Model", y_title="Association (percentage points)")


def probability_contrast(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.copy()
    rows = []
    for _, row in d.iterrows():
        rows.extend([
            {"contrast": row["contrast_name"], "condition": row["condition_a"], "probability": row["probability_a"]},
            {"contrast": row["contrast_name"], "condition": row["condition_b"], "probability": row["probability_b"]},
        ])
    plot = pd.DataFrame(rows)
    fig = px.bar(plot, x="condition", y="probability", color="condition", text=plot["probability"].map(lambda value: f"{value:.1%}"))
    fig.update_yaxes(tickformat=".0%", range=[0, min(1, max(.5, plot["probability"].max() * 1.25))])
    fig.update_layout(showlegend=False, title=dict(text="Model-implied probability contrast"))
    return _layout(fig, height=390, x_title="Condition", y_title="Model-implied probability")


def subgroup_heatmap(data: pd.DataFrame, dimension: str, version: str) -> go.Figure:
    d = data.copy()
    pivot = d.pivot_table(index="subgroup_name", columns="feature_state", values="cited_rate", aggfunc="first")
    annotations = d.pivot_table(index="subgroup_name", columns="feature_state", values="n_rows", aggfunc="first")
    text = annotations.map(lambda value: f"{value:.0f}" if pd.notna(value) else "")
    fig = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=pivot.columns, y=pivot.index, text=text.to_numpy(), texttemplate="%{z:.1%}<br>n=%{text}", colorscale="RdYlGn", zmin=0, zmax=1, colorbar_title="Cited rate", hovertemplate="%{y}<br>%{x}<br>Cited rate %{z:.1%}<br>n=%{text}<extra></extra>"))
    fig.update_layout(title=dict(text="Descriptive subgroup comparison"))
    return _layout(fig, height=max(380, 34 * len(pivot) + 150), x_title="Feature state", y_title=dimension.replace("_", " ").title())


def related_association_chart(data: pd.DataFrame, version: str) -> go.Figure:
    d = data.dropna(subset=["association"]).copy().sort_values("association")
    d["color"] = np.where(d["association"] >= 0, COLORS["cited"], "#d97706")
    fig = go.Figure(go.Bar(x=d["association"], y=d["related_feature"], orientation="h", marker_color=d["color"], text=d["association"].map(lambda value: f"{value:+.2f}"), textposition="outside", customdata=np.stack([d["association_measure"], d["pairwise_n"], d["missing_rate"]], axis=-1), hovertemplate="%{y}<br>%{x:+.3f}<br>Measure: %{customdata[0]}<br>Pairwise n=%{customdata[1]:,.0f}<br>Missing %{customdata[2]:.1%}<extra></extra>"))
    fig.add_vline(x=0, line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Related measured features"))
    return _layout(fig, height=max(370, 48 * len(d) + 110), x_title="Association", y_title="")


def sample_flow(data: pd.DataFrame, version: str) -> go.Figure:
    d = data.sort_values("stage_order")
    fig = go.Figure(go.Funnel(y=d["stage"].str.replace("_", " ").str.title(), x=d["n_rows"], textinfo="value+percent initial", marker_color=[COLORS["primary"], "#6366f1", "#0891b2", "#0d9488", COLORS["cited"]][:len(d)], customdata=np.stack([d["cited_rate"], d["n_prompts"], d["n_urls"]], axis=-1), hovertemplate="%{y}<br>Rows %{x:,}<br>Cited rate %{customdata[0]:.1%}<br>Prompts %{customdata[1]:,.0f}<br>URLs %{customdata[2]:,.0f}<extra></extra>"))
    fig.update_layout(title=dict(text="Sample and measurement flow"))
    return _layout(fig, height=430, x_title="Rows", y_title="")


def comparable_difference(pair: pd.Series, version: str) -> go.Figure:
    rows = []
    for label, cited_key, uncited_key in [
        ("Page length (words)", "cited_word_count", "uncited_word_count"),
        ("Prompt-page relevance", "cited_relevance", "uncited_relevance"),
    ]:
        rows.append({"variable": label, "page": "Cited page", "value": pair[cited_key]})
        rows.append({"variable": label, "page": "Not-cited page", "value": pair[uncited_key]})
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="variable", y="value", color="page", barmode="group", text_auto=".2f",
                 color_discrete_map={"Cited page": COLORS["cited"], "Not-cited page": COLORS["noncited"]})
    fig.update_layout(title=dict(text="Displayed numeric differences"))
    return _layout(fig, height=360, x_title="", y_title="Measured value")


def cross_model_path(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.copy()
    rank = {value: index for index, value in enumerate(MODEL_ORDER)}
    d["_rank"] = d["model_id"].map(rank).fillna(99)
    d = d.sort_values(["_rank", "source_model_id"])
    d["display"] = d["model_id"] + " | " + d["source_model_id"].astype(str)
    d["role_color"] = d["model_role"].map({
        "headline": COLORS["primary"], "robustness": "#0891b2", "sensitivity": "#d97706",
        "diagnostic": COLORS["noncited"], "cross_check": "#7c3aed",
    }).fillna(COLORS["muted"])
    d["support"] = d.apply(
        lambda row: f"rows={int(row.n_rows):,}; prompts={int(row.n_prompts):,}; URLs={int(row.n_urls):,}; domains={int(row.n_domains):,}",
        axis=1,
    )
    d["change_detail"] = d.apply(
        lambda row: (
            f"FE: {row.fixed_effects}<br>Controls: {row.controls}<br>Sample: {row.sample_restriction}"
            f"<br>SE: {row.se_method}<br>Unit: {row.interpretation_unit}<br>Status: {row.comparability_status}"
        ),
        axis=1,
    )
    fig = go.Figure(go.Scatter(
        x=d["estimate_pp"], y=d["display"], mode="markers",
        marker=dict(size=11, color=d["role_color"], line=dict(width=1, color="white")),
        error_x=dict(type="data", symmetric=False, array=d["ci_upper_pp"] - d["estimate_pp"], arrayminus=d["estimate_pp"] - d["ci_lower_pp"], thickness=1.5),
        customdata=np.stack([d["model_role"], d["support"], d["change_detail"], d["ci_lower_pp"], d["ci_upper_pp"]], axis=-1),
        hovertemplate="%{y}<br>%{x:+.2f} pp (95% CI %{customdata[3]:+.2f} to %{customdata[4]:+.2f})<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Association across model specifications"))
    fig.update_yaxes(automargin=True)
    return _layout(fig, height=max(430, 45 * len(d) + 130), x_title="Association (percentage points)", y_title="")


def model_transition_matrix(data: pd.DataFrame, version: str) -> go.Figure:
    d = data.copy()
    d["transition"] = d["baseline_model_id"] + " to " + d["comparison_model_id"]
    pivot = d.pivot_table(index="feature_label", columns="transition", values="estimate_change_pp", aggfunc="first")
    labels = d.pivot_table(index="feature_label", columns="transition", values="primary_label", aggfunc="first")
    text = np.empty(pivot.shape, dtype=object)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            label = labels.iloc[row, column] if row < labels.shape[0] and column < labels.shape[1] else ""
            text[row, column] = "" if pd.isna(value) else f"{value:+.1f} pp<br>{label}"
    finite = np.abs(pivot.to_numpy(dtype=float))
    limit = max(2.0, float(np.nanpercentile(finite, 90))) if np.isfinite(finite).any() else 2.0
    fig = go.Figure(go.Heatmap(
        z=pivot.to_numpy(), x=pivot.columns, y=pivot.index, text=text, texttemplate="%{text}",
        colorscale="RdBu", reversescale=True, zmid=0, zmin=-limit, zmax=limit,
        colorbar_title="Estimate change (pp)",
        hovertemplate="%{y}<br>%{x}<br>Estimate change %{z:+.2f} pp<extra></extra>",
    ))
    fig.update_layout(title=dict(text="Cross-model transition matrix"))
    fig.update_xaxes(tickangle=-25, automargin=True)
    fig.update_yaxes(automargin=True)
    return _layout(fig, height=max(480, 45 * len(pivot) + 180), x_title="Predefined model transition", y_title="Feature")


def covariance_forest(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data.copy().sort_values("comparison_se_method")
    rows = []
    reference = d.iloc[0]
    rows.append({
        "se_method": reference["reference_se_method"], "estimate_pp": reference["estimate_pp"],
        "ci_low": reference["estimate_pp"] - reference["reference_ci_width_pp"] / 2,
        "ci_high": reference["estimate_pp"] + reference["reference_ci_width_pp"] / 2,
        "status": "reference",
    })
    for _, row in d.iterrows():
        rows.append({
            "se_method": row["comparison_se_method"], "estimate_pp": row["comparison_estimate_pp"],
            "ci_low": row["comparison_estimate_pp"] - row["comparison_ci_width_pp"] / 2,
            "ci_high": row["comparison_estimate_pp"] + row["comparison_ci_width_pp"] / 2,
            "status": row["inference_status"],
        })
    plot = pd.DataFrame(rows).drop_duplicates("se_method")
    finite = np.isfinite(plot[["estimate_pp", "ci_low", "ci_high"]]).all(axis=1)
    plot = plot[finite]
    fig = go.Figure(go.Scatter(
        x=plot["estimate_pp"], y=plot["se_method"], mode="markers",
        marker=dict(size=10, color=plot["status"].map({"reference": COLORS["primary"], "inference_stable": "#0891b2", "inference_sensitive": "#d97706"}).fillna(COLORS["muted"])),
        error_x=dict(type="data", symmetric=False, array=plot["ci_high"] - plot["estimate_pp"], arrayminus=plot["estimate_pp"] - plot["ci_low"]),
        customdata=np.stack([plot["ci_low"], plot["ci_high"], plot["status"]], axis=-1),
        hovertemplate="%{y}<br>%{x:+.2f} pp (95% CI %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<br>%{customdata[2]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Point estimate versus uncertainty"))
    fig.update_yaxes(automargin=True)
    return _layout(fig, height=max(360, 48 * len(plot) + 130), x_title="Association (percentage points)", y_title="")


def intent_slope_forest(data: pd.DataFrame, human_label: str, version: str) -> go.Figure:
    d = data[np.isfinite(pd.to_numeric(data["ci_lower_pp"], errors="coerce")) & np.isfinite(pd.to_numeric(data["ci_upper_pp"], errors="coerce"))].copy()
    d = d.sort_values("estimate_pp")
    fig = go.Figure(go.Scatter(
        x=d["estimate_pp"], y=d["intent"], mode="markers",
        marker=dict(size=10, color=COLORS["primary"]),
        error_x=dict(type="data", symmetric=False, array=d["ci_upper_pp"] - d["estimate_pp"], arrayminus=d["estimate_pp"] - d["ci_lower_pp"]),
        customdata=np.stack([d["ci_lower_pp"], d["ci_upper_pp"], d["se_method"]], axis=-1),
        hovertemplate="%{y}<br>%{x:+.2f} pp (95% CI %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<br>SE: %{customdata[2]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    fig.update_layout(title=dict(text="Intent-specific slopes"))
    fig.update_yaxes(automargin=True)
    return _layout(fig, height=max(430, 42 * len(d) + 130), x_title="Intent-specific association (percentage points)", y_title="")
