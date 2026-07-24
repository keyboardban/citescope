"""Intent-stratified descriptive diagnostics for the SCOPE pre-LPM EDA."""

from __future__ import annotations

from pathlib import Path
from re import sub
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.econometrics_eda_v2.pre_lpm_diagnostics import _bool, _category, wilson_interval
from src.econometrics_eda_v2.pre_lpm_readable_graphs_v5 import apply_readable_plotly_layout, save_plotly_figure


INTENT_CANDIDATES = ("intent", "prompt_intent", "question_intent", "query_intent", "theme", "prompt_theme", "prompt_category", "expected_intent")
TYPE_FEATURES = ("source_type_real_estate", "page_type_family_real_estate")


def _slug(value: str) -> str:
    return sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_") or "missing"


def _category_label(value: object) -> str:
    return fill(str(value).replace("_", " "), width=24)


def _heatmap_label(value: object) -> str:
    return fill(str(value).replace("_", " "), width=10)


def _intent_audit(lpm: pd.DataFrame, eda: pd.DataFrame | None) -> tuple[pd.DataFrame, str | None, pd.DataFrame, str]:
    """Select intent from the model frame, or attach an EDA prompt-level mapping."""
    rows = []
    selected: str | None = None
    selected_source = "none"
    work = lpm.copy()
    for column in INTENT_CANDIDATES:
        values = work[column] if column in work else pd.Series(dtype=object)
        non_null = values.notna() & values.astype(str).str.strip().ne("") if column in work else pd.Series(dtype=bool)
        rows.append({
            "candidate_column": column,
            "exists": column in work,
            "non_null_count": int(non_null.sum()) if column in work else 0,
            "unique_count": int(values.loc[non_null].nunique()) if column in work else 0,
            "example_values": " | ".join(values.loc[non_null].astype(str).drop_duplicates().head(5).tolist()) if column in work else "",
            "selected_as_intent_group": False,
            "notes": "available_in_lpm_ready" if column in work else "not_in_lpm_ready",
        })
    for row in rows:
        if row["exists"] and row["non_null_count"] and row["unique_count"] >= 1:
            selected = str(row["candidate_column"])
            selected_source = "lpm_ready"
            break
    if selected is None and eda is not None and "prompt_id" in work and "prompt_id" in eda:
        for column in INTENT_CANDIDATES:
            if column not in eda:
                continue
            source = eda[["prompt_id", column]].dropna().copy()
            source = source[source[column].astype(str).str.strip().ne("")]
            if source.empty:
                continue
            mapping = source.groupby("prompt_id", dropna=False)[column].agg(lambda values: values.astype(str).mode().iloc[0]).rename("intent_group")
            work = work.merge(mapping, left_on="prompt_id", right_index=True, how="left", validate="many_to_one")
            selected = column
            selected_source = "eda_prompt_mapping"
            for row in rows:
                if row["candidate_column"] == column:
                    row["notes"] = "selected_from_eda_by_prompt_id"
            break
    if selected is not None:
        if selected_source == "lpm_ready":
            work["intent_group"] = work[selected]
        work["intent_group"] = _category(work, "intent_group")
        for row in rows:
            if row["candidate_column"] == selected:
                row["selected_as_intent_group"] = True
    return pd.DataFrame(rows), selected, work, selected_source


def _intent_distribution(df: pd.DataFrame) -> pd.DataFrame:
    cited = _bool(df, "cited")
    rows = []
    for intent, group in df.groupby("intent_group", dropna=False):
        outcome = cited.loc[group.index]
        rows.append({
            "intent_group": intent,
            "n_rows": int(len(group)),
            "unique_prompts": int(group["prompt_id"].nunique()),
            "unique_urls": int(group["normalized_url"].nunique()),
            "cited_rows": int(outcome.sum()),
            "cited_rate": float(outcome.mean()),
            "share_of_rows": len(group) / len(df),
        })
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False, kind="stable").reset_index(drop=True)


def _cell_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    cited = _bool(df, "cited")
    category = _category(df, feature)
    overall = float(cited.mean())
    rows = []
    for (intent, value), indices in pd.DataFrame({"intent": df["intent_group"], "value": category}).groupby(["intent", "value"], dropna=False).groups.items():
        group = df.loc[indices]
        n = len(group)
        cited_rows = int(cited.loc[indices].sum())
        more_only = n - cited_rows
        prompts = int(group["prompt_id"].nunique())
        ci_low, ci_high = wilson_interval(cited_rows, n)
        if prompts < 3:
            status = "weak_prompt_coverage"
        elif n < 20:
            status = "descriptive_only_sparse"
        elif cited_rows < 5 or more_only < 5:
            status = "unstable_cited_rate"
        else:
            status = "eligible"
        intent_mask = df["intent_group"].eq(intent)
        cited_in_intent = int(cited.loc[intent_mask].sum())
        rate = cited_rows / n if n else np.nan
        rows.append({
            "intent_group": intent,
            feature: value,
            "n_rows": int(n),
            "cited_rows": cited_rows,
            "more_only_rows": more_only,
            "cited_rate": rate,
            "cited_rate_pct": rate * 100,
            "overall_cited_rate": overall,
            "diff_from_overall_pp": (rate - overall) * 100,
            "ci_low_pct": ci_low * 100,
            "ci_high_pct": ci_high * 100,
            "unique_urls": int(group["normalized_url"].nunique()),
            "unique_domains": int(group["source_root_domain"].nunique()),
            "unique_prompts": prompts,
            "row_share_within_intent": n / int(intent_mask.sum()),
            "cited_share_within_intent": cited_rows / cited_in_intent if cited_in_intent else np.nan,
            "regression_eligible": status == "eligible",
            "sparse_flag": "sparse_n_lt_20" if n < 20 else ("unstable_cited_or_more_only_lt_5" if cited_rows < 5 or more_only < 5 else ""),
            "status": status,
        })
    return pd.DataFrame(rows).sort_values(["intent_group", "n_rows"], ascending=[True, False], kind="stable").reset_index(drop=True)


def _orders(summary: pd.DataFrame, feature: str) -> tuple[list[str], list[str]]:
    intents = summary.groupby("intent_group")["n_rows"].sum().sort_values(ascending=False).index.tolist()
    values = summary.groupby(feature)["n_rows"].sum().sort_values(ascending=False).index.tolist()
    return intents, values


def _heatmap(summary: pd.DataFrame, feature: str, frequency: bool) -> go.Figure:
    intents, values = _orders(summary, feature)
    field = "n_rows" if frequency else "cited_rate_pct"
    matrix = summary.pivot(index="intent_group", columns=feature, values=field).reindex(index=intents, columns=values)
    labels = summary.pivot(index="intent_group", columns=feature, values="n_rows").reindex(index=intents, columns=values)
    custom = []
    for intent in intents:
        row = []
        for value in values:
            cell = summary[(summary["intent_group"] == intent) & (summary[feature] == value)]
            if cell.empty:
                row.append([0, 0, 0, np.nan, np.nan, 0, "no_rows"])
            else:
                item = cell.iloc[0]
                row.append([item.n_rows, item.cited_rows, item.more_only_rows, item.cited_rate_pct, item.diff_from_overall_pp, item.unique_prompts, item.status])
        custom.append(row)
    text = np.where(labels.fillna(0).to_numpy() > 0, np.vectorize(lambda rate, n: f"{rate:.1f}%<br>(n={int(n)})")(matrix.fillna(0).to_numpy() if not frequency else summary.pivot(index="intent_group", columns=feature, values="cited_rate_pct").reindex(index=intents, columns=values).fillna(0).to_numpy(), labels.fillna(0).to_numpy()), "")
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=values, y=intents, text=text, texttemplate="%{text}", textfont={"size": 11},
        colorscale="Viridis" if not frequency else "Cividis", colorbar={"title": "Cited rate (%)" if not frequency else "Rows"}, customdata=np.array(custom, dtype=object),
        hovertemplate=(
            "Intent: %{y}<br>Category: %{x}<br>Rows: %{customdata[0]:,}<br>Cited rows: %{customdata[1]:,}<br>"
            "More-only rows: %{customdata[2]:,}<br>Cited rate: %{customdata[3]:.1f}%<br>Difference from overall: %{customdata[4]:+.1f} pp<br>"
            "Unique prompts: %{customdata[5]:,}<br>Status: %{customdata[6]}<extra></extra>"
        ),
    ))
    title = f"Intent × {feature.replace('_real_estate', '').replace('_', ' ')} " + ("frequency" if frequency else "cited rate")
    apply_readable_plotly_layout(fig, title.title(), "read cited rate together with the supporting row count")
    fig.update_layout(
        width=max(1200, len(values) * 105 + 260),
        height=max(650, len(intents) * 72 + 160),
        margin={"l": 190, "r": 150, "t": 105, "b": 205},
    )
    fig.update_xaxes(
        title=feature.replace("_real_estate", "").replace("_", " ").title(),
        tickmode="array",
        tickvals=values,
        ticktext=[_heatmap_label(value).replace("\n", "<br>") for value in values],
        tickangle=-45,
        tickfont={"size": 11},
    )
    fig.update_yaxes(title="Intent")
    return fig


def _composition(summary: pd.DataFrame, feature: str, cited_only: bool) -> go.Figure:
    intents, values = _orders(summary, feature)
    share_column = "cited_share_within_intent" if cited_only else "row_share_within_intent"
    fig = go.Figure()
    for value in values:
        data = summary[summary[feature].eq(value)].set_index("intent_group").reindex(intents)
        label = str(value).replace("_", " ")
        fig.add_bar(name=label, x=intents, y=data[share_column].fillna(0), customdata=np.column_stack([data["n_rows"].fillna(0), data["cited_rows"].fillna(0)]), hovertemplate="Intent: %{x}<br>Category: " + label + "<br>Share: %{y:.1%}<br>Rows: %{customdata[0]:,}<br>Cited rows: %{customdata[1]:,}<extra></extra>")
    label = "Cited " if cited_only else "All-source "
    apply_readable_plotly_layout(fig, f"{label}{feature.replace('_real_estate', '').replace('_', ' ').title()} composition by intent", "composition is a surfaced/cited mix, not an adjusted effect")
    fig.update_layout(barmode="stack", legend_title="Category", width=1180, height=650, margin={"l": 80, "r": 300, "t": 90, "b": 140})
    fig.update_yaxes(title="Share within intent", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="Intent", tickangle=-20)
    return fig


def _forest_dropdown(summary: pd.DataFrame, feature: str) -> go.Figure:
    intents, _ = _orders(summary, feature)
    traces = []
    for index, intent in enumerate(intents):
        data = summary[summary["intent_group"].eq(intent)].sort_values("diff_from_overall_pp", kind="stable")
        labels = data[feature].map(_category_label)
        traces.append(go.Scatter(
            x=data["diff_from_overall_pp"], y=labels, mode="markers+text", visible=index == 0,
            text=[f"n={int(n)}" for n in data["n_rows"]], textposition="middle right",
            marker={"size": 10, "color": np.where(data["regression_eligible"], "#287a8e", "#94a3b8"), "symbol": np.where(data["regression_eligible"], "circle", "x")},
            error_x={"type": "data", "array": (data["ci_high_pct"] - data["cited_rate_pct"]).clip(lower=0), "arrayminus": (data["cited_rate_pct"] - data["ci_low_pct"]).clip(lower=0), "color": "#64748b"},
            customdata=np.column_stack([data["n_rows"], data["cited_rows"], data["cited_rate_pct"], data["status"]]),
            hovertemplate="Category: %{y}<br>Rows: %{customdata[0]:,}<br>Cited rows: %{customdata[1]:,}<br>Cited rate: %{customdata[2]:.1f}%<br>Difference: %{x:+.1f} pp<br>Status: %{customdata[3]}<extra></extra>",
            name=str(intent),
        ))
    fig = go.Figure(traces)
    buttons = [{"label": str(intent), "method": "update", "args": [{"visible": [i == index for i in range(len(intents))]}, {"title.text": f"Difference from overall cited rate by {feature.replace('_real_estate', '').replace('_', ' ').title()}<br><sup>Intent: {intent}; unadjusted descriptive comparison</sup>"}]} for index, intent in enumerate(intents)]
    fig.add_vline(x=0, line_dash="dash", line_color="#5d6670")
    apply_readable_plotly_layout(fig, f"Difference from overall cited rate by {feature.replace('_real_estate', '').replace('_', ' ').title()}", f"Intent: {intents[0]}; unadjusted descriptive comparison")
    fig.update_layout(updatemenus=[{"buttons": buttons, "direction": "down", "x": 1, "xanchor": "right", "y": 1.18, "yanchor": "top"}])
    fig.update_xaxes(title="Difference from overall cited rate (percentage points)", ticksuffix=" pp")
    fig.update_yaxes(title="Category", automargin=True)
    return fig


def _save_heatmap_preview(summary: pd.DataFrame, feature: str, frequency: bool, path: Path) -> None:
    intents, values = _orders(summary, feature)
    field = "n_rows" if frequency else "cited_rate_pct"
    matrix = summary.pivot(index="intent_group", columns=feature, values=field).reindex(index=intents, columns=values).fillna(0)
    n = summary.pivot(index="intent_group", columns=feature, values="n_rows").reindex(index=intents, columns=values).fillna(0)
    fig, ax = plt.subplots(figsize=(max(12, len(values) * 1.05), max(4, len(intents) * 0.75 + 2.4)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis" if not frequency else "cividis")
    ax.set(title=f"Intent × {feature.replace('_real_estate', '').replace('_', ' ').title()} " + ("frequency" if frequency else "cited rate"), xticks=range(len(values)), xticklabels=[_heatmap_label(value) for value in values], yticks=range(len(intents)), yticklabels=intents)
    ax.tick_params(axis="x", labelsize=8)
    for row, col in np.ndindex(matrix.shape):
        if n.iloc[row, col] > 0:
            label = f"{matrix.iloc[row, col]:.1f}%\nn={int(n.iloc[row, col])}" if not frequency else f"n={int(matrix.iloc[row, col])}"
            ax.text(col, row, label, ha="center", va="center", fontsize=8, color="white")
    fig.colorbar(image, ax=ax, label="Rows" if frequency else "Cited rate (%)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_composition_preview(summary: pd.DataFrame, feature: str, cited_only: bool, path: Path) -> None:
    intents, values = _orders(summary, feature)
    share_column = "cited_share_within_intent" if cited_only else "row_share_within_intent"
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(intents))
    colors = plt.cm.tab20(np.linspace(0, 1, len(values)))
    for value, color in zip(values, colors):
        data = summary[summary[feature].eq(value)].set_index("intent_group").reindex(intents)[share_column].fillna(0).to_numpy()
        ax.bar(intents, data, bottom=bottom, label=str(value).replace("_", " "), color=color)
        bottom += data
    ax.set(title=("Cited " if cited_only else "All-source ") + f"{feature.replace('_real_estate', '').replace('_', ' ').title()} composition by intent", ylabel="Share within intent", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_forest_preview(summary: pd.DataFrame, feature: str, intent: str, path: Path) -> None:
    data = summary[summary["intent_group"].eq(intent)].sort_values("diff_from_overall_pp", kind="stable")
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, max(4, 0.5 * len(data) + 1.5)))
    ax.errorbar(data["diff_from_overall_pp"], y, xerr=[(data["cited_rate_pct"] - data["ci_low_pct"]).clip(lower=0), (data["ci_high_pct"] - data["cited_rate_pct"]).clip(lower=0)], fmt="o", color="#287a8e", ecolor="#64748b", capsize=3)
    ax.axvline(0, color="#5d6670", linestyle="--")
    ax.set(title=f"Difference from overall cited rate by {feature.replace('_real_estate', '').replace('_', ' ').title()}\nIntent: {intent}", xlabel="Difference from overall cited rate (percentage points)", yticks=y, yticklabels=data[feature].map(_category_label))
    for yi, row in enumerate(data.itertuples(index=False)):
        ax.annotate(f"n={row.n_rows}", (row.diff_from_overall_pp, yi), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _prompt_balance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cited = _bool(df, "cited")
    rows = []
    for (intent, prompt), group in df.groupby(["intent_group", "prompt_id"], dropna=False):
        outcome = cited.loc[group.index]
        rows.append({"intent_group": intent, "prompt_id": prompt, "n_sources": len(group), "cited_rows": int(outcome.sum()), "cited_rate": float(outcome.mean()), "unique_source_types": int(_category(group, "source_type_real_estate").nunique()), "unique_page_types": int(_category(group, "page_type_family_real_estate").nunique())})
    audit = pd.DataFrame(rows)
    summary = audit.groupby("intent_group", as_index=False).agg(n_prompts=("prompt_id", "nunique"), avg_sources_per_prompt=("n_sources", "mean"), avg_cited_rate_per_prompt=("cited_rate", "mean"), min_sources_per_prompt=("n_sources", "min"), max_sources_per_prompt=("n_sources", "max"))
    return audit, summary.sort_values("n_prompts", ascending=False, kind="stable")


def run_intent_stratified_diagnostics_v6(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    audit, selected, df, selected_source = _intent_audit(lpm, eda)
    output_dir.mkdir(parents=True, exist_ok=True)
    interactive = figure_dir / "interactive"
    preview = figure_dir / "preview"
    interactive.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "intent_column_audit.csv", index=False)
    if selected is None:
        (output_dir / "intent_stratified_diagnostics_summary.md").write_text("# Intent-stratified diagnostics\n\nNo usable intent column was found. Intent-stratified plots were skipped without changing earlier diagnostics.\n", encoding="utf-8")
        return {"status": "skipped_no_intent", "output_dir": str(output_dir)}

    distribution = _intent_distribution(df)
    distribution.to_csv(output_dir / "intent_distribution.csv", index=False)
    summaries = {feature: _cell_summary(df, feature) for feature in TYPE_FEATURES}
    summaries["page_type_detail_real_estate"] = _cell_summary(df, "page_type_detail_real_estate")
    summaries["source_type_real_estate"].to_csv(output_dir / "intent_source_type_cell_summary.csv", index=False)
    summaries["page_type_family_real_estate"].to_csv(output_dir / "intent_page_type_family_cell_summary.csv", index=False)
    summaries["page_type_detail_real_estate"].to_csv(output_dir / "intent_page_type_detail_cell_summary.csv", index=False)

    manifest = []
    for feature, stem in (("source_type_real_estate", "source_type"), ("page_type_family_real_estate", "page_type_family")):
        summary = summaries[feature]
        for frequency in (False, True):
            suffix = "frequency" if frequency else "cited_rate"
            status = save_plotly_figure(_heatmap(summary, feature, frequency), interactive / f"heatmap_intent_by_{stem}_{suffix}.html", figure_dir / f"heatmap_intent_by_{stem}_{suffix}.png")
            _save_heatmap_preview(summary, feature, frequency, preview / f"heatmap_intent_by_{stem}_{suffix}.png")
            manifest.append({"plot_type": "heatmap", "feature": feature, "role": suffix, "png_export": status})
        for cited_only in (False, True):
            suffix = "cited_sources" if cited_only else "all_sources"
            status = save_plotly_figure(_composition(summary, feature, cited_only), interactive / f"stacked_{stem}_composition_by_intent_{suffix}.html", figure_dir / f"stacked_{stem}_composition_by_intent_{suffix}.png")
            _save_composition_preview(summary, feature, cited_only, preview / f"stacked_{stem}_composition_by_intent_{suffix}.png")
            manifest.append({"plot_type": "composition", "feature": feature, "role": suffix, "png_export": status})
        status = save_plotly_figure(_forest_dropdown(summary, feature), interactive / f"forest_{stem}_diff_by_intent_dropdown.html", figure_dir / f"forest_{stem}_diff_by_intent_dropdown.png")
        manifest.append({"plot_type": "forest_dropdown", "feature": feature, "role": "all_intents", "png_export": status})
        for intent in distribution["intent_group"]:
            _save_forest_preview(summary, feature, intent, preview / f"forest_{stem}_{_slug(intent)}.png")

    forest_data = pd.concat([summaries[feature].assign(feature_name=feature, category=summaries[feature][feature]) for feature in TYPE_FEATURES], ignore_index=True)
    forest_data[["intent_group", "feature_name", "category", "n_rows", "cited_rows", "cited_rate_pct", "diff_from_overall_pp", "ci_low_pct", "ci_high_pct", "status"]].to_csv(output_dir / "intent_stratified_forest_plot_data.csv", index=False)
    interactions = forest_data.copy()
    interactions["interaction_type"] = interactions["feature_name"].map({"source_type_real_estate": "intent_x_source_type", "page_type_family_real_estate": "intent_x_page_type_family"})
    interactions["recommendation"] = np.where(
        interactions["regression_eligible"] & interactions["diff_from_overall_pp"].abs().ge(10) & interactions["unique_domains"].ge(2),
        "candidate_for_lpm_interaction_sensitivity",
        np.where(interactions["status"].ne("eligible"), "descriptive_only_sparse", "not_interesting"),
    )
    interactions[["interaction_type", "intent_group", "category", "n_rows", "cited_rows", "more_only_rows", "cited_rate_pct", "diff_from_overall_pp", "unique_prompts", "unique_domains", "status", "recommendation"]].to_csv(output_dir / "intent_interaction_candidate_summary.csv", index=False)
    prompt_audit, prompt_summary = _prompt_balance(df)
    prompt_audit.to_csv(output_dir / "prompt_intent_balance_audit.csv", index=False)
    prompt_summary.to_csv(output_dir / "intent_prompt_balance_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_dir / "intent_graph_artifact_manifest.csv", index=False)

    strongest = {}
    for feature in TYPE_FEATURES:
        eligible = summaries[feature].query("regression_eligible").copy()
        strongest[feature] = eligible.loc[eligible["diff_from_overall_pp"].abs().nlargest(5).index] if not eligible.empty else pd.DataFrame()
    (output_dir / "intent_stratified_graph_guide.md").write_text(
        "# Intent-Stratified Graph Guide\n\n"
        "1. **Intent × source-type heatmap:** Within each question intent, which source types have higher cited rates?\n"
        "2. **Intent × page-type heatmap:** Within each intent, which page types have higher cited rates?\n"
        "3. **Source-type composition:** What source mix is surfaced for each intent?\n"
        "4. **Cited source-type composition:** What source mix is actually cited for each intent?\n"
        "5. **Forest plot by intent:** Which source/page categories differ most from the overall cited rate within an intent?\n"
        "6. **Prompt balance audit:** Are intent-level patterns based on enough prompts?\n\n"
        "All figures are descriptive and unadjusted. Read cited rates together with cell counts and prompt coverage.\n",
        encoding="utf-8",
    )
    top_intents = ", ".join(f"`{row.intent_group}` (n={int(row.n_rows)})" for row in distribution.head(5).itertuples(index=False))
    strong_lines = []
    for feature, table in strongest.items():
        if not table.empty:
            row = table.iloc[0]
            strong_lines.append(f"- `{feature}`: `{row.intent_group}` × `{row[feature]}` = {row.diff_from_overall_pp:+.1f} pp (n={int(row.n_rows)}, {row.status}).")
    candidates = int(interactions["recommendation"].eq("candidate_for_lpm_interaction_sensitivity").sum())
    (output_dir / "intent_stratified_diagnostics_summary.md").write_text(
        "# Intent-Stratified Diagnostics Summary\n\n"
        f"- Selected intent column: `{selected}` via `{selected_source}`.\n"
        f"- Intent groups: {len(distribution)}. Top intents: {top_intents}.\n"
        f"- Sparse intent cells are retained and labelled descriptive-only; they are not reliable interaction evidence.\n"
        + ("\n".join(strong_lines) if strong_lines else "- No regression-eligible intent × category cell was available for a strongest-pattern summary.")
        + f"\n- Interaction candidates for later sensitivity testing: {candidates}. Main LPM should remain simpler first.\n"
        "- Intent stratification does not change basic LPM readiness; it adds prechecks for future sensitivity interactions.\n\n"
        "**Status:** ready_for_LPM_v1_with_intent_stratified_prechecks\n",
        encoding="utf-8",
    )
    return {"status": "completed", "selected_intent_column": selected, "intent_source": selected_source, "intent_groups": int(len(distribution)), "rows": int(len(df)), "output_dir": str(output_dir), "figure_dir": str(figure_dir), "interactive_dir": str(interactive)}
