"""Read-only frontend for the separate position-focused econometric analysis."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "outputs/position_model_v1"
PLACEMENT_FEATURES = {
    "Direct-answer placement": "direct_answer_placement",
    "Table placement": "table_placement",
    "Question-heading placement": "question_heading_placement",
    "Numeric-evidence total density": "numeric_evidence_total_density",
    "Numeric-evidence early share (position extension)": "numeric_evidence_early_share",
    "External-source placement (exploratory)": "external_source_placement",
}
RATIO_COLUMNS = {
    "direct_answer_placement": "direct_answer_position_ratio",
    "table_placement": "first_table_position_ratio",
    "question_heading_placement": "first_question_heading_position_ratio",
    "external_source_placement": "external_source_position_ratio",
}
STATUS_COLUMNS = {
    "direct_answer_placement": "direct_answer_extraction_status",
    "table_placement": "table_extraction_status",
    "question_heading_placement": "question_heading_extraction_status",
    "numeric_evidence_total_density": "numeric_evidence_extraction_status",
    "numeric_evidence_early_share": "numeric_evidence_extraction_status",
    "external_source_placement": "external_source_extraction_status",
}
COLORS = {"Cited": "#167D3E", "Not cited": "#C43D32"}


@st.cache_data(show_spinner=False)
def _load(root_text: str, artifact_version: int) -> dict[str, Any]:
    del artifact_version  # Included in the cache key so regenerated outputs reload.
    root = Path(root_text)
    required = {
        "dataset": "position_model_dataset.parquet",
        "coverage": "position_model_feature_coverage.csv",
        "page_type_mapping": "position_model_page_type_6_mapping.csv",
        "source_type_mapping": "position_model_source_type_6_mapping.csv",
        "source_domain_audit": "position_model_source_type_domain_audit.csv",
        "domain": "position_model_domain_concentration.csv",
        "prompt": "position_model_prompt_concentration.csv",
        "within": "position_model_within_domain_variation.csv",
        "multicollinearity": "position_model_multicollinearity.csv",
        "ci": "position_model_ci_diagnostics.csv",
        "influence": "position_model_influence_diagnostics.csv",
        "results": "position_model_results_long.csv",
        "robustness": "position_model_robustness_results.csv",
        "flow": "position_model_sample_flow.csv",
        "audit": "position_model_feature_audit.csv",
        "manual_qa": "position_model_manual_validation_examples.csv",
        "clusters": "position_model_cluster_support.csv",
        "predicted": "position_model_predicted_probability_diagnostics.csv",
    }
    missing = [filename for filename in required.values() if not (root / filename).exists()]
    if missing:
        raise FileNotFoundError(f"Missing position model artifacts: {', '.join(missing)}")
    bundle: dict[str, Any] = {
        key: pd.read_parquet(root / filename) if filename.endswith(".parquet") else pd.read_csv(root / filename, low_memory=False)
        for key, filename in required.items()
    }
    bundle["manifest"] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    bundle["findings"] = (root / "POSITION_MODEL_FINDINGS.txt").read_text(encoding="utf-8")
    return bundle


def _metric_row(rows: pd.DataFrame) -> None:
    columns = st.columns(4)
    columns[0].metric("Rows", f"{len(rows):,}")
    columns[1].metric("Cited", f"{int(pd.to_numeric(rows['cited'], errors='coerce').sum()):,}")
    columns[2].metric("Domains", f"{rows['source_root_domain'].nunique():,}")
    columns[3].metric("Prompts", f"{rows['prompt_id'].nunique():,}")


def _clean_term(term: str) -> str:
    match = re.search(r"\[T\.([^\]]+)\]", str(term))
    if match:
        return match.group(1).replace("_", " ").title()
    if "numeric_evidence_total_density" in str(term):
        return "Total numeric-evidence density (1 SD)"
    return str(term)


def _association_label(term: str) -> str:
    text = str(term)
    label = _clean_term(text)
    if "page_type_model_6" in text:
        return f"Page: {label}"
    if "source_type_model_6" in text:
        return f"Source: {label}"
    if "direct_answer_placement" in text:
        return f"Direct answer: {label.replace('Direct Answer ', '')}"
    if "table_placement" in text:
        return f"Table: {label.replace('Table ', '')}"
    if "question_heading_placement" in text:
        return f"Question heading: {label.replace('Question Heading ', '')}"
    if text == "z_numeric_evidence_total_density":
        return "Numeric evidence density (z)"
    if text == "log_word_count":
        return "Log word count"
    return label


def _filtered_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    st.markdown("**EDA Filters**")
    left, right = st.columns(2)
    feature_label = left.selectbox("Feature", list(PLACEMENT_FEATURES), key="position_filter_feature")
    feature = PLACEMENT_FEATURES[feature_label]
    categories = ["All"]
    if feature in rows and feature not in {
        "numeric_evidence_total_density", "numeric_evidence_early_share"
    }:
        categories += sorted(rows[feature].dropna().astype(str).unique().tolist())
    category = right.selectbox("Placement category", categories, key="position_filter_category")

    left, right = st.columns(2)
    citation = left.selectbox("Citation status", ["All", "Cited", "Not cited"])
    quality = right.selectbox(
        "Extraction-quality status",
        ["All"] + sorted(rows.get(STATUS_COLUMNS[feature], pd.Series(dtype=str)).dropna().astype(str).unique().tolist()),
    )
    selectors = []
    for label, column in (
        ("Page type", "page_type"), ("Source type", "source_type"),
        ("Intent", "intent"), ("Domain", "source_root_domain"), ("Prompt", "prompt_id"),
    ):
        options = ["All"] + sorted(rows[column].dropna().astype(str).unique().tolist())
        selectors.append((column, st.selectbox(label, options)))
    minimum = st.number_input("Minimum pages per domain", min_value=1, value=1, step=1)

    filtered = rows.copy()
    if category != "All":
        filtered = filtered[filtered[feature].astype(str).eq(category)]
    if citation != "All":
        filtered = filtered[pd.to_numeric(filtered["cited"], errors="coerce").eq(1 if citation == "Cited" else 0)]
    if quality != "All":
        filtered = filtered[filtered[STATUS_COLUMNS[feature]].astype(str).eq(quality)]
    for column, value in selectors:
        if value != "All":
            filtered = filtered[filtered[column].astype(str).eq(value)]
    counts = filtered.groupby("source_root_domain")["normalized_url"].nunique()
    filtered = filtered[filtered["source_root_domain"].isin(counts[counts.ge(minimum)].index)]
    st.caption(
        f"EDA filters retain {len(filtered):,} rows, {filtered.normalized_url.nunique():,} URLs, "
        f"{filtered.source_root_domain.nunique():,} domains, and {filtered.prompt_id.nunique():,} prompts. "
        "Model Results remain the frozen unfiltered estimation output."
    )
    return filtered, feature, feature_label


def _overview(bundle: dict[str, Any]) -> None:
    manifest = bundle["manifest"]
    st.subheader("Position Model — New")
    st.info(
        "Among surfaced webpages with measurable content, this separate analysis estimates adjusted "
        "associations between feature placement and AI citation probability. It is observational and not causal."
    )
    columns = st.columns(5)
    columns[0].metric("All rows", f"{manifest['rows']:,}")
    columns[1].metric("M5 rows", f"{manifest['M5_rows']:,}")
    columns[2].metric("Citation rate", f"{manifest['citation_rate']:.1%}")
    columns[3].metric("M5 domains", f"{manifest['M5_domains']:,}")
    columns[4].metric("M5 prompts", f"{manifest['M5_prompts']:,}")
    st.markdown("**Primary features**")
    st.markdown(
        "Direct answer: no feature / first detected block in first half / second half.  "
        "Table: no verified main-content table / first table in first half / second half.  "
        "Question heading: first valid H2 or H3 question in first half / second half.  "
        "Numeric evidence: validated numeric-evidence blocks per 1,000 total main-content tokens, standardized.  "
        "Early share is a separate position extension and is undefined when a page contains no numeric evidence."
    )
    st.code(
        "M5: cited ~ direct-answer placement + table placement + question-heading placement + "
        "z total numeric-evidence density + log word count + 6-class page type + "
        "6-class source type + prompt FE"
    )
    with st.expander("Six-class taxonomy controls", expanded=False):
        st.caption(
            "Detailed Gemini labels remain available for QA. The model uses deterministic "
            "six-class controls; source type is stabilized to one modal class per domain "
            "using unique URLs, with exact ties assigned to other_or_unknown."
        )
        st.markdown("**Page-type mapping**")
        st.dataframe(
            bundle["page_type_mapping"][[
                "page_type_detailed", "page_type_model_6", "n_rows", "unique_urls",
                "citation_rate",
            ]], width="stretch", hide_index=True,
        )
        st.markdown("**Source-type mapping before domain consensus**")
        st.dataframe(
            bundle["source_type_mapping"][[
                "source_type_detailed", "source_type_row_collapsed", "n_rows",
                "unique_urls", "citation_rate",
            ]], width="stretch", hide_index=True,
        )
        st.caption(
            f"Low-confidence source domains (<60% URL agreement): "
            f"{manifest['source_type_low_confidence_domains']}. "
            f"Exact-tie domains: {manifest['source_type_tied_domains']}."
        )
    st.caption(f"Main inference: `{manifest['primary_se_method']}`. {manifest['primary_se_reason']}")
    st.warning(
        "External-source placement is excluded from M5/M6: "
        + manifest["external_source_eligibility_reason"]
    )
    primary = bundle["results"]
    primary = primary[
        primary["model_id"].eq("M5") & primary["is_primary_inference"].astype(bool)
        & primary["term"].str.contains("placement|numeric_evidence", case=False, regex=True)
    ].copy()
    primary["Coefficient"] = primary["term"].map(_clean_term)
    st.dataframe(
        primary[["Coefficient", "estimate_pp", "ci_lower_pp", "ci_upper_pp", "p_value", "n_obs"]]
        .rename(columns={
            "estimate_pp": "Estimate (pp)", "ci_lower_pp": "CI low (pp)",
            "ci_upper_pp": "CI high (pp)", "p_value": "p-value", "n_obs": "Rows",
        }),
        width="stretch", hide_index=True,
    )


def _categorical_eda(rows: pd.DataFrame, feature: str) -> None:
    eligible = rows[rows[feature].notna()].copy()
    if eligible.empty:
        st.warning("No measured rows remain under these filters.")
        return
    eligible["Outcome"] = np.where(pd.to_numeric(eligible["cited"], errors="coerce").eq(1), "Cited", "Not cited")
    summary = eligible.groupby(feature, observed=True).agg(
        n_rows=("cited", "size"), cited_rows=("cited", "sum"),
        n_domains=("source_root_domain", "nunique"), n_prompts=("prompt_id", "nunique"),
    ).reset_index()
    summary["share"] = summary["n_rows"] / len(eligible)
    summary["citation_rate"] = summary["cited_rows"] / summary["n_rows"]
    summary["non_cited_rows"] = summary["n_rows"] - summary["cited_rows"]
    intervals = [
        _wilson(int(row.cited_rows), int(row.n_rows)) for row in summary.itertuples()
    ]
    summary["ci_lower"] = [item[0] for item in intervals]
    summary["ci_upper"] = [item[1] for item in intervals]
    _metric_row(eligible)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(summary, x=feature, y="n_rows", text="n_rows", title="Category counts"), width="stretch")
    right.plotly_chart(px.bar(summary, x=feature, y="share", text=summary["share"].map(lambda x: f"{x:.1%}"), title="Category share"), width="stretch")
    fig = go.Figure(go.Scatter(
        x=summary[feature], y=summary["citation_rate"], mode="markers+lines+text",
        text=summary["n_rows"].map(lambda value: f"n={value:,}"), textposition="top center",
        error_y={"type": "data", "symmetric": False,
                 "array": summary["ci_upper"] - summary["citation_rate"],
                 "arrayminus": summary["citation_rate"] - summary["ci_lower"]},
    ))
    fig.update_layout(title="Citation rate by placement with Wilson 95% CI", yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    counts = eligible.groupby([feature, "Outcome"], observed=True).size().rename("Rows").reset_index()
    st.plotly_chart(px.bar(counts, x=feature, y="Rows", color="Outcome", barmode="group", color_discrete_map=COLORS, title="Cited versus not-cited counts"), width="stretch")
    for dimension, label in (("page_type", "Page type"), ("source_type", "Source type"), ("prompt_id", "Prompt")):
        composition = eligible.groupby([feature, dimension], observed=True).size().rename("Rows").reset_index()
        if dimension == "prompt_id":
            composition = composition.sort_values("Rows", ascending=False).groupby(feature).head(12)
        fig = px.bar(composition, x=feature, y="Rows", color=dimension, title=f"{label} composition", barmode="stack")
        fig.update_layout(legend_title_text=label)
        st.plotly_chart(fig, width="stretch")
    ratio = RATIO_COLUMNS.get(feature)
    if ratio and ratio in eligible:
        st.plotly_chart(px.histogram(eligible.dropna(subset=[ratio]), x=ratio, color="Outcome", nbins=30, marginal="box", color_discrete_map=COLORS, title="Feature-position ratio distribution"), width="stretch")


def _wilson(cited: int, n: int) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = cited / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return center - half, center + half


def _numeric_eda(rows: pd.DataFrame, feature: str) -> None:
    data = rows.dropna(subset=[feature]).copy()
    if data.empty:
        st.warning("No measured numeric-density rows remain under these filters.")
        return
    data["Outcome"] = np.where(pd.to_numeric(data["cited"], errors="coerce").eq(1), "Cited", "Not cited")
    _metric_row(data)
    title = (
        "Total numeric-evidence density per 1,000 main-content tokens"
        if feature == "numeric_evidence_total_density"
        else "Share of numeric evidence appearing in the first half"
    )
    st.plotly_chart(px.histogram(data, x=feature, color="Outcome", nbins=50, marginal="box", color_discrete_map=COLORS, title=title), width="stretch")
    percentiles = data[feature].quantile([0, .25, .5, .75, .9, .95, .99, 1]).rename("Value").reset_index()
    percentiles.columns = ["Percentile", "Value"]
    st.dataframe(percentiles, width="stretch", hide_index=True)
    left, right = st.columns(2)
    left.plotly_chart(px.box(data, x="Outcome", y=feature, color="Outcome", color_discrete_map=COLORS, title="Density by citation status", points=False), width="stretch")
    top_types = data["page_type"].value_counts().head(10).index
    right.plotly_chart(px.box(data[data.page_type.isin(top_types)], x="page_type", y=feature, title="Density by page type", points=False), width="stretch")
    detail_columns = list(dict.fromkeys([
        "normalized_url", "source_root_domain", "page_type", "source_type", feature,
        "numeric_evidence_total_count", "numeric_evidence_early_count",
        "total_main_content_tokens", "numeric_evidence_total_density",
        "numeric_evidence_early_share", "cited",
    ]))
    outliers = data.sort_values(feature, ascending=False).head(50)[detail_columns]
    st.markdown(
        "**Highest-density observations**"
        if feature == "numeric_evidence_total_density"
        else "**Highest early-share observations**"
    )
    st.dataframe(outliers, width="stretch", hide_index=True)


def _feature_eda(bundle: dict[str, Any], filtered: pd.DataFrame, feature: str) -> None:
    st.caption("Every chart below uses the displayed filtered EDA sample and reports its support above the chart group.")
    if feature in {"numeric_evidence_total_density", "numeric_evidence_early_share"}:
        _numeric_eda(filtered, feature)
    else:
        _categorical_eda(filtered, feature)


def _balance(bundle: dict[str, Any], feature: str) -> None:
    if feature in {"numeric_evidence_total_density", "numeric_evidence_early_share"}:
        key_feature = (
            "z_numeric_evidence_total_density"
            if feature == "numeric_evidence_total_density"
            else "numeric_evidence_early_share"
        )
        categories = ["top_10_percent"]
    else:
        key_feature = feature
        categories = sorted(bundle["domain"].loc[bundle["domain"].feature.eq(feature), "category"].unique())
    category = st.selectbox("Feature placement category", categories, key="balance_category")
    domain = bundle["domain"][(bundle["domain"].feature.eq(key_feature)) & (bundle["domain"].category.eq(category))]
    prompt = bundle["prompt"][(bundle["prompt"].feature.eq(key_feature)) & (bundle["prompt"].category.eq(category))]
    left, right = st.columns(2)
    left.dataframe(domain, width="stretch", hide_index=True)
    right.dataframe(prompt, width="stretch", hide_index=True)
    metrics = pd.concat([domain, prompt], ignore_index=True)
    if not metrics.empty:
        st.plotly_chart(px.bar(metrics, x="dimension", y="effective_groups", color="dimension", title="Effective number of domains and prompts", text_auto=".1f"), width="stretch")
    st.download_button(
        "Download domain concentration table", bundle["domain"].to_csv(index=False),
        file_name="position_model_domain_concentration.csv", mime="text/csv",
    )
    st.markdown("**Within-domain variation and fixed-effect readiness**")
    st.dataframe(bundle["within"], width="stretch", hide_index=True)
    st.markdown("**Cluster-size distribution**")
    rows = bundle["dataset"]
    cluster_sizes = pd.concat([
        rows.groupby("source_root_domain").size().rename("size").reset_index().assign(dimension="Domain"),
        rows.groupby("prompt_id").size().rename("size").reset_index().assign(dimension="Prompt"),
    ], ignore_index=True)
    st.plotly_chart(px.histogram(cluster_sizes, x="size", color="dimension", barmode="overlay", log_y=True, title="Cluster-size distribution"), width="stretch")


def _model_results(bundle: dict[str, Any]) -> None:
    results = bundle["results"].copy()
    model = st.segmented_control("Model", ["M0", "M1", "M2", "M3", "M4", "M5"] , default="M5")
    method = st.selectbox("Standard-error method", sorted(results["se_method"].dropna().unique()), index=sorted(results["se_method"].dropna().unique()).index(bundle["manifest"]["primary_se_method"]))
    selected = results[results.model_id.eq(model) & results.se_method.eq(method)].copy()
    focal = selected[selected.term.str.contains("placement|numeric_evidence", case=False, regex=True)].copy()
    if focal.empty:
        st.info("M0 contains controls only and has no position-feature coefficient.")
    else:
        focal["Coefficient"] = focal.term.map(_clean_term)
        fig = go.Figure(go.Scatter(
            x=focal["estimate_pp"], y=focal["Coefficient"], mode="markers",
            error_x={"type": "data", "symmetric": False,
                     "array": focal["ci_upper_pp"] - focal["estimate_pp"],
                     "arrayminus": focal["estimate_pp"] - focal["ci_lower_pp"]},
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#555")
        fig.update_layout(title=f"{model} coefficient forest plot ({method})", xaxis_title="Adjusted association (percentage points)")
        st.plotly_chart(fig, width="stretch")
    st.dataframe(selected[[
        "term", "estimate_pp", "standard_error", "ci_lower_pp", "ci_upper_pp", "p_value",
        "bh_q_value", "n_obs", "n_cited", "n_domains", "n_prompts", "fixed_effects",
    ]], width="stretch", hide_index=True)
    comparison = results[results.is_primary_inference.astype(bool) & results.term.map(lambda term: "placement" in str(term) or "numeric_evidence" in str(term))].copy()
    comparison["Coefficient"] = comparison.term.map(_clean_term)
    st.plotly_chart(px.line(comparison, x="model_id", y="estimate_pp", color="Coefficient", markers=True, title="Coefficient comparison across M1–M5"), width="stretch")
    st.caption(
        "Reference groups: no direct answer, no table, no question heading, "
        "blog/guide/editorial page type, and official company/brand source type. "
        "Estimates are percentage-point differences conditional on included controls and prompt fixed effects."
    )


def _ci_diagnostics(bundle: dict[str, Any]) -> None:
    data = bundle["ci"].copy()
    data["label"] = data["model_id"] + " · " + data["term"].map(_clean_term)
    label = st.selectbox("Coefficient", data["label"].tolist())
    row = data[data.label.eq(label)].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Estimate", f"{row.estimate_pp:.2f} pp")
    columns[1].metric("95% CI", f"{row.ci_lower_pp:.2f} to {row.ci_upper_pp:.2f} pp")
    columns[2].metric("CI width", f"{row.ci_width_pp:.2f} pp")
    columns[3].metric("p-value", f"{row.p_value:.4g}")
    st.info(row.grounded_ci_explanation)
    diagnostics = pd.DataFrame({
        "Metric": ["Category rows", "Cited events", "Domains", "Prompts", "Max domain share", "Max prompt share", "VIF", "LODO min", "LODO max", "LODO sign changes"],
        "Value": [row.get("category_sample_size"), row.get("category_cited_count"), row.get("contributing_domains"), row.get("contributing_prompts"), row.get("maximum_domain_share"), row.get("maximum_prompt_share"), row.get("vif"), row.get("leave_one_domain_out_min_pp"), row.get("leave_one_domain_out_max_pp"), row.get("leave_one_domain_out_sign_changes")],
    })
    st.dataframe(diagnostics, width="stretch", hide_index=True)
    se = pd.DataFrame({
        "Method": ["HC3", "Domain cluster", "Prompt cluster", "Two-way cluster"],
        "Standard error": [row.get("hc3_standard_error"), row.get("domain_clustered_standard_error"), row.get("prompt_clustered_standard_error"), row.get("two_way_clustered_standard_error")],
    })
    st.plotly_chart(px.bar(se, x="Method", y="Standard error", title="Standard-error comparison", text_auto=".3f"), width="stretch")
    influence = bundle["influence"]
    influence = influence[influence.term.eq(row.term) & influence.influence_dimension.eq("source_root_domain")]
    if not influence.empty:
        st.plotly_chart(px.scatter(influence, x="removed_group", y="estimate_pp", hover_data=["removed_rows", "change_pp", "sign_changed"], title="Leave-one-domain-out stability"), width="stretch")


def _multicollinearity(bundle: dict[str, Any]) -> None:
    diagnostics = bundle["multicollinearity"]
    vif = diagnostics[diagnostics.row_type.eq("vif")].copy()
    st.metric("Condition number", f"{vif.condition_number.dropna().iloc[0]:.2f}" if len(vif) else "NA")
    st.dataframe(vif.sort_values("vif", ascending=False), width="stretch", hide_index=True)
    pairs = diagnostics[
        diagnostics.row_type.isin(
            ["pairwise_predictor_association", "pairwise_dummy_association"]
        )
    ].copy()
    variables = sorted(set(pairs.variable.dropna()) | set(pairs.related_variable.dropna()))
    if variables:
        matrix = pd.DataFrame(np.eye(len(variables)), index=variables, columns=variables)
        for row in pairs.itertuples():
            matrix.loc[row.variable, row.related_variable] = row.association
            matrix.loc[row.related_variable, row.variable] = row.association
        labels = {_variable: _association_label(_variable) for _variable in variables}
        matrix = matrix.rename(index=labels, columns=labels)
        figure = px.imshow(
            matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
            aspect="auto",
            title="Full M5 predictor association heatmap",
        )
        figure.update_layout(
            height=max(680, 36 * len(variables)),
            margin=dict(l=220, r=30, t=80, b=220),
            xaxis_tickangle=-45,
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Every non-reference M5 predictor is shown. Page/source reference classes "
            "are omitted by regression coding, not by an association threshold."
        )
    st.markdown("**Highly associated pairs and warnings**")
    flagged_pairs = pairs[pairs.association.abs().ge(.30)].copy()
    st.dataframe(
        flagged_pairs.sort_values(
            "association", key=lambda values: values.abs(), ascending=False
        ),
        width="stretch",
        hide_index=True,
    )


def _robustness(bundle: dict[str, Any]) -> None:
    robustness = bundle["robustness"].copy()
    st.dataframe(robustness, width="stretch", hide_index=True)
    estimates = robustness[
        pd.to_numeric(robustness.get("estimate_pp"), errors="coerce").notna()
        & robustness.term.astype(str).str.contains("placement|position_ratio|numeric_evidence", case=False, regex=True)
    ].copy()
    if not estimates.empty:
        estimates["Coefficient"] = estimates.term.map(_clean_term)
        st.plotly_chart(px.scatter(estimates, x="estimate_pp", y="Coefficient", color="model_id", title="Robustness coefficient comparison", labels={"estimate_pp": "Estimate (percentage points)"}), width="stretch")
    st.markdown("**Predicted-probability diagnostic**")
    st.dataframe(bundle["predicted"], width="stretch", hide_index=True)
    st.caption("A small number of LPM fitted values outside [0, 1] is reported as a limitation; logit AMEs provide a functional-form cross-check.")


def _data_quality(bundle: dict[str, Any], rows: pd.DataFrame) -> None:
    st.dataframe(bundle["audit"], width="stretch", hide_index=True)
    status_columns = list(dict.fromkeys(STATUS_COLUMNS.values()))
    status = rows[status_columns].melt(var_name="Feature", value_name="Extraction status").value_counts().rename("Rows").reset_index()
    st.plotly_chart(px.bar(status, x="Feature", y="Rows", color="Extraction status", title="Extraction-status distribution"), width="stretch")
    missing = pd.DataFrame({
        "Feature": list(PLACEMENT_FEATURES.values()),
        "Missing rows": [rows[column].isna().sum() for column in PLACEMENT_FEATURES.values()],
        "Missing rate": [rows[column].isna().mean() for column in PLACEMENT_FEATURES.values()],
    })
    st.dataframe(missing, width="stretch", hide_index=True)
    st.markdown("**Sample flow and exclusions**")
    st.dataframe(bundle["flow"], width="stretch", hide_index=True)
    st.markdown("**Purposive manual validation examples**")
    st.dataframe(bundle["manual_qa"], width="stretch", hide_index=True)
    st.markdown("**Source-type domain consensus audit**")
    st.dataframe(bundle["source_domain_audit"], width="stretch", hide_index=True)
    st.warning(
        "Placement-specific formal precision/recall labels are not yet available. Gemini evidence was manually spot-checked, "
        "and the HTML table detector was previously audited, but these are not substitutes for a blinded labeled validation sample."
    )


def render(output_dir: str | Path | None = None) -> None:
    root = Path(output_dir or DEFAULT_ROOT).resolve()
    st.title("Position Model — New")
    st.caption(f"Separate read-only artifacts: `{root}`")
    try:
        manifest_path = root / "manifest.json"
        artifact_version = manifest_path.stat().st_mtime_ns if manifest_path.exists() else 0
        bundle = _load(str(root), artifact_version)
    except Exception as exc:
        st.error(f"Position-model outputs are unavailable: {type(exc).__name__}: {exc}")
        st.code(".venv/bin/python scripts/v2_run_position_model.py")
        return

    with st.expander("Filters", expanded=False):
        filtered, feature, _ = _filtered_rows(bundle["dataset"])

    tabs = st.tabs([
        "Overview", "Feature EDA", "Domain and Prompt Balance", "Model Results",
        "Confidence-Interval Diagnostics", "Multicollinearity", "Robustness", "Data Quality",
    ])
    with tabs[0]:
        _overview(bundle)
    with tabs[1]:
        _feature_eda(bundle, filtered, feature)
    with tabs[2]:
        _balance(bundle, feature)
    with tabs[3]:
        _model_results(bundle)
    with tabs[4]:
        _ci_diagnostics(bundle)
    with tabs[5]:
        _multicollinearity(bundle)
    with tabs[6]:
        _robustness(bundle)
    with tabs[7]:
        _data_quality(bundle, bundle["dataset"])
    with st.expander("Automated findings"):
        st.text(bundle["findings"])
