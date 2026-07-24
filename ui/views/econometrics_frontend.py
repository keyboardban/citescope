"""Feature-centered, read-only econometrics research interface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.econometrics_eda_v2.econometrics_frontend import FrontendArtifacts, load_frontend_artifacts
from ui import econometrics_charts as charts
from ui import components as C


SCOPE = "Results describe associations among sources already surfaced in this audit. They are not causal effects or web-wide citation probabilities."
UNOBSERVED = [
    "source authority", "publisher reputation", "official status", "backlink authority",
    "content freshness", "retrieval-stage relevance", "hidden ranking signals",
    "user-location context", "audit-time page content", "template quality not captured by domain",
    "non-surfaced candidate sources",
]


@st.cache_data(show_spinner="Loading validated econometrics frontend artifacts...")
def _load(root: str, manifest_mtime: float) -> FrontendArtifacts:
    del manifest_mtime
    return load_frontend_artifacts(root, verify_hashes=True)


def _frontend(bundle) -> FrontendArtifacts | None:
    root = bundle.package_dir / "tables/econometrics_frontend"
    manifest = root / "econometrics_frontend_manifest.json"
    if not manifest.exists():
        st.warning("The validated frontend artifact bundle has not been generated yet.")
        st.code(".venv/bin/python scripts/v2_build_econometrics_frontend_artifacts.py", language="bash")
        return None
    try:
        return _load(str(root), manifest.stat().st_mtime)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Frontend artifacts failed validation: {type(exc).__name__}: {exc}")
        st.caption("The interface stopped before displaying potentially stale or incompatible statistics.")
        return None


def _pp(value) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "Not available" if pd.isna(numeric) else f"{numeric:+.1f} pp"


def _status(value: object) -> str:
    return str(value or "not_available").replace("_", " ")


def _registry_row(bundle, feature: str) -> pd.Series | None:
    try:
        registry = pd.read_csv(bundle.package_dir / "tables/core_general_content_feature_dictionary.csv", low_memory=False)
    except FileNotFoundError:
        return None
    rows = registry[registry["feature_name"].astype(str).eq(feature)]
    if "registry_record_type" in rows:
        canonical = rows[rows["registry_record_type"].astype(str).eq("canonical")]
        rows = canonical if not canonical.empty else rows
    return None if rows.empty else rows.iloc[0]


def render_overview(bundle) -> None:
    artifacts = _frontend(bundle)
    if artifacts is None:
        return
    overview = artifacts.overview
    scorecard = artifacts.tables["core_general_feature_scorecard.csv"].copy()
    st.info(SCOPE)
    C.metric_cards([
        {"value": f"{overview['surfaced_source_rows']:,}", "label": "surfaced-source rows"},
        {"value": f"{overview['measurable_content_rows']:,}", "label": "measurable-content rows"},
        {"value": f"{overview['cited_rows']:,}", "label": "cited rows"},
        {"value": f"{overview['overall_cited_rate']:.1%}", "label": "overall cited rate"},
        {"value": f"{overview['prompts']:,}", "label": "prompts"},
        {"value": f"{overview['unique_urls']:,}", "label": "unique URLs"},
        {"value": f"{overview['domains']:,}", "label": "domains"},
        {"value": f"{overview['model_ready_features']:,}", "label": "model-ready features"},
        {"value": f"{overview['features_blocked_pending_qa']:,}", "label": "blocked pending QA"},
    ])
    st.caption(f"Artifact generated {overview['generated_at']} | Contract {overview['contract_version']} | Full audit = {overview['prompts']} prompts; measurable-content sample = {overview['measurable_prompts']} prompts.")

    st.markdown("#### Interactive feature scorecard")
    filters = st.columns([1.2, 1.2, 1.2, 1.2])
    group = filters[0].multiselect("Feature group", sorted(scorecard["feature_group"].unique()), default=sorted(scorecard["feature_group"].unique()), key="econ_overview_group")
    kind = filters[1].multiselect("Feature type", sorted(scorecard["feature_type"].unique()), default=sorted(scorecard["feature_type"].unique()), key="econ_overview_type")
    robustness = filters[2].multiselect("Robustness", sorted(scorecard["robustness_status"].unique()), default=sorted(scorecard["robustness_status"].unique()), key="econ_overview_robust")
    readiness = filters[3].selectbox("Model readiness", ["All", "Approved", "Blocked"], key="econ_overview_ready")
    direction = st.segmented_control("Adjusted direction", ["All", "Positive", "Negative", "Unavailable"], default="All", key="econ_overview_direction")
    visible = scorecard[scorecard["feature_group"].isin(group) & scorecard["feature_type"].isin(kind) & scorecard["robustness_status"].isin(robustness)].copy()
    if readiness == "Approved":
        visible = visible[visible["approved_for_model"]]
    elif readiness == "Blocked":
        visible = visible[~visible["approved_for_model"]]
    if direction == "Positive":
        visible = visible[pd.to_numeric(visible["g2_estimate_pp"], errors="coerce").gt(0)]
    elif direction == "Negative":
        visible = visible[pd.to_numeric(visible["g2_estimate_pp"], errors="coerce").lt(0)]
    elif direction == "Unavailable":
        visible = visible[visible["g2_estimate_pp"].isna()]
    display = visible[["human_label", "feature_name", "feature_group", "feature_type", "measurement_status", "support_rows", "raw_association_pp", "g1_estimate_pp", "g2_estimate_pp", "domain_fe_estimate_pp", "g2_ci_lower_pp", "g2_ci_upper_pp", "robustness_status", "approved_for_model", "primary_warning"]].rename(columns={
        "human_label": "Feature", "feature_name": "Technical name", "feature_group": "Group", "feature_type": "Type", "measurement_status": "Measurement", "support_rows": "Rows", "raw_association_pp": "Raw difference (pp)", "g1_estimate_pp": "G1 (pp)", "g2_estimate_pp": "G2 (pp)", "domain_fe_estimate_pp": "Domain FE (pp)", "g2_ci_lower_pp": "G2 CI low", "g2_ci_upper_pp": "G2 CI high", "robustness_status": "Robustness", "approved_for_model": "Approved", "primary_warning": "Primary warning",
    })
    st.caption(f"{len(display):,} of {len(scorecard):,} supported features remain after filters.")
    st.dataframe(display, width="stretch", hide_index=True, column_config={
        "Raw difference (pp)": st.column_config.NumberColumn(format="%+.1f"), "G1 (pp)": st.column_config.NumberColumn(format="%+.1f"), "G2 (pp)": st.column_config.NumberColumn(format="%+.1f"), "Domain FE (pp)": st.column_config.NumberColumn(format="%+.1f"), "G2 CI low": st.column_config.NumberColumn(format="%+.1f"), "G2 CI high": st.column_config.NumberColumn(format="%+.1f"),
    })
    st.download_button("Download filtered scorecard", display.to_csv(index=False).encode("utf-8"), "econometrics_feature_scorecard_filtered.csv", "text/csv", key="econ_scorecard_download")

    comparisons = artifacts.tables["feature_model_comparisons.parquet"].copy()
    if not comparisons.empty:
        comparisons = comparisons[comparisons["feature_name"].isin(visible["feature_name"])].copy()
        comparisons["primary_label"] = comparisons["diagnostic_labels"].fillna("not_available").str.split(";").str[-1]
        matrix_rows = (
            comparisons.sort_values("estimate_change_pp", key=lambda values: values.abs(), ascending=False)
            .drop_duplicates(["feature_name", "baseline_model_id", "comparison_model_id"])
        )
        st.markdown("#### Cross-model comparison matrix")
        st.caption("Each cell is the coefficient change for the largest displayed term in that feature-transition. It is a robustness map, not a p-value ranking.")
        st.plotly_chart(charts.model_transition_matrix(matrix_rows, artifacts.manifest["contract_version"]), width="stretch")


def _feature_header(bundle, score: pd.Series) -> None:
    registry = _registry_row(bundle, str(score["feature_name"]))
    definition = str(registry.get("definition", "Definition is not available in the registry.")) if registry is not None else "Definition is not available in the registry."
    provenance = str(registry.get("source_provenance", "precomputed measured-row dataset")) if registry is not None else "precomputed measured-row dataset"
    extraction = str(registry.get("extraction_requirement", "existing extracted content")) if registry is not None else "existing extracted content"
    missing = str(registry.get("missing_value_meaning", "Not measured or unavailable.")) if registry is not None else "Not measured or unavailable."
    st.markdown(f"### {score['human_label']}")
    st.caption(f"Technical name: `{score['feature_name']}`")
    st.write(definition)
    columns = st.columns(4)
    columns[0].metric("Layer", _status(score["feature_layer"]))
    columns[1].metric("Type", _status(score["feature_type"]))
    columns[2].metric("Measured rows", f"{int(score['support_rows']):,}")
    columns[3].metric("Model approval", "Approved" if bool(score["approved_for_model"]) else "Pending QA")
    with st.expander("Measurement definition and registry status"):
        st.dataframe(pd.DataFrame([
            {"Field": "Feature group", "Value": score["feature_group"]},
            {"Field": "Source provenance", "Value": provenance},
            {"Field": "Extraction requirement", "Value": extraction},
            {"Field": "Measurement status", "Value": score["measurement_status"]},
            {"Field": "QA status", "Value": score["qa_status"]},
            {"Field": "Missing value means", "Value": missing},
        ]), width="stretch", hide_index=True)
    warning = str(score.get("primary_warning") or "").strip()
    if warning and warning.casefold() != "nan":
        st.warning(warning)
    if not bool(score["approved_for_model"]):
        st.warning("This feature is not approved for Core-General model v1. Estimates are shown as existing evidence, not as a model-readiness endorsement.")


def _descriptive(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    rates = artifacts.tables["feature_cited_rate_summary.csv"]
    rates = rates[rates["feature_name"].eq(feature)].copy()
    C.section("Unadjusted cited rates", "Descriptive association only. Bins are not the regression specification.")
    if rates.empty or len(rates) < 2:
        st.info("A multi-level cited-rate comparison is unavailable because this feature has too little observed variation.")
        return
    support = rates[["feature_level", "n_rows", "n_cited", "n_more_only", "cited_rate", "ci_lower", "ci_upper", "n_prompts", "n_urls", "n_domains", "support_flag"]]
    left, right = st.columns([1.4, 1])
    left.plotly_chart(charts.cited_rate_plot(rates, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
    right.plotly_chart(charts.distribution_plot(rates, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
    with st.expander("Descriptive support table"):
        st.dataframe(support, width="stretch", hide_index=True, column_config={"cited_rate": st.column_config.NumberColumn(format="%.1%%"), "ci_lower": st.column_config.NumberColumn(format="%.1%%"), "ci_upper": st.column_config.NumberColumn(format="%.1%%")})


def _transition_card(row: pd.Series, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"#### {row['baseline_model_id']} to {row['comparison_model_id']} | `{row['comparison_source_model_id']}`")
        metrics = st.columns(5)
        metrics[0].metric("Baseline", f"{row['baseline_estimate_pp']:+.1f} pp")
        metrics[1].metric("Comparison", f"{row['comparison_estimate_pp']:+.1f} pp")
        metrics[2].metric("Change", f"{row['estimate_change_pp']:+.1f} pp")
        metrics[3].metric("CI width", f"{row['ci_width_change_pp']:+.1f} pp")
        metrics[4].metric("Rows change", f"{int(row['rows_change']):+,}")
        st.caption(
            f"Comparability: {_status(row['comparability_status'])} | Fixed effects changed: {'No' if row['same_fixed_effects'] else 'Yes'} | "
            f"Controls changed: {'No' if row['same_controls'] else 'Yes'} | Sample changed: {'No' if row['same_sample'] else 'Yes'} | "
            f"Functional form changed: {'No' if row['same_functional_form'] else 'Yes'} | "
            f"Rows: {int(row['baseline_n_rows']):,} to {int(row['comparison_n_rows']):,}"
        )
        st.write(row["explanation"])
        st.info("Diagnostics: " + _status(str(row["diagnostic_labels"]).replace(";", ", ")))
        if str(row["comparability_status"]) != "directly_comparable":
            st.warning(str(row["comparability_warning"]))


def _selected_term_narrative(
    estimates: pd.DataFrame,
    comparisons: pd.DataFrame,
    selected_term: str,
) -> str:
    aligned = estimates[
        estimates["is_preferred_covariance"].fillna(False)
        & estimates["model_status"].eq("available")
        & estimates["term_label"].astype(str).eq(selected_term)
    ]

    def model_estimate(model_id: str) -> str:
        rows = aligned[aligned["model_id"].eq(model_id)]
        return "unavailable" if rows.empty else _pp(rows.iloc[0]["estimate_pp"])

    raw = model_estimate("G0")
    opening = (
        f"For the selected contrast `{selected_term}`, the aligned raw contrast is {raw}, "
        f"the G1 within-prompt estimate is {model_estimate('G1')}, and the G2 joint estimate is {model_estimate('G2')}."
    )
    if raw == "unavailable":
        opening += " No directly aligned G0 contrast is available for this regression unit."

    meaningful = comparisons[comparisons["comparability_status"].ne("not_directly_comparable")]
    if meaningful.empty:
        transition = "No predefined compatible transition is available for this selected contrast."
    else:
        largest = meaningful.loc[meaningful["estimate_change_pp"].abs().idxmax()]
        transition = (
            f" The largest displayed compatible change for this contrast is "
            f"{largest['baseline_model_id']} to {largest['comparison_model_id']} "
            f"({_pp(largest['estimate_change_pp'])})."
        )
    return opening + transition + " This is a robustness diagnostic among surfaced sources, not a causal pathway."


def _models(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    estimates = artifacts.tables["feature_model_estimates_harmonized.parquet"]
    estimates = estimates[estimates["feature_name"].eq(feature)].copy()
    comparisons = artifacts.tables["feature_model_comparisons.parquet"]
    comparisons = comparisons[comparisons["feature_name"].eq(feature)].copy()
    summary_table = artifacts.tables["feature_model_comparison_summary.parquet"]
    summary = summary_table[summary_table["feature_name"].eq(feature)].iloc[0]
    C.section("Association across model specifications", "G1 and G2 are the headline models. Every other available estimate is a predefined robustness, sensitivity, diagnostic, or cross-check result.")
    if estimates.empty:
        st.info("No validated adjusted estimate exists for this feature.")
        return
    regression_terms = estimates[estimates["model_id"].ne("G0")]["term_label"].dropna().astype(str).drop_duplicates().tolist()
    selected_term = st.selectbox("Feature contrast", regression_terms, key=f"econ_model_term_{feature}") if regression_terms else ""
    preferred = estimates[
        estimates["is_preferred_covariance"].fillna(False)
        & estimates["model_status"].eq("available")
        & estimates["model_id"].ne("G0")
        & estimates["term_label"].astype(str).eq(selected_term)
    ].copy()
    preferred["comparability_status"] = "not_yet_compared"
    for index, model_row in preferred.iterrows():
        match = comparisons[
            comparisons["comparison_model_id"].eq(model_row["model_id"])
            & comparisons["comparison_source_model_id"].eq(model_row["source_model_id"])
            & comparisons["term_label"].astype(str).eq(selected_term)
        ]
        if not match.empty:
            preferred.loc[index, "comparability_status"] = match.iloc[0]["comparability_status"]
        elif model_row["model_id"] in {"G1", "G2"}:
            preferred.loc[index, "comparability_status"] = "headline_model"
    if preferred.empty:
        st.info("No finite precomputed estimate is available for this contrast.")
    else:
        st.plotly_chart(charts.cross_model_path(preferred, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
        st.caption(f"{score['human_label']} | Harmonized percentage-point units | Artifact {artifacts.manifest['contract_version']}")
    st.caption(f"Available aliases: {summary['available_model_aliases'] or 'none'}. Missing coefficient aliases: {summary['missing_model_aliases'] or 'none'}. Missing models are not synthesized.")

    selected_comparisons = comparisons[comparisons["term_label"].astype(str).eq(selected_term)].copy()
    st.markdown("#### What changed between models?")
    if selected_comparisons.empty:
        st.info("No predefined pairwise transition is available for this contrast.")
    else:
        transition_labels = {
            f"{row.baseline_model_id} to {row.comparison_model_id} | {row.comparison_source_model_id}": index
            for index, row in selected_comparisons.iterrows()
        }
        chosen = st.selectbox("Model transition", list(transition_labels), key=f"econ_transition_{feature}_{selected_term}")
        _transition_card(selected_comparisons.loc[transition_labels[chosen]], f"{feature}_{selected_term}")

        meaningful = selected_comparisons[selected_comparisons["comparability_status"].ne("not_directly_comparable")]
        largest = meaningful.loc[meaningful["estimate_change_pp"].abs().idxmax()] if not meaningful.empty else selected_comparisons.loc[selected_comparisons["ci_width_change_pp"].abs().idxmax()]
        st.markdown("#### Largest sensitivity")
        C.metric_cards([
            {"value": f"{largest['baseline_model_id']} to {largest['comparison_model_id']}", "label": "largest transition"},
            {"value": _pp(largest["estimate_change_pp"]), "label": "estimate change"},
            {"value": f"{largest['ci_width_change_pp']:+.1f} pp", "label": "CI width change"},
            {"value": f"{int(largest['rows_change']):+,}", "label": "rows changed"},
        ])
        st.write(largest["explanation"])

        st.markdown("#### Specification versus sample change")
        change_table = selected_comparisons[[
            "baseline_model_id", "comparison_model_id", "comparison_source_model_id", "same_controls",
            "same_fixed_effects", "same_sample", "same_functional_form", "rows_change_percent",
            "prompts_changed", "urls_changed", "domains_changed", "comparability_status",
        ]].copy()
        change_table["rows_change_percent"] = change_table["rows_change_percent"] * 100
        st.dataframe(change_table, width="stretch", hide_index=True, column_config={"rows_change_percent": st.column_config.NumberColumn("Rows change (%)", format="%+.1f")})

    covariance = artifacts.tables["feature_covariance_comparisons.parquet"]
    covariance = covariance[(covariance["feature_name"].eq(feature)) & covariance["term_label"].astype(str).eq(selected_term)].copy()
    st.markdown("#### Point estimate versus uncertainty")
    if covariance.empty:
        st.info("No multi-estimator covariance comparison is available for this feature contrast.")
    else:
        covariance["model_key"] = covariance["model_id"] + " | " + covariance["source_model_id"]
        model_key = st.selectbox("Specification for covariance comparison", covariance["model_key"].drop_duplicates().tolist(), key=f"econ_covariance_{feature}_{selected_term}")
        cov_selected = covariance[covariance["model_key"].eq(model_key)]
        st.plotly_chart(charts.covariance_forest(cov_selected, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
        inference_counts = cov_selected["inference_status"].value_counts()
        st.caption(f"Point estimates are unchanged when only covariance changes. Inference-sensitive comparisons: {int(inference_counts.get('inference_sensitive', 0))}. Clustering changes uncertainty assumptions; it does not add regression controls or remove confounding.")

    intents = artifacts.tables["feature_intent_interaction_contrasts.parquet"]
    intents = intents[(intents["feature_name"].eq(feature)) & intents["interaction_supported"]].copy()
    if not intents.empty:
        st.markdown("#### Supported intent interactions")
        se_methods = intents["se_method"].drop_duplicates().tolist()
        intent_se = st.selectbox("Intent-model covariance", se_methods, key=f"econ_intent_se_{feature}")
        intent_selected = intents[intents["se_method"].eq(intent_se)]
        st.plotly_chart(charts.intent_slope_forest(intent_selected, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
        st.warning("These are subgroup-specific slopes from a formal interaction model. Pairwise between-intent contrasts are unavailable because their covariance was not exported; different zero-inclusion patterns do not establish heterogeneity.")

    st.markdown("#### Diagnostic interpretation")
    st.write(_selected_term_narrative(estimates, selected_comparisons, selected_term))
    st.warning(summary["interpretation_boundary"])

    contrasts = artifacts.tables["feature_probability_contrasts.csv"]
    contrasts = contrasts[contrasts["feature_name"].eq(feature)]
    if not contrasts.empty:
        st.markdown("#### Model-implied probability contrast")
        contrast_label = st.selectbox("Contrast", contrasts["contrast_name"].tolist(), key=f"econ_contrast_{feature}")
        contrast = contrasts[contrasts["contrast_name"].eq(contrast_label)]
        st.plotly_chart(charts.probability_contrast(contrast, score["human_label"], artifacts.manifest["contract_version"]), width="stretch")
        st.caption("This is an observed-covariate model contrast, not expected gain, content improvement, or causal impact.")

    with st.expander("Cross-model technical details"):
        technical = preferred[[
            "model_id", "source_model_id", "model_role", "term_label", "estimate_pp", "ci_lower_pp",
            "ci_upper_pp", "p_value", "interpretation_unit", "reference_group", "n_rows", "n_prompts",
            "n_urls", "n_domains", "prompt_clusters", "url_clusters", "se_method", "fixed_effects",
            "controls", "sample_restriction", "functional_form", "taxonomy_version", "model_warning", "formula",
        ]]
        st.dataframe(technical, width="stretch", hide_index=True)
        st.download_button("Download cross-model comparisons", selected_comparisons.to_csv(index=False).encode("utf-8"), f"{feature}_{selected_term}_model_comparisons.csv", "text/csv", key=f"econ_download_comparison_{feature}_{selected_term}")


def _subgroups(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    data = artifacts.tables["feature_subgroup_statistics.csv"]
    data = data[data["feature_name"].eq(feature)].copy()
    C.section("Subgroups", "Descriptive subgroup comparison unless a formal interaction model is explicitly displayed.")
    if data.empty:
        st.info("No subgroup artifact is available for this feature.")
        return
    dimension = st.selectbox("Subgroup dimension", data["subgroup_dimension"].drop_duplicates().tolist(), key=f"econ_subgroup_{feature}")
    show_low = st.toggle("Show low-support cells", value=False, key=f"econ_subgroup_low_{feature}")
    selected = data[data["subgroup_dimension"].eq(dimension)]
    if not show_low:
        selected = selected[selected["support_flag"].eq("supported")]
    if selected.empty:
        st.info("No subgroup cells meet the support rule under the active selection.")
        return
    st.plotly_chart(charts.subgroup_heatmap(selected, dimension, artifacts.manifest["contract_version"]), width="stretch")
    st.caption("Differences between subgroup estimates are not formal interaction tests. One significant subgroup and one non-significant subgroup do not establish heterogeneity.")
    with st.expander("Subgroup support table"):
        st.dataframe(selected, width="stretch", hide_index=True)


def _example_card(row: pd.Series, index: int) -> None:
    label = "Cited" if int(row["cited"]) == 1 else "Not cited"
    st.markdown(f"**{row.get('title') or row['source_root_domain']}**")
    st.caption(f"{label} | {row['source_root_domain']} | prompt `{row['prompt_id']}` | {row['example_group'].replace('_', ' ')}")
    st.write(str(row.get("relevant_excerpt") or "No compact excerpt available."))
    fields = pd.DataFrame([
        {"Context": "Feature value", "Value": row["feature_value"]}, {"Context": "Intent", "Value": row["intent"]},
        {"Context": "Page type", "Value": row["page_type"]}, {"Context": "Page family", "Value": row["page_type_family"]},
        {"Context": "Source type", "Value": row["source_type"]}, {"Context": "Content strength", "Value": row["content_strength"]},
        {"Context": "Extraction scope", "Value": row["extraction_scope"]}, {"Context": "Language", "Value": row["language"]},
    ])
    st.dataframe(fields, width="stretch", hide_index=True)
    st.link_button("Open public page", row["normalized_url"], key=f"econ_example_open_{row['feature_name']}_{index}")


def _examples(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    data = artifacts.tables["feature_example_pages.csv"]
    data = data[data["feature_name"].eq(feature)].copy()
    C.section("Website examples", "Compact, precomputed evidence. The excerpt explains measurement, not citation.")
    if data.empty:
        st.info("No example artifact is available for this feature.")
        return
    controls = st.columns(4)
    group = controls[0].selectbox("Evidence group", ["All"] + sorted(data["example_group"].unique()), key=f"econ_example_group_{feature}")
    intent = controls[1].selectbox("Intent", ["All"] + sorted(data["intent"].dropna().astype(str).unique()), key=f"econ_example_intent_{feature}")
    page_type = controls[2].selectbox("Page type", ["All"] + sorted(data["page_type"].dropna().astype(str).unique()), key=f"econ_example_page_{feature}")
    source_type = controls[3].selectbox("Source type", ["All"] + sorted(data["source_type"].dropna().astype(str).unique()), key=f"econ_example_source_{feature}")
    filtered = data
    for column, value in (("example_group", group), ("intent", intent), ("page_type", page_type), ("source_type", source_type)):
        if value != "All":
            filtered = filtered[filtered[column].astype(str).eq(value)]
    st.caption(f"{len(filtered):,} compact examples remain. These filters do not change the precomputed model estimates.")
    page_size = 4
    pages = max(1, math.ceil(len(filtered) / page_size))
    page = st.number_input("Example page", min_value=1, max_value=pages, value=1, step=1, key=f"econ_example_page_number_{feature}")
    for index, (_, row) in enumerate(filtered.iloc[(page - 1) * page_size: page * page_size].iterrows(), start=(page - 1) * page_size):
        with st.container(border=True):
            _example_card(row, index)


def _pairs(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    data = artifacts.tables["feature_comparable_pairs.csv"]
    data = data[data["feature_name"].eq(feature)].copy()
    C.section("Compare similar pages", "Deterministic offline pairs use same prompt first, then displayed page, source, intent, extraction, length, relevance, and domain signals.")
    if data.empty:
        st.info("No cited-versus-not-cited pair with differing feature values met the deterministic pairing rules.")
        return
    quality = st.selectbox("Minimum match view", ["All", "strong", "moderate", "weak"], key=f"econ_pair_quality_{feature}")
    shown = data if quality == "All" else data[data["match_quality"].eq(quality)]
    if shown.empty:
        st.info("No comparison pair matches that quality filter.")
        return
    labels = {f"{row.pair_id} | {row.match_quality} | distance {row.distance_score:.2f}": row.pair_id for row in shown.itertuples()}
    selected_label = st.selectbox("Comparison pair", list(labels), key=f"econ_pair_{feature}")
    pair = shown[shown["pair_id"].eq(labels[selected_label])].iloc[0]
    st.info("These pages are observationally similar on the displayed variables.")
    comparison_table = pd.DataFrame([
        {"Variable": "Selected feature", "Cited page": pair["cited_feature_value"], "Not-cited page": pair["uncited_feature_value"], "Difference": "observed"},
        {"Variable": "Page family", "Cited page": pair["cited_page_type_family"], "Not-cited page": pair["uncited_page_type_family"], "Difference": "same" if pair["cited_page_type_family"] == pair["uncited_page_type_family"] else "different"},
        {"Variable": "Source type", "Cited page": pair["cited_source_type"], "Not-cited page": pair["uncited_source_type"], "Difference": "same" if pair["cited_source_type"] == pair["uncited_source_type"] else "different"},
        {"Variable": "Domain", "Cited page": pair["cited_domain"], "Not-cited page": pair["uncited_domain"], "Difference": "same" if pair["cited_domain"] == pair["uncited_domain"] else "different"},
        {"Variable": "Content strength", "Cited page": pair["cited_content_strength"], "Not-cited page": pair["uncited_content_strength"], "Difference": "same" if pair["cited_content_strength"] == pair["uncited_content_strength"] else "different"},
        {"Variable": "Page length", "Cited page": pair["cited_word_count"], "Not-cited page": pair["uncited_word_count"], "Difference": float(pair["cited_word_count"]) - float(pair["uncited_word_count"])},
        {"Variable": "Prompt-page relevance", "Cited page": pair["cited_relevance"], "Not-cited page": pair["uncited_relevance"], "Difference": float(pair["cited_relevance"]) - float(pair["uncited_relevance"])},
    ])
    comparison_table = comparison_table.map(lambda value: "" if pd.isna(value) else str(value))
    st.dataframe(comparison_table, width="stretch", hide_index=True)
    st.plotly_chart(charts.comparable_difference(pair, artifacts.manifest["contract_version"]), width="stretch")
    link_columns = st.columns(2)
    link_columns[0].link_button("Open cited page", pair["cited_url"], width="stretch")
    link_columns[1].link_button("Open not-cited page", pair["uncited_url"], width="stretch")
    st.markdown("#### Observed differences that may help explain the citation contrast")
    st.write(f"Exact match fields: {pair['exact_match_fields'] or 'none beyond prompt'}. Unresolved displayed differences: {pair['unmatched_differences'] or 'none listed'}. Match quality is {pair['match_quality']} with distance {pair['distance_score']:.2f}.")
    st.warning("Unobserved authority, freshness, trust, retrieval signals, and other omitted variables may remain. The observed variables do not fully determine citation status.")


def _diagnostics(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    C.section("Related features and diagnostic risks", "Association, precision, sensitivity, and omitted-variable diagnostics are kept separate.")
    related = artifacts.tables["feature_related_associations.csv"]
    related = related[related["feature_name"].eq(feature)].copy()
    if not related.empty:
        st.plotly_chart(charts.related_association_chart(related, artifacts.manifest["contract_version"]), width="stretch")
    mult = artifacts.tables["feature_multicollinearity_diagnostics.csv"]
    mult = mult[mult["feature_name"].eq(feature)]
    if not mult.empty:
        row = mult.iloc[0]
        st.markdown("#### Multicollinearity")
        C.metric_cards([
            {"value": _status(row["risk_classification"]), "label": "classification"},
            {"value": "Not assessable" if pd.isna(row["vif"]) else f"{row['vif']:.2f}", "label": "VIF"},
            {"value": "Not available" if pd.isna(row["condition_number"]) else f"{row['condition_number']:.1f}", "label": "condition number"},
            {"value": _pp(row["coefficient_change_pp"]), "label": "G1 to G2 change"},
        ])
        st.caption(row["explanation"])
    confounding = artifacts.tables["feature_confounding_diagnostics.csv"]
    confounding = confounding[confounding["feature_name"].eq(feature)]
    st.markdown("#### Confounding and sensitivity checks")
    st.dataframe(confounding[["risk_dimension", "comparison_model", "baseline_estimate_pp", "comparison_estimate_pp", "absolute_change_pp", "relative_change", "sign_flip", "sample_change", "over_control_risk", "classification", "explanation"]], width="stretch", hide_index=True)
    st.caption("A changed estimate may reflect adjustment, sample change, precision, or overlap. It does not prove that a single confounder explains the relationship.")
    st.markdown("#### Important unobserved or incompletely observed variables")
    st.write(", ".join(UNOBSERVED) + ".")
    st.warning("The frontend cannot detect every omitted variable and does not solve omitted-variable bias.")

    quality = artifacts.tables["feature_evidence_quality.csv"]
    quality = quality[quality["feature_name"].eq(feature)]
    st.markdown("#### Evidence quality by dimension")
    st.dataframe(quality[["dimension", "status", "supporting_statistic", "explanation", "limitation"]], width="stretch", hide_index=True)


def _sample_and_technical(artifacts: FrontendArtifacts, score: pd.Series) -> None:
    feature = score["feature_name"]
    sample = artifacts.tables["feature_sample_audit.csv"]
    sample = sample[sample["feature_name"].eq(feature)]
    C.section("Sample and measurement", "Availability is a missingness and selection issue, not random noise.")
    st.plotly_chart(charts.sample_flow(sample, artifacts.manifest["contract_version"]), width="stretch")
    st.dataframe(sample, width="stretch", hide_index=True)
    with st.expander("Technical model appendix and downloads"):
        estimates = artifacts.tables["feature_model_estimates.csv"]
        estimates = estimates[estimates["feature_name"].eq(feature)]
        st.dataframe(estimates, width="stretch", hide_index=True)
        st.download_button("Download selected feature model estimates", estimates.to_csv(index=False).encode("utf-8"), f"{feature}_model_estimates.csv", "text/csv", key=f"econ_download_models_{feature}")
        st.json({
            "artifact_contract": artifacts.manifest["contract_version"],
            "generated_at": artifacts.manifest["generated_at"],
            "dataset_version": score["dataset_version"],
            "feature_registry_version": score["feature_registry_version"],
            "models_fit_in_streamlit": False,
            "raw_brightdata_loaded_in_streamlit": False,
        })


def render_feature_explorer(bundle) -> None:
    artifacts = _frontend(bundle)
    if artifacts is None:
        return
    scorecard = artifacts.tables["core_general_feature_scorecard.csv"].copy()
    st.info(SCOPE)
    label_to_name = dict(scorecard.sort_values("human_label")[["human_label", "feature_name"]].itertuples(index=False, name=None))
    selected_label = st.selectbox("Feature", list(label_to_name), key="econ_explorer_feature")
    feature = label_to_name[selected_label]
    score = scorecard[scorecard["feature_name"].eq(feature)].iloc[0]
    _feature_header(bundle, score)
    st.info(str(score["interpretation_summary"]))
    _descriptive(artifacts, score)
    _models(artifacts, score)
    _subgroups(artifacts, score)
    _examples(artifacts, score)
    _pairs(artifacts, score)
    _diagnostics(artifacts, score)
    _sample_and_technical(artifacts, score)
