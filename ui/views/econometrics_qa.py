"""ChatGPT content-econometrics QA and page comparison interface."""

from __future__ import annotations

import html
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import econometrics_qa as qa
from src import storage
from src.econometrics_eda_v2 import feature_distribution_support as feature_support
from src.econometrics_eda_v2 import manual_feature_validation as feature_qa

from .. import components as C


@st.cache_data(show_spinner="Loading validated econometrics package...")
def _load_bundle(
    package_dir: str,
    prompt_manifest_path: str,
    gemini_taxonomy_path: str,
    gemini_taxonomy_mtime: float,
    taxonomy_version: str = qa.GENERAL_TAXONOMY_VERSION,
) -> qa.QABundle:
    del taxonomy_version, gemini_taxonomy_mtime
    return qa.load_bundle(
        package_dir,
        prompt_manifest_path or None,
        gemini_taxonomy_path or None,
    )


@st.cache_data(show_spinner=False)
def _load_snapshot(source_url: str, snapshot_root: str) -> tuple[dict | None, str]:
    snapshot, path = qa.load_snapshot(source_url, snapshot_root=snapshot_root)
    return snapshot, str(path or "")


@st.cache_data(show_spinner="Loading manual feature-validation evidence...")
def _load_feature_validation_artifacts(
    frontend_dir: str,
    manifest_mtime: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    del manifest_mtime
    return feature_qa.load_validated_artifacts(Path(frontend_dir))


@st.cache_data(show_spinner="Loading feature distribution and support evidence...")
def _load_feature_support_artifacts(
    manifest_path: str,
    manifest_mtime: float,
) -> tuple[dict[str, pd.DataFrame], dict]:
    del manifest_mtime
    return feature_support.load_support_artifacts(Path(manifest_path))


def _snapshot_root(bundle: qa.QABundle) -> Path:
    return bundle.package_dir.parent / qa.SNAPSHOT_RELATIVE_DIR


def _format_url_option(row: pd.Series) -> str:
    title = str(row.get("page_title") or "").strip()
    domain = str(row.get("source_root_domain") or "unknown")
    normalized = str(row.get("normalized_url") or "")
    label = title[:68] if title else normalized[:68]
    return f"{domain} | {label} | {normalized[-22:]}"


def _safe_text(value, limit: int = 30000) -> str:
    text = str(value or "").strip()
    return text[:limit] + ("\n\n[preview truncated]" if len(text) > limit else "")


def _int_value(value) -> int:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0 if pd.isna(numeric) else int(numeric)


def _float_value(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return 0.0 if pd.isna(numeric) else float(numeric)


def _label_value(row: pd.Series, column: str, fallback: str = "unknown") -> str:
    value = row.get(column)
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return fallback
    text = str(value).strip()
    return text if text and text.casefold() != "nan" else fallback


def _taxonomy_comparison(row: pd.Series, *, heading: str = "Classification check") -> None:
    """Show deterministic, LLM, and historical labels next to the live page."""
    records = [
        {
            "Layer": "Rule v2",
            "Page type": _label_value(row, "page_type_general_rule_v2"),
            "Page family": _label_value(row, "page_type_family_general_rule_v2"),
            "Source / site type": _label_value(row, "site_type_general_rule_v2"),
            "Confidence": _label_value(row, "page_type_general_confidence_rule_v2"),
            "Basis": _label_value(row, "page_type_general_source_rule_v2"),
        },
        {
            "Layer": "Gemini 3.1 Flash-Lite",
            "Page type": _label_value(row, "llm_page_type_general", "not available"),
            "Page family": _label_value(row, "llm_page_type_family_general", "not available"),
            "Source / site type": _label_value(row, "llm_site_type_general", "not available"),
            "Confidence": _label_value(row, "llm_confidence", "not available"),
            "Basis": _label_value(row, "llm_evidence_source", "not available"),
        },
        {
            "Layer": "Historical",
            "Page type": _label_value(row, "page_type_general"),
            "Page family": _label_value(row, "page_type_family_general"),
            "Source / site type": _label_value(
                row,
                "site_type_general",
                _label_value(row, "source_type_real_estate"),
            ),
            "Confidence": _label_value(row, "page_type_general_confidence"),
            "Basis": "stored model-reproducibility label",
        },
    ]
    st.markdown(f"#### {heading}")
    st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)

    details = st.columns(3)
    details[0].caption(
        "URL seed: " + _label_value(row, "page_type_url_seed_general_rule_v2")
    )
    details[1].caption(
        "Content: "
        + _label_value(row, "content_strength")
        + " / "
        + _label_value(row, "content_quality_flag")
    )
    details[2].caption(
        "Gemini input: " + _label_value(row, "classification_input_mode", "not available")
    )

    if _label_value(row, "llm_page_type_general", ""):
        agrees = row.get("llm_agrees_with_rule_v2")
        if pd.notna(agrees) and not bool(agrees):
            st.warning("Gemini and Rule v2 disagree on the detailed page type. Check the webpage before adopting either label.")
        flags = []
        if bool(row.get("llm_abstain")) if pd.notna(row.get("llm_abstain")) else False:
            flags.append("abstained")
        if bool(row.get("llm_needs_review")) if pd.notna(row.get("llm_needs_review")) else False:
            flags.append("needs review")
        if bool(row.get("family_repaired")) if pd.notna(row.get("family_repaired")) else False:
            flags.append("family derived from detailed type")
        evidence = _label_value(row, "llm_evidence", "")
        if evidence or flags:
            with st.expander("Gemini evidence and QA flags"):
                if evidence:
                    st.write(evidence)
                if flags:
                    st.caption("Flags: " + ", ".join(flags))


def _live_page_panel(source_url: str, key_prefix: str, height: int = 650) -> None:
    """Render an on-demand iframe with policy diagnostics and a durable fallback."""
    st.link_button("Open original page", source_url, width="stretch", key=f"{key_prefix}_open")
    actions = st.columns(2)
    policy_key = f"{key_prefix}_frame_policy"
    policy_url_key = f"{key_prefix}_frame_policy_url"
    live_url_key = f"{key_prefix}_live_url"
    if actions[0].button("Check iframe policy", width="stretch", key=f"{key_prefix}_check"):
        st.session_state[policy_key] = qa.inspect_live_frame_policy(source_url)
        st.session_state[policy_url_key] = source_url
    if actions[1].button(
        "Load live webpage",
        type="primary",
        width="stretch",
        key=f"{key_prefix}_load",
    ):
        st.session_state[live_url_key] = source_url

    policy = st.session_state.get(policy_key)
    if policy and st.session_state.get(policy_url_key) == source_url:
        message = f"{policy['status']}: {policy['reason']}"
        if policy["status"] == "blocked":
            st.warning(message)
        else:
            st.caption(message)
    if st.session_state.get(live_url_key) == source_url:
        st.caption(
            "If this panel is blank or reports that it refused to connect, the target site's "
            "X-Frame-Options or CSP policy blocks embedding. Use Open original page instead."
        )
        st.iframe(source_url, height=height)
    else:
        with st.container(height=height, border=True):
            st.markdown("Select **Load live webpage** to attempt an embedded comparison.")
            st.caption("Loading is manual and does not spend Bright Data credits.")


def render(package_dir: str = "", prompt_manifest_path: str = "") -> None:
    package = Path(package_dir) if package_dir else qa.default_package_dir()
    C.section(
        "ChatGPT Content Econometrics QA",
        "Inspect surfaced sources, compare crawler output with live pages, and review model evidence.",
    )
    if not package.exists():
        st.error("The external econometrics package is not configured or cannot be found.")
        recommended_root = qa.default_package_dir().parents[4]
        st.code(
            f"CITESCOPE_RESEARCH_DATA_DIR={recommended_root}",
            language="bash",
        )
        st.caption(f"Resolved package path: {package}")
        return

    try:
        gemini_path = qa.default_gemini_taxonomy_path(package)
        gemini_mtime = gemini_path.stat().st_mtime if gemini_path.exists() else 0.0
        try:
            bundle = _load_bundle(
                str(package),
                prompt_manifest_path,
                str(gemini_path) if gemini_path.exists() else "",
                gemini_mtime,
            )
        except PermissionError:
            # Sandboxed/headless sessions may not be allowed to read a Downloads manifest.
            # The package's prompt_reference.csv is the validated fallback.
            bundle = _load_bundle(
                str(package),
                "",
                str(gemini_path) if gemini_path.exists() else "",
                gemini_mtime,
            )
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load the validated econometrics package: {type(exc).__name__}: {exc}")
        return

    tabs = st.tabs(
        [
            "Overview",
            "Page comparison",
            "Prompt explorer",
            "Taxonomy",
            "Taxonomy analysis",
            "Econometrics",
            "Feature validation",
            "Reviews",
        ]
    )
    with tabs[0]:
        _overview(bundle)
    with tabs[1]:
        _page_comparison(bundle)
    with tabs[2]:
        _prompt_explorer(bundle)
    with tabs[3]:
        _taxonomy_explorer(bundle)
    with tabs[4]:
        _taxonomy_analysis(bundle)
    with tabs[5]:
        _econometrics(bundle)
    with tabs[6]:
        _feature_validation(bundle)
    with tabs[7]:
        _reviews()


def _overview(bundle: qa.QABundle) -> None:
    summary = qa.bundle_summary(bundle)
    reviews = storage.list_econometrics_reviews()
    C.metric_cards(
        [
            {"value": f"{summary['full_audit_prompts']:,}", "label": "full audit prompts"},
            {"value": f"{summary['surfaced_rows']:,}", "label": "surfaced rows"},
            {"value": f"{summary['unique_urls']:,}", "label": "unique URLs"},
            {"value": f"{summary['measurable_rows']:,}", "label": "measurable rows"},
            {"value": f"{summary['cited_rate']:.1%}", "label": "cited rate"},
            {"value": f"{len(reviews):,}", "label": "manually reviewed"},
        ]
    )
    st.info(
        "Full audit = 500 prompts; measurable-content LPM sample = 498 prompts. "
        "Content associations are conditional on a source being surfaced and having measurable content."
    )

    left, right = st.columns(2)
    strength = (
        bundle.url_evidence["content_strength"]
        .fillna("unknown")
        .value_counts()
        .rename_axis("content_strength")
        .reset_index(name="urls")
    )
    fig = px.bar(
        strength,
        x="content_strength",
        y="urls",
        color="content_strength",
        category_orders={"content_strength": ["strong", "medium", "weak", "failed", "unknown"]},
        color_discrete_map={
            "strong": "#198754",
            "medium": "#d39e00",
            "weak": "#dc6b35",
            "failed": "#b23a48",
            "unknown": "#77808f",
        },
        title="Unique URLs by extracted-content strength",
        text_auto=True,
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="URLs", height=390)
    left.plotly_chart(fig, width="stretch")

    availability = bundle.all_rows.copy()
    availability["citation_status"] = availability["cited"].map({1: "Cited", 0: "More-only"})
    availability["content_available"] = availability["content_feature_available"].fillna(False).astype(bool)
    availability = (
        availability.groupby("citation_status", observed=True)["content_available"]
        .agg(rows="size", available="sum")
        .reset_index()
    )
    availability["availability_rate"] = availability["available"] / availability["rows"]
    fig = px.bar(
        availability,
        x="citation_status",
        y="availability_rate",
        color="citation_status",
        color_discrete_map={"Cited": "#198754", "More-only": "#77808f"},
        title="Measurable content by observed citation status",
        text=availability["availability_rate"].map(lambda value: f"{value:.1%}"),
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Share", height=390)
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    right.plotly_chart(fig, width="stretch")

    with st.expander("Data contract and external paths"):
        preset = qa.previous_area_condo_preset()
        st.dataframe(
            pd.DataFrame(
                [
                    {"asset": "Econometrics package", "path": str(bundle.package_dir), "available": bundle.package_dir.exists()},
                    {"asset": "Full prompt manifest", "path": str(bundle.prompt_manifest_path or "using package prompt_reference.csv"), "available": bundle.prompt_manifest_path is not None},
                    {"asset": "Bright Data input", "path": str(preset.brightdata_input_path or "not configured"), "available": preset.brightdata_input_path is not None},
                    {"asset": "Raw Bright Data output", "path": str(preset.brightdata_output_path or "not configured"), "available": preset.brightdata_output_path is not None},
                    {"asset": "Crawler snapshots", "path": str(_snapshot_root(bundle)), "available": _snapshot_root(bundle).exists()},
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.json(
            {
                "study": bundle.manifest["study_name"],
                "full_audit_prompts": bundle.manifest["full_audit_prompt_count"],
                "measurable_content_prompts": bundle.manifest["measurable_content_prompt_count"],
                "canonical_notebooks": bundle.manifest["canonical_notebooks"],
            }
        )


def _filtered_url_evidence(bundle: qa.QABundle) -> pd.DataFrame:
    evidence = bundle.url_evidence.copy()
    controls = st.columns([2.2, 1, 1, 1])
    query = controls[0].text_input(
        "Find URL, title, or domain",
        placeholder="Search 2,881 URLs",
        key="qa_url_search",
    ).strip()
    strengths = ["All"] + sorted(evidence["content_strength"].dropna().astype(str).unique().tolist())
    strength = controls[1].selectbox("Content", strengths, key="qa_strength_filter")
    citation = controls[2].selectbox("Citation", ["All", "Cited", "More-only"], key="qa_cited_filter")
    scrape = controls[3].selectbox("Scrape", ["All", "Success", "Failed"], key="qa_scrape_filter")

    if query:
        mask = pd.Series(False, index=evidence.index)
        for column in ("normalized_url", "source_url", "source_root_domain", "page_title"):
            mask |= evidence[column].fillna("").astype(str).str.contains(query, case=False, regex=False)
        evidence = evidence[mask]
    if strength != "All":
        evidence = evidence[evidence["content_strength"].astype(str).eq(strength)]
    if citation == "Cited":
        evidence = evidence[pd.to_numeric(evidence["cited_appearances"], errors="coerce").fillna(0).gt(0)]
    elif citation == "More-only":
        evidence = evidence[pd.to_numeric(evidence["cited_appearances"], errors="coerce").fillna(0).eq(0)]
    if scrape == "Success":
        evidence = evidence[evidence["scrape_success"].fillna(False).astype(bool)]
    elif scrape == "Failed":
        evidence = evidence[~evidence["scrape_success"].fillna(False).astype(bool)]
    return evidence.reset_index(drop=True)


def _page_comparison(bundle: qa.QABundle) -> None:
    C.section(
        "Scraped versus live page",
        "The crawler snapshot is historical. The live page may have changed after the scrape.",
    )
    evidence = _filtered_url_evidence(bundle)
    if evidence.empty:
        st.warning("No URLs match these filters.")
        return
    labels = {_format_url_option(row): str(row["normalized_url"]) for _, row in evidence.iterrows()}
    selected_label = st.selectbox(
        "Page",
        list(labels),
        key="qa_selected_page",
        help=f"{len(evidence):,} URLs match the active filters.",
    )
    normalized_url = labels[selected_label]
    row = evidence[evidence["normalized_url"].astype(str).eq(normalized_url)].iloc[0]
    source_url = str(row.get("source_url") or normalized_url)
    snapshot, snapshot_path = _load_snapshot(source_url, str(_snapshot_root(bundle)))

    C.metric_cards(
        [
            {"value": str(row.get("content_strength") or "unknown"), "label": "content strength"},
            {"value": f"{_int_value(row.get('word_count')):,}", "label": "words"},
            {"value": f"{_int_value(row.get('heading_count')):,}", "label": "headings"},
            {"value": f"{_int_value(row.get('source_appearances')):,}", "label": "appearances"},
            {"value": f"{_float_value(row.get('cited_rate')):.1%}", "label": "cited rate"},
        ]
    )

    _taxonomy_comparison(row)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Stored crawler snapshot")
        st.caption(snapshot_path or "No normalized snapshot file was found.")
        if snapshot:
            st.caption(
                f"Fetched: {snapshot.get('fetched_at') or 'unknown'} | "
                f"Provider: {snapshot.get('provider_mode') or snapshot.get('provider') or 'unknown'}"
            )
            extracted, metadata, raw = st.tabs(["Extracted page", "Metadata", "Raw HTML"])
            with extracted:
                content = snapshot.get("text") or row.get("page_text_excerpt")
                with st.container(height=570, border=True):
                    st.markdown(
                        '<div style="white-space:pre-wrap;line-height:1.55">'
                        + html.escape(_safe_text(content))
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            with metadata:
                st.json(
                    {
                        "requested_url": snapshot.get("requested_url"),
                        "final_url": snapshot.get("final_url"),
                        "status_code": snapshot.get("status_code"),
                        "success": snapshot.get("success"),
                        "title": snapshot.get("title"),
                        "meta_description": snapshot.get("meta_description"),
                        "content_quality_flag": snapshot.get("content_quality_flag"),
                        "word_count": snapshot.get("word_count"),
                        "heading_count": snapshot.get("heading_count"),
                        "table_count": snapshot.get("table_count"),
                        "link_count": snapshot.get("link_count"),
                        "error": snapshot.get("error"),
                    }
                )
            with raw:
                st.code(_safe_text(snapshot.get("html"), 20000), language="html")
        else:
            st.warning("The page-level evidence exists, but its normalized crawler snapshot is unavailable.")

    with right:
        st.markdown("#### Current live webpage")
        _live_page_panel(source_url, "qa_compare", height=650)

    _review_form(row, snapshot)


def _review_form(row: pd.Series, snapshot: dict | None) -> None:
    normalized_url = str(row["normalized_url"])
    source_url = str(row.get("source_url") or normalized_url)
    existing = storage.get_econometrics_review(normalized_url) or {}
    st.divider()
    st.markdown("#### Manual review")
    statuses = ["unreviewed", "correct", "needs_rescrape", "wrong_page", "blocked", "stale", "dynamic_js"]
    completeness = ["unknown", "complete", "mostly_complete", "incomplete", "empty"]
    changed = ["unknown", "no", "yes"]
    with st.form(f"qa_review_{qa.snapshot_key(source_url)}"):
        columns = st.columns(3)
        current_status = existing.get("review_status", "unreviewed")
        status = columns[0].selectbox(
            "Review status",
            statuses,
            index=statuses.index(current_status) if current_status in statuses else 0,
        )
        current_completeness = existing.get("scrape_completeness", "unknown") or "unknown"
        completeness_value = columns[1].selectbox(
            "Scrape completeness",
            completeness,
            index=completeness.index(current_completeness) if current_completeness in completeness else 0,
        )
        current_changed = "unknown"
        if existing.get("live_page_changed") is True:
            current_changed = "yes"
        elif existing.get("live_page_changed") is False:
            current_changed = "no"
        changed_value = columns[2].selectbox("Live page changed", changed, index=changed.index(current_changed))
        taxonomy = st.text_input("Taxonomy suggestion", value=existing.get("taxonomy_suggestion", ""))
        notes = st.text_area("Review notes", value=existing.get("notes", ""), height=90)
        submitted = st.form_submit_button("Save review", type="primary")
    if submitted:
        storage.save_econometrics_review(
            {
                "normalized_url": normalized_url,
                "source_url": source_url,
                "snapshot_key": qa.snapshot_key(source_url) if snapshot else "",
                "review_status": status,
                "scrape_completeness": completeness_value,
                "live_page_changed": None if changed_value == "unknown" else changed_value == "yes",
                "taxonomy_suggestion": taxonomy,
                "notes": notes,
            }
        )
        st.success("Review saved separately from the model dataset.")


def _prompt_explorer(bundle: qa.QABundle) -> None:
    C.section("Prompt explorer", "Inspect all surfaced sources for one prompt.")
    query = st.text_input("Find prompt", placeholder="Search prompt text or ID", key="qa_prompt_search").strip()
    prompts = bundle.prompts.copy()
    if query:
        mask = prompts["prompt_id"].astype(str).str.contains(query, case=False, regex=False)
        mask |= prompts["prompt"].fillna("").astype(str).str.contains(query, case=False, regex=False)
        prompts = prompts[mask]
    if prompts.empty:
        st.warning("No prompts match this search.")
        return
    labels = {
        f"{row['prompt_id']} | {str(row.get('prompt') or '')[:95]}": str(row["prompt_id"])
        for _, row in prompts.iterrows()
    }
    prompt_id = labels[st.selectbox("Prompt", list(labels), key="qa_selected_prompt")]
    prompt = prompts[prompts["prompt_id"].astype(str).eq(prompt_id)].iloc[0]
    st.markdown(f"**{html.escape(str(prompt.get('prompt') or ''))}**")
    st.caption(
        f"Intent: {prompt.get('intent') or 'unknown'} | Area: {prompt.get('area_tag') or 'unknown'} | "
        f"Expansion: {prompt.get('expansion_group') or 'unknown'}"
    )
    rows = bundle.all_rows[bundle.all_rows["prompt_id"].astype(str).eq(prompt_id)].copy()
    evidence_columns = [
            "normalized_url",
            "page_title",
            "content_strength",
            "page_type_url_seed_general_rule_v2",
            "page_type_general_rule_v2",
            "page_type_family_general_rule_v2",
            "site_type_general_rule_v2",
            "page_type_general_confidence_rule_v2",
            "page_type_general_source_rule_v2",
            "page_type_general",
            "page_type_family_general",
            "site_type_general",
            "page_type_general_confidence",
            "source_type_real_estate",
            "content_quality_flag",
        ]
    evidence_columns += [
        column
        for column in (
            "llm_page_type_general",
            "llm_page_type_family_general",
            "llm_site_type_general",
            "llm_confidence",
            "llm_evidence_source",
            "llm_evidence",
            "llm_abstain",
            "llm_needs_review",
            "llm_agrees_with_rule_v2",
            "family_repaired",
            "classification_input_mode",
        )
        if column in bundle.url_evidence
    ]
    evidence = bundle.url_evidence[evidence_columns]
    rows = rows.merge(evidence, on="normalized_url", how="left", suffixes=("", "_url"))
    rows["citation_status"] = rows["cited"].map({1: "Cited", 0: "More-only"})
    C.metric_cards(
        [
            {"value": len(rows), "label": "surfaced rows"},
            {"value": int(rows["cited"].sum()), "label": "cited rows"},
            {"value": int(rows["normalized_url"].nunique()), "label": "unique URLs"},
            {"value": int(rows["content_feature_available"].fillna(False).sum()), "label": "measurable rows"},
        ]
    )
    display_columns = [
        "citation_status",
        "page_title",
        "source_root_domain",
        "source_url",
        "content_strength",
        "page_type_general_rule_v2",
        "page_type_family_general_rule_v2",
        "site_type_general_rule_v2",
    ]
    for column in ("llm_page_type_general", "llm_page_type_family_general", "llm_site_type_general", "llm_confidence"):
        if column in rows:
            display_columns.append(column)
    st.dataframe(
        rows[display_columns],
        width="stretch",
        hide_index=True,
        column_config={"source_url": st.column_config.LinkColumn("URL", display_text="Open")},
    )

    st.divider()
    st.markdown("#### Inspect a prompt source")
    source_rows = rows.drop_duplicates("normalized_url").copy()
    source_labels = {}
    for _, source in source_rows.iterrows():
        title = str(source.get("page_title") or source.get("normalized_url") or "")[:65]
        normalized = str(source.get("normalized_url") or "")
        label = (
            f"{source.get('citation_status', 'Unknown')} | "
            f"{source.get('source_root_domain', 'unknown')} | {title} | {normalized[-18:]}"
        )
        source_labels[label] = normalized
    selected_source_label = st.selectbox(
        "Source webpage",
        list(source_labels),
        key="qa_prompt_source_page",
    )
    selected_normalized = source_labels[selected_source_label]
    source = source_rows[source_rows["normalized_url"].astype(str).eq(selected_normalized)].iloc[0]
    source_url = str(source.get("source_url") or selected_normalized)
    st.caption(
        f"{source.get('citation_status', 'Unknown')} | "
        f"{source.get('source_root_domain', 'unknown')}"
    )
    _taxonomy_comparison(source, heading="Selected source classification")
    _live_page_panel(source_url, "qa_prompt_source", height=720)


def _taxonomy_explorer(bundle: qa.QABundle) -> None:
    C.section(
        "Taxonomy explorer",
        "Rule v2 is the current deterministic preview. Historical labels remain available for model reproducibility.",
    )
    feature_labels = {
        "page_type_url_seed_general_rule_v2": "Rule v2: URL-seed page type",
        "page_type_general_rule_v2": "Rule v2: final detailed page type",
        "page_type_family_general_rule_v2": "Rule v2: final page family",
        "site_type_general_rule_v2": "Rule v2: site type",
        "page_type_url_seed_general": "Historical: URL-seed page type",
        "page_type_family_general": "Historical: final page family",
        "site_type_general": "Historical: site type",
        "source_type_real_estate": "Historical: real-estate source type",
        "page_type_family_real_estate": "Historical: real-estate page family",
    }
    if "llm_page_type_general" in bundle.url_evidence:
        feature_labels = {
            "llm_page_type_general": "Gemini: detailed page type",
            "llm_page_type_family_general": "Gemini: page family",
            "llm_site_type_general": "Gemini: source / site type",
            "llm_confidence": "Gemini: confidence",
            **feature_labels,
        }
    feature = st.selectbox(
        "Taxonomy level",
        list(feature_labels),
        key="qa_taxonomy_feature",
        format_func=feature_labels.get,
    )
    evidence = bundle.url_evidence.copy()
    evidence[feature] = evidence[feature].fillna("unknown").astype(str)
    distribution = (
        evidence.groupby(feature, dropna=False)
        .agg(
            unique_urls=("normalized_url", "size"),
            source_appearances=("source_appearances", "sum"),
            cited_appearances=("cited_appearances", "sum"),
        )
        .reset_index()
    )
    distribution["cited_rate"] = distribution["cited_appearances"] / distribution["source_appearances"]
    distribution = distribution.sort_values("unique_urls", ascending=True)
    fig = px.bar(
        distribution,
        x="unique_urls",
        y=feature,
        orientation="h",
        color="cited_rate",
        color_continuous_scale="Tealgrn",
        title=f"Unique URLs and cited rate by {feature}",
        text="unique_urls",
    )
    fig.update_layout(height=max(420, len(distribution) * 34), yaxis_title=None, xaxis_title="Unique URLs")
    fig.update_coloraxes(colorbar_tickformat=".0%", colorbar_title="Cited rate")
    st.plotly_chart(fig, width="stretch")
    selected = st.selectbox("Inspect category", distribution[feature].tolist(), key="qa_taxonomy_level")
    sample = evidence[evidence[feature].eq(selected)].sort_values(
        ["cited_appearances", "source_appearances"], ascending=False
    )
    st.dataframe(
        sample[
            [
                "page_title",
                "source_root_domain",
                "source_url",
                "content_strength",
                "word_count",
                "source_appearances",
                "cited_appearances",
                "cited_rate",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "source_url": st.column_config.LinkColumn("URL", display_text="Open"),
            "cited_rate": st.column_config.NumberColumn("Cited rate", format="percent"),
        },
    )

    st.divider()
    st.markdown("#### Inspect a category webpage")
    source_rows = sample.drop_duplicates("normalized_url").copy()
    source_labels = {
        _format_url_option(row): str(row["normalized_url"])
        for _, row in source_rows.iterrows()
    }
    selected_source_label = st.selectbox(
        "Taxonomy source webpage",
        list(source_labels),
        key="qa_taxonomy_source_page",
    )
    selected_normalized = source_labels[selected_source_label]
    source = source_rows[source_rows["normalized_url"].astype(str).eq(selected_normalized)].iloc[0]
    source_url = str(source.get("source_url") or selected_normalized)
    st.caption(
        f"{source.get('source_root_domain', 'unknown')} | "
        f"{_int_value(source.get('source_appearances')):,} appearances | "
        f"{_float_value(source.get('cited_rate')):.1%} cited rate"
    )
    _taxonomy_comparison(source, heading="Selected webpage classification")
    _live_page_panel(source_url, "qa_taxonomy_source", height=720)


def _taxonomy_analysis(bundle: qa.QABundle) -> None:
    C.section(
        "Aggregate taxonomy comparison",
        "Compare classifier agreement across URLs. Agreement is not accuracy until manual labels are available.",
    )
    evidence = bundle.url_evidence.copy()
    summary = qa.taxonomy_comparison_summary(evidence)
    if not summary:
        st.warning("The Gemini taxonomy artifact is not available for aggregate comparison.")
        return

    C.metric_cards(
        [
            {
                "value": f"{summary['known_exact_agreement_rate']:.1%}",
                "label": "known Rule-v2 exact agreement",
            },
            {"value": f"{summary['rule_unknown']:,}", "label": "Rule-v2 unknown"},
            {
                "value": f"{summary['rule_unknown_resolved']:,}",
                "label": "unknown resolved by Gemini",
            },
            {"value": f"{summary['llm_unknown']:,}", "label": "Gemini unknown"},
            {
                "value": f"{summary['high_medium_confidence_disagreements']:,}",
                "label": "high/medium disagreements",
            },
            {"value": f"{summary['metadata_only_urls']:,}", "label": "metadata-only URLs"},
        ]
    )
    st.info(
        "These figures describe agreement between two classification methods. Neither Rule v2 nor Gemini is "
        "treated as ground truth. Use the live-page review workflow to establish manual labels."
    )

    baseline_options = {
        "Rule v2": {
            "Detailed page type": "page_type_general_rule_v2",
            "Page family": "page_type_family_general_rule_v2",
            "Source / site type": "site_type_general_rule_v2",
        },
        "Historical": {
            "Detailed page type": "page_type_general",
            "Page family": "page_type_family_general",
            "Source / site type": "site_type_general",
        },
    }
    llm_columns = {
        "Detailed page type": "llm_page_type_general",
        "Page family": "llm_page_type_family_general",
        "Source / site type": "llm_site_type_general",
    }
    controls = st.columns([1, 1, 1.2])
    baseline_name = controls[0].selectbox(
        "Baseline taxonomy",
        list(baseline_options),
        key="qa_taxonomy_analysis_baseline",
    )
    level = controls[1].selectbox(
        "Comparison level",
        list(llm_columns),
        index=1,
        key="qa_taxonomy_analysis_level",
    )
    display_mode = controls[2].segmented_control(
        "Matrix values",
        ["URL count", "Baseline-row share"],
        default="URL count",
        key="qa_taxonomy_analysis_matrix_mode",
    )
    baseline_column = baseline_options[baseline_name][level]
    llm_column = llm_columns[level]

    comparison = qa.taxonomy_confusion_table(evidence, baseline_column, llm_column)
    baseline_totals = comparison.groupby("baseline_label")["unique_urls"].sum().sort_values(ascending=False)
    llm_totals = comparison.groupby("llm_label")["unique_urls"].sum().sort_values(ascending=False)
    max_categories = 14 if level == "Detailed page type" else 18
    keep_baseline = set(baseline_totals.head(max_categories).index) | {"unknown"}
    keep_llm = set(llm_totals.head(max_categories).index) | {"unknown"}
    plotted = comparison.copy()
    plotted["baseline_plot"] = plotted["baseline_label"].where(
        plotted["baseline_label"].isin(keep_baseline), "rare_other"
    )
    plotted["llm_plot"] = plotted["llm_label"].where(
        plotted["llm_label"].isin(keep_llm), "rare_other"
    )
    matrix = plotted.pivot_table(
        index="baseline_plot",
        columns="llm_plot",
        values="unique_urls",
        aggfunc="sum",
        fill_value=0,
    )
    matrix = matrix.loc[
        matrix.sum(axis=1).sort_values(ascending=False).index,
        matrix.sum(axis=0).sort_values(ascending=False).index,
    ]
    if display_mode == "Baseline-row share":
        plotted_matrix = matrix.div(matrix.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
        text_auto = ".0%"
        color_title = "Row share"
    else:
        plotted_matrix = matrix
        text_auto = True
        color_title = "URLs"
    fig = px.imshow(
        plotted_matrix,
        text_auto=text_auto,
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "Gemini label", "y": f"{baseline_name} label", "color": color_title},
        title=f"{baseline_name} versus Gemini: {level.casefold()}",
    )
    fig.update_layout(
        height=max(520, len(matrix) * 34 + 190),
        margin={"l": 210, "r": 30, "t": 80, "b": 180},
    )
    fig.update_xaxes(tickangle=45, automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, width="stretch")
    with st.expander("Show comparison matrix as a table"):
        st.dataframe(plotted_matrix, width="stretch")

    left, right = st.columns(2, gap="large")
    rule_unknown = evidence[baseline_column].fillna("unknown").astype(str).eq("unknown")
    resolved = evidence[
        rule_unknown & evidence[llm_column].fillna("unknown").astype(str).ne("unknown")
    ]
    resolved_distribution = (
        resolved[llm_column]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .head(15)
        .rename_axis("Gemini label")
        .reset_index(name="URLs")
        .sort_values("URLs")
    )
    resolved_fig = px.bar(
        resolved_distribution,
        x="URLs",
        y="Gemini label",
        orientation="h",
        text="URLs",
        title=f"How Gemini resolved {baseline_name} unknowns",
        color_discrete_sequence=["#287a8e"],
    )
    resolved_fig.update_layout(height=520, yaxis_title=None, margin={"l": 180, "r": 20, "t": 65, "b": 40})
    left.plotly_chart(resolved_fig, width="stretch")

    agreement = evidence[baseline_column].fillna("unknown").astype(str).eq(
        evidence[llm_column].fillna("unknown").astype(str)
    )
    quality = evidence.assign(
        comparison_status=agreement.map({True: "Exact agreement", False: "Different label"}),
        comparison_group=evidence["content_strength"].fillna("unknown").astype(str),
    )
    quality_table = (
        quality.groupby(["comparison_group", "comparison_status"], dropna=False)
        .size()
        .reset_index(name="URLs")
    )
    quality_table["share"] = quality_table["URLs"] / quality_table.groupby("comparison_group")["URLs"].transform("sum")
    quality_fig = px.bar(
        quality_table,
        x="comparison_group",
        y="share",
        color="comparison_status",
        barmode="stack",
        text=quality_table["share"].map(lambda value: f"{value:.0%}"),
        title="Agreement by extracted-content strength",
        color_discrete_map={"Exact agreement": "#198754", "Different label": "#d7dde5"},
        category_orders={"comparison_group": ["strong", "medium", "weak", "failed", "unknown"]},
    )
    quality_fig.update_layout(height=520, xaxis_title=None, yaxis_title="Share of URLs", legend_title=None)
    quality_fig.update_yaxes(tickformat=".0%", range=[0, 1])
    right.plotly_chart(quality_fig, width="stretch")

    st.markdown("#### Disagreement review queue")
    confidence_options = ["All"] + sorted(evidence["llm_confidence"].dropna().astype(str).unique().tolist())
    queue_controls = st.columns([1, 1, 2])
    confidence_filter = queue_controls[0].selectbox(
        "Gemini confidence",
        confidence_options,
        key="qa_taxonomy_analysis_confidence",
    )
    input_options = ["All"] + sorted(
        evidence["classification_input_mode"].dropna().astype(str).unique().tolist()
    )
    input_filter = queue_controls[1].selectbox(
        "Classification input",
        input_options,
        key="qa_taxonomy_analysis_input",
    )
    domain_query = queue_controls[2].text_input(
        "Filter domain",
        placeholder="example.com",
        key="qa_taxonomy_analysis_domain",
    ).strip()
    queue = evidence[~agreement].copy()
    if confidence_filter != "All":
        queue = queue[queue["llm_confidence"].astype(str).eq(confidence_filter)]
    if input_filter != "All":
        queue = queue[queue["classification_input_mode"].astype(str).eq(input_filter)]
    if domain_query:
        queue = queue[
            queue["source_root_domain"].fillna("").astype(str).str.contains(
                domain_query, case=False, regex=False
            )
        ]
    queue = queue.sort_values(["source_appearances", "cited_appearances"], ascending=False)
    queue_display = pd.DataFrame(
        {
            "Title": queue["page_title"],
            "Domain": queue["source_root_domain"],
            f"{baseline_name} label": queue[baseline_column],
            "Gemini label": queue[llm_column],
            "Gemini confidence": queue["llm_confidence"],
            "Input": queue["classification_input_mode"],
            "Content strength": queue["content_strength"],
            "Gemini evidence": queue["llm_evidence"],
            "URL": queue["source_url"],
        }
    )
    st.caption(f"{len(queue):,} URLs match the active disagreement filters.")
    st.dataframe(
        queue_display.head(500),
        width="stretch",
        hide_index=True,
        column_config={"URL": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    st.download_button(
        "Download filtered disagreement queue",
        queue_display.to_csv(index=False).encode("utf-8"),
        file_name=f"taxonomy_disagreements_{baseline_name.casefold().replace(' ', '_')}_{level.casefold().replace(' ', '_').replace('/', '_')}.csv",
        mime="text/csv",
        key="qa_taxonomy_analysis_download",
    )
    if queue.empty:
        st.info("No disagreement webpages match the active filters.")
        return

    st.divider()
    st.markdown("#### Inspect a disagreement webpage")
    source_rows = queue.drop_duplicates("normalized_url").copy()
    source_labels = {}
    for _, source in source_rows.iterrows():
        normalized = str(source.get("normalized_url") or "")
        domain = str(source.get("source_root_domain") or "unknown")
        baseline_label = _label_value(source, baseline_column)
        gemini_label = _label_value(source, llm_column)
        title = _label_value(source, "page_title", normalized)[:55]
        label = f"{domain} | {baseline_label} -> {gemini_label} | {title} | {normalized[-16:]}"
        source_labels[label] = normalized
    selected_source_label = st.selectbox(
        "Disagreement webpage",
        list(source_labels),
        key="qa_taxonomy_analysis_source_page",
    )
    selected_normalized = source_labels[selected_source_label]
    source = source_rows[
        source_rows["normalized_url"].astype(str).eq(selected_normalized)
    ].iloc[0]
    source_url = str(source.get("source_url") or selected_normalized)
    st.caption(
        f"{baseline_name}: {_label_value(source, baseline_column)} | "
        f"Gemini: {_label_value(source, llm_column)} | "
        f"Gemini confidence: {_label_value(source, 'llm_confidence')}"
    )
    _taxonomy_comparison(source, heading="Disagreement classification")
    _live_page_panel(source_url, "qa_taxonomy_analysis_source", height=760)


def _econometrics(bundle: qa.QABundle) -> None:
    del bundle
    C.section(
        "Econometric analysis layers",
        "The redesigned pipeline is frozen to D0, FE1, FE2, FE3, and FE4.",
    )
    st.warning(
        "Results describe conditional associations among surfaced sources. They are not causal effects "
        "or web-wide citation probabilities."
    )
    st.code("D0 -> FE1 -> FE2\n              |-> FE3\n              |-> FE4", language="text")
    st.caption(
        "FE3 and FE4 are separate branches from FE2. FE4 does not include domain fixed effects. "
        "Changing the clustered standard-error estimator does not create another model."
    )

    repo = Path(__file__).resolve().parents[2]
    artifact_dir = repo / "outputs/econometrics_redesign_v2_20260722/frontend"
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        st.error(f"Validated econometrics artifact manifest is missing: {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = ["D0", "FE1", "FE2", "FE3", "FE4"]
    if manifest.get("layers") != expected or not manifest.get("validated"):
        st.error("The artifact manifest failed the five-layer scope or validation check.")
        return
    for filename, metadata in manifest.get("files", {}).items():
        path = artifact_dir / filename
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != metadata.get("sha256"):
            st.error(f"Validated artifact hash mismatch: {filename}")
            return

    registry = pd.read_csv(artifact_dir / "model_layers.csv", low_memory=False)
    observed = registry["analysis_layer"].astype(str).tolist()
    if observed != expected:
        st.error("The model registry failed the five-layer scope check, so no model view is displayed.")
        return

    display = registry.rename(
        columns={
            "analysis_layer": "Layer",
            "analysis_type": "Analysis type",
            "formula_scope": "Formula scope",
            "focal_predictors": "Focal predictors",
            "controls": "Controls",
            "fixed_effects": "Fixed effects",
            "branch_from": "Branch from",
            "role": "Role",
            "implementation_status": "Status",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download five-layer model registry",
        registry.to_csv(index=False).encode("utf-8"),
        "econometrics_model_registry_v2.csv",
        "text/csv",
        key="qa_five_layer_model_registry_download",
    )

    st.divider()
    estimates = pd.read_csv(artifact_dir / "model_estimates.csv", low_memory=False)
    focal = estimates[estimates["term"].isin([
        "log2_word_count_plus1",
        "has_verified_html_table",
        "factual_numeric_density_score",
        "writing_structure_score",
    ])].copy()
    selected_layer = st.segmented_control(
        "Regression layer",
        ["FE1", "FE2", "FE3", "FE4"],
        default="FE2",
        key="qa_econometrics_layer_v2",
    )
    selected = focal[focal["analysis_layer"].eq(selected_layer or "FE2")].copy()
    selected["Estimate (pp)"] = selected["estimate_pp"]
    selected["95% CI (pp)"] = selected.apply(
        lambda row: f"{row['conf_low_pp']:.2f} to {row['conf_high_pp']:.2f}", axis=1
    )
    selected["Feature"] = selected["term"]
    selected["N"] = selected["n_obs"].astype(int)
    st.markdown("#### Two-way prompt/URL clustered estimates")
    st.dataframe(
        selected[["model_id", "Feature", "Estimate (pp)", "95% CI (pp)", "p_value", "N"]],
        width="stretch",
        hide_index=True,
    )
    if not selected.empty:
        figure = px.scatter(
            selected,
            x="estimate_pp",
            y="term",
            error_x=selected["conf_high_pp"] - selected["estimate_pp"],
            error_x_minus=selected["estimate_pp"] - selected["conf_low_pp"],
            labels={"estimate_pp": "Conditional association (percentage points)", "term": "Feature"},
        )
        figure.add_vline(x=0, line_dash="dash", line_color="#6b7280")
        figure.update_layout(height=max(320, 62 * len(selected)), showlegend=False)
        st.plotly_chart(figure, width="stretch")
    st.caption(
        "Extraction Strength is a measurement-quality control. FE4 taxonomy is Gemini content-informed "
        "and may over-control content signals. External Evidence Structure is blocked and was not fitted."
    )


def _not_missing(value: object) -> bool:
    return value is not None and not (not isinstance(value, (list, dict)) and pd.isna(value))


def _display_measurement(value: object, *, digits: int = 3) -> str:
    if not _not_missing(value):
        return "Unmeasured"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{float(numeric):,.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _binary_result(value: object, true_label: str, false_label: str) -> str:
    if not _not_missing(value):
        return "Unmeasured"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Unmeasured"
    return true_label if float(numeric) == 1 else false_label


def _feature_formula(feature_name: str) -> str:
    formulas = {
        "log2_word_count_plus1": "log2(word_count + 1)",
        "has_verified_html_table": "has_html_table when html_available = 1; otherwise Unmeasured",
        "factual_numeric_density_score": (
            "min(number_token_per_1000_words / 10, 5) + I(percent_mention_count > 0) + "
            "I(year_mention_count > 0) + I(range_mention_count > 0) + log1p(measurement_mention_count)"
        ),
        "writing_structure_score": " + ".join(feature_qa.COMPONENTS),
    }
    return formulas.get(feature_name, "Binary component: 1 = detected, 0 = not detected, missing = unmeasured")


def _apply_feature_value_filter(frame: pd.DataFrame, feature_name: str, selection: str) -> pd.DataFrame:
    if selection == "All":
        return frame
    if feature_name == "writing_structure_score":
        score = pd.to_numeric(frame[feature_name], errors="coerce")
        masks = {
            "Score 0": score.eq(0),
            "Score 1-2": score.between(1, 2),
            "Score 3-4": score.between(3, 4),
            "Score 5-6": score.between(5, 6),
            "Unmeasured": score.isna(),
        }
        return frame[masks[selection]]
    if feature_name in {"has_verified_html_table", *feature_qa.COMPONENTS}:
        states = frame[feature_name].map(feature_qa.nullable_binary_status)
        return frame[states.eq(selection)]
    numeric = pd.to_numeric(frame[feature_name], errors="coerce")
    if selection == "Unmeasured":
        return frame[numeric.isna()]
    return frame


def _order_validation_sample(frame: pd.DataFrame, mode: str, feature_name: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    bin_data = feature_support.feature_bins(data, feature_name)
    data["_feature_bin"] = bin_data["bin_key"]
    bin_counts = data["_feature_bin"].value_counts(dropna=False)
    if mode == "Random sample":
        return data.sample(frac=1, random_state=20260723).drop(columns="_feature_bin")
    if mode == "High-score examples":
        return data.sort_values(feature_name, ascending=False, na_position="last").drop(columns="_feature_bin")
    if mode == "Low-score examples":
        return data.sort_values(feature_name, ascending=True, na_position="last").drop(columns="_feature_bin")
    if mode == "Disagreement / suspicious examples":
        return data.sort_values(
            ["suspicious_measurement", "writing_score_matches_components"],
            ascending=[False, True],
        ).drop(columns="_feature_bin")
    if mode == "Rare feature examples":
        data["_bin_frequency"] = data["_feature_bin"].map(bin_counts)
        return data.sort_values(["_bin_frequency", "_feature_bin"]).drop(
            columns=["_feature_bin", "_bin_frequency"]
        )
    if mode == "Common feature examples":
        data["_bin_frequency"] = data["_feature_bin"].map(bin_counts)
        return data.sort_values(["_bin_frequency", "_feature_bin"], ascending=[False, True]).drop(
            columns=["_feature_bin", "_bin_frequency"]
        )
    if mode == "Missing or unmeasured examples":
        data["_missing_order"] = data["_feature_bin"].ne("unmeasured")
        return data.sort_values(["_missing_order", "_feature_bin"]).drop(
            columns=["_feature_bin", "_missing_order"]
        )
    if mode == "Extreme continuous values":
        numeric = pd.to_numeric(data[feature_name], errors="coerce")
        data["_extreme_distance"] = (numeric - numeric.median()).abs()
        return data.sort_values("_extreme_distance", ascending=False, na_position="last").drop(
            columns=["_feature_bin", "_extreme_distance"]
        )
    if mode == "Prompts with no within-prompt variation":
        variation = feature_support.prompt_variation(data, feature_name)
        no_variation = ~data["prompt_id"].astype(str).isin(variation["_varying_prompts"])
        return data[no_variation].drop(columns="_feature_bin")
    grouping = {
        "Stratified by selected feature": "_feature_bin",
        "Stratified by cited status": "cited",
        "Stratified by content_strength": "content_strength",
    }.get(mode)
    if grouping:
        data["_stratum_order"] = data.groupby(grouping, dropna=False).cumcount()
        return data.sort_values(["_stratum_order", grouping]).drop(
            columns=["_stratum_order", "_feature_bin"]
        )
    return data.drop(columns="_feature_bin")


def _plot_selection_points(event: object) -> list[dict]:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    return list(points or [])


def _selected_custom_value(event: object) -> str | None:
    points = _plot_selection_points(event)
    if not points:
        return None
    custom = points[0].get("customdata")
    if isinstance(custom, (list, tuple)):
        custom = custom[0] if custom else None
    return None if custom is None else str(custom)


def _support_summary_metrics(summary_row: pd.Series) -> None:
    metrics = [
        {"value": f"{int(summary_row['total_observations']):,}", "label": "total observations"},
        {"value": f"{int(summary_row['measured_n']):,}", "label": "measured"},
        {"value": f"{int(summary_row['missing_n']):,}", "label": "missing / unmeasured"},
        {"value": f"{summary_row['missing_pct']:.1%}", "label": "missing"},
        {"value": f"{int(summary_row['unique_values']):,}", "label": "unique values"},
        {
            "value": f"{summary_row['prompts_with_variation_pct']:.1%}",
            "label": "prompts with FE variation",
        },
    ]
    C.metric_cards(metrics)
    numeric_fields = [
        ("mean", "Mean"),
        ("standard_deviation", "Std. dev."),
        ("minimum", "Minimum"),
        ("p05", "P05"),
        ("p25", "P25"),
        ("median", "Median"),
        ("p75", "P75"),
        ("p95", "P95"),
        ("maximum", "Maximum"),
    ]
    values = [
        (label, _display_measurement(summary_row.get(field)))
        for field, label in numeric_fields
    ]
    values.extend(
        [
            ("Prompts", f"{int(summary_row['number_of_prompts']):,}"),
            (
                "Rows in identifying prompts",
                f"{int(summary_row['rows_within_identifying_prompts']):,} "
                f"({summary_row['rows_within_identifying_prompts_pct']:.1%})",
            ),
        ]
    )
    st.dataframe(pd.DataFrame(values, columns=["Statistic", "Value"]), width="stretch", hide_index=True)


def _record_linked_selection(
    event: object,
    *,
    feature: str,
    content_strength: str | None = None,
) -> None:
    selected = _selected_custom_value(event)
    if selected is None:
        return
    st.session_state["feature_qa_linked_filter"] = {
        "feature": feature,
        "bin_key": selected,
        "content_strength": content_strength,
    }


def _feature_distribution_dashboard(
    evidence: pd.DataFrame,
    artifacts: dict[str, pd.DataFrame],
    selected_feature: str,
) -> dict:
    summary = artifacts["feature_distribution_support_summary.csv"]
    bins = artifacts["feature_distribution_bins.csv"]
    strength = artifacts["feature_distribution_by_content_strength.csv"]
    within = artifacts["feature_within_prompt_variation.csv"]
    selected_summary = summary[summary["feature_name"].eq(selected_feature)].iloc[0]
    selected_bins = bins[bins["feature_name"].eq(selected_feature)].sort_values("bin_order")
    selected_strength = strength[strength["feature_name"].eq(selected_feature)].sort_values(
        ["content_strength", "bin_order"]
    )

    st.markdown("### A. Feature Distribution and Support Dashboard")
    role = selected_summary["model_role"]
    if role == "D0_QA_diagnostic_only":
        st.warning("`heading_count_group` is an optional D0/QA diagnostic and is not an active LPM predictor.")
    st.caption(
        "Zeros and false/absent measurements remain distinct from missing or unmeasured values. "
        "Citation status is used only for descriptive comparison."
    )
    _support_summary_metrics(selected_summary)

    warnings = [
        str(selected_summary.get("low_support_warning") or "").strip(),
        str(selected_summary.get("imbalance_warning") or "").strip(),
        str(selected_summary.get("extraction_dependence_warning") or "").strip(),
    ]
    warnings = [warning for warning in warnings if warning and warning.casefold() != "nan"]
    if warnings:
        st.warning("Diagnostic warning: " + " | ".join(warnings))
    if feature_support.feature_type(selected_feature) == "binary":
        binary_support = selected_bins[
            [
                "bin_label",
                "n_rows",
                "percentage_all_rows",
                "cited_rate",
                "unique_prompts",
                "prompts_with_usable_variation",
            ]
        ].rename(
            columns={
                "bin_label": "Measurement",
                "n_rows": "Rows",
                "percentage_all_rows": "Row share",
                "cited_rate": "Cited rate",
                "unique_prompts": "Prompts",
                "prompts_with_usable_variation": "Prompts with usable variation",
            }
        )
        binary_support["Row share"] = binary_support["Row share"].map(lambda value: f"{value:.1%}")
        binary_support["Cited rate"] = binary_support["Cited rate"].map(
            lambda value: "No rows" if pd.isna(value) else f"{value:.1%}"
        )
        st.dataframe(binary_support, width="stretch", hide_index=True)

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.markdown("#### Overall distribution")
        overall = go.Figure(
            go.Bar(
                x=selected_bins["bin_label"],
                y=selected_bins["n_rows"],
                customdata=np.asarray(selected_bins["bin_key"]).reshape(-1, 1),
                text=[
                    f"{int(n):,}<br>{pct:.1%}" if pd.notna(pct) else f"{int(n):,}<br>unmeasured"
                    for n, pct in zip(
                        selected_bins["n_rows"], selected_bins["percentage_measured_rows"]
                    )
                ],
                textposition="outside",
                marker_color="#3178a8",
            )
        )
        overall.update_layout(
            xaxis_title=feature_support.FEATURE_LABELS[selected_feature],
            yaxis_title="Observations",
            height=410,
            showlegend=False,
        )
        overall_event = st.plotly_chart(
            overall,
            width="stretch",
            key=f"feature_support_overall_{selected_feature}",
            on_select="rerun",
            selection_mode="points",
        )
        _record_linked_selection(overall_event, feature=selected_feature)

    with chart_right:
        st.markdown("#### Distribution by citation status")
        citation = go.Figure()
        citation.add_bar(
            name="More-only",
            x=selected_bins["bin_label"],
            y=selected_bins["more_only_rows"],
            customdata=np.asarray(selected_bins["bin_key"]).reshape(-1, 1),
            marker_color="#8fa8bb",
        )
        citation.add_bar(
            name="Cited",
            x=selected_bins["bin_label"],
            y=selected_bins["cited_rows"],
            customdata=np.asarray(selected_bins["bin_key"]).reshape(-1, 1),
            marker_color="#d65f4c",
        )
        citation.update_layout(
            barmode="group",
            xaxis_title=feature_support.FEATURE_LABELS[selected_feature],
            yaxis_title="Observations",
            height=410,
        )
        citation_event = st.plotly_chart(
            citation,
            width="stretch",
            key=f"feature_support_citation_{selected_feature}",
            on_select="rerun",
            selection_mode="points",
        )
        _record_linked_selection(citation_event, feature=selected_feature)

    extraction_left, cited_right = st.columns(2, gap="large")
    with extraction_left:
        st.markdown("#### Distribution by Extraction Strength")
        strength_order = ["strong", "medium", "weak", "failed_or_unknown"]
        colors = {
            "strong": "#2f855a",
            "medium": "#d69e2e",
            "weak": "#c05621",
            "failed_or_unknown": "#718096",
        }
        if feature_support.feature_type(selected_feature) == "continuous":
            box_data = evidence[["content_strength", selected_feature]].copy()
            box_data[selected_feature] = pd.to_numeric(box_data[selected_feature], errors="coerce")
            extraction = px.box(
                box_data,
                x="content_strength",
                y=selected_feature,
                color="content_strength",
                category_orders={"content_strength": strength_order},
                color_discrete_map=colors,
                points=False,
                labels={
                    "content_strength": "Extraction Strength",
                    selected_feature: feature_support.FEATURE_LABELS[selected_feature],
                },
            )
            extraction.update_layout(height=410, showlegend=False)
            st.plotly_chart(
                extraction,
                width="stretch",
                key=f"feature_support_strength_{selected_feature}",
            )
            strength_buttons = st.columns(3)
            for column, level in zip(strength_buttons, ["strong", "medium", "weak"]):
                if column.button(
                    f"Review {level}",
                    key=f"feature_support_strength_filter_{selected_feature}_{level}",
                    width="stretch",
                ):
                    st.session_state["feature_qa_linked_filter"] = {
                        "feature": selected_feature,
                        "bin_key": None,
                        "content_strength": level,
                    }
                    st.rerun()
        else:
            extraction = go.Figure()
            visible_levels = []
            all_bin_keys = selected_bins["bin_key"].astype(str).tolist()
            bin_labels_by_key = selected_bins.set_index("bin_key")["bin_label"].to_dict()
            for level in strength_order:
                group = selected_strength[selected_strength["content_strength"].eq(level)]
                if group.empty:
                    continue
                visible_levels.append(level)
                group_by_key = group.set_index(group["bin_key"].astype(str))
                counts = [
                    int(group_by_key.loc[key, "n_rows"]) if key in group_by_key.index else 0
                    for key in all_bin_keys
                ]
                extraction.add_bar(
                    name=level,
                    x=[bin_labels_by_key.get(key, key) for key in all_bin_keys],
                    y=counts,
                    customdata=np.asarray(all_bin_keys).reshape(-1, 1),
                    marker_color=colors[level],
                )
            extraction.update_layout(
                barmode="group",
                xaxis_title=feature_support.FEATURE_LABELS[selected_feature],
                yaxis_title="Observations",
                height=410,
            )
            extraction_event = st.plotly_chart(
                extraction,
                width="stretch",
                key=f"feature_support_strength_{selected_feature}",
                on_select="rerun",
                selection_mode="points",
            )
            selected_point = _selected_custom_value(extraction_event)
            if selected_point:
                points = _plot_selection_points(extraction_event)
                curve = int(points[0].get("curve_number", 0))
                selected_level = visible_levels[curve] if curve < len(visible_levels) else None
                st.session_state["feature_qa_linked_filter"] = {
                    "feature": selected_feature,
                    "bin_key": selected_point,
                    "content_strength": selected_level,
                }

    with cited_right:
        st.markdown("#### Descriptive cited rate by value or governed bin")
        valid_rate = selected_bins[selected_bins["n_rows"].gt(0)].copy()
        cited_figure = go.Figure(
            go.Scatter(
                x=valid_rate["bin_label"],
                y=valid_rate["cited_rate"],
                mode="lines+markers+text",
                customdata=np.asarray(valid_rate["bin_key"]).reshape(-1, 1),
                text=[f"n={int(value):,}" for value in valid_rate["n_rows"]],
                textposition="top center",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": valid_rate["ci_high"] - valid_rate["cited_rate"],
                    "arrayminus": valid_rate["cited_rate"] - valid_rate["ci_low"],
                },
                marker={"size": 9, "color": "#d65f4c"},
                line={"color": "#d65f4c"},
            )
        )
        cited_figure.update_layout(
            xaxis_title=feature_support.FEATURE_LABELS[selected_feature],
            yaxis_title="Cited rate (descriptive)",
            yaxis_tickformat=".0%",
            height=410,
            showlegend=False,
        )
        cited_event = st.plotly_chart(
            cited_figure,
            width="stretch",
            key=f"feature_support_cited_rate_{selected_feature}",
            on_select="rerun",
            selection_mode="points",
        )
        _record_linked_selection(cited_event, feature=selected_feature)
        if selected_feature == "writing_structure_score":
            sparse = selected_bins[selected_bins["n_rows"].between(1, 19)]
            if not sparse.empty:
                st.warning("One or more observed scores have fewer than 20 rows; cited rates are unstable.")
        st.caption("Descriptive association among surfaced, measurable rows. This is not a causal effect.")

    st.markdown("#### Prompt Fixed-Effect identifying support")
    support_plot = within.copy()
    support_plot["Feature"] = support_plot["feature_name"].map(feature_support.FEATURE_LABELS)
    support_plot["Prompts with variation"] = support_plot["prompts_with_variation_pct"]
    support_figure = px.bar(
        support_plot.sort_values("Prompts with variation"),
        x="Prompts with variation",
        y="Feature",
        orientation="h",
        text=support_plot.sort_values("Prompts with variation")[
            "prompts_with_usable_variation"
        ],
        labels={"Prompts with variation": "Share of prompts with usable within-prompt variation"},
    )
    support_figure.update_traces(marker_color="#3178a8", texttemplate="%{text} prompts")
    support_figure.update_layout(
        xaxis_tickformat=".0%",
        height=330,
        showlegend=False,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
    )
    st.plotly_chart(support_figure, width="stretch", key="feature_support_within_prompt")
    within_display = within.rename(
            columns={
                "feature_name": "Feature",
                "prompts_with_no_variation": "Prompts without variation",
                "prompts_with_usable_variation": "Prompts with variation",
                "rows_in_prompts_with_usable_variation": "Rows in identifying prompts",
                "rows_in_prompts_with_usable_variation_pct": "Identifying-row share",
                "median_within_prompt_std": "Median within-prompt SD",
                "prompts_containing_both_0_and_1": "Prompts with both 0 and 1",
            }
        )[
            [
                "Feature",
                "Prompts without variation",
                "Prompts with variation",
                "Rows in identifying prompts",
                "Identifying-row share",
                "Median within-prompt SD",
                "Prompts with both 0 and 1",
            ]
        ].copy()
    within_display["Identifying-row share"] = within_display[
        "Identifying-row share"
    ].map(lambda value: f"{value:.1%}")
    st.dataframe(
        within_display,
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Feature commonness and imbalance")
    sort_mode = st.selectbox(
        "Sort feature overview",
        [
            "Highest missingness",
            "Lowest within-prompt variation",
            "Strongest imbalance",
            "Lowest support",
        ],
        key="feature_support_summary_sort",
    )
    if sort_mode == "Highest missingness":
        overview = summary.sort_values("missing_pct", ascending=False)
    elif sort_mode == "Lowest within-prompt variation":
        overview = summary.sort_values("prompts_with_variation_pct")
    elif sort_mode == "Strongest imbalance":
        overview = summary.sort_values("imbalance_score", ascending=False)
    else:
        overview = summary.sort_values(["low_support_warning", "measured_n"], ascending=[False, True])
    display_columns = [
        "feature_name",
        "feature_type",
        "measured_n",
        "missing_n",
        "missing_pct",
        "prevalence_or_mean",
        "standard_deviation",
        "unique_values",
        "prompts_with_variation",
        "prompts_with_variation_pct",
        "rows_within_identifying_prompts",
        "zero_pct",
        "p05",
        "median",
        "p95",
        "low_support_warning",
        "imbalance_warning",
        "extraction_dependence_warning",
    ]
    overview_event = st.dataframe(
        overview[display_columns],
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="feature_support_overview_table",
    )
    selected_rows = getattr(getattr(overview_event, "selection", None), "rows", [])
    if selected_rows:
        requested = str(overview.iloc[int(selected_rows[0])]["feature_name"])
        if requested != selected_feature:
            st.session_state["feature_qa_requested_feature"] = requested
            st.rerun()
    st.download_button(
        "Download feature support summary",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="feature_distribution_support_summary.csv",
        mime="text/csv",
        key="download_feature_support_summary",
    )

    linked = st.session_state.get("feature_qa_linked_filter", {})
    if linked and linked.get("feature") == selected_feature:
        selected_label = (
            selected_bins.set_index("bin_key")["bin_label"].to_dict().get(
                linked.get("bin_key"), linked.get("bin_key")
            )
            if linked.get("bin_key")
            else "All feature values"
        )
        strength_note = (
            f"; Extraction Strength = {linked['content_strength']}"
            if linked.get("content_strength")
            else ""
        )
        notice, clear = st.columns([5, 1])
        notice.info(f"Chart-linked reviewer filter: {selected_label}{strength_note}")
        if clear.button("Clear", key="clear_feature_qa_linked_filter", width="stretch"):
            st.session_state.pop("feature_qa_linked_filter", None)
            st.rerun()
    return linked


def _feature_validation(bundle: qa.QABundle) -> None:
    del bundle
    C.section(
        "Manual feature validation",
        "Compare governed feature values with the stored text used by the producer. Reviews never alter model inputs.",
    )
    repo = Path(__file__).resolve().parents[2]
    frontend_dir = repo / "outputs/econometrics_redesign_v2_20260722/frontend"
    manifest_path = frontend_dir / "manual_feature_validation_manifest.json"
    if not manifest_path.exists():
        st.warning("The manual feature-validation artifact has not been built.")
        st.code(".venv/bin/python scripts/v2_build_manual_feature_validation_artifact.py", language="bash")
        return
    try:
        rows, content, manifest = _load_feature_validation_artifacts(
            str(frontend_dir), manifest_path.stat().st_mtime
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Could not load validated feature-review artifacts: {type(exc).__name__}: {exc}")
        return
    join_keys = ["normalized_url", "source_url"]
    duplicated_provenance = [
        column for column in content.columns if column in rows.columns and column not in join_keys
    ]
    evidence = rows.merge(
        content.drop(columns=duplicated_provenance),
        on=join_keys,
        how="left",
        validate="many_to_one",
    )
    evidence["qa_row_key"] = evidence["prompt_id"].astype(str) + " | " + evidence["normalized_url"].astype(str)

    support_manifest_path = frontend_dir / "feature_distribution_support_manifest.json"
    if not support_manifest_path.exists():
        st.warning("The feature distribution and support artifacts have not been built.")
        st.code(".venv/bin/python scripts/v2_build_manual_feature_validation_artifact.py", language="bash")
        return
    try:
        support_artifacts, _ = _load_feature_support_artifacts(
            str(support_manifest_path), support_manifest_path.stat().st_mtime
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        st.error(f"Could not load feature distribution artifacts: {type(exc).__name__}: {exc}")
        return

    requested_feature = st.session_state.pop("feature_qa_requested_feature", None)
    feature_state_key = "feature_qa_selected_feature"
    feature_options = list(feature_support.DASHBOARD_FEATURES)
    if requested_feature in feature_options:
        st.session_state[feature_state_key] = requested_feature
    selected_feature = st.selectbox(
        "Feature to inspect",
        feature_options,
        format_func=lambda feature: feature_support.FEATURE_LABELS[feature],
        key=feature_state_key,
    )
    linked_filter = _feature_distribution_dashboard(
        evidence, support_artifacts, selected_feature
    )

    st.divider()
    st.markdown("### B. Webpage-Level Manual Review")
    st.caption(
        "Review rows are source appearances because citation status and prompt fixed-effect support vary by prompt. "
        "Stored producer content remains authoritative."
    )

    with st.expander("Filters and sampling", expanded=True):
        first = st.columns([2.2, 1, 1, 1])
        query = first[0].text_input(
            "Search URL or domain", placeholder="example.com or URL fragment", key="feature_qa_search"
        ).strip()
        prompt_options = ["All"] + sorted(evidence["prompt_id"].dropna().astype(str).unique().tolist())
        prompt_filter = first[1].selectbox("Prompt", prompt_options, key="feature_qa_prompt")
        cited_filter = first[2].selectbox("Citation", ["All", "Cited", "More-only"], key="feature_qa_cited")
        measured_filter = first[3].selectbox(
            "Measurement status",
            ["All", "Measured only", "Unmeasured only"],
            key="feature_qa_measured",
        )

        second = st.columns(4)
        strength_options = ["All"] + sorted(evidence["content_strength"].dropna().astype(str).unique().tolist())
        strength = second[0].selectbox("Extraction Strength", strength_options, key="feature_qa_strength")
        page_options = ["All"] + sorted(evidence["page_type_family_gemini_v1_collapsed"].dropna().astype(str).unique().tolist())
        page_type = second[1].selectbox("Page type", page_options, key="feature_qa_page_type")
        source_options = ["All"] + sorted(evidence["source_type_general_gemini_v1_collapsed"].dropna().astype(str).unique().tolist())
        source_type = second[2].selectbox("Source type", source_options, key="feature_qa_source_type")
        domain_options = ["All"] + sorted(
            evidence["source_root_domain"].dropna().astype(str).unique().tolist()
        )
        domain_filter = second[3].selectbox("Domain", domain_options, key="feature_qa_domain")

        third = st.columns(4)
        selected_bins = support_artifacts["feature_distribution_bins.csv"]
        selected_bins = selected_bins[selected_bins["feature_name"].eq(selected_feature)].sort_values(
            "bin_order"
        )
        bin_labels = {
            str(row.bin_label): str(row.bin_key)
            for row in selected_bins.itertuples(index=False)
        }
        value_label = third[0].selectbox(
            "Feature value or governed range",
            ["All", *bin_labels.keys()],
            key=f"feature_qa_value_{selected_feature}",
        )
        variation_filter = third[1].selectbox(
            "Within-prompt variation",
            ["All prompts", "Prompts with variation only", "Prompts without variation"],
            key="feature_qa_variation",
        )
        suspicious_filter = third[2].selectbox(
            "Measurement QA",
            ["All", "Suspicious only", "Missing any active feature"],
            key="feature_qa_suspicious",
        )
        scope_options = ["All"] + sorted(evidence["feature_extraction_text_scope"].dropna().astype(str).unique().tolist())
        scope_filter = third[3].selectbox("Producer text scope", scope_options, key="feature_qa_scope")

        sampling = st.selectbox(
            "Sampling mode",
            [
                "Random sample", "Stratified by selected feature", "Stratified by cited status",
                "Stratified by content_strength", "Rare feature examples", "Common feature examples",
                "Missing or unmeasured examples", "High-score examples", "Low-score examples",
                "Extreme continuous values", "Prompts with no within-prompt variation",
                "Disagreement / suspicious examples",
            ],
            key="feature_qa_sampling",
        )

    filtered = evidence.copy()
    if query:
        filtered = filtered[
            filtered["normalized_url"].fillna("").astype(str).str.contains(query, case=False, regex=False)
            | filtered["source_root_domain"].fillna("").astype(str).str.contains(query, case=False, regex=False)
        ]
    if prompt_filter != "All":
        filtered = filtered[filtered["prompt_id"].astype(str).eq(prompt_filter)]
    if cited_filter != "All":
        cited = pd.to_numeric(filtered["cited"], errors="coerce")
        filtered = filtered[cited.eq(1 if cited_filter == "Cited" else 0)]
    if strength != "All":
        filtered = filtered[filtered["content_strength"].astype(str).eq(strength)]
    if page_type != "All":
        filtered = filtered[filtered["page_type_family_gemini_v1_collapsed"].astype(str).eq(page_type)]
    if source_type != "All":
        filtered = filtered[filtered["source_type_general_gemini_v1_collapsed"].astype(str).eq(source_type)]
    if domain_filter != "All":
        filtered = filtered[filtered["source_root_domain"].astype(str).eq(domain_filter)]
    if scope_filter != "All":
        filtered = filtered[filtered["feature_extraction_text_scope"].astype(str).eq(scope_filter)]
    if feature_support.feature_type(selected_feature) in {"binary", "continuous", "score"}:
        selected_measured = pd.to_numeric(filtered[selected_feature], errors="coerce").notna()
    else:
        selected_measured = filtered[selected_feature].notna()
    if measured_filter == "Measured only":
        filtered = filtered[selected_measured]
    elif measured_filter == "Unmeasured only":
        filtered = filtered[~selected_measured]
    if suspicious_filter == "Suspicious only":
        filtered = filtered[filtered["suspicious_measurement"].fillna(False).astype(bool)]
    elif suspicious_filter == "Missing any active feature":
        filtered = filtered[filtered[list(feature_qa.ACTIVE_FEATURES)].isna().any(axis=1)]
    variation_mode = {
        "All prompts": "all",
        "Prompts with variation only": "with_variation",
        "Prompts without variation": "without_variation",
    }[variation_filter]
    selected_bin_key = None if value_label == "All" else bin_labels[value_label]
    filtered = feature_support.apply_review_filter(
        filtered,
        selected_feature,
        bin_key=selected_bin_key,
        variation_mode=variation_mode,
        variation_reference=evidence,
        bin_reference=evidence,
    )
    if linked_filter and linked_filter.get("feature") == selected_feature:
        filtered = feature_support.apply_review_filter(
            filtered,
            selected_feature,
            bin_key=linked_filter.get("bin_key"),
            content_strength=linked_filter.get("content_strength"),
            bin_reference=evidence,
        )
    filtered = _order_validation_sample(filtered, sampling, selected_feature).reset_index(drop=True)
    if filtered.empty:
        st.warning("No source appearances match the active review filters.")
        return

    labels = {
        key: f"{row.source_root_domain} | {row.prompt_id} | {str(row.url_title)[:65]} | {row.normalized_url[-20:]}"
        for key, row in zip(filtered["qa_row_key"], filtered.itertuples(index=False))
    }
    keys = filtered["qa_row_key"].tolist()
    state_key = "feature_qa_selected_row"
    if st.session_state.get(state_key) not in keys:
        st.session_state[state_key] = keys[0]
    current = keys.index(st.session_state[state_key])
    navigation = st.columns([1, 1, 1, 3])
    if navigation[0].button("Previous", icon=":material/arrow_back:", width="stretch", disabled=len(keys) == 1):
        st.session_state[state_key] = keys[(current - 1) % len(keys)]
        st.rerun()
    if navigation[1].button("Next", icon=":material/arrow_forward:", width="stretch", disabled=len(keys) == 1):
        st.session_state[state_key] = keys[(current + 1) % len(keys)]
        st.rerun()
    if navigation[2].button("Random", icon=":material/shuffle:", width="stretch", disabled=len(keys) == 1):
        st.session_state[state_key] = random.choice([key for key in keys if key != keys[current]])
        st.rerun()
    selected_key = navigation[3].selectbox(
        f"Webpage ({len(filtered):,} matching appearances)",
        keys,
        format_func=lambda key: labels[key],
        key=state_key,
    )
    row = filtered[filtered["qa_row_key"].eq(selected_key)].iloc[0]
    source_url = str(row.get("source_url") or row["normalized_url"])

    left, right = st.columns([0.92, 1.38], gap="large")
    with left:
        st.markdown("#### Feature scores and metadata")
        metadata = pd.DataFrame(
            [
                ("Normalized URL", row["normalized_url"]),
                ("Page title", row.get("url_title")),
                ("Domain", row.get("source_root_domain")),
                ("Prompt ID", row.get("prompt_id")),
                ("Citation status", "Cited" if pd.to_numeric(pd.Series([row.get("cited")]), errors="coerce").iloc[0] == 1 else "More-only"),
                ("Extraction Strength", row.get("content_strength")),
                ("Page type", row.get("page_type_family_gemini_v1_collapsed")),
                ("Source type", row.get("source_type_general_gemini_v1_collapsed")),
                ("Scrape status", _binary_result(row.get("scrape_success"), "Success", "Failed")),
                ("Producer content scope", row.get("feature_extraction_text_scope")),
                ("Exact source field", row.get("authoritative_content_source")),
                (
                    "Secondary captured format",
                    "Markdown/body available"
                    if _not_missing(row.get("captured_markdown_or_body"))
                    else "Unavailable",
                ),
                (
                    "HTML-derived preview",
                    "Available" if _not_missing(row.get("sanitized_html_preview")) else "Unavailable",
                ),
            ],
            columns=["Field", "Value"],
        )
        st.dataframe(metadata, width="stretch", hide_index=True)

        feature_rows = [
            ("Measured Content Length (log2)", _display_measurement(row.get("log2_word_count_plus1"))),
            ("Original word count", _display_measurement(row.get("word_count"), digits=0)),
            ("Verified HTML Table Presence", feature_qa.nullable_binary_status(row.get("has_verified_html_table"))),
            ("Factual and Numeric Specificity Score", _display_measurement(row.get("factual_numeric_density_score"))),
            ("Answer-Oriented Writing Structure Score", _display_measurement(row.get("writing_structure_score"), digits=0)),
        ]
        st.markdown("##### Active model features")
        st.dataframe(pd.DataFrame(feature_rows, columns=["Feature", "Calculated value"]), width="stretch", hide_index=True)

        st.markdown("##### Writing-score components")
        component_rows = []
        for component in feature_qa.COMPONENTS:
            status = feature_qa.nullable_binary_status(row.get(component))
            contribution = "Unmeasured" if status == "Unmeasured" else ("+1" if status == "Detected" else "+0")
            component_rows.append((component, status, contribution))
        st.dataframe(
            pd.DataFrame(component_rows, columns=["Component", "Measurement", "Contribution"]),
            width="stretch",
            hide_index=True,
        )
        score = _display_measurement(row.get("writing_structure_score"), digits=0)
        component_sum = _display_measurement(row.get("writing_component_sum"), digits=0)
        st.code(f"writing_structure_score = {score}\nsum(governed components) = {component_sum}")
        score_match = row.get("writing_score_matches_components")
        if not _not_missing(score_match):
            st.warning("The score/component consistency check is unmeasured.")
        elif not bool(score_match):
            st.error("Stored score does not equal the governed component sum.")

        st.markdown("##### Factual/numeric-score components")
        factual_contributions = feature_qa.factual_component_contributions(pd.DataFrame([row]))
        contribution = factual_contributions.iloc[0]
        factual_rows = [
            (
                "number_token_per_1000_words",
                _display_measurement(row.get("number_token_per_1000_words")),
                _display_measurement(contribution["numeric_rate_contribution"]),
            ),
            (
                "I(percent_mention_count > 0)",
                _display_measurement(row.get("percent_mention_count"), digits=0),
                _display_measurement(contribution["percent_indicator_contribution"], digits=0),
            ),
            (
                "I(year_mention_count > 0)",
                _display_measurement(row.get("year_mention_count"), digits=0),
                _display_measurement(contribution["year_indicator_contribution"], digits=0),
            ),
            (
                "I(range_mention_count > 0)",
                _display_measurement(row.get("range_mention_count"), digits=0),
                _display_measurement(contribution["range_indicator_contribution"], digits=0),
            ),
            (
                "log1p(measurement_mention_count)",
                _display_measurement(row.get("measurement_mention_count"), digits=0),
                _display_measurement(contribution["measurement_log_contribution"]),
            ),
        ]
        st.dataframe(
            pd.DataFrame(factual_rows, columns=["Formula term", "Observed input", "Contribution"]),
            width="stretch",
            hide_index=True,
        )
        factual_score = _display_measurement(row.get("factual_numeric_density_score"))
        factual_sum = _display_measurement(row.get("factual_component_sum"))
        st.code(
            f"factual_numeric_density_score = {factual_score}\n"
            f"sum(governed contributions) = {factual_sum}"
        )
        factual_match = row.get("factual_score_matches_components")
        if not _not_missing(factual_match):
            st.warning("The factual-score consistency check is unmeasured.")
        elif not bool(factual_match):
            st.error("Stored factual score does not equal the governed contribution sum.")
        with st.expander("Exact governed formulas"):
            for feature in feature_qa.ACTIVE_FEATURES:
                st.markdown(f"**`{feature}`**")
                st.code(_feature_formula(feature), language="text")

    with right:
        st.markdown("#### Stored content review")
        st.info(
            f"Authoritative feature input: {row.get('authoritative_content_source')} "
            f"({row.get('feature_extraction_text_scope')}). The live page is not authoritative."
        )
        rendered, raw_text, opening_text, html_preview, original = st.tabs(
            [
                "Rendered Content",
                "Raw Extracted Text",
                "Opening Text",
                "HTML Preview",
                "Original Webpage",
            ]
        )
        authoritative = row.get("authoritative_feature_content")
        with rendered:
            highlighted = feature_qa.highlighted_content(authoritative, row)
            st.markdown(
                "<style>.feature-evidence{white-space:pre-wrap;line-height:1.6}.feature-evidence mark{padding:1px 3px;border-radius:2px}.list-evidence{background:#d9f2e6}.pattern-evidence{background:#fff0b8}.numeric-evidence{background:#dcecff}</style>"
                f'<div class="feature-evidence">{highlighted}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Highlights use the governed producer's list and writing-pattern terms. "
                "The historical producer did not retain character-level evidence spans."
            )
        with raw_text:
            st.caption(
                "Exact `url_text_for_features` value used by the writing/factual producer. "
                "This may include title and meta description appended by the governed assembly rule."
            )
            st.code("" if not _not_missing(authoritative) else str(authoritative), language="text", wrap_lines=True)
            secondary = row.get("captured_markdown_or_body")
            if _not_missing(secondary) and str(secondary).strip() and str(secondary) != str(authoritative):
                with st.expander("Captured Markdown/body (secondary, not used for these stored feature values)"):
                    st.code(str(secondary), language="markdown", wrap_lines=True)
        with opening_text:
            opening = row.get("opening_100_words")
            st.caption(
                "Configured window: first 100 producer tokens. The same exact window is used by "
                "`opening_has_summary_signal` and `opening_has_direct_answer_signal`."
            )
            with st.container(border=True):
                st.write("Unmeasured" if not _not_missing(opening) else str(opening))
            opening_components = pd.DataFrame(
                [
                    (
                        "opening_has_summary_signal",
                        feature_qa.nullable_binary_status(row.get("opening_has_summary_signal")),
                    ),
                    (
                        "opening_has_direct_answer_signal",
                        feature_qa.nullable_binary_status(
                            row.get("opening_has_direct_answer_signal")
                        ),
                    ),
                ],
                columns=["Opening feature", "Measurement"],
            )
            st.dataframe(opening_components, width="stretch", hide_index=True)
        with html_preview:
            sanitized = row.get("sanitized_html_preview")
            if not _not_missing(sanitized) or not str(sanitized).strip():
                st.info("No sanitized HTML preview is available for this stored page.")
            else:
                st.caption("Sanitized and script-free preview. Forms, event handlers, frames, and executable content were removed.")
                document = (
                    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src https: data:; style-src \'unsafe-inline\'">'
                    '<style>body{font-family:system-ui,sans-serif;line-height:1.5;padding:12px;color:#1f2937}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #d1d5db;padding:5px}</style>'
                    + str(sanitized)
                )
                with st.container(height=620, border=True):
                    st.html(document, unsafe_allow_javascript=False)
        with original:
            st.warning("The current webpage may differ from the stored scrape and may block iframe embedding.")
            _live_page_panel(source_url, "feature_qa_original", height=610)

    st.divider()
    st.markdown("#### Manual feature review")
    reviewable = list(dict.fromkeys([selected_feature, *feature_qa.ACTIVE_FEATURES, *feature_qa.COMPONENTS, "content_strength", "heading_count_group"]))
    review_file = Path(manifest["review_file"])
    with st.form(f"manual_feature_review_{qa.snapshot_key(source_url)}_{row.get('prompt_id')}"):
        form_columns = st.columns(3)
        reviewed_feature = form_columns[0].selectbox("Feature reviewed", reviewable)
        automated_value = row.get(reviewed_feature)
        if reviewed_feature in {"has_verified_html_table", *feature_qa.COMPONENTS}:
            automated_label = feature_qa.nullable_binary_status(automated_value)
        else:
            automated_label = _display_measurement(automated_value)
        form_columns[1].text_input("Automated value", value=automated_label, disabled=True)
        decision = form_columns[2].selectbox("Reviewer decision", ["correct", "incorrect", "uncertain", "cannot verify"])
        error_type = st.selectbox(
            "Error type",
            ["", "false positive", "false negative", "wrong text scope", "navigation/footer contamination",
             "excerpt too short", "formatting lost", "HTML unavailable",
             "missing incorrectly coded", "other"],
        )
        note = st.text_area("Reviewer note", height=90)
        submitted = st.form_submit_button("Save feature review", type="primary")
    if submitted:
        feature_qa.append_review(
            review_file,
            {
                "normalized_url": row["normalized_url"],
                "prompt_id": row["prompt_id"],
                "feature_name": reviewed_feature,
                "automated_value": "" if not _not_missing(automated_value) else automated_value,
                "reviewer_decision": decision,
                "error_type": error_type,
                "reviewer_note": note,
                "content_source_used": row.get("authoritative_content_source"),
                "feature_producer_version": feature_qa.producer_version(reviewed_feature),
            },
        )
        st.success("Review appended to the separate QA artifact. Feature and model values were not changed.")
    if review_file.exists():
        reviews = pd.read_csv(review_file, low_memory=False)
        st.caption(f"{len(reviews):,} feature-review annotations saved in `{review_file.name}`.")
        st.download_button(
            "Download feature-validation reviews",
            reviews.to_csv(index=False).encode("utf-8"),
            file_name=review_file.name,
            mime="text/csv",
            key="download_manual_feature_validation_reviews",
        )


def _feature_contribution(bundle: qa.QABundle) -> None:
    path = "tables/frontend/econometric_feature_ablation.csv"
    try:
        ablation = qa.load_model_table(bundle, path)
    except FileNotFoundError:
        st.warning("The feature-ablation artifact has not been generated yet.")
        st.code(".venv/bin/python scripts/v2_run_econometric_feature_ablation.py", language="bash")
        return

    labels = dict(ablation[["model_label", "model_family"]].drop_duplicates().itertuples(index=False))
    selected_label = st.selectbox("Model", list(labels), key="qa_ablation_model")
    selected = ablation[ablation["model_family"].eq(labels[selected_label])].copy()
    metric_options = {
        "R-squared gain": ("r_squared_gain", "R-squared gain"),
        "Partial R-squared": ("partial_r_squared", "Partial R-squared"),
        "RMSE reduction": ("rmse_reduction", "RMSE reduction"),
        "Brier-score reduction": ("brier_reduction", "Brier-score reduction"),
        "MAE reduction": ("mae_reduction", "MAE reduction"),
    }
    metric_label = st.selectbox("Comparison metric", list(metric_options), key="qa_ablation_metric")
    metric, axis_label = metric_options[metric_label]
    selected = selected.sort_values(metric, ascending=True)

    best = selected.loc[selected[metric].idxmax()]
    C.metric_cards(
        [
            {"value": f"{selected['n_obs'].iloc[0]:,.0f}", "label": "model rows"},
            {"value": f"{selected['n_prompts'].iloc[0]:,.0f}", "label": "prompts"},
            {"value": f"{selected['with_r_squared'].iloc[0]:.3f}", "label": "full-model R-squared"},
            {"value": str(best["feature_label"]), "label": f"largest {metric_label.lower()}"},
        ]
    )
    fig = px.bar(
        selected,
        x=metric,
        y="feature_label",
        orientation="h",
        text=metric,
        color=metric,
        color_continuous_scale="RdYlGn",
        title=f"With-feature improvement over the model without that feature: {metric_label}",
    )
    fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig.update_layout(
        xaxis_title=axis_label,
        yaxis_title="",
        coloraxis_showscale=False,
        height=max(390, 62 * len(selected)),
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#77808f")
    st.plotly_chart(fig, width="stretch")

    detail_options = selected.sort_values(metric, ascending=False)
    feature_labels = dict(detail_options[["feature_label", "feature"]].itertuples(index=False))
    detail_label = st.selectbox("Inspect feature", list(feature_labels), key="qa_ablation_feature")
    detail = selected[selected["feature"].eq(feature_labels[detail_label])].iloc[0]
    comparison = pd.DataFrame(
        {
            "model": ["Without feature", "With feature"],
            "R-squared": [detail["without_r_squared"], detail["with_r_squared"]],
            "RMSE": [detail["without_rmse"], detail["with_rmse"]],
            "Brier score": [detail["without_brier"], detail["with_brier"]],
            "MAE": [detail["without_mae"], detail["with_mae"]],
        }
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
    st.caption(
        f"Nested-model F-test p-value: {detail['nested_f_p_value']:.3g}. "
        "This conventional nested OLS test is supplementary; robust coefficient uncertainty remains in the notebook tables."
    )
    with st.expander("All feature-ablation results and formulas"):
        st.dataframe(selected, width="stretch", hide_index=True)
    st.info(
        "Positive values mean the full model fits this same sample better after the feature is included. "
        "These are in-sample nested-model diagnostics, not causal effects or out-of-sample prediction guarantees."
    )


def _feature_registry(bundle: qa.QABundle) -> None:
    path = "tables/core_general_content_feature_dictionary.csv"
    try:
        registry = qa.load_model_table(bundle, path)
    except FileNotFoundError:
        st.warning("The Core-General feature registry has not been generated yet.")
        st.code(
            ".venv/bin/python scripts/v2_build_core_general_feature_registry.py",
            language="bash",
        )
        return

    C.metric_cards(
        [
            {"value": f"{len(registry):,}", "label": "registry entries"},
            {
                "value": f"{registry['feature_layer'].eq('core_general').sum():,}",
                "label": "Core-General",
            },
            {
                "value": f"{registry['feature_layer'].eq('commerce_general').sum():,}",
                "label": "Commerce-General",
            },
            {
                "value": f"{registry['feature_status'].eq('exclude_leakage').sum():,}",
                "label": "leakage exclusions",
            },
        ]
    )
    st.caption(
        "This is a pre-estimation specification registry. Status and formulas are frozen before "
        "examining outcomes; inclusion here does not mean a feature is ready for a headline model."
    )

    layer_options = sorted(registry["feature_layer"].dropna().astype(str).unique())
    status_options = sorted(registry["feature_status"].dropna().astype(str).unique())
    role_options = sorted(registry["recommended_model_role"].dropna().astype(str).unique())
    implementation_options = sorted(
        registry["current_implementation_status"].dropna().astype(str).unique()
    )
    qa_options = sorted(registry["qa_status"].dropna().astype(str).unique())
    approval_options = sorted(registry["approved_for_model_v1"].astype(str).unique())
    granularity_options = sorted(
        registry["feature_granularity"].dropna().astype(str).unique()
    )
    record_type_options = sorted(
        registry["registry_record_type"].dropna().astype(str).unique()
    )
    c1, c2 = st.columns(2)
    with c1:
        layers = st.multiselect(
            "Feature layer",
            layer_options,
            default=layer_options,
            key="qa_feature_registry_layer",
        )
        roles = st.multiselect(
            "Model role",
            role_options,
            default=role_options,
            key="qa_feature_registry_role",
        )
    with c2:
        statuses = st.multiselect(
            "Feature status",
            status_options,
            default=status_options,
            key="qa_feature_registry_status",
        )
        implementation = st.multiselect(
            "Implementation status",
            implementation_options,
            default=implementation_options,
            key="qa_feature_registry_implementation",
        )
        qa_statuses = st.multiselect(
            "QA status",
            qa_options,
            default=qa_options,
            key="qa_feature_registry_qa_status",
        )
        approvals = st.multiselect(
            "Approved for model v1",
            approval_options,
            default=approval_options,
            key="qa_feature_registry_approval",
        )
        granularities = st.multiselect(
            "Feature granularity",
            granularity_options,
            default=granularity_options,
            key="qa_feature_registry_granularity",
        )
        record_types = st.multiselect(
            "Registry record type",
            record_type_options,
            default=record_type_options,
            key="qa_feature_registry_record_type",
        )
    search = st.text_input(
        "Search feature specification",
        placeholder="table, relevance, extraction, real estate...",
        key="qa_feature_registry_search",
    ).strip()

    filtered = registry[
        registry["feature_layer"].isin(layers)
        & registry["feature_status"].isin(statuses)
        & registry["recommended_model_role"].isin(roles)
        & registry["current_implementation_status"].isin(implementation)
        & registry["qa_status"].isin(qa_statuses)
        & registry["approved_for_model_v1"].astype(str).isin(approvals)
        & registry["feature_granularity"].isin(granularities)
        & registry["registry_record_type"].isin(record_types)
    ].copy()
    if search:
        searchable = filtered.astype(str).agg(" ".join, axis=1)
        filtered = filtered[searchable.str.contains(search, case=False, regex=False)]
    filtered = filtered.sort_values(
        ["feature_layer", "feature_group", "feature_name"],
        kind="stable",
    )

    st.caption(f"Showing {len(filtered):,} of {len(registry):,} registry entries.")
    display_columns = [
        "feature_name",
        "feature_layer",
        "feature_group",
        "feature_granularity",
        "registry_record_type",
        "replacement_feature_name",
        "primitive_or_composite",
        "definition",
        "source_provenance",
        "recommended_model_role",
        "feature_status",
        "current_implementation_status",
        "qa_status",
        "approved_for_model_v1",
        "model_entry_blocker",
        "page_aggregation_rule",
        "leakage_status",
        "validation_requirement",
    ]
    st.dataframe(
        filtered[display_columns],
        width="stretch",
        hide_index=True,
        height=560,
    )
    st.download_button(
        "Download filtered feature registry",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="core_general_content_feature_dictionary_filtered.csv",
        mime="text/csv",
        key="qa_feature_registry_download",
    )
    with st.expander("Full registry columns and formulas"):
        st.dataframe(filtered, width="stretch", hide_index=True, height=560)


def _model_tables(bundle: qa.QABundle) -> None:
    artifacts = [
        (
            "Notebook 09 minimum reporting table",
            "tables/09_content_feature_econometrics/09_minimum_reporting_table.csv",
        ),
        (
            "Notebook 09 interpretation patch",
            "tables/09_content_feature_econometrics/interp_patch/09_minimum_reporting_table_v2_interpretation_patch.csv",
        ),
        (
            "Notebook 09 robustness classification",
            "tables/09_content_feature_econometrics/interp_patch/focal_feature_robustness_classification.csv",
        ),
        (
            "Notebook 11 minimum reporting table",
            "tables/11_writing_factual_density_econometrics/11_minimum_reporting_table.csv",
        ),
        (
            "Notebook 11 robustness classification",
            "tables/11_writing_factual_density_econometrics/11_writing_factual_robustness_classification.csv",
        ),
    ]
    for title, path in artifacts:
        with st.expander(title, expanded="interpretation patch" in title.lower()):
            try:
                st.dataframe(qa.load_model_table(bundle, path), width="stretch", hide_index=True)
            except FileNotFoundError:
                st.warning(f"Missing model artifact: {path}")
    with st.expander("Main model rules"):
        for rule in bundle.manifest["main_model_rules"]:
            st.markdown(f"- {rule}")


def _reviews() -> None:
    C.section("Manual reviews", "Annotations are local and separate from analytical datasets.")
    reviews = pd.DataFrame(storage.list_econometrics_reviews())
    if reviews.empty:
        st.info("No pages have been manually reviewed yet.")
        return
    st.dataframe(
        reviews,
        width="stretch",
        hide_index=True,
        column_config={"source_url": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    st.download_button(
        "Export reviews as CSV",
        reviews.to_csv(index=False).encode("utf-8"),
        file_name="econometrics_manual_reviews.csv",
        mime="text/csv",
    )
