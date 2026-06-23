"""Feature Analysis: cited vs non-cited comparison, charts, and insights."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.analysis import (
    FEATURE_LABELS,
    correlation_with_citation,
    features_df,
    group_compare,
    official_compare,
    source_breakdown,
)
from src.features import NUMERIC_FEATURES

from .. import charts
from .. import components as C
from ..state import get_clients, get_run, recompute_downstream

SIM_KEYS = ["title_query_sim", "snippet_query_sim", "page_query_sim",
            "page_output_sim", "max_chunk_output_sim", "max_chunk_query_sim"]


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _insights(df: pd.DataFrame, gc_records: list[dict], off: dict) -> list[str]:
    gc = {r["key"]: r for r in gc_records}
    out: list[str] = []

    r = gc.get("serp_rank", {})
    if r.get("cited_mean") is not None and r.get("noncited_mean") is not None:
        better = "higher" if r["cited_mean"] < r["noncited_mean"] else "lower"
        out.append(f"Cited sites sat **{better}** in the reconstructed SERP on average "
                   f"({r['cited_mean']:.1f} vs {r['noncited_mean']:.1f} mean rank).")

    p = gc.get("page_output_sim", {})
    if p.get("delta") is not None:
        comp = "higher" if p["delta"] > 0 else "comparable/lower"
        out.append(f"Cited pages showed **{comp}** page–answer similarity (proxy): "
                   f"{_fmt(p.get('cited_mean'))} vs {_fmt(p.get('noncited_mean'))}.")

    if off:
        o, n = off.get("official", {}), off.get("non_official", {})
        out.append(f"Official-source cite-rate **{o.get('cite_rate',0)*100:.0f}%** (n={o.get('candidates',0)}) "
                   f"vs non-official **{n.get('cite_rate',0)*100:.0f}%** (n={n.get('candidates',0)}).")

    if "page_output_sim" in df.columns:
        nc = df[df["cited"] == 0]
        if nc["page_output_sim"].notna().any():
            mx = nc["page_output_sim"].max()
            if mx and mx > 0.3:
                dom = nc.loc[nc["page_output_sim"].idxmax(), "domain"]
                out.append(f"Some **non-cited** candidates still had high answer overlap "
                           f"(max {mx:.2f}, e.g. `{dom}`) — high overlap ≠ citation.")
    return out


def render() -> None:
    run = get_run()
    C.section("Feature Analysis",
              "Compare cited websites against non-cited reconstructed SERP candidates.", "📈")
    if not (run and run.get("features")):
        C.empty_state("Run Gemini + reconstruct a SERP to build the feature table.", "📈")
        return

    inputs = st.session_state["inputs"]
    a = inputs["analysis"]
    clients = get_clients()
    cc1, cc2 = st.columns([3, 1])
    methods = config.SIMILARITY_METHODS
    a["similarity_method"] = cc1.selectbox(
        "Similarity method", methods,
        index=methods.index(a["similarity_method"]) if a["similarity_method"] in methods else 0,
        help="Lexical = offline bag-of-words. Embeddings = Gemini vectors (uses API quota).")
    if cc2.button("↻ Recompute", width="stretch"):
        recompute_downstream()
        st.rerun()
    if "embed" in a["similarity_method"].lower() and clients.get("gemini") is None:
        st.warning("Embeddings need `GEMINI_API_KEY`. Falls back to lexical similarity.")

    df = features_df(run["features"])
    an = run.get("analysis") or {}
    m = an.get("summary", {})
    C.metric_cards([
        {"value": m.get("n_candidates", len(df)), "label": "candidates"},
        {"value": m.get("n_cited_candidates", int(df["cited"].sum())), "label": "cited"},
        {"value": m.get("n_scraped", 0), "label": "scraped"},
        {"value": C.pct(m.get("recall_10")), "label": "recall@10"},
    ])
    C.proxy_note("All comparisons are cited vs non-cited reconstructed candidates. Differences are "
                 "observable associations, not causal explanations.")

    gc_records = an.get("group_compare") or group_compare(df).to_dict(orient="records")
    st.plotly_chart(charts.grouped_means(gc_records, SIM_KEYS), width="stretch")

    col1, col2 = st.columns([3, 2])
    with col1:
        gdf = pd.DataFrame(gc_records)
        if not gdf.empty:
            st.dataframe(gdf[["feature", "cited_mean", "noncited_mean", "delta", "n_cited", "n_noncited"]],
                         width="stretch", hide_index=True)
    with col2:
        st.plotly_chart(charts.rank_box(df), width="stretch")

    C.section("Distribution by feature", icon="📦")
    feat_opts = {FEATURE_LABELS.get(k, k): k for k in NUMERIC_FEATURES
                 if k in df.columns and df[k].notna().any()}
    if feat_opts:
        chosen = st.selectbox("Feature", list(feat_opts))
        st.plotly_chart(charts.distribution_box(df, feat_opts[chosen], chosen), width="stretch")

    C.section("Source types", icon="🗂️")
    sb = source_breakdown(df)
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(charts.source_stacked(sb), width="stretch")
    with s2:
        st.plotly_chart(charts.cite_rate_by_source(sb), width="stretch")
    if not sb.empty:
        st.dataframe(sb, width="stretch", hide_index=True)

    off = an.get("official") or official_compare(df)
    if off:
        C.section("Official vs non-official", icon="🏛️")
        st.dataframe(pd.DataFrame([{"group": k, **v} for k, v in off.items()]),
                     width="stretch", hide_index=True)

    C.section("Feature ↔ citation correlation", "Point-biserial-style correlation with the cited label.", "📐")
    corr = correlation_with_citation(df)
    if not corr.empty:
        st.dataframe(corr, width="stretch", hide_index=True)

    C.section("Feature heatmap", icon="🌡️")
    st.plotly_chart(charts.feature_heatmap(df), width="stretch")

    C.section("Query → candidate → citation flow", icon="🔀")
    st.plotly_chart(charts.citation_sankey(run), width="stretch")

    C.section("Observable patterns (read carefully)", icon="💡")
    bullets = _insights(df, gc_records, off)
    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    C.proxy_note("These describe associations in this single run's observable data. They do not "
                 "reveal the AI's internal retrieval or citation mechanism.")

    with st.expander("🗂️ Candidate cards"):
        rows = sorted(run["features"], key=lambda x: x.get("serp_rank", 999))[:12]
        cols = st.columns(2)
        for i, row in enumerate(rows):
            with cols[i % 2]:
                C.site_card(row)
