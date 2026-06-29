"""Feature Analysis: cited vs non-cited, split into pre-answer vs post-output."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.analysis import (
    FEATURE_LABELS,
    POST_OUTPUT_FEATURES,
    PRE_ANSWER_FEATURES,
    correlation_with_citation,
    features_df,
    group_compare,
    length_sim_correlation,
    official_compare,
    source_breakdown,
)

from .. import charts
from .. import components as C
from ..state import get_clients, get_run, recompute_downstream

PRE_SIM_KEYS = ["title_query_sim", "snippet_query_sim", "page_query_sim", "max_chunk_query_sim"]
POST_KEYS = POST_OUTPUT_FEATURES


def _fmt(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _insights(gc_records: list[dict], off: dict, df: pd.DataFrame) -> list[str]:
    gc = {r["key"]: r for r in gc_records}
    out: list[str] = []

    r = gc.get("serp_rank", {})
    if r.get("cited_median") is not None and r.get("noncited_median") is not None:
        better = "higher" if r["cited_median"] < r["noncited_median"] else "lower/similar"
        out.append(f"**[pre-answer]** Cited sites sat **{better}** in the reconstructed SERP "
                   f"(median rank {r['cited_median']} vs {r['noncited_median']}).")
    pq = gc.get("page_query_sim", {})
    if pq.get("delta") is not None:
        comp = "higher" if pq["delta"] > 0 else "comparable/lower"
        out.append(f"**[pre-answer]** Cited pages had **{comp}** page–query similarity "
                   f"({_fmt(pq.get('cited_mean'))} vs {_fmt(pq.get('noncited_mean'))}).")
    if off:
        inst = off.get("institutional_official", {})
        brand = off.get("brand_official_candidate", {})
        other = off.get("other", {})
        out.append(f"**[pre-answer]** Cite-rate — institutional {inst.get('cite_rate',0)*100:.0f}% "
                   f"(n={inst.get('candidates',0)}), brand-candidate {brand.get('cite_rate',0)*100:.0f}% "
                   f"(n={brand.get('candidates',0)}), other {other.get('cite_rate',0)*100:.0f}% "
                   f"(n={other.get('candidates',0)}).")
    po = gc.get("page_output_sim", {})
    if po.get("delta") is not None:
        out.append(f"**[post-output · circular]** Cited pages showed higher page–answer overlap "
                   f"({_fmt(po.get('cited_mean'))} vs {_fmt(po.get('noncited_mean'))}) — expected if the "
                   "answer was generated from them; not independent evidence.")
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
        help="Lexical = offline bag-of-words. Embeddings = Gemini vectors (cached persistently).")
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
        {"value": m.get("n_cited_candidates", int(df["cited"].sum())), "label": "cited", "sub": "strong"},
        {"value": m.get("n_weak_candidates", 0), "label": "weak domain", "sub": "not cited"},
        {"value": m.get("n_scraped", 0), "label": "scraped"},
        {"value": C.pct(m.get("recall_strict_10")), "label": "strict recall@10"},
    ])
    C.proxy_note("All comparisons are cited vs non-cited reconstructed candidates. Differences are "
                 "observable associations, not causal explanations.")

    gc_records = an.get("group_compare") or group_compare(df).to_dict(orient="records")
    gcols = ["feature", "cited_mean", "noncited_mean", "cited_median", "noncited_median", "delta"]
    gdf = pd.DataFrame(gc_records)

    # ---- Pre-answer signals (non-circular) ----
    C.section("Pre-answer signals (non-circular)",
              "Observable before the answer exists — the cleaner signals.", "✅")
    p1, p2 = st.columns([3, 2])
    with p1:
        st.plotly_chart(charts.grouped_means(gc_records, PRE_SIM_KEYS), width="stretch")
    with p2:
        st.plotly_chart(charts.rank_box(df), width="stretch")
    if not gdf.empty:
        st.dataframe(gdf[gdf["phase"] == "pre_answer"][gcols], width="stretch", hide_index=True)

    # ---- Post-output overlap (circular) ----
    C.section("Post-output semantic overlap (may be partly circular)", icon="🌀")
    C.caveat_box(config.CAVEAT_POST_OUTPUT)
    pp1, pp2 = st.columns([2, 3])
    with pp1:
        st.plotly_chart(charts.grouped_means(gc_records, POST_KEYS), width="stretch")
    with pp2:
        st.plotly_chart(charts.length_vs_sim_scatter(df), width="stretch")
    lc = an.get("length_sim_corr") or length_sim_correlation(df)
    if lc:
        corr_txt = ", ".join(f"{k}={'—' if v is None else v}" for k, v in lc.items())
        st.caption(f"Length ↔ page–answer similarity correlation: {corr_txt}. {config.CAVEAT_LENGTH}")
    if not gdf.empty:
        st.dataframe(gdf[gdf["phase"] == "post_output"][gcols], width="stretch", hide_index=True)

    # ---- Distribution explorer ----
    C.section("Distribution by feature", icon="📦")
    feat_opts = {FEATURE_LABELS.get(k, k): k for k in (PRE_ANSWER_FEATURES + POST_OUTPUT_FEATURES)
                 if k in df.columns and df[k].notna().any()}
    if feat_opts:
        chosen = st.selectbox("Feature", list(feat_opts))
        st.plotly_chart(charts.distribution_box(df, feat_opts[chosen], chosen), width="stretch")

    # ---- Source types ----
    C.section("Source types", icon="🗂️")
    sb = source_breakdown(df)
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(charts.source_stacked(sb), width="stretch")
    with s2:
        st.plotly_chart(charts.cite_rate_by_source(sb), width="stretch")
    if not sb.empty:
        st.dataframe(sb, width="stretch", hide_index=True)

    # ---- Official signals ----
    off = an.get("official") or official_compare(df)
    if off:
        C.section("Official signals (institutional vs brand-candidate)", icon="🏛️")
        o1, o2 = st.columns([2, 3])
        with o1:
            st.plotly_chart(charts.official_bar(off), width="stretch")
        with o2:
            st.dataframe(pd.DataFrame([{"group": k, **v} for k, v in off.items()]),
                         width="stretch", hide_index=True)
        st.caption("`institutional` = .gov/.edu/.mil/.int. `brand candidate` = looks like the entity's "
                   "own site (heuristic, lower confidence).")

    # ---- Position-adjusted citation model (the rigorous headline) ----
    C.section("Position-adjusted citation model",
              "Multivariate LPM: Δ probability of citation per feature, holding others (incl. rank) fixed.", "📐")
    C.regression_block(an.get("regression"))

    # ---- Unadjusted correlation (quick descriptive, kept beneath) ----
    C.section("Feature ↔ citation correlation (unadjusted)", "Point-biserial-style; no controls; small n is noisy.", "📈")
    corr = correlation_with_citation(df)
    if not corr.empty:
        st.dataframe(corr, width="stretch", hide_index=True)

    C.section("Feature heatmap", icon="🌡️")
    st.plotly_chart(charts.feature_heatmap(df), width="stretch")

    C.section("Query → candidate → citation flow", icon="🔀")
    st.plotly_chart(charts.citation_sankey(run), width="stretch")

    # ---- Insights ----
    C.section("Observable patterns (read carefully)", icon="💡")
    for b in _insights(gc_records, off, df):
        st.markdown(f"- {b}")
    C.proxy_note("These describe associations in this single run. Pre-answer signals are prioritized; "
                 "post-output overlap is flagged as potentially circular. Use Batch mode for aggregates.")

    with st.expander("🗂️ Candidate cards"):
        rows = sorted(run["features"], key=lambda x: x.get("serp_rank", 999))[:12]
        cols = st.columns(2)
        for i, row in enumerate(rows):
            with cols[i % 2]:
                C.site_card(row)
