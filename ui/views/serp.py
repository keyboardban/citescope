"""SERP Reconstruction: run Apify SERP actor for the observed queries."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.analysis import features_df
from src.pipeline import stage_serp
from src.url_utils import domain, normalize_url

from .. import charts
from .. import components as C
from ..state import get_clients, get_run, recompute_downstream, set_run


def render() -> None:
    run = get_run()
    C.section("SERP Reconstruction",
              "Independently fetch Google results (Apify) for the observed queries — your candidate set.", "🌐")
    if not (run and run.get("gemini")):
        C.empty_state("Run a grounded Gemini query first (Run AI Search), then reconstruct the SERP.", "🌐")
        return

    C.proxy_note("The reconstructed SERP is a parallel candidate set — not the AI's internal results.")
    inputs = st.session_state["inputs"]
    s = inputs["serp"]

    observed = [q["query"] for q in run["gemini"].get("search_queries", []) or []]
    g = run["gemini"]
    if g.get("error") or not (g.get("output_text") or observed or g.get("citations")):
        st.warning(
            "The Gemini run produced no usable output or queries"
            + (f" — {g['error']}" if g.get("error") else "")
            + ". You can still add **fallback queries** manually below to reconstruct a SERP."
        )
    st.session_state.setdefault("manual_queries", [])

    cc1, cc2 = st.columns([3, 1])
    mq = cc1.text_input("Add a manual / fallback query",
                        placeholder=(run["inputs"]["prompt"] if not observed else "extra query…"))
    if cc2.button("➕ Add", width="stretch") and mq.strip():
        st.session_state.manual_queries.append(mq.strip())
        st.rerun()

    options = list(dict.fromkeys(observed + st.session_state.manual_queries)) or [run["inputs"]["prompt"]]
    default = observed if observed else options
    selected = st.multiselect("Queries to reconstruct", options, default=default)
    if not observed:
        C.proxy_note("Gemini exposed no queries — these are clearly-marked fallback queries.")
    s["selected_queries"] = selected

    c1, c2, c3 = st.columns(3)
    ks = [10, 20, 30, 50]
    s["top_k"] = c1.select_slider("Top-K results", ks, value=s["top_k"] if s["top_k"] in ks else 20)
    s["country"] = c2.text_input("Country code", s["country"], max_chars=2)
    s["language"] = c3.text_input("Language code", s["language"], max_chars=5)

    clients = get_clients()
    ready = clients.get("apify") is not None
    if st.button("🌐 Reconstruct SERP", type="primary", disabled=not ready or not selected):
        with st.spinner("Running Apify SERP actor…"):
            serp = stage_serp(clients["apify"], selected, s, run.get("run_id", ""),
                              use_cache=inputs["scrape"]["use_cache"])
        run["serp"] = serp
        set_run(run)
        if serp.get("error") and not serp.get("candidates"):
            st.error(f"SERP error: {serp['error']}")
        else:
            recompute_downstream()
            st.success(f"Reconstructed {len(serp.get('candidates', []))} candidate rows"
                       + (" (from cache)" if serp.get("cached") else ""))
    if not ready:
        st.warning("Set `APIFY_TOKEN` in `.env` to run the SERP actor.")

    serp = run.get("serp")
    if not serp or not serp.get("candidates"):
        C.empty_state("No reconstructed SERP yet. Pick queries and click **Reconstruct SERP**.", "🌐")
        return

    st.caption(
        f"actor `{serp.get('actor','')}` · run `{serp.get('run_id')}` · "
        f"dataset `{serp.get('dataset_id')}` · status `{serp.get('status')}`"
    )

    cited_set = {normalize_url(r["url"]) for r in (run.get("features") or []) if r.get("cited")}
    cands = serp["candidates"]
    df = pd.DataFrame([{
        "cited": normalize_url(c["url"]) in cited_set,
        "rank": c["rank"], "query": c["query"], "title": c["title"],
        "domain": domain(c["url"]), "snippet": c["snippet"], "url": c["url"],
    } for c in cands])
    st.dataframe(
        df, width="stretch", hide_index=True,
        column_config={
            "cited": st.column_config.CheckboxColumn("cited", help="Matched to a Gemini citation"),
            "url": st.column_config.LinkColumn("url"),
            "snippet": st.column_config.TextColumn("snippet", width="medium"),
        },
    )

    if run.get("features"):
        st.plotly_chart(charts.rank_box(features_df(run["features"])), width="stretch")
