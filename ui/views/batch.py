"""Batch Mode: run multiple prompts and view aggregated observable associations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import batch as batch_mod
from src import config, report

from .. import charts
from .. import components as C
from ..state import get_clients

_SIM_KEYS = ["title_query_sim", "snippet_query_sim", "page_query_sim",
             "page_output_sim", "max_chunk_output_sim"]

_DEFAULT_PROMPTS = (
    "What are the best tailors in Bangkok for custom suits?\n"
    "Best running shoes for flat feet?\n"
    "Top project management tools for small teams?"
)


def render() -> None:
    C.section("Batch Mode",
              "Run multiple prompts and aggregate cited vs non-cited associations across runs.", "📚")
    C.caveat_box(config.CAVEAT_BATCH)

    ss = st.session_state
    text = st.text_area("Prompts (one per line)", value=ss.get("batch_prompts_text", _DEFAULT_PROMPTS),
                        height=160, key="batch_prompts_text")
    prompts = [p.strip() for p in text.splitlines() if p.strip()]
    use_cache = st.toggle("Use cached results when available", value=True)
    st.caption(f"{len(prompts)} prompt(s). Each runs the full pipeline (Gemini + Apify SERP + scrape) — "
               "this can take a while and consumes API credits.")

    clients = get_clients()
    ready = clients.get("gemini") is not None and clients.get("apify") is not None
    if st.button("⚡ Run batch", type="primary", disabled=not ready or not prompts, width="stretch"):
        bar = st.progress(0.0, text="Starting…")
        try:
            b = batch_mod.run_batch(clients, prompts, ss["inputs"],
                                    progress=lambda s, f: bar.progress(min(f, 1.0), text=s),
                                    use_cache=use_cache)
        except Exception as exc:  # keep UI alive
            bar.empty()
            st.error(f"Batch failed: {type(exc).__name__}: {exc}")
            return
        bar.progress(1.0, text="Done")
        ss["batch"] = b
        ok = b["aggregate"]["sample_sizes"]["n_runs_ok"]
        st.success(f"Batch complete — {ok}/{len(prompts)} runs succeeded.")
    if not ready:
        st.warning("Batch mode needs both `GEMINI_API_KEY` and `APIFY_TOKEN` (it runs real audits).")

    b = ss.get("batch")
    if not b:
        C.empty_state("No batch yet. Add prompts and click **Run batch**.", "📚")
        return
    _show_batch(b)


def _show_batch(b: dict) -> None:
    agg = b.get("aggregate") or {}
    s = agg.get("sample_sizes") or {}

    C.section("Sample sizes", icon="📏")
    C.metric_cards([
        {"value": b.get("n_prompts", 0), "label": "prompts"},
        {"value": s.get("n_runs_ok", 0), "label": "runs ok"},
        {"value": s.get("n_candidates", 0), "label": "candidates"},
        {"value": s.get("n_cited", 0), "label": "cited", "sub": "strong"},
        {"value": s.get("n_citations", 0), "label": "citations"},
        {"value": s.get("n_scraped", 0), "label": "scraped"},
    ])

    pp = pd.DataFrame(b.get("per_prompt") or [])
    if not pp.empty:
        with st.expander("Per-prompt results"):
            st.dataframe(pp, width="stretch", hide_index=True)

    C.section("Recall@K (averaged across runs)", icon="📊")
    st.plotly_chart(charts.recall_grouped(agg.get("recall") or {}), width="stretch")

    C.section("Cited vs non-cited (pooled across prompts)",
              "Median + Mann-Whitney U p-value + 95% bootstrap CI of the median difference.", "📈")
    gs = pd.DataFrame(agg.get("group_stats") or [])
    if not gs.empty:
        st.plotly_chart(charts.grouped_means(agg["group_stats"], _SIM_KEYS), width="stretch")
        keep = ["feature", "phase", "cited_median", "noncited_median", "median_diff",
                "mwu_p", "ci_low", "ci_high", "n_cited", "n_noncited"]
        st.dataframe(gs[[c for c in keep if c in gs.columns]], width="stretch", hide_index=True)
        st.caption("`mwu_p` < 0.05 indicates the cited/non-cited difference is unlikely under no effect "
                   "— still an observational association, not causal. Pre-answer features are the cleaner signals.")

    sb = pd.DataFrame(agg.get("source_breakdown") or [])
    if not sb.empty:
        C.section("Source-type breakdown (pooled)", icon="🗂️")
        st.plotly_chart(charts.source_stacked(sb), width="stretch")
        st.dataframe(sb, width="stretch", hide_index=True)

    C.section("Export", icon="📤")
    c = st.columns(2)
    c[0].download_button("⬇️ Batch report (Markdown)", report.batch_markdown_report(b),
                         f"{b['batch_id']}_report.md", "text/markdown", width="stretch")
    c[1].download_button("⬇️ Pooled features (CSV)", report.batch_features_csv(b),
                         f"{b['batch_id']}_features.csv", "text/csv", width="stretch")
