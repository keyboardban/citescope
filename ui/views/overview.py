"""Overview: pipeline diagram, headline metrics, and the black-box framing."""

from __future__ import annotations

import streamlit as st

from src.analysis import _normalize_recall, summary_metrics

from .. import components as C
from .. import charts
from ..state import get_run


def _counts(run: dict | None) -> dict:
    if not run:
        return {"prompt": 1}
    g = run.get("gemini") or {}
    serp = run.get("serp") or {}
    scrape = run.get("scrape") or {}
    matching = run.get("matching") or {}
    feats = run.get("features") or []
    return {
        "prompt": 1,
        "queries": len(g.get("search_queries", []) or []),
        "citations": matching.get("n_citations", len(g.get("citations", []) or [])),
        "candidates": len(feats) or len(serp.get("candidates", []) or []),
        "scraped": sum(1 for p in (scrape.get("pages") or {}).values() if p.get("status") == "success"),
        "matched": len(matching.get("cited_candidate_ids", []) or []),
        "features": len(feats),
    }


def render() -> None:
    run = get_run()
    C.hero(
        "AI Search Citation Audit",
        "A black-box observational audit: compare websites Gemini cites against an "
        "independently reconstructed SERP for the same search queries.",
    )

    if run:
        tag = "🧪 demo run" if run.get("is_demo") else "live run"
        st.caption(f"Current run: `{run.get('run_id','')}` · {tag} · {run.get('created_at','')}")
    else:
        st.caption("No run yet — go to **Run AI Search**, or load the demo run from the sidebar.")

    C.section("Pipeline", "Each prompt flows through these observable stages.", "🧭")
    C.pipeline_diagram(_counts(run))

    m = summary_metrics(run) if run else {}
    C.section("Key metrics", icon="📊")
    C.metric_cards([
        {"value": m.get("n_queries", 0), "label": "search queries", "sub": "observed"},
        {"value": m.get("n_citations", 0), "label": "citations", "sub": "distinct URLs"},
        {"value": m.get("n_candidates", 0), "label": "SERP candidates", "sub": "reconstructed"},
        {"value": m.get("n_scraped", 0), "label": "pages scraped"},
        {"value": C.pct(m.get("recall_strict_10")) if run else "—", "label": "strict recall@10"},
        {"value": C.pct(m.get("recall_domain_10")) if run else "—", "label": "domain-incl@10"},
    ])

    if run and (run.get("matching") or {}).get("recall"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(charts.recall_grouped(_normalize_recall(run["matching"]["recall"])), width="stretch")
        with col2:
            rc = run["matching"].get("rate_counts") or {}
            st.plotly_chart(charts.match_type_bar(rc), width="stretch")

    C.section("What this measures (and what it doesn't)", icon="🔬")
    C.limitation_box(long=True)
    C.glossary_expander()

    if not run:
        st.divider()
        if st.button("🧪 Load demo run (no API calls)", type="primary"):
            from src.demo import make_demo_run
            from ..state import set_run
            set_run(make_demo_run())
            st.rerun()
