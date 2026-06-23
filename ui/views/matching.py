"""Citation Matching: tiered matches, recall@K, and match-type breakdown."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.matching import match_all, unique_candidates

from .. import charts
from .. import components as C
from ..state import get_run, recompute_downstream, set_run


def render() -> None:
    run = get_run()
    C.section("Citation Matching",
              "Link cited URLs to reconstructed SERP candidates with tiered rules, then measure recall@K.", "🎯")
    if not (run and run.get("gemini") and run.get("serp") and run["serp"].get("candidates")):
        C.empty_state("Need a Gemini run and a reconstructed SERP first.", "🎯")
        return

    if not run.get("matching"):
        recompute_downstream()
        run = get_run()

    inputs = st.session_state["inputs"]
    a = inputs["analysis"]
    new_weak = st.toggle("Count domain-only (weak) matches as cited",
                         value=a["include_weak"],
                         help="Weak matches share only the registrable domain. Off by default.")
    if new_weak != a["include_weak"]:
        a["include_weak"] = new_weak
        recompute_downstream()
        st.rerun()

    matching = run["matching"]
    recall = matching.get("recall", {})
    C.metric_cards([
        {"value": C.pct(recall.get("5")), "label": "recall@5"},
        {"value": C.pct(recall.get("10")), "label": "recall@10"},
        {"value": C.pct(recall.get("20")), "label": "recall@20"},
        {"value": C.pct(recall.get("50")), "label": "recall@50"},
        {"value": matching.get("n_citations", 0), "label": "citations", "sub": "distinct"},
        {"value": len(matching.get("unmatched", [])), "label": "unmatched"},
    ])
    C.proxy_note("recall@K = share of citations whose matched candidate appears within the top-K "
                 "reconstructed SERP ranks. Unmatched citations were simply not found in the "
                 "reconstruction — not evidence the model didn't use them.")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.recall_bar(recall), width="stretch")
    with c2:
        st.plotly_chart(charts.match_type_bar(matching.get("rate_counts", {})), width="stretch")

    # strict vs weak recall
    cands = unique_candidates(run["serp"]["candidates"])
    pages = (run.get("scrape") or {}).get("pages", {})
    strict = match_all(run["gemini"]["citations"], cands, pages, False)["recall"]
    weak = match_all(run["gemini"]["citations"], cands, pages, True)["recall"]
    comp = pd.DataFrame([
        {"matching": "strict (strong only)", **{f"@{k}": strict.get(str(k), 0) for k in (5, 10, 20, 50)}},
        {"matching": "with domain-only", **{f"@{k}": weak.get(str(k), 0) for k in (5, 10, 20, 50)}},
    ])
    C.section("Strict vs weak matching", icon="⚖️")
    st.dataframe(comp, width="stretch", hide_index=True)
    st.caption("Recall values are fractions (0–1). 'with domain-only' also counts weak domain-level matches.")

    # match-type legend
    st.markdown(
        "**Tiers (strongest → weakest):** "
        + C.match_badge("exact") + " " + C.match_badge("normalized") + " "
        + C.match_badge("final_redirect") + " " + C.match_badge("canonical") + " "
        + C.match_badge("amp_canonical") + " " + C.match_badge("domain_only") + " "
        + C.match_badge("no_match"),
        unsafe_allow_html=True,
    )

    C.section("Per-citation matches", icon="🔗")
    matches = matching.get("matches", [])
    if matches:
        df = pd.DataFrame([{
            "match_type": m["match_type"], "strong": m["strong"],
            "matched_rank": m.get("matched_rank"),
            "citation_url": m["citation_url"], "matched_url": m.get("matched_url"),
        } for m in matches])
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"citation_url": st.column_config.LinkColumn("citation_url"),
                                    "matched_url": st.column_config.LinkColumn("matched_url")})
    unmatched = matching.get("unmatched", [])
    if unmatched:
        st.warning(f"{len(unmatched)} citation(s) not found in the reconstructed top-K:")
        for u in unmatched:
            st.markdown(f"- `{u}`")
