"""Citation Matching: tiered matches, three recall variants, match-type breakdown."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import config
from src.analysis import _normalize_recall

from .. import charts
from .. import components as C
from ..state import get_run, recompute_downstream


def render() -> None:
    run = get_run()
    C.section("Citation Matching",
              "Link cited URLs to reconstructed SERP candidates with tiered rules; measure recall@K.", "🎯")
    if not (run and run.get("gemini") and run.get("serp") and run["serp"].get("candidates")):
        C.empty_state("Need a Gemini run and a reconstructed SERP first.", "🎯")
        return

    if not run.get("matching"):
        recompute_downstream()
        run = get_run()

    matching = run["matching"]
    recall = _normalize_recall(matching.get("recall") or {})
    strict, canon, dom = recall.get("strict", {}), recall.get("canonical", {}), recall.get("domain_inclusive", {})

    C.metric_cards([
        {"value": C.pct(strict.get("10")), "label": "strict recall@10", "sub": "URL identity"},
        {"value": C.pct(canon.get("10")), "label": "canonical recall@10", "sub": "+ canonical/amp"},
        {"value": C.pct(dom.get("10")), "label": "domain-incl recall@10", "sub": "weak / exploratory"},
        {"value": matching.get("n_citations", 0), "label": "citations", "sub": "distinct"},
        {"value": len(matching.get("cited_candidate_ids", [])), "label": "cited", "sub": "strong only"},
        {"value": len(matching.get("unmatched", [])), "label": "unmatched"},
    ])
    C.caveat_box(config.CAVEAT_RECALL)
    st.caption("Only **strong** matches (exact/normalized/redirect/canonical/amp) set cited = 1. "
               "**Weak domain-only** matches never flip the cited label and appear only in "
               "domain-inclusive recall.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.recall_grouped(recall), width="stretch")
    with c2:
        st.plotly_chart(charts.match_type_bar(matching.get("rate_counts", {})), width="stretch")

    C.section("Recall@K — three variants", icon="📊")
    rdf = pd.DataFrame([{
        "K": k, "strict_recall": strict.get(str(k), 0.0),
        "canonical_recall": canon.get(str(k), 0.0),
        "domain_inclusive_recall (weak)": dom.get(str(k), 0.0),
    } for k in (5, 10, 20, 50)])
    st.dataframe(rdf, width="stretch", hide_index=True)
    st.caption("Values are fractions (0–1). domain-inclusive is exploratory and counts weak domain matches.")

    C.section("Match-type counts", icon="🔢")
    rc = matching.get("rate_counts", {})
    order = ["exact", "normalized", "final_redirect", "canonical", "amp_canonical", "domain_only", "no_match"]
    cdf = pd.DataFrame([{"match_type": t.replace("_", " "), "count": rc.get(t, 0),
                        "strength": ("weak" if t == "domain_only" else ("—" if t == "no_match" else "strong"))}
                       for t in order])
    cdf.loc[len(cdf)] = ["TOTAL distinct citations", matching.get("n_citations", 0), ""]
    st.dataframe(cdf, width="stretch", hide_index=True)

    st.markdown(
        "**Tiers (strongest → weakest):** "
        + " ".join(C.match_badge(t) for t in
                   ["exact", "normalized", "final_redirect", "canonical", "amp_canonical", "domain_only", "no_match"]),
        unsafe_allow_html=True,
    )

    C.section("Per-citation matches", icon="🔗")
    matches = matching.get("matches", [])
    if matches:
        df = pd.DataFrame([{
            "match_type": m["match_type"], "strong": m["strong"],
            "weak_domain_match": m.get("weak_domain_match", False),
            "matched_rank": m.get("matched_rank"),
            "citation_url": m["citation_url"], "matched_url": m.get("matched_url"),
        } for m in matches])
        st.dataframe(df, width="stretch", hide_index=True,
                     column_config={"citation_url": st.column_config.LinkColumn("citation_url"),
                                    "matched_url": st.column_config.LinkColumn("matched_url")})
    unmatched = matching.get("unmatched", [])
    if unmatched:
        st.warning(f"{len(unmatched)} citation(s) not recovered in the reconstructed top-K "
                   "(not evidence the model didn't use them):")
        for u in unmatched:
            st.markdown(f"- `{u}`")
