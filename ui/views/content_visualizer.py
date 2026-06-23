"""Content Visualizer: per-page metadata, content, chunk relevance, radar."""

from __future__ import annotations

import streamlit as st

from src.chunking import extract_headings
from src.features import NUMERIC_FEATURES
from src.url_utils import normalize_url

from .. import charts
from .. import components as C
from ..state import get_run


def _avg_profile(feats: list[dict]) -> dict:
    dims = ["title_query_sim", "snippet_query_sim", "page_query_sim",
            "page_output_sim", "max_chunk_output_sim"]
    out = {}
    for d in dims:
        vals = [r[d] for r in feats if r.get(d) is not None]
        out[d] = sum(vals) / len(vals) if vals else 0.0
    return out


def render() -> None:
    run = get_run()
    C.section("Content Visualizer",
              "Inspect a candidate page, its content, and which chunks most overlap the AI answer.", "🔬")
    if not (run and run.get("features")):
        C.empty_state("Reconstruct a SERP (and scrape pages) to inspect candidate content.", "🔬")
        return

    feats = run["features"]
    chunks_map = run.get("chunks", {})
    pages = (run.get("scrape") or {}).get("pages", {})

    ordered = sorted(feats, key=lambda x: (0 if x.get("scrape_success") else 1, x.get("serp_rank", 999)))
    opts = {}
    for r in ordered:
        mark = "✓" if r.get("scrape_success") else "·"
        cited = "● " if r.get("cited") else ""
        opts[f"{mark} {cited}#{r.get('serp_rank')} · {r.get('domain')}"] = r
    sel = st.selectbox("Select a candidate website", list(opts))
    row = opts[sel]
    page = pages.get(normalize_url(row["url"]), {})

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {row.get('title') or row.get('domain')}")
        st.caption(row["url"])
        badges = C.cited_badge(bool(row.get("cited"))) + " " + C.badge(row.get("source_type", "unknown"), "src")
        if row.get("official_source"):
            badges += " " + C.badge("official", "src")
        if row.get("cited"):
            badges += " " + C.match_badge(row.get("match_type", "no_match"))
        badges += " " + C.badge(f"#{row.get('serp_rank','?')} rank", "rank")
        st.markdown(badges, unsafe_allow_html=True)
        if page:
            st.caption(
                f"words: {row.get('word_count') or '—'} · headings: {row.get('heading_count') or '—'} · "
                f"published: {page.get('published_date') or '—'} · lang: {page.get('language') or '—'} · "
                f"canonical: {page.get('canonical_url') or '—'}"
            )
    with right:
        st.plotly_chart(charts.similarity_radar(row, avg=_avg_profile(feats)), width="stretch")

    if not page or page.get("status") != "success":
        st.info("This candidate was not scraped — chunk relevance is unavailable. Scrape it in **Web Scraping**.")
        return

    heads = extract_headings(page.get("markdown", ""))
    if heads:
        with st.expander(f"Headings ({len(heads)})"):
            for h in heads[:40]:
                st.markdown(f"- {h}")
    with st.expander("Main content preview"):
        md = page.get("markdown") or page.get("text") or ""
        st.markdown(md[:6000] + ("…" if len(md) > 6000 else ""))

    cid = row["candidate_id"]
    chunks = chunks_map.get(cid, [])
    C.section("Chunk relevance", "Which passages overlap the AI output most (semantic overlap proxy).", "🧩")
    if not chunks:
        st.info("No chunks computed for this page.")
        return

    target = st.radio("Compare chunks to", ["AI answer", "search query"], horizontal=True)
    key = "output_sim" if target == "AI answer" else "query_sim"
    label = "answer" if target == "AI answer" else "query"
    st.plotly_chart(charts.chunk_relevance(chunks, key, label), width="stretch")
    C.proxy_note("High overlap suggests the passage is topically related to the AI output. It is NOT "
                 "evidence the model read this passage.")

    best = max(chunks, key=lambda c: c.get(key, 0) or 0)
    st.markdown(f"**Most similar chunk** — score `{best.get(key, 0):.3f}` · "
                f"_{best.get('heading') or '(no heading)'}_")
    st.success(best.get("text", "")[:900])

    with st.expander(f"All chunks ({len(chunks)})"):
        for c in sorted(chunks, key=lambda x: x.get(key, 0) or 0, reverse=True):
            st.markdown(f"`{c.get(key, 0):.3f}` · chunk {c['index']} · _{c.get('heading') or '(no heading)'}_")
            st.write(c.get("text", "")[:500] + "…")
            st.divider()
