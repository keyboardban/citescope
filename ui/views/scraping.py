"""Web Scraping: extract page content for candidate websites via Apify."""

from __future__ import annotations

import streamlit as st

from src import config
from src.matching import unique_candidates
from src.pipeline import select_scrape_urls, stage_scrape
from src.url_utils import pretty_url

from .. import components as C
from ..state import get_clients, get_run, recompute_downstream, set_run


def render() -> None:
    run = get_run()
    C.section("Web Scraping",
              "Extract page content (title, headings, text, markdown, metadata) for candidate websites.", "🕸️")
    if not (run and run.get("serp") and run["serp"].get("candidates")):
        C.empty_state("Reconstruct a SERP first, then choose what to scrape.", "🕸️")
        return

    inputs = st.session_state["inputs"]
    sc = inputs["scrape"]
    cands = unique_candidates(run["serp"]["candidates"])
    cited_ids = set((run.get("matching") or {}).get("cited_candidate_ids", []))

    scope_label = {"top_k": "Top-K candidates", "cited": "Only cited", "all": "All candidates", "selected": "Selected URLs"}
    c1, c2 = st.columns([2, 1])
    keys = list(scope_label)
    sc["scope"] = c1.radio("Scope", keys, format_func=lambda k: scope_label[k], horizontal=True,
                           index=keys.index(sc["scope"]) if sc["scope"] in keys else 0)
    ct = config.CRAWLER_TYPES
    sc["crawler_type"] = c2.selectbox("Crawler type", ct,
                                      index=ct.index(sc["crawler_type"]) if sc["crawler_type"] in ct else 0,
                                      help="cheerio = fast static HTML; playwright = JS-rendered (slower).")

    if sc["scope"] == "top_k":
        sc["top_k"] = st.slider("How many top candidates", 1, max(1, len(cands)),
                                min(sc["top_k"], len(cands)))
    if sc["scope"] == "selected":
        opts = {f"#{c['best_rank']:>2} · {c['domain']}": c["url"] for c in cands}
        chosen = st.multiselect("Pick URLs to scrape", list(opts))
        sc["selected_urls"] = [opts[k] for k in chosen]

    sc["use_cache"] = st.toggle("Use cached pages when available", value=sc["use_cache"])

    urls = select_scrape_urls(sc["scope"], cands, cited_ids, sc["top_k"], sc["selected_urls"])
    st.caption(f"**{len(urls)}** page(s) selected for scraping.")

    clients = get_clients()
    ready = clients.get("apify") is not None
    if st.button("🕸️ Scrape pages", type="primary", disabled=not ready or not urls):
        with st.spinner(f"Scraping {len(urls)} page(s) via Apify…"):
            scrape = stage_scrape(clients["apify"], urls, sc, run.get("run_id", ""), use_cache=sc["use_cache"])
        run["scrape"] = scrape
        set_run(run)
        recompute_downstream()
        st.success("Scraping complete.")
    if not ready:
        st.warning("Set `APIFY_TOKEN` in `.env` to scrape.")

    scrape = run.get("scrape")
    if not scrape or not scrape.get("pages"):
        C.empty_state("No scraped pages yet. Choose a scope and click **Scrape pages**.", "🕸️")
        return

    pages = scrape["pages"]
    ok = [p for p in pages.values() if p.get("status") == "success"]
    fail = [p for p in pages.values() if p.get("status") != "success"]
    apify_meta = scrape.get("apify") if isinstance(scrape.get("apify"), dict) else {}
    C.metric_cards([
        {"value": len(pages), "label": "requested"},
        {"value": len(ok), "label": "succeeded"},
        {"value": len(fail), "label": "failed"},
        {"value": apify_meta.get("from_cache", 0), "label": "from cache"},
    ])

    if ok:
        labels = {f"{(p.get('title') or pretty_url(p.get('url','')))[:70]}": p for p in ok}
        sel = st.selectbox("Preview a scraped page", list(labels))
        p = labels[sel]
        st.markdown(f"**{p.get('title','')}**")
        st.caption(f"{p.get('url','')}  ·  canonical: {p.get('canonical_url') or '—'}  ·  "
                   f"published: {p.get('published_date') or '—'}  ·  lang: {p.get('language') or '—'}")
        from src.chunking import extract_headings
        heads = extract_headings(p.get("markdown", ""))
        if heads:
            with st.expander(f"Headings ({len(heads)})"):
                for h in heads[:40]:
                    st.markdown(f"- {h}")
        with st.expander("Main content (markdown)", expanded=False):
            md = p.get("markdown") or p.get("text") or ""
            st.markdown(md[:6000] + ("…" if len(md) > 6000 else ""))

    if fail:
        with st.expander(f"⚠️ Failed / skipped pages ({len(fail)})"):
            for p in fail:
                st.markdown(f"- `{pretty_url(p.get('url',''))}` — {p.get('error') or 'no content'}")
