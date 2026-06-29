"""ChatGPT Bright Data Source Audit — upload, parse, scrape, compare cited vs more-only."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import brand_visibility as bv, brightdata, chatgpt_pipeline as cgp
from src import cluster, config, demo, report, storage
from src.analysis import features_df
from src.config import CRAWLER_TYPES
from src.pipeline import make_sim_engine

from .. import charts
from .. import components as C
from ..state import get_clients

_PRE_SIM = ["title_prompt_similarity", "description_prompt_similarity",
            "page_prompt_similarity", "max_chunk_prompt_similarity"]
_POST_SIM = ["page_answer_similarity", "max_chunk_answer_similarity"]
_GS_COLS = ["feature", "cited_mean", "noncited_mean", "cited_median", "noncited_median", "delta"]


# --------------------------------------------------------------------------- #
def _sim_engine():
    a = st.session_state["inputs"]["analysis"]
    clients = get_clients()
    return make_sim_engine(a.get("similarity_method", "lexical (offline)"),
                           clients.get("gemini"), a.get("embedding_model", "text-embedding-004"))


def _parse_terms(text: str | None) -> list[str]:
    """Split a pasted brand-terms box on ; | newline (commas kept inside a term)."""
    return brightdata._split_terms(text or "")


def _recompute() -> None:
    ss = st.session_state
    run = ss.get("cg_run")
    if not run:
        return
    sim = _sim_engine()
    pages = ss.get("cg_pages") or {}
    out = cgp.recompute(run, pages, sim)
    ss["cg_features"], ss["cg_chunks"], ss["cg_analysis"] = out["features"], out["chunks"], out["analysis"]
    # Non-branded Brand Visibility layer — manifest terms, plus any fallback terms
    # pasted in the Brand Visibility tab (used only where a record has no terms).
    ss["cg_brand"] = bv.build_brand_visibility(
        run, out["features"], pages, sim,
        default_client_terms=_parse_terms(ss.get("cg_brand_client_terms")),
        default_competitor_terms=_parse_terms(ss.get("cg_brand_competitor_terms")),
    )


def render() -> None:
    ss = st.session_state
    C.section("ChatGPT Bright Data Source Audit",
              "Compare sources ChatGPT cited against sources shown but not cited (more-only).", "🟢")
    st.info(config.CHATGPT_INTRO, icon="🟢")

    # ensure features exist if a run is loaded (e.g. from sidebar) but not yet computed
    if ss.get("cg_run") and ss.get("cg_features") is None:
        _recompute()

    tabs = st.tabs(["📤 Upload", "📄 Records", "🔗 Source Table", "🕸️ Scrape Sources",
                    "📈 Feature Analysis", "🧩 Questions", "🎯 Intent", "🏷️ Brand Visibility",
                    "🔬 Content", "📊 Report"])
    with tabs[0]:
        _tab_upload(ss)
    with tabs[1]:
        _tab_records(ss.get("cg_run"))
    with tabs[2]:
        _tab_sources(ss.get("cg_run"))
    with tabs[3]:
        _tab_scrape(ss)
    with tabs[4]:
        _tab_analysis(ss)
    with tabs[5]:
        _tab_questions(ss)
    with tabs[6]:
        _tab_intent(ss)
    with tabs[7]:
        _tab_brand(ss)
    with tabs[8]:
        _tab_content(ss)
    with tabs[9]:
        _tab_report(ss)


# --------------------------------------------------------------------------- #
def _tab_upload(ss) -> None:
    C.section("Upload Bright Data export", "JSON or CSV array of ChatGPT runs.", "📤")
    st.caption("Upload the Bright Data **OUTPUT / results** export (the large JSON with `citations` and "
               "`search_sources`, e.g. `sd_*.json`) — **not** the input prompt list (`*_prompts.csv`).")
    up = st.file_uploader("Bright Data file", type=["json", "csv"], key="cg_upload")
    c1, c2 = st.columns([1, 3])
    if c1.button("🧪 Load sample", width="stretch"):
        run = demo.make_demo_brightdata()
        ss.update(cg_run=run, cg_pages={}, cg_features=None, cg_chunks={}, cg_analysis=None)
        _recompute()
        st.success(f"Loaded sample — {run['n_records']} records.")
    if up is not None and st.button("📥 Parse uploaded file", type="primary", width="stretch"):
        raw = up.getvalue()
        run = brightdata.parse_run(raw, up.name)
        ss.update(cg_run=run, cg_pages={}, cg_features=None, cg_chunks={}, cg_analysis=None)
        storage.save_raw(run["run_id"], "brightdata_source", raw.decode("utf-8", "replace"))
        storage.save_chatgpt_run(run)
        _recompute()
        if run.get("looks_like_input") or run.get("n_sources", 0) == 0:
            st.error(f"Parsed {run['n_records']} records from `{up.name}` but found **0 sources**. "
                     "This looks like a prompt/input file — upload the **results** export instead.")
        else:
            st.success(f"Parsed {run['n_records']} records · {run['n_cited']} cited · "
                       f"{run['n_more_only']} more-only from `{up.name}`.")

    # ---- Prompt Manifest (attaches topic + intent) ----
    st.divider()
    st.markdown("**Optional — Prompt Manifest** "
                "(`prompt_id, topic, intent, prompt[, country, prompt_language, expected_source_types]`)")
    st.caption("Matched to records by prompt text / prompt_hash → attaches **topic + intent** to every record, "
               "source, and feature row, enabling the **🎯 Intent** analysis.")
    mf = st.file_uploader("Prompt Manifest (CSV or JSON)", type=["csv", "json"], key="cg_manifest")
    mc1, mc2 = st.columns(2)
    if mc1.button("🔗 Apply manifest", disabled=mf is None, width="stretch"):
        if not ss.get("cg_run"):
            st.warning("Parse a results file first.")
        else:
            man = brightdata.parse_manifest(mf.getvalue(), mf.name)
            stats = brightdata.apply_manifest(ss["cg_run"], man)
            storage.save_chatgpt_run(ss["cg_run"])
            _recompute()
            st.success(f"Manifest matched **{stats['matched']}/{stats['total']}** records by prompt.")
            if stats["unmatched"]:
                st.warning(f"{stats['unmatched']} unmatched (prompt text/hash differs): "
                           + "; ".join(stats["unmatched_prompts"][:4]))
    if mc2.button("🧪 Load sample manifest", width="stretch"):
        if not ss.get("cg_run"):
            st.warning("Load the sample results first.")
        else:
            stats = brightdata.apply_manifest(ss["cg_run"], demo.make_demo_manifest())
            storage.save_chatgpt_run(ss["cg_run"])
            _recompute()
            st.success(f"Sample manifest matched {stats['matched']}/{stats['total']} records.")
    _man = (ss.get("cg_run") or {}).get("manifest")
    if _man and _man.get("applied"):
        st.info(f"Manifest applied — {_man['matched']}/{_man['total']} matched. Intent analysis is in the **🎯 Intent** tab.")

    run = ss.get("cg_run")
    if not run:
        C.empty_state("Upload a Bright Data JSON/CSV, or load the sample.", "📤")
        return
    C.metric_cards([
        {"value": run["n_records"], "label": "records"},
        {"value": run["n_cited"], "label": "cited sources"},
        {"value": run["n_more_only"], "label": "more-only sources"},
        {"value": run.get("source_file_name", ""), "label": "file"},
    ])
    if run.get("looks_like_input"):
        C.caveat_box("This file has prompts but **no sources** (no `citations` / `search_sources`). "
                     "It looks like a Bright Data **input/prompt** file — upload the **output/results** "
                     "export (the large JSON with citations, e.g. `sd_*.json`) to run the audit.")
    if run.get("warnings"):
        with st.expander(f"⚠️ Parsing warnings ({len(run['warnings'])})"):
            for w in run["warnings"]:
                st.markdown(f"- {w}")
    # detected fields per record
    rows = [{"record_id": r["record_id"], "prompt": r["prompt"][:60],
             "cited": sum(1 for s in r["sources"] if s["cited_label"] == 1),
             "more_only": sum(1 for s in r["sources"] if s["cited_label"] == 0),
             "queries": len(r["web_search_query"]), "answer_chars": len(r["answer_text"])}
            for r in run["records"]]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _tab_records(run) -> None:
    C.section("Parsed records", icon="📄")
    if not run:
        C.empty_state("No file parsed yet.", "📄")
        return
    df = pd.DataFrame([{
        "record_id": r["record_id"], "prompt": r["prompt"], "model": r.get("model"),
        "web_search_query": "; ".join(r.get("web_search_query", [])),
        "cited": sum(1 for s in r["sources"] if s["cited_label"] == 1),
        "more_only": sum(1 for s in r["sources"] if s["cited_label"] == 0),
    } for r in run["records"]])
    st.dataframe(df, width="stretch", hide_index=True)
    labels = {f"{r['record_id']} · {r['prompt'][:50]}": r for r in run["records"]}
    sel = st.selectbox("Preview answer", list(labels))
    rec = labels[sel]
    st.markdown(rec["answer_text"][:4000] + ("…" if len(rec["answer_text"]) > 4000 else "")
                or "_no answer text_")


def _tab_sources(run) -> None:
    C.section("Source table", "All sources, labeled cited vs more-only.", "🔗")
    C.caveat_box(config.CAVEAT_MORE_ONLY)
    if not run:
        C.empty_state("No file parsed yet.", "🔗")
        return
    rows = []
    for rec in run["records"]:
        for s in rec["sources"]:
            rows.append({
                "cited": bool(s["cited_label"]), "source_group": s["source_group"],
                "intent": rec.get("intent"), "topic": rec.get("topic"),
                "record_id": rec["record_id"], "domain": s.get("domain"), "title": s.get("title"),
                "source_origin": s.get("source_origin"), "source_position": s.get("source_position"),
                "observed_rank": s.get("observed_rank"), "url": s["url"], "description": s.get("description"),
            })
    df = pd.DataFrame(rows)
    grp = st.radio("Show", ["all", "cited", "more_only"], horizontal=True, key="cg_src_filter")
    if grp == "cited":
        df = df[df["cited"]]
    elif grp == "more_only":
        df = df[~df["cited"]]
    st.dataframe(df, width="stretch", hide_index=True,
                 column_config={"cited": st.column_config.CheckboxColumn("cited"),
                                "url": st.column_config.LinkColumn("url"),
                                "description": st.column_config.TextColumn("description", width="medium")})


def _tab_scrape(ss) -> None:
    C.section("Scrape sources", "Fetch page content for the source URLs (shared Apify crawler).", "🕸️")
    run = ss.get("cg_run")
    if not run:
        C.empty_state("Parse a file first.", "🕸️")
        return
    scope_label = {"all": "All sources", "cited": "Cited only", "more_only": "More-only only", "selected": "Selected URLs"}
    keys = list(scope_label)
    c1, c2 = st.columns([2, 1])
    scope = c1.radio("Scope", keys, format_func=lambda k: scope_label[k], horizontal=True, key="cg_scope")
    crawler = c2.selectbox("Crawler type", CRAWLER_TYPES, key="cg_crawler")
    selected_norm = []
    if scope == "selected":
        flat = cgp.flatten_sources(run)
        opts = {f"{s['domain']} · {s['title'][:40]}": s["normalized_url"] for s in flat}
        chosen = st.multiselect("Pick sources", list(opts))
        selected_norm = [opts[k] for k in chosen]
    use_cache = st.toggle("Use cached pages when available", value=True, key="cg_cache")

    urls = cgp.select_scrape_urls(run, scope, selected_norm)
    st.caption(f"**{len(urls)}** unique URL(s) selected for scraping.")
    clients = get_clients()
    if st.button("🕸️ Scrape sources", type="primary", disabled=clients.get("apify") is None or not urls):
        with st.spinner(f"Scraping {len(urls)} page(s)…"):
            res = cgp.scrape_sources(clients, urls, {"crawler_type": crawler, "scope": scope},
                                     run["run_id"], use_cache=use_cache)
        ss["cg_pages"] = {**(ss.get("cg_pages") or {}), **res.get("pages", {})}
        _recompute()
        st.success("Scraping complete.")
    if clients.get("apify") is None:
        st.warning("Set `APIFY_TOKEN` in `.env` to scrape source pages.")

    pages = ss.get("cg_pages") or {}
    if pages:
        ok = [p for p in pages.values() if p.get("status") == "success"]
        fail = [p for p in pages.values() if p.get("status") != "success"]
        C.metric_cards([
            {"value": len(pages), "label": "scraped"},
            {"value": len(ok), "label": "succeeded"},
            {"value": len(fail), "label": "failed"},
        ])
        if fail:
            with st.expander(f"⚠️ Failed pages ({len(fail)})"):
                for p in fail:
                    st.markdown(f"- `{p.get('url','')}` — {p.get('error') or 'no content'}")


def _tab_analysis(ss) -> None:
    C.section("Feature Analysis", "Cited vs more-only (shown-but-not-cited) sources.", "📈")
    feats = ss.get("cg_features")
    an = ss.get("cg_analysis")
    if not feats or not an:
        C.empty_state("Parse a file (and optionally scrape) to build features.", "📈")
        return
    m = an["summary"]
    C.metric_cards([
        {"value": m["n_records"], "label": "prompts"},
        {"value": m["n_sources"], "label": "sources"},
        {"value": m["n_cited"], "label": "cited"},
        {"value": m["n_more_only"], "label": "more-only"},
        {"value": m["n_scraped"], "label": "scraped"},
        {"value": C.pct(m["scrape_success_rate"]), "label": "scrape rate"},
    ])
    C.proxy_note("Cited vs more-only differences are observable associations, not proof of selection. "
                 "Pre-answer signals are the cleaner ones.")

    C.section("Position-adjusted citation model",
              "Δ probability of being cited per feature, holding others (incl. source position) fixed — "
              "clustered by prompt. The rigorous read; the correlation below is an unadjusted screen.", "📐")
    C.regression_block(an.get("regression"))

    df = features_df(feats, cgp.CHATGPT_NUMERIC)
    gc = an["group_compare"]
    gdf = pd.DataFrame(gc)

    C.section("Pre-answer signals (non-circular)", icon="✅")
    st.plotly_chart(charts.grouped_means(gc, _PRE_SIM), width="stretch")
    if not gdf.empty:
        st.dataframe(gdf[gdf["phase"] == "pre_answer"][_GS_COLS], width="stretch", hide_index=True)

    C.section("Post-output overlap (may be circular)", icon="🌀")
    C.caveat_box(config.CAVEAT_ANSWER_CG)
    p1, p2 = st.columns([2, 3])
    with p1:
        st.plotly_chart(charts.grouped_means(gc, _POST_SIM), width="stretch")
    with p2:
        st.plotly_chart(charts.length_vs_sim_scatter(df, "page_answer_similarity"), width="stretch")
    if not gdf.empty:
        st.dataframe(gdf[gdf["phase"] == "post_output"][_GS_COLS], width="stretch", hide_index=True)

    C.section("Distribution by feature", icon="📦")
    opts = {cgp.CHATGPT_LABELS.get(k, k): k for k in cgp.CHATGPT_NUMERIC
            if k in df.columns and df[k].notna().any()}
    if opts:
        chosen = st.selectbox("Feature", list(opts), key="cg_dist")
        st.plotly_chart(charts.distribution_box(df, opts[chosen], chosen), width="stretch")

    C.section("Source types", icon="🗂️")
    sb = pd.DataFrame(an["source_breakdown"])
    if not sb.empty:
        s1, s2 = st.columns(2)
        s1.plotly_chart(charts.source_stacked(sb), width="stretch")
        s2.plotly_chart(charts.cite_rate_by_source(sb), width="stretch")
        st.dataframe(sb, width="stretch", hide_index=True)

    off = an.get("official") or {}
    if off:
        C.section("Official signals", icon="🏛️")
        st.plotly_chart(charts.official_bar(off), width="stretch")

    C.section("Top domains", icon="🌐")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Cited**")
        st.dataframe(pd.DataFrame(an.get("top_domains_cited") or []), width="stretch", hide_index=True)
    with d2:
        st.markdown("**More-only**")
        st.dataframe(pd.DataFrame(an.get("top_domains_more") or []), width="stretch", hide_index=True)


def _tab_questions(ss) -> None:
    C.section("Questions & clusters",
              "Each question's cited/searched sites, and questions grouped by shared websites.", "🧩")
    feats = ss.get("cg_features")
    if not feats:
        C.empty_state("Parse a file first (Upload tab).", "🧩")
        return
    grp_label = {"cited": "cited sources", "all": "all surfaced sources", "more_only": "more-only sources"}
    grp = st.radio("Group questions by", list(grp_label), format_func=lambda g: grp_label[g],
                   horizontal=True, key="cg_q_group")

    qt = cluster.question_table(feats)
    st.dataframe(qt, width="stretch", hide_index=True,
                 column_config={"prompt": st.column_config.TextColumn("prompt", width="large")})

    C.section("Inspect one question", icon="🔎")
    labels = {f"{r.qid[:8]} · {r.prompt[:60]}": r.qid for r in qt.itertuples()}
    sel = st.selectbox("Question", list(labels))
    rid = labels[sel]
    rows = [r for r in feats if (r.get("record_id") or r.get("run_id")) == rid]
    drill = pd.DataFrame([{"cited": bool(r["cited"]), "domain": r.get("domain"),
                           "source_type": r.get("source_type"), "title": r.get("title"),
                           "url": r.get("url")} for r in rows])
    if not drill.empty:
        c1, c2 = st.columns(2)
        c1.markdown(f"**Cited ({int(drill['cited'].sum())})**")
        c1.dataframe(drill[drill["cited"]][["domain", "source_type", "title"]], width="stretch", hide_index=True)
        c2.markdown(f"**More-only ({int((~drill['cited']).sum())})**")
        c2.dataframe(drill[~drill["cited"]][["domain", "source_type", "title"]], width="stretch", hide_index=True)

    C.section("Cluster questions by shared websites", icon="🔠")
    n_q = len(qt)
    if n_q < 3:
        st.info("Need at least 3 questions to cluster.")
        return
    k = st.slider("Number of clusters", 2, min(8, n_q - 1), min(3, n_q - 1), key="cg_k")
    st.plotly_chart(charts.question_domain_heatmap(
        cluster.clustered_question_matrix(feats, grp, k),
        f"Questions × top domains ({grp_label[grp]}; rows grouped by cluster)"), width="stretch")
    for c in cluster.cluster_questions(feats, grp, k):
        tag = f"Cluster {c['cluster']}" if c["cluster"] >= 0 else "Unclustered (no sources)"
        top = ", ".join(d["domain"] for d in c["top_domains"][:4])
        with st.expander(f"{tag} — {c['size']} question(s)" + (f" · top: {top}" if top else "")):
            for m in c["members"]:
                extra = f"  _{m['intent']}_" if m.get("intent") else ""
                st.markdown(f"- {m['prompt']}{extra}")
            if c["top_domains"]:
                st.markdown("**Most-shared domains:** "
                            + ", ".join(f"{d['domain']} ({d['n_questions']})" for d in c["top_domains"]))
    C.proxy_note("Clusters group questions by overlap in the websites they "
                 + ("cite" if grp == "cited" else "surface")
                 + " (Jaccard distance, agglomerative). Observable grouping — not causal.")


def _tab_intent(ss) -> None:
    C.section("Intent → Source Type Analysis",
              "For each intent, which website types does the AI search, surface, cite, or leave as more-only?", "🎯")
    feats = ss.get("cg_features")
    run = ss.get("cg_run")
    if not feats:
        C.empty_state("Parse a file first (Upload tab).", "🎯")
        return
    if not (run and run.get("has_intent")):
        st.info("Upload & **Apply** a Prompt Manifest (Upload tab) to attach intent/topic — then this analysis activates.")
        return
    C.proxy_note("Observable source-placement patterns by intent — not the AI's internal retrieval process. "
                 "'more-only' = shown-but-not-cited (not rejected).")

    long = cgp.intent_source_long(feats)
    ldf = pd.DataFrame(long)

    # 1) Intent × Source Type (counts + %)
    C.section("Intent × Source Type (all surfaced)", icon="📊")
    counts = ldf.pivot_table(index="intent", columns="source_type", values="n", aggfunc="sum", fill_value=0)
    st.plotly_chart(charts.intent_sourcetype_bar(long, None, "All surfaced source types by intent"), width="stretch")
    st.dataframe(counts, width="stretch")
    pct = counts.div(counts.sum(axis=1).replace(0, 1), axis=0).round(3)
    with st.expander("Row % within each intent"):
        st.dataframe(pct, width="stretch")

    # 2) Cited source types by intent
    C.section("Cited source types by intent", icon="✅")
    cdf = ldf[ldf["group"] == "cited"]
    if not cdf.empty:
        st.plotly_chart(charts.intent_sourcetype_bar(long, "cited", "Cited source types by intent"), width="stretch")
        st.dataframe(cdf.pivot_table(index="intent", columns="source_type", values="n", aggfunc="sum", fill_value=0),
                     width="stretch")

    # 3) More-only source types by intent
    C.section("More-only (shown-but-not-cited) source types by intent", icon="🟡")
    mdf = ldf[ldf["group"] == "more_only"]
    if not mdf.empty:
        st.dataframe(mdf.pivot_table(index="intent", columns="source_type", values="n", aggfunc="sum", fill_value=0),
                     width="stretch")

    # 4) Cited vs more-only comparison by intent
    C.section("Cited vs more-only — composition by intent", icon="⚖️")
    st.dataframe(pd.DataFrame(cgp.intent_summary(feats)), width="stretch", hide_index=True)
    st.caption("Shares are of each intent's **cited** sources. official_cited_pct = % of cited that are institutional "
               "or brand-official (heuristic).")
    comp = ldf.pivot_table(index=["intent", "source_type"], columns="group", values="n",
                           aggfunc="sum", fill_value=0).reset_index()
    for col in ("cited", "more_only"):
        if col not in comp.columns:
            comp[col] = 0
    comp["cite_rate"] = (comp["cited"] / (comp["cited"] + comp["more_only"]).replace(0, 1)).round(2)
    with st.expander("Per intent × source-type (cited vs more-only + cite-rate)"):
        st.dataframe(comp, width="stretch", hide_index=True)

    # 5) Expected vs actual
    if run.get("manifest", {}).get("has_expected"):
        C.section("Expected vs actual cited source types", icon="🧭")
        ev = cgp.expected_vs_actual(feats)
        if ev:
            st.dataframe(pd.DataFrame(ev), width="stretch", hide_index=True)
            st.caption("Heuristic mapping of manifest `expected_source_types` onto observed source types + "
                       "official/brand flags. coverage = share of expected types that appeared among cited sources.")


def _tab_brand(ss) -> None:
    C.section("Non-branded Brand Visibility Audit",
              "For non-branded prompts: does the client / competitor appear or get cited, and which "
              "content features separate cited from more-only pages?", "🏷️")
    st.info(config.BRAND_VISIBILITY_INTRO, icon="🏷️")

    run = ss.get("cg_run")
    with st.expander("🔧 Brand terms (optional fallback) · load sample", expanded=not run):
        st.caption("Brand terms normally come from the **Prompt Manifest** "
                   "(`client_brand_terms_to_detect_in_output` / `competitor_terms_to_detect_in_output`, "
                   "semicolon-separated). You can also paste fallback terms here — they apply only to "
                   "records that carry no manifest terms.")
        c1, c2 = st.columns(2)
        c1.text_area("Client brand terms (`;`-separated)", key="cg_brand_client_terms",
                     placeholder="ศิริราช;Siriraj;SIPH;siphhospital.com", height=80)
        c2.text_area("Competitor terms (`;`-separated)", key="cg_brand_competitor_terms",
                     placeholder="Bumrungrad;bumrungrad.com;Bangkok Hospital", height=80)
        b1, b2 = st.columns(2)
        if b1.button("🔁 Apply terms / recompute", disabled=run is None, width="stretch"):
            _recompute()
            st.success("Recomputed brand visibility with the current terms.")
        if b2.button("🧪 Load brand-visibility sample", width="stretch"):
            d = demo.make_demo_brand_run()
            ss.update(cg_run=d["run"], cg_pages=d["pages"], cg_features=None, cg_chunks={}, cg_analysis=None)
            _recompute()
            st.success(f"Loaded brand sample — {d['run']['n_records']} non-branded prompts "
                       "with client/competitor sources + scraped pages.")
            run = ss.get("cg_run")

    if not run:
        C.empty_state("Upload a Bright Data results file + a Prompt Manifest (Upload tab), "
                      "or load the brand-visibility sample above.", "🏷️")
        return

    brand = ss.get("cg_brand")
    if brand is None:
        _recompute()
        brand = ss.get("cg_brand")
    if not brand:
        return

    man = run.get("manifest") or {}
    s = brand["summary"]

    # 1) status
    C.section("Status", icon="✅")
    C.metric_cards([
        {"value": run.get("n_records", 0), "label": "result records"},
        {"value": "yes" if man.get("applied") else "no", "label": "manifest applied"},
        {"value": s["nonbranded_prompts"], "label": "non-branded prompts"},
        {"value": len(brand.get("client_terms") or []), "label": "client terms"},
        {"value": len(brand.get("competitor_terms") or []), "label": "competitor terms"},
    ])
    if not brand.get("has_terms"):
        C.caveat_box("No client/competitor brand terms detected. Apply a Prompt Manifest with "
                     "`client_brand_terms_to_detect_in_output` / `competitor_terms_to_detect_in_output`, "
                     "or paste fallback terms above. The record-level table below still works as the "
                     "visibility denominator.")
    else:
        st.caption("**Client terms:** " + ", ".join(brand["client_terms"])
                   + "  ·  **Competitor terms:** " + ", ".join(brand["competitor_terms"]))
    C.caveat_box(config.CAVEAT_BRAND_VISIBILITY)

    # 2) overall visibility
    C.section("Observable brand visibility", icon="📊")
    C.metric_cards([
        {"value": C.pct(s["client_appeared_rate"]), "label": "client appeared"},
        {"value": C.pct(s["client_cited_rate"]), "label": "client cited"},
        {"value": C.pct(s["competitor_appeared_rate"]), "label": "competitor appeared"},
        {"value": C.pct(s["competitor_cited_rate"]), "label": "competitor cited"},
        {"value": f'{s["client_vs_competitor_cited_delta"]:+.2f}', "label": "client−comp cited Δ"},
    ])
    st.plotly_chart(charts.brand_overall_bar(s), width="stretch")

    # 3) by intent
    bi = brand.get("by_intent") or []
    if bi:
        C.section("Visibility by intent", icon="🎯")
        st.caption("Which intents cause the client or a competitor to appear organically? "
                   "Rates use **non-branded prompts** in each intent as the denominator.")
        st.plotly_chart(charts.brand_visibility_intent_bar(bi), width="stretch")
        bidf = pd.DataFrame(bi)
        show = ["topic", "intent", "total_prompts", "nonbranded_prompts",
                "client_appeared_rate", "client_cited_rate", "client_more_only_rate",
                "competitor_appeared_rate", "competitor_cited_rate", "competitor_more_only_rate",
                "client_vs_competitor_cited_delta"]
        st.dataframe(bidf[[c for c in show if c in bidf.columns]], width="stretch", hide_index=True)

    # 4) client vs competitor
    cvc = brand.get("client_vs_competitor") or []
    if cvc:
        C.section("Client vs competitor", icon="⚖️")
        st.dataframe(pd.DataFrame(cvc), width="stretch", hide_index=True)

    # 5) examples
    C.section("Example prompts", icon="💡")
    ex = brand.get("examples") or {}
    ex_specs = [("✅ Client was cited", "client_cited"),
                ("🟠 Competitor cited, client absent", "competitor_cited_client_absent"),
                ("🟡 Client appeared only as more-only", "client_more_only"),
                ("⚪ Neither appeared", "neither_appeared")]
    ecols = st.columns(2)
    for i, (title, key) in enumerate(ex_specs):
        with ecols[i % 2]:
            items = ex.get(key) or []
            st.markdown(f"**{title}** ({len(items)})")
            if items:
                st.dataframe(pd.DataFrame([{"intent": it.get("intent"), "prompt": it.get("prompt")}
                                           for it in items]), width="stretch", hide_index=True)
            else:
                st.caption("_none_")

    # 6) cited vs more-only content
    sp = brand.get("source_pages") or []
    C.section("Cited vs more-only content features", icon="🧬")
    if not sp:
        st.info("No client/competitor-matched sources yet. Add brand terms (and ideally scrape pages) to populate this.")
    else:
        st.caption(f"{len(sp)} brand-matched source page(s) · {brand.get('n_scraped_source_pages', 0)} scraped. "
                   "Content features need scraped pages (Scrape Sources tab).")
        grp = st.radio("Brand group", ["all", "client", "competitor"], horizontal=True, key="cg_brand_cv_group")
        st.plotly_chart(charts.brand_feature_compare(brand.get("cited_vs_moreonly") or [], grp), width="stretch")
        cv = pd.DataFrame(brand.get("cited_vs_moreonly") or [])
        if not cv.empty:
            keep = ["feature", "cited_mean", "more_only_mean", "cited_median", "more_only_median",
                    "delta", "n_cited", "n_more_only"]
            st.dataframe(cv[cv["group"] == grp][keep], width="stretch", hide_index=True)
        C.proxy_note("Positive delta = feature more common/higher among CITED brand pages; negative = more "
                     "common among MORE-ONLY (shown-but-not-cited). Observable association, not proof of selection.")

        # 7) position-controlled
        C.section("Position-controlled comparison", icon="📏")
        st.caption("Within similar source-position bands (1-3 / 4-6 / 7-10 / 11+), which content features still "
                   "differ between cited and more-only pages? Controls for the strong position effect.")
        pb = pd.DataFrame(brand.get("by_position_band") or [])
        if pb.empty:
            st.info("Not enough positioned brand pages to band yet.")
        else:
            pgrp = st.radio("Brand group ", ["all", "client", "competitor"], horizontal=True, key="cg_brand_pb_group")
            cols = ["position_band", "feature", "cited_mean", "more_only_mean", "delta", "n_cited", "n_more_only"]
            st.dataframe(pb[pb["brand_match_group"] == pgrp][cols], width="stretch", hide_index=True)

    # 7b) position-adjusted regression — the rigorous companion to the band comparison
    if sp:
        C.section("Position-adjusted content model (regression)", icon="📐")
        st.caption("LPM: Δ probability of citation per content feature, holding source position fixed and "
                   "clustered by prompt. The rigorous version of the position-band comparison above.")
        C.regression_block(brand.get("position_adjusted"))

    # 8) downloads
    C.section("Download brand-visibility exports", icon="⬇️")
    rid = run["run_id"]
    d1 = st.columns(3)
    d1[0].download_button("records.csv", report.brand_visibility_records_csv(brand),
                          f"{rid}_brand_visibility_records.csv", "text/csv", width="stretch")
    d1[1].download_button("by_intent.csv", report.brand_visibility_by_intent_csv(brand),
                          f"{rid}_brand_visibility_by_intent.csv", "text/csv", width="stretch")
    d1[2].download_button("source_pages.csv", report.brand_source_pages_csv(brand),
                          f"{rid}_brand_source_pages.csv", "text/csv", width="stretch")
    d2 = st.columns(3)
    d2[0].download_button("client_vs_competitor.csv", report.client_vs_competitor_visibility_csv(brand),
                          f"{rid}_client_vs_competitor_visibility.csv", "text/csv", width="stretch")
    d2[1].download_button("cited_vs_moreonly_content.csv", report.cited_vs_moreonly_content_features_csv(brand),
                          f"{rid}_cited_vs_moreonly_content_features.csv", "text/csv", width="stretch")
    d2[2].download_button("by_position_band.csv", report.content_features_by_position_band_csv(brand),
                          f"{rid}_content_features_by_position_band.csv", "text/csv", width="stretch")

    with st.expander("📄 Record-level visibility table (all prompts kept — visibility denominator)"):
        recs = pd.DataFrame(brand.get("records") or [])
        if "answer_text" in recs.columns:
            recs = recs.drop(columns=["answer_text"])
        st.dataframe(recs, width="stretch", hide_index=True)


def _tab_content(ss) -> None:
    C.section("Content / Chunk Visualizer", icon="🔬")
    feats = ss.get("cg_features")
    chunks_map = ss.get("cg_chunks") or {}
    pages = ss.get("cg_pages") or {}
    if not feats:
        C.empty_state("Parse and scrape sources to inspect content.", "🔬")
        return
    scraped = [f for f in feats if f.get("scrape_success")]
    if not scraped:
        st.info("No scraped pages yet — scrape sources in the **Scrape Sources** tab.")
        return
    labels = {f"{'● ' if f['cited'] else ''}{f['domain']} · {f['title'][:40]}": f for f in scraped}
    sel = st.selectbox("Select a scraped source", list(labels))
    row = labels[sel]
    badges = C.cited_badge(bool(row["cited"])) + " " + C.badge(row.get("source_group", ""), "cited" if row["cited"] else "noncited")
    badges += " " + C.badge(row.get("source_type", "unknown"), "src")
    if row.get("institutional_official"):
        badges += " " + C.badge("official", "src")
    if row.get("brand_official_candidate"):
        badges += " " + C.badge("official?", "brand")
    st.markdown(f"**{row.get('title') or row.get('domain')}**")
    st.caption(row["url"])
    st.markdown(badges, unsafe_allow_html=True)
    trunc = " · ✂️ truncated for similarity" if row.get("truncated") else ""
    st.caption(f"words: {row.get('word_count') or '—'} · chars used "
               f"{row.get('used_char_count') or '—'}/{row.get('original_char_count') or '—'}{trunc} · "
               f"origin: {row.get('source_origin')}")

    page = pages.get(row["normalized_url"], {})
    with st.expander("Main content preview"):
        md = page.get("markdown") or page.get("text") or ""
        st.markdown(md[:6000] + ("…" if len(md) > 6000 else ""))

    chunks = chunks_map.get(row["source_id"], [])
    if not chunks:
        st.info("No chunks for this page.")
        return
    target = st.radio("Compare chunks to", ["prompt", "answer"], horizontal=True, key="cg_chunk_target")
    key = "prompt_sim" if target == "prompt" else "answer_sim"
    if target == "answer":
        C.caveat_box(config.CAVEAT_ANSWER_CG)
    st.plotly_chart(charts.chunk_relevance(chunks, key, target), width="stretch")
    best = max(chunks, key=lambda c: c.get(key, 0) or 0)
    st.markdown(f"**Most similar chunk** — `{best.get(key, 0):.3f}` · _{best.get('heading') or '(no heading)'}_")
    st.success(best.get("text", "")[:900])


def _tab_report(ss) -> None:
    C.section("Report & Export", icon="📊")
    run = ss.get("cg_run")
    if not run:
        C.empty_state("No run to export yet.", "📊")
        return
    an = ss.get("cg_analysis") or {}
    feats = ss.get("cg_features") or []
    brand = ss.get("cg_brand")
    rid = run["run_id"]
    md = report.chatgpt_markdown_report(run, an, feats, brand)
    st.caption("The **AI-ready report** embeds a feature dictionary, a feature↔citation correlation table, "
               "intent → source-type breakdowns, the **Non-branded Brand Visibility Audit**, and the raw "
               "per-source data — paste it (or the JSON bundle) into an AI to find correlations.")
    c = st.columns(3)
    c[0].download_button("⬇️ AI-ready report (Markdown)", md, f"{rid}_report.md", "text/markdown", width="stretch")
    c[1].download_button("⬇️ Analysis bundle (JSON)", report.chatgpt_analysis_json(run, an, feats, brand),
                         f"{rid}_analysis.json", "application/json", width="stretch")
    c[2].download_button("⬇️ Per-source dataset (CSV)", report.chatgpt_dataset_csv(feats),
                         f"{rid}_dataset.csv", "text/csv", width="stretch")
    c2 = st.columns(3)
    c2[0].download_button("⬇️ Source table (CSV)", report.chatgpt_sources_csv(run),
                          f"{rid}_sources.csv", "text/csv", width="stretch")
    c2[1].download_button("⬇️ Feature table (CSV)", report.chatgpt_features_csv(feats),
                          f"{rid}_features.csv", "text/csv", width="stretch")
    if run.get("has_intent"):
        c2[2].download_button("⬇️ Intent × source-type (CSV)",
                              report.chatgpt_intent_csv(cgp.intent_source_long(feats)),
                              f"{rid}_intent.csv", "text/csv", width="stretch")

    if brand and brand.get("has_terms"):
        st.markdown("**Non-branded Brand Visibility exports**")
        b1 = st.columns(3)
        b1[0].download_button("⬇️ brand_visibility_records.csv", report.brand_visibility_records_csv(brand),
                              f"{rid}_brand_visibility_records.csv", "text/csv", width="stretch")
        b1[1].download_button("⬇️ brand_visibility_by_intent.csv", report.brand_visibility_by_intent_csv(brand),
                              f"{rid}_brand_visibility_by_intent.csv", "text/csv", width="stretch")
        b1[2].download_button("⬇️ brand_source_pages.csv", report.brand_source_pages_csv(brand),
                              f"{rid}_brand_source_pages.csv", "text/csv", width="stretch")
        b2 = st.columns(3)
        b2[0].download_button("⬇️ client_vs_competitor_visibility.csv",
                              report.client_vs_competitor_visibility_csv(brand),
                              f"{rid}_client_vs_competitor_visibility.csv", "text/csv", width="stretch")
        b2[1].download_button("⬇️ cited_vs_moreonly_content_features.csv",
                              report.cited_vs_moreonly_content_features_csv(brand),
                              f"{rid}_cited_vs_moreonly_content_features.csv", "text/csv", width="stretch")
        b2[2].download_button("⬇️ content_features_by_position_band.csv",
                              report.content_features_by_position_band_csv(brand),
                              f"{rid}_content_features_by_position_band.csv", "text/csv", width="stretch")

    if st.button("💾 Save run snapshot to data/chatgpt/"):
        path = storage.save_chatgpt_run(run)
        st.success(f"Saved: {path}")
    C.section("Report preview", icon="📄")
    with st.container(border=True):
        st.markdown(md)
