"""Topic Studies: throw many questions across topics, find cited vs non-cited patterns."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import batch as batch_mod
from src import cluster, config, demo, question_sets, report
from src.analysis import FEATURE_LABELS, PRE_ANSWER_FEATURES

from .. import charts
from .. import components as C
from ..state import get_clients

_GS_COLS = ["feature", "phase", "cited_median", "noncited_median", "median_diff",
            "mwu_p", "ci_low", "ci_high", "n_cited", "n_noncited"]


def render() -> None:
    C.section("Topic Studies",
              "Throw many questions across topics; find cited vs non-cited website patterns.", "🧪")
    C.caveat_box(config.CAVEAT_BATCH)
    ss = st.session_state

    packs = list(question_sets.TOPIC_SETS)
    mode = st.radio("Add questions from", ["📦 Topic packs", "📝 Paste prompts", "Both"],
                    horizontal=True, key="topic_mode")
    pack_items, paste_items = [], []

    if mode in ("📦 Topic packs", "Both"):
        chosen = st.multiselect(
            "Topic packs", packs, default=packs,
            format_func=lambda t: f"{question_sets.TOPIC_EMOJI.get(t,'•')} {t} ({len(question_sets.TOPIC_SETS[t])})")
        pack_items = question_sets.items_for(chosen)

    if mode in ("📝 Paste prompts", "Both"):
        custom_topic = st.text_input("Topic label for pasted prompts", "Custom")
        paste = st.text_area(
            "Paste prompts — one per line (no ID or intent needed)", height=180, key="topic_paste",
            placeholder=("What ingredients should I look for in a moisturizer for dry skin?\n"
                         "Hybrid vs electric cars: which is more cost-effective in Bangkok?\n"
                         "Is buying a condo in Bangkok still a good investment?"))
        structured = st.checkbox("My lines include 'ID | Intent | Prompt' columns", value=False)
        if paste.strip():
            paste_items = (question_sets.parse_prompt_block(paste, default_topic=custom_topic)
                           if structured else question_sets.simple_prompts(paste, default_topic=custom_topic))

    items = pack_items + paste_items
    if not items:
        C.empty_state("Pick a topic pack or paste prompts (one per line).", "🧪")
        return

    n_topics = len({i["topic"] for i in items})
    st.caption(f"**{len(items)}** question(s) across **{n_topics}** topic(s).")
    with st.expander("Preview question set"):
        st.dataframe(pd.DataFrame(items)[["topic", "id", "intent", "prompt"]],
                     width="stretch", hide_index=True)

    max_q = st.slider("Max questions to run (cost control)", 1, len(items), min(len(items), 12),
                      help="Each question runs the full pipeline (Gemini + Apify SERP + scrape).")
    use_cache = st.toggle("Use cached results when available", value=True)
    run_items = items[:max_q]

    clients = get_clients()
    ready = clients.get("gemini") is not None and clients.get("apify") is not None
    b1, b2 = st.columns(2)
    if b1.button("⚡ Run topic study", type="primary", disabled=not ready or not run_items, width="stretch"):
        bar = st.progress(0.0, text="Starting…")
        try:
            study = batch_mod.run_batch(clients, run_items, ss["inputs"],
                                        progress=lambda s, f: bar.progress(min(f, 1.0), text=s),
                                        use_cache=use_cache)
        except Exception as exc:
            bar.empty()
            st.error(f"Topic study failed: {type(exc).__name__}: {exc}")
            return
        bar.progress(1.0, text="Done")
        ss["topic_study"] = study
        ok = study["aggregate"]["sample_sizes"]["n_runs_ok"]
        st.success(f"Topic study complete — {ok}/{len(run_items)} runs succeeded.")
    if b2.button("🧪 Load demo topic study (no API)", width="stretch"):
        ss["topic_study"] = demo.make_demo_topic_study()
        st.rerun()
    if not ready:
        st.warning("Live runs need `GEMINI_API_KEY` + `APIFY_TOKEN`. Use the demo to explore the layout.")

    study = ss.get("topic_study")
    if not study:
        C.empty_state("No topic study yet. Pick packs and **Run** (or load the demo).", "🧪")
        return
    _show(study)


def _show(study: dict) -> None:
    agg = study.get("aggregate") or {}
    s = agg.get("sample_sizes") or {}
    by_topic = agg.get("by_topic") or {}

    C.metric_cards([
        {"value": study.get("n_prompts", 0), "label": "prompts"},
        {"value": s.get("n_runs_ok", 0), "label": "runs ok"},
        {"value": s.get("n_candidates", 0), "label": "candidates"},
        {"value": s.get("n_cited", 0), "label": "cited", "sub": "strong"},
        {"value": C.pct((s.get("n_cited", 0) / s["n_candidates"]) if s.get("n_candidates") else None), "label": "overall cite-rate"},
        {"value": C.pct(agg.get("recall", {}).get("strict", {}).get("10")), "label": "strict recall@10"},
    ])

    # The headline answer.
    C.section("Observable patterns — cited vs non-cited", icon="💡")
    for p in agg.get("patterns", []):
        st.markdown(f"- {p}")
    C.proxy_note("Patterns are observable associations pooled across prompts — not causal evidence of how "
                 "the AI selects or cites sources. Pre-answer signals (rank, query similarity, source type) "
                 "are the cleaner ones; page–answer similarity may be partly circular.")

    # By topic.
    if by_topic:
        C.section("By topic", icon="🗂️")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.topic_compare(by_topic, "cite_rate", "Cite-rate by topic"), width="stretch")
        with c2:
            st.plotly_chart(charts.topic_compare(by_topic, "recall_strict_10", "Strict recall@10 by topic"), width="stretch")

        feat_opts = {FEATURE_LABELS.get(k, k): k for k in PRE_ANSWER_FEATURES}
        chosen = st.selectbox("Compare a pre-answer feature across topics", list(feat_opts))
        st.plotly_chart(charts.topic_feature_compare(by_topic, feat_opts[chosen], chosen), width="stretch")

        tabs = st.tabs([f"{question_sets.TOPIC_EMOJI.get(t,'•')} {t}" for t in by_topic])
        for tab, (t, info) in zip(tabs, by_topic.items()):
            with tab:
                ss_t = info.get("sample_sizes", {})
                C.metric_cards([
                    {"value": ss_t.get("n_candidates", 0), "label": "candidates"},
                    {"value": ss_t.get("n_cited", 0), "label": "cited"},
                    {"value": C.pct(info.get("cite_rate")), "label": "cite-rate"},
                    {"value": C.pct(info.get("recall", {}).get("strict", {}).get("10")), "label": "strict recall@10"},
                ])
                sb = pd.DataFrame(info.get("source_breakdown") or [])
                if not sb.empty:
                    cc1, cc2 = st.columns(2)
                    cc1.plotly_chart(charts.cite_rate_by_source(sb), width="stretch")
                    cc2.plotly_chart(charts.source_stacked(sb), width="stretch")
                off = info.get("official") or {}
                if off:
                    st.plotly_chart(charts.official_bar(off), width="stretch")
                gs = pd.DataFrame(info.get("group_stats") or [])
                if not gs.empty:
                    st.dataframe(gs[[c for c in _GS_COLS if c in gs.columns]], width="stretch", hide_index=True)

    # By intent.
    by_intent = agg.get("by_intent") or {}
    if by_intent:
        C.section("By intent", icon="🎯")
        idf = pd.DataFrame([{"intent": k, **v} for k, v in by_intent.items()]).sort_values("cite_rate", ascending=False)
        st.dataframe(idf, width="stretch", hide_index=True)
        st.caption("Cite-rate per intent across all topics. 'Official Source Test' / 'Brand…' intents are "
                   "interesting to watch for official-source behavior.")

    # Overall pooled comparison + correlation.
    C.section("Overall cited vs non-cited (pooled)",
              "Median + Mann-Whitney U p-value + 95% bootstrap CI of the median difference.", "📈")
    gs = pd.DataFrame(agg.get("group_stats") or [])
    if not gs.empty:
        st.dataframe(gs[[c for c in _GS_COLS if c in gs.columns]], width="stretch", hide_index=True)
    corr = pd.DataFrame(agg.get("correlation") or [])
    if not corr.empty:
        st.plotly_chart(charts.recall_grouped(agg.get("recall") or {}), width="stretch")
        st.caption("Feature↔citation correlations are in the per-topic tables; small samples are noisy.")

    # Position-adjusted citation model (pooled across prompts, clustered by run).
    C.section("Position-adjusted citation model (pooled)",
              "Multivariate LPM clustered by prompt — Δ probability of citation per feature, rank held fixed.", "📐")
    C.regression_block(agg.get("regression"))

    # Question clusters across topics.
    feats = study.get("features")
    if feats:
        C.section("Question clusters (across topics)",
                  "Group questions by the websites they cite — do they cluster by topic, or cut across?", "🔠")
        n_q = len({(r.get("record_id") or r.get("run_id")) for r in feats})
        if n_q >= 3:
            kk = st.slider("Number of clusters", 2, min(8, n_q - 1), min(3, n_q - 1), key="topic_k")
            st.plotly_chart(charts.question_domain_heatmap(
                cluster.clustered_question_matrix(feats, "cited", kk),
                "Questions × top cited domains (rows grouped by cluster)"), width="stretch")
            crows = [{
                "cluster": c["cluster"], "size": c["size"],
                "topics": ", ".join(sorted({m["topic"] for m in c["members"] if m["topic"]})),
                "top_domains": ", ".join(d["domain"] for d in c["top_domains"][:5]),
            } for c in cluster.cluster_questions(feats, "cited", kk)]
            st.dataframe(pd.DataFrame(crows), width="stretch", hide_index=True)
        else:
            st.caption("Run ≥3 prompts to enable clustering.")

    # Export.
    C.section("Export", icon="📤")
    cexp = st.columns(2)
    cexp[0].download_button("⬇️ Topic-study report (Markdown)", report.batch_markdown_report(study),
                            f"{study['batch_id']}_topics.md", "text/markdown", width="stretch")
    cexp[1].download_button("⬇️ Pooled features (CSV)", report.batch_features_csv(study),
                            f"{study['batch_id']}_features.csv", "text/csv", width="stretch")
