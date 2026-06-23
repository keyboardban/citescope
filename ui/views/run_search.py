"""Run AI Search: grounded Gemini call + observable trace display."""

from __future__ import annotations

import copy

import pandas as pd
import streamlit as st

from src import config
from src.ids import new_run_id, now_iso
from src.pipeline import run_full, stage_gemini
from src.url_utils import is_redirect_wrapper

from .. import components as C
from ..state import get_clients, get_run, recompute_downstream, set_run


def _snapshot_inputs() -> dict:
    return copy.deepcopy(st.session_state["inputs"])


def _reset_downstream(run: dict) -> None:
    for k in ("serp", "scrape", "matching", "analysis"):
        run[k] = None
    run["features"] = []
    run["chunks"] = {}


def _run_gemini_only(inputs: dict, clients: dict) -> None:
    with st.spinner("Querying Gemini with Google Search Grounding…"):
        run_id = (get_run() or {}).get("run_id") or new_run_id()
        trace = stage_gemini(clients["gemini"], inputs, run_id, use_cache=inputs["scrape"]["use_cache"])
    run = get_run() or {}
    run.update({
        "run_id": run.get("run_id") or run_id, "created_at": now_iso(),
        "is_demo": False, "inputs": _snapshot_inputs(), "gemini": trace,
    })
    _reset_downstream(run)
    set_run(run)
    if trace.get("error"):
        st.error(f"Gemini error: {trace['error']}")
    else:
        st.success("Gemini run complete" + (" (from cache)" if trace.get("cached") else ""))


def _run_full(inputs: dict, clients: dict) -> None:
    bar = st.progress(0.0, text="Starting…")
    try:
        run = run_full(clients, _snapshot_inputs(), progress=lambda s, f: bar.progress(min(f, 1.0), text=s),
                       use_cache=inputs["scrape"]["use_cache"])
    except Exception as exc:  # surface, don't crash the app
        bar.empty()
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        return
    bar.progress(1.0, text="Done")
    set_run(run)
    st.success(f"Full audit complete — run `{run['run_id']}`. Explore the other sections →")


def _show_gemini(run: dict) -> None:
    g = run["gemini"]
    st.divider()
    C.section("Observable trace", icon="🔎")
    if g.get("error"):
        st.error(f"⚠️ Gemini stage: {g['error']}")
    queries = g.get("search_queries", []) or []
    cites = g.get("citations", []) or []
    C.metric_cards([
        {"value": len(queries), "label": "search queries"},
        {"value": len(cites), "label": "citation URLs"},
        {"value": f"{len(g.get('output_text','')):,}", "label": "answer chars"},
        {"value": "on" if g.get("grounding") else "off", "label": "grounding"},
    ])

    t_ans, t_q, t_c, t_raw = st.tabs(["💬 Answer", "🔍 Search queries", "🔗 Citations", "🧱 Grounding / raw"])

    with t_ans:
        if g.get("output_text"):
            st.markdown(g["output_text"])
        else:
            st.warning("No answer text returned.")
            if g.get("finish_reason"):
                st.caption(f"finish_reason: `{g['finish_reason']}`")
            if g.get("prompt_feedback"):
                st.caption(f"prompt_feedback: `{g['prompt_feedback']}`")
            st.info("Likely causes: the selected **model isn't available** for your account/region, "
                    "grounding **quota**, or a **blocked** prompt. Try another model in the selector "
                    "above (e.g. `gemini-2.5-pro` or `gemini-3-flash`), then re-run.")

    with t_q:
        if queries:
            for q in queries:
                tag = " · _fallback_" if q.get("is_fallback") else ""
                st.markdown(f"- **{q.get('query','')}**{tag}")
            C.proxy_note("These are the queries Gemini exposed. We do not fabricate queries; "
                         "if none were exposed you can fall back to the prompt in the SERP step.")
        else:
            st.info("Gemini exposed no search queries. Use the prompt as a fallback query in "
                    "**SERP Reconstruction**.")

    with t_c:
        if cites:
            df = pd.DataFrame([{
                "title": c.get("title", ""),
                "domain": c.get("domain", ""),
                "resolved_url": c.get("resolved_url", ""),
                "redirect_wrapper": is_redirect_wrapper(c.get("raw_uri", "")),
            } for c in cites])
            st.dataframe(df, width="stretch", hide_index=True)
            C.proxy_note("Citation URLs come from grounding metadata. Vertex redirect wrappers are "
                         "resolved to the real publisher URL before matching.")
        else:
            st.info("No citation URLs were exposed for this run.")

    with t_raw:
        sep = g.get("search_entry_point_html")
        if sep:
            with st.expander("Search entry point (Google-rendered)"):
                st.components.v1.html(sep, height=120, scrolling=True)
        with st.expander("Raw response (audit trail)"):
            st.json(g.get("raw") or {"note": "no raw payload"})


def render() -> None:
    inputs = st.session_state["inputs"]
    C.section("Run AI Search",
              "Send your prompt to Gemini with Google Search Grounding and capture the observable trace.", "🤖")
    C.limitation_box()

    inputs["prompt"] = st.text_area("Prompt", value=inputs["prompt"], height=90, key="w_prompt")

    g = inputs["gemini"]
    c1, c2, c3 = st.columns([2, 1, 1])
    models = config.GEMINI_MODELS
    g["model"] = c1.selectbox("Model", models,
                              index=models.index(g["model"]) if g["model"] in models else 0,
                              help="The proven grounding path uses generate_content; switch models if your account differs.")
    g["temperature"] = c2.slider("Temperature", 0.0, 1.0, float(g["temperature"]), 0.05)
    g["grounding"] = c3.toggle("Grounding", value=bool(g["grounding"]),
                               help="Google Search Grounding. Off = no citations to audit.")
    with st.expander("System prompt (optional)"):
        sp = st.text_area("System instruction", value=g.get("system_prompt") or "", key="w_sys")
        g["system_prompt"] = sp.strip() or None

    clients = get_clients()
    ready = clients.get("gemini") is not None
    b1, b2, _ = st.columns([1.1, 1, 2])
    run_one = b1.button("▶ Run grounded query", type="primary", disabled=not ready, width="stretch")
    run_all = b2.button("⚡ Run full audit", disabled=not ready, width="stretch",
                        help="Runs every stage end-to-end with the current settings.")
    if not ready:
        st.warning("Set `GEMINI_API_KEY` in `.env` to run live. You can still explore the **demo run** "
                   "from the sidebar.")

    if run_one:
        _run_gemini_only(inputs, clients)
    if run_all:
        _run_full(inputs, clients)
        if get_run():
            recompute_downstream()

    run = get_run()
    if run and run.get("gemini"):
        _show_gemini(run)
