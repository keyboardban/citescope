"""AI Search Citation Audit — Streamlit entry point.

A black-box observational audit: compare websites Gemini cites against an
independently reconstructed SERP for the same search queries.
"""

from __future__ import annotations

import streamlit as st

from src import config, storage
from src.demo import make_demo_run
from ui import components as C
from ui.state import get_run, init_state, set_run
from ui.theme import inject_css
from ui.views import (
    content_visualizer,
    feature_analysis,
    matching,
    overview,
    report,
    run_search,
    scraping,
    serp,
)

st.set_page_config(
    page_title="AI Search Citation Audit",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
config.ensure_dirs()
init_state()

VIEWS = {
    "Overview": ("🧭", overview.render),
    "Run AI Search": ("🤖", run_search.render),
    "SERP Reconstruction": ("🌐", serp.render),
    "Web Scraping": ("🕸️", scraping.render),
    "Citation Matching": ("🎯", matching.render),
    "Content Visualizer": ("🔬", content_visualizer.render),
    "Feature Analysis": ("📈", feature_analysis.render),
    "Report / Export": ("📤", report.render),
}


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🔎 Citation Audit")
        st.caption("Black-box AI-search citation audit")

        nav = st.radio(
            "Navigate", list(VIEWS),
            format_func=lambda k: f"{VIEWS[k][0]}  {k}",
            key="nav_radio", label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**API keys** _(from `.env`)_")
        for name in config.REQUIRED_SECRETS:
            ok = config.secret_present(name)
            st.markdown(("✅ " if ok else "⛔ ") + f"`{name}`")
        if not all(config.secret_present(n) for n in config.REQUIRED_SECRETS):
            st.caption("Missing keys → use the demo run below.")

        st.divider()
        if st.button("🧪 Load demo run", width="stretch",
                     help="Explore the full dashboard with synthetic data — no API calls."):
            set_run(make_demo_run())
            st.rerun()

        run = get_run()
        if run:
            tag = "🧪 demo" if run.get("is_demo") else "live"
            st.caption(f"Active: `{run.get('run_id','')[:20]}` · {tag}")
            if st.button("🗑️ Clear current run", width="stretch"):
                set_run(None)
                st.rerun()

        with st.expander("📂 Previous runs"):
            runs = storage.list_runs(20)
            if runs:
                labels = {f"{r['run_id'][:18]} · {(r.get('prompt') or '')[:22]}": r["run_id"] for r in runs}
                pick = st.selectbox("Load a saved run", list(labels), index=None, placeholder="select a run…")
                if pick and st.button("Load run", width="stretch"):
                    loaded = storage.load_run(labels[pick])
                    if loaded:
                        set_run(loaded)
                        st.rerun()
            else:
                st.caption("No saved runs yet.")

        with st.expander("⚙️ Cache & data"):
            st.caption(f"DB: `{config.DB_PATH.name}` · exports: `data/exports/`")
            if st.button("Clear API cache", width="stretch"):
                n = storage.cache_clear()
                st.success(f"Cleared {n} cached entries.")

        st.divider()
        st.caption(config.DISCLAIMER_SHORT)

    return st.session_state.get("nav_radio", "Overview")


def main() -> None:
    nav = _sidebar()
    try:
        VIEWS[nav][1]()
    except Exception as exc:  # keep the app alive; show the error in-page
        st.error(f"Something went wrong rendering **{nav}**: {type(exc).__name__}: {exc}")
        with st.expander("Traceback"):
            import traceback
            st.code(traceback.format_exc())


main()
