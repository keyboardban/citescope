"""ChatGPT source audit and content-econometrics QA entry point."""

from __future__ import annotations

import streamlit as st

from src import config, storage
from src import econometrics_qa as qa_data
from ui.state import init_state
from ui.theme import inject_css
from ui.views import chatgpt, econometrics_qa

st.set_page_config(
    page_title="CiteScope ChatGPT Content Audit",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
config.ensure_dirs()
init_state()

MODES = ["ChatGPT Bright Data Audit", "Content Econometrics QA"]


def _clear_cg() -> None:
    st.session_state.update(cg_run=None, cg_pages={}, cg_features=None, cg_chunks={}, cg_analysis=None)


def _open_previous_area_condo() -> None:
    st.session_state["audit_mode"] = MODES[1]
    st.session_state["qa_data_source"] = "Previous Area Condo 500"


def _sidebar() -> tuple[str, str, str]:
    package_dir = ""
    prompt_manifest_path = ""
    with st.sidebar:
        st.markdown("## CiteScope")
        mode = st.radio("Audit mode", MODES,
                        key="audit_mode")

        st.divider()
        if mode == MODES[0]:
            st.caption("Upload and analyze observable ChatGPT Bright Data source records.")
            st.button(
                "Use previous Area Condo 500 data",
                width="stretch",
                on_click=_open_previous_area_condo,
            )
        else:
            st.caption("Inspect scrape quality, prompts, taxonomy, and econometric outputs.")
            preset = qa_data.previous_area_condo_preset()
            data_source = st.selectbox(
                "QA data source",
                [preset.label, "Custom econometrics package"],
                key="qa_data_source",
            )
            if data_source == preset.label:
                package_dir = str(preset.package_dir)
                prompt_manifest_path = str(preset.prompt_manifest_path or "")
                st.caption(
                    f"500 prompts | 2,881 crawler snapshots | "
                    f"manifest {'available' if preset.prompt_manifest_path else 'not found'}"
                )
            else:
                package_dir = st.text_input(
                    "Package directory",
                    value=str(qa_data.default_package_dir()),
                    key="qa_custom_package_dir",
                ).strip()
                prompt_manifest_path = st.text_input(
                    "Prompt manifest (optional)",
                    key="qa_custom_prompt_manifest",
                ).strip()

        st.divider()
        st.markdown("**Scraping credentials** _(from `.env`)_")
        for name in ("BRIGHTDATA_API_KEY", "APIFY_TOKEN"):
            ok = config.secret_present(name)
            st.markdown(("Available: " if ok else "Missing: ") + f"`{name}`")
        st.caption("The QA explorer is read-only and does not spend scraping credits.")

        st.divider()
        if mode == MODES[0]:
            cg = st.session_state.get("cg_run")
            if cg:
                st.caption(f"Active file: `{(cg.get('source_file_name') or '')[:24]}` · {cg.get('n_records', 0)} records")
                if st.button("Clear ChatGPT run", width="stretch"):
                    _clear_cg()
                    st.rerun()
            with st.expander("Previous Bright Data runs"):
                cruns = storage.list_chatgpt_runs(20)
                if cruns:
                    labels = {f"{r['run_id'][:18]} · {(r.get('source_file_name') or '')[:18]}": r["run_id"] for r in cruns}
                    pick = st.selectbox("Load a saved ChatGPT run", list(labels), index=None, placeholder="select")
                    if pick and st.button("Load ChatGPT run", width="stretch"):
                        loaded = storage.load_chatgpt_run(labels[pick])
                        if loaded:
                            _clear_cg()
                            st.session_state["cg_run"] = loaded
                            st.rerun()
                else:
                    st.caption("No saved Bright Data runs yet.")
        else:
            st.caption("Econometrics data is resolved through `CITESCOPE_RESEARCH_DATA_DIR`.")
            st.caption(f"Manual reviews: {len(storage.list_econometrics_reviews()):,}")

        with st.expander("Cache and local data"):
            st.caption(f"DB: `{config.DB_PATH.name}` · exports: `data/exports/`")
            if st.button("Clear API cache", width="stretch"):
                n = storage.cache_clear()
                st.success(f"Cleared {n} cached entries.")

        st.divider()
        st.caption(
            "Black-box observational audit. Cited and more-only describe sources exposed in the "
            "ChatGPT Bright Data output, not ChatGPT's complete internal retrieval set."
        )

    return mode, package_dir, prompt_manifest_path


def main() -> None:
    mode, package_dir, prompt_manifest_path = _sidebar()
    try:
        if mode == MODES[0]:
            chatgpt.render()
        else:
            econometrics_qa.render(package_dir, prompt_manifest_path)
    except Exception as exc:  # keep the app alive; show the error in-page
        st.error(f"Something went wrong rendering **{mode}**: {type(exc).__name__}: {exc}")
        with st.expander("Traceback"):
            import traceback
            st.code(traceback.format_exc())


main()
