"""Session-state management and cached API clients."""

from __future__ import annotations

import streamlit as st

from src import apify_runner, config, gemini_client
from src.pipeline import (
    make_sim_engine,
    stage_analyze,
    stage_features,
    stage_match,
)


# --------------------------------------------------------------------------- #
# cached clients (built once per key, reused across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _build_gemini(api_key: str):
    return gemini_client.build_client(api_key)


@st.cache_resource(show_spinner=False)
def _build_apify(token: str):
    return apify_runner.build_client(token)


def get_clients() -> dict:
    gk = config.get_secret("GEMINI_API_KEY")
    tk = config.get_secret("APIFY_TOKEN")
    return {
        "gemini": _build_gemini(gk) if gk else None,
        "apify": _build_apify(tk) if tk else None,
    }


# --------------------------------------------------------------------------- #
# defaults + init
# --------------------------------------------------------------------------- #
def default_inputs() -> dict:
    return {
        "prompt": "What are the best tailors in Bangkok for custom suits?",
        "gemini": {
            "model": config.DEFAULT_GEMINI_MODEL,
            "temperature": config.DEFAULT_TEMPERATURE,
            "grounding": True,
            "system_prompt": None,
        },
        "serp": {
            "top_k": config.DEFAULT_SERP_TOP_K,
            "country": config.DEFAULT_COUNTRY,
            "language": config.DEFAULT_LANGUAGE,
            "selected_queries": [],
        },
        "scrape": {
            "scope": "top_k",
            "top_k": config.DEFAULT_SCRAPE_TOP_K,
            "selected_urls": [],
            "use_cache": True,
            "crawler_type": config.DEFAULT_CRAWLER_TYPE,
        },
        "analysis": {
            "include_weak": False,
            "similarity_method": config.SIMILARITY_METHODS[0],
            "embedding_model": config.DEFAULT_EMBED_MODEL,
        },
    }


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("run", None)
    ss.setdefault("inputs", default_inputs())
    ss.setdefault("nav", "Overview")
    ss.setdefault("last_error", None)


def get_run() -> dict | None:
    return st.session_state.get("run")


def set_run(run: dict) -> None:
    st.session_state["run"] = run


def has_stage(*keys: str) -> bool:
    run = get_run()
    return bool(run) and all(run.get(k) for k in keys)


# --------------------------------------------------------------------------- #
# recompute matching -> features -> analysis (cheap for lexical similarity)
# --------------------------------------------------------------------------- #
def recompute_downstream() -> None:
    run = get_run()
    inp = st.session_state["inputs"]
    if not run or not run.get("gemini") or not run.get("serp"):
        return
    clients = get_clients()
    matching = stage_match(run["gemini"], run["serp"], run.get("scrape"), inp["analysis"])
    sim = make_sim_engine(
        inp["analysis"]["similarity_method"], clients.get("gemini"),
        inp["analysis"]["embedding_model"],
    )
    feat = stage_features(
        run["gemini"], matching, run.get("scrape"), sim,
        fallback_query=run.get("inputs", {}).get("prompt", inp["prompt"]),
    )
    run["matching"] = matching
    run["features"] = feat["features"]
    run["chunks"] = feat["chunks"]
    run["analysis"] = stage_analyze(run)
    set_run(run)
