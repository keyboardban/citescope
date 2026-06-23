"""Pipeline orchestration.

Each stage is a cache-aware function that can be called on its own (the
interactive multi-page flow) or chained by `run_full` (the one-click audit).
Expensive Gemini/Apify calls are cached in SQLite so they are never repeated by
accident; pass force=True to refresh.
"""

from __future__ import annotations

from typing import Callable

from . import apify_runner, gemini_client, storage
from .analysis import (
    correlation_with_citation,
    features_df,
    group_compare,
    official_compare,
    source_breakdown,
    summary_metrics,
)
from .config import (
    APIFY_SCRAPER_ACTOR,
    APIFY_SERP_ACTOR,
)
from .features import build_features
from .ids import new_run_id, now_iso, stable_hash
from .matching import match_all, unique_candidates
from .similarity import SimilarityEngine
from .url_utils import domain, is_redirect_wrapper, normalize_url, resolve_redirect

ProgressCB = Callable[[str, float], None]


# --------------------------------------------------------------------------- #
# similarity engine factory
# --------------------------------------------------------------------------- #
def make_sim_engine(method: str, gem_client=None, embed_model: str = "text-embedding-004") -> SimilarityEngine:
    wants_embed = bool(method) and ("embed" in method.lower())
    if wants_embed and gem_client is not None:
        def embed_fn(texts):
            return gemini_client.embed_texts(gem_client, texts, embed_model)
        return SimilarityEngine("embedding", embed_fn)
    return SimilarityEngine("lexical")


# --------------------------------------------------------------------------- #
# stage 1 — Gemini grounded run + citation redirect resolution
# --------------------------------------------------------------------------- #
def _resolve_citations(trace: dict, use_cache: bool = True) -> dict:
    for c in trace.get("citations", []):
        raw = c.get("raw_uri", "")
        if is_redirect_wrapper(raw):
            key = "redirect:" + raw
            r = storage.cache_get(key) if use_cache else None
            if not r:
                r = resolve_redirect(raw) or raw
                storage.cache_set(key, r, stage="redirect")
            c["resolved_url"] = r
        else:
            c["resolved_url"] = raw
        c["domain"] = domain(c["resolved_url"])
    return trace


def stage_gemini(gem_client, inputs: dict, run_id: str = "", use_cache: bool = True, force: bool = False) -> dict:
    g = inputs["gemini"]
    prompt = inputs["prompt"]
    key = "gemini:" + stable_hash(
        {"prompt": prompt, "model": g["model"], "temp": g["temperature"],
         "grounding": g["grounding"], "system": g.get("system_prompt")}
    )
    if use_cache and not force:
        cached = storage.cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    if gem_client is None:
        raise RuntimeError("Gemini client unavailable — set GEMINI_API_KEY in .env")

    trace = gemini_client.run_grounded(
        gem_client, prompt, g["model"], g["temperature"], g["grounding"], g.get("system_prompt")
    )
    trace = _resolve_citations(trace, use_cache=use_cache)
    trace["search_queries"] = [{"query": q, "is_fallback": False} for q in trace.get("search_queries", [])]
    trace["ts"] = now_iso()
    trace["cached"] = False
    if run_id and trace.get("raw"):
        storage.save_raw(run_id, "gemini_raw", trace["raw"])
    if not trace.get("error"):  # never cache a failed/empty run
        storage.cache_set(key, trace, stage="gemini")
    return trace


# --------------------------------------------------------------------------- #
# stage 2 — reconstructed SERP
# --------------------------------------------------------------------------- #
def stage_serp(apify_client, queries: list[str], serp_inputs: dict, run_id: str = "",
               use_cache: bool = True, force: bool = False) -> dict:
    actor = serp_inputs.get("actor", APIFY_SERP_ACTOR)
    key = "serp:" + stable_hash(
        {"q": sorted(queries), "k": serp_inputs["top_k"],
         "country": serp_inputs["country"], "lang": serp_inputs["language"], "actor": actor}
    )
    if use_cache and not force:
        cached = storage.cache_get(key)
        if cached:
            cached["cached"] = True
            return cached

    if apify_client is None:
        raise RuntimeError("Apify client unavailable — set APIFY_TOKEN in .env")

    res = apify_runner.run_serp(
        apify_client, queries, serp_inputs["top_k"], serp_inputs["country"],
        serp_inputs["language"], actor,
    )
    res["cached"] = False
    if run_id:
        storage.save_raw(run_id, "serp_items", res.get("items", []))
    if res.get("candidates"):  # don't cache an empty/failed SERP
        storage.cache_set(key, res, stage="serp")
    return res


# --------------------------------------------------------------------------- #
# stage 3 — scraping (per-URL cache)
# --------------------------------------------------------------------------- #
def select_scrape_urls(scope: str, cands: list[dict], cited_ids, top_k: int,
                       selected_urls: list[str] | None) -> list[str]:
    cited = set(cited_ids or [])
    if scope == "selected":
        return list(selected_urls or [])
    if scope == "cited":
        return [c["url"] for c in cands if c["candidate_id"] in cited]
    if scope == "all":
        return [c["url"] for c in cands]
    return [c["url"] for c in sorted(cands, key=lambda x: x["best_rank"])[:top_k]]


def stage_scrape(apify_client, urls: list[str], scrape_inputs: dict, run_id: str = "",
                 use_cache: bool = True, force: bool = False) -> dict:
    crawler = scrape_inputs.get("crawler_type", "cheerio")
    actor = scrape_inputs.get("actor", APIFY_SCRAPER_ACTOR)
    pages: dict[str, dict] = {}
    to_fetch: list[str] = []

    for u in urls:
        nurl = normalize_url(u)
        if not nurl:
            continue
        ck = "scrape:" + stable_hash({"u": nurl, "crawler": crawler})
        cached = storage.cache_get(ck) if (use_cache and not force) else None
        if cached:
            pages[nurl] = cached
        else:
            to_fetch.append(u)

    meta = {"run_id": None, "dataset_id": None, "actor": actor, "ts": now_iso(),
            "fetched": len(to_fetch), "from_cache": len(pages)}

    if to_fetch:
        if apify_client is None:
            raise RuntimeError("Apify client unavailable — set APIFY_TOKEN in .env")
        res = apify_runner.run_scrape(apify_client, to_fetch, crawler, actor)
        for p in res.get("pages", []):
            nurl = normalize_url(p.get("url") or p.get("final_url") or "")
            if not nurl:
                continue
            pages[nurl] = p
            if p.get("status") == "success":  # don't cache failed scrapes (allow retry)
                storage.cache_set("scrape:" + stable_hash({"u": nurl, "crawler": crawler}), p, stage="scrape")
        meta.update({"run_id": res.get("run_id"), "dataset_id": res.get("dataset_id"),
                     "status": res.get("status"), "error": res.get("error")})
        if run_id:
            storage.save_raw(run_id, "scrape_items", res.get("items", []))

    # Mark any requested URL that produced no page as failed (so the UI is honest).
    for u in urls:
        nurl = normalize_url(u)
        if nurl and nurl not in pages:
            pages[nurl] = {"url": u, "final_url": u, "status": "failed",
                           "error": "no result returned", "text": "", "markdown": "",
                           "title": "", "canonical_url": None, "published_date": None}

    return {"pages": pages, "apify": meta, "scope": scrape_inputs.get("scope"),
            "cached": len(to_fetch) == 0}


# --------------------------------------------------------------------------- #
# stage 4 — matching, stage 5 — features, stage 6 — analysis
# --------------------------------------------------------------------------- #
def stage_match(gemini: dict, serp: dict, scrape: dict | None, analysis_inputs: dict) -> dict:
    cands = unique_candidates(serp.get("candidates", []))
    pages = (scrape or {}).get("pages", {})
    res = match_all(gemini.get("citations", []), cands, pages, analysis_inputs.get("include_weak", False))
    res["unique_candidates"] = cands
    return res


def stage_features(gemini: dict, matching: dict, scrape: dict | None,
                   sim_engine: SimilarityEngine, fallback_query: str = "") -> dict:
    cands = matching.get("unique_candidates") or unique_candidates([])
    pages = (scrape or {}).get("pages", {})
    return build_features(cands, pages, matching, gemini.get("output_text", ""), sim_engine, fallback_query)


def stage_analyze(run: dict) -> dict:
    df = features_df(run.get("features") or [])
    return {
        "summary": summary_metrics(run),
        "group_compare": group_compare(df).to_dict(orient="records"),
        "source_breakdown": source_breakdown(df).to_dict(orient="records"),
        "official": official_compare(df),
        "correlation": correlation_with_citation(df).to_dict(orient="records"),
    }


# --------------------------------------------------------------------------- #
# full one-click orchestration
# --------------------------------------------------------------------------- #
def run_full(clients: dict, inputs: dict, progress: ProgressCB | None = None,
             use_cache: bool = True, force: bool = False) -> dict:
    def p(stage: str, frac: float) -> None:
        if progress:
            progress(stage, frac)

    run_id = new_run_id()
    p("Querying Gemini (grounded)…", 0.05)
    gemini = stage_gemini(clients.get("gemini"), inputs, run_id, use_cache, force)

    observed = [q["query"] for q in gemini.get("search_queries", [])]
    queries = inputs["serp"].get("selected_queries") or observed or [inputs["prompt"]]
    used_fallback = not observed and not inputs["serp"].get("selected_queries")

    p("Reconstructing SERP (Apify)…", 0.3)
    serp = stage_serp(clients.get("apify"), queries, inputs["serp"], run_id, use_cache, force)

    cands = unique_candidates(serp.get("candidates", []))
    pre = match_all(gemini.get("citations", []), cands, {}, inputs["analysis"].get("include_weak", False))
    urls = select_scrape_urls(
        inputs["scrape"].get("scope", "top_k"), cands, pre["cited_candidate_ids"],
        inputs["scrape"].get("top_k", 12), inputs["scrape"].get("selected_urls"),
    )

    p("Scraping candidate pages (Apify)…", 0.55)
    scrape = stage_scrape(clients.get("apify"), urls, inputs["scrape"], run_id,
                          use_cache, force) if urls else {"pages": {}, "apify": {}, "cached": True}

    p("Matching citations…", 0.8)
    matching = stage_match(gemini, serp, scrape, inputs["analysis"])

    p("Extracting features…", 0.9)
    sim = make_sim_engine(inputs["analysis"].get("similarity_method", "lexical"),
                          clients.get("gemini"), inputs["analysis"].get("embedding_model", "text-embedding-004"))
    feat = stage_features(gemini, matching, scrape, sim, fallback_query=inputs["prompt"])

    run = assemble_run(run_id, inputs, gemini, serp, scrape, matching, feat, used_fallback)
    p("Done", 1.0)
    storage.save_run(run)
    return run


def assemble_run(run_id, inputs, gemini, serp, scrape, matching, feat, used_fallback=False) -> dict:
    run = {
        "run_id": run_id,
        "created_at": now_iso(),
        "is_demo": False,
        "used_fallback_query": used_fallback,
        "inputs": inputs,
        "gemini": gemini,
        "serp": serp,
        "scrape": scrape,
        "matching": matching,
        "features": feat.get("features", []),
        "chunks": feat.get("chunks", {}),
    }
    run["analysis"] = stage_analyze(run)
    return run
