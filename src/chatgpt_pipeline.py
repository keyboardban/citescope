"""ChatGPT Bright Data Source Audit pipeline.

Reuses the shared engine (scraping, chunking, similarity, source-type, analysis)
but with ChatGPT framing: compare **cited sources** vs **more-only**
(shown-but-not-cited) sources. No SERP reconstruction, no recall@K.
"""

from __future__ import annotations

import pandas as pd

from . import analysis, pipeline
from .config import MAX_SIM_CHARS
from .chunking import chunk_text, extract_headings
from .features import _freshness_days, _parse_date
from .similarity import SimilarityEngine, summarize_scores
from .source_type import brand_official_candidate, classify
from .url_utils import normalize_url

# Feature columns (mirrors gemini split: pre-answer = non-circular, post = circular)
CHATGPT_PRE = [
    "title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity",
    "max_chunk_prompt_similarity", "word_count", "char_count", "heading_count",
    "freshness_days", "source_position", "observed_rank",
]
CHATGPT_POST = ["page_answer_similarity", "max_chunk_answer_similarity"]
CHATGPT_NUMERIC = CHATGPT_PRE + CHATGPT_POST
CHATGPT_PHASE = {**{f: "pre_answer" for f in CHATGPT_PRE}, **{f: "post_output" for f in CHATGPT_POST}}
CHATGPT_LABELS = {
    "title_prompt_similarity": "Title–prompt similarity",
    "description_prompt_similarity": "Description–prompt similarity",
    "page_prompt_similarity": "Page–prompt similarity",
    "max_chunk_prompt_similarity": "Best chunk–prompt similarity",
    "page_answer_similarity": "Page–answer similarity",
    "max_chunk_answer_similarity": "Best chunk–answer similarity",
    "word_count": "Word count", "char_count": "Char count", "heading_count": "Heading count",
    "freshness_days": "Age (days)", "source_position": "Source position", "observed_rank": "Observed rank",
}


# --------------------------------------------------------------------------- #
# source flattening / scrape selection
# --------------------------------------------------------------------------- #
def flatten_sources(run: dict) -> list[dict]:
    """All sources across records, each enriched with its record's prompt/answer."""
    rows: list[dict] = []
    for rec in run.get("records", []):
        ctx = {
            "run_id": run.get("run_id"), "record_id": rec.get("record_id"),
            "prompt": rec.get("prompt", ""), "answer_text": rec.get("answer_text", ""),
            "web_search_query": rec.get("web_search_query", []),
            "intent": rec.get("intent") or "", "topic": rec.get("topic") or "",
            "prompt_id": rec.get("prompt_id"), "expected_source_types": rec.get("expected_source_types") or [],
        }
        for s in rec.get("sources", []):
            rows.append({**s, **ctx})
    return rows


def select_scrape_urls(run: dict, scope: str = "all", selected_norm: list[str] | None = None) -> list[str]:
    sources = flatten_sources(run)
    selected = set(selected_norm or [])
    seen, urls = set(), []
    for s in sources:
        nurl = s.get("normalized_url")
        if not nurl or nurl in seen:
            continue
        keep = (
            scope == "all"
            or (scope == "cited" and s["cited_label"] == 1)
            or (scope == "more_only" and s["cited_label"] == 0)
            or (scope == "selected" and nurl in selected)
        )
        if keep:
            seen.add(nurl)
            urls.append(s["url"])
    return urls


def scrape_sources(clients: dict, urls: list[str], scrape_inputs: dict, run_id: str = "",
                   use_cache: bool = True) -> dict:
    """Reuse the shared Apify content-crawler stage to scrape source URLs."""
    if not urls:
        return {"pages": {}, "apify": {}, "cached": True}
    return pipeline.stage_scrape(clients.get("apify"), urls, scrape_inputs, run_id, use_cache=use_cache)


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def build_features(run: dict, pages: dict[str, dict], sim_engine: SimilarityEngine) -> dict:
    """One feature row per source (cited vs more-only). Returns features + chunks."""
    features: list[dict] = []
    chunks_map: dict[str, list[dict]] = {}

    for s in flatten_sources(run):
        prompt = s.get("prompt") or (s.get("web_search_query") or [""])[0]
        answer = s.get("answer_text", "")
        page = pages.get(s["normalized_url"]) or {}
        scraped = page.get("status") == "success"
        text = (page.get("text") or page.get("markdown") or "") if scraped else ""
        markdown = (page.get("markdown") or text) if scraped else ""

        stype, institutional = classify(s["url"])
        brand = brand_official_candidate(s["url"], s.get("title", ""), prompt, answer)
        headings = extract_headings(markdown) if markdown else []

        row = {
            "run_id": s.get("run_id"), "record_id": s.get("record_id"), "source_id": s.get("source_id"),
            "prompt": prompt, "answer_text": answer[:1500],
            "url": s["url"], "normalized_url": s["normalized_url"], "domain": s.get("domain", ""),
            "title": s.get("title", ""), "description": s.get("description", ""),
            "intent": s.get("intent") or "", "topic": s.get("topic") or "",
            "expected_source_types": s.get("expected_source_types") or [],
            # labels
            "cited_label": s["cited_label"], "cited": s["cited_label"],   # 'cited' alias -> analysis reuse
            "source_group": s.get("source_group", "more_only"),
            "source_origin": s.get("source_origin", ""),
            "source_position": s.get("source_position"), "observed_rank": s.get("observed_rank"),
            # pre-answer
            "title_prompt_similarity": sim_engine.score(s.get("title", ""), prompt),
            "description_prompt_similarity": sim_engine.score(s.get("description", ""), prompt),
            "page_prompt_similarity": None, "max_chunk_prompt_similarity": None,
            "source_type": stype, "institutional_official": institutional,
            "official_source": institutional, "brand_official_candidate": brand,
            "freshness_days": _freshness_days(_parse_date(s.get("date_published"))),
            "word_count": None, "char_count": None,
            "original_char_count": None, "used_char_count": None, "truncated": False,
            "heading_count": len(headings) if scraped else None,
            "scrape_success": scraped,
            # post-output (may be circular)
            "page_answer_similarity": None, "max_chunk_answer_similarity": None,
        }

        if scraped and text:
            original = len(text)
            used = text[:MAX_SIM_CHARS]
            row["word_count"] = len(text.split())
            row["char_count"] = original
            row["original_char_count"] = original
            row["used_char_count"] = len(used)
            row["truncated"] = original > MAX_SIM_CHARS
            row["page_prompt_similarity"] = sim_engine.score(used, prompt)
            row["page_answer_similarity"] = sim_engine.score(used, answer)
            if page.get("published_date"):
                row["freshness_days"] = _freshness_days(_parse_date(page.get("published_date")))
            chunks = chunk_text(markdown)
            if chunks:
                texts = [c["text"] for c in chunks]
                p_scores = sim_engine.score_many(prompt, texts)
                a_scores = sim_engine.score_many(answer, texts)
                chunks_map[s["source_id"]] = [
                    {**c, "prompt_sim": p, "answer_sim": a}
                    for c, p, a in zip(chunks, p_scores, a_scores)
                ]
                row["max_chunk_prompt_similarity"] = summarize_scores(p_scores)["max"]
                row["max_chunk_answer_similarity"] = summarize_scores(a_scores)["max"]

        features.append(row)

    return {"features": features, "chunks": chunks_map}


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def _top_domains(df: pd.DataFrame, cited_value: int, n: int = 8) -> list[dict]:
    if df.empty or "domain" not in df.columns:
        return []
    sub = df[df["cited"] == cited_value]
    if sub.empty:
        return []
    g = sub.groupby("domain").size().reset_index(name="count").sort_values("count", ascending=False).head(n)
    return g.to_dict(orient="records")


def analyze(features: list[dict]) -> dict:
    df = analysis.features_df(features, CHATGPT_NUMERIC)
    n_sources = len(df)
    n_cited = int(df["cited"].sum()) if not df.empty and "cited" in df else 0
    n_scraped = int(df["scrape_success"].sum()) if not df.empty and "scrape_success" in df else 0
    summary = {
        "n_records": int(df["record_id"].nunique()) if not df.empty and "record_id" in df else 0,
        "n_sources": n_sources,
        "n_cited": n_cited,
        "n_more_only": n_sources - n_cited,
        "n_scraped": n_scraped,
        "scrape_success_rate": round(n_scraped / n_sources, 3) if n_sources else 0.0,
    }
    return {
        "summary": summary,
        "group_compare": analysis.group_compare(df, CHATGPT_NUMERIC, CHATGPT_LABELS, CHATGPT_PHASE).to_dict(orient="records"),
        "source_breakdown": analysis.source_breakdown(df).to_dict(orient="records"),
        "official": analysis.official_compare(df),
        "correlation": analysis.correlation_with_citation(df, CHATGPT_NUMERIC, CHATGPT_LABELS, CHATGPT_PHASE).to_dict(orient="records"),
        "regression": analysis.econometric_analysis(
            df, CHATGPT_NUMERIC, CHATGPT_LABELS, CHATGPT_PHASE,
            position_col="source_position", position_fallbacks=["observed_rank"],
            cluster_key="record_id", context="chatgpt"),
        "length_sim_corr": analysis.length_sim_correlation(df, "page_answer_similarity"),
        "top_domains_cited": _top_domains(df, 1),
        "top_domains_more": _top_domains(df, 0),
    }


def recompute(run: dict, pages: dict, sim_engine: SimilarityEngine) -> dict:
    """Build features + analysis for the current run/pages (used after upload & scrape)."""
    feat = build_features(run, pages, sim_engine)
    return {"features": feat["features"], "chunks": feat["chunks"], "analysis": analyze(feat["features"])}


# --------------------------------------------------------------------------- #
# Intent -> Source Type analysis (requires a Prompt Manifest applied)
# --------------------------------------------------------------------------- #
# Map free-form expected_source_types tokens onto our taxonomy + official flags.
_EXP_SYN = {
    "gov": "government", "government": "government", "official": "official",
    "institutional": "government", "edu": "education", "education": "education",
    "academic": "education", "university": "education",
    "official_brand": "official_brand", "brand": "official_brand",
    "manufacturer": "official_brand", "oem": "official_brand", "vendor": "official_brand",
    "dealer": "ecommerce", "dealership": "ecommerce", "shop": "ecommerce", "store": "ecommerce",
    "ecommerce": "ecommerce", "e-commerce": "ecommerce", "marketplace": "ecommerce", "retail": "ecommerce",
    "review": "review", "reviews": "review", "news": "news", "media": "news", "press": "news",
    "forum": "forum", "community": "forum", "wiki": "reference", "reference": "reference",
    "encyclopedia": "reference", "blog": "blog", "video": "video", "social": "social",
    "hospital": "hospital", "clinic": "hospital", "medical": "hospital", "health": "hospital",
}


def _canon(token: str) -> str:
    t = (token or "").strip().lower().replace(" ", "_")
    return _EXP_SYN.get(t, t)


def _effective_canon_types(row: dict) -> set[str]:
    """Canonical type tags for a source row (source_type + official/brand flags)."""
    out = {_canon(row.get("source_type") or "unknown")}
    if row.get("institutional_official"):
        out.add("official")
        out.add("government")
    if row.get("brand_official_candidate"):
        out.add("official_brand")
    return out


def intent_source_long(features: list[dict]) -> list[dict]:
    """Aggregated counts per (intent, source_type, group=cited|more_only)."""
    from collections import Counter
    c: Counter = Counter()
    for r in features:
        intent = r.get("intent") or "Unspecified"
        st = r.get("source_type") or "unknown"
        grp = "cited" if r.get("cited") == 1 else "more_only"
        c[(intent, st, grp)] += 1
    return [{"intent": i, "source_type": s, "group": g, "n": n} for (i, s, g), n in sorted(c.items())]


def intent_summary(features: list[dict]) -> list[dict]:
    """Per-intent rollup: cite-rate + cited composition (official/review/ecommerce/forum)."""
    from collections import Counter, defaultdict
    by: dict[str, list] = defaultdict(list)
    for r in features:
        by[r.get("intent") or "Unspecified"].append(r)
    rows = []
    for intent, rs in sorted(by.items()):
        cited = [r for r in rs if r.get("cited") == 1]
        n, nc = len(rs), len(cited)

        def pct(pred):
            return round(sum(1 for r in cited if pred(r)) / nc, 3) if nc else 0.0

        top = Counter(r.get("source_type") for r in cited).most_common(1)
        rows.append({
            "intent": intent, "n_sources": n, "n_cited": nc,
            "cite_rate": round(nc / n, 3) if n else 0.0,
            "official_cited_pct": pct(lambda r: r.get("institutional_official") or r.get("brand_official_candidate")),
            "review_cited_pct": pct(lambda r: r.get("source_type") == "review"),
            "ecommerce_cited_pct": pct(lambda r: r.get("source_type") == "ecommerce"),
            "forum_cited_pct": pct(lambda r: r.get("source_type") == "forum"),
            "top_cited_type": top[0][0] if top else None,
        })
    return rows


def expected_vs_actual(features: list[dict]) -> list[dict]:
    """Per question: compare manifest expected_source_types with actually-cited types (heuristic)."""
    from collections import defaultdict
    by: dict[str, list] = defaultdict(list)
    for r in features:
        by[(r.get("record_id") or r.get("run_id"))].append(r)
    rows = []
    for rs in by.values():
        expected = next((r["expected_source_types"] for r in rs if r.get("expected_source_types")), [])
        if not expected:
            continue
        cited = [r for r in rs if r.get("cited") == 1]
        cited_canon: set[str] = set()
        for r in cited:
            cited_canon |= _effective_canon_types(r)
        exp_pairs = [(t, _canon(t)) for t in expected]
        found = [t for t, c in exp_pairs if c in cited_canon]
        missing = [t for t, c in exp_pairs if c not in cited_canon]
        exp_canon = {c for _, c in exp_pairs}
        unexpected = sorted({(r.get("source_type") or "unknown") for r in cited
                             if _canon(r.get("source_type") or "unknown") not in exp_canon})
        rows.append({
            "prompt_id": rs[0].get("prompt_id"), "intent": rs[0].get("intent"),
            "prompt": (rs[0].get("prompt") or "")[:60],
            "expected": "; ".join(expected), "cited_found": "; ".join(found),
            "expected_missing": "; ".join(missing), "unexpected_cited": "; ".join(unexpected[:6]),
            "coverage": round(len(found) / len(expected), 2) if expected else 0.0,
        })
    return rows
