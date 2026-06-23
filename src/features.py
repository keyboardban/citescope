"""Per-candidate feature extraction.

Produces one feature row per unique SERP candidate, plus chunk-level scores for
the content visualizer. Similarity features are a semantic overlap proxy.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .chunking import chunk_text, extract_headings
from .similarity import SimilarityEngine, summarize_scores
from .source_type import classify


def _best_query(cand: dict, fallback: str) -> str:
    qs = cand.get("queries") or []
    if not qs:
        return fallback
    best = min(qs, key=lambda q: q.get("rank", 999))
    return best.get("query") or fallback


def _parse_date(value) -> str | None:
    """Tolerant ISO/date parser -> normalized ISO string, or None."""
    if not value or not isinstance(value, str):
        return None
    raw = value.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%b %d, %Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.fromisoformat(raw) if fmt is None else datetime.strptime(value.strip(), fmt)
            return dt.isoformat()
        except (ValueError, TypeError):
            continue
    return None


def _freshness_days(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    days = (datetime.now(timezone.utc) - dt).days
    return float(max(days, 0))


NUMERIC_FEATURES = [
    "serp_rank", "title_query_sim", "snippet_query_sim", "page_query_sim",
    "page_output_sim", "max_chunk_output_sim", "max_chunk_query_sim",
    "word_count", "heading_count", "freshness_days",
]


def build_features(
    unique_cands: list[dict],
    pages: dict[str, dict],
    matching_result: dict,
    output_text: str,
    sim_engine: SimilarityEngine,
    fallback_query: str = "",
) -> dict:
    """Return {'features': [...], 'chunks': {candidate_id: [...]}}."""
    cited_ids = set(matching_result.get("cited_candidate_ids", []))
    match_by_cand = {
        m["matched_candidate_id"]: m
        for m in matching_result.get("matches", [])
        if m.get("matched_candidate_id")
    }

    features: list[dict] = []
    chunks_map: dict[str, list[dict]] = {}

    for cand in unique_cands:
        cid = cand["candidate_id"]
        query = _best_query(cand, fallback_query)
        page = pages.get(cand["normalized_url"]) or {}
        scraped = page.get("status") == "success"
        text = (page.get("text") or page.get("markdown") or "") if scraped else ""
        markdown = (page.get("markdown") or text) if scraped else ""

        source_type, is_official = classify(cand["url"])
        headings = extract_headings(markdown) if markdown else []

        row = {
            "candidate_id": cid,
            "url": cand["url"],
            "domain": cand["domain"],
            "root_domain": cand["root_domain"],
            "title": cand.get("title", ""),
            "cited": 1 if cid in cited_ids else 0,
            "match_type": match_by_cand.get(cid, {}).get("match_type", "no_match"),
            "strong_match": bool(match_by_cand.get(cid, {}).get("strong", False)),
            "serp_rank": cand["best_rank"],
            "source_type": source_type,
            "official_source": is_official,
            "scrape_success": scraped,
            "title_query_sim": sim_engine.score(cand.get("title", ""), query),
            "snippet_query_sim": sim_engine.score(cand.get("snippet", ""), query),
            "page_query_sim": None,
            "page_output_sim": None,
            "max_chunk_output_sim": None,
            "max_chunk_query_sim": None,
            "word_count": None,
            "heading_count": len(headings) if scraped else None,
            "freshness_days": None,
        }

        if scraped and text:
            row["word_count"] = len(text.split())
            row["page_query_sim"] = sim_engine.score(text[:8000], query)
            row["page_output_sim"] = sim_engine.score(text[:8000], output_text)
            row["freshness_days"] = _freshness_days(_parse_date(page.get("published_date")))

            chunks = chunk_text(markdown)
            if chunks:
                texts = [c["text"] for c in chunks]
                out_scores = sim_engine.score_many(output_text, texts)
                q_scores = sim_engine.score_many(query, texts)
                enriched = []
                for c, o, q in zip(chunks, out_scores, q_scores):
                    enriched.append({**c, "output_sim": o, "query_sim": q})
                chunks_map[cid] = enriched
                row["max_chunk_output_sim"] = summarize_scores(out_scores)["max"]
                row["max_chunk_query_sim"] = summarize_scores(q_scores)["max"]

        features.append(row)

    return {"features": features, "chunks": chunks_map}
