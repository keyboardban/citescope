"""Synthetic demo run.

Builds a realistic run by feeding fabricated Gemini/SERP/scrape data through the
*real* matching, feature, and analysis code — so the dashboard is fully
explorable with no API keys or spend, and the engine gets an end-to-end check.
"""

from __future__ import annotations

from .features import build_features
from .matching import match_all, unique_candidates
from .pipeline import stage_analyze
from .similarity import SimilarityEngine
from .ids import new_run_id, now_iso
from .url_utils import domain, normalize_url

PROMPT = "What are the best tailors in Bangkok for custom suits?"

OUTPUT_TEXT = (
    "Bangkok is well known for affordable bespoke tailoring. Among the most "
    "frequently recommended custom tailors are Nick's Tailor, Ravi's Custom Tailor "
    "and Narry Tailor, all praised for hand-stitched canvas construction, quality "
    "wool and linen fabrics, and multiple fittings. A good custom suit in Bangkok "
    "typically involves choosing a fabric, two or three fittings, and one to two "
    "weeks of turnaround. Local guides and the Bangkok Post highlight strong value "
    "for money, though quality varies between tailor shops, so reading recent "
    "reviews before booking a fitting is recommended."
)

# Candidate URLs (reused by citations so matching lines up).
U_NICK = "https://www.nickstailorbangkok.com/custom-suits"
U_RAVI = "https://ravistailorbangkok.com/bespoke-suits"
U_NARRY = "https://narrytailor.com/services/custom-suits"
U_POST = "https://www.bangkokpost.com/life/social-and-lifestyle/2451/best-tailors"
U_TRIP = "https://www.tripadvisor.com/best-tailors-bangkok"
U_REDDIT = "https://www.reddit.com/r/Bangkok/comments/abc/best_tailor"
U_WIKI = "https://en.wikipedia.org/wiki/Bespoke_tailoring"
U_FB = "https://www.facebook.com/nickstailorbkk"
U_YT = "https://www.youtube.com/watch?v=demo123"
U_MEDIUM = "https://medium.com/@traveler/bangkok-tailor-guide"
U_TRUST = "https://www.trustpilot.com/review/ravistailorbangkok.com"
U_YELP = "https://www.yelp.com/biz/narry-tailor-bangkok"
U_QUORA = "https://www.quora.com/Whats-the-best-tailor-in-Bangkok"
U_THE = "https://thetailorbangkok.com/suits"
U_ESQ = "https://www.esquire.com/style/bangkok-tailors"
U_GQ = "https://www.gqthailand.com/style/best-bespoke-tailors"


def _cited_md(name: str) -> str:
    return (
        f"# {name} — Custom Suits in Bangkok\n\n"
        "## Bespoke tailoring\n"
        f"{name} is a custom tailor in Bangkok specialising in bespoke suits, "
        "hand-stitched canvas construction, and made-to-measure shirts. Customers "
        "choose from quality wool and linen fabrics imported for each suit.\n\n"
        "## Fittings and turnaround\n"
        "A typical custom suit involves selecting a fabric, two or three fittings, "
        "and a one to two week turnaround. The tailor shop offers good value for "
        "money compared with bespoke tailoring elsewhere.\n\n"
        "## Reviews\n"
        "Recent reviews highlight friendly fittings, durable hand-stitched details, "
        "and consistent quality across wool and linen suits in Bangkok.\n"
    )


def _generic_md(title: str) -> str:
    return (
        f"# {title}\n\n"
        "## Overview\n"
        "This page collects general travel and shopping notes about visiting "
        "Bangkok, including markets, food courts, transport tips and nightlife. "
        "It mentions shopping districts in passing.\n\n"
        "## Comments\n"
        "Users discuss a wide range of unrelated topics, from hotels to street food, "
        "with occasional remarks about clothing and souvenirs.\n"
    )


def _page(url: str, title: str, md: str, published: str | None, ok: bool = True) -> dict:
    return {
        "url": url,
        "final_url": url,
        "canonical_url": url,
        "title": title,
        "description": title,
        "author": None,
        "language": "en",
        "published_date": published,
        "text": md.replace("#", " ").replace("\n", " "),
        "markdown": md,
        "http_status": 200 if ok else 503,
        "status": "success" if ok else "failed",
        "error": None if ok else "navigation timeout",
    }


def _cand(query: str, rank: int, url: str, title: str, snippet: str, rtype: str = "organic") -> dict:
    return {"query": query, "rank": rank, "url": url, "title": title,
            "snippet": snippet, "displayed_url": url, "result_type": rtype}


def make_demo_run() -> dict:
    q1 = "best custom tailors Bangkok"
    q2 = "bespoke suits Bangkok reviews"

    flat = [
        _cand(q1, 1, U_NICK, "Nick's Tailor — Custom Suits Bangkok", "Bespoke suits, wool and linen fabrics, multiple fittings."),
        _cand(q1, 2, U_RAVI, "Ravi's Custom Tailor Bangkok", "Hand-stitched bespoke suits and made-to-measure shirts."),
        _cand(q1, 3, U_TRIP, "10 Best Tailors in Bangkok - Tripadvisor", "Traveler reviews of custom tailor shops in Bangkok."),
        _cand(q1, 4, U_REDDIT, "Best tailor in Bangkok? : r/Bangkok", "Reddit thread discussing tailor recommendations."),
        _cand(q1, 5, U_POST, "Where to get a suit made in Bangkok | Bangkok Post", "Guide to custom tailoring and value for money."),
        _cand(q1, 6, U_WIKI, "Bespoke tailoring - Wikipedia", "General article on bespoke tailoring."),
        _cand(q1, 7, U_NARRY, "Narry Tailor — Custom Suits", "Custom suits with hand-stitched canvas and fittings."),
        _cand(q1, 8, U_FB, "Nick's Tailor BKK - Facebook", "Official Facebook page."),
        _cand(q1, 9, U_YT, "Bangkok Tailor Suit Review - YouTube", "Video review of a Bangkok suit."),
        _cand(q1, 10, U_MEDIUM, "My Bangkok Tailor Guide - Medium", "Blog post about getting a suit made."),
        _cand(q2, 1, U_TRUST, "Ravi's Tailor Reviews | Trustpilot", "Customer reviews and ratings."),
        _cand(q2, 2, U_RAVI, "Ravi's Custom Tailor Bangkok", "Bespoke suits and shirts."),
        _cand(q2, 3, U_NICK, "Nick's Tailor — Custom Suits Bangkok", "Bespoke suits, fittings."),
        _cand(q2, 4, U_YELP, "Narry Tailor - Yelp", "Reviews for Narry Tailor."),
        _cand(q2, 5, U_QUORA, "What's the best tailor in Bangkok? - Quora", "Crowd answers."),
        _cand(q2, 6, U_THE, "The Tailor Bangkok — Suits", "Custom suits and fabrics."),
        _cand(q2, 7, U_ESQ, "Bangkok's Best Tailors | Esquire", "Editorial style guide."),
        _cand(q2, 8, U_GQ, "Best Bespoke Tailors | GQ Thailand", "Editorial recommendations."),
    ]

    # Citations chosen to exercise every match tier:
    #   final_redirect (Vertex wrapper, normalizes to a candidate), exact, normalized,
    #   domain_only (weak), and no_match. resolved_url is the publisher URL.
    def cite(i, url, title, wrapper=True):
        raw = f"https://vertexaisearch.cloud.google.com/grounding-api-redirect/{i:03d}xyz" if wrapper else url
        return {"index": i, "raw_uri": raw, "resolved_url": url, "title": title, "domain": domain(url)}

    citations = [
        cite(0, "https://nickstailorbangkok.com/custom-suits?utm_source=ai", "Nick's Tailor"),   # final_redirect
        cite(1, U_RAVI, "Ravi's Custom Tailor", wrapper=False),                                   # exact
        cite(2, U_NARRY + "/", "Narry Tailor"),                                                   # final_redirect
        cite(3, "http://www.bangkokpost.com/life/social-and-lifestyle/2451/best-tailors/",        # normalized
             "Bangkok Post", wrapper=False),
        cite(4, "https://www.tripadvisor.com/Restaurant_Review-bangkok", "Tripadvisor list"),     # domain_only (weak)
        cite(5, "https://www.timeout.com/bangkok/shopping/best-tailors-in-bangkok",               # no_match
             "Time Out Bangkok"),
    ]

    gemini = {
        "output_text": OUTPUT_TEXT,
        "search_queries": [{"query": q1, "is_fallback": False}, {"query": q2, "is_fallback": False}],
        "citations": citations,
        "supports": [
            {"text": "Among the most frequently recommended custom tailors are Nick's Tailor, "
                     "Ravi's Custom Tailor and Narry Tailor.", "chunk_indices": [0, 1, 2], "confidence": [0.92, 0.88, 0.81]},
            {"text": "Local guides and the Bangkok Post highlight strong value for money.",
             "chunk_indices": [3], "confidence": [0.76]},
        ],
        "search_entry_point_html": None,
        "raw": {"note": "demo run — no real API response"},
        "error": None,
        "model": "demo",
        "grounding": True,
        "ts": now_iso(),
        "cached": False,
    }

    pages_list = [
        _page(U_NICK, "Nick's Tailor — Custom Suits Bangkok", _cited_md("Nick's Tailor"), "2026-04-12"),
        _page(U_RAVI, "Ravi's Custom Tailor Bangkok", _cited_md("Ravi's Custom Tailor"), "2026-03-02"),
        _page(U_NARRY, "Narry Tailor — Custom Suits", _cited_md("Narry Tailor"), "2026-05-20"),
        _page(U_POST, "Where to get a suit made in Bangkok", _cited_md("Bangkok Post — tailoring guide"), "2026-02-18"),
        _page(U_TRIP, "10 Best Tailors in Bangkok", _generic_md("Tripadvisor: Bangkok tailors"), "2025-11-05"),
        _page(U_REDDIT, "Best tailor in Bangkok?", _generic_md("Reddit r/Bangkok thread"), "2026-01-09"),
        _page(U_WIKI, "Bespoke tailoring", _generic_md("Bespoke tailoring (encyclopedia)"), "2024-09-01"),
        _page(U_MEDIUM, "My Bangkok Tailor Guide", _generic_md("Medium travel blog"), "2026-06-01"),
        _page(U_TRUST, "Ravi's Tailor Reviews", _generic_md("Trustpilot reviews"), "2026-05-30"),
        _page(U_THE, "The Tailor Bangkok — Suits", _cited_md("The Tailor Bangkok"), "2026-04-28"),
        _page(U_ESQ, "Bangkok's Best Tailors", _generic_md("Esquire style"), "2025-12-15"),
        _page(U_YT, "Bangkok Tailor Suit Review", "", None, ok=False),  # failed scrape demo
    ]
    pages = {normalize_url(p["url"]): p for p in pages_list}

    cands = unique_candidates(flat)
    matching = match_all(citations, cands, pages, include_weak=False)
    matching["unique_candidates"] = cands

    sim = SimilarityEngine("lexical")
    feat = build_features(cands, pages, matching, OUTPUT_TEXT, sim, fallback_query=q1)

    inputs = {
        "prompt": PROMPT,
        "gemini": {"model": "gemini-2.5-flash", "temperature": 0.2, "grounding": True, "system_prompt": None},
        "serp": {"top_k": 20, "country": "th", "language": "en", "selected_queries": [q1, q2]},
        "scrape": {"scope": "top_k", "top_k": 12, "selected_urls": [], "use_cache": True, "crawler_type": "cheerio"},
        "analysis": {"include_weak": False, "similarity_method": "lexical (offline)", "embedding_model": "text-embedding-004"},
    }

    run = {
        "run_id": "DEMO-" + new_run_id(),
        "created_at": now_iso(),
        "is_demo": True,
        "used_fallback_query": False,
        "inputs": inputs,
        "gemini": gemini,
        "serp": {"candidates": flat, "items": [], "run_id": "demo", "dataset_id": "demo",
                 "actor": "demo", "ts": now_iso(), "error": None, "cached": False},
        "scrape": {"pages": pages, "apify": {"run_id": "demo", "fetched": len(pages)},
                   "scope": "top_k", "cached": False},
        "matching": matching,
        "features": feat["features"],
        "chunks": feat["chunks"],
    }
    run["analysis"] = stage_analyze(run)
    return run
