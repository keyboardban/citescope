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
from .ids import new_run_id, now_iso, short_id
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
    matching = match_all(citations, cands, pages)
    matching["unique_candidates"] = cands

    sim = SimilarityEngine("lexical")
    feat = build_features(cands, pages, matching, OUTPUT_TEXT, sim, fallback_query=q1)

    inputs = {
        "prompt": PROMPT,
        "gemini": {"model": "gemini-2.5-flash", "temperature": 0.2, "grounding": True, "system_prompt": None},
        "serp": {"top_k": 20, "country": "th", "language": "en", "selected_queries": [q1, q2]},
        "scrape": {"scope": "top_k", "top_k": 12, "selected_urls": [], "use_cache": True, "crawler_type": "cheerio"},
        "analysis": {"similarity_method": "lexical (offline)", "embedding_model": "text-embedding-004"},
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


# --------------------------------------------------------------------------- #
# Synthetic Topic Study (offline) — explore Topic Studies mode without keys.
# Fabricated to illustrate plausible patterns (official/brand + top ranks cited).
# --------------------------------------------------------------------------- #
_TOPIC_DOMAINS = {
    "Healthcare / Skincare": [
        ("fda.gov", "government", True, False), ("nih.gov", "government", True, False),
        ("healthline.com", "news", False, False), ("byrdie.com", "blog", False, False),
        ("reddit.com", "forum", False, False), ("pantip.com", "forum", False, False),
        ("wikipedia.org", "reference", False, False), ("cerave.com", "unknown", False, True),
        ("eucerin.com", "unknown", False, True), ("watsons.co.th", "ecommerce", False, False),
    ],
    "Automotive": [
        ("dlt.go.th", "government", True, False), ("toyota.co.th", "unknown", False, True),
        ("byd.com", "unknown", False, True), ("autolifethailand.com", "news", False, False),
        ("headlightmag.com", "news", False, False), ("caranddriver.com", "news", False, False),
        ("pantip.com", "forum", False, False), ("reddit.com", "forum", False, False),
        ("one2car.com", "ecommerce", False, False), ("youtube.com", "video", False, False),
    ],
    "Real Estate": [
        ("dol.go.th", "government", True, False), ("sansiri.com", "unknown", False, True),
        ("ananda.co.th", "unknown", False, True), ("ddproperty.com", "ecommerce", False, False),
        ("hipflat.co.th", "ecommerce", False, False), ("bangkokpost.com", "news", False, False),
        ("thinkofliving.com", "blog", False, False), ("pantip.com", "forum", False, False),
        ("reddit.com", "forum", False, False), ("propwise.co", "review", False, False),
    ],
}


def _demo_row(rid, topic, item, rank, dom, stype, inst, brand, cited, ti, pi):
    jit = rank * 0.001 + ti * 0.002 + pi * 0.0015
    pq = round(0.16 + (0.22 if cited else 0.0) + max(0, (6 - rank)) * 0.012 + jit, 3)
    po = round(0.15 + (0.30 if cited else 0.0) + 0.003 * rank + jit, 3)
    wc = 500 + rank * 40 + (250 if cited else 0) + pi * 20
    return {
        "candidate_id": short_id(f"{rid}:{dom}:{rank}"),
        "url": f"https://{dom}/p{rank}", "domain": dom, "root_domain": dom, "title": dom,
        "cited": cited, "weak_domain_match": False,
        "match_type": "normalized" if cited else "no_match", "strong_match": bool(cited),
        "serp_rank": rank, "source_type": stype,
        "institutional_official": inst, "official_source": inst, "brand_official_candidate": brand,
        "scrape_success": True,
        "title_query_sim": round(pq * 0.9 + 0.03, 3), "snippet_query_sim": round(pq * 0.8, 3),
        "page_query_sim": pq, "max_chunk_query_sim": round(pq + 0.05, 3),
        "page_output_sim": po, "max_chunk_output_sim": round(po + 0.03, 3),
        "word_count": wc, "char_count": wc * 6, "original_char_count": wc * 6,
        "used_char_count": min(wc * 6, 8000), "truncated": wc * 6 > 8000,
        "heading_count": 3 + (rank % 4), "freshness_days": float(20 + rank * 18 + pi * 5 - cited * 8),
        "run_id": rid, "topic": topic, "intent": item["intent"], "id": item["id"], "prompt": item["prompt"],
    }


def make_demo_topic_study(per_topic: int = 4) -> dict:
    from . import batch, question_sets
    runs, combined, per_prompt, items = [], [], [], []
    for ti, (topic, doms) in enumerate(_TOPIC_DOMAINS.items()):
        for pi, item in enumerate(question_sets.TOPIC_SETS[topic][:per_topic]):
            rid = f"DEMO-{topic.split()[0][:4]}-{pi}"
            items.append({**item, "topic": topic})
            cited_set = {1, 2}
            for rank in range(3, 6):  # add one official/brand page in ranks 3-5
                _, _, inst, brand = doms[(pi + rank - 1) % len(doms)]
                if inst or brand:
                    cited_set.add(rank)
                    break
            for rank in range(1, 9):
                dom, stype, inst, brand = doms[(pi + rank - 1) % len(doms)]
                combined.append(_demo_row(rid, topic, item, rank, dom, stype, inst, brand,
                                          1 if rank in cited_set else 0, ti, pi))
            citations = len(cited_set) + 1  # one unmatched citation
            strict = {str(k): round(sum(1 for r in cited_set if r <= k) / citations, 4) for k in (5, 10, 20, 50)}
            recall = {"strict": strict, "canonical": strict,
                      "domain_inclusive": {k: round(min(1.0, v + 0.06), 4) for k, v in strict.items()}}
            runs.append({"run_id": rid, "matching": {"recall": recall, "n_citations": citations}})
            per_prompt.append({**item, "topic": topic, "run_id": rid, "error": None,
                               "n_candidates": 8, "n_citations": citations, "n_scraped": 8,
                               "recall_strict_10": strict["10"]})
    return {
        "batch_id": "DEMO-" + new_run_id(), "created_at": now_iso(), "is_demo": True,
        "n_prompts": len(runs), "n_candidates": len(combined), "items": items,
        "prompts": [it["prompt"] for it in items], "run_ids": [r["run_id"] for r in runs],
        "per_prompt": per_prompt, "features": combined, "aggregate": batch.aggregate(runs, combined),
    }


# --------------------------------------------------------------------------- #
# Synthetic Bright Data (ChatGPT) export — for offline exploration + tests.
# Includes cited/more-only mixes, a cited↔search_sources duplicate (cited wins),
# and a tracking-param duplicate (dedup by normalized URL).
# --------------------------------------------------------------------------- #
SAMPLE_BRIGHTDATA = [
    {
        "url": "https://chatgpt.com/?q=Top%20hotels%20in%20New%20York",
        "prompt": "Top hotels in New York",
        "answer_text_markdown": ("New York has many highly rated hotels. Among the most frequently "
                                 "recommended are The Plaza, the Marriott Marquis in Times Square, and "
                                 "boutique options reviewed by travel guides. Booking.com and major travel "
                                 "publications list strong options across price ranges."),
        "web_search_query": ["best hotels in New York", "top rated NYC hotels"],
        "web_search_triggered": True,
        "model": "gpt-4o",
        "timestamp": "2026-06-20T10:00:00Z",
        "citations": [
            {"url": "https://www.booking.com/city/us/new-york.html", "title": "Hotels in New York | Booking.com",
             "description": "Compare hotels in New York.", "domain": "booking.com", "cited": True},
            {"url": "https://www.nytimes.com/guides/nyc-hotels", "title": "Where to Stay in NYC - The New York Times",
             "description": "Editorial hotel guide.", "domain": "nytimes.com", "cited": True},
            {"url": "https://www.tripadvisor.com/Hotels-g60763-New_York_City.html", "title": "NYC Hotels - Tripadvisor",
             "description": "Traveler reviews.", "domain": "tripadvisor.com", "cited": False},
            {"url": "https://www.reddit.com/r/nyc/comments/abc/best_hotels", "title": "Best hotels? r/nyc",
             "description": "Reddit thread.", "domain": "reddit.com", "cited": False},
        ],
        "search_sources_more": [
            {"url": "https://www.hotels.com/de1234/new-york", "title": "New York Hotels - Hotels.com", "domain": "hotels.com"},
            {"url": "https://www.tripadvisor.com/Hotels-g60763-New_York_City.html?utm_source=chatgpt", "title": "NYC Hotels"},
        ],
        "search_sources": [
            {"url": "https://www.booking.com/city/us/new-york.html", "title": "Booking NYC", "rank": 1},
            {"url": "https://www.marriott.com/new-york", "title": "Marriott New York", "rank": 2, "date_published": "2026-01-10"},
        ],
        "links_attached": [],
        "response_raw": {},
    },
    {
        "url": "https://chatgpt.com/?q=EV%20incentives%20Thailand",
        "prompt": "What government incentives exist for electric vehicles in Thailand?",
        "answer_text_markdown": ("Thailand offers EV incentives including tax reductions and subsidies under "
                                 "government EV programs. Automakers such as Toyota and BYD participate, and "
                                 "official agencies publish the current incentive details."),
        "web_search_query": "Thailand EV incentives 2026",
        "web_search_triggered": True,
        "model": "gpt-4o",
        "timestamp": "2026-06-21T09:00:00Z",
        "citations": [
            {"url": "https://www.dlt.go.th/ev-incentives", "title": "EV incentives - Department of Land Transport",
             "description": "Official incentive details.", "domain": "dlt.go.th", "cited": True},
            {"url": "https://www.caranddriver.com/thailand-ev", "title": "Thailand EV guide - Car and Driver",
             "description": "EV overview.", "domain": "caranddriver.com", "cited": True},
            {"url": "https://pantip.com/topic/ev-th", "title": "EV subsidy discussion - Pantip", "cited": False},
        ],
        "search_sources_more": [
            {"url": "https://www.byd.com/th", "title": "BYD Thailand", "domain": "byd.com"},
            {"url": "https://www.headlightmag.com/ev-incentive", "title": "EV incentive news"},
        ],
        "search_sources": [],
        "links_attached": ["https://www.dlt.go.th/ev-incentives"],
        "response_raw": {},
    },
    {
        "url": "https://chatgpt.com/?q=Buying%20a%20condo%20in%20Bangkok",
        "prompt": "What should I check before buying a condominium in Bangkok?",
        "answer_text": ("Before buying a Bangkok condo, check the developer reputation, the title and land "
                        "documents, foreign ownership quota, common fees, and the project location. Property "
                        "portals and the land department provide official document verification."),
        "web_search_query": [],
        "web_search_triggered": True,
        "model": "gpt-4o",
        "timestamp": "2026-06-22T08:00:00Z",
        "citations": [
            {"url": "https://www.ddproperty.com/buy-condo-bangkok", "title": "Buying a condo - DDproperty",
             "description": "Buyer guide.", "domain": "ddproperty.com", "cited": True},
            {"url": "https://www.bangkokpost.com/property/condo-guide", "title": "Condo guide - Bangkok Post", "cited": True},
            {"url": "https://www.reddit.com/r/Thailand/comments/xyz/condo", "title": "Condo advice r/Thailand", "cited": False},
        ],
        "search_sources_more": [
            {"url": "https://www.sansiri.com/condominium", "title": "Sansiri Condominiums", "domain": "sansiri.com"},
            {"url": "https://www.hipflat.co.th/en/", "title": "Hipflat property", "domain": "hipflat.co.th"},
        ],
        "search_sources": [
            {"url": "https://www.dol.go.th/verify", "title": "Land document verification", "rank": 1},
        ],
        "links_attached": [],
        "response_raw": {},
    },
]


def make_demo_brightdata() -> dict:
    """Parse the synthetic Bright Data sample into a normalized ChatGPT run."""
    import json as _json
    from . import brightdata
    return brightdata.parse_run(_json.dumps(SAMPLE_BRIGHTDATA), "sample_brightdata.json")


# Sample Prompt Manifest matching the SAMPLE_BRIGHTDATA prompts (for offline/tests).
SAMPLE_MANIFEST = (
    "prompt_id,topic,intent,prompt,country,prompt_language,expected_source_types\n"
    "P1,Travel,Product/Recommendation,Top hotels in New York,US,en,review;ecommerce;official_brand\n"
    "P2,Automotive,Regulation/Policy,What government incentives exist for electric vehicles in Thailand?,TH,en,government;news\n"
    "P3,Real Estate,Buyer Guide,What should I check before buying a condominium in Bangkok?,TH,en,government;official_brand;review;ecommerce\n"
)


def make_demo_manifest() -> dict:
    from . import brightdata
    return brightdata.parse_manifest(SAMPLE_MANIFEST, "sample_manifest.csv")
