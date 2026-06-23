"""Citation matching: link cited URLs to reconstructed SERP candidates.

Tiered, strongest-first:
  exact -> normalized -> final_redirect -> canonical -> amp_canonical -> domain_only -> no_match

Only 'domain_only' is weak; by default it does not count as a cited positive.
From the matches we derive citation_recall@K and per-tier match rates.

This is correlational bookkeeping over *observable* URLs — never a claim that the
model saw or rejected any particular page.
"""

from __future__ import annotations

from collections import defaultdict

from .config import RECALL_KS, WEAK_TIERS
from .ids import short_id
from .url_utils import (
    domain,
    is_redirect_wrapper,
    normalize_url,
    root_domain,
    strip_amp,
)


def unique_candidates(flat_candidates: list[dict]) -> list[dict]:
    """Collapse per-query SERP rows into unique candidate sites (best rank kept)."""
    groups: dict[str, dict] = {}
    for c in flat_candidates:
        nurl = normalize_url(c["url"])
        if not nurl:
            continue
        g = groups.get(nurl)
        rank = int(c.get("rank") or 999)
        if g is None:
            groups[nurl] = {
                "candidate_id": short_id(nurl),
                "url": c["url"],
                "normalized_url": nurl,
                "domain": domain(c["url"]),
                "root_domain": root_domain(c["url"]),
                "title": c.get("title", ""),
                "snippet": c.get("snippet", ""),
                "displayed_url": c.get("displayed_url", ""),
                "result_type": c.get("result_type", "organic"),
                "best_rank": rank,
                "queries": [{"query": c.get("query", ""), "rank": rank}],
            }
        else:
            g["queries"].append({"query": c.get("query", ""), "rank": rank})
            if rank < g["best_rank"]:
                g["best_rank"] = rank
                g["title"] = c.get("title", "") or g["title"]
                g["snippet"] = c.get("snippet", "") or g["snippet"]
    return sorted(groups.values(), key=lambda x: x["best_rank"])


def _build_indexes(cands: list[dict], pages: dict[str, dict]):
    norm_map: dict[str, dict] = {}
    raw_map: dict[str, dict] = {}
    amp_map: dict[str, dict] = {}
    domain_map: dict[str, list[dict]] = defaultdict(list)
    canon_map: dict[str, dict] = {}
    for c in cands:
        norm_map.setdefault(c["normalized_url"], c)
        raw_map.setdefault(c["url"], c)
        amp_map.setdefault(strip_amp(c["url"]), c)
        domain_map[c["root_domain"]].append(c)
        page = pages.get(c["normalized_url"]) if pages else None
        if page and page.get("canonical_url"):
            canon_map.setdefault(normalize_url(page["canonical_url"]), c)
    return norm_map, raw_map, amp_map, domain_map, canon_map


def _match_one(cit: dict, idx) -> dict:
    norm_map, raw_map, amp_map, domain_map, canon_map = idx
    raw = cit.get("raw_uri", "")
    effective = cit.get("resolved_url") or raw
    cnorm = normalize_url(effective)
    croot = root_domain(effective)

    # exact (raw string equality on either the raw or resolved URL)
    for key in (effective, raw):
        if key in raw_map:
            return _result(cit, raw_map[key], "exact")
    # normalized / final_redirect
    if cnorm in norm_map:
        tier = "final_redirect" if is_redirect_wrapper(raw) else "normalized"
        return _result(cit, norm_map[cnorm], tier)
    # canonical (uses scraped page canonical URL)
    if cnorm in canon_map:
        return _result(cit, canon_map[cnorm], "canonical")
    # amp/canonical variant
    if strip_amp(effective) in amp_map:
        return _result(cit, amp_map[strip_amp(effective)], "amp_canonical")
    # domain-only (weak)
    if croot in domain_map:
        best = min(domain_map[croot], key=lambda c: c["best_rank"])
        return _result(cit, best, "domain_only")
    return _result(cit, None, "no_match")


def _result(cit: dict, cand: dict | None, tier: str) -> dict:
    return {
        "citation_index": cit.get("index"),
        "citation_url": cit.get("resolved_url") or cit.get("raw_uri", ""),
        "raw_uri": cit.get("raw_uri", ""),
        "title": cit.get("title", ""),
        "match_type": tier,
        "strong": tier not in WEAK_TIERS and tier != "no_match",
        "matched_candidate_id": cand["candidate_id"] if cand else None,
        "matched_url": cand["url"] if cand else None,
        "matched_rank": cand["best_rank"] if cand else None,
    }


def match_all(
    citations: list[dict],
    cands: list[dict],
    pages: dict[str, dict] | None = None,
    include_weak: bool = False,
) -> dict:
    """Match every (deduped) citation and compute recall@K + per-tier rates."""
    pages = pages or {}
    idx = _build_indexes(cands, pages)

    # Dedupe citations by their effective normalized URL.
    seen: dict[str, dict] = {}
    for cit in citations:
        eff = normalize_url(cit.get("resolved_url") or cit.get("raw_uri", ""))
        if eff and eff not in seen:
            seen[eff] = cit
    distinct = list(seen.values())

    matches = [_match_one(c, idx) for c in distinct]
    n = len(distinct)

    def counts_as_cited(m: dict) -> bool:
        return m["strong"] or (include_weak and m["match_type"] == "domain_only")

    cited_ids = {m["matched_candidate_id"] for m in matches
                 if counts_as_cited(m) and m["matched_candidate_id"]}

    recall: dict[str, float] = {}
    for k in RECALL_KS:
        hit = sum(
            1 for m in matches
            if counts_as_cited(m) and m["matched_rank"] is not None and m["matched_rank"] <= k
        )
        recall[str(k)] = round(hit / n, 4) if n else 0.0

    tiers = ["exact", "normalized", "final_redirect", "canonical",
             "amp_canonical", "domain_only", "no_match"]
    rates = {t: 0 for t in tiers}
    for m in matches:
        rates[m["match_type"]] = rates.get(m["match_type"], 0) + 1
    rate_pct = {t: (round(c / n, 4) if n else 0.0) for t, c in rates.items()}

    unmatched = [m["citation_url"] for m in matches if m["match_type"] == "no_match"]

    return {
        "matches": matches,
        "unmatched": unmatched,
        "recall": recall,
        "rates": rate_pct,
        "rate_counts": rates,
        "cited_candidate_ids": sorted(cited_ids),
        "n_citations": n,
        "include_weak": include_weak,
    }
