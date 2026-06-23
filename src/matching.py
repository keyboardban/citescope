"""Citation matching: link cited URLs to reconstructed SERP candidates.

Tiered, strongest-first:
  exact -> normalized -> final_redirect -> canonical -> amp_canonical -> domain_only -> no_match

Match-strength rules:
- STRONG = exact / normalized / final_redirect / canonical / amp_canonical.
  Only strong matches set cited_label = 1.
- WEAK = domain_only. Stored as weak_domain_match=True; it does NOT flip cited_label
  and is NOT counted in strict/canonical recall. For a weak match we pick the
  *closest-path* candidate on the domain (not just the best-ranked one) so we don't
  credit a citation to an arbitrary homepage.

Three recall variants are reported:
- strict_recall@K          : STRICT_TIERS only (URL identity)
- canonical_recall@K       : STRONG_TIERS (identity + canonical/amp equivalence)
- domain_inclusive_recall@K: strong + weak domain-only (exploratory)

This is correlational bookkeeping over *observable* URLs — never a claim that the
model saw or rejected any particular page.
"""

from __future__ import annotations

from collections import defaultdict
from urllib.parse import urlparse

from .config import RECALL_KS, STRICT_TIERS, STRONG_TIERS
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


def _closest_path_candidate(cands_on_domain: list[dict], citation_norm: str) -> dict:
    """Pick the same-domain candidate whose path best matches the citation path.

    Prefers the most shared leading path segments, then the better SERP rank.
    """
    cpath = urlparse(citation_norm).path.strip("/").split("/")

    def score(c: dict) -> tuple[int, int]:
        cand_path = urlparse(c["normalized_url"]).path.strip("/").split("/")
        common = 0
        for a, b in zip(cpath, cand_path):
            if a == b and a != "":
                common += 1
            else:
                break
        return (common, -int(c.get("best_rank", 999)))

    return max(cands_on_domain, key=score)


def _match_one(cit: dict, idx) -> dict:
    norm_map, raw_map, amp_map, domain_map, canon_map = idx
    raw = cit.get("raw_uri", "")
    effective = cit.get("resolved_url") or raw
    cnorm = normalize_url(effective)
    croot = root_domain(effective)

    strong_cand = None
    strong_tier = None
    # exact (raw string equality on either the raw or resolved URL)
    for key in (effective, raw):
        if key in raw_map:
            strong_cand, strong_tier = raw_map[key], "exact"
            break
    # normalized / final_redirect
    if strong_cand is None and cnorm in norm_map:
        strong_cand = norm_map[cnorm]
        strong_tier = "final_redirect" if is_redirect_wrapper(raw) else "normalized"
    # canonical (uses scraped page canonical URL)
    if strong_cand is None and cnorm in canon_map:
        strong_cand, strong_tier = canon_map[cnorm], "canonical"
    # amp/canonical variant
    if strong_cand is None and strip_amp(effective) in amp_map:
        strong_cand, strong_tier = amp_map[strip_amp(effective)], "amp_canonical"

    # weak domain-only match (closest path) — independent of strong match
    weak_cand = None
    if croot in domain_map:
        weak_cand = _closest_path_candidate(domain_map[croot], cnorm)

    if strong_cand is not None:
        match_type, cand, strong = strong_tier, strong_cand, True
    elif weak_cand is not None:
        match_type, cand, strong = "domain_only", weak_cand, False
    else:
        match_type, cand, strong = "no_match", None, False

    strong_rank = strong_cand["best_rank"] if strong_cand else None
    weak_rank = weak_cand["best_rank"] if weak_cand else None
    matched_rank = strong_rank if strong else (weak_rank if match_type == "domain_only" else None)

    return {
        "citation_index": cit.get("index"),
        "citation_url": effective,
        "raw_uri": raw,
        "title": cit.get("title", ""),
        "match_type": match_type,
        "strong": strong,
        "weak_domain_match": (not strong) and weak_cand is not None,
        "matched_candidate_id": cand["candidate_id"] if cand else None,
        "matched_url": cand["url"] if cand else None,
        "matched_rank": matched_rank,
        "strong_rank": strong_rank,
        "weak_rank": weak_rank,
    }


def _recalled(m: dict, k: int, mode: str) -> bool:
    if mode in ("strict", "canonical"):
        tiers = STRICT_TIERS if mode == "strict" else STRONG_TIERS
        return m["match_type"] in tiers and m["strong_rank"] is not None and m["strong_rank"] <= k
    # domain_inclusive: strong rank if strong, else weak rank
    if m["match_type"] in STRONG_TIERS:
        r = m["strong_rank"]
    elif m["match_type"] == "domain_only":
        r = m["weak_rank"]
    else:
        return False
    return r is not None and r <= k


def match_all(
    citations: list[dict],
    cands: list[dict],
    pages: dict[str, dict] | None = None,
) -> dict:
    """Match every (deduped) citation and compute the three recall variants."""
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

    # Cited label comes from STRONG matches only.
    cited_ids = sorted({m["matched_candidate_id"] for m in matches
                        if m["strong"] and m["matched_candidate_id"]})
    weak_ids = sorted({m["matched_candidate_id"] for m in matches
                       if m["weak_domain_match"] and m["matched_candidate_id"]})

    recall: dict[str, dict[str, float]] = {"strict": {}, "canonical": {}, "domain_inclusive": {}}
    for mode in recall:
        for k in RECALL_KS:
            hit = sum(1 for m in matches if _recalled(m, k, mode))
            recall[mode][str(k)] = round(hit / n, 4) if n else 0.0

    tiers = ["exact", "normalized", "final_redirect", "canonical",
             "amp_canonical", "domain_only", "no_match"]
    rate_counts = {t: 0 for t in tiers}
    for m in matches:
        rate_counts[m["match_type"]] = rate_counts.get(m["match_type"], 0) + 1
    rates = {t: (round(c / n, 4) if n else 0.0) for t, c in rate_counts.items()}

    unmatched = [m["citation_url"] for m in matches if m["match_type"] == "no_match"]

    return {
        "matches": matches,
        "unmatched": unmatched,
        "recall": recall,                  # nested: {strict|canonical|domain_inclusive: {K: rate}}
        "rates": rates,
        "rate_counts": rate_counts,
        "cited_candidate_ids": cited_ids,  # strong only
        "weak_candidate_ids": weak_ids,    # domain-only (exploratory)
        "n_citations": n,
    }
