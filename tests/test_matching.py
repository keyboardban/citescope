"""Domain-only weak matching + recall@K variants."""

from __future__ import annotations

from src.config import STRONG_TIERS
from src.features import build_features
from src.matching import match_all, unique_candidates
from src.similarity import SimilarityEngine


def _flat():
    return [
        {"query": "q", "rank": 1, "url": "https://example.com/", "title": "Home", "snippet": "home"},
        {"query": "q", "rank": 9, "url": "https://example.com/deep/article", "title": "Deep", "snippet": "deep"},
        {"query": "q", "rank": 2, "url": "https://other.com/page", "title": "Other", "snippet": "o"},
    ]


def _cites():
    return [
        {"index": 0, "raw_uri": "https://other.com/page",
         "resolved_url": "https://other.com/page", "title": "strong"},
        {"index": 1, "raw_uri": "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
         "resolved_url": "https://example.com/some-other-path", "title": "weak"},
        {"index": 2, "raw_uri": "https://nowhere.com/x",
         "resolved_url": "https://nowhere.com/x", "title": "none"},
    ]


def test_domain_only_not_cited():
    cands = unique_candidates(_flat())
    m = match_all(_cites(), cands)
    by = {x["title"]: x for x in m["matches"]}

    assert by["strong"]["strong"] and by["strong"]["match_type"] in STRONG_TIERS
    assert by["weak"]["match_type"] == "domain_only"
    assert by["weak"]["strong"] is False
    assert by["weak"]["weak_domain_match"] is True
    assert by["none"]["match_type"] == "no_match"

    other_id = next(c["candidate_id"] for c in cands if c["domain"] == "other.com")
    example_ids = {c["candidate_id"] for c in cands if c["domain"] == "example.com"}
    assert other_id in m["cited_candidate_ids"]
    assert not (example_ids & set(m["cited_candidate_ids"]))     # weak page never cited
    assert example_ids & set(m["weak_candidate_ids"])            # weak recorded separately


def test_recall_variants_separate_and_ordered():
    cands = unique_candidates(_flat())
    m = match_all(_cites(), cands)
    r = m["recall"]
    assert m["n_citations"] == 3
    # strict@10: only the strong match (other.com, rank 2) -> 1/3 (rounded to 4dp)
    assert abs(r["strict"]["10"] - 1 / 3) < 1e-3
    assert r["canonical"]["10"] == r["strict"]["10"]            # no canonical-only here
    # domain_inclusive@10: strong + weak (example, rank<=10) -> 2/3
    assert abs(r["domain_inclusive"]["10"] - 2 / 3) < 1e-3
    for k in ("5", "10", "20", "50"):                           # weak never lowers recall
        assert r["domain_inclusive"][k] >= r["strict"][k]
    assert any("nowhere.com" in u for u in m["unmatched"])


def test_domain_only_does_not_flip_cited_label_in_features():
    cands = unique_candidates(_flat())
    m = match_all(_cites(), cands)
    m["unique_candidates"] = cands
    feat = build_features(cands, {}, m, "answer text", SimilarityEngine("lexical"), "q")
    feats = feat["features"]
    ex = [r for r in feats if r["domain"] == "example.com"]
    assert all(r["cited"] == 0 for r in ex)            # weak domain match -> not cited
    assert any(r["weak_domain_match"] for r in ex)
    other = next(r for r in feats if r["domain"] == "other.com")
    assert other["cited"] == 1
