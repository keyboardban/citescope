"""Truncation metadata in feature extraction."""

from __future__ import annotations

from src.config import MAX_SIM_CHARS
from src.features import build_features
from src.matching import unique_candidates
from src.similarity import SimilarityEngine


def _cand(url):
    return unique_candidates([{"query": "q", "rank": 1, "url": url, "title": "t", "snippet": "s"}])


def _page(url, text):
    return {"url": url, "final_url": url, "canonical_url": url, "title": "t", "language": "en",
            "published_date": None, "text": text, "markdown": "# t\n\n" + text,
            "status": "success", "error": None}


def _empty_matching(cands):
    return {"cited_candidate_ids": [], "weak_candidate_ids": [], "matches": [], "unique_candidates": cands}


def test_truncation_flag_and_counts():
    url = "https://ex.com/long"
    cands = _cand(url)
    long_text = ("word " * 5000).strip()  # ~25k chars > MAX_SIM_CHARS
    pages = {cands[0]["normalized_url"]: _page(url, long_text)}
    row = build_features(cands, pages, _empty_matching(cands), "answer",
                         SimilarityEngine("lexical"), "q")["features"][0]
    assert row["truncated"] is True
    assert row["original_char_count"] == len(long_text)
    assert row["used_char_count"] == MAX_SIM_CHARS
    assert row["char_count"] == len(long_text)


def test_no_truncation_for_short_page():
    url = "https://ex.com/short"
    cands = _cand(url)
    short = "short page about tailoring"
    pages = {cands[0]["normalized_url"]: _page(url, short)}
    row = build_features(cands, pages, _empty_matching(cands), "answer",
                         SimilarityEngine("lexical"), "q")["features"][0]
    assert row["truncated"] is False
    assert row["used_char_count"] == row["original_char_count"] == len(short)
