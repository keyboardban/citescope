from __future__ import annotations

import pandas as pd
import pytest

from src.econometrics_eda_v2.normalize_sources import build_source_rows, can_construct_cited


def test_ai_json_without_cited_outcome_fails():
    ai = {"records": [{"record_id": "r1", "prompt": "p"}]}
    assert not can_construct_cited(ai)
    with pytest.raises(ValueError, match="Cannot construct cited outcome"):
        build_source_rows(ai, pd.DataFrame())


def test_ai_json_with_surfaced_and_cited_sources_builds_binary_cited():
    ai = {
        "run_id": "run",
        "records": [
            {
                "record_id": "r1",
                "prompt": "best hospital",
                "answer": "answer",
                "citations": [{"url": "https://example.com/a", "cited": True}],
                "search_sources": [
                    {"url": "https://example.com/a?utm_source=x", "title": "A"},
                    {"url": "https://example.com/b", "title": "B"},
                ],
            }
        ],
    }
    df, summary = build_source_rows(ai, pd.DataFrame([{"record_id": "r1", "intent": "info", "prompt": "best hospital"}]))
    assert df["cited"].tolist() == [1, 0]
    assert summary["cited_count"] == 1
    assert summary["more_only_count"] == 1


def test_ai_json_with_cited_false_citation_rows_produces_more_only():
    ai = {
        "run_id": "run",
        "records": [
            {
                "record_id": "r1",
                "prompt": "best hospital",
                "citations": [
                    {"url": "https://example.com/a", "title": "A", "cited": True},
                    {"url": "https://example.com/b", "title": "B", "cited": False},
                ],
                "search_sources": [{"url": "https://example.com/a"}],
                "search_sources_more": [{"url": "https://example.com/b"}],
            }
        ],
    }
    df, summary = build_source_rows(ai, pd.DataFrame())
    assert df.sort_values("normalized_url")["cited"].tolist() == [1, 0]
    assert summary["more_only_sources_found"] == 1


def test_ai_json_with_only_cited_sources_warns_strongly():
    ai = {
        "run_id": "run",
        "records": [
            {
                "record_id": "r1",
                "prompt": "best hospital",
                "citations": [{"url": "https://example.com/a", "cited": True}],
                "search_sources": [{"url": "https://example.com/a"}],
            }
        ],
    }
    _, summary = build_source_rows(ai, pd.DataFrame())
    assert summary["more_only_count"] == 0
    assert "outcome_has_single_class_or_no_more_only" in summary["warnings"]
