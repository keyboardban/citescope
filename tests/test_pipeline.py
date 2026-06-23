"""Gemini failure short-circuit: do not call Apify on an unusable Gemini run."""

from __future__ import annotations

import pytest

from src import gemini_client, pipeline


class FakeApify:
    def __init__(self):
        self.called = False

    def actor(self, *a, **k):
        self.called = True
        raise RuntimeError("no network in tests")

    def dataset(self, *a, **k):
        self.called = True
        raise RuntimeError("no network in tests")


def _inputs():
    return {
        "prompt": "p",
        "gemini": {"model": "m", "temperature": 0.0, "grounding": True, "system_prompt": None},
        "serp": {"top_k": 10, "country": "us", "language": "en", "selected_queries": []},
        "scrape": {"scope": "top_k", "top_k": 5, "selected_urls": [], "use_cache": False, "crawler_type": "cheerio"},
        "analysis": {"similarity_method": "lexical (offline)", "embedding_model": "m"},
    }


def _trace(**over):
    base = {"output_text": "", "search_queries": [], "citations": [], "supports": [],
            "search_entry_point_html": None, "finish_reason": None, "prompt_feedback": None,
            "raw": None, "error": None, "model": "m", "grounding": True}
    base.update(over)
    return base


def test_pipeline_aborts_when_gemini_unusable(monkeypatch):
    monkeypatch.setattr(gemini_client, "run_grounded",
                        lambda *a, **k: _trace(error="429 RESOURCE_EXHAUSTED"))
    fake = FakeApify()
    with pytest.raises(pipeline.PipelineError):
        pipeline.run_full({"gemini": object(), "apify": fake}, _inputs(), use_cache=False)
    assert fake.called is False  # Apify never touched


def test_pipeline_reaches_apify_when_gemini_usable(monkeypatch):
    monkeypatch.setattr(gemini_client, "run_grounded",
                        lambda *a, **k: _trace(output_text="some answer",
                                               search_queries=["q"], finish_reason="STOP"))
    fake = FakeApify()
    # usable run -> proceeds to the SERP stage (run_serp swallows the network error)
    pipeline.run_full({"gemini": object(), "apify": fake}, _inputs(), use_cache=False)
    assert fake.called is True
