"""Persistent embedding cache: identical text/model is embedded only once."""

from __future__ import annotations

from src import gemini_client, pipeline


def test_embedding_cache_avoids_recompute(monkeypatch):
    calls: list[str] = []

    def fake_embed(client, texts, model):
        calls.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(gemini_client, "embed_texts", fake_embed)

    eng = pipeline.make_sim_engine("gemini embeddings", gem_client=object(), embed_model="m")
    eng.score("hello world", "hello world")  # embeds "hello world" once

    # A fresh engine (empty in-memory cache) must reuse the persistent SQLite cache.
    eng2 = pipeline.make_sim_engine("gemini embeddings", gem_client=object(), embed_model="m")
    eng2.score("hello world", "brand new text")

    assert calls.count("hello world") == 1   # never re-embedded
    assert calls.count("brand new text") == 1
