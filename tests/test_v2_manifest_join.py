from __future__ import annotations

import pandas as pd

from src.econometrics_eda_v2.normalize_sources import MISSING_INTENT_LABEL, build_source_rows, read_manifest


def _ai(record):
    base = {
        "answer": "a",
        "citations": [{"url": "https://a.com"}],
        "search_sources": [{"url": "https://a.com", "title": "A"}, {"url": "https://b.com", "title": "B"}],
    }
    base.update(record)
    return {"records": [base]}


def test_join_by_prompt_text(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text("prompt_id,prompt,intent,topic,country,prompt_language\np1,Best hospital in Bangkok,info,health,TH,en\n", "utf-8")
    mf = read_manifest(path)
    df, _ = build_source_rows(_ai({"record_id": "r1", "prompt": "Best hospital in Bangkok"}), mf)
    assert set(df["intent"]) == {"info"}
    assert set(df["prompt_id"]) == {"p1"}


def test_failed_join_becomes_missing_intent():
    df, _ = build_source_rows(_ai({"record_id": "r1", "prompt": "x"}), pd.DataFrame([{"prompt": "other", "intent": "x"}]))
    assert set(df["intent"]) == {"(unmatched)"}
