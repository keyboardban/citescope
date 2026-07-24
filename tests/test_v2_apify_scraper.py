from __future__ import annotations

import json

import pandas as pd
import pytest

from src.econometrics_eda_v2.apify_scraper import scrape_queue_with_apify


def _queue(tmp_path):
    return pd.DataFrame(
        [{"scrape_id": "s1", "normalized_url": "https://example.com", "should_scrape": True, "cache_path": str(tmp_path / "s1.json")}]
    )


def test_missing_apify_token_fails_unless_dry_run(monkeypatch, tmp_path):
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="APIFY_TOKEN"):
        scrape_queue_with_apify(_queue(tmp_path), tmp_path)
    summary = scrape_queue_with_apify(_queue(tmp_path), tmp_path, dry_run=True)
    assert summary["dry_run"] is True
    assert not (tmp_path / "s1.json").exists()


def test_cached_success_skipped_unless_force_rescrape(tmp_path):
    (tmp_path / "s1.json").write_text(json.dumps({"provider_status": "success", "error": None}), "utf-8")
    summary = scrape_queue_with_apify(_queue(tmp_path), tmp_path, dry_run=True)
    assert summary["urls_cached_skipped"] == 1


def test_raw_cache_file_written_for_successful_scrape(tmp_path):
    class Dataset:
        def iterate_items(self):
            return iter([{"url": "https://example.com", "text": "Hello body", "title": "Hello"}])

    class Actor:
        def call(self, run_input):
            return {"id": "run1", "default_dataset_id": "ds1"}

    class Client:
        def __init__(self, token):
            self.token = token
        def actor(self, actor_id):
            return Actor()
        def dataset(self, dataset_id):
            return Dataset()

    summary = scrape_queue_with_apify(_queue(tmp_path), tmp_path, token="tok", client_cls=Client, batch_mode=False)
    assert summary["urls_success"] == 1
    raw = json.loads((tmp_path / "s1.json").read_text("utf-8"))
    assert raw["provider"] == "apify"
    assert raw["text"] == "Hello body"
