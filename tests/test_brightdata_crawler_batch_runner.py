from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "v2_run_area_condo_brightdata_content_pilot.py"
    spec = importlib.util.spec_from_file_location("brightdata_crawler_batch_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_async_batches_write_manifest_coverage_and_reuse_completed_cache(tmp_path, monkeypatch):
    runner = _runner_module()
    queue = pd.DataFrame([
        {"source_url": "https://example.com/a?utm_source=test", "normalized_url": "https://example.com/a", "source_root_domain": "example.com", "source_type": "other", "source_rows": 1, "cited_rows": 1, "intent_n": 1, "cited": 1, "initial_mode": "crawler_api", "selection_reason": "all_unique_urls"},
        {"source_url": "https://example.com/b", "normalized_url": "https://example.com/b", "source_root_domain": "example.com", "source_type": "other", "source_rows": 1, "cited_rows": 0, "intent_n": 1, "cited": 0, "initial_mode": "crawler_api", "selection_reason": "all_unique_urls"},
    ])
    settings = SimpleNamespace()
    calls = []

    def trigger(urls, _settings):
        calls.append(urls)
        return {"snapshot_id": f"s_{len(calls)}", "request_payload": urls, "request_params": {}, "response_headers": {}}

    monkeypatch.setattr(runner, "trigger_brightdata_crawler_async", trigger)
    monkeypatch.setattr(runner, "wait_for_brightdata_snapshot", lambda snapshot_id, _settings, **_kwargs: {"status": "ready", "snapshot_id": snapshot_id})
    monkeypatch.setattr(runner, "download_brightdata_snapshot", lambda snapshot_id, _settings: [{"url": url, "page_title": "Page", "markdown": "# Page\\n\\nUseful content " * 20} for url in calls[int(snapshot_id.split("_")[-1]) - 1]])

    out = tmp_path / "out"
    results, coverage = runner._run_crawler_async_batches(queue, settings, out / "raw", out / "normalized", out, batch_size=1, resume=False, attempt_type="all_unique_urls")
    assert len(results) == 2
    assert len(calls) == 2
    assert coverage["missing_urls"].sum() == 0
    manifest = pd.read_csv(out / "crawler_async_batch_manifest.csv")
    assert manifest["batch_status"].eq("completed").all()
    checkpoint = json.loads((out / "raw/crawler_api/async_batch_checkpoint.json").read_text())
    assert len(checkpoint["batches"]) == 2

    monkeypatch.setattr(runner, "trigger_brightdata_crawler_async", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("completed cache should be reused")))
    resumed, _ = runner._run_crawler_async_batches(queue, settings, out / "raw", out / "normalized", out, batch_size=1, resume=True, attempt_type="all_unique_urls")
    assert len(resumed) == 2
