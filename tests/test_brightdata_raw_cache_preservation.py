from __future__ import annotations

import json

import pandas as pd
import pytest

import scripts.v2_benchmark_brightdata_scrape as bench
from src.econometrics_eda_v2.brightdata_config import BrightDataSettings
from src.econometrics_eda_v2.brightdata_response_parser import write_brightdata_raw_cache


def _settings() -> BrightDataSettings:
    return BrightDataSettings(
        api_key="super-secret-token",
        provider_mode="browser_api",
        endpoint="https://api.brightdata.com/request",
        zone="web_unlocker1",
        render_js=True,
        country="us",
        timeout_seconds=60,
        max_retries=2,
    )


def _input_csv(tmp_path):
    path = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {
                "benchmark_id": "bd_live_001",
                "source_url": "https://example.com/page?utm_source=chatgpt.com",
                "normalized_url": "https://example.com/page",
                "recommended_brightdata_mode": "browser_api",
            }
        ]
    ).to_csv(path, index=False)
    return path


def test_raw_cache_writer_refuses_normalized_result_without_raw_response(tmp_path):
    with pytest.raises(ValueError, match="Refusing to save normalized result as raw cache: raw_response missing"):
        write_brightdata_raw_cache(tmp_path / "bad.raw.json", {"benchmark_id": "bd_bad", "success": False})


def test_live_benchmark_saves_raw_response_before_parsed_cache(tmp_path, monkeypatch):
    input_path = _input_csv(tmp_path)
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    dry_dir = tmp_path / "dry"

    monkeypatch.setattr(
        bench,
        "check_brightdata_config",
        lambda live=False: {
            "api_key_present": True,
            "provider_mode": "browser_api",
            "endpoint": "https://api.brightdata.com/request",
            "render_js": True,
            "missing_env_vars": [],
        },
    )
    monkeypatch.setattr(bench, "load_brightdata_settings", _settings)

    def fake_scrape(url, mode, settings, *, live=False, raw_response_path="", request_fn=None):
        assert live is True
        assert raw_response_path == ""
        return {
            "raw_response": {"data": {"text": "Real provider body " * 80}, "status_code": 200},
            "request_payload": {"url": url, "zone": settings.zone, "api_key": settings.api_key},
            "response_headers": {"x-request-id": "abc"},
            "normalized_result": {
                "provider": "brightdata",
                "provider_mode": "brightdata_browser_api",
                "requested_url": url,
                "final_url": url,
                "normalized_url": url,
                "status_code": 200,
                "success": True,
                "error": "",
                "fetched_at": "2026-07-09T00:00:00Z",
            },
        }

    monkeypatch.setattr(bench, "scrape_url_brightdata", fake_scrape)
    rc = bench.main(
        [
            "--input",
            str(input_path),
            "--raw-cache-dir",
            str(raw_dir),
            "--parsed-cache-dir",
            str(parsed_dir),
            "--dry-run-dir",
            str(dry_dir),
            "--execute-live-brightdata",
            "--force",
        ]
    )
    assert rc == 0
    raw_path = raw_dir / "bd_live_001.raw.json"
    parsed_path = parsed_dir / "bd_live_001.parsed.json"
    assert raw_path.exists()
    assert parsed_path.exists()
    raw = json.loads(raw_path.read_text("utf-8"))
    parsed = json.loads(parsed_path.read_text("utf-8"))
    assert raw["raw_response"] == {"data": {"text": "Real provider body " * 80}, "status_code": 200}
    assert raw["request_payload_sanitized"]["api_key"] == "***"
    assert parsed["parse_success"] is True
    assert "raw_response" not in parsed


def test_dry_run_does_not_pollute_raw_cache_and_does_not_print_api_key(tmp_path, monkeypatch, capsys):
    input_path = _input_csv(tmp_path)
    raw_dir = tmp_path / "raw"
    parsed_dir = tmp_path / "parsed"
    dry_dir = tmp_path / "dry"
    monkeypatch.setattr(
        bench,
        "check_brightdata_config",
        lambda live=False: {
            "api_key_present": True,
            "provider_mode": "browser_api",
            "endpoint": "https://api.brightdata.com/request",
            "render_js": True,
            "missing_env_vars": [],
        },
    )
    monkeypatch.setattr(bench, "load_brightdata_settings", _settings)

    rc = bench.main(
        [
            "--input",
            str(input_path),
            "--raw-cache-dir",
            str(raw_dir),
            "--parsed-cache-dir",
            str(parsed_dir),
            "--dry-run-dir",
            str(dry_dir),
            "--dry-run",
            "--force",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert not list(raw_dir.glob("*.raw.json"))
    assert not list(parsed_dir.glob("*.parsed.json"))
    assert list(dry_dir.glob("*.dry_run.json"))
    assert "super-secret-token" not in captured.out
    assert "super-secret-token" not in captured.err
