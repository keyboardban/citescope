from __future__ import annotations

import pytest

from src.econometrics_eda_v2.brightdata_config import check_brightdata_config, require_live_brightdata_config


def test_dry_run_config_does_not_require_api_key():
    status = check_brightdata_config(live=False, env={})
    assert status["config_available"] is True
    assert status["missing_env_vars"] == []
    assert status["provider_mode"] == "browser_api"
    assert status["endpoint"] == "https://api.brightdata.com/request"
    assert status["render_js"] is True


def test_live_mode_fails_clearly_without_api_key():
    with pytest.raises(RuntimeError, match="Live Bright Data execution requires BRIGHTDATA_API_KEY"):
        require_live_brightdata_config(env={})


def test_config_status_masks_secret():
    status = check_brightdata_config(live=True, env={"BRIGHTDATA_API_KEY": "super-secret-token", "BRIGHTDATA_ZONE": "web_unlocker1"})
    rendered = repr(status)
    assert "super-secret-token" not in rendered
    assert status["api_key_present"] is True
    assert status["live_ready"] is True


def test_crawler_api_requires_dataset_id_but_not_proxy_zone():
    status = check_brightdata_config(
        live=True,
        env={
            "BRIGHTDATA_API_KEY": "super-secret-token",
            "BRIGHTDATA_PROVIDER_MODE": "crawler_api",
            "BRIGHTDATA_CRAWLER_DATASET_ID": "gd_example",
        },
    )
    assert status["provider_mode"] == "crawler_api"
    assert status["endpoint"] == "https://api.brightdata.com/datasets/v3/scrape"
    assert status["dataset_id_present"] is True
    assert status["live_ready"] is True
    assert status["zone_present"] is False
