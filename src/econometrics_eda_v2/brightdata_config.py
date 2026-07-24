from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_BRIGHTDATA_PROVIDER_MODES = {"browser_api", "unlocker_api", "crawler_api"}
DEFAULT_UNLOCKER_ENDPOINT = "https://api.brightdata.com/request"
DEFAULT_CRAWLER_ENDPOINT = "https://api.brightdata.com/datasets/v3/scrape"
DEFAULT_CRAWLER_TRIGGER_ENDPOINT = "https://api.brightdata.com/datasets/v3/trigger"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass


@dataclass(frozen=True)
class BrightDataSettings:
    api_key: str
    provider_mode: str
    endpoint: str
    zone: str
    render_js: bool
    country: str
    timeout_seconds: int
    max_retries: int
    dataset_id: str = ""
    crawler_async: bool = False
    crawler_poll_seconds: int = 10
    crawler_wait_seconds: int = 1800

    def masked(self) -> dict[str, Any]:
        return {
            "api_key_present": bool(self.api_key),
            "provider_mode": self.provider_mode,
            "endpoint": self.endpoint,
            "zone_present": bool(self.zone),
            "dataset_id_present": bool(self.dataset_id),
            "crawler_async": self.crawler_async,
            "render_js": self.render_js,
            "country": self.country,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }


def _bool_env(value: str | None, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


def _int_env(value: str | None, default: int) -> int:
    try:
        return int(str(value).strip()) if value is not None and str(value).strip() else default
    except ValueError:
        return default


def load_brightdata_settings(env: dict[str, str] | None = None) -> BrightDataSettings:
    env = os.environ if env is None else env
    mode = env.get("BRIGHTDATA_PROVIDER_MODE", "browser_api").strip().casefold() or "browser_api"
    if mode not in ALLOWED_BRIGHTDATA_PROVIDER_MODES:
        mode = "browser_api"
    crawler_async = _bool_env(env.get("BRIGHTDATA_CRAWLER_ASYNC"), False)
    endpoint = (
        env.get("BRIGHTDATA_CRAWLER_ENDPOINT", DEFAULT_CRAWLER_TRIGGER_ENDPOINT if crawler_async else DEFAULT_CRAWLER_ENDPOINT).strip()
        or (DEFAULT_CRAWLER_TRIGGER_ENDPOINT if crawler_async else DEFAULT_CRAWLER_ENDPOINT)
        if mode == "crawler_api"
        else env.get("BRIGHTDATA_ENDPOINT", DEFAULT_UNLOCKER_ENDPOINT).strip() or DEFAULT_UNLOCKER_ENDPOINT
    )
    return BrightDataSettings(
        api_key=env.get("BRIGHTDATA_API_KEY", ""),
        provider_mode=mode,
        endpoint=endpoint,
        zone=env.get("BRIGHTDATA_ZONE", "").strip(),
        render_js=_bool_env(env.get("BRIGHTDATA_RENDER_JS"), True),
        country=env.get("BRIGHTDATA_COUNTRY", "us").strip() or "us",
        timeout_seconds=_int_env(env.get("BRIGHTDATA_TIMEOUT_SECONDS"), 60),
        max_retries=_int_env(env.get("BRIGHTDATA_MAX_RETRIES"), 2),
        dataset_id=env.get("BRIGHTDATA_CRAWLER_DATASET_ID", env.get("BRIGHTDATA_DATASET_ID", "")).strip(),
        crawler_async=crawler_async,
        crawler_poll_seconds=_int_env(env.get("BRIGHTDATA_CRAWLER_POLL_SECONDS"), 10),
        crawler_wait_seconds=_int_env(env.get("BRIGHTDATA_CRAWLER_WAIT_SECONDS"), 1800),
    )


def check_brightdata_config(live: bool = False, env: dict[str, str] | None = None) -> dict[str, Any]:
    settings = load_brightdata_settings(env)
    missing = []
    if live and not settings.api_key:
        missing.append("BRIGHTDATA_API_KEY")
    if live and settings.provider_mode == "crawler_api" and not settings.dataset_id:
        missing.append("BRIGHTDATA_CRAWLER_DATASET_ID")
    if live and settings.provider_mode != "crawler_api" and settings.endpoint.rstrip("/") == DEFAULT_UNLOCKER_ENDPOINT.rstrip("/") and not settings.zone:
        missing.append("BRIGHTDATA_ZONE")
    return {
        "config_available": True,
        "live_ready": bool(settings.api_key) and not missing,
        "missing_env_vars": missing,
        "provider_mode": settings.provider_mode,
        "endpoint": settings.endpoint,
        "zone_present": bool(settings.zone),
        "dataset_id_present": bool(settings.dataset_id),
        "crawler_async": settings.crawler_async,
        "render_js": settings.render_js,
        "country": settings.country,
        "timeout_seconds": settings.timeout_seconds,
        "max_retries": settings.max_retries,
        "api_key_present": bool(settings.api_key),
    }


def require_live_brightdata_config(env: dict[str, str] | None = None) -> BrightDataSettings:
    status = check_brightdata_config(live=True, env=env)
    if status["missing_env_vars"]:
        missing = ", ".join(status["missing_env_vars"])
        if "BRIGHTDATA_API_KEY" in status["missing_env_vars"]:
            raise RuntimeError("Live Bright Data execution requires BRIGHTDATA_API_KEY.")
        raise RuntimeError(f"Live Bright Data execution missing required env vars: {missing}.")
    return load_brightdata_settings(env)
