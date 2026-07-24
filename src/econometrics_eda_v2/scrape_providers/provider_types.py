from __future__ import annotations

from enum import StrEnum


class ProviderMode(StrEnum):
    APIFY_CHEERIO = "apify_cheerio"
    APIFY_PLAYWRIGHT_ADAPTIVE = "apify_playwright_adaptive"
    APIFY_PLAYWRIGHT_FIREFOX = "apify_playwright_firefox"
    BRIGHTDATA_BROWSER_API = "brightdata_browser_api"
    BRIGHTDATA_UNLOCKER_API = "brightdata_unlocker_api"
    BRIGHTDATA_CRAWLER_API = "brightdata_crawler_api"
    SERPER_METADATA_ONLY = "serper_metadata_only"


BRIGHTDATA_MODE_TO_PROVIDER_MODE = {
    "browser_api": ProviderMode.BRIGHTDATA_BROWSER_API.value,
    "unlocker_api": ProviderMode.BRIGHTDATA_UNLOCKER_API.value,
    "crawler_api": ProviderMode.BRIGHTDATA_CRAWLER_API.value,
}
