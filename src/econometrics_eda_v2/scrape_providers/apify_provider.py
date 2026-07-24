from __future__ import annotations

from src.econometrics_eda_v2.scrape_providers.provider_types import ProviderMode


PROVIDER_ROLE_NOTES = {
    ProviderMode.APIFY_CHEERIO.value: (
        "Apify Website Content Crawler in Cheerio mode is the current primary static full-page scraper. "
        "It enters target URLs but does not fully render JavaScript."
    ),
    ProviderMode.APIFY_PLAYWRIGHT_ADAPTIVE.value: (
        "Apify Playwright adaptive is a rendered full-page fallback for JS-heavy pages."
    ),
    ProviderMode.APIFY_PLAYWRIGHT_FIREFOX.value: (
        "Apify Playwright Firefox is a rendered fallback for pages that look blocked, captcha-gated, or browser-sensitive."
    ),
    ProviderMode.SERPER_METADATA_ONLY.value: (
        "Serper is only a SERP metadata provider for titles/snippets/search results. It is not a full-page scraper."
    ),
}


def crawler_type_to_provider_mode(crawler_type: str) -> str:
    crawler = str(crawler_type or "cheerio").strip().casefold()
    if crawler == "playwright:adaptive":
        return ProviderMode.APIFY_PLAYWRIGHT_ADAPTIVE.value
    if crawler == "playwright:firefox":
        return ProviderMode.APIFY_PLAYWRIGHT_FIREFOX.value
    return ProviderMode.APIFY_CHEERIO.value
