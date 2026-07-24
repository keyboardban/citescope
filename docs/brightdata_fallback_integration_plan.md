# Bright Data Fallback Integration Plan

This plan prepares CiteScope v2 for a future Bright Data benchmark without changing the current full export. The current primary scraper remains Apify Website Content Crawler in Cheerio mode.

## Provider roles

- `apify_cheerio`: current primary static full-page scraper. It enters the target URL and extracts page content, but it does not fully render JavaScript.
- `apify_playwright_adaptive` / `apify_playwright_firefox`: rendered Apify fallback modes for JS-heavy, browser-sensitive, or blocked-looking pages.
- `brightdata_browser_api` / `brightdata_unlocker_api`: future managed rendered/unblocking scraper modes. These should be benchmarked on problematic URLs before any full-pipeline use.
- `serper_metadata_only`: SERP metadata fallback only. Serper is not a full-page scraper and should be used only for title/snippet/search metadata when page content cannot be retrieved.

## When to use Bright Data fallback

Bright Data should only be considered after the current Apify result is observably weak and after a small benchmark shows that Bright Data improves those cases. Candidate fallback cases are URLs with `parse_failed`, `empty_text`, `dynamic_js_likely`, `blocked_or_error_page`, `very_short_text`, `boilerplate_only`, or `nav_footer_only`.

## Trigger flags

Use `brightdata_browser_api` first for:

- `dynamic_js_likely`
- `parse_failed` without clear blocking
- `empty_text`
- `very_short_text`
- `boilerplate_only`
- `nav_footer_only`

Use `brightdata_unlocker_api` first for:

- `blocked_or_error_page`
- captcha-like title/excerpt
- access denied / Cloudflare / human verification pages

Do not send PDF/binary URLs to Bright Data browser/unlocker first. Mark them `pdf_parser_needed`.

## Domain strategy

Do not promote Bright Data for all domains. Domain fallback should require repeated benchmark wins for the same domain, such as higher parse success, better content quality, or resolving unknown page types with medium/high confidence. Domains with good Apify output should stay on Apify.

## Raw cache layout

Future Bright Data benchmark cache:

```text
data/econometrics_v2/scrape_cache/brightdata_benchmark/raw/{benchmark_id}.json
data/econometrics_v2/scrape_cache/brightdata_benchmark/parsed/
```

Production fallback, if approved later, should use provider-specific cache paths and never overwrite good Apify cache:

```text
data/econometrics_v2/scrape_cache/provider_fallback/brightdata/raw/
data/econometrics_v2/scrape_cache/provider_fallback/brightdata/parsed/
```

## Provider result selection

Select the final provider result only when the fallback has clear observable improvement:

- Apify failed and Bright Data succeeded.
- Bright Data content quality improves.
- Bright Data word count is at least 1.5x Apify and excerpt appears to be main content.
- Apify page type is `unknown` and Bright Data resolves it to non-unknown with medium/high confidence.

Do not select Bright Data if it only returns more navigation/footer text, boilerplate, or irrelevant long text.

## Future final export columns

Do not add these columns to the full final CSV until fallback integration is approved:

- `scrape_provider_primary`
- `scrape_provider_final`
- `fallback_used`
- `fallback_reason`
- `provider_quality_score`
- `apify_word_count`
- `brightdata_word_count`
- `apify_content_quality_flag`
- `brightdata_content_quality_flag`
- `content_quality_before`
- `content_quality_after`
- `page_type_before_provider_fallback`
- `page_type_after_provider_fallback`
- `provider_raw_cache_path`

## Avoid overwriting good Apify results

Never replace Apify when `content_quality_flag = ok`, `word_count >= 300`, and the page text excerpt looks like main content. If `page_type_final = unknown` in that situation, treat the issue as likely taxonomy/classifier coverage, not scraping.

## Cost controls

- Start with the 40-URL benchmark only.
- Require explicit `--execute-live-brightdata` for live Bright Data calls.
- Default all Bright Data commands to dry-run.
- Set `--max-urls` for any live benchmark.
- Cache one raw response per benchmark URL.
- Do not rescrape successful Bright Data cache unless `--force` is passed.

## Rollback plan

Fallback integration should be reversible by switching `scrape_provider_final` back to Apify and ignoring Bright Data cache paths. Since Bright Data results are stored separately, rollback should not require deleting or mutating existing Apify raw cache, parsed rows, page features, or final exports.
