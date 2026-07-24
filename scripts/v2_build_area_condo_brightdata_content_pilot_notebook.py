#!/usr/bin/env python3
"""Build the Bright Data content-pilot notebook."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/06_area_condo_brightdata_content_pilot.ipynb"
def md(text: str): return nbf.v4.new_markdown_cell(text.strip())
def py(text: str): return nbf.v4.new_code_cell(text.strip())

def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# Area Condo Bright Data Content Pilot

This isolated live-scrape pilot records the configured Bright Data provider, its request URL, and the resulting content quality. Browser/Unlocker runs may use a fallback comparison; Crawler API runs use the Bright Data Dataset endpoint directly. The final content record is selected only when it is observably at least as usable as the available provider attempt.

This is a scrape-quality and descriptive content check, not a final LPM or causal analysis."""),
        py("""from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import json
import pandas as pd
from IPython.display import Image, Markdown, display
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists(): ROOT = ROOT.parent
BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
OUT = BASE / 'tables/area_condo_brightdata_content_pilot'
FIG = BASE / 'figures/area_condo_brightdata_content_pilot'
def table(name): return pd.read_csv(OUT / name)
def show(name): display(Image(filename=str(FIG / name)))
TRACKING_PARAMETERS = {'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'igshid', 'ref', 'spm'}
def strip_tracking_params(url):
    parts = urlsplit(str(url))
    query = [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
             if not (key.casefold().startswith('utm_') or key.casefold() in TRACKING_PARAMETERS)]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query, doseq=True), parts.fragment))
summary = json.loads((OUT / 'brightdata_content_pilot_summary.json').read_text())
display(pd.DataFrame([summary]))"""),
        md("""### Latest pilot status

The table above is read from the latest pilot output. These are pilot results only; they do not replace the wider 846-URL post-scrape EDA dataset."""),
        md("""## 1. Provider execution and fallback

The cache stores raw provider responses separately by provider mode. Browser/Unlocker fallback records are retained rather than overwritten. Crawler API runs have one Crawler attempt per URL in this pilot runner."""),
        py("detail = table('brightdata_content_pilot_url_results.csv')\nexecution_cols = ['initial_mode','browser_attempted','unlocker_attempted','crawler_attempted','fallback_used','final_provider_mode','scrape_success','content_strength']\ndisplay(detail.reindex(columns=execution_cols).value_counts().rename('urls').reset_index())\nshow('final_provider_selection.png')"),
        md("""### Provider error audit

Bright Data can encode an upstream proxy failure in `x-brd-*` response headers even when the outer HTTP response is 200. Crawler API errors are returned in the Dataset response instead. This table is the authoritative configuration/error diagnosis for the pilot."""),
        py("display(table('brightdata_content_pilot_provider_error_audit.csv'))"),
        md("""### URL cleaning and tracking-parameter retry audit

The source URL is retained exactly as observed. A separate `tracking_clean_url` removes known tracking query parameters such as `utm_*`, `fbclid`, `gclid`, `mc_*`, `igshid`, `ref`, and `spm`; it does not change the site domain or the URL path. The final request URL records the URL actually sent to Bright Data. Only non-policy failures were retried with the cleaned URL, so a raw source URL in `final_request_url` means no clean retry was needed or attempted.

Removing a tracker is a request-normalization step, not evidence that the tracker caused a failure. Robots, KYC/compliance, blocking, and upstream gateway failures are diagnosed separately."""),
        py("""detail = table('brightdata_content_pilot_url_results.csv').copy()
detail['source_url_raw'] = detail['source_url']
detail['tracking_clean_url'] = detail['source_url_raw'].map(strip_tracking_params)
detail['tracking_parameters_removed'] = detail['source_url_raw'].ne(detail['tracking_clean_url'])
detail['final_request_url'] = detail['final_request_url'].fillna(detail['source_url_raw'])
detail['tracking_clean_retry_used'] = detail.get('tracking_clean_retry_used', pd.Series(False, index=detail.index)).fillna(False)
detail['final_request_used_clean_url'] = detail['final_request_url'].eq(detail['tracking_clean_url'])
detail['request_url_strategy'] = 'source URL retained'
detail.loc[detail['tracking_clean_retry_used'].fillna(False), 'request_url_strategy'] = 'tracking-clean retry selected'
detail['tracking_parameter_keys'] = detail['source_url_raw'].map(lambda u: ', '.join(k for k, _ in parse_qsl(urlsplit(u).query) if k.casefold().startswith('utm_') or k.casefold() in {'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'igshid', 'ref', 'spm'}))
tracking_summary = pd.DataFrame([{
    'pilot_urls': len(detail),
    'urls_with_tracker_parameters': int(detail['tracking_parameters_removed'].sum()),
    'tracking_clean_retries_selected': int(detail['tracking_clean_retry_used'].fillna(False).sum()),
    'tracking_clean_retry_successes': int((detail['tracking_clean_retry_used'].fillna(False) & detail['scrape_success'].fillna(False)).sum()),
    'source_domains': int(detail['source_root_domain'].nunique()),
}])
display(tracking_summary)
display(detail[['source_url_raw', 'tracking_clean_url', 'final_request_url', 'request_url_strategy', 'tracking_parameter_keys', 'source_root_domain', 'scrape_success', 'content_strength']].sort_values(['request_url_strategy', 'source_root_domain', 'source_url_raw']))
request_audit = OUT / 'tracking_parameter_request_audit.csv'
if request_audit.exists():
    display(pd.read_csv(request_audit))"""),
        py("retry_audit = OUT / 'tracking_param_clean_retry_audit.csv'\nif retry_audit.exists(): display(pd.read_csv(retry_audit))"),
        md("""## 2. Scrape and content quality

`strong` means successful extraction, `content_quality_flag = ok`, and at least 300 words. `medium` is at least 100 words; `weak` is shorter usable text; `failed` has no usable extracted text."""),
        py("display(table('brightdata_content_pilot_quality_summary.csv'))\nshow('cited_rate_by_content_strength.png')"),
        md("""## 3. Cited versus more-only content comparison

These pilot summaries are unadjusted. Content values describe only pages successfully measured by the selected provider, so missingness and provider choice remain part of the interpretation."""),
        py("display(table('brightdata_content_pilot_cited_comparison.csv'))\nshow('word_count_by_cited_status.png')"),
        md("""## 4. URL-level content and failure review

Use this table to inspect source versus request URLs, provider mode, the explicit provider error, quality flags, word/heading/table/link counts, and short text excerpts. It contains no answer text and is appropriate for manual scrape-quality review. A success flag only confirms retrieval; `content_strength` and `content_quality_flag` determine whether content features are usable."""),
        py("display(detail[['source_url_raw','tracking_clean_url','final_request_url','request_url_strategy','tracking_clean_retry_used','source_root_domain','cited_rows','initial_mode','final_provider_mode','scrape_success','content_strength','content_quality_flag','status_code','provider_error_code','provider_error_message','scrape_error','word_count','heading_count','table_count','link_count','page_title','page_text_excerpt']])"),
        md("""## 5. Interpretation boundary

This pilot supports a decision about whether Bright Data can provide usable webpage content for this topic. It does not establish why ChatGPT cited a page, and it should not be used to include answer-derived features, rank/position, source origin, or citation labels as main LPM predictors."""),
    ]
    nb.metadata['area_condo_brightdata_content_pilot'] = {'purpose': 'live Bright Data scrape-quality pilot; no LPM'}
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbf.write(nb, TARGET)
    return TARGET
if __name__ == '__main__': print(build())
