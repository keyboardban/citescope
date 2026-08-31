# CiteScope ChatGPT Content Audit

CiteScope is a Streamlit research application for auditing observable ChatGPT
sources exported through Bright Data. It compares sources marked as cited with
sources shown but not cited (more-only), validates scraped page content, and
estimates content-feature associations using the econometric notebooks 07-11.

This is a black-box observational audit. More-only does not mean rejected, and
the observed Bright Data source panel is not ChatGPT's complete internal
retrieval set. Model estimates are conditional associations among surfaced
sources, not causal effects or web-wide citation probabilities.

## Where to start

| You want to | Read |
|---|---|
| understand the whole pipeline | [`docs/CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md`](docs/CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md) |
| find your way around 25 documents | [`docs/README.md`](docs/README.md) |
| know what changed or was superseded | [`docs/CHANGELOG.md`](docs/CHANGELOG.md) |
| run something | [Setup](#setup), then [Validate and run](#validate-and-run) |
| interpret an estimate | [Econometric guardrails](#econometric-guardrails) — read before quoting numbers |

Two model families live here and are deliberately kept apart: the **governed
content model** (D0, FE1-FE4) and the **separate position model** (M0-M6). Merging
them would break the comparability rules each was estimated under.

## Application modes

### ChatGPT Bright Data Audit

Upload a Bright Data ChatGPT results export, optionally join a prompt manifest,
and inspect records, sources, scraping, questions, intent, brand visibility,
content features, and reports.

### Content Econometrics QA

The QA workspace reads the validated Area Condo econometrics package and offers:

- scrape and measurable-content coverage;
- scraped-versus-live split-screen page inspection;
- prompt-level source exploration;
- general and real-estate taxonomy exploration;
- read-only notebook 09 and notebook 11 model tables;
- with-versus-without feature contribution diagnostics for the validated M2 and W1 models;
- manual scrape and taxonomy reviews stored separately in SQLite.

Choose **Previous Area Condo 500** in the sidebar to reuse the existing 500-prompt
manifest, normalized source tables, model package, and 2,881 crawler snapshots.
The 1.3 GB raw Bright Data output is retained for lineage but is not loaded into
the interactive app. A custom econometrics package path can also be selected.

Live pages are embedded only on demand. Some sites block iframe embedding with
`X-Frame-Options` or CSP `frame-ancestors`; the interface retains an open-page
fallback for those sites.

## Pipeline

```text
Bright Data ChatGPT output + prompt manifest
  -> normalize prompts and surfaced sources
  -> normalize URLs and remove tracking parameters
  -> Bright Data Crawler API scrape and raw snapshot cache
  -> parse and audit extraction quality
  -> general page-function and real-estate taxonomy
  -> source-appearance and URL-level feature tables
  -> notebook 07 scrape/content analysis
  -> notebook 08 final pre-LPM diagnostics
  -> notebook 09 content-feature econometrics + interpretation patch
  -> notebook 10 writing/factual-density feature layer
  -> notebook 11 writing/factual-density econometrics
  -> notebook 12 HTML document-structure and generated-Markdown QA
  -> read-only QA frontend and manual review export
```

The full audit contains 500 prompts. The measurable-content LPM sample contains
498 prompts because two prompts have no measurable content observations.

## Repository boundary

Source code and clean notebooks are versioned here. Raw crawler responses,
generated CSVs, model outputs, and figures stay outside Git. The existing
research archive is configured through `.env`:

```bash
# example — point these at wherever your research archive actually lives
CITESCOPE_RESEARCH_DATA_DIR=/path/to/CompareSearch-v2-clean
```

Use `CITESCOPE_ECONOMETRICS_DATA_DIR` and
`CITESCOPE_ECONOMETRICS_OUTPUT_DIR` when inputs and generated outputs live in
different locations.

## Setup

```bash
git clone https://github.com/keyboardban/citescope.git
cd citescope
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

Add a freshly rotated Bright Data key only when live scraping is required:

```bash
BRIGHTDATA_API_KEY=
BRIGHTDATA_PROVIDER_MODE=crawler_api
BRIGHTDATA_CRAWLER_DATASET_ID=
BRIGHTDATA_CRAWLER_ASYNC=true
BRIGHTDATA_CRAWLER_ENDPOINT=https://api.brightdata.com/datasets/v3/trigger
```

Never commit `.env` or paste bearer tokens into source files or notebooks.

## Validate and run

Validate the external econometrics package before opening or rerunning models:

```bash
.venv/bin/python scripts/v2_validate_econometrics_migration.py
```

Expected status:

```text
migration_parity_passed
```

Run the application:

```bash
.venv/bin/streamlit run app.py
```

Then open `http://localhost:8501`.

Run all tests:

```bash
.venv/bin/python -m compileall -q src ui scripts tests app.py
.venv/bin/pytest -q
```

## Econometric guardrails

- Start substantive reporting with M1 and M2.
- Do not use final enriched page type as a main control.
- Use `page_type_url_seed_general_collapsed` only as a sensitivity.
- Report M5 strong-content and M10 outlier sensitivity before interpretation.
- Keep answer-derived similarity, source position, observed rank, source origin,
  and citation-rate proxies out of the main content model.
- Treat content availability as structured missingness, not random noise.

The full notebook sequence and data contract are documented in
[`docs/CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md`](docs/CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md).
The deterministic website/page classification method is documented in
[`docs/GENERAL_PAGE_TAXONOMY_RULE_V2.md`](docs/GENERAL_PAGE_TAXONOMY_RULE_V2.md).
Machine-readable sample and artifact checks live in
[`config/econometrics_pipeline_manifest.json`](config/econometrics_pipeline_manifest.json).

## Separate position model

The position-focused M0-M6 analysis is intentionally isolated from the governed
D0-FE4 content model. It uses direct-answer placement, verified-table placement,
H2/H3 question-heading placement, and standardized total numeric-evidence
density, with prompt fixed effects and allowed controls. Total density is the
number of validated numeric-evidence blocks per 1,000 total main-content tokens.
The separate `numeric_evidence_early_share` position extension is the proportion
of those blocks in the first half and is missing when a page has no numeric
evidence; it is not included in primary M5.

The position model preserves the original Gemini taxonomy in
`page_type_detailed` and `source_type_detailed`, while regressions use
deterministic six-class controls. `page_type_model_6` combines editorial/news,
landing/contact/support, and small residual page functions. `source_type_model_6`
combines marketplace/directory, blog/news, review/community, and residual
sources. Source type is stabilized to one modal class per domain using unique
URLs; exact top-class ties become `other_or_unknown`, and confidence is written
to the domain audit table.

```bash
.venv/bin/python scripts/v2_run_position_model.py
```

Outputs are written only to `outputs/position_model_v1/`. The Streamlit option
`Position Model — New` reads those files without recomputing the old model.
