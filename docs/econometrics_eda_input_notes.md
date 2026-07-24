# Econometrics EDA — input data notes

_Repo inspection for the offline EDA / feature-diagnostics layer. Describes where row-level source
data comes from and what each column means, so the notebook has a documented input contract. No
model, no causal claim._

## 1. Where the row-level source data comes from
There is no single stored "econometric table" in the repo; it is **assembled from the ChatGPT
Bright Data pipeline** at run time:

- **Parser** — `src/brightdata.py::parse_run()` / `extract_sources()` turns a Bright Data export
  (array of ChatGPT responses) into `records[] → sources[]`. **This is where the outcome `cited`
  is built** (citations-flag ∪ `links_attached` fallback, deduped by `normalized_url`, "cited wins").
- **Main per-source feature table** — `src/chatgpt_pipeline.py::build_features()` → one row per
  surfaced source (labels, placement, source_type, similarities; scrape-dependent stats when
  scraped). Exported by `src/report.py::chatgpt_features_csv()`.
- **Content features** — `src/brand_visibility.py::build_source_pages()` +
  `extract_content_features()` add `has_*` flags and `page_type`, **only for brand-matched +
  scraped sources**. Exported by `report.py::brand_source_pages_csv()`.
- **Row builder for EDA** — `src/econometrics_eda/diagnostics.py::build_rows_from_run()` joins the
  two tables + record metadata into one row-level table.
- **Real row-level exporter** — `src/econometrics_eda/export_rows.py` and
  `scripts/export_econometrics_rows.py` read `data/chatgpt/*.json` plus saved scrape payloads and
  write `data/exports/econometrics_row_level_sources.csv`.
- **Loader policy** — `load_econometric_rows()` resolves a data source in this order: explicit CSV
  path → real repo CSV under `data/exports/`, `data/runs/`, or `data/chatgpt/`. It does **not**
  auto-select review-packet, sample, demo, synthetic, fixture, or test CSVs.

Persisted runs live under `data/chatgpt/*.json` (gitignored). The real EDA CSV should be generated
at `data/exports/econometrics_row_level_sources.csv`. The review-packet sample
(`econometrics_review_packet/03_sample_data/sample_row_level_data.csv`) is demo data and requires
explicit `--allow-demo true`.

## 2. Columns currently available (row-level table)
Identity/prompt: `run_id, record_id, prompt_id, prompt_text, intent, language, topic, country`.
Source identity: `url, normalized_url, canonical_url, domain, title, description`.
Placement/labels: `source_origin, source_position, observed_rank, source_group, cited,
is_more_only`. Type: `source_type, institutional_official, brand_official_candidate, page_type`.
Scrape stats: `scraped_ok/scrape_success, word_count, char_count, heading_count, freshness_days,
truncated, used_char_count, original_char_count`. Content flags (brand-matched+scraped only):
`has_faq, has_price_or_package, has_contact_info, has_bullets, has_table, has_author, has_reviewer,
has_step_by_step, has_booking_or_appointment, has_schema, has_location_info, has_opening_hours,
has_phone_number, has_email, has_published_date, has_updated_date, has_many_headings,
heading_prompt_match, title_contains_intent_terms, answer_like_text_in_first_500_chars, page_type`.
Similarities: `title_prompt_similarity, description_prompt_similarity, page_prompt_similarity,
max_chunk_prompt_similarity` (prompt-side) and `page_answer_similarity, max_chunk_answer_similarity`
(answer-side). Record context: `prompt_is_nonbranded, brand_appeared_in_answer`.

## 3. Which column is the outcome `cited`
**`cited`** (1 = explicitly cited, 0 = surfaced / more-only but not cited). Built in
`brightdata.extract_sources()`. `cited_label` is an identical alias; `source_group`
(`cited`/`more_only`) and `is_more_only` are the outcome in other forms → **not predictors**.

## 4. Which columns are content features
The heuristic `has_*` flags + `page_type` + `heading_prompt_match` + `title_contains_intent_terms`
from `brand_visibility.extract_content_features()`. **Caveat:** present only for **brand-matched +
scraped** sources → large structural missingness (in the demo table, 13 of 35 rows). Treat as
proxies, and scope any content model to the measured subset.

## 5. Which columns are metadata / controls
Categorical: `intent, language, country, topic, page_type, source_type`. Binary control:
`institutional_official`. Numeric: `word_count, heading_count, freshness_days` (scrape-dependent).
Prompt-side similarities are pre-output relevance **proxies**.

## 6. Which columns are leakage or diagnostic-only
- **Outcome-derived / post-output (exclude as predictors):** `cited_label, is_more_only,
  source_group, source_origin, page_answer_similarity, max_chunk_answer_similarity,
  brand_appeared_in_answer`, non-LOO `domain_citation_rate`, and **`brand_official_candidate`**
  (it reads `answer_text` → post-output contamination).
- **Placement, diagnostic-only (never a main predictor):** `source_position`, `observed_rank`
  (observed source-panel placement, mediator-sensitive; **not** an internal/Google rank).
- **Selection indicators (diagnostic-only):** `scraped_ok/scrape_success, truncated,
  used_char_count, original_char_count, content_feature_available,
  brand_matched_for_content_features`.
- **Identities (cluster/FE, not predictors):** `domain, prompt_id, record_id, run_id,
  canonical_url` (+ `answer_id` if added).

## 7. Missing / ambiguous fields
- **`answer_like_text_in_first_500_chars`** — name suggests answer-side but it is computed vs the
  **prompt** (`brand_visibility.extract_content_features`). Ambiguous name → flagged; treated as a
  prompt-side proxy, not leakage.
- **Absent recommended columns:** `scrape_success` (only `scraped_ok` exists),
  `top3_chunk_prompt_similarity_mean`, `answer_id`, `content_feature_missing_reason`,
  `domain_seen_count(_loo)`, `relevance_score_prompt_only`, composites — the last several are
  **engineered** by `diagnostics.engineer_proposed_features()`; `answer_id` maps to
  `(run_id, record_id)`.
- **`record_id` is not globally unique** (resets per run) → use `(run_id, record_id)` or
  `prompt_id` as the prompt key.
- **`page_answer_similarity` circular** — the answer may be generated from cited sources.
- **`freshness_days`** — high, possibly non-random missingness (only some sites expose dates).
