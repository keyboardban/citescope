# CiteScope Econometrics EDA Pipeline

This document describes the offline EDA pipeline used before building CiteScope's
future econometric layer. The pipeline is intentionally pre-model: it audits row-level
source data, feature availability, leakage risk, missingness, shape, sparsity,
collinearity, and possible feature-engineering choices. It does not estimate the final
LPM, fixed-effects, stratified, or causal model layer.

## 1. Purpose

The EDA pipeline answers practical design questions before econometric modeling:

- Which source-level features exist in the current data?
- Which features are missing, structurally sparse, or only available after scraping?
- Which columns are leakage, outcome-derived, answer-derived, or diagnostic-only?
- Which binary content flags are associated with higher or lower cited rate?
- Which numeric features look roughly linear, nonlinear, threshold-like, or too sparse?
- Which features may need log transforms, bins, thresholds, splines, or composites?
- Which intent and page-type cells have enough sample size for future regression?
- Which variables should be excluded, diagnostic-only, sensitivity-only, or candidates for future LPM models?

All language is descriptive. The pipeline reports observational associations among
surfaced sources. It does not claim that a feature causes citation, and it does not
call more-only sources rejected.

## 2. Unit Of Analysis

One row represents one surfaced source appearance.

The outcome is:

```text
cited = 1  explicitly cited source
cited = 0  surfaced / more-only source that was not explicitly cited
```

The comparison set is therefore surfaced sources only. It is not a reconstruction of
ChatGPT's internal retrieval set, and it cannot tell us why a source was or was not
cited.

## 3. High-Level Data Flow

```text
data/chatgpt/*.json
  saved ChatGPT Bright Data audit snapshots

data/raw/{run_id}__scrape_items.json
  optional saved Apify scrape payloads for the same runs

scripts/export_econometrics_rows.py
  flattens JSON snapshots into one source-appearance table

data/exports/econometrics_row_level_sources.csv
  real row-level CSV used by EDA

src/econometrics_eda/diagnostics.py
  reusable loader, schema checks, feature groups, diagnostics, recommendations

notebooks/econometrics_eda/01_feature_diagnostics_before_lpm.ipynb
scripts/run_econometrics_eda.py
  EDA execution surfaces

outputs/econometrics_eda/
  CSV summaries, warnings, metadata, and plots
```

## 4. Main Files

| File | Role |
|---|---|
| `src/econometrics_eda/export_rows.py` | Offline JSON-to-row-level CSV exporter. Reads saved ChatGPT snapshots and saved scrape payloads. Does not call external APIs. |
| `scripts/export_econometrics_rows.py` | CLI wrapper for the exporter. Default output is `data/exports/econometrics_row_level_sources.csv`. |
| `src/econometrics_eda/diagnostics.py` | Core EDA library: loading, strict outcome validation, feature classification, leakage exclusion, availability, binary diagnostics, numeric shape checks, intent/page-type cells, correlation/VIF, LightGBM discovery, recommendations. |
| `scripts/run_econometrics_eda.py` | Headless CLI EDA runner for real row-level CSVs. Supports `--input`, `--output`, `--allow-demo`, bins, thresholds, and optional LightGBM. |
| `notebooks/econometrics_eda/01_feature_diagnostics_before_lpm.ipynb` | Notebook version of the EDA with plots and narrative sections. |
| `outputs/econometrics_eda/` | Generated EDA artifacts. |
| `tests/test_econometrics_export_rows.py` | Tests for exporter, strict `cited` parsing, demo detection, real CSV preference, and no-real-CSV errors. |

## 5. Creating The Real Row-Level CSV

Run:

```bash
.venv/bin/python scripts/export_econometrics_rows.py
```

By default, this reads:

```text
data/chatgpt/*.json
data/raw/{run_id}__scrape_items.json
```

and writes:

```text
data/exports/econometrics_row_level_sources.csv
data/exports/econometrics_row_level_sources.summary.json
```

The exporter:

1. Reads each saved ChatGPT run snapshot.
2. Skips stale zero-source snapshots by default.
3. Loads matching saved scrape payloads when present.
4. Recomputes ChatGPT source features with the lexical similarity engine.
5. Joins record metadata, source labels, prompt-side features, scrape-dependent features, and brand/content features when available.
6. Writes one row per surfaced source appearance.

The current generated export has:

```text
rows: 11764
cited: 4398
more-only: 7366
cited_rate: 0.374
runs exported: 17
empty snapshots skipped: 9
```

These counts describe the current local workspace and can change when new snapshots
are added.

## 6. Loader Policy

The EDA loader no longer auto-selects the review-packet sample CSV.

Default behavior:

1. If `--input` or `CITESCOPE_ROWS_CSV` is provided, use that explicit path.
2. Otherwise, search for real CSV exports only under:
   - `data/exports/`
   - `data/runs/`
   - `data/chatgpt/`
3. Prefer files with names like:
   - `econometrics_row_level_sources.csv`
   - `row_level_sources.csv`
   - `source_features.csv`
   - `features.csv`
   - `analysis_export.csv`
4. If no real CSV exists, stop with a helpful error that suggests running:

```bash
.venv/bin/python scripts/export_econometrics_rows.py
```

The loader does not silently fall back to demo data.

## 7. Demo And Sample Data Policy

Paths containing any of these markers are treated as demo/sample:

```text
econometrics_review_packet
03_sample_data
sample
demo
synthetic
fixture
test
```

That means this file is always demo:

```text
econometrics_review_packet/03_sample_data/sample_row_level_data.csv
```

If demo/sample data is selected while demo mode is disabled, the loader raises:

```text
Refusing to run EDA on demo/sample data without --allow-demo true.
```

Demo mode is only for smoke testing:

```bash
.venv/bin/python scripts/run_econometrics_eda.py \
  --input econometrics_review_packet/03_sample_data/sample_row_level_data.csv \
  --allow-demo true
```

When demo is allowed, outputs include clear demo warnings in:

```text
eda_warnings.csv
run_metadata.json
console output
```

## 8. Running EDA On Real Data

Preferred workflow:

```bash
.venv/bin/python scripts/export_econometrics_rows.py
.venv/bin/python scripts/run_econometrics_eda.py --allow-demo false
```

Run with an explicit CSV:

```bash
.venv/bin/python scripts/run_econometrics_eda.py \
  --input data/exports/econometrics_row_level_sources.csv \
  --output outputs/econometrics_eda/latest \
  --allow-demo false
```

Run the notebook headlessly:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --inplace \
  notebooks/econometrics_eda/01_feature_diagnostics_before_lpm.ipynb
```

Or open and run the notebook interactively.

The notebook also supports:

```bash
export CITESCOPE_ROWS_CSV=/absolute/path/to/real_row_level_sources.csv
export CITESCOPE_ALLOW_DEMO=false
```

## 9. Input Schema

The only required outcome column is:

```text
cited
```

`cited` is strictly validated. Accepted values are binary labels such as:

```text
1, 0, true, false, yes, no, cited, more_only
```

Missing or non-binary values fail fast.

Common row-level columns include:

| Group | Columns |
|---|---|
| Identity | `run_id`, `record_id`, `source_id`, `prompt_id`, `url`, `normalized_url`, `canonical_url`, `domain` |
| Prompt context | `prompt`, `prompt_text`, `intent`, `topic`, `language`, `country`, `expected_source_types` |
| Labels | `cited`, `cited_label`, `source_group`, `is_more_only`, `source_origin` |
| Placement | `source_position`, `observed_rank`, `log1p_source_position` |
| Source metadata | `title`, `description`, `source_type`, `institutional_official`, `official_source`, `brand_official_candidate` |
| Scrape stats | `scrape_success`, `scraped_ok`, `word_count`, `char_count`, `heading_count`, `freshness_days`, `truncated`, `used_char_count`, `original_char_count` |
| Prompt-side similarity | `title_prompt_similarity`, `description_prompt_similarity`, `page_prompt_similarity`, `max_chunk_prompt_similarity`, `relevance_score_prompt_only` |
| Answer-side similarity | `page_answer_similarity`, `max_chunk_answer_similarity` |
| Content flags | `has_faq`, `has_price_or_package`, `has_contact_info`, `has_bullets`, `has_table`, `has_author`, `has_reviewer`, `has_schema`, `has_step_by_step`, `has_booking_or_appointment`, `has_phone_number`, `has_email`, `has_location_info`, `has_opening_hours`, `page_type` |

Missing optional columns are skipped gracefully and listed in warnings.

## 10. Leakage And Diagnostic-Only Rules

The pipeline separates descriptive diagnostics from safe candidate predictors.

Excluded as leakage or outcome-derived:

```text
cited_label
is_more_only
source_group
source_origin
page_answer_similarity
max_chunk_answer_similarity
page_output_sim
max_chunk_output_sim
brand_appeared_in_answer
answer_overlap
domain_citation_rate
domain_citation_rate_loo
brand_official_candidate
```

Diagnostic-only placement features:

```text
source_position
log1p_source_position
observed_rank
```

Selection and availability diagnostics:

```text
scraped_ok
scrape_success
content_feature_available
brand_matched_for_content_features
content_feature_missing_reason
truncated
used_char_count
original_char_count
char_count
```

Identity columns are candidates for clustering or fixed effects, not ordinary
predictors:

```text
domain
answer_id
prompt_id
record_id
run_id
canonical_url
```

## 11. Feature Engineering Performed By EDA

`diagnostics.engineer_proposed_features()` adds diagnostic features used for EDA:

| Feature | Meaning |
|---|---|
| `structure_score` | Sum of available structure flags such as bullets, tables, headings, schema. |
| `answer_ready_score` | Sum of FAQ, step-by-step, table, and bullet flags. |
| `access_score` | Sum of contact, location, opening-hours, booking, phone, and email flags. |
| `trust_signal_score` | Sum of author, reviewer, published/updated date, and institutional flags. |
| `commercial_info_score` | Sum of price/package and booking flags. |
| `relevance_score_prompt_only` | Row-wise max of prompt-side similarity features. |
| `domain_seen_count` | Domain visibility count among surfaced sources, not true authority. |
| `domain_seen_count_loo` | Leave-one-out domain visibility count. |
| `domain_citation_rate_loo` | Outcome-derived leave-one-out proxy; diagnostic/sensitivity only, never safe predictor. |
| `log1p_*` | Shape helpers for skewed numeric variables. |
| `content_feature_available` | Whether content flags were measurable. |
| `content_feature_missing_reason` | Diagnostic reason content features are missing, when inferable. |

These are EDA helpers, not final approved model variables.

## 12. EDA Steps

### 12.1 Feature Inventory

Creates:

```text
feature_inventory_detected.csv
```

This lists every detected feature with dtype, missing rate, unique count, role guess,
and risk flags.

### 12.2 Availability And Missingness

Creates:

```text
feature_availability_summary.csv
plots/02_top_missing.png
plots/03_content_avail_by_cited.png
plots/03b_scrape_by_cited.png
```

This identifies high-missingness features and whether availability differs by cited
status. Such differences are selection signals, not feature effects.

### 12.3 Binary Feature Cited-Rate Diagnostics

Creates:

```text
binary_feature_cited_rate_summary.csv
plots/04_binary_cited_rate.png
```

For each binary feature, the pipeline computes:

```text
n_available
n_0
n_1
cited_rate_0
cited_rate_1
diff_pp
min_group_size
warning
```

`diff_pp` is a descriptive cited-rate gap among surfaced sources, not a causal
effect.

### 12.4 Numeric Shape Diagnostics

Creates:

```text
numeric_feature_shape_summary.csv
plots/05_numeric_shapes.png
```

The pipeline quantile-bins numeric features and estimates whether the cited-rate
shape looks:

```text
raw
log1p
threshold
bins
spline_or_bins
diagnostic_only
```

Small samples and high missingness are flagged.

### 12.5 Intent By Page-Type Cell Diagnostics

Creates:

```text
intent_page_type_cell_summary.csv
plots/06_cell_n.png
plots/06b_cell_rate.png
```

For each `intent x page_type` cell, it computes:

```text
n
cited_n
more_only_n
cited_rate
unique_domain
unique_prompt_or_answer
regression_eligible
status
```

Cells below thresholds remain descriptive-only. The notebook does not fit regressions
inside sparse cells.

### 12.6 Interaction Candidate Diagnostics

Creates:

```text
plots/07_interactions.png
```

This checks whether selected content flag associations appear concentrated within
intent families, such as price, comparison, local access, medical safety, official,
or service intents.

This is a prioritization heuristic for future model design, not an interaction model.

### 12.7 Correlation And VIF

Creates:

```text
correlation_matrix.csv
vif_summary.csv
plots/08_correlation.png
```

These are computed only on low-missing, leakage-safe numeric and binary candidates.
Highly correlated content flags suggest composite scores or narrower model scopes.

### 12.8 Optional LightGBM Discovery

Creates, when enabled and successful:

```text
lightgbm_feature_importance.csv
plots/09_lightgbm_importance.png
```

LightGBM is discovery-only. It uses leakage-safe pre-output features and group-aware
validation when possible. The current implementation prefers grouping by `prompt_id`,
then `answer_id`, then `record_id`, then `domain`.

Missing feature values are imputed inside this discovery diagnostic only. Missingness
tables remain the source of truth for econometric design.

LightGBM importance is not causal importance.

### 12.9 Feature Engineering Recommendations

Creates:

```text
feature_engineering_recommendations.csv
```

This combines availability, binary associations, numeric shapes, role guesses,
small-cell warnings, collinearity, and leakage rules into recommendations such as:

```text
exclude_leakage
use_as_diagnostic_only
use_in_sensitivity_only
keep_as_control
keep_as_focal
transform_log1p
transform_threshold
transform_bins
combine_into_composite_score
```

The recommendations are suggestions for human review, not automatic model decisions.

## 13. Output Artifacts

Typical notebook outputs:

```text
outputs/econometrics_eda/
  binary_feature_cited_rate_summary.csv
  correlation_matrix.csv
  eda_warnings.csv
  eda_warnings.md
  feature_availability_summary.csv
  feature_engineering_recommendations.csv
  feature_inventory_detected.csv
  intent_page_type_cell_summary.csv
  lightgbm_feature_importance.csv
  numeric_feature_shape_summary.csv
  run_metadata.json
  vif_summary.csv
  plots/
    01_outcome_balance.png
    02_top_missing.png
    03_content_avail_by_cited.png
    03b_scrape_by_cited.png
    04_binary_cited_rate.png
    05_numeric_shapes.png
    06_cell_n.png
    06b_cell_rate.png
    07_interactions.png
    08_correlation.png
    09_lightgbm_importance.png
```

The CLI runner writes the same core CSV diagnostics, plus:

```text
eda_warnings.csv
run_metadata.json
```

## 14. Warning And Metadata Contract

`eda_warnings.csv` is machine-readable. It contains:

```text
warning_id
level
message
is_demo
```

`run_metadata.json` contains:

```text
source
path
is_demo
rows
cited
more_only
notes
warnings
```

When demo data is explicitly allowed, both files mark `is_demo=true` and include the
smoke-test warning.

## 15. Current Loader Behavior Checks

Expected behavior:

```bash
# Real export is selected by default when present.
.venv/bin/python scripts/run_econometrics_eda.py --allow-demo false

# Review-packet sample is refused by default.
.venv/bin/python scripts/run_econometrics_eda.py \
  --input econometrics_review_packet/03_sample_data/sample_row_level_data.csv \
  --allow-demo false

# Review-packet sample runs only as explicit smoke test.
.venv/bin/python scripts/run_econometrics_eda.py \
  --input econometrics_review_packet/03_sample_data/sample_row_level_data.csv \
  --allow-demo true
```

The tests cover:

- sample row-level path is detected as demo
- loader does not auto-select demo by default
- `--allow-demo true` permits demo data
- no real CSV produces a helpful error
- real CSV under `data/exports/econometrics_row_level_sources.csv` is preferred

Run tests:

```bash
.venv/bin/python -m pytest -q tests/test_econometrics_export_rows.py
.venv/bin/python -m pytest -q
```

## 16. What This Pipeline Does Not Do

This EDA pipeline does not:

- infer ChatGPT's internal retrieval set
- explain why a page was or was not cited
- call more-only sources rejected
- treat similarity as proof of source use
- use answer-derived features as predictors
- use outcome-derived features as safe predictors
- use `source_position` as a main LPM feature
- estimate final LPM, fixed-effects, stratified, or causal models

Those steps belong to a later econometric implementation after a human reviews the
EDA outputs.

## 17. Recommended Operating Procedure

For real analysis:

1. Refresh saved ChatGPT snapshots under `data/chatgpt/`.
2. Refresh scrape payloads under `data/raw/` if content features are needed.
3. Run:

```bash
.venv/bin/python scripts/export_econometrics_rows.py
```

4. Run:

```bash
.venv/bin/python scripts/run_econometrics_eda.py --allow-demo false
```

5. Review:

```text
outputs/econometrics_eda/run_metadata.json
outputs/econometrics_eda/eda_warnings.csv
outputs/econometrics_eda/feature_inventory_detected.csv
outputs/econometrics_eda/feature_engineering_recommendations.csv
```

6. Only after that review, decide which variables move into the future econometric
model layer.
