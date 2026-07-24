# ChatGPT Content Econometrics Pipeline

## Interactive Econometrics Frontend

The read-only Streamlit Econometrics view consumes versioned lightweight artifacts from `tables/econometrics_frontend/`. Generate and validate them after precomputed notebook/model outputs change:

```bash
.venv/bin/python scripts/v2_build_econometrics_frontend_artifacts.py
.venv/bin/python scripts/v2_validate_econometrics_frontend_artifacts.py \
  /path/to/content_econometrics_ai_package/tables/econometrics_frontend
```

The builder does not fit a model. It normalizes existing model tables, computes descriptive Wilson intervals and diagnostics, and prepares deterministic compact examples and comparison pairs. Streamlit does not load raw Bright Data JSON or execute notebooks.

See `docs/econometrics_frontend_product_spec.md`, `docs/econometrics_frontend_data_contract.md`, `docs/econometrics_frontend_interpretation_rules.md`, and `docs/econometrics_frontend_user_guide.md`.

## Scope

This workspace studies observable ChatGPT sources from the Bright Data audit. The
comparison is cited versus more-only among surfaced sources. It does not claim to
observe ChatGPT's complete retrieval set, and more-only does not mean rejected.

The current study is an area-condo / SCOPE-relevant nonbranded audit. The full
audit contains 500 prompts. The measurable-content LPM sample contains 498
prompts because two prompts have no measurable content observations.

## Data boundary

Code and source notebooks live in this repository. Raw crawler responses,
generated tables, figures, and model outputs remain outside Git. Configure their
location in `.env`:

```bash
CITESCOPE_RESEARCH_DATA_DIR=/Volumes/ExtremeSD/Metier/Research/CompareSearch-v2-clean
```

The more specific `CITESCOPE_ECONOMETRICS_DATA_DIR` and
`CITESCOPE_ECONOMETRICS_OUTPUT_DIR` variables override that root when needed.

The frontend provides a **Previous Area Condo 500** preset. It loads the full
prompt manifest plus the validated lightweight tables and crawler snapshots. It
does not deserialize the 1.3 GB raw Bright Data JSON during a Streamlit rerun.

## Canonical notebook sequence

1. `07_area_condo_brightdata_content_analysis_master.ipynb`: scrape and content QA.
2. `08_area_condo_final_pre_lpm_master_notebook.ipynb`: taxonomy, missingness, and pre-LPM diagnostics.
3. `09_area_condo_content_feature_econometrics.ipynb`: M1/M2 content-feature LPMs and sensitivities.
4. `09_area_condo_content_feature_econometrics_v2_interpretation_patch.ipynb`: canonical interpretation patch.
5. `10_area_condo_writing_factual_density_feature_layer.ipynb`: writing and factual-density extraction.
6. `11_area_condo_writing_factual_density_econometrics.ipynb`: writing-feature models and robustness.
7. `12_area_condo_document_structure_features.ipynb`: HTML-first structure and generated-Markdown QA.

Notebook 09's interpretation patch is required after the base notebook. It does
not replace the original estimates; it strengthens covariance and robustness
reporting.

Notebook 12 is an additive descriptive layer. It extracts general document
structure from the archived HTML and does not change the frozen notebook 10/11
features or estimates. Run the extractor before opening it:

```bash
.venv/bin/python scripts/v2_run_document_structure_features.py
```

## Main model rules

- Start substantive reporting with M1 and M2.
- Use versioned Gemini page-function family and source/site type in taxonomy-adjusted sensitivity models.
- Do not use Gemini taxonomy as a focal M1/M2 predictor because its classification may use scraped body content.
- Retain `page_type_url_seed_general_collapsed` only as a rule-v2 robustness comparison.
- Report M5 strong-content and M10 outlier sensitivity before interpretation.
- Keep answer-derived similarity, source position, observed rank, and citation-rate
  proxies out of the main content model.
- Interpret results as conditional associations among surfaced sources, not causal
  effects or web-wide citation probabilities.

## Core-General feature registry

The pre-estimation feature specification is stored in
`config/core_general_content_feature_dictionary.csv` and copied into the external
AI package as `tables/core_general_content_feature_dictionary.csv`. It separates
Core-General, Commerce-General, paused vertical-specific, diagnostic, sensitivity,
and leakage-excluded fields. Regenerate it without running any model:

```bash
.venv/bin/python scripts/v2_build_core_general_feature_registry.py
```

### Provenance-aware table layer

The Core-General table specification separates detection provenance, verification
state, structure, table-level function, multi-label content signals, and page-level
aggregates. HTML-verified, Markdown-inferred, text-inferred, verified-absent, and
unmeasured states remain distinct; unmeasured values are never silently encoded as
zero. `has_any_data_table` excludes layout/navigation-only tables.

The stable table-level vocabulary is factual/specification, comparison, pricing/plan,
directory/listing, schedule/timeline, transactional/form, layout/navigation, and
unknown/other. Page-level `dominant_table_type` uses a strict unique-winner rule and
returns `mixed_or_unknown` for ties or insufficient evidence.

Pricing table fields are Commerce-General. Real-estate unit-size, bedroom mix,
floor-plan, price-per-area, and project-inventory table fields are paused vertical
extensions. Conceptual `feature_status` does not imply model approval; consult
`qa_status`, `approved_for_model_v1`, `minimum_qa_gate`, and `model_entry_blocker`.

This registry revision does not rerun an econometric model. Generate the complete
table specification and QA artifacts with:

```bash
.venv/bin/python scripts/v2_revise_core_general_table_registry.py
```

`has_verified_html_table` is the sole canonical verified HTML-presence feature.
`has_any_verified_table` and `has_table` are deprecated alias records. Alias records
use `registry_record_type = deprecated_alias` and cannot be selected automatically
for modeling.

Every table-related row declares `feature_granularity` as table-level, page-level,
extraction diagnostic, or registry metadata. Table-level primitives must provide a
`page_aggregation_rule` before they can enter a page-level dataset. Multi-label table
content primitives have separate page-level presence and count aggregates.

Categorical states do not use numeric zero for missingness. `no_table` means measured
absence; `unknown_or_other` means a measured but uncertain table type;
`mixed_or_unknown` means measured tables without a unique dominant type; and NA means
extraction was unavailable. The controlled vocabularies live in
`tables/core_general_table_taxonomy_allowed_values.csv`.

The staged future model starts with one validated page-level field:
`has_any_data_table`, or temporarily `has_verified_html_table` if semantic data/layout
classification has not passed QA. Raw dimensions and table-level primitives remain
diagnostic until extraction, support, rank, VIF, and condition-number checks pass.

## Setup and parity check

```bash
cd /Volumes/ExtremeSD/Metier/Research/CiteScope-content-audit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-econometrics.txt
cp .env.example .env
```

After setting the external research path, validate the migrated package:

```bash
.venv/bin/python scripts/v2_validate_econometrics_migration.py
```

The expected status is `migration_parity_passed`. The contract is stored in
`config/econometrics_pipeline_manifest.json` and includes sample counts, required
columns, canonical notebooks, model artifacts, and interpretation rules.

## Frontend boundary

The future QA frontend will read these validated outputs. It will not execute
notebooks or refit models during a Streamlit rerun. Manual scrape and taxonomy
reviews will be stored separately from the model-ready datasets.

The Econometrics tab reads a precomputed with-versus-without feature comparison
for the validated notebook 09 M2 and notebook 11 W1 models. Regenerate it after
either model dataset changes:

```bash
.venv/bin/python scripts/v2_run_econometric_feature_ablation.py
```

Its R-squared, RMSE, Brier, and MAE changes are in-sample nested-model
diagnostics. They are not causal effects or out-of-sample performance claims.
