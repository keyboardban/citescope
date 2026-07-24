# Econometrics EDA plan

_How the offline EDA / feature-diagnostics layer works and why it exists. This layer fits **no**
model and makes **no** causal claim. Every relationship is an **observational association among
surfaced sources** (cited vs surfaced-but-not-cited)._

## 1. Purpose
Decide — **before** writing any econometric model — which features to **keep / transform /
exclude / engineer**, by measuring how each existing and proposed feature relates to the binary
outcome `cited` (linear, nonlinear, sparse, missing, collinear, or interaction-specific). Outputs
feed a human feature-decision review, not a model.

## 2. Why it is separate from the frontend
The engine (`src/`) is deliberately Streamlit-free and testable headlessly. This EDA lives in
`src/econometrics_eda/` (reusable, plot-free) + a notebook (`notebooks/econometrics_eda/`) +
static exports (`outputs/econometrics_eda/`). **No frontend changes.** A human econometrician can
run and read it offline; nothing depends on the dashboard.

## 3. Why it runs before econometric modeling
The previous econometric layer was rolled back because methodology was not planned first. This
layer surfaces the data-generating problems that dictate the model design — **selection**
(surfaced-only comparison set; scrape selection; brand-match selection), **leakage** (answer-side
and outcome-derived features), **few clusters**, **collinearity**, and **nonlinear shape** — so
those decisions are made by a human, on evidence, up front.

## 4. How binary features are evaluated
`binary_feature_cited_rate_summary()` reports, per flag: `n_0, n_1, cited_rate_0, cited_rate_1,
diff_pp = (rate_1 − rate_0)·100`, `min_group_size`, and warnings (`small_group`, `no_variation`,
`high_missing`). `diff_pp` is an **association in percentage points**, not an effect. Decision
logic: no variation → diagnostic-only; small minority group → sensitivity/descriptive; sufficient
variation → candidate **focal** (content flags) in the scoped content model; correlated siblings →
prefer a **composite score**.

## 5. How numeric features are evaluated
`numeric_feature_shape_summary()` quantile-bins each feature and plots cited-rate per bin, plus
Pearson/Spearman hints and `roughly_monotonic` / `roughly_nonlinear` flags, yielding a
`transform_candidate`: **raw** (roughly linear) · **log1p** (rises then plateaus) · **threshold**
(single step) · **bins/spline** (non-monotonic/complex) · **diagnostic_only** (too few points or
high missingness). Shapes are unstable at small N and are flagged as such.

## 6. How intent × page_type cells are evaluated
`intent_page_type_cell_summary()` reports `n, cited_n, more_only_n, cited_rate, unique_domain,
unique_prompt_or_answer` and a **regression-eligibility** gate (defaults: `n≥50, cited_n≥10,
more_only_n≥10, unique_domain≥10, unique_prompt≥10`). Ineligible cells stay **descriptive-only**;
the future layer must not silently fit them. Heatmaps show sample size and cited-rate per cell.

## 7. How LightGBM is used (discovery only)
Optional. Group-aware CV (GroupKFold by `prompt_id`/`answer_id`/`domain`; falls back to row-level
KFold with a loud warning) over **leakage-safe pre-output features only**. Gain importance is used
to spot **nonlinearities, thresholds, and interaction candidates** — **never** as causal or even
"model" importance. Skips gracefully if LightGBM (or `libomp`) is unavailable.

## 8. How leakage features are excluded
`LEAKAGE_EXCLUDE` (outcome-derived / post-output: `cited_label, is_more_only, source_group,
source_origin, page_answer_similarity, max_chunk_answer_similarity, page_output_sim,
max_chunk_output_sim, brand_appeared_in_answer, answer_overlap`, non-LOO `domain_citation_rate`,
and answer-reading `brand_official_candidate`) and `PLACEMENT_DIAGNOSTIC_ONLY`
(`source_position, log1p_source_position, observed_rank`) are removed from every predictive
diagnostic by `safe_predictor_features()`. `source_position` is **diagnostic-only**.
`domain_citation_rate` is allowed **only** in a leave-one-out form, labelled proxy/sensitivity.

## 9. How outputs feed the future econometric layer
`feature_engineering_recommendations.csv` (per-feature `recommended_action` +
`recommended_lpm_status`) and `feature_inventory_detected.csv` become the **feature decision
registry** (`docs/econometrics_feature_engineering_decisions.md`). Only after a human reviews
those does the model layer get built. Model design targets (pooled LPM, scoped content LPM,
fixed-effects / intent- / page-type-stratified models, two-way clustered inference, few-cluster
wild bootstrap) live in the review packet's `human_decisions_needed.md` — **not implemented here**.

## Files
| file | role |
|---|---|
| `src/econometrics_eda/export_rows.py` | offline JSON-to-row-level CSV exporter for `data/exports/econometrics_row_level_sources.csv` |
| `src/econometrics_eda/diagnostics.py` | reusable, plot-free diagnostics (load, classify, availability, binary/numeric summaries, cells, corr/VIF, LightGBM, recommendations) |
| `scripts/export_econometrics_rows.py` | CLI exporter from `data/chatgpt/*.json` to the real row-level EDA CSV |
| `scripts/run_econometrics_eda.py` | headless CLI EDA runner with `--input` and `--allow-demo` |
| `notebooks/econometrics_eda/01_feature_diagnostics_before_lpm.ipynb` | runnable EDA (13 sections, plots) |
| `outputs/econometrics_eda/*.csv, eda_warnings.md, plots/*.png` | exported diagnostics |
| `docs/econometrics_eda_input_notes.md` | input-data contract |
| `docs/econometrics_eda_pipeline.md` | detailed end-to-end pipeline guide |
| `docs/econometrics_feature_engineering_decisions.md` | decision registry (filled from the run) |

## How to run
```bash
source .venv/bin/activate
pip install pandas numpy matplotlib statsmodels scikit-learn nbformat nbconvert ipykernel
# optional discovery model: pip install lightgbm   (macOS also: brew install libomp)

# Step 1 — build the real row-level CSV from saved ChatGPT JSON snapshots:
python scripts/export_econometrics_rows.py

# option A — execute the notebook headlessly:
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/econometrics_eda/01_feature_diagnostics_before_lpm.ipynb

# option B — run the headless EDA CLI:
python scripts/run_econometrics_eda.py --allow-demo false
```
With no real row-level CSV present, `load_econometric_rows()` now stops with a clear message that
suggests running `scripts/export_econometrics_rows.py`. It does **not** silently fall back to the
review-packet sample. Demo/sample inputs require explicit `allow_demo=True` or
`--allow-demo true`, and outputs are marked as smoke-test only.
