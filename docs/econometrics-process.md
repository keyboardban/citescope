# CiteScope — Econometrics Layer: Process & Per-Update Change Log

> How the **position-adjusted citation model** was built, audited, and hardened — update by
> update, with the methodology, the rationale, the before→after, and the files touched.
> Companion to `docs/DEVELOPMENT.md` (Iterations J–K) and the engine source `src/econometrics.py`.
>
> _Last updated: 2026-06-30._

---

## Table of contents

1. [What this layer is (and the one rule it bends)](#1-what-this-layer-is)
2. [The estimator stack (methodology primer)](#2-the-estimator-stack)
3. [Update J.0 — Position-adjusted citation model (foundation)](#3-update-j0)
4. [Update J.1 — Sensitivity & diagnostics extension](#4-update-j1)
5. [Update J.2 — Bug fix: fitted-UI `NameError`](#5-update-j2)
6. [Update J.3 — Calculation-correctness audit](#6-update-j3)
7. [Update K — Careful-reporting upgrade](#7-update-k)
8. [The model ladder (A / B / C / D / FULL)](#8-the-model-ladder)
9. [Output catalog](#9-output-catalog)
10. [Wording rules (the safety contract)](#10-wording-rules)
11. [Test coverage map](#11-test-coverage-map)

---

## 1. What this layer is

CiteScope is a **black-box, observational** AI-citation audit. Its golden rule: *report only
observable patterns; never claim we know the AI's internal retrieval set or why a page was/wasn't
cited.* `more-only` = **surfaced but not cited**, never "rejected".

The econometrics layer is a **scoped, explicit exception** to that rule. It fits a regression of
the citation outcome on observable features and lets a coefficient be read as a **cautious effect
estimate** — but *only* under stated assumptions (exogeneity, positivity, functional form) and a
**signed omitted-variable caveat**. Robust error bars are honest about **noise**, never about an
unobserved confounder.

- **Outcome:** `cited = 1` (explicitly cited) vs `cited = 0` (surfaced / more-only, not cited).
- **Engine:** `src/econometrics.py` — pure functions, no Streamlit, deterministic, behind a
  `statsmodels` import guard so the app degrades to a message if the dep is missing.
- **Wired into:** all three audit modes (Gemini `pipeline`, ChatGPT `chatgpt_pipeline`, `batch`,
  `brand_visibility`) via `analysis.econometric_analysis`, plus `report.py` exports and the UI.

---

## 2. The estimator stack

The methods used (and where each lives), so the change log below is readable:

| Method | What it does | Code |
|---|---|---|
| **Linear Probability Model (LPM)** | OLS on the 0/1 `cited` outcome → coefficients are in **probability points** (easy to read). | `fit_citation_model` |
| **HC3 robust SEs** | Heteroskedasticity-robust error bars (a 0/1 outcome is always heteroskedastic). Default. | `cov_type="HC3"` |
| **Cluster-robust SEs** | When sources nest in prompts/domains, errors correlate within cluster. | `cov_type="cluster"` |
| **Wild cluster bootstrap** | Resampling SE for the few-cluster case (Rademacher signs per cluster). | `_wild_cluster_bootstrap_se` |
| **VIF** | Variance-inflation factor — flags overlapping predictors (wide bars, not bias). | `_vif_map`, `vif_level` |
| **Benjamini–Hochberg (FDR)** | Controls false discoveries across a family of tested features. | `multipletests(..., "fdr_bh")` |
| **Logit + AME** | Logistic cross-check; Average Marginal Effects put it back on the probability scale. | `_logit_ame` |
| **Separation handling** | Detects perfect/quasi-separation (logit diverges) → keep the LPM. | `_logit_ame`, `_separation_rows` |
| **Design matrix** | One-hot (reference level), `log1p`/bins/linear position, median-fill + missing indicator, collinearity drop. | `design_matrix`, `_drop_collinear` |

---

## 3. Update J.0

### Position-adjusted citation model (foundation) · commit `b2cafea`

**Goal.** Replace the univariate "feature ↔ citation correlation" screen with a **multivariate,
error-bar-bearing** model that answers: *what makes a source more likely to be cited, holding the
other features — especially position — fixed, and how sure are we?*

**What was added**

- **`src/econometrics.py`** (new). Core pieces:
  - `build_spec(...)` — declares one regression (focal features, controls, position transform,
    categoricals, cluster key, phase filter, labels, logit/bootstrap toggles).
  - `design_matrix(df, spec)` — builds `(X, y)`: coerces numerics, drops low-coverage /
    zero-variance / low-support columns, one-hot encodes categoricals with the **most-common level
    as the omitted reference**, transforms position (`log1p` default, or `bins`/`linear`),
    **median-fills position + adds a missing indicator**, drops perfectly-collinear (aliased)
    columns, adds the constant, aligns cluster groups.
  - `fit_citation_model(df, spec)` — the orchestrator returning a stable result schema:
    coefficients (estimate / se / ci / t / p / `p_adj` / vif / support / focal flag), diagnostics,
    AME cross-check, assumptions, signed OVB caveat, warnings.
- **Standard errors.** HC3 by default; **cluster-robust** when a usable cluster key exists; for
  **fewer than `MIN_CLUSTERS` (40)** clusters the focal coefficients' SE/CI/p are replaced by the
  **wild cluster bootstrap** (precompute `P = (XᵀX)⁻¹Xᵀ`, draw Rademacher ±1 per cluster, 1,999
  iters, fixed seed → deterministic).
- **Multiplicity.** Benjamini–Hochberg over the focal family → `p_adj` (q-value).
- **Multicollinearity.** Per-coefficient VIF + condition number; high-VIF features flagged.
- **Robustness cross-check.** A **logit** is fit and converted to **Average Marginal Effects**
  (probability points) that should land near the LPM coefficients; **perfect/quasi-separation** is
  detected (warnings captured, non-convergence, `|param|>25`) and the LPM is kept as the headline.
- **Framing.** `config.py` caveats — `CAVEAT_REGRESSION`, `CAVEAT_ASSUMPTIONS`, signed
  `CAVEAT_OVB_GEMINI/CHATGPT/BRAND`, `CAVEAT_FEW_CLUSTERS`, `CAVEAT_SEPARATION`.
- **Wiring.** `analysis.econometric_analysis` → used by `pipeline` (single run, exploratory),
  `chatgpt_pipeline.analyze` (cluster `record_id`), `batch.aggregate` (cluster `run_id`),
  `brand_visibility.position_adjusted_regression`. Rendered by `ui/charts.coefficient_forest` +
  `ui/components.regression_block`; exported in `report.py` (regression section + JSON).
- **Deps.** `statsmodels`, `scipy` (cp314 wheels verified for Python 3.14).

**Before → after.** Before: a single correlation column, no controls, no uncertainty. After: a
position-adjusted coefficient **with a 95% CI**, a clustered/robust SE, a multiplicity correction,
a collinearity flag, and a logit cross-check — plus the old correlation table kept but relabeled
**"unadjusted."**

---

## 4. Update J.1

### Sensitivity & diagnostics extension · commit `4937e53`

**Goal.** One fitted model is a point of view, not an answer. Add a **sensitivity analysis** — fit
the same outcome under several specifications and see which coefficients are **stable**.

**What was added**

- **`econometrics.model_comparison(df)`** — fits a ladder of specs and returns a comparison table
  plus diagnostics:
  - **Model A** — content/page features only.
  - **Model B** — A + source-type / official / brand / `page_type` / `intent`.
  - **Model C** — B + `log1p(source_position)`.
  - **Model D** — C with the collinear similarity features reduced to one combined score.
  - **FULL** — everything incl. all raw similarities, **as a diagnostic to surface VIF**.
- **Diagnostic tables.** VIF table, **anomaly diagnostics** (6 auto-flags: position dominates,
  similarity severe-VIF, negative access features, negative authorship, positive age,
  large page-type), **grouped-feature** interpretation, and an **executive summary** (which excludes
  severe-VIF coefficients from the "strongest" picks because their magnitude is unreliable).
- **Content features for every ChatGPT source.** `chatgpt_pipeline.build_features` now computes the
  heuristic content booleans + `page_type` on **every** source (not just brand pages), so Model A is
  actually fittable.
- **Outputs & UI.** `report.forest_png` (matplotlib, guarded) + 4 CSV exporters
  (`model_comparison` / `vif_diagnostics` / `anomaly_diagnostics` / `feature_group_summary`) +
  `report._sensitivity_section`; `ui/components.sensitivity_block` on the ChatGPT analysis tab + the
  Report-tab downloads.
- **Business caveats.** `config.py` — observational / position-mediator / contact-location /
  similarity / age. **Dep:** `matplotlib`.

**Before → after.** Before: one model, take-it-or-leave-it. After: an **A→D ladder** that shows
whether `has_faq` (etc.) keeps its sign and size as you add source-type, position, and a clean
relevance score — with VIF/anomaly/group context and a forest plot.

---

## 5. Update J.2

### Bug fix — fitted-UI `NameError` · commit `e8d9bb6`

**Symptom.** *"Something went wrong rendering ChatGPT Bright Data Audit: NameError: name 'f' is not
defined."*

**Root cause.** When `sensitivity_block` was inserted into `ui/components.py`, the **tail of
`regression_block`** (the signed-OVB caveat + warnings + assumptions, which reference the loop
variable `f`) was accidentally orphaned into `sensitivity_block`, where `f` doesn't exist. That code
only runs on a **fitted** model with group rows — so the tiny offline demos (which fit nothing)
passed `AppTest`, but a real upload (178+ sources → fitted) crashed.

**Fix.** Moved the block back into `regression_block`'s loop (which also **restored the signed OVB
caveat** that had silently vanished from the UI), removed the orphan, and added
**`test_fitted_chatgpt_ui_renders`** — an `AppTest` that injects a *fitted* synthetic analysis so the
fitted regression path is finally exercised in CI (the gap the demos couldn't reach).

---

## 6. Update J.3

### Calculation-correctness audit (verification pass)

**Goal.** Independently confirm the math before trusting the numbers — recompute every quantity from
first principles (or a different library path) and compare.

**Result — 21/21 mechanical checks exact:**

| Quantity | Reference | Agreement |
|---|---|---|
| LPM coefficients | normal equations `(XᵀX)⁻¹Xᵀy` | max Δ ≈ 1e-16 |
| HC3 SE | manual sandwich `(XᵀX)⁻¹ Xᵀ diag(uᵢ²/(1-hᵢ)²) X (XᵀX)⁻¹` | max Δ ≈ 4e-17 |
| Cluster SE | manual CR sandwich incl. `G/(G-1)·(N-1)/(N-K)` | max Δ ≈ 3e-17 |
| VIF | manual `1/(1-R²)` | matches |
| Benjamini–Hochberg | textbook step-up | max Δ ≈ 3e-17 |
| two-sided p | `scipy.norm.sf` | < 1e-12 |
| log1p / median-fill / one-hot | hand-computed | exact |
| logit AME | column-index check + AME ≈ LPM; separation caught | exact / no crash |

**One methodology finding (not an arithmetic bug).** The **unrestricted** wild cluster bootstrap, as
implemented, mathematically reproduces the **CR0** cluster variance (no finite-sample correction) —
so its SE comes out ~10% **smaller** than the analytic cluster SE it replaces, and it pairs with a
hard-coded `1.96` rather than `t(G-1)`. Net effect for few clusters: focal CIs run a touch
**too narrow**, the opposite of the "treat significance cautiously" intent. The honest fix is the
**restricted** wild cluster bootstrap-t (impose H₀, bootstrap the t-stat, invert). _Left as a
deliberate, documented follow-up_ — see Update K's clustering note.

---

## 7. Update K

### Careful-reporting upgrade · branch `econometrics-layer` (latest)

**Goal.** Make the layer **statistically safer, harder to misread, and more useful for business
reporting** — driven by a structured review. 17 work-items; every change integrated into the
existing engine → report → UI path (no standalone script).

### K.1 — Safer clustering
- **`choose_cluster(df)`** — prefer `domain` → `prompt_id` → `record_id` → a **repeated** page key
  (`canonical_url`/`page_id`), and **skip degenerate** candidates: a **unique id** (every row its own
  cluster) or a **single `run_id`** (one cluster). Returns `cluster_variable`, `cluster_count`,
  `cluster_warning`. Below `MIN_CLUSTERS` (~40) it emits *"Cluster-robust SE may be unstable with few
  clusters; consider wild cluster bootstrap or interpret cautiously."*
- **Why.** The old loop could silently cluster on `record_id` even when each record was unique, or on
  a single `run_id` — both useless. The new logic refuses those and says why.
- **Clustering note.** The wild-bootstrap *mechanism* from J.0 is **unchanged** here (a unit test
  depends on it). K only makes the **variable choice** safer and surfaces the warning; the
  anti-conservativeness flagged in J.3 remains a separate, larger follow-up.

### K.2 — Source-position language
- Described everywhere as the **observable source panel position** (observed source order / source
  panel placement) — **never** internal AI rank, retrieval rank, or Google rank (`CAVEAT_POSITION_PANEL`).
- Models are **always run with and without** position (A/B vs C/D) so content effects can be read both
  ways — position may be a **mediator / post-treatment** variable.

### K.3 — LPM wording
- Coefficients are reported as *"associated with a ±X **percentage-point** difference in citation
  probability, **controlling for the included variables**"* — a controlled association, **not** "this
  feature increases citation by X" (`CAVEAT_LPM_INTERPRET`, applied in the report + exec summary).

### K.4 — Similarity separation (prompt vs answer)
- **Prompt-based** similarity (`_PROMPT_SIM`: title/description/page/best-chunk ↔ prompt, query sims)
  is admissible in the main model. **Answer-derived / circular** similarity (`_ANSWER_SIM`: page/chunk
  ↔ answer, answer overlap, `answer_like_text_in_first_500_chars`) is **barred** from A/B/C/D and is a
  post-output diagnostic only.
- **`relevance_score`** (Model D) is now the z-scored mean of the **prompt** sims **only**, plus a
  `relevance_n_missing` count.
- **Why.** Answer-derived features leak the outcome (the answer is downstream of citation); keeping
  them out of the effect-style model removes circularity.

### K.5 — Missing data
- `design_matrix` now **median-fills every numeric feature and adds a `{feature}_missing` indicator**
  (a new **`missingness`** feature group), not just position; categoricals keep an explicit `unknown`
  level. New **`missingness_diagnostics`** by cited-status / source-type + `CAVEAT_MISSINGNESS`
  ("missingness may be informative…").

### K.6 — Reference categories
- **`reference_categories`** table (`variable`, `reference_category`, `all_categories`, `notes`) +
  `CAVEAT_REFERENCE_CATEGORY` — so a `page_type=article` coefficient is read **relative to the omitted
  level** ("+30 pp vs the reference page type").

### K.7 — VIF, two ways
- **`vif_focal_rows`** (content features only — easy to read) **and** **`vif_full_rows`** (full design
  matrix — inflated by sparse dummies) + the **condition number**. Labels: `<2` low, `2–5` moderate,
  `5–10` high, `≥10` severe. Warning: *"High VIF indicates overlapping predictors and wider error
  bars, not necessarily biased coefficients."*

### K.8 — Multiple testing, per family
- BH is now applied **within each model × feature family** (e.g. `content_structure`, `access`,
  `authority`, `relevance`), not all focal features mixed. `fit_citation_model` records
  `bh_families`; exported as **`multiple_testing_summary`** (`model_name`, `feature_family`,
  `num_tests`, `bh_applied`, `notes`). `p` = raw, `q_bh` = adjusted **within the stated family/model**.

### K.9 — Logit AME cross-check table
- **`logit_ame_check`** — side-by-side LPM Δprob vs logit AME per focal feature, with `sign_agrees`
  and a `logit_status` (`ok` / `failed_perfect_separation` / `skipped`). The **LPM stays the
  headline**; logit is robustness only; a penalized fit would be labeled, never read as a clean effect.

### K.10 — Separation diagnostics
- **`separation_diagnostics`** — per binary feature: `cited_rate_when_feature_1/0`, `n_feature_1/0`,
  `possible_separation`, notes. Detected **directly from the data** (not just "did the logit
  converge"), so a near-perfect predictor is flagged before it produces unstable coefficients.

### K.11 — OVB / confounding caveats
- Signed but **hedged**: *"Possible upward bias if unobserved writing quality is positively correlated
  with `has_faq`…"* (`CAVEAT_OVB_SIGNED_EXAMPLE`) + a named-confounder list (writing quality, domain
  authority, source-panel placement, scrape success, page type) in `CAVEAT_OVB_CONFOUNDERS`.

### K.12 — New data-quality diagnostics
- **`dedup_diagnostics`** — duplicate raw/normalized URLs, http/https + trailing-slash dupes, UTM /
  tracking params, same domain + path, repeated canonical/page ids.
- **`scrape_success_diagnostics`** — scrape rate by cited-status / source_type / page_type / domain,
  **with a warning if cited vs more-only scrape rates differ strongly** (informative-missingness /
  selection signal).
- **`overlap_diagnostics`** — positivity check: is a feature (e.g. `has_price_or_package`) almost
  only present in one `page_type`/`source_type`/`intent`? If so its effect can't be separated from
  that category.
- **`rare_feature_diagnostics`** — features with prevalence `<5%` or `>95%` (unstable, wide CIs) +
  `CAVEAT_RARE_FEATURES`.
- **`outcome_definition`** (`.txt`) — the cited / more-only definition, verbatim, so no reader treats
  more-only as "rejected" or as "not retrieved."

### K.13 — Anomaly diagnostics
- Retained the six J.1 checks (position dominance, similarity severe-VIF, negative access, negative
  authorship, positive age, large page-type), each with a careful, business-safe interpretation note.

### K.14 — Feature grouping
- Groups split for clarity: `structure` → **`content_structure`**, **`source_type`** pulled out of
  `authority`, and a **`missingness`** group added. Group summary now reports `num_p_lt_05`,
  `num_q_lt_10`, and `warnings`.

### K.15 — Forest plots
- Two named PNGs: **`forest_plot_focal.png`** (focal features **with** position, Model C) and
  **`forest_plot_no_position.png`** (Model B, or Model C with the position group excluded). Both show
  Δprob ± 95% CI in **percentage points**, a zero line, and color-code whether the CI crosses zero.

### K.16 — Report wording
- `report._sensitivity_section` rebuilt with: **Executive summary**, **Interpretation-safety**
  (`CAVEAT_MODEL_OBSERVATIONAL` — "…does not prove that adding a feature will cause citation
  probability to increase"), **Model-specification** (A/B/C/D defs + cluster var/count + SE type +
  reference categories + whether position is included), **Diagnostics** (VIF focal/full, missing,
  scrape, dedup, rare, overlap, multiple-testing, logit-AME, separation), and a **Business-safe
  recommendation** (e.g. *don't* read a negative contact coefficient as "remove contact info" —
  embed it inside answer-ready product/service pages).

### K.17 — Tests
- 6 new tests (78 total): every new diagnostic + exporter generated; separation handled without
  crashing; BH within model × family; observable-panel wording; answer-similarity excluded from the
  main model; safe clustering avoids degenerate ids.

**Verification.** `pytest -q` → **78 passed**; `AppTest` renders both modes incl. the sensitivity
section; **end-to-end on a real 1,270-source ChatGPT audit** (fits, clusters on 350 domains, renders
the full report + JSON + both forests with **no banned causal / AI-rejection wording**).

---

## 8. The model ladder

`model_comparison()` is **not one model** — it's the **same outcome (`cited`) fit five times**, each
adding one control layer, so you can watch whether a feature's effect is **stable**.

| Model | Content | Source/authority + page_type/intent | Source position | Similarity | Purpose |
|---|:--:|:--:|:--:|---|---|
| **A · content only** | ✅ | — | — | — | Raw content associations |
| **B · + source/authority** | ✅ | ✅ | — | — | Does the effect survive *what kind of source/page* it is? |
| **C · + source position** | ✅ | ✅ | ✅ `log1p` | — | Headline model; logit AME cross-check runs here |
| **D · reduced similarity** | ✅ | ✅ | ✅ | one `relevance_score` | Clean relevance signal (no collinear sims) |
| **FULL · diagnostic** | ✅ | ✅ | ✅ | **all raw** sims | Deliberately surfaces VIF — diagnostic, not for reading |

**Reading the comparison row by row** (`econometrics_model_comparison.csv`):

```
feature        A      B      C      D
has_faq      +0.12  +0.10  +0.09  +0.09   ← stable across A→D → trustworthy
has_contact  -0.06  -0.02  +0.01  +0.01   ← shrinks/flips → was confounded by page type
```

A coefficient **stable A→D** is trustworthy; a big swing signals **confounding** (B/C) or
**collinearity** (D vs FULL).

---

## 9. Output catalog

Generated by `model_comparison()` and exported from `report.py` (ChatGPT Report tab). All 16:

| File | Content |
|---|---|
| `econometrics_model_comparison.csv` | Per-model focal Δprob / se / CI / p / q_bh / VIF across A→D |
| `econometrics_reference_categories.csv` | Omitted dummy level per categorical |
| `econometrics_vif_focal.csv` | VIF on focal content features only |
| `econometrics_vif_full.csv` | VIF on the full design matrix |
| `econometrics_multiple_testing_summary.csv` | BH families per model (num_tests, bh_applied) |
| `econometrics_logit_ame_check.csv` | LPM Δprob vs logit AME, sign agreement, logit status |
| `econometrics_separation_diagnostics.csv` | Per-feature cited-rate split + separation flag |
| `econometrics_dedup_diagnostics.csv` | Duplicate / canonical / UTM / scheme URL checks |
| `econometrics_scrape_success_diagnostics.csv` | Scrape rate by cited-status / type / domain + selection warning |
| `econometrics_overlap_diagnostics.csv` | Positivity: feature near-exclusive to one category |
| `econometrics_rare_feature_diagnostics.csv` | Prevalence <5% / >95% features |
| `econometrics_anomaly_diagnostics.csv` | 6 auto-flagged anomalies + safe interpretations |
| `econometrics_feature_group_summary.csv` | Per-group top ±, num p<.05 / q<.10, warnings |
| `econometrics_outcome_definition.txt` | cited / more-only definition (verbatim) |
| `econometrics_forest_plot_focal.png` | Focal features **with** source position |
| `econometrics_forest_plot_no_position.png` | Focal features **without** source position |

Plus the structured `regression_comparison` block inside the JSON analysis bundle.

---

## 10. Wording rules (the safety contract)

Enforced by config caveats + a unit test that scans the rendered report:

- ✅ "**associated with** a ±X percentage-point difference … controlling for the included variables."
- ✅ `more-only` = **surfaced but not cited** (never "rejected" / "ignored").
- ✅ `source_position` = **observable source panel position** (never internal AI / retrieval / Google rank).
- ✅ "It **does not prove** that adding a feature will cause citation probability to increase."
- ❌ Never: "causes citation", "caused by", "AI rejected", "AI ignored", "increases citation by X".

The only places the strings "internal AI rank" / "will cause citation" appear are inside the
**required negations** ("**not** an internal AI rank…", "**does not prove** … will cause …").

---

## 11. Test coverage map

`tests/test_econometrics.py` (78 total). Highlights:

| Area | Tests |
|---|---|
| Recover known coefficient / CI coverage ≈95% | `test_recover_known_coefficient`, `test_ci_coverage_about_95pct` |
| Cluster SE > HC3 under within-cluster correlation | `test_cluster_se_exceeds_hc3` |
| BH ≥ raw p; BH within model **and** family | `test_bh_more_conservative_than_raw`, `test_bh_within_each_model_spec`, `test_bh_within_model_and_family` |
| VIF flags collinearity; functional form; aliasing | `test_vif_flags_collinearity`, `test_position_functional_form`, `test_collinear_column_dropped` |
| Parity vs direct statsmodels HC3 | `test_matches_direct_statsmodels` |
| Logit AME ≈ LPM; separation flagged, AME suppressed | `test_logit_ame_tracks_lpm`, `test_separation_flagged_lpm_fallback`, `test_logit_ame_check_handles_separation` |
| Wild bootstrap engages with few clusters | `test_wild_bootstrap_with_few_clusters` |
| All new diagnostics + exporters generated | `test_new_diagnostics_and_exporters_generated` |
| Safe clustering avoids degenerate ids | `test_choose_cluster_is_safe` |
| Answer similarity excluded from the main model | `test_answer_similarity_excluded_from_main_model` |
| Observable-panel wording; no banned causal language | `test_position_described_as_observable_panel`, `test_sensitivity_report_safe_wording` |
| Fitted ChatGPT UI renders (guards the J.2 bug class) | `test_fitted_chatgpt_ui_renders` |
| Pipelines still carry a graceful `regression_comparison` | `test_pipeline_carries_regression_comparison` |

---

_See also: `src/econometrics.py` (engine), `src/report.py` (exporters + report sections),
`ui/components.py` (`regression_block` / `sensitivity_block`), `src/config.py` (caveats &
thresholds), and `docs/DEVELOPMENT.md` §4 (Iterations J–K)._
