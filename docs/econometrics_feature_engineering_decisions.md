# Econometrics feature-engineering decisions

_Decision registry produced from the EDA notebook. **Status: DRAFT on DEMO data.** The numbers
below come from the synthetic 35-row demo table (`load_econometric_rows` fallback) and are
**machinery/descriptive checks only — NOT findings**. Re-run on a real Bright Data export
(`CITESCOPE_ROWS_CSV=...`) and replace the DEMO tables before treating any row as a decision._

> Wording: every relationship is an **observational association among surfaced sources** (cited vs
> surfaced-but-not-cited), never a causal effect. Content flags are heuristic proxies.
> `source_position` is observed source-panel placement, not a rank.

## How to refresh this file
1. Run the notebook on real data.
2. Read `outputs/econometrics_eda/feature_engineering_recommendations.csv` (per-feature
   `recommended_action` + `recommended_lpm_status` + rationale) and `.../eda_warnings.md`.
3. A human confirms/edits each row → move it from **Proposed** to **Decided** below.

---

## A. Decision summary from the last run (DEMO — 35 rows, 13 cited, 22 more-only)

**Recommended actions (counts):**

| recommended_action | n |
|---|---|
| use_in_sensitivity_only | 28 |
| use_as_diagnostic_only | 28 |
| exclude_leakage | 7 |
| combine_into_composite_score | 3 |
| keep_as_control | 3 |
| transform_bins | 2 |

**Recommended LPM status (counts):** include_main 5 · include_content_model_only 3 ·
include_fixed_effects_model 5 · include_sensitivity 26 · diagnostic_only 21 · exclude 11.

> On demo data almost everything lands in *sensitivity/diagnostic* because N is tiny and the
> minority group of most binary flags is < 20. This is the machinery behaving correctly, not a
> statement about the real data.

## B. Excluded as leakage / outcome-derived (firm — independent of N)
`cited_label, is_more_only, source_group, source_origin, page_answer_similarity,
max_chunk_answer_similarity, page_output_sim, max_chunk_output_sim, brand_appeared_in_answer,
answer_overlap`, non-LOO `domain_citation_rate`, and **`brand_official_candidate`** (reads
`answer_text`). → `recommended_lpm_status = exclude`. **Do not** use in any predictive model.

## C. Placement — diagnostic-only (firm)
`source_position, log1p_source_position, observed_rank`. Observed source-panel placement,
mediator-sensitive, co-determined with the outcome via `source_origin`. → sensitivity/diagnostic
only; never a main predictor. The model layer should compare specifications **with and without**
placement, and prefer **position-band stratification**.

## D. Numeric transform candidates (DEMO shapes — re-verify on real data)

| feature | transform_candidate (demo) | missing_rate | note |
|---|---|---|---|
| title_prompt_similarity | spline_or_bins | 0.00 | prompt-side; always available → main-model candidate |
| relevance_score_prompt_only | spline_or_bins | 0.00 | engineered prompt-only composite; preferred relevance proxy |
| description_prompt_similarity | diagnostic_only (demo) | 0.00 | sparse snippets |
| word_count / heading_count | diagnostic_only (demo) | 0.63 | scrape-selected; high missing on demo |
| page/max_chunk_prompt_similarity | diagnostic_only (demo) | 0.63 | scrape-selected |
| freshness_days | diagnostic_only (demo) | 0.60 | missingness likely non-random |
| domain_seen_count(_loo) | diagnostic_only (demo) | 0.00 | visibility **proxy**, not authority; prefer domain FE |

**Rule (independent of N):** high-missingness numerics (scrape/brand-match selected) → content
model / sensitivity only, never impute; prompt-side similarity → prefer over answer-side; domain
aggregates → leave-one-out + label proxy, prefer **domain fixed effects** in the model layer.

## E. Composite scores (engineered; reduce content-flag multicollinearity)
`structure_score, answer_ready_score, access_score, trust_signal_score, commercial_info_score` —
sums of available member flags. Recommended `combine_into_composite_score` /
`include_content_model_only`. Real data showed many `has_*` flags highly correlated (|r|≥0.8), so
composites are preferred over raw flags in a content model.

## F. Controls / fixed effects
Categorical FE / stratification: `intent, page_type, source_type, topic, language, country`.
Numeric controls: `word_count, heading_count, freshness_days` (scope to scraped rows). Binary
control: `institutional_official`.

## G. Intent × page_type eligibility (DEMO)
15 cells, **0 regression-eligible** on demo (all below thresholds). → descriptive-only for now.
Real data must re-check; only eligible cells get their own regression.

## H. Prompt-relevance features
Prefer **prompt-based** relevance (`relevance_score_prompt_only` = max of available prompt-side
similarities). **Exclude answer-based** similarity from main models. Future improvement: an
LLM-calibrated prompt-only relevance proxy (advanced, out of scope here).

## I. Domain authority proxies
Do **not** label as true authority. Use `domain_seen_count(_loo)` only as a visibility **proxy**;
`domain_citation_rate` only in **leave-one-out** form and only in sensitivity. Prefer **domain
fixed effects** in the final econometric layer.

---

## TODO — before the model layer is written (human)
- [ ] Re-run notebook on a **real** Bright Data export; replace sections A/D/G tables.
- [ ] Confirm the **outcome variant** (citations-only vs as-built incl. `links_attached`; whether
      `references` counts). *(EDA does not decide this.)*
- [ ] Harmonise the free-text **intent taxonomy** into canonical groups before using as FE.
- [ ] Decide **content-feature scope**: keep scoped to scraped+brand-matched, or generalise content
      extraction to all scraped sources (pipeline change).
- [ ] Confirm the **composite-score** definitions (members) with the domain owner.
- [ ] Pick **cluster/inference** rule (two-way `domain × prompt_id`; wild bootstrap if few clusters).
- [ ] Confirm placement stays **diagnostic-only** and define the position-band stratification.
- [ ] Decide **multiple-testing** families + 3–5 pre-registered primary hypotheses.

_None of the final models (pooled/content/fixed-effects/stratified LPM) are implemented in this
EDA task — see `docs/econometrics_eda_plan.md`._
