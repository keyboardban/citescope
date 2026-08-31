# Documentation index

25 documents accumulate over a long project. This is the map. Start with the
repository `README.md`, then come here for depth.

## Read these first

| Document | Why |
|---|---|
| [`CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md`](CHATGPT_CONTENT_ECONOMETRICS_PIPELINE.md) | the full notebook sequence and data contract — the single best overview |
| [`CHANGELOG.md`](CHANGELOG.md) | what changed and, more usefully, **what was superseded and why** |
| [`econometrics_core_model_specification_v2.md`](econometrics_core_model_specification_v2.md) | the governed D0–FE4 model definitions currently in force |
| [`GENERAL_PAGE_TAXONOMY_RULE_V2.md`](GENERAL_PAGE_TAXONOMY_RULE_V2.md) | how pages are classified deterministically |

## Model layers

Two model families live side by side and are deliberately **not** merged.

| Layer | Models | Specification |
|---|---|---|
| Governed content model | D0, FE1–FE4 | [`econometrics_core_model_specification_v2.md`](econometrics_core_model_specification_v2.md) |
| Separate position model | M0–M6 | see "Separate position model" in the root `README.md` |

The position model reads frozen HTML-position and Gemini block-classification
artifacts and writes only to `outputs/position_model_v1/`. It does not touch the
governed registry.

## By topic

**Pipeline design**
[`econometrics_pipeline_redesign_v2.md`](econometrics_pipeline_redesign_v2.md) ·
[`econometrics_eda_pipeline.md`](econometrics_eda_pipeline.md) ·
[`econometrics_eda_plan.md`](econometrics_eda_plan.md) ·
[`econometrics_eda_input_notes.md`](econometrics_eda_input_notes.md)

**Features and decisions**
[`econometrics_feature_engineering_decisions.md`](econometrics_feature_engineering_decisions.md)

**Interpretation guardrails** — read before quoting an estimate
[`econometrics_frontend_interpretation_rules.md`](econometrics_frontend_interpretation_rules.md) ·
[`econometrics_model_comparability_rules.md`](econometrics_model_comparability_rules.md) ·
[`econometrics_model_comparison_thresholds.md`](econometrics_model_comparison_thresholds.md) ·
[`econometrics_model_transition_interpretation.md`](econometrics_model_transition_interpretation.md)

**Frontend**
[`econometrics_frontend_product_spec.md`](econometrics_frontend_product_spec.md) ·
[`econometrics_frontend_user_guide.md`](econometrics_frontend_user_guide.md) ·
[`econometrics_frontend_data_contract.md`](econometrics_frontend_data_contract.md) ·
[`econometrics_frontend_model_comparison_guide.md`](econometrics_frontend_model_comparison_guide.md) ·
[`econometrics_model_comparison_spec.md`](econometrics_model_comparison_spec.md)

**Engineering**
[`DEVELOPMENT.md`](DEVELOPMENT.md) ·
[`ARCHITECTURE_BEFORE_AFTER.md`](ARCHITECTURE_BEFORE_AFTER.md) ·
[`brightdata_fallback_integration_plan.md`](brightdata_fallback_integration_plan.md)

## Superseded — kept for reproducibility, not for new work

| Document | Superseded by |
|---|---|
| [`econometrics_core_model_specification_v1.md`](econometrics_core_model_specification_v1.md) | `..._v2.md` |
| [`econometrics_pipeline_redesign_v1.md`](econometrics_pipeline_redesign_v1.md) | `..._v2.md` |

The same rule applies in code: `src/econometrics_eda_v2/writing_structure_v2.py`
is superseded by `writing_structure_v3.py` and is imported nowhere. It survives
so that estimates published under `writing_structure_score_v2` stay reproducible
— a score definition has to outlive the results quoted from it.

## The boundary that governs every number here

This is a **black-box observational audit**. "More-only" does not mean rejected,
and the Bright Data source panel is not ChatGPT's complete internal retrieval
set. Estimates are conditional associations among surfaced sources — not causal
effects, and not web-wide citation probabilities.
