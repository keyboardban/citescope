# Econometrics Frontend Data Contract

## Canonical Location

Frontend artifacts live under:

`<content_econometrics_ai_package>/tables/econometrics_frontend/`

`econometrics_frontend_manifest.json` is the entry point. It records contract version, generation time, source artifacts, file hashes, row counts, and schemas. CSV remains canonical for the original feature artifacts; versioned Parquet is canonical for the cross-model comparison layer. The frontend never loads crawler JSON.

## Artifacts

| Artifact | Grain | Purpose |
|---|---|---|
| `econometrics_overview_summary.json` | audit | KPI and scope metadata |
| `core_general_feature_scorecard.csv` | feature | overview and evidence classification |
| `feature_cited_rate_summary.csv` | feature x level/bin | Wilson descriptive rates |
| `feature_model_estimates.csv` | feature x term x model x covariance | normalized model path |
| `feature_probability_contrasts.csv` | feature x contrast | precomputed intuitive contrasts |
| `feature_subgroup_statistics.csv` | feature x subgroup x state | descriptive heterogeneity support |
| `feature_related_associations.csv` | ordered feature pair | mixed-type association diagnostics |
| `feature_multicollinearity_diagnostics.csv` | feature | overlap, VIF, and coefficient movement |
| `feature_confounding_diagnostics.csv` | feature x risk | observed sensitivity evidence |
| `feature_evidence_quality.csv` | feature x dimension | transparent evidence-quality components |
| `feature_sample_audit.csv` | feature x flow stage | missingness and selection flow |
| `feature_example_pages.csv` | feature x observation | compact website evidence |
| `feature_comparable_pairs.csv` | feature x pair | deterministic cited/uncited comparisons |
| `feature_model_estimates_harmonized.parquet` | feature x term x model x covariance | unit-harmonized comparison estimates |
| `feature_model_comparisons.parquet` | feature x term x predefined transition | coefficient, uncertainty, and sample changes |
| `feature_model_transition_labels.parquet` | transition | deterministic stability labels |
| `feature_covariance_comparisons.parquet` | model term x covariance pair | point-estimate versus inference stability |
| `feature_intent_interaction_contrasts.parquet` | feature x intent x covariance | supported intent slopes and contrast availability |
| `feature_model_comparability.parquet` | transition | explicit comparability decisions |
| `feature_model_comparison_summary.parquet` | feature | largest changes and deterministic narrative |
| `econometrics_model_comparison_manifest.json` | bundle | comparison hashes, aliases, gaps, and guardrails |
| `model_comparison_thresholds.yaml` | bundle | frozen deterministic thresholds |

## Validation

The loader rejects a missing manifest, unsupported contract version, absent required files, schema mismatch, hash mismatch, duplicate keys where prohibited, invalid binary outcomes, non-finite primary estimates, or causal/prohibited interpretation language. Optional panels may be empty, but their schema must remain valid.

Every displayed statistic is traceable through `source_artifact`, `dataset_version`, `feature_registry_version`, `model_version`, and the manifest SHA-256 digest.

Both manifests are validated before display. Streamlit does not fit a model or modify thresholds.

## Fixed-Effect Safety

Adjusted estimates are displayed only from precomputed model artifacts. Frontend filters never refit a model. Therefore a filter may narrow descriptive examples and subgroup summaries, but it cannot silently relabel a whole-sample fixed-effect estimate as a filtered estimate. Unsupported filtered FE comparisons are explicitly unavailable.
