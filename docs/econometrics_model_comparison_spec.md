# Econometrics Model Comparison Specification

## Scope

The comparison layer describes changes in estimated associations among sources already surfaced in this audit:

`P(cited = 1 | source surfaced in this audit)`

It is not causal, web-wide, or evidence that changing page content will change citation outcomes. It reads validated Notebook 09–11 artifacts offline. Streamlit never fits a model.

## Canonical aliases

| Alias | Role | Available input |
|---|---|---|
| G0 | Descriptive | Unadjusted cited-rate contrasts |
| G1 | Headline | M1 / Notebook 11 F one-feature prompt-FE models |
| G2 | Headline | M2 / Notebook 11 W3 joint prompt-FE models |
| G2R | Sensitivity | Unavailable: no otherwise-identical relevance-only artifact |
| G3 | Robustness | M3 / D_W3 prompt and domain FE |
| G4A | Sensitivity | Rule-v2 metadata taxonomy |
| G4B | Sensitivity | Gemini content-informed taxonomy |
| G5A | Sensitivity | Strong-content sample |
| G5B | Diagnostic | Full-text-equivalent sample |
| G5C | Diagnostic | Excerpt-only sample |
| G6 | Missingness audit | M6 descriptive selection audit; no coefficient |
| G7 | Cross-check | Simplified logit AME with intent/area controls |
| G8 | Sensitivity | Every available predefined removal/winsorization treatment |
| G9 | Interaction | Supported intent-specific slopes; pairwise intent contrasts unavailable |

Missing models are never synthesized.

## Model pairs

The fixed pair list is G0→G1, G1→G2, G2→G2R, G2→G3, G2→G4A, G2→G4B, G4A→G4B, G2→G5A, G5B→G5C, G2→G7, and G2→every G8 treatment. A pair is emitted only when both estimates exist.

## Unit harmonization

- Binary features: present versus absent.
- Categorical features: category versus the declared reference.
- Page length: approximate doubling because the predictor is log base 2.
- Continuous scores: p25-to-p75 model-implied contrast. Original one-unit coefficients remain in the artifact.
- Logit: average marginal effect in percentage points.
- G0 continuous summaries: highest displayed quartile versus lowest. This is marked not directly comparable with a one-unit regression coefficient.

## Derived statistics

`estimate_change_pp = comparison_estimate_pp - baseline_estimate_pp`

`absolute_magnitude_change_pp = abs(comparison_estimate_pp) - abs(baseline_estimate_pp)`

Relative magnitude change is emitted only when the baseline magnitude exceeds the configured minimum and interpretation units match.

Every transition also records confidence-interval width, zero inclusion, row/prompt/URL/domain changes, cluster changes, controls, fixed effects, functional form, sample restriction, and covariance estimator.

## Covariance comparisons

HC3, prompt-clustered, URL-clustered, and two-way prompt-by-URL covariance estimates are compared where exported. Coefficient equality and uncertainty changes are separate classifications. Invalid or non-positive focal variance remains unavailable with its original warning.

Clustering changes uncertainty assumptions; it does not add controls or remove confounding.

## Interaction output

Notebook 09 exports intent-specific slopes from models with intent interactions for page length and table presence. It does not export the covariance needed for formal pairwise differences between intents. The artifact therefore labels these as subgroup-specific slopes and sets `formal_contrast_available = false`.

## Offline artifacts

The frontend bundle contains harmonized estimates, pairwise comparisons, transition labels, covariance comparisons, intent output, comparability decisions, a feature summary, frozen thresholds, and a dedicated manifest. All files are hashed and validated before display.
