# Econometrics Model Comparability Rules

## Directly comparable

Requires the same focal feature definition, interpretation unit, outcome, sample, controls, fixed effects, and functional form. Covariance-only changes are handled separately rather than represented as specification changes.

## Partially comparable

Used when feature units remain compatible but at least one of these changes:

- sample restriction;
- controls;
- fixed effects;
- taxonomy;
- extraction scope;
- outlier treatment.

The frontend must display every changed component. A partially comparable result can diagnose sensitivity but cannot attribute the full coefficient change to one mechanism.

## Not directly comparable

Used when:

- feature definitions or interpretation units differ;
- a continuous G0 top-versus-bottom contrast is compared with a one-unit regression coefficient;
- G7 logit AMEs are compared with G2 while G7 also replaces prompt fixed effects with intent/area controls;
- reference groups cannot be aligned;
- required metadata is missing.

## Model-specific warnings

- G0→G1: prompt composition may contribute; prompt FE do not remove every confounder.
- G1→G2: change may reflect observed confounding, predictor overlap, suppression, or multicollinearity. It is not proof of mediation.
- G2→G3: domain FE and sample restriction occur together. Domain attenuation cannot be isolated from sample selection.
- G2→G4B: Gemini taxonomy may improve page-function adjustment and may also over-control because it can encode scraped content.
- G2→G5A: extraction reliability and sample composition both change.
- G5B→G5C: text scope and selected samples differ.
- G2→G7: functional form and controls differ. G7 is a cross-check, not a replacement.
- G2→G8: every predefined treatment is retained; none is selected by result.

## Reference groups

Categorical estimates are joined only on the same category contrast and declared reference. The current references are 0–1 headings, 9+ links, and strong extraction. A mismatched reference is not silently transformed.

## Missing models

No comparison row is created when either estimate is absent. Model metadata still records the unavailable alias and reason.
