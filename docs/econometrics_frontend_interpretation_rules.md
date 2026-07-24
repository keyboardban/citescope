# Econometrics Frontend Interpretation Rules

## Permanent Scope

Results describe associations among sources already surfaced in this audit. They are not causal effects or web-wide citation probabilities.

Use “associated with,” “observed difference,” and “model-adjusted association.” Never use “impact,” “uplift,” “AI preference,” or wording that says adding a feature will change citation probability.

## Evidence Order

1. Measurement and sample support
2. Unadjusted descriptive cited rates
3. G1 one-feature prompt-FE association
4. G2 joint prompt-FE association
5. G5 extraction-quality and G8 outlier checks
6. Domain, relevance, taxonomy, functional-form, and inference sensitivities
7. Examples and observationally similar pairs
8. Technical details

## Stability

Robustness is not statistical significance. Labels use direction, magnitude, interval width, support, sample change, and stability across predefined models. Domain attenuation may indicate domain/template confounding. It does not prove the domain is the only confounder.

`content_strength` measures extraction quality, not prose quality. `has_table` may be a domain/template-confounded legacy proxy. Taxonomy controls based on scraped content can over-control features derived from the same page body.

## Examples

Examples illustrate measured observations. Highlighted text or structure explains feature measurement, not citation. Comparable pages are observationally similar on displayed variables, and important unobserved differences may remain.

The fixed ending for comparison diagnostics is: “The observed variables do not fully determine citation status.”
