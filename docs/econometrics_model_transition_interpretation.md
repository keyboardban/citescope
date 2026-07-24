# Econometrics Model Transition Interpretation

## Read order

1. Start with G1 and G2.
2. Read coefficient direction and magnitude.
3. Read confidence interval and covariance sensitivity separately.
4. Check sample, prompt, URL, and domain changes.
5. Read the transition-specific warning.
6. Review G5A and every G8 treatment before interpretation.

## Deterministic labels

Labels use predeclared coefficient, uncertainty, sample, and support thresholds. They do not use p-values alone.

- `stable_direction`: the displayed estimates retain the same sign.
- `direction_change_below_threshold`: signs differ, but at least one estimate is below the predeclared magnitude required for a `sign_flip` label.
- `stable_magnitude`: absolute magnitude changes by no more than the configured tolerance.
- `substantial_attenuation`: magnitude falls by both absolute and relative thresholds.
- `substantial_amplification`: magnitude rises by both thresholds.
- `sign_flip`: both estimates exceed the minimum magnitude and signs differ.
- `wider_uncertainty` / `narrower_uncertainty`: CI width changes materially.
- `sample_sensitive`: row count changes materially.
- transition labels identify prompt, domain/template, taxonomy, extraction, text-scope, functional-form, or outlier sensitivity.

Near-zero sign changes are labeled `direction_change_below_threshold`, not `sign_flip` or `stable_direction`. Relative change is suppressed when the baseline is near zero.

## Point estimate versus inference

Covariance changes should leave a regression coefficient unchanged. A stable point estimate with a wider interval is `stable_point_estimate` plus `inference_sensitive`, not coefficient instability.

## Largest change

The largest-change panel reports the largest compatible coefficient change, uncertainty increase, sample loss, and first meaningful sign flip. “Largest” means diagnostically largest, not best or most significant.

## Remaining confounding

No transition observes all source authority, publisher reputation, official status, backlinks, freshness, hidden relevance/ranking signals, personalization, audit-time content, non-surfaced candidates, or within-domain template variation. The model path is evidence about sensitivity, not proof that one transition identifies the true mechanism.
