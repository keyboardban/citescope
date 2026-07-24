# Core Econometric Model Specification v1

## 1. Scope freeze

The complete redesigned sequence is:

```text
D0 -> FE1 -> FE2
             |-> FE3
             |-> FE4
```

No other analysis layer is part of this specification. `FE3` and `FE4` are separate branches from `FE2`.

The outcome for source appearance `i` in prompt `p` is `cited_ip` in `{0, 1}`. The estimand is citation probability conditional on the source being surfaced in this audit.

## 2. Notation

- `alpha_p`: prompt fixed effect.
- `delta_d`: source-root-domain fixed effect.
- `T_page`: approved collapsed page-type category.
- `T_source`: approved collapsed source-type category.
- `Q_extract`: extraction-quality control, provisionally `C(content_strength)`.
- `X_core`: small approved Core-General feature vector.

The provisional `X_core` representatives are:

1. `log2_word_count_plus1`: page length.
2. `has_verified_html_table`: validated semantic HTML-table presence.
3. `heading_count_group`: nonlinear heading structure.
4. `external_evidence_structure_score`: outbound/evidence-link structure after navigation-link separation.
5. `factual_numeric_density_score`: validated general factual and numeric specificity.

Every representative remains blocked until its feature-registry QA gate and human approval pass. The final manifest must select one representative per construct.

## 3. D0

`D0` is descriptive and is not a regression.

For each approved or diagnostic feature, report:

- available and missing rows;
- cited and more-only support;
- unique prompts, URLs, and domains;
- distributions, tails, sparse levels, and unknown shares;
- raw cited rate and Wilson 95% interval by predeclared level or bin;
- cited-versus-more-only descriptive differences;
- within-prompt and within-domain variation.

`D0` answers whether the surfaced-source sample contains enough reliable variation for adjusted analysis. Bins and thresholds must be defined from predictor distributions or prior rules, never from citation outcomes.

## 4. FE1

`FE1` is one model family run separately for each approved focal feature `x_j`:

```text
cited ~ focal_feature + C(prompt_id)
```

Categorical features use a declared reference group:

```text
cited ~ C(focal_feature, Treatment(reference='declared_reference')) + C(prompt_id)
```

- Predictor: one approved focal feature.
- Fixed effects: prompt.
- Other content controls: none.
- Question: within the same prompt, is the focal feature associated with citation?

`FE1` does not adjust for other page characteristics. Its coefficient may reflect correlated domain, template, page function, extraction, or omitted page-level attributes.

## 5. FE2

The provisional joint formula is:

```text
cited ~ log2_word_count_plus1
      + has_verified_html_table
      + C(heading_count_group, Treatment(reference='0-1'))
      + external_evidence_structure_score
      + factual_numeric_density_score
      + C(content_strength, Treatment(reference='strong'))
      + C(prompt_id)
```

- Predictors: the approved Core-General representatives.
- Measurement control: extraction quality.
- Fixed effects: prompt.
- Question: within the same prompt, what association remains for each feature after joint adjustment for the other approved Core-General features and extraction quality?

If verified table extraction has not passed QA, the table construct must be omitted. A legacy proxy must not be silently renamed as verified table presence.

Joint adjustment does not establish causality. Overlapping features may create instability, suppression, or over-control, so the joint set must remain deliberately small.

## 6. FE3

`FE3` branches directly from `FE2`:

```text
cited ~ FE2_core_terms
      + C(content_strength, Treatment(reference='strong'))
      + C(prompt_id)
      + C(source_root_domain)
```

- Predictors and controls: identical to `FE2`.
- Fixed effects: prompt and source root domain.
- Sample: domains meeting the frozen support rule, provisionally at least two distinct normalized URLs plus usable within-domain feature variation.
- Question: do `FE2` associations remain after stable domain-level differences are absorbed?

Required reporting includes retained rows, prompts, URLs, and domains; the support rule; sample loss from `FE2`; coefficient changes; interval-width changes; and within-domain variation.

`FE3` is not a replacement for `FE2`. Differences can reflect both domain adjustment and changed sample composition.

## 7. FE4

`FE4` also branches directly from `FE2`:

```text
cited ~ FE2_core_terms
      + C(content_strength, Treatment(reference='strong'))
      + C(page_type_family_general_collapsed, Treatment(reference='unknown'))
      + C(source_type_general_collapsed, Treatment(reference='unknown'))
      + C(prompt_id)
```

- Predictors: identical to `FE2`.
- Measurement control: extraction quality.
- Categorical controls: page type and source type.
- Fixed effects: prompt only.
- Question: do `FE2` associations remain after adjustment for observed page function and publisher/source role?

`FE4` does not include domain fixed effects. Page type and source type are controls, not primary fixed effects. Because content-informed taxonomy may encode part of the focal content construct, `FE4` is a sensitivity branch and must report classifier provenance and potential over-control.

## 8. Inference

The primary standard errors are clustered by `prompt_id`. URL-clustered standard errors may be shown as an inference check when valid.

Changing the clustered standard-error estimator does not change the model formula, predictors, controls, fixed effects, or analysis-layer name. It must be stored as inference metadata rather than displayed as another model.

## 9. Sample and reporting rules

All five layers must use versioned inputs and report:

- formula or descriptive specification;
- exact feature versions;
- rows, prompts, URLs, and domains;
- cited support and cited rate;
- missingness and dropped-row audit;
- fixed effects and categorical controls;
- reference groups;
- covariance estimator for regressions;
- noncausal interpretation boundary.

Content features are interpreted only where their required evidence is measurable. `content_strength` and feature-availability fields describe extraction measurement, not page writing quality. Unknown taxonomy remains a valid category.

## 10. Approval gate

Estimation must not begin until:

1. the final `FE2` representatives and reference groups are approved;
2. the verified-table definition passes extraction QA or is removed;
3. the `FE3` domain-support rule is frozen;
4. the `FE4` taxonomy classifier, evidence scope, and collapse policy are approved;
5. answer-derived fields are physically absent from model-ready inputs;
6. formula and registry validators enforce the five-layer scope.

| Layer | Analysis | Predictors and controls | Fixed effects | Purpose |
|---|---|---|---|---|
| D0 | Descriptive analysis; no regression | Approved features summarized without adjustment | None | Establish support, missingness, and variation |
| FE1 | One focal feature per run | One approved focal Core-General feature | Prompt | Estimate one-feature within-prompt association |
| FE2 | Small joint Core-General LPM | Approved Core-General features plus extraction-quality control | Prompt | Estimate joint within-prompt associations |
| FE3 | FE2 with domain adjustment | Same predictors and controls as FE2 | Prompt and domain | Assess domain/template confounding |
| FE4 | FE2 with taxonomy adjustment | Same predictors as FE2 plus page-type and source-type categorical controls | Prompt | Assess page/source taxonomy sensitivity |
