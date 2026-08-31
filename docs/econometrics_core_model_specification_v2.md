# Core-General LPM Specification v2

## Exact formulas

For each focal feature `x`, FE1 is `cited ~ x + C(prompt_id)`.

FE2 is:

```text
cited ~ log2_word_count_plus1
      + has_verified_html_table
      + factual_numeric_density_score
      + writing_structure_score_v3
      + has_direct_answer_gemini_v1
      + has_definition_gemini_v1
      + has_comparison_gemini_v1
      + has_steps_gemini_v1
      + has_numeric_evidence_gemini_v1
      + has_question_heading_gemini_v1
      + C(content_strength, Treatment(reference='strong'))
      + C(prompt_id)
```

FE3 adds `C(source_root_domain)` to FE2 and keeps only domains with at least two unique normalized URLs.

FE4 adds `C(page_type_family_gemini_v1_collapsed)` and `C(source_type_general_gemini_v1_collapsed)` to FE2. FE4 does not include domain fixed effects.

## Interpretation

`log2_word_count_plus1` is measured extracted-text length; a one-unit change is approximately a doubling away from zero. `has_verified_html_table` is broad verified table structure. `factual_numeric_density_score` measures numeric specificity, not truth. `writing_structure_score_v3` sums HTML-first main-content unordered-list and ordered-list indicators with FAQ, opening-summary, and opening-direct-answer indicators. It is `NA` unless all five components are measured, and it is not total writing quality. Q&A is excluded because the governed detector duplicated FAQ exactly. The six `*_gemini_v1` terms are manually reviewed binary semantic-presence measurements from main-content blocks. They are `NA` unless Gemini classification succeeded; confidence, counts, and positions are diagnostic-only. `content_strength` controls extraction strength.

The regressions compare sources already surfaced in this audit. Coefficients must not be presented as causal effects or instructions that editing a page will change citation probability.
