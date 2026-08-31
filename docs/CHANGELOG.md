# Change Log

## 2026-08-03

- Promoted six manually reviewed Gemini semantic-presence indicators into D0 and FE1-FE4.
- Retained the four governed deterministic core predictors as a separately named core group.
- Defined `0` only for successful measured absence; failed, partial, unavailable, and unmatched semantic classifications remain `NA`.
- Kept Gemini confidence, counts, first-block IDs, and page-relative position ratios diagnostic-only.
- Changed FE1 estimation to feature-specific complete cases; FE2-FE4 use the explicitly audited joint complete-case sample.

## 2026-07-27

- Added `writing_structure_score_v3` with five active components.
- Removed `has_question_answer_structure` from the active composite and FE1-FE4
  model data because it was identical to `has_faq_pattern` in all governed rows.
- Marked `writing_structure_score_v2` and its model outputs as superseded.

## 2026-07-24 - Structured list detector and writing score v2

- Replaced the active short-text list indicators with `has_main_content_unordered_list` and `has_main_content_ordered_list`.
- Added HTML-first detection after main-content selection and page-chrome filtering.
- Required at least two visible, non-empty direct `<li>` items inside a valid `<ul>` or `<ol>`.
- Added generated Markdown as the only fallback when HTML structure is unavailable.
- Removed the old list indicators from active registries, model-ready data, frontend views, and FE1-FE4 formulas.
- Added `writing_structure_score_v2`, which is `NA` unless all six governed components are measured.
- Rebuilt document features, selected/model-ready rows, frontend QA evidence, distribution and variation diagnostics.
- Reran D0, FE1, FE2, FE3, and FE4 under the unchanged fixed-effect, clustering, and taxonomy-control policies.
- Marked the previous writing score and the 2026-07-22 FE1-FE4 result files as superseded.

Historical source producers and output files remain available only to reproduce the old estimates. They are not part of the active model flow.
