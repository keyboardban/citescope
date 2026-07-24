# Econometrics Pipeline Redesign v2

## Scope

The active redesign contains exactly five layers:

```mermaid
flowchart TD
    D0["D0"] --> FE1["FE1"]
    FE1 --> FE2["FE2"]
    FE2 --> FE3["FE3"]
    FE2 --> FE4["FE4"]
```

`D0` is descriptive, not a regression. `FE3` and `FE4` are separate branches from `FE2`. The unit is one surfaced source appearance for one prompt and the estimand is `P(cited = 1 | source surfaced in this audit)`. Results are observational conditional associations, not causal effects or web-wide probabilities.

## Governed Features

The four active focal predictors are `log2_word_count_plus1`, `has_verified_html_table`, `factual_numeric_density_score`, and `writing_structure_score`. `content_strength` is an extraction-quality control in FE2-FE4. It is not writing quality. `heading_count_group` and heading primitives are D0/QA only. No heading substitute enters a regression.

`external_evidence_structure_score` is blocked: no canonical implemented producer, formula, or current model column was found. The older `external_evidence_score` is not substituted.

Verified HTML-table presence is nullable: `1` means measured present, `0` means measured absent, and `NA` means HTML was not measured. It is a broad structural proxy, not proof of a semantic data table.

## Taxonomy

FE4 uses `page_type_family_gemini_v1_collapsed` and `source_type_general_gemini_v1_collapsed` from `gemini_3_1_flash_lite_taxonomy_v1`. Categories with fewer than 20 URL classifications are collapsed without using the outcome; `unknown` remains explicit. Gemini used Markdown/body content where available and metadata otherwise, so FE4 is a sensitivity analysis with over-control risk.

## Execution

The offline runner is:

```bash
.venv/bin/python scripts/v2_run_redesigned_content_econometrics.py \
  --output-root outputs/econometrics_redesign_v2_20260722
```

It joins versioned writing, HTML-document, and Gemini-taxonomy evidence; performs model-entry and leakage gates; writes a physically restricted model-ready table; fits models offline; and creates hash-validated lightweight frontend artifacts. Streamlit never reads raw Bright Data JSON for econometric display.

## Inference

Each regression is fit once. The same model is reported with HC3, prompt-clustered, normalized-URL-clustered, and two-way prompt/URL-clustered covariance where valid. A covariance choice is not a new model layer.

| Layer | Purpose | Predictors and controls | Fixed effects |
|---|---|---|---|
| D0 | Distribution, support, missingness, raw cited-rate and extraction QA | Approved focal features, Extraction Strength, headings for QA | None |
| FE1 | One-feature conditional association | One approved focal feature per run | Prompt |
| FE2 | Headline joint Core-General association | Four focal features plus Extraction Strength | Prompt |
| FE3 | Domain/template-confounding robustness | Same predictors and control as FE2 | Prompt and domain; domains need at least two unique URLs |
| FE4 | Taxonomy-control sensitivity | Same FE2 terms plus collapsed Gemini page/source controls | Prompt |
