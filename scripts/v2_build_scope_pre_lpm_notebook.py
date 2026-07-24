#!/usr/bin/env python3
"""Build the executable SCOPE condo pre-LPM notebook from the existing EDA template."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/econometrics_eda/02_real_apify_scrape_feature_diagnostics_before_lpm.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Notebook template not found: {TEMPLATE}")
    template = nbf.read(TEMPLATE, as_version=4)
    cells = [
        markdown("""
# SCOPE Condo Pre-LPM Feature Diagnostics

This notebook is a descriptive, pre-model diagnostic notebook for the SCOPE condo citation study. It is adapted from the existing Apify feature-diagnostics notebook but uses only the SCOPE condo final-LPM-prep and post-scrape data.

**Purpose**
1. Verify the final LPM-prep dataset is usable.
2. Examine citation rate by feature category.
3. Identify descriptive associations, sparse categories, and unstable measurements before modeling.
4. Run taxonomy-confidence and scrape/content-availability sensitivities.
5. Separate main-LPM variables from diagnostic-only and forbidden variables.

This is descriptive pre-model analysis, not causal inference. The unit is one surfaced source appearance. `cited = 1` means explicitly cited; `cited = 0` means surfaced/more-only but not cited. A more-only source is not evidence of rejection by a hidden retrieval system. No final LPM is fit here.
"""),
        code("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    raise RuntimeError(f'Cannot locate the project root from {Path.cwd().resolve()}')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.metric_recheck import aggregate_urls
from src.econometrics_eda_v2.pre_lpm_diagnostics import (
    MAIN_FEATURES, NUMERIC_FEATURES, citation_rate_by_category,
    enrich_lpm_diagnostics, run_pre_lpm_diagnostics,
)

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
OUT = BASE / 'tables/pre_lpm_eda'
FIG = BASE / 'figures/pre_lpm_eda'
INPUT_PATH = LPM_PATH if LPM_PATH.exists() else EDA_PATH
if not INPUT_PATH.exists():
    raise FileNotFoundError(f'No SCOPE LPM-ready or EDA-ready file found: {LPM_PATH}; {EDA_PATH}')
print(f'Input: {INPUT_PATH}')
print('Mode:', 'final_lpm_prep' if INPUT_PATH == LPM_PATH else 'eda_ready_fallback')
"""),
        markdown("""
## 1. Load data and verify schema

The final-LPM-prep CSV is preferred. The post-scrape EDA CSV is read only to attach diagnostic titles, excerpts, detailed labels, and raw numeric fields when row order and identifiers match. Those fields are never added to the main LPM feature set.
"""),
        code("""
lpm = pd.read_csv(INPUT_PATH, low_memory=False)
eda = pd.read_csv(EDA_PATH, low_memory=False) if EDA_PATH.exists() else None
df = enrich_lpm_diagnostics(lpm, eda)

urls = aggregate_urls(df)
usable = ((df['content_quality_flag'].fillna('').astype(str).str.casefold() == 'ok') & (pd.to_numeric(df['word_count'], errors='coerce') >= 300))
confidence = df.get('taxonomy_confidence', df.get('re_page_type_confidence', '')).fillna('').astype(str).str.casefold()
metrics = pd.DataFrame([
    ('total rows', len(df)),
    ('unique URLs', df['normalized_url'].nunique()),
    ('unique prompts', df['prompt_id'].nunique()),
    ('cited rows', int(df['cited'].sum())),
    ('cited rate', float(df['cited'].mean())),
    ('more-only rows', int((df['cited'] == 0).sum())),
    ('scrape success rate, URL level', float(urls['scrape_success'].mean())),
    ('parse success rate, URL level', float(urls['parse_success'].mean())),
    ('usable content rate, URL level', float(((urls['content_quality_flag'].eq('ok')) & (urls['word_count_max'] >= 300)).mean())),
    ('unknown page type rate, URL level', float(urls['page_type_family_real_estate'].eq('unknown').mean())),
    ('unknown source type rate, row level', float(df['source_type_real_estate'].eq('unknown').mean())),
    ('unknown source type rate, URL level', float(urls['source_type_real_estate'].eq('unknown').mean())),
    ('high/medium taxonomy confidence rate, URL level', float(urls['re_page_type_confidence'].isin(['high', 'medium']).mean())),
], columns=['metric', 'value'])
display(metrics)
display(Markdown('Reference values: 1,139 rows; 846 URLs; 411 cited rows (36.1%); URL scrape 87.8%; URL parse 87.2%; URL usable content 58.9%; URL unknown page type 18.8%; post-prep URL unknown source type 29.3%; high/medium confidence 77.2%.'))

expected = ['cited','prompt_id','normalized_url','source_url','source_root_domain','page_type_family_real_estate','page_type_detail_real_estate','source_type_real_estate','re_page_type_confidence','content_quality_flag','scrape_success','parse_success','scraped_body_available','word_count','content_feature_available','taxonomy_confidence_high_or_medium']
schema_check = pd.DataFrame({'expected_column': expected})
schema_check['status'] = schema_check.expected_column.map(lambda c: 'present' if c in df.columns else 'missing')
schema_check['notes'] = schema_check.expected_column.map(lambda c: 'diagnostic field attached from post-scrape EDA when present' if c in {'source_url','page_type_detail_real_estate','word_count'} else '')
display(schema_check)
"""),
        markdown("""
## 2. Variable safety

Broad page family, conservative source type, source-type flags, and prompt fixed effects are candidates for the main model. Detailed taxonomy and raw content measurements remain diagnostic or sensitivity-only. Answer-derived, rank/position, provenance, and outcome-duplicate fields are forbidden main predictors.
"""),
        code("""
result = run_pre_lpm_diagnostics(INPUT_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
lpm_variable_use_table = pd.read_csv(OUT / 'lpm_variable_use_table.csv')
display(lpm_variable_use_table)
print('All tables and figures were regenerated by the descriptive helper. No final LPM was fit.')
"""),
        markdown("""
## 3. Citation rate by categorical feature

`citation_rate_by_category(df, feature, min_n=20)` reports category counts, cited rates, differences from the full-sample rate, Wilson 95% confidence intervals, and sparse-category warnings. Categories with `n < 20` or very few cited rows are descriptive only.
"""),
        code("""
def describe_category_table(table):
    stable = table[table['n_rows'] >= 20]
    if stable.empty:
        return 'No non-sparse category is available for interpretation.'
    high = stable.iloc[stable['cited_rate'].argmax()]
    low = stable.iloc[stable['cited_rate'].argmin()]
    sparse_n = int((table['n_rows'] < 20).sum())
    return (f"Highest non-sparse rate: `{high.category}` ({high.cited_rate:.1%}, n={int(high.n_rows)}). "
            f"Lowest non-sparse rate: `{low.category}` ({low.cited_rate:.1%}, n={int(low.n_rows)}). "
            f"Sparse categories: {sparse_n}. These are descriptive associations, not causal effects.")

for feature in MAIN_FEATURES:
    table = citation_rate_by_category(df, feature)
    display(Markdown(f'### {feature}'))
    display(table)
    display(Markdown(describe_category_table(table)))
"""),
        markdown("""
## 4. Citation-rate plots with Wilson confidence intervals

Bars are sorted by cited rate and labelled with row counts. Sparse categories are identified by labels and the companion tables, not visually dramatized.
"""),
        code("""
for filename in ['citation_rate_page_type_family.png','citation_rate_source_type.png','citation_rate_content_quality.png','citation_rate_taxonomy_confidence.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        markdown("""
## 5. Sparse-category diagnostics

The main LPM should use family-level page categories rather than the much more granular detail labels. The action recommendations identify categories that should be collapsed, held for diagnostics, or manually reviewed.
"""),
        code("""
sparse_category_diagnostics = pd.read_csv(OUT / 'sparse_category_diagnostics.csv')
display(sparse_category_diagnostics.sort_values(['feature_name','n_rows']))
"""),
        markdown("""
## 6. Numeric feature diagnostics

Numeric content measures are evaluated only where page content is available. Quantile-bin plots avoid assuming linearity; log transforms are suggested as sensitivities for skewed counts, not adopted automatically.
"""),
        code("""
numeric_feature_summary = pd.read_csv(OUT / 'numeric_feature_summary.csv')
numeric_feature_bins = pd.read_csv(OUT / 'numeric_feature_bin_diagnostics.csv')
display(numeric_feature_summary)
display(numeric_feature_bins)
for feature in numeric_feature_summary['feature']:
    path = FIG / f'citation_rate_by_{feature}_quantile.png'
    if path.exists():
        display(Image(filename=str(path)))
"""),
        markdown("""
## 7. Taxonomy-confidence sensitivity

This compares all rows with the high/medium-confidence subset. Similar patterns support robustness; changed or sparse categories require caution rather than a forced taxonomy decision.
"""),
        code("""
confidence_sensitivity = pd.read_csv(OUT / 'sensitivity_all_vs_high_medium_confidence.csv')
display(confidence_sensitivity)
display(Markdown('Stable means both samples have at least 20 rows and their cited rates differ by no more than 10 percentage points. It is a diagnostic heuristic, not a significance test.'))
"""),
        markdown("""
## 8. Scraped-content availability sensitivity

Content-feature analysis is conditional on successful scraping/extraction. It should not be generalized to all surfaced sources without this qualification.
"""),
        code("""
content_availability_sensitivity = pd.read_csv(OUT / 'sensitivity_scraped_content_availability.csv')
display(content_availability_sensitivity)
"""),
        markdown("""
## 9. Domain-level diagnostics

Domains are inspected for concentration, taxonomy gaps, and potential clustering choices. They are not a main causal claim, and high-cardinality domain fixed effects are not the default model specification.
"""),
        code("""
domain_level_citation_diagnostics = pd.read_csv(OUT / 'domain_level_citation_diagnostics.csv')
display(domain_level_citation_diagnostics.head(30))
display(Markdown('Top domains by cited rows:'))
display(domain_level_citation_diagnostics.sort_values('cited_rows', ascending=False).head(20))
"""),
        markdown("""
## 10. Pre-LPM readiness checks

The expected outcome is `ready_for_pre_LPM_EDA` with a more cautious final recommendation of `near_lpm_ready_after_taxonomy_QA` while residual unknown source types and scrape-selection issues remain.
"""),
        code("""
pre_lpm_readiness_checklist = pd.read_csv(OUT / 'pre_lpm_readiness_checklist.csv')
display(pre_lpm_readiness_checklist)
display(Markdown((OUT / 'pre_lpm_eda_summary.md').read_text()))
"""),
        markdown("""
## 11. Draft LPM plan (not fit here)

1. `cited ~ page_type_family_real_estate + prompt fixed effects`
2. Add `source_type_real_estate`, retaining `unknown`.
3. Repeat Model 2 in the high/medium-confidence subset.
4. Use the content-available subset for structural page-content features.
5. Treat source position/rank only as a diagnostic sensitivity, never a main claim.

Use robust standard errors, preferably clustered by prompt, with source-domain clustering as a robustness check when supported. Coefficients will be descriptive associations in citation probability, not causal estimates.
"""),
        code("""
display(Markdown((OUT / 'lpm_model_design_plan.md').read_text()))
print('Notebook complete. Outputs:', OUT)
"""),
    ]
    template.cells = cells
    template.metadata["scope_pre_lpm"] = {"template": str(TEMPLATE.relative_to(ROOT)), "purpose": "SCOPE condo descriptive pre-LPM diagnostics; no model fit"}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(template, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
