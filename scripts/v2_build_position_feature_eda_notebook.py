#!/usr/bin/env python3
"""Build the executable final position-feature EDA notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/04_scope_position_feature_eda_final.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3"}
    nb.cells = [
        markdown("""
# SCOPE Position Feature EDA Final

This notebook measures where content structures appear in cleaned main-page content and evaluates whether those measurements are ready for later econometric modeling.

**Boundary:** descriptive associations among sources already surfaced in the audit. This notebook does not fit a new LPM or logistic regression, does not use answer text or ranking variables, and makes no causal claim.
"""),
        code("""
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
from IPython.display import display, HTML

ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.position_feature_eda import run_position_feature_eda

OUT = ROOT / 'outputs/position_feature_eda_final_20260731'
result = run_position_feature_eda(output_dir=OUT)
result
"""),
        markdown("""
## 1. Sample and extraction status

The row-level estimand remains citation conditional on a source being surfaced. URL-level coverage is reported separately so repeated source appearances are not described as additional webpages.
"""),
        code("""
rows = pd.read_parquet(OUT / 'data/scope_condo_eda_ready_with_position_features.parquet')
coverage = pd.read_csv(OUT / 'tables/position_feature_coverage.csv')
missingness = pd.read_csv(OUT / 'tables/position_feature_missingness.csv')
sample_summary = pd.DataFrame([{
    'surfaced_rows': len(rows),
    'unique_urls': rows.normalized_url.nunique(),
    'unique_domains': rows.source_root_domain.nunique(),
    'unique_prompts': rows.prompt_id.nunique(),
    'citation_rate': rows.cited.mean(),
}])
display(sample_summary)
display(missingness)
"""),
        markdown("""
## 2. Feature audit and coverage

Baseline features are not automatically repeated. The audit distinguishes previously available variables from genuinely new position, intensity, density, and interaction measurements.
"""),
        code("""
audit = pd.read_csv(OUT / 'tables/position_feature_audit.csv')
display(audit)
fig = px.bar(
    coverage.sort_values('percentage_of_eligible_pages'),
    x='percentage_of_eligible_pages', y='feature', orientation='h',
    text='pages_with_feature',
    title=f'Feature coverage among eligible pages (n={int(coverage.applicable_page_count.max()):,})',
)
fig.update_xaxes(tickformat='.0%')
fig.show()
display(coverage)
"""),
        markdown("""
## 3. Position distributions and citation rates

`No feature` is distinct from Q1. Position zero means the feature begins at the start of main content. Continuous comparisons are conditional on the feature being present.
"""),
        code("""
distribution = pd.read_csv(OUT / 'tables/position_feature_distribution_summary.csv')
citation = pd.read_csv(OUT / 'tables/citation_rate_by_feature_position.csv')
display(distribution)

for feature in ['table', 'faq', 'direct_answer', 'definition', 'comparison']:
    chart = citation[
        citation.feature.eq(feature)
        & citation.grouping.eq('quartile')
        & citation.sample_scope.eq('full_sample')
    ].sort_values('category_order')
    fig = px.bar(
        chart, x='category', y='citation_rate', text='n_observations',
        error_y=chart.ci_high - chart.citation_rate,
        error_y_minus=chart.citation_rate - chart.ci_low,
        title=f'Citation rate by {feature.replace("_", " ")} position (n={int(chart.n_observations.sum()):,})',
    )
    fig.update_yaxes(tickformat='.0%')
    fig.show()
"""),
        markdown("""
## 4. Cited versus not-cited positions

Means and medians below use only pages containing the relevant feature. Skewed position distributions make medians especially important.
"""),
        code("""
outcome = pd.read_csv(OUT / 'tables/position_feature_outcome_comparison.csv')
display(outcome)
"""),
        markdown("""
## 5. Domain fixed-effect readiness

Readiness requires position variation and citation-outcome variation within the same domains. The thresholds are documented in the table and are planning rules, not statistical significance tests.
"""),
        code("""
within = pd.read_csv(OUT / 'tables/position_feature_within_domain_diagnostics.csv')
fig = px.scatter(
    within, x='within_domain_standard_deviation',
    y='domains_with_both_position_and_outcome_variation',
    size='informative_observations', color='fixed_effect_readiness',
    hover_name='feature', title=f'Domain fixed-effect readiness (n={len(rows):,} surfaced rows)',
)
fig.show()
display(within)
"""),
        markdown("""
## 6. Sparse cells, taxonomy confounding, and page length

Four quartiles are retained only when observed cells support them. Taxonomy cross-tabs use normalized percentages. Relative and absolute positions are compared with page length because the same token index can imply a different relative location on short and long pages.
"""),
        code("""
sparse = pd.read_csv(OUT / 'tables/position_feature_sparse_cell_diagnostics.csv')
taxonomy = pd.read_csv(OUT / 'tables/position_feature_taxonomy_crosstab.csv')
page_length = pd.read_csv(OUT / 'tables/position_feature_page_length_relationship.csv')
display(sparse)
display(taxonomy.head(50))
display(page_length)
"""),
        markdown("""
## 7. Correlation and redundancy

Continuous measures use Spearman correlations. Binary presence indicators use phi correlations. Arbitrary category codes are never treated as continuous values.
"""),
        code("""
associations = pd.read_csv(OUT / 'tables/position_feature_associations.csv')
high_pairs = pd.read_csv(OUT / 'tables/position_feature_high_correlation_pairs.csv')
categorical_associations = pd.read_csv(OUT / 'tables/position_feature_categorical_association.csv')
display(high_pairs)
for association_type in associations.association_type.unique():
    subset = associations[associations.association_type.eq(association_type)]
    pivot = subset.pivot(index='feature_a', columns='feature_b', values='association')
    px.imshow(
        pivot, zmin=-1, zmax=1, color_continuous_scale='RdBu_r', aspect='auto',
        title=f'{association_type.replace("_", " ").title()} association matrix',
    ).show()
display(categorical_associations)
"""),
        markdown("""
## 8. Manual stored-evidence QA and validation checks

The review sample contains 10 detected and 10 measured-negative pages for each major feature. It is based on stored cleaned HTML/Markdown; live-page drift remains possible.
"""),
        code("""
manual = pd.read_csv(OUT / 'tables/position_feature_manual_validation.csv')
checks = pd.read_csv(OUT / 'tables/position_feature_validation_checks.csv')
display(manual.groupby(['feature', 'review_stratum', 'manual_validation_result']).size().reset_index(name='n'))
display(manual)
display(checks)
assert checks.passed.astype(bool).all()
"""),
        markdown("""
## 9. Model-readiness handoff

This is a recommendation table only. No new regression is estimated. Presence and position should remain separate in any later specification.
"""),
        code("""
readiness = pd.read_csv(OUT / 'tables/position_feature_model_readiness.csv')
display(readiness)
"""),
        markdown("""
## 10. Evidence-based conclusion

The conclusion is generated directly from the validated diagnostics and is also exported as plain text for downstream review.
"""),
        code("""
findings = (OUT / 'POSITION_FEATURE_EDA_FINDINGS.txt').read_text(encoding='utf-8')
print(findings)
"""),
    ]
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
