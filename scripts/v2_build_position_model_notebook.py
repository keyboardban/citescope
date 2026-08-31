#!/usr/bin/env python3
"""Build the reproducible position-model notebook from source cells."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks/05_scope_position_lpm_model_final.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# SCOPE Position LPM Model - Final

This notebook runs the **separate** position-focused M0-M6 analysis. It does not
modify or mix with the governed D0-FE4 content model. The estimand is
`P(cited = 1 | source surfaced in this audit, measurable content)`.

All estimates are observational adjusted associations, not causal effects."""
        ),
        nbf.v4.new_markdown_cell(
            """## Numeric-evidence definition

The primary continuous numeric feature is:

`numeric_evidence_total_density = numeric_evidence_total_count / total_main_content_tokens * 1,000`

It is standardized over the row-level analysis sample using the sample standard
deviation (`ddof=1`) to create `z_numeric_evidence_total_density`, which enters
M4 and M5. The position extension is
`numeric_evidence_early_share = numeric_evidence_early_count / numeric_evidence_total_count`.
Early share is `NaN`, not zero, when total numeric evidence is zero, and it does
not enter primary M5."""
        ),
        nbf.v4.new_markdown_cell(
            """## Six-class taxonomy controls

Detailed Gemini labels are preserved for QA. M0-M5 use these page-type controls:

- `blog_guide_or_editorial`
- `directory_or_listing`
- `commercial_product_or_service`
- `comparison_or_review`
- `landing_contact_or_support`
- `other_page_function`

The source-type controls are:

- `official_company_or_brand`
- `marketplace_or_directory_platform`
- `blog_or_news_publisher`
- `review_or_community_platform`
- `government_or_public_institution`
- `other_or_unknown`

Source type is made domain-stable using the modal collapsed class across unique
URLs. Exact ties become `other_or_unknown`; low-agreement domains remain flagged
in `position_model_source_type_domain_audit.csv`."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, Markdown

REPO = Path.cwd().resolve()
if REPO.name == 'notebooks':
    REPO = REPO.parent
if not (REPO / 'src').exists():
    candidates = [Path.cwd(), *Path.cwd().parents]
    REPO = next(path for path in candidates if (path / 'src').exists())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUTPUT = REPO / 'outputs/position_model_v1'

from src.econometrics_eda_v2.position_model import run_position_model
manifest = run_position_model(REPO, OUTPUT)
display(pd.DataFrame([manifest]).T.rename(columns={0: 'value'}))"""
        ),
        nbf.v4.new_markdown_cell("## 1. Sample and extraction flow"),
        nbf.v4.new_code_cell(
            """dataset = pd.read_parquet(OUTPUT / 'position_model_dataset.parquet')
flow = pd.read_csv(OUTPUT / 'position_model_sample_flow.csv')
audit = pd.read_csv(OUTPUT / 'position_model_feature_audit.csv')
display(flow)
display(audit)"""
        ),
        nbf.v4.new_markdown_cell("## 2. Feature coverage and imbalance"),
        nbf.v4.new_code_cell(
            """coverage = pd.read_csv(OUTPUT / 'position_model_feature_coverage.csv')
categorical = coverage[coverage['n_rows'].notna()].copy()
display(categorical)
px.bar(categorical, x='category', y='n_rows', color='feature', barmode='group',
       title='Position-feature category counts').show()"""
        ),
        nbf.v4.new_code_cell(
            """fig = go.Figure()
for feature, group in categorical.groupby('feature'):
    fig.add_trace(go.Scatter(
        x=group['category'], y=group['citation_rate'], mode='markers+lines', name=feature,
        error_y=dict(type='data', symmetric=False,
                     array=group['ci_upper'] - group['citation_rate'],
                     arrayminus=group['citation_rate'] - group['ci_lower']),
        text=group['n_rows'].map(lambda n: f'n={n:,}'),
    ))
fig.update_layout(title='Citation rate by placement with Wilson 95% CI', yaxis_tickformat='.0%')
fig.show()"""
        ),
        nbf.v4.new_markdown_cell("## 3. Domain, prompt, and within-domain support"),
        nbf.v4.new_code_cell(
            """domain = pd.read_csv(OUTPUT / 'position_model_domain_concentration.csv')
prompt = pd.read_csv(OUTPUT / 'position_model_prompt_concentration.csv')
within = pd.read_csv(OUTPUT / 'position_model_within_domain_variation.csv')
clusters = pd.read_csv(OUTPUT / 'position_model_cluster_support.csv')
display(domain)
display(prompt)
display(within)
display(clusters)"""
        ),
        nbf.v4.new_markdown_cell("## 4. M0-M6 model results"),
        nbf.v4.new_code_cell(
            """results = pd.read_csv(OUTPUT / 'position_model_results_long.csv')
primary = results[results['is_primary_inference'].astype(bool)].copy()
focal = primary[primary['term'].str.contains('placement|numeric_evidence', case=False, regex=True)]
display(focal[['model_id','term','estimate_pp','ci_lower_pp','ci_upper_pp','p_value','bh_q_value',
               'n_obs','n_cited','n_domains','n_prompts','se_method']])

plot = focal[focal['model_id'].eq('M5')].copy()
fig = go.Figure(go.Scatter(
    x=plot['estimate_pp'], y=plot['term'], mode='markers',
    error_x=dict(type='data', symmetric=False,
                 array=plot['ci_upper_pp'] - plot['estimate_pp'],
                 arrayminus=plot['estimate_pp'] - plot['ci_lower_pp']),
))
fig.add_vline(x=0, line_dash='dash')
fig.update_layout(title='M5 adjusted associations', xaxis_title='Percentage points')
fig.show()"""
        ),
        nbf.v4.new_markdown_cell("## 5. Multicollinearity and confidence-interval diagnostics"),
        nbf.v4.new_code_cell(
            """multi = pd.read_csv(OUTPUT / 'position_model_multicollinearity.csv')
ci = pd.read_csv(OUTPUT / 'position_model_ci_diagnostics.csv')
display(multi[multi['row_type'].eq('vif')].sort_values('vif', ascending=False))
display(ci[['model_id','term','ci_width_pp','category_sample_size','category_cited_count',
            'maximum_domain_share','maximum_prompt_share','vif',
            'leave_one_domain_out_min_pp','leave_one_domain_out_max_pp',
            'grounded_ci_explanation']])"""
        ),
        nbf.v4.new_markdown_cell("## 6. Stability and robustness"),
        nbf.v4.new_code_cell(
            """influence = pd.read_csv(OUTPUT / 'position_model_influence_diagnostics.csv')
robustness = pd.read_csv(OUTPUT / 'position_model_robustness_results.csv')
predicted = pd.read_csv(OUTPUT / 'position_model_predicted_probability_diagnostics.csv')
display(robustness)
display(predicted)

m5_term = focal.loc[focal['model_id'].eq('M5'), 'term'].iloc[0]
stability = influence[(influence['term'].eq(m5_term)) &
                      (influence['influence_dimension'].eq('source_root_domain'))]
px.scatter(stability, x='removed_group', y='estimate_pp',
           hover_data=['removed_rows','change_pp','sign_changed'],
           title=f'Leave-one-domain-out: {m5_term}').show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. Interpretation boundary

- Coefficients are percentage-point adjusted associations among surfaced sources.
- A coefficient whose interval includes zero is retained as an inconclusive/null result.
- Placement categories can reflect domain templates, page function, prompt mix, or extraction behavior.
- Stability across M5, alternative SEs, logit AMEs, and leave-one-domain-out checks improves reportability but does not establish causality.
- Moving a feature earlier on a webpage is **not** proven to change citation probability.
- External-source placement remains descriptive because formal placement-specific manual validation is unavailable."""
        ),
        nbf.v4.new_code_cell(
            """display(Markdown('```text\\n' + (OUTPUT / 'POSITION_MODEL_FINDINGS.txt').read_text() + '\\n```'))"""
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
