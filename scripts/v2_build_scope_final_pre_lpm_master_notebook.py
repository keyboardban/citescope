#!/usr/bin/env python3
"""Build the final consolidated SCOPE pre-LPM master notebook."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/04_scope_final_pre_lpm_master_notebook.ipynb'

def md(text: str): return nbf.v4.new_markdown_cell(text.strip())
def py(text: str): return nbf.v4.new_code_cell(text.strip())

def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md('''# SCOPE Condo Final Pre-LPM Master Notebook

This is the final descriptive and readiness checkpoint for the SCOPE condo citation study. It consolidates the validated work from v4-v8, creates the final pre-LPM package, and does **not** fit a final LPM.

All results are observational. Citation patterns are descriptive associations, not retrieval mechanisms or causal effects. Answer-derived variables, rank/position, provenance, and outcome duplicates remain outside the main model.'''),
        py('''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists(): ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.econometrics_eda_v2.final_pre_lpm_master import run_final_master

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
INPUT = BASE / 'tables/pre_lpm_feature_form/scope_condo_lpm_ready_general_taxonomy_feature_form.csv'
OUT = BASE / 'tables/final_pre_lpm_master'
FIG = BASE / 'figures/final_pre_lpm_master'
def show(name):
    path = FIG / name
    if not path.exists(): raise FileNotFoundError(path)
    display(Image(filename=str(path)))
def table(name, **kwargs): return pd.read_csv(OUT / name, **kwargs)
result = run_final_master(INPUT, BASE, OUT, FIG)
display(pd.DataFrame([result]))'''),
        md('''## 1. Dataset consistency

The master uses the v8 feature-form table, which is built on the v7 general taxonomy output. Counts are validated before diagnostics are interpreted.'''),
        py("display(table('dataset_consistency_summary.csv'))"),
        md('''## 2. Notebook lineage

Earlier notebooks are preserved as traceable sources. This notebook reuses their clearest validated outputs instead of re-running unrelated exploratory work.'''),
        py("display(table('notebook_lineage_map.csv'))"),
        md('''## 3. General taxonomy distribution

`page_type_family_general` is the main broad page-function candidate. `site_type_general` is a website-level companion. `unknown` is retained as a valid category.'''),
        py("display(table('page_type_family_general_distribution.csv'))\ndisplay(table('page_type_general_distribution.csv'))\ndisplay(table('site_type_general_distribution.csv'))\ndisplay(table('page_type_general_confidence_distribution.csv'))"),
        md('''## 4. Suspicious-label QA

This rule-based list is for manual checking only. It does not alter taxonomy labels or force ambiguous pages into a class.'''),
        py("display(table('general_taxonomy_suspicious_rows_final.csv'))"),
        md('''## 5. Feature-form inventory

Raw counts are preserved with `_raw` suffixes for diagnostics. Binary flags, threshold groups, and logged lengths are the interpretable forms considered for modeling.'''),
        py("display(table('feature_form_inventory_final.csv'))"),
        md('''## 6. Binary feature summary

These are unadjusted cited-rate differences with confidence intervals. They are not model coefficients.'''),
        py("display(table('binary_feature_cited_rate_summary_final.csv'))\nshow('binary_feature_diff_forest_final.png')"),
        md('''## 7. Categorical and binned diagnostics

Sparse categories are flagged for collapse or sensitivity use. `unknown` remains distinct rather than being treated as missing or reassigned.'''),
        py("display(table('categorical_feature_cited_rate_summary_final.csv'))\nshow('forest_page_type_family_general_final.png')\nshow('forest_site_type_general_final.png')\nshow('forest_content_quality_final.png')\nshow('forest_taxonomy_confidence_final.png')"),
        md('''## 8. Exact numeric diagnostics

The exact-count and rolling plots are descriptive shape checks. Content measurements are conditional on pages with measurable content; they are not assumed to be randomly observed.'''),
        py("show('exact_scatter_heading_count_final.png')\nshow('exact_scatter_table_count_final.png')\nshow('exact_scatter_link_count_final.png')\nshow('heatmap_heading_count_by_cited_final.png')\nshow('heatmap_table_count_by_cited_final.png')\nshow('heatmap_link_count_by_cited_final.png')\nshow('rolling_log1p_word_count_final.png')"),
        md('''## 9. Intent diagnostics

Intent is a prompt-level descriptive stratum and a candidate fixed effect. Heatmaps pair cited rate with frequency so small cells remain visible as small cells.'''),
        py("display(table('intent_column_audit_final.csv'))\ndisplay(table('intent_distribution_final.csv'))\ndisplay(table('intent_page_type_family_general_cell_summary_final.csv'))\ndisplay(table('intent_site_type_general_cell_summary_final.csv'))\nshow('heatmap_intent_by_page_type_family_general_cited_rate_final.png')\nshow('heatmap_intent_by_page_type_family_general_frequency_final.png')\nshow('heatmap_intent_by_site_type_general_cited_rate_final.png')\nshow('heatmap_intent_by_site_type_general_frequency_final.png')\nshow('stacked_page_type_family_general_by_intent_all_final.png')\nshow('stacked_page_type_family_general_by_intent_cited_final.png')\nshow('stacked_site_type_general_by_intent_all_final.png')\nshow('stacked_site_type_general_by_intent_cited_final.png')"),
        md('''## 10. Content missingness

Content-derived variables must be interpreted only where content is measurable. The all-row model uses an availability flag; the content model is explicitly restricted to available content.'''),
        py("display(table('missingness_content_availability_final.csv'))"),
        md('''## 10A. Detailed Scrape Failure Diagnostics

HTTP 200 means a server returned a response, but that response can still be a soft 404 or moved-page message. HTTP 502 means the browser, proxy, or server path failed before usable content was retrieved. Therefore scrape success is based on usable content, not HTTP status alone.'''),
        py("display(table('scrape_failure_category_summary.csv'))\ndisplay(table('soft_404_or_moved_audit.csv').head(20))\ndisplay(table('http_error_timeout_audit.csv').head(20))\ndisplay(table('url_normalization_issue_audit.csv').head(20))"),
        md('''## 11. Correlation, VIF, and redundancy

The VIF table is an early warning, not a fitted-model result. Do not include redundant length, content, or table transformations in the same specification.'''),
        py("display(table('correlation_matrix_final.csv', index_col=0))\ndisplay(table('vif_summary_final.csv'))\ndisplay(table('redundant_feature_recommendations_final.csv'))"),
        md('''## 12. Leakage guardrails

Answer text, answer similarity, labels, source provenance, and observed position/rank remain forbidden as main predictors. The audit verifies that the candidate list obeys that rule.'''),
        py("display(table('leakage_guardrail_final.csv'))"),
        md('''## 13. Sparse-category plan

Categories with low support, few cited rows, or few prompts are marked for `other`, sensitivity-only use, or diagnostic-only use. `unknown` is never silently collapsed.'''),
        py("display(table('sparse_category_collapse_plan_final.csv'))"),
        md('''## 14. Final candidate variable dictionary

This is the modeling handoff: it separates all-row predictors, content-subset candidates, sensitivity terms, diagnostics, and forbidden fields.'''),
        py("display(table('final_lpm_candidate_variable_dictionary.csv'))\ndisplay(Markdown('```json\\n' + (OUT / 'final_lpm_candidate_columns.json').read_text() + '\\n```'))"),
        md('''## 15. Recommended formula

This notebook only writes the formula. It does not estimate it. Run the main all-row model before adding content features, then use the content subset as a clearly labeled sensitivity analysis.'''),
        py("display(Markdown((OUT / 'recommended_lpm_v1_spec_final.md').read_text()))"),
        md('''## 16. Final readiness checklist

LPM estimation should begin only after every required check passes. A warning signals a known caveat rather than permission to ignore it.'''),
        py("display(table('final_pre_lpm_master_readiness_checklist.csv'))"),
        md('''## 17. Final report

The final table includes collapsed category fields required by the recommended formula. Raw counts, rank/position, source provenance, answer features, and outcome duplicates stay out of the main feature set.'''),
        py("display(Markdown((OUT / 'final_pre_lpm_master_report.md').read_text()))\nfinal_table = table('scope_condo_lpm_ready_final_pre_lpm_master.csv')\nprint('FINAL MASTER SUMMARY')\nprint(f\"Rows: {len(final_table):,}\")\nprint(f\"Unique URLs: {final_table['normalized_url'].nunique():,}\")\nprint(f\"Prompts: {final_table['prompt_id'].nunique():,}\")\nprint(f\"Cited rows: {int(final_table['cited'].sum()):,} ({final_table['cited'].mean():.1%})\")\nprint(f\"Unknown page family: {(final_table['page_type_family_general'] == 'unknown').mean():.1%}\")\nprint(f\"Final readiness: {result['final_readiness_status']}\")"),
    ]
    notebook.metadata['scope_final_pre_lpm_master'] = {'purpose': 'final consolidated descriptive EDA and LPM readiness; no model fit'}
    notebook.metadata['kernelspec'] = {'display_name': 'CiteScope Plotly (.venv)', 'language': 'python', 'name': 'citescope-v4-plotly'}
    nbf.write(notebook, TARGET)
    return TARGET

if __name__ == '__main__': print(build())
