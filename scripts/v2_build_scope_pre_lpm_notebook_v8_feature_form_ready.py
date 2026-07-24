#!/usr/bin/env python3
"""Build the v8 feature-form readiness notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v8_feature_form_ready.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Diagnostics v8: Feature-Form Ready

V8 converts raw content/count measurements into interpretable dummies, thresholds, bins, and logs without fitting a final LPM. Raw counts remain diagnostics unless later EDA supports a well-supported linear form.
"""),
        py("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.econometrics_eda_v2.pre_lpm_feature_form_v8 import run_feature_form_v8

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
INPUT = BASE / 'tables/general_page_taxonomy/scope_condo_lpm_ready_with_general_page_taxonomy.csv'
EDA = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
OUT = BASE / 'tables/pre_lpm_feature_form'
FIG = BASE / 'figures/pre_lpm_feature_form'
def show(path):
    if not path.exists():
        raise FileNotFoundError(path)
    display(Image(filename=str(path)))
"""),
        md("""
## 1. Generate transformed features

Unavailable scraped content remains missing in content-derived feature columns. Availability and missingness indicators state whether the feature can be interpreted; missing content is not treated as absence.
"""),
        py("""
result = run_feature_form_v8(INPUT, EDA if EDA.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(pd.read_csv(OUT / 'feature_form_inventory.csv'))
"""),
        md("""
## 2. Binary and threshold feature diagnostics

These are unadjusted associations. Sparse or imbalanced features are marked and should not be elevated to main-model terms without support.
"""),
        py("""
display(pd.read_csv(OUT / 'binary_feature_cited_rate_summary.csv'))
show(FIG / 'preview/binary_feature_diff_forest.png')
"""),
        md("""
## 3. Binned and categorical forms

Heading, link, and word-count groups are sensitivity forms. `page_type_family_general` and `site_type_general` remain the broad categorical candidates for cross-domain analysis.
"""),
        py("""
display(pd.read_csv(OUT / 'binned_feature_cited_rate_summary.csv'))
for feature in ['heading_count_group', 'link_count_group', 'word_count_group', 'page_type_family_general', 'site_type_general']:
    show(FIG / 'preview' / f'cited_rate_by_{feature}.png')
"""),
        md("""
## 4. Correlation and VIF precheck

High correlation or VIF means related transformations should not be piled into the same model. In particular, word and character length logs should be alternatives unless a later specification check supports both.
"""),
        py("""
display(pd.read_csv(OUT / 'correlation_matrix_feature_form.csv', index_col=0))
display(pd.read_csv(OUT / 'vif_feature_form_summary.csv'))
"""),
        md("""
## 5. LPM-ready v2 table and guardrails

The table contains transformed forms, availability flags, and raw diagnostic fields with `_raw` suffixes. It excludes answer-derived, outcome-derived, provenance, and rank/position predictors.
"""),
        py("""
display(pd.read_json(OUT / 'lpm_main_candidate_columns.json', typ='series').to_frame('columns'))
display(Markdown((OUT / 'feature_form_readiness_report.md').read_text()))
display(Markdown('```json\\n' + (OUT / 'feature_form_validation.json').read_text() + '\\n```'))
"""),
        md("""
## 6. Final status

Status: **ready_for_LPM_v1_after_feature_form_layer** when the validation file reports `pass`. No final LPM is fit in this notebook.
"""),
    ]
    notebook.metadata['scope_pre_lpm_v8_feature_form'] = {'purpose': 'feature-form readiness; no final LPM'}
    notebook.metadata['kernelspec'] = {'display_name': 'CiteScope Plotly (.venv)', 'language': 'python', 'name': 'citescope-v4-plotly'}
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == '__main__':
    print(build())
