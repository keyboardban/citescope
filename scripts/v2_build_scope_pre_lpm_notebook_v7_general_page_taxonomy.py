#!/usr/bin/env python3
"""Build the general-page-taxonomy v7 pre-LPM notebook."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v7_general_page_taxonomy.ipynb"

def md(text: str): return nbf.v4.new_markdown_cell(text.strip())
def py(text: str): return nbf.v4.new_code_cell(text.strip())

def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Diagnostics v7: General Website Page Taxonomy

This additive layer replaces vertical-specific page function labels with a reusable cross-domain taxonomy. `site_type_general` describes the website/source, while `page_type_general` describes the individual page function. `page_type_family_general` is the recommended broad feature for cross-domain EDA and future LPM work.

No final LPM is fit. All graphs are descriptive, unadjusted, and do not use citation outcome, rank, position, answer text, or answer-derived features for classification.
"""),
        py("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists(): ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.econometrics_eda_v2.general_page_taxonomy_pipeline import run_general_page_taxonomy

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
OUT = BASE / 'tables/general_page_taxonomy'
FIG = BASE / 'figures/general_page_taxonomy'
def show(path):
    if not path.exists(): raise FileNotFoundError(path)
    display(Image(filename=str(path)))
"""),
        md("""
## 1. Generate general taxonomy outputs

The classifier combines URL, title/meta, domain, headings, and usable scraped content with evidence scoring. Weak or conflicting evidence remains `unknown`; poor content cannot override a useful URL seed.
"""),
        py("""
result = run_general_page_taxonomy(LPM, EDA if EDA.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(pd.read_csv(OUT / 'general_page_taxonomy_summary.csv').query("summary_type == 'distribution'"))
"""),
        md("""
## 2. General taxonomy versus vertical diagnostics

The original real-estate labels remain available only as vertical diagnostics. The comparison table makes it possible to inspect disagreements without treating the old taxonomy as the main cross-domain feature.
"""),
        py("""
display(pd.read_csv(OUT / 'general_page_taxonomy_comparison.csv'))
"""),
        md("""
## 3. Main cross-domain page-function graph

`page_type_family_general` is the default main graph because it is broad enough for cross-domain analysis. The detailed page type is retained as a more granular diagnostic and may be sparse.
"""),
        py("""
show(FIG / 'preview/cited_rate_by_page_type_family_general.png')
show(FIG / 'preview/cited_rate_by_page_type_general.png')
show(FIG / 'preview/cited_rate_by_site_type_general.png')
"""),
        md("""
## 4. Intent × general page-function diagnostics

The heatmap pairs cited rates with a frequency version. High-rate cells with low support should remain descriptive only. `unknown` is kept visible rather than forced into another category.
"""),
        py("""
for filename in [
    'intent_page_type_family_general_cited_rate.png',
    'intent_page_type_family_general_frequency.png',
    'intent_site_type_general_cited_rate.png',
    'intent_site_type_general_frequency.png',
]: show(FIG / filename)
display(pd.read_csv(OUT / 'intent_page_type_family_general_cell_summary.csv'))
display(pd.read_csv(OUT / 'intent_site_type_general_cell_summary.csv'))
"""),
        md("""
## 5. Composition by intent

These charts show the page-function or site-type mix surfaced within each intent. They are composition diagnostics, not adjusted effects.
"""),
        py("""
show(FIG / 'composition_page_type_family_general_by_intent.png')
show(FIG / 'composition_site_type_general_by_intent.png')
"""),
        md("""
## 6. Evidence and manual QA

The review sample is stratified by confidence and includes high-impact cited sources. Evidence columns are retained in the LPM-ready output for auditability, not as main predictors.
"""),
        py("""
display(pd.read_csv(OUT / 'general_page_taxonomy_manual_review_sample_150.csv'))
display(Markdown((OUT / 'general_page_taxonomy_report.md').read_text()))
"""),
        md("""
## 7. Final guidance

- Main cross-domain EDA/LPM candidates: `page_type_family_general`, `site_type_general`, general taxonomy-confidence flags, and prompt fixed effects.
- Diagnostic or sensitivity variables: `page_type_general`, real-estate-specific taxonomy fields, source/rank fields, and numeric content features.
- Forbidden main predictors remain answer-derived similarity, outcome duplicates, provenance fields, and rank/position variables.

Status: **general taxonomy ready for cross-domain pre-LPM EDA; vertical-specific taxonomy retained as optional diagnostics.**
"""),
    ]
    notebook.metadata['scope_pre_lpm_v7_general_page_taxonomy'] = {'purpose': 'general cross-domain page-function taxonomy; no final LPM'}
    notebook.metadata['kernelspec'] = {'display_name': 'CiteScope Plotly (.venv)', 'language': 'python', 'name': 'citescope-v4-plotly'}
    nbf.write(notebook, TARGET)
    return TARGET

if __name__ == '__main__': print(build())
