#!/usr/bin/env python3
"""Build the readable-graphs v5 copy of the SCOPE pre-LPM diagnostics notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v4_plotly.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v5_readable_graphs.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"V4 notebook not found: {TEMPLATE}")
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Feature Diagnostics v5: Readable Graphs

V5 preserves the v4 diagnostic methodology and inputs while promoting cleaner, tail-capped presentation plots. Full-resolution exact-value and rolling-window sensitivity views are retained as appendix-style diagnostics.

No final LPM is fit. All associations are descriptive, unadjusted, and conditional on scrape/content availability for content-derived features. Nothing here supports causal claims or claims about a hidden retrieval system.
"""),
        py("""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    raise RuntimeError(f'Cannot locate project root from {Path.cwd().resolve()}')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.pre_lpm_readable_graphs_v5 import run_readable_graph_diagnostics_v5

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
V4_FIG = BASE / 'figures/pre_lpm_eda_v4_plotly'
OUT = BASE / 'tables/pre_lpm_eda_v5_readable_graphs'
FIG = BASE / 'figures/pre_lpm_eda_v5_readable_graphs'
PREVIEW = FIG / 'preview'
INTERACTIVE = FIG / 'interactive'
if not LPM_PATH.exists():
    raise FileNotFoundError(f'LPM-ready input not found: {LPM_PATH}')

def show_preview(path):
    if not path.exists():
        raise FileNotFoundError(f'Expected image preview not found: {path}')
    display(Image(filename=str(path)))

print('Input:', LPM_PATH)
"""),
        md("""
## 1. V4 preservation

V4 remains unchanged. Its plots and detailed tables are retained as the previous diagnostic layer; V5 adds a cleaner default presentation layer using the same source data.
"""),
        py("""
for filename in [
    'preview/plotly_exact_frequency_scatter_heading_count_clean.png',
    'preview/plotly_count_heatmap_heading_count_by_cited.png',
    'preview/plotly_rolling_cited_rate_log1p_word_count.png',
]:
    path = V4_FIG / filename
    if path.exists():
        show_preview(path)
"""),
        md("""
## 2. Generate V5 readable diagnostics

The generator writes presentation-ready HTML artifacts, full diagnostic HTML artifacts, CSV tables, and embedded PNG previews. A missing optional Kaleido PNG exporter does not prevent the notebook previews from rendering.
"""),
        py("""
result = run_readable_graph_diagnostics_v5(LPM_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(pd.read_csv(OUT / 'graph_readability_checklist_v5.csv'))
"""),
        md("""
## 3. Exact-value cited-rate scatter plots

Each point is an exact feature value, and the y-axis is the cited rate among rows with that value. Color represents the number of observations supporting the estimate. Sparse or unstable estimates use a distinct marker in the interactive version and should not be over-interpreted.

The tail-capped scatter is the default readable presentation plot. Full exact-value plots remain appendix diagnostics.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(Markdown(f'### {feature}'))
    display(pd.read_csv(OUT / f'readable_exact_scatter_{feature}.csv'))
    show_preview(PREVIEW / f'readable_exact_scatter_{feature}.png')
"""),
        md("""
## 4. Raw concentration heatmaps

These heatmaps show raw row concentration before converting counts into cited rates. They use the same capped tails as the default scatter plots.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(Markdown(f'### {feature} concentration'))
    display(pd.read_csv(OUT / f'readable_heatmap_{feature}_by_cited.csv'))
    show_preview(PREVIEW / f'readable_heatmap_{feature}_by_cited.png')
"""),
        md("""
## 5. Rolling cited-rate curves

The default curves use overlapping 75-row windows. They diagnose possible shape, thresholds, or noise; they are not model estimates. Word content is displayed as `log1p_word_count`, not exact raw word-count cells.
"""),
        py("""
rolling = pd.read_csv(OUT / 'rolling_cited_rate_sensitivity_v5.csv')
display(rolling.groupby(['feature_name', 'window_size']).size().rename('rolling_windows').reset_index())
for feature in ['log1p_word_count', 'heading_count', 'link_count']:
    display(Markdown(f'### {feature}, 75-row window'))
    show_preview(PREVIEW / f'readable_rolling_{feature}.png')
"""),
        md("""
## 6. Forest-style categorical differences

These graphs show each category's unadjusted percentage-point difference from the overall cited rate, with Wilson intervals. The default plots show categories with at least 20 rows. Sparse categories remain in the accompanying table and are not removed from the data.

Final LPM work, if pursued later, would adjust for prompt fixed effects and other safe controls.
"""),
        py("""
forest = [
    ('page_type_family_real_estate', 'page_type_family'),
    ('source_type_real_estate', 'source_type'),
    ('content_quality_flag', 'content_quality'),
    ('re_page_type_confidence', 'taxonomy_confidence'),
]
for feature, slug in forest:
    display(Markdown(f'### {feature}'))
    display(pd.read_csv(OUT / f'forest_diff_{feature}.csv'))
    show_preview(PREVIEW / f'forest_diff_{slug}.png')
display(Markdown('### Sparse categories retained outside the default forest plots'))
display(pd.read_csv(OUT / 'sparse_categories_v5.csv'))
"""),
        md("""
## 7. Which graph should be used for what?

1. **Exact-value scatter**: count features such as heading count, table count, and link count. It answers: at this exact value, what is the cited rate?
2. **Heatmap**: raw concentration. It answers: where are the observations concentrated?
3. **Rolling curve**: numeric shape. It answers: does the relationship look linear, nonlinear, threshold-like, or noisy?
4. **Forest-style difference plot**: categorical features. It answers: which categories have higher or lower citation rates than the overall average?
5. **Distribution plots**: overlap between cited and non-cited values. They answer: do those rows occupy different numeric ranges?
"""),
        md("""
## 8. Appendix-style diagnostics

The artifact table identifies the full exact-value scatter plots and the 50-row and 100-row rolling sensitivity curves. They are preserved for detailed inspection but intentionally kept out of the main visual narrative.
"""),
        py("""
manifest = pd.read_csv(OUT / 'graph_artifact_manifest_v5.csv')
display(manifest)
display(Markdown((OUT / 'plot_readability_summary_v5.md').read_text()))
"""),
        md("""
## 9. Final guardrails

Status: **ready_for_pre_LPM_EDA_with_readable_interactive_graphs**.

No final LPM is fit, no answer-derived or outcome-derived predictors are introduced, and content-feature diagnostics remain conditional on scrape/content availability.
"""),
    ]
    notebook.metadata['scope_pre_lpm_v5_readable_graphs'] = {
        'source_notebook': str(TEMPLATE.relative_to(ROOT)),
        'purpose': 'readable descriptive Plotly graphs; no final LPM',
    }
    notebook.metadata['kernelspec'] = {
        'display_name': 'CiteScope Plotly (.venv)',
        'language': 'python',
        'name': 'citescope-v4-plotly',
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == '__main__':
    print(build())
