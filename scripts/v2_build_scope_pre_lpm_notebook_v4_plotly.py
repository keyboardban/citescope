#!/usr/bin/env python3
"""Build the Plotly v4 copy of the SCOPE pre-LPM diagnostics notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v3.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v4_plotly.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"V3 notebook not found: {TEMPLATE}")
    notebook = nbf.read(TEMPLATE, as_version=4)
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Feature Diagnostics v4: Interactive Plotly Shapes

This v4 copy preserves all v3 outputs and adds interactive Plotly diagnostics. Section 3 replaces bubble-size frequency encoding with **fixed-size markers colored by row frequency**, making exact-value cited rates easier to inspect through hover.

No final LPM is fit. These are descriptive pre-model diagnostics, conditional on content availability/scrape success for page-content counts. They do not support causal claims or claims about a hidden retrieval system.
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

from src.econometrics_eda_v2.pre_lpm_diagnostics import enrich_lpm_diagnostics
from src.econometrics_eda_v2.pre_lpm_plotly_v4 import (
    _add_logs,
    exact_frequency_stats,
    rolling_stats,
    run_plotly_diagnostics_v4,
)

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
V3_FIG = BASE / 'figures/pre_lpm_eda_v3'
OUT = BASE / 'tables/pre_lpm_eda_v4_plotly'
FIG = BASE / 'figures/pre_lpm_eda_v4_plotly'
INTERACTIVE = FIG / 'interactive'
if not LPM_PATH.exists():
    raise FileNotFoundError(f'LPM-ready input not found: {LPM_PATH}')

def show_preview(path):
    if not path.exists():
        raise FileNotFoundError(f'Expected PNG preview not found: {path}')
    display(Image(filename=str(path)))

print('Input:', LPM_PATH)
"""),
        md("""
## 1. V3 output preservation

The static v3 bubble, jitter, rolling, distribution, and heatmap outputs remain available unchanged. V4 adds visible numeric-shape previews without relying on browser JavaScript support.
"""),
        py("""
for filename in ['bubble_exact_cited_rate_heading_count.png', 'rolling_cited_rate_log1p_word_count.png', 'heatmap_heading_count_by_cited.png']:
    path = V3_FIG / filename
    if path.exists():
        display(Image(filename=str(path)))
"""),
        md("""
## 2. Generate Plotly v4 diagnostics

The generator writes detailed tables, interactive HTML artifacts, and visible PNG previews for this notebook.
"""),
        py("""
result = run_plotly_diagnostics_v4(LPM_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
_lpm = pd.read_csv(LPM_PATH, low_memory=False)
_eda = pd.read_csv(EDA_PATH, low_memory=False) if EDA_PATH.exists() else None
df_plotly = _add_logs(enrich_lpm_diagnostics(_lpm, _eda))
display(pd.DataFrame([result]))
summary = pd.read_csv(OUT / 'plotly_numeric_diagnostics_summary_v4.csv')
display(summary)
display(Markdown((OUT / 'plotly_exact_frequency_scatter_summary_v4.md').read_text()))
"""),
        md("""
## 3. Exact-value cited-rate scatter plots with frequency color

The **clean** plot is the primary readable view: every point has a fixed marker size, x is the exact count, y is cited rate (%), and color shows the supporting row frequency. Hover reveals exact counts, cited/more-only totals, rate difference, Wilson interval, and sparse/unstable flags.

### How to read these plots
1. Each point is one exact feature value, not one webpage.
2. The y-axis is cited rate among all rows with that exact value.
3. Color shows how many rows support the rate; darker points have more observations.
4. Pale points with low `n` are sparse and should not be over-interpreted.
5. The plot is descriptive only and helps choose raw, log, threshold, bins, or diagnostic-only treatment.

Treat `n_rows < 20` as sparse. Treat fewer than five cited or more-only rows as unstable.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(Markdown(f'### {feature}'))
    stats = exact_frequency_stats(df_plotly, feature)
    display(stats)
    show_preview(FIG / 'preview' / f'plotly_exact_frequency_scatter_{feature}_clean.png')
"""),
        md("""
## 3B. Raw row concentration by exact value and cited status

These heatmaps show the raw counts before transforming them into cited rates. Long tails are grouped (heading 26+, table 6+, link 31+) so concentration is legible.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(Markdown(f'### {feature} concentration'))
    display(pd.read_csv(OUT / f'plotly_count_heatmap_{feature}_by_cited.csv'))
    show_preview(FIG / 'preview' / f'plotly_count_heatmap_{feature}_by_cited.png')
"""),
        md("""
## 4. Tail-capped and CI variants

Full exact-value plots preserve numeric x-values. Tail-capped plots intentionally use a categorical x-axis for grouped tails. The CI version is retained for uncertainty inspection, while the clean color-frequency scatter remains the preferred notebook view.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(pd.read_csv(OUT / f'exact_value_frequency_scatter_{feature}_capped.csv'))
"""),
        md("""
## 5. Interactive rolling cited-rate curves

Rolling curves use a sorted, overlapping 50-row window. `log1p_word_count` is used for word content because exact raw word values are too granular and frequently have tiny cell counts.
"""),
        py("""
rolling = pd.read_csv(OUT / 'plotly_rolling_cited_rate_numeric_features.csv')
display(rolling.groupby('feature_name').size().rename('rolling_windows').reset_index())
for feature in ['log1p_word_count', 'heading_count', 'link_count']:
    stats = rolling_stats(df_plotly, feature)
    show_preview(FIG / 'preview' / f'plotly_rolling_cited_rate_{feature}.png')
"""),
        md("""
## 6. Final guardrails

- Clean scatter: readability through fixed-size markers and frequency color.
- CI scatter: uncertainty diagnostic, potentially dense in sparse tails.
- Heatmap: raw row concentration by count and cited status.
- HTML is the preferred exploration format; static PNG remains optional for reports.

Final status: **ready_for_pre_LPM_EDA_with_interactive_numeric_shape_diagnostics**. No final LPM is fit, no answer-derived/leakage fields are introduced, and all numeric interpretations remain conditional on scrape/content availability.
"""),
    ]
    notebook.metadata['scope_pre_lpm_v4_plotly'] = {'source_notebook': str(TEMPLATE.relative_to(ROOT)), 'purpose': 'interactive Plotly numeric shape diagnostics; no final LPM'}
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
