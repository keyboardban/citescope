#!/usr/bin/env python3
"""Build the detailed numeric-shape v3 copy of the SCOPE pre-LPM notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v2.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v3.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"V2 notebook not found: {TEMPLATE}")
    notebook = nbf.read(TEMPLATE, as_version=4)
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Feature Diagnostics v3

This v3 copy preserves the v2 ordered-bin diagnostics and adds scatter-style numeric shape checks: raw-row jitter, exact-count bubble rates, rolling curves, distribution overlap, and count heatmaps. **No final LPM is fit.**

All diagnostics are descriptive. The unit is one surfaced source appearance; `cited = 0` means surfaced/more-only, not rejected from a hidden retrieval set. Numeric page-content features are interpreted only where content is available or scrape extraction succeeded.
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

from src.econometrics_eda_v2.pre_lpm_numeric_shape_v3 import run_numeric_shape_diagnostics_v3

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
V2_FIG = BASE / 'figures/pre_lpm_eda_v2'
OUT = BASE / 'tables/pre_lpm_eda_v3'
FIG = BASE / 'figures/pre_lpm_eda_v3'
if not LPM_PATH.exists():
    raise FileNotFoundError(f'LPM-ready input not found: {LPM_PATH}')
print('Input:', LPM_PATH)
"""),
        md("""
## 1. Preserve v2 ordered-bin diagnostics

V2 remains the compact check for readable thresholds and ordered citation rates. The v3 outputs below add detail rather than replacing those views.
"""),
        py("""
for filename in ['cited_rate_by_word_count_ordered.png', 'cited_rate_by_heading_count_ordered.png', 'cited_rate_by_table_count_threshold.png', 'cited_rate_by_link_count_ordered.png']:
    display(Image(filename=str(V2_FIG / filename)))
"""),
        md("""
## 2. Generate detailed numeric-shape diagnostics

1. Row-level jitter shows raw observations but the y-axis is binary, so it is not itself a cited-rate curve.
2. Exact-value bubbles show cited rate for discrete counts and scale point area by concentration.
3. Continuous word/text features are not grouped by exact value because small cells create unstable 0%/100% rates.
4. Rolling curves inspect whether broad shape is increasing, decreasing, threshold-like, or noisy.
5. These exploratory plots guide possible functional-form sensitivity choices; they do not make causal claims.
"""),
        py("""
result = run_numeric_shape_diagnostics_v3(LPM_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(Markdown((OUT / 'numeric_shape_interpretation_v3.md').read_text()))
"""),
        md("""
## 3. Exact-value cited-rate bubble plots

Use these only for count-like variables. Bubble size shows the number of rows behind each exact cited rate; Wilson intervals and sparse flags identify values that should not be over-interpreted.
"""),
        py("""
for feature in ['heading_count', 'table_count', 'link_count']:
    display(Markdown(f'### {feature}'))
    display(pd.read_csv(OUT / f'exact_value_cited_rate_{feature}.csv'))
    display(Image(filename=str(FIG / f'bubble_exact_cited_rate_{feature}.png')))
"""),
        md("""
## 4. Row-level jitter plots

Jitter exposes raw concentration and overlap between cited/non-cited rows. For skewed content counts, log transforms are used on the x-axis. Vertical jitter is visual only.
"""),
        py("""
for filename in ['jitter_row_level_heading_count_by_cited.png', 'jitter_row_level_log1p_word_count_by_cited.png', 'jitter_row_level_log1p_text_char_count_by_cited.png', 'jitter_row_level_link_count_by_cited.png', 'jitter_row_level_table_count_by_cited.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 5. Rolling cited-rate curves

Each curve uses sorted, overlapping windows of 50 rows. Curves are a smoothed descriptive check, not independent observations or fitted causal relationships.
"""),
        py("""
rolling = pd.read_csv(OUT / 'rolling_cited_rate_numeric_features.csv')
display(rolling.groupby('feature_name').size().rename('rolling_windows').reset_index())
for filename in ['rolling_cited_rate_log1p_word_count.png', 'rolling_cited_rate_heading_count.png', 'rolling_cited_rate_link_count.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 6. Distribution overlap and count heatmaps

Distribution overlap indicates how weakly a feature separates cited from surfaced/more-only observations on its own. Heatmaps directly show count concentration by cited status without first converting counts to cited rates.
"""),
        py("""
for filename in ['distribution_log1p_word_count_by_cited.png', 'distribution_heading_count_by_cited.png', 'distribution_link_count_by_cited.png', 'distribution_table_count_by_cited.png', 'heatmap_heading_count_by_cited.png', 'heatmap_table_count_by_cited.png', 'heatmap_link_count_by_cited.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 7. Shape recommendations

Use this table as a functional-form guide for later sensitivity analysis, not as a final specification selection rule. In particular, `table_count` should be thresholded (`has_table` or 0/1/2+) and `heading_count` remains diagnostic/binned at most unless later model checks justify more.
"""),
        py("""
recommendations = pd.read_csv(OUT / 'numeric_feature_visual_shape_recommendations_v3.csv')
display(recommendations)
print('Final notebook status: ready_for_pre_LPM_EDA_with_detailed_numeric_shape_diagnostics')
print('No final LPM was fitted.')
"""),
    ]
    notebook.metadata['scope_pre_lpm_v3'] = {'source_notebook': str(TEMPLATE.relative_to(ROOT)), 'purpose': 'detailed numeric shape diagnostics; no final LPM'}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == '__main__':
    print(build())
