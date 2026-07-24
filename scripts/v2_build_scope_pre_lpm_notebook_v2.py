#!/usr/bin/env python3
"""Build the ordered numeric-diagnostics v2 copy of the SCOPE pre-LPM notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v2.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Working notebook not found: {TEMPLATE}")
    notebook = nbf.read(TEMPLATE, as_version=4)
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Feature Diagnostics v2

This is a patched copy of the SCOPE pre-LPM diagnostics notebook. It remains descriptive and pre-model: **no final LPM is fit**.

V2 fixes numeric-bin presentation. Old labels such as `(-0.001, 1.0]` were pandas `qcut` display artifacts used to include zero; they never represented negative page measurements or negative citation rates. V2 uses observed raw ranges or explicit threshold labels and preserves numeric order in all numeric charts.

The unit is one surfaced source appearance. `cited = 1` means explicitly cited; `cited = 0` means surfaced/more-only but not cited. These associations do not identify hidden retrieval behavior or causal effects.
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

from src.econometrics_eda_v2.pre_lpm_numeric_diagnostics_v2 import run_numeric_diagnostics_v2

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
OUT = BASE / 'tables/pre_lpm_eda_v2'
FIG = BASE / 'figures/pre_lpm_eda_v2'
if not LPM_PATH.exists():
    raise FileNotFoundError(f'LPM-ready input not found: {LPM_PATH}')
print('Input:', LPM_PATH)
"""),
        md("""
## 1. Generate v2 numeric diagnostics

All numeric content diagnostics are conditional on `content_feature_available == 1` or `scraped_body_available == 1`. Missing content is not imputed as zero. The final LPM-ready table remains unchanged and no answer-derived, rank, or outcome-duplicate feature is introduced.
"""),
        py("""
result = run_numeric_diagnostics_v2(LPM_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(Markdown((OUT / 'numeric_feature_interpretation_v2.md').read_text()))
"""),
        md("""
## 2. Readable ordered numeric bins

`word_count` and `text_char_count` use quartiles only when they form multiple bins; their labels reflect the observed minimum/maximum in each bin. Zero-inflated count variables use fixed thresholds:

- `heading_count`: 0–1, 2–6, 7–12, 13+
- `table_count`: 0, 1, 2+
- `link_count`: 0–3, 4–8, 9+

Sparse bins (`n < 20` or fewer than 5 cited rows) are flagged in the tables. Numeric plots are ordered by bin, not by cited rate.
"""),
        py("""
numeric_bins = pd.read_csv(OUT / 'numeric_feature_bin_diagnostics_v2.csv')
threshold_bins = pd.read_csv(OUT / 'numeric_feature_threshold_diagnostics_v2.csv')
shape_summary = pd.read_csv(OUT / 'numeric_feature_shape_summary_v2.csv')
display(numeric_bins)
display(threshold_bins)
display(shape_summary)
"""),
        md("""
## 3. Ordered citation-rate point plots

Each point is a cited rate with a Wilson 95% confidence interval. The dashed horizontal line is the all-row cited rate. The line joins bins only to make their order visible; it is not a fitted trend or evidence of linearity.
"""),
        py("""
for filename in ['cited_rate_by_word_count_ordered.png', 'cited_rate_by_heading_count_ordered.png', 'cited_rate_by_table_count_threshold.png', 'cited_rate_by_link_count_ordered.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 4. Difference-from-overall diagnostics

These plots express category/bin differences in percentage points relative to the overall cited rate, with Wilson intervals shifted by that reference rate. The zero line marks no descriptive difference. They are association diagnostics only.
"""),
        py("""
difference_from_overall = pd.read_csv(OUT / 'difference_from_overall_by_feature_v2.csv')
display(difference_from_overall)
for filename in ['diff_from_overall_page_type_family.png', 'diff_from_overall_source_type.png', 'diff_from_overall_content_quality.png', 'diff_from_overall_heading_count.png', 'diff_from_overall_word_count.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 5. Distribution diagnostics

These diagnostic boxplots compare log-transformed count distributions for cited versus surfaced/more-only rows. They are intentionally conditional on measurable page content and should not be generalized to all surfaced sources.
"""),
        py("""
for filename in ['distribution_log1p_word_count_by_cited.png', 'distribution_log1p_heading_count_by_cited.png', 'distribution_log1p_link_count_by_cited.png']:
    display(Image(filename=str(FIG / filename)))
"""),
        md("""
## 6. Interpretation and guardrails

`heading_count does not show a clear monotonic or linear association with cited rate. Treat it as diagnostic or as a possible control/sensitivity variable, not as a main focal predictor.`

`table_count is zero-inflated or duplicate-heavy, so quantile binning collapsed. Use has_table or table_count_group instead of raw table_count for EDA and LPM sensitivity.`

Keep the main LPM focused on broad page family, conservative source type with an explicit unknown category, and prompt fixed effects. Numeric content features belong in content-available sensitivity analyses. Do not use answer-derived similarity, source origin/group, source position/rank, cited-label duplicates, or more-only outcome complements as main predictors.
"""),
    ]
    notebook.metadata["scope_pre_lpm_v2"] = {"source_notebook": str(TEMPLATE.relative_to(ROOT)), "purpose": "ordered zero-aware numeric diagnostics; no final LPM"}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
