#!/usr/bin/env python3
"""Build the intent-stratified v6 copy of the SCOPE pre-LPM notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v5_readable_graphs.ipynb"
TARGET = ROOT / "notebooks/03_scope_pre_lpm_feature_diagnostics_v6_intent_stratified.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"V5 notebook not found: {TEMPLATE}")
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""
# SCOPE Condo Pre-LPM Feature Diagnostics v6: Intent-Stratified Graphs

V6 preserves the v5 readable-graphs diagnostics and adds a descriptive layer showing how citation patterns, source types, and page types vary across question intents.

No final LPM is fit. These are unadjusted pre-model diagnostics. They do not make causal claims, and intent interactions are sensitivity candidates rather than headline model terms.
"""),
        py("""
from pathlib import Path
import sys
import re
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src').exists():
    raise RuntimeError(f'Cannot locate project root from {Path.cwd().resolve()}')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.pre_lpm_intent_stratified_v6 import run_intent_stratified_diagnostics_v6

BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
LPM_PATH = BASE / 'tables/final_lpm_prep/scope_condo_lpm_ready.csv'
EDA_PATH = BASE / 'tables/scope_condo_eda_ready_post_scrape.csv'
V5_PREVIEW = BASE / 'figures/pre_lpm_eda_v5_readable_graphs/preview'
OUT = BASE / 'tables/pre_lpm_eda_v6_intent_stratified'
FIG = BASE / 'figures/pre_lpm_eda_v6_intent_stratified'
PREVIEW = FIG / 'preview'
INTERACTIVE = FIG / 'interactive'
if not LPM_PATH.exists():
    raise FileNotFoundError(f'LPM-ready input not found: {LPM_PATH}')

def show_preview(path):
    if not path.exists():
        raise FileNotFoundError(f'Expected image preview not found: {path}')
    display(Image(filename=str(path)))
"""),
        md("""
## 1. V5 preservation

V5 remains unchanged. Its capped count plots, rolling curves, and categorical forest plots remain the main non-intent diagnostic layer.
"""),
        py("""
for filename in ['readable_exact_scatter_heading_count.png', 'forest_diff_page_type_family.png']:
    path = V5_PREVIEW / filename
    if path.exists():
        show_preview(path)
"""),
        md("""
## 2. Intent column validation

The LPM-ready table is preferred. If it does not already contain an intent field, intent is attached at the prompt level from the post-scrape EDA table. The audit below documents the selected column and source.
"""),
        py("""
result = run_intent_stratified_diagnostics_v6(LPM_PATH, EDA_PATH if EDA_PATH.exists() else None, OUT, FIG)
display(pd.DataFrame([result]))
display(pd.read_csv(OUT / 'intent_column_audit.csv'))
display(pd.read_csv(OUT / 'intent_distribution.csv'))
"""),
        md("""
## 3. Intent × source-type diagnostics

Each cell retains `unknown` source type. A cell is eligible for future interaction sensitivity work only with adequate rows, cited and more-only observations, and prompt coverage. Sparse cells remain descriptive only.
"""),
        py("""
display(pd.read_csv(OUT / 'intent_source_type_cell_summary.csv'))
"""),
        md("""
## 4. Intent × source-type heatmaps

Darker high-rate cells may simply have low support. Always read cited rate together with `n_rows`; sparse cells are descriptive only. The frequency view shows where the observations are concentrated.
"""),
        py("""
for suffix in ['cited_rate', 'frequency']:
    show_preview(PREVIEW / f'heatmap_intent_by_source_type_{suffix}.png')
"""),
        md("""
## 5. Source-type composition by intent

All-source composition shows what the AI surfaced. Cited-source composition shows what it explicitly cited. Differences are suggestive patterns, not causal effects.
"""),
        py("""
for suffix in ['all_sources', 'cited_sources']:
    show_preview(PREVIEW / f'stacked_source_type_composition_by_intent_{suffix}.png')
"""),
        md("""
## 6. Intent × page-type diagnostics

Page-type family cells use the same support rules. The detail-level page-type table is retained as diagnostic only because it is more likely to be sparse.
"""),
        py("""
display(pd.read_csv(OUT / 'intent_page_type_family_cell_summary.csv'))
display(Markdown('### Detail-level diagnostic table'))
display(pd.read_csv(OUT / 'intent_page_type_detail_cell_summary.csv'))
"""),
        md("""
## 7. Intent × page-type heatmaps and composition

Read cited rates alongside cell counts. Composition charts describe the source/page mix surfaced or cited within each intent; they are not adjusted comparisons.
"""),
        py("""
for suffix in ['cited_rate', 'frequency']:
    show_preview(PREVIEW / f'heatmap_intent_by_page_type_family_{suffix}.png')
for suffix in ['all_sources', 'cited_sources']:
    show_preview(PREVIEW / f'stacked_page_type_family_composition_by_intent_{suffix}.png')
"""),
        md("""
## 8. Difference-from-overall forest diagnostics by intent

Forest plots show unadjusted within-intent category differences from the overall cited rate. Sparse categories should not be treated as stable effects. The interactive artifacts provide a dropdown across all intents; visible previews below show each intent separately.
"""),
        py("""
intent_order = pd.read_csv(OUT / 'intent_distribution.csv')['intent_group'].tolist()
for feature, stem in [('source_type_real_estate', 'source_type'), ('page_type_family_real_estate', 'page_type_family')]:
    display(Markdown(f'### {feature}'))
    for intent in intent_order:
        safe_intent = re.sub(r'[^a-z0-9]+', '_', intent.casefold()).strip('_') or 'missing'
        show_preview(PREVIEW / f'forest_{stem}_{safe_intent}.png')
display(pd.read_csv(OUT / 'intent_stratified_forest_plot_data.csv'))
"""),
        md("""
## 9. Interaction candidates and prompt balance

Interaction candidates are for future sensitivity models only. The main LPM should remain simpler first. Prompt-level balance checks whether a pattern is represented by enough independent prompts rather than being dominated by a small number.
"""),
        py("""
display(pd.read_csv(OUT / 'intent_interaction_candidate_summary.csv'))
display(pd.read_csv(OUT / 'prompt_intent_balance_audit.csv'))
display(pd.read_csv(OUT / 'intent_prompt_balance_summary.csv'))
"""),
        md("""
## 10. Intent-stratified graph guide

1. **Intent × source-type heatmap:** Within each question intent, which source types have higher cited rates?
2. **Intent × page-type heatmap:** Within each intent, which page types have higher cited rates?
3. **Source-type composition:** What source mix is surfaced for each intent?
4. **Cited source-type composition:** What source mix is actually cited for each intent?
5. **Forest plot by intent:** Which source/page categories differ most from the overall cited rate within an intent?
6. **Prompt balance audit:** Are intent-level patterns based on enough prompts?
"""),
        py("""
display(Markdown((OUT / 'intent_stratified_graph_guide.md').read_text()))
display(Markdown((OUT / 'intent_stratified_diagnostics_summary.md').read_text()))
display(pd.read_csv(OUT / 'intent_graph_artifact_manifest.csv'))
"""),
        md("""
## 11. Final status

Status: **ready_for_LPM_v1_with_intent_stratified_prechecks**.

Intent interactions should be sensitivity models first, not main headline model terms. No final LPM is fit here.
"""),
    ]
    notebook.metadata['scope_pre_lpm_v6_intent_stratified'] = {
        'source_notebook': str(TEMPLATE.relative_to(ROOT)),
        'purpose': 'intent-stratified descriptive diagnostics; no final LPM',
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
