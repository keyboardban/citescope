#!/usr/bin/env python3
"""Build the final Area Condo descriptive and pre-LPM master notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/08_area_condo_final_pre_lpm_master_notebook.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# Area Condo Final Pre-LPM Master Notebook

This notebook consolidates the latest **500-prompt Area Condo** source data, the completed Bright Data crawler scrape, and the recomputed general and real-estate taxonomies. Its structure follows the SCOPE notebook 04, but every table and figure below is recalculated from the latest Area Condo data.

The analysis is observational. A cited-rate difference is an association in the observed source appearances, not evidence about an AI system's internal retrieval or selection mechanism. More-only sources are shown-but-not-cited observations, not rejected pages. This notebook prepares an LPM design but does **not** fit the final model."""
        ),
        py(
            """from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import Image, Markdown, display

CODE_ROOT = Path.cwd().resolve()
if not (CODE_ROOT / 'src').exists() and (CODE_ROOT.parent / 'src').exists():
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.econometrics_eda_v2.area_condo_final_pre_lpm_master import run_area_condo_final_master
from src.econometrics_eda_v2.paths import topic_output_dir

BASE = topic_output_dir()
TAXONOMY = BASE / 'tables/area_condo_lpm_prep'
INPUT = TAXONOMY / 'area_condo_lpm_ready_with_taxonomy.csv'
OUT = BASE / 'tables/area_condo_final_pre_lpm_master'
FIG = BASE / 'figures/area_condo_final_pre_lpm_master'

result = run_area_condo_final_master(INPUT, TAXONOMY, OUT, FIG)
display(pd.DataFrame([result]).T.rename(columns={0: 'value'}))

def table(name, n=None):
    frame = pd.read_csv(OUT / name, low_memory=False)
    return frame if n is None else frame.head(n)

def figure(name, width=1000):
    display(Image(filename=str(FIG / name), width=width))"""
        ),
        md(
            """## 1. Dataset consistency and coverage

The unit of analysis is a **source appearance within a prompt**, so the same normalized URL may occur more than once. The coverage chart separates successful retrieval, measurable content, and taxonomy availability. These are related but not interchangeable checks."""
        ),
        py("display(table('dataset_consistency_summary.csv'))\nfigure('coverage_scrape_content_taxonomy.png', 950)"),
        md(
            """**Reading boundary:** scrape success says the provider returned a usable page record. `content_feature_available` says content-derived counts are measurable. A page can be scraped but still have weak, blocked, or insufficient content. Missing content is handled as selection/missingness, not random measurement noise."""
        ),
        md("""## 2. Data lineage

This compact lineage prevents the notebook from silently mixing the previous SCOPE data with the Area Condo experiment."""),
        py("display(table('data_lineage_map.csv'))"),
        md(
            """## 3. General page-function taxonomy

The general taxonomy is the primary cross-domain feature layer. It has three distinct levels:

- `site_type_general`: what kind of website/source it is, such as official company, marketplace, or forum.
- `page_type_family_general`: a broad family, such as support/help or contact/location.
- `page_type_general`: the detailed page function, such as `faq_page`, `contact_page`, `landing_page`, or `listing_page`.

The common-function view below keeps familiar detailed functions visible, retains `unknown`, and combines uncommon residual functions as `other_page_function`. Rare functions such as `landing_page` remain visible but are marked sparse. Points show cited-rate differences from the overall sample; bars are Wilson 95% intervals."""
        ),
        py(
            """display(table('distribution_page_type_family_general.csv'))
figure('difference_from_overall_page_type_family_general.png', 1050)
display(table('general_page_function_hierarchy.csv'))
display(table('distribution_page_type_general_common.csv'))
figure('difference_from_overall_common_page_function.png', 1050)
display(table('distribution_site_type_general.csv'))
figure('difference_from_overall_site_type_general.png', 1050)"""
        ),
        md(
            """## 4. Real-estate taxonomy sensitivity

The domain-specific taxonomy is retained as a sensitivity layer. It should be tested separately from the general taxonomy before both systems are placed in one model, because several categories encode overlapping page and site functions."""
        ),
        py(
            """display(table('distribution_page_type_family_real_estate.csv'))
figure('difference_from_overall_page_type_family_real_estate.png', 1050)
display(table('distribution_source_type_real_estate.csv'))
figure('difference_from_overall_source_type_real_estate.png', 1050)"""
        ),
        md(
            """## 5. Taxonomy confidence and manual QA

Confidence is a diagnostic property of the rule evidence, not a probability that a label is correct. The review sample deliberately covers high-impact and uncertain URLs. It should be manually audited before the final econometric claims are frozen."""
        ),
        py(
            """display(table('distribution_page_type_general_confidence.csv'))
figure('difference_from_overall_taxonomy_confidence.png', 950)
review = table('taxonomy_manual_review_sample_150.csv')
display(review[['source_url','source_root_domain','page_type_family_general','page_type_general','page_type_general_confidence','source_type_real_estate','page_type_family_real_estate','review_reason']].head(30))"""
        ),
        md(
            """## 6. Binary feature diagnostics

These are unadjusted descriptive comparisons. Content-derived indicators such as tables and headings are meaningful only when `content_feature_available = true`; the final content model enforces that restriction."""
        ),
        py("display(table('binary_feature_cited_rate_summary.csv'))\nfigure('binary_feature_difference_forest.png', 1050)"),
        md(
            """## 7. Content quality and failure modes

Content quality is shown separately from the provider's scrape-success flag. This makes blocked pages, parse failures, very short pages, and apparently dynamic pages visible instead of treating them as ordinary zero-valued content."""
        ),
        py(
            """display(table('distribution_content_strength.csv'))
display(table('distribution_content_quality_flag.csv'))
figure('difference_from_overall_content_quality.png', 950)
display(table('scrape_failure_summary.csv'))"""
        ),
        md(
            """## 8. Ordered numeric content diagnostics

Numeric bins remain in numeric order. Labels are readable thresholds rather than pandas interval artifacts, and each point is annotated with its row count. These plots use only rows where content features are measurable."""
        ),
        py(
            """display(table('numeric_feature_shape_summary.csv'))
display(table('numeric_feature_bin_diagnostics.csv'))
figure('cited_rate_by_heading_count_ordered.png', 950)
figure('cited_rate_by_word_count_ordered.png', 950)
figure('cited_rate_by_table_count_ordered.png', 950)
figure('cited_rate_by_link_count_ordered.png', 950)
display(table('numeric_feature_scatter_diagnostics.csv'))
figure('scatter_cited_rate_vs_word_count.png', 950)
figure('scatter_cited_rate_vs_heading_count.png', 950)
figure('scatter_cited_rate_vs_table_count.png', 950)
figure('scatter_cited_rate_vs_link_count.png', 950)"""
        ),
        md(
            """`heading_count` does not by itself establish a monotonic or linear relationship with cited rate. Treat it as diagnostic or as a possible control/sensitivity variable. `table_count` is zero-inflated, so `has_table` or a threshold group is preferable to the raw count."""
        ),
        md("""## 9. Content-length distribution

The log transform prevents a small number of very long pages from dominating the display. This remains a content-available-subset diagnostic."""),
        py("figure('distribution_log1p_word_count_by_cited.png', 900)"),
        md(
            """## 10. Intent-stratified diagnostics

Intent cells show where taxonomy composition and cited rates differ across prompt groups. Frequency heatmaps print row counts; cited-rate heatmaps print percentages. An asterisk marks cells with fewer than 20 source appearances. Blank or low-frequency cells should not be interpreted as stable effects. Intent is descriptive here; prompt fixed effects absorb prompt-level differences in the proposed LPM."""
        ),
        py(
            """display(table('distribution_intent_group.csv'))
figure('heatmap_intent_by_page_type_family_general_frequency.png', 1150)
figure('heatmap_intent_by_page_type_family_general_cited_rate.png', 1150)
figure('heatmap_intent_by_page_type_general_common_frequency.png', 1250)
figure('heatmap_intent_by_page_type_general_common_cited_rate.png', 1250)
figure('heatmap_intent_by_site_type_general_frequency.png', 1150)
figure('heatmap_intent_by_site_type_general_cited_rate.png', 1150)"""
        ),
        md(
            """## 11. Scrape/content missingness

The cited and more-only groups have slightly different content availability. The domain table helps identify whether missingness is concentrated in a few high-volume websites."""
        ),
        py(
            """display(table('scrape_content_availability_by_cited.csv'))
display(table('content_missingness_summary.csv'))
domains = table('scrape_content_availability_by_domain.csv')
display(domains.head(40))"""
        ),
        md(
            """## 12. Correlation, VIF, and redundant forms

Correlation and VIF are calculated only where content features are available. High collinearity between page-length measures is expected. Raw counts and transformed versions should not be entered together merely because both are available."""
        ),
        py(
            """display(table('content_feature_spearman_correlation.csv'))
display(table('vif_summary.csv'))
display(table('redundancy_recommendations.csv'))"""
        ),
        md(
            """## 13. Sparse-category plan

Rare categories are collapsed to `other` for the proposed main formula, while `unknown` remains explicit. This preserves uncertainty without creating unstable dummy coefficients from very small cells."""
        ),
        py("sparse = table('sparse_category_collapse_plan.csv')\ndisplay(sparse[sparse['sparse_flag']].sort_values(['feature','n_rows']))"),
        md(
            """## 14. Leakage guardrail

Answer text, answer overlap/similarity, source provenance, rank, position, and outcome-derived fields are excluded from the main candidate set. Position/rank may appear only in a separately labeled diagnostic sensitivity model using the original source table."""
        ),
        py(
            """display(table('leakage_guardrail.csv'))
with open(OUT / 'final_lpm_candidate_columns.json', encoding='utf-8') as handle:
    display(json.load(handle))
display(table('final_lpm_candidate_variable_dictionary.csv'))"""
        ),
        md(
            """## 15. Proposed LPM design

The first model uses all source appearances and includes prompt fixed effects. The content model is a separate conditional analysis restricted to pages where content features are measurable. Neither model has been estimated in this notebook."""
        ),
        py("display(Markdown((OUT / 'recommended_lpm_model_design.md').read_text(encoding='utf-8')))"),
        md("""## 16. Final readiness checklist"""),
        py("display(table('final_pre_lpm_readiness_checklist.csv'))"),
        md(
            """## 17. Interpretation boundary and report

This notebook does not fit a final LPM because its job is to verify feature forms, taxonomy quality, missingness, sparse cells, and leakage first. The next econometric notebook can fit the proposed specifications after the manual taxonomy sample is reviewed. All content coefficients must be described as conditional on successful measurable extraction."""
        ),
        py(
            """display(Markdown((OUT / 'final_pre_lpm_master_report.md').read_text(encoding='utf-8')))
final_table = table('area_condo_lpm_ready_final_pre_lpm_master.csv')
print('FINAL AREA CONDO PRE-LPM SUMMARY')
print(f"Rows: {len(final_table):,}")
print(f"Unique URLs: {final_table['normalized_url'].nunique():,}")
print(f"Prompts with sources: {final_table['prompt_id'].nunique():,}")
print(f"Cited rows: {int(final_table['cited'].sum()):,} ({final_table['cited'].mean():.1%})")
print(f"Scrape success: {final_table['scrape_success'].mean():.1%}")
print(f"Content available: {final_table['content_feature_available'].mean():.1%}")
print(f"Readiness: {result['readiness']}")"""
        ),
    ]
    notebook.metadata["area_condo_final_pre_lpm_master"] = {
        "purpose": "latest Area Condo descriptive EDA and LPM readiness; no final model fit",
        "source_rows": 5758,
        "source_prompts": 500,
    }
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata["language_info"] = {"name": "python", "version": "3"}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
