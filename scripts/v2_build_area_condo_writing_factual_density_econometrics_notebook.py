#!/usr/bin/env python3
"""Build notebook 11 for writing/factual-density econometrics."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/11_area_condo_writing_factual_density_econometrics.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# 11 Area Condo Writing and Factual-Density Econometrics

Notebook 09 found `has_table` as the clearest suggestive structural signal. Notebook 10 created leakage-safe writing/factual-density features. Notebook 11 estimates whether these richer features are associated with citation probability and whether they explain or modify the descriptive `has_table` association.

The unit is one surfaced source appearance and the estimand is `P(cited = 1 | source surfaced in this audit)`.

**These models estimate conditional associations among surfaced sources, not causal effects of changing page content.**

Table presence is treated as an observational association, not a causal mechanism. Results are not web-wide, and they should not be converted into promises about content changes."""
        ),
        py(
            """from pathlib import Path
import inspect
import json
import sys

import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display

CODE_ROOT = Path.cwd().resolve()
if not (CODE_ROOT / "src").exists() and (CODE_ROOT.parent / "src").exists():
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.econometrics_eda_v2.writing_factual_density_econometrics import (
    compare_covariance_types,
    fit_lpm,
    focal_terms_only,
    make_forest_plot,
    make_has_table_path_plot,
    run_writing_factual_density_econometrics,
    save_model_result,
    tidy_results,
)
from src.econometrics_eda_v2.paths import topic_output_dir

PACKAGE = topic_output_dir() / "content_econometrics_ai_package"
TABLES = PACKAGE / "tables/11_writing_factual_density_econometrics"
FIGURES = PACKAGE / "figures/11_writing_factual_density_econometrics"
REPORTS = PACKAGE / "reports/11_writing_factual_density_econometrics"

result = run_writing_factual_density_econometrics(PACKAGE)
display(pd.DataFrame([result]).T.rename(columns={0: "value"}))

def table(name, n=None):
    frame = pd.read_csv(TABLES / name, low_memory=False)
    return frame if n is None else frame.head(n)

def show_plot(name):
    display(pio.read_json(FIGURES / name.replace(".html", ".plotly.json")))

def focal(name):
    frame = table(name)
    return frame[frame["term"].str.contains(
        "has_table|factual_numeric_density|price_unit_detail|location_transit|"
        "prompt_page_relevance|amenity_project_detail|external_evidence",
        regex=True,
        na=False,
    )]"""
        ),
        md("""## 1. Load data and verify readiness"""),
        py(
            """display(table("11_dataset_readiness_summary.csv"))
display(table("11_required_feature_check.csv"))"""
        ),
        md(
            """The readiness table reports source-appearance rows and unique URLs separately by extraction scope. Most observations use captured excerpts/previews rather than guaranteed full webpage text."""
        ),
        md("""## 2. Leakage and feature-scope guardrail"""),
        py("display(table('11_leakage_and_scope_guardrail.csv'))"),
        md(
            """The model formulas exclude answer-derived measures, outcome-derived features, final enriched page type, headline rank/position controls, and domain citation-rate proxies. Prompt-page relevance uses prompt and page text only."""
        ),
        md("""## 3. Descriptive feature distributions"""),
        py(
            """display(table("11_feature_distribution_summary.csv"))
display(table("11_feature_correlation_with_has_table.csv"))
show_plot("11_feature_distributions.html")
show_plot("11_cited_rate_by_feature_quartile.html")
show_plot("11_has_table_feature_boxplots.html")"""
        ),
        md(
            """Quartile cited rates and correlations are unadjusted descriptions. For excerpt-based features, zero means “not observed in captured text,” not proof of absence from the full page."""
        ),
        md("""## 4. Modeling helper functions"""),
        py(
            """helpers = [
    fit_lpm,
    tidy_results,
    focal_terms_only,
    compare_covariance_types,
    save_model_result,
    make_forest_plot,
    make_has_table_path_plot,
]
display(pd.DataFrame({
    "helper": [helper.__name__ for helper in helpers],
    "signature": [str(inspect.signature(helper)) for helper in helpers],
}))"""
        ),
        md(
            """Models are OLS Linear Probability Models. Coefficient tables retain HC3, prompt-clustered, URL-clustered, and feasible two-way prompt-by-URL clustered inference."""
        ),
        md("""## 5. Baseline replication from notebook 09"""),
        py("display(focal('B0_notebook09_baseline_replication.csv'))"),
        md(
            """B0 repeats notebook 09's preferred structural specification using the expanded notebook 10 dataset. The added columns do not change the model sample."""
        ),
        md("""## 6. One-feature writing/factual screening with prompt fixed effects"""),
        py(
            """display(focal("F_one_feature_prompt_fe_results.csv"))
show_plot("F_one_feature_prompt_fe_forest.html")"""
        ),
        md("""These are screening associations only."""),
        md("""## 7. Joint writing/factual feature models"""),
        py(
            """display(focal("W_joint_writing_factual_results.csv"))
show_plot("W_joint_writing_factual_forest.html")"""
        ),
        md(
            """Interpret direction, size, uncertainty, and stability. The coefficients do not identify causal effects of modifying page content."""
        ),
        md("""## 8. Has-table proxy / attenuation test"""),
        py(
            """display(focal("T_has_table_proxy_ladder.csv"))
display(table("T_has_table_coefficient_path_summary.csv"))
show_plot("T_has_table_coefficient_path.html")"""
        ),
        md(
            """The T0-T4 ladder is a descriptive coefficient path and proxy/attenuation pattern. It is suggestive of omitted structure, not mediation."""
        ),
        md("""## 9. Domain-FE robustness"""),
        py("display(focal('D_domain_fe_writing_factual_results.csv'))"),
        md(
            """Domain fixed effects use domains with at least two unique URLs. Attenuation here is consistent with domain, publisher, or template confounding."""
        ),
        md("""## 10. Gemini taxonomy sensitivity"""),
        py("display(focal('P_page_function_sensitivity_results.csv'))"),
        md(
            """The primary taxonomy sensitivity uses Gemini page-function family and source/site type. Because Gemini may use scraped body content, this is not the headline writing model. The same table retains a separately named rule-v2 URL-seed robustness comparison."""
        ),
        md("""## 11. Strong-content and text-scope sensitivity"""),
        py(
            """display(table("S_text_scope_content_strength_support.csv"))
display(focal("S_text_scope_content_strength_sensitivity.csv"))"""
        ),
        md(
            """`content_strength` is extraction quality, not writing quality. Full-text and excerpt-only estimates test whether captured-text scope materially changes the associations."""
        ),
        md("""## 12. Outlier and distribution sensitivity"""),
        py(
            """display(table("11_outlier_distribution_audit.csv"))
display(focal("O_outlier_sensitivity_writing_factual_results.csv"))"""
        ),
        md(
            """The sensitivity removes each pre-specified p99 tail and repeats the joint equation with p99-winsorized main scores."""
        ),
        md("""## 13. Standard-error robustness comparison"""),
        py(
            """display(table("SE_focal_term_covariance_comparison.csv"))
show_plot("SE_focal_term_covariance_forest.html")"""
        ),
        md(
            """Two-way clustered inference is retained when feasible. Negative diagonal variance is flagged, and interpretation does not rely on the unavailable two-way standard error alone."""
        ),
        md("""## 14. Robustness classification"""),
        py("display(table('11_writing_factual_robustness_classification.csv'))"),
        md(
            """Classification combines specification direction, domain sensitivity, extraction scope, outliers, and covariance robustness. It summarizes stability, not causal credibility."""
        ),
        md("""## 15. Minimum reporting table"""),
        py("display(table('11_minimum_reporting_table.csv'))"),
        md(
            """This is the preferred compact table for AI-assisted interpretation and final content write-up."""
        ),
        md("""## 16. Final report"""),
        py(
            """display(Markdown(
    (REPORTS / "11_writing_factual_density_econometrics_report.md").read_text()
))"""
        ),
        md("""## 17. Executive summary"""),
        py(
            """display(Markdown(
    (REPORTS / "11_writing_factual_density_econometrics_executive_summary.md").read_text()
))"""
        ),
        md("""## 18. Run manifest"""),
        py(
            """manifest = json.loads(
    (REPORTS / "11_writing_factual_density_econometrics_manifest.json").read_text()
)
display(pd.DataFrame({
    "field": [
        "row_count", "url_count", "prompt_count", "domain_count", "cited_rate",
        "has_table_proxy_path_pattern", "has_table_precision_pattern", "final_status"
    ],
    "value": [
        manifest["row_count"], manifest["url_count"], manifest["prompt_count"],
        manifest["domain_count"], manifest["cited_rate"],
        manifest["has_table_proxy_path_pattern"],
        manifest["has_table_precision_pattern"],
        manifest["final_status"],
    ],
}))"""
        ),
        md("""## Final terminal summary"""),
        py(
            """print(f"rows: {result['rows']:,}")
print(f"unique URLs: {result['unique_urls']:,}")
print(f"unique prompts: {result['unique_prompts']:,}")
print(f"unique domains: {result['unique_domains']:,}")
print(f"cited rows: {result['cited_rows']:,}")
print(f"cited rate: {result['cited_rate']:.2%}")
print(f"excerpt-only rows: {result['excerpt_only_rows']:,}")
print(f"full-text-equivalent rows: {result['full_text_rows']:,}")
print(f"has_table path: {result['has_table_proxy_path_pattern']}")
print(f"has_table precision: {result['has_table_precision_pattern']}")
print(f"leakage check passed: {result['leakage_check_passed']}")
print(f"minimum reporting table: {result['minimum_reporting_table']}")
print(f"report: {result['report']}")
print(f"final status: {result['final_status']}")"""
        ),
    ]
    notebook.metadata.kernelspec = {
        "display_name": "CiteScope Plotly (.venv)",
        "language": "python",
        "name": "citescope-v4-plotly",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.14"}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
