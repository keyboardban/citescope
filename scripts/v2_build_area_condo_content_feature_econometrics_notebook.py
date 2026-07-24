#!/usr/bin/env python3
"""Build notebook 09 for area-condo content-feature econometrics."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/09_area_condo_content_feature_econometrics.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# 09 Area Condo Content Feature Econometrics

This notebook estimates descriptive content-feature Linear Probability Models for surfaced sources in the **area-condo / SCOPE-relevant nonbranded audit**.

- The model estimates associations with citation probability among surfaced sources.
- The analysis is conditional on sources already being surfaced.
- It does not estimate causal effects of changing a page.
- It does not represent all webpages on the internet.

The outcome is `cited`, and the unit of analysis is one surfaced source appearance."""
        ),
        md("""## 1. Load data and verify package readiness"""),
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

from src.econometrics_eda_v2.content_feature_econometrics import (
    add_cluster_counts,
    fit_lpm,
    make_coefficient_forest,
    make_prediction_contrast,
    run_content_feature_econometrics,
    run_model_and_save,
    tidy_lpm_result,
)
from src.econometrics_eda_v2.paths import topic_output_dir

PACKAGE = topic_output_dir() / "content_econometrics_ai_package"
TABLES = PACKAGE / "tables/09_content_feature_econometrics"
INTERACTIVE = PACKAGE / "figures/09_content_feature_econometrics/interactive"
REPORTS = PACKAGE / "reports/09_content_feature_econometrics"

run_summary = run_content_feature_econometrics(PACKAGE)
display(pd.DataFrame([run_summary]).T.rename(columns={0: "value"}))

def table(name, n=None):
    frame = pd.read_csv(TABLES / name, low_memory=False)
    return frame if n is None else frame.head(n)

def show_plot(name):
    figure = pio.read_json(INTERACTIVE / name.replace(".html", ".plotly.json"))
    display(figure)

display(table("09_dataset_readiness_summary.csv"))
display(table("09_required_column_check.csv"))"""
        ),
        md(
            """The expected measurable sample is 5,264 source appearances, 2,600 normalized URLs, 498 prompts, 541 domains, and 1,708 cited rows. Count differences are flagged but do not stop execution unless the outcome or required model columns are invalid."""
        ),
        md("""## 2. Leakage guardrail check"""),
        py("display(table('09_leakage_guardrail_runtime_check.csv'))"),
        md(
            """Every focal formula is scanned before fitting. Answer-derived similarity, answer overlap, outcome-derived labels, source position or observed rank, source origin/group, and domain citation-rate proxies are forbidden. A formula match raises a hard error."""
        ),
        md("""## 3. Descriptive overview / M0"""),
        py(
            """display(table("M0_descriptive_cited_rate_summary.csv"))
display(table("M0_feature_distribution_summary.csv"))
show_plot("M0_cited_rate_by_has_table.html")
show_plot("M0_cited_rate_by_heading_count_group.html")
show_plot("M0_cited_rate_by_link_count_group.html")
show_plot("M0_word_count_distribution.html")
show_plot("M0_word_length_binned_cited_rate.html")"""
        ),
        md(
            """All M0 results are unadjusted descriptive associations only. Missing scraped content is not treated as feature absence."""
        ),
        md("""## 4. Modeling helper functions"""),
        py(
            """helpers = [
    fit_lpm,
    tidy_lpm_result,
    add_cluster_counts,
    run_model_and_save,
    make_coefficient_forest,
    make_prediction_contrast,
]
display(pd.DataFrame({
    "helper": [helper.__name__ for helper in helpers],
    "signature": [str(inspect.signature(helper)) for helper in helpers],
}))"""
        ),
        md(
            """The model is an OLS Linear Probability Model. Coefficients are reported in probability points and percentage points. Tables retain HC3, prompt-clustered, URL-clustered, and two-way prompt/URL clustered inference when feasible."""
        ),
        md("""## 5. M1: One feature at a time with prompt fixed effects"""),
        py(
            """m1 = table("M1_one_feature_prompt_fe_results.csv")
display(m1[m1["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength|low_link_count",
    regex=True,
    na=False,
)])
show_plot("M1_focal_feature_forest.html")"""
        ),
        md("""M1 shows prompt-adjusted single-feature associations and is interpreted before adding the joint controls."""),
        md("""## 6. M2: Preferred joint structural-content LPM"""),
        py(
            """m2 = table("M2_preferred_joint_lpm_results.csv")
display(m2[m2["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength",
    regex=True,
    na=False,
)])
show_plot("M2_preferred_joint_lpm_forest.html")"""
        ),
        md(
            """`log2_word_count_plus1` approximates the association from doubling page length. `has_table` compares pages with and without detected tables. Heading and link groups allow nonlinear structural comparisons. `content_strength` is extraction-quality control, not writing quality."""
        ),
        md("""## 7. M3: Domain fixed-effects robustness"""),
        py(
            """m3 = table("M3_domain_fe_results.csv")
display(m3[m3["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength",
    regex=True,
    na=False,
)])"""
        ),
        md(
            """M3 is robustness, not the headline. It filters to domains with at least two unique URLs, then adds source-root-domain fixed effects to absorb stable publisher and site-template factors."""
        ),
        md("""## 8. M4: Gemini taxonomy sensitivity"""),
        py(
            """m4 = table("M4_gemini_taxonomy_sensitivity_results.csv")
display(m4[m4["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength|gemini",
    regex=True,
    na=False,
)])"""
        ),
        md(
            """This sensitivity uses the versioned Gemini page-function family and source/site type. Since Gemini can use scraped page content, M4 remains a sensitivity model; M1/M2 are still the primary content results. The older rule-v2 URL-seed model is retained in `M4R_rule_v2_taxonomy_robustness_results.csv` for comparison only."""
        ),
        md("""## 9. M5: Strong-content sensitivity"""),
        py(
            """m5 = table("M5_strong_content_sensitivity_results.csv")
display(m5[m5["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength",
    regex=True,
    na=False,
)])"""
        ),
        md(
            """M5 repeats M2, M3, and M4 among pages with strong extraction quality. This is mandatory before interpretation. `content_strength` is an extraction-quality classification, not a measure of writing quality."""
        ),
        md("""## 10. M6: Availability and missingness sensitivity"""),
        py("display(table('M6_measurable_selection_audit.csv'))"),
        md(
            """The measurable-content model sample is selected from all surfaced sources. Failed, weak, or unavailable extraction is retained as a missingness and selection issue, not recoded as zero content."""
        ),
        md("""## 11. M7: Logit average marginal effects cross-check"""),
        py("display(table('M7_logit_ame_crosscheck_results.csv'))"),
        md(
            """Prompt-fixed-effect logit is pre-screened for separation. When prompt groups have no within-prompt outcome variation, the notebook reports that limitation and uses a simplified logit with intent/area controls only as an AME cross-check. It does not replace M2."""
        ),
        md("""## 12. M8: Prompt-page relevance without answer text"""),
        py("display(table('M8_prompt_page_relevance_sensitivity_results.csv'))"),
        md(
            """Only prompt-page relevance computed from prompt text and page text would be allowed. Answer similarity, answer overlap, or any answer-derived measure remains forbidden."""
        ),
        md("""## 13. M9: Limited intent interactions"""),
        py(
            """display(table("M9_intent_interaction_cell_support.csv"))
display(table("M9_intent_interaction_sensitivity_results.csv", 80))"""
        ),
        md(
            """Interactions are sensitivity-only and are skipped when any required cell has fewer than 20 rows, fewer than 5 cited rows, or fewer than 5 more-only rows."""
        ),
        md("""## 14. M10: Outlier and winsorized sensitivity"""),
        py(
            """m10 = table("M10_outlier_winsorized_sensitivity_results.csv")
display(m10[m10["term"].str.contains(
    "log2_word_count|has_table|heading_count_group|link_count_group|content_strength",
    regex=True,
    na=False,
)])"""
        ),
        md(
            """M10 is mandatory before interpretation because page length and link counts have extreme tails. It removes each p99 tail separately and also uses the winsorized page-length transform."""
        ),
        md("""## 15. Robustness comparison"""),
        py(
            """display(table("09_focal_feature_robustness_comparison.csv"))
show_plot("09_focal_feature_robustness_forest.html")"""
        ),
        md("""The forest is a specification comparison, not a ranking of feature importance."""),
        md("""## 16. Actionable predicted probability contrasts"""),
        py("display(table('09_actionable_predicted_probability_contrasts.csv'))"),
        md(
            """These are model-implied contrasts holding the observed covariate distribution fixed. They are not causal effects and should not be read as promises from webpage rewriting."""
        ),
        md("""## 17. Minimum reporting table"""),
        py("display(table('09_minimum_reporting_table.csv'))"),
        md(
            """This table combines unadjusted, prompt-FE, preferred joint, domain-FE, Gemini-taxonomy, strong-content, logit AME, and model-implied contrast evidence for each focal feature."""
        ),
        md("""## 18. Final interpretation report"""),
        py("display(Markdown((REPORTS / '09_content_feature_econometrics_report.md').read_text()))"),
        md("""## 19. Model run manifest"""),
        py(
            """manifest = json.loads((REPORTS / "09_model_run_manifest.json").read_text())
display(pd.DataFrame({
    "field": [
        "timestamp",
        "number_of_fitted_models",
        "leakage_check_passed",
        "final_notebook_status",
    ],
    "value": [
        manifest["timestamp"],
        manifest["number_of_fitted_models"],
        manifest["leakage_check_passed"],
        manifest["final_notebook_status"],
    ],
}))
display(pd.DataFrame.from_dict(manifest["model_completion"], orient="index", columns=["completed"]))"""
        ),
        md("""## 20. Final notebook terminal summary"""),
        py(
            """print(f"input dataset path: {run_summary['input_dataset_path']}")
print(f"rows: {run_summary['rows']:,}")
print(f"unique URLs: {run_summary['unique_urls']:,}")
print(f"unique prompts: {run_summary['unique_prompts']:,}")
print(f"cited rows: {run_summary['cited_rows']:,}")
print(f"cited rate: {run_summary['cited_rate']:.2%}")
print(f"number of fitted models: {run_summary['number_of_fitted_models']}")
print(f"leakage check passed: {run_summary['leakage_check_passed']}")
for model in ("M2", "M3", "M4", "M5", "M7", "M10"):
    print(f"{model} completed: {run_summary[f'{model}_completed']}")
print(f"minimum reporting table: {run_summary['minimum_reporting_table']}")
print(f"report: {run_summary['report']}")
print(f"final status: {run_summary['final_status']}")"""
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
