#!/usr/bin/env python3
"""Build the notebook 09 interpretation-patch working copy."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/09_area_condo_content_feature_econometrics_v2_interpretation_patch.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# 09 Area Condo Content Feature Econometrics: V2 Interpretation Patch

This notebook patches the robustness reporting of notebook 09 without changing the estimand or refitting the original model ladder.

- Outcome: `cited`
- Unit: one surfaced source appearance
- Estimand: citation probability conditional on a source already being surfaced
- Interpretation: observational association, not causal and not web-wide

The patch does not use answer-derived variables, does not introduce final enriched page type as a main control, and does not overwrite raw inputs or original model outputs."""
        ),
        py(
            """from pathlib import Path
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

from src.econometrics_eda_v2.content_feature_interpretation_patch import (
    run_content_feature_interpretation_patch,
)
from src.econometrics_eda_v2.paths import topic_output_dir

PACKAGE = topic_output_dir() / "content_econometrics_ai_package"
TABLES = PACKAGE / "tables/09_content_feature_econometrics/interp_patch"
FIGURES = PACKAGE / "figures/09_content_feature_econometrics/interp_patch"
REPORTS = PACKAGE / "reports/09_content_feature_econometrics/interp_patch"

result = run_content_feature_interpretation_patch(PACKAGE)
display(pd.DataFrame([result]).T.rename(columns={0: "value"}))

def table(name, n=None):
    frame = pd.read_csv(TABLES / name, low_memory=False)
    return frame if n is None else frame.head(n)

def show_plot(name):
    display(pio.read_json(FIGURES / name.replace(".html", ".plotly.json")))"""
        ),
        md("""## 1. Focal-term standard-error comparison"""),
        py(
            """display(table("focal_term_se_comparison.csv"))
display(table("focal_term_se_stability_summary.csv"))
show_plot("focal_term_se_comparison_forest.html")"""
        ),
        md(
            """The same coefficient is compared under HC3, prompt-cluster, URL-cluster, and two-way prompt-by-URL clustered inference. A result is not treated as definitive when interval conclusions depend on one covariance estimator."""
        ),
        md("""## 2. Two-way cluster warning audit"""),
        py("display(table('two_way_cluster_warning_audit.csv'))"),
        md(
            """Two-way clustered covariance can be unstable in high-dimensional fixed-effect models with repeated prompts and URLs. Negative diagonal variances indicate that some reported SEs are not valid for affected terms. Therefore, focal content estimates should be checked against HC3, prompt-cluster, and URL-cluster alternatives."""
        ),
        md("""## 3. Robustness classification"""),
        py("display(table('focal_feature_robustness_classification.csv'))"),
        md(
            """The classification is intentionally conservative:

- `suggestive`: direction is informative, but uncertainty or attenuation prevents a definitive statement.
- `unstable`: important specifications change sign, magnitude, or interval conclusion.
- `descriptive_only`: the feature is a diagnostic/control rather than substantive writing quality."""
        ),
        md("""## 4. Domain-FE attenuation"""),
        py(
            """display(table("domain_fe_attenuation_summary.csv"))
show_plot("domain_fe_attenuation_focal_terms.html")"""
        ),
        md(
            """Heading-count categories show large negative associations in prompt-FE models, but these estimates attenuate strongly under domain fixed effects. This suggests domain/template or page-function differences may explain much of the pattern."""
        ),
        md("""## 5. Outlier sensitivity"""),
        py("display(table('outlier_sensitivity_focal_terms.csv'))"),
        md(
            """Page length is sensitive to extreme word-count tails. The preferred estimate is small and imprecise, while removing the top 1% word-count tail changes its magnitude and interval conclusion."""
        ),
        md("""## 6. Revised interpretation report"""),
        py(
            """display(Markdown(
    (REPORTS / "09_content_feature_econometrics_report_v2_interpretation_patch.md").read_text()
))"""
        ),
        md("""## 7. Patched minimum reporting table"""),
        py("display(table('09_minimum_reporting_table_v2_interpretation_patch.csv'))"),
        md(
            """The final interpretation buckets are `suggestive_positive`, `domain_template_confounded`, `unstable_diagnostic`, `extraction_quality_control`, and `insufficient_support`."""
        ),
        md("""## 8. Executive summary"""),
        py(
            """display(Markdown(
    (REPORTS / "09_content_feature_econometrics_executive_summary_v2.md").read_text()
))"""
        ),
        md("""## 9. Next feature layer"""),
        py("display(table('next_feature_layer_priority_plan.csv'))"),
        md(
            """All proposed next-layer features use page text, page structure, links, structured data, or prompt text only. Answer text and citation outcomes must not define the features."""
        ),
        md("""## 10. Final status"""),
        py(
            """print(f"number of focal terms checked: {result['number_of_focal_terms_checked']}")
print(f"number of models included in SE comparison: {result['number_of_models_included_in_se_comparison']}")
print(f"number of two-way cluster warnings: {result['number_of_two_way_cluster_warnings']}")
print(f"number of features classified as suggestive: {result['number_of_features_classified_as_suggestive']}")
print(
    "number of features classified as domain/template-confounded: "
    f"{result['number_of_features_classified_as_domain_template_confounded']}"
)
print(
    "number of features classified as unstable/diagnostic: "
    f"{result['number_of_features_classified_as_unstable_diagnostic']}"
)
print(f"path to revised report: {result['revised_report']}")
print(f"path to revised minimum reporting table: {result['revised_minimum_reporting_table']}")
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
