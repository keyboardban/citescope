#!/usr/bin/env python3
"""Build notebook 10 for writing and factual-density features."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/10_area_condo_writing_factual_density_feature_layer.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# 10 Area Condo Writing and Factual-Density Feature Layer

This notebook builds the next leakage-safe content feature layer to test whether table presence is proxying for factual specificity, numeric density, project/location detail, and prompt-page relevance.

Notebook 09 found `has_table` as the most suggestive structural signal. Table presence is treated as an observational association, not a causal mechanism. This notebook creates deterministic explanatory features and an econometric-ready v2 dataset for notebook 11.

The estimand remains `P(cited = 1 | source surfaced in this audit)`. Results are conditional associations among surfaced sources, not causal effects and not web-wide citation likelihood."""
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

from src.econometrics_eda_v2.writing_factual_density_features import (
    run_writing_factual_density_feature_layer,
)
from src.econometrics_eda_v2.paths import topic_output_dir

PACKAGE = topic_output_dir() / "content_econometrics_ai_package"
TABLES = PACKAGE / "tables/10_writing_factual_density_features"
FIGURES = PACKAGE / "figures/10_writing_factual_density_features"
REPORTS = PACKAGE / "reports/10_writing_factual_density_features"

result = run_writing_factual_density_feature_layer(PACKAGE)
display(pd.DataFrame([result]).T.rename(columns={0: "value"}))

def table(name, n=None):
    frame = pd.read_csv(TABLES / name, low_memory=False)
    return frame if n is None else frame.head(n)

def show_plot(name):
    display(pio.read_json(FIGURES / name.replace(".html", ".plotly.json")))"""
        ),
        md("""## 1. Load data and audit available text fields"""),
        py(
            """display(table("available_text_field_audit.csv"))
display(table("url_text_assembly_summary.csv"))
show_plot("feature_extraction_text_scope.html")"""
        ),
        md(
            """The package normally provides a title, a 1,200-character excerpt, and a 3,000-character preview rather than guaranteed full body text. `excerpt_only`, `title_description_only`, and `no_text` remain explicit extraction scopes."""
        ),
        md("""## 2. Build base text assembly"""),
        py(
            """assembly = table("url_text_assembly_audit.csv")
display(assembly[[
    "normalized_url", "url_title", "url_text_length_chars", "url_text_length_words",
    "full_page_text_available", "limited_excerpt_only", "text_source_used",
    "feature_extraction_text_scope"
]].head(30))"""
        ),
        md(
            """A zero extracted from an excerpt means the pattern was not observed in the captured excerpt. It does not establish absence from the full webpage."""
        ),
        md("""## 3. Structural writing features"""),
        py(
            """display(table("url_writing_structure_features.csv", 30))
display(table("content_lpm_with_writing_structure_features.csv", 10))"""
        ),
        md(
            """Sentence, paragraph, list, FAQ, question, and opening-summary features are deterministic. List and paragraph diagnostics are lower-reliability when crawler normalization has flattened line structure."""
        ),
        md("""## 4. Factual and numeric-density features"""),
        py(
            """display(table("url_factual_numeric_features.csv", 30))
display(table("content_lpm_with_factual_numeric_features.csv", 10))"""
        ),
        md(
            """Price, measurement, unit-size, percentage, year, and range features use bilingual regex rules. Composite weights are pre-specified and were not selected using `cited`."""
        ),
        md("""## 5. Location, transit, and neighborhood features"""),
        py("display(table('url_location_transit_features.csv', 30))"),
        md(
            """The location layer counts named neighborhoods, landmarks, BTS/MRT signals, distances, and walking-time details observed in the captured page text."""
        ),
        md("""## 6. Amenity and project-detail features"""),
        py("display(table('url_amenity_project_features.csv', 30))"),
        md(
            """Amenity/project entity rules support Thai and English but remain dictionary-based. The manual review sample should be checked before these enter notebook 11."""
        ),
        md("""## 7. External evidence and citation-like webpage features"""),
        py("display(table('url_external_evidence_features.csv', 30))"),
        md(
            """These features use webpage text and visible page-level links only. They do not use citations from an AI answer. Link counts may be incomplete when the compact text preview omits href destinations."""
        ),
        md("""## 8. Prompt-page similarity without answer text"""),
        py(
            """display(table("source_appearance_prompt_page_relevance_features.csv", 30))
display(table("prompt_page_relevance_merge_audit.csv"))"""
        ),
        md(
            """Prompt-page relevance uses deterministic character TF-IDF and keyword matching between prompt text and page title/description/body preview. It never uses answer text or answer overlap."""
        ),
        md("""## 9. Composite feature groups"""),
        py(
            """display(table("writing_factual_feature_dictionary.csv"))
show_plot("10_composite_feature_distributions.html")"""
        ),
        md(
            """All component variables are retained. Composite formulas are documented and use fixed additive/log rules rather than outcome-tuned weights."""
        ),
        md("""## 10. Validation and manual evidence audit"""),
        py(
            """display(table("writing_factual_feature_validation_summary.csv"))
display(table("writing_factual_feature_manual_review_sample.csv"))"""
        ),
        md(
            """The review sample contains high factual-density, price/unit, location/transit, and prompt-page similarity rows plus a deterministic random sample. `cited` is included for review context only and was not used to construct features."""
        ),
        md("""## 11. Merge final feature dataset"""),
        py(
            """display(table("writing_factual_feature_leakage_check.csv"))
display(table("10_writing_factual_feature_merge_audit.csv"))"""
        ),
        md(
            """The final dataset preserves every original measurable row. Missing text remains missing; it is not converted to feature absence."""
        ),
        md("""## 12. First-pass econometric screening, not final claims"""),
        py(
            """screening = table("10_first_pass_feature_screening_lpm.csv")
display(screening)
show_plot("10_first_pass_feature_screening_forest.html")"""
        ),
        md(
            """These prompt-fixed-effect models screen one feature group at a time. They prioritize candidates for notebook 11 and do not produce final causal or substantive claims."""
        ),
        md("""### Has-table proxy test"""),
        py(
            """display(table("10_has_table_proxy_test.csv"))
display(table("10_has_table_proxy_attenuation_summary.csv"))
show_plot("10_has_table_proxy_attenuation.html")"""
        ),
        md(
            """T1, T2, and T3 ask whether the `has_table` coefficient attenuates after adding factual/detail and prompt-page relevance features. Attenuation is descriptive screening evidence, not proof of mediation."""
        ),
        md("""## 13. Feature-priority recommendations"""),
        py("display(table('10_feature_priority_for_11_econometrics.csv'))"),
        md("""Recommended buckets are `priority_main_candidate`, `sensitivity_candidate`, `diagnostic_only`, `needs_extraction_fix`, and `forbidden`."""),
        md("""## 14. Final report"""),
        py(
            """display(Markdown(
    (REPORTS / "10_writing_factual_density_feature_layer_report.md").read_text()
))"""
        ),
        md("""## 15. Final run manifest"""),
        py(
            """manifest = json.loads(
    (REPORTS / "10_writing_factual_density_feature_layer_manifest.json").read_text()
)
display(pd.DataFrame({
    "field": [
        "input_rows", "output_rows", "unique_urls", "unique_prompts",
        "leakage_status", "validation_status", "first_pass_model_status", "final_status"
    ],
    "value": [
        manifest["row_counts"]["input_rows"],
        manifest["row_counts"]["output_rows"],
        manifest["row_counts"]["unique_urls"],
        manifest["row_counts"]["unique_prompts"],
        manifest["leakage_status"],
        manifest["validation_status"],
        manifest["first_pass_model_status"],
        manifest["final_status"],
    ],
}))"""
        ),
        md("""## Final terminal summary"""),
        py(
            """print(f"input rows: {result['input_rows']:,}")
print(f"output rows: {result['output_rows']:,}")
print(f"unique URLs: {result['unique_urls']:,}")
print(f"URLs with usable text: {result['urls_with_usable_text']:,}")
print(f"excerpt-only URLs: {result['excerpt_only_urls']:,}")
print(f"features created: {result['features_created']}")
print(f"leakage check passed: {result['leakage_check_passed']}")
print(f"first-pass models completed: {result['first_pass_models_completed']}")
print(f"has_table T1 estimate: {result['has_table_T1_estimate_pp']:.2f} pp")
print(f"has_table T3 estimate: {result['has_table_T3_estimate_pp']:.2f} pp")
print(f"has_table T3 coefficient change: {result['has_table_T3_coefficient_change_pp']:+.2f} pp")
print(f"has_table T3 proxy pattern: {result['has_table_T3_proxy_pattern']}")
print(f"final dataset: {result['final_dataset']}")
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
