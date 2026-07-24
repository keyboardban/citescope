#!/usr/bin/env python3
"""Build notebook 12 for HTML-first document-structure QA."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/12_area_condo_document_structure_features.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md(
            """# 12 Area Condo Document-Structure Features

This notebook audits general webpage structure extracted directly from captured HTML: full and main body text, tables, headings, paragraphs, lists, links, external domains, and schema.org metadata. Markdown is generated from the HTML for human inspection.

**This is a descriptive QA layer. It does not fit an LPM and does not alter notebooks 10 or 11.** All cited comparisons are unadjusted associations among surfaced source appearances, not causal effects or web-wide citation probabilities."""
        ),
        py(
            """from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Markdown, display

CODE_ROOT = Path.cwd().resolve()
if not (CODE_ROOT / "src").exists() and (CODE_ROOT.parent / "src").exists():
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.econometrics_eda_v2.paths import topic_output_dir

PACKAGE = topic_output_dir() / "content_econometrics_ai_package"
TABLES = PACKAGE / "tables/12_document_structure_features"
DATA = PACKAGE / "data/content_lpm_measurable_rows_with_document_structure_features.csv"

rows = pd.read_csv(DATA, low_memory=False)
urls = pd.read_csv(TABLES / "url_document_structure_features.csv", low_memory=False)
coverage = pd.read_csv(TABLES / "document_structure_coverage_summary.csv")
dictionary = pd.read_csv(TABLES / "document_structure_feature_dictionary.csv")
review = pd.read_csv(TABLES / "document_structure_manual_review_sample_100.csv", low_memory=False)
measurable = rows[rows["document_features_measurable"].fillna(False).astype(bool)].copy()
overall = measurable["cited"].mean()

print(f"source appearances: {len(rows):,}")
print(f"unique URLs: {urls['normalized_url'].nunique():,}")
print(f"structure-measurable URLs: {int(urls['document_features_measurable'].sum()):,}")
print(f"structure-measurable appearances: {len(measurable):,}")"""
        ),
        md("""## 1. Coverage and missingness"""),
        py(
            """display(coverage)

coverage_plot = coverage[coverage["value_type"].eq("rate")].copy()
coverage_plot["percent"] = coverage_plot["value"] * 100
fig = px.bar(coverage_plot, x="metric", y="percent", text="percent", title="Document-structure coverage")
fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(yaxis_title="URLs (%)", xaxis_title="", yaxis_range=[0, 105])
fig.show()"""
        ),
        md(
            """Missing HTML is retained as missing, not converted to structural zeros. `document_features_measurable` additionally requires a successful scrape with strong or medium extracted content."""
        ),
        md("""## 2. Feature dictionary"""),
        py("display(dictionary)"),
        md("""## 3. Tables and factual row structure"""),
        py(
            """table_features = [
    "html_table_count", "table_row_count", "table_column_max",
    "price_row_count", "unit_size_row_count", "comparison_row_count",
]
display(measurable.groupby("cited")[table_features].agg(["count", "mean", "median"]).round(2))

plot_data = measurable[["cited", *table_features]].copy()
plot_data["citation_status"] = plot_data["cited"].map({0: "More-only", 1: "Cited"})
long = plot_data.melt(id_vars="citation_status", value_vars=table_features, var_name="feature", value_name="value")
long["value_clipped_p99"] = long.groupby("feature")["value"].transform(lambda s: s.clip(upper=s.quantile(.99)))
fig = px.box(long, x="feature", y="value_clipped_p99", color="citation_status", points=False,
             title="Table and row features by cited status (p99 display cap)")
fig.update_layout(xaxis_title="", yaxis_title="Count", xaxis_tickangle=-30)
fig.show()"""
        ),
        md(
            """Price, unit-size, and comparison rows are deterministic text-pattern counts within HTML table rows. They are useful for QA and sensitivity analysis, but should not be treated as perfect semantic labels."""
        ),
        md("""## 4. Heading, paragraph, and list structure"""),
        py(
            """structure_features = [
    "dom_heading_count", "heading_max_depth", "heading_level_skip_count",
    "paragraph_count", "median_paragraph_words", "list_item_count", "max_list_depth",
]
display(measurable.groupby("cited")[structure_features].agg(["mean", "median"]).round(2))

scatter = measurable.drop_duplicates("normalized_url").copy()
scatter["citation_status"] = scatter["cited"].map({0: "More-only", 1: "Cited"})
fig = px.scatter(
    scatter,
    x="paragraph_count",
    y="dom_heading_count",
    size="main_content_word_count",
    color="citation_status",
    hover_data=["source_root_domain", "normalized_url", "list_item_count"],
    log_x=True,
    title="Heading and paragraph structure by URL",
)
fig.update_layout(xaxis_title="Paragraph count (log scale)", yaxis_title="DOM heading count")
fig.show()"""
        ),
        md("""## 5. Outbound links and external domains"""),
        py(
            """link_features = ["link_count_total", "internal_link_count", "outbound_link_count", "external_link_domain_count"]
display(measurable.groupby("cited")[link_features].agg(["mean", "median"]).round(2))

link_plot = measurable.drop_duplicates("normalized_url").copy()
link_plot["citation_status"] = link_plot["cited"].map({0: "More-only", 1: "Cited"})
fig = px.scatter(
    link_plot,
    x="outbound_link_count",
    y="external_link_domain_count",
    color="citation_status",
    hover_data=["source_root_domain", "normalized_url", "external_link_domains"],
    title="Outbound links and unique external domains",
)
fig.update_layout(xaxis_title="Outbound links", yaxis_title="Unique external domains")
fig.show()"""
        ),
        md("""## 6. Schema.org and FAQ structure"""),
        py(
            """binary_features = [
    "has_html_table", "has_heading_hierarchy", "has_paragraph_structure",
    "has_list_structure", "has_outbound_links", "has_jsonld", "has_faqpage_schema",
    "has_article_schema", "has_product_schema", "has_breadcrumb_schema",
]
summary = []
for feature in binary_features:
    for value, group in measurable.groupby(feature, dropna=False):
        summary.append({
            "feature": feature,
            "present": bool(value),
            "n_rows": len(group),
            "cited_rate": group["cited"].mean(),
            "difference_pp": 100 * (group["cited"].mean() - overall),
        })
binary_summary = pd.DataFrame(summary)
display(binary_summary.round(3))

present = binary_summary[binary_summary["present"]].sort_values("difference_pp")
fig = px.bar(present, x="difference_pp", y="feature", orientation="h", text="n_rows",
             title="Unadjusted difference from overall cited rate when structure is present")
fig.add_vline(x=0, line_dash="dash", line_color="gray")
fig.update_layout(xaxis_title="Difference from overall cited rate (percentage points)", yaxis_title="")
fig.show()"""
        ),
        md(
            """These differences are descriptive and can reflect prompt mix, domain, page function, scrape quality, and repeated URLs. They are not feature effects."""
        ),
        md("""## 7. Manual review and generated Markdown"""),
        py(
            """review_columns = [
    "normalized_url", "source_root_domain", "cited", "document_features_measurable",
    "html_table_count", "dom_heading_count", "paragraph_count", "list_item_count",
    "outbound_link_count", "schema_types", "main_content_preview", "generated_markdown_path",
]
display(review[[column for column in review_columns if column in review.columns]])

example_path = next((Path(value) for value in review["generated_markdown_path"].dropna() if Path(value).exists()), None)
if example_path:
    display(Markdown(f"### Generated Markdown example: `{example_path.name}`"))
    display(Markdown(example_path.read_text(encoding="utf-8")[:5000]))"""
        ),
        md("""## 8. Interpretation boundary

- Full body text and generated Markdown are audit evidence, not direct model predictors.
- Structural variables are interpretable only where `document_features_measurable = true`.
- Absence in failed or weak extractions is missingness, not a zero.
- URL-level content repeats across prompt-source appearances; future inference must account for prompt and URL/domain clustering.
- Existing notebook 11 results remain frozen. Any econometric use of these new features requires a separately specified model and renewed leakage, sparse-cell, outlier, and covariance checks."""),
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
