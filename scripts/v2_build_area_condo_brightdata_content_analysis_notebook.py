#!/usr/bin/env python3
"""Build the Bright Data Crawler content-analysis master notebook."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/07_area_condo_brightdata_content_analysis_master.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def py(text: str):
    return nbf.v4.new_code_cell(text.strip())


ANALYSIS = r"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

CODE_ROOT = Path.cwd().resolve()
if not (CODE_ROOT / "src").exists() and (CODE_ROOT.parent / "src").exists():
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.econometrics_eda_v2.paths import topic_output_dir

BASE = topic_output_dir()
SOURCE_OUT = BASE / "tables/area_condo_brightdata_content_pilot"
OUT = BASE / "tables/area_condo_brightdata_content_analysis"
FIG = BASE / "figures/area_condo_brightdata_content_analysis"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

detail = pd.read_csv(SOURCE_OUT / "brightdata_content_pilot_url_results.csv", low_memory=False)
summary = json.loads((SOURCE_OUT / "brightdata_content_pilot_summary.json").read_text())
detail["cited"] = pd.to_numeric(detail["cited"], errors="coerce").fillna(0).astype(int)
detail["content_chars"] = pd.to_numeric(detail["content_chars"], errors="coerce").fillna(0)
for column in ["word_count", "heading_count", "table_count", "link_count"]:
    detail[column] = pd.to_numeric(detail[column], errors="coerce").fillna(0)
detail["content_feature_available"] = detail["scrape_success"].fillna(False) & detail["content_chars"].gt(0)
detail["log1p_word_count"] = np.log1p(detail["word_count"])
detail["word_count_group"] = pd.cut(detail["word_count"], [-1, 299, 999, 2999, np.inf], labels=["0-299", "300-999", "1,000-2,999", "3,000+"])
detail["heading_count_group"] = pd.cut(detail["heading_count"], [-1, 1, 6, 12, np.inf], labels=["0-1", "2-6", "7-12", "13+"])
detail["link_count_group"] = pd.cut(detail["link_count"], [-1, 9, 49, 99, np.inf], labels=["0-9", "10-49", "50-99", "100+"])
detail["table_count_group"] = pd.cut(detail["table_count"], [-1, 0, 1, np.inf], labels=["0", "1", "2+"])
overall_rate = detail["cited"].mean()

def ci(rate, n):
    if not n:
        return (np.nan, np.nan)
    z = 1.96
    denominator = 1 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2)) / denominator
    return max(0, centre - margin), min(1, centre + margin)

def rate_table(frame, feature, order=None):
    rows = []
    for value, group in frame.groupby(feature, dropna=False, observed=False):
        n = len(group)
        rate = group["cited"].mean()
        lo, hi = ci(rate, n)
        rows.append({"feature": feature, "level": str(value), "urls": n, "cited_urls": int(group["cited"].sum()), "cited_rate": rate, "ci_low": lo, "ci_high": hi, "difference_from_overall_pp": (rate - overall_rate) * 100, "sparse": n < 20 or int(group["cited"].sum()) < 5})
    result = pd.DataFrame(rows)
    if order:
        result["level"] = pd.Categorical(result["level"], categories=order, ordered=True)
        result = result.sort_values("level")
        result["level"] = result["level"].astype(str)
    return result

def point_plot(table, title, path):
    table = table.dropna(subset=["cited_rate"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(7, len(table) * 1.4), 4.8))
    x = np.arange(len(table))
    ax.errorbar(x, table["cited_rate"] * 100, yerr=[(table["cited_rate"] - table["ci_low"]) * 100, (table["ci_high"] - table["cited_rate"]) * 100], fmt="o-", color="#187b8d", capsize=4)
    ax.axhline(overall_rate * 100, color="#a64646", linestyle="--", linewidth=1.2, label="overall cited rate")
    for i, row in table.iterrows():
        ax.annotate(f"n={int(row['urls'])}", (i, row["cited_rate"] * 100), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, table["level"], rotation=28, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Cited rate (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / path, dpi=170)
    plt.close(fig)

coverage = pd.DataFrame([{
    "unique_urls": len(detail), "cited_urls": int(detail["cited"].sum()), "cited_rate": overall_rate,
    "crawler_attempted": int(detail.get("crawler_attempted", pd.Series(False, index=detail.index)).sum()),
    "scrape_success_urls": int(detail["scrape_success"].sum()), "content_feature_available_urls": int(detail["content_feature_available"].sum()),
    "strong_content_urls": int(detail["content_strength"].eq("strong").sum()), "empty_text_urls": int(detail["content_chars"].eq(0).sum()),
    "tracking_clean_request_urls": int(detail.get("tracking_parameters_removed", pd.Series(False, index=detail.index)).sum()),
}])
coverage.to_csv(OUT / "crawler_content_coverage_summary.csv", index=False)

strength = rate_table(detail, "content_strength", ["strong", "medium", "weak", "failed"])
quality = rate_table(detail, "content_quality_flag")
source_type = rate_table(detail, "source_type")
word = rate_table(detail, "word_count_group", ["0-299", "300-999", "1,000-2,999", "3,000+"])
heading = rate_table(detail, "heading_count_group", ["0-1", "2-6", "7-12", "13+"])
links = rate_table(detail, "link_count_group", ["0-9", "10-49", "50-99", "100+"])
tables = rate_table(detail, "table_count_group", ["0", "1", "2+"])
for name, table in {"content_strength_cited_rate.csv": strength, "content_quality_cited_rate.csv": quality, "source_type_cited_rate.csv": source_type, "word_count_group_cited_rate.csv": word, "heading_count_group_cited_rate.csv": heading, "link_count_group_cited_rate.csv": links, "table_count_group_cited_rate.csv": tables}.items():
    table.to_csv(OUT / name, index=False)

domain = detail.groupby("source_root_domain", dropna=False).agg(urls=("normalized_url", "size"), cited_urls=("cited", "sum"), scrape_success_rate=("scrape_success", "mean"), content_available_rate=("content_feature_available", "mean"), strong_content_rate=("content_strength", lambda x: x.eq("strong").mean()), median_word_count=("word_count", "median"), median_heading_count=("heading_count", "median"), median_link_count=("link_count", "median")).reset_index()
domain["cited_rate"] = domain["cited_urls"] / domain["urls"]
domain.sort_values(["urls", "cited_urls"], ascending=False).to_csv(OUT / "domain_content_availability_summary.csv", index=False)
detail.sort_values(["scrape_success", "content_strength", "cited_rows"], ascending=[True, True, False]).to_csv(OUT / "url_level_content_review.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.6))
counts = detail["content_strength"].value_counts().reindex(["strong", "medium", "weak", "failed"]).fillna(0)
ax.bar(counts.index, counts.values, color=["#187b8d", "#c79032", "#b25a3c", "#a64646"])
ax.set_ylabel("URLs"); ax.set_title("Crawler content strength coverage")
for i, value in enumerate(counts.values): ax.text(i, value + .25, str(int(value)), ha="center")
fig.tight_layout(); fig.savefig(FIG / "crawler_content_strength_coverage.png", dpi=170); plt.close(fig)

point_plot(strength, "Cited rate by content strength", "cited_rate_by_content_strength.png")
point_plot(word, "Cited rate by extracted word-count group", "cited_rate_by_word_count_group.png")
point_plot(heading, "Cited rate by heading-count group", "cited_rate_by_heading_count_group.png")
point_plot(links, "Cited rate by link-count group", "cited_rate_by_link_count_group.png")
point_plot(tables, "Cited rate by table-count group", "cited_rate_by_table_count_group.png")

fig, ax = plt.subplots(figsize=(7, 4.8))
box = [detail.loc[detail["cited"].eq(value), "log1p_word_count"].dropna() for value in [0, 1]]
ax.boxplot(box, tick_labels=["More-only", "Cited"], showfliers=False)
ax.set_ylabel("log(1 + extracted words)"); ax.set_title("Extracted content length by cited status")
fig.tight_layout(); fig.savefig(FIG / "distribution_log1p_word_count_by_cited.png", dpi=170); plt.close(fig)

top_domain = domain.sort_values("urls", ascending=False).head(15).sort_values("strong_content_rate")
fig, ax = plt.subplots(figsize=(9, max(5, len(top_domain) * .38)))
ax.barh(top_domain["source_root_domain"], top_domain["strong_content_rate"] * 100, color="#187b8d")
ax.set_xlabel("Strong content rate (%)"); ax.set_title("Crawler content availability by top domain")
fig.tight_layout(); fig.savefig(FIG / "strong_content_rate_by_top_domain.png", dpi=170); plt.close(fig)

report = f'''# Bright Data Crawler Content Analysis\n\nCurrent data: {len(detail):,} unique URLs, {int(detail.scrape_success.sum()):,} successful Crawler extractions, and {int(detail.content_strength.eq("strong").sum()):,} strong-content pages.\n\nThe content measurements are descriptive and are conditional on successful extraction. They do not show why ChatGPT cited or did not cite a source, and no answer text, source position, rank, or source origin is used as a predictor.\n\nRaw content length can include navigation and linked-page text. Use the raw-cache evidence and page excerpts for manual QA before treating structural features as model-ready.\n'''
(OUT / "crawler_content_analysis_report.md").write_text(report, encoding="utf-8")
display(coverage)
"""


def build() -> Path:
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""# Area Condo Bright Data Crawler Content Analysis Master

This notebook reads the latest completed Bright Data Crawler result table and builds a full descriptive content-analysis surface. Re-run it after a resumed scrape to refresh every table and figure from the current cache.

All results are observational. Content measurements are only available for pages successfully extracted by Bright Data; they are not evidence of an AI retrieval mechanism or causal effects."""),
        py(ANALYSIS),
        md("""## 1. Coverage and content availability

The first check is whether the Crawler returned usable content. Extraction availability is a missingness diagnostic, not a content feature that may be silently treated as zero."""),
        py("display(pd.read_csv(OUT / 'crawler_content_coverage_summary.csv'))\nshow = lambda name: display(Image(filename=str(FIG / name)))\nshow('crawler_content_strength_coverage.png')"),
        md("""## 2. Content-strength and quality diagnostics

These rates are unadjusted descriptive comparisons. Sparse categories are marked in the table and should not be overinterpreted."""),
        py("display(pd.read_csv(OUT / 'content_strength_cited_rate.csv'))\ndisplay(pd.read_csv(OUT / 'content_quality_cited_rate.csv'))\nshow('cited_rate_by_content_strength.png')"),
        md("""## 3. Extracted content length and structure

Word, heading, link, and table counts describe the extracted response only. They can include page chrome, so use them as diagnostics or clearly labeled content-subset controls rather than causal claims."""),
        py("display(pd.read_csv(OUT / 'word_count_group_cited_rate.csv'))\ndisplay(pd.read_csv(OUT / 'heading_count_group_cited_rate.csv'))\ndisplay(pd.read_csv(OUT / 'link_count_group_cited_rate.csv'))\ndisplay(pd.read_csv(OUT / 'table_count_group_cited_rate.csv'))\nshow('cited_rate_by_word_count_group.png')\nshow('cited_rate_by_heading_count_group.png')\nshow('cited_rate_by_link_count_group.png')\nshow('cited_rate_by_table_count_group.png')\nshow('distribution_log1p_word_count_by_cited.png')"),
        md("""## 4. Source-type and domain diagnostics

Domain patterns show where the Crawler has robust content coverage. They do not establish why a domain was cited."""),
        py("display(pd.read_csv(OUT / 'source_type_cited_rate.csv'))\ndisplay(pd.read_csv(OUT / 'domain_content_availability_summary.csv').head(30))\nshow('strong_content_rate_by_top_domain.png')"),
        md("""## 5. URL normalization and manual review

The original source URL is retained for auditability. The final request URL is tracker-cleaned. Review source titles, excerpts, and raw-cache records before relying on any surprising feature result."""),
        py("display(pd.read_csv(SOURCE_OUT / 'tracking_parameter_request_audit.csv'))\ndisplay(pd.read_csv(OUT / 'url_level_content_review.csv').head(100))"),
        md("""## 6. Interpretation boundary

This notebook does not fit an LPM. It does not use answer text, answer similarity, source origin, observed rank, source position, or citation labels as predictors. For any future model, analyze raw content features only among `content_feature_available = true` pages and retain scrape availability separately."""),
        py("display(Markdown((OUT / 'crawler_content_analysis_report.md').read_text()))"),
    ]
    notebook.metadata["area_condo_brightdata_content_analysis"] = {"purpose": "Crawler content EDA; no causal claims and no LPM fit"}
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(notebook, TARGET)
    return TARGET


if __name__ == "__main__":
    print(build())
