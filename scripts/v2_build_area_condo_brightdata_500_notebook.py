#!/usr/bin/env python3
"""Build a no-scrape overview notebook for the 500-prompt Bright Data run."""
from __future__ import annotations
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/05_area_condo_nonbranded_brightdata_500_overview.ipynb"

def md(text: str): return nbf.v4.new_markdown_cell(text.strip())
def py(text: str): return nbf.v4.new_code_cell(text.strip())

def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""# Area Condo Non-branded: Bright Data 500-Prompt Overview

This notebook describes the observable sources surfaced in the supplied ChatGPT/Bright Data export. It compares **cited** sources with **more-only** sources, which were shown but not cited. It does not reconstruct a SERP, infer the model's internal retrieval set, scrape websites, or fit a regression model."""),
        py("""from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists(): ROOT = ROOT.parent
BASE = ROOT / 'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
OUT = BASE / 'tables/area_condo_brightdata_500'
FIG = BASE / 'figures/area_condo_brightdata_500'
def table(name): return pd.read_csv(OUT / name)
def show(name): display(Image(filename=str(FIG / name)))
summary = json.loads((OUT / 'run_summary.json').read_text())
display(pd.DataFrame([summary]))"""),
        md("""## 1. Run and manifest validation

The Bright Data JSON is a result export, not a prompt-input CSV. The manifest join attaches prompt IDs, topic, and intent without using answer text as a predictor."""),
        py("display(table('manifest_coverage_audit.csv'))\nrecords = table('area_condo_brightdata_500_records.csv')\ndisplay(records[['has_sources', 'source_rows', 'cited_sources']].describe())"),
        md("""## 2. Observable source-set overview

Each row is one observed source appearance. The same URL can appear for more than one prompt; URL counts should therefore not be read as independent observations."""),
        py("sources = table('area_condo_brightdata_500_normalized_sources.csv')\nprint(f\"Source appearances: {len(sources):,}\")\nprint(f\"Unique URLs: {sources['normalized_url'].nunique():,}\")\nprint(f\"Cited rate: {sources['cited'].mean():.1%}\")\ndisplay(sources.head(20))"),
        md("""## 3. Cited versus more-only origin

`citations` are the directly observed cited source field. `search_sources_more` and `search_sources` are observable non-cited source fields. `links_attached` is a parser fallback used only when no explicit citation exists for that record."""),
        py("display(table('source_origin_summary.csv'))"),
        md("""## 4. Intent diagnostics

Intent is a prompt-level descriptive grouping. Cited-rate differences here are unadjusted and may reflect different question types and source mixes."""),
        py("display(table('intent_summary.csv'))\nshow('cited_rate_by_intent.png')"),
        md("""## 5. Source-type diagnostics

Source type is a deterministic URL/domain heuristic. `unknown` is retained rather than guessed, and this layer does not require scraping."""),
        py("display(table('source_type_summary.csv'))\nshow('cited_rate_by_source_type.png')"),
        md("""## 6. Top observed domains

This is an exposure summary, not an inference that any domain was internally retrieved or preferred by ChatGPT."""),
        py("display(table('top_domain_summary.csv'))\nshow('cited_rate_by_top_domain.png')"),
        md("""## 7. What is and is not ready

This 500-prompt export is ready for no-scrape descriptive citation analysis and source-mix analysis. Website content, page-function taxonomy, and content-feature analyses require a separate scraping stage. Any later LPM must exclude answer text, answer similarity, source origin, position/rank, and outcome duplicates from its main predictor set."""),
    ]
    nb.metadata['area_condo_brightdata_500'] = {'purpose': 'no-scrape descriptive overview of latest Bright Data export'}
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbf.write(nb, TARGET)
    return TARGET

if __name__ == '__main__': print(build())
