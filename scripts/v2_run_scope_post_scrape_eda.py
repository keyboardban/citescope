#!/usr/bin/env python3
"""Run post-scrape diagnostics and exploratory LPM sensitivities for SCOPE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.scope_post_scrape_eda import (
    availability_by,
    cited_vs_more_only_comparison,
    page_type_distribution,
    prepare_post_scrape_eda,
    run_exploratory_lpm,
    scrape_quality_summary,
    sensitivity_descriptive_summary,
    sensitivity_subsets,
    unknown_page_type_diagnostics,
)


TOPIC = "scope_condo_nonbranded"
OUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity" / TOPIC
TABLES = OUT / "tables"
PROCESSED = ROOT / "data/econometrics_v2/topics" / TOPIC / "processed"
DEFAULT_EDA = TABLES / "scope_condo_eda_ready_with_real_estate_taxonomy.csv"
DEFAULT_PARSE = PROCESSED / "apify_page_parse_rows.csv"


def _write_csv(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def _write_report(path: Path, df: pd.DataFrame, lpm_summary: pd.DataFrame) -> None:
    n = len(df)
    cited = int(df["is_cited"].sum())
    strong = float(df["content_strength"].eq("strong").mean()) if n else 0.0
    medium_or_strong = float(df["content_strength"].isin(["medium", "strong"]).mean()) if n else 0.0
    unknown = float(df["page_type_family_real_estate"].eq("unknown").mean()) if n else 0.0
    scraped_ok = float(df["scraped_ok"].mean()) if n else 0.0
    lpm_ok = lpm_summary[lpm_summary["model_status"].eq("ok")] if not lpm_summary.empty else pd.DataFrame()
    lines = [
        "# SCOPE Post-Scrape EDA Report",
        "",
        "## Scope",
        "This is an observational post-scrape EDA pass over source rows. It does not make claims about an AI system's hidden retrieval set or the reason any page was cited or not cited.",
        "",
        "## Data availability",
        f"- Source rows: {n}",
        f"- Cited rows: {cited}",
        f"- `scraped_ok` rate: {scraped_ok:.1%}",
        f"- Strong content rate: {strong:.1%}",
        f"- Medium-or-strong content rate: {medium_or_strong:.1%}",
        f"- Unknown broad page-type rate: {unknown:.1%}",
        "",
        "## Taxonomy",
        "`page_type_family_real_estate` is the broad EDA taxonomy: project, listing, developer, aggregator, blog/guide/news, review/article, directory/contact, social/video/forum, official/corporate, and unknown. `page_type_detail_real_estate` remains the diagnostic-level label. Unknown rows are retained.",
        "",
        "## LPM interpretation",
        f"- Exploratory LPMs successfully fit: {len(lpm_ok)} of {len(lpm_summary)} requested scenarios.",
        "- The dependent variable is `is_cited`. Models use page-type family, `scraped_ok`, content strength, domain family, and prompt fixed effects when estimable.",
        "- Answer text, source position, observed rank, and answer-derived similarity are excluded from the main model.",
        "- Coefficients are descriptive conditional associations, not causal effects or evidence of hidden retrieval behavior.",
        "",
        "## Missingness limitation",
        "Content-derived variables are interpretable only where `content_features_available = 1` (medium or strong measurable content). Scrape and content availability are reported as missingness patterns, not assumed to be random; use the `scraped_ok_only` and `strong_content_only` sensitivity outputs alongside all-row results.",
        "",
        "## Recommended reading order",
        "1. Review scrape quality and availability tables before interpreting content-related findings.",
        "2. Use broad page type in the main EDA/LPM and detailed type only for diagnostics.",
        "3. Compare the all-row and availability-restricted sensitivity models before drawing conclusions.",
    ]
    path.write_text("\n".join(lines), "utf-8")


def run(eda_path: Path = DEFAULT_EDA, parse_path: Path = DEFAULT_PARSE, output_dir: Path = TABLES) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(eda_path, low_memory=False)
    parse = pd.read_csv(parse_path, low_memory=False) if parse_path.exists() else pd.DataFrame()
    df = prepare_post_scrape_eda(source, parse)

    updated_path = output_dir / "scope_condo_eda_ready_post_scrape.csv"
    _write_csv(updated_path, df)
    _write_csv(output_dir / "scope_scrape_quality_summary.csv", scrape_quality_summary(df))
    _write_csv(output_dir / "scope_cited_vs_more_only_comparison.csv", cited_vs_more_only_comparison(df))
    _write_csv(output_dir / "scope_page_type_family_real_estate_distribution.csv", page_type_distribution(df))
    _write_csv(output_dir / "scope_unknown_page_type_diagnostics.csv", unknown_page_type_diagnostics(df))
    _write_csv(output_dir / "scope_scrape_content_availability_by_cited_status.csv", availability_by(df.assign(citation_status=df["is_cited"].map({1: "cited", 0: "more_only"})), "citation_status"))
    _write_csv(output_dir / "scope_scrape_content_availability_by_domain.csv", availability_by(df, "source_root_domain"))
    _write_csv(output_dir / "scope_scrape_content_availability_by_page_type.csv", availability_by(df, "page_type_family_real_estate"))
    _write_csv(output_dir / "scope_taxonomy_diagnostics.csv", df[[
        "normalized_url", "source_root_domain", "source_type_real_estate",
        "page_type_family_real_estate_taxonomy_v1", "page_type_family_real_estate",
        "page_type_detail_real_estate", "page_type_family_real_estate_reason",
        "page_type_available", "content_strength", "scrape_error_type",
    ]].drop_duplicates("normalized_url"))

    subsets = sensitivity_subsets(df)
    all_terms = []
    summaries = []
    sensitivity_page_types = []
    for scenario, subset in subsets.items():
        scenario_dir = output_dir / "sensitivities" / scenario
        _write_csv(scenario_dir / "scrape_quality_summary.csv", scrape_quality_summary(subset))
        _write_csv(scenario_dir / "cited_vs_more_only_comparison.csv", cited_vs_more_only_comparison(subset))
        _write_csv(scenario_dir / "page_type_family_real_estate_distribution.csv", page_type_distribution(subset))
        _write_csv(scenario_dir / "scrape_content_availability_by_cited_status.csv", availability_by(subset.assign(citation_status=subset["is_cited"].map({1: "cited", 0: "more_only"})), "citation_status"))
        terms, summary = run_exploratory_lpm(subset, scenario)
        if not terms.empty:
            all_terms.append(terms)
            _write_csv(scenario_dir / "lpm_coefficients.csv", terms)
        summaries.append(summary)
        dist = page_type_distribution(subset).assign(scenario=scenario)
        sensitivity_page_types.append(dist)
    lpm_terms = pd.concat(all_terms, ignore_index=True) if all_terms else pd.DataFrame()
    lpm_summary = pd.DataFrame(summaries)
    _write_csv(output_dir / "scope_exploratory_lpm_coefficients.csv", lpm_terms)
    _write_csv(output_dir / "scope_exploratory_lpm_main_terms.csv", lpm_terms[~lpm_terms.get("term_group", pd.Series(index=lpm_terms.index, dtype=object)).eq("prompt_fixed_effect")] if not lpm_terms.empty else lpm_terms)
    _write_csv(output_dir / "scope_exploratory_lpm_summary.csv", lpm_summary)
    _write_csv(output_dir / "scope_sensitivity_descriptive_comparison.csv", sensitivity_descriptive_summary(subsets))
    _write_csv(output_dir / "scope_sensitivity_page_type_distribution.csv", pd.concat(sensitivity_page_types, ignore_index=True))
    _write_report(OUT / "scope_post_scrape_eda_report.md", df, lpm_summary)
    return {
        "source_rows": int(len(df)),
        "unique_urls": int(df["normalized_url"].nunique()),
        "scraped_ok_rate": float(df["scraped_ok"].mean()),
        "strong_content_rate": float(df["content_strength"].eq("strong").mean()),
        "unknown_page_type_rate": float(df["page_type_family_real_estate"].eq("unknown").mean()),
        "lpm_scenarios_ok": int(lpm_summary["model_status"].eq("ok").sum()),
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eda", type=Path, default=DEFAULT_EDA)
    parser.add_argument("--parse", type=Path, default=DEFAULT_PARSE)
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.eda, args.parse, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
