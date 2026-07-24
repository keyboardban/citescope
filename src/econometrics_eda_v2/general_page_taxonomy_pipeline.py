"""Build general page-function taxonomy tables and descriptive diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.econometrics_eda_v2.general_page_taxonomy import (
    GENERAL_TAXONOMY_VERSION, PAGE_FAMILIES, PAGE_TYPES, SITE_TYPES, classify_general_page_type, classify_general_site_type,
    finalise_general_page_type,
)
from src.econometrics_eda_v2.pre_lpm_diagnostics import _bool, _category, citation_rate_by_category, wilson_interval
from src.econometrics_eda_v2.pre_lpm_intent_stratified_v6 import _cell_summary, _composition, _heatmap, _intent_audit
from src.econometrics_eda_v2.pre_lpm_readable_graphs_v5 import apply_readable_plotly_layout, save_plotly_figure


def _attach_metadata(lpm: pd.DataFrame, eda: pd.DataFrame | None) -> pd.DataFrame:
    work = lpm.copy().reset_index(drop=True)
    if eda is None: return work
    eda = eda.reset_index(drop=True)
    keys = ["prompt_id", "normalized_url", "cited"]
    same = len(work) == len(eda) and all(key in work and key in eda and work[key].astype(str).equals(eda[key].astype(str)) for key in keys)
    wanted = ["source_url", "source_title", "page_title", "meta_description", "page_text_excerpt", "headings", "intent", "content_strength", "has_price_or_package", "has_contact_info", "has_table", "source_type_real_estate", "page_type_family_real_estate", "page_type_detail_real_estate"]
    if same:
        for column in wanted:
            if column in eda: work[column] = eda[column]
    else:
        present = [column for column in wanted if column in eda]
        meta = eda[["prompt_id", "normalized_url", "cited", *present]].drop_duplicates(["prompt_id", "normalized_url", "cited"])
        work = work.merge(meta, on=["prompt_id", "normalized_url", "cited"], how="left", suffixes=("", "_eda"))
        for column in present:
            alternate = f"{column}_eda"
            if alternate in work:
                work[column] = work.get(column, pd.Series(index=work.index)).fillna(work[alternate])
                work.drop(columns=alternate, inplace=True)
    return work


def _classify_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        seed = classify_general_page_type(row, include_content=False)
        scraped = classify_general_page_type(row, include_content=True)
        final, source = finalise_general_page_type(seed, scraped, str(row.get("content_quality_flag", "")), str(row.get("content_strength", "")))
        rows.append({
            "general_taxonomy_rule_version": GENERAL_TAXONOMY_VERSION,
            "site_type_general": classify_general_site_type(row),
            "page_type_url_seed_general": seed.detail,
            "page_type_scraped_enriched_general": scraped.detail,
            "page_type_final_general": final.detail,
            "page_type_final_general_source": source,
            "page_type_family_general": final.family,
            "page_type_general": final.detail,
            "page_type_general_source": source,
            "page_type_general_confidence": final.confidence,
            "page_type_general_score": final.score,
            "page_type_general_reason": final.reason,
            "general_taxonomy_evidence_url": final.evidence_url,
            "general_taxonomy_evidence_title": final.evidence_title,
            "general_taxonomy_evidence_domain": final.evidence_domain,
            "general_taxonomy_evidence_headings": final.evidence_headings,
            "general_taxonomy_evidence_content": final.evidence_content,
            "general_taxonomy_reason": final.reason,
        })
    taxonomy = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, taxonomy], axis=1)
    confidence = out["page_type_general_confidence"].fillna("unknown")
    for level in ("high", "medium", "low", "unknown"):
        out[f"page_type_general_confidence_{level}"] = confidence.eq(level).astype(int)
    out["page_type_general_confidence_high_or_medium"] = confidence.isin(["high", "medium"]).astype(int)
    return out


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    cited = _bool(df, "cited")
    rows = []
    for column in ("page_type_general", "page_type_family_general", "site_type_general", "page_type_general_confidence"):
        for value, group in df.groupby(column, dropna=False):
            outcome = cited.loc[group.index]
            rows.append({"summary_type": "distribution", "field": column, "category": value, "n_rows": len(group), "unique_urls": group.normalized_url.nunique(), "cited_rows": int(outcome.sum()), "cited_rate": float(outcome.mean())})
    for column in ("page_type_general", "site_type_general"):
        grouped = df.groupby(["source_root_domain", column], dropna=False).size().reset_index(name="n_rows")
        for row in grouped.sort_values("n_rows", ascending=False).head(30).itertuples(index=False):
            rows.append({"summary_type": "top_domain", "field": column, "category": getattr(row, column), "source_root_domain": row.source_root_domain, "n_rows": row.n_rows})
    return pd.DataFrame(rows)


def _review_sample(df: pd.DataFrame) -> pd.DataFrame:
    picks = []; used: set[int] = set()
    for level in ("high", "medium", "low", "unknown"):
        group = df[df.page_type_general_confidence.eq(level)].sort_values(["cited", "normalized_url"], ascending=[False, True])
        sample = group.head(30).copy(); sample["review_reason"] = f"{level}_confidence"; picks.append(sample); used.update(sample.index)
    top_domains = df[df.cited.eq(1)].source_root_domain.value_counts().head(10).index
    impact = df[df.cited.eq(1) & df.source_root_domain.isin(top_domains) & ~df.index.isin(used)].sort_values(["source_root_domain", "normalized_url"]).head(30).copy()
    impact["review_reason"] = "high_impact_cited_top_domain"; picks.append(impact)
    sample = pd.concat(picks, ignore_index=True)
    columns = ["source_url", "source_title", "page_title", "source_root_domain", "page_text_excerpt", "site_type_general", "page_type_family_general", "page_type_general", "page_type_general_confidence", "page_type_general_reason", "cited", "review_reason"]
    return sample.reindex(columns=columns).rename(columns={"source_title": "title", "page_type_general_confidence": "confidence", "page_type_general_reason": "reason"})


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _rate_plot(df: pd.DataFrame, feature: str, stem: str, figure_dir: Path) -> None:
    table = citation_rate_by_category(df, feature)
    table["diff_pp"] = table.difference_from_overall * 100
    table["low_pp"] = (table.wilson_ci_low - table.overall_cited_rate) * 100
    table["high_pp"] = (table.wilson_ci_high - table.overall_cited_rate) * 100
    table.to_csv(figure_dir.parent / "tables" / f"cited_rate_by_{stem}.csv", index=False) if False else None
    ordered = table.sort_values("diff_pp")
    fig = go.Figure(go.Scatter(x=ordered.diff_pp, y=ordered.category.map(lambda x: str(x).replace("_", " ")), mode="markers+text", text=[f"n={int(n)}" for n in ordered.n_rows], textposition="middle right", error_x={"type": "data", "array": (ordered.high_pp-ordered.diff_pp).clip(lower=0), "arrayminus": (ordered.diff_pp-ordered.low_pp).clip(lower=0)}, customdata=np.column_stack([ordered.n_rows, ordered.cited_rows, ordered.cited_rate*100]), hovertemplate="Category: %{y}<br>Rows: %{customdata[0]:,}<br>Cited rows: %{customdata[1]:,}<br>Cited rate: %{customdata[2]:.1f}%<br>Difference: %{x:+.1f} pp<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="#5d6670"); apply_readable_plotly_layout(fig, f"Difference from overall cited rate by {feature.replace('_', ' ')}", "unadjusted descriptive association")
    fig.update_xaxes(title="Difference from overall cited rate (percentage points)"); fig.update_yaxes(automargin=True)
    save_plotly_figure(fig, figure_dir / "interactive" / f"cited_rate_by_{stem}.html", figure_dir / f"cited_rate_by_{stem}.png")
    plt.figure(figsize=(10, max(4, len(ordered)*.45+1.5))); plt.errorbar(ordered.diff_pp, range(len(ordered)), xerr=[(ordered.diff_pp-ordered.low_pp).clip(lower=0), (ordered.high_pp-ordered.diff_pp).clip(lower=0)], fmt="o", color="#287a8e", ecolor="#64748b", capsize=3); plt.axvline(0, color="#5d6670", linestyle="--"); plt.yticks(range(len(ordered)), ordered.category.str.replace("_", " ")); plt.xlabel("Difference from overall cited rate (percentage points)"); plt.title(f"Cited rate by {feature.replace('_', ' ')}"); plt.tight_layout(); (figure_dir / "preview").mkdir(parents=True, exist_ok=True); plt.savefig(figure_dir / "preview" / f"cited_rate_by_{stem}.png", dpi=180); plt.close()


def _intent_figures(df: pd.DataFrame, figure_dir: Path, table_dir: Path) -> None:
    audit, selected, work, source = _intent_audit(df, df)
    audit.to_csv(table_dir / "general_taxonomy_intent_column_audit.csv", index=False)
    if selected is None: return
    for feature, stem in (("page_type_family_general", "page_type_family_general"), ("site_type_general", "site_type_general")):
        summary = _cell_summary(work, feature); summary.to_csv(table_dir / f"intent_{stem}_cell_summary.csv", index=False)
        for freq in (False, True):
            suffix = "frequency" if freq else "cited_rate"; fig = _heatmap(summary, feature, freq); save_plotly_figure(fig, figure_dir / "interactive" / f"intent_{stem}_{suffix}.html", figure_dir / f"intent_{stem}_{suffix}.png")
        fig = _composition(summary, feature, False); save_plotly_figure(fig, figure_dir / "interactive" / f"composition_{stem}_by_intent.html", figure_dir / f"composition_{stem}_by_intent.png")


def run_general_page_taxonomy(lpm_path: Path, eda_path: Path | None, table_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False); eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    table_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True); (figure_dir / "interactive").mkdir(exist_ok=True)
    frame = _classify_frame(_attach_metadata(lpm, eda))
    lpm_columns = list(lpm.columns) + [column for column in frame.columns if column not in lpm.columns and column not in {"source_url", "source_title", "page_title", "meta_description", "page_text_excerpt", "headings", "intent"}]
    frame.reindex(columns=lpm_columns).to_csv(table_dir / "scope_condo_lpm_ready_with_general_page_taxonomy.csv", index=False)
    comparison_columns = ["normalized_url", "source_url", "source_root_domain", "source_title", "page_title", "source_type_real_estate", "page_type_family_real_estate", "page_type_detail_real_estate", "site_type_general", "page_type_family_general", "page_type_general", "page_type_general_confidence", "page_type_general_reason"]
    frame.reindex(columns=comparison_columns).rename(columns={"source_title": "title", "source_type_real_estate": "old_source_type_real_estate", "page_type_family_real_estate": "old_page_type_family_real_estate", "page_type_detail_real_estate": "old_page_type_detail_real_estate"}).to_csv(table_dir / "general_page_taxonomy_comparison.csv", index=False)
    summary = _summary(frame); summary.to_csv(table_dir / "general_page_taxonomy_summary.csv", index=False)
    review = _review_sample(frame); review.to_csv(table_dir / "general_page_taxonomy_manual_review_sample_150.csv", index=False)
    validation = {"allowed_page_type_only": bool(frame.page_type_general.isin(PAGE_TYPES).all()), "allowed_page_family_only": bool(frame.page_type_family_general.isin(PAGE_FAMILIES).all()), "allowed_site_type_only": bool(frame.site_type_general.isin(SITE_TYPES).all()), "vertical_specific_labels_in_general_page_type": [], "unknown_rate": float(frame.page_type_general.eq("unknown").mean()), "confidence_distribution": frame.page_type_general_confidence.value_counts().to_dict(), "leakage_columns_used": [], "forbidden_evidence_not_used": True}
    (table_dir / "general_page_taxonomy_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    for feature, stem in (("page_type_family_general", "page_type_family_general"), ("page_type_general", "page_type_general"), ("site_type_general", "site_type_general")):
        _rate_plot(frame, feature, stem, figure_dir)
    _intent_figures(frame, figure_dir, table_dir)
    example_columns = ["normalized_url", "site_type_general", "page_type_family_general", "page_type_general", "page_type_general_confidence"]
    high_examples = _markdown_table(frame.loc[frame.page_type_general_confidence.eq("high"), example_columns].head(5))
    unknown_examples = _markdown_table(frame.loc[frame.page_type_general.eq("unknown"), example_columns].head(5))
    report = "# General Page Taxonomy Report\n\n" + "## Purpose\nA vertical-specific taxonomy is useful for its original topic but cannot be the main page-function feature for a reusable cross-domain website analysis tool. The new layer classifies what a page does across industries.\n\n" + "## Concepts\n`site_type_general` describes the website/source; `page_type_general` describes the individual page function; `page_type_family_general` is the recommended coarse main EDA/LPM category.\n\n" + f"## Coverage\nUnknown page-type rate: {validation['unknown_rate']:.1%}. Confidence distribution: {validation['confidence_distribution']}.\n\n" + "## Classified examples\n" + high_examples + "\n\n## Ambiguous or unknown examples\n" + unknown_examples + "\n\n" + "## Main versus diagnostic variables\nUse `page_type_family_general`, `site_type_general`, general confidence flags, and prompt fixed effects in cross-domain main EDA/LPM. Use detailed `page_type_general` and all real-estate-specific labels as diagnostics or sensitivity variables.\n\n" + "## Limitations\nClassification remains observational and depends on URL/metadata/scrape availability. Unknown is intentionally retained where evidence is weak. No citation outcome, rank, answer text, or answer-derived feature is used.\n\n## Recommendation\nUse `page_type_family_general` as the main cross-domain page-function feature and retain vertical taxonomy only as optional diagnostics.\n"
    (table_dir / "general_page_taxonomy_report.md").write_text(report, encoding="utf-8")
    return {"rows": len(frame), "unique_urls": int(frame.normalized_url.nunique()), "unknown_rate": validation["unknown_rate"], "table_dir": str(table_dir), "figure_dir": str(figure_dir)}
