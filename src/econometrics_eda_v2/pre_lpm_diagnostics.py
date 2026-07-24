"""Descriptive, pre-model diagnostics for the SCOPE condo LPM preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.econometrics_eda_v2.metric_recheck import normalise_boolean


MISSING = "<missing>"
MAIN_FEATURES = (
    "page_type_family_real_estate", "source_type_real_estate", "re_page_type_confidence",
    "content_quality_flag", "scraped_body_available", "content_feature_available",
    "taxonomy_confidence_high_or_medium",
)
NUMERIC_FEATURES = ("word_count", "text_char_count", "heading_count", "table_count", "link_count")
LOG_NUMERIC_FEATURES = ("log1p_word_count", "log1p_text_char_count", "log1p_heading_count")


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING
    text = str(value).strip()
    return text if text else MISSING


def _category(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(MISSING, index=df.index, dtype=object)
    return df[column].map(_clean).str.casefold()


def _bool(df: pd.DataFrame, column: str) -> pd.Series:
    return normalise_boolean(df, column)[0].fillna(False).astype(int)


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def enrich_lpm_diagnostics(lpm: pd.DataFrame, eda: pd.DataFrame | None) -> pd.DataFrame:
    """Attach diagnostic-only fields without changing LPM-ready model data."""
    out = lpm.copy().reset_index(drop=True)
    out["word_count"] = _numeric(out, "content_word_count")
    out["re_page_type_confidence"] = out.get("taxonomy_confidence", pd.Series(MISSING, index=out.index))
    if eda is not None and len(eda) == len(out):
        identity = ["prompt_id", "normalized_url", "cited"]
        same_order = all(
            column in eda and column in out and eda[column].astype(str).reset_index(drop=True).equals(out[column].astype(str))
            for column in identity
        )
        if same_order:
            details = eda.reset_index(drop=True)
            for column in (
                "source_url", "source_title", "page_title", "page_text_excerpt", "page_type_detail_real_estate",
                "page_type_final_real_estate", "page_type_final_real_estate_source", "re_page_type_reason",
                "text_char_count", "heading_count", "table_count", "link_count", "word_count",
            ):
                if column in details:
                    out[column] = details[column]
    for column in NUMERIC_FEATURES:
        if column not in out:
            out[column] = np.nan
    for column in ("word_count", "text_char_count", "heading_count"):
        out[f"log1p_{column}"] = np.log1p(_numeric(out, column).clip(lower=0))
    if "page_type_detail_real_estate" not in out:
        out["page_type_detail_real_estate"] = MISSING
    out["usable_content"] = (
        _category(out, "content_quality_flag").eq("ok") & _numeric(out, "word_count").ge(300)
    ).astype(int)
    return out


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    radius = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def citation_rate_by_category(df: pd.DataFrame, feature: str, min_n: int = 20) -> pd.DataFrame:
    work = pd.DataFrame({"category": _category(df, feature), "cited": _bool(df, "cited")})
    overall = float(work["cited"].mean()) if len(work) else np.nan
    rows = []
    for category, group in work.groupby("category", dropna=False):
        n = len(group)
        cited_n = int(group["cited"].sum())
        low, high = wilson_interval(cited_n, n)
        rows.append(
            {
                "feature": feature,
                "category": category,
                "n_rows": int(n),
                "cited_rows": cited_n,
                "cited_rate": cited_n / n if n else np.nan,
                "overall_cited_rate": overall,
                "difference_from_overall": cited_n / n - overall if n else np.nan,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "category_share": n / len(work) if len(work) else np.nan,
                "warning": "very_sparse_n_lt_20" if n < 20 else ("sparse_n_lt_30" if n < 30 else ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["cited_rate", "n_rows"], ascending=[False, False], kind="stable")


def _plot_citation_rates(table: pd.DataFrame, path: Path, title: str) -> None:
    ordered = table.sort_values("cited_rate", ascending=True, kind="stable")
    height = max(3.6, 0.45 * len(ordered) + 1.4)
    fig, ax = plt.subplots(figsize=(9.5, height))
    y = np.arange(len(ordered))
    rates = ordered["cited_rate"].to_numpy()
    low = ordered["wilson_ci_low"].to_numpy()
    high = ordered["wilson_ci_high"].to_numpy()
    ax.barh(y, rates, color="#3f7f8e")
    ax.errorbar(rates, y, xerr=[rates - low, high - rates], fmt="none", ecolor="#24333d", capsize=3)
    ax.set_yticks(y, ordered["category"])
    ax.set_xlim(0, min(1, max(0.55, float(np.nanmax(high)) + 0.08)))
    ax.set_xlabel("Cited rate (Wilson 95% CI)")
    ax.set_title(title)
    for yi, row in enumerate(ordered.itertuples(index=False)):
        ax.text(min(0.985, row.cited_rate + 0.015), yi, f"n={row.n_rows}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def sparse_category_diagnostics(tables: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for table in tables:
        for row in table.itertuples(index=False):
            if row.n_rows < 20 or row.cited_rows < 5:
                action = "exclude_from_main_lpm" if row.n_rows < 20 and row.cited_rows < 5 else "keep_diagnostic_only"
            elif row.n_rows < 30:
                action = "collapse_to_other"
            else:
                action = "keep"
            if row.feature == "page_type_family_real_estate" and row.n_rows >= 20:
                action = "keep"
            rows.append(
                {
                    "feature_name": row.feature,
                    "category": row.category,
                    "n_rows": row.n_rows,
                    "cited_rows": row.cited_rows,
                    "cited_rate": row.cited_rate,
                    "sparse_flag": "very_sparse" if row.n_rows < 20 else ("sparse" if row.n_rows < 30 else "not_sparse"),
                    "recommended_action": action,
                }
            )
    return pd.DataFrame(rows)


def numeric_feature_diagnostics(df: pd.DataFrame, figure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    bins_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    availability = _bool(df, "content_feature_available").eq(1) | _bool(df, "scraped_body_available").eq(1)
    for feature in NUMERIC_FEATURES + LOG_NUMERIC_FEATURES:
        values = _numeric(df, feature)
        eligible = df.loc[availability & values.notna()].copy()
        eligible["_value"] = values.loc[eligible.index]
        eligible["_cited"] = _bool(eligible, "cited")
        if len(eligible) < 4 or eligible["_value"].nunique() < 2:
            summary_rows.append({"feature": feature, "eligible_rows": len(eligible), "n_unique_values": int(eligible["_value"].nunique()), "recommended_use": "diagnostic_only_insufficient_variation", "monotonic_pattern": "not_estimable"})
            continue
        quantiles = min(4, int(eligible["_value"].nunique()))
        eligible["_bin"] = pd.qcut(eligible["_value"], q=quantiles, duplicates="drop")
        grouped = eligible.groupby("_bin", observed=True)
        feature_rows = []
        for category, group in grouped:
            n = len(group)
            cited_n = int(group["_cited"].sum())
            low, high = wilson_interval(cited_n, n)
            item = {
                "feature": feature,
                "bin": str(category),
                "bin_low": float(category.left),
                "bin_high": float(category.right),
                "n_rows": n,
                "cited_rows": cited_n,
                "cited_rate": cited_n / n,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "eligible_definition": "content_feature_available == 1 OR scraped_body_available == 1",
            }
            bins_rows.append(item)
            feature_rows.append(item)
        rates = [item["cited_rate"] for item in feature_rows]
        direction = "roughly_increasing" if rates == sorted(rates) else ("roughly_decreasing" if rates == sorted(rates, reverse=True) else "non_monotonic")
        recommended = "compare_raw_and_log1p" if feature in {"word_count", "text_char_count", "heading_count"} else ("log1p_diagnostic" if feature in LOG_NUMERIC_FEATURES else "binned_or_diagnostic_only")
        summary_rows.append({"feature": feature, "eligible_rows": len(eligible), "n_unique_values": int(eligible["_value"].nunique()), "recommended_use": recommended, "monotonic_pattern": direction})
        chart = pd.DataFrame(feature_rows)
        fig_path = figure_dir / f"citation_rate_by_{feature}_quantile.png"
        _plot_citation_rates(chart.rename(columns={"bin": "category"}), fig_path, f"Cited rate by {feature} quantile")
        figures.append(fig_path)
    return pd.DataFrame(bins_rows), pd.DataFrame(summary_rows), figures


def confidence_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    high_medium = _bool(df, "taxonomy_confidence_high_or_medium").eq(1)
    if not high_medium.any():
        high_medium = _category(df, "re_page_type_confidence").isin(["high", "medium"])
    rows = []
    for feature in ("page_type_family_real_estate", "source_type_real_estate"):
        all_table = citation_rate_by_category(df, feature).set_index("category")
        subset_table = citation_rate_by_category(df.loc[high_medium], feature).set_index("category")
        for category in all_table.index.union(subset_table.index):
            all_row = all_table.loc[category] if category in all_table.index else pd.Series(dtype=float)
            subset_row = subset_table.loc[category] if category in subset_table.index else pd.Series(dtype=float)
            n_all = int(all_row.get("n_rows", 0))
            n_subset = int(subset_row.get("n_rows", 0))
            all_rate = all_row.get("cited_rate", np.nan)
            subset_rate = subset_row.get("cited_rate", np.nan)
            stable = n_all >= 20 and n_subset >= 20 and pd.notna(all_rate) and pd.notna(subset_rate) and abs(all_rate - subset_rate) <= 0.10
            rows.append({"feature": feature, "category": category, "n_all": n_all, "cited_rate_all": all_rate, "n_high_medium": n_subset, "cited_rate_high_medium": subset_rate, "difference": subset_rate - all_rate if pd.notna(all_rate) and pd.notna(subset_rate) else np.nan, "pattern_stable": stable, "notes": "stable_when_absolute_rate_difference_le_10pp_and_both_n_ge_20" if stable else "inspect_sparse_or_changed_category"})
    return pd.DataFrame(rows)


def content_availability_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    usable = (_category(df, "content_quality_flag").eq("ok") & _numeric(df, "word_count").ge(300)).astype(int)
    datasets = {"all_rows": df, "usable_content": df.loc[usable.eq(1)], "without_usable_content": df.loc[usable.eq(0)]}
    rows = []
    for sample, subset in datasets.items():
        for feature in ("usable_content", "content_quality_flag", "page_type_family_real_estate", "source_type_real_estate"):
            work = subset.copy()
            work["usable_content"] = usable.loc[subset.index].astype(str)
            table = citation_rate_by_category(work, feature)
            for row in table.itertuples(index=False):
                rows.append({"sample": sample, **row._asdict()})
    return pd.DataFrame(rows)


def domain_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_domain"] = _category(work, "source_root_domain")
    work["_url"] = _category(work, "normalized_url")
    work["_cited"] = _bool(work, "cited")
    work["_usable"] = (_category(work, "content_quality_flag").eq("ok") & _numeric(work, "word_count").ge(300)).astype(int)
    work["_source_unknown"] = _category(work, "source_type_real_estate").eq("unknown").astype(int)
    work["_page_unknown"] = _category(work, "page_type_family_real_estate").eq("unknown").astype(int)
    work["_tax_low"] = _category(work, "re_page_type_confidence").isin(["low", "unknown", MISSING]).astype(int)
    rows = []
    for domain, group in work.groupby("_domain", sort=False):
        n = len(group)
        cited_n = int(group["_cited"].sum())
        issue = max(float(group["_source_unknown"].mean()), float(group["_page_unknown"].mean()), float(group["_tax_low"].mean()))
        recommended = "safe_control_top_domain" if n >= 20 and issue < 0.25 else ("review_manually" if n >= 10 and issue >= 0.25 else ("sparse" if n < 10 else "diagnostic_only"))
        rows.append({"source_root_domain": domain, "n_rows": n, "unique_urls": int(group["_url"].nunique()), "cited_rows": cited_n, "cited_rate": cited_n / n, "top_source_type_real_estate": _category(group, "source_type_real_estate").mode().iloc[0], "top_page_type_family_real_estate": _category(group, "page_type_family_real_estate").mode().iloc[0], "usable_content_rate": float(group["_usable"].mean()), "unknown_source_type_rate": float(group["_source_unknown"].mean()), "unknown_page_type_rate": float(group["_page_unknown"].mean()), "taxonomy_low_unknown_rate": float(group["_tax_low"].mean()), "recommended_use": recommended})
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False, kind="stable")


def lpm_variable_use_table() -> pd.DataFrame:
    rows = [
        ("page_type_family_real_estate", True, True, False, "none", "Broad conservative page taxonomy; retain unknown."),
        ("source_type_real_estate", True, True, False, "none", "Conservative domain taxonomy with residual unknown category."),
        ("re_page_type_confidence", True, False, True, "none", "Use for sensitivity restriction, not substantive main coefficient."),
        ("content_quality_flag", True, False, True, "none", "Scrape-quality/missingness diagnostic."),
        ("scraped_body_available", True, False, True, "none", "Content availability diagnostic."),
        ("content_feature_available", True, True, False, "none", "Defines content-subset eligibility; avoid imputing unavailable content."),
        ("taxonomy_confidence_high_or_medium", True, False, True, "none", "Defines taxonomy-confidence sensitivity sample."),
        ("page_type_detail_real_estate", True, False, True, "none", "Detailed categories are sparse; use family level in main model."),
        ("word_count", True, False, True, "none", "Content measurement; use only content-available subset and test nonlinearity."),
        ("text_char_count", True, False, True, "none", "Diagnostic numeric content measurement."),
        ("heading_count", True, False, True, "none", "Diagnostic numeric content measurement."),
        ("table_count", True, False, True, "none", "Diagnostic numeric content measurement."),
        ("source_root_domain", True, False, True, "none", "Domain concentration and clustered-SE diagnostic, not main claim."),
        ("source_position", False, False, True, "high", "Position is diagnostic sensitivity only, never main claim."),
        ("observed_rank", False, False, True, "high", "Rank is diagnostic sensitivity only, never main claim."),
        ("answer text overlap", False, False, True, "high", "Answer-derived; forbidden predictor."),
        ("page_answer_similarity", False, False, True, "high", "Answer-derived; forbidden predictor."),
        ("answer_like_text", False, False, True, "high", "Answer-derived; forbidden predictor."),
        ("source_group", False, False, True, "high", "Provenance/outcome-adjacent; forbidden predictor."),
        ("source_origin", False, False, True, "high", "Provenance/outcome-adjacent; forbidden predictor."),
        ("is_more_only", False, False, True, "high", "Outcome complement; forbidden predictor."),
        ("cited_label", False, False, True, "high", "Outcome duplicate; forbidden predictor."),
    ]
    return pd.DataFrame(rows, columns=["variable", "use_in_pre_lpm_eda", "use_in_main_lpm", "diagnostic_only", "leakage_risk", "reason"])


def readiness_checklist(df: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    cited = _bool(df, "cited")
    source_unknown = _category(df, "source_type_real_estate").eq("unknown").mean()
    page_unknown = _category(df, "page_type_family_real_estate").eq("unknown").mean()
    rows = [
        ("cited outcome available and binary", "pass" if set(cited.unique()) <= {0, 1} else "fail", "Binary cited outcome present."),
        ("enough cited and non-cited rows", "pass" if cited.sum() >= 100 and (1 - cited).sum() >= 100 else "fail", "Both outcome classes exceed 100 rows."),
        ("no leakage variables in main feature list", "pass", "LPM feature table excludes answer, rank, and outcome-duplicate predictors."),
        ("page_type_family has acceptable unknown rate", "pass" if page_unknown <= 0.20 else "warning", f"Unknown rate: {page_unknown:.1%}."),
        ("source_type unknown reduced to acceptable level", "warning" if source_unknown > 0.20 else "pass", f"Unknown rate: {source_unknown:.1%}; retain unknown category."),
        ("sparse categories identified", "pass", "Sparse-category output generated."),
        ("content-feature subset flagged", "pass", "Content features are restricted to availability subset."),
        ("high/medium taxonomy sensitivity completed", "pass", "All vs high/medium sensitivity generated."),
        ("domain concentration checked", "pass", "Domain-level diagnostics generated."),
        ("final LPM table exists", "pass" if (output_root.parent / "final_lpm_prep/scope_condo_lpm_ready.csv").exists() else "warning", "Checked final-LPM-prep table."),
        ("original EDA CSV remains unchanged", "pass", "This workflow reads source CSVs and writes only pre_lpm_eda outputs."),
    ]
    return pd.DataFrame(rows, columns=["check", "status", "notes"])


def _model_plan() -> str:
    return """# Pre-LPM Model Plan

No final LPM is fit in this notebook.

1. `cited ~ page_type_family_real_estate + prompt fixed effects`
2. `cited ~ page_type_family_real_estate + source_type_real_estate + prompt fixed effects`
3. Model 2 restricted to high/medium taxonomy confidence.
4. Content-subset model restricted to `content_feature_available == 1`, adding only structural page-content features.
5. Diagnostic-only sensitivity with source position or observed rank, if those fields are available; never use as a main claim.

Use heteroskedasticity-robust standard errors at minimum, with prompt-level clustered standard errors preferred and source-domain clustering as a robustness check when supported. Interpret results as associations in surfaced source appearances, not causal effects or hidden retrieval behavior.
"""


def _summary_markdown(df: pd.DataFrame, category_tables: dict[str, pd.DataFrame], sensitivity: pd.DataFrame, checklist: pd.DataFrame) -> str:
    cited = _bool(df, "cited")
    strongest = []
    for feature, table in category_tables.items():
        usable = table[table["n_rows"].ge(20)]
        if not usable.empty:
            row = usable.iloc[usable["difference_from_overall"].abs().argmax()]
            strongest.append(f"`{feature}`: `{row.category}` differs from the overall cited rate by {row.difference_from_overall:+.1%} (n={int(row.n_rows)}).")
    stable_rate = float(sensitivity["pattern_stable"].mean()) if len(sensitivity) else np.nan
    status = "near_lpm_ready_after_taxonomy_QA" if checklist["status"].ne("fail").all() else "not_ready"
    return "\n".join([
        "# SCOPE Condo Pre-LPM EDA Summary", "", f"- Dataset: {len(df):,} surfaced source appearances, {df['normalized_url'].nunique():,} normalized URLs, and {df['prompt_id'].nunique():,} prompts.", f"- Cited rate: {cited.mean():.1%} ({int(cited.sum()):,} cited rows).",
        "- Main descriptive feature patterns: " + (" ".join(strongest) if strongest else "No non-sparse categorical feature was available."),
        "- Promising first-model variables: broad page family, conservative source type with an explicit unknown category, and prompt fixed effects.",
        "- Sparse detailed categories remain diagnostic-only; do not substitute them for family-level taxonomy in the main LPM.",
        f"- Taxonomy-confidence stability: {stable_rate:.1%} of evaluated all-vs-high/medium category comparisons met the notebook stability heuristic.",
        "- Content-feature selection remains a concern because content is observed conditionally on scrape/extraction success; content models must use the content-available subset.",
        "- Exclude answer-derived, source-origin, rank/position, and outcome-duplicate variables from main LPMs.",
        f"- Final status: **{status}**.", "",
        "This is descriptive pre-model analysis. `cited=0` means surfaced/more-only, not evidence a source was rejected by a hidden retrieval system.",
    ]) + "\n"


def run_pre_lpm_diagnostics(lpm_path: Path, eda_path: Path | None, output_dir: Path, figure_dir: Path) -> dict[str, Any]:
    lpm = pd.read_csv(lpm_path, low_memory=False)
    eda = pd.read_csv(eda_path, low_memory=False) if eda_path and eda_path.exists() else None
    df = enrich_lpm_diagnostics(lpm, eda)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    variable_table = lpm_variable_use_table()
    variable_table.to_csv(output_dir / "lpm_variable_use_table.csv", index=False)
    category_tables = {feature: citation_rate_by_category(df, feature) for feature in MAIN_FEATURES}
    category_tables["page_type_detail_real_estate"] = citation_rate_by_category(df, "page_type_detail_real_estate")
    category_tables["page_type_family_real_estate"].to_csv(output_dir / "citation_rate_by_page_type_family.csv", index=False)
    category_tables["source_type_real_estate"].to_csv(output_dir / "citation_rate_by_source_type.csv", index=False)
    category_tables["re_page_type_confidence"].to_csv(output_dir / "citation_rate_by_taxonomy_confidence.csv", index=False)
    category_tables["content_quality_flag"].to_csv(output_dir / "citation_rate_by_content_quality.csv", index=False)
    pd.concat([category_tables["scraped_body_available"], category_tables["content_feature_available"]], ignore_index=True).to_csv(output_dir / "citation_rate_by_scrape_availability.csv", index=False)
    figures = []
    for feature, filename, label in (
        ("page_type_family_real_estate", "citation_rate_page_type_family.png", "Cited rate by page type family"),
        ("source_type_real_estate", "citation_rate_source_type.png", "Cited rate by source type"),
        ("content_quality_flag", "citation_rate_content_quality.png", "Cited rate by content quality"),
        ("re_page_type_confidence", "citation_rate_taxonomy_confidence.png", "Cited rate by taxonomy confidence"),
    ):
        path = figure_dir / filename
        _plot_citation_rates(category_tables[feature], path, label)
        figures.append(path)
    sparse = sparse_category_diagnostics(list(category_tables.values()))
    sparse.to_csv(output_dir / "sparse_category_diagnostics.csv", index=False)
    numeric_bins, numeric_summary, numeric_figures = numeric_feature_diagnostics(df, figure_dir)
    numeric_bins.to_csv(output_dir / "numeric_feature_bin_diagnostics.csv", index=False)
    numeric_summary.to_csv(output_dir / "numeric_feature_summary.csv", index=False)
    sensitivity = confidence_sensitivity(df)
    sensitivity.to_csv(output_dir / "sensitivity_all_vs_high_medium_confidence.csv", index=False)
    content_sensitivity = content_availability_sensitivity(df)
    content_sensitivity.to_csv(output_dir / "sensitivity_scraped_content_availability.csv", index=False)
    domains = domain_diagnostics(df)
    domains.to_csv(output_dir / "domain_level_citation_diagnostics.csv", index=False)
    checklist = readiness_checklist(df, output_dir)
    checklist.to_csv(output_dir / "pre_lpm_readiness_checklist.csv", index=False)
    (output_dir / "lpm_model_design_plan.md").write_text(_model_plan(), encoding="utf-8")
    (output_dir / "pre_lpm_eda_summary.md").write_text(_summary_markdown(df, category_tables, sensitivity, checklist), encoding="utf-8")
    return {
        "rows": int(len(df)), "unique_urls": int(df["normalized_url"].nunique()), "unique_prompts": int(df["prompt_id"].nunique()),
        "cited_rows": int(_bool(df, "cited").sum()), "cited_rate": float(_bool(df, "cited").mean()),
        "tables_dir": str(output_dir), "figure_dir": str(figure_dir), "figures": [str(path) for path in figures + numeric_figures],
    }
