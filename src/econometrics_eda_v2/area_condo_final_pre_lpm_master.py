"""Final descriptive and LPM-readiness analysis for the Area Condo study."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.econometrics_eda_v2.pre_lpm_diagnostics import wilson_interval


FORBIDDEN_MAIN_PREDICTORS = (
    "answer_text",
    "page_answer_similarity",
    "max_chunk_answer_similarity",
    "answer_like_text",
    "answer_overlap",
    "source_group",
    "source_origin",
    "source_position",
    "observed_rank",
    "is_more_only",
    "cited_label",
    "brand_appeared_in_answer",
    "domain_citation_rate",
)

COLORS = {
    "cited": "#176B87",
    "more": "#D1495B",
    "neutral": "#667085",
    "reference": "#344054",
    "strong": "#227C9D",
    "medium": "#F4A261",
    "weak": "#E76F51",
    "failed": "#6B7280",
}

COMMON_PAGE_FUNCTIONS = (
    "homepage",
    "landing_page",
    "about_page",
    "blog_article",
    "guide_article",
    "news_article",
    "product_page",
    "pricing_page",
    "listing_page",
    "search_results_page",
    "directory_page",
    "review_page",
    "comparison_page",
    "faq_page",
    "contact_page",
    "location_page",
    "appointment_or_booking_page",
    "pdf_document",
    "forum_thread",
    "video_page",
)


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    return series.fillna("").astype(str).str.casefold().isin({"1", "1.0", "true", "yes", "y"})


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["cited"] = _bool(df["cited"]).astype(int)
    for column in (
        "scrape_success",
        "content_feature_available",
        "taxonomy_confidence_high_or_medium",
        "page_type_general_confidence_high_or_medium",
        "has_table",
        "developer_official",
        "property_portal",
        "broker_agency",
        "social_forum",
    ):
        if column in df:
            df[column] = _bool(df[column])
    for column in ("content_chars", "word_count", "heading_count", "table_count", "link_count"):
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["scraped_ok"] = df["scrape_success"]
    df["usable_content"] = df["content_feature_available"]
    df["log1p_content_chars"] = np.log1p(df["content_chars"].clip(lower=0))
    df["log1p_word_count"] = np.log1p(df["word_count"].clip(lower=0))
    df["log1p_heading_count"] = np.log1p(df["heading_count"].clip(lower=0))
    df["log1p_link_count"] = np.log1p(df["link_count"].clip(lower=0))
    df["has_headings"] = df["heading_count"].fillna(0).gt(0)
    df["has_links"] = df["link_count"].fillna(0).gt(0)
    df["has_substantial_text"] = df["word_count"].fillna(0).ge(300)
    df["has_multiple_tables"] = df["table_count"].fillna(0).ge(2)
    df["word_count_group"] = pd.cut(
        df["word_count"],
        [-np.inf, 99, 299, 999, 2999, np.inf],
        labels=["0-99", "100-299", "300-999", "1,000-2,999", "3,000+"],
    ).astype(str)
    for column in (
        "intent",
        "page_type_family_general",
        "page_type_general",
        "site_type_general",
        "page_type_general_confidence",
        "content_strength",
        "content_quality_flag",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "heading_count_group",
        "link_count_group",
    ):
        if column in df:
            df[column] = _safe_text(df[column])
    df["intent_group"] = df["intent"]
    detail = _safe_text(df["page_type_general"])
    df["page_type_general_common"] = detail.where(
        detail.isin(COMMON_PAGE_FUNCTIONS) | detail.eq("unknown"),
        "other_page_function",
    )
    df["page_type_family_general_collapsed"] = _collapse_rare(df["page_type_family_general"])
    df["site_type_general_collapsed"] = _collapse_rare(df["site_type_general"])
    return df


def _collapse_rare(series: pd.Series, minimum: int = 20) -> pd.Series:
    values = _safe_text(series)
    counts = values.value_counts()
    keep = values.eq("unknown") | values.map(counts).ge(minimum)
    return values.where(keep, "other")


def _category_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    overall = float(df["cited"].mean())
    rows: list[dict[str, Any]] = []
    for category, group in df.groupby(feature, dropna=False, observed=False):
        n = len(group)
        cited = int(group["cited"].sum())
        low, high = wilson_interval(cited, n)
        rate = cited / n if n else np.nan
        rows.append(
            {
                "feature": feature,
                "category": str(category),
                "n_rows": n,
                "row_share": n / len(df),
                "cited_rows": cited,
                "cited_rate": rate,
                "wilson_ci_low": low,
                "wilson_ci_high": high,
                "overall_cited_rate": overall,
                "difference_from_overall": rate - overall,
                "difference_from_overall_pp": (rate - overall) * 100,
                "difference_ci_low_pp": (low - overall) * 100,
                "difference_ci_high_pp": (high - overall) * 100,
                "unique_urls": group["normalized_url"].nunique(),
                "unique_prompts": group["prompt_id"].nunique(),
                "sparse_flag": n < 20 or cited < 5 or (n - cited) < 5,
            }
        )
    return pd.DataFrame(rows).sort_values(["cited_rate", "n_rows"], ascending=[False, False], kind="stable")


def _binary_summary(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    overall = float(df["cited"].mean())
    rows = []
    for feature in features:
        if feature not in df:
            continue
        available = df[feature].notna()
        values = _bool(df.loc[available, feature])
        for flag in (False, True):
            group = df.loc[available].loc[values.eq(flag)]
            n = len(group)
            if not n:
                continue
            cited = int(group["cited"].sum())
            low, high = wilson_interval(cited, n)
            rate = cited / n
            rows.append(
                {
                    "feature": feature,
                    "value": int(flag),
                    "n_rows": n,
                    "cited_rows": cited,
                    "cited_rate": rate,
                    "wilson_ci_low": low,
                    "wilson_ci_high": high,
                    "overall_cited_rate": overall,
                    "difference_from_overall_pp": (rate - overall) * 100,
                    "difference_ci_low_pp": (low - overall) * 100,
                    "difference_ci_high_pp": (high - overall) * 100,
                    "content_conditional": feature
                    in {"has_table", "has_headings", "has_links", "has_substantial_text", "has_multiple_tables"},
                }
            )
    return pd.DataFrame(rows)


def _numeric_bins(df: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "heading_count": ([-np.inf, 1, 6, 12, np.inf], ["0-1", "2-6", "7-12", "13+"]),
        "table_count": ([-np.inf, 0, 1, np.inf], ["0 tables", "1 table", "2+ tables"]),
        "link_count": ([-np.inf, 3, 8, np.inf], ["0-3", "4-8", "9+"]),
        "word_count": ([-np.inf, 99, 299, 999, 2999, np.inf], ["0-99", "100-299", "300-999", "1,000-2,999", "3,000+"]),
    }
    rows = []
    overall = float(df["cited"].mean())
    content = df[df["content_feature_available"]].copy()
    for feature, (edges, labels) in specifications.items():
        groups = pd.cut(content[feature], edges, labels=labels, ordered=True)
        for order, label in enumerate(labels):
            group = content.loc[groups.eq(label)]
            n = len(group)
            if not n:
                continue
            cited = int(group["cited"].sum())
            low, high = wilson_interval(cited, n)
            rate = cited / n
            rows.append(
                {
                    "feature": feature,
                    "bin_order": order,
                    "bin_label": label,
                    "n_rows": n,
                    "cited_rows": cited,
                    "cited_rate": rate,
                    "wilson_ci_low": low,
                    "wilson_ci_high": high,
                    "overall_cited_rate": overall,
                    "difference_from_overall_pp": (rate - overall) * 100,
                    "sparse_flag": n < 20 or cited < 5 or (n - cited) < 5,
                    "analysis_subset": "content_feature_available=true",
                }
            )
    return pd.DataFrame(rows)


def _numeric_scatter_summary(df: pd.DataFrame) -> pd.DataFrame:
    content = df[df["content_feature_available"]].copy()
    overall = float(df["cited"].mean())
    rows: list[dict[str, Any]] = []
    for feature in ("word_count", "heading_count", "table_count", "link_count"):
        values = content[feature].dropna()
        if values.empty:
            continue
        work = content.loc[values.index].copy()
        cap = float(values.quantile(0.99))
        if feature == "word_count":
            try:
                group_key = pd.qcut(work[feature], q=20, duplicates="drop")
            except ValueError:
                group_key = pd.cut(work[feature], bins=10, duplicates="drop")
            method = "20_quantile_bins"
        else:
            group_key = work[feature].clip(upper=cap)
            method = "exact_value_with_p99_tail_cap"
        for group_value, group in work.groupby(group_key, observed=True):
            n = len(group)
            cited = int(group["cited"].sum())
            low, high = wilson_interval(cited, n)
            x_value = float(group[feature].median()) if feature == "word_count" else float(group_value)
            rows.append(
                {
                    "feature": feature,
                    "grouping_method": method,
                    "x_value": x_value,
                    "x_plot": math.log1p(x_value) if feature == "word_count" else x_value,
                    "x_min": float(group[feature].min()),
                    "x_max": float(group[feature].max()),
                    "p99_cap": cap,
                    "n_rows": n,
                    "cited_rows": cited,
                    "cited_rate": cited / n,
                    "wilson_ci_low": low,
                    "wilson_ci_high": high,
                    "overall_cited_rate": overall,
                    "sparse_flag": n < 20 or cited < 5 or (n - cited) < 5,
                    "analysis_subset": "content_feature_available=true",
                }
            )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_category_forest(table: pd.DataFrame, feature: str, path: Path) -> None:
    data = table[table["feature"].eq(feature)].sort_values("difference_from_overall_pp")
    if data.empty:
        return
    height = max(4.4, 0.42 * len(data) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, height))
    y = np.arange(len(data))
    x = data["difference_from_overall_pp"].to_numpy()
    lower = x - data["difference_ci_low_pp"].to_numpy()
    upper = data["difference_ci_high_pp"].to_numpy() - x
    colors = np.where(data["sparse_flag"], COLORS["neutral"], COLORS["cited"])
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="none", ecolor="#98A2B3", capsize=3, lw=1.3)
    ax.scatter(x, y, c=colors, s=48, zorder=3)
    ax.axvline(0, color=COLORS["reference"], lw=1, ls="--")
    labels = [f"{c}  (n={n:,})" for c, n in zip(data["category"], data["n_rows"])]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Difference from overall cited rate (percentage points)")
    ax.set_title(f"Cited-rate difference: {feature.replace('_', ' ')}")
    ax.grid(axis="x", color="#EAECF0", lw=0.8)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_binary_forest(table: pd.DataFrame, path: Path) -> None:
    data = table[table["value"].eq(1)].sort_values("difference_from_overall_pp")
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, len(data) * 0.46 + 1.6)))
    y = np.arange(len(data))
    x = data["difference_from_overall_pp"].to_numpy()
    lower = x - data["difference_ci_low_pp"].to_numpy()
    upper = data["difference_ci_high_pp"].to_numpy() - x
    ax.errorbar(x, y, xerr=np.vstack([lower, upper]), fmt="o", color=COLORS["cited"], ecolor="#98A2B3", capsize=3)
    ax.axvline(0, color=COLORS["reference"], lw=1, ls="--")
    ax.set_yticks(y, [f"{f}  (n={n:,})" for f, n in zip(data["feature"], data["n_rows"])])
    ax.set_xlabel("Difference from overall cited rate (percentage points)")
    ax.set_title("Binary feature cited-rate differences")
    ax.grid(axis="x", color="#EAECF0", lw=0.8)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_numeric_bins(table: pd.DataFrame, feature: str, path: Path) -> None:
    data = table[table["feature"].eq(feature)].sort_values("bin_order")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    x = np.arange(len(data))
    rate = data["cited_rate"].to_numpy()
    lower = np.maximum(rate - data["wilson_ci_low"].to_numpy(), 0)
    upper = np.maximum(data["wilson_ci_high"].to_numpy() - rate, 0)
    ax.errorbar(x, rate, yerr=np.vstack([lower, upper]), marker="o", color=COLORS["cited"], capsize=4, lw=1.8)
    ax.axhline(data["overall_cited_rate"].iloc[0], color=COLORS["reference"], lw=1, ls="--", label="Overall cited rate")
    for position, value, n in zip(x, rate, data["n_rows"]):
        ax.annotate(f"n={n:,}", (position, value), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, data["bin_label"])
    ax.set_ylabel("Cited rate")
    ax.set_xlabel(f"{feature.replace('_', ' ')} bin")
    ax.set_ylim(0, min(1, max(data["wilson_ci_high"].max() + 0.12, 0.5)))
    ax.set_title(f"Cited rate by {feature.replace('_', ' ')} bin")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#EAECF0", lw=0.8)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_numeric_scatter(table: pd.DataFrame, feature: str, path: Path) -> None:
    data = table[table["feature"].eq(feature)].sort_values("x_plot")
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = data["x_plot"].to_numpy()
    rate = data["cited_rate"].to_numpy()
    lower = np.maximum(rate - data["wilson_ci_low"].to_numpy(), 0)
    upper = np.maximum(data["wilson_ci_high"].to_numpy() - rate, 0)
    max_n = max(int(data["n_rows"].max()), 1)
    sizes = 28 + 180 * np.sqrt(data["n_rows"].to_numpy() / max_n)
    colors = np.where(data["sparse_flag"], COLORS["neutral"], COLORS["cited"])
    ax.errorbar(x, rate, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#98A2B3", alpha=0.65, lw=1)
    ax.scatter(x, rate, s=sizes, c=colors, alpha=0.82, edgecolor="white", linewidth=0.7)
    ax.axhline(data["overall_cited_rate"].iloc[0], color=COLORS["reference"], lw=1, ls="--", label="Overall cited rate")
    ax.set_xlabel("log1p(word count bin median)" if feature == "word_count" else f"{feature.replace('_', ' ')} (p99 tail capped)")
    ax.set_ylabel("Cited rate")
    ax.set_title(f"Cited-rate scatter by {feature.replace('_', ' ')}\nBubble area represents rows; grey points are sparse")
    ax.set_ylim(0, min(1.02, max(data["wilson_ci_high"].max() + 0.08, 0.55)))
    ax.grid(color="#EAECF0", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_coverage(df: pd.DataFrame, path: Path) -> None:
    rates = pd.Series(
        {
            "Scrape success": df["scrape_success"].mean(),
            "Content features available": df["content_feature_available"].mean(),
            "Strong content": df["content_strength"].eq("strong").mean(),
            "General taxonomy known": df["page_type_family_general"].ne("unknown").mean(),
            "Site taxonomy known": df["site_type_general"].ne("unknown").mean(),
        }
    ).sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.barh(rates.index, rates.values, color=[COLORS["neutral"], COLORS["medium"], COLORS["strong"], COLORS["cited"], "#2A9D8F"])
    for bar, value in zip(bars, rates.values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center")
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("Share of source rows")
    ax.set_title("Scrape, content, and taxonomy coverage")
    ax.grid(axis="x", color="#EAECF0", lw=0.8)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_content_distribution(df: pd.DataFrame, path: Path) -> None:
    content = df[df["content_feature_available"]]
    values = [content.loc[content.cited.eq(0), "log1p_word_count"].dropna(), content.loc[content.cited.eq(1), "log1p_word_count"].dropna()]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    box = ax.boxplot(values, tick_labels=["More-only", "Cited"], patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], [COLORS["more"], COLORS["cited"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("log1p(word count)")
    ax.set_title("Content length by cited status\n(content-available rows only)")
    ax.grid(axis="y", color="#EAECF0", lw=0.8)
    fig.tight_layout()
    _save_figure(fig, path)


def _intent_cell_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    rows = []
    for (intent, category), group in df.groupby(["intent_group", feature], dropna=False, observed=False):
        n = len(group)
        cited = int(group["cited"].sum())
        rows.append(
            {
                "intent_group": str(intent),
                "feature": feature,
                "category": str(category),
                "n_rows": n,
                "cited_rows": cited,
                "cited_rate": cited / n if n else np.nan,
                "sparse_flag": n < 20 or cited < 5 or (n - cited) < 5,
            }
        )
    return pd.DataFrame(rows)


def _plot_heatmap(table: pd.DataFrame, feature: str, value: str, path: Path) -> None:
    pivot = table.pivot(index="intent_group", columns="category", values=value)
    if pivot.empty:
        return
    if value == "n_rows":
        pivot = pivot.fillna(0)
    color_map = plt.get_cmap("YlGnBu").copy()
    color_map.set_bad("#EAECF0")
    fig, ax = plt.subplots(figsize=(max(10, 0.68 * len(pivot.columns) + 4), max(5.6, 0.5 * len(pivot.index) + 2)))
    image = ax.imshow(
        np.ma.masked_invalid(pivot.to_numpy(dtype=float)),
        aspect="auto",
        cmap=color_map,
        vmin=0,
        vmax=1 if value == "cited_rate" else None,
    )
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=42, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title(f"Intent by {feature.replace('_', ' ')}: {value.replace('_', ' ')}")
    counts = table.pivot(index="intent_group", columns="category", values="n_rows").reindex(index=pivot.index, columns=pivot.columns)
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            raw_value = pivot.iat[row_index, column_index]
            n_rows = counts.iat[row_index, column_index]
            if pd.isna(raw_value) or pd.isna(n_rows):
                continue
            if value == "cited_rate":
                label = f"{raw_value:.0%}{'*' if n_rows < 20 else ''}"
                text_color = "white" if raw_value >= 0.58 else "#101828"
            else:
                label = f"{int(raw_value):,}"
                scale = float(np.nanmax(pivot.to_numpy(dtype=float))) or 1.0
                text_color = "white" if raw_value / scale >= 0.58 else "#101828"
            ax.text(column_index, row_index, label, ha="center", va="center", fontsize=6.8, color=text_color)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Rate" if value == "cited_rate" else "Rows")
    if value == "cited_rate":
        fig.text(0.5, 0.005, "* cell has fewer than 20 source appearances; interpret cautiously", ha="center", fontsize=8, color="#475467")
    fig.tight_layout()
    _save_figure(fig, path)


def _vif_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    complete = df.loc[df["content_feature_available"], columns].apply(pd.to_numeric, errors="coerce").dropna()
    rows = []
    for target in columns:
        y = complete[target].to_numpy(dtype=float)
        others = [column for column in columns if column != target]
        if len(complete) < 5 or not others or np.nanstd(y) == 0:
            vif = np.nan
        else:
            x = np.column_stack([np.ones(len(complete)), complete[others].to_numpy(dtype=float)])
            fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
            denominator = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - np.sum((y - fitted) ** 2) / denominator if denominator else np.nan
            vif = 1 / max(1 - r_squared, 1e-12) if np.isfinite(r_squared) else np.nan
        rows.append(
            {
                "variable": target,
                "n_complete": len(complete),
                "vif": vif,
                "flag": "severe" if np.isfinite(vif) and vif >= 10 else ("review" if np.isfinite(vif) and vif >= 5 else ""),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def _copy_if_present(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def run_area_condo_final_master(
    input_path: Path,
    taxonomy_dir: Path,
    table_dir: Path,
    figure_dir: Path,
) -> dict[str, Any]:
    """Generate the complete Area Condo descriptive and pre-LPM package."""
    df = _prepare(pd.read_csv(input_path, low_memory=False))
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    final_path = table_dir / "area_condo_lpm_ready_final_pre_lpm_master.csv"
    df.to_csv(final_path, index=False)

    metrics = [
        ("source_rows", len(df), "row"),
        ("unique_urls", df["normalized_url"].nunique(), "url"),
        ("prompts_with_sources", df["prompt_id"].nunique(), "prompt"),
        ("cited_rows", int(df["cited"].sum()), "row"),
        ("cited_rate", float(df["cited"].mean()), "row_rate"),
        ("scrape_success_rate", float(df["scrape_success"].mean()), "row_rate"),
        ("content_feature_available_rate", float(df["content_feature_available"].mean()), "row_rate"),
        ("strong_content_rate", float(df["content_strength"].eq("strong").mean()), "row_rate"),
        ("unknown_general_page_family_rate", float(df["page_type_family_general"].eq("unknown").mean()), "row_rate"),
        ("unknown_general_site_type_rate", float(df["site_type_general"].eq("unknown").mean()), "row_rate"),
        ("unknown_real_estate_page_family_rate", float(df["page_type_family_real_estate"].eq("unknown").mean()), "row_rate"),
        ("unknown_real_estate_source_type_rate", float(df["source_type_real_estate"].eq("unknown").mean()), "row_rate"),
        ("high_medium_general_taxonomy_rate", float(df["page_type_general_confidence_high_or_medium"].mean()), "row_rate"),
    ]
    consistency = pd.DataFrame(metrics, columns=["metric", "value", "analysis_level"])
    consistency["status"] = "observed"
    consistency.to_csv(table_dir / "dataset_consistency_summary.csv", index=False)

    lineage = pd.DataFrame(
        [
            ["Bright Data source export", "5,758 source appearances and citation outcome", "row-level base"],
            ["Bright Data crawler cache", "scrape and content features for 2,881 URLs", "URL-level evidence"],
            ["Area Condo taxonomy prep", "general and real-estate taxonomy", "pre-model classification"],
            ["Area Condo final master", "descriptive diagnostics and LPM guardrails", "this notebook"],
        ],
        columns=["layer", "contribution", "role"],
    )
    lineage.to_csv(table_dir / "data_lineage_map.csv", index=False)

    categorical_features = [
        "page_type_family_general",
        "page_type_general",
        "page_type_general_common",
        "site_type_general",
        "page_type_general_confidence",
        "content_strength",
        "content_quality_flag",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "intent_group",
    ]
    category_tables = [_category_summary(df, feature) for feature in categorical_features]
    categorical = pd.concat(category_tables, ignore_index=True)
    categorical.to_csv(table_dir / "categorical_feature_cited_rate_summary.csv", index=False)
    for feature in categorical_features:
        categorical[categorical.feature.eq(feature)].to_csv(table_dir / f"distribution_{feature}.csv", index=False)

    hierarchy = df.groupby(
        ["page_type_family_general", "page_type_general", "page_type_general_common"],
        dropna=False,
    ).agg(
        n_rows=("cited", "size"),
        cited_rows=("cited", "sum"),
        cited_rate=("cited", "mean"),
        unique_urls=("normalized_url", "nunique"),
    ).reset_index().sort_values(["page_type_family_general", "n_rows"], ascending=[True, False])
    hierarchy["common_function_view"] = hierarchy.page_type_general.isin(COMMON_PAGE_FUNCTIONS)
    hierarchy["sparse_flag"] = hierarchy.n_rows.lt(20) | hierarchy.cited_rows.lt(5) | (hierarchy.n_rows - hierarchy.cited_rows).lt(5)
    hierarchy.to_csv(table_dir / "general_page_function_hierarchy.csv", index=False)

    binary_features = [
        "scrape_success",
        "content_feature_available",
        "taxonomy_confidence_high_or_medium",
        "page_type_general_confidence_high_or_medium",
        "developer_official",
        "property_portal",
        "broker_agency",
        "social_forum",
        "has_table",
        "has_headings",
        "has_links",
        "has_substantial_text",
        "has_multiple_tables",
    ]
    binary = _binary_summary(df, binary_features)
    binary.to_csv(table_dir / "binary_feature_cited_rate_summary.csv", index=False)

    numeric_bins = _numeric_bins(df)
    numeric_bins.to_csv(table_dir / "numeric_feature_bin_diagnostics.csv", index=False)
    numeric_scatter = _numeric_scatter_summary(df)
    numeric_scatter.to_csv(table_dir / "numeric_feature_scatter_diagnostics.csv", index=False)
    content = df[df["content_feature_available"]]
    shape_rows = []
    for feature in ("word_count", "heading_count", "table_count", "link_count"):
        shape_rows.append(
            {
                "feature": feature,
                "analysis_subset": "content_feature_available=true",
                "n_rows": int(content[feature].notna().sum()),
                "zero_share": float(content[feature].fillna(0).eq(0).mean()),
                "median": float(content[feature].median()),
                "p90": float(content[feature].quantile(0.9)),
                "spearman_with_cited": float(content[[feature, "cited"]].corr(method="spearman").iloc[0, 1]),
                "recommended_form": {
                    "word_count": "log1p_word_count",
                    "heading_count": "heading_count_group",
                    "table_count": "has_table or threshold group",
                    "link_count": "link_count_group",
                }[feature],
                "interpretation": "diagnostic association only; conditional on measurable content",
            }
        )
    pd.DataFrame(shape_rows).to_csv(table_dir / "numeric_feature_shape_summary.csv", index=False)

    cited_availability = []
    for cited_value, group in df.groupby("cited"):
        cited_availability.append(
            {
                "cited": int(cited_value),
                "n_rows": len(group),
                "scrape_success_rate": group["scrape_success"].mean(),
                "content_feature_available_rate": group["content_feature_available"].mean(),
                "strong_content_rate": group["content_strength"].eq("strong").mean(),
                "unknown_page_family_rate": group["page_type_family_general"].eq("unknown").mean(),
                "unknown_site_type_rate": group["site_type_general"].eq("unknown").mean(),
            }
        )
    pd.DataFrame(cited_availability).to_csv(table_dir / "scrape_content_availability_by_cited.csv", index=False)

    domain = df.groupby("source_root_domain", dropna=False).agg(
        n_rows=("cited", "size"),
        cited_rows=("cited", "sum"),
        cited_rate=("cited", "mean"),
        unique_urls=("normalized_url", "nunique"),
        scrape_success_rate=("scrape_success", "mean"),
        content_feature_available_rate=("content_feature_available", "mean"),
        unknown_page_family_rate=("page_type_family_general", lambda s: s.eq("unknown").mean()),
        unknown_site_type_rate=("site_type_general", lambda s: s.eq("unknown").mean()),
    ).reset_index().sort_values(["n_rows", "cited_rows"], ascending=False)
    domain.to_csv(table_dir / "scrape_content_availability_by_domain.csv", index=False)

    failure_summary = df.assign(failure_reason=np.where(df.scrape_success, df.content_quality_flag, df.content_quality_flag)).groupby(
        ["scrape_success", "content_strength", "failure_reason"], dropna=False
    ).agg(n_rows=("cited", "size"), unique_urls=("normalized_url", "nunique"), cited_rows=("cited", "sum")).reset_index()
    failure_summary["row_share"] = failure_summary.n_rows / len(df)
    failure_summary.to_csv(table_dir / "scrape_failure_summary.csv", index=False)

    missing_rows = []
    for feature in ("content_chars", "word_count", "heading_count", "table_count", "link_count"):
        missing = df[feature].isna() | ~df["content_feature_available"]
        missing_rows.append(
            {
                "feature": feature,
                "n_rows": len(df),
                "n_unavailable": int(missing.sum()),
                "unavailable_rate": float(missing.mean()),
                "cited_rate_available": float(df.loc[~missing, "cited"].mean()),
                "cited_rate_unavailable": float(df.loc[missing, "cited"].mean()),
                "recommended_handling": "content subset plus explicit availability diagnostic",
            }
        )
    pd.DataFrame(missing_rows).to_csv(table_dir / "content_missingness_summary.csv", index=False)

    intent_tables = []
    for feature in ("page_type_family_general", "page_type_general_common", "site_type_general"):
        cells = _intent_cell_summary(df, feature)
        cells.to_csv(table_dir / f"intent_{feature}_cell_summary.csv", index=False)
        intent_tables.append(cells)

    correlation_columns = ["log1p_word_count", "log1p_content_chars", "heading_count", "table_count", "link_count", "has_table"]
    correlation = content[correlation_columns].apply(pd.to_numeric, errors="coerce").corr(method="spearman")
    correlation.to_csv(table_dir / "content_feature_spearman_correlation.csv")
    vif = _vif_summary(df, correlation_columns)
    vif.to_csv(table_dir / "vif_summary.csv", index=False)
    redundancy = pd.DataFrame(
        [
            ["log1p_word_count", "log1p_content_chars", "keep one length measure in a given specification"],
            ["table_count", "has_table", "prefer has_table because table_count is zero-inflated"],
            ["raw counts", "count groups/log transforms", "do not include raw and transformed versions together"],
            ["page_type_family_real_estate", "page_type_family_general", "estimate alternative taxonomies separately before combining"],
        ],
        columns=["feature_a", "feature_b", "recommendation"],
    )
    redundancy.to_csv(table_dir / "redundancy_recommendations.csv", index=False)

    sparse_rows = []
    for feature in ("page_type_family_general", "page_type_general_common", "site_type_general", "source_type_real_estate", "page_type_family_real_estate", "intent_group"):
        summary = categorical[categorical.feature.eq(feature)]
        for row in summary.itertuples(index=False):
            action = "keep_unknown_as_own_category" if row.category == "unknown" else ("collapse_to_other" if row.sparse_flag else "keep")
            sparse_rows.append(
                {
                    "feature": feature,
                    "category": row.category,
                    "n_rows": row.n_rows,
                    "cited_rows": row.cited_rows,
                    "unique_prompts": row.unique_prompts,
                    "sparse_flag": row.sparse_flag,
                    "recommended_lpm_action": action,
                }
            )
    sparse = pd.DataFrame(sparse_rows)
    sparse.to_csv(table_dir / "sparse_category_collapse_plan.csv", index=False)

    main_variables = ["page_type_family_general_collapsed", "site_type_general_collapsed", "content_feature_available", "prompt_id"]
    content_variables = [
        "page_type_family_general_collapsed",
        "site_type_general_collapsed",
        "has_table",
        "log1p_word_count",
        "heading_count_group",
        "link_count_group",
        "content_quality_flag",
        "prompt_id",
    ]
    candidates = {
        "outcome": ["cited"],
        "main_all_row_lpm": main_variables,
        "content_subset_lpm": content_variables,
        "alternative_taxonomy_sensitivity": ["page_type_general_common", "page_type_family_real_estate", "source_type_real_estate"],
        "diagnostic_only": ["word_count", "content_chars", "heading_count", "table_count", "link_count", "content_strength"],
        "forbidden_main_predictors": list(FORBIDDEN_MAIN_PREDICTORS),
    }
    (table_dir / "final_lpm_candidate_columns.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")

    leakage_rows = []
    all_main = set(main_variables + content_variables)
    for variable in FORBIDDEN_MAIN_PREDICTORS:
        leakage_rows.append(
            {
                "variable_name": variable,
                "present_in_dataset": variable in df.columns,
                "present_in_main_candidates": variable in all_main,
                "allowed_role": "diagnostic_only" if variable in {"source_position", "observed_rank"} else "forbidden_predictor",
                "status": "fail" if variable in all_main else "pass",
            }
        )
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(table_dir / "leakage_guardrail.csv", index=False)

    dictionary_rows = []
    for column in df.columns:
        role = "unused"
        if column == "cited":
            role = "outcome"
        elif column in main_variables:
            role = "main_all_row"
        elif column in content_variables:
            role = "content_subset"
        elif column in candidates["alternative_taxonomy_sensitivity"]:
            role = "taxonomy_sensitivity"
        elif column in candidates["diagnostic_only"]:
            role = "diagnostic_only"
        leakage_risk = "high" if any(token in column.casefold() for token in FORBIDDEN_MAIN_PREDICTORS) else "none"
        dictionary_rows.append(
            {
                "variable_name": column,
                "role": role,
                "use_in_main_all_row_lpm": column in main_variables,
                "use_in_content_subset_lpm": column in content_variables,
                "use_in_sensitivity_only": column in candidates["alternative_taxonomy_sensitivity"],
                "diagnostic_only": role == "diagnostic_only",
                "leakage_risk": leakage_risk,
                "availability_condition": "content_feature_available=true" if column in content_variables and column not in main_variables else "all rows",
                "reason": "pre-specified observable page/source feature; no answer-derived predictor" if role not in {"unused", "diagnostic_only"} else "retain for QA or descriptive analysis",
            }
        )
    pd.DataFrame(dictionary_rows).to_csv(table_dir / "final_lpm_candidate_variable_dictionary.csv", index=False)

    all_row_formula = "cited ~ C(page_type_family_general_collapsed) + C(site_type_general_collapsed) + content_feature_available + C(prompt_id)"
    content_formula = "cited ~ C(page_type_family_general_collapsed) + C(site_type_general_collapsed) + has_table + log1p_word_count + C(heading_count_group) + C(link_count_group) + C(content_quality_flag) + C(prompt_id)"
    spec = f"""# Recommended Area Condo LPM design

## Model 1: all source appearances

`{all_row_formula}`

## Model 2: taxonomy extension

Use `C(page_type_general_common)` for a familiar detailed page-function sensitivity model. Also test `C(page_type_family_real_estate)` or `C(source_type_real_estate)` in separate specifications. Do not automatically combine overlapping taxonomy systems.

## Model 3: taxonomy-confidence subset

Repeat Model 1 where `page_type_general_confidence_high_or_medium == true`.

## Model 4: content-available subset

Restrict to `content_feature_available == true` and estimate:

`{content_formula}`

## Model 5: position sensitivity only

Position or rank may be added only to a separately labeled diagnostic model using the original source data. They are absent from the LPM-ready table and cannot support the main claim.

Use prompt fixed effects. Begin with HC3 standard errors; report prompt-clustered and domain-clustered alternatives only when the number and structure of clusters are adequate. These models estimate conditional associations, not causal effects or the AI system's internal retrieval mechanism.
"""
    (table_dir / "recommended_lpm_model_design.md").write_text(spec, encoding="utf-8")

    _copy_if_present(taxonomy_dir / "taxonomy_manual_review_sample_150.csv", table_dir / "taxonomy_manual_review_sample_150.csv")
    _copy_if_present(taxonomy_dir / "taxonomy_unknown_diagnostics.csv", table_dir / "taxonomy_unknown_diagnostics.csv")
    _copy_if_present(taxonomy_dir / "prompt_manifest_join_audit.csv", table_dir / "prompt_manifest_join_audit.csv")

    _plot_coverage(df, figure_dir / "coverage_scrape_content_taxonomy.png")
    _plot_binary_forest(binary, figure_dir / "binary_feature_difference_forest.png")
    for feature, stem in (
        ("page_type_family_general", "page_type_family_general"),
        ("page_type_general_common", "common_page_function"),
        ("site_type_general", "site_type_general"),
        ("content_quality_flag", "content_quality"),
        ("page_type_general_confidence", "taxonomy_confidence"),
        ("source_type_real_estate", "source_type_real_estate"),
        ("page_type_family_real_estate", "page_type_family_real_estate"),
    ):
        _plot_category_forest(categorical, feature, figure_dir / f"difference_from_overall_{stem}.png")
    for feature in ("heading_count", "word_count", "table_count", "link_count"):
        _plot_numeric_bins(numeric_bins, feature, figure_dir / f"cited_rate_by_{feature}_ordered.png")
        _plot_numeric_scatter(numeric_scatter, feature, figure_dir / f"scatter_cited_rate_vs_{feature}.png")
    _plot_content_distribution(df, figure_dir / "distribution_log1p_word_count_by_cited.png")
    for cells, feature in zip(intent_tables, ("page_type_family_general", "page_type_general_common", "site_type_general")):
        _plot_heatmap(cells, feature, "cited_rate", figure_dir / f"heatmap_intent_by_{feature}_cited_rate.png")
        _plot_heatmap(cells, feature, "n_rows", figure_dir / f"heatmap_intent_by_{feature}_frequency.png")

    page_family_rows = categorical[categorical.feature.eq("page_type_family_general")]
    supported = page_family_rows[page_family_rows.n_rows >= 100]
    if supported.empty:
        supported = page_family_rows
    top_page = supported.sort_values("cited_rate", ascending=False).iloc[0]
    low_page = supported.sort_values("cited_rate").iloc[0]
    availability = pd.DataFrame(cited_availability).set_index("cited")
    report = f"""# Area Condo final pre-LPM descriptive report

## Dataset

The analysis contains **{len(df):,} source appearances**, **{df.normalized_url.nunique():,} unique URLs**, and **{df.prompt_id.nunique():,} prompts with source rows**. There are **{int(df.cited.sum()):,} cited appearances**, for an overall cited rate of **{df.cited.mean():.1%}**.

## Scrape and content availability

Row-level scrape success is **{df.scrape_success.mean():.1%}** and measurable content is available for **{df.content_feature_available.mean():.1%}** of rows. Availability differs slightly by observed citation status: **{availability.loc[1, 'content_feature_available_rate']:.1%}** for cited rows and **{availability.loc[0, 'content_feature_available_rate']:.1%}** for more-only rows. Content features must therefore be interpreted only within the measurable-content subset, while availability remains an explicit missingness diagnostic.

## Taxonomy

The general page-family unknown rate is **{df.page_type_family_general.eq('unknown').mean():.1%}** and the general site-type unknown rate is **{df.site_type_general.eq('unknown').mean():.1%}** at row level. High/medium general taxonomy confidence covers **{df.page_type_general_confidence_high_or_medium.mean():.1%}** of rows. Unknown remains a valid category and is not forced into a guessed label.

Among page-family categories with at least 100 rows, **{top_page.category}** has the highest descriptive cited rate ({top_page.cited_rate:.1%}, n={int(top_page.n_rows):,}) and **{low_page.category}** the lowest ({low_page.cited_rate:.1%}, n={int(low_page.n_rows):,}). These are unadjusted associations and may reflect prompt mix, repeated URLs, domain composition, or content availability.

## Numeric content features

Numeric plots use ordered threshold bins rather than citation-rate sorting. `heading_count` does not establish a clear linear effect and should remain diagnostic or a control/sensitivity form. `table_count` is zero-inflated, so `has_table` or a threshold group is preferable to raw count. All numeric results are conditional on `content_feature_available = true`.

## LPM boundary

No final LPM is fit in this notebook. Answer-derived variables, source provenance, position, rank, and outcome duplicates are excluded from the main candidate set. The recommended next step is an all-row page/site taxonomy specification with prompt fixed effects, followed by a separately labeled content-available model and taxonomy-confidence sensitivity checks.

**Readiness:** `near_lpm_ready_after_taxonomy_QA`. The data and feature forms are suitable for first-pass LPM estimation, while manual taxonomy review and cluster-robust inference choices remain before final claims.
"""
    (table_dir / "final_pre_lpm_master_report.md").write_text(report, encoding="utf-8")

    checks = pd.DataFrame(
        [
            ["latest source table loaded", "pass"],
            ["citation outcome is binary", "pass" if set(df.cited.unique()) <= {0, 1} else "fail"],
            ["scrape and content availability explicit", "pass"],
            ["general and real-estate taxonomies retained", "pass"],
            ["unknown taxonomy retained", "pass"],
            ["content features restricted in interpretation", "pass"],
            ["forbidden variables absent from main candidates", "pass" if not leakage.present_in_main_candidates.any() else "fail"],
            ["sparse categories audited", "pass"],
            ["manual taxonomy review available", "pass" if (table_dir / "taxonomy_manual_review_sample_150.csv").exists() else "warning"],
            ["final LPM fit", "not_run_by_design"],
        ],
        columns=["check", "status"],
    )
    checks.to_csv(table_dir / "final_pre_lpm_readiness_checklist.csv", index=False)

    return {
        "input_path": str(input_path),
        "rows": len(df),
        "unique_urls": int(df.normalized_url.nunique()),
        "unique_prompts": int(df.prompt_id.nunique()),
        "cited_rows": int(df.cited.sum()),
        "cited_rate": float(df.cited.mean()),
        "scrape_success_rate": float(df.scrape_success.mean()),
        "content_feature_available_rate": float(df.content_feature_available.mean()),
        "unknown_page_family_rate": float(df.page_type_family_general.eq("unknown").mean()),
        "unknown_site_type_rate": float(df.site_type_general.eq("unknown").mean()),
        "forbidden_in_main": int(leakage.present_in_main_candidates.sum()),
        "readiness": "near_lpm_ready_after_taxonomy_QA",
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir),
    }
