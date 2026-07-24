"""Conservative final-LPM preparation for the SCOPE condo EDA dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.metric_recheck import aggregate_urls, normalise_boolean
from src.econometrics_eda_v2.real_estate_taxonomy import SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES


MISSING = "<missing>"
SAFE_CONTENT_FEATURES = ("has_price_or_package", "has_contact_info", "has_table")
LEAKAGE_TOKENS = (
    "answer",
    "similarity",
    "source_group",
    "source_origin",
    "source_position",
    "observed_rank",
    "cited_label",
    "is_more_only",
)


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


def apply_safe_source_type_patch(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only exact, reviewed domain rules to current unknown source types."""
    out = df.copy()
    before = _category(out, "source_type_real_estate")
    domain = _category(out, "source_root_domain")
    mapped_type = domain.map(lambda value: SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES.get(value, (None, ""))[0])
    mapped_reason = domain.map(lambda value: SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES.get(value, (None, ""))[1])
    apply = before.eq("unknown") & mapped_type.notna()
    out["source_type_real_estate_before_patch"] = before
    out["source_type_real_estate"] = before.where(~apply, mapped_type)
    out["source_type_patch_applied"] = apply.astype(int)
    out["source_type_patch_rule"] = np.where(apply, "safe_exact_root_domain_override", "")
    out["source_type_patch_reason"] = np.where(apply, mapped_reason, "")
    # This preparation patch deliberately does not infer a page's content type
    # from a new domain source type. Its existing page taxonomy remains intact.
    out["page_type_family_real_estate_before_patch"] = _category(out, "page_type_family_real_estate")
    return out


def source_type_unknown_reduction_audit(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    unknown_before = _category(before, "source_type_real_estate").eq("unknown")
    work = after.loc[unknown_before].copy()
    if work.empty:
        return pd.DataFrame()
    work["_domain"] = _category(work, "source_root_domain")
    work["_url"] = _category(work, "normalized_url")
    work["_cited"] = _bool(work, "cited")
    work["_before"] = _category(work, "source_type_real_estate_before_patch")
    work["_after"] = _category(work, "source_type_real_estate")
    rows = []
    for domain, group in work.groupby("_domain", sort=False):
        row_count = len(group)
        cited_count = int(group["_cited"].sum())
        mapped = group["source_type_patch_applied"].eq(1).any()
        priority_score = row_count + 2 * cited_count + int(group["_url"].nunique())
        if row_count >= 5 or cited_count >= 3:
            priority = "high"
        elif row_count >= 3 or cited_count >= 2:
            priority = "medium"
        else:
            priority = "low"
        title = group.get("page_title", pd.Series(dtype=object)).dropna().astype(str).head(2).tolist()
        rows.append(
            {
                "source_root_domain": domain,
                "row_count": int(row_count),
                "source_rows_n": int(row_count),
                "unique_url_count": int(group["_url"].nunique()),
                "cited_count": cited_count,
                "cited_rate": float(group["_cited"].mean()),
                "priority_score": int(priority_score),
                "recommended_review_priority": priority,
                "source_type_before": "unknown",
                "source_type_after": group["_after"].iloc[0] if group["_after"].nunique() == 1 else "mixed",
                "patch_status": "applied_safe_domain_rule" if mapped else "retained_unknown_no_safe_domain_rule",
                "patch_rule": group.get("source_type_patch_rule", pd.Series([""])).replace("", np.nan).dropna().iloc[0] if mapped else "",
                "patch_reason": group.get("source_type_patch_reason", pd.Series([""])).replace("", np.nan).dropna().iloc[0] if mapped else "No exact reviewed domain-level rule; retained unknown.",
                "example_titles": " | ".join(title),
            }
        )
    priority_order = {"high": 0, "medium": 1, "low": 2}
    return pd.DataFrame(rows).sort_values(
        ["recommended_review_priority", "priority_score", "row_count"],
        ascending=[True, False, False],
        key=lambda series: series.map(priority_order) if series.name == "recommended_review_priority" else series,
        kind="stable",
    )


def _summary_values(df: pd.DataFrame) -> dict[str, float]:
    urls = aggregate_urls(df)
    n_rows = len(df)
    n_urls = len(urls)
    family = _category(df, "page_type_family_real_estate")
    source_type = _category(df, "source_type_real_estate")
    confidence = _category(df, "re_page_type_confidence")
    url_family = _category(urls, "page_type_family_real_estate")
    url_source_type = _category(urls, "source_type_real_estate")
    url_confidence = _category(urls, "re_page_type_confidence")
    return {
        "source_type_unknown_n_row": float(source_type.eq("unknown").sum()),
        "source_type_unknown_rate_row": float(source_type.eq("unknown").mean()),
        "page_type_unknown_n_row": float(family.eq("unknown").sum()),
        "page_type_unknown_rate_row": float(family.eq("unknown").mean()),
        "high_medium_confidence_n_row": float(confidence.isin(["high", "medium"]).sum()),
        "high_medium_confidence_rate_row": float(confidence.isin(["high", "medium"]).mean()),
        "source_type_unknown_n_url": float(url_source_type.eq("unknown").sum()),
        "source_type_unknown_rate_url": float(url_source_type.eq("unknown").mean()),
        "page_type_unknown_n_url": float(url_family.eq("unknown").sum()),
        "page_type_unknown_rate_url": float(url_family.eq("unknown").mean()),
        "high_medium_confidence_n_url": float(url_confidence.isin(["high", "medium"]).sum()),
        "high_medium_confidence_rate_url": float(url_confidence.isin(["high", "medium"]).mean()),
        "row_denominator": float(n_rows),
        "url_denominator": float(n_urls),
    }


def taxonomy_before_after_summary(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    old = _summary_values(before)
    new = _summary_values(after)
    metric_specs = (
        ("source_type_unknown", "row", "source_type_unknown_n_row", "source_type_unknown_rate_row", "Unknown source types decline only through exact domain overrides."),
        ("page_type_unknown", "row", "page_type_unknown_n_row", "page_type_unknown_rate_row", "Unchanged by design: source-type patch does not infer page content type."),
        ("high_medium_taxonomy_confidence", "row", "high_medium_confidence_n_row", "high_medium_confidence_rate_row", "Unchanged by design: confidence remains tied to the page-type classifier."),
        ("source_type_unknown", "url", "source_type_unknown_n_url", "source_type_unknown_rate_url", "URL aggregation uses the most frequent non-null source type."),
        ("page_type_unknown", "url", "page_type_unknown_n_url", "page_type_unknown_rate_url", "Unchanged by design: source-type patch does not infer page content type."),
        ("high_medium_taxonomy_confidence", "url", "high_medium_confidence_n_url", "high_medium_confidence_rate_url", "URL aggregation uses the most frequent non-null confidence."),
    )
    rows = []
    for metric, level, count_key, rate_key, note in metric_specs:
        denominator = old["row_denominator"] if level == "row" else old["url_denominator"]
        rows.append(
            {
                "metric_name": metric,
                "level": level,
                "denominator": int(denominator),
                "before_n": int(old[count_key]),
                "before_rate": old[rate_key],
                "after_n": int(new[count_key]),
                "after_rate": new[rate_key],
                "n_change": int(new[count_key] - old[count_key]),
                "rate_change": new[rate_key] - old[rate_key],
                "notes": note,
            }
        )
    return pd.DataFrame(rows)


def taxonomy_manual_review_sample(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    work = df.copy()
    work["_row_index"] = work.index
    cited = _bool(work, "cited").eq(1)
    confidence = _category(work, "re_page_type_confidence")
    source_type = _category(work, "source_type_real_estate")
    page_type = _category(work, "page_type_family_real_estate")
    top_domains = _category(work, "source_root_domain").value_counts().head(5).index
    strata = (
        ("cited_high_confidence", cited & confidence.eq("high"), 20),
        ("low_or_unknown_confidence", confidence.isin(["low", "unknown", MISSING]), 20),
        ("unknown_source_type", source_type.eq("unknown"), 20),
        ("unknown_page_type", page_type.eq("unknown"), 20),
        ("top_domain", _category(work, "source_root_domain").isin(top_domains), 20),
    )
    chosen: list[int] = []
    reasons: dict[int, list[str]] = {}
    for sequence, (reason, mask, target) in enumerate(strata):
        candidates = work.loc[mask & ~work["_row_index"].isin(chosen), "_row_index"]
        take = min(target, len(candidates))
        if take:
            picks = pd.Series(candidates).sample(n=take, random_state=20260713 + sequence).tolist()
            chosen.extend(picks)
            for pick in picks:
                reasons.setdefault(pick, []).append(reason)
    if len(chosen) < n:
        candidates = work.loc[~work["_row_index"].isin(chosen), "_row_index"]
        picks = pd.Series(candidates).sample(n=min(n - len(chosen), len(candidates)), random_state=20260800).tolist()
        chosen.extend(picks)
        for pick in picks:
            reasons.setdefault(pick, []).append("coverage_fill")
    selected = work.set_index("_row_index").loc[chosen].reset_index()
    title = selected.get("page_title", pd.Series(MISSING, index=selected.index)).map(_clean)
    if "source_title" in selected:
        title = title.where(title.ne(MISSING), selected["source_title"].map(_clean))
    columns = [
        "prompt_id", "source_url", "source_root_domain", "cited", "source_type_real_estate_before_patch",
        "source_type_real_estate", "source_type_patch_applied", "source_type_patch_reason",
        "page_type_family_real_estate", "page_type_detail_real_estate", "page_type_final_real_estate",
        "re_page_type_confidence", "re_page_type_reason", "content_quality_flag", "word_count",
    ]
    out = selected[[column for column in columns if column in selected]].copy()
    out.insert(3, "title", title)
    out.insert(4, "page_text_excerpt", selected.get("page_text_excerpt", pd.Series(MISSING, index=selected.index)).map(_clean))
    out["review_reason"] = selected["_row_index"].map(lambda value: ";".join(reasons[value]))
    return out


def lpm_variable_dictionary() -> pd.DataFrame:
    rows = [
        ("cited", "binary outcome", True, False, False, "none", "Dependent variable only; never a predictor."),
        ("prompt_id", "categorical fixed effect / cluster", True, False, False, "none", "Controls for prompt-level comparison set and supports prompt-clustered standard errors."),
        ("normalized_url", "identifier", False, False, True, "none", "Deduplication and audit key, not a model regressor."),
        ("source_root_domain", "categorical / cluster", False, True, True, "none", "Use for domain-clustered robustness or grouped diagnostics; high-cardinality domain fixed effects are not the default."),
        ("page_type_family_real_estate", "categorical", True, False, False, "none", "Broad conservative taxonomy; retain unknown as a reference category."),
        ("source_type_real_estate", "categorical", True, True, False, "none", "Use with an explicit unknown category; compare Model 1 and Model 2 because residual unknown coverage remains material."),
        ("developer_official_flag", "binary", True, False, False, "none", "Exact source-type indicator derived from the conservative taxonomy."),
        ("property_portal_flag", "binary", True, False, False, "none", "Exact source-type indicator derived from the conservative taxonomy."),
        ("broker_agency_flag", "binary", True, False, False, "none", "Exact source-type indicator derived from the conservative taxonomy."),
        ("social_forum_flag", "binary", True, False, False, "none", "Exact source-type indicator derived from the conservative taxonomy."),
        ("taxonomy_confidence_high_or_medium", "binary", False, True, False, "none", "Use to define the taxonomy-confidence sensitivity subset, not as a substantive causal predictor."),
        ("scrape_success", "binary availability", True, True, False, "none", "Observed scrape-availability control; interpret as non-random missingness, not page quality."),
        ("parse_success", "binary availability", False, True, False, "none", "Diagnostic scrape pipeline status; highly collinear with scrape success."),
        ("scraped_body_available", "binary availability", False, True, False, "none", "Diagnostic content-availability status."),
        ("content_feature_available", "binary availability", True, True, False, "none", "Safe content-subset eligibility flag; Model 4 restricts to true rather than imputing missing content."),
        ("content_features_available", "binary availability", False, True, False, "none", "Stricter medium/strong measurable-content eligibility flag."),
        ("content_quality_flag", "categorical diagnostic", False, True, True, "none", "Use for missingness diagnostics, not as a primary content-quality claim."),
        ("content_strength", "categorical diagnostic", False, True, True, "none", "Use in content-restricted sensitivity models only."),
        ("has_price_or_package", "binary page-content feature", False, True, False, "none", "Structural page feature; analyze only where content_feature_available is true."),
        ("has_contact_info", "binary page-content feature", False, True, False, "none", "Structural page feature; analyze only where content_feature_available is true."),
        ("has_table", "binary page-content feature", False, True, False, "none", "Structural page feature; analyze only where content_feature_available is true."),
        ("answer_text_overlap", "forbidden answer-derived feature", False, False, True, "high", "Answer-derived and excluded from the LPM-ready table."),
        ("page_answer_similarity", "forbidden answer-derived feature", False, False, True, "high", "Answer-derived and excluded from the LPM-ready table."),
        ("answer_like_text", "forbidden answer-derived feature", False, False, True, "high", "Answer-derived and excluded from the LPM-ready table."),
        ("source_group", "forbidden provenance feature", False, False, True, "high", "Source-set membership is not a pre-outcome page characteristic."),
        ("source_origin", "forbidden provenance feature", False, False, True, "high", "Source provenance is not used as a citation predictor."),
        ("source_position", "forbidden position feature", False, True, True, "high", "Allowed only as a clearly labelled diagnostic sensitivity, never a main claim."),
        ("observed_rank", "forbidden position feature", False, True, True, "high", "Allowed only as a clearly labelled diagnostic sensitivity, never a main claim."),
        ("cited_label", "forbidden outcome duplicate", False, False, True, "high", "Outcome duplicate; not a predictor."),
        ("is_more_only", "forbidden outcome complement", False, False, True, "high", "Complement of the outcome; not a predictor."),
    ]
    return pd.DataFrame(rows, columns=["variable_name", "variable_type", "use_in_main_lpm", "use_in_sensitivity_only", "diagnostic_only", "leakage_risk", "reason"])


def lpm_ready_table(df: pd.DataFrame) -> pd.DataFrame:
    source_type = _category(df, "source_type_real_estate")
    confidence = _category(df, "re_page_type_confidence")
    out = pd.DataFrame(
        {
            "cited": _bool(df, "cited"),
            "prompt_id": df.get("prompt_id", pd.Series(MISSING, index=df.index)).map(_clean),
            "normalized_url": df.get("normalized_url", pd.Series(MISSING, index=df.index)).map(_clean),
            "source_root_domain": df.get("source_root_domain", pd.Series(MISSING, index=df.index)).map(_clean),
            "page_type_family_real_estate": _category(df, "page_type_family_real_estate"),
            "source_type_real_estate": source_type,
            "taxonomy_confidence": confidence,
            "taxonomy_confidence_high": confidence.eq("high").astype(int),
            "taxonomy_confidence_medium": confidence.eq("medium").astype(int),
            "taxonomy_confidence_low": confidence.eq("low").astype(int),
            "taxonomy_confidence_unknown": confidence.isin(["unknown", MISSING]).astype(int),
            "taxonomy_confidence_high_or_medium": confidence.isin(["high", "medium"]).astype(int),
            "scrape_success": _bool(df, "scrape_success"),
            "parse_success": _bool(df, "parse_success"),
            "scraped_body_available": _bool(df, "scraped_body_available"),
            "scraped_ok": _bool(df, "scraped_ok"),
            "content_feature_available": _bool(df, "content_feature_available"),
            "content_features_available": _bool(df, "content_features_available"),
            "content_quality_flag": _category(df, "content_quality_flag"),
            "content_strength": _category(df, "content_strength"),
            "content_word_count": _numeric(df, "word_count"),
            "developer_official_flag": source_type.eq("developer_official").astype(int),
            "property_portal_flag": source_type.eq("property_portal").astype(int),
            "broker_agency_flag": source_type.eq("broker_agency").astype(int),
            "social_forum_flag": source_type.eq("social_forum").astype(int),
            "source_type_patch_applied": _bool(df, "source_type_patch_applied"),
        }
    )
    for feature in SAFE_CONTENT_FEATURES:
        out[feature] = _numeric(df, feature)
    forbidden = [column for column in out if any(token in column.casefold() for token in LEAKAGE_TOKENS)]
    if forbidden:
        raise ValueError(f"LPM-ready table unexpectedly includes leakage-risk columns: {forbidden}")
    return out


def _model_design_plan() -> str:
    return """# SCOPE Condo LPM Model Design Plan

## Outcome and framing
The dependent variable is `cited`. These are observational linear probability models (LPMs): coefficients describe conditional associations in the observed citation table, not the hidden retrieval process or causal reasons a source was cited.

## Model 1
`cited ~ C(page_type_family_real_estate) + C(prompt_id)`

Primary descriptive model. Keep `unknown` page type as an explicit reference category; do not drop it.

## Model 2
`cited ~ C(page_type_family_real_estate) + C(source_type_real_estate) + C(prompt_id)`

Adds the conservative source-type taxonomy. Retain residual `unknown` source type as a category and compare estimates with Model 1 because unknown source type remains nontrivial after the exact-domain patch.

## Model 3
Same specification as Model 2, restricted to `taxonomy_confidence_high_or_medium == 1`.

This is a taxonomy-quality sensitivity analysis, not the sole preferred sample.

## Model 4
Content-subset model, restricted to `content_feature_available == 1`. Start with Model 2 and add only structural page features measured from page content, such as `has_price_or_package`, `has_contact_info`, and `has_table`.

Do not impute unavailable content features as zero. Report the availability restriction and compare it with Models 1–3.

## Model 5
Diagnostic-only sensitivity: augment Model 2 with `source_position` or `observed_rank` only if present in the source table.

Do not use this as a main result or causal claim. Position/rank may be downstream of source presentation and is excluded from `scope_condo_lpm_ready.csv`.

## Standard errors and reporting
Use heteroskedasticity-robust standard errors at minimum. Prefer cluster-robust standard errors by `prompt_id`; run `source_root_domain` clustering as a robustness check when the implementation supports it. Report the number of clusters, observations, fixed-effect reference categories, and all sample restrictions. Avoid two-way clustering unless the estimator/version is explicitly verified for it.

## Exclusions
Never use answer text, answer overlap, answer-like text, page-answer similarity, source origin/group, cited-label duplicates, or more-only outcome complements as predictors. These fields are absent from the LPM-ready table by design.
"""


def _readiness_report(before_after: pd.DataFrame, patched: pd.DataFrame, audit: pd.DataFrame) -> str:
    row_source = before_after[(before_after.metric_name.eq("source_type_unknown")) & (before_after.level.eq("row"))].iloc[0]
    url_source = before_after[(before_after.metric_name.eq("source_type_unknown")) & (before_after.level.eq("url"))].iloc[0]
    page_unknown = before_after[(before_after.metric_name.eq("page_type_unknown")) & (before_after.level.eq("url"))].iloc[0]
    confidence = before_after[(before_after.metric_name.eq("high_medium_taxonomy_confidence")) & (before_after.level.eq("url"))].iloc[0]
    rules = int(patched["source_type_patch_applied"].sum())
    domains = int(audit["patch_status"].eq("applied_safe_domain_rule").sum())
    return f"""# SCOPE Condo Final LPM Readiness

## Status
Recommended status: **near_lpm_ready_after_taxonomy_QA**.

The dataset remains ready for first-pass EDA. It is not marked `final_lpm_ready` because the conservative patch intentionally leaves a meaningful unknown source-type category and page-type unknowns in place rather than forcing labels.

## What changed
- Exact reviewed root-domain rules were applied to {rules} source rows across {domains} recurring domains, and only where the prior source type was `unknown`.
- Row-level unknown source type moved from {int(row_source.before_n)}/{int(row_source.denominator)} ({row_source.before_rate:.1%}) to {int(row_source.after_n)}/{int(row_source.denominator)} ({row_source.after_rate:.1%}).
- URL-level unknown source type moved from {int(url_source.before_n)}/{int(url_source.denominator)} ({url_source.before_rate:.1%}) to {int(url_source.after_n)}/{int(url_source.denominator)} ({url_source.after_rate:.1%}).
- Page-type unknown remains {int(page_unknown.after_n)}/{int(page_unknown.denominator)} ({page_unknown.after_rate:.1%}) by design; this patch does not infer page content type from a domain label.
- High/medium taxonomy confidence remains {int(confidence.after_n)}/{int(confidence.denominator)} ({confidence.after_rate:.1%}) because page-taxonomy confidence was not overwritten.

## Main-LPM variables
Use `page_type_family_real_estate`, `source_type_real_estate` (with `unknown` retained), exact source-type flags, prompt fixed effects, and scrape/content availability indicators. The variable dictionary separates these from sensitivity-only and diagnostic fields.

## Diagnostic-only variables
Detailed page labels, taxonomy reasons, titles/excerpts, content-quality labels, source/domain identifiers, and all position/rank fields are not in the main feature set. Answer-derived or outcome-derived variables are forbidden and excluded from the LPM-ready table.

## Remaining caveats
- Scrape/content availability is non-random; content-feature effects must be interpreted only in the content-available subset.
- Source type is still unknown for {int(url_source.after_n)} URL units. Model 2 should retain this category and be compared with Model 1 rather than treated as a fully resolved taxonomy.
- Review the 100-row stratified taxonomy sample and the retained-unknown high-priority domains before final model sign-off.
- This remains an observational citation study; no result identifies an AI system's hidden retrieval or ranking mechanism.
"""


def run_final_lpm_prep(input_path: Path, output_dir: Path) -> dict[str, Any]:
    before = pd.read_csv(input_path, low_memory=False)
    after = apply_safe_source_type_patch(before)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = source_type_unknown_reduction_audit(before, after)
    summary = taxonomy_before_after_summary(before, after)
    sample = taxonomy_manual_review_sample(after)
    dictionary = lpm_variable_dictionary()
    ready = lpm_ready_table(after)
    audit.to_csv(output_dir / "source_type_unknown_reduction_audit.csv", index=False)
    sample.to_csv(output_dir / "taxonomy_manual_review_sample_100.csv", index=False)
    summary.to_csv(output_dir / "taxonomy_before_after_summary.csv", index=False)
    dictionary.to_csv(output_dir / "lpm_variable_dictionary.csv", index=False)
    ready.to_csv(output_dir / "scope_condo_lpm_ready.csv", index=False)
    (output_dir / "lpm_model_design_plan.md").write_text(_model_design_plan(), encoding="utf-8")
    (output_dir / "final_lpm_readiness_report.md").write_text(_readiness_report(summary, after, audit), encoding="utf-8")
    return {
        "input_rows": int(len(before)),
        "lpm_ready_rows": int(len(ready)),
        "safe_domain_rules_applied": int(after["source_type_patch_applied"].sum()),
        "mapped_domains": int(audit["patch_status"].eq("applied_safe_domain_rule").sum()),
        "source_type_unknown_rate_after": float(_category(after, "source_type_real_estate").eq("unknown").mean()),
        "output_dir": str(output_dir),
    }
