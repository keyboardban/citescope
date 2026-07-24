"""Independent, read-only metric verification for the SCOPE EDA-ready CSV."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.real_estate_taxonomy import (
    BROKER_DOMAINS,
    DEVELOPER_DOMAINS,
    INVESTMENT_DOMAINS,
    LISTING_MARKETPLACE_DOMAINS,
    MAP_DOMAINS,
    NEIGHBORHOOD_DOMAINS,
    NEWS_DOMAINS,
    PROJECT_DOMAINS,
    PROPERTY_PORTAL_DOMAINS,
    REVIEW_DOMAINS,
    SOCIAL_DOMAINS,
    VIDEO_DOMAINS,
)


MISSING = "<missing>"
EXPECTED_COLUMNS = (
    "prompt_id",
    "source_url",
    "normalized_url",
    "cited",
    "cited_label",
    "scrape_success",
    "parse_success",
    "scraped_body_available",
    "content_quality_flag",
    "word_count",
    "text_char_count",
    "page_type_family_real_estate",
    "page_type_detail_real_estate",
    "page_type_final_real_estate",
    "page_type_final_real_estate_source",
    "source_type_real_estate",
    "re_page_type_confidence",
)
BOOLEAN_COLUMNS = ("cited", "scrape_success", "parse_success", "scraped_body_available")
QUALITY_PRIORITY = {
    "ok": 8,
    "very_short_text": 7,
    "dynamic_js_likely": 6,
    "nav_footer_only": 5,
    "boilerplate_only": 5,
    "blocked_or_error_page": 4,
    "parse_failed": 3,
    "empty_text": 2,
    "unknown": 1,
    "no_raw_cache": 0,
    MISSING: 0,
}
MEDICAL_STYLE_LABELS = frozenset(
    {
        "article_health_info",
        "service_or_treatment_page",
        "department_or_center_page",
        "disease_condition_page",
        "doctor_profile",
        "treatment_page",
        "appointment_page",
        "hospital",
        "clinic",
    }
)
KNOWN_REAL_ESTATE_DOMAIN_TOKENS = tuple(
    sorted(
        set(
            DEVELOPER_DOMAINS
            + PROJECT_DOMAINS
            + PROPERTY_PORTAL_DOMAINS
            + LISTING_MARKETPLACE_DOMAINS
            + BROKER_DOMAINS
            + REVIEW_DOMAINS
            + INVESTMENT_DOMAINS
            + NEIGHBORHOOD_DOMAINS
            + SOCIAL_DOMAINS
            + VIDEO_DOMAINS
            + NEWS_DOMAINS
            + MAP_DOMAINS
        )
    )
)


@dataclass(frozen=True)
class ReportedMetric:
    name: str
    level: str
    reported_numerator: float
    reported_denominator: float | None
    reported_display: str
    recomputed_key: str
    notes: str = ""


def _clean_scalar(value: Any) -> str:
    if value is None or pd.isna(value):
        return MISSING
    value = str(value).strip()
    return value if value else MISSING


def _category(df: pd.DataFrame, column: str, lower: bool = True) -> pd.Series:
    if column not in df:
        return pd.Series(MISSING, index=df.index, dtype=object)
    result = df[column].map(_clean_scalar)
    return result.str.casefold() if lower else result


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normalise_boolean(df: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Return nullable booleans and an invalid-token mask without coercing blanks."""
    raw = _category(df, column)
    true_tokens = {"true", "1", "1.0", "yes", "y"}
    false_tokens = {"false", "0", "0.0", "no", "n"}
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    result.loc[raw.isin(true_tokens)] = True
    result.loc[raw.isin(false_tokens)] = False
    invalid = ~raw.isin(true_tokens | false_tokens | {MISSING})
    return result, invalid


def _truth(df: pd.DataFrame, column: str) -> pd.Series:
    return normalise_boolean(df, column)[0].fillna(False).astype(bool)


def _value_row(metric_name: str, value: Any, numerator: Any = "", denominator: Any = "", notes: str = "") -> dict[str, Any]:
    return {
        "metric_name": metric_name,
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "notes": notes,
    }


def input_schema_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"audit_item": "total_columns", "value": int(len(df.columns)), "notes": ""},
        {"audit_item": "row_count", "value": int(len(df)), "notes": ""},
        {"audit_item": "duplicated_columns_n", "value": int(df.columns.duplicated().sum()), "notes": ""},
        {"audit_item": "duplicated_rows_n", "value": int(df.duplicated().sum()), "notes": "Exact duplicate CSV rows."},
        {"audit_item": "unique_normalized_urls_n", "value": int(_category(df, "normalized_url", lower=False).nunique()), "notes": "Blank values, if any, are grouped as <missing>."},
        {"audit_item": "unique_source_urls_n", "value": int(_category(df, "source_url", lower=False).nunique()), "notes": "Blank values, if any, are grouped as <missing>."},
        {"audit_item": "unique_prompt_ids_n", "value": int(_category(df, "prompt_id", lower=False).nunique()), "notes": "Blank values, if any, are grouped as <missing>."},
    ]
    for expected in EXPECTED_COLUMNS:
        rows.append(
            {
                "audit_item": "expected_column",
                "value": expected,
                "notes": "present" if expected in df.columns else "MISSING",
            }
        )
    for position, column in enumerate(df.columns, start=1):
        rows.append({"audit_item": "input_column", "value": column, "notes": f"position={position}"})
    return pd.DataFrame(rows)


def row_level_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    n = len(df)
    cited, cited_invalid = normalise_boolean(df, "cited")
    scrape, scrape_invalid = normalise_boolean(df, "scrape_success")
    parse, parse_invalid = normalise_boolean(df, "parse_success")
    body, body_invalid = normalise_boolean(df, "scraped_body_available")
    quality = _category(df, "content_quality_flag")
    family = _category(df, "page_type_family_real_estate")
    detail = _category(df, "page_type_detail_real_estate")
    source_type = _category(df, "source_type_real_estate")
    confidence = _category(df, "re_page_type_confidence")
    word_count = _numeric(df, "word_count")
    usable = quality.eq("ok") & word_count.ge(300)
    values: dict[str, float] = {
        "total_rows": float(n),
        "cited_rows": float(cited.eq(True).sum()),
        "more_only_rows": float(cited.eq(False).sum()),
        "cited_rate": float(cited.eq(True).mean()) if n else np.nan,
        "scrape_success_n": float(scrape.eq(True).sum()),
        "scrape_success_rate": float(scrape.eq(True).mean()) if n else np.nan,
        "parse_success_n": float(parse.eq(True).sum()),
        "parse_success_rate": float(parse.eq(True).mean()) if n else np.nan,
        "scraped_body_available_n": float(body.eq(True).sum()),
        "scraped_body_available_rate": float(body.eq(True).mean()) if n else np.nan,
        "content_quality_ok_n": float(quality.eq("ok").sum()),
        "content_quality_ok_rate": float(quality.eq("ok").mean()) if n else np.nan,
        "usable_content_n": float(usable.sum()),
        "usable_content_rate": float(usable.mean()) if n else np.nan,
        "page_type_family_unknown_n": float(family.eq("unknown").sum()),
        "page_type_family_unknown_rate": float(family.eq("unknown").mean()) if n else np.nan,
        "page_type_detail_unknown_n": float(detail.eq("unknown").sum()),
        "page_type_detail_unknown_rate": float(detail.eq("unknown").mean()) if n else np.nan,
        "source_type_unknown_n": float(source_type.eq("unknown").sum()),
        "source_type_unknown_rate": float(source_type.eq("unknown").mean()) if n else np.nan,
        "high_confidence_n": float(confidence.eq("high").sum()),
        "medium_confidence_n": float(confidence.eq("medium").sum()),
        "low_confidence_n": float(confidence.eq("low").sum()),
        "unknown_confidence_n": float(confidence.eq("unknown").sum()),
        "high_or_medium_confidence_n": float(confidence.isin(["high", "medium"]).sum()),
        "high_or_medium_confidence_rate": float(confidence.isin(["high", "medium"]).mean()) if n else np.nan,
    }
    rows = []
    count_metrics = {name for name in values if name.endswith("_n") or name.endswith("_rows") or name == "total_rows"}
    for name, value in values.items():
        denominator = n if name not in count_metrics else ""
        numerator = value if name not in {"total_rows"} else ""
        rows.append(_value_row(name, int(value) if name in count_metrics else value, numerator, denominator))
    for name, series, invalid in (
        ("cited", cited, cited_invalid),
        ("scrape_success", scrape, scrape_invalid),
        ("parse_success", parse, parse_invalid),
        ("scraped_body_available", body, body_invalid),
    ):
        rows.append(_value_row(f"{name}_blank_or_nan_n", int(series.isna().sum()), notes="Blank/NaN boolean values are not counted as true or false."))
        rows.append(_value_row(f"{name}_unrecognized_token_n", int(invalid.sum()), notes="Nonblank values outside accepted true/false tokens."))
    rows.append(_value_row("word_count_missing_n", int(word_count.isna().sum())))
    return pd.DataFrame(rows), values


def _mode_with_tie(values: Iterable[Any]) -> tuple[str, bool, str]:
    cleaned = [_clean_scalar(value).casefold() for value in values]
    nonmissing = [value for value in cleaned if value != MISSING]
    if not nonmissing:
        return MISSING, False, ""
    counts = pd.Series(nonmissing, dtype=object).value_counts()
    winners = sorted(counts[counts.eq(counts.max())].index.tolist())
    return winners[0], len(winners) > 1, " | ".join(winners)


def _best_quality(values: Iterable[Any]) -> str:
    cleaned = [_clean_scalar(value).casefold() for value in values]
    if not cleaned:
        return MISSING
    return sorted(cleaned, key=lambda value: (-QUALITY_PRIORITY.get(value, 1), value))[0]


def aggregate_urls(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_url_key"] = _category(work, "normalized_url", lower=False)
    work["_cited"] = _truth(work, "cited")
    work["_scrape"] = _truth(work, "scrape_success")
    work["_parse"] = _truth(work, "parse_success")
    work["_body"] = _truth(work, "scraped_body_available")
    work["_word_count"] = _numeric(work, "word_count")
    rows: list[dict[str, Any]] = []
    taxonomy_columns = (
        "page_type_family_real_estate",
        "page_type_detail_real_estate",
        "page_type_final_real_estate",
        "page_type_final_real_estate_source",
        "source_type_real_estate",
        "re_page_type_confidence",
    )
    for url, group in work.groupby("_url_key", sort=False, dropna=False):
        numeric_words = group["_word_count"].dropna()
        record: dict[str, Any] = {
            "normalized_url": url,
            "source_url": _clean_scalar(group.get("source_url", pd.Series(dtype=object)).replace("", np.nan).dropna().iloc[0]) if "source_url" in group and group["source_url"].replace("", np.nan).notna().any() else MISSING,
            "source_root_domain": _clean_scalar(group.get("source_root_domain", pd.Series(dtype=object)).replace("", np.nan).dropna().iloc[0]) if "source_root_domain" in group and group["source_root_domain"].replace("", np.nan).notna().any() else MISSING,
            "source_row_count": int(len(group)),
            "cited_url": bool(group["_cited"].any()),
            "scrape_success": bool(group["_scrape"].any()),
            "parse_success": bool(group["_parse"].any()),
            "scraped_body_available": bool(group["_body"].any()),
            "word_count_max": float(numeric_words.max()) if not numeric_words.empty else np.nan,
            "content_quality_flag": _best_quality(group.get("content_quality_flag", pd.Series(dtype=object))),
        }
        for column in taxonomy_columns:
            value, tied, tie_values = _mode_with_tie(group.get(column, pd.Series(dtype=object)))
            record[column] = value
            record[f"{column}_tied"] = tied
            record[f"{column}_tie_values"] = tie_values
        rows.append(record)
    return pd.DataFrame(rows)


def url_level_metrics(urls: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    n = len(urls)
    quality = _category(urls, "content_quality_flag")
    family = _category(urls, "page_type_family_real_estate")
    source_type = _category(urls, "source_type_real_estate")
    confidence = _category(urls, "re_page_type_confidence")
    words = _numeric(urls, "word_count_max")
    usable = quality.eq("ok") & words.ge(300)
    values: dict[str, float] = {
        "unique_urls": float(n),
        "cited_unique_urls": float(_truth(urls, "cited_url").sum()),
        "more_only_unique_urls": float((~_truth(urls, "cited_url")).sum()),
        "scrape_success_unique_url_n": float(_truth(urls, "scrape_success").sum()),
        "parse_success_unique_url_n": float(_truth(urls, "parse_success").sum()),
        "scraped_body_available_unique_url_n": float(_truth(urls, "scraped_body_available").sum()),
        "content_quality_ok_unique_url_n": float(quality.eq("ok").sum()),
        "usable_content_unique_url_n": float(usable.sum()),
        "median_word_count": float(words.median()) if words.notna().any() else np.nan,
        "p25_word_count": float(words.quantile(0.25)) if words.notna().any() else np.nan,
        "p75_word_count": float(words.quantile(0.75)) if words.notna().any() else np.nan,
        "mean_word_count": float(words.mean()) if words.notna().any() else np.nan,
        "page_type_family_unknown_unique_url_n": float(family.eq("unknown").sum()),
        "page_type_family_unknown_unique_url_rate": float(family.eq("unknown").mean()) if n else np.nan,
        "source_type_unknown_unique_url_n": float(source_type.eq("unknown").sum()),
        "source_type_unknown_unique_url_rate": float(source_type.eq("unknown").mean()) if n else np.nan,
        "high_or_medium_confidence_unique_url_n": float(confidence.isin(["high", "medium"]).sum()),
        "high_or_medium_confidence_unique_url_rate": float(confidence.isin(["high", "medium"]).mean()) if n else np.nan,
    }
    rows = []
    counts = {name for name in values if name.endswith("_n") or name.endswith("_urls")}
    for name, value in values.items():
        rows.append(_value_row(name, int(value) if name in counts else value, value if name not in counts else "", n if name not in counts else ""))
    for column in (
        "page_type_family_real_estate",
        "page_type_detail_real_estate",
        "page_type_final_real_estate",
        "page_type_final_real_estate_source",
        "source_type_real_estate",
        "re_page_type_confidence",
    ):
        rows.append(_value_row(f"{column}_tied_unique_url_n", int(urls.get(f"{column}_tied", pd.Series(False, index=urls.index)).sum()), notes="The URL-level value uses a deterministic lexical tie break; see url_level_taxonomy_resolution_audit.csv."))
    return pd.DataFrame(rows), values


def _distribution(df: pd.DataFrame, column: str, include_cited: bool) -> pd.DataFrame:
    work = df.copy()
    work[column] = _category(work, column)
    work["_cited"] = _truth(work, "cited")
    work["_url"] = _category(work, "normalized_url", lower=False)
    out = (
        work.groupby(column, dropna=False)
        .agg(row_count=(column, "size"), unique_url_count=("_url", "nunique"), cited_rows=("_cited", "sum"), cited_rate=("_cited", "mean"))
        .reset_index()
    )
    out["row_rate"] = out["row_count"] / len(work) if len(work) else np.nan
    out["unique_url_rate"] = out["unique_url_count"] / work["_url"].nunique() if len(work) else np.nan
    if not include_cited:
        out = out.drop(columns=["cited_rows", "cited_rate"])
    return out.sort_values("row_count", ascending=False, kind="stable")


def content_quality_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _distribution(df, "content_quality_flag", include_cited=False)


def inconsistency_audit(df: pd.DataFrame, urls: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_row_index"] = work.index
    scrape = _truth(work, "scrape_success")
    parse = _truth(work, "parse_success")
    body = _truth(work, "scraped_body_available")
    quality = _category(work, "content_quality_flag")
    words = _numeric(work, "word_count")
    family = _category(work, "page_type_family_real_estate")
    detail = _category(work, "page_type_detail_real_estate")
    source_type = _category(work, "source_type_real_estate")
    confidence = _category(work, "re_page_type_confidence")
    domain = _category(work, "source_root_domain")
    taxonomy_columns = (
        "page_type_family_real_estate",
        "page_type_detail_real_estate",
        "page_type_final_real_estate",
        "page_type_final_real_estate_source",
        "source_type_real_estate",
        "re_page_type_confidence",
    )
    ties = urls.set_index("normalized_url")[[f"{column}_tied" for column in taxonomy_columns]].any(axis=1)
    url_key = _category(work, "normalized_url", lower=False)
    medical_fields = [column for column in ("page_type_family_real_estate", "page_type_detail_real_estate", "page_type_final_real_estate", "page_type_final") if column in work]
    medical = pd.Series(False, index=work.index)
    for column in medical_fields:
        medical |= _category(work, column).isin(MEDICAL_STYLE_LABELS)
    detail_source_conflict = pd.Series(False, index=work.index)
    allowed = {
        "developer_brand_page": {"developer_official", "project_official"},
        "condo_project_page": {"developer_official", "project_official", "property_portal", "listing_marketplace"},
        "project_listing_page": {"property_portal", "listing_marketplace", "broker_agency"},
        "resale_listing_page": {"property_portal", "listing_marketplace", "broker_agency"},
        "rental_listing_page": {"property_portal", "listing_marketplace", "broker_agency"},
        "broker_property_page": {"broker_agency", "property_portal", "listing_marketplace"},
        "forum_discussion": {"social_forum"},
        "video_page": {"video_platform", "social_forum"},
        "pdf_brochure": {"pdf_document", "developer_official", "project_official"},
    }
    for page_detail, allowed_types in allowed.items():
        detail_source_conflict |= detail.eq(page_detail) & ~source_type.isin(allowed_types | {"unknown", MISSING})
    known_domain = domain.map(lambda value: value != MISSING and any(token in value for token in KNOWN_REAL_ESTATE_DOMAIN_TOKENS))
    flags = {
        "scrape_success_true_parse_success_false": scrape & ~parse,
        "parse_success_true_body_unavailable": parse & ~body,
        "content_quality_ok_word_count_lt_300": quality.eq("ok") & words.lt(300),
        "usable_content_true_quality_not_ok": (quality.eq("ok") & words.ge(300)) & ~quality.eq("ok"),
        "word_count_missing_quality_ok": words.isna() & quality.eq("ok"),
        "known_family_unknown_confidence": family.ne("unknown") & family.ne(MISSING) & confidence.eq("unknown"),
        "unknown_family_high_confidence": family.eq("unknown") & confidence.eq("high"),
        "unknown_source_type_known_real_estate_domain": source_type.eq("unknown") & known_domain,
        "medical_style_page_type": medical,
        "detail_source_type_contradiction": detail_source_conflict,
        "duplicate_normalized_url_conflicting_taxonomy": url_key.map(ties).fillna(False),
    }
    flag_frame = pd.DataFrame(flags, index=work.index)
    selected = flag_frame.any(axis=1)
    wanted = [
        "_row_index", "prompt_id", "source_url", "normalized_url", "source_root_domain", "source_title", "page_title", "cited",
        "scrape_success", "parse_success", "scraped_body_available", "content_quality_flag", "word_count",
        "source_type_real_estate", "page_type_family_real_estate", "page_type_detail_real_estate",
        "page_type_final_real_estate", "re_page_type_confidence", "re_page_type_reason",
    ]
    out = work[[column for column in wanted if column in work]].loc[selected].copy()
    for column in flag_frame:
        out[column] = flag_frame.loc[selected, column].astype(int)
    out["inconsistency_flags"] = flag_frame.loc[selected].apply(lambda row: ";".join(row.index[row].tolist()), axis=1)
    return out


def _top_value(values: Iterable[Any]) -> str:
    return _mode_with_tie(values)[0]


def domain_level_quality_audit(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_domain"] = _category(work, "source_root_domain")
    work["_url"] = _category(work, "normalized_url", lower=False)
    work["_cited"] = _truth(work, "cited")
    work["_scrape"] = _truth(work, "scrape_success")
    work["_parse"] = _truth(work, "parse_success")
    work["_word_count"] = _numeric(work, "word_count")
    work["_quality"] = _category(work, "content_quality_flag")
    work["_source_type"] = _category(work, "source_type_real_estate")
    work["_family"] = _category(work, "page_type_family_real_estate")
    work["_confidence"] = _category(work, "re_page_type_confidence")
    work["_usable"] = work["_quality"].eq("ok") & work["_word_count"].ge(300)
    rows = []
    for domain, group in work.groupby("_domain", sort=False):
        count = len(group)
        unknown_source = float(group["_source_type"].eq("unknown").mean())
        unknown_page = float(group["_family"].eq("unknown").mean())
        low_confidence = float(group["_confidence"].isin(["low", "unknown", MISSING]).mean())
        usable = float(group["_usable"].mean())
        issue = max(unknown_source, unknown_page, low_confidence, 1 - usable)
        if count >= 10 and (issue >= 0.40 or usable < 0.40):
            priority = "high"
        elif count >= 5 and (issue >= 0.20 or usable < 0.60):
            priority = "medium"
        else:
            priority = "low"
        rows.append(
            {
                "source_root_domain": domain,
                "row_count": int(count),
                "unique_url_count": int(group["_url"].nunique()),
                "cited_rows": int(group["_cited"].sum()),
                "cited_rate": float(group["_cited"].mean()),
                "scrape_success_rate": float(group["_scrape"].mean()),
                "parse_success_rate": float(group["_parse"].mean()),
                "usable_content_rate": usable,
                "median_word_count": float(group["_word_count"].median()) if group["_word_count"].notna().any() else np.nan,
                "top_content_quality_flag": _top_value(group["_quality"]),
                "top_source_type_real_estate": _top_value(group["_source_type"]),
                "top_page_type_family_real_estate": _top_value(group["_family"]),
                "unknown_source_type_rate": unknown_source,
                "unknown_page_type_rate": unknown_page,
                "low_confidence_rate": low_confidence,
                "recommended_review_priority": priority,
            }
        )
    return pd.DataFrame(rows).sort_values(["recommended_review_priority", "row_count"], ascending=[True, False], key=lambda series: series.map({"high": 0, "medium": 1, "low": 2}) if series.name == "recommended_review_priority" else series, kind="stable")


def manual_review_sample(df: pd.DataFrame, inconsistencies: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    work = df.copy()
    work["_row_index"] = work.index
    cited = _truth(work, "cited")
    confidence = _category(work, "re_page_type_confidence")
    family = _category(work, "page_type_family_real_estate")
    source_type = _category(work, "source_type_real_estate")
    top_domains = _category(work, "source_root_domain").value_counts().head(5).index
    inconsistent_indexes = set(inconsistencies.get("_row_index", pd.Series(dtype=int)).tolist())
    strata = (
        ("cited_high_confidence", cited & confidence.eq("high"), 20),
        ("cited_low_or_unknown_confidence", cited & confidence.isin(["low", "unknown", MISSING]), 20),
        ("unknown_page_type", family.eq("unknown"), 20),
        ("unknown_source_type", source_type.eq("unknown"), 20),
        ("top_domain", _category(work, "source_root_domain").isin(top_domains), 10),
        ("inconsistent_row", work["_row_index"].isin(inconsistent_indexes), 10),
    )
    chosen: list[int] = []
    reasons: dict[int, list[str]] = {}
    for sequence, (reason, mask, target) in enumerate(strata):
        candidates = work.loc[mask & ~work["_row_index"].isin(chosen), "_row_index"]
        take = min(target, len(candidates))
        if take:
            selected = pd.Series(candidates).sample(n=take, random_state=20260713 + sequence).tolist()
            chosen.extend(selected)
            for index in selected:
                reasons.setdefault(index, []).append(reason)
    if len(chosen) < n:
        candidates = work.loc[~work["_row_index"].isin(chosen), "_row_index"]
        take = min(n - len(chosen), len(candidates))
        if take:
            selected = pd.Series(candidates).sample(n=take, random_state=20260800).tolist()
            chosen.extend(selected)
            for index in selected:
                reasons.setdefault(index, []).append("coverage_fill")
    selected = work.set_index("_row_index").loc[chosen].reset_index()
    title = selected.get("page_title", pd.Series(MISSING, index=selected.index)).map(_clean_scalar)
    if "source_title" in selected:
        title = title.where(title.ne(MISSING), selected["source_title"].map(_clean_scalar))
    columns = [
        "prompt_id", "source_url", "source_root_domain", "cited", "content_quality_flag", "word_count",
        "source_type_real_estate", "page_type_family_real_estate", "page_type_detail_real_estate",
        "re_page_type_confidence", "re_page_type_reason",
    ]
    out = selected[[column for column in columns if column in selected]].copy()
    out.insert(3, "title", title)
    out.insert(4, "page_text_excerpt", selected.get("page_text_excerpt", pd.Series(MISSING, index=selected.index)).map(_clean_scalar))
    out["review_reason"] = selected["_row_index"].map(lambda index: ";".join(reasons.get(index, ["coverage_fill"])))
    return out


def _comparison_status(reported: ReportedMetric, recomputed: float, denominator: float | None) -> tuple[str, float, float]:
    if pd.isna(recomputed):
        return "cannot_verify", np.nan, np.nan
    if reported.reported_denominator is None:
        absolute = abs(recomputed - reported.reported_numerator)
        relative = absolute / abs(reported.reported_numerator) if reported.reported_numerator else np.nan
        return ("match" if absolute == 0 else "mismatch"), absolute, relative
    recomputed_rate = recomputed / denominator if denominator else np.nan
    reported_rate = reported.reported_numerator / reported.reported_denominator
    absolute = abs(recomputed_rate - reported_rate)
    relative = absolute / reported_rate if reported_rate else np.nan
    if recomputed == reported.reported_numerator and denominator == reported.reported_denominator:
        return "match", absolute, relative
    if absolute <= 0.0005:
        return "minor_rounding_difference", absolute, relative
    return "mismatch", absolute, relative


def reported_vs_recomputed(row_values: dict[str, float], url_values: dict[str, float]) -> pd.DataFrame:
    reported = (
        ReportedMetric("row_level_source_appearances", "row_level", 1139, None, "1,139", "total_rows"),
        ReportedMetric("unique_urls", "url_level", 846, None, "846", "unique_urls"),
        ReportedMetric("cited_rows", "row_level", 411, None, "411", "cited_rows"),
        ReportedMetric("more_only_rows", "row_level", 728, None, "728", "more_only_rows"),
        ReportedMetric("cited_rate", "row_level", 361, 1000, "36.1%", "cited_rows", "Reported as a rounded row-level rate."),
        ReportedMetric("scrape_success", "url_level", 743, 846, "743/846 (87.8%)", "scrape_success_unique_url_n"),
        ReportedMetric("parse_success", "url_level", 738, 846, "738/846 (87.2%)", "parse_success_unique_url_n"),
        ReportedMetric("scraped_body_available", "url_level", 738, 846, "738/846 (87.2%)", "scraped_body_available_unique_url_n"),
        ReportedMetric("usable_content", "url_level", 498, 846, "498/846 (58.9%)", "usable_content_unique_url_n", "Recomputed using content_quality_flag == ok and max URL word_count >= 300."),
        ReportedMetric("page_type_family_unknown", "row_level", 191, 1139, "191/1,139 (16.8%)", "page_type_family_unknown_n"),
        ReportedMetric("page_type_family_unknown", "url_level", 159, 846, "159/846 (18.8%)", "page_type_family_unknown_unique_url_n"),
        ReportedMetric("source_type_unknown", "row_level", 367, 1139, "367/1,139 (32.2%)", "source_type_unknown_n"),
        ReportedMetric("source_type_unknown", "url_level", 304, 846, "304/846 (35.9%)", "source_type_unknown_unique_url_n"),
        ReportedMetric("high_confidence", "row_level", 386, 1000, "38.6%", "high_confidence_n", "Reported only as a rounded rate."),
        ReportedMetric("medium_confidence", "row_level", 411, 1000, "41.1%", "medium_confidence_n", "Reported only as a rounded rate."),
        ReportedMetric("low_confidence", "row_level", 120, 1000, "12.0%", "low_confidence_n", "Reported only as a rounded rate."),
        ReportedMetric("unknown_confidence", "row_level", 83, 1000, "8.3%", "unknown_confidence_n", "Reported only as a rounded rate."),
        ReportedMetric("high_or_medium_confidence", "row_level", 797, 1000, "79.7%", "high_or_medium_confidence_n", "Reported only as a rounded rate."),
        ReportedMetric("high_or_medium_confidence", "url_level", 772, 1000, "77.2%", "high_or_medium_confidence_unique_url_n", "Reported only as a rounded rate."),
    )
    rows = []
    for metric in reported:
        values = row_values if metric.level == "row_level" else url_values
        recomputed = values.get(metric.recomputed_key, np.nan)
        denominator = values.get("total_rows") if metric.level == "row_level" else values.get("unique_urls")
        status, absolute, relative = _comparison_status(metric, recomputed, denominator)
        if metric.reported_denominator is None:
            recomputed_display = str(int(recomputed)) if not pd.isna(recomputed) else ""
        elif metric.reported_denominator == 1000:
            recomputed_display = f"{recomputed / denominator:.1%}" if denominator else ""
        else:
            recomputed_display = f"{int(recomputed)}/{int(denominator)} ({recomputed / denominator:.1%})" if denominator else ""
        rows.append(
            {
                "metric_name": metric.name,
                "reported_value": metric.reported_display,
                "recomputed_value": recomputed_display,
                "level": metric.level,
                "absolute_difference": absolute,
                "relative_difference": relative,
                "status": status,
                "notes": metric.notes,
            }
        )
    return pd.DataFrame(rows)


def _report_text(
    row_values: dict[str, float],
    url_values: dict[str, float],
    comparison: pd.DataFrame,
    inconsistencies: pd.DataFrame,
) -> str:
    mismatches = comparison[comparison["status"].eq("mismatch")]
    rounded = comparison[comparison["status"].eq("minor_rounding_difference")]
    exact = comparison[comparison["status"].eq("match")]
    core_matches = comparison[comparison["metric_name"].isin(["row_level_source_appearances", "unique_urls", "cited_rows", "more_only_rows", "scrape_success", "parse_success", "scraped_body_available", "usable_content"])]
    ready = core_matches["status"].isin(["match", "minor_rounding_difference"]).all()
    recommendation = "ready_for_first_pass_eda" if ready else "needs_taxonomy_review"
    metric_label = lambda row: f"{row.metric_name} ({row.level})"
    exact_labels = ", ".join(metric_label(row) for row in exact.itertuples(index=False)) or "none"
    rounded_labels = ", ".join(metric_label(row) for row in rounded.itertuples(index=False)) or "none"
    mismatch_labels = ", ".join(metric_label(row) for row in mismatches.itertuples(index=False)) or "none"
    flag_count = lambda column: int(pd.to_numeric(inconsistencies.get(column, pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    legacy_medical = flag_count("medical_style_page_type")
    short_ok = flag_count("content_quality_ok_word_count_lt_300")
    scrape_parse_gap = flag_count("scrape_success_true_parse_success_false")
    known_family_unknown_confidence = flag_count("known_family_unknown_confidence")
    tied_taxonomy = flag_count("duplicate_normalized_url_conflicting_taxonomy")
    lines = [
        "# SCOPE Condo Metric Recheck",
        "",
        "## Verdict",
        f"Final recommendation: **{recommendation}**.",
        "",
        "## Recomputed headline metrics",
        f"- Total source rows: {int(row_values['total_rows'])}",
        f"- Unique normalized URLs: {int(url_values['unique_urls'])}",
        f"- Cited rows: {int(row_values['cited_rows'])} ({row_values['cited_rate']:.1%})",
        f"- URL-level scrape success: {int(url_values['scrape_success_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['scrape_success_unique_url_n'] / url_values['unique_urls']:.1%})",
        f"- URL-level parse success: {int(url_values['parse_success_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['parse_success_unique_url_n'] / url_values['unique_urls']:.1%})",
        f"- URL-level usable content: {int(url_values['usable_content_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['usable_content_unique_url_n'] / url_values['unique_urls']:.1%})",
        f"- URL-level unknown page type: {int(url_values['page_type_family_unknown_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['page_type_family_unknown_unique_url_rate']:.1%})",
        f"- URL-level unknown source type: {int(url_values['source_type_unknown_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['source_type_unknown_unique_url_rate']:.1%})",
        f"- URL-level high/medium taxonomy confidence: {int(url_values['high_or_medium_confidence_unique_url_n'])}/{int(url_values['unique_urls'])} ({url_values['high_or_medium_confidence_unique_url_rate']:.1%})",
        "",
        "## Verification",
        f"- Exact matches: {int(comparison['status'].eq('match').sum())}",
        f"  {exact_labels}",
        f"- Rounding-only differences: {int(len(rounded))}",
        f"  {rounded_labels}",
        f"- Mismatches: {int(len(mismatches))}",
        f"  {mismatch_labels}",
        "- The comparison file records every prior headline metric and its denominator. No prior metric was accepted without a direct CSV recomputation.",
        "",
        "## Denominators",
        "- Citation counts/rates and the row-level taxonomy/confidence distributions use 1,139 source appearances.",
        "- Scrape, parse, usable-content, and URL-level taxonomy figures use the one-normalized-URL-per-unit aggregation (846 URLs in the reported summary).",
        "- URL aggregation takes any successful scrape/parse/body value, the maximum word count, the stated content-quality priority, and a most-frequent non-null taxonomy label. Ties are explicitly retained in `url_level_taxonomy_resolution_audit.csv`.",
        "- No verified prior headline metric mixes a row-level numerator with a URL-level denominator. The audit labels the level of every comparison row explicitly.",
        "",
        "## EDA readiness",
        "- Safe for first-pass EDA: `cited`, prompt/domain identifiers, normalized URL, broad real-estate page family, scrape/parse/body flags, content quality, word count, and confidence, with their documented denominators.",
        "- Diagnostic only: page excerpt/title/reason fields, detailed taxonomy labels, URL-level tie indicators, and all inconsistency flags. Content-derived analyses should be restricted to measurable, sufficiently strong content rather than treating missing scraped content as random.",
        "- Do not include answer text, source position, rank, or answer-derived similarity variables in the main LPM; they are not part of this verification and may create leakage.",
        f"- Inconsistency audit rows: {int(len(inconsistencies))}. Review `inconsistency_audit.csv` and the high-priority domains before a final LPM.",
        f"- QA priorities: {legacy_medical} rows retain a legacy generic medical-style page-type label, so those legacy generic page-type columns must remain diagnostic-only; {short_ok} rows have `content_quality_flag=ok` but fewer than 300 words, which is not a contradiction but excludes them from the defined usable-content subset; {scrape_parse_gap} rows scraped successfully but did not parse; {known_family_unknown_confidence} rows have a known broad family with unknown confidence; {tied_taxonomy} rows have duplicated-URL taxonomy conflicts.",
        "",
        "## Final interpretation",
    ]
    if ready:
        lines.append("The dataset is ready for first-pass EDA if the recomputed scrape and taxonomy metrics match the reported values, but source_type_real_estate unknown rate and low-confidence taxonomy rows should be reviewed before final LPM.")
    else:
        lines.append("The direct recomputation found one or more substantive headline mismatches. Resolve the flagged definitions or source rows before treating the table as EDA-ready.")
    return "\n".join(lines) + "\n"


def run_metric_recheck(input_path: Path, output_dir: Path) -> dict[str, Any]:
    """Write an independent audit without mutating the input CSV or parent EDA outputs."""
    df = pd.read_csv(input_path, low_memory=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = input_schema_audit(df)
    row_metrics, row_values = row_level_metrics(df)
    urls = aggregate_urls(df)
    url_metrics, url_values = url_level_metrics(urls)
    comparison = reported_vs_recomputed(row_values, url_values)
    inconsistencies = inconsistency_audit(df, urls)
    distributions = {
        "content_quality_distribution.csv": content_quality_distribution(df),
        "page_type_family_distribution.csv": _distribution(df, "page_type_family_real_estate", include_cited=True),
        "page_type_detail_distribution.csv": _distribution(df, "page_type_detail_real_estate", include_cited=True),
        "source_type_real_estate_distribution.csv": _distribution(df, "source_type_real_estate", include_cited=True),
        "confidence_distribution.csv": _distribution(df, "re_page_type_confidence", include_cited=False),
    }
    schema.to_csv(output_dir / "input_schema_audit.csv", index=False)
    row_metrics.to_csv(output_dir / "row_level_metric_recheck.csv", index=False)
    url_metrics.to_csv(output_dir / "url_level_metric_recheck.csv", index=False)
    comparison.to_csv(output_dir / "reported_vs_recomputed_metrics.csv", index=False)
    urls.to_csv(output_dir / "url_level_taxonomy_resolution_audit.csv", index=False)
    for filename, table in distributions.items():
        table.to_csv(output_dir / filename, index=False)
    inconsistencies.to_csv(output_dir / "inconsistency_audit.csv", index=False)
    domain_level_quality_audit(df).to_csv(output_dir / "domain_level_quality_audit.csv", index=False)
    manual_review_sample(df, inconsistencies).to_csv(output_dir / "manual_review_sample_100.csv", index=False)
    (output_dir / "scope_condo_metric_recheck_report.md").write_text(_report_text(row_values, url_values, comparison, inconsistencies), encoding="utf-8")
    return {
        "total_rows": int(row_values["total_rows"]),
        "unique_urls": int(url_values["unique_urls"]),
        "cited_rows": int(row_values["cited_rows"]),
        "cited_rate": float(row_values["cited_rate"]),
        "scrape_success_url_rate": float(url_values["scrape_success_unique_url_n"] / url_values["unique_urls"]),
        "parse_success_url_rate": float(url_values["parse_success_unique_url_n"] / url_values["unique_urls"]),
        "usable_content_url_rate": float(url_values["usable_content_unique_url_n"] / url_values["unique_urls"]),
        "page_type_unknown_url_rate": float(url_values["page_type_family_unknown_unique_url_rate"]),
        "source_type_unknown_url_rate": float(url_values["source_type_unknown_unique_url_rate"]),
        "high_medium_confidence_url_rate": float(url_values["high_or_medium_confidence_unique_url_rate"]),
        "mismatched_reported_metrics": int(comparison["status"].eq("mismatch").sum()),
        "output_dir": str(output_dir),
    }
