#!/usr/bin/env python3
"""Apply the SCOPE condo real-estate taxonomy to existing topic outputs."""

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

from src.econometrics_eda_v2.real_estate_taxonomy import (
    MEDICAL_STYLE_PAGE_TYPES,
    PAGE_TYPE_FAMILY_REAL_ESTATE,
    PAGE_TYPE_DETAIL_REAL_ESTATE,
    classify_real_estate_page_type,
    classify_source_type_real_estate,
    finalise_real_estate_page_type,
    is_real_estate_looking_url,
)


TOPIC = "scope_condo_nonbranded"
OUT = ROOT / "outputs/econometrics_eda_v2/topic_sensitivity" / TOPIC
TABLES = OUT / "tables"
PROCESSED = ROOT / "data/econometrics_v2/topics" / TOPIC / "processed"
DEFAULT_SOURCES = TABLES / "scope_condo_sources_with_manifest.csv"
DEFAULT_AUDIT = TABLES / "scope_condo_scrape_quality_audit.csv"
DEFAULT_PARSE = PROCESSED / "apify_page_parse_rows.csv"
DEFAULT_FEATURES = PROCESSED / "apify_page_features.csv"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _bool(value: Any) -> bool:
    return _clean(value).casefold() in {"1", "1.0", "true", "yes", "y"}


def _number(value: Any, default: float = 0.0) -> float:
    return float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(default).iloc[0])


def _unique_by_url(df: pd.DataFrame, key: str = "normalized_url") -> pd.DataFrame:
    work = df.copy()
    if key not in work:
        work[key] = ""
    return work[work[key].fillna("").astype(str).str.strip() != ""].drop_duplicates(key)


def _parse_evidence(parse: pd.DataFrame) -> pd.DataFrame:
    """Align parsed rows to the request URL, which is the audit's join key."""
    work = parse.copy()
    if "requested_normalized_url" in work:
        request_key = work["requested_normalized_url"].fillna("").astype(str).str.strip()
        work.loc[request_key != "", "normalized_url"] = request_key[request_key != ""]
    wanted = [
        "normalized_url",
        "page_title",
        "meta_description",
        "page_text",
        "heading_count",
        "table_count",
    ]
    for col in wanted:
        if col not in work:
            work[col] = ""
    return _unique_by_url(work[wanted])


def _choose_title(row: pd.Series) -> str:
    return _clean(row.get("page_title")) or _clean(row.get("source_title"))


def _classify_unique_urls(sources: pd.DataFrame, audit: pd.DataFrame, parse: pd.DataFrame) -> pd.DataFrame:
    source_url = _unique_by_url(sources)[
        [c for c in ["normalized_url", "source_url", "source_root_domain", "source_title", "source_description", "source_type_url"] if c in sources]
    ].copy()
    audit_one = _unique_by_url(audit)
    parsed = _parse_evidence(parse)
    work = source_url.merge(audit_one, on="normalized_url", how="outer", suffixes=("", "_audit"))
    work = work.merge(parsed, on="normalized_url", how="left", suffixes=("", "_parse"))
    for col in ["source_url", "source_root_domain"]:
        audit_col = f"{col}_audit"
        if audit_col in work:
            work[col] = work.get(col, pd.Series(index=work.index, dtype=object)).fillna(work[audit_col])
    if "source_url" not in work:
        work["source_url"] = work["normalized_url"]
    if "source_root_domain" not in work:
        work["source_root_domain"] = ""
    if "page_title_parse" in work:
        work["page_title"] = work.get("page_title", pd.Series(index=work.index, dtype=object)).fillna(work["page_title_parse"])
    if "meta_description_parse" in work:
        work["meta_description"] = work.get("meta_description", pd.Series(index=work.index, dtype=object)).fillna(work["meta_description_parse"])
    if "page_text_parse" in work:
        work["page_text"] = work["page_text_parse"]

    result_rows: list[dict[str, Any]] = []
    for _, row in work.iterrows():
        record = row.to_dict()
        url = _clean(record.get("source_url")) or _clean(record.get("normalized_url"))
        domain = _clean(record.get("source_root_domain"))
        source_type = classify_source_type_real_estate(url, domain)
        record["source_type_real_estate"] = source_type
        record["page_title"] = _choose_title(pd.Series(record))
        seed = classify_real_estate_page_type(record, source_type=source_type, include_content=False)
        scraped = classify_real_estate_page_type(record, source_type=source_type, include_content=True)
        final, final_source = finalise_real_estate_page_type(seed, scraped, _clean(record.get("content_quality_flag")))
        record.update(
            {
                "page_type_url_seed_real_estate": seed.detail,
                "page_type_scraped_enriched_real_estate": scraped.detail,
                "page_type_final_real_estate": final.detail,
                "page_type_final_real_estate_source": final_source,
                "page_type_family_real_estate": final.family,
                "page_type_detail_real_estate": final.detail,
                "re_page_type_evidence_url": final.evidence_url,
                "re_page_type_evidence_title": final.evidence_title,
                "re_page_type_evidence_domain": final.evidence_domain,
                "re_page_type_evidence_content": final.evidence_content,
                "re_page_type_score": final.score,
                "re_page_type_confidence": final.confidence,
                "re_page_type_reason": final.reason,
                "real_estate_looking_url": is_real_estate_looking_url(url, domain),
            }
        )
        result_rows.append(record)
    return pd.DataFrame(result_rows)


def _error_audit(urls: pd.DataFrame) -> pd.DataFrame:
    out = urls.copy()
    out["current_source_type"] = out.get("source_type_url", "unknown").fillna("unknown")
    out["current_page_type_final"] = out.get("page_type_final", "unknown").fillna("unknown")
    out["old_page_type_is_medical_style"] = out["current_page_type_final"].isin(MEDICAL_STYLE_PAGE_TYPES)
    out["old_page_type_invalid_for_scope"] = out["old_page_type_is_medical_style"]
    out["medical_style_real_estate_url"] = out["old_page_type_is_medical_style"] & out["real_estate_looking_url"].astype(bool)
    out["title"] = out.apply(_choose_title, axis=1)
    out["suggested_real_estate_page_type"] = out["page_type_final_real_estate"]
    out["reason"] = out.apply(
        lambda row: (
            "legacy_medical_label_on_real_estate_looking_url; " + _clean(row.get("re_page_type_reason"))
            if bool(row.get("medical_style_real_estate_url"))
            else _clean(row.get("re_page_type_reason"))
        ),
        axis=1,
    )
    columns = [
        "source_url",
        "normalized_url",
        "source_root_domain",
        "title",
        "page_text_excerpt",
        "current_source_type",
        "current_page_type_final",
        "current_content_quality_flag",
        "content_quality_flag",
        "word_count",
        "old_page_type_is_medical_style",
        "real_estate_looking_url",
        "medical_style_real_estate_url",
        "suggested_real_estate_page_type",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "re_page_type_confidence",
        "reason",
    ]
    out["current_content_quality_flag"] = out.get("content_quality_flag", "")
    for col in columns:
        if col not in out:
            out[col] = ""
    return out[columns].sort_values(
        ["old_page_type_is_medical_style", "medical_style_real_estate_url", "word_count"],
        ascending=[False, False, False],
    )


def _summary_table(urls: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(section: str, label: str, n: int | float, share: float | None = None, detail: str = "") -> None:
        rows.append({"section": section, "label": label, "n": n, "share": share, "detail": detail})

    n_urls = len(urls)
    add("coverage", "total_urls", n_urls, 1.0 if n_urls else 0.0)
    old = urls.get("page_type_final", pd.Series(dtype=object)).fillna("unknown")
    old_family = urls.get("page_type_family", pd.Series(dtype=object)).fillna("unknown")
    new_family = urls.get("page_type_family_real_estate", pd.Series(dtype=object)).fillna("unknown")
    new_detail = urls.get("page_type_detail_real_estate", pd.Series(dtype=object)).fillna("unknown")
    confidence = urls.get("re_page_type_confidence", pd.Series(dtype=object)).fillna("unknown")
    for section, series in [
        ("old_page_type_distribution", old),
        ("old_page_type_family_distribution", old_family),
        ("new_page_type_family_distribution", new_family),
        ("new_page_type_detail_distribution", new_detail),
        ("confidence_distribution", confidence),
    ]:
        for label, count in series.value_counts(dropna=False).items():
            add(section, str(label), int(count), float(count) / n_urls if n_urls else 0.0)
    add("metrics", "old_unknown_rate", int(old.eq("unknown").sum()), float(old.eq("unknown").mean()) if n_urls else 0.0)
    add("metrics", "new_unknown_rate", int(new_detail.eq("unknown").sum()), float(new_detail.eq("unknown").mean()) if n_urls else 0.0)
    old_medical = old.isin(MEDICAL_STYLE_PAGE_TYPES)
    add("metrics", "old_medical_style_misclassification_count", int(old_medical.sum()), float(old_medical.mean()) if n_urls else 0.0)
    add("metrics", "new_medical_style_misclassification_count", 0, 0.0)
    add(
        "metrics",
        "real_estate_looking_urls_with_legacy_medical_labels",
        int((old_medical & urls["real_estate_looking_url"].astype(bool)).sum()),
        float((old_medical & urls["real_estate_looking_url"].astype(bool)).mean()) if n_urls else 0.0,
    )
    for category, group in urls.assign(_old_page_type=old).groupby("_old_page_type", dropna=False):
        for domain, count in group["source_root_domain"].fillna("unknown").value_counts().head(10).items():
            add("top_domains_by_current_page_type", str(domain), int(count), float(count) / len(group) if len(group) else 0.0, str(category))
    for category, group in urls.groupby("source_type_real_estate", dropna=False):
        for domain, count in group["source_root_domain"].fillna("unknown").value_counts().head(10).items():
            add("top_domains_by_new_source_type", str(domain), int(count), float(count) / len(group) if len(group) else 0.0, str(category))
    for category, group in urls.groupby("page_type_family_real_estate", dropna=False):
        for domain, count in group["source_root_domain"].fillna("unknown").value_counts().head(10).items():
            add("top_domains_by_new_page_type_family", str(domain), int(count), float(count) / len(group) if len(group) else 0.0, str(category))
    return pd.DataFrame(rows)


def _review_sample(rows: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    work = rows.copy()
    cited = pd.to_numeric(work.get("cited"), errors="coerce").fillna(0)
    old_medical = work.get("page_type_final", pd.Series(index=work.index, dtype=object)).fillna("unknown").isin(MEDICAL_STYLE_PAGE_TYPES)
    old_unknown = work.get("page_type_final", pd.Series(index=work.index, dtype=object)).fillna("unknown").eq("unknown")
    changed_unknown = old_unknown & work.get("page_type_final_real_estate", pd.Series(index=work.index, dtype=object)).fillna("unknown").ne("unknown")
    low_confidence = work.get("re_page_type_confidence", pd.Series(index=work.index, dtype=object)).fillna("unknown").isin(["low", "unknown"])
    domain_rank = work.get("source_root_domain", pd.Series(index=work.index, dtype=object)).fillna("").map(work.get("source_root_domain", pd.Series(index=work.index, dtype=object)).value_counts())
    work["_priority"] = (
        cited * 10000
        + old_medical.astype(int) * 1000
        + changed_unknown.astype(int) * 500
        + low_confidence.astype(int) * 100
        + pd.to_numeric(domain_rank, errors="coerce").fillna(0)
    )
    sample = work.sort_values(["_priority", "word_count"], ascending=[False, False]).drop_duplicates("normalized_url").head(n).copy()
    sample["title"] = sample.apply(_choose_title, axis=1)
    columns = [
        "source_url",
        "source_root_domain",
        "title",
        "page_text_excerpt",
        "old_source_type",
        "old_page_type_final",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "page_type_detail_real_estate",
        "re_page_type_confidence",
        "re_page_type_reason",
        "content_quality_flag",
        "word_count",
        "cited",
    ]
    sample["old_source_type"] = (
        sample["source_type_url"].fillna("unknown")
        if "source_type_url" in sample
        else "unknown"
    )
    sample["old_page_type_final"] = (
        sample["page_type_final"].fillna("unknown")
        if "page_type_final" in sample
        else "unknown"
    )
    for col in columns:
        if col not in sample:
            sample[col] = ""
    return sample[columns]


def _validation(urls: pd.DataFrame) -> dict[str, Any]:
    old = urls.get("page_type_final", pd.Series(dtype=object)).fillna("unknown")
    new = urls.get("page_type_final_real_estate", pd.Series(dtype=object)).fillna("unknown")
    quality = urls.get("content_quality_flag", pd.Series(dtype=object)).fillna("")
    final_source = urls.get("page_type_final_real_estate_source", pd.Series(dtype=object)).fillna("")
    source_type = urls.get("source_type_real_estate", pd.Series(dtype=object)).fillna("unknown")
    seed = urls.get("page_type_url_seed_real_estate", pd.Series(dtype=object)).fillna("unknown")
    confidence = urls.get("re_page_type_confidence", pd.Series(dtype=object)).fillna("unknown")
    errors: list[str] = []
    warnings: list[str] = []
    remaining_medical = int(new.isin(MEDICAL_STYLE_PAGE_TYPES).sum())
    poor_content_overrides = int((quality.ne("ok") & final_source.eq("scraped_content")).sum())
    if remaining_medical:
        errors.append(f"{remaining_medical} medical-style labels remain in the real-estate taxonomy.")
    if poor_content_overrides:
        errors.append(f"{poor_content_overrides} poor-content rows were overridden by scraped-content labels.")
    old_unknown_rate = float(old.eq("unknown").mean()) if len(old) else 0.0
    new_unknown_rate = float(new.eq("unknown").mean()) if len(new) else 0.0
    if new_unknown_rate >= old_unknown_rate:
        warnings.append("Unknown rate did not decrease; inspect weak-domain and non-real-estate URLs before broadening rules.")
    common_types = {"developer_official", "project_official", "property_portal", "listing_marketplace", "broker_agency"}
    common = urls[source_type.isin(common_types)]
    seed_coverage = float(seed.loc[common.index].ne("unknown").mean()) if len(common) else 1.0
    if seed_coverage < 0.9:
        warnings.append(f"URL/domain seed coverage for common real-estate source types is {seed_coverage:.1%}, below 90%.")
    family_counts = urls.get("page_type_family_real_estate", pd.Series(dtype=object)).value_counts()
    detail_counts = urls.get("page_type_detail_real_estate", pd.Series(dtype=object)).value_counts()
    sparse_families = int((family_counts < 5).sum())
    sparse_details = int((detail_counts < 5).sum())
    if sparse_families > sparse_details:
        warnings.append("Family taxonomy is more sparse than detail taxonomy; review family mapping.")
    return {
        "passed": not errors,
        "warnings": warnings,
        "errors": errors,
        "medical_label_remaining_count": remaining_medical,
        "unknown_rate_before": old_unknown_rate,
        "unknown_rate_after": new_unknown_rate,
        "changed_rows_count": int(old.ne(new).sum()),
        "low_confidence_count": int(confidence.isin(["low", "unknown"]).sum()),
        "poor_content_scraped_override_count": poor_content_overrides,
        "common_real_estate_url_seed_coverage": seed_coverage,
        "sparse_family_count": sparse_families,
        "sparse_detail_count": sparse_details,
        "recommendation": (
            "Use page_type_family_real_estate in main EDA/LPM, retain page_type_detail_real_estate for diagnostics, "
            "and use source_type_real_estate as a source control or stratifier. Keep source position and answer-derived "
            "features out of the main model."
        ),
    }


def _eda_ready(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["scraped_ok"] = (
        work.get("content_quality_flag", pd.Series(index=work.index, dtype=object)).fillna("").eq("ok")
        & work.get("scraped_body_available", pd.Series(index=work.index, dtype=object)).map(_bool)
    ).astype(int)
    available = work.get("content_feature_available", pd.Series(index=work.index, dtype=object)).map(_bool)
    work["content_feature_available"] = (available | work["scraped_ok"].eq(1)).astype(int)
    work["real_estate_taxonomy_available"] = work["page_type_family_real_estate"].fillna("unknown").ne("unknown").astype(int)
    work["page_type_confidence_high_or_medium"] = work["re_page_type_confidence"].isin(["high", "medium"]).astype(int)
    # Keep the outcome itself, but remove its duplicates, rank/position fields,
    # raw answer text, and all answer-/prompt-similarity features from the EDA table.
    forbidden = {
        "source_position",
        "observed_rank",
        "answer_text",
        "cited_label",
        "is_more_only",
        "source_group",
        "source_origin",
        "page_prompt_similarity",
        "max_chunk_prompt_similarity",
    }
    similarity_columns = {col for col in work.columns if "answer" in col.casefold() or "similarity" in col.casefold()}
    keep = [col for col in work.columns if col not in forbidden and col not in similarity_columns]
    return work[keep]


def _report(urls: pd.DataFrame, validation: dict[str, Any], path: Path) -> None:
    old = urls["page_type_final"].fillna("unknown")
    new = urls["page_type_detail_real_estate"].fillna("unknown")
    medical = urls[old.isin(MEDICAL_STYLE_PAGE_TYPES)].copy()
    corrected = medical[medical["page_type_detail_real_estate"].fillna("unknown").ne("unknown")]
    examples = medical[
        ["source_url", "page_title", "page_type_final", "page_type_detail_real_estate", "re_page_type_confidence"]
    ] if corrected.empty else corrected[
        ["source_url", "page_title", "page_type_final", "page_type_detail_real_estate", "re_page_type_confidence"]
    ]
    examples = examples.head(10)
    remaining = urls[new.eq("unknown")].copy()
    lines = [
        "# SCOPE Condo Real-Estate Taxonomy Report",
        "",
        "## Why the old taxonomy failed",
        "The previous page-type vocabulary was designed around hospital pages. Its broad article, service, center, and profile rules labelled condo guides, broker pages, and market reports with medical-style categories. This is a taxonomy mismatch, not evidence about AI retrieval or citation decisions.",
        "",
        "## New taxonomy design",
        "The new layer uses a real-estate-specific source taxonomy plus a two-level page taxonomy. URL and domain evidence create a seed; good scraped content can refine it only at medium or high confidence. Bad, short, blocked, or unparsed content cannot overwrite a useful seed.",
        "",
        "## Old vs new page types",
        f"- URLs: {len(urls)}",
        f"- Old unknown rate: {validation['unknown_rate_before']:.1%}",
        f"- New unknown rate: {validation['unknown_rate_after']:.1%}",
        f"- Legacy medical-style labels: {int(old.isin(MEDICAL_STYLE_PAGE_TYPES).sum())}",
        f"- Legacy medical-style labels on real-estate-looking URLs: {int((old.isin(MEDICAL_STYLE_PAGE_TYPES) & urls['real_estate_looking_url'].astype(bool)).sum())}",
        f"- Medical-style labels remaining in new taxonomy: {validation['medical_label_remaining_count']}",
        "",
        "### Old distribution",
        old.value_counts().to_string(),
        "",
        "### New family distribution",
        urls["page_type_family_real_estate"].value_counts().to_string(),
        "",
        "### New detail distribution",
        new.value_counts().to_string(),
        "",
        "## Corrected examples",
        examples.to_string(index=False),
        "",
        "## Remaining difficult cases",
        f"{len(remaining)} URLs remain unknown. These are retained as unknown when evidence is weak or conflicting, including pages with incomplete scraping, generic external content, or no reliable real-estate signal.",
        "",
        "## EDA use",
        "Use `page_type_family_real_estate` as the main categorical variable in the SCOPE EDA/LPM. Keep `source_type_real_estate` as a control or descriptive stratifier. Use `page_type_detail_real_estate`, evidence strings, confidence, and source selection fields for diagnostics only.",
        "",
        "Do not include source position, observed rank, answer text, or answer-derived similarity in the main model. The taxonomy is feature engineering for later observational analysis, not a causal interpretation.",
        "",
        "## Provider fallback",
        "For remaining unknown pages with a useful URL seed, improve classification only after verifying page evidence. Use a JS-capable fallback selectively for parse-failed, dynamic-JS, or very-short-content URLs; do not rescrape merely because the old medical taxonomy was wrong.",
    ]
    path.write_text("\n".join(lines), "utf-8")


def run(
    sources_path: Path = DEFAULT_SOURCES,
    audit_path: Path = DEFAULT_AUDIT,
    parse_path: Path = DEFAULT_PARSE,
    features_path: Path = DEFAULT_FEATURES,
    output_dir: Path = TABLES,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(sources_path, low_memory=False)
    audit = pd.read_csv(audit_path, low_memory=False)
    parse = pd.read_csv(parse_path, low_memory=False) if parse_path.exists() else pd.DataFrame()
    features = pd.read_csv(features_path, low_memory=False) if features_path.exists() else pd.DataFrame()
    urls = _classify_unique_urls(sources, audit, parse)
    error_audit = _error_audit(urls)

    join_cols = [
        "normalized_url",
        "source_type_real_estate",
        "page_type_family_real_estate",
        "page_type_detail_real_estate",
        "page_type_url_seed_real_estate",
        "page_type_scraped_enriched_real_estate",
        "page_type_final_real_estate",
        "page_type_final_real_estate_source",
        "re_page_type_evidence_url",
        "re_page_type_evidence_title",
        "re_page_type_evidence_domain",
        "re_page_type_evidence_content",
        "re_page_type_score",
        "re_page_type_confidence",
        "re_page_type_reason",
    ]
    taxonomy = sources.merge(urls[join_cols], on="normalized_url", how="left")
    audit_cols = [
        "normalized_url",
        "scrape_success",
        "parse_success",
        "scraped_body_available",
        "word_count",
        "text_char_count",
        "heading_count",
        "table_count",
        "link_count",
        "content_quality_flag",
        "page_title",
        "meta_description",
        "page_text_excerpt",
        "page_type_url_seed",
        "page_type_scraped_enriched",
        "page_type_final",
        "page_type_final_source",
        "page_type_family",
        "source_type_url",
    ]
    available_audit_cols = [col for col in audit_cols if col in urls]
    taxonomy = taxonomy.merge(
        _unique_by_url(urls)[available_audit_cols],
        on="normalized_url",
        how="left",
        suffixes=("", "_audit"),
    )
    if not features.empty:
        flags = [
            "normalized_url",
            "content_feature_available",
            "has_price_or_package",
            "has_contact_info",
            "has_table",
        ]
        present = [col for col in flags if col in features]
        taxonomy = taxonomy.merge(_unique_by_url(features)[present], on="normalized_url", how="left")

    summary = _summary_table(urls)
    review = _review_sample(taxonomy)
    validation = _validation(urls)
    eda = _eda_ready(taxonomy)
    error_audit.to_csv(output_dir / "scope_taxonomy_error_audit.csv", index=False)
    taxonomy.to_csv(output_dir / "scope_condo_sources_with_real_estate_taxonomy.csv", index=False)
    summary.to_csv(output_dir / "scope_real_estate_taxonomy_summary.csv", index=False)
    review.to_csv(output_dir / "scope_real_estate_taxonomy_review_sample.csv", index=False)
    (output_dir / "scope_real_estate_taxonomy_validation.json").write_text(json.dumps(validation, indent=2), "utf-8")
    eda.to_csv(output_dir / "scope_condo_eda_ready_with_real_estate_taxonomy.csv", index=False)
    _report(urls, validation, output_dir.parent / "scope_real_estate_taxonomy_report.md")
    return {
        "total_urls": int(len(urls)),
        "source_rows": int(len(taxonomy)),
        "old_medical_style_count": int(urls["page_type_final"].fillna("unknown").isin(MEDICAL_STYLE_PAGE_TYPES).sum()),
        "old_unknown_rate": validation["unknown_rate_before"],
        "new_unknown_rate": validation["unknown_rate_after"],
        "validation_passed": validation["passed"],
        "output_dir": str(output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--parse", type=Path, default=DEFAULT_PARSE)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=TABLES)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.sources, args.audit, args.parse, args.features, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
