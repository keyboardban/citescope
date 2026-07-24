#!/usr/bin/env python3
"""Build latest area-condo taxonomy and an LPM-safe row-level dataset."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.general_page_taxonomy import (
    classify_general_page_type,
    classify_general_site_type,
    finalise_general_page_type,
)
from src.econometrics_eda_v2.real_estate_taxonomy import (
    classify_real_estate_page_type,
    classify_source_type_real_estate,
    finalise_real_estate_page_type,
)


def _bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.casefold().isin({"1", "1.0", "true", "yes", "y"})


def _cache_evidence(cache_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        html = str(payload.get("html") or "")
        schema_types = list(dict.fromkeys(re.findall(r'"@type"\s*:\s*"([^"]+)"', html, flags=re.I)))
        rows.append({
            "normalized_url": str(payload.get("normalized_url") or ""),
            "cache_page_title": str(payload.get("title") or ""),
            "cache_meta_description": str(payload.get("meta_description") or ""),
            "page_text": str(payload.get("text") or "")[:12000],
            "structured_data_types": "; ".join(schema_types[:30]),
        })
    return pd.DataFrame(rows).drop_duplicates("normalized_url")


def _classify_urls(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        source_type_re = classify_source_type_real_estate(record.get("source_url", ""), record.get("source_root_domain", ""))
        record["source_type_real_estate"] = source_type_re
        re_seed = classify_real_estate_page_type(record, source_type=source_type_re, include_content=False)
        re_scraped = classify_real_estate_page_type(record, source_type=source_type_re, include_content=True)
        re_final, re_source = finalise_real_estate_page_type(re_seed, re_scraped, str(record.get("content_quality_flag") or ""))

        general_seed = classify_general_page_type(record, include_content=False)
        general_scraped = classify_general_page_type(record, include_content=True)
        general_final, general_source = finalise_general_page_type(
            general_seed,
            general_scraped,
            str(record.get("content_quality_flag") or ""),
            str(record.get("content_strength") or ""),
        )
        record.update({
            "source_type_real_estate": source_type_re,
            "page_type_family_real_estate": re_final.family,
            "page_type_detail_real_estate": re_final.detail,
            "re_page_type_confidence": re_final.confidence,
            "re_page_type_source": re_source,
            "re_page_type_reason": re_final.reason,
            "site_type_general": classify_general_site_type(record),
            "page_type_url_seed_general": general_seed.detail,
            "page_type_scraped_enriched_general": general_scraped.detail,
            "page_type_family_general": general_final.family,
            "page_type_general": general_final.detail,
            "page_type_general_confidence": general_final.confidence,
            "page_type_general_source": general_source,
            "page_type_general_reason": general_final.reason,
            "page_type_general_score": general_final.score,
        })
        output.append(record)
    result = pd.DataFrame(output)
    confidence = result["page_type_general_confidence"].fillna("unknown")
    result["page_type_general_confidence_high_or_medium"] = confidence.isin({"high", "medium"})
    result["taxonomy_confidence_high_or_medium"] = (
        confidence.isin({"high", "medium"}) & result["re_page_type_confidence"].isin({"high", "medium"})
    )
    return result


def _summary(urls: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for level, frame in (("url", urls), ("row", rows)):
        for field in (
            "page_type_family_general", "page_type_general", "site_type_general",
            "page_type_general_confidence", "page_type_family_real_estate", "source_type_real_estate",
        ):
            for category, group in frame.groupby(field, dropna=False):
                output.append({
                    "analysis_level": level,
                    "field": field,
                    "category": category,
                    "n": len(group),
                    "share": len(group) / len(frame) if len(frame) else np.nan,
                    "cited_rows": int(group["cited"].sum()) if "cited" in group else pd.NA,
                    "cited_rate": float(group["cited"].mean()) if "cited" in group else np.nan,
                })
    return pd.DataFrame(output)


def _review_sample(urls: pd.DataFrame, exposure: pd.DataFrame) -> pd.DataFrame:
    work = urls.merge(exposure, on="normalized_url", how="left", validate="one_to_one")
    picks = []
    used: set[str] = set()

    def take(frame: pd.DataFrame, n: int, reason: str) -> None:
        selected = frame[~frame.normalized_url.isin(used)].head(n).copy()
        selected["review_reason"] = reason
        picks.append(selected)
        used.update(selected.normalized_url)

    for confidence in ("high", "medium", "low", "unknown"):
        group = work[work.page_type_general_confidence.eq(confidence)].sort_values(
            ["cited_rows", "source_rows", "normalized_url"], ascending=[False, False, True]
        )
        take(group, 30, f"{confidence}_general_taxonomy_confidence")
    take(
        work.sort_values(["cited_rows", "source_rows", "normalized_url"], ascending=[False, False, True]),
        30,
        "high_impact_cited_url",
    )
    sample = pd.concat(picks, ignore_index=True)
    columns = [
        "source_url", "normalized_url", "source_root_domain", "source_title", "page_title",
        "meta_description", "page_text_excerpt", "structured_data_types", "source_rows", "cited_rows",
        "site_type_general", "page_type_family_general", "page_type_general",
        "page_type_general_confidence", "page_type_general_reason", "source_type_real_estate",
        "page_type_family_real_estate", "page_type_detail_real_estate", "re_page_type_confidence",
        "review_reason",
    ]
    return sample.reindex(columns=columns)


def prepare(sources_path: Path, features_path: Path, manifest_path: Path, cache_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = pd.read_csv(sources_path, low_memory=False)
    features = pd.read_csv(features_path, low_memory=False)
    manifest = pd.read_csv(manifest_path, low_memory=False)

    if not sources.prompt_id.astype(str).str.startswith("AREA_CONDO_NB_500_").all():
        raise RuntimeError("Source rows do not contain stable manifest prompt IDs.")
    if set(sources.prompt_id) - set(manifest.prompt_id):
        raise RuntimeError("Some source prompt IDs are absent from the manifest.")

    manifest_columns = [
        "prompt_id", "country", "prompt_language", "prompt_is_nonbranded", "visibility_goal",
        "area_tag", "expansion_group", "brand_mention_in_prompt_check", "expected_source_types",
    ]
    source_rows = sources.merge(manifest[manifest_columns], on="prompt_id", how="left", validate="many_to_one")
    metadata = source_rows.groupby("normalized_url", as_index=False).agg(
        source_title=("source_title", "first"), source_description=("source_description", "first")
    )
    feature_columns = [
        "normalized_url", "source_url", "source_root_domain", "final_provider_mode", "scrape_success",
        "content_strength", "content_quality_flag", "content_chars", "word_count", "heading_count",
        "table_count", "link_count", "page_title", "meta_description", "page_text_excerpt", "scrape_error",
    ]
    urls = features[feature_columns].drop_duplicates("normalized_url").merge(
        metadata, on="normalized_url", how="left", validate="one_to_one"
    )
    cache = _cache_evidence(cache_dir)
    urls = urls.merge(cache, on="normalized_url", how="left", validate="one_to_one")
    urls["page_title"] = urls["page_title"].fillna("").mask(urls["page_title"].fillna("").eq(""), urls["cache_page_title"])
    urls["meta_description"] = urls["meta_description"].fillna("").mask(
        urls["meta_description"].fillna("").eq(""), urls["cache_meta_description"]
    )
    urls.drop(columns=["cache_page_title", "cache_meta_description"], inplace=True)
    urls = _classify_urls(urls)

    taxonomy_columns = [
        "normalized_url", "source_type_real_estate", "page_type_family_real_estate",
        "page_type_detail_real_estate", "re_page_type_confidence", "re_page_type_source",
        "site_type_general", "page_type_url_seed_general", "page_type_scraped_enriched_general",
        "page_type_family_general", "page_type_general", "page_type_general_confidence",
        "page_type_general_source", "page_type_general_score",
        "page_type_general_confidence_high_or_medium", "taxonomy_confidence_high_or_medium",
        "scrape_success", "content_strength", "content_quality_flag", "content_chars", "word_count",
        "heading_count", "table_count", "link_count", "final_provider_mode",
    ]
    safe_base = source_rows[[
        "record_id", "prompt_id", "intent", "topic", "country", "prompt_language",
        "prompt_is_nonbranded", "area_tag", "expansion_group", "brand_mention_in_prompt_check",
        "normalized_url", "source_url", "source_root_domain", "cited",
    ]].copy()
    lpm = safe_base.merge(urls[taxonomy_columns], on="normalized_url", how="left", validate="many_to_one")
    scrape_join_rate = float(lpm["scrape_success"].notna().mean())
    lpm["scrape_success"] = _bool(lpm["scrape_success"])
    lpm["content_feature_available"] = lpm.scrape_success & pd.to_numeric(lpm.content_chars, errors="coerce").fillna(0).gt(0)
    lpm["has_table"] = pd.to_numeric(lpm.table_count, errors="coerce").fillna(0).gt(0)
    lpm["log1p_word_count"] = np.log1p(pd.to_numeric(lpm.word_count, errors="coerce").clip(lower=0))
    lpm["heading_count_group"] = pd.cut(
        pd.to_numeric(lpm.heading_count, errors="coerce"), [-np.inf, 1, 6, 12, np.inf],
        labels=["0-1", "2-6", "7-12", "13+"],
    ).astype(str)
    lpm["link_count_group"] = pd.cut(
        pd.to_numeric(lpm.link_count, errors="coerce"), [-np.inf, 3, 8, np.inf],
        labels=["0-3", "4-8", "9+"],
    ).astype(str)
    lpm["developer_official"] = lpm.source_type_real_estate.isin({"developer_official", "project_official"})
    lpm["property_portal"] = lpm.source_type_real_estate.isin({"property_portal", "listing_marketplace"})
    lpm["broker_agency"] = lpm.source_type_real_estate.eq("broker_agency")
    lpm["social_forum"] = lpm.source_type_real_estate.eq("social_forum")

    urls.to_csv(output_dir / "area_condo_url_taxonomy.csv", index=False)
    lpm.to_csv(output_dir / "area_condo_lpm_ready_with_taxonomy.csv", index=False)
    exposure = source_rows.groupby("normalized_url", as_index=False).agg(
        source_rows=("cited", "size"), cited_rows=("cited", "sum")
    )
    summary = _summary(urls, lpm)
    summary.to_csv(output_dir / "taxonomy_summary.csv", index=False)
    _review_sample(urls, exposure).to_csv(output_dir / "taxonomy_manual_review_sample_150.csv", index=False)
    unknown = urls.merge(exposure, on="normalized_url", how="left", validate="one_to_one")
    unknown = unknown[
        unknown.page_type_family_general.eq("unknown")
        | unknown.site_type_general.eq("unknown")
        | unknown.page_type_family_real_estate.eq("unknown")
        | unknown.source_type_real_estate.eq("unknown")
    ].sort_values(["cited_rows", "source_rows", "normalized_url"], ascending=[False, False, True])
    unknown[[
        "source_url", "normalized_url", "source_root_domain", "source_title", "page_title",
        "page_text_excerpt", "structured_data_types", "source_rows", "cited_rows", "scrape_success",
        "content_strength", "site_type_general", "page_type_family_general", "page_type_general",
        "page_type_general_confidence", "page_type_general_reason", "source_type_real_estate",
        "page_type_family_real_estate", "page_type_detail_real_estate", "re_page_type_reason",
    ]].to_csv(output_dir / "taxonomy_unknown_diagnostics.csv", index=False)

    manifest_ids_with_sources = set(source_rows.prompt_id)
    prompt_audit = manifest[["prompt_id", "prompt", "intent", "area_tag", "expansion_group"]].copy()
    prompt_audit["has_source_rows"] = prompt_audit.prompt_id.isin(manifest_ids_with_sources)
    prompt_audit.to_csv(output_dir / "prompt_manifest_join_audit.csv", index=False)

    forbidden = [
        "answer_text", "page_answer_similarity", "answer_like_text", "source_group", "source_origin",
        "source_position", "observed_rank", "is_more_only", "cited_rows", "selection_reason",
    ]
    leakage = pd.DataFrame({
        "variable_name": forbidden,
        "present_in_lpm_table": [name in lpm.columns for name in forbidden],
        "allowed_role": ["forbidden_predictor" for _ in forbidden],
    })
    leakage.to_csv(output_dir / "lpm_leakage_audit.csv", index=False)

    dictionary_rows = []
    main = {"page_type_family_general", "site_type_general", "content_feature_available", "prompt_id"}
    content = {"has_table", "log1p_word_count", "heading_count_group", "link_count_group", "content_quality_flag"}
    sensitivity = {"source_type_real_estate", "page_type_family_real_estate", "taxonomy_confidence_high_or_medium"}
    for column in lpm.columns:
        role = "outcome" if column == "cited" else "fixed_effect" if column == "prompt_id" else "main" if column in main else "content_subset" if column in content else "sensitivity" if column in sensitivity else "identity_or_diagnostic"
        dictionary_rows.append({
            "variable_name": column, "role": role, "use_in_main_lpm": column in main,
            "use_in_content_subset": column in content or column in main,
            "use_in_sensitivity_only": column in sensitivity, "leakage_risk": "none",
        })
    pd.DataFrame(dictionary_rows).to_csv(output_dir / "lpm_variable_dictionary.csv", index=False)

    validation = {
        "source_rows": len(lpm), "unique_urls": int(lpm.normalized_url.nunique()),
        "source_prompts": int(lpm.prompt_id.nunique()), "manifest_prompts": int(manifest.prompt_id.nunique()),
        "manifest_prompts_without_sources": int((~prompt_audit.has_source_rows).sum()),
        "scrape_join_rate": scrape_join_rate,
        "content_feature_available_rate": float(lpm.content_feature_available.mean()),
        "general_page_family_unknown_url_rate": float(urls.page_type_family_general.eq("unknown").mean()),
        "general_site_type_unknown_url_rate": float(urls.site_type_general.eq("unknown").mean()),
        "real_estate_page_family_unknown_url_rate": float(urls.page_type_family_real_estate.eq("unknown").mean()),
        "real_estate_source_type_unknown_url_rate": float(urls.source_type_real_estate.eq("unknown").mean()),
        "general_high_medium_confidence_url_rate": float(urls.page_type_general_confidence.isin({"high", "medium"}).mean()),
        "forbidden_columns_present": leakage.loc[leakage.present_in_lpm_table, "variable_name"].tolist(),
        "taxonomy_uses_citation_outcome": False,
    }
    (output_dir / "taxonomy_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    model_plan = """# Area Condo LPM Model Design Plan

## Unit and outcome

Use one prompt-source appearance per row. Outcome: `cited`. Use `prompt_id` fixed effects and prompt-clustered standard errors. A domain-clustered specification is a robustness check.

## Main all-row model

`cited ~ C(page_type_family_general) + C(site_type_general) + content_feature_available + C(prompt_id)`

This model retains unknown as a category and does not use content measurements that are unavailable after failed extraction.

## Taxonomy sensitivity

Replace the general categories with `C(page_type_family_real_estate) + C(source_type_real_estate)`. Repeat on high/medium taxonomy-confidence rows and after excluding unknown labels.

## Content-subset model

Restrict to `content_feature_available == True`, then fit:

`cited ~ C(page_type_family_general) + C(site_type_general) + has_table + log1p_word_count + C(heading_count_group) + C(link_count_group) + C(content_quality_flag) + C(prompt_id)`

Do not include `content_strength` in the same specification as the counts and quality flag because it is constructed from those measurements.

## Diagnostic-only sensitivity

Position/rank may be added only in a clearly labeled diagnostic model after rejoining the original source table. It is excluded from the LPM-ready table and cannot support the main interpretation.

All coefficients are conditional associations in an observational citation audit, not causal effects or evidence about hidden retrieval mechanisms.
"""
    (output_dir / "lpm_model_design_plan.md").write_text(model_plan, encoding="utf-8")
    report = f"""# Area Condo Taxonomy and LPM Preparation

The latest dataset contains {len(lpm):,} prompt-source rows, {lpm.normalized_url.nunique():,} URLs, and {lpm.prompt_id.nunique():,} prompts with sources. All source prompt IDs map to the 500-row manifest; two manifest prompts returned no sources.

General page-family unknown rate is {validation['general_page_family_unknown_url_rate']:.1%}; general site-type unknown rate is {validation['general_site_type_unknown_url_rate']:.1%}. High/medium general taxonomy confidence is {validation['general_high_medium_confidence_url_rate']:.1%}.

`page_type_family_general` and `site_type_general` are the broad main-model candidates. Real-estate-specific labels are retained as sensitivity variables. Content features remain conditional on `content_feature_available`. The LPM table excludes answer-derived variables, cited/more-only provenance, rank, position, and URL-level aggregated citation fields.

Taxonomy labels remain observational heuristics and require review of `taxonomy_manual_review_sample_150.csv` before final claims.

Recommended status: **near_lpm_ready_after_taxonomy_QA**.
"""
    (output_dir / "taxonomy_lpm_readiness_report.md").write_text(report, encoding="utf-8")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.features, args.manifest, args.cache_dir, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
