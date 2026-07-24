"""Build and persist the lightweight manual feature-validation layer."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Comment

from src.econometrics_eda_v2.gemini_taxonomy_features import attach_gemini_taxonomy
from src.econometrics_eda_v2.redesigned_pipeline_v2 import source_paths
from src.econometrics_eda_v2.writing_factual_density_features import (
    ANSWER_SIGNAL_TERMS,
    DIRECT_ANSWER_TERMS,
    FAQ_TERMS,
    NUMBER_RE,
    SUMMARY_TERMS,
)
from src.econometrics_qa import load_snapshot


ARTIFACT_VERSION = "manual_feature_validation_v1"
WRITING_VERSION = "writing_factual_density_v1"
DOCUMENT_VERSION = "document_structure_v1"
CONTENT_TRANSFORM_VERSION = "content_feature_transform_v1"
TAXONOMY_VERSION = "gemini_3_1_flash_lite_taxonomy_v1"
COMPONENTS = (
    "has_bullet_list",
    "has_numbered_list",
    "has_faq_pattern",
    "has_question_answer_structure",
    "opening_has_summary_signal",
    "opening_has_direct_answer_signal",
)
FACTUAL_COMPONENTS = (
    "number_token_per_1000_words",
    "percent_mention_count",
    "year_mention_count",
    "range_mention_count",
    "measurement_mention_count",
)
ACTIVE_FEATURES = (
    "log2_word_count_plus1",
    "has_verified_html_table",
    "factual_numeric_density_score",
    "writing_structure_score",
)
REVIEW_COLUMNS = (
    "normalized_url",
    "prompt_id",
    "feature_name",
    "automated_value",
    "reviewer_decision",
    "error_type",
    "reviewer_note",
    "content_source_used",
    "feature_producer_version",
    "reviewed_at",
)
FORBIDDEN_ARTIFACT_TERMS = (
    "answer_text",
    "page_answer_similarity",
    "answer_overlap",
    "authorization",
    "api_key",
    "cookie",
    "request_headers",
    "raw_response",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def nullable_binary_status(value: object) -> str:
    """Return a three-state label without coercing missing to false."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Unmeasured"
    return "Detected" if float(numeric) == 1 else "Not detected"


def writing_component_sum(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[list(COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=len(COMPONENTS))


def factual_component_contributions(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[list(FACTUAL_COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame(
        {
            "numeric_rate_contribution": (numeric["number_token_per_1000_words"] / 10).clip(upper=5),
            "percent_indicator_contribution": numeric["percent_mention_count"].gt(0).where(
                numeric["percent_mention_count"].notna()
            ).astype("Float64"),
            "year_indicator_contribution": numeric["year_mention_count"].gt(0).where(
                numeric["year_mention_count"].notna()
            ).astype("Float64"),
            "range_indicator_contribution": numeric["range_mention_count"].gt(0).where(
                numeric["range_mention_count"].notna()
            ).astype("Float64"),
            "measurement_log_contribution": np.log1p(numeric["measurement_mention_count"]),
        },
        index=frame.index,
    )


def producer_version(feature_name: str) -> str:
    if feature_name == "has_verified_html_table":
        return DOCUMENT_VERSION
    if feature_name in {"writing_structure_score", "factual_numeric_density_score", *COMPONENTS}:
        return WRITING_VERSION
    if feature_name == "log2_word_count_plus1":
        return CONTENT_TRANSFORM_VERSION
    return ARTIFACT_VERSION


def sanitize_html_preview(raw_html: object, *, max_chars: int = 20000) -> str:
    """Return inert HTML suitable for a sandboxed preview component."""
    source = "" if raw_html is None else str(raw_html)
    if not source.strip():
        return ""
    soup = BeautifulSoup(source, "html.parser")
    for node in soup.find_all(["script", "style", "noscript", "iframe", "object", "embed", "form", "input", "button", "link", "meta"]):
        node.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
        comment.extract()
    safe_attrs = {"href", "src", "alt", "title", "colspan", "rowspan", "scope"}
    for tag in soup.find_all(True):
        for attribute in list(tag.attrs):
            if attribute.casefold().startswith("on") or attribute.casefold() not in safe_attrs:
                del tag.attrs[attribute]
        for attribute in ("href", "src"):
            value = str(tag.attrs.get(attribute, "")).strip()
            if value and not re.match(r"^(?:https?:|/|#)", value, flags=re.IGNORECASE):
                del tag.attrs[attribute]
        if tag.name == "a":
            tag.attrs["target"] = "_blank"
            tag.attrs["rel"] = "noopener noreferrer"
    body = soup.body or soup
    rendered = str(body)
    if len(rendered) <= max_chars:
        return rendered
    truncated = BeautifulSoup(rendered[:max_chars], "html.parser")
    return str(truncated) + '<p><em>HTML preview truncated for frontend QA.</em></p>'


def highlighted_content(text: object, row: pd.Series) -> str:
    """Escape producer text and mark evidence using governed producer terms."""
    escaped = html.escape("" if text is None else str(text))
    if nullable_binary_status(row.get("has_bullet_list")) == "Detected":
        escaped = re.sub(
            r"(?im)^([ \t]*(?:[-*•●▪◦])[ \t]+[^\n]+)",
            r'<mark class="list-evidence">\1</mark>',
            escaped,
        )
    if nullable_binary_status(row.get("has_numbered_list")) == "Detected":
        escaped = re.sub(
            r"(?im)^([ \t]*(?:\d+[.)]|[A-Za-z][.)])[ \t]+[^\n]+)",
            r'<mark class="list-evidence">\1</mark>',
            escaped,
        )
    terms: list[str] = []
    if nullable_binary_status(row.get("has_faq_pattern")) == "Detected":
        terms.extend(FAQ_TERMS)
    if nullable_binary_status(row.get("has_question_answer_structure")) == "Detected":
        terms.extend(ANSWER_SIGNAL_TERMS)
    if nullable_binary_status(row.get("opening_has_summary_signal")) == "Detected":
        terms.extend(SUMMARY_TERMS)
    if nullable_binary_status(row.get("opening_has_direct_answer_signal")) == "Detected":
        terms.extend(DIRECT_ANSWER_TERMS)
    if terms:
        pattern = "|".join(
            sorted((re.escape(html.escape(term)) for term in set(terms)), key=len, reverse=True)
        )
        escaped = re.sub(
            rf"(?i)({pattern})",
            r'<mark class="pattern-evidence">\1</mark>',
            escaped,
        )
    if pd.notna(pd.to_numeric(pd.Series([row.get("factual_numeric_density_score")]), errors="coerce").iloc[0]):
        escaped = NUMBER_RE.sub(r'<mark class="numeric-evidence">\g<0></mark>', escaped)
    return escaped.replace("\n", "<br>")


def _safe_snapshot_fields(snapshot: dict[str, Any] | None) -> dict[str, str]:
    if not snapshot:
        return {"captured_markdown_or_body": "", "sanitized_html_preview": ""}
    markdown = str(snapshot.get("markdown") or "").strip()
    body = markdown or str(snapshot.get("text") or "").strip()
    return {
        "captured_markdown_or_body": body[:12000],
        "sanitized_html_preview": sanitize_html_preview(snapshot.get("html")),
    }


def build_artifacts(repo: Path, output_dir: Path) -> dict[str, Any]:
    paths = source_paths(repo)
    package = paths["base"].parents[1]
    writing = pd.read_csv(paths["base"], low_memory=False)
    assembly_path = package / "tables/10_writing_factual_density_features/url_text_assembly_audit.csv"
    document_path = package / "tables/12_document_structure_features/url_document_structure_features.csv"
    assembly = pd.read_csv(assembly_path, low_memory=False).drop_duplicates("normalized_url")
    document = pd.read_csv(document_path, low_memory=False).drop_duplicates("normalized_url")

    document_columns = [
        "normalized_url", "html_available", "has_html_table", "main_content_available",
        "main_content_extraction_method", "main_content_text_path", "generated_markdown_path",
    ]
    rows = writing.merge(
        document[document_columns], on="normalized_url", how="left", validate="many_to_one",
    ).copy()
    html_measured = pd.to_numeric(rows["html_available"], errors="coerce").eq(1)
    rows["has_verified_html_table"] = pd.Series(pd.NA, index=rows.index, dtype="Int64")
    rows.loc[html_measured, "has_verified_html_table"] = (
        pd.to_numeric(rows.loc[html_measured, "has_html_table"], errors="coerce").fillna(0).gt(0).astype(int)
    )
    rows, _, taxonomy_audit = attach_gemini_taxonomy(
        rows, package, taxonomy_path=paths["taxonomy"], min_rows=20,
    )
    component_sum = writing_component_sum(rows)
    rows["writing_component_sum"] = component_sum
    rows["writing_score_matches_components"] = (
        pd.to_numeric(rows["writing_structure_score"], errors="coerce").eq(component_sum)
    )
    factual_contributions = factual_component_contributions(rows)
    rows["factual_component_sum"] = factual_contributions.sum(
        axis=1, min_count=len(factual_contributions.columns)
    )
    rows["factual_score_matches_components"] = (
        pd.to_numeric(rows["factual_numeric_density_score"], errors="coerce")
        .sub(rows["factual_component_sum"])
        .abs()
        .le(1e-9)
    )
    rows["suspicious_measurement"] = (
        ~rows["writing_score_matches_components"]
        | ~rows["factual_score_matches_components"]
        | rows["text_source_used"].fillna("none").eq("none")
        | rows[list(COMPONENTS)].isna().any(axis=1)
        | (rows["has_verified_html_table"].isna() & html_measured)
    )
    row_columns = [
        "prompt_id", "normalized_url", "source_url", "source_root_domain", "cited",
        "content_strength", "scrape_success", "content_quality_flag", "word_count",
        "log2_word_count_plus1", "has_verified_html_table", "factual_numeric_density_score",
        "writing_structure_score", *FACTUAL_COMPONENTS, *COMPONENTS,
        "opening_100_words", "text_source_used",
        "feature_extraction_text_scope", "text_feature_available", "heading_count_group",
        "writing_component_sum",
        "writing_score_matches_components", "factual_component_sum",
        "factual_score_matches_components", "suspicious_measurement",
        "page_type_family_gemini_v1_collapsed", "source_type_general_gemini_v1_collapsed",
    ]
    frontend_rows = rows[row_columns].copy()
    frontend_rows["artifact_version"] = ARTIFACT_VERSION
    frontend_rows["feature_producer_version"] = WRITING_VERSION

    content_columns = [
        "normalized_url", "source_url", "url_title", "url_description", "url_text_for_features",
        "page_text_excerpt", "text_source_used", "feature_extraction_text_scope",
        "full_page_text_available", "limited_excerpt_only",
    ]
    content = assembly[content_columns].copy()
    snapshot_root = package.parent / "tables/area_condo_brightdata_content_pilot/normalized"
    snapshot_fields = []
    for source in content["source_url"].fillna(content["normalized_url"]).astype(str):
        snapshot, _ = load_snapshot(source, snapshot_root=snapshot_root)
        snapshot_fields.append(_safe_snapshot_fields(snapshot))
    content = pd.concat([content.reset_index(drop=True), pd.DataFrame(snapshot_fields)], axis=1)
    content["authoritative_feature_content"] = content["url_text_for_features"]
    content["authoritative_content_source"] = content["text_source_used"]
    content["artifact_version"] = ARTIFACT_VERSION

    output_dir.mkdir(parents=True, exist_ok=True)
    for legacy_name in ("manual_feature_validation_rows.csv", "manual_feature_validation_content.csv"):
        legacy_path = output_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
    rows_path = output_dir / "manual_feature_validation_rows.csv.gz"
    content_path = output_dir / "manual_feature_validation_content.csv.gz"
    frontend_rows.to_csv(rows_path, index=False, compression="gzip")
    content.to_csv(content_path, index=False, compression="gzip")
    reviews_path = output_dir.parent / "tables/manual_feature_validation_reviews.csv"
    if not reviews_path.exists():
        pd.DataFrame(columns=REVIEW_COLUMNS).to_csv(reviews_path, index=False)

    combined_headers = " ".join(frontend_rows.columns.tolist() + content.columns.tolist()).casefold()
    found_forbidden = [term for term in FORBIDDEN_ARTIFACT_TERMS if term in combined_headers]
    if found_forbidden:
        raise ValueError(f"Sensitive/leakage fields entered frontend artifacts: {found_forbidden}")
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": utc_now(),
        "validated": True,
        "producer_versions": [CONTENT_TRANSFORM_VERSION, DOCUMENT_VERSION, WRITING_VERSION, TAXONOMY_VERSION],
        "authoritative_content_field": "authoritative_feature_content",
        "authoritative_content_source_field": "authoritative_content_source",
        "files": {
            rows_path.name: {"sha256": sha256(rows_path), "rows": len(frontend_rows)},
            content_path.name: {"sha256": sha256(content_path), "rows": len(content)},
        },
        "review_file": str(reviews_path),
        "taxonomy_join_audit": taxonomy_audit,
        "warnings": [
            "Stored feature content is authoritative; live webpages may have changed.",
            "HTML preview is sanitized, script-free, and capped for frontend QA.",
            "The historical writing producer usually used a 3,000-character preview plus title/meta, not the full body.",
        ],
    }
    manifest_path = output_dir / "manual_feature_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_validated_artifacts(frontend_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = frontend_dir / "manual_feature_validation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("validated"):
        raise ValueError("Manual feature-validation artifact is not validated.")
    for filename, metadata in manifest["files"].items():
        path = frontend_dir / filename
        if not path.exists() or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Manual feature-validation artifact hash mismatch: {filename}")
    return (
        pd.read_csv(frontend_dir / "manual_feature_validation_rows.csv.gz", low_memory=False),
        pd.read_csv(frontend_dir / "manual_feature_validation_content.csv.gz", low_memory=False),
        manifest,
    )


def append_review(path: Path, review: dict[str, Any]) -> None:
    """Append an annotation atomically without touching model or feature files."""
    row = {column: review.get(column, "") for column in REVIEW_COLUMNS}
    row["reviewed_at"] = str(row["reviewed_at"] or utc_now())
    if not str(row["normalized_url"]).strip() or not str(row["feature_name"]).strip():
        raise ValueError("normalized_url and feature_name are required")
    existing = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame(columns=REVIEW_COLUMNS)
    output = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, path)
