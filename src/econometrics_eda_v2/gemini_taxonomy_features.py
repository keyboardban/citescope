"""Versioned Gemini taxonomy fields for econometric model inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


GEMINI_TAXONOMY_VERSION = "gemini_3_1_flash_lite_taxonomy_v1"
GEMINI_TAXONOMY_FILENAME = "all_pages_gemini_taxonomy_classifications.csv"

GEMINI_PAGE_TYPE = "page_type_general_gemini_v1"
GEMINI_PAGE_FAMILY = "page_type_family_gemini_v1"
GEMINI_SOURCE_TYPE = "source_type_general_gemini_v1"
GEMINI_SITE_TYPE = "site_type_general_gemini_v1"
GEMINI_CONFIDENCE = "taxonomy_confidence_gemini_v1"
GEMINI_INPUT_MODE = "taxonomy_input_mode_gemini_v1"
GEMINI_NEEDS_REVIEW = "taxonomy_needs_review_gemini_v1"

GEMINI_PAGE_FAMILY_COLLAPSED = f"{GEMINI_PAGE_FAMILY}_collapsed"
GEMINI_SOURCE_TYPE_COLLAPSED = f"{GEMINI_SOURCE_TYPE}_collapsed"


def default_gemini_taxonomy_path(package: Path | str) -> Path:
    """Return the canonical URL-level Gemini taxonomy output for this package."""
    package_path = Path(package).resolve()
    return (
        package_path.parent
        / "tables/gemini_page_taxonomy_batch"
        / GEMINI_TAXONOMY_FILENAME
    )


def _clean_category(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).str.strip().replace("", "unknown")


def _collapse_mapping(series: pd.Series, *, min_rows: int) -> dict[str, str]:
    counts = _clean_category(series).value_counts(dropna=False)
    return {
        str(category): (
            "unknown"
            if str(category) == "unknown"
            else (str(category) if int(count) >= min_rows else "rare_other")
        )
        for category, count in counts.items()
    }


def attach_gemini_taxonomy(
    frame: pd.DataFrame,
    package: Path | str,
    *,
    taxonomy_path: Path | str | None = None,
    min_rows: int = 20,
    collapse_mappings: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], dict[str, int | float | str]]:
    """Join Gemini labels by URL and create LPM-safe collapsed categories.

    Collapsing is based only on category support, never on the citation outcome.
    Pass mappings learned on the main measurable sample when applying the same
    taxonomy encoding to another sample.
    """
    if "normalized_url" not in frame:
        raise ValueError("Gemini taxonomy join requires normalized_url.")

    path = Path(taxonomy_path).resolve() if taxonomy_path else default_gemini_taxonomy_path(package)
    if not path.exists():
        raise FileNotFoundError(f"Gemini taxonomy output not found: {path}")

    source_columns = (
        "normalized_url",
        "llm_page_type_general",
        "llm_page_type_family_general",
        "llm_site_type_general",
        "llm_confidence",
        "classification_input_mode",
        "llm_needs_review",
        "result_valid",
    )
    taxonomy = pd.read_csv(path, usecols=lambda column: column in source_columns, low_memory=False)
    if taxonomy["normalized_url"].duplicated().any():
        duplicates = int(taxonomy["normalized_url"].duplicated(keep=False).sum())
        raise ValueError(f"Gemini taxonomy URL key is not unique ({duplicates} duplicate rows).")

    taxonomy = taxonomy.rename(
        columns={
            "llm_page_type_general": GEMINI_PAGE_TYPE,
            "llm_page_type_family_general": GEMINI_PAGE_FAMILY,
            "llm_site_type_general": GEMINI_SOURCE_TYPE,
            "llm_confidence": GEMINI_CONFIDENCE,
            "classification_input_mode": GEMINI_INPUT_MODE,
            "llm_needs_review": GEMINI_NEEDS_REVIEW,
            "result_valid": "taxonomy_result_valid_gemini_v1",
        }
    )
    taxonomy[GEMINI_SITE_TYPE] = taxonomy[GEMINI_SOURCE_TYPE]
    label_columns = (GEMINI_PAGE_TYPE, GEMINI_PAGE_FAMILY, GEMINI_SOURCE_TYPE, GEMINI_SITE_TYPE)
    for column in label_columns:
        taxonomy[column] = _clean_category(taxonomy[column])

    existing = [column for column in taxonomy.columns if column != "normalized_url" and column in frame]
    data = frame.drop(columns=existing).merge(
        taxonomy,
        on="normalized_url",
        how="left",
        validate="many_to_one",
    ).copy()
    matched = data[GEMINI_PAGE_TYPE].notna()
    for column in label_columns:
        data[column] = _clean_category(data[column])
    data[GEMINI_CONFIDENCE] = _clean_category(data[GEMINI_CONFIDENCE])
    data[GEMINI_INPUT_MODE] = _clean_category(data[GEMINI_INPUT_MODE])
    data[GEMINI_NEEDS_REVIEW] = (
        data[GEMINI_NEEDS_REVIEW]
        .replace({"true": True, "True": True, "false": False, "False": False, 1: True, 0: False})
        .fillna(True)
        .astype(bool)
    )

    mappings = {key: dict(value) for key, value in (collapse_mappings or {}).items()}
    for source, target in (
        (GEMINI_PAGE_FAMILY, GEMINI_PAGE_FAMILY_COLLAPSED),
        (GEMINI_SOURCE_TYPE, GEMINI_SOURCE_TYPE_COLLAPSED),
    ):
        mapping = mappings.get(source) or _collapse_mapping(data[source], min_rows=min_rows)
        mappings[source] = mapping
        data[target] = data[source].map(mapping).fillna(
            data[source].where(data[source].eq("unknown"), "rare_other")
        )

    audit: dict[str, int | float | str] = {
        "taxonomy_version": GEMINI_TAXONOMY_VERSION,
        "taxonomy_path": str(path),
        "rows": int(len(data)),
        "matched_rows": int(matched.sum()),
        "unmatched_rows": int((~matched).sum()),
        "match_rate": float(matched.mean()) if len(data) else 0.0,
        "unique_urls": int(data["normalized_url"].nunique()),
        "matched_unique_urls": int(data.loc[matched, "normalized_url"].nunique()),
    }
    return data, mappings, audit
