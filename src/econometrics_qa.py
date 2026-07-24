"""Read-only data adapter for the ChatGPT content-econometrics QA interface."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import requests

from .econometrics_eda_v2.paths import CODE_ROOT, topic_output_dir
from .econometrics_eda_v2.general_page_taxonomy import (
    GENERAL_TAXONOMY_VERSION,
    classify_general_page_type,
    classify_general_site_type,
    finalise_general_page_type,
)


PACKAGE_NAME = "content_econometrics_ai_package"
SNAPSHOT_RELATIVE_DIR = Path("tables/area_condo_brightdata_content_pilot/normalized")
SNAPSHOT_MODES = ("crawler_api", "browser_api", "unlocker_api")
GEMINI_TAXONOMY_RELATIVE_PATH = Path(
    "tables/gemini_page_taxonomy_batch/all_pages_gemini_taxonomy_classifications.csv"
)


@dataclass(frozen=True)
class QABundle:
    package_dir: Path
    manifest: dict
    all_rows: pd.DataFrame
    measurable_rows: pd.DataFrame
    writing_rows: pd.DataFrame
    url_evidence: pd.DataFrame
    prompts: pd.DataFrame
    prompt_manifest_path: Path | None
    gemini_taxonomy_path: Path | None


@dataclass(frozen=True)
class QAPreset:
    label: str
    package_dir: Path
    prompt_manifest_path: Path | None
    brightdata_input_path: Path | None
    brightdata_output_path: Path | None
    snapshot_root: Path


def default_package_dir() -> Path:
    return topic_output_dir() / PACKAGE_NAME


def default_gemini_taxonomy_path(package_dir: str | Path | None = None) -> Path:
    configured = os.getenv("CITESCOPE_GEMINI_TAXONOMY_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    package = Path(package_dir) if package_dir else default_package_dir()
    return package.parent / GEMINI_TAXONOMY_RELATIVE_PATH


def _optional_path(env_name: str, default: Path) -> Path | None:
    value = os.getenv(env_name, "").strip()
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.exists() else None


def previous_area_condo_preset() -> QAPreset:
    downloads = Path.home() / "Downloads" / "scope"
    package = default_package_dir()
    return QAPreset(
        label="Previous Area Condo 500",
        package_dir=package,
        prompt_manifest_path=_optional_path(
            "CITESCOPE_AREA_CONDO_PROMPT_MANIFEST",
            downloads / "area_condo_nonbranded_prompt_manifest_500.csv",
        ),
        brightdata_input_path=_optional_path(
            "CITESCOPE_AREA_CONDO_BRIGHTDATA_INPUT",
            downloads / "area_condo_nonbranded_brightdata_input_500.csv",
        ),
        brightdata_output_path=_optional_path(
            "CITESCOPE_AREA_CONDO_BRIGHTDATA_OUTPUT",
            downloads / "01_area_condo_nonbranded_brightdata_run_1.json",
        ),
        snapshot_root=package.parent / SNAPSHOT_RELATIVE_DIR,
    )


def add_general_taxonomy_v2(evidence: pd.DataFrame) -> pd.DataFrame:
    """Attach versioned rule-v2 labels without replacing historical model inputs."""
    out = evidence.copy()
    classified = []
    for _, row in out.iterrows():
        seed = classify_general_page_type(row, include_content=False)
        enriched = classify_general_page_type(row, include_content=True)
        final, source = finalise_general_page_type(
            seed,
            enriched,
            str(row.get("content_quality_flag", "")),
            str(row.get("content_strength", "")),
        )
        classified.append(
            {
                "general_taxonomy_rule_version": GENERAL_TAXONOMY_VERSION,
                "page_type_url_seed_general_rule_v2": seed.detail,
                "page_type_scraped_enriched_general_rule_v2": enriched.detail,
                "page_type_general_rule_v2": final.detail,
                "page_type_family_general_rule_v2": final.family,
                "page_type_general_confidence_rule_v2": final.confidence,
                "page_type_general_source_rule_v2": source,
                "page_type_general_reason_rule_v2": final.reason,
                "site_type_general_rule_v2": classify_general_site_type(row),
            }
        )
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(classified)], axis=1)


def add_gemini_taxonomy(
    evidence: pd.DataFrame,
    classifications: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach optional LLM audit labels without replacing deterministic labels."""
    out = evidence.copy()
    if classifications is None or classifications.empty:
        return out
    if "normalized_url" not in classifications:
        raise ValueError("Gemini taxonomy file is missing normalized_url")
    if classifications["normalized_url"].astype(str).duplicated().any():
        raise ValueError("Gemini taxonomy file contains duplicate normalized_url values")

    status_columns = {
        "result_valid",
        "validation_error",
        "family_repaired",
        "markdown_available",
        "classification_input_mode",
        "llm_agrees_with_rule_v2",
        "rule_v2_unknown_resolved_by_llm",
    }
    columns = [
        column
        for column in classifications.columns
        if column == "normalized_url" or column.startswith("llm_") or column in status_columns
    ]
    attached = classifications[columns].copy()
    attached["normalized_url"] = attached["normalized_url"].astype(str)
    out["normalized_url"] = out["normalized_url"].astype(str)
    collisions = [column for column in attached.columns if column != "normalized_url" and column in out]
    if collisions:
        out = out.drop(columns=collisions)
    return out.merge(attached, on="normalized_url", how="left", validate="one_to_one")


def load_bundle(
    package_dir: str | Path | None = None,
    prompt_manifest_path: str | Path | None = None,
    gemini_taxonomy_path: str | Path | None = None,
) -> QABundle:
    package = Path(package_dir) if package_dir else default_package_dir()
    manifest_path = CODE_ROOT / "config/econometrics_pipeline_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    data = package / "data"
    prompt_reference = data / "prompt_reference.csv"
    full_manifest = Path(prompt_manifest_path) if prompt_manifest_path else None
    prompts_path = full_manifest if full_manifest and full_manifest.exists() else prompt_reference
    url_evidence = add_general_taxonomy_v2(
        pd.read_csv(data / "url_content_evidence_compact.csv", low_memory=False)
    )
    gemini_path = (
        Path(gemini_taxonomy_path)
        if gemini_taxonomy_path
        else default_gemini_taxonomy_path(package)
    )
    gemini = pd.read_csv(gemini_path, low_memory=False) if gemini_path.exists() else None
    url_evidence = add_gemini_taxonomy(url_evidence, gemini)
    return QABundle(
        package_dir=package,
        manifest=manifest,
        all_rows=pd.read_csv(data / "content_lpm_all_surfaced_rows.csv", low_memory=False),
        measurable_rows=pd.read_csv(data / "content_lpm_measurable_rows.csv", low_memory=False),
        writing_rows=pd.read_csv(
            data / "content_lpm_measurable_rows_with_writing_factual_features.csv",
            low_memory=False,
        ),
        url_evidence=url_evidence,
        prompts=pd.read_csv(prompts_path, low_memory=False),
        prompt_manifest_path=full_manifest if full_manifest and full_manifest.exists() else None,
        gemini_taxonomy_path=gemini_path.resolve() if gemini_path.exists() else None,
    )


def bundle_summary(bundle: QABundle) -> dict[str, float | int]:
    cited = pd.to_numeric(bundle.all_rows["cited"], errors="coerce").fillna(0)
    return {
        "full_audit_prompts": int(bundle.prompts["prompt_id"].nunique()),
        "surfaced_rows": len(bundle.all_rows),
        "unique_urls": int(bundle.url_evidence["normalized_url"].nunique()),
        "measurable_rows": len(bundle.measurable_rows),
        "measurable_prompts": int(bundle.measurable_rows["prompt_id"].nunique()),
        "cited_rows": int(cited.sum()),
        "cited_rate": float(cited.mean()),
    }


def taxonomy_comparison_summary(evidence: pd.DataFrame) -> dict[str, float | int]:
    """Summarize agreement without treating either classifier as ground truth."""
    required = {"page_type_general_rule_v2", "llm_page_type_general"}
    if not required.issubset(evidence.columns):
        return {}
    rule = evidence["page_type_general_rule_v2"].fillna("unknown").astype(str)
    llm = evidence["llm_page_type_general"].fillna("unknown").astype(str)
    confidence = evidence.get("llm_confidence", pd.Series("unknown", index=evidence.index)).fillna("unknown").astype(str)
    valid = evidence.get("result_valid", pd.Series(True, index=evidence.index)).fillna(False).astype(bool)
    known = rule.ne("unknown")
    agreement = rule.eq(llm) & valid
    resolved = rule.eq("unknown") & llm.ne("unknown") & valid
    high_disagreement = known & ~agreement & confidence.isin(["high", "medium"]) & valid
    markdown = evidence.get("markdown_available", pd.Series(False, index=evidence.index)).fillna(False).astype(bool)
    return {
        "unique_urls": len(evidence),
        "valid_results": int(valid.sum()),
        "rule_unknown": int(rule.eq("unknown").sum()),
        "llm_unknown": int(llm.eq("unknown").sum()),
        "rule_unknown_resolved": int(resolved.sum()),
        "rule_unknown_still_unknown": int((rule.eq("unknown") & llm.eq("unknown")).sum()),
        "known_rule_urls": int(known.sum()),
        "known_exact_agreement": int((known & agreement).sum()),
        "known_exact_agreement_rate": float(agreement[known].mean()) if known.any() else 0.0,
        "high_medium_confidence_disagreements": int(high_disagreement.sum()),
        "metadata_only_urls": int((~markdown).sum()),
    }


def taxonomy_confusion_table(
    evidence: pd.DataFrame,
    baseline_column: str,
    llm_column: str,
) -> pd.DataFrame:
    """Return URL counts for baseline-by-LLM labels."""
    if baseline_column not in evidence or llm_column not in evidence:
        return pd.DataFrame(columns=["baseline_label", "llm_label", "unique_urls"])
    frame = pd.DataFrame(
        {
            "baseline_label": evidence[baseline_column].fillna("unknown").astype(str),
            "llm_label": evidence[llm_column].fillna("unknown").astype(str),
        }
    )
    return (
        frame.groupby(["baseline_label", "llm_label"], dropna=False)
        .size()
        .reset_index(name="unique_urls")
    )


def snapshot_key(source_url: str) -> str:
    return hashlib.sha256(str(source_url).encode("utf-8")).hexdigest()[:20]


def snapshot_path_for(
    source_url: str,
    *,
    snapshot_root: str | Path | None = None,
) -> Path | None:
    if not str(source_url or "").strip():
        return None
    root = Path(snapshot_root) if snapshot_root else topic_output_dir() / SNAPSHOT_RELATIVE_DIR
    filename = f"{snapshot_key(source_url)}.json"
    for mode in SNAPSHOT_MODES:
        candidate = root / mode / filename
        if candidate.exists():
            return candidate
    return None


def load_snapshot(
    source_url: str,
    *,
    snapshot_root: str | Path | None = None,
) -> tuple[dict | None, Path | None]:
    path = snapshot_path_for(source_url, snapshot_root=snapshot_root)
    if path is None:
        return None, None
    return json.loads(path.read_text("utf-8")), path


def load_model_table(bundle: QABundle, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(bundle.package_dir / relative_path, low_memory=False)


def classify_frame_policy(headers: Mapping[str, str]) -> tuple[str, str]:
    """Classify whether response headers permit cross-origin iframe embedding."""
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    x_frame = normalized.get("x-frame-options", "").strip().lower()
    if x_frame == "deny":
        return "blocked", "X-Frame-Options: DENY"
    if x_frame.startswith("sameorigin"):
        return "blocked", "X-Frame-Options: SAMEORIGIN"

    csp = normalized.get("content-security-policy", "")
    match = re.search(r"(?:^|;)\s*frame-ancestors\s+([^;]+)", csp, flags=re.IGNORECASE)
    if match:
        ancestors = match.group(1).strip().lower()
        if "'none'" in ancestors or ("'self'" in ancestors and "*" not in ancestors):
            return "blocked", f"CSP frame-ancestors {match.group(1).strip()}"
        if "*" in ancestors:
            return "allowed", f"CSP frame-ancestors {match.group(1).strip()}"
        return "restricted", f"CSP frame-ancestors {match.group(1).strip()}"
    if x_frame:
        return "restricted", f"X-Frame-Options: {x_frame}"
    return "unknown", "No blocking frame header detected; browser enforcement may still differ."


def inspect_live_frame_policy(url: str, timeout: float = 8.0) -> dict:
    """Perform an on-demand header check without downloading the response body."""
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "CiteScope-QA/1.0"},
        )
        status, reason = classify_frame_policy(response.headers)
        return {
            "status": status,
            "reason": reason,
            "http_status": response.status_code,
            "final_url": response.url,
            "error": "",
        }
    except requests.RequestException as exc:
        return {
            "status": "unknown",
            "reason": "Header check failed.",
            "http_status": None,
            "final_url": url,
            "error": f"{type(exc).__name__}: {exc}",
        }
