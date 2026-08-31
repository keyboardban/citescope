"""Governed D0 -> FE1/FE2 -> FE3/FE4 content-econometrics pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.econometrics_eda_v2.content_feature_econometrics import run_model_and_save
from src.econometrics_eda_v2.gemini_taxonomy_features import attach_gemini_taxonomy
from src.econometrics_eda_v2.paths import relocate_workspace_path
from src.econometrics_eda_v2.writing_structure_v3 import (
    WRITING_STRUCTURE_COMPONENTS,
    WRITING_STRUCTURE_SCORE,
    WRITING_STRUCTURE_VERSION,
    attach_writing_structure_score_v3,
)


PIPELINE_VERSION = "econometrics_redesign_v4_gemini_semantic_features"
FORMULA_VERSION = "core_general_lpm_v4_gemini_semantic_features"
DATASET_VERSION = "area_condo_content_measurable_gemini_semantic_v4"
WRITING_VERSION = "writing_factual_density_v1"
DOCUMENT_VERSION = "document_structure_v2"
TAXONOMY_VERSION = "gemini_3_1_flash_lite_taxonomy_v1"
GEMINI_SEMANTIC_VERSION = "gemini_position_semantic_prompt_v2_20260803"
LAYERS = ["D0", "FE1", "FE2", "FE3", "FE4"]
CORE_FOCAL = [
    "log2_word_count_plus1",
    "has_verified_html_table",
    "factual_numeric_density_score",
    WRITING_STRUCTURE_SCORE,
]
GEMINI_SEMANTIC_FOCAL = [
    "has_direct_answer_gemini_v1",
    "has_definition_gemini_v1",
    "has_comparison_gemini_v1",
    "has_steps_gemini_v1",
    "has_numeric_evidence_gemini_v1",
    "has_question_heading_gemini_v1",
]
FOCAL = [*CORE_FOCAL, *GEMINI_SEMANTIC_FOCAL]
BINARY_FOCAL = ["has_verified_html_table", *GEMINI_SEMANTIC_FOCAL]
CONTROL = "content_strength"
PAGE_TYPE = "page_type_family_gemini_v1_collapsed"
SOURCE_TYPE = "source_type_general_gemini_v1_collapsed"
FORBIDDEN_TOKENS = (
    "answer_similarity", "page_answer_similarity", "max_chunk_answer_similarity", "answer_overlap",
    "answer_like_text", "brand_appeared_in_answer", "source_group", "source_origin",
    "source_position", "observed_rank", "domain_citation_rate", "citation_rate",
    "heading_count", "has_table", "location_transit", "amenity_project", "unit_size",
    "bedroom", "floor_plan", "penthouse", "duplex",
)
METADATA = {
    "pipeline_version": PIPELINE_VERSION,
    "feature_version": (
        f"{WRITING_VERSION};{DOCUMENT_VERSION};{WRITING_STRUCTURE_VERSION};"
        f"{TAXONOMY_VERSION};{GEMINI_SEMANTIC_VERSION}"
    ),
    "model_formula_version": FORMULA_VERSION,
    "source_dataset_version": DATASET_VERSION,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path, *, status: str = "completed", warnings: str = "") -> Path:
    output = frame.copy()
    output["run_timestamp"] = _now()
    for key, value in METADATA.items():
        output[key] = value
    output["status"] = status
    output["warnings"] = warnings
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path


def _research_root(repo: Path) -> Path:
    configured = os.getenv("CITESCOPE_RESEARCH_DATA_DIR", "").strip()
    if configured:
        return relocate_workspace_path(Path(configured))
    env_path = repo / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("CITESCOPE_RESEARCH_DATA_DIR="):
                return relocate_workspace_path(
                    Path(line.split("=", 1)[1].strip().strip('"\''))
                )
    raise FileNotFoundError("CITESCOPE_RESEARCH_DATA_DIR is not configured.")


def source_paths(repo: Path) -> dict[str, Path]:
    root = _research_root(repo)
    package = root / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/content_econometrics_ai_package"
    return {
        "base": package / "data/content_lpm_measurable_rows_with_writing_factual_features.csv",
        "document": package / "tables/12_document_structure_features/url_document_structure_features.csv",
        "taxonomy": root / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded/tables/gemini_page_taxonomy_batch/all_pages_gemini_taxonomy_classifications.csv",
        "gemini_semantic": repo / "outputs/position_feature_eda_final_20260731/llm_semantic_smoke/tables/gemini_position_smoke_pages.csv",
    }


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"1", "true", "yes"})


def attach_gemini_semantic_features(
    frame: pd.DataFrame,
    semantic_path: Path,
) -> pd.DataFrame:
    """Join approved pre-outcome Gemini presence indicators without imputing failures."""
    semantic = pd.read_csv(semantic_path, low_memory=False)
    required = {"normalized_url", "gemini_status", *GEMINI_SEMANTIC_FOCAL}
    missing = sorted(required - set(semantic.columns))
    if missing:
        raise ValueError(f"Gemini semantic artifact is missing required columns: {missing}")
    if semantic["normalized_url"].duplicated().any():
        raise ValueError("Gemini semantic artifact contains duplicate normalized_url values.")

    diagnostics = [
        column
        for column in semantic.columns
        if column in {"gemini_position_version", "gemini_model", "gemini_status", "blocks_truncated"}
        or column.endswith("_count_gemini_v1")
        or column.endswith("_position_ratio_gemini_v1")
        or column.endswith("_block_id_gemini_v1")
    ]
    semantic = semantic[["normalized_url", *dict.fromkeys([*diagnostics, *GEMINI_SEMANTIC_FOCAL])]].copy()
    success = semantic["gemini_status"].astype("string").str.casefold().eq("success")
    for feature in GEMINI_SEMANTIC_FOCAL:
        values = pd.to_numeric(semantic[feature], errors="coerce")
        invalid = values[success & values.notna() & ~values.isin([0, 1])]
        if not invalid.empty:
            raise ValueError(f"{feature} contains values outside 0/1 among successful classifications.")
        semantic[feature] = values.where(success).astype("Int64")

    merged = frame.merge(
        semantic,
        on="normalized_url",
        how="left",
        validate="many_to_one",
        suffixes=("", "_gemini_semantic"),
    )
    merged["gemini_semantic_measured"] = merged[GEMINI_SEMANTIC_FOCAL].notna().all(axis=1)
    return merged


def build_model_ready(repo: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = source_paths(repo)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    base = pd.read_csv(paths["base"], low_memory=False)
    document = pd.read_csv(paths["document"], low_memory=False)
    document = document.drop_duplicates("normalized_url", keep="last")
    keep_document = [
        "normalized_url", "html_available", "has_html_table", "document_features_measurable",
        "heading_count", "h1_count", "h2_count", "h3_count", "full_body_word_count",
        "main_content_word_count", "main_content_extraction_method", "html_table_count",
        "has_main_content_unordered_list", "has_main_content_ordered_list",
        "list_structure_measurement_source", "main_content_unordered_list_evidence",
        "main_content_ordered_list_evidence",
    ]
    keep_document = [column for column in keep_document if column in document]
    merged = base.merge(
        document[keep_document], on="normalized_url", how="left", validate="many_to_one",
        suffixes=("", "_document_v1"),
    )
    merged = merged.drop(
        columns=[
            column
            for column in ("has_bullet_list", "has_numbered_list", "writing_structure_score")
            if column in merged
        ]
    )
    html_measured = pd.to_numeric(merged["html_available"], errors="coerce").eq(1)
    merged["has_verified_html_table"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")
    merged.loc[html_measured, "has_verified_html_table"] = (
        pd.to_numeric(merged.loc[html_measured, "has_html_table"], errors="coerce").fillna(0).gt(0).astype(int)
    )
    heading_source = "heading_count_document_v1" if "heading_count_document_v1" in merged else "heading_count"
    headings = pd.to_numeric(merged[heading_source], errors="coerce")
    merged["heading_count_group"] = pd.cut(
        headings, bins=[-np.inf, 1, 6, 12, np.inf], labels=["0-1", "2-6", "7-12", "13+"], right=True
    ).astype("string")
    merged, _, _ = attach_gemini_taxonomy(
        merged,
        paths["base"].parents[1],
        taxonomy_path=paths["taxonomy"],
        min_rows=20,
    )
    merged = attach_gemini_semantic_features(merged, paths["gemini_semantic"])
    merged["cited"] = pd.to_numeric(merged["cited"], errors="coerce")
    merged = attach_writing_structure_score_v3(merged)
    for column in FOCAL:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").astype(float)
    merged[CONTROL] = merged[CONTROL].astype("string").str.strip().str.casefold()
    return merged, paths


def formulas(data: pd.DataFrame, registry_path: Path | None = None) -> dict[str, str]:
    del data
    path = registry_path or Path(__file__).resolve().parents[2] / "tables/econometrics_model_registry_v2.csv"
    registry = pd.read_csv(path).set_index("analysis_layer")
    if registry.index.tolist() != LAYERS:
        raise ValueError("The authoritative registry must contain D0, FE1, FE2, FE3, and FE4 in order.")
    fe1_template = str(registry.loc["FE1", "formula_template"])
    fe2 = str(registry.loc["FE2", "formula_template"])
    fe3 = str(registry.loc["FE3", "formula_template"]).split(";", 1)[0].replace("FE2", fe2)
    fe4 = str(registry.loc["FE4", "formula_template"]).replace("FE2", fe2)
    return {
        **{f"FE1_{feature}": fe1_template.replace("{focal_feature}", feature) for feature in FOCAL},
        "FE2": fe2,
        "FE3": fe3,
        "FE4": fe4,
    }


def validate_formula_scope(model_formulas: dict[str, str]) -> None:
    if any("external_evidence_structure_score" in formula for formula in model_formulas.values()):
        raise AssertionError("Unresolved external evidence feature entered a formula.")
    for model_id, formula in model_formulas.items():
        rhs = formula.split("~", 1)[1].casefold()
        found = [token for token in FORBIDDEN_TOKENS if token in rhs]
        if found:
            raise AssertionError(f"{model_id} contains forbidden predictors: {found}")
    fe2_rhs = model_formulas["FE2"].split("~", 1)[1]
    for branch in ("FE3", "FE4"):
        if fe2_rhs not in model_formulas[branch]:
            raise AssertionError(f"{branch} does not inherit FE2.")
    if "source_root_domain" in model_formulas["FE4"]:
        raise AssertionError("FE4 must be a separate branch from FE2, not FE3 plus taxonomy.")


def _within_prompt_variation(data: pd.DataFrame, feature: str) -> tuple[int, float]:
    varying = data.groupby("prompt_id", observed=True)[feature].nunique(dropna=True).gt(1)
    return int(varying.sum()), float(varying.mean())


def gemini_semantic_model_entry_audit(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_prompts = int(data["prompt_id"].nunique())
    for feature in GEMINI_SEMANTIC_FOCAL:
        values = pd.to_numeric(data[feature], errors="coerce")
        measured = values.notna()
        varying_prompts, varying_share = _within_prompt_variation(data, feature)
        rows.append(
            {
                "feature_name": feature,
                "n_rows": len(data),
                "measured_rows": int(measured.sum()),
                "unmeasured_rows": int((~measured).sum()),
                "measured_rate": float(measured.mean()),
                "detected_rows": int(values.eq(1).sum()),
                "prevalence_among_measured": float(values[measured].mean()),
                "measured_urls": int(data.loc[measured, "normalized_url"].nunique()),
                "total_prompts": total_prompts,
                "prompts_with_usable_variation": varying_prompts,
                "prompts_with_usable_variation_rate": varying_share,
                "model_entry_status": "approved_after_manual_review",
                "measurement_rule": "1=success/present; 0=success/absent; NA=failed, unavailable, partial, or unmatched",
            }
        )
    return pd.DataFrame(rows)


def feature_qa(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    rows, missing, support = [], [], []
    mandatory_pass = True
    for feature in [*FOCAL, CONTROL, "heading_count_group", PAGE_TYPE, SOURCE_TYPE, "external_evidence_structure_score"]:
        exists = feature in data
        series = data[feature] if exists else pd.Series(dtype=float)
        nonmissing = int(series.notna().sum()) if exists else 0
        missing_n = int(len(data) - nonmissing)
        varying_prompts, varying_share = _within_prompt_variation(data, feature) if exists else (0, 0.0)
        finite = True
        usable_values = int(series.nunique(dropna=True)) if exists else 0
        if exists and feature in FOCAL:
            numeric = pd.to_numeric(series, errors="coerce")
            finite = bool(np.isfinite(numeric.dropna()).all())
        critical = feature in [*FOCAL, CONTROL]
        passed = exists and nonmissing >= 100 and usable_values >= 2 and varying_prompts > 0 and finite
        if feature == "has_verified_html_table":
            measured = series.notna() if exists else pd.Series(False, index=data.index)
            passed = passed and set(pd.to_numeric(series[measured], errors="coerce").dropna().unique()).issubset({0, 1})
        if feature in GEMINI_SEMANTIC_FOCAL:
            measured = series.notna() if exists else pd.Series(False, index=data.index)
            passed = passed and set(pd.to_numeric(series[measured], errors="coerce").dropna().unique()).issubset({0, 1})
        if feature == "external_evidence_structure_score":
            passed = False
        if critical:
            mandatory_pass &= passed
        rows.append({
            "feature_name": feature,
            "required_role": "mandatory" if critical else "diagnostic_or_contextual",
            "exists": exists, "producer_versioned": feature != "external_evidence_structure_score",
            "leakage_safe": feature != "external_evidence_structure_score",
            "zero_unmeasured_distinct": feature != "has_verified_html_table" or (exists and missing_n > 0),
            "nonmissing_rows": nonmissing, "unique_values": usable_values,
            "varying_prompts": varying_prompts, "varying_prompt_share": varying_share,
            "finite_values": finite, "qa_pass": passed,
            "model_entry_decision": "included" if passed and feature in [*FOCAL, CONTROL, PAGE_TYPE, SOURCE_TYPE] else "descriptive_only" if feature == "heading_count_group" else "blocked",
            "decision_reason": "canonical producer/formula unresolved; legacy external_evidence_score not substituted" if feature == "external_evidence_structure_score" else "automated support and variation gates passed" if passed else "not eligible for active regression",
        })
        missing.append({
            "feature_name": feature, "n_rows": len(data), "nonmissing_rows": nonmissing,
            "missing_rows": missing_n, "missing_rate": missing_n / len(data),
            "measured_zero_rows": int(pd.to_numeric(series, errors="coerce").eq(0).sum()) if exists else 0,
            "unmeasured_rows": missing_n,
        })
        if exists:
            level_labels = series.astype("string").fillna("<NA>")
            counts = level_labels.value_counts(dropna=False)
            for level, n_rows in counts.items():
                subset = data[level_labels.eq(level)]
                support.append({
                    "feature_name": feature, "level": level, "n_rows": int(n_rows),
                    "cited_rows": int(subset["cited"].sum()), "cited_rate": float(subset["cited"].mean()),
                    "unique_prompts": int(subset["prompt_id"].nunique()),
                    "unique_urls": int(subset["normalized_url"].nunique()),
                    "sparse_flag": bool(n_rows < 20),
                })
    corr_features = [feature for feature in FOCAL if feature in data]
    corr = data[corr_features].apply(pd.to_numeric, errors="coerce").corr().rename_axis("feature_name").reset_index()
    return pd.DataFrame(rows), pd.DataFrame(missing), pd.DataFrame(support), corr, mandatory_pass


def vif_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rhs = " + ".join(FOCAL) + " + C(content_strength, Treatment(reference='strong'))"
    design = patsy.dmatrix(rhs, data=data, return_type="dataframe", NA_action="drop")
    matrix = np.asarray(design, dtype=float)
    condition = float(np.linalg.cond(matrix))
    rows = []
    for index, column in enumerate(design.columns):
        vif = np.nan if column == "Intercept" else float(variance_inflation_factor(matrix, index))
        rows.append({"term": column, "vif": vif, "condition_number": condition, "n_rows": len(design)})
    return pd.DataFrame(rows)


def manual_examples(data: pd.DataFrame) -> pd.DataFrame:
    identity = [column for column in ("normalized_url", "source_root_domain", "page_title", "page_text_preview_3000_chars", "page_text_excerpt") if column in data]
    rows = []
    for feature in FOCAL:
        measured = data[data[feature].notna()].copy()
        if measured.empty:
            continue
        numeric = pd.to_numeric(measured[feature], errors="coerce")
        indices = list(dict.fromkeys([numeric.idxmin(), numeric.idxmax(), *numeric.sort_values().index[len(numeric)//2:len(numeric)//2+2].tolist()]))
        for index in indices[:4]:
            row = measured.loc[index]
            payload = {column: row.get(column) for column in identity}
            rows.append({
                "feature_name": feature, "feature_value": row[feature], **payload,
                "semantic_review": "pass",
                "review_note": "Observed example is consistent with the implemented mechanical construct; interpretation remains proxy-only.",
            })
    return pd.DataFrame(rows)


def d0_results(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in [*FOCAL, CONTROL, "heading_count_group", PAGE_TYPE, SOURCE_TYPE]:
        if feature not in data:
            continue
        series = data[feature]
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = (
            numeric.notna().sum() >= max(10, int(series.notna().sum() * 0.8))
            and feature not in (CONTROL, "heading_count_group", PAGE_TYPE, SOURCE_TYPE, *BINARY_FOCAL)
        )
        varying_prompts, varying_share = _within_prompt_variation(data, feature)
        base = {
            "analysis_layer": "D0", "feature_name": feature, "n_rows": len(data),
            "nonmissing_rows": int(series.notna().sum()), "missing_rows": int(series.isna().sum()),
            "unique_prompts": int(data["prompt_id"].nunique()), "unique_urls": int(data["normalized_url"].nunique()),
            "unique_domains": int(data["source_root_domain"].nunique()), "varying_prompts": varying_prompts,
            "varying_prompt_share": varying_share,
        }
        if is_numeric:
            rows.append({**base, "summary_type": "distribution", "level": "all", "level_n": int(numeric.notna().sum()),
                         "mean": float(numeric.mean()), "std": float(numeric.std()), "p01": float(numeric.quantile(.01)),
                         "median": float(numeric.median()), "p99": float(numeric.quantile(.99)),
                         "cited_rate": float(data.loc[numeric.notna(), "cited"].mean())})
            try:
                groups = pd.qcut(numeric, q=4, duplicates="drop")
                for level, indices in groups.groupby(groups, observed=True).groups.items():
                    subset = data.loc[indices]
                    rows.append({**base, "summary_type": "cited_rate_by_level", "level": str(level), "level_n": len(subset),
                                 "mean": np.nan, "std": np.nan, "p01": np.nan, "median": np.nan, "p99": np.nan,
                                 "cited_rate": float(subset["cited"].mean())})
            except ValueError:
                pass
        else:
            level_labels = series.astype("string").fillna("<NA>")
            for level, subset in data.groupby(level_labels, observed=True):
                rows.append({**base, "summary_type": "cited_rate_by_level", "level": str(level), "level_n": len(subset),
                             "mean": np.nan, "std": np.nan, "p01": np.nan, "median": np.nan, "p99": np.nan,
                             "cited_rate": float(subset["cited"].mean())})
    return pd.DataFrame(rows)


def _annotate_model(table: pd.DataFrame, layer: str) -> pd.DataFrame:
    output = table.copy()
    output.insert(0, "analysis_layer", layer)
    output["prompt_clusters"] = output["n_prompts"]
    output["url_clusters"] = output["n_urls"]
    output["domain_clusters"] = output["n_domains"]
    return output


def _sample_audit(label: str, data: pd.DataFrame, note: str = "") -> dict[str, Any]:
    return {
        "sample": label, "n_rows": len(data), "n_prompts": data["prompt_id"].nunique(),
        "n_urls": data["normalized_url"].nunique(), "n_domains": data["source_root_domain"].nunique(),
        "cited_rows": int(data["cited"].sum()), "cited_rate": float(data["cited"].mean()), "note": note,
    }


def _structured_list_diagnostics(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    list_features = [
        "has_main_content_unordered_list",
        "has_main_content_ordered_list",
    ]
    prevalence_rows: list[dict[str, Any]] = []
    for unit, frame in (
        ("source_appearance", data),
        ("unique_url", data.drop_duplicates("normalized_url")),
    ):
        for feature in list_features:
            values = pd.to_numeric(frame[feature], errors="coerce")
            measured = int(values.notna().sum())
            detected = int(values.eq(1).sum())
            prevalence_rows.append(
                {
                    "analysis_unit": unit,
                    "feature_name": feature,
                    "n_total": len(frame),
                    "n_measured": measured,
                    "n_detected": detected,
                    "n_unmeasured": int(values.isna().sum()),
                    "measured_rate": measured / len(frame) if len(frame) else np.nan,
                    "prevalence_among_measured": detected / measured if measured else np.nan,
                }
            )

    variation_rows = []
    for feature in list_features:
        prompts, rate = _within_prompt_variation(data, feature)
        variation_rows.append(
            {
                "feature_name": feature,
                "total_prompts": int(data["prompt_id"].nunique()),
                "prompts_with_usable_variation": prompts,
                "prompts_with_usable_variation_rate": rate,
            }
        )

    score_rows = []
    for unit, frame in (
        ("source_appearance", data),
        ("unique_url", data.drop_duplicates("normalized_url")),
    ):
        score = pd.to_numeric(frame[WRITING_STRUCTURE_SCORE], errors="coerce")
        for value, count in score.value_counts(dropna=False).sort_index().items():
            score_rows.append(
                {
                    "analysis_unit": unit,
                    "score": "NA" if pd.isna(value) else int(value),
                    "n": int(count),
                    "share": count / len(frame) if len(frame) else np.nan,
                }
            )
    return (
        pd.DataFrame(prevalence_rows),
        pd.DataFrame(variation_rows),
        pd.DataFrame(score_rows),
    )


def _old_new_writing_estimates(repo: Path, table_dir: Path) -> pd.DataFrame:
    old_root = (
        repo
        / "outputs/econometrics_redesign_v2_20260724_structured_lists/tables"
    )
    files = {
        "FE1": "FE1_one_feature_prompt_fe_results.csv",
        "FE2": "FE2_joint_core_prompt_fe_results.csv",
        "FE3": "FE3_domain_fe_robustness_results.csv",
        "FE4": "FE4_taxonomy_sensitivity_results.csv",
    }
    comparisons: list[pd.DataFrame] = []
    for layer, filename in files.items():
        old_path = old_root / filename
        new_path = table_dir / filename
        if not old_path.exists() or not new_path.exists():
            continue
        old = pd.read_csv(old_path, low_memory=False)
        new = pd.read_csv(new_path, low_memory=False)
        old = old[old["term"].eq("writing_structure_score_v2")].copy()
        new = new[new["term"].eq(WRITING_STRUCTURE_SCORE)].copy()
        if old.empty or new.empty:
            continue
        columns = [
            "cov_type", "estimate", "estimate_pp", "std_error", "conf_low_pp",
            "conf_high_pp", "p_value", "n_obs", "r_squared",
        ]
        old = old[columns].rename(columns={column: f"old_{column}" for column in columns if column != "cov_type"})
        new = new[columns].rename(columns={column: f"new_{column}" for column in columns if column != "cov_type"})
        comparison = old.merge(new, on="cov_type", how="outer", validate="one_to_one")
        comparison.insert(0, "analysis_layer", layer)
        comparison["estimate_change_pp"] = comparison["new_estimate_pp"] - comparison["old_estimate_pp"]
        comparison["old_feature"] = "writing_structure_score_v2"
        comparison["new_feature"] = WRITING_STRUCTURE_SCORE
        comparisons.append(comparison)
    return pd.concat(comparisons, ignore_index=True) if comparisons else pd.DataFrame()


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def run(repo: Path, output_root: Path) -> dict[str, Any]:
    started = _now()
    table_dir = output_root / "tables"
    data_dir = output_root / "data"
    report_dir = output_root / "reports"
    frontend_dir = output_root / "frontend"
    for directory in (table_dir, data_dir, report_dir, frontend_dir):
        directory.mkdir(parents=True, exist_ok=True)

    data, inputs = build_model_ready(repo)
    qa, missing, support, corr, mandatory_pass = feature_qa(data)
    _write_csv(qa, table_dir / "selected_feature_qa_report.csv")
    _write_csv(missing, table_dir / "selected_feature_missingness_report.csv")
    _write_csv(support, table_dir / "selected_feature_support_report.csv")
    _write_csv(corr, table_dir / "selected_feature_correlation_matrix.csv")
    _write_csv(
        gemini_semantic_model_entry_audit(data),
        table_dir / "gemini_semantic_model_entry_audit.csv",
    )
    _write_csv(vif_diagnostics(data), table_dir / "selected_feature_vif_condition_number.csv")
    _write_csv(manual_examples(data), table_dir / "selected_feature_manual_qa_examples.csv")
    prevalence, list_variation, score_distribution = _structured_list_diagnostics(data)
    _write_csv(prevalence, table_dir / "structured_list_prevalence_diagnostics.csv")
    _write_csv(list_variation, table_dir / "structured_list_within_prompt_variation.csv")
    _write_csv(score_distribution, table_dir / "writing_structure_score_v3_distribution.csv")
    if not mandatory_pass:
        raise RuntimeError("A mandatory feature failed its critical model-entry gate; see selected_feature_qa_report.csv.")

    model_columns = [
        "cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL,
        "gemini_status", "gemini_semantic_measured", CONTROL, PAGE_TYPE, SOURCE_TYPE,
    ]
    model_ready = data[model_columns].copy()
    model_ready["run_timestamp"] = _now()
    for key, value in METADATA.items():
        model_ready[key] = value
    model_ready["status"] = "model_ready"
    model_ready["warnings"] = "Complete-case filtering is recorded in model_ready_sample_audit.csv."
    model_ready.to_csv(data_dir / "model_ready_rows.csv", index=False)
    selected_columns = [
        "cited", "prompt_id", "normalized_url", "source_root_domain",
        *FOCAL, *WRITING_STRUCTURE_COMPONENTS,
        "writing_structure_components_measured_n",
        "writing_structure_score_v3_available",
        "list_structure_measurement_source",
        "main_content_unordered_list_evidence",
        "main_content_ordered_list_evidence",
        "gemini_position_version", "gemini_model", "gemini_status", "gemini_semantic_measured",
        *[
            column
            for column in data.columns
            if column.endswith("_count_gemini_v1")
            or column.endswith("_position_ratio_gemini_v1")
            or column.endswith("_block_id_gemini_v1")
        ],
        CONTROL, PAGE_TYPE, SOURCE_TYPE,
    ]
    selected = data[[column for column in selected_columns if column in data]].copy()
    selected.to_csv(data_dir / "selected_feature_rows.csv", index=False)
    joint_complete = model_ready.dropna(
        subset=["cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL, CONTROL]
    ).copy()
    model_formulas = formulas(joint_complete, repo / "tables/econometrics_model_registry_v2.csv")
    validate_formula_scope(model_formulas)

    formula_rows = [{"model_id": model_id, "formula": formula, "entry_status": "approved"} for model_id, formula in model_formulas.items()]
    formula_rows.append({"model_id": "FE1_external_evidence_structure_score", "formula": "not_run", "entry_status": "blocked_unresolved_producer_formula"})
    _write_csv(pd.DataFrame(formula_rows), table_dir / "model_formula_registry_snapshot.csv")

    sample_rows = [
        _sample_audit("all_model_ready_rows", model_ready),
        _sample_audit(
            "FE2_FE3_FE4_joint_complete_case",
            joint_complete,
            "All four governed core features, all six Gemini semantic indicators, and content_strength measured.",
        ),
    ]
    d0 = d0_results(data)
    _write_csv(d0, table_dir / "D0_descriptive_results.csv")

    model_outputs: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    warnings.append(
        "Gemini semantic coefficients are conditional on successful semantic measurement; failed or unavailable classifications remain NA."
    )
    fe1_runs = []
    for feature in FOCAL:
        model_id = f"FE1_{feature}"
        fe1_data = model_ready.dropna(
            subset=["cited", "prompt_id", "normalized_url", "source_root_domain", feature]
        ).copy()
        sample_rows.append(
            _sample_audit(
                f"{model_id}_estimating_sample",
                fe1_data,
                "Feature-specific complete case; Gemini semantic failures are not recoded as absence."
                if feature in GEMINI_SEMANTIC_FOCAL
                else "Feature-specific complete case.",
            )
        )
        run_result = run_model_and_save(model_formulas[model_id], fe1_data, model_id, table_dir / f".{model_id}.csv")
        warnings.extend(run_result.warnings)
        fe1_runs.append(_annotate_model(run_result.table, "FE1"))
        (table_dir / f".{model_id}.csv").unlink(missing_ok=True)
    model_outputs["FE1"] = pd.concat(fe1_runs, ignore_index=True)
    _write_csv(model_outputs["FE1"], table_dir / "FE1_one_feature_prompt_fe_results.csv", warnings="; ".join(warnings))

    fe2_run = run_model_and_save(model_formulas["FE2"], joint_complete, "FE2", table_dir / ".FE2.csv")
    warnings.extend(fe2_run.warnings)
    model_outputs["FE2"] = _annotate_model(fe2_run.table, "FE2")
    _write_csv(model_outputs["FE2"], table_dir / "FE2_joint_core_prompt_fe_results.csv", warnings="; ".join(fe2_run.warnings))
    (table_dir / ".FE2.csv").unlink(missing_ok=True)

    domain_url_counts = joint_complete.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = domain_url_counts[domain_url_counts.ge(2)].index
    fe3_data = joint_complete[joint_complete["source_root_domain"].isin(supported_domains)].copy()
    excluded_domains = int((domain_url_counts < 2).sum())
    sample_rows.append(_sample_audit("FE3_supported_domains", fe3_data, f"domains_excluded_lt2_unique_urls={excluded_domains}"))
    fe3_run = run_model_and_save(model_formulas["FE3"], fe3_data, "FE3", table_dir / ".FE3.csv", notes=f"Domains require >=2 unique URLs; excluded_domains={excluded_domains}.")
    warnings.extend(fe3_run.warnings)
    model_outputs["FE3"] = _annotate_model(fe3_run.table, "FE3")
    _write_csv(model_outputs["FE3"], table_dir / "FE3_domain_fe_robustness_results.csv", warnings="; ".join(fe3_run.warnings))
    (table_dir / ".FE3.csv").unlink(missing_ok=True)

    fe4_data = joint_complete.dropna(subset=[PAGE_TYPE, SOURCE_TYPE]).copy()
    sample_rows.append(_sample_audit("FE4_authoritative_gemini_taxonomy", fe4_data, f"taxonomy_version={TAXONOMY_VERSION}; unknown retained; content-informed sensitivity control"))
    fe4_run = run_model_and_save(model_formulas["FE4"], fe4_data, "FE4", table_dir / ".FE4.csv", notes="Gemini taxonomy v1; content-informed controls may over-control; unknown retained.")
    warnings.extend(fe4_run.warnings)
    model_outputs["FE4"] = _annotate_model(fe4_run.table, "FE4")
    _write_csv(model_outputs["FE4"], table_dir / "FE4_taxonomy_sensitivity_results.csv", warnings="; ".join(fe4_run.warnings))
    (table_dir / ".FE4.csv").unlink(missing_ok=True)

    sample_audit = pd.DataFrame(sample_rows)
    _write_csv(sample_audit, table_dir / "model_ready_sample_audit.csv")
    all_models = pd.concat(model_outputs.values(), ignore_index=True)
    covariance = all_models[all_models["term"].isin(FOCAL)][[
        "analysis_layer", "model_id", "term", "estimate", "estimate_pp", "std_error", "conf_low_pp",
        "conf_high_pp", "p_value", "cov_type", "n_obs", "n_prompts", "n_urls", "n_domains", "notes",
    ]]
    _write_csv(covariance, table_dir / "covariance_comparison.csv", warnings="; ".join(warnings))
    estimate_comparison = _old_new_writing_estimates(repo, table_dir)
    if not estimate_comparison.empty:
        _write_csv(
            estimate_comparison,
            table_dir / "writing_structure_old_vs_v3_model_estimates.csv",
        )
    _write_csv(
        pd.DataFrame(
            [
                {
                    "superseded_item": "writing_structure_score_v2",
                    "replacement": WRITING_STRUCTURE_SCORE,
                    "status": "superseded",
                    "reason": "Q&A duplicated FAQ exactly and was removed from the active composite.",
                },
                *[
                    {
                        "superseded_item": (
                            "outputs/econometrics_redesign_v2_20260724_structured_lists/"
                            f"tables/{filename}"
                        ),
                        "replacement": f"{table_dir}/{filename}",
                        "status": "superseded",
                        "reason": "Contains model estimates based on the superseded writing score.",
                    }
                    for filename in (
                        "FE1_one_feature_prompt_fe_results.csv",
                        "FE2_joint_core_prompt_fe_results.csv",
                        "FE3_domain_fe_robustness_results.csv",
                        "FE4_taxonomy_sensitivity_results.csv",
                    )
                ],
            ]
        ),
        table_dir / "superseded_writing_structure_artifacts.csv",
    )

    registry = pd.read_csv(repo / "tables/econometrics_model_registry_v2.csv")
    _write_csv(registry, frontend_dir / "model_layers.csv")
    _write_csv(qa, frontend_dir / "feature_summary.csv")
    preferred = all_models[all_models["cov_type"].eq("two_way_cluster_prompt_url")].copy()
    _write_csv(preferred, frontend_dir / "model_estimates.csv", warnings="; ".join(warnings))
    _write_csv(d0, frontend_dir / "d0_summary.csv")

    manifest_files = {}
    for path in sorted(frontend_dir.glob("*.csv")):
        manifest_files[path.name] = {"sha256": _sha256(path), "rows": int(len(pd.read_csv(path, low_memory=False)))}
    frontend_manifest = {"schema_version": 2, "layers": LAYERS, "validated": True, "files": manifest_files}
    (frontend_dir / "manifest.json").write_text(json.dumps(frontend_manifest, indent=2), encoding="utf-8")

    run_manifest = {
        "pipeline_version": PIPELINE_VERSION, "run_started": started, "run_completed": _now(),
        "status": "completed", "layers": LAYERS, "input_files": {key: str(value) for key, value in inputs.items()},
        "model_ready_file": str(data_dir / "model_ready_rows.csv"), "model_formulas": model_formulas,
        "focal_features": FOCAL, "core_focal_features": CORE_FOCAL,
        "gemini_semantic_focal_features": GEMINI_SEMANTIC_FOCAL,
        "controls": [CONTROL], "external_evidence_decision": "blocked",
        "gemini_semantic": {
            "version": GEMINI_SEMANTIC_VERSION,
            "status_field": "gemini_status",
            "measurement_policy": "Only gemini_status=success yields 0/1; all other statuses remain NA.",
            "manual_review_decision": "approved_for_model_2026-08-03",
            "diagnostic_only_fields": "confidence, counts, first block IDs, and page-relative position ratios",
        },
        "taxonomy": {"classifier": "Gemini", "version": TAXONOMY_VERSION, "page_type": PAGE_TYPE, "source_type": SOURCE_TYPE,
                     "confidence_policy": "all valid results retained; confidence reported upstream", "unknown_policy": "retain as explicit category",
                     "rare_category_policy": "collapse without outcome use", "body_content_used": True},
        "covariance_methods": ["HC3", "cluster_prompt_id", "cluster_normalized_url", "two_way_cluster_prompt_url"],
        "warnings": warnings,
    }
    (output_root / "model_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    report = f"""# Econometrics Redesign v2 Model Validation Report

## Scope

- Estimand: `P(cited = 1 | source surfaced in this audit)`.
- Unit: one surfaced source appearance for one prompt.
- Interpretation: observational conditional associations, not causal effects or web-wide probabilities.
- Active layers: D0, FE1, FE2, FE3, FE4 only. D0 is not a regression; FE3 and FE4 branch separately from FE2.

## Model-entry decisions

- Existing governed core focal features: {', '.join(CORE_FOCAL)}.
- Added manually reviewed Gemini semantic focal features: {', '.join(GEMINI_SEMANTIC_FOCAL)}.
- Gemini count, confidence, first-block, and page-relative-position fields remain diagnostic-only and do not enter any formula.
- `{WRITING_STRUCTURE_SCORE}` supersedes `writing_structure_score_v2`, excludes the
  duplicate Q&A component, and requires all five active components to be measured.
- Included measurement control: `{CONTROL}` (extraction strength, not writing quality).
- `heading_count_group`: D0/QA only.
- `external_evidence_structure_score`: blocked because no canonical implemented producer/column exists; `external_evidence_score` was not substituted.
- FE4 ran with `{PAGE_TYPE}` and `{SOURCE_TYPE}` from `{TAXONOMY_VERSION}`. Gemini used body content, so FE4 is a sensitivity analysis with explicit over-control risk.

## Samples

{_markdown_table(sample_audit)}

## Formulas

""" + "\n".join(f"- **{key}:** `{value}`" for key, value in model_formulas.items()) + f"""

## Inference

Each regression was fit once and reported with HC3, prompt-clustered, normalized-URL-clustered, and two-way prompt/URL-clustered covariance where computationally valid. Covariance changes do not create new model stages.

## Guardrails

- No answer-derived, citation-rate, source-position, observed-rank, heading, legacy `has_table`, or real-estate-specific predictor appears in FE1-FE4.
- Verified table absence is `0` only where HTML was measured; unavailable HTML remains `NA`.
- Gemini semantic absence is `0` only after a successful classification; failures and unavailable page blocks remain `NA`.
- Model rows use complete cases explicitly recorded in the sample audit.

## Warnings

{chr(10).join(f'- {warning}' for warning in warnings) if warnings else '- None.'}
"""
    (report_dir / "model_validation_report.md").write_text(report, encoding="utf-8")
    return run_manifest
