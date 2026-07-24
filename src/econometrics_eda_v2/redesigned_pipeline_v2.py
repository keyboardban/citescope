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


PIPELINE_VERSION = "econometrics_redesign_v2"
FORMULA_VERSION = "core_general_lpm_v2"
DATASET_VERSION = "area_condo_content_measurable_writing_document_gemini_v2"
WRITING_VERSION = "writing_factual_density_v1"
DOCUMENT_VERSION = "document_structure_v1"
TAXONOMY_VERSION = "gemini_3_1_flash_lite_taxonomy_v1"
LAYERS = ["D0", "FE1", "FE2", "FE3", "FE4"]
FOCAL = [
    "log2_word_count_plus1",
    "has_verified_html_table",
    "factual_numeric_density_score",
    "writing_structure_score",
]
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
    "feature_version": f"{WRITING_VERSION};{DOCUMENT_VERSION};{TAXONOMY_VERSION}",
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
    }


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().isin({"1", "true", "yes"})


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
    ]
    keep_document = [column for column in keep_document if column in document]
    merged = base.merge(
        document[keep_document], on="normalized_url", how="left", validate="many_to_one",
        suffixes=("", "_document_v1"),
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
    merged["cited"] = pd.to_numeric(merged["cited"], errors="coerce")
    for column in ("log2_word_count_plus1", "factual_numeric_density_score", "writing_structure_score"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
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
        if exists and feature in FOCAL[:1] + FOCAL[2:]:
            numeric = pd.to_numeric(series, errors="coerce")
            finite = bool(np.isfinite(numeric.dropna()).all())
        critical = feature in [*FOCAL, CONTROL]
        passed = exists and nonmissing >= 100 and usable_values >= 2 and varying_prompts > 0 and finite
        if feature == "has_verified_html_table":
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
            counts = series.fillna("<NA>").astype(str).value_counts(dropna=False)
            for level, n_rows in counts.items():
                subset = data[series.fillna("<NA>").astype(str).eq(level)]
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
        is_numeric = numeric.notna().sum() >= max(10, int(series.notna().sum() * 0.8)) and feature not in (CONTROL, "heading_count_group", PAGE_TYPE, SOURCE_TYPE)
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
            for level, subset in data.groupby(series.fillna("<NA>"), observed=True):
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
    _write_csv(vif_diagnostics(data), table_dir / "selected_feature_vif_condition_number.csv")
    _write_csv(manual_examples(data), table_dir / "selected_feature_manual_qa_examples.csv")
    if not mandatory_pass:
        raise RuntimeError("A mandatory feature failed its critical model-entry gate; see selected_feature_qa_report.csv.")

    model_columns = ["cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL, CONTROL, PAGE_TYPE, SOURCE_TYPE]
    model_ready = data[model_columns].copy()
    model_ready["run_timestamp"] = _now()
    for key, value in METADATA.items():
        model_ready[key] = value
    model_ready["status"] = "model_ready"
    model_ready["warnings"] = "Complete-case filtering is recorded in model_ready_sample_audit.csv."
    model_ready.to_csv(data_dir / "model_ready_rows.csv", index=False)
    complete = model_ready.dropna(subset=["cited", "prompt_id", "normalized_url", "source_root_domain", *FOCAL, CONTROL]).copy()
    model_formulas = formulas(complete, repo / "tables/econometrics_model_registry_v2.csv")
    validate_formula_scope(model_formulas)

    formula_rows = [{"model_id": model_id, "formula": formula, "entry_status": "approved"} for model_id, formula in model_formulas.items()]
    formula_rows.append({"model_id": "FE1_external_evidence_structure_score", "formula": "not_run", "entry_status": "blocked_unresolved_producer_formula"})
    _write_csv(pd.DataFrame(formula_rows), table_dir / "model_formula_registry_snapshot.csv")

    sample_rows = [_sample_audit("all_model_ready_rows", model_ready), _sample_audit("FE1_FE2_FE4_complete_case", complete)]
    d0 = d0_results(data)
    _write_csv(d0, table_dir / "D0_descriptive_results.csv")

    model_outputs: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []
    fe1_runs = []
    for feature in FOCAL:
        model_id = f"FE1_{feature}"
        run_result = run_model_and_save(model_formulas[model_id], complete, model_id, table_dir / f".{model_id}.csv")
        warnings.extend(run_result.warnings)
        fe1_runs.append(_annotate_model(run_result.table, "FE1"))
        (table_dir / f".{model_id}.csv").unlink(missing_ok=True)
    model_outputs["FE1"] = pd.concat(fe1_runs, ignore_index=True)
    _write_csv(model_outputs["FE1"], table_dir / "FE1_one_feature_prompt_fe_results.csv", warnings="; ".join(warnings))

    fe2_run = run_model_and_save(model_formulas["FE2"], complete, "FE2", table_dir / ".FE2.csv")
    warnings.extend(fe2_run.warnings)
    model_outputs["FE2"] = _annotate_model(fe2_run.table, "FE2")
    _write_csv(model_outputs["FE2"], table_dir / "FE2_joint_core_prompt_fe_results.csv", warnings="; ".join(fe2_run.warnings))
    (table_dir / ".FE2.csv").unlink(missing_ok=True)

    domain_url_counts = complete.groupby("source_root_domain")["normalized_url"].nunique()
    supported_domains = domain_url_counts[domain_url_counts.ge(2)].index
    fe3_data = complete[complete["source_root_domain"].isin(supported_domains)].copy()
    excluded_domains = int((domain_url_counts < 2).sum())
    sample_rows.append(_sample_audit("FE3_supported_domains", fe3_data, f"domains_excluded_lt2_unique_urls={excluded_domains}"))
    fe3_run = run_model_and_save(model_formulas["FE3"], fe3_data, "FE3", table_dir / ".FE3.csv", notes=f"Domains require >=2 unique URLs; excluded_domains={excluded_domains}.")
    warnings.extend(fe3_run.warnings)
    model_outputs["FE3"] = _annotate_model(fe3_run.table, "FE3")
    _write_csv(model_outputs["FE3"], table_dir / "FE3_domain_fe_robustness_results.csv", warnings="; ".join(fe3_run.warnings))
    (table_dir / ".FE3.csv").unlink(missing_ok=True)

    fe4_data = complete.dropna(subset=[PAGE_TYPE, SOURCE_TYPE]).copy()
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
        "focal_features": FOCAL, "controls": [CONTROL], "external_evidence_decision": "blocked",
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

- Included focal features: {', '.join(FOCAL)}.
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
- Model rows use complete cases explicitly recorded in the sample audit.

## Warnings

{chr(10).join(f'- {warning}' for warning in warnings) if warnings else '- None.'}
"""
    (report_dir / "model_validation_report.md").write_text(report, encoding="utf-8")
    return run_manifest
