from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.econometrics_eda_v2 import econometrics_frontend as frontend
from src.econometrics_eda_v2 import model_comparison
from ui.views import econometrics_frontend as frontend_view


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _empty_contract(tmp_path):
    artifacts = {}
    for filename, columns in frontend.REQUIRED_SCHEMAS.items():
        path = tmp_path / filename
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        artifacts[filename] = {"sha256": _digest(path), "rows": 0, "columns": list(columns)}
    overview = tmp_path / "econometrics_overview_summary.json"
    overview.write_text(json.dumps({"contract_version": frontend.CONTRACT_VERSION}), "utf-8")
    artifacts[overview.name] = {"sha256": _digest(overview), "rows": 1, "columns": ["contract_version"]}
    comparison_artifacts = {}
    for filename, columns in model_comparison.COMPARISON_SCHEMAS.items():
        path = tmp_path / filename
        pd.DataFrame(columns=columns).to_parquet(path, index=False)
        comparison_artifacts[filename] = {"sha256": _digest(path), "rows": 0, "columns": list(columns)}
    thresholds = {
        "schema_version": "test", "minimum_baseline_magnitude_pp": 1.0,
        "stable_magnitude_absolute_threshold_pp": 1.0, "attenuation_absolute_threshold_pp": 2.0,
        "attenuation_relative_threshold": .4, "amplification_absolute_threshold_pp": 2.0,
        "amplification_relative_threshold": .4, "sign_flip_minimum_magnitude_pp": 1.0,
        "large_sample_change_percent": .1, "large_ci_width_change_percent": .25,
        "estimate_equality_tolerance_pp": 1e-6, "low_support_rows": 100,
        "low_support_prompts": 30, "low_support_urls": 50, "low_support_domains": 20,
        "covariance_standard_error_ratio_threshold": 1.25,
        "preferred_covariance_order": ["HC3"],
    }
    threshold_path = tmp_path / "model_comparison_thresholds.yaml"
    threshold_path.write_text(yaml.safe_dump(thresholds), "utf-8")
    comparison_manifest = {
        "contract_version": model_comparison.COMPARISON_CONTRACT_VERSION,
        "generated_at": "2026-07-20T00:00:00+00:00",
        "artifacts": comparison_artifacts,
        "thresholds": {"path": threshold_path.name, "sha256": _digest(threshold_path)},
    }
    comparison_manifest_path = tmp_path / "econometrics_model_comparison_manifest.json"
    comparison_manifest_path.write_text(json.dumps(comparison_manifest), "utf-8")
    manifest = {
        "contract_version": frontend.CONTRACT_VERSION,
        "generated_at": "2026-07-20T00:00:00+00:00",
        "artifacts": artifacts,
    }
    (tmp_path / "econometrics_frontend_manifest.json").write_text(json.dumps(manifest), "utf-8")


def test_frontend_contract_accepts_versioned_empty_optional_tables(tmp_path):
    _empty_contract(tmp_path)

    manifest = frontend.validate_frontend_artifacts(tmp_path)

    assert manifest["contract_version"] == frontend.CONTRACT_VERSION


def test_frontend_contract_rejects_tampered_artifact(tmp_path):
    _empty_contract(tmp_path)
    path = tmp_path / "feature_model_estimates.csv"
    path.write_text(path.read_text("utf-8") + "tampered", "utf-8")

    with pytest.raises(ValueError, match="does not match the manifest"):
        frontend.validate_frontend_artifacts(tmp_path)


def test_term_mapping_excludes_fixed_effects_and_maps_supported_terms():
    assert frontend._feature_from_term("C(prompt_id)[T.p2]") is None
    assert frontend._feature_from_term("has_table") == "has_table"
    assert frontend._feature_from_term(
        "C(heading_count_group, Treatment(reference='0-1'))[T.2-6]"
    ) == "heading_count_group"
    assert frontend._feature_from_term(
        "prompt_page_relevance_score_winsorized_p99"
    ) == "prompt_page_relevance_score"


def test_binary_rate_rows_accept_boolean_strings():
    rows = pd.DataFrame(
        {
            "has_table": ["true"] * 20 + ["false"] * 20,
            "cited": [1] * 10 + [0] * 10 + [1] * 5 + [0] * 15,
            "prompt_id": [f"p{i % 10}" for i in range(40)],
            "normalized_url": [f"https://example.com/{i}" for i in range(40)],
            "source_root_domain": ["example.com"] * 40,
        }
    )
    spec = frontend.SPEC_BY_NAME["has_table"]

    rates = pd.DataFrame(frontend._rate_rows(rows, spec))

    assert set(rates["feature_level"]) == {"absent", "present"}
    assert rates["n_rows"].sum() == 40


def test_negligible_sign_change_is_not_classified_as_sign_flip():
    thresholds = model_comparison.load_thresholds(
        Path(__file__).resolve().parents[1] / "config/model_comparison_thresholds.yaml"
    )
    base = pd.Series({"estimate_pp": .2, "ci_width_pp": 3.0, "ci_lower_pp": -1.3, "ci_upper_pp": 1.7, "n_rows": 1000})
    comparison = pd.Series({"estimate_pp": -.3, "ci_width_pp": 3.2, "ci_lower_pp": -1.9, "ci_upper_pp": 1.3, "n_rows": 1000})

    labels, point_status, _ = model_comparison._labels(base, comparison, "one_feature_to_joint", thresholds)

    assert "sign_flip" not in labels
    assert "stable_direction" not in labels
    assert "direction_change_below_threshold" in labels
    assert point_status == "stable_point_estimate"


def test_relative_change_is_suppressed_for_near_zero_baseline():
    thresholds = model_comparison.load_thresholds(
        Path(__file__).resolve().parents[1] / "config/model_comparison_thresholds.yaml"
    )
    common = {
        "feature_name": "has_table", "feature_label": "Detected table", "term_label": "present versus absent",
        "contrast_key": "present_vs_absent", "model_role": "headline", "is_preferred_covariance": True,
        "model_status": "available", "focal_feature_definition": "has_table", "interpretation_unit": "present versus absent",
        "n_rows": 1000, "n_prompts": 100, "n_urls": 500, "n_domains": 80, "prompt_clusters": 100,
        "url_clusters": 500, "fixed_effects": "prompt_id", "functional_form": "LPM", "ci_width_pp": 2.0,
        "standard_error_pp": .5, "ci_lower_pp": -.8, "ci_upper_pp": 1.2, "se_method": "HC3",
        "dataset_version": "test", "model_version": "test",
    }
    estimates = pd.DataFrame([
        {**common, "model_id": "G1", "source_model_id": "M1", "estimate_pp": .2, "controls": "none"},
        {**common, "model_id": "G2", "source_model_id": "M2", "estimate_pp": -.3, "controls": "joint"},
    ])

    comparisons = model_comparison.build_model_comparisons(estimates, thresholds, "2026-07-20T00:00:00+00:00")

    row = comparisons.iloc[0]
    assert row["relative_change_status"] == "relative_change_not_stable"
    assert pd.isna(row["relative_magnitude_change"])
    assert not row["sign_changed"]


def test_selected_term_narrative_does_not_mix_categorical_contrasts():
    estimates = pd.DataFrame([
        {"model_id": "G0", "term_label": "medium", "estimate_pp": 3.1, "is_preferred_covariance": True, "model_status": "available"},
        {"model_id": "G0", "term_label": "weak", "estimate_pp": 9.6, "is_preferred_covariance": True, "model_status": "available"},
        {"model_id": "G1", "term_label": "medium", "estimate_pp": 5.6, "is_preferred_covariance": True, "model_status": "available"},
        {"model_id": "G1", "term_label": "weak", "estimate_pp": 14.4, "is_preferred_covariance": True, "model_status": "available"},
        {"model_id": "G2", "term_label": "medium", "estimate_pp": .5, "is_preferred_covariance": True, "model_status": "available"},
        {"model_id": "G2", "term_label": "weak", "estimate_pp": .5, "is_preferred_covariance": True, "model_status": "available"},
    ])
    comparisons = pd.DataFrame([
        {"baseline_model_id": "G1", "comparison_model_id": "G2", "estimate_change_pp": -5.1, "comparability_status": "partially_comparable"},
    ])

    narrative = frontend_view._selected_term_narrative(estimates, comparisons, "medium")

    assert "+3.1 pp" in narrative
    assert "+5.6 pp" in narrative
    assert "+0.5 pp" in narrative
    assert "-5.1 pp" in narrative
    assert "+9.6 pp" not in narrative
    assert "+14.4 pp" not in narrative
