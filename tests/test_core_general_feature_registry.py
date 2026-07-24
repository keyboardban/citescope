from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.core_general_feature_registry import (
    ALLOWED_STATUSES,
    REGISTRY_COLUMNS,
    build_core_general_feature_registry,
    validate_core_general_feature_registry,
    write_core_general_feature_registry,
)


def test_core_general_registry_has_required_schema_and_layers():
    registry = build_core_general_feature_registry()
    assert tuple(registry.columns) == REGISTRY_COLUMNS
    assert registry["feature_name"].is_unique
    assert set(registry["feature_status"]).issubset(ALLOWED_STATUSES)
    assert {"core_general", "commerce_general", "vertical_specific", "excluded"}.issubset(
        set(registry["feature_layer"])
    )


def test_registry_splits_price_and_pauses_vertical_features():
    registry = build_core_general_feature_registry().set_index("feature_name")
    assert registry.loc["price_detail_score", "feature_layer"] == "commerce_general"
    assert (
        registry.loc["vertical_specific_unit_detail_score", "feature_status"]
        == "pause_vertical_specific"
    )
    assert registry.loc["location_transit_specificity_score", "feature_status"] == "pause_vertical_specific"
    assert registry.loc["amenity_project_detail_score", "feature_status"] == "pause_vertical_specific"


def test_registry_keeps_leakage_features_excluded():
    registry = build_core_general_feature_registry()
    excluded = registry[registry["feature_status"].eq("exclude_leakage")]
    assert len(excluded) >= 3
    assert excluded["leakage_status"].str.contains("forbidden").all()
    validate_core_general_feature_registry(registry)


def test_registry_writer_round_trips(tmp_path: Path):
    path = write_core_general_feature_registry(tmp_path / "registry.csv")
    written = pd.read_csv(path)
    assert len(written) == len(build_core_general_feature_registry())
    assert tuple(written.columns) == REGISTRY_COLUMNS
