from pathlib import Path

import pandas as pd

from src import econometrics_qa as qa
from src.econometrics_eda_v2.core_general_feature_registry import (
    CANONICAL_REGISTRY_PATH,
    build_core_general_feature_registry,
)


REQUIRED_TABLE_FEATURES = {
    "inferred_text_table_detected",
    "table_detection_source",
    "table_verification_status",
    "table_type_primary",
    "table_has_numeric_facts",
    "table_has_measurements",
    "table_has_specifications",
    "table_has_comparison",
    "has_any_verified_table",
    "has_any_data_table",
    "dominant_table_type",
    "table_has_price",
    "real_estate_table_contains_unit_size",
    "has_any_table_with_numeric_facts",
    "numeric_fact_table_count",
    "has_any_table_with_measurements",
    "measurement_table_count",
    "has_any_table_with_specifications",
    "specification_table_count",
    "has_any_comparison_table",
    "has_any_table_with_textual_facts",
    "textual_fact_table_count",
}


def _table_rows(registry: pd.DataFrame) -> pd.DataFrame:
    return registry[registry["feature_group"].str.startswith("table_")].copy()


def test_table_registry_has_expanded_schema_and_required_features():
    registry = build_core_general_feature_registry()
    assert REQUIRED_TABLE_FEATURES.issubset(set(registry["feature_name"]))
    assert {
        "canonical_column_name",
        "legacy_aliases",
        "registry_record_type",
        "replacement_feature_name",
        "feature_granularity",
        "page_aggregation_rule",
        "taxonomy_or_rule_version",
        "qa_status",
        "approved_for_model_v1",
        "minimum_qa_gate",
        "model_entry_blocker",
    }.issubset(registry.columns)


def test_legacy_table_aliases_remain_mapped():
    registry = build_core_general_feature_registry().set_index("feature_name")
    expected = {
        "has_any_verified_table": "has_verified_html_table",
        "has_table": "has_verified_html_table",
        "table_row_count": "table_row_count_total",
        "table_column_count": "table_column_count_max",
        "table_contains_numeric_facts": "table_has_numeric_facts",
        "table_contains_comparison": "table_has_comparison",
        "table_is_layout_or_navigation": "table_has_layout_or_navigation_structure",
    }
    for legacy, canonical in expected.items():
        assert registry.loc[legacy, "registry_record_type"] == "deprecated_alias"
        assert registry.loc[legacy, "replacement_feature_name"] == canonical
        assert registry.loc[legacy, "approved_for_model_v1"].casefold() == "false"


def test_canonical_columns_are_unique_and_table_aggregation_is_explicit():
    registry = build_core_general_feature_registry()
    canonical = registry[registry["registry_record_type"].eq("canonical")]
    assert canonical["canonical_column_name"].is_unique
    table_level = canonical[canonical["feature_granularity"].eq("table_level")]
    assert table_level["page_aggregation_rule"].str.strip().ne("").all()
    assert not table_level["page_aggregation_rule"].eq("not_applicable").any()


def test_single_canonical_verified_presence_feature():
    registry = build_core_general_feature_registry().set_index("feature_name")
    assert registry.loc["has_verified_html_table", "registry_record_type"] == "canonical"
    assert registry.loc["has_verified_html_table", "feature_granularity"] == "page_level"
    assert registry.loc["has_any_verified_table", "replacement_feature_name"] == "has_verified_html_table"
    assert registry.loc["has_table", "replacement_feature_name"] == "has_verified_html_table"


def test_table_layer_assignment_and_model_approval_are_separate():
    table = _table_rows(build_core_general_feature_registry()).set_index("feature_name")
    for feature in (
        "table_has_price",
        "table_has_pricing_plan",
        "pricing_or_plan_table_count",
        "table_has_availability",
        "table_has_rating_or_review",
    ):
        assert table.loc[feature, "feature_layer"] == "commerce_general"
    vertical = table[table.index.str.startswith("real_estate_table_")]
    assert vertical["feature_layer"].eq("vertical_specific").all()
    assert vertical["feature_status"].eq("pause_vertical_specific").all()
    assert vertical["recommended_model_role"].eq("vertical_extension_only").all()
    assert table["approved_for_model_v1"].str.casefold().eq("false").all()


def test_core_table_definitions_exclude_vertical_vocabulary_and_leakage():
    table = _table_rows(build_core_general_feature_registry())
    core = table[table["feature_layer"].eq("core_general")]
    definitions = core["definition"].str.casefold()
    forbidden_vertical = (
        "scope",
        "condo",
        "neighborhood",
        "transit",
        "amenity",
        "bedroom",
        "floor plan",
        "sqm",
        "sq.m",
        "square metre",
        "ตร.ม",
    )
    for token in forbidden_vertical:
        assert not definitions.str.contains(token, regex=False).any()
    inspected = table[["formula", "required_input", "source_provenance"]].astype(str).agg(" ".join, axis=1).str.casefold()
    for token in ("cited_label", "citation rate", "answer text", "answer_similarity", "source_position", "observed_rank", "domain_citation_rate", "p-value", "coefficient"):
        assert not inspected.str.contains(token, regex=False).any()


def test_verified_absent_and_unmeasured_vocabularies_are_distinct():
    package = qa.default_package_dir()
    allowed = pd.read_csv(package / "tables/core_general_table_taxonomy_allowed_values.csv")
    status = set(
        allowed.loc[
            allowed["field_name"].eq("table_verification_status"), "allowed_value"
        ]
    )
    assert status == {
        "verified_html",
        "inferred_markdown",
        "inferred_text",
        "verified_absent",
        "unmeasured",
    }
    assert {
        "field_name",
        "feature_granularity",
        "allowed_value",
        "definition",
        "is_reference_candidate",
        "is_missing_state",
        "is_unknown_state",
        "is_no_table_state",
        "is_mixed_state",
        "taxonomy_version",
        "notes",
    } == set(allowed.columns)
    dominant = set(
        allowed.loc[allowed["field_name"].eq("dominant_table_type"), "allowed_value"]
    )
    assert {"no_table", "mixed_or_unknown", "__NA__"}.issubset(dominant)


def test_commerce_naming_cleanup_preserves_old_aliases():
    registry = build_core_general_feature_registry().set_index("feature_name")
    assert registry.loc["transactional_action_signal", "feature_layer"] == "commerce_general"
    assert registry.loc["contact_access_signal", "feature_layer"] == "commerce_general"
    assert registry.loc["commercial_offer_comparison_signal", "feature_layer"] == "commerce_general"
    assert registry.loc["purchase_or_contact_signal", "registry_record_type"] == "deprecated_alias"
    assert registry.loc["product_comparison_signal", "replacement_feature_name"] == "commercial_offer_comparison_signal"


def test_schema_validation_artifact_passes():
    validation = pd.read_csv(
        qa.default_package_dir() / "tables/core_general_table_schema_validation.csv"
    )
    assert validation["status"].eq("pass").all()


def test_generated_registry_matches_canonical_source():
    canonical = pd.read_csv(CANONICAL_REGISTRY_PATH, dtype=str, keep_default_na=False)
    generated = pd.read_csv(
        qa.default_package_dir() / "tables/core_general_content_feature_dictionary.csv",
        dtype=str,
        keep_default_na=False,
    )
    pd.testing.assert_frame_equal(canonical, generated)
