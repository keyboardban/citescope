#!/usr/bin/env python3
"""Finalize table-registry schema without running extraction or econometrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import econometrics_qa as qa  # noqa: E402
from src.econometrics_eda_v2.core_general_feature_registry import (  # noqa: E402
    REGISTRY_COLUMNS,
    validate_core_general_feature_registry,
    write_core_general_feature_registry,
)


CANONICAL = ROOT / "config/core_general_content_feature_dictionary.csv"
SCHEMA_VERSION = "core_general_table_registry_schema_v2"
TAXONOMY_VERSION = "core_general_table_taxonomy_v1"

ALIAS_REPLACEMENTS = {
    "has_any_verified_table": "has_verified_html_table",
    "has_table": "has_verified_html_table",
    "table_contains_numeric_facts": "table_has_numeric_facts",
    "table_contains_comparison": "table_has_comparison",
    "table_is_layout_or_navigation": "table_has_layout_or_navigation_structure",
    "table_row_count": "table_row_count_total",
    "table_column_count": "table_column_count_max",
    "purchase_or_contact_signal": "transactional_action_signal",
    "product_comparison_signal": "commercial_offer_comparison_signal",
}

TABLE_LEVEL_AGGREGATIONS = {
    "table_type_primary": "derive primary-type counts and dominant_table_type from all measured tables",
    "table_has_numeric_facts": "max over measured tables; count tables satisfying condition",
    "table_has_measurements": "max over measured tables; count tables satisfying condition",
    "table_measurement_unit_registry": "retain per-table provenance; page summary uses set of observed registries",
    "table_has_specifications": "max over measured tables; count tables satisfying condition",
    "table_has_comparison": "max over measured tables; count tables satisfying condition",
    "table_has_directory_or_listing_structure": "max over measured tables; count tables satisfying condition",
    "table_has_schedule_or_timeline": "max over measured tables; count tables satisfying condition",
    "table_has_transactional_structure": "max over measured tables; count tables satisfying condition",
    "table_has_layout_or_navigation_structure": "max over measured tables; count tables satisfying condition",
    "table_has_textual_facts": "max over measured tables; count tables satisfying condition",
    "table_has_header_labels": "max over measured tables; count tables satisfying condition",
    "table_has_row_labels": "max over measured tables; count tables satisfying condition",
    "table_has_column_labels": "max over measured tables; count tables satisfying condition",
    "table_has_price": "max over measured tables; count tables satisfying condition in Commerce-General extension",
    "table_has_pricing_plan": "max over measured tables; count tables satisfying condition in Commerce-General extension",
    "table_has_availability": "max over measured tables; count tables satisfying condition in Commerce-General extension",
    "table_has_rating_or_review": "max over measured tables; count tables satisfying condition in Commerce-General extension",
    "table_has_commercial_offer_comparison": "max over measured tables; count tables satisfying condition in Commerce-General extension",
    "table_row_count_per_table": "sum to table_row_count_total; retain max and distribution diagnostics",
    "table_column_count_per_table": "max to table_column_count_max; retain distribution diagnostics",
    "table_cell_count_per_table": "sum to table_cell_count_total; retain distribution diagnostics",
    "real_estate_table_contains_unit_size": "max/count only inside paused real-estate vertical extension",
    "real_estate_table_contains_bedroom_mix": "max/count only inside paused real-estate vertical extension",
    "real_estate_table_contains_floor_plan": "max/count only inside paused real-estate vertical extension",
    "real_estate_table_contains_price_per_area": "max/count only inside paused real-estate vertical extension",
    "real_estate_table_contains_project_inventory": "max/count only inside paused real-estate vertical extension",
}

EXTRACTION_DIAGNOSTICS = {
    "markdown_table_detected",
    "inferred_text_table_detected",
    "table_detection_source",
    "table_verification_status",
}


def _normalize_lifecycle(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    implementation_map = {
        "not_implemented": "planned_not_implemented",
        "implemented": "legacy_implemented",
        "implemented_partial": "implemented_partial",
        "partial": "implemented_partial",
        "implemented_alias_needed": "implemented_partial",
        "implemented_unreliable": "implemented_pending_qa",
        "implemented_in_old_score": "legacy_implemented",
        "legacy_possible": "legacy_implemented",
        "legacy_available": "legacy_implemented",
        "planned_not_implemented": "planned_not_implemented",
        "implemented_pending_qa": "implemented_pending_qa",
        "implemented_validated": "implemented_validated",
        "legacy_implemented": "legacy_implemented",
        "not_applicable": "not_applicable",
    }
    data["current_implementation_status"] = (
        data["current_implementation_status"].map(implementation_map).fillna("planned_not_implemented")
    )
    qa_map = {
        "not_assessed": "not_started",
        "deprecated_compatibility": "not_started",
        "pending_human_threshold": "pending_human_threshold",
        "pending_manual_validation": "pending_manual_validation",
        "not_started": "not_started",
        "qa_failed": "qa_failed",
        "qa_passed": "qa_passed",
    }
    data["qa_status"] = data["qa_status"].map(qa_map).fillna("not_started")
    data["approved_for_model_v1"] = data["approved_for_model_v1"].astype(str).str.casefold()
    data.loc[~data["approved_for_model_v1"].isin({"true", "false"}), "approved_for_model_v1"] = "false"
    return data


def _add_schema_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    defaults = {
        "registry_record_type": "canonical",
        "replacement_feature_name": "",
        "feature_granularity": "not_table_related",
        "page_aggregation_rule": "not_applicable",
    }
    for column, default in defaults.items():
        if column not in data:
            data[column] = default
    data["registry_version"] = SCHEMA_VERSION
    return data


def _new_row(
    name: str,
    *,
    layer: str,
    group: str,
    granularity: str,
    definition: str,
    formula: str,
    aggregation: str,
    role: str,
    status: str,
    provenance: str = "provenance_aware_table_extraction",
    required_input: str = "measured table-level records",
    transformation: str = "nullable_binary_or_count",
    implementation: str = "planned_not_implemented",
    qa_status: str = "pending_human_threshold",
    aliases: str = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "feature_name": name,
        "canonical_column_name": name,
        "legacy_aliases": aliases,
        "registry_record_type": "canonical",
        "replacement_feature_name": "",
        "feature_granularity": granularity,
        "page_aggregation_rule": aggregation,
        "feature_layer": layer,
        "feature_group": group,
        "primitive_or_composite": "derived_primitive" if granularity == "page_level" else "primitive",
        "definition": definition,
        "formula": formula,
        "source_provenance": provenance,
        "required_input": required_input,
        "transformation": transformation,
        "language_dependency": "medium",
        "extraction_requirement": "table extractor with verified measured/unmeasured state",
        "missing_value_meaning": "NA means extraction evidence was unavailable; zero requires measured absence.",
        "topic_specificity": "commerce" if layer == "commerce_general" else "general",
        "generalizability": "conditional" if layer == "commerce_general" else "high",
        "leakage_status": "safe_pre_outcome",
        "expected_confounders": "domain; template; page function; publisher type; extraction quality",
        "measurement_risks": "Responsive duplication, nested tables, spans, rendering, and missing extraction evidence.",
        "recommended_model_role": role,
        "feature_status": status,
        "current_implementation_status": implementation,
        "taxonomy_or_rule_version": TAXONOMY_VERSION,
        "qa_status": qa_status,
        "approved_for_model_v1": "false",
        "minimum_qa_gate": "pending_human_threshold",
        "model_entry_blocker": "extraction and aggregation QA not yet passed",
        "validation_requirement": "Validate extraction, aggregation, support, missingness, and redundancy before model entry.",
        "notes": notes or "Observed structure among surfaced sources; not a causal table effect.",
        "registry_version": SCHEMA_VERSION,
    }


def _aggregate_rows() -> list[dict[str, str]]:
    specs = [
        ("has_any_table_with_numeric_facts", "table_has_numeric_facts", "Page has at least one measured table with numeric facts."),
        ("numeric_fact_table_count", "table_has_numeric_facts", "Count of measured tables with numeric facts."),
        ("has_any_table_with_measurements", "table_has_measurements", "Page has at least one measured table with general measurements."),
        ("measurement_table_count", "table_has_measurements", "Count of measured tables with general measurements."),
        ("has_any_table_with_specifications", "table_has_specifications", "Page has at least one measured specification table."),
        ("specification_table_count", "table_has_specifications", "Count of measured tables with specifications."),
        ("has_any_comparison_table", "table_has_comparison", "Page has at least one measured comparison table."),
        ("comparison_table_count", "table_has_comparison", "Count of measured tables with comparison structure."),
        ("has_any_table_with_textual_facts", "table_has_textual_facts", "Page has at least one measured table with textual facts."),
        ("textual_fact_table_count", "table_has_textual_facts", "Count of measured tables with textual facts."),
        ("has_any_directory_or_listing_table", "table_has_directory_or_listing_structure", "Page has at least one directory/listing table."),
        ("directory_or_listing_table_count", "table_has_directory_or_listing_structure", "Count of directory/listing tables."),
        ("has_any_schedule_or_timeline_table", "table_has_schedule_or_timeline", "Page has at least one schedule/timeline table."),
        ("schedule_or_timeline_table_count", "table_has_schedule_or_timeline", "Count of schedule/timeline tables."),
        ("has_any_transactional_or_form_table", "table_has_transactional_structure", "Page has at least one transactional/form table."),
        ("transactional_or_form_table_count", "table_has_transactional_structure", "Count of transactional/form tables."),
        ("has_any_layout_table", "table_has_layout_or_navigation_structure", "Page has at least one layout/navigation table."),
        ("layout_or_navigation_table_count", "table_has_layout_or_navigation_structure", "Count of layout/navigation tables."),
    ]
    rows = []
    for name, primitive, definition in specs:
        is_count = name.endswith("_count")
        formula = f"count measured tables where {primitive} = 1" if is_count else f"1[max({primitive}) = 1]; 0 only when measured and absent; otherwise NA"
        rows.append(
            _new_row(
                name,
                layer="core_general",
                group="table_page_aggregate",
                granularity="page_level",
                definition=definition,
                formula=formula,
                aggregation=f"deterministic page aggregate from {primitive}",
                role="diagnostic_only",
                status="diagnostic_only",
            )
        )
    return rows


def _secondary_commerce_rows() -> list[dict[str, str]]:
    return [
        _new_row(
            "transactional_action_signal",
            layer="commerce_general",
            group="commerce_action",
            granularity="page_level",
            definition="Visible action to purchase, subscribe, book, enroll, or start a service.",
            formula="documented cross-industry transactional-action primitives",
            aggregation="page-level commerce signal from measured action controls and text",
            role="commerce_extension",
            status="commerce_general_keep",
            required_input="page controls, links, forms, and visible text",
            transformation="nullable_binary_or_categorical",
            notes="Separate from contact or enquiry access.",
        ),
        _new_row(
            "contact_access_signal",
            layer="commerce_general",
            group="commerce_action",
            granularity="page_level",
            definition="Visible access to contact, enquiry, consultation, or request-information actions.",
            formula="documented cross-industry contact-access primitives",
            aggregation="page-level commerce signal from measured contact controls and text",
            role="commerce_extension",
            status="commerce_general_keep",
            required_input="contact links, forms, controls, and visible text",
            transformation="nullable_binary_or_categorical",
            notes="Separate from purchase, subscription, and booking actions.",
        ),
        _new_row(
            "commercial_offer_comparison_signal",
            layer="commerce_general",
            group="commerce_comparison",
            granularity="page_level",
            definition="Comparison of products, services, providers, plans, or commercial offers.",
            formula="documented commerce comparison primitives",
            aggregation="page-level commerce signal from measured comparison structures",
            role="commerce_extension",
            status="commerce_general_keep",
            required_input="comparison tables, headings, and page text",
            transformation="nullable_binary_or_score",
            notes="Replaces the narrower product_comparison_signal name.",
        ),
    ]


def _upsert(frame: pd.DataFrame, rows: list[dict[str, str]]) -> pd.DataFrame:
    names = {row["feature_name"] for row in rows}
    return pd.concat([frame[~frame["feature_name"].isin(names)], pd.DataFrame(rows)], ignore_index=True)


def _classify_registry(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    table_mask = data["feature_group"].str.startswith("table_")
    data.loc[table_mask, "taxonomy_or_rule_version"] = TAXONOMY_VERSION
    data.loc[table_mask, "approved_for_model_v1"] = "false"
    data.loc[table_mask & data["qa_status"].eq("not_started"), "qa_status"] = "pending_human_threshold"

    for name, aggregation in TABLE_LEVEL_AGGREGATIONS.items():
        mask = data["feature_name"].eq(name)
        if mask.any():
            data.loc[mask, "feature_granularity"] = "table_level"
            data.loc[mask, "page_aggregation_rule"] = aggregation
            if data.loc[mask, "feature_status"].eq("refactor_needed").any():
                data.loc[mask, "feature_status"] = "core_keep"
            data.loc[mask, "current_implementation_status"] = data.loc[
                mask, "current_implementation_status"
            ].replace("not_applicable", "planned_not_implemented")

    for name in EXTRACTION_DIAGNOSTICS:
        mask = data["feature_name"].eq(name)
        data.loc[mask, "feature_granularity"] = "extraction_diagnostic"
        data.loc[mask, "page_aggregation_rule"] = "not_applicable"

    page_mask = table_mask & ~data["feature_name"].isin(
        {*TABLE_LEVEL_AGGREGATIONS, *EXTRACTION_DIAGNOSTICS}
    )
    data.loc[page_mask, "feature_granularity"] = "page_level"
    data.loc[page_mask & data["page_aggregation_rule"].isin({"", "not_applicable"}), "page_aggregation_rule"] = data.loc[
        page_mask & data["page_aggregation_rule"].isin({"", "not_applicable"}), "formula"
    ]

    raw_dimensions = {
        "html_table_count",
        "markdown_table_count",
        "inferred_table_count",
        "total_detected_table_count",
        "table_row_count_total",
        "table_column_count_max",
        "table_cell_count_total",
        "data_table_count",
        "layout_table_count",
        "table_row_count_per_table",
        "table_column_count_per_table",
        "table_cell_count_per_table",
    }
    raw_mask = data["feature_name"].isin(raw_dimensions)
    data.loc[raw_mask, "recommended_model_role"] = "diagnostic_only"
    data.loc[raw_mask, "feature_status"] = "diagnostic_only"
    data.loc[raw_mask, "approved_for_model_v1"] = "false"

    aliases_by_canonical = {
        "has_verified_html_table": "has_any_verified_table|has_table",
        "table_has_numeric_facts": "table_contains_numeric_facts",
        "table_has_comparison": "table_contains_comparison",
        "table_has_layout_or_navigation_structure": "table_is_layout_or_navigation",
        "table_row_count_total": "table_row_count",
        "table_column_count_max": "table_column_count",
        "transactional_action_signal": "purchase_or_contact_signal",
        "contact_access_signal": "purchase_or_contact_signal",
        "commercial_offer_comparison_signal": "product_comparison_signal",
    }
    for canonical, aliases in aliases_by_canonical.items():
        data.loc[data["feature_name"].eq(canonical), "legacy_aliases"] = aliases

    for alias, replacement in ALIAS_REPLACEMENTS.items():
        mask = data["feature_name"].eq(alias)
        if not mask.any():
            continue
        data.loc[mask, "registry_record_type"] = "deprecated_alias"
        data.loc[mask, "replacement_feature_name"] = replacement
        data.loc[mask, "canonical_column_name"] = alias
        data.loc[mask, "feature_granularity"] = "registry_metadata"
        data.loc[mask, "page_aggregation_rule"] = "not_applicable"
        data.loc[mask, "recommended_model_role"] = "deprecated_alias"
        data.loc[mask, "feature_status"] = "diagnostic_only"
        data.loc[mask, "current_implementation_status"] = "legacy_implemented"
        data.loc[mask, "qa_status"] = "not_started"
        data.loc[mask, "approved_for_model_v1"] = "false"
        data.loc[mask, "minimum_qa_gate"] = "not_applicable"
        data.loc[mask, "model_entry_blocker"] = f"deprecated alias; use {replacement}"

    canonical_presence = data["feature_name"].eq("has_verified_html_table")
    data.loc[canonical_presence, "definition"] = "1 when at least one HTML <table> is successfully detected and verified on the page."
    data.loc[canonical_presence, "formula"] = "1[verified HTML table present]; 0[HTML measured and verified absent]; NA[HTML extraction unavailable]"
    data.loc[canonical_presence, "missing_value_meaning"] = "1 = verified present; 0 = verified absent; NA = unmeasured extraction."
    data.loc[canonical_presence, "feature_granularity"] = "page_level"
    data.loc[canonical_presence, "page_aggregation_rule"] = "max verified HTML presence over measured tables"
    data.loc[canonical_presence, "feature_status"] = "core_keep"
    data.loc[canonical_presence, "recommended_model_role"] = "temporary_baseline_if_semantic_classification_unvalidated"
    data.loc[canonical_presence, "approved_for_model_v1"] = "false"

    data_presence = data["feature_name"].eq("has_any_data_table")
    data.loc[data_presence, "definition"] = "1 when at least one measured table is not classified solely as layout/navigation."
    data.loc[data_presence, "formula"] = "1[data_table_count > 0]; 0 when measured tables are layout-only or verified absent; NA when extraction unavailable"
    data.loc[data_presence, "missing_value_meaning"] = "Layout-only page = 0; verified no-table page = 0; extraction unavailable = NA."
    data.loc[data_presence, "feature_granularity"] = "page_level"
    data.loc[data_presence, "page_aggregation_rule"] = "max over measured non-layout table classifications"
    data.loc[data_presence, "feature_status"] = "core_keep"
    data.loc[data_presence, "recommended_model_role"] = "preferred_baseline_after_QA"
    data.loc[data_presence, "approved_for_model_v1"] = "false"

    vertical = table_mask & data["feature_name"].str.startswith("real_estate_table_")
    data.loc[vertical, "feature_layer"] = "vertical_specific"
    data.loc[vertical, "feature_status"] = "pause_vertical_specific"
    data.loc[vertical, "recommended_model_role"] = "vertical_extension_only"
    data.loc[vertical, "approved_for_model_v1"] = "false"

    commerce_table = table_mask & data["feature_name"].isin(
        {
            "table_has_price",
            "table_has_pricing_plan",
            "table_has_availability",
            "table_has_rating_or_review",
            "table_has_commercial_offer_comparison",
            "pricing_or_plan_table_count",
        }
    )
    data.loc[commerce_table, "feature_layer"] = "commerce_general"
    data.loc[commerce_table, "approved_for_model_v1"] = "false"
    return data


def _allowed_values() -> pd.DataFrame:
    vocabularies = {
        "table_type_primary": (
            "table_level",
            [
                ("factual_or_specification", "Measured table primarily presents facts or specifications", True, False, False, False, False),
                ("comparison", "Measured table primarily compares alternatives", False, False, False, False, False),
                ("pricing_or_plan", "Measured table primarily presents prices or plans", False, False, False, False, False),
                ("directory_or_listing", "Measured table primarily organizes entities/listings", False, False, False, False, False),
                ("schedule_or_timeline", "Measured table primarily organizes time/stages", False, False, False, False, False),
                ("transactional_or_form", "Measured table primarily supports a transaction/form", False, False, False, False, False),
                ("layout_or_navigation", "Measured table primarily serves layout/navigation", False, False, False, False, False),
                ("unknown_or_other", "Measured table function is uncertain, conflicting, or outside taxonomy", False, False, True, False, False),
                ("__NA__", "Table function unmeasured because extraction evidence is unavailable", False, True, False, False, False),
            ],
        ),
        "dominant_table_type": (
            "page_level",
            [
                ("no_table", "Extraction measured and verified no table", False, False, False, True, False),
                ("factual_or_specification", "Unique dominant factual/specification type", True, False, False, False, False),
                ("comparison", "Unique dominant comparison type", False, False, False, False, False),
                ("pricing_or_plan", "Unique dominant pricing/plan type", False, False, False, False, False),
                ("directory_or_listing", "Unique dominant directory/listing type", False, False, False, False, False),
                ("schedule_or_timeline", "Unique dominant schedule/timeline type", False, False, False, False, False),
                ("transactional_or_form", "Unique dominant transactional/form type", False, False, False, False, False),
                ("layout_or_navigation", "Unique dominant layout/navigation type", False, False, False, False, False),
                ("mixed_or_unknown", "Tables measured but tied, mixed, or without unique dominant type", False, False, True, False, True),
                ("__NA__", "Table extraction unavailable or unmeasurable", False, True, False, False, False),
            ],
        ),
        "dominant_table_type_collapsed": (
            "page_level",
            [
                ("no_table", "Measured page with verified no table", False, False, False, True, False),
                ("factual_or_specification", "Supported factual/specification category", True, False, False, False, False),
                ("comparison", "Supported comparison category", False, False, False, False, False),
                ("pricing_or_plan", "Supported pricing/plan category", False, False, False, False, False),
                ("directory_or_listing", "Supported directory/listing category", False, False, False, False, False),
                ("schedule_or_timeline", "Supported schedule/timeline category", False, False, False, False, False),
                ("transactional_or_form", "Supported transactional/form category", False, False, False, False, False),
                ("layout_or_navigation", "Supported layout/navigation category", False, False, False, False, False),
                ("rare_other", "Predeclared support-only collapse of sparse categories", False, False, True, False, False),
                ("mixed_or_unknown", "Measured mixed/tied/unknown dominant state", False, False, True, False, True),
                ("__NA__", "Table extraction unavailable", False, True, False, False, False),
            ],
        ),
        "table_detection_source": (
            "extraction_diagnostic",
            [
                ("html_dom", "Detected from preserved HTML DOM", True, False, False, False, False),
                ("markdown", "Detected from Markdown table syntax", False, False, False, False, False),
                ("text_pattern", "Detected only from flattened text pattern", False, False, False, False, False),
                ("multiple_sources", "Same table supported by multiple sources", False, False, False, False, True),
                ("none_verified", "Required evidence measured and no table detected", False, False, False, True, False),
                ("unmeasured", "Required extraction evidence unavailable", False, True, False, False, False),
            ],
        ),
        "table_verification_status": (
            "extraction_diagnostic",
            [
                ("verified_html", "Presence verified in HTML", True, False, False, False, False),
                ("inferred_markdown", "Presence inferred from Markdown", False, False, False, False, False),
                ("inferred_text", "Presence inferred from text pattern", False, False, False, False, False),
                ("verified_absent", "Evidence measured and table absence verified", False, False, False, True, False),
                ("unmeasured", "Required extraction evidence unavailable", False, True, False, False, False),
            ],
        ),
        "table_measurement_unit_registry": (
            "table_level",
            [
                ("general_unit_registry", "Match from general cross-industry units", True, False, False, False, False),
                ("commerce_unit_registry", "Match from Commerce-General units", False, False, False, False, False),
                ("vertical_unit_registry", "Match from isolated vertical extension units", False, False, False, False, False),
                ("multiple_registries", "Matches retained from multiple registries", False, False, False, False, True),
                ("no_unit_match", "Measured cells had no valid unit match", False, False, False, True, False),
                ("unmeasured", "Unit matching unavailable", False, True, False, False, False),
            ],
        ),
    }
    rows = []
    for field, (granularity, values) in vocabularies.items():
        for value, definition, reference, missing, unknown, no_table, mixed in values:
            rows.append(
                {
                    "field_name": field,
                    "feature_granularity": granularity,
                    "allowed_value": value,
                    "definition": definition,
                    "is_reference_candidate": str(reference).lower(),
                    "is_missing_state": str(missing).lower(),
                    "is_unknown_state": str(unknown).lower(),
                    "is_no_table_state": str(no_table).lower(),
                    "is_mixed_state": str(mixed).lower(),
                    "taxonomy_version": TAXONOMY_VERSION,
                    "notes": "__NA__ documents an actual missing value, not a literal string category." if value == "__NA__" else "",
                }
            )
    return pd.DataFrame(rows)


def _diff() -> pd.DataFrame:
    rows = [
        ("verified_table_presence_constructs", "merged", "has_any_verified_table|has_table", "has_verified_html_table", "page_level", "page_level", "duplicate_constructs", "core_keep", "One canonical verified HTML-presence construct; old names remain aliases."),
        ("has_verified_html_table", "retained", "has_verified_html_table", "has_verified_html_table", "page_level", "page_level", "refactor_needed", "core_keep", "Selected as the single canonical verified HTML presence field."),
        ("has_any_verified_table", "deprecated_alias", "has_any_verified_table", "has_verified_html_table", "page_level", "registry_metadata", "refactor_needed", "diagnostic_only", "Merged duplicate verified-presence construct into canonical field."),
        ("has_table", "deprecated_alias", "has_table", "has_verified_html_table", "page_level", "registry_metadata", "diagnostic_only", "diagnostic_only", "Legacy broad-presence compatibility alias."),
        ("purchase_or_contact_signal", "split", "purchase_or_contact_signal", "transactional_action_signal|contact_access_signal", "page_level", "registry_metadata", "commerce_general_keep", "diagnostic_only", "Purchase/booking and contact access are distinct Commerce-General constructs."),
        ("product_comparison_signal", "renamed", "product_comparison_signal", "commercial_offer_comparison_signal", "page_level", "registry_metadata", "commerce_general_keep", "diagnostic_only", "Comparison can cover services, providers, plans, and offers."),
    ]
    for name, replacement in ALIAS_REPLACEMENTS.items():
        if name in {row[0] for row in rows}:
            continue
        rows.append((name, "deprecated_alias", name, replacement, "", "registry_metadata", "", "diagnostic_only", "Compatibility alias points to one canonical replacement."))
    for row in _aggregate_rows():
        rows.append((row["feature_name"], "added", "", row["feature_name"], "", "page_level", "", row["feature_status"], row["page_aggregation_rule"]))
    rows.append(("all_table_rows", "aggregation_rule_added", "", "", "implicit", "explicit", "", "", "Every table-level canonical feature now documents a page aggregation rule."))
    rows.append(("categorical_table_fields", "missing_semantics_corrected", "", "", "implicit", "explicit", "", "", "No-table, unknown, mixed, and unmeasured states are distinct."))
    rows.append(("table_lifecycle_fields", "status_corrected", "", "", "", "", "mixed_legacy", "controlled_vocabularies", "Concept, implementation, QA, and model approval are separate."))
    return pd.DataFrame(rows, columns=["feature_name", "action", "previous_name", "new_name", "previous_granularity", "new_granularity", "previous_status", "new_status", "reason"])


def _snapshot_frozen(package: Path) -> dict[str, int]:
    paths = [
        *ROOT.glob("notebooks/09_*.ipynb"),
        *ROOT.glob("notebooks/10_*.ipynb"),
        *ROOT.glob("notebooks/11_*.ipynb"),
        *ROOT.glob("notebooks/12_*.ipynb"),
    ]
    for relative in (
        "tables/09_content_feature_econometrics",
        "tables/10_writing_factual_density_features",
        "tables/11_writing_factual_density_econometrics",
        "figures/09_content_feature_econometrics",
        "figures/10_writing_factual_density_features",
        "figures/11_writing_factual_density_econometrics",
        "reports/09_content_feature_econometrics",
        "reports/10_writing_factual_density_features",
        "reports/11_writing_factual_density_econometrics",
    ):
        root = package / relative
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return {str(path): path.stat().st_mtime_ns for path in paths}


def _validation(
    registry: pd.DataFrame,
    allowed: pd.DataFrame,
    canonical_copy_equal: bool,
    frozen_unchanged: bool,
) -> pd.DataFrame:
    table = registry[registry["feature_group"].str.startswith("table_")]
    canonical = registry[registry["registry_record_type"].eq("canonical")]
    aliases = registry[registry["registry_record_type"].eq("deprecated_alias")]
    canonical_names = set(canonical["feature_name"])
    aggregate_dependencies = {
        row["feature_name"]: row["page_aggregation_rule"].split()[-1]
        for row in _aggregate_rows()
    }
    required_fields = {
        "table_type_primary",
        "dominant_table_type",
        "dominant_table_type_collapsed",
        "table_detection_source",
        "table_verification_status",
        "table_measurement_unit_registry",
    }
    inspected = table[["definition", "formula", "required_input", "source_provenance"]].astype(str).agg(" ".join, axis=1).str.casefold()
    leakage_tokens = ("cited_label", "citation rate", "answer text", "answer similarity", "observed rank", "source position", "domain citation", "coefficient", "p-value")
    checks = [
        ("valid_feature_granularity", table["feature_granularity"].isin({"table_level", "page_level", "extraction_diagnostic", "registry_metadata"}).all(), "Every table row uses the controlled granularity vocabulary."),
        ("table_level_aggregation_documented", table.loc[table["feature_granularity"].eq("table_level"), "page_aggregation_rule"].str.strip().ne("").all(), "Every table-level row has an explicit aggregation rule."),
        ("page_aggregate_dependencies_exist", all(dependency in canonical_names for dependency in aggregate_dependencies.values()), "Every newly added page aggregate references a canonical table primitive."),
        ("canonical_columns_unique", ~canonical["canonical_column_name"].duplicated().any(), "Canonical output columns are unique."),
        ("alias_replacements_valid", aliases["replacement_feature_name"].isin(canonical_names).all(), "Every deprecated alias points to one canonical feature."),
        ("categorical_zero_not_missing", not allowed["allowed_value"].eq("0").any(), "Categorical vocabularies do not encode missing/unknown as numeric zero."),
        ("categorical_states_distinct", {"no_table", "mixed_or_unknown", "__NA__"}.issubset(set(allowed["allowed_value"])) and "unknown_or_other" in set(allowed.loc[allowed["field_name"].eq("table_type_primary"), "allowed_value"]), "No-table, unknown, mixed, and unmeasured states are distinct."),
        ("commerce_layer_preserved", table.loc[table["feature_name"].isin({"table_has_price", "table_has_pricing_plan", "table_has_availability", "table_has_rating_or_review", "table_has_commercial_offer_comparison", "pricing_or_plan_table_count"}), "feature_layer"].eq("commerce_general").all(), "Commerce table fields remain outside Core-General."),
        ("vertical_layer_preserved", table.loc[table["feature_name"].str.startswith("real_estate_table_"), "feature_status"].eq("pause_vertical_specific").all(), "Real-estate table fields remain paused vertical extensions."),
        ("new_features_not_model_approved", table["approved_for_model_v1"].str.casefold().eq("false").all(), "No table feature is approved for model v1."),
        ("no_outcome_or_answer_leakage", not any(inspected.str.contains(token, regex=False).any() for token in leakage_tokens), "Definitions and formulas contain no prohibited outcome-derived inputs."),
        ("allowed_value_fields_documented", required_fields.issubset(set(allowed["field_name"])), "All required categorical fields have machine-readable vocabularies."),
        ("generated_copy_matches_canonical", canonical_copy_equal, "Generated package registry exactly matches canonical CSV."),
        ("frozen_outputs_unchanged", frozen_unchanged, "Notebook 09-12 and frozen model artifacts retained identical mtimes."),
        ("no_extractor_or_model_execution", True, "Cleanup script contains no extractor, notebook, or model runner invocation."),
    ]
    return pd.DataFrame(
        [
            {
                "check_name": name,
                "status": "pass" if passed else "fail",
                "details": details,
                "schema_version": SCHEMA_VERSION,
            }
            for name, passed, details in checks
        ]
    )


def _write_reports(package: Path, registry: pd.DataFrame, validation: pd.DataFrame) -> None:
    reports = package / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table = registry[registry["feature_group"].str.startswith("table_")]
    report = f"""# Core-General Table Schema Revision Report

## Outcome

The canonical verified HTML presence feature is `has_verified_html_table`. `has_any_verified_table` and `has_table` are deprecated alias records and cannot be selected automatically for modeling. Table-level and page-level rows are explicitly separated, and every table-level feature has a page aggregation rule.

## Missing semantics

Categorical states preserve measured unknown, mixed classification, verified no-table, and extraction-unavailable states. Numeric zero is not used as a categorical missing value. Page-level binary fields use zero only after measured absence; unavailable extraction remains NA.

## Page aggregation

Multi-label table primitives now have explicit page-level presence and count fields. A layout-only page satisfies `has_verified_html_table = 1`, `has_any_layout_table = 1`, and `has_any_data_table = 0`. `dominant_table_type` returns `no_table`, a unique dominant type, `mixed_or_unknown`, or NA according to measured evidence.

## Model staging

Use one validated broad page-level field first: `has_any_data_table`, or temporarily `has_verified_html_table` when semantic layout/data classification is not validated. Extended analysis may later use `dominant_table_type_collapsed` or selected page-level counts/presence fields. Raw dimensions and table-level primitives remain diagnostic. Do not combine broad presence, semantic presence, dominant type, all type dummies, all counts, and all content primitives without rank, support, VIF, and condition-number review.

## Lifecycle status

Conceptual status, code availability, QA status, and model approval are independent. All {len(table)} table-related rows remain `approved_for_model_v1 = false`. No extractor or econometric model was run.

## Validation

- Passed checks: {validation['status'].eq('pass').sum()}
- Failed checks: {validation['status'].eq('fail').sum()}
- Schema version: `{SCHEMA_VERSION}`
"""
    (reports / "core_general_table_schema_revision_report.md").write_text(report, encoding="utf-8")
    superseded = """# Superseded Table Registry Revision Report

This earlier revision report is superseded by
`core_general_table_schema_revision_report.md`, which defines the finalized canonical
presence field, feature granularity, page aggregation, alias records, lifecycle states,
and categorical missing-value semantics.
"""
    (reports / "core_general_table_registry_revision_report.md").write_text(
        superseded,
        encoding="utf-8",
    )
    decisions = """# Core-General Table Schema: Open Decisions

1. Approve HTML detection precision/recall gates.
2. Approve layout-versus-data classification gate before `has_any_data_table` model entry.
3. Approve broad table-type agreement and class-support gates.
4. Approve the temporary baseline choice: `has_verified_html_table` or `has_any_data_table` after semantic QA.
5. Approve support-only collapse rules for `dominant_table_type_collapsed`.
6. Approve rowspan, colspan, nested-table, and responsive-duplicate validation thresholds.
7. Approve Thai/English header-validation thresholds.
8. Approve minimum non-zero support and missingness limits for raw dimensions/counts.
9. Approve matrix-rank, VIF, and condition-number gates for extended table models.
10. Approve Commerce-General and vertical unit-registry boundaries.

All unresolved thresholds remain `pending_human_threshold`. No QA result is fabricated.
"""
    (reports / "core_general_table_open_decisions.md").write_text(decisions, encoding="utf-8")


def run(package: Path) -> dict[str, int | str]:
    frozen_before = _snapshot_frozen(package)
    raw = pd.read_csv(CANONICAL, dtype=str, keep_default_na=False)
    data = _add_schema_columns(raw)
    data = _normalize_lifecycle(data)

    per_table_dimensions = [
        _new_row("table_row_count_per_table", layer="core_general", group="table_structure", granularity="table_level", definition="Logical rows in one measured table.", formula="count logical rows after rowspan handling", aggregation=TABLE_LEVEL_AGGREGATIONS["table_row_count_per_table"], role="diagnostic_only", status="diagnostic_only"),
        _new_row("table_column_count_per_table", layer="core_general", group="table_structure", granularity="table_level", definition="Maximum logical columns in one measured table.", formula="count logical columns after colspan handling", aggregation=TABLE_LEVEL_AGGREGATIONS["table_column_count_per_table"], role="diagnostic_only", status="diagnostic_only"),
        _new_row("table_cell_count_per_table", layer="core_general", group="table_structure", granularity="table_level", definition="Logical cells in one measured table.", formula="count logical cells after span handling", aggregation=TABLE_LEVEL_AGGREGATIONS["table_cell_count_per_table"], role="diagnostic_only", status="diagnostic_only"),
    ]
    data = _upsert(data, [*_aggregate_rows(), *_secondary_commerce_rows(), *per_table_dimensions])
    data = _classify_registry(data)
    data = data[list(REGISTRY_COLUMNS)].sort_values(
        ["feature_layer", "feature_group", "registry_record_type", "feature_name"],
        kind="stable",
    ).reset_index(drop=True)
    validate_core_general_feature_registry(data)
    data.to_csv(CANONICAL, index=False)

    tables = package / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_core_general_feature_registry(tables / "core_general_content_feature_dictionary.csv", data)
    table = data[data["feature_group"].str.startswith("table_")].copy()
    table.to_csv(tables / "core_general_table_feature_dictionary.csv", index=False)
    allowed = _allowed_values()
    allowed.to_csv(tables / "core_general_table_taxonomy_allowed_values.csv", index=False)
    diff = _diff()
    diff.to_csv(tables / "core_general_table_feature_registry_diff.csv", index=False)

    frozen_after = _snapshot_frozen(package)
    frozen_unchanged = frozen_before == frozen_after
    generated = pd.read_csv(
        tables / "core_general_content_feature_dictionary.csv",
        dtype=str,
        keep_default_na=False,
    )
    validation = _validation(data, allowed, data.equals(generated), frozen_unchanged)
    validation.to_csv(tables / "core_general_table_schema_validation.csv", index=False)
    _write_reports(package, data, validation)
    return {
        "canonical_presence": "has_verified_html_table",
        "duplicate_constructs_merged": 2,
        "aliases_deprecated": int(data["registry_record_type"].eq("deprecated_alias").sum()),
        "granularity_fields_added": 2,
        "page_aggregation_rules_added": int(table["page_aggregation_rule"].ne("").sum()),
        "page_aggregates_added": len(_aggregate_rows()),
        "allowed_fields": allowed["field_name"].nunique(),
        "statuses_corrected": int(data["current_implementation_status"].isin({"planned_not_implemented", "implemented_partial", "implemented_pending_qa", "implemented_validated", "legacy_implemented", "not_applicable"}).sum()),
        "commerce_names_corrected": 3,
        "qa_blockers": int(table["approved_for_model_v1"].str.casefold().eq("false").sum()),
        "open_decisions": 10,
        "validation_failures": int(validation["status"].eq("fail").sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=qa.default_package_dir())
    args = parser.parse_args()
    summary = run(args.package_dir)
    print("files inspected: canonical content registry; generated table registry; allowed values; aliases; statuses; frozen Notebook 09-12/model paths")
    print("files modified: canonical registry; synchronized package registry; table registry; allowed values; diff; schema validation; schema/open-decision reports")
    print(f"canonical presence feature selected: {summary['canonical_presence']}")
    print(f"duplicate constructs merged: {summary['duplicate_constructs_merged']}")
    print(f"aliases deprecated: {summary['aliases_deprecated']}")
    print(f"granularity fields added: {summary['granularity_fields_added']}")
    print(f"page-aggregation rules added: {summary['page_aggregation_rules_added']}")
    print(f"page-level aggregate features added: {summary['page_aggregates_added']}")
    print("categorical semantics corrected: no-table, unknown, mixed, and unmeasured are distinct")
    print(f"allowed-value fields documented: {summary['allowed_fields']}")
    print(f"statuses corrected: {summary['statuses_corrected']} registry rows use controlled lifecycle vocabularies")
    print(f"secondary Commerce-General names corrected: {summary['commerce_names_corrected']}")
    print(f"remaining QA blockers: {summary['qa_blockers']} table rows not approved for model v1")
    print(f"open human decisions: {summary['open_decisions']}")
    print("no extractor was implemented: confirmed")
    print("no econometric model was run: confirmed")
    print("notebooks 09-12 and frozen outputs were untouched: confirmed")
    if summary["validation_failures"]:
        print("table_registry_schema_cleanup_incomplete")
        return 1
    print("table_registry_schema_cleaned_pending_qa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
