#!/usr/bin/env python3
"""Revise the canonical Core-General registry with table taxonomy v1."""

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


TAXONOMY_VERSION = "core_general_table_taxonomy_v1"
REGISTRY_VERSION = "core_general_content_features_v2_table_taxonomy"
CANONICAL = ROOT / "config/core_general_content_feature_dictionary.csv"


def _base_feature(
    name: str,
    *,
    layer: str = "core_general",
    group: str = "table_content",
    kind: str = "primitive",
    definition: str,
    formula: str,
    provenance: str = "table_level_extraction",
    required_input: str = "preserved HTML/Markdown table evidence",
    transformation: str = "binary_or_count",
    language: str = "medium",
    extraction: str = "provenance-aware table extraction",
    missing: str = "NA means required table evidence was unmeasured; 0 is allowed only after verified absence.",
    specificity: str = "general",
    generalizability: str = "high",
    leakage: str = "safe_pre_outcome",
    confounders: str = "domain; website template; page function; publisher type; extraction quality",
    risks: str = "JavaScript rendering, flattened tables, responsive duplicates, and layout markup can alter measurement.",
    role: str = "diagnostic_or_feature_group",
    status: str = "refactor_needed",
    implementation: str = "planned_not_implemented",
    qa_status: str = "pending_human_threshold",
    minimum_gate: str = "pending_human_threshold",
    blocker: str = "table extraction and classification QA not yet passed",
    validation: str = "Validate on labeled HTML/Markdown tables before model entry.",
    aliases: str = "",
    canonical: str | None = None,
    notes: str = "Table structure among surfaced sources; not a causal effect of adding a table.",
) -> dict[str, str]:
    return {
        "feature_name": name,
        "canonical_column_name": canonical or name,
        "legacy_aliases": aliases,
        "feature_layer": layer,
        "feature_group": group,
        "primitive_or_composite": kind,
        "definition": definition,
        "formula": formula,
        "source_provenance": provenance,
        "required_input": required_input,
        "transformation": transformation,
        "language_dependency": language,
        "extraction_requirement": extraction,
        "missing_value_meaning": missing,
        "topic_specificity": specificity,
        "generalizability": generalizability,
        "leakage_status": leakage,
        "expected_confounders": confounders,
        "measurement_risks": risks,
        "recommended_model_role": role,
        "feature_status": status,
        "current_implementation_status": implementation,
        "taxonomy_or_rule_version": TAXONOMY_VERSION,
        "qa_status": qa_status,
        "approved_for_model_v1": "false",
        "minimum_qa_gate": minimum_gate,
        "model_entry_blocker": blocker,
        "validation_requirement": validation,
        "notes": notes,
        "registry_version": REGISTRY_VERSION,
    }


def _table_features() -> list[dict[str, str]]:
    f = _base_feature
    core = [
        f("has_verified_html_table", group="table_detection", definition="At least one HTML table node is verified present.", formula="1[verified HTML table present]; 0[HTML evidence measured and verified absent]; NA[HTML evidence unmeasured]", provenance="html_dom", transformation="nullable_binary", language="low", required_input="preserved HTML DOM", role="baseline_candidate_if_semantic_QA_unavailable", status="refactor_needed", implementation="planned_not_implemented", validation="Measure HTML detection precision and recall where a labeled sample permits.", notes="Broad HTML presence can include layout tables; it is not equivalent to has_any_data_table."),
        f("markdown_table_detected", group="table_detection", definition="At least one valid Markdown table is inferred present.", formula="1[valid Markdown table present]; 0[Markdown evidence measured and verified absent]; NA[Markdown evidence unmeasured]", provenance="scraped_markdown", transformation="nullable_binary", language="low", required_input="Markdown preserving table syntax", role="diagnostic_sensitivity", status="diagnostic_only", implementation="implemented_partial", validation="Compare Markdown inference with verified HTML where both exist.", notes="Never treat Markdown-inferred presence as identical to HTML verification without an explicit sensitivity comparison."),
        f("html_table_count", group="table_structure", definition="Count of HTML-verified table nodes before semantic layout/data classification.", formula="count(verified deduplicated HTML table nodes); NA when HTML unmeasured", provenance="html_dom", transformation="nullable_count", language="low", required_input="preserved HTML DOM", role="diagnostic_only", status="diagnostic_only", implementation="implemented_partial", validation="Validate responsive duplicate and nested-table handling.", notes="Raw count is not approved as a headline feature."),
        f("has_table", group="table_compatibility", definition="Deprecated broad table-presence compatibility field.", formula="legacy pipeline table presence rule", provenance="legacy_mixed", role="deprecated_alias", status="diagnostic_only", implementation="implemented_partial", qa_status="deprecated_compatibility", minimum_gate="not_applicable", blocker="use provenance-aware canonical fields", validation="Map legacy values to has_any_verified_table only when provenance confirms equivalence.", canonical="has_any_verified_table", notes="Compatibility output only; may mix substantive and layout tables."),
        f("inferred_text_table_detected", group="table_detection", definition="Table-like structure inferred only from flattened text.", formula="1[predeclared text-row/column pattern passes]", provenance="flattened_text", transformation="nullable_binary", role="diagnostic_only", status="diagnostic_only", validation="Estimate false-positive rate against HTML/Markdown ground truth."),
        f("table_detection_source", group="table_detection", kind="categorical_provenance", definition="Evidence source used to detect each table/page table state.", formula="enum from extraction evidence", provenance="extraction_provenance", transformation="categorical", language="low", role="diagnostic_provenance", status="diagnostic_only", validation="Confirm every detected table has exactly one or an explicit multiple source."),
        f("table_verification_status", group="table_detection", kind="categorical_provenance", definition="Verified, inferred, absent, or unmeasured table state.", formula="deterministic evidence-state rule", provenance="extraction_provenance", transformation="categorical", language="low", role="diagnostic_and_sample_definition", status="diagnostic_only", validation="Audit verified_absent versus unmeasured semantics with no unexplained row loss."),
        f("markdown_table_count", group="table_structure", definition="Count of valid Markdown tables.", formula="count(valid Markdown table blocks)", provenance="scraped_markdown", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Compare with HTML tables where both representations exist."),
        f("inferred_table_count", group="table_structure", definition="Count of table-like blocks inferred from flattened text.", formula="count(valid inferred text-table blocks)", provenance="flattened_text", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate precision and avoid treating inferred counts as HTML counts."),
        f("total_detected_table_count", group="table_structure", kind="derived_primitive", definition="Deduplicated count across verified and inferred detection sources.", formula="count(deduplicated table identities across sources)", provenance="html_markdown_text_reconciliation", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate cross-source deduplication and responsive duplicates."),
        f("table_row_count_total", group="table_structure", definition="Total rows across deduplicated detected tables.", formula="sum(rows after rowspan expansion and duplicate removal)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", aliases="table_row_count", validation="Validate rowspan, nested-table and responsive-duplicate handling."),
        f("table_column_count_max", group="table_structure", definition="Maximum columns across deduplicated detected tables.", formula="max(columns after colspan expansion)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", aliases="table_column_count", validation="Validate colspan and responsive/mobile table variants."),
        f("table_cell_count_total", group="table_structure", definition="Total logical cells across deduplicated detected tables.", formula="sum(logical cells after span handling)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate rowspan/colspan expansion without double counting."),
        f("data_table_count", group="table_structure", definition="Count of tables not classified as layout/navigation.", formula="count(table_type_primary != layout_or_navigation and verification measured)", transformation="nullable_count", role="diagnostic_then_candidate", validation="Pass layout-versus-data precision gate."),
        f("layout_table_count", group="table_structure", definition="Count of layout or navigation tables.", formula="count(table_type_primary == layout_or_navigation)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate legacy layout and navigation pages."),
        f("table_with_header_count", group="table_structure", definition="Count of tables with verified header cells or equivalent labels.", formula="count(tables with TH or validated header row)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate header detection in Thai and English."),
        f("table_caption_count", group="table_structure", definition="Count of tables with visible captions or equivalent labels.", formula="count(tables with caption/ARIA-labelled description)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate visible caption and ARIA label handling."),
        f("table_nested_count", group="table_structure", definition="Count of tables nested within another table.", formula="count(table nodes with table ancestor)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Ensure nested tables do not inflate parent row/cell counts."),
        f("table_type_primary", group="table_type", kind="categorical_classification", definition="Main observed function of one table; abstains when evidence is weak or conflicting.", formula="frozen deterministic classifier using structure and content only", transformation="categorical", role="diagnostic_table_level", status="refactor_needed", validation="Measure broad table-type agreement on a manually labeled sample."),
        f("table_has_numeric_facts", definition="Table contains numeric factual values beyond navigation identifiers.", formula="1[validated numeric-fact cells present]", aliases="table_contains_numeric_facts", role="diagnostic_then_candidate", validation="Validate identifiers, years, percentages, and locale formats."),
        f("table_has_measurements", definition="Table contains general measurement classes.", formula="1[general unit registry match in factual cell]", required_input="table cells and versioned general unit registry", role="diagnostic_then_candidate", validation="Validate length, mass, volume, duration, temperature, percentage, speed, storage, energy, and concentration units.", notes="Store unit-match provenance; vertical unit matches do not automatically activate this field."),
        f("table_measurement_unit_registry", group="table_detection", kind="categorical_provenance", definition="Registry that produced table unit matches.", formula="enum general_unit_registry|commerce_unit_registry|vertical_unit_registry|multiple|none|unmeasured", provenance="unit_match_provenance", transformation="categorical", role="diagnostic_provenance", status="diagnostic_only", validation="Confirm vertical matches remain isolated from Core-General specification rules."),
        f("table_has_specifications", definition="Table presents labeled attributes or specifications using general, non-vertical constructs.", formula="1[attribute-value/specification structure passes]", role="diagnostic_then_candidate", validation="Validate across industries and ensure vertical-only units do not activate it."),
        f("table_has_comparison", definition="Table compares alternatives or attributes across multiple entities.", formula="1[validated comparison structure passes]", aliases="table_contains_comparison", role="diagnostic_then_candidate", validation="Validate commercial and non-commercial comparisons."),
        f("table_has_directory_or_listing_structure", definition="Table organizes multiple entities as a directory or listing.", formula="1[repeated entity-row/listing structure passes]", role="diagnostic_then_candidate", validation="Distinguish substantive listings from navigation menus."),
        f("table_has_schedule_or_timeline", definition="Table organizes dates, times, stages, or milestones.", formula="1[validated schedule/timeline structure passes]", role="diagnostic_then_candidate", validation="Validate date/time formats in Thai and English."),
        f("table_has_transactional_structure", definition="Table contains form-like or transaction-action structure.", formula="1[validated input/action/transaction cells present]", role="diagnostic_only", status="diagnostic_only", validation="Distinguish forms and account controls from factual tables."),
        f("table_has_layout_or_navigation_structure", definition="Table is used primarily for page layout or navigation.", formula="1[layout/navigation classifier passes]", aliases="table_is_layout_or_navigation", role="diagnostic_exclusion", status="diagnostic_only", validation="Prioritize precision because errors contaminate has_any_data_table."),
        f("table_has_textual_facts", definition="Table contains substantive textual factual cells.", formula="1[non-navigation factual text pattern passes]", role="diagnostic_then_candidate", validation="Validate across page functions and languages."),
        f("table_has_header_labels", group="table_labels", definition="Table contains explicit header labels.", formula="1[TH or validated header row present]", role="diagnostic_only", status="diagnostic_only", validation="Validate semantic and inferred headers."),
        f("table_has_row_labels", group="table_labels", definition="Table contains labels identifying row meaning.", formula="1[validated row-label structure present]", role="diagnostic_only", status="diagnostic_only", validation="Validate first-column labels and multi-level headers."),
        f("table_has_column_labels", group="table_labels", definition="Table contains labels identifying column meaning.", formula="1[validated column-label structure present]", role="diagnostic_only", status="diagnostic_only", validation="Validate header rows and colspan labels."),
        f("has_any_verified_table", group="table_page_aggregate", kind="derived_primitive", definition="Page has at least one HTML-verified table, including layout tables.", formula="1[html_table_count > 0]; 0 only if HTML verified absent; otherwise NA", provenance="html_dom", transformation="nullable_binary", role="baseline_candidate_if_semantic_QA_unavailable", status="refactor_needed", aliases="has_table", validation="Pass HTML detection and absent-versus-unmeasured gates."),
        f("has_any_data_table", group="table_page_aggregate", kind="derived_primitive", definition="Page has at least one detected table not classified as layout/navigation.", formula="1[data_table_count > 0]; 0 only when all measured tables are layout or verified absent; otherwise NA", transformation="nullable_binary", role="preferred_baseline_after_QA", status="refactor_needed", validation="Pass layout-versus-data precision and multi-table aggregation gates."),
        f("has_any_layout_table", group="table_page_aggregate", kind="derived_primitive", definition="Page has at least one layout/navigation table.", formula="1[layout_table_count > 0] with verified evidence-state semantics", transformation="nullable_binary", role="diagnostic_only", status="diagnostic_only", validation="Validate layout classification and unmeasured handling."),
        f("has_multiple_tables", group="table_page_aggregate", kind="derived_primitive", definition="Page has at least two deduplicated detected tables.", formula="1[total_detected_table_count >= 2]", transformation="nullable_binary", role="diagnostic_or_sensitivity", status="diagnostic_only", validation="Pass responsive-duplicate and nested-table gates."),
        f("has_multiple_table_types", group="table_page_aggregate", kind="derived_primitive", definition="Page has at least two distinct non-unknown table types.", formula="1[distinct_table_type_count >= 2]", transformation="nullable_binary", role="diagnostic_or_sensitivity", status="diagnostic_only", validation="Pass table-type agreement and aggregation gates."),
        f("dominant_table_type", group="table_page_aggregate", kind="derived_categorical", definition="Unique most frequent qualifying table type on a page.", formula="type with strictly greater count than runner-up; otherwise mixed_or_unknown", transformation="categorical", role="extended_table_model_after_QA", status="sensitivity_only", validation="Validate deterministic tie/abstention rule and sparse category support.", notes="Derived without citation outcomes; layout-only pages remain layout_or_navigation."),
        f("dominant_table_type_collapsed", group="table_page_aggregate", kind="derived_categorical", definition="Support-collapsed dominant table type for later sensitivity models.", formula="predeclared support-only collapse of dominant_table_type; mixed/unknown explicit", transformation="categorical", role="extended_table_model_after_QA", status="sensitivity_only", validation="Human-approve support thresholds before outcomes."),
        f("distinct_table_type_count", group="table_page_aggregate", kind="derived_primitive", definition="Distinct qualifying primary table types on a page.", formula="nunique(table_type_primary excluding unknown where measured)", transformation="nullable_count", role="diagnostic_only", status="diagnostic_only", validation="Validate unknown handling and multi-table aggregation."),
    ]
    count_specs = {
        "factual_or_specification_table_count": "factual_or_specification",
        "comparison_table_count": "comparison",
        "directory_or_listing_table_count": "directory_or_listing",
        "schedule_or_timeline_table_count": "schedule_or_timeline",
        "transactional_or_form_table_count": "transactional_or_form",
        "layout_or_navigation_table_count": "layout_or_navigation",
        "unknown_or_other_table_count": "unknown_or_other",
    }
    for name, value in count_specs.items():
        core.append(
            f(name, group="table_page_aggregate", kind="derived_primitive", definition=f"Page count of {value} tables.", formula=f"count(table_type_primary == {value})", transformation="nullable_count", role="diagnostic_or_feature_group", status="diagnostic_only", validation="Pass type-classification and multi-table aggregation QA.")
        )

    commerce_specs = {
        "table_has_price": ("Table contains explicit price or currency details.", "1[validated price detail in table cells]"),
        "table_has_pricing_plan": ("Table presents commercial plans, tiers, or packages.", "1[validated plan/tier structure]"),
        "table_has_availability": ("Table presents stock, booking, service, or enrollment availability.", "1[validated availability structure]"),
        "table_has_rating_or_review": ("Table presents ratings, review counts, or review summaries.", "1[validated rating/review structure]"),
        "table_has_commercial_offer_comparison": ("Table compares commercial offers, products, plans, or providers.", "1[validated commercial comparison structure]"),
        "pricing_or_plan_table_count": ("Page count of pricing_or_plan primary tables.", "count(table_type_primary == pricing_or_plan)"),
    }
    commerce = [
        f(name, layer="commerce_general", group="table_commerce", definition=definition, formula=formula, specificity="commerce", generalizability="conditional", role="commerce_extension_only", status="commerce_general_keep", validation="Validate only on commercial samples; exclude from universal Core-General headline models.", notes="Pricing and commerce structure may primarily identify commercial page function.")
        for name, (definition, formula) in commerce_specs.items()
    ]

    vertical_names = {
        "real_estate_table_contains_unit_size": "Real-estate unit-size details such as sqm or square metres.",
        "real_estate_table_contains_bedroom_mix": "Bedroom, studio, duplex, penthouse, or unit-mix details.",
        "real_estate_table_contains_floor_plan": "Floor-plan or layout details.",
        "real_estate_table_contains_price_per_area": "Price-per-area details using real-estate units.",
        "real_estate_table_contains_project_inventory": "Project-specific unit inventory details.",
    }
    vertical = [
        f(name, layer="vertical_specific", group="table_real_estate", definition=definition, formula="versioned real-estate table dictionary rule", provenance="table_cells_and_vertical_unit_registry", required_input="table cells and real-estate vertical registry", language="high", extraction="vertical extension parser", specificity="real_estate", generalizability="low_outside_vertical", leakage="safe_but_topic_specific", confounders="real-estate page function; inventory; market", risks="Vertical vocabulary has no universal Core-General interpretation.", role="vertical_extension_only", status="pause_vertical_specific", implementation="planned_not_implemented", validation="Preserve under vertical_extensions/real_estate and exclude from Core-General models.", notes="Must not activate Core-General table specifications solely from vertical-unit matches.")
        for name, definition in vertical_names.items()
    ]
    return [*core, *commerce, *vertical]


def _upgrade_schema(frame: pd.DataFrame) -> pd.DataFrame:
    defaults = {
        "canonical_column_name": frame["feature_name"] if "feature_name" in frame else "",
        "legacy_aliases": "",
        "taxonomy_or_rule_version": "core_general_content_features_v1",
        "qa_status": "not_assessed",
        "approved_for_model_v1": "false",
        "minimum_qa_gate": "pending_human_threshold",
        "model_entry_blocker": "not yet reviewed for Core-General model v1",
    }
    upgraded = frame.copy()
    for column, default in defaults.items():
        if column not in upgraded:
            upgraded[column] = default
    upgraded["registry_version"] = REGISTRY_VERSION
    return upgraded


def _apply_alias_updates(frame: pd.DataFrame) -> pd.DataFrame:
    updates = {
        "table_row_count": ("table_row_count_total", "deprecated_alias", "diagnostic_only"),
        "table_column_count": ("table_column_count_max", "deprecated_alias", "diagnostic_only"),
        "table_contains_numeric_facts": ("table_has_numeric_facts", "deprecated_alias", "diagnostic_only"),
        "table_contains_comparison": ("table_has_comparison", "deprecated_alias", "diagnostic_only"),
        "table_is_layout_or_navigation": ("table_has_layout_or_navigation_structure", "deprecated_alias", "diagnostic_only"),
    }
    output = frame.copy()
    for legacy, (canonical, role, status) in updates.items():
        mask = output["feature_name"].eq(legacy)
        if not mask.any():
            continue
        output.loc[mask, "canonical_column_name"] = canonical
        output.loc[mask, "recommended_model_role"] = role
        output.loc[mask, "feature_status"] = status
        output.loc[mask, "qa_status"] = "deprecated_compatibility"
        output.loc[mask, "approved_for_model_v1"] = "false"
        output.loc[mask, "minimum_qa_gate"] = "not_applicable"
        output.loc[mask, "model_entry_blocker"] = f"use canonical field {canonical}"
        output.loc[mask, "taxonomy_or_rule_version"] = TAXONOMY_VERSION
        output.loc[mask, "notes"] = f"Backward-compatible alias retained; canonical field is {canonical}."
    return output


def _upsert(frame: pd.DataFrame, rows: list[dict[str, str]]) -> pd.DataFrame:
    names = {row["feature_name"] for row in rows}
    retained = frame[~frame["feature_name"].isin(names)].copy()
    return pd.concat([retained, pd.DataFrame(rows)], ignore_index=True)[list(REGISTRY_COLUMNS)]


def _allowed_values() -> pd.DataFrame:
    definitions = {
        "table_detection_source": {
            "verified_html": "Detected from preserved HTML table nodes.",
            "inferred_markdown": "Detected from valid Markdown table structure.",
            "inferred_text": "Detected only from flattened text patterns.",
            "multiple_sources": "Same deduplicated table supported by multiple representations.",
            "none_verified_absent": "Required evidence was measured and no table was found.",
            "unmeasured": "Required extraction evidence was unavailable.",
        },
        "table_verification_status": {
            "verified_html": "Presence verified in HTML.",
            "inferred_markdown": "Presence inferred from Markdown.",
            "inferred_text": "Presence inferred from text only.",
            "verified_absent": "Evidence was measured and absence verified.",
            "unmeasured": "Required evidence unavailable; must remain NA.",
        },
        "table_type_primary": {
            "factual_or_specification": "Facts, attributes, or general specifications.",
            "comparison": "Compares alternatives or attributes.",
            "pricing_or_plan": "Commercial prices, plans, or tiers.",
            "directory_or_listing": "Organizes multiple entities or listings.",
            "schedule_or_timeline": "Dates, times, stages, or milestones.",
            "transactional_or_form": "Form, account, or transaction structure.",
            "layout_or_navigation": "Layout/navigation rather than substantive content.",
            "unknown_or_other": "Evidence weak, conflicting, or outside stable vocabulary.",
        },
        "dominant_table_type": {
            "factual_or_specification": "Unique most frequent factual/specification type.",
            "comparison": "Unique most frequent comparison type.",
            "pricing_or_plan": "Unique most frequent pricing/plan type.",
            "directory_or_listing": "Unique most frequent directory/listing type.",
            "schedule_or_timeline": "Unique most frequent schedule/timeline type.",
            "transactional_or_form": "Unique most frequent transactional/form type.",
            "layout_or_navigation": "Page tables are uniquely dominated by layout/navigation.",
            "unknown_or_other": "Unique most frequent unknown/other type.",
            "mixed_or_unknown": "Tie for highest count, no qualifying type, or insufficient evidence.",
        },
        "table_measurement_unit_registry": {
            "general_unit_registry": "General cross-industry units.",
            "commerce_unit_registry": "Commerce-specific units.",
            "vertical_unit_registry": "Vertical extension units only.",
            "multiple": "Matches from more than one registry, retained separately.",
            "none": "Measured with no valid unit match.",
            "unmeasured": "Unit matching unavailable.",
        },
    }
    rows = []
    for field, values in definitions.items():
        for order, (value, definition) in enumerate(values.items(), start=1):
            rows.append({"field_name": field, "allowed_value": value, "display_order": order, "definition": definition, "taxonomy_or_rule_version": TAXONOMY_VERSION})
    return pd.DataFrame(rows)


def _qa_plan() -> pd.DataFrame:
    checks = [
        ("html_table_detection_precision", "HTML table presence", "precision"),
        ("html_table_detection_recall", "HTML table presence where labeled recall is feasible", "recall"),
        ("layout_vs_data_precision", "layout/navigation versus substantive data tables", "precision"),
        ("broad_table_type_agreement", "table_type_primary", "human agreement and per-class support"),
        ("html_markdown_consistency", "pages with both HTML and Markdown", "agreement and explained disagreement"),
        ("multi_table_page_aggregation", "page aggregates from table-level labels", "exact aggregation agreement"),
        ("rowspan_colspan_handling", "logical row/column/cell counts", "manual count agreement"),
        ("responsive_duplicate_tables", "desktop/mobile duplicate structures", "duplicate-removal accuracy"),
        ("nested_tables", "nested table measurement", "parent/child count accuracy"),
        ("javascript_rendered_tables", "JS-rendered table availability", "detection coverage by rendering mode"),
        ("thai_table_headers", "Thai header and label extraction", "precision/recall or agreement"),
        ("english_table_headers", "English header and label extraction", "precision/recall or agreement"),
        ("absent_vs_unmeasured", "verified absence versus unavailable evidence", "state-classification agreement"),
        ("no_unexplained_row_loss", "registry/extraction joins", "input-output row reconciliation"),
        ("vertical_unit_isolation", "general versus commerce versus vertical unit registries", "cross-registry contamination rate"),
        ("model_rank_collinearity", "proposed baseline and extended table feature sets", "rank, VIF, condition number"),
    ]
    return pd.DataFrame(
        [
            {
                "qa_check": name,
                "scope": scope,
                "metric": metric,
                "sample_design": "stratified manual sample by source, page type, language, extraction mode, and table state",
                "minimum_qa_gate": "pending_human_threshold",
                "model_entry_blocker": "yes",
                "qa_status": "pending_human_threshold",
                "notes": "No validation result claimed during registry revision.",
            }
            for name, scope, metric in checks
        ]
    )


def _registry_diff() -> pd.DataFrame:
    rows = [
        ("has_verified_html_table", "retained", "core_general", "core_general", "refactor_needed", "refactor_needed", "Retained as provenance-specific baseline candidate pending QA."),
        ("markdown_table_detected", "retained", "core_general", "core_general", "diagnostic_only", "diagnostic_only", "Retained separately from HTML verification."),
        ("table_row_count", "deprecated_alias", "core_general", "core_general", "refactor_needed", "diagnostic_only", "Canonical field is table_row_count_total."),
        ("table_column_count", "deprecated_alias", "core_general", "core_general", "refactor_needed", "diagnostic_only", "Canonical field is table_column_count_max."),
        ("table_contains_numeric_facts", "renamed", "core_general", "core_general", "refactor_needed", "diagnostic_only", "Canonical field is table_has_numeric_facts."),
        ("table_contains_comparison", "renamed", "core_general", "core_general", "refactor_needed", "diagnostic_only", "Canonical field is table_has_comparison."),
        ("table_is_layout_or_navigation", "renamed", "core_general", "core_general", "refactor_needed", "diagnostic_only", "Canonical field is table_has_layout_or_navigation_structure."),
        ("table_contains_price_or_specs", "split", "mixed_or_unclassified", "core_general|commerce_general", "not_registered", "blocked_pending_qa", "Split into table_has_specifications and Commerce-General table_has_price."),
    ]
    for feature in _table_features():
        name = feature["feature_name"]
        if name in {row[0] for row in rows}:
            continue
        action = "added"
        if feature["feature_layer"] == "commerce_general":
            action = "moved_to_commerce"
        elif feature["feature_layer"] == "vertical_specific":
            action = "moved_to_vertical"
        rows.append((name, action, "", feature["feature_layer"], "", feature["feature_status"], feature["notes"] or feature["definition"]))
    return pd.DataFrame(rows, columns=["feature_name", "action", "previous_layer", "new_layer", "previous_status", "new_status", "reason"])


def _write_reports(package: Path, registry: pd.DataFrame, diff: pd.DataFrame, qa: pd.DataFrame) -> None:
    reports = package / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    table_rows = registry[registry["feature_group"].str.startswith("table_")]
    report = f"""# Core-General Table Registry Revision Report

## Scope

This registry revision defines a provenance-aware table taxonomy without rerunning any econometric model or changing frozen Notebook 09-12 outputs. Table variables describe observed structure among surfaced sources and do not identify a causal effect of adding a table.

## Result

- Registry version: `{REGISTRY_VERSION}`
- Table taxonomy version: `{TAXONOMY_VERSION}`
- Table-related registry rows: {len(table_rows)}
- Core-General table rows: {(table_rows['feature_layer'] == 'core_general').sum()}
- Commerce-General table rows: {(table_rows['feature_layer'] == 'commerce_general').sum()}
- Paused real-estate table rows: {(table_rows['feature_layer'] == 'vertical_specific').sum()}
- Approved for model v1: {table_rows['approved_for_model_v1'].astype(str).str.casefold().eq('true').sum()}

## Design

Detection provenance, verification state, structure, primary function, multi-label content, and page-level aggregates are separate. `has_any_data_table` excludes layout-only tables. `dominant_table_type` uses a strict unique-winner rule and returns `mixed_or_unknown` on ties or insufficient evidence. Pricing fields are Commerce-General. Real-estate unit tables remain paused vertical extensions.

## Initial model staging

1. Use `has_any_data_table` only after semantic/layout QA. If semantic classification is not validated, use `has_verified_html_table` only as a broad sensitivity proxy.
2. Add `C(dominant_table_type_collapsed)` or selected non-overlapping functions only after support and classification QA.
3. Keep raw dimensions, type counts, and multi-label indicators diagnostic until reliability, rank, VIF, and condition-number checks pass.
4. Never include broad presence, data-table presence, dominant type, every type dummy, and all type counts simultaneously without a documented rank review.

## QA status

All {len(qa)} predefined QA checks remain `pending_human_threshold`. No precision, recall, agreement, or model-readiness result is invented here.
"""
    (reports / "core_general_table_registry_revision_report.md").write_text(report, encoding="utf-8")
    decisions = """# Core-General Table Registry: Open Decisions

1. Approve minimum HTML detection precision and, where feasible, recall thresholds.
2. Approve layout-versus-data classification precision required for `has_any_data_table`.
3. Approve minimum broad table-type agreement and per-class sample support.
4. Approve HTML-versus-Markdown consistency tolerance and reconciliation precedence.
5. Approve handling rules for rowspan, colspan, nested and responsive duplicated tables.
6. Approve Thai and English header-validation thresholds.
7. Approve table-type category collapse thresholds without viewing model outcomes.
8. Approve whether the first baseline is `has_any_data_table` or the broader `has_verified_html_table`.
9. Approve general, commerce and vertical unit-registry boundaries.
10. Approve the labeled QA sample size and stratification plan.

Until approval, every affected gate is `pending_human_threshold` and table features remain unapproved for model v1.
"""
    (reports / "core_general_table_open_decisions.md").write_text(decisions, encoding="utf-8")


def run(package: Path) -> dict[str, int]:
    original = pd.read_csv(CANONICAL, dtype=str, keep_default_na=False)
    upgraded = _upgrade_schema(original)
    upgraded = _apply_alias_updates(upgraded)
    revised = _upsert(upgraded, _table_features())
    revised = revised.sort_values(["feature_layer", "feature_group", "feature_name"], kind="stable")
    validate_core_general_feature_registry(revised)
    revised.to_csv(CANONICAL, index=False)

    tables = package / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_core_general_feature_registry(tables / "core_general_content_feature_dictionary.csv", revised)
    table_dictionary = revised[
        revised["feature_group"].str.startswith("table_")
        | revised["feature_name"].isin(
            {"has_verified_html_table", "markdown_table_detected", "html_table_count"}
        )
    ].copy()
    table_dictionary.to_csv(tables / "core_general_table_feature_dictionary.csv", index=False)
    allowed = _allowed_values()
    allowed.to_csv(tables / "core_general_table_taxonomy_allowed_values.csv", index=False)
    diff = _registry_diff()
    diff.to_csv(tables / "core_general_table_feature_registry_diff.csv", index=False)
    qa = _qa_plan()
    qa.to_csv(tables / "core_general_table_feature_qa_plan.csv", index=False)
    _write_reports(package, revised, diff, qa)
    return {
        "registry_rows": len(revised),
        "table_rows": len(table_dictionary),
        "features_added": int(diff["action"].isin({"added", "moved_to_commerce", "moved_to_vertical"}).sum()),
        "renamed_or_split": int(diff["action"].isin({"renamed", "split"}).sum()),
        "moved_to_commerce": int(diff["action"].eq("moved_to_commerce").sum()),
        "moved_to_vertical": int(diff["action"].eq("moved_to_vertical").sum()),
        "deprecated_aliases": int(diff["action"].eq("deprecated_alias").sum()),
        "qa_blockers": len(qa),
        "open_decisions": 10,
    }


def main() -> int:
    from v2_cleanup_core_general_table_registry_schema import main as cleanup_main

    return cleanup_main()


if __name__ == "__main__":
    raise SystemExit(main())
