"""Specification registry for the future Core-General content feature layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REGISTRY_VERSION = "core_general_content_features_v1"
CANONICAL_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "config/core_general_content_feature_dictionary.csv"
)
ALLOWED_STATUSES = {
    "core_keep",
    "commerce_general_keep",
    "pause_vertical_specific",
    "refactor_needed",
    "diagnostic_only",
    "sensitivity_only",
    "needs_extraction_fix",
    "exclude_leakage",
}
ALLOWED_IMPLEMENTATION_STATUSES = {
    "planned_not_implemented",
    "implemented_partial",
    "implemented_pending_qa",
    "implemented_validated",
    "legacy_implemented",
    "not_applicable",
}
ALLOWED_QA_STATUSES = {
    "not_started",
    "pending_human_threshold",
    "pending_manual_validation",
    "qa_failed",
    "qa_passed",
}
ALLOWED_GRANULARITIES = {
    "table_level",
    "page_level",
    "extraction_diagnostic",
    "registry_metadata",
    "not_table_related",
}
REGISTRY_COLUMNS = (
    "feature_name",
    "canonical_column_name",
    "legacy_aliases",
    "registry_record_type",
    "replacement_feature_name",
    "feature_granularity",
    "page_aggregation_rule",
    "feature_layer",
    "feature_group",
    "primitive_or_composite",
    "definition",
    "formula",
    "source_provenance",
    "required_input",
    "transformation",
    "language_dependency",
    "extraction_requirement",
    "missing_value_meaning",
    "topic_specificity",
    "generalizability",
    "leakage_status",
    "expected_confounders",
    "measurement_risks",
    "recommended_model_role",
    "feature_status",
    "current_implementation_status",
    "taxonomy_or_rule_version",
    "qa_status",
    "approved_for_model_v1",
    "minimum_qa_gate",
    "model_entry_blocker",
    "validation_requirement",
    "notes",
    "registry_version",
)


def _row(
    name: str,
    layer: str,
    group: str,
    kind: str,
    definition: str,
    formula: str,
    provenance: str,
    required_input: str,
    transformation: str,
    language_dependency: str,
    extraction_requirement: str,
    missing_meaning: str,
    specificity: str,
    generalizability: str,
    leakage: str,
    confounders: str,
    risks: str,
    role: str,
    status: str,
    implementation: str,
    validation: str,
    notes: str = "",
) -> dict[str, str]:
    return dict(
        zip(
            REGISTRY_COLUMNS,
            (
                name,
                layer,
                group,
                kind,
                definition,
                formula,
                provenance,
                required_input,
                transformation,
                language_dependency,
                extraction_requirement,
                missing_meaning,
                specificity,
                generalizability,
                leakage,
                confounders,
                risks,
                role,
                status,
                implementation,
                validation,
                notes,
                REGISTRY_VERSION,
            ),
        )
    )


def _build_initial_core_general_feature_registry() -> pd.DataFrame:
    """Return the frozen pre-estimation feature specification."""
    r = _row
    rows = [
        r("log2_word_count_plus1", "core_general", "page_length", "derived_primitive", "Primary page-length measure.", "log2(word_count + 1)", "extracted_page_text", "word_count", "log2", "medium", "measurable page text", "Unavailable means length was not measurable; never zero-fill.", "general", "high", "safe_pre_outcome", "page type; domain template; extraction scope", "Truncation and boilerplate alter length.", "focal_G1_G2", "core_keep", "implemented", "Compare against full-text word counts and inspect tails.", "Do not include jointly with log2_content_chars_plus1 without a collinearity justification."),
        r("log2_content_chars_plus1", "core_general", "page_length", "derived_primitive", "Alternative character-length measure.", "log2(content_chars + 1)", "extracted_page_text", "content_chars", "log2", "low", "measurable page text", "Unavailable means content was not measurable.", "general", "high", "safe_pre_outcome", "page type; domain template; extraction scope", "HTML noise and script text may inflate characters.", "sensitivity_alternative", "sensitivity_only", "implemented", "Compare with word length; never include both automatically.", "Alternative to, not companion for, the primary length measure."),
        r("has_verified_html_table", "core_general", "table_structure", "primitive", "At least one semantic HTML table verified in the DOM.", "1[verified semantic table count > 0]", "html_dom", "HTML table nodes and layout classifier", "binary", "low", "preserved HTML DOM", "Missing means HTML table verification was unavailable, not no table.", "general", "high", "safe_pre_outcome", "domain template; page function", "Layout tables may be mistaken for factual tables.", "focal_G1_G2", "refactor_needed", "not_implemented", "Manually validate semantic versus layout tables across domains.", "Treat as a structural proxy, not a causal table effect."),
        r("markdown_table_detected", "core_general", "table_structure", "primitive", "Markdown pipe-table structure is present.", "1[valid header/separator/row pattern]", "scraped_markdown", "markdown blocks", "binary", "low", "markdown preserving table syntax", "Missing means Markdown was unavailable.", "general", "high", "safe_pre_outcome", "crawler conversion; page function", "Flattened or malformed tables can be missed.", "diagnostic_or_sensitivity", "diagnostic_only", "partial", "Compare with verified HTML tables and report disagreement.", "Never silently combine with HTML-confirmed tables."),
        r("html_table_count", "core_general", "table_structure", "primitive", "Count of verified semantic HTML tables.", "count(semantic HTML table nodes)", "html_dom", "verified table nodes", "count_or_threshold", "low", "preserved HTML DOM", "Missing means HTML was unavailable.", "general", "high", "safe_pre_outcome", "domain template; page function", "Duplicate responsive tables can inflate counts.", "diagnostic_then_nonlinear", "refactor_needed", "partial", "Deduplicate responsive copies and compare with has_verified_html_table."),
        r("table_row_count", "core_general", "table_structure", "primitive", "Rows across verified semantic tables.", "sum(non-layout rows)", "html_dom", "verified table cells", "count_or_log2", "low", "preserved table structure", "Missing means table structure was unavailable.", "general", "high", "safe_pre_outcome", "page function; template", "Headers and nested tables complicate counts.", "candidate_focal", "refactor_needed", "not_implemented", "Validate row parsing and duplicate tables."),
        r("table_column_count", "core_general", "table_structure", "primitive", "Maximum or summarized columns in verified semantic tables.", "max(non-layout columns)", "html_dom", "verified table cells", "count_or_bins", "low", "preserved table structure", "Missing means table structure was unavailable.", "general", "high", "safe_pre_outcome", "page function; mobile templates", "Colspan and responsive markup complicate width.", "candidate_focal", "refactor_needed", "not_implemented", "Validate colspan handling and mobile variants."),
        r("table_contains_numeric_facts", "core_general", "table_structure", "primitive", "Verified table contains numeric factual cells.", "1[numeric factual cell pattern present]", "html_dom", "table cell text", "binary", "medium", "semantic table cells", "Missing means table content was unavailable.", "general", "high", "safe_pre_outcome", "page function; commerce status", "Identifiers may be mistaken for facts.", "candidate_focal", "refactor_needed", "not_implemented", "Multilingual manual validation of numeric cell rules."),
        r("table_contains_comparison", "core_general", "table_structure", "primitive", "Verified table compares multiple alternatives or attributes.", "1[comparison structure rule passes]", "html_dom", "headers and table cells", "binary", "medium", "semantic table structure", "Missing means comparison structure was unavailable.", "general", "conditional", "safe_pre_outcome", "page function; commercial intent", "Comparison intent is difficult to infer from headers alone.", "sensitivity", "refactor_needed", "not_implemented", "Validate across commercial and non-commercial tables."),
        r("table_is_layout_or_navigation", "core_general", "table_structure", "primitive", "Table node appears to be layout or navigation rather than content.", "layout classifier from links/cells/ARIA/context", "html_dom", "table DOM and context", "binary", "low", "preserved DOM context", "Missing means classification was unavailable.", "general", "high", "safe_pre_outcome", "legacy site technology", "Classifier error contaminates verified-table features.", "diagnostic_exclusion", "refactor_needed", "not_implemented", "High-recall manual QA on legacy and navigation tables."),
        r("heading_count", "core_general", "heading_structure", "primitive", "Count of preserved content headings.", "count(H1-H6 content nodes)", "html_dom_or_markdown", "heading nodes", "count_or_prespecified_bins", "medium", "HTML tags or preserved Markdown headings", "Missing means heading structure was unavailable.", "general", "high", "safe_pre_outcome", "domain template; page type; length", "Navigation headings and flattened text bias counts.", "focal_nonlinear", "core_keep", "implemented_partial", "Separate content headings from navigation and check nonlinear support."),
        r("heading_depth", "core_general", "heading_structure", "primitive", "Maximum verified H1-H6 depth used in content.", "max(level of content heading nodes)", "html_dom", "H1-H6 nodes", "count_or_bins", "low", "preserved HTML heading tags", "Missing means tag depth was unavailable.", "general", "high", "safe_pre_outcome", "CMS template", "Cannot be inferred reliably from flattened text.", "candidate_focal", "refactor_needed", "not_implemented", "HTML-only validation across CMS platforms."),
        r("heading_hierarchy_consistency", "core_general", "heading_structure", "composite", "Degree to which content headings follow a coherent hierarchy.", "documented score from skipped levels; repeated H1; order", "html_dom", "ordered H1-H6 nodes", "bounded_score", "low", "preserved ordered heading tags", "Missing means hierarchy was unverifiable.", "general", "high", "safe_pre_outcome", "CMS template", "Score formula can hide component behavior.", "sensitivity", "refactor_needed", "not_implemented", "Retain skipped-level and repeated-H1 primitives; test component dominance."),
        r("question_heading_count", "core_general", "heading_structure", "primitive", "Content headings phrased as visible questions.", "count(question headings)", "html_dom_or_markdown", "heading text", "count_or_binary", "high", "preserved headings and language-specific question rules", "Missing means headings were unavailable.", "general", "conditional", "safe_pre_outcome", "language; page type", "Thai questions may omit question marks.", "candidate_focal", "core_keep", "implemented_partial", "Thai/English manual validation and segmentation confidence."),
        r("sentence_count", "core_general", "writing_structure", "primitive", "Number of segmented sentences in measurable page text.", "count(segmented sentences)", "page_text", "page text and segmenter", "count_or_log2", "high", "language-aware segmentation", "Missing means sentence segmentation was unavailable.", "general", "high", "safe_pre_outcome", "length; language; page type", "Thai sentence boundaries are uncertain.", "candidate_focal_or_diagnostic", "core_keep", "implemented", "Benchmark Thai and English segmentation separately."),
        r("median_sentence_length_words", "core_general", "writing_structure", "primitive", "Median segmented sentence length.", "median(words per sentence)", "page_text", "sentence segmentation", "continuous_or_spline", "high", "language-aware segmentation", "Missing means insufficient segmented sentences.", "general", "high", "safe_pre_outcome", "language; genre", "Tokenization differs across Thai and English.", "candidate_focal", "core_keep", "implemented", "Validate tokenization and inspect nonlinear shape."),
        r("p90_sentence_length_words", "core_general", "writing_structure", "primitive", "Upper-tail sentence length.", "p90(words per sentence)", "page_text", "sentence segmentation", "continuous_or_bins", "high", "language-aware segmentation", "Missing means insufficient sentences.", "general", "high", "safe_pre_outcome", "language; genre", "Sparse pages produce unstable quantiles.", "diagnostic_or_sensitivity", "core_keep", "implemented", "Set a minimum sentence count before reporting."),
        r("paragraph_count", "core_general", "writing_structure", "primitive", "Number of preserved content paragraphs.", "count(content paragraph blocks)", "html_dom_or_markdown", "paragraph blocks", "count_or_log2", "medium", "block boundaries preserved", "Missing means paragraph boundaries were unavailable.", "general", "high", "safe_pre_outcome", "length; template", "Flattened text destroys paragraph boundaries.", "candidate_focal", "core_keep", "implemented_partial", "Compare HTML and Markdown paragraph boundaries."),
        r("median_paragraph_length_words", "core_general", "writing_structure", "primitive", "Median words per preserved paragraph.", "median(words per paragraph)", "html_dom_or_markdown", "paragraph blocks", "continuous_or_spline", "high", "block boundaries and tokenization", "Missing means insufficient paragraphs.", "general", "high", "safe_pre_outcome", "language; page type", "Markdown conversion may merge blocks.", "candidate_focal", "core_keep", "implemented", "Validate against HTML blocks in both languages."),
        r("bullet_list_count", "core_general", "list_structure", "primitive", "Count of content bullet lists.", "count(UL content nodes or validated Markdown lists)", "html_dom_or_markdown", "list nodes", "count_or_binary", "low", "preserved list structure", "Missing means list structure was unavailable.", "general", "high", "safe_pre_outcome", "template; page type", "Navigation menus may be counted as lists.", "candidate_focal", "refactor_needed", "implemented_partial", "Exclude navigation/footer lists using DOM context."),
        r("numbered_list_count", "core_general", "list_structure", "primitive", "Count of content ordered lists.", "count(OL content nodes or validated Markdown lists)", "html_dom_or_markdown", "list nodes", "count_or_binary", "low", "preserved list structure", "Missing means list structure was unavailable.", "general", "high", "safe_pre_outcome", "how-to page function", "Numbered paragraphs may be false positives.", "candidate_focal", "refactor_needed", "implemented_partial", "Compare DOM and Markdown detection."),
        r("visible_qa_structure", "core_general", "qa_and_schema", "composite", "Visible paired question and answer structure.", "documented rule from question heading and following answer block", "html_dom_or_markdown", "ordered content blocks", "binary", "high", "block order and language rules", "Missing means structure was unavailable.", "general", "conditional", "safe_pre_outcome", "support/FAQ page function", "Question phrases alone do not prove Q&A structure.", "candidate_focal", "refactor_needed", "implemented_partial", "Retain question and answer primitives and validate multilingual pairs."),
        r("faq_schema_present", "core_general", "qa_and_schema", "primitive", "FAQPage structured data is present.", "1[FAQPage in JSON-LD/microdata/RDFa]", "html_structured_data", "schema.org payload", "binary", "low", "raw HTML structured data", "Missing means structured data was not captured.", "general", "high", "safe_pre_outcome", "CMS/plugin use; page type", "Schema can be stale or invisible to users.", "candidate_focal_or_sensitivity", "refactor_needed", "not_implemented", "Validate parsed schema types against raw HTML."),
        r("opening_summary_signal", "core_general", "opening_structure", "composite", "Opening content provides a summary or direct orientation.", "documented score from first content block primitives", "page_text_or_blocks", "opening content blocks", "binary_or_score", "high", "reliable main-content ordering", "Missing means opening content was unavailable.", "general", "conditional", "safe_pre_outcome", "page type; language", "Keyword dictionaries can become topic-specific.", "sensitivity", "refactor_needed", "implemented_partial", "Freeze cross-industry multilingual rules and retain components."),
        r("number_token_count", "core_general", "numeric_detail", "primitive", "Count of numeric tokens in measurable content.", "count(valid numeric tokens)", "page_text", "page text", "count", "medium", "language-aware number parsing", "Missing means text was unavailable.", "general", "high", "safe_pre_outcome", "length; page type", "IDs, dates and navigation numbers inflate counts.", "diagnostic_primitive", "core_keep", "implemented", "Validate token classes and use with a preselected density/count form."),
        r("number_token_density", "core_general", "numeric_detail", "derived_primitive", "Numeric tokens normalized by measurable words.", "1000 * number_token_count / word_count", "page_text", "number_token_count and word_count", "density_per_1000_words", "medium", "measurable content and number parser", "Missing means numerator or denominator was unavailable.", "general", "high", "safe_pre_outcome", "page type; language", "Unstable on very short pages.", "candidate_focal", "core_keep", "implemented", "Set minimum word support; do not pair automatically with raw count."),
        r("percentage_count", "core_general", "numeric_detail", "primitive", "Count of percentage expressions.", "count(percent patterns)", "page_text", "page text", "count_or_density", "medium", "multilingual regex/parser", "Missing means text was unavailable.", "general", "high", "safe_pre_outcome", "topic; page type", "Locale-specific formats may be missed.", "candidate_component", "core_keep", "implemented", "Validate Thai/English and decimal formats."),
        r("measurement_count", "core_general", "numeric_detail", "primitive", "Count of number-plus-unit measurements using a general unit registry.", "count(general measurement patterns)", "page_text", "page text and general unit registry", "count_or_density", "high", "versioned cross-industry unit parser", "Missing means text was unavailable.", "general", "conditional", "safe_pre_outcome", "industry mix; language", "A unit dictionary can silently become vertical-specific.", "candidate_component", "refactor_needed", "implemented_partial", "Separate universal units from vertical extension registries."),
        r("year_count", "core_general", "numeric_detail", "primitive", "Count of plausible year expressions.", "count(year patterns in approved range)", "page_text", "page text", "count_or_binary", "low", "measurable text", "Missing means text was unavailable.", "general", "high", "safe_pre_outcome", "news/document page function", "Model numbers can resemble years.", "candidate_component", "core_keep", "implemented", "Validate plausible-year range before freezing."),
        r("range_pattern_count", "core_general", "numeric_detail", "primitive", "Count of explicit numeric ranges.", "count(number-range-number patterns)", "page_text", "page text", "count_or_density", "medium", "multilingual range parser", "Missing means text was unavailable.", "general", "high", "safe_pre_outcome", "topic; page type", "Hyphens and identifiers produce false positives.", "candidate_component", "core_keep", "implemented", "Validate punctuation and Thai range words."),
        r("external_link_count", "core_general", "external_evidence", "primitive", "Count of content links to external root domains.", "count(content href root_domain != source_root_domain)", "html_dom", "resolved href destinations", "count_or_log2", "low", "reliable HTML links and DOM context", "Missing means link destinations were unavailable, not zero.", "general", "high", "safe_pre_outcome", "domain template; page function", "Navigation, tracking and redirects distort destinations.", "sensitivity", "needs_extraction_fix", "implemented_unreliable", "Preserve resolved hrefs and exclude navigation/footer links."),
        r("unique_external_domain_count", "core_general", "external_evidence", "primitive", "Unique external root domains linked from content.", "nunique(external resolved root domains)", "html_dom", "resolved external hrefs", "count_or_log2", "low", "reliable HTML links", "Missing means link destinations were unavailable.", "general", "high", "safe_pre_outcome", "page function; template", "Affiliate and tracking redirects inflate domain variety.", "sensitivity", "needs_extraction_fix", "partial", "Canonicalize redirects and validate content-only links."),
        r("official_domain_link_count", "core_general", "external_evidence", "primitive", "Links to verified official, government, standards or institutional domains.", "count(external links matching versioned official registry/rule)", "html_dom", "resolved hrefs and domain-role registry", "count_or_binary", "medium", "reliable links and versioned domain roles", "Missing means links or roles were unavailable.", "general", "conditional", "safe_pre_outcome", "source authority; topic", "Official-domain classification is imperfect and jurisdiction-specific.", "sensitivity", "needs_extraction_fix", "not_implemented", "Document domain-role rules and manually validate across countries."),
        r("reference_language_count", "core_general", "external_evidence", "primitive", "Count of visible citation/reference phrases.", "count(versioned multilingual reference phrases)", "page_text", "page text and phrase registry", "count_or_density", "high", "multilingual versioned phrase rules", "Missing means text was unavailable.", "general", "conditional", "safe_pre_outcome", "genre; language", "Phrases may not correspond to actual external evidence.", "diagnostic_or_sensitivity", "needs_extraction_fix", "implemented_partial", "Validate phrase-to-link agreement; never label as verified evidence alone."),
        r("prompt_url_similarity", "core_general", "prompt_page_relevance", "derived_primitive", "Leakage-safe similarity between prompt and decoded URL tokens.", "predeclared lexical similarity(prompt, URL tokens)", "prompt_and_url", "prompt text and normalized URL", "bounded_similarity", "high", "multilingual tokenization", "Missing means prompt or URL tokens were unavailable.", "general", "high", "safe_no_answer", "surfaced-source selection; topic", "URL slugs can be sparse or transliterated.", "G2R_sensitivity", "sensitivity_only", "not_implemented", "Freeze tokenizer and similarity method without outcome tuning."),
        r("prompt_title_similarity", "core_general", "prompt_page_relevance", "derived_primitive", "Leakage-safe prompt-title similarity.", "predeclared similarity(prompt, page title)", "prompt_and_metadata", "prompt text and page title", "bounded_similarity", "high", "multilingual tokenizer or embedding", "Missing means title or prompt was unavailable.", "general", "high", "safe_no_answer", "selection into surfaced sample", "Language/model choice changes scores.", "G2R_sensitivity", "sensitivity_only", "implemented_partial", "Version similarity model and validate cross-language comparability."),
        r("prompt_meta_similarity", "core_general", "prompt_page_relevance", "derived_primitive", "Leakage-safe prompt-meta-description similarity.", "predeclared similarity(prompt, meta description)", "prompt_and_metadata", "prompt text and meta description", "bounded_similarity", "high", "multilingual tokenizer or embedding", "Missing means metadata was unavailable.", "general", "high", "safe_no_answer", "selection; page type", "Metadata can be templated, stale or absent.", "G2R_sensitivity", "sensitivity_only", "not_implemented", "Version method and report metadata coverage."),
        r("prompt_body_similarity", "core_general", "prompt_page_relevance", "derived_primitive", "Leakage-safe prompt-page-body similarity.", "predeclared similarity(prompt, measurable page body)", "prompt_and_page_text", "prompt text and page body/Markdown", "bounded_similarity", "high", "reliable measurable text and versioned similarity", "Missing means page body or prompt was unavailable.", "general", "high", "safe_no_answer", "surfaced-source selection; extraction scope", "Excerpt/full-text differences alter relevance.", "G2R_sensitivity", "sensitivity_only", "implemented_partial", "Compare excerpt and full-text scores; never use answer text."),
        r("general_keyword_overlap", "core_general", "prompt_page_relevance", "derived_primitive", "General prompt-page token overlap without vertical dictionaries.", "documented overlap(prompt tokens, page tokens)", "prompt_and_page_text", "prompt and page text", "bounded_overlap", "high", "multilingual tokenization and stopwords", "Missing means prompt/page tokens were unavailable.", "general", "high", "safe_no_answer", "selection; language", "Token overlap favors repeated wording over semantic relevance.", "G2R_sensitivity", "sensitivity_only", "implemented_partial", "Freeze stopwords and tokenizer before outcomes."),
        r("prompt_page_relevance_score", "core_general", "prompt_page_relevance", "composite", "Combined leakage-safe prompt-page relevance score.", "predeclared combination of URL/title/meta/body/overlap components", "prompt_and_page", "primitive relevance components", "bounded_score", "high", "all selected components", "Missing means required relevance components were unavailable.", "general", "conditional", "safe_no_answer", "surfaced-source selection", "Composite may be dominated by body similarity and extraction scope.", "G2R_sensitivity", "sensitivity_only", "implemented", "Document weights; retain components; test dominance; never model with all components automatically."),
        r("page_type_family_rule_v2_url_seed", "core_general", "taxonomy_control", "primitive_classification", "Broad page family inferred only from URL/title/meta/domain signals.", "versioned Rule-v2 metadata classifier", "url_title_meta", "URL, title, meta and domain", "collapsed_categorical", "medium", "metadata availability", "Unknown remains an explicit category.", "general", "conditional", "safe_pre_outcome", "page function; domain", "Rule errors and unknowns create measurement error.", "G4A_sensitivity", "sensitivity_only", "implemented_alias_needed", "Freeze taxonomy version and collapse levels using support only."),
        r("source_type_general_rule_v2", "core_general", "taxonomy_control", "primitive_classification", "General source/site role inferred without page body.", "versioned Rule-v2 source/site classifier", "url_title_meta", "URL, title, meta and domain", "collapsed_categorical", "medium", "metadata availability", "Unknown remains explicit.", "general", "conditional", "safe_pre_outcome", "domain authority; platform type", "Broad roles can combine heterogeneous publishers.", "G4A_sensitivity", "sensitivity_only", "implemented_alias_needed", "Freeze version and report category support."),
        r("page_type_family_gemini_v1_collapsed", "core_general", "taxonomy_control", "primitive_classification", "Collapsed Gemini page-function family.", "Gemini taxonomy v1; support-only rare collapse", "url_metadata_and_content", "versioned Gemini classification", "categorical", "medium", "taxonomy join and classification evidence", "Unknown remains explicit.", "general", "conditional", "content_informed_overcontrol_risk", "page function; source authority", "Uses content and may absorb focal writing variation.", "G4B_secondary_sensitivity", "sensitivity_only", "implemented", "Keep secondary to Rule-v2 in Core-General design; report taxonomy version."),
        r("source_type_general_gemini_v1_collapsed", "core_general", "taxonomy_control", "primitive_classification", "Collapsed Gemini source/site type.", "Gemini taxonomy v1; support-only rare collapse", "url_metadata_and_content", "versioned Gemini classification", "categorical", "medium", "taxonomy join and classification evidence", "Unknown remains explicit.", "general", "conditional", "content_informed_overcontrol_risk", "domain authority; platform type", "May use page content and over-control structural variation.", "G4B_secondary_sensitivity", "sensitivity_only", "implemented", "Report as content-informed sensitivity, not focal content."),
        r("content_feature_available", "core_general", "extraction_quality", "primitive", "Content features can be measured for the row/page.", "1[required extraction fields measurable]", "scrape_and_parse", "scrape/parse/extraction outputs", "binary", "low", "documented availability rule", "False means unavailable, not feature absence.", "general", "high", "safe_diagnostic", "platform; blocking; JavaScript; page type", "Availability is plausibly non-random.", "G6_selection_diagnostic", "diagnostic_only", "implemented", "Audit availability by cited status, domain, taxonomy and language."),
        r("content_strength", "core_general", "extraction_quality", "composite", "Extraction strength, not writing quality.", "versioned rule from text scope/length/quality signals", "scrape_and_parse", "content quality primitives", "categorical", "medium", "documented quality rules", "Missing means quality was not classified.", "general", "high", "safe_diagnostic", "platform; page type; domain technology", "Can reflect page accessibility rather than content quality.", "G5_covariate_and_restriction", "diagnostic_only", "implemented", "Run main model without mandatory control, then add covariate and strong-only sensitivity."),
        r("feature_extraction_text_scope", "core_general", "extraction_quality", "primitive", "Whether features use full text, excerpt, metadata only or no text.", "recorded extraction provenance", "scrape_and_parse", "text source provenance", "categorical", "low", "provenance retained", "Missing means provenance was not recorded.", "general", "high", "safe_diagnostic", "crawler and page technology", "Scope labels may not guarantee complete main content.", "G5B_G5C_diagnostic", "diagnostic_only", "implemented", "Manually verify full-text-equivalent classification."),
        r("language", "core_general", "language_quality", "primitive", "Detected or source-provided page language.", "versioned language detector plus metadata", "page_text_and_metadata", "page text and lang metadata", "categorical", "high", "language detection evidence", "Unknown means language could not be established.", "general", "high", "safe_diagnostic", "topic and market", "Mixed-language pages and short excerpts reduce accuracy.", "diagnostic_or_interaction_support", "diagnostic_only", "partial", "Validate Thai, English and mixed pages."),
        r("segmentation_method", "core_general", "language_quality", "primitive", "Sentence/token segmentation method used.", "recorded algorithm/model version", "feature_extraction", "segmenter metadata", "categorical", "high", "provenance recorded", "Missing means segmentation provenance was lost.", "general", "high", "safe_diagnostic", "language", "Different methods change structural counts.", "diagnostic", "diagnostic_only", "not_implemented", "Record exact method and version for every row."),
        r("segmentation_confidence", "core_general", "language_quality", "primitive", "Confidence or quality class for segmentation.", "predefined validation-based confidence rule", "feature_extraction", "language and segmentation diagnostics", "categorical", "high", "validated segmenter", "Missing means confidence was not assessed.", "general", "high", "safe_diagnostic", "language and extraction scope", "Uncalibrated confidence can mislead.", "diagnostic_or_restriction", "diagnostic_only", "not_implemented", "Calibrate against manually segmented Thai/English samples."),
        r("audit_scrape_gap_days", "core_general", "freshness", "derived_primitive", "Days between citation audit and content scrape.", "date(scrape_timestamp) - date(audit_timestamp)", "audit_and_scrape_metadata", "audit_timestamp and scrape_timestamp", "continuous_or_bins", "low", "timestamps retained", "Missing means one timestamp was unavailable.", "general", "high", "safe_diagnostic", "crawl scheduling", "Does not measure actual page-change timing.", "diagnostic_or_sensitivity", "diagnostic_only", "not_implemented", "Validate timestamp timezone and sign."),
        r("freshness_days", "core_general", "freshness", "derived_primitive", "Age of page publication/update at audit time.", "audit_timestamp - best validated published/modified date", "html_metadata_or_schema", "audit timestamp and extracted date", "continuous_or_bins", "medium", "reliable date extraction", "Missing means no reliable page date was available.", "general", "conditional", "safe_pre_outcome", "page type; publisher", "Dates may refer to templates, events or updates.", "diagnostic_or_sensitivity", "refactor_needed", "not_implemented", "Add date_extraction_quality and manual validation."),
        r("template_hash", "core_general", "template_duplication", "derived_primitive", "Hash of normalized DOM structure after content/text removal.", "hash(normalized DOM tag/class skeleton)", "html_dom", "raw HTML DOM", "categorical_hash", "low", "stable DOM normalization", "Missing means DOM was unavailable.", "general", "high", "safe_diagnostic", "CMS and platform", "Minor experiments or dynamic markup split templates.", "diagnostic_cluster", "diagnostic_only", "not_implemented", "Check within-domain stability and duplicate detection."),
        r("dom_structure_cluster", "core_general", "template_duplication", "composite", "Cluster of similar normalized DOM structures.", "predeclared clustering of layout fingerprints", "html_dom", "layout fingerprint", "categorical_cluster", "low", "sufficient HTML coverage", "Missing means clustering input was unavailable.", "general", "high", "safe_diagnostic", "platform and CMS", "Clustering choices can be outcome-tuned if not frozen.", "optional_template_sensitivity", "diagnostic_only", "not_implemented", "Freeze distance and clustering parameters before outcomes."),
        r("price_detail_score", "commerce_general", "commerce_detail", "composite", "General price/currency/range/detail signal for commercial pages.", "documented combination of price primitives only", "page_text_or_tables", "currency, amount, range and price-label primitives", "bounded_score", "high", "multilingual currency/price parser", "Missing means content was unavailable; zero means measured absence.", "commerce", "conditional", "safe_pre_outcome", "commercial page type; market", "Currency patterns and subscription prices vary by industry.", "commerce_extension_focal", "commerce_general_keep", "refactor_needed", "Split from unit terms; retain every price primitive and test dominance."),
        r("availability_signal", "commerce_general", "commerce_detail", "composite", "Visible stock, booking, service or enrollment availability.", "versioned cross-industry availability primitives", "page_text_or_schema", "text and structured data", "categorical_or_binary", "high", "commerce schema/text parsing", "Missing means availability could not be measured.", "commerce", "conditional", "safe_pre_outcome", "page type; current inventory", "Meaning differs for products, services and appointments.", "commerce_extension", "commerce_general_keep", "not_implemented", "Validate separately by commerce subtype."),
        r("purchase_or_contact_signal", "commerce_general", "commerce_action", "composite", "Visible action to buy, book, request, contact or start service.", "versioned CTA primitives from content controls/text", "html_dom_or_text", "buttons, forms, links and visible text", "categorical_or_binary", "high", "DOM context and multilingual CTA rules", "Missing means action structure was unavailable.", "commerce", "conditional", "safe_pre_outcome", "page function; template", "Navigation CTAs and persistent headers create false positives.", "commerce_extension", "commerce_general_keep", "not_implemented", "Separate content CTA from global navigation."),
        r("rating_review_signal", "commerce_general", "commerce_trust", "composite", "Visible rating value, rating count or review structure.", "documented components from schema and visible content", "html_schema_and_text", "ratings/reviews primitives", "categorical_or_score", "medium", "schema and visible-content validation", "Missing means review evidence was unavailable.", "commerce", "conditional", "safe_pre_outcome", "platform and product type", "Self-published ratings may not be independent evidence.", "commerce_extension", "commerce_general_keep", "not_implemented", "Preserve rating value/count/source primitives."),
        r("product_comparison_signal", "commerce_general", "commerce_comparison", "composite", "Commercial comparison of products, plans or providers.", "documented table/text comparison primitives", "html_dom_or_text", "comparison primitives", "binary_or_score", "high", "reliable tables and multilingual rules", "Missing means comparison evidence was unavailable.", "commerce", "conditional", "safe_pre_outcome", "page type; transactional intent", "Editorial comparisons may overlap Core-General table structure.", "commerce_extension", "commerce_general_keep", "not_implemented", "Validate commerce versus informational comparisons."),
        r("vertical_specific_unit_detail_score", "vertical_specific", "real_estate_units", "composite", "Real-estate unit-size and layout detail formerly mixed with price.", "versioned real-estate unit dictionary", "page_text_or_tables", "sqm, bedroom, floor-plan and unit primitives", "bounded_score", "high", "vertical parser", "Missing means vertical detail was unavailable.", "real_estate", "low_outside_vertical", "safe_but_topic_specific", "property page type", "No cross-industry meaning.", "vertical_extension_only", "pause_vertical_specific", "implemented_in_old_score", "Split from price detail and preserve under real-estate extension."),
        r("location_transit_specificity_score", "vertical_specific", "real_estate_location", "composite", "Transit and location specificity for real estate.", "existing real-estate rule", "page_text", "location/transit dictionary", "bounded_score", "high", "vertical parser", "Missing means feature was unavailable.", "real_estate", "low_outside_vertical", "safe_but_topic_specific", "prompt geography; property page type", "Area and transit dictionaries are market-specific.", "vertical_extension_only", "pause_vertical_specific", "implemented", "Freeze existing condo implementation; exclude from Core-General."),
        r("amenity_project_detail_score", "vertical_specific", "real_estate_project", "composite", "Amenities and project/developer specificity.", "existing real-estate rule", "page_text", "amenity/project dictionary", "bounded_score", "high", "vertical parser", "Missing means feature was unavailable.", "real_estate", "low_outside_vertical", "safe_but_topic_specific", "listing/project page function", "Confounds product category with detail.", "vertical_extension_only", "pause_vertical_specific", "implemented", "Freeze existing condo implementation; exclude from Core-General."),
        r("prompt_area_keyword_match", "vertical_specific", "real_estate_location", "composite", "Prompt-page overlap on predefined condo area names.", "existing area dictionary match", "prompt_and_page_text", "area-name registry", "binary", "high", "vertical dictionary", "Missing means prompt/page was unavailable.", "real_estate_local", "none_outside_vertical", "safe_but_topic_specific", "prompt geography", "Dictionary is SCOPE/area specific.", "vertical_extension_only", "pause_vertical_specific", "implemented", "Exclude from Core-General and preserve in real-estate extension."),
        r("answer_similarity", "excluded", "leakage", "derived", "Any page-answer similarity or overlap measure.", "forbidden", "answer_text", "answer text and page", "none", "high", "not_allowed", "Not applicable.", "outcome_derived", "none", "forbidden_answer_derived", "post-outcome information", "Direct leakage into citation analysis.", "excluded", "exclude_leakage", "legacy_possible", "Formula guardrail must reject all aliases."),
        r("source_position_or_observed_rank", "excluded", "leakage", "primitive", "Source position or observed rank in the surfaced output.", "forbidden in headline content model", "post_surface_output", "position/rank", "none", "low", "not_allowed_in_main", "Not applicable.", "outcome_process", "none", "forbidden_main_model", "ranking process", "Post-selection and potentially post-answer information.", "diagnostic_only_if_separately_approved", "exclude_leakage", "legacy_available", "Formula guardrail must reject main-model use."),
        r("domain_citation_rate", "excluded", "leakage", "derived", "Domain-level citation-rate proxy.", "forbidden", "citation_outcome", "cited labels", "none", "low", "not_allowed", "Not applicable.", "outcome_derived", "none", "forbidden_outcome_derived", "domain identity", "Directly encodes the outcome.", "excluded", "exclude_leakage", "legacy_possible", "Reject raw, leave-one-out and smoothed aliases."),
    ]
    frame = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    validate_core_general_feature_registry(frame)
    return frame


def validate_core_general_feature_registry(frame: pd.DataFrame) -> None:
    """Raise when the registry violates its schema or frozen guardrails."""
    missing = set(REGISTRY_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature registry missing columns: {', '.join(sorted(missing))}")
    if frame["feature_name"].duplicated().any():
        raise ValueError("Feature registry contains duplicate feature names.")
    invalid = sorted(set(frame["feature_status"]) - ALLOWED_STATUSES)
    if invalid:
        raise ValueError(f"Invalid feature_status values: {', '.join(invalid)}")
    invalid_implementation = sorted(
        set(frame["current_implementation_status"]) - ALLOWED_IMPLEMENTATION_STATUSES
    )
    if invalid_implementation:
        raise ValueError(
            "Invalid current_implementation_status values: "
            + ", ".join(invalid_implementation)
        )
    invalid_qa = sorted(set(frame["qa_status"]) - ALLOWED_QA_STATUSES)
    if invalid_qa:
        raise ValueError(f"Invalid qa_status values: {', '.join(invalid_qa)}")
    if not frame["registry_record_type"].isin({"canonical", "deprecated_alias"}).all():
        raise ValueError("registry_record_type must be canonical or deprecated_alias.")
    invalid_granularity = sorted(set(frame["feature_granularity"]) - ALLOWED_GRANULARITIES)
    if invalid_granularity:
        raise ValueError(f"Invalid feature_granularity values: {', '.join(invalid_granularity)}")
    canonical = frame[frame["registry_record_type"].eq("canonical")]
    duplicates = canonical["canonical_column_name"].duplicated(keep=False)
    if duplicates.any():
        names = sorted(canonical.loc[duplicates, "canonical_column_name"].unique())
        raise ValueError(f"Canonical output columns are duplicated: {', '.join(names)}")
    aliases = frame[frame["registry_record_type"].eq("deprecated_alias")]
    canonical_names = set(canonical["feature_name"])
    bad_replacements = sorted(set(aliases["replacement_feature_name"]) - canonical_names)
    if bad_replacements:
        raise ValueError(
            "Deprecated aliases reference missing canonical features: "
            + ", ".join(bad_replacements)
        )
    table_level = canonical[canonical["feature_granularity"].eq("table_level")]
    undocumented = table_level["page_aggregation_rule"].astype(str).str.strip().eq("")
    if undocumented.any():
        names = sorted(table_level.loc[undocumented, "feature_name"])
        raise ValueError(
            "Table-level features lack page_aggregation_rule: " + ", ".join(names)
        )
    approved = frame["approved_for_model_v1"].astype(str).str.casefold()
    if not approved.isin({"true", "false"}).all():
        raise ValueError("approved_for_model_v1 must contain only true/false values.")
    forbidden = frame["feature_status"].eq("exclude_leakage")
    if not frame.loc[forbidden, "recommended_model_role"].isin(
        {"excluded", "diagnostic_only_if_separately_approved"}
    ).all():
        raise ValueError("Leakage exclusions cannot have an econometric predictor role.")


def build_core_general_feature_registry(
    path: Path | str = CANONICAL_REGISTRY_PATH,
) -> pd.DataFrame:
    """Load and validate the canonical CSV source of truth."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Canonical Core-General feature registry not found: {source}")
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    validate_core_general_feature_registry(frame)
    return frame


def write_core_general_feature_registry(
    path: Path | str,
    frame: pd.DataFrame | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    registry = build_core_general_feature_registry() if frame is None else frame.copy()
    validate_core_general_feature_registry(registry)
    registry.to_csv(output, index=False)
    return output
