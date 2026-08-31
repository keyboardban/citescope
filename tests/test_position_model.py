import numpy as np
import pandas as pd

from src.econometrics_eda_v2.position_model import (
    _collapse_page_type_for_model,
    _collapse_source_type_for_model,
    _domain_source_type_assignments,
    _gemini_url_features,
    _placement,
    _sample_zscore,
    external_source_eligibility,
    model_formulas,
    multicollinearity_diagnostics,
    select_main_se_method,
)


def test_placement_keeps_absence_distinct_from_position_zero():
    category, status = _placement(
        pd.Series([0, 1, 1, 1]),
        pd.Series([np.nan, 0.0, 0.75, np.nan]),
        pd.Series(["success_absent"] * 4),
        no_label="no_feature",
        early_label="early",
        late_label="late",
    )

    assert category.tolist()[:3] == ["no_feature", "early", "late"]
    assert pd.isna(category.iloc[3])
    assert status.tolist() == ["success_absent", "success_present", "success_present", "ambiguous"]


def test_primary_formula_uses_only_position_features_and_allowed_controls():
    formula = model_formulas()["M5"]

    assert "direct_answer_placement" in formula
    assert "table_placement" in formula
    assert "question_heading_placement" in formula
    assert "z_numeric_evidence_total_density" in formula
    assert "numeric_evidence_early_density" not in formula
    assert "numeric_evidence_early_share" not in formula
    assert "C(prompt_id)" in formula
    assert "C(page_type_model_6" in formula
    assert "reference='blog_guide_or_editorial'" in formula
    assert "C(source_type_model_6" in formula
    assert "writing_structure" not in formula
    assert "factual_numeric_density_score" not in formula
    assert "has_verified_html_table" not in formula
    assert "position_ratio" not in formula


def test_approved_page_and_source_collapses_have_six_classes():
    page_types = pd.Series([
        "informational_content", "news_or_press", "directory_or_listing",
        "commercial_product_or_service", "comparison_or_review",
        "landing_or_brand_page", "contact_or_location", "support_or_help",
        "document_or_media", "social_or_user_generated", "search_or_results",
        "unknown", "rare_other",
    ])
    source_types = pd.Series([
        "official_company_or_brand", "marketplace_or_platform",
        "directory_or_listing_platform", "blog_or_content_site", "news_media",
        "review_platform", "social_or_forum", "government", "rare_other", "unknown",
    ])

    assert set(_collapse_page_type_for_model(page_types)) == {
        "blog_guide_or_editorial", "directory_or_listing",
        "commercial_product_or_service", "comparison_or_review",
        "landing_contact_or_support", "other_page_function",
    }
    assert set(_collapse_source_type_for_model(source_types)) == {
        "official_company_or_brand", "marketplace_or_directory_platform",
        "blog_or_news_publisher", "review_or_community_platform",
        "government_or_public_institution", "other_or_unknown",
    }


def test_source_type_domain_consensus_uses_unique_urls_and_sends_ties_to_other():
    rows = pd.DataFrame({
        "source_root_domain": ["majority.test"] * 3 + ["tie.test"] * 2,
        "normalized_url": ["a", "b", "c", "d", "e"],
        "source_type_row_collapsed": [
            "blog_or_news_publisher", "blog_or_news_publisher",
            "official_company_or_brand", "official_company_or_brand",
            "marketplace_or_directory_platform",
        ],
    })

    result = _domain_source_type_assignments(rows).set_index("source_root_domain")

    assert result.loc["majority.test", "source_type_model_6"] == "blog_or_news_publisher"
    assert np.isclose(result.loc["majority.test", "dominant_url_share"], 2 / 3)
    assert result.loc["tie.test", "source_type_model_6"] == "other_or_unknown"
    assert bool(result.loc["tie.test", "top_class_tie"])


def test_total_numeric_density_and_early_share_are_distinct():
    pages = pd.DataFrame(
        {
            "normalized_url": ["https://example.com/with", "https://example.com/without"],
            "gemini_status": ["success", "success"],
            "block_extraction_status": ["measured", "measured"],
            "total_main_content_tokens": [100, 200],
        }
    )
    evidence = pd.DataFrame(
        [
            {"normalized_url": "https://example.com/with", "feature": "direct_answer", "tag": "p", "block_id": "d1", "position_ratio": .1, "start_token": 10, "evidence_text": "answer"},
            {"normalized_url": "https://example.com/with", "feature": "question_heading", "tag": "h2", "block_id": "q1", "position_ratio": .2, "start_token": 20, "evidence_text": "question"},
            {"normalized_url": "https://example.com/with", "feature": "numeric_evidence", "tag": "p", "block_id": "n1", "position_ratio": .25, "start_token": 25, "evidence_text": "10 percent"},
            {"normalized_url": "https://example.com/with", "feature": "numeric_evidence", "tag": "p", "block_id": "n2", "position_ratio": .75, "start_token": 75, "evidence_text": "20 percent"},
        ]
    )

    result = _gemini_url_features(pages, evidence).set_index("normalized_url")

    assert result.loc["https://example.com/with", "numeric_evidence_total_density"] == 20
    assert result.loc["https://example.com/with", "numeric_evidence_early_share"] == .5
    assert result.loc["https://example.com/without", "numeric_evidence_total_density"] == 0
    assert pd.isna(result.loc["https://example.com/without", "numeric_evidence_early_share"])


def test_total_numeric_density_zscore_uses_sample_standard_deviation():
    standardized = _sample_zscore(pd.Series([1.0, 2.0, 4.0, np.nan]))

    assert np.isclose(standardized.mean(), 0)
    assert np.isclose(standardized.std(ddof=1), 1)
    assert pd.isna(standardized.iloc[-1])


def test_two_way_cluster_selected_only_with_adequate_support():
    adequate = pd.DataFrame(
        {
            "dimension": ["domain", "prompt"],
            "n_clusters": [40, 50],
            "singleton_observation_share": [.10, .05],
        }
    )
    weak = adequate.copy()
    weak.loc[weak["dimension"].eq("prompt"), "n_clusters"] = 10

    assert select_main_se_method(adequate)[0] == "two_way_cluster_domain_prompt"
    assert select_main_se_method(weak)[0] == "cluster_domain"


def test_external_source_requires_formal_manual_validation():
    coverage = pd.DataFrame(
        {
            "feature": ["external_source_placement"] * 3,
            "n_rows": [100, 100, 100],
            "category_share": [.34, .33, .33],
        }
    )
    concentration = pd.DataFrame(
        {
            "feature": ["external_source_placement"] * 3,
            "top_group_share": [.10, .15, .20],
        }
    )

    eligible, reason = external_source_eligibility(coverage, concentration)

    assert not eligible
    assert "formal_manual_validation_available=fail" in reason


def test_multicollinearity_diagnostics_retains_every_predictor_pair():
    rng = np.random.default_rng(42)
    n_rows = 240
    data = pd.DataFrame(
        {
            "direct_answer_placement": rng.choice(
                ["no_direct_answer", "direct_answer_early", "direct_answer_late"],
                n_rows,
            ),
            "table_placement": rng.choice(
                ["no_table", "table_early", "table_late"], n_rows
            ),
            "question_heading_placement": rng.choice(
                [
                    "no_question_heading",
                    "question_heading_early",
                    "question_heading_late",
                ],
                n_rows,
            ),
            "z_numeric_evidence_total_density": rng.normal(size=n_rows),
            "log_word_count": rng.normal(size=n_rows),
            "page_type_model_6": rng.choice(
                [
                    "blog_guide_or_editorial",
                    "directory_or_listing",
                    "commercial_product_or_service",
                    "comparison_or_review",
                    "landing_contact_or_support",
                    "other_page_function",
                ],
                n_rows,
            ),
            "source_type_model_6": rng.choice(
                [
                    "official_company_or_brand",
                    "marketplace_or_directory_platform",
                    "blog_or_news_publisher",
                    "review_or_community_platform",
                    "government_or_public_institution",
                    "other_or_unknown",
                ],
                n_rows,
            ),
        }
    )

    diagnostics = multicollinearity_diagnostics(data)
    vif_rows = diagnostics[diagnostics.row_type.eq("vif")]
    pair_rows = diagnostics[
        diagnostics.row_type.eq("pairwise_predictor_association")
    ]

    assert len(vif_rows) == 18
    assert len(pair_rows) == 18 * 17 // 2
    assert pair_rows.association.notna().all()
    assert (pair_rows.warning == "below_review_threshold").any()
