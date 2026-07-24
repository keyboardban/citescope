import pandas as pd

from src.econometrics_eda_v2.writing_factual_density_econometrics import (
    build_leakage_scope_guardrail,
    classify_has_table_path,
    compare_covariance_types,
    formula_guardrail_matches,
)


def test_formula_guardrail_blocks_rank_answer_and_enriched_page_type():
    formula = "cited ~ page_answer_similarity + source_position + page_type_family_general"
    assert formula_guardrail_matches(formula) == [
        "page_answer_similarity",
        "page_type_family_general",
        "source_position",
    ]
    assert formula_guardrail_matches(
        "cited ~ prompt_page_relevance_score + C(page_type_url_seed_general_collapsed)"
    ) == []
    assert formula_guardrail_matches(
        "cited ~ prompt_page_relevance_score + C(page_type_family_gemini_v1_collapsed) "
        "+ C(source_type_general_gemini_v1_collapsed)"
    ) == []


def test_has_table_path_separates_coefficient_pattern_from_precision():
    path = pd.DataFrame(
        {
            "path_order": [0, 1, 2],
            "estimate_pp": [2.0, 3.0, 5.0],
            "conf_low_pp": [-1.0, -0.5, 0.1],
            "conf_high_pp": [5.0, 6.5, 9.9],
        }
    )
    assert classify_has_table_path(path) == ("amplified", "imprecise")


def test_covariance_comparison_flags_unreliable_two_way():
    table = pd.DataFrame(
        {
            "model_id": ["W3", "W3"],
            "formula": ["cited ~ has_table", "cited ~ has_table"],
            "term": ["has_table", "has_table"],
            "estimate": [0.03, 0.03],
            "estimate_pp": [3.0, 3.0],
            "std_error": [0.01, float("nan")],
            "conf_low_pp": [1.0, float("nan")],
            "conf_high_pp": [5.0, float("nan")],
            "p_value": [0.01, float("nan")],
            "n_obs": [100, 100],
            "n_prompts": [10, 10],
            "n_urls": [40, 40],
            "n_domains": [5, 5],
            "cov_type": ["HC3", "two_way_cluster_prompt_url"],
            "notes": ["", "negative diagonal"],
            "warning": ["", "negative diagonal"],
            "model_status": ["completed", "completed"],
        }
    )
    result = compare_covariance_types(table)
    assert set(result["se_robustness_classification"]) == {"unreliable_two_way"}


def test_scope_guardrail_uses_prior_notebook10_check(tmp_path):
    prior = tmp_path / "tables/10_writing_factual_density_features"
    prior.mkdir(parents=True)
    pd.DataFrame([{"status": "pass"}]).to_csv(
        prior / "writing_factual_feature_leakage_check.csv",
        index=False,
    )
    data = pd.DataFrame({"feature_extraction_text_scope": ["excerpt_only"]})
    formulas = {
        "W3_structural_plus_writing_factual": (
            "cited ~ factual_numeric_density_score + prompt_page_relevance_score + C(prompt_id)"
        )
    }
    result = build_leakage_scope_guardrail(data, formulas, tmp_path)
    assert result["status"].eq("pass").all()
