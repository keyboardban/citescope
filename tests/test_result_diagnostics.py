import pandas as pd

from src.econometrics_eda_v2.result_diagnostics import (
    PREFERRED_COVARIANCE,
    _multiple_testing,
    _variation_block,
)


def test_variation_block_counts_only_groups_with_within_group_change():
    data = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2", "p3"],
            "normalized_url": ["u1", "u2", "u3", "u4", "u5"],
            "feature": [0, 1, 1, 1, pd.NA],
        }
    )

    result = _variation_block(data, "feature", "prompt_id")

    assert result["total_groups"] == 3
    assert result["groups_with_usable_variation"] == 1
    assert result["rows_in_varying_groups"] == 2
    assert result["unique_urls_in_varying_groups"] == 2


def test_multiple_testing_keeps_headline_and_exploratory_families_separate():
    focal = pd.DataFrame(
        {
            "analysis_layer": ["FE2"] * 4 + ["FE3"] * 4,
            "cov_type": [PREFERRED_COVARIANCE] * 8,
            "term": [f"core_{index}" for index in range(4)] * 2,
            "p_value": [0.001, 0.02, 0.20, 0.80] * 2,
        }
    )
    components = pd.DataFrame(
        {
            "cov_type": [PREFERRED_COVARIANCE] * 5,
            "term": [f"component_{index}" for index in range(5)],
            "p_value": [0.001, 0.01, 0.05, 0.10, 0.90],
        }
    )

    result = _multiple_testing(focal, components)

    sizes = result.groupby("hypothesis_family")["family_size"].first().to_dict()
    assert sizes == {
        "headline_FE2_four_core_features": 4,
        "exploratory_FE1_five_writing_components": 5,
    }
    assert set(result["correction_method"]) == {"Benjamini-Hochberg FDR"}
    assert result.loc[result["term"].str.startswith("core_"), "term"].is_unique
