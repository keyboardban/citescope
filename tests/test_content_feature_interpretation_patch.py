import numpy as np
import pandas as pd

from src.econometrics_eda_v2.content_feature_interpretation_patch import (
    build_domain_fe_attenuation,
    build_focal_term_se_comparison,
    build_se_stability_summary,
    patch_minimum_reporting_table,
)


def _coefficient_rows(model_id: str, estimate: float, term: str = "has_table") -> list[dict]:
    rows = []
    for covariance, standard_error in (
        ("HC3", 0.02),
        ("cluster_prompt_id", 0.025),
        ("cluster_normalized_url", 0.03),
        ("two_way_cluster_prompt_url", 0.035),
    ):
        rows.append(
            {
                "model_id": model_id,
                "formula": "cited ~ has_table + C(prompt_id)",
                "term": term,
                "estimate": estimate,
                "estimate_pp": estimate * 100,
                "std_error": standard_error,
                "conf_low_pp": (estimate - 1.96 * standard_error) * 100,
                "conf_high_pp": (estimate + 1.96 * standard_error) * 100,
                "p_value": 0.1,
                "n_obs": 100,
                "n_prompts": 10,
                "n_urls": 50,
                "n_domains": 8,
                "r_squared": 0.2,
                "cov_type": covariance,
                "notes": "",
            }
        )
    return rows


def test_se_comparison_flags_unavailable_focal_two_way_se():
    frame = pd.DataFrame(_coefficient_rows("M2", 0.03))
    mask = frame["cov_type"].eq("two_way_cluster_prompt_url")
    frame.loc[mask, ["std_error", "conf_low_pp", "conf_high_pp", "p_value"]] = np.nan
    comparison = build_focal_term_se_comparison(frame)
    two_way = comparison[comparison["cov_type"].eq("two_way_cluster_prompt_url")].iloc[0]
    assert not bool(two_way.se_available)
    assert two_way.focal_term_warning == "focal_term_se_unavailable"
    summary = build_se_stability_summary(comparison)
    assert summary.iloc[0].classification == "unavailable_or_unreliable_se"


def test_domain_attenuation_flags_large_heading_reduction():
    term = "C(heading_count_group, Treatment(reference='0-1'))[T.13+]"
    frame = pd.DataFrame(_coefficient_rows("M2", -0.20, term) + _coefficient_rows("M3_domain_fe", -0.02, term))
    attenuation = build_domain_fe_attenuation(frame).iloc[0]
    assert attenuation.percent_attenuation_if_defined == 90
    assert bool(attenuation.domain_fe_attenuation_flag)
    assert "domain/template confounding" in attenuation.interpretation


def test_minimum_table_receives_requested_interpretation_bucket():
    original = pd.DataFrame([{"feature": "Has table", "term": "has_table"}])
    classification = pd.DataFrame(
        [
            {
                "feature": "Has table",
                "classification": "suggestive",
                "domain_fe_attenuation": True,
                "outlier_sensitive": False,
                "se_sensitive": True,
                "recommended_wording": "Suggestive positive association.",
            }
        ]
    )
    patched = patch_minimum_reporting_table(original, classification).iloc[0]
    assert patched.final_interpretation_bucket == "suggestive_positive"
    assert patched.robustness_classification == "suggestive"
