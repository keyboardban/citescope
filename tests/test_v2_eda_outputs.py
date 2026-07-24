from __future__ import annotations

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.diagnostics import write_eda_outputs
from src.econometrics_eda_v2.leakage import safe_predictor_columns


def _eda_df(n=160):
    rows = []
    for i in range(n):
        cited = int(i % 3 == 0)
        rows.append(
            {
                "run_id": "run",
                "answer_id": f"a{i//4}",
                "prompt_id": f"p{i//4}",
                "record_id": f"r{i//4}",
                "source_row_id": f"sr{i}",
                "normalized_url": f"https://example{i%8}.com/path/{i}",
                "source_domain": f"example{i%8}.com",
                "cited": cited,
                "intent": "info" if i % 2 else "compare",
                "intent_plot_label": "info" if i % 2 else "compare",
                "topic": "health",
                "language": "en",
                "country": "TH",
                "source_title": "Title",
                "source_description": "Description",
                "source_snippet": "Snippet",
                "source_url": f"https://example{i%8}.com/path/{i}",
                "source_position": (i % 10) + 1,
                "log1p_source_position": np.log1p((i % 10) + 1),
                "scrape_success": True,
                "parse_success": True,
                "scraped_body_available": True,
                "content_feature_available": True,
                "content_feature_missing_reason": "",
                "word_count": 200 + i * 5,
                "heading_count": i % 6,
                "table_count": i % 2,
                "link_count": 3 + i % 7,
                "has_faq": i % 2,
                "has_price_or_package": i % 4 == 0,
                "has_contact_info": i % 3 == 0,
                "has_table": i % 2,
                "has_bullets": i % 5 == 0,
                "has_author": i % 7 == 0,
                "has_reviewer": i % 9 == 0,
                "has_schema": i % 6 == 0,
                "has_phone_number": i % 3 == 0,
                "has_email": i % 4 == 0,
                "has_address": i % 5 == 0,
                "has_opening_hours": i % 6 == 0,
                "has_booking_or_appointment": i % 7 == 0,
                "has_step_by_step": i % 8 == 0,
                "has_medical_disclaimer": i % 9 == 0,
                "has_references": i % 4 == 0,
                "has_updated_date": i % 5 == 0,
                "page_type": "article" if i % 2 else "price_package_page",
                "page_type_confidence": "high" if i % 2 else "medium",
                "title_prompt_similarity": i / n,
                "description_prompt_similarity": (n - i) / n,
                "page_prompt_similarity": (i % 10) / 10,
                "max_chunk_prompt_similarity": (i % 8) / 8,
                "relevance_score_prompt_only": max(i / n, (i % 10) / 10),
                "domain_seen_count": 5,
                "domain_seen_count_loo": 4,
                "log1p_domain_seen_count": np.log1p(5),
                "url_length": 30 + i,
                "url_path_depth": 2,
                "https_flag": 1,
                "url_has_query_params": i % 2,
            }
        )
    return pd.DataFrame(rows)


def test_leakage_guard_excludes_answer_features_and_position():
    cols = safe_predictor_columns(["has_faq", "page_answer_similarity", "source_position", "answer_text"])
    assert cols == ["has_faq"]


def test_eda_output_contract(tmp_path):
    result = write_eda_outputs(_eda_df(), tmp_path, enable_lightgbm=False)
    assert (tmp_path / "plots" / "01_outcome_balance.png").exists()
    assert (tmp_path / "plots" / "02_feature_coverage_by_group.png").exists()
    assert (tmp_path / "plots" / "04_binary_feature_forest_diff_pp.png").exists()
    assert list((tmp_path / "plots").glob("05_numeric_binned_*.png"))
    assert list((tmp_path / "plots").glob("06_categorical_*_cited_rate.png"))
    assert (tmp_path / "plots" / "07_intent_page_type_cell_n.png").exists()
    assert (tmp_path / "tables" / "correlation_matrix.csv").exists()
    assert (tmp_path / "tables" / "vif_summary.csv").exists()
    assert (tmp_path / "tables" / "plot_skip_reasons.csv").exists()
    assert (tmp_path / "tables" / "numeric_shape_recommendations.csv").exists()
    assert (tmp_path / "tables" / "lpm_feature_readiness.csv").exists()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "eda_warnings.csv").exists()
    assert result["metadata"]["binary_plot_count"] > 0
    assert result["metadata"]["numeric_plot_count"] > 0
    assert result["metadata"]["categorical_plot_count"] > 0


def test_eda_skips_association_plots_for_single_class_outcome(tmp_path):
    df = _eda_df()
    df["cited"] = 1
    result = write_eda_outputs(df, tmp_path, enable_lightgbm=True)
    assert (tmp_path / "plots" / "01_outcome_balance.png").exists()
    assert not (tmp_path / "plots" / "04_binary_feature_forest_diff_pp.png").exists()
    skips = pd.read_csv(tmp_path / "tables" / "plot_skip_reasons.csv")
    assert "outcome has one class" in set(skips["reason"])
    readiness = pd.read_csv(tmp_path / "tables" / "lpm_feature_readiness.csv")
    assert not readiness["recommended_for_lpm"].any()
    assert "lightgbm_skipped_single_class_outcome" in result["metadata"]["warnings"]
