from __future__ import annotations

import json

import pandas as pd

from src.econometrics_eda_v2 import feature_distribution_support as support


def _sample_rows() -> pd.DataFrame:
    rows = []
    values = [
        ("p1", 0, "strong", 0.0, 0, 0.0, 0),
        ("p1", 1, "medium", 1.0, 1, 2.0, 2),
        ("p2", 0, "strong", 2.0, 0, 4.0, 1),
        ("p2", 1, "weak", 2.0, 0, 4.0, 1),
        ("p3", 0, "weak", 4.0, pd.NA, 8.0, 4),
        ("p3", 1, "strong", 5.0, 1, 10.0, 6),
    ]
    for index, (prompt, cited, strength, length, table, factual, writing) in enumerate(values):
        rows.append(
            {
                "prompt_id": prompt,
                "normalized_url": f"https://example.com/{index}",
                "source_url": f"https://example.com/{index}?utm_source=test",
                "source_root_domain": "example.com",
                "cited": cited,
                "content_strength": strength,
                "heading_count_group": ["0-1", "2-6", "7-12", "13+", "13+", "2-6"][index],
                "log2_word_count_plus1": length,
                "has_verified_html_table": table,
                "factual_numeric_density_score": factual,
                "writing_structure_score": writing,
                "has_bullet_list": [0, 1, 0, 0, 1, 1][index],
                "has_numbered_list": [0, 0, 0, 0, 0, 1][index],
                "has_faq_pattern": [0, 0, 1, 1, 0, 1][index],
                "has_question_answer_structure": [0, 0, 1, 1, 0, 1][index],
                "opening_has_summary_signal": [0, 1, 0, 0, 0, 1][index],
                "opening_has_direct_answer_signal": [0, 0, 0, 0, 1, 1][index],
            }
        )
    return pd.DataFrame(rows)


def test_every_dashboard_feature_appears_in_summary():
    summary = support.feature_summary(_sample_rows())

    assert summary["feature_name"].tolist() == list(support.DASHBOARD_FEATURES)
    assert "external_evidence_structure_score" not in set(summary["feature_name"])


def test_binary_distribution_distinguishes_zero_one_and_missing():
    distribution = support.distribution_table(_sample_rows(), "has_verified_html_table")
    counts = dict(distribution.set_index("bin_key")["n_rows"])

    assert counts == {"not_detected": 3, "detected": 2, "unmeasured": 1}
    assert distribution["n_rows"].sum() == 6


def test_writing_score_distribution_includes_zero_through_six_and_na():
    distribution = support.distribution_table(_sample_rows(), "writing_structure_score")

    assert distribution["bin_label"].tolist() == ["0", "1", "2", "3", "4", "5", "6", "NA"]
    assert distribution.loc[distribution["bin_label"].eq("0"), "n_rows"].iloc[0] == 1
    assert distribution.loc[distribution["bin_label"].eq("3"), "n_rows"].iloc[0] == 0
    assert distribution.loc[distribution["bin_label"].eq("NA"), "n_rows"].iloc[0] == 0


def test_distribution_outcome_counts_match_stored_citation_status():
    frame = _sample_rows()
    distribution = support.distribution_table(frame, "writing_structure_score")

    assert distribution["cited_rows"].sum() == int(frame["cited"].sum())
    assert distribution["more_only_rows"].sum() == int(frame["cited"].eq(0).sum())


def test_prompt_fixed_effect_variation_is_computed_from_prompt_groups():
    variation = support.prompt_variation(_sample_rows(), "log2_word_count_plus1")

    assert variation["total_prompts"] == 3
    assert variation["prompts_with_usable_variation"] == 2
    assert variation["prompts_with_no_variation"] == 1
    assert variation["rows_in_prompts_with_usable_variation"] == 4


def test_chart_bin_selection_filters_review_rows_with_same_governed_bins():
    frame = _sample_rows()
    bins = support.feature_bins(frame, "factual_numeric_density_score")
    highest = bins.sort_values("bin_order").iloc[-1]["bin_key"]
    cited_only = frame[frame["cited"].eq(1)]

    filtered = support.apply_review_filter(
        cited_only,
        "factual_numeric_density_score",
        bin_key=str(highest),
        bin_reference=frame,
    )

    assert not filtered.empty
    assert filtered["factual_numeric_density_score"].min() == frame["factual_numeric_density_score"].max()


def test_variation_filter_uses_full_reference_not_post_filter_subset():
    frame = _sample_rows()
    cited_only = frame[frame["cited"].eq(1)]

    filtered = support.apply_review_filter(
        cited_only,
        "log2_word_count_plus1",
        variation_mode="with_variation",
        variation_reference=frame,
    )

    assert set(filtered["prompt_id"]) == {"p1", "p3"}


def test_support_artifacts_are_count_consistent_and_leakage_safe(tmp_path):
    frame = _sample_rows()
    tables = tmp_path / "tables"
    manifest_path = tmp_path / "frontend" / "feature_distribution_support_manifest.json"

    manifest = support.build_support_artifacts(frame, tables, manifest_path)
    artifacts, loaded_manifest = support.load_support_artifacts(manifest_path)

    assert manifest["model_ready_rows"] == len(frame)
    assert loaded_manifest["unique_prompts"] == frame["prompt_id"].nunique()
    assert set(artifacts) == set(support.SUPPORT_FILES)
    assert len(artifacts["manual_qa_review_rows.csv"]) == len(frame)
    headers = " ".join(
        column
        for artifact in artifacts.values()
        for column in artifact.columns
    ).casefold()
    assert "answer_text" not in headers
    assert "api_key" not in headers
    assert "authorization" not in headers
    assert json.loads(manifest_path.read_text())["blocked_features"] == [
        "external_evidence_structure_score"
    ]
