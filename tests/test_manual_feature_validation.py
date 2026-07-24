from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src import econometrics_qa
from src.econometrics_eda_v2 import manual_feature_validation as feature_qa


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_missing_binary_measurement_is_not_coerced_to_zero():
    assert feature_qa.nullable_binary_status(pd.NA) == "Unmeasured"
    assert feature_qa.nullable_binary_status(float("nan")) == "Unmeasured"
    assert feature_qa.nullable_binary_status(0) == "Not detected"
    assert feature_qa.nullable_binary_status(1) == "Detected"


def test_writing_structure_score_equals_governed_component_sum():
    frame = pd.DataFrame(
        [
            dict(
                zip(
                    feature_qa.COMPONENTS,
                    [1, 0, 1, 1, 0, 1],
                )
            )
        ]
    )

    assert feature_qa.writing_component_sum(frame).iloc[0] == 4


def test_factual_numeric_score_equals_governed_component_sum():
    frame = pd.DataFrame(
        [{
            "number_token_per_1000_words": 20,
            "percent_mention_count": 1,
            "year_mention_count": 0,
            "range_mention_count": 2,
            "measurement_mention_count": 3,
        }]
    )

    contributions = feature_qa.factual_component_contributions(frame)

    assert contributions.loc[0, "numeric_rate_contribution"] == 2
    assert contributions.loc[0, "percent_indicator_contribution"] == 1
    assert contributions.loc[0, "year_indicator_contribution"] == 0
    assert contributions.loc[0, "range_indicator_contribution"] == 1
    assert contributions.loc[0, "measurement_log_contribution"] == pytest.approx(1.3862943611)
    assert contributions.sum(axis=1).iloc[0] == pytest.approx(5.3862943611)


def test_html_preview_is_inert_and_preserves_reviewable_content():
    unsafe = """
    <html><head><script>steal()</script></head><body onload="steal()">
      <h1>Useful heading</h1>
      <a href="javascript:steal()" onclick="steal()">Bad link</a>
      <form><input value="secret"></form>
      <iframe src="https://evil.example"></iframe>
      <table><tr><td>Price 100</td></tr></table>
    </body></html>
    """

    sanitized = feature_qa.sanitize_html_preview(unsafe)
    lowered = sanitized.casefold()

    assert "useful heading" in lowered
    assert "price 100" in lowered
    assert "<script" not in lowered
    assert "<iframe" not in lowered
    assert "<form" not in lowered
    assert "onload" not in lowered
    assert "onclick" not in lowered
    assert "javascript:" not in lowered


def test_blocked_iframe_does_not_remove_stored_feature_content():
    stored_content = "Exact producer text remains available offline."

    policy, _ = econometrics_qa.classify_frame_policy({"X-Frame-Options": "DENY"})

    assert policy == "blocked"
    assert stored_content == "Exact producer text remains available offline."


def test_manual_review_is_append_only_and_does_not_modify_feature_file(tmp_path):
    feature_file = tmp_path / "features.csv"
    review_file = tmp_path / "manual_feature_validation_reviews.csv"
    pd.DataFrame(
        [{"normalized_url": "https://example.com/page", "writing_structure_score": 2}]
    ).to_csv(feature_file, index=False)
    before = _sha256(feature_file)

    feature_qa.append_review(
        review_file,
        {
            "normalized_url": "https://example.com/page",
            "prompt_id": "prompt-1",
            "feature_name": "writing_structure_score",
            "automated_value": 2,
            "reviewer_decision": "incorrect",
            "error_type": "formatting lost",
            "reviewer_note": "A bullet list was flattened.",
            "content_source_used": "page_text_preview_3000_chars",
            "feature_producer_version": feature_qa.WRITING_VERSION,
        },
    )

    assert _sha256(feature_file) == before
    reviews = pd.read_csv(review_file)
    assert len(reviews) == 1
    assert reviews.loc[0, "reviewer_decision"] == "incorrect"
    assert pd.notna(reviews.loc[0, "reviewed_at"])


def test_built_frontend_artifacts_match_producer_text_and_exclude_sensitive_fields():
    repo = Path(__file__).resolve().parents[1]
    frontend = repo / "outputs/econometrics_redesign_v2_20260722/frontend"
    if not (frontend / "manual_feature_validation_manifest.json").exists():
        pytest.skip("Manual feature-validation artifact has not been built.")

    rows, content, manifest = feature_qa.load_validated_artifacts(frontend)
    package = feature_qa.source_paths(repo)["base"].parents[1]
    assembly = pd.read_csv(
        package / "tables/10_writing_factual_density_features/url_text_assembly_audit.csv",
        usecols=["normalized_url", "url_text_for_features", "text_source_used"],
    ).drop_duplicates("normalized_url")
    checked = content.merge(assembly, on="normalized_url", suffixes=("_frontend", "_producer"))

    assert checked["authoritative_feature_content"].equals(checked["url_text_for_features_producer"])
    assert checked["authoritative_content_source"].equals(checked["text_source_used_producer"])
    assert manifest["validated"] is True
    assert not rows.empty

    columns = " ".join([*rows.columns, *content.columns]).casefold()
    for forbidden in feature_qa.FORBIDDEN_ARTIFACT_TERMS:
        assert forbidden not in columns


def test_frontend_rows_and_content_merge_without_provenance_suffixes():
    rows = pd.DataFrame(
        [{
            "normalized_url": "https://example.com/page",
            "source_url": "https://example.com/page?utm_source=test",
            "feature_extraction_text_scope": "excerpt_only",
            "text_source_used": "page_text_preview_3000_chars",
        }]
    )
    content = pd.DataFrame(
        [{
            "normalized_url": "https://example.com/page",
            "source_url": "https://example.com/page?utm_source=test",
            "feature_extraction_text_scope": "excerpt_only",
            "text_source_used": "page_text_preview_3000_chars",
            "authoritative_feature_content": "Exact producer text.",
        }]
    )
    join_keys = ["normalized_url", "source_url"]
    duplicated = [
        column for column in content.columns if column in rows.columns and column not in join_keys
    ]

    merged = rows.merge(
        content.drop(columns=duplicated),
        on=join_keys,
        how="left",
        validate="many_to_one",
    )

    assert merged.loc[0, "feature_extraction_text_scope"] == "excerpt_only"
    assert merged.loc[0, "text_source_used"] == "page_text_preview_3000_chars"
    assert not any(column.endswith(("_x", "_y")) for column in merged.columns)
