import math

import pandas as pd

from src.econometrics_eda_v2.position_feature_eda import CITATION_COLORS, extract_position_features


def test_citation_palette_uses_distinct_categorical_colors():
    assert CITATION_COLORS == {
        "Cited": "#009E73",
        "Not cited": "#D55E00",
    }


def test_html_position_extraction_keeps_absence_separate_from_zero():
    html = """
    <html><body><nav><ul><li>Home</li><li>About</li></ul></nav>
    <main>
      <h1>Example guide</h1>
      <p>Opening context contains enough words to precede the structured evidence.</p>
      <h2>What is the answer?</h2>
      <p>In short, the answer is supported by 25% growth.</p>
      <table><tr><th>Year</th><th>Value</th></tr><tr><td>2025</td><td>25%</td></tr></table>
      <h2>Comparison</h2><p>Compare option A versus option B.</p>
      <ol><li>Check the source</li><li>Record the result</li></ol>
    </main></body></html>
    """
    result = extract_position_features(html, "https://example.com/guide")

    assert result["position_features_available"] == 1
    assert result["has_table"] == 1
    assert result["table_count"] == 1
    assert 0 <= result["first_table_position_ratio"] <= 1
    assert result["first_table_position_quartile"] in {"Q1", "Q2", "Q3", "Q4"}
    assert result["has_bullets"] == 1
    assert result["has_question_heading"] == 1
    assert result["has_direct_answer"] == 1
    assert result["has_comparison"] == 1
    assert result["has_steps"] == 1
    assert result["has_numeric_evidence"] == 1

    absent = extract_position_features(
        "<main><h1>Plain page</h1><p>This page contains plain explanatory prose only.</p></main>",
        "https://example.com/plain",
    )
    assert absent["has_table"] == 0
    assert pd.isna(absent["first_table_position_ratio"])
    assert absent["first_table_position_quartile"] == "No feature"


def test_unmeasured_page_does_not_impute_feature_absence():
    result = extract_position_features("", "https://example.com")

    assert result["position_features_available"] == 0
    assert pd.isna(result["has_table"])
    assert pd.isna(result["first_table_position_ratio"])
    assert result["first_table_position_quartile"] == "Unmeasured"


def test_markdown_fallback_measures_structured_lists_and_tables():
    markdown = """# Guide

Introductory words before the feature.

## FAQ

- First answer
- Second answer

| Item | Value |
| --- | --- |
| A | 10% |
"""
    result = extract_position_features("", "https://example.com", markdown)

    assert result["position_measurement_source"] == "generated_markdown_fallback"
    assert result["has_bullets"] == 1
    assert result["has_table"] == 1
    assert result["has_faq"] == 1
    assert not math.isnan(result["first_list_position_ratio"])
