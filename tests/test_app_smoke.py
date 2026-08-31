from streamlit.testing.v1 import AppTest

from src.econometrics_eda_v2 import feature_distribution_support as feature_support
from ui.views import econometrics_qa


def test_feature_label_falls_back_during_partial_hot_reload(monkeypatch):
    monkeypatch.delitem(
        feature_support.FEATURE_LABELS,
        "has_main_content_unordered_list",
        raising=False,
    )

    assert (
        econometrics_qa._feature_label("has_main_content_unordered_list")
        == "Has Main Content Unordered List"
    )


def test_chatgpt_and_econometrics_modes_render_without_uncaught_exceptions():
    app = AppTest.from_file("app.py").run(timeout=30)
    assert not app.exception
    assert app.sidebar.radio[0].value == "ChatGPT Bright Data Audit"

    app.sidebar.radio[0].set_value("Content Econometrics QA").run(timeout=60)
    assert not app.exception
    assert any(selectbox.label == "Source webpage" for selectbox in app.selectbox)
    assert any(selectbox.label == "Taxonomy source webpage" for selectbox in app.selectbox)
    taxonomy = next(selectbox for selectbox in app.selectbox if selectbox.label == "Taxonomy level")
    assert "Gemini: detailed page type" in taxonomy.options
    assert "Gemini: source / site type" in taxonomy.options
    assert any(selectbox.label == "Baseline taxonomy" for selectbox in app.selectbox)
    assert any(selectbox.label == "Comparison level" for selectbox in app.selectbox)
    assert any(selectbox.label == "Disagreement webpage" for selectbox in app.selectbox)
    assert any(code.value.startswith("D0 -> FE1 -> FE2") for code in app.code)
    model_tables = [
        dataframe.value
        for dataframe in app.dataframe
        if "Analysis type" in getattr(dataframe.value, "columns", [])
    ]
    assert len(model_tables) == 1
    assert model_tables[0]["Layer"].tolist() == ["D0", "FE1", "FE2", "FE3", "FE4"]
    assert not any(selectbox.label == "Model transition" for selectbox in app.selectbox)
    assert not any(selectbox.label == "Specification for covariance comparison" for selectbox in app.selectbox)
    assert sum(button.label == "Load live webpage" for button in app.button) == 5
    assert any(tab.label == "Feature validation" for tab in app.tabs)
    assert any(tab.label == "Verified table diagnostics" for tab in app.tabs)
    assert any(tab.label == "Position Feature EDA" for tab in app.tabs)
    assert any(
        selectbox.label == "Diagnostic table" for selectbox in app.selectbox
    )
    feature_select = next(
        selectbox for selectbox in app.selectbox if selectbox.label == "Feature to inspect"
    )
    assert "Answer-Oriented Writing Structure Score v3" in feature_select.options
    assert "Question-Answer Structure Detected" not in feature_select.options
    assert "Main-Content Unordered List" in feature_select.options
    assert "Main-Content Ordered List" in feature_select.options
    assert "Bullet List Detected" not in feature_select.options
    assert "Numbered List Detected" not in feature_select.options
    assert any(tab.label == "Structured List Evidence" for tab in app.tabs)
    assert any(selectbox.label == "Feature value or governed range" for selectbox in app.selectbox)
    assert sum(button.label == "Check iframe policy" for button in app.button) == 5


def test_separate_position_model_mode_renders_all_requested_sections():
    app = AppTest.from_file("app.py").run(timeout=30)
    app.sidebar.radio[0].set_value("Position Model — New").run(timeout=60)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Feature EDA",
        "Domain and Prompt Balance",
        "Model Results",
        "Confidence-Interval Diagnostics",
        "Multicollinearity",
        "Robustness",
        "Data Quality",
    ]
    assert any(selectbox.label == "Feature" for selectbox in app.selectbox)
    assert any(selectbox.label == "Coefficient" for selectbox in app.selectbox)
    assert any(code.value.startswith("M5: cited ~") for code in app.code)
