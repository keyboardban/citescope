from streamlit.testing.v1 import AppTest


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
    assert any(selectbox.label == "Feature to inspect" for selectbox in app.selectbox)
    assert any(selectbox.label == "Feature value or governed range" for selectbox in app.selectbox)
    assert sum(button.label == "Check iframe policy" for button in app.button) == 5
