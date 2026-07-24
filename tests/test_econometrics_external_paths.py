from __future__ import annotations

import importlib

from src.econometrics_eda_v2 import paths


def test_legacy_workspace_paths_relocate_to_current_workspace():
    legacy = (
        paths.LEGACY_WORKSPACE_ROOT
        / "CompareSearch-v2-clean"
        / "outputs"
        / "econometrics_eda_v2"
    )

    assert paths.relocate_workspace_path(legacy) == (
        paths.WORKSPACE_ROOT
        / "CompareSearch-v2-clean"
        / "outputs"
        / "econometrics_eda_v2"
    ).resolve()


def test_external_research_root_controls_default_data_and_output_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("CITESCOPE_RESEARCH_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("CITESCOPE_ECONOMETRICS_DATA_DIR", raising=False)
    monkeypatch.delenv("CITESCOPE_ECONOMETRICS_OUTPUT_DIR", raising=False)

    reloaded = importlib.reload(paths)

    assert reloaded.RESEARCH_ROOT == tmp_path.resolve()
    assert reloaded.ECONOMETRICS_DATA_DIR == (tmp_path / "data/econometrics_v2").resolve()
    assert reloaded.ECONOMETRICS_OUTPUT_DIR == (tmp_path / "outputs/econometrics_eda_v2").resolve()
    assert reloaded.topic_output_dir() == (
        tmp_path / "outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded"
    ).resolve()

    monkeypatch.delenv("CITESCOPE_RESEARCH_DATA_DIR", raising=False)
    importlib.reload(paths)


def test_specific_econometrics_paths_override_research_root(tmp_path, monkeypatch):
    data_dir = tmp_path / "input-data"
    output_dir = tmp_path / "generated-results"
    monkeypatch.setenv("CITESCOPE_RESEARCH_DATA_DIR", str(tmp_path / "archive"))
    monkeypatch.setenv("CITESCOPE_ECONOMETRICS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CITESCOPE_ECONOMETRICS_OUTPUT_DIR", str(output_dir))

    reloaded = importlib.reload(paths)

    assert reloaded.ECONOMETRICS_DATA_DIR == data_dir.resolve()
    assert reloaded.ECONOMETRICS_OUTPUT_DIR == output_dir.resolve()

    monkeypatch.delenv("CITESCOPE_RESEARCH_DATA_DIR", raising=False)
    monkeypatch.delenv("CITESCOPE_ECONOMETRICS_DATA_DIR", raising=False)
    monkeypatch.delenv("CITESCOPE_ECONOMETRICS_OUTPUT_DIR", raising=False)
    importlib.reload(paths)
