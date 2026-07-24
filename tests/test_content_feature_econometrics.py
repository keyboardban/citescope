from pathlib import Path

import pandas as pd
import pytest

from src.econometrics_eda_v2.content_feature_econometrics import (
    forbidden_formula_matches,
    run_model_and_save,
)


def _model_frame() -> pd.DataFrame:
    rows = []
    for index in range(80):
        rows.append(
            {
                "cited": int((index % 7) in {0, 1, 3}),
                "x": index / 10,
                "prompt_id": f"p{index % 8}",
                "normalized_url": f"https://example.com/{index % 20}",
                "source_root_domain": f"d{index % 5}.example",
            }
        )
    return pd.DataFrame(rows)


def test_formula_guardrail_scans_predictor_side_only():
    assert forbidden_formula_matches("cited ~ x + source_position") == ["source_position"]
    assert forbidden_formula_matches("cited ~ x") == []


def test_clustered_lpm_writes_shared_schema(tmp_path: Path):
    run = run_model_and_save(
        "cited ~ x + C(prompt_id)",
        _model_frame(),
        "test_model",
        tmp_path / "result.csv",
    )
    assert (tmp_path / "result.csv").exists()
    assert {"HC3", "cluster_prompt_id", "cluster_normalized_url"}.issubset(set(run.table["cov_type"]))
    assert {
        "model_id",
        "formula",
        "term",
        "estimate_pp",
        "conf_low_pp",
        "conf_high_pp",
        "n_prompts",
        "n_urls",
    }.issubset(run.table.columns)


def test_model_runner_stops_on_forbidden_predictor(tmp_path: Path):
    frame = _model_frame().assign(source_position=1)
    with pytest.raises(ValueError, match="Leakage guardrail"):
        run_model_and_save(
            "cited ~ x + source_position",
            frame,
            "unsafe",
            tmp_path / "unsafe.csv",
        )
