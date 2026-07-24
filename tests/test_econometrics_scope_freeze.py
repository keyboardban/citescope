import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LAYERS = ["D0", "FE1", "FE2", "FE3", "FE4"]


def test_model_registry_contains_exactly_five_frozen_layers():
    registry = pd.read_csv(ROOT / "tables/econometrics_model_registry_v1.csv")

    assert registry["analysis_layer"].tolist() == LAYERS
    assert registry.set_index("analysis_layer").loc["FE3", "branch_from"] == "FE2"
    assert registry.set_index("analysis_layer").loc["FE4", "branch_from"] == "FE2"
    assert "source_root_domain" not in registry.set_index("analysis_layer").loc["FE4", "fixed_effects"]


def test_redesign_mermaid_contains_only_frozen_layers():
    diagram = (ROOT / "docs/econometrics_pipeline_redesign_v1.mmd").read_text(encoding="utf-8")
    node_ids = set(re.findall(r"\b(?:D0|FE[0-9]+)\b", diagram))

    assert node_ids == set(LAYERS)
    assert 'FE2 --> FE3["FE3"]' in diagram
    assert 'FE2 --> FE4["FE4"]' in diagram
    assert "FE3 --> FE4" not in diagram


def test_redesign_documents_end_with_exactly_five_layer_rows():
    for relative in (
        "docs/econometrics_pipeline_redesign_v1.md",
        "docs/econometrics_core_model_specification_v1.md",
    ):
        lines = (ROOT / relative).read_text(encoding="utf-8").strip().splitlines()
        final_rows = [line for line in lines[-7:] if re.match(r"\| (?:D0|FE[0-9]+) \|", line)]
        assert [row.split("|")[1].strip() for row in final_rows] == LAYERS
