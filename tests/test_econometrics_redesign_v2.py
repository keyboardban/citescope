from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.redesigned_pipeline_v2 import (
    CORE_FOCAL,
    FOCAL,
    GEMINI_SEMANTIC_FOCAL,
    attach_gemini_semantic_features,
    formulas,
    validate_formula_scope,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_has_exact_five_layers_and_branches():
    registry = pd.read_csv(ROOT / "tables/econometrics_model_registry_v2.csv")
    assert registry.analysis_layer.tolist() == ["D0", "FE1", "FE2", "FE3", "FE4"]
    assert registry.loc[registry.analysis_layer.eq("D0"), "analysis_type"].item() == "descriptive"
    assert registry.set_index("analysis_layer").loc["FE3", "branch_from"] == "FE2"
    assert registry.set_index("analysis_layer").loc["FE4", "branch_from"] == "FE2"


def test_formulas_enforce_feature_scope_and_separate_branches():
    model_formulas = formulas(pd.DataFrame())
    validate_formula_scope(model_formulas)
    assert set(key for key in model_formulas if key.startswith("FE1_")) == {f"FE1_{x}" for x in FOCAL}
    joined = " ".join(model_formulas.values())
    assert "writing_structure_score_v3" in joined
    assert "writing_structure_score_v2" not in joined
    assert "has_question_answer_structure" not in joined
    assert "has_bullet_list" not in joined
    assert "has_numbered_list" not in joined
    assert "heading_count" not in joined
    assert "has_table" not in joined
    assert "external_evidence" not in joined
    assert set(CORE_FOCAL).issubset(FOCAL)
    assert set(GEMINI_SEMANTIC_FOCAL).issubset(FOCAL)
    assert "position_ratio_gemini" not in joined
    assert "_count_gemini" not in joined
    assert "content_strength" not in " ".join(model_formulas[key] for key in model_formulas if key.startswith("FE1_"))
    assert "source_root_domain" in model_formulas["FE3"]
    assert "source_root_domain" not in model_formulas["FE4"]


def test_selected_feature_registry_matches_executable_membership():
    registry = pd.read_csv(ROOT / "config/econometrics_selected_features_v2.csv").set_index("feature_name")
    for feature in FOCAL:
        assert registry.loc[feature, "FE1"] == "Included"
        assert registry.loc[feature, "FE2"] == "Included"
    assert registry.loc["content_strength", "FE1"] == "Excluded"
    assert registry.loc["content_strength", "FE2"] == "Included"
    assert registry.loc["heading_count_group", "D0"] == "Included"
    assert set(registry.loc["heading_count_group", ["FE1", "FE2", "FE3", "FE4"]]) == {"Excluded"}
    assert registry.loc["external_evidence_structure_score", "entry_status"] == "blocked"


def test_precomputed_frontend_manifest_matches_completed_outputs():
    import hashlib
    import json

    frontend = ROOT / "outputs/econometrics_redesign_v4_20260803_gemini_semantic_features/frontend"
    manifest = json.loads((frontend / "manifest.json").read_text())
    assert manifest["layers"] == ["D0", "FE1", "FE2", "FE3", "FE4"]
    assert manifest["validated"] is True
    for filename, metadata in manifest["files"].items():
        assert hashlib.sha256((frontend / filename).read_bytes()).hexdigest() == metadata["sha256"]


def test_verified_table_preserves_measured_absence_and_unmeasured_semantics():
    source = (ROOT / "src/econometrics_eda_v2/redesigned_pipeline_v2.py").read_text()
    assert 'dtype="Int64"' in source
    assert 'html_measured' in source
    assert 'merged.loc[html_measured, "has_verified_html_table"]' in source


def test_gemini_semantic_join_preserves_unmeasured_as_na(tmp_path):
    semantic = pd.DataFrame(
        {
            "normalized_url": ["https://measured.test", "https://failed.test"],
            "gemini_status": ["success", "partial_failure"],
            **{
                feature: [1, 0]
                for feature in GEMINI_SEMANTIC_FOCAL
            },
        }
    )
    path = tmp_path / "semantic.csv"
    semantic.to_csv(path, index=False)
    source = pd.DataFrame(
        {"normalized_url": ["https://measured.test", "https://failed.test", "https://missing.test"]}
    )
    result = attach_gemini_semantic_features(source, path).set_index("normalized_url")
    assert result.loc["https://measured.test", GEMINI_SEMANTIC_FOCAL].eq(1).all()
    assert result.loc["https://failed.test", GEMINI_SEMANTIC_FOCAL].isna().all()
    assert result.loc["https://missing.test", GEMINI_SEMANTIC_FOCAL].isna().all()
