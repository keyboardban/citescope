import pandas as pd

from scripts.v2_prepare_area_condo_content_econometrics_package import (
    _apply_taxonomy_collapses,
    _leakage_guardrail,
    _model_ladder,
    _sparse_category_audit,
)


def _taxonomy_frame() -> pd.DataFrame:
    rows = []
    for index in range(25):
        rows.append(
            {
                "cited": index % 2,
                "prompt_id": f"p{index}",
                "content_feature_available": True,
                "page_type_url_seed_general": "unknown",
                "page_type_family_general": "unknown",
                "site_type_general": "unknown",
                "heading_count_group": "2-6",
                "link_count_group": "9+",
                "word_count_group": "300-999",
                "content_strength": "strong",
                "intent": "Area / Location",
            }
        )
    for index in range(5):
        rows.append(
            {
                "cited": 1,
                "prompt_id": f"v{index}",
                "content_feature_available": True,
                "page_type_url_seed_general": "video_page",
                "page_type_family_general": "document_or_media",
                "site_type_general": "video_platform",
                "heading_count_group": "13+",
                "link_count_group": "0-3",
                "word_count_group": "100-299",
                "content_strength": "strong",
                "intent": "Document / Brochure",
            }
        )
    return pd.DataFrame(rows)


def test_taxonomy_collapse_keeps_unknown_and_combines_tiny_levels():
    collapsed = _apply_taxonomy_collapses(_taxonomy_frame())
    assert set(collapsed.page_type_url_seed_general_collapsed) == {"unknown", "rare_other"}
    assert set(collapsed.site_type_general_collapsed) == {"unknown", "rare_other"}
    assert collapsed.loc[
        collapsed.page_type_url_seed_general.eq("unknown"),
        "page_type_url_seed_general_collapsed",
    ].eq("unknown").all()


def test_sparse_audit_flags_separating_video_levels():
    audit = _sparse_category_audit(_taxonomy_frame())
    video = audit[
        audit.feature.eq("site_type_general")
        & audit.category.eq("video_platform")
    ].iloc[0]
    assert bool(video.sparse_flag)
    assert bool(video.perfect_prediction_flag)
    assert video.recommended_action == "collapse_to_rare_other"


def test_model_ladder_and_leakage_guardrail():
    ladder = _model_ladder()
    assert ladder.model_id.tolist() == [f"M{index}" for index in range(11)]
    clean = pd.DataFrame(
        {
            "cited": [0, 1],
            "prompt_id": ["p1", "p1"],
            "normalized_url": ["https://a", "https://b"],
            "log2_word_count_plus1": [8.0, 9.0],
        }
    )
    guardrail = _leakage_guardrail(clean, clean, ladder)
    assert guardrail.status.eq("pass").all()
