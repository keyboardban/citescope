from pathlib import Path

import pandas as pd

from src.econometrics_eda_v2.gemini_taxonomy_features import (
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    attach_gemini_taxonomy,
)


def test_attach_gemini_taxonomy_preserves_unknown_and_collapses_rare(tmp_path: Path):
    package = tmp_path / "content_econometrics_ai_package"
    taxonomy_dir = tmp_path / "tables/gemini_page_taxonomy_batch"
    taxonomy_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "normalized_url": "https://example.com/a",
                "llm_page_type_general": "article_page",
                "llm_page_type_family_general": "editorial_content",
                "llm_site_type_general": "publisher_media",
                "llm_confidence": "high",
                "classification_input_mode": "markdown",
                "llm_needs_review": False,
                "result_valid": True,
            },
            {
                "normalized_url": "https://example.com/b",
                "llm_page_type_general": "unknown",
                "llm_page_type_family_general": "unknown",
                "llm_site_type_general": "unknown",
                "llm_confidence": "low",
                "classification_input_mode": "metadata_only",
                "llm_needs_review": True,
                "result_valid": True,
            },
        ]
    ).to_csv(taxonomy_dir / "all_pages_gemini_taxonomy_classifications.csv", index=False)
    frame = pd.DataFrame(
        {
            "normalized_url": ["https://example.com/a", "https://example.com/b"],
            "cited": [1, 0],
        }
    )

    result, _, audit = attach_gemini_taxonomy(frame, package, min_rows=2)

    assert result.loc[0, GEMINI_PAGE_FAMILY_COLLAPSED] == "rare_other"
    assert result.loc[1, GEMINI_PAGE_FAMILY_COLLAPSED] == "unknown"
    assert result.loc[0, GEMINI_SOURCE_TYPE_COLLAPSED] == "rare_other"
    assert audit["match_rate"] == 1.0
