"""Canonical writing-structure v3 score and missing-value policy."""

from __future__ import annotations

import pandas as pd


WRITING_STRUCTURE_VERSION = "writing_structure_v3"
WRITING_STRUCTURE_SCORE = "writing_structure_score_v3"
WRITING_STRUCTURE_COMPONENTS = (
    "has_main_content_unordered_list",
    "has_main_content_ordered_list",
    "has_faq_pattern",
    "opening_has_summary_signal",
    "opening_has_direct_answer_signal",
)


def attach_writing_structure_score_v3(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the five-component score without treating missing components as zero.

    ``has_question_answer_structure`` is intentionally excluded because it is
    identical to ``has_faq_pattern`` in the governed sample. The score is
    measured only when all five active components are measured.
    """
    missing = [column for column in WRITING_STRUCTURE_COMPONENTS if column not in frame]
    if missing:
        raise KeyError(f"Missing writing-structure v3 components: {missing}")

    output = frame.copy()
    numeric = output[list(WRITING_STRUCTURE_COMPONENTS)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    output["writing_structure_components_measured_n"] = (
        numeric.notna().sum(axis=1).astype("Int64")
    )
    available = numeric.notna().all(axis=1)
    output["writing_structure_score_v3_available"] = available.astype("Int64")
    score = numeric.sum(axis=1, min_count=len(WRITING_STRUCTURE_COMPONENTS))
    output[WRITING_STRUCTURE_SCORE] = score.astype("Int64")
    return output
