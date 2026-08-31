"""Writing-structure v2 score and missing-value policy.

SUPERSEDED by :mod:`writing_structure_v3` (2026-07-27), which drops
``has_question_answer_structure`` because it was identical to ``has_faq_pattern``
in every governed row. Nothing in the active pipeline imports this module.

It is kept so that estimates published under ``writing_structure_score_v2`` stay
reproducible: the score's definition has to survive as long as results computed
from it are quoted. Do not use it for new work. See ``docs/CHANGELOG.md``.
"""

from __future__ import annotations

import pandas as pd


WRITING_STRUCTURE_VERSION = "writing_structure_v2"
WRITING_STRUCTURE_SCORE = "writing_structure_score_v2"
WRITING_STRUCTURE_COMPONENTS = (
    "has_main_content_unordered_list",
    "has_main_content_ordered_list",
    "has_faq_pattern",
    "has_question_answer_structure",
    "opening_has_summary_signal",
    "opening_has_direct_answer_signal",
)


def attach_writing_structure_score_v2(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the six-component score without treating missing components as zero.

    The score is measured only when all six binary components are measured.
    ``writing_structure_components_measured_n`` records partial availability,
    while ``writing_structure_score_v2_available`` identifies complete scores.
    """
    missing = [column for column in WRITING_STRUCTURE_COMPONENTS if column not in frame]
    if missing:
        raise KeyError(f"Missing writing-structure v2 components: {missing}")

    output = frame.copy()
    numeric = output[list(WRITING_STRUCTURE_COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    output["writing_structure_components_measured_n"] = numeric.notna().sum(axis=1).astype("Int64")
    available = numeric.notna().all(axis=1)
    output["writing_structure_score_v2_available"] = available.astype("Int64")
    score = numeric.sum(axis=1, min_count=len(WRITING_STRUCTURE_COMPONENTS))
    output[WRITING_STRUCTURE_SCORE] = score.astype("Int64")
    return output
