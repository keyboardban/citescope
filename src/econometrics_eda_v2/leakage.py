from __future__ import annotations

LEAKAGE_EXCLUSIONS = {
    "page_answer_similarity",
    "max_chunk_answer_similarity",
    "answer_overlap",
    "brand_appeared_in_answer",
    "answer_like_text_in_first_500_chars",
    "cited_label",
    "is_more_only",
    "source_group",
    "source_origin",
}
DIAGNOSTIC_ONLY = {"source_position", "observed_rank", "log1p_source_position", "page_type_final_source"}


def safe_predictor_columns(columns: list[str]) -> list[str]:
    blocked = LEAKAGE_EXCLUSIONS | DIAGNOSTIC_ONLY | {"cited", "answer_text"}
    return [c for c in columns if c not in blocked]
