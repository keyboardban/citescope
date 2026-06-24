"""Topic question packs: parser, tagging, and per-topic aggregation."""

from __future__ import annotations

from src import question_sets
from src.demo import make_demo_topic_study


def test_parse_prompt_block_formats():
    text = (
        "ID | Intent | Prompt\n"                                  # header (skipped)
        "H01 | Informational | What ingredients for dry skin?\n"  # pipe, with id
        "A09\tBrand Comparison\tToyota vs BYD?\n"                 # tab, with id
        "Just a bare prompt about condos\n"                       # bare prompt
        "Comparison | Hybrid vs electric?"                        # intent + prompt
    )
    items = question_sets.parse_prompt_block(text, default_topic="Custom")
    by = {i["prompt"]: i for i in items}
    assert len(items) == 4  # header skipped
    assert by["What ingredients for dry skin?"]["id"] == "H01"
    assert by["What ingredients for dry skin?"]["intent"] == "Informational"
    assert by["Toyota vs BYD?"]["id"] == "A09"
    assert by["Just a bare prompt about condos"]["intent"] == "Custom"
    assert by["Hybrid vs electric?"]["intent"] == "Comparison"
    assert all(i["topic"] == "Custom" for i in items)


def test_simple_prompts_need_no_id_or_intent():
    text = "What is X?\nA vs B: which is better?\nLine with | pipe stays whole\n\n"
    items = question_sets.simple_prompts(text)
    assert len(items) == 3                                   # blank line skipped
    assert items[2]["prompt"] == "Line with | pipe stays whole"  # not split on '|'
    assert all(i["id"] == "" and i["intent"] == "Custom" and i["topic"] == "Custom" for i in items)


def test_items_for_tags_topic():
    items = question_sets.items_for(["Automotive"])
    assert len(items) == 12
    assert items[0]["id"] == "A01"
    assert all(i["topic"] == "Automotive" for i in items)


def test_demo_topic_study_aggregates_by_topic_and_intent():
    b = make_demo_topic_study()
    agg = b["aggregate"]
    assert set(agg["by_topic"]) == set(question_sets.TOPIC_SETS)  # all 3 topics present
    assert agg["patterns"]            # non-empty pattern strings
    assert agg["by_intent"]           # intent breakdown present
    for _t, info in agg["by_topic"].items():
        assert 0.0 <= info["cite_rate"] <= 1.0
        assert info["sample_sizes"]["n_candidates"] > 0
        # cited never exceeds candidates (strong-only labels)
        assert info["sample_sizes"]["n_cited"] <= info["sample_sizes"]["n_candidates"]
