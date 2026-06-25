"""Prompt Manifest parsing/matching + Intent → Source Type analysis."""

from __future__ import annotations

from src import brightdata
from src import chatgpt_pipeline as cgp
from src import demo
from src.similarity import SimilarityEngine


def test_parse_manifest_csv_and_expected():
    man = brightdata.parse_manifest(demo.SAMPLE_MANIFEST, "m.csv")
    assert man["n"] == 3 and man["has_expected"]
    e = {x["prompt_id"]: x for x in man["entries"]}
    assert e["P1"]["intent"] == "Product/Recommendation"
    assert e["P1"]["expected_source_types"] == ["review", "ecommerce", "official_brand"]


def test_apply_manifest_matches_and_attaches():
    run = demo.make_demo_brightdata()
    stats = brightdata.apply_manifest(run, demo.make_demo_manifest())
    assert stats["matched"] == 3 and stats["unmatched"] == 0
    assert run["has_intent"] is True
    assert "Regulation/Policy" in {r["intent"] for r in run["records"]}
    # intent propagated to source rows
    assert run["records"][0]["sources"][0]["intent"] == "Product/Recommendation"


def test_match_is_whitespace_and_case_insensitive():
    run = {"records": [{"record_id": "r1", "prompt": "  TOP   Hotels in New York? ", "sources": []}]}
    man = brightdata.parse_manifest("prompt_id,intent,prompt\nP1,Rec,Top hotels in New York\n", "m.csv")
    stats = brightdata.apply_manifest(run, man)
    assert stats["matched"] == 1 and run["records"][0]["intent"] == "Rec"


def test_unmatched_marked_when_prompt_differs():
    run = {"records": [{"record_id": "r1", "prompt": "totally different question", "sources": []}]}
    man = brightdata.parse_manifest("prompt_id,intent,prompt\nP1,Rec,Top hotels in New York\n", "m.csv")
    stats = brightdata.apply_manifest(run, man)
    assert stats["unmatched"] == 1 and run["records"][0]["intent"] == "(unmatched)"


def test_intent_analysis_outputs():
    run = demo.make_demo_brightdata()
    brightdata.apply_manifest(run, demo.make_demo_manifest())
    feats = cgp.build_features(run, {}, SimilarityEngine("lexical"))["features"]
    long = cgp.intent_source_long(feats)
    assert long and all(set(r) == {"intent", "source_type", "group", "n"} for r in long)
    summ = cgp.intent_summary(feats)
    assert any(r["intent"] == "Buyer Guide" for r in summ)
    ev = cgp.expected_vs_actual(feats)
    assert ev and all("coverage" in r for r in ev)
