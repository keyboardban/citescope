"""AI-ready reports: feature dictionary, correlation, intent breakdowns, embedded data."""

from __future__ import annotations

import json

from src import brightdata
from src import chatgpt_pipeline as cgp
from src import demo, report
from src.similarity import SimilarityEngine


def _cg():
    run = demo.make_demo_brightdata()
    brightdata.apply_manifest(run, demo.make_demo_manifest())
    feats = cgp.build_features(run, {}, SimilarityEngine("lexical"))["features"]
    return run, cgp.analyze(feats), feats


def test_chatgpt_report_is_ai_ready():
    run, an, feats = _cg()
    md = report.chatgpt_markdown_report(run, an, feats)
    for marker in ["Feature dictionary", "Feature ↔ citation correlation", "Intent × Source Type",
                   "Expected vs actual", "How to analyze this", "```csv", "Raw per-source data"]:
        assert marker in md, f"missing: {marker}"
    assert "more-only" in md and "AI rejected" not in md   # safe wording


def test_chatgpt_analysis_json_bundle():
    run, an, feats = _cg()
    b = json.loads(report.chatgpt_analysis_json(run, an, feats))
    assert b["summary"]["n_sources"] > 0
    assert b["feature_dictionary"] and b["sources"] and len(b["sources"]) > 0
    assert "intent_summary" in b and "correlation" in b


def test_chatgpt_dataset_csv_columns():
    _run, _an, feats = _cg()
    header = report.chatgpt_dataset_csv(feats).splitlines()[0]
    for col in ("cited", "intent", "source_type", "source_position", "domain"):
        assert col in header


def test_gemini_report_has_correlation_and_embedded_data():
    md = report.markdown_report(demo.make_demo_run())
    assert "Feature ↔ citation correlation" in md
    assert "Feature dictionary" in md and "```csv" in md
