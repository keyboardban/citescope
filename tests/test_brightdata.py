"""ChatGPT Bright Data parser, source labeling, dedup, CSV, and features."""

from __future__ import annotations

import csv as csvmod
import io
import json

from src import brightdata
from src import chatgpt_pipeline as cgp
from src.similarity import SimilarityEngine
from src.url_utils import normalize_url


def _run(records):
    return brightdata.parse_run(json.dumps(records), "f.json")


def test_parse_array_and_missing_fields():
    run = _run([
        {"prompt": "Q1", "answer_text": "A1", "citations": [{"url": "https://a.com/x", "cited": True}]},
        {"url": "https://chatgpt.com/?q=hello%20world"},  # no prompt -> from URL; no sources
    ])
    assert run["n_records"] == 2
    r0, r1 = run["records"]
    assert r0["prompt"] == "Q1" and r0["answer_text"] == "A1"
    assert len(r0["sources"]) == 1 and r0["sources"][0]["cited_label"] == 1
    assert r1["prompt"] == "hello world"          # parsed from url ?q=
    assert r1["sources"] == []


def test_source_labeling_and_cited_wins():
    run = _run([{
        "prompt": "P",
        "citations": [{"url": "https://x.com/a", "cited": True},
                      {"url": "https://y.com/b", "cited": False}],
        "search_sources_more": [{"url": "https://z.com/c"},
                                {"url": "https://x.com/a"}],   # dup of cited -> cited wins
    }])
    srcs = {s["normalized_url"]: s for s in run["records"][0]["sources"]}
    a = srcs[normalize_url("https://x.com/a")]
    assert a["cited_label"] == 1 and a["source_group"] == "cited"
    assert len(a["appearances"]) == 2                          # both appearances preserved
    assert srcs[normalize_url("https://y.com/b")]["cited_label"] == 0
    z = srcs[normalize_url("https://z.com/c")]
    assert z["cited_label"] == 0 and z["source_origin"] == "search_sources_more"


def test_url_dedup_tracking_params():
    run = _run([{
        "prompt": "P",
        "citations": [{"url": "https://e.com/p?utm_source=g", "cited": False}],
        "search_sources_more": [{"url": "https://e.com/p"}],
    }])
    assert len(run["records"][0]["sources"]) == 1              # deduped despite tracking param


def test_links_attached_fallback_only_without_citations():
    with_links = _run([{"prompt": "P", "links_attached": ["https://l.com/1"]}])
    s = with_links["records"][0]["sources"]
    assert len(s) == 1 and s[0]["cited_label"] == 1 and s[0]["source_origin"] == "links_attached"

    with_cit = _run([{"prompt": "P", "citations": [{"url": "https://c.com/1", "cited": True}],
                      "links_attached": ["https://l.com/1"]}])
    urls = {x["url"] for x in with_cit["records"][0]["sources"]}
    assert "https://l.com/1" not in urls                       # citations present -> no fallback


def test_csv_import_with_json_cells():
    cits = json.dumps([{"url": "https://a.com/x", "cited": True},
                       {"url": "https://b.com/y", "cited": False}])
    buf = io.StringIO()
    w = csvmod.writer(buf)
    w.writerow(["prompt", "answer_text", "citations"])
    w.writerow(["Hello", "Ans", cits])
    run = brightdata.parse_run(buf.getvalue(), "f.csv")
    assert run["n_records"] == 1
    s = run["records"][0]["sources"]
    assert len(s) == 2 and sum(x["cited_label"] for x in s) == 1


def test_detects_input_prompt_file():
    # prompts only (no citations/sources) -> flagged as an input file
    run = _run([{"prompt": "Q1"}, {"prompt": "Q2"}])
    assert run["n_sources"] == 0 and run["looks_like_input"] is True
    assert any("input" in w.lower() for w in run["warnings"])
    # has sources -> not flagged
    run2 = _run([{"prompt": "Q", "citations": [{"url": "https://a.com", "cited": True}]}])
    assert run2["looks_like_input"] is False


def test_feature_table_one_row_per_source_no_scrape():
    run = _run([{"prompt": "P", "citations": [
        {"url": "https://a.com/x", "cited": True, "title": "A"},
        {"url": "https://b.com/y", "cited": False, "title": "B"}]}])
    feat = cgp.build_features(run, {}, SimilarityEngine("lexical"))
    rows = feat["features"]
    by = {r["url"]: r for r in rows}
    assert len(rows) == 2
    assert by["https://a.com/x"]["cited"] == 1 and by["https://b.com/y"]["cited"] == 0
    assert by["https://a.com/x"]["page_answer_similarity"] is None   # no scrape -> None, no crash
    an = cgp.analyze(rows)
    assert an["summary"]["n_sources"] == 2 and an["summary"]["n_cited"] == 1
