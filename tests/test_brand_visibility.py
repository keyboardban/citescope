"""Non-branded Brand Visibility Audit: manifest fields, detection, record/intent/
source tables, position bands, safe wording, and zero-match exports."""

from __future__ import annotations

import json

from src import brand_visibility as bv
from src import brightdata, config, demo, report
from src import chatgpt_pipeline as cgp
from src.similarity import SimilarityEngine
from src.url_utils import domain as _dom, normalize_url


# --------------------------------------------------------------------------- #
# tiny builders (no scraping needed)
# --------------------------------------------------------------------------- #
def _src(url, cited, title="", description="", domain=None, position=1):
    return {"url": url, "normalized_url": normalize_url(url), "domain": domain or _dom(url),
            "title": title, "description": description, "cited_label": 1 if cited else 0,
            "source_group": "cited" if cited else "more_only", "source_position": position,
            "observed_rank": None, "source_id": url, "canonical_url": None, "final_url": None}


def _rec(record_id, prompt, sources, answer="", intent="Intent", topic="T",
         client_terms=None, competitor_terms=None, nonbranded=None):
    return {"record_id": record_id, "prompt": prompt, "answer_text": answer,
            "intent": intent, "topic": topic, "sources": sources,
            "client_terms": client_terms or [], "competitor_terms": competitor_terms or [],
            "prompt_is_nonbranded": nonbranded}


def _run(records, run_id="CG-test"):
    return {"run_id": run_id, "records": records}


def _records(run):
    return bv.compute_records(run, {}, [], [])


# --------------------------------------------------------------------------- #
# 1. Manifest parser reads client + competitor term fields
# --------------------------------------------------------------------------- #
def test_manifest_reads_brand_term_fields():
    man = demo.make_demo_brand_manifest()
    assert man["has_brand_terms"] is True
    e = {x["prompt_id"]: x for x in man["entries"]}
    assert "Siriraj" in e["BV1"]["client_terms"]
    assert "si.mahidol.ac.th" in e["BV1"]["client_terms"]          # domain-style term kept verbatim
    assert "Bumrungrad" in e["BV1"]["competitor_terms"]
    assert e["BV1"]["prompt_is_nonbranded"] is True
    assert e["BV1"]["visibility_goal"]
    # semicolon-separated terms split (and a Thai term preserved)
    assert "ศิริราช" in e["BV3"]["client_terms"]


# --------------------------------------------------------------------------- #
# 2. Non-branded prompt detection works (derived + explicit override)
# --------------------------------------------------------------------------- #
def test_nonbranded_detection_derived_and_explicit():
    run = _run([
        _rec("r1", "best hospitals in Bangkok", [], client_terms=["Siriraj"]),       # no brand in prompt
        _rec("r2", "is Siriraj good for heart surgery?", [], client_terms=["Siriraj"]),  # brand in prompt
        _rec("r3", "Siriraj review", [], client_terms=["Siriraj"], nonbranded=True),  # explicit override
    ])
    recs = {r["record_id"]: r for r in _records(run)}
    assert recs["r1"]["prompt_contains_client_brand"] is False and recs["r1"]["is_nonbranded_prompt"] is True
    assert recs["r2"]["prompt_contains_client_brand"] is True and recs["r2"]["is_nonbranded_prompt"] is False
    assert recs["r3"]["prompt_contains_client_brand"] is True and recs["r3"]["is_nonbranded_prompt"] is True


# --------------------------------------------------------------------------- #
# 3. Brand detector matches terms in answer text
# --------------------------------------------------------------------------- #
def test_detect_terms_in_answer_text():
    assert bv.detect_terms("Siriraj Hospital (SIPH) is recommended.", ["Siriraj", "SIPH", "Globex"]) == ["Siriraj", "SIPH"]
    run = _run([_rec("r1", "best heart hospital", [], answer="Both Siriraj and others are good.",
                     client_terms=["Siriraj"])])
    rec = _records(run)[0]
    assert rec["client_appeared_in_answer"] is True and rec["client_appeared"] is True


# --------------------------------------------------------------------------- #
# 4. Brand detector matches terms in source URL / domain / title
# --------------------------------------------------------------------------- #
def test_detect_terms_in_source_fields():
    # domain-style term matches a URL; brand word matches a title
    assert bv.detect_terms("https://www.siphhospital.com/heart", ["siphhospital.com"]) == ["siphhospital.com"]
    run = _run([_rec("r1", "best heart hospital", [
        _src("https://siphhospital.com/heart", cited=True, title="Heart Center"),
        _src("https://example.com/x", cited=False, title="About Siriraj cardiology"),
    ], client_terms=["siphhospital.com", "Siriraj"])])
    rec = _records(run)[0]
    assert rec["n_client_sources"] == 2 and rec["client_appeared_in_sources"] is True


# --------------------------------------------------------------------------- #
# 5. Brand detector separates client vs competitor (+ 'both')
# --------------------------------------------------------------------------- #
def test_separates_client_and_competitor():
    run = _run([_rec("r1", "best dealers", [
        _src("https://tccars.com/showroom", cited=True, title="TCC showroom"),
        _src("https://benzbkk.com/dealer", cited=False, title="Benz BKK"),
        _src("https://blog.com/x", cited=False, title="TCC vs Benz BKK comparison"),
        _src("https://neutral.com/y", cited=False, title="random page"),
    ], client_terms=["TCC", "tccars"], competitor_terms=["Benz BKK", "benzbkk.com"])])
    pages = bv.build_source_pages(run, [], {}, SimilarityEngine("lexical"), [], [])
    by_url = {r["url"]: r for r in pages}
    assert by_url["https://tccars.com/showroom"]["brand_match_group"] == "client"
    assert by_url["https://benzbkk.com/dealer"]["brand_match_group"] == "competitor"
    assert by_url["https://blog.com/x"]["brand_match_group"] == "both"
    assert "https://neutral.com/y" not in by_url                      # neutral excluded


# --------------------------------------------------------------------------- #
# 6. Record-level table KEEPS prompts where no brand appears (the denominator)
# --------------------------------------------------------------------------- #
def test_records_keep_prompts_with_no_brand():
    run = _run([
        _rec("r1", "totally unrelated question", [_src("https://x.com/a", cited=True)],
             client_terms=["Siriraj"], competitor_terms=["Bumrungrad"]),
        _rec("r2", "another unrelated question", [], client_terms=["Siriraj"]),
    ])
    recs = _records(run)
    assert len(recs) == 2                                             # all prompts kept
    assert all(r["client_appeared"] is False for r in recs)
    assert all(r["is_nonbranded_prompt"] is True for r in recs)


# --------------------------------------------------------------------------- #
# 7. Intent denominator = total NON-BRANDED prompts (not only matched prompts)
# --------------------------------------------------------------------------- #
def test_intent_denominator_is_nonbranded_prompts():
    cterms = ["Acme"]
    run = _run([
        _rec("r1", "best widget shops", [_src("https://acme.com/a", cited=True, title="Acme")],
             intent="Shopping", client_terms=cterms),               # client appears
        _rec("r2", "where to buy widgets", [_src("https://other.com/b", cited=True)],
             intent="Shopping", client_terms=cterms),               # client absent
        _rec("r3", "cheap widgets online", [], intent="Shopping", client_terms=cterms),  # client absent
    ])
    recs = _records(run)
    by_intent = {(r["topic"], r["intent"]): r for r in bv.summarize_by_intent(recs)}
    row = by_intent[("T", "Shopping")]
    assert row["nonbranded_prompts"] == 3                            # denominator = all 3 non-branded
    assert row["client_appeared_prompts"] == 1
    assert row["client_appeared_rate"] == round(1 / 3, 3)            # 1/3, NOT 1/1


# --------------------------------------------------------------------------- #
# 8. Source-level table includes ONLY client/competitor-matched sources
# --------------------------------------------------------------------------- #
def test_source_pages_only_brand_matched():
    run = _run([_rec("r1", "q", [
        _src("https://siphhospital.com/a", cited=True, title="client"),
        _src("https://bumrungrad.com/b", cited=False, title="competitor"),
        _src("https://reddit.com/c", cited=False, title="forum noise"),
    ], client_terms=["siphhospital.com"], competitor_terms=["bumrungrad.com"])])
    pages = bv.build_source_pages(run, [], {}, SimilarityEngine("lexical"), [], [])
    assert len(pages) == 2
    assert {p["brand_match_group"] for p in pages} == {"client", "competitor"}


# --------------------------------------------------------------------------- #
# 9. More-only is never labeled "rejected"/"ignored" in the report wording
# --------------------------------------------------------------------------- #
def test_more_only_never_called_rejected():
    d = demo.make_demo_brand_run()
    sim = SimilarityEngine("lexical")
    feats = cgp.build_features(d["run"], d["pages"], sim)["features"]
    brand = bv.build_brand_visibility(d["run"], feats, d["pages"], sim)
    md = report.brand_visibility_markdown(brand).lower()
    banned = ["rejected this source", "ignored this source", "chatgpt rejected",
              "chatgpt ignored", "proves why", "reveals the internal retrieval"]
    assert all(b not in md for b in banned)
    assert "shown but not cited" in md                              # safe framing present
    # the caveat only uses the *safe negation* ("does NOT mean ... rejected or ignored"), never an affirmative label
    cav = config.CAVEAT_BRAND_VISIBILITY.lower()
    assert "rejected or ignored" in cav and "shown but not cited" in cav


# --------------------------------------------------------------------------- #
# 10. Existing ChatGPT Bright Data parsing still works on the brand sample
# --------------------------------------------------------------------------- #
def test_brand_sample_is_valid_output_export():
    d = demo.make_demo_brand_run()
    run = d["run"]
    assert run["n_sources"] > 0 and run["looks_like_input"] is False
    assert run["has_intent"] is True and run["manifest"]["has_brand_terms"] is True
    # all 6 non-branded prompts matched the manifest
    assert run["manifest"]["matched"] == 6


# --------------------------------------------------------------------------- #
# 11. Existing Gemini + ChatGPT report paths unaffected by the new brand param
# --------------------------------------------------------------------------- #
def test_existing_report_paths_unaffected():
    grun = demo.make_demo_run()
    assert isinstance(report.markdown_report(grun), str)            # gemini report still builds
    crun = demo.make_demo_brightdata()
    feats = cgp.build_features(crun, {}, SimilarityEngine("lexical"))["features"]
    an = cgp.analyze(feats)
    # brand omitted entirely -> no brand section, no crash
    md = report.chatgpt_markdown_report(crun, an, feats)
    assert "ChatGPT Bright Data Source Audit" in md
    assert "Non-branded Brand Visibility Audit" not in md


# --------------------------------------------------------------------------- #
# 12. Exports generate even with ZERO brand matches (no terms anywhere)
# --------------------------------------------------------------------------- #
def test_exports_generate_with_zero_brand_matches():
    crun = demo.make_demo_brightdata()                              # no manifest, no brand terms
    sim = SimilarityEngine("lexical")
    feats = cgp.build_features(crun, {}, sim)["features"]
    brand = bv.build_brand_visibility(crun, feats, {}, sim)
    assert brand["has_terms"] is False and brand["source_pages"] == []
    assert len(brand["records"]) == crun["n_records"]               # records still present
    exporters = [report.brand_visibility_records_csv, report.brand_visibility_by_intent_csv,
                 report.brand_source_pages_csv, report.client_vs_competitor_visibility_csv,
                 report.cited_vs_moreonly_content_features_csv, report.content_features_by_position_band_csv]
    for fn in exporters:
        out = fn(brand)
        assert isinstance(out, str) and out                         # non-empty string, no crash
    # JSON bundle tolerates an empty-brand dict
    js = json.loads(report.chatgpt_analysis_json(crun, cgp.analyze(feats), feats, brand))
    assert "brand_visibility" not in js                             # not attached when no terms
