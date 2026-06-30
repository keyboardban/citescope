"""Confounder framework — registry, proxy derivation, audit, models E–H, wording,
and integration with the existing econometrics/report pipeline (Iteration L)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import chatgpt_pipeline as cgp, confounders as CF, demo, econometrics as E, report
from src.similarity import SimilarityEngine


def _df(seed=0, n_prompts=40, per=8):
    """A rich source-level frame (Thai + English, domains, URLs, content flags) — enough
    to derive every no-scrape proxy and fit the model ladder."""
    rng = np.random.default_rng(seed)
    doms = [f"d{i}.co.th" if i % 3 else f"shop{i}.com" for i in range(25)]
    rows = []
    for r in range(n_prompts):
        for j in range(per):
            d = str(rng.choice(doms)); pos = int(rng.integers(1, 21))
            faq, contact, sim = int(rng.random() < .4), int(rng.random() < .4), rng.random()
            p = 0.30 + 0.12 * faq - 0.06 * contact + 0.20 * sim - 0.10 * np.log1p(pos)
            rows.append(dict(
                record_id=f"p{r}", run_id="run1", domain=d,
                url=(f"https://{d}/product/{1000 + j}" if j % 4 == 0 else f"https://{d}/บริการ/{j}"),
                normalized_url=f"https://{d}/x{j}",
                prompt=("ราคา รักษา near me hospital" if r % 2 else "best clinic compare price"),
                title=("โรงพยาบาล ราคา" if r % 2 else "Best Clinic"), description="desc",
                cited=int(rng.random() < min(.97, max(.03, p))), source_position=pos, observed_rank=pos,
                has_faq=faq, has_contact_info=contact, has_location_info=int(rng.random() < .3),
                has_price_or_package=int(rng.random() < .3), product_page=int(rng.random() < .2),
                has_table=int(rng.random() < .3), has_bullets=int(rng.random() < .5),
                has_author=int(rng.random() < .3), has_reviewer=int(rng.random() < .2),
                has_schema=int(rng.random() < .2), freshness_days=float(rng.integers(0, 900)),
                word_count=int(rng.integers(100, 3000)), heading_count=int(rng.integers(0, 12)),
                source_type=str(rng.choice(["news", "forum", "review"])),
                page_type=str(rng.choice(["article", "product_page", "unknown"])),
                intent=str(rng.choice(["Buy", "Compare", "Info"])),
                institutional_official=False, brand_official_candidate=False,
                title_prompt_similarity=sim * .9 + rng.normal(0, .03),
                description_prompt_similarity=sim * .8 + rng.normal(0, .03),
                page_prompt_similarity=sim + rng.normal(0, .03),
                max_chunk_prompt_similarity=sim + .05 + rng.normal(0, .02)))
    return pd.DataFrame(rows)


# 1. registry: 32 entries, all required fields, CSV-ready
def test_registry_complete():
    assert len(CF.CONFOUNDER_REGISTRY) == 32
    needed = {"confounder", "category", "bias_mechanism", "availability_status", "available_now",
              "proxy_possible", "requires_external_data", "recommended_columns", "model_role",
              "diagnostic_role", "caveat"}
    assert all(needed <= set(c) for c in CF.CONFOUNDER_REGISTRY)
    rows = CF.registry_rows()
    assert len(rows) == 32 and rows[0]["confounder"]
    names = {c["confounder"] for c in CF.CONFOUNDER_REGISTRY}
    for must in ("writing_quality", "domain_authority", "brand_authority",
                 "source_panel_placement", "index_history", "scrape_success", "page_type"):
        assert must in names


# 2. missing columns never crash derivation
def test_derive_minimal_no_crash():
    out, notes = CF.derive_proxy_features(pd.DataFrame({"cited": [0, 1, 0, 1, 1, 0]}))
    assert isinstance(out, pd.DataFrame) and isinstance(notes, list)
    assert any(n["status"] == "skipped" for n in notes)


# 3. rich data → visibility / url / prompt / language / score proxies derived
def test_derive_proxies_on_rich_data():
    out, _ = CF.derive_proxy_features(_df())
    for c in ("domain_seen_count", "domain_citation_rate", "url_seen_count",
              "citescope_visibility_history_score", "url_path_depth", "prompt_has_price_terms",
              "is_thai_domain", "content_completeness_score", "answer_ready_score"):
        assert c in out.columns, c
    assert "index_history_score" not in out.columns           # never the (false) true-index-history name


# 4. all 7 audit tables generated + JSON-safe
def test_audit_tables():
    out, notes = CF.derive_proxy_features(_df())
    a = CF.confounder_audit(out, derivation_notes=notes)
    for k in ("registry", "feature_availability", "proxy_summary", "balance_by_cited",
              "correlation_matrix", "confounder_vif", "unmeasured_confounders"):
        assert isinstance(a[k], list) and a[k], k
    json.dumps(a, default=str)


# 5. domain / brand / index-history authority are labelled as proxies, not the true construct
def test_authority_labelled_proxy():
    reg = {c["confounder"]: c for c in CF.CONFOUNDER_REGISTRY}
    for name in ("domain_authority", "brand_authority", "index_history"):
        c = reg[name]
        assert c["availability_status"] == CF.PROXY and c["requires_external_data"]
        assert "not" in c["caveat"].lower()
    assert "visibility" in reg["index_history"]["caveat"].lower()   # CiteScope visibility, not true index history


# 6. post-output / answer-derived variables never enter the main OR sensitivity models
def test_post_output_excluded_from_models():
    mc = E.model_comparison(_df(), context="chatgpt")
    used = set()
    for grp in ("models", "confounder_models"):
        for m in mc.get(grp, []):
            for c in m["fit"].get("coefficients", []):
                used.add(c["name"].lower())
    for banned in E._ANSWER_SIM + ["brand_appeared_in_answer"]:
        assert not any(banned in u for u in used), banned


# 7. Models E–H are optional sensitivity models; the A/B/C/D headline ladder is intact
def test_eh_optional_not_headline():
    mc = E.model_comparison(_df(), context="chatgpt")
    assert sum(1 for m in mc["models"] if m["fit"].get("fitted")) == 4     # A/B/C/D unchanged
    assert mc["model_c"] and mc["model_c"].get("fitted")                   # headline stays Model C
    letters = {m["model_name"][0] for m in mc["confounder_models"]}
    assert letters <= {"E", "F", "G", "H"} and "A" not in letters


# 8. confounder exporters produce valid CSV / md with the right headers
def test_confounder_exporters():
    mc = E.model_comparison(_df(), context="chatgpt")
    for fn, head in [
        (report.econometrics_confounder_registry_csv, "confounder,category"),
        (report.econometrics_confounder_feature_availability_csv, "feature,confounder,available"),
        (report.econometrics_confounder_proxy_summary_csv, "confounder,proxy_features"),
        (report.econometrics_confounder_balance_by_cited_csv, "confounder_proxy,feature"),
        (report.econometrics_confounder_vif_csv, "confounder_proxy,vif"),
    ]:
        out = fn(mc)
        assert out.startswith(head) and len(out.splitlines()) > 1, fn.__name__
    assert report.econometrics_confounder_correlation_matrix_csv(mc).startswith("feature,")
    md = report.econometrics_unmeasured_confounders_md(mc)
    assert "Confounder & Proxy Audit" in md and "requires external data" in md.lower()


# 9. report wording: observable panel + proxy labels, no banned causal/AI-rejection language
def test_confounder_report_wording():
    mc = E.model_comparison(_df(), context="chatgpt")
    lines = []
    report._confounder_section(mc, lines.append)
    md = "\n".join(lines) + "\n" + report.econometrics_unmeasured_confounders_md(mc)
    low = md.lower()
    assert "observable source panel position" in low
    assert "citescope-observed visibility" in low                      # proxy label present
    assert "not an internal ai" in low or "not external seo" in low    # safe negation present
    for banned in ("causes citation", "ai rejected", "ai ignored", "caused by"):
        assert banned not in low


# 10. ChatGPT pipeline carries the confounder audit (integration; demos still work)
def test_pipeline_carries_confounder_audit():
    d = demo.make_demo_brand_run()
    feats = cgp.build_features(d["run"], d["pages"], SimilarityEngine("lexical"))["features"]
    an = cgp.analyze(feats)
    mc = an["regression_comparison"]
    assert (mc.get("confounder_audit") or {}).get("available") is True
    json.dumps(mc.get("confounder_audit"), default=str)
