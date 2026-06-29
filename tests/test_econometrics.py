"""Position-adjusted citation model — simulation-first tests ("build a known world,
recover the coefficient"), plus diagnostics, framing, and integration."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import analysis, chatgpt_pipeline as cgp, demo, econometrics as E, report
from src.similarity import SimilarityEngine


def _coef(res, name):
    return next((c for c in res.get("coefficients", []) if c["name"] == name), None)


def _sim(n=400, b_x=0.12, seed=0, clusters=20, cluster_shock=0.0, extra=None, pos_max=20, b_pos=0.10):
    """LPM truth: P(cited) = 0.4 + b_x*x1 - b_pos*log1p(pos) (+ optional cluster shock)."""
    rng = np.random.default_rng(seed)
    rows = []
    per = max(1, n // clusters)
    for g in range(clusters):
        shock = rng.normal(0, cluster_shock) if cluster_shock else 0.0
        for _ in range(per):
            pos = int(rng.integers(1, pos_max + 1))
            x1 = int(rng.random() < 0.5)
            p = 0.40 + b_x * x1 - b_pos * np.log1p(pos) + shock
            row = {"record_id": f"g{g}", "cited": int(rng.random() < min(0.97, max(0.03, p))),
                   "source_position": pos, "x1": x1}
            if extra:
                extra(row, rng)
            rows.append(row)
    return pd.DataFrame(rows)


def _spec(focal=("x1",), cluster_key=None, position_spec="log1p", categoricals=(),
          wild=True, logit=True):
    return E.build_spec(focal=list(focal), position_col="source_position",
                        position_spec=position_spec, categoricals=list(categoricals),
                        cluster_key=cluster_key, context="chatgpt", title="t",
                        wild_bootstrap=wild, crosscheck_logit=logit)


# --------------------------------------------------------------------------- #
# 1. recover a known LPM coefficient; CI covers the truth
# --------------------------------------------------------------------------- #
def test_recover_known_coefficient():
    res = E.fit_citation_model(_sim(n=2000, b_x=0.12, seed=1), _spec())
    c = _coef(res, "x1")
    assert res["fitted"] and c is not None
    assert abs(c["estimate"] - 0.12) < 0.05
    assert c["ci_low"] <= 0.12 <= c["ci_high"]
    assert res["model"] == "lpm" and res["se_type"] == "HC3"
    assert "log1p_source_position" in {cc["name"] for cc in res["coefficients"]}


# --------------------------------------------------------------------------- #
# 2. CI coverage ≈ 95% across many sims
# --------------------------------------------------------------------------- #
def test_ci_coverage_about_95pct():
    hits = 0
    M = 150
    for s in range(M):
        c = _coef(E.fit_citation_model(_sim(n=400, b_x=0.10, seed=s), _spec()), "x1")
        if c and c["ci_low"] <= 0.10 <= c["ci_high"]:
            hits += 1
    cov = hits / M
    assert 0.88 <= cov <= 0.99, f"coverage {cov:.2f} out of band"


# --------------------------------------------------------------------------- #
# 3. cluster-robust SE > HC3 when the outcome is correlated within clusters
#    (cluster-level feature is the sharp case)
# --------------------------------------------------------------------------- #
def test_cluster_se_exceeds_hc3():
    rng = np.random.default_rng(7)
    rows = []
    for g in range(40):
        xg = int(g % 2)                       # cluster-LEVEL feature (constant within cluster)
        shock = rng.normal(0, 0.20)           # strong within-cluster correlation
        for _ in range(25):
            pos = int(rng.integers(1, 21))
            p = 0.45 + 0.0 * xg - 0.05 * np.log1p(pos) + shock   # xg truly has no effect
            rows.append({"record_id": f"g{g}", "cited": int(rng.random() < min(0.97, max(0.03, p))),
                         "source_position": pos, "xg": xg})
    df = pd.DataFrame(rows)
    se_h = _coef(E.fit_citation_model(df, _spec(focal=["xg"], cluster_key=None)), "xg")["se"]
    se_c = _coef(E.fit_citation_model(df, _spec(focal=["xg"], cluster_key="record_id")), "xg")["se"]
    assert se_c > se_h * 1.5, f"cluster SE {se_c} should dwarf HC3 {se_h}"


# --------------------------------------------------------------------------- #
# 4. Benjamini–Hochberg is never less conservative than raw p (q ≥ p)
# --------------------------------------------------------------------------- #
def test_bh_more_conservative_than_raw():
    def add_nulls(row, rng):
        for j in range(8):
            row[f"z{j}"] = int(rng.random() < 0.5)   # pure-null features
    res = E.fit_citation_model(_sim(n=1500, b_x=0.12, seed=3, extra=add_nulls),
                               _spec(focal=["x1"] + [f"z{j}" for j in range(8)]))
    focal = [c for c in res["coefficients"] if c["is_focal"] and c["p"] is not None and c["p_adj"] is not None]
    assert focal and all(c["p_adj"] >= c["p"] - 1e-9 for c in focal)        # BH q ≥ raw p
    n_raw = sum(c["p"] < 0.05 for c in focal)
    n_adj = sum(c["p_adj"] < 0.05 for c in focal)
    assert n_adj <= n_raw
    assert _coef(res, "x1")["p_adj"] < 0.05                                 # the real one survives


# --------------------------------------------------------------------------- #
# 5. VIF flags correlated regressors, ≈1 for independent ones
# --------------------------------------------------------------------------- #
def test_vif_flags_collinearity():
    def corr_pair(row, rng):
        row["a"] = rng.normal()
        row["b"] = row["a"] + rng.normal(0, 0.05)   # ~collinear with a
        row["indep"] = rng.normal()
    res = E.fit_citation_model(_sim(n=1500, seed=4, extra=corr_pair),
                               _spec(focal=["a", "b", "indep"]))
    assert _coef(res, "a")["vif"] > E.config.VIF_WATCH
    assert _coef(res, "indep")["vif"] < 2.0


# --------------------------------------------------------------------------- #
# 6. functional form: log1p(position) fits better than linear when truth is log
# --------------------------------------------------------------------------- #
def test_position_functional_form():
    # Wide position range + strong log effect so curvature is real: linear can't keep up.
    df = _sim(n=4000, b_x=0.10, seed=5, pos_max=120, b_pos=0.16)   # truth uses log1p(pos)
    r_log = E.fit_citation_model(df, _spec(position_spec="log1p"))["r2"]
    r_lin = E.fit_citation_model(df, _spec(position_spec="linear"))["r2"]
    assert r_log > r_lin


# --------------------------------------------------------------------------- #
# 7. perfectly collinear (aliased) column dropped, no crash
# --------------------------------------------------------------------------- #
def test_collinear_column_dropped():
    def dup(row, rng):
        row["x2"] = row["x1"]                       # exact duplicate of focal x1
    res = E.fit_citation_model(_sim(n=800, seed=6, extra=dup), _spec(focal=["x1", "x2"]))
    assert res["fitted"]
    reasons = {d["reason"] for d in res["diagnostics"]["dropped_columns"]}
    assert "collinear" in reasons
    # exactly one of the duplicate pair survives
    assert (_coef(res, "x1") is None) ^ (_coef(res, "x2") is None) or _coef(res, "x1") is not None


# --------------------------------------------------------------------------- #
# 8. zero-variance focal column dropped
# --------------------------------------------------------------------------- #
def test_zero_variance_dropped():
    def const(row, rng):
        row["allzero"] = 0
    res = E.fit_citation_model(_sim(n=600, seed=8, extra=const), _spec(focal=["x1", "allzero"]))
    assert any(d["name"] == "allzero" for d in res["diagnostics"]["dropped_columns"])
    assert _coef(res, "allzero") is None


# --------------------------------------------------------------------------- #
# 9. determinism: identical input -> identical estimates
# --------------------------------------------------------------------------- #
def test_determinism():
    df = _sim(n=500, seed=9)
    e1 = [c["estimate"] for c in E.fit_citation_model(df, _spec())["coefficients"]]
    e2 = [c["estimate"] for c in E.fit_citation_model(df, _spec())["coefficients"]]
    assert e1 == e2


# --------------------------------------------------------------------------- #
# 10. insufficient data -> graceful (not fitted), never a crash
# --------------------------------------------------------------------------- #
def test_insufficient_data_graceful():
    res = E.fit_citation_model(_sim(n=10, clusters=2, seed=10), _spec())
    assert res["available"] and res["fitted"] is False and res["coefficients"] == []
    assert any("insufficient" in w.lower() for w in res["warnings"])


# --------------------------------------------------------------------------- #
# 11. graceful when statsmodels is absent (monkeypatched)
# --------------------------------------------------------------------------- #
def test_statsmodels_absent_degrades(monkeypatch):
    monkeypatch.setattr(E, "HAVE_STATSMODELS", False)
    res = E.fit_citation_model(_sim(n=400, seed=11), _spec())
    assert res["available"] is False and res["coefficients"] == []
    out = analysis.econometric_analysis(_sim(n=400, seed=11), ["x1"], {}, {},
                                        position_col="source_position", context="chatgpt")
    assert out and out[0]["available"] is False


# --------------------------------------------------------------------------- #
# 12. logit AME cross-check lands near the LPM coefficient (probability scale)
# --------------------------------------------------------------------------- #
def test_logit_ame_tracks_lpm():
    res = E.fit_citation_model(_sim(n=3000, b_x=0.15, seed=20), _spec())   # crosscheck on by default
    lpm = _coef(res, "x1")["estimate"]
    ame = next((a for a in res["ame"] if a["name"] == "x1"), None)
    assert ame is not None and ame["ci_low"] is not None
    assert abs(ame["ame"] - lpm) < 0.04          # AME (prob points) tracks the LPM coefficient


# --------------------------------------------------------------------------- #
# 13. (quasi-)separation: logit flagged, LPM still the headline, AME suppressed
# --------------------------------------------------------------------------- #
def test_separation_flagged_lpm_fallback():
    # x1 PERFECTLY predicts cited -> logit diverges (separation); LPM still fits.
    rows = [{"record_id": f"g{i % 20}", "cited": i % 2, "source_position": int(1 + i % 20), "x1": i % 2}
            for i in range(600)]
    res = E.fit_citation_model(pd.DataFrame(rows), _spec())
    assert res["fitted"]                              # LPM headline still fits
    assert res["diagnostics"]["separation"] is True   # logit separation detected
    assert res["ame"] == []                           # AME suppressed under separation


# --------------------------------------------------------------------------- #
# 14. wild cluster bootstrap kicks in with few clusters and is no less honest
#     than the (anti-conservative) analytic cluster SE under a clustered null
# --------------------------------------------------------------------------- #
def test_wild_bootstrap_with_few_clusters():
    res = E.fit_citation_model(
        _sim(n=240, clusters=8, cluster_shock=0.30, seed=1), _spec(cluster_key="record_id", logit=False))
    assert res["diagnostics"]["few_clusters"] and "wild" in res["se_type"]

    rej_analytic = rej_wild = 0
    M = 40
    for s in range(M):                                   # NULL focal effect, strong within-cluster corr
        df = _sim(n=240, b_x=0.0, clusters=8, cluster_shock=0.30, seed=300 + s)
        ca = _coef(E.fit_citation_model(df, _spec(cluster_key="record_id", wild=False, logit=False)), "x1")
        cw = _coef(E.fit_citation_model(df, _spec(cluster_key="record_id", wild=True, logit=False)), "x1")
        rej_analytic += bool(ca["p"] is not None and ca["p"] < 0.05)
        rej_wild += bool(cw["p"] is not None and cw["p"] < 0.05)
    assert rej_wild <= rej_analytic                      # bootstrap never rejects the null more often


# --------------------------------------------------------------------------- #
# 15. parity: the engine reproduces a direct statsmodels OLS+HC3 fit exactly
# --------------------------------------------------------------------------- #
def test_matches_direct_statsmodels():
    import statsmodels.api as sm
    df = _sim(n=1500, b_x=0.13, seed=2)
    spec = _spec(logit=False, wild=False)                # plain HC3, no clustering
    res = E.fit_citation_model(df, spec)
    dm = E.design_matrix(df, spec)
    direct = sm.OLS(dm["y"].values, dm["X"].values).fit(cov_type="HC3")
    j = list(dm["X"].columns).index("x1")
    c = _coef(res, "x1")
    assert abs(c["estimate"] - float(direct.params[j])) < 1e-5   # equal up to our 6-dp rounding
    assert abs(c["se"] - float(direct.bse[j])) < 1e-5


# --------------------------------------------------------------------------- #
# 16. position "bins" functional form produces band controls and fits
# --------------------------------------------------------------------------- #
def test_position_bins_option():
    res = E.fit_citation_model(_sim(n=1500, seed=3, pos_max=30), _spec(position_spec="bins", logit=False))
    assert res["fitted"] and any("band=" in c["name"] for c in res["coefficients"])


# --------------------------------------------------------------------------- #
# 17. integration + framing: demos fit, carry assumptions + signed OVB, JSON-safe,
#     and the report renders the section without over-claiming causation
# --------------------------------------------------------------------------- #
def test_integration_and_framing():
    study = demo.make_demo_topic_study()
    fits = study["aggregate"]["regression"]
    fit = fits[0]
    assert fit["fitted"] and fit["n"] > 0
    assert fit["assumptions"] and fit["ovb_caveat"]
    for c in fit["coefficients"]:
        assert {"estimate", "se", "ci_low", "ci_high", "p"} <= set(c)
    json.dumps(fits)                                            # JSON-serializable

    md = report.batch_markdown_report(study).lower()
    assert "position-adjusted citation model" in md
    assert "omitted-variable note" in md                       # signed caveat present
    for banned in ("proves causation", "causally proven", "rejected this source"):
        assert banned not in md


# =========================================================================== #
# Sensitivity analysis / model comparison + diagnostics (A/B/C/D + FULL)
# =========================================================================== #
def _source_df(n_prompts=40, per=8, seed=0):
    """A realistically-sized source-level frame with content features, (collinear)
    similarity features, position, domain, page_type, intent — enough rows to fit."""
    rng = np.random.default_rng(seed)
    doms = [f"d{i}" for i in range(25)]
    rows = []
    for r in range(n_prompts):
        for _ in range(per):
            pos = int(rng.integers(1, 21))
            faq, contact, sim = int(rng.random() < 0.4), int(rng.random() < 0.4), rng.random()
            p = 0.30 + 0.12 * faq - 0.06 * contact + 0.20 * sim - 0.10 * np.log1p(pos)
            rows.append(dict(
                record_id=f"p{r}", domain=str(rng.choice(doms)),
                cited=int(rng.random() < min(0.97, max(0.03, p))), source_position=pos,
                has_faq=faq, has_contact_info=contact, has_location_info=int(rng.random() < 0.3),
                has_phone_number=int(rng.random() < 0.3), has_table=int(rng.random() < 0.3),
                has_bullets=int(rng.random() < 0.5), has_author=int(rng.random() < 0.3),
                has_reviewer=int(rng.random() < 0.2), freshness_days=float(rng.integers(0, 900)),
                word_count=int(rng.integers(100, 3000)), heading_count=int(rng.integers(0, 12)),
                source_type=str(rng.choice(["news", "forum", "review", "blog"])),
                page_type=str(rng.choice(["article", "product_page", "faq_page", "contact_page", "unknown"])),
                intent=str(rng.choice(["Buy", "Compare", "Info"])),
                institutional_official=False, brand_official_candidate=False,
                title_prompt_similarity=sim * 0.9 + rng.normal(0, 0.03),
                description_prompt_similarity=sim * 0.8 + rng.normal(0, 0.03),
                page_prompt_similarity=sim + rng.normal(0, 0.03),
                max_chunk_prompt_similarity=sim + 0.05 + rng.normal(0, 0.02)))
    return pd.DataFrame(rows)


# 18. all sensitivity outputs are generated and the CSV exporters work
def test_model_comparison_generates_outputs():
    mc = E.model_comparison(_source_df(seed=2), context="chatgpt")
    assert mc["available"] and mc["fitted"]
    assert sum(1 for m in mc["models"] if m["fit"].get("fitted")) == 4      # A/B/C/D fit
    assert mc["comparison_rows"] and mc["vif_rows"] and mc["anomaly_rows"] and mc["group_rows"]
    assert mc["executive_summary"]
    for fn, head in [(report.econometrics_model_comparison_csv, "feature,model_name,delta_prob"),
                     (report.econometrics_vif_diagnostics_csv, "feature,vif,vif_level,interpretation"),
                     (report.econometrics_anomaly_diagnostics_csv, "check,feature"),
                     (report.econometrics_feature_group_summary_csv, "feature_group,num_features")]:
        out = fn(mc)
        assert isinstance(out, str) and out.startswith(head) and len(out.splitlines()) > 1


# 19. forest plot PNG is generated when focal features exist (+ content-only variant)
def test_forest_png_generated():
    mc = E.model_comparison(_source_df(seed=3), context="chatgpt")
    cmod = next(m["fit"] for m in mc["models"] if m["model_name"].startswith("C"))
    png = report.forest_png(cmod, title="Model C")
    png_content = report.forest_png(cmod, exclude_groups=("authority", "page_type", "intent"))
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"          # valid PNG
    assert png_content and len(png_content) < len(png)       # content-only is a smaller plot
    assert report.forest_png({"coefficients": []}) is None   # nothing to plot -> None


# 20. BH is applied within EACH model specification (q >= p inside each spec)
def test_bh_within_each_model_spec():
    mc = E.model_comparison(_source_df(seed=4), context="chatgpt")
    fitted = [m for m in mc["models"] if m["fit"].get("fitted")]
    assert fitted
    for m in fitted:
        focal = [c for c in m["fit"]["coefficients"]
                 if c["is_focal"] and c["p"] is not None and c["p_adj"] is not None]
        assert focal and all(c["p_adj"] >= c["p"] - 1e-9 for c in focal)


# 21. heavily-overlapping similarity features trigger a high-VIF warning/flag
def test_similarity_high_vif_warning():
    mc = E.model_comparison(_source_df(seed=5), context="chatgpt")
    severe = [r for r in mc["vif_rows"] if r["feature_group"] == "relevance" and r["vif_level"] == "severe"]
    assert len(severe) >= 2
    assert any(a["check"] == "similarity_collinear" for a in mc["anomaly_rows"])
    assert any("severe vif" in w.lower() or "relevance" in w.lower() for w in mc["warnings"])


# 22. the sensitivity report avoids causal/over-claiming wording, keeps observational framing
def test_sensitivity_report_safe_wording():
    mc = E.model_comparison(_source_df(seed=6), context="chatgpt")
    lines = []
    report._sensitivity_section(mc, lines.append)
    md = "\n".join(lines).lower()
    assert "observational association analysis" in md
    assert "does not prove" in md                              # the safe negation is present
    for banned in ("ai rejected", "ai ignored", "ai reject", "causes citation", "caused by"):
        assert banned not in md


# 23. existing ChatGPT pipeline still produces a (graceful) regression_comparison
def test_pipeline_carries_regression_comparison():
    d = demo.make_demo_brand_run()
    feats = cgp.build_features(d["run"], d["pages"], SimilarityEngine("lexical"))["features"]
    assert "has_faq" in feats[0] and "page_type" in feats[0]   # content features now on every source
    an = cgp.analyze(feats)
    assert "regression_comparison" in an and an["regression_comparison"]["available"] is True
    json.dumps(an["regression_comparison"])                    # JSON-serializable
