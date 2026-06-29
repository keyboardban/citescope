"""Export helpers: CSV datasets, JSON, and Markdown / HTML reports."""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .analysis import (
    correlation_with_citation,
    features_df,
    group_compare,
    length_sim_correlation,
    official_compare,
    source_breakdown,
    summary_metrics,
)


# --------------------------------------------------------------------------- #
# tabular exports
# --------------------------------------------------------------------------- #
def features_csv(run: dict) -> str:
    df = features_df(run.get("features") or [])
    return df.to_csv(index=False) if not df.empty else "no features\n"


def gemini_dataset_csv(run: dict) -> str:
    """Compact per-candidate dataset (key correlation columns only)."""
    df = features_df(run.get("features") or [])
    if df.empty:
        return "no features\n"
    cols = [c for c in _GEM_DATASET_COLS if c in df.columns]
    return df[cols].to_csv(index=False)


def serp_csv(run: dict) -> str:
    cands = (run.get("serp") or {}).get("candidates") or []
    return pd.DataFrame(cands).to_csv(index=False) if cands else "no serp candidates\n"


def matches_csv(run: dict) -> str:
    matches = (run.get("matching") or {}).get("matches") or []
    return pd.DataFrame(matches).to_csv(index=False) if matches else "no matches\n"


def run_json(run: dict) -> str:
    return json.dumps(run, indent=2, default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, sep, *rows]) + "\n"


def _recall_table(recall: dict) -> pd.DataFrame:
    ks = (5, 10, 20, 50)
    return pd.DataFrame([{
        "K": k,
        "strict": recall.get("strict", {}).get(str(k), 0.0),
        "canonical": recall.get("canonical", {}).get(str(k), 0.0),
        "domain_inclusive (weak)": recall.get("domain_inclusive", {}).get(str(k), 0.0),
    } for k in ks])


# --------------------------------------------------------------------------- #
# AI-ready helpers: feature dictionary, embedded raw data, analysis guide
# --------------------------------------------------------------------------- #
_CG_DATASET_COLS = [
    "record_id", "prompt_id", "intent", "topic", "cited", "source_group", "source_type",
    "institutional_official", "brand_official_candidate", "source_position", "observed_rank",
    "freshness_days", "word_count", "title_prompt_similarity", "description_prompt_similarity",
    "page_prompt_similarity", "page_answer_similarity", "domain",
]
_GEM_DATASET_COLS = [
    "candidate_id", "cited", "match_type", "strong_match", "serp_rank", "source_type",
    "institutional_official", "brand_official_candidate", "title_query_sim", "snippet_query_sim",
    "page_query_sim", "page_output_sim", "max_chunk_output_sim", "word_count", "freshness_days", "domain",
]


def _embed_csv(csv_text: str, max_rows: int = 1500) -> str:
    lines = csv_text.splitlines()
    note = ""
    if len(lines) > max_rows + 1:
        lines = [lines[0]] + lines[1:max_rows + 1]
        note = f"\n_(showing first {max_rows} rows; full data via the CSV download)_\n"
    return "```csv\n" + "\n".join(lines).rstrip("\n") + "\n```\n" + note


def _data_dict_chatgpt() -> list[dict]:
    g = {"label": "label", "meta": "meta", "pre": "pre-answer", "post": "post-output (may be circular)", "id": "id"}
    return [
        {"column": "cited", "group": g["label"], "meaning": "1 = ChatGPT cited; 0 = more-only (shown-but-not-cited)"},
        {"column": "source_group", "group": g["label"], "meaning": "cited | more_only"},
        {"column": "intent", "group": g["meta"], "meaning": "user intent (from Prompt Manifest)"},
        {"column": "topic", "group": g["meta"], "meaning": "topic (from Prompt Manifest)"},
        {"column": "source_type", "group": g["pre"], "meaning": "heuristic class: news/forum/review/ecommerce/government/reference/blog/..."},
        {"column": "institutional_official", "group": g["pre"], "meaning": "gov/edu/mil/int domain (bool)"},
        {"column": "brand_official_candidate", "group": g["pre"], "meaning": "looks like the entity's own site (heuristic, bool)"},
        {"column": "source_position", "group": g["pre"], "meaning": "order in the source panel — NOT Google rank"},
        {"column": "observed_rank", "group": g["pre"], "meaning": "rank from search_sources if present — NOT Google rank"},
        {"column": "freshness_days", "group": g["pre"], "meaning": "page age in days (if a date was found)"},
        {"column": "word_count", "group": g["pre"], "meaning": "scraped page word count (if scraped)"},
        {"column": "title_prompt_similarity", "group": g["pre"], "meaning": "title↔prompt overlap proxy [0–1]"},
        {"column": "description_prompt_similarity", "group": g["pre"], "meaning": "snippet↔prompt overlap [0–1]"},
        {"column": "page_prompt_similarity", "group": g["pre"], "meaning": "page↔prompt overlap [0–1] (if scraped)"},
        {"column": "page_answer_similarity", "group": g["post"], "meaning": "page↔ChatGPT-answer overlap [0–1] — may be circular"},
        {"column": "domain", "group": g["id"], "meaning": "host domain"},
    ]


def _data_dict_gemini() -> list[dict]:
    return [
        {"column": "cited", "group": "label", "meaning": "1 = matched a Gemini citation (strong); 0 = non-cited candidate"},
        {"column": "match_type", "group": "label", "meaning": "exact/normalized/final_redirect/canonical/amp/domain_only/no_match"},
        {"column": "serp_rank", "group": "pre-answer", "meaning": "best rank in the reconstructed SERP (lower = higher)"},
        {"column": "source_type", "group": "pre-answer", "meaning": "heuristic site class"},
        {"column": "institutional_official", "group": "pre-answer", "meaning": "gov/edu/mil/int (bool)"},
        {"column": "brand_official_candidate", "group": "pre-answer", "meaning": "entity's own site (heuristic, bool)"},
        {"column": "title_query_sim", "group": "pre-answer", "meaning": "title↔query overlap [0–1]"},
        {"column": "snippet_query_sim", "group": "pre-answer", "meaning": "snippet↔query overlap [0–1]"},
        {"column": "page_query_sim", "group": "pre-answer", "meaning": "page↔query overlap [0–1] (if scraped)"},
        {"column": "page_output_sim", "group": "post-output (may be circular)", "meaning": "page↔answer overlap [0–1]"},
        {"column": "max_chunk_output_sim", "group": "post-output (may be circular)", "meaning": "best chunk↔answer overlap [0–1]"},
        {"column": "freshness_days", "group": "pre-answer", "meaning": "page age in days (if a date was found)"},
        {"column": "domain", "group": "id", "meaning": "host domain"},
    ]


def _analysis_guide(kind: str) -> str:
    target = "more-only" if kind == "chatgpt" else "non-cited SERP candidate"
    pos = "source_position / observed_rank" if kind == "chatgpt" else "serp_rank"
    lines = [
        "## How to analyze this (for an AI / analyst)",
        f"Target = **`cited`** (1/0); each row is one source. Look for associations with cited vs **{target}**:",
        "",
        "1. **Per-feature correlation with `cited`** — use the correlation table + the raw CSV below. Prefer **pre-answer** features (non-circular).",
        ("2. **Cite-rate by `source_type`, and by `intent` × `source_type`** — do different intents cite different site types?"
         if kind == "chatgpt" else "2. **Cite-rate by `source_type`** — which site types are cited more?"),
        f"3. **Position effect** — are cited sources higher up (`{pos}` lower) than {target}?",
        "4. **Official/brand effect** — higher cite-rate for `institutional_official` / `brand_official_candidate`?",
    ]
    if kind == "chatgpt":
        lines.append("5. **Expected vs actual** — where a manifest gave `expected_source_types`, what coverage and what unexpected types were cited?")
    lines += [
        "",
        "**Caveats (please keep wording safe):**",
        f"- '{target}' = *shown-but-not-cited*, **not rejected**.",
        "- These are **observable associations, not causal** claims about the AI's internal retrieval.",
        "- **Post-output** similarity (page/chunk ↔ answer) may be **circular** — the answer can be generated from cited sources.",
        ("- `source_position`/`observed_rank` are panel order, **not** Google SERP rank."
         if kind == "chatgpt" else "- The reconstructed SERP can differ from the AI's internal results."),
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# position-adjusted citation model (regression) rendering
# --------------------------------------------------------------------------- #
def _regression_table(fit: dict) -> pd.DataFrame:
    rows = []
    for c in fit.get("coefficients", []):
        rows.append({
            "feature": c["label"], "Δ prob": c["estimate"], "se": c["se"],
            "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p": c["p"],
            "p_adj(BH)": c.get("p_adj"), "VIF": c.get("vif"),
            "focal": "✓" if c.get("is_focal") else "",
        })
    return pd.DataFrame(rows)


def _regression_section(fits, a, *, header: str = "Position-adjusted citation model") -> None:
    """Render a list of fit_results (or {group: fit} dict) into the markdown report."""
    if isinstance(fits, dict):
        fits = list(fits.values())
    fits = [f for f in (fits or []) if f]
    if not fits:
        return
    a(f"## {header} (LPM — cautious effect estimates)\n")
    a(f"> {config.CAVEAT_REGRESSION}\n")
    for f in fits:
        if not f.get("available", True):
            a(f"_{(f.get('warnings') or ['statsmodels not installed'])[0]}_\n")
            continue
        if not f.get("fitted"):
            a(f"_{f.get('title', 'model')}: {(f.get('warnings') or ['not fitted'])[0]}_\n")
            continue
        meta = f"n={f['n']}"
        meta += (f", {f['n_clusters']} clusters ({f['se_type']} SE)"
                 if f.get("n_clusters") else f", {f['se_type']} SE")
        if f.get("r2") is not None:
            meta += f", R²={f['r2']}"
        a(f"**{f.get('title', 'model')}** — {meta}. Coefficients are Δ probability of citation per feature, "
          "holding the others (incl. position) fixed.\n")
        a(_md_table(_regression_table(f)))
        if f.get("ame"):
            a("_Logit AME cross-check (Δ probability; should track the LPM coefficients above):_\n")
            a(_md_table(pd.DataFrame([
                {"feature": r["label"], "AME": r["ame"], "se": r["se"],
                 "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]} for r in f["ame"]])))
        if f.get("ovb_caveat"):
            a(f"> **Omitted-variable note (signed).** {f['ovb_caveat']}\n")
        for asm in f.get("assumptions", []):
            a(f"> {asm}\n")
        for w in f.get("warnings", []):
            a(f"> ⚠️ {w}\n")


# --------------------------------------------------------------------------- #
# single-run report
# --------------------------------------------------------------------------- #
def markdown_report(run: dict) -> str:
    m = summary_metrics(run)
    df = features_df(run.get("features") or [])
    prompt = (run.get("inputs") or {}).get("prompt", "")
    g = run.get("gemini") or {}
    matching = run.get("matching") or {}
    gc = group_compare(df)

    lines: list[str] = []
    a = lines.append
    a(f"# AI Search Citation Audit — {run.get('run_id','')}\n")
    a(f"_Generated {run.get('created_at','')}_\n")
    a("> **Black-box observational audit.** Cited websites come from Gemini's grounding "
      "metadata; candidates come from an independently **reconstructed SERP** (Apify) — a "
      "parallel candidate set, not the AI's internal results. **Non-cited SERP candidates** "
      "were not \"rejected\". Similarity is a *semantic overlap proxy*.\n")

    a("## Prompt\n")
    a(f"> {prompt}\n")

    a("## Headline metrics\n")
    a(_md_table(pd.DataFrame([
        {"metric": "Observed search queries", "value": m["n_queries"]},
        {"metric": "Citations (distinct)", "value": m["n_citations"]},
        {"metric": "SERP candidates", "value": m["n_candidates"]},
        {"metric": "Pages scraped", "value": m["n_scraped"]},
        {"metric": "Cited candidates (strong)", "value": m["n_cited_candidates"]},
        {"metric": "Weak domain-only matches", "value": m["n_weak_candidates"]},
        {"metric": "Unmatched citations", "value": m["unmatched"]},
        {"metric": "strict_recall@10", "value": m["recall_strict_10"]},
    ])))

    a("## Observed search queries\n")
    for q in g.get("search_queries", []):
        tag = " _(fallback)_" if q.get("is_fallback") else ""
        a(f"- {q.get('query','')}{tag}")
    a("")

    a("## Citation recall@K (three variants)\n")
    a(_md_table(_recall_table(m["recall"])))
    a(f"> {config.CAVEAT_RECALL}\n")
    a("## Match-type counts\n")
    a(_md_table(pd.DataFrame([{"match_type": k, "count": v}
                              for k, v in (matching.get("rate_counts") or {}).items()])))

    cols = ["feature", "cited_mean", "noncited_mean", "cited_median", "noncited_median", "delta"]
    if not gc.empty:
        a("## Pre-answer signals (non-circular)\n")
        a("_Observable before the answer exists: rank, query similarity, content stats._\n")
        a(_md_table(gc[gc["phase"] == "pre_answer"][cols]))
        a("## Post-output semantic overlap (may be partly circular)\n")
        a(f"> {config.CAVEAT_POST_OUTPUT}\n")
        a(_md_table(gc[gc["phase"] == "post_output"][cols]))
        lc = length_sim_correlation(df)
        if lc:
            a(f"_Length vs page–answer similarity correlation: {lc}. {config.CAVEAT_LENGTH}_\n")

    a("## Source-type breakdown\n")
    a(_md_table(source_breakdown(df)))

    off = official_compare(df)
    if off:
        a("## Official signals (institutional vs brand-candidate)\n")
        a(_md_table(pd.DataFrame([{"group": k, **v} for k, v in off.items()])))

    unmatched = matching.get("unmatched") or []
    if unmatched:
        a("## Unmatched citations (not recovered in reconstructed top-K)\n")
        for u in unmatched:
            a(f"- {u}")
        a("")

    _regression_section((run.get("analysis") or {}).get("regression"), a)

    corr = correlation_with_citation(df)
    if not corr.empty:
        a("## Feature ↔ citation correlation (point-biserial, unadjusted)\n")
        a("_Quick **unadjusted** screen (no controls, no error bar). The position-adjusted model above "
          "is the rigorous read; small |r| with few rows is noisy._\n")
        a(_md_table(corr[["feature", "phase", "corr"]]))

    a("## Feature dictionary\n")
    a(_md_table(pd.DataFrame(_data_dict_gemini())))
    a(_analysis_guide("gemini"))

    a("## Limitations\n")
    a("- We observe only what the Gemini API exposes; the true internal retrieval set is unknown.\n"
      "- The reconstructed SERP can differ from the AI's results by time, region, personalization, ranking.\n"
      "- Post-output similarity may be partly circular; prefer pre-answer signals and rank.\n"
      "- Source-type / official flags and brand-candidate detection are heuristics.\n"
      "- Single-run results are anecdotal — use Batch mode for aggregated associations.\n")

    if not df.empty:
        a("## Raw per-candidate data (CSV) — for your own correlation analysis\n")
        a("_One row per SERP candidate. Columns are defined in the feature dictionary above._\n")
        a(_embed_csv(gemini_dataset_csv(run)))
    return "\n".join(lines)


def html_report(run: dict) -> str:
    df = features_df(run.get("features") or [])
    m = summary_metrics(run)
    prompt = (run.get("inputs") or {}).get("prompt", "")
    gc = group_compare(df)

    def tbl(d: pd.DataFrame) -> str:
        return d.to_html(index=False, border=0, classes="t") if d is not None and not d.empty else "<p><i>no data</i></p>"

    cards = "".join(
        f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in [
            ("queries", m["n_queries"]), ("citations", m["n_citations"]),
            ("candidates", m["n_candidates"]), ("scraped", m["n_scraped"]),
            ("strict recall@10", m["recall_strict_10"]),
            ("domain-incl@10", m["recall_domain_10"]),
        ]
    )
    cols = ["feature", "cited_mean", "noncited_mean", "delta"]
    pre = gc[gc["phase"] == "pre_answer"][cols] if not gc.empty else pd.DataFrame()
    post = gc[gc["phase"] == "post_output"][cols] if not gc.empty else pd.DataFrame()

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Citation Audit — {run.get('run_id','')}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Inter,sans-serif;margin:40px auto;max-width:980px;color:#1e2330;background:#f7f8fc}}
h1{{font-size:24px}} h2{{margin-top:30px;border-bottom:2px solid #eee;padding-bottom:6px}}
.note{{background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 16px;border-radius:8px;font-size:14px}}
.warn{{background:#fef3c7;border-left:4px solid #f59e0b;padding:12px 16px;border-radius:8px;font-size:14px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border:1px solid #e8eaf2;border-radius:12px;padding:14px 18px;min-width:120px}}
.card .v{{font-size:24px;font-weight:700;color:#4f46e5}} .card .k{{font-size:12px;color:#6b7280;text-transform:uppercase}}
table.t{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
table.t th,table.t td{{border:1px solid #e8eaf2;padding:8px 10px;text-align:left}} table.t th{{background:#f3f4f6}}
blockquote{{color:#374151;border-left:3px solid #c7d2fe;padding-left:12px}}
</style></head><body>
<h1>AI Search Citation Audit — {run.get('run_id','')}</h1>
<p class="note"><b>Black-box observational audit.</b> Cited websites come from grounding metadata;
candidates come from a reconstructed SERP. Non-cited candidates were not "rejected".</p>
<h2>Prompt</h2><blockquote>{prompt}</blockquote>
<div class="cards">{cards}</div>
<h2>Citation recall@K</h2>{tbl(_recall_table(m["recall"]))}
<h2>Pre-answer signals (non-circular)</h2>{tbl(pre)}
<h2>Post-output semantic overlap</h2>
<p class="warn">{config.CAVEAT_POST_OUTPUT}</p>{tbl(post)}
<h2>Source-type breakdown</h2>{tbl(source_breakdown(df))}
</body></html>"""


# --------------------------------------------------------------------------- #
# batch report
# --------------------------------------------------------------------------- #
def batch_markdown_report(batch: dict) -> str:
    agg = batch.get("aggregate") or {}
    s = agg.get("sample_sizes") or {}
    lines: list[str] = []
    a = lines.append
    a(f"# Batch Citation Audit — {batch.get('batch_id','')}\n")
    a(f"_Generated {batch.get('created_at','')}_\n")
    a(f"> {config.CAVEAT_BATCH}\n")
    a("## Sample sizes\n")
    a(_md_table(pd.DataFrame([
        {"metric": "Prompts", "value": batch.get("n_prompts", 0)},
        {"metric": "Successful runs", "value": s.get("n_runs_ok", 0)},
        {"metric": "Candidate rows", "value": s.get("n_candidates", 0)},
        {"metric": "Cited (strong)", "value": s.get("n_cited", 0)},
        {"metric": "Citations", "value": s.get("n_citations", 0)},
        {"metric": "Scraped pages", "value": s.get("n_scraped", 0)},
    ])))
    a("## Recall@K (averaged across runs)\n")
    a(_md_table(_recall_table(agg.get("recall") or {})))
    gs = pd.DataFrame(agg.get("group_stats") or [])
    if not gs.empty:
        a("## Cited vs non-cited (pooled across prompts)\n")
        a("_Mann-Whitney U p-value and 95% bootstrap CI for the median difference._\n")
        keep = ["feature", "phase", "cited_median", "noncited_median", "median_diff",
                "mwu_p", "ci_low", "ci_high", "n_cited", "n_noncited"]
        a(_md_table(gs[[c for c in keep if c in gs.columns]]))

    _regression_section(agg.get("regression"), a, header="Position-adjusted citation model (pooled)")

    sb = pd.DataFrame(agg.get("source_breakdown") or [])
    if not sb.empty:
        a("## Source-type breakdown (pooled)\n")
        a(_md_table(sb))

    patterns = agg.get("patterns") or []
    if patterns:
        a("## Observable patterns\n")
        for p in patterns:
            a(f"- {p}")
        a("")

    by_topic = agg.get("by_topic") or {}
    if by_topic:
        a("## By topic\n")
        a(_md_table(pd.DataFrame([{
            "topic": t,
            "candidates": info.get("sample_sizes", {}).get("n_candidates", 0),
            "cited": info.get("sample_sizes", {}).get("n_cited", 0),
            "cite_rate": info.get("cite_rate", 0.0),
            "strict_recall@10": info.get("recall", {}).get("strict", {}).get("10", 0.0),
        } for t, info in by_topic.items()])))

    by_intent = agg.get("by_intent") or {}
    if by_intent:
        a("## By intent\n")
        a(_md_table(pd.DataFrame([{"intent": k, **v} for k, v in by_intent.items()])))

    a("## Limitations\n")
    a("- Observable associations across runs, not causal evidence.\n"
      "- Reconstructed SERP ≠ the AI's internal results; post-output similarity may be circular.\n")
    return "\n".join(lines)


def batch_features_csv(batch: dict) -> str:
    feats = batch.get("features") or []
    return pd.DataFrame(feats).to_csv(index=False) if feats else "no features\n"


# --------------------------------------------------------------------------- #
# ChatGPT Bright Data exports
# --------------------------------------------------------------------------- #
def chatgpt_sources_csv(run: dict) -> str:
    rows = []
    for rec in run.get("records", []):
        for s in rec.get("sources", []):
            rows.append({
                "record_id": rec.get("record_id"), "prompt": rec.get("prompt", ""),
                "intent": rec.get("intent"), "topic": rec.get("topic"),
                "url": s["url"], "normalized_url": s["normalized_url"], "domain": s.get("domain"),
                "title": s.get("title"), "description": s.get("description"),
                "source_group": s.get("source_group"), "cited_label": s["cited_label"],
                "source_origin": s.get("source_origin"), "source_position": s.get("source_position"),
                "observed_rank": s.get("observed_rank"), "date_published": s.get("date_published"),
            })
    return pd.DataFrame(rows).to_csv(index=False) if rows else "no sources\n"


def chatgpt_features_csv(features: list[dict]) -> str:
    return pd.DataFrame(features).to_csv(index=False) if features else "no features\n"


def chatgpt_intent_csv(intent_long: list[dict]) -> str:
    """Intent × (group, source_type) count matrix as CSV."""
    if not intent_long:
        return "no intent data\n"
    df = pd.DataFrame(intent_long)
    piv = df.pivot_table(index="intent", columns=["group", "source_type"],
                         values="n", aggfunc="sum", fill_value=0)
    return piv.to_csv()


def chatgpt_dataset_csv(features: list[dict]) -> str:
    """Compact per-source dataset (key correlation columns only)."""
    if not features:
        return "no features\n"
    df = pd.DataFrame(features)
    cols = [c for c in _CG_DATASET_COLS if c in df.columns]
    return df[cols].to_csv(index=False)


# --------------------------------------------------------------------------- #
# Non-branded Brand Visibility Audit exports
# --------------------------------------------------------------------------- #
def _rows_csv(rows: list[dict] | None, empty_msg: str) -> str:
    return pd.DataFrame(rows).to_csv(index=False) if rows else empty_msg


def brand_visibility_records_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("records"), "no records\n")


def brand_visibility_by_intent_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("by_intent"), "no intent rows\n")


def brand_source_pages_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("source_pages"), "no brand-matched source pages\n")


def client_vs_competitor_visibility_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("client_vs_competitor"), "no comparison rows\n")


def cited_vs_moreonly_content_features_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("cited_vs_moreonly"), "no content-feature comparison\n")


def content_features_by_position_band_csv(brand: dict) -> str:
    return _rows_csv((brand or {}).get("by_position_band"), "no position-band comparison\n")


def brand_visibility_markdown(brand: dict) -> str:
    """The 'Non-branded Brand Visibility Audit' report section (observable wording)."""
    if not brand:
        return ""
    s = brand.get("summary") or {}
    lines: list[str] = []
    a = lines.append
    a("## Non-branded Brand Visibility Audit\n")
    a("This section analyzes non-branded prompts that do not directly mention the client brand. "
      "It measures whether the client or competitor appears in the observable ChatGPT/Bright Data "
      "answer or source panel.\n")
    a("This does not reveal ChatGPT's internal retrieval process. It only studies observable source "
      "and citation behavior.\n")
    a(f"> {config.CAVEAT_BRAND_VISIBILITY}\n")
    a("**Detected brand terms** — client: "
      + (", ".join(brand.get("client_terms") or []) or "_none_")
      + " · competitor: " + (", ".join(brand.get("competitor_terms") or []) or "_none_") + "\n")

    a("### Overall visibility (denominator = non-branded prompts)\n")
    a(_md_table(pd.DataFrame([
        {"metric": "Total prompts", "value": s.get("total_prompts")},
        {"metric": "Non-branded prompts", "value": s.get("nonbranded_prompts")},
        {"metric": "Client appeared rate", "value": s.get("client_appeared_rate")},
        {"metric": "Client cited rate", "value": s.get("client_cited_rate")},
        {"metric": "Client more-only rate", "value": s.get("client_more_only_rate")},
        {"metric": "Competitor appeared rate", "value": s.get("competitor_appeared_rate")},
        {"metric": "Competitor cited rate", "value": s.get("competitor_cited_rate")},
        {"metric": "Client − competitor cited delta", "value": s.get("client_vs_competitor_cited_delta")},
    ])))

    bi = pd.DataFrame(brand.get("by_intent") or [])
    if not bi.empty:
        a("### Visibility by intent\n")
        cols = ["topic", "intent", "nonbranded_prompts", "client_appeared_rate", "client_cited_rate",
                "client_more_only_rate", "competitor_appeared_rate", "competitor_cited_rate",
                "competitor_more_only_rate", "client_vs_competitor_cited_delta"]
        a(_md_table(bi[[c for c in cols if c in bi.columns]]))

    ex = brand.get("examples") or {}

    def _ex_block(title: str, key: str) -> None:
        items = ex.get(key) or []
        if not items:
            return
        a(f"### {title}\n")
        for it in items:
            a(f"- _{it.get('intent') or ''}_ — {it.get('prompt') or ''}")
        a("")

    _ex_block("Example prompts that triggered client citation", "client_cited")
    _ex_block("Example prompts where a competitor was cited but the client did not appear",
              "competitor_cited_client_absent")
    _ex_block("Example prompts where the client appeared only as more-only (shown but not cited)",
              "client_more_only")
    _ex_block("Example prompts where neither client nor competitor appeared", "neither_appeared")

    cv = pd.DataFrame(brand.get("cited_vs_moreonly") or [])
    if not cv.empty:
        a("### Content features associated with cited vs more-only pages (all brand-matched)\n")
        keep = ["feature", "cited_mean", "more_only_mean", "delta", "n_cited", "n_more_only"]
        a(_md_table(cv[cv["group"] == "all"][keep]))
        a("_Positive delta = feature more common/higher among **cited** brand pages; negative = more common "
          "among **more-only** (shown-but-not-cited) pages. Boolean features are shown as rates._\n")

    pb = pd.DataFrame(brand.get("by_position_band") or [])
    if not pb.empty:
        a("### Position-controlled content comparison (brand_match_group = all)\n")
        keep = ["position_band", "feature", "cited_mean", "more_only_mean", "delta", "n_cited", "n_more_only"]
        a(_md_table(pb[pb["brand_match_group"] == "all"][keep]))
        a("_Cited vs more-only compared **within** similar source-position bands, so differences are not "
          "merely position effects. `source_position` is panel order, not Google rank._\n")

    _regression_section(brand.get("position_adjusted"), a, header="Position-adjusted content model")
    return "\n".join(lines)


def chatgpt_analysis_json(run: dict, an: dict, features: list[dict] | None = None,
                          brand: dict | None = None) -> str:
    """Structured bundle (summary + comparisons + correlation + intent + raw rows) for an AI to parse."""
    bundle = {
        "run_id": run.get("run_id"), "source_file": run.get("source_file_name"),
        "manifest": run.get("manifest"), "summary": (an or {}).get("summary"),
        "group_compare": (an or {}).get("group_compare"),
        "correlation": (an or {}).get("correlation"),
        "regression": (an or {}).get("regression"),
        "source_breakdown": (an or {}).get("source_breakdown"),
        "official": (an or {}).get("official"),
        "top_domains_cited": (an or {}).get("top_domains_cited"),
        "top_domains_more": (an or {}).get("top_domains_more"),
        "feature_dictionary": _data_dict_chatgpt(),
        "caveats": [config.CHATGPT_INTRO, config.CAVEAT_MORE_ONLY, config.CAVEAT_ANSWER_CG],
    }
    if features and run.get("has_intent"):
        from . import chatgpt_pipeline as cgp
        bundle["intent_source_long"] = cgp.intent_source_long(features)
        bundle["intent_summary"] = cgp.intent_summary(features)
        if (run.get("manifest") or {}).get("has_expected"):
            bundle["expected_vs_actual"] = cgp.expected_vs_actual(features)
    if brand and brand.get("has_terms"):
        bundle["brand_visibility"] = {
            "summary": brand.get("summary"),
            "by_intent": brand.get("by_intent"),
            "client_vs_competitor": brand.get("client_vs_competitor"),
            "examples": brand.get("examples"),
            "cited_vs_moreonly_content_features": brand.get("cited_vs_moreonly"),
            "content_features_by_position_band": brand.get("by_position_band"),
            "position_adjusted_regression": brand.get("position_adjusted"),
            "source_pages": brand.get("source_pages"),
            "records": brand.get("records"),
            "client_terms": brand.get("client_terms"),
            "competitor_terms": brand.get("competitor_terms"),
        }
        bundle["caveats"].append(config.CAVEAT_BRAND_VISIBILITY)
    if features:
        bundle["sources"] = [{k: r.get(k) for k in _CG_DATASET_COLS if k in r} for r in features]
    return json.dumps(bundle, indent=2, default=str, ensure_ascii=False)


def chatgpt_markdown_report(run: dict, an: dict, features: list[dict] | None = None,
                            brand: dict | None = None) -> str:
    s = (an or {}).get("summary", {})
    lines: list[str] = []
    a = lines.append
    a(f"# ChatGPT Bright Data Source Audit — {run.get('run_id','')}\n")
    a(f"_Generated {run.get('created_at','')} · source file: {run.get('source_file_name','')}_\n")
    a(f"> {config.CHATGPT_INTRO}\n")
    a(f"> {config.CAVEAT_MORE_ONLY}\n")
    man = run.get("manifest") or {}
    if man.get("applied"):
        a(f"_Prompt Manifest applied: {man.get('matched')}/{man.get('total')} records matched → intent/topic attached._\n")

    a("## Sample sizes\n")
    a(_md_table(pd.DataFrame([
        {"metric": "Records / prompts", "value": s.get("n_records", 0)},
        {"metric": "Sources (total)", "value": s.get("n_sources", 0)},
        {"metric": "Cited sources", "value": s.get("n_cited", 0)},
        {"metric": "More-only sources", "value": s.get("n_more_only", 0)},
        {"metric": "Scraped OK", "value": s.get("n_scraped", 0)},
        {"metric": "Scrape success rate", "value": s.get("scrape_success_rate", 0.0)},
    ])))

    a("## Feature dictionary\n")
    a(_md_table(pd.DataFrame(_data_dict_chatgpt())))

    gc = pd.DataFrame(an.get("group_compare") or [])
    cols = ["feature", "cited_mean", "noncited_mean", "cited_median", "noncited_median", "delta"]
    if not gc.empty:
        a("## Cited vs more-only — pre-answer signals (non-circular)\n")
        a(_md_table(gc[gc["phase"] == "pre_answer"][cols]))
        a("## Cited vs more-only — post-output overlap (may be circular)\n")
        a(f"> {config.CAVEAT_ANSWER_CG}\n")
        a(_md_table(gc[gc["phase"] == "post_output"][cols]))

    _regression_section((an or {}).get("regression"), a)

    corr = pd.DataFrame(an.get("correlation") or [])
    if not corr.empty:
        a("## Feature ↔ citation correlation (point-biserial, unadjusted)\n")
        a("_Quick **unadjusted** screen (no controls, no error bar) — the position-adjusted model above is "
          "the rigorous read; small |r| with few rows is noisy._\n")
        a(_md_table(corr[["feature", "phase", "corr"]]))

    sb = pd.DataFrame(an.get("source_breakdown") or [])
    if not sb.empty:
        a("## Source-type breakdown (cite-rate per type)\n")
        a(_md_table(sb))

    off = an.get("official") or {}
    if off:
        a("## Official signals (institutional vs brand-candidate)\n")
        a(_md_table(pd.DataFrame([{"group": k, **v} for k, v in off.items()])))

    tc = pd.DataFrame(an.get("top_domains_cited") or [])
    tm = pd.DataFrame(an.get("top_domains_more") or [])
    if not tc.empty:
        a("## Top domains — cited\n")
        a(_md_table(tc))
    if not tm.empty:
        a("## Top domains — more-only\n")
        a(_md_table(tm))

    # ---- Intent → Source Type (needs a manifest applied) ----
    if features and run.get("has_intent"):
        from . import chatgpt_pipeline as cgp
        ldf = pd.DataFrame(cgp.intent_source_long(features))
        if not ldf.empty:
            counts = ldf.pivot_table(index="intent", columns="source_type", values="n", aggfunc="sum", fill_value=0)
            a("## Intent × Source Type — counts (all surfaced)\n")
            a(_md_table(counts.reset_index()))
            pct = counts.div(counts.sum(axis=1).replace(0, 1), axis=0).round(3)
            a("## Intent × Source Type — row % within intent\n")
            a(_md_table(pct.reset_index()))
            cited = ldf[ldf["group"] == "cited"]
            if not cited.empty:
                a("## Cited source types by intent\n")
                a(_md_table(cited.pivot_table(index="intent", columns="source_type",
                                              values="n", aggfunc="sum", fill_value=0).reset_index()))
            more = ldf[ldf["group"] == "more_only"]
            if not more.empty:
                a("## More-only (shown-but-not-cited) source types by intent\n")
                a(_md_table(more.pivot_table(index="intent", columns="source_type",
                                             values="n", aggfunc="sum", fill_value=0).reset_index()))
        summ = cgp.intent_summary(features)
        if summ:
            a("## Per-intent cited composition\n")
            a(_md_table(pd.DataFrame(summ)))
        if man.get("has_expected"):
            ev = cgp.expected_vs_actual(features)
            if ev:
                a("## Expected vs actual cited source types (heuristic)\n")
                a(_md_table(pd.DataFrame(ev)))

    # ---- Non-branded Brand Visibility Audit (needs brand terms in the manifest) ----
    if brand and brand.get("has_terms"):
        a(brand_visibility_markdown(brand))

    a(_analysis_guide("chatgpt"))

    a("## Limitations\n")
    a("- Observable source placement only — not ChatGPT's full internal retrieval set.\n"
      "- More-only sources were not 'rejected'; they were surfaced but not marked cited.\n"
      "- Post-output similarity may be partly circular; prefer pre-answer signals.\n"
      "- No SERP recall@K here; any ordering is `source_position`/`observed_rank`, not Google rank.\n")

    if features:
        a("## Raw per-source data (CSV) — for your own correlation analysis\n")
        a("_One row per source; `cited` is the target. Columns are in the feature dictionary above._\n")
        a(_embed_csv(chatgpt_dataset_csv(features)))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# write-all
# --------------------------------------------------------------------------- #
def write_all(run: dict) -> dict[str, str]:
    """Write CSV/JSON/MD/HTML to data/exports and return their paths."""
    rid = run.get("run_id", "run")
    return {
        "features_csv": config.write_export(f"{rid}_features.csv", features_csv(run)),
        "serp_csv": config.write_export(f"{rid}_serp.csv", serp_csv(run)),
        "matches_csv": config.write_export(f"{rid}_matches.csv", matches_csv(run)),
        "run_json": config.write_export(f"{rid}_run.json", run_json(run)),
        "report_md": config.write_export(f"{rid}_report.md", markdown_report(run)),
        "report_html": config.write_export(f"{rid}_report.html", html_report(run)),
    }
