"""Export helpers: CSV datasets, JSON, and Markdown / HTML reports."""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .analysis import (
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

    a("## Limitations\n")
    a("- We observe only what the Gemini API exposes; the true internal retrieval set is unknown.\n"
      "- The reconstructed SERP can differ from the AI's results by time, region, personalization, ranking.\n"
      "- Post-output similarity may be partly circular; prefer pre-answer signals and rank.\n"
      "- Source-type / official flags and brand-candidate detection are heuristics.\n"
      "- Single-run results are anecdotal — use Batch mode for aggregated associations.\n")
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


def chatgpt_markdown_report(run: dict, an: dict, intent_summary: list | None = None,
                            expected: list | None = None) -> str:
    s = (an or {}).get("summary", {})
    lines: list[str] = []
    a = lines.append
    a(f"# ChatGPT Bright Data Source Audit — {run.get('run_id','')}\n")
    a(f"_Generated {run.get('created_at','')} · source file: {run.get('source_file_name','')}_\n")
    a(f"> {config.CHATGPT_INTRO}\n")
    a(f"> {config.CAVEAT_MORE_ONLY}\n")

    a("## Sample sizes\n")
    a(_md_table(pd.DataFrame([
        {"metric": "Records / prompts", "value": s.get("n_records", 0)},
        {"metric": "Sources (total)", "value": s.get("n_sources", 0)},
        {"metric": "Cited sources", "value": s.get("n_cited", 0)},
        {"metric": "More-only sources", "value": s.get("n_more_only", 0)},
        {"metric": "Scraped OK", "value": s.get("n_scraped", 0)},
        {"metric": "Scrape success rate", "value": s.get("scrape_success_rate", 0.0)},
    ])))

    gc = pd.DataFrame(an.get("group_compare") or [])
    cols = ["feature", "cited_mean", "noncited_mean", "cited_median", "noncited_median", "delta"]
    if not gc.empty:
        a("## Cited vs more-only — pre-answer signals (non-circular)\n")
        a(_md_table(gc[gc["phase"] == "pre_answer"][cols]))
        a("## Cited vs more-only — post-output overlap (may be circular)\n")
        a(f"> {config.CAVEAT_ANSWER_CG}\n")
        a(_md_table(gc[gc["phase"] == "post_output"][cols]))

    sb = pd.DataFrame(an.get("source_breakdown") or [])
    if not sb.empty:
        a("## Source-type breakdown\n")
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

    if intent_summary:
        a("## Intent → Source Type (per-intent cited composition)\n")
        a(_md_table(pd.DataFrame(intent_summary)))
    if expected:
        a("## Expected vs actual cited source types (heuristic)\n")
        a(_md_table(pd.DataFrame(expected)[["intent", "expected", "cited_found", "expected_missing", "coverage"]]))

    a("## Limitations\n")
    a("- Observable source placement only — not ChatGPT's full internal retrieval set.\n"
      "- More-only sources were not 'rejected'; they were surfaced but not marked cited.\n"
      "- Post-output similarity may be partly circular; prefer pre-answer signals.\n"
      "- No SERP recall@K here; any ordering is `source_position`/`observed_rank`, not Google rank.\n")
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
