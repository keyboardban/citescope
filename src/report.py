"""Export helpers: CSV datasets, JSON, and Markdown / HTML reports."""

from __future__ import annotations

import json

import pandas as pd

from . import config
from .analysis import (
    features_df,
    group_compare,
    official_compare,
    source_breakdown,
    summary_metrics,
)
from .matching import unique_candidates


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
# narrative reports
# --------------------------------------------------------------------------- #
def _md_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_no data_\n"
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join([head, sep, *rows]) + "\n"


def markdown_report(run: dict) -> str:
    m = summary_metrics(run)
    df = features_df(run.get("features") or [])
    prompt = (run.get("inputs") or {}).get("prompt", "")
    g = run.get("gemini") or {}
    matching = run.get("matching") or {}

    recall = matching.get("recall") or {}
    recall_df = pd.DataFrame(
        [{"K": k, "citation_recall@K": recall.get(str(k), 0.0)} for k in (5, 10, 20, 50)]
    )
    rates = matching.get("rates") or {}
    rates_df = pd.DataFrame([{"match_type": k, "share": v} for k, v in rates.items()])

    lines: list[str] = []
    a = lines.append
    a(f"# AI Search Citation Audit — {run.get('run_id','')}\n")
    a(f"_Generated {run.get('created_at','')}_\n")
    a("> **Black-box observational audit.** Cited websites come from Gemini's grounding "
      "metadata; candidates come from an independently **reconstructed SERP** (Apify). "
      "This reconstruction is a parallel candidate set, not the AI's internal results. "
      "Non-cited candidates were **not** \"rejected\". Similarity is a semantic overlap "
      "proxy, not proof of use.\n")

    a("## Prompt\n")
    a(f"> {prompt}\n")

    a("## Headline metrics\n")
    metric_df = pd.DataFrame([
        {"metric": "Observed search queries", "value": m["n_queries"]},
        {"metric": "Citations (distinct)", "value": m["n_citations"]},
        {"metric": "SERP candidates", "value": m["n_candidates"]},
        {"metric": "Pages scraped", "value": m["n_scraped"]},
        {"metric": "Cited candidates matched", "value": m["n_cited_candidates"]},
        {"metric": "Unmatched citations", "value": m["unmatched"]},
        {"metric": "citation_recall@10", "value": m["recall_10"]},
        {"metric": "citation_recall@20", "value": m["recall_20"]},
    ])
    a(_md_table(metric_df))

    a("## Observed search queries\n")
    for q in g.get("search_queries", []):
        tag = " _(fallback)_" if q.get("is_fallback") else ""
        a(f"- {q.get('query','')}{tag}")
    a("")

    a("## Citation recall@K\n")
    a(_md_table(recall_df))
    a("## Match-type distribution\n")
    a(_md_table(rates_df))

    a("## Cited vs non-cited reconstructed candidates\n")
    a(_md_table(group_compare(df)[["feature", "cited_mean", "noncited_mean", "delta",
                                   "n_cited", "n_noncited"]] if not df.empty else pd.DataFrame()))

    a("## Source-type breakdown\n")
    a(_md_table(source_breakdown(df)))

    off = official_compare(df)
    if off:
        a("## Official vs non-official\n")
        a(_md_table(pd.DataFrame([{"group": k, **v} for k, v in off.items()])))

    unmatched = matching.get("unmatched") or []
    if unmatched:
        a("## Unmatched citations (not found in reconstructed top-K)\n")
        for u in unmatched:
            a(f"- {u}")
        a("")

    a("## Limitations\n")
    a("- We observe only what the Gemini API exposes; the true internal retrieval set is unknown.\n"
      "- The reconstructed SERP can differ from the AI's results by time, region, "
      "personalization, and ranking changes.\n"
      "- Chunk/page similarity is a semantic overlap proxy, not evidence the model read the text.\n"
      "- Source-type and official-source flags are heuristics.\n")
    return "\n".join(lines)


def html_report(run: dict) -> str:
    df = features_df(run.get("features") or [])
    m = summary_metrics(run)
    prompt = (run.get("inputs") or {}).get("prompt", "")
    matching = run.get("matching") or {}
    recall = matching.get("recall") or {}

    def tbl(d: pd.DataFrame) -> str:
        return d.to_html(index=False, border=0, classes="t") if d is not None and not d.empty else "<p><i>no data</i></p>"

    cards = "".join(
        f'<div class="card"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in [
            ("queries", m["n_queries"]), ("citations", m["n_citations"]),
            ("candidates", m["n_candidates"]), ("scraped", m["n_scraped"]),
            ("recall@10", m["recall_10"]), ("recall@20", m["recall_20"]),
        ]
    )
    recall_df = pd.DataFrame([{"K": k, "recall@K": recall.get(str(k), 0.0)} for k in (5, 10, 20, 50)])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Citation Audit — {run.get('run_id','')}</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Inter,sans-serif;margin:40px auto;max-width:960px;color:#1e2330;background:#f7f8fc}}
h1{{font-size:24px}} h2{{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:6px}}
.note{{background:#eef2ff;border-left:4px solid #4f46e5;padding:12px 16px;border-radius:8px;font-size:14px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#fff;border:1px solid #e8eaf2;border-radius:12px;padding:14px 18px;min-width:120px;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
.card .v{{font-size:26px;font-weight:700;color:#4f46e5}} .card .k{{font-size:12px;color:#6b7280;text-transform:uppercase}}
table.t{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}}
table.t th,table.t td{{border:1px solid #e8eaf2;padding:8px 10px;text-align:left}}
table.t th{{background:#f3f4f6}}
blockquote{{color:#374151;border-left:3px solid #c7d2fe;padding-left:12px}}
</style></head><body>
<h1>AI Search Citation Audit — {run.get('run_id','')}</h1>
<p class="note"><b>Black-box observational audit.</b> Cited websites come from Gemini's grounding
metadata; candidates come from an independently reconstructed SERP (Apify). Non-cited candidates
were not "rejected"; similarity is a semantic overlap proxy, not proof of use.</p>
<h2>Prompt</h2><blockquote>{prompt}</blockquote>
<div class="cards">{cards}</div>
<h2>Citation recall@K</h2>{tbl(recall_df)}
<h2>Cited vs non-cited reconstructed candidates</h2>{tbl(group_compare(df))}
<h2>Source-type breakdown</h2>{tbl(source_breakdown(df))}
</body></html>"""


def write_all(run: dict) -> dict[str, str]:
    """Write CSV/JSON/MD/HTML to data/exports and return their paths."""
    rid = run.get("run_id", "run")
    paths = {
        "features_csv": config.write_export(f"{rid}_features.csv", features_csv(run)),
        "serp_csv": config.write_export(f"{rid}_serp.csv", serp_csv(run)),
        "matches_csv": config.write_export(f"{rid}_matches.csv", matches_csv(run)),
        "run_json": config.write_export(f"{rid}_run.json", run_json(run)),
        "report_md": config.write_export(f"{rid}_report.md", markdown_report(run)),
        "report_html": config.write_export(f"{rid}_report.html", html_report(run)),
    }
    return paths
