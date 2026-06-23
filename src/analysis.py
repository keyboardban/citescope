"""Cited vs non-cited comparison and headline metrics.

All comparisons are between *cited websites* and *non-cited reconstructed SERP
candidates*. Differences are observable associations, not causal explanations.

Features are split into:
- PRE_ANSWER_FEATURES  : observable before the answer exists (rank, query similarity,
  content stats). The "cleaner", non-circular signals.
- POST_OUTPUT_FEATURES : page/chunk similarity to the AI answer — may be partly
  circular because the answer can be generated from cited sources.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import NUMERIC_FEATURES

FEATURE_LABELS = {
    "serp_rank": "SERP rank (lower = higher)",
    "title_query_sim": "Title–query similarity",
    "snippet_query_sim": "Snippet–query similarity",
    "page_query_sim": "Page–query similarity",
    "max_chunk_query_sim": "Best chunk–query similarity",
    "page_output_sim": "Page–answer similarity",
    "max_chunk_output_sim": "Best chunk–answer similarity",
    "word_count": "Word count",
    "char_count": "Char count",
    "heading_count": "Heading count",
    "freshness_days": "Age (days)",
}

PRE_ANSWER_FEATURES = [
    "serp_rank", "title_query_sim", "snippet_query_sim", "page_query_sim",
    "max_chunk_query_sim", "word_count", "char_count", "heading_count", "freshness_days",
]
POST_OUTPUT_FEATURES = ["page_output_sim", "max_chunk_output_sim"]

FEATURE_PHASE = {f: "pre_answer" for f in PRE_ANSWER_FEATURES}
FEATURE_PHASE.update({f: "post_output" for f in POST_OUTPUT_FEATURES})


def features_df(features: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(features)
    if df.empty:
        return df
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _round(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else round(float(x), 4)


def group_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Mean & median of each numeric feature for cited vs non-cited candidates."""
    if df.empty or "cited" not in df.columns:
        return pd.DataFrame()
    rows = []
    cited = df[df["cited"] == 1]
    noncited = df[df["cited"] == 0]
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        cm, nm = cited[col].mean(skipna=True), noncited[col].mean(skipna=True)
        cmd, nmd = cited[col].median(skipna=True), noncited[col].median(skipna=True)
        rows.append({
            "feature": FEATURE_LABELS.get(col, col),
            "key": col,
            "phase": FEATURE_PHASE.get(col, "pre_answer"),
            "cited_mean": _round(cm),
            "noncited_mean": _round(nm),
            "cited_median": _round(cmd),
            "noncited_median": _round(nmd),
            "n_cited": int(cited[col].notna().sum()),
            "n_noncited": int(noncited[col].notna().sum()),
            "delta": None if (pd.isna(cm) or pd.isna(nm)) else round(float(cm - nm), 4),
        })
    return pd.DataFrame(rows)


def source_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "source_type" not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby("source_type")
        .agg(candidates=("cited", "size"), cited=("cited", "sum"))
        .reset_index()
    )
    g["non_cited"] = g["candidates"] - g["cited"]
    g["cite_rate"] = (g["cited"] / g["candidates"]).round(3)
    return g.sort_values("candidates", ascending=False).reset_index(drop=True)


def official_compare(df: pd.DataFrame) -> dict:
    """Cite-rate for institutional-official vs brand-official-candidate vs other."""
    if df.empty:
        return {}

    def stats(sub: pd.DataFrame) -> dict:
        return {
            "candidates": int(len(sub)),
            "cited": int(sub["cited"].sum()) if not sub.empty else 0,
            "cite_rate": round(float(sub["cited"].mean()), 3) if not sub.empty else 0.0,
        }

    out = {}
    inst = df.get("institutional_official")
    brand = df.get("brand_official_candidate")
    if inst is not None:
        out["institutional_official"] = stats(df[inst == True])  # noqa: E712
    if brand is not None:
        out["brand_official_candidate"] = stats(df[brand == True])  # noqa: E712
    if inst is not None and brand is not None:
        out["other"] = stats(df[(inst != True) & (brand != True)])  # noqa: E712
    return out


def _normalize_recall(recall: dict) -> dict:
    """Accept nested {strict:{...}} or legacy flat {'10':...}; return nested."""
    if recall and "strict" not in recall and any(k in recall for k in ("5", "10", "20", "50")):
        return {"strict": recall, "canonical": recall, "domain_inclusive": recall}
    return recall or {"strict": {}, "canonical": {}, "domain_inclusive": {}}


def summary_metrics(run: dict) -> dict:
    """Headline numbers for the Overview cards."""
    g = run.get("gemini") or {}
    serp = run.get("serp") or {}
    scrape = run.get("scrape") or {}
    matching = run.get("matching") or {}
    pages = scrape.get("pages") or {}
    recall = _normalize_recall(matching.get("recall") or {})
    strict = recall.get("strict", {})
    domain_inc = recall.get("domain_inclusive", {})
    df = features_df(run.get("features") or [])

    n_cited = int(df["cited"].sum()) if not df.empty and "cited" in df else 0
    n_weak = (int(df["weak_domain_match"].sum()) if not df.empty and "weak_domain_match" in df
              else len(matching.get("weak_candidate_ids", []) or []))
    return {
        "n_queries": len(g.get("search_queries", []) or []),
        "n_citations": matching.get("n_citations", len(g.get("citations", []) or [])),
        "n_candidates": len(df) if not df.empty else len(serp.get("candidates", []) or []),
        "n_scraped": sum(1 for p in pages.values() if p.get("status") == "success"),
        "n_cited_candidates": n_cited,
        "n_weak_candidates": n_weak,
        "recall": recall,
        "recall_strict_5": float(strict.get("5", 0.0)),
        "recall_strict_10": float(strict.get("10", 0.0)),
        "recall_strict_20": float(strict.get("20", 0.0)),
        "recall_strict_50": float(strict.get("50", 0.0)),
        "recall_domain_10": float(domain_inc.get("10", 0.0)),
        "unmatched": len(matching.get("unmatched", []) or []),
    }


def correlation_with_citation(df: pd.DataFrame) -> pd.DataFrame:
    """Point-biserial-style correlation of each numeric feature with `cited`."""
    if df.empty or "cited" not in df.columns:
        return pd.DataFrame()
    rows = []
    y = df["cited"].astype(float)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna()
        phase = FEATURE_PHASE.get(col, "pre_answer")
        if mask.sum() < 3 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
            rows.append({"feature": FEATURE_LABELS.get(col, col), "key": col, "phase": phase, "corr": None})
            continue
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        rows.append({"feature": FEATURE_LABELS.get(col, col), "key": col, "phase": phase, "corr": round(r, 3)})
    return pd.DataFrame(rows)


def length_sim_correlation(df: pd.DataFrame) -> dict:
    """Correlation of page length with page-answer similarity (length-bias check)."""
    out: dict[str, float | None] = {}
    if df.empty or "page_output_sim" not in df.columns:
        return out
    y = pd.to_numeric(df["page_output_sim"], errors="coerce")
    for length_col in ("word_count", "char_count"):
        if length_col not in df.columns:
            continue
        x = pd.to_numeric(df[length_col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() >= 3 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
            out[length_col] = round(float(np.corrcoef(x[mask], y[mask])[0, 1]), 3)
        else:
            out[length_col] = None
    return out
