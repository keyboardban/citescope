"""Cited vs non-cited comparison and headline metrics.

All comparisons are between *cited websites* and *non-cited reconstructed SERP
candidates*. Differences are observable associations, not causal explanations.
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
    "page_output_sim": "Page–answer similarity",
    "max_chunk_output_sim": "Best chunk–answer similarity",
    "max_chunk_query_sim": "Best chunk–query similarity",
    "word_count": "Word count",
    "heading_count": "Heading count",
    "freshness_days": "Age (days)",
}


def features_df(features: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(features)
    if df.empty:
        return df
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def group_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of each numeric feature for cited vs non-cited candidates."""
    if df.empty or "cited" not in df.columns:
        return pd.DataFrame()
    rows = []
    cited = df[df["cited"] == 1]
    noncited = df[df["cited"] == 0]
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            continue
        cm = cited[col].mean(skipna=True)
        nm = noncited[col].mean(skipna=True)
        rows.append(
            {
                "feature": FEATURE_LABELS.get(col, col),
                "key": col,
                "cited_mean": None if pd.isna(cm) else round(float(cm), 4),
                "noncited_mean": None if pd.isna(nm) else round(float(nm), 4),
                "n_cited": int(cited[col].notna().sum()),
                "n_noncited": int(noncited[col].notna().sum()),
                "delta": None if (pd.isna(cm) or pd.isna(nm)) else round(float(cm - nm), 4),
            }
        )
    return pd.DataFrame(rows)


def source_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Counts and cite-rate per source type."""
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
    if df.empty or "official_source" not in df.columns:
        return {}
    out = {}
    for flag, label in [(True, "official"), (False, "non_official")]:
        sub = df[df["official_source"] == flag]
        out[label] = {
            "candidates": int(len(sub)),
            "cited": int(sub["cited"].sum()) if not sub.empty else 0,
            "cite_rate": round(float(sub["cited"].mean()), 3) if not sub.empty else 0.0,
        }
    return out


def summary_metrics(run: dict) -> dict:
    """Headline numbers for the Overview cards."""
    g = run.get("gemini") or {}
    serp = run.get("serp") or {}
    scrape = run.get("scrape") or {}
    matching = run.get("matching") or {}
    pages = scrape.get("pages") or {}
    recall = matching.get("recall") or {}
    df = features_df(run.get("features") or [])

    n_cited = int(df["cited"].sum()) if not df.empty and "cited" in df else 0
    return {
        "n_queries": len(g.get("search_queries", []) or []),
        "n_citations": matching.get("n_citations", len(g.get("citations", []) or [])),
        "n_candidates": len(df) if not df.empty else len(serp.get("candidates", []) or []),
        "n_scraped": sum(1 for p in pages.values() if p.get("status") == "success"),
        "n_cited_candidates": n_cited,
        "recall_5": float(recall.get("5", 0.0)),
        "recall_10": float(recall.get("10", 0.0)),
        "recall_20": float(recall.get("20", 0.0)),
        "recall_50": float(recall.get("50", 0.0)),
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
        if mask.sum() < 3 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
            rows.append({"feature": FEATURE_LABELS.get(col, col), "key": col, "corr": None})
            continue
        r = float(np.corrcoef(x[mask], y[mask])[0, 1])
        rows.append({"feature": FEATURE_LABELS.get(col, col), "key": col, "corr": round(r, 3)})
    return pd.DataFrame(rows)
