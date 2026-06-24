"""Plotly visualizations for the dashboard.

Every chart compares *cited websites* against *non-cited reconstructed SERP
candidates*. Differences shown are observable associations, not causal claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.matching import unique_candidates

from .theme import CITED_SEQ, COLORS, SOURCE_COLORS, TIER_COLORS


def _style(fig: go.Figure, height: int = 320, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=COLORS["text"]),
        title_font=dict(size=14),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
    )
    fig.update_xaxes(gridcolor="#eef0f6", zeroline=False)
    fig.update_yaxes(gridcolor="#eef0f6", zeroline=False)
    return fig


def _with_group(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["group"] = d["cited"].map({1: "cited", 0: "non-cited"})
    return d


# --------------------------------------------------------------------------- #
# recall
# --------------------------------------------------------------------------- #
def recall_bar(recall: dict) -> go.Figure:
    ks = [5, 10, 20, 50]
    vals = [float(recall.get(str(k), 0.0)) for k in ks]
    fig = go.Figure(
        go.Bar(
            x=[f"@{k}" for k in ks], y=vals,
            text=[f"{v*100:.0f}%" for v in vals], textposition="outside",
            marker_color=COLORS["primary"], marker_line_width=0,
        )
    )
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title="Citation recall@K")
    return _style(fig, 300, legend=False)


# --------------------------------------------------------------------------- #
# rank
# --------------------------------------------------------------------------- #
def rank_box(df: pd.DataFrame) -> go.Figure:
    d = _with_group(df.dropna(subset=["serp_rank"]))
    fig = px.box(
        d, x="serp_rank", y="group", color="group", points="all",
        color_discrete_map=CITED_SEQ, orientation="h",
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7), jitter=0.35)
    fig.update_layout(title="Where do cited sites sit in the reconstructed SERP rank?")
    fig.update_xaxes(title="SERP rank (lower = higher on the page)")
    fig.update_yaxes(title="")
    return _style(fig, 300)


# --------------------------------------------------------------------------- #
# cited vs non-cited comparison
# --------------------------------------------------------------------------- #
def grouped_means(records: list[dict], keys: list[str]) -> go.Figure:
    rows = []
    for r in records:
        if r["key"] not in keys:
            continue
        rows.append({"feature": r["feature"], "group": "cited", "mean": r["cited_mean"] or 0})
        rows.append({"feature": r["feature"], "group": "non-cited", "mean": r["noncited_mean"] or 0})
    d = pd.DataFrame(rows)
    if d.empty:
        return _style(go.Figure(), 320)
    fig = px.bar(d, x="feature", y="mean", color="group", barmode="group",
                 color_discrete_map=CITED_SEQ, text_auto=".2f")
    fig.update_layout(title="Cited vs non-cited — mean feature values")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="mean")
    return _style(fig, 340)


def distribution_box(df: pd.DataFrame, feature: str, label: str) -> go.Figure:
    d = _with_group(df.dropna(subset=[feature]))
    fig = px.box(d, x="group", y=feature, color="group", points="all",
                 color_discrete_map=CITED_SEQ)
    fig.update_traces(marker=dict(size=7, opacity=0.65), jitter=0.3)
    fig.update_layout(title=f"{label} — distribution")
    fig.update_xaxes(title="")
    fig.update_yaxes(title=label)
    return _style(fig, 320, legend=False)


# --------------------------------------------------------------------------- #
# source types
# --------------------------------------------------------------------------- #
def source_stacked(sb_df: pd.DataFrame) -> go.Figure:
    if sb_df.empty:
        return _style(go.Figure(), 320)
    rows = []
    for _, r in sb_df.iterrows():
        rows.append({"source_type": r["source_type"], "status": "cited", "count": int(r["cited"])})
        rows.append({"source_type": r["source_type"], "status": "non-cited", "count": int(r["non_cited"])})
    d = pd.DataFrame(rows)
    fig = px.bar(d, x="source_type", y="count", color="status", barmode="stack",
                 color_discrete_map=CITED_SEQ)
    fig.update_layout(title="Source-type distribution (cited vs non-cited)")
    fig.update_xaxes(title="")
    return _style(fig, 340)


def cite_rate_by_source(sb_df: pd.DataFrame) -> go.Figure:
    if sb_df.empty:
        return _style(go.Figure(), 300)
    d = sb_df[sb_df["candidates"] > 0].copy()
    fig = go.Figure(
        go.Bar(
            x=d["source_type"], y=d["cite_rate"],
            marker_color=[SOURCE_COLORS.get(s, COLORS["primary"]) for s in d["source_type"]],
            text=[f"{v*100:.0f}%" for v in d["cite_rate"]], textposition="outside",
            customdata=d["candidates"],
            hovertemplate="%{x}<br>cite-rate %{y:.0%}<br>%{customdata} candidates<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title="Cite-rate within each source type")
    return _style(fig, 300, legend=False)


# --------------------------------------------------------------------------- #
# match-type distribution
# --------------------------------------------------------------------------- #
def match_type_bar(rate_counts: dict) -> go.Figure:
    order = ["exact", "normalized", "final_redirect", "canonical",
             "amp_canonical", "domain_only", "no_match"]
    items = [(t, rate_counts.get(t, 0)) for t in order if rate_counts.get(t, 0)]
    if not items:
        return _style(go.Figure(), 260)
    labels = [t.replace("_", " ") for t, _ in items]
    fig = go.Figure(
        go.Bar(
            x=[c for _, c in items], y=labels, orientation="h",
            marker_color=[TIER_COLORS.get(t, COLORS["primary"]) for t, _ in items],
            text=[c for _, c in items], textposition="outside",
        )
    )
    fig.update_layout(title="Citation match-type distribution")
    fig.update_xaxes(title="citations")
    return _style(fig, 280, legend=False)


# --------------------------------------------------------------------------- #
# chunk relevance
# --------------------------------------------------------------------------- #
def chunk_relevance(chunks: list[dict], key: str = "output_sim", label: str = "answer") -> go.Figure:
    if not chunks:
        return _style(go.Figure(), 280)
    vals = [c.get(key, 0) or 0 for c in chunks]
    best = int(np.argmax(vals)) if vals else -1
    colors = [COLORS["primary"] if i == best else "#c7d2fe" for i in range(len(vals))]
    hover = [
        f"chunk {c['index']} · {c.get('n_words', 0)} words<br>"
        f"<b>{(c.get('heading') or '(no heading)')[:60]}</b><br>"
        f"{(c.get('text', '')[:140])}…"
        for c in chunks
    ]
    fig = go.Figure(
        go.Bar(x=[c["index"] for c in chunks], y=vals, marker_color=colors,
               hovertext=hover, hoverinfo="text")
    )
    fig.update_layout(title=f"Chunk similarity to AI {label} (semantic overlap proxy)")
    fig.update_xaxes(title="chunk index", dtick=1)
    fig.update_yaxes(title="similarity", range=[0, max(0.1, max(vals) * 1.15)])
    return _style(fig, 300, legend=False)


# --------------------------------------------------------------------------- #
# similarity radar
# --------------------------------------------------------------------------- #
def similarity_radar(row: dict, avg: dict | None = None) -> go.Figure:
    dims = [
        ("title_query_sim", "title–query"),
        ("snippet_query_sim", "snippet–query"),
        ("page_query_sim", "page–query"),
        ("page_output_sim", "page–answer"),
        ("max_chunk_output_sim", "chunk–answer"),
    ]
    labels = [d[1] for d in dims]
    vals = [float(row.get(d[0]) or 0) for d in dims]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]], fill="toself",
        name="this site", line_color=COLORS["primary"]))
    if avg:
        av = [float(avg.get(d[0]) or 0) for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=av + [av[0]], theta=labels + [labels[0]], fill="toself",
            name="all-candidate avg", line_color=COLORS["noncited"], opacity=0.5))
    fig.update_layout(title="Similarity profile", polar=dict(radialaxis=dict(range=[0, 1], visible=True)))
    return _style(fig, 340)


# --------------------------------------------------------------------------- #
# feature heatmap
# --------------------------------------------------------------------------- #
def feature_heatmap(df: pd.DataFrame, max_rows: int = 24) -> go.Figure:
    feats = ["serp_rank", "title_query_sim", "snippet_query_sim", "page_query_sim",
             "page_output_sim", "max_chunk_output_sim", "word_count", "freshness_days"]
    feats = [f for f in feats if f in df.columns]
    if df.empty or not feats:
        return _style(go.Figure(), 360)
    d = df.sort_values(["cited", "serp_rank"], ascending=[False, True]).head(max_rows).copy()
    norm = pd.DataFrame(index=d.index)
    for f in feats:
        col = pd.to_numeric(d[f], errors="coerce")
        lo, hi = col.min(), col.max()
        norm[f] = 0.5 if (pd.isna(lo) or hi == lo) else (col - lo) / (hi - lo)
    labels = [("● " if c == 1 else "  ") + (dom[:26]) for c, dom in zip(d["cited"], d["domain"])]
    indigo_scale = [[0.0, "#f7f8fc"], [0.5, "#a5b4fc"], [1.0, COLORS["primary"]]]
    fig = go.Figure(go.Heatmap(
        z=norm.values, x=[f.replace("_", " ") for f in feats], y=labels,
        colorscale=indigo_scale, zmin=0, zmax=1, colorbar=dict(title="norm"),
        hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(title="Feature heatmap (● = cited · values min-max normalized per column)")
    return _style(fig, max(360, 26 + 18 * len(d)), legend=False)


# --------------------------------------------------------------------------- #
# query → candidate → citation flow (Sankey)
# --------------------------------------------------------------------------- #
def citation_sankey(run: dict, top_n: int = 16) -> go.Figure:
    serp = run.get("serp") or {}
    cands = unique_candidates(serp.get("candidates", []))
    cited_ids = set((run.get("matching") or {}).get("cited_candidate_ids", []))
    cands = sorted(cands, key=lambda c: c["best_rank"])[:top_n]
    if not cands:
        return _style(go.Figure(), 360)

    queries = sorted({(q.get("query") or "?") for c in cands for q in c.get("queries", [])})
    q_idx = {q: i for i, q in enumerate(queries)}
    base = len(queries)
    dom_list = list(dict.fromkeys(c["domain"] for c in cands))
    d_idx = {dom: base + i for i, dom in enumerate(dom_list)}
    cited_node = base + len(dom_list)
    noncited_node = cited_node + 1

    node_labels = queries + dom_list + ["✓ cited", "non-cited"]
    node_colors = (
        [COLORS["primary"]] * len(queries)
        + [COLORS["primary_soft"]] * len(dom_list)
        + [COLORS["cited"], COLORS["noncited"]]
    )

    src, tgt, val, lcol = [], [], [], []
    for c in cands:
        best_q = min(c.get("queries", [{"query": "?", "rank": 999}]), key=lambda x: x.get("rank", 999))
        q = best_q.get("query") or "?"
        src.append(q_idx.get(q, 0)); tgt.append(d_idx[c["domain"]]); val.append(1)
        lcol.append("rgba(79,70,229,0.18)")
        is_cited = c["candidate_id"] in cited_ids
        src.append(d_idx[c["domain"]])
        tgt.append(cited_node if is_cited else noncited_node)
        val.append(1)
        lcol.append("rgba(22,163,74,0.35)" if is_cited else "rgba(148,163,184,0.30)")

    fig = go.Figure(go.Sankey(
        node=dict(label=node_labels, color=node_colors, pad=14, thickness=14,
                  line=dict(width=0)),
        link=dict(source=src, target=tgt, value=val, color=lcol),
    ))
    fig.update_layout(title="Query → reconstructed candidate → citation status")
    return _style(fig, 420, legend=False)


# --------------------------------------------------------------------------- #
# recall variants + validity charts
# --------------------------------------------------------------------------- #
def recall_grouped(recall: dict) -> go.Figure:
    """Grouped bar: strict / canonical / domain-inclusive recall across K."""
    modes = [("strict", "strict"), ("canonical", "canonical"),
             ("domain_inclusive", "domain-incl (weak)")]
    rows = []
    for key, label in modes:
        d = recall.get(key, {}) or {}
        for k in (5, 10, 20, 50):
            rows.append({"K": f"@{k}", "mode": label, "recall": float(d.get(str(k), 0.0))})
    dfp = pd.DataFrame(rows)
    if dfp.empty:
        return _style(go.Figure(), 320)
    fig = px.bar(dfp, x="K", y="recall", color="mode", barmode="group",
                 color_discrete_map={"strict": COLORS["cited"], "canonical": COLORS["primary"],
                                     "domain-incl (weak)": COLORS["weak"]})
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title="Citation recall@K — strict vs canonical vs domain-inclusive")
    return _style(fig, 320)


def length_vs_sim_scatter(df: pd.DataFrame, sim_col: str = "page_output_sim") -> go.Figure:
    """Page length vs page-answer similarity — visualizes the length-bias confound."""
    if df.empty or "word_count" not in df.columns or sim_col not in df.columns:
        return _style(go.Figure(), 320)
    d = _with_group(df.dropna(subset=["word_count", sim_col]))
    if d.empty:
        return _style(go.Figure(), 320)
    hover = ["domain"] if "domain" in d.columns else None
    fig = px.scatter(d, x="word_count", y=sim_col, color="group",
                     color_discrete_map=CITED_SEQ, hover_data=hover)
    fig.update_traces(marker=dict(size=10, opacity=0.75))
    fig.update_layout(title="Page length vs page–answer similarity (length-bias check)")
    fig.update_xaxes(title="word count")
    fig.update_yaxes(title="page–answer similarity")
    return _style(fig, 320)


_TOPIC_PALETTE = ["#4f46e5", "#0891b2", "#16a34a", "#db2777", "#ca8a04", "#7c3aed"]


def topic_compare(by_topic: dict, metric: str = "cite_rate", title: str | None = None) -> go.Figure:
    """One bar per topic for a chosen rate metric (cite_rate or strict recall@10)."""
    rows = []
    for t, info in by_topic.items():
        if metric == "cite_rate":
            v = info.get("cite_rate", 0.0)
        elif metric == "recall_strict_10":
            v = info.get("recall", {}).get("strict", {}).get("10", 0.0)
        else:
            v = 0.0
        rows.append({"topic": t, "value": v})
    d = pd.DataFrame(rows)
    if d.empty:
        return _style(go.Figure(), 300)
    fig = go.Figure(go.Bar(
        x=d["topic"], y=d["value"],
        marker_color=[_TOPIC_PALETTE[i % len(_TOPIC_PALETTE)] for i in range(len(d))],
        text=[f"{v*100:.0f}%" for v in d["value"]], textposition="outside"))
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title=title or "By topic")
    return _style(fig, 300, legend=False)


def topic_feature_compare(by_topic: dict, key: str, label: str) -> go.Figure:
    """Cited vs non-cited median of one feature, grouped per topic."""
    rows = []
    for t, info in by_topic.items():
        gs = {g["key"]: g for g in info.get("group_stats", [])}
        r = gs.get(key, {})
        rows.append({"topic": t, "group": "cited", "median": r.get("cited_median") or 0})
        rows.append({"topic": t, "group": "non-cited", "median": r.get("noncited_median") or 0})
    d = pd.DataFrame(rows)
    if d.empty:
        return _style(go.Figure(), 320)
    fig = px.bar(d, x="topic", y="median", color="group", barmode="group",
                 color_discrete_map=CITED_SEQ, text_auto=".2f")
    fig.update_layout(title=f"{label} — cited vs non-cited, by topic")
    fig.update_xaxes(title="")
    return _style(fig, 320)


def official_bar(off: dict) -> go.Figure:
    """Cite-rate for institutional-official vs brand-candidate vs other."""
    if not off:
        return _style(go.Figure(), 260)
    labels = {"institutional_official": "institutional", "brand_official_candidate": "brand candidate", "other": "other"}
    d = pd.DataFrame([{"group": labels.get(k, k), "cite_rate": v.get("cite_rate", 0.0),
                       "candidates": v.get("candidates", 0)} for k, v in off.items()])
    fig = go.Figure(go.Bar(
        x=d["group"], y=d["cite_rate"], marker_color=COLORS["primary"],
        text=[f"{v*100:.0f}%" for v in d["cite_rate"]], textposition="outside",
        customdata=d["candidates"],
        hovertemplate="%{x}<br>cite-rate %{y:.0%}<br>%{customdata} candidates<extra></extra>"))
    fig.update_yaxes(range=[0, 1.08], tickformat=".0%")
    fig.update_layout(title="Cite-rate: official signals (heuristic)")
    return _style(fig, 280, legend=False)
