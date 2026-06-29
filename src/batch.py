"""Batch / Topic-study mode: run many prompts and aggregate observable associations.

Single-run findings are anecdotal. This pools candidate feature rows across many
prompts (optionally tagged with topic + intent + id) so cited-vs-non-cited
differences can be summarised with sample sizes, a non-parametric test
(Mann-Whitney U), bootstrap median-difference CIs, and per-topic / per-intent
breakdowns.

Results are observable associations — NOT causal evidence about how the AI selects
or cites sources.
"""

from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np

from . import storage
from .analysis import (
    FEATURE_LABELS,
    FEATURE_PHASE,
    _normalize_recall,
    correlation_with_citation,
    econometric_analysis,
    features_df,
    official_compare,
    source_breakdown,
)
from .features import NUMERIC_FEATURES
from .ids import new_run_id, now_iso
from .pipeline import run_full

ProgressCB = Callable[[str, float], None]
_RNG_SEED = 12345


# --------------------------------------------------------------------------- #
# statistics (scipy-free)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i, n = 0, len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mann_whitney_u(a, b) -> dict:
    a = np.asarray([v for v in a if v == v], dtype=float)
    b = np.asarray([v for v in b if v == v], dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return {"U": None, "p": None, "n1": n1, "n2": n2}
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2.0
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2.0
    N = n1 + n2
    _, counts = np.unique(allv, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts))
    var = (n1 * n2 / 12.0) * ((N + 1) - tie / (N * (N - 1)))
    if var <= 0:
        return {"U": float(u), "p": None, "n1": n1, "n2": n2}
    z = (u - mu) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return {"U": float(u), "p": round(min(1.0, max(0.0, p)), 4), "n1": n1, "n2": n2}


def bootstrap_median_diff(a, b, iters: int = 1500) -> tuple[float | None, float | None]:
    a = np.asarray([v for v in a if v == v], dtype=float)
    b = np.asarray([v for v in b if v == v], dtype=float)
    if len(a) < 3 or len(b) < 3:
        return (None, None)
    rng = np.random.default_rng(_RNG_SEED)
    diffs = np.empty(iters)
    for i in range(iters):
        diffs[i] = np.median(rng.choice(a, len(a), replace=True)) - np.median(rng.choice(b, len(b), replace=True))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def _med(x):
    return round(float(np.median(x)), 4) if len(x) else None


def _mean(x):
    return round(float(np.mean(x)), 4) if len(x) else None


def _group_stats(df) -> list[dict]:
    rows: list[dict] = []
    if df.empty or "cited" not in df.columns:
        return rows
    cited = df[df["cited"] == 1]
    non = df[df["cited"] == 0]
    for key in NUMERIC_FEATURES:
        if key not in df.columns:
            continue
        ca = cited[key].dropna().tolist()
        na = non[key].dropna().tolist()
        mwu = mann_whitney_u(ca, na)
        lo, hi = bootstrap_median_diff(ca, na)
        cmd, nmd = _med(ca), _med(na)
        rows.append({
            "feature": FEATURE_LABELS.get(key, key), "key": key, "phase": FEATURE_PHASE.get(key, "pre_answer"),
            "cited_median": cmd, "noncited_median": nmd, "cited_mean": _mean(ca), "noncited_mean": _mean(na),
            "median_diff": (None if cmd is None or nmd is None else round(cmd - nmd, 4)),
            "mwu_p": mwu["p"], "ci_low": lo, "ci_high": hi, "n_cited": len(ca), "n_noncited": len(na),
        })
    return rows


def _recall_avg(runs) -> dict:
    out = {"strict": {}, "canonical": {}, "domain_inclusive": {}}
    per = [_normalize_recall((r.get("matching") or {}).get("recall") or {}) for r in runs]
    for mode in out:
        for k in ("5", "10", "20", "50"):
            vals = [p.get(mode, {}).get(k) for p in per if p.get(mode, {}).get(k) is not None]
            out[mode][k] = round(float(np.mean(vals)), 4) if vals else 0.0
    return out


def _sample(df, runs) -> dict:
    return {
        "n_runs_ok": len(runs),
        "n_candidates": int(len(df)),
        "n_cited": int(df["cited"].sum()) if not df.empty and "cited" in df else 0,
        "n_citations": int(sum((r.get("matching") or {}).get("n_citations", 0) for r in runs)),
        "n_scraped": int(df["scrape_success"].sum()) if not df.empty and "scrape_success" in df else 0,
    }


def _cite_patterns(df, label: str) -> list[str]:
    if df.empty or "cited" not in df.columns or df["cited"].sum() == 0:
        return [f"**{label}:** no cited candidates to compare."]
    cited = df[df["cited"] == 1]
    non = df[df["cited"] == 0]

    def med(series):
        v = series.dropna()
        return float(np.median(v)) if len(v) else None

    out: list[str] = []
    if "serp_rank" in df.columns:
        rc, rn = med(cited["serp_rank"]), med(non["serp_rank"])
        if rc is not None and rn is not None:
            out.append(f"**{label}:** cited sites ranked "
                       f"{'higher' if rc < rn else 'lower/similar'} (median SERP rank {rc:.0f} vs {rn:.0f}).")
    if "page_query_sim" in df.columns:
        pc, pn = med(cited["page_query_sim"]), med(non["page_query_sim"])
        if pc is not None and pn is not None:
            out.append(f"**{label}:** cited pages had "
                       f"{'higher' if pc > pn else 'comparable'} page–query similarity "
                       f"({pc:.2f} vs {pn:.2f}).")
    sb = source_breakdown(df)
    if not sb.empty:
        top = sb[sb["candidates"] >= 2].sort_values("cite_rate", ascending=False).head(2)
        if not top.empty:
            out.append(f"**{label}:** highest cite-rate source types — "
                       + ", ".join(f"{r['source_type']} {r['cite_rate']*100:.0f}%" for _, r in top.iterrows()) + ".")
    off = official_compare(df)
    if off:
        inst = off.get("institutional_official", {})
        brand = off.get("brand_official_candidate", {})
        other = off.get("other", {})
        out.append(f"**{label}:** cite-rate — institutional {inst.get('cite_rate',0)*100:.0f}% / "
                   f"brand-candidate {brand.get('cite_rate',0)*100:.0f}% / other {other.get('cite_rate',0)*100:.0f}%.")
    corr = correlation_with_citation(df)
    corr = corr.dropna(subset=["corr"]) if not corr.empty else corr
    if not corr.empty:
        top = corr.reindex(corr["corr"].abs().sort_values(ascending=False).index).head(1).iloc[0]
        out.append(f"**{label}:** strongest feature↔citation correlation = {top['feature']} (r={top['corr']}).")
    return out


def aggregate(runs: list[dict], combined: list[dict]) -> dict:
    df = features_df(combined)
    run_by_id = {r["run_id"]: r for r in runs}

    overall = {
        "sample_sizes": _sample(df, runs),
        "group_stats": _group_stats(df),
        "recall": _recall_avg(runs),
        "source_breakdown": source_breakdown(df).to_dict(orient="records"),
        "correlation": correlation_with_citation(df).to_dict(orient="records"),
        # pooled across prompts → cluster by run_id (many runs ⇒ valid cluster-robust SEs)
        "regression": econometric_analysis(
            df, NUMERIC_FEATURES, FEATURE_LABELS, FEATURE_PHASE,
            position_col="serp_rank", cluster_key="run_id", context="gemini",
            title="Position-adjusted citation model (pooled across prompts)"),
        "official": official_compare(df),
    }

    by_topic: dict[str, dict] = {}
    if not df.empty and "topic" in df.columns:
        for topic, sub in df.groupby("topic"):
            t_runs = [run_by_id[i] for i in sub["run_id"].unique() if i in run_by_id]
            by_topic[str(topic)] = {
                "sample_sizes": _sample(sub, t_runs),
                "recall": _recall_avg(t_runs),
                "group_stats": _group_stats(sub),
                "source_breakdown": source_breakdown(sub).to_dict(orient="records"),
                "official": official_compare(sub),
                "correlation": correlation_with_citation(sub).to_dict(orient="records"),
                "cite_rate": round(float(sub["cited"].mean()), 3) if len(sub) else 0.0,
            }

    by_intent: dict[str, dict] = {}
    if not df.empty and "intent" in df.columns:
        for intent, sub in df.groupby("intent"):
            if not str(intent):
                continue
            c, n = sub[sub["cited"] == 1], sub[sub["cited"] == 0]
            by_intent[str(intent)] = {
                "n_candidates": int(len(sub)),
                "n_cited": int(sub["cited"].sum()),
                "cite_rate": round(float(sub["cited"].mean()), 3) if len(sub) else 0.0,
                "rank_cited_median": (round(float(c["serp_rank"].median()), 1) if len(c) and "serp_rank" in sub else None),
                "rank_noncited_median": (round(float(n["serp_rank"].median()), 1) if len(n) and "serp_rank" in sub else None),
            }

    patterns = _cite_patterns(df, "Overall")
    for topic in by_topic:
        patterns += _cite_patterns(df[df["topic"] == topic], topic)

    return {**overall, "by_topic": by_topic, "by_intent": by_intent, "patterns": patterns}


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _normalize_items(items) -> list[dict]:
    out: list[dict] = []
    for it in items or []:
        if isinstance(it, str):
            out.append({"prompt": it.strip(), "id": "", "intent": "Custom", "topic": "Custom"})
        elif isinstance(it, dict) and it.get("prompt"):
            out.append({
                "prompt": it["prompt"].strip(), "id": it.get("id", "") or "",
                "intent": it.get("intent", "") or "Custom", "topic": it.get("topic", "") or "Custom",
            })
    return [it for it in out if it["prompt"]]


def run_batch(clients: dict, items, base_inputs: dict,
              progress: ProgressCB | None = None, use_cache: bool = True) -> dict:
    """Run each item (string or {prompt,id,intent,topic}) through the full pipeline."""
    items = _normalize_items(items)
    runs: list[dict] = []
    per_prompt: list[dict] = []
    combined: list[dict] = []
    total = max(1, len(items))

    for i, it in enumerate(items):
        if progress:
            progress(f"{i + 1}/{len(items)} [{it['topic']}] {it['prompt'][:42]}", i / total)
        inp = copy.deepcopy(base_inputs)
        inp["prompt"] = it["prompt"]
        inp["serp"]["selected_queries"] = []
        try:
            run = run_full(clients, inp, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 - keep the batch going on one failure
            per_prompt.append({**it, "run_id": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        runs.append(run)
        for row in run.get("features", []):
            combined.append({**row, "run_id": run["run_id"], "prompt": it["prompt"],
                             "id": it["id"], "intent": it["intent"], "topic": it["topic"]})
        m = (run.get("analysis") or {}).get("summary", {})
        per_prompt.append({**it, "run_id": run["run_id"], "error": None,
                           "n_candidates": m.get("n_candidates", 0), "n_citations": m.get("n_citations", 0),
                           "n_scraped": m.get("n_scraped", 0), "recall_strict_10": m.get("recall_strict_10", 0.0)})

    if progress:
        progress("Aggregating…", 0.97)

    batch = {
        "batch_id": "BATCH-" + new_run_id(),
        "created_at": now_iso(),
        "is_demo": False,
        "n_prompts": len(items),
        "n_candidates": len(combined),
        "items": items,
        "prompts": [it["prompt"] for it in items],
        "run_ids": [r["run_id"] for r in runs],
        "per_prompt": per_prompt,
        "features": combined,
        "aggregate": aggregate(runs, combined),
    }
    storage.save_batch(batch)
    return batch
