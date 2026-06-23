"""Batch mode: run multiple prompts and aggregate observable associations.

Single-run findings are anecdotal. Batch mode pools candidate feature rows across
many prompts so cited-vs-non-cited differences can be summarised with sample sizes
and a non-parametric test (Mann-Whitney U) plus bootstrapped median-difference CIs.

Results are observable associations across runs — NOT causal evidence about how the
AI selects or cites sources.
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
    features_df,
    source_breakdown,
)
from .features import NUMERIC_FEATURES
from .ids import new_run_id, now_iso
from .pipeline import PipelineError, run_full

ProgressCB = Callable[[str, float], None]
_RNG_SEED = 12345


# --------------------------------------------------------------------------- #
# statistics (scipy-free)
# --------------------------------------------------------------------------- #
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank (1-based), ties averaged — like scipy.stats.rankdata."""
    order = a.argsort(kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
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
    """Two-sided Mann-Whitney U with tie correction + normal approximation."""
    a = np.asarray([v for v in a if v == v], dtype=float)
    b = np.asarray([v for v in b if v == v], dtype=float)
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return {"U": None, "p": None, "n1": n1, "n2": n2}
    allv = np.concatenate([a, b])
    ranks = _rankdata(allv)
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
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


def bootstrap_median_diff(a, b, iters: int = 2000) -> tuple[float | None, float | None]:
    """95% bootstrap CI for median(cited) - median(non-cited)."""
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
# aggregation
# --------------------------------------------------------------------------- #
def aggregate(runs: list[dict], combined: list[dict]) -> dict:
    df = features_df(combined)
    group_stats: list[dict] = []
    if not df.empty and "cited" in df.columns:
        cited = df[df["cited"] == 1]
        noncited = df[df["cited"] == 0]
        for key in NUMERIC_FEATURES:
            if key not in df.columns:
                continue
            ca = cited[key].dropna().tolist()
            na = noncited[key].dropna().tolist()
            mwu = mann_whitney_u(ca, na)
            lo, hi = bootstrap_median_diff(ca, na)

            def _med(x):
                return round(float(np.median(x)), 4) if x else None

            def _mean(x):
                return round(float(np.mean(x)), 4) if x else None

            group_stats.append({
                "feature": FEATURE_LABELS.get(key, key),
                "key": key,
                "phase": FEATURE_PHASE.get(key, "pre_answer"),
                "cited_median": _med(ca),
                "noncited_median": _med(na),
                "cited_mean": _mean(ca),
                "noncited_mean": _mean(na),
                "median_diff": (None if _med(ca) is None or _med(na) is None else round(_med(ca) - _med(na), 4)),
                "mwu_p": mwu["p"],
                "ci_low": lo,
                "ci_high": hi,
                "n_cited": len(ca),
                "n_noncited": len(na),
            })

    # recall averaged across runs (per-run recall comes from matching)
    recall_avg = {"strict": {}, "canonical": {}, "domain_inclusive": {}}
    per_run = [_normalize_recall((r.get("matching") or {}).get("recall") or {}) for r in runs]
    for mode in recall_avg:
        for k in ("5", "10", "20", "50"):
            vals = [pr.get(mode, {}).get(k) for pr in per_run if pr.get(mode, {}).get(k) is not None]
            recall_avg[mode][k] = round(float(np.mean(vals)), 4) if vals else 0.0

    sample = {
        "n_runs_ok": len(runs),
        "n_candidates": int(len(df)),
        "n_cited": int(df["cited"].sum()) if not df.empty and "cited" in df else 0,
        "n_citations": int(sum((r.get("matching") or {}).get("n_citations", 0) for r in runs)),
        "n_scraped": int(df["scrape_success"].sum()) if not df.empty and "scrape_success" in df else 0,
    }

    return {
        "sample_sizes": sample,
        "group_stats": group_stats,
        "recall": recall_avg,
        "source_breakdown": source_breakdown(df).to_dict(orient="records"),
    }


def run_batch(clients: dict, prompts: list[str], base_inputs: dict,
              progress: ProgressCB | None = None, use_cache: bool = True) -> dict:
    """Run each prompt through the full pipeline; aggregate across successful runs."""
    prompts = [p.strip() for p in prompts if p and p.strip()]
    runs: list[dict] = []
    per_prompt: list[dict] = []
    combined: list[dict] = []
    total = max(1, len(prompts))

    for i, prompt in enumerate(prompts):
        if progress:
            progress(f"Prompt {i + 1}/{len(prompts)}: {prompt[:48]}", i / total)
        inp = copy.deepcopy(base_inputs)
        inp["prompt"] = prompt
        inp["serp"]["selected_queries"] = []  # use each prompt's own observed queries
        try:
            run = run_full(clients, inp, use_cache=use_cache)
        except PipelineError as exc:
            per_prompt.append({"prompt": prompt, "run_id": None, "error": str(exc)})
            continue
        except Exception as exc:  # noqa: BLE001 - keep batch going on one failure
            per_prompt.append({"prompt": prompt, "run_id": None, "error": f"{type(exc).__name__}: {exc}"})
            continue
        runs.append(run)
        for row in run.get("features", []):
            combined.append({**row, "run_id": run["run_id"], "prompt": prompt})
        m = (run.get("analysis") or {}).get("summary", {})
        per_prompt.append({
            "prompt": prompt, "run_id": run["run_id"], "error": None,
            "n_candidates": m.get("n_candidates", 0), "n_citations": m.get("n_citations", 0),
            "n_scraped": m.get("n_scraped", 0), "recall_strict_10": m.get("recall_strict_10", 0.0),
        })

    if progress:
        progress("Aggregating…", 0.97)

    batch = {
        "batch_id": "BATCH-" + new_run_id(),
        "created_at": now_iso(),
        "n_prompts": len(prompts),
        "n_candidates": len(combined),
        "prompts": prompts,
        "run_ids": [r["run_id"] for r in runs],
        "per_prompt": per_prompt,
        "features": combined,
        "aggregate": aggregate(runs, combined),
    }
    storage.save_batch(batch)
    return batch
