"""Per-question separation + clustering of questions by the websites they touch.

Multi-question runs (ChatGPT Bright Data uploads, Topic Studies / Batch) pool
sources together. This module lets you look at each question individually and
group questions by the **set of websites they cite or surface** (Jaccard +
agglomerative average-linkage, pure numpy — no scikit-learn).

This is observable grouping of source overlap, not a causal claim about the AI.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _qid(row: dict) -> str:
    return str(row.get("record_id") or row.get("run_id") or "?")


def _question_records(features: list[dict]) -> dict[str, dict]:
    """Group feature rows by question (record_id for ChatGPT, run_id for batch)."""
    by: dict[str, dict] = {}
    for r in features:
        q = by.setdefault(_qid(r), {
            "qid": _qid(r), "prompt": r.get("prompt", ""),
            "intent": r.get("intent") or "", "topic": r.get("topic") or "", "rows": [],
        })
        q["rows"].append(r)
    return by


def _filter(rows: list[dict], group: str) -> list[dict]:
    if group == "cited":
        return [r for r in rows if r.get("cited") == 1]
    if group == "more_only":
        return [r for r in rows if r.get("cited") == 0]
    return rows  # "all"


def question_table(features: list[dict]) -> pd.DataFrame:
    """One row per question: counts, cite-rate, and its top cited domains."""
    rows = []
    for q in _question_records(features).values():
        cited = _filter(q["rows"], "cited")
        more = _filter(q["rows"], "more_only")
        cdom = Counter(r["domain"] for r in cited if r.get("domain"))
        rows.append({
            "qid": q["qid"], "prompt": q["prompt"],
            "intent": q["intent"], "topic": q["topic"],
            "n_cited": len(cited), "n_more": len(more), "n_total": len(q["rows"]),
            "cite_rate": round(len(cited) / len(q["rows"]), 2) if q["rows"] else 0.0,
            "top_cited_domains": ", ".join(d for d, _ in cdom.most_common(5)),
        })
    return pd.DataFrame(rows)


def domain_sets(features: list[dict], group: str = "cited") -> dict[str, set[str]]:
    """qid -> set of domains it cites / surfaces (per `group`)."""
    out: dict[str, set[str]] = {}
    for q in _question_records(features).values():
        out[q["qid"]] = {r["domain"] for r in _filter(q["rows"], group) if r.get("domain")}
    return out


def _top_domains(features: list[dict], group: str, n: int) -> list[str]:
    freq: Counter = Counter()
    for doms in domain_sets(features, group).values():
        freq.update(doms)
    return [d for d, _ in freq.most_common(n)]


# --------------------------------------------------------------------------- #
# clustering (Jaccard distance + agglomerative average linkage)
# --------------------------------------------------------------------------- #
def _agglomerative(dist: np.ndarray, k: int) -> dict[int, int]:
    """Average-linkage agglomerative clustering -> {row index: cluster id}."""
    n = len(dist)
    k = max(1, min(k, n))
    D = dist.astype(float).copy()
    np.fill_diagonal(D, np.inf)
    active = list(range(n))
    members: dict[int, list[int]] = {i: [i] for i in range(n)}

    while len(active) > k:
        best, best_d = None, np.inf
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                i, j = active[ai], active[aj]
                if D[i, j] < best_d:
                    best_d, best = D[i, j], (i, j)
        i, j = best  # type: ignore[misc]
        mi, mj = len(members[i]), len(members[j])
        for x in active:
            if x in (i, j):
                continue
            D[i, x] = D[x, i] = (D[i, x] * mi + D[j, x] * mj) / (mi + mj)
        members[i].extend(members[j])
        active.remove(j)
        D[j, :] = np.inf
        D[:, j] = np.inf

    assign: dict[int, int] = {}
    for cid, rep in enumerate(active):
        for m in members[rep]:
            assign[m] = cid
    return assign


def cluster_questions(features: list[dict], group: str = "cited", k: int = 3) -> list[dict]:
    """Group questions by overlap in the websites they cite/surface.

    Returns one dict per cluster: {cluster, size, members[{qid,prompt,intent,topic}],
    top_domains[{domain,n_questions}]}. Questions with no domains form a trailing
    'unclustered' group (cluster = -1).
    """
    sets = domain_sets(features, group)
    qrec = _question_records(features)
    items = [(qid, doms) for qid, doms in sets.items() if doms]
    empty = [qid for qid, doms in sets.items() if not doms]

    assign: dict[str, int] = {}
    if items:
        ids = [i for i, _ in items]
        S = [d for _, d in items]
        n = len(ids)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                u = len(S[i] | S[j])
                D[i, j] = D[j, i] = (1 - len(S[i] & S[j]) / u) if u else 1.0
        idx_assign = _agglomerative(D, k)
        assign = {ids[i]: c for i, c in idx_assign.items()}

    grouped: dict[int, list[str]] = defaultdict(list)
    for qid, c in assign.items():
        grouped[c].append(qid)
    for qid in empty:
        grouped[-1].append(qid)

    out = []
    for cid in sorted(grouped, key=lambda c: (c < 0, c)):
        memids = grouped[cid]
        domc: Counter = Counter()
        for qid in memids:
            domc.update(sets[qid])
        out.append({
            "cluster": cid, "size": len(memids),
            "members": [{"qid": q, "prompt": qrec[q]["prompt"],
                         "intent": qrec[q]["intent"], "topic": qrec[q]["topic"]} for q in memids],
            "top_domains": [{"domain": d, "n_questions": c} for d, c in domc.most_common(8)],
        })
    return out


def clustered_question_matrix(features: list[dict], group: str = "cited",
                              k: int = 3, top_n: int = 20) -> pd.DataFrame:
    """Questions × top-domains count matrix, rows ordered/labeled by cluster."""
    qrec = _question_records(features)
    clusters = cluster_questions(features, group, k)
    top = _top_domains(features, group, top_n)
    rows, index = [], []
    for c in clusters:
        for m in c["members"]:
            qid = m["qid"]
            rc = Counter(r["domain"] for r in _filter(qrec[qid]["rows"], group) if r.get("domain"))
            rows.append([rc.get(d, 0) for d in top])
            tag = f"C{c['cluster']} ▸ " if c["cluster"] >= 0 else "· "
            index.append(tag + f"{qid[:8]}: {qrec[qid]['prompt'][:34]}")
    return pd.DataFrame(rows, index=index, columns=top)
