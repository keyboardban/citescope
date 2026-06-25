"""Per-question table + question clustering by shared websites."""

from __future__ import annotations

from src import cluster


def _row(rid, dom, cited=1, key="record_id"):
    return {key: rid, "prompt": f"prompt {rid}", "domain": dom, "cited": cited, "intent": ""}


def test_question_table_counts():
    feats = [_row("Q1", "a.com", 1), _row("Q1", "b.com", 0), _row("Q2", "c.com", 1)]
    by = {r.qid: r for r in cluster.question_table(feats).itertuples()}
    assert by["Q1"].n_cited == 1 and by["Q1"].n_more == 1 and by["Q1"].n_total == 2
    assert by["Q2"].n_cited == 1 and by["Q2"].n_more == 0


def test_cluster_splits_by_shared_domains():
    feats = []
    for rid in ("Q1", "Q2"):
        feats += [_row(rid, "a.com"), _row(rid, "b.com")]
    for rid in ("Q3", "Q4"):
        feats += [_row(rid, "x.com"), _row(rid, "y.com")]
    cl = cluster.cluster_questions(feats, "cited", k=2)
    groups = [set(m["qid"] for m in c["members"]) for c in cl]
    assert len(cl) == 2
    assert {"Q1", "Q2"} in groups and {"Q3", "Q4"} in groups
    # each cluster reports its shared domains
    by = {frozenset(m["qid"] for m in c["members"]): c for c in cl}
    doms = {d["domain"] for d in by[frozenset({"Q1", "Q2"})]["top_domains"]}
    assert {"a.com", "b.com"} <= doms


def test_cluster_works_with_run_id_key():
    feats = [_row("R1", "a.com", key="run_id"), _row("R2", "a.com", key="run_id"),
             _row("R3", "z.com", key="run_id")]
    assert len(cluster.question_table(feats)) == 3
    cl = cluster.cluster_questions(feats, "cited", k=2)
    assert sum(c["size"] for c in cl) == 3


def test_empty_questions_are_unclustered_and_matrix_builds():
    feats = [{"record_id": "Q1", "prompt": "p", "domain": "a.com", "cited": 0},  # more-only only
             {"record_id": "Q2", "prompt": "p", "domain": None, "cited": 1}]     # no domain
    cl = cluster.cluster_questions(feats, "cited", k=2)            # no cited domains anywhere
    assert any(c["cluster"] == -1 for c in cl)
    mat = cluster.clustered_question_matrix(feats, "more_only", 2)
    assert "a.com" in list(mat.columns) and len(mat) == 2
