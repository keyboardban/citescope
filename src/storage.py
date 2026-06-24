"""Local persistence: a SQLite index + cache, plus JSON run snapshots on disk.

Design:
- SQLite (`data/audit.db`) holds a lightweight run index and an API result cache
  so expensive Gemini/Apify calls are never repeated by accident.
- Full run state is also written as a JSON snapshot under `data/runs/` (easy to
  inspect, diff, and reload). Raw API payloads are preserved under `data/raw/`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from . import config
from .ids import now_iso


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    config.ensure_dirs()
    con = sqlite3.connect(config.DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _conn() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                stage      TEXT,
                value      TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id        TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                prompt        TEXT,
                model         TEXT,
                n_queries     INTEGER,
                n_citations   INTEGER,
                n_candidates  INTEGER,
                n_scraped     INTEGER,
                recall_10     REAL,
                is_demo       INTEGER DEFAULT 0,
                snapshot_path TEXT
            );
            CREATE TABLE IF NOT EXISTS embeddings (
                key        TEXT PRIMARY KEY,
                vector     TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id      TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                n_prompts     INTEGER,
                n_candidates  INTEGER,
                snapshot_path TEXT
            );
            """
        )


# --------------------------------------------------------------------------- #
# API result cache
# --------------------------------------------------------------------------- #
def cache_get(key: str) -> Any | None:
    init_db()
    with _conn() as con:
        row = con.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
    return json.loads(row["value"]) if row else None


def cache_set(key: str, value: Any, stage: str = "") -> None:
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO cache (key, stage, value, created_at) VALUES (?,?,?,?)",
            (key, stage, json.dumps(value, default=str, ensure_ascii=False), now_iso()),
        )


def cache_clear() -> int:
    init_db()
    with _conn() as con:
        n = con.execute("SELECT COUNT(*) AS c FROM cache").fetchone()["c"]
        con.execute("DELETE FROM cache")
    return int(n)


# --------------------------------------------------------------------------- #
# Embedding cache (persistent; keyed by text-hash + model + provider)
# --------------------------------------------------------------------------- #
def embedding_get(key: str) -> list[float] | None:
    init_db()
    with _conn() as con:
        row = con.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
    return json.loads(row["vector"]) if row else None


def embedding_set(key: str, vector) -> None:
    init_db()
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO embeddings (key, vector, created_at) VALUES (?,?,?)",
            (key, json.dumps(list(vector)), now_iso()),
        )


def embedding_count() -> int:
    init_db()
    with _conn() as con:
        return int(con.execute("SELECT COUNT(*) AS c FROM embeddings").fetchone()["c"])


# --------------------------------------------------------------------------- #
# Raw payload audit trail
# --------------------------------------------------------------------------- #
def save_raw(run_id: str, name: str, payload: Any) -> str:
    config.ensure_dirs()
    path = config.RAW_DIR / f"{run_id}__{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False), "utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Run snapshots
# --------------------------------------------------------------------------- #
def save_run(run: dict) -> str:
    """Persist a full run snapshot and index it in SQLite."""
    init_db()
    run_id = run["run_id"]
    path = config.RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(run, indent=2, default=str, ensure_ascii=False), "utf-8")

    g = run.get("gemini") or {}
    serp = run.get("serp") or {}
    scrape = run.get("scrape") or {}
    matching = run.get("matching") or {}
    recall = (matching.get("recall") or {})
    # recall may be nested {strict:{...}} or flat {"10":...}
    strict = recall.get("strict") if isinstance(recall.get("strict"), dict) else recall
    recall_10 = float((strict or {}).get("10") or 0.0)

    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, created_at, prompt, model, n_queries, n_citations,
                n_candidates, n_scraped, recall_10, is_demo, snapshot_path)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                run_id,
                run.get("created_at", now_iso()),
                (run.get("inputs", {}).get("prompt") or "")[:500],
                run.get("inputs", {}).get("gemini", {}).get("model"),
                len(g.get("search_queries", []) or []),
                len(g.get("citations", []) or []),
                len(serp.get("candidates", []) or []),
                sum(1 for p in (scrape.get("pages") or {}).values() if p.get("status") == "success"),
                recall_10,
                1 if run.get("is_demo") else 0,
                str(path),
            ),
        )
    return str(path)


def load_run(run_id: str) -> dict | None:
    path = config.RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text("utf-8"))


def list_runs(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Batch snapshots (multi-prompt runs)
# --------------------------------------------------------------------------- #
def save_batch(batch: dict) -> str:
    init_db()
    config.ensure_dirs()
    bid = batch["batch_id"]
    path = config.BATCHES_DIR / f"{bid}.json"
    path.write_text(json.dumps(batch, indent=2, default=str, ensure_ascii=False), "utf-8")
    with _conn() as con:
        con.execute(
            """INSERT OR REPLACE INTO batches
               (batch_id, created_at, n_prompts, n_candidates, snapshot_path)
               VALUES (?,?,?,?,?)""",
            (bid, batch.get("created_at", now_iso()), batch.get("n_prompts", 0),
             batch.get("n_candidates", 0), str(path)),
        )
    return str(path)


def load_batch(batch_id: str) -> dict | None:
    path = config.BATCHES_DIR / f"{batch_id}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def list_batches(limit: int = 50) -> list[dict]:
    init_db()
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# ChatGPT Bright Data run snapshots (file-based; separate from gemini runs)
# --------------------------------------------------------------------------- #
def save_chatgpt_run(run: dict) -> str:
    config.ensure_dirs()
    path = config.CHATGPT_DIR / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2, default=str, ensure_ascii=False), "utf-8")
    return str(path)


def load_chatgpt_run(run_id: str) -> dict | None:
    path = config.CHATGPT_DIR / f"{run_id}.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def list_chatgpt_runs(limit: int = 50) -> list[dict]:
    config.ensure_dirs()
    out = []
    for p in config.CHATGPT_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text("utf-8"))
        except (ValueError, OSError):
            continue
        out.append({"run_id": d.get("run_id", p.stem), "created_at": d.get("created_at", ""),
                    "source_file_name": d.get("source_file_name", ""), "n_records": d.get("n_records", 0)})
    return sorted(out, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]


def write_export(filename: str, content: str | bytes) -> str:
    config.ensure_dirs()
    path = config.EXPORTS_DIR / filename
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(content)
    return str(path)
