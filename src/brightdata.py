"""Bright Data (ChatGPT) export parser.

Defensive parsing of a Bright Data JSON or CSV export where each record is one
ChatGPT response. Produces a normalized run object with per-record sources split
into *cited* vs *more-only* (shown-but-not-cited) groups.

Framing: this is the **observable source set** ChatGPT surfaced via Bright Data —
NOT a reconstructed SERP, and NOT ChatGPT's full internal candidate set.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from .ids import new_run_id, now_iso, short_id
from .url_utils import domain, normalize_url


# --------------------------------------------------------------------------- #
# small coercers
# --------------------------------------------------------------------------- #
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "y")
    return False


def _jsonish(v: Any) -> Any:
    """Parse a value that may be a JSON string (common in CSV cells)."""
    if isinstance(v, str):
        s = v.strip()
        if s[:1] in ("[", "{"):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return v
    return v


def _as_list(v: Any) -> list:
    v = _jsonish(v)
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, dict):
        return [v]
    return []


def _normalize_queries(v: Any) -> list[str]:
    v = _jsonish(v)
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                q = x.get("query") or x.get("q") or x.get("text")
                if q:
                    out.append(str(q))
        return out
    return [str(v)]


def _prompt_from_url(url: str) -> str:
    try:
        qs = parse_qs(urlparse(url).query)
    except ValueError:
        return ""
    for key in ("q", "prompt", "query"):
        if qs.get(key):
            return qs[key][0]
    return ""


# --------------------------------------------------------------------------- #
# source extraction
# --------------------------------------------------------------------------- #
def _item_url(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("url") or item.get("link") or item.get("source") or ""
    return ""


def _walk_response_raw(node: Any, found: list[dict], limit: int = 60) -> None:
    """Best-effort recursive scan of response_raw for {url,title,...} entries."""
    if len(found) >= limit:
        return
    if isinstance(node, dict):
        if (node.get("url") or node.get("link")) and (node.get("title") or node.get("snippet") or node.get("text")):
            found.append(node)
        for v in node.values():
            _walk_response_raw(v, found, limit)
    elif isinstance(node, list):
        for v in node:
            _walk_response_raw(v, found, limit)


def extract_sources(record: dict, record_id: str) -> list[dict]:
    """Return deduped sources for one record (cited wins; appearances preserved)."""
    acc: dict[str, dict] = {}

    def add(item: Any, origin: str, cited_label: int, position=None, observed_rank=None) -> None:
        if isinstance(item, str):
            item = {"url": item}
        url = _item_url(item)
        if not url:
            return
        nurl = normalize_url(url)
        if not nurl:
            return
        appearance = {"origin": origin, "cited": bool(cited_label), "position": position}
        if nurl in acc:
            s = acc[nurl]
            s["appearances"].append(appearance)
            if cited_label == 1 and s["cited_label"] == 0:  # cited wins
                s.update(source_group="cited", cited_label=1, source_origin=origin, cited_flag_raw=True)
            # backfill missing descriptive fields
            for src_k, item_ks in (("title", ("title",)), ("description", ("description", "snippet")),
                                   ("domain", ("domain",)), ("icon", ("icon",)),
                                   ("date_published", ("date_published", "pub_date", "date"))):
                if not s.get(src_k):
                    for ik in item_ks:
                        if item.get(ik):
                            s[src_k] = item.get(ik)
                            break
            if s.get("observed_rank") is None and observed_rank is not None:
                s["observed_rank"] = observed_rank
            if s.get("source_position") is None and position is not None:
                s["source_position"] = position
            return
        acc[nurl] = {
            "source_id": short_id(f"{record_id}:{nurl}"),
            "record_id": record_id,
            "url": url, "normalized_url": nurl, "canonical_url": None, "final_url": None,
            "title": (item.get("title") or "").strip(),
            "description": (item.get("description") or item.get("snippet") or "").strip(),
            "domain": item.get("domain") or domain(url),
            "icon": item.get("icon"),
            "source_origin": origin,
            "source_group": "cited" if cited_label == 1 else "more_only",
            "cited_label": cited_label,
            "cited_flag_raw": (item.get("cited") if origin == "citations" else (cited_label == 1)),
            "source_position": position,
            "observed_rank": observed_rank,
            "date_published": item.get("date_published") or item.get("pub_date") or item.get("date"),
            "appearances": [appearance],
            "raw": {k: item.get(k) for k in ("url", "title", "cited", "domain", "rank") if k in item},
        }

    # 1) citations — cited flag is authoritative
    for i, it in enumerate(_as_list(record.get("citations"))):
        add(it, "citations", 1 if _coerce_bool((it or {}).get("cited")) else 0, position=i + 1)
    # 2) search_sources_more — more-only by default
    for i, it in enumerate(_as_list(record.get("search_sources_more"))):
        add(it, "search_sources_more", 0, position=i + 1)
    # 3) search_sources — observable; keep rank as observed_rank
    for it in _as_list(record.get("search_sources")):
        rk = (it or {}).get("rank") if isinstance(it, dict) else None
        add(it, "search_sources", 0, position=rk, observed_rank=rk)
    # 4) links_attached — fallback cited only if no cited source found yet
    if not any(s["cited_label"] == 1 for s in acc.values()):
        for i, it in enumerate(_as_list(record.get("links_attached"))):
            add(it, "links_attached", 1, position=i + 1)
    # 5) response_raw — last-resort fallback only if nothing else found
    if not acc:
        found: list[dict] = []
        _walk_response_raw(_jsonish(record.get("response_raw")), found)
        for it in found:
            rk = it.get("rank")
            add(it, "response_raw", 0, position=rk, observed_rank=rk)

    return list(acc.values())


# --------------------------------------------------------------------------- #
# record + run
# --------------------------------------------------------------------------- #
def load_records(raw: str | bytes, filename: str = "") -> tuple[list[dict], list[str]]:
    """Load raw records from a JSON or CSV export. Returns (records, warnings)."""
    warnings: list[str] = []
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    name = (filename or "").lower()
    is_csv = name.endswith(".csv") or (not name.endswith(".json") and not text.lstrip()[:1] in ("[", "{"))

    if is_csv:
        try:
            rows = list(csv.DictReader(io.StringIO(text)))
        except Exception as exc:  # noqa: BLE001
            return [], [f"CSV parse failed: {exc}"]
        # JSON-decode any cell that holds a JSON blob (citations, sources, raw…)
        for r in rows:
            for k, v in list(r.items()):
                r[k] = _jsonish(v)
        return rows, warnings

    try:
        data = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return [], [f"JSON parse failed: {exc}"]
    if isinstance(data, dict):
        # tolerate {"data":[...]} / {"records":[...]} / single record
        data = data.get("data") or data.get("records") or data.get("results") or [data]
    if not isinstance(data, list):
        return [], ["Top-level JSON is not an array of records."]
    return data, warnings


def parse_run(raw: str | bytes, filename: str = "") -> dict:
    """Parse a Bright Data export into a normalized ChatGPT run object."""
    records_raw, warnings = load_records(raw, filename)
    norm_records: list[dict] = []
    n_cited = n_more = 0

    for idx, rec in enumerate(records_raw):
        if not isinstance(rec, dict):
            warnings.append(f"Record {idx + 1} is not an object — skipped.")
            continue
        record_id = str(rec.get("record_id") or rec.get("id") or f"rec-{idx + 1}")
        prompt = (rec.get("prompt") or "").strip() or _prompt_from_url(rec.get("url", ""))
        answer_md = (rec.get("answer_text_markdown") or "").strip()
        answer_text = (rec.get("answer_text") or "").strip()
        sources = extract_sources(rec, record_id)
        n_cited += sum(1 for s in sources if s["cited_label"] == 1)
        n_more += sum(1 for s in sources if s["cited_label"] == 0)
        if not prompt:
            warnings.append(f"Record {idx + 1}: no prompt found (used URL/empty).")
        norm_records.append({
            "record_id": record_id,
            "prompt": prompt,
            "answer_text": answer_md or answer_text,          # prefer markdown
            "answer_markdown": answer_md,
            "web_search_query": _normalize_queries(rec.get("web_search_query")),
            "web_search_triggered": _coerce_bool(rec.get("web_search_triggered")),
            "model": rec.get("model"),
            "timestamp": rec.get("timestamp"),
            "url": rec.get("url"),
            "input": rec.get("input") if isinstance(rec.get("input"), dict) else None,
            "raw": {k: rec.get(k) for k in ("model", "timestamp", "url", "web_search_triggered") if k in rec},
            "sources": sources,
        })

    total_sources = n_cited + n_more
    looks_like_input = bool(norm_records) and total_sources == 0
    if looks_like_input:
        warnings.append(
            "No sources found in any record (0 citations / search_sources). This looks like a "
            "Bright Data INPUT / prompt file, not an OUTPUT / results export. Upload the results "
            "file — usually a large JSON (e.g. sd_*.json) that contains 'citations' and "
            "'search_sources'."
        )

    return {
        "run_id": "CG-" + new_run_id(),
        "mode": "chatgpt_brightdata",
        "created_at": now_iso(),
        "source_file_name": filename or "uploaded",
        "records": norm_records,
        "warnings": warnings,
        "n_records": len(norm_records),
        "n_sources": total_sources,
        "n_cited": n_cited,
        "n_more_only": n_more,
        "looks_like_input": looks_like_input,
    }


# --------------------------------------------------------------------------- #
# Prompt Manifest (CSV/JSON): prompt_id, topic, intent, prompt, country,
# prompt_language [, expected_source_types]. Matched to records to attach intent.
# --------------------------------------------------------------------------- #
def _prompt_key(text: str) -> str:
    """Normalized prompt for matching (case/whitespace/trailing punctuation-insensitive)."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    return t.strip(" ?.!\"'")


def _split_expected(v: Any) -> list[str]:
    if not v:
        return []
    toks = v if isinstance(v, list) else re.split(r"[;,|]", str(v))
    return [t.strip() for t in toks if t and t.strip()]


def parse_manifest(raw: str | bytes, filename: str = "") -> dict:
    """Parse a Prompt Manifest into match-ready entries."""
    records, warnings = load_records(raw, filename)
    entries: list[dict] = []
    cols: set[str] = set()
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            continue
        cols.update(r.keys())
        prompt = (r.get("prompt") or "").strip()
        phash = (str(r.get("prompt_hash")).strip() if r.get("prompt_hash") else None)
        if not prompt and not phash:
            warnings.append(f"Manifest row {i + 1}: no prompt or prompt_hash — skipped.")
            continue
        entries.append({
            "prompt_id": str(r.get("prompt_id") or r.get("id") or f"P{i + 1}"),
            "topic": (r.get("topic") or "").strip(),
            "intent": (r.get("intent") or "").strip() or "Unspecified",
            "prompt": prompt,
            "country": r.get("country"),
            "prompt_language": r.get("prompt_language") or r.get("language"),
            "expected_source_types": _split_expected(r.get("expected_source_types")),
            "prompt_hash": phash,
            "_key": _prompt_key(prompt),
        })
    has_expected = any(e["expected_source_types"] for e in entries)
    return {"entries": entries, "warnings": warnings, "columns": sorted(cols),
            "has_expected": has_expected, "n": len(entries),
            "source_file_name": filename or "manifest"}


def apply_manifest(run: dict, manifest: dict) -> dict:
    """Attach prompt_id/topic/intent/expected to each record + its sources (match by prompt/hash)."""
    by_key, by_hash = {}, {}
    for e in manifest.get("entries", []):
        if e["_key"]:
            by_key.setdefault(e["_key"], e)
        if e["prompt_hash"]:
            by_hash.setdefault(e["prompt_hash"], e)

    matched, unmatched = 0, []
    for rec in run.get("records", []):
        rh = (str(rec.get("prompt_hash")).strip() if rec.get("prompt_hash") else None)
        e = (by_hash.get(rh) if rh else None) or by_key.get(_prompt_key(rec.get("prompt", "")))
        if e is None:
            rec["intent"] = "(unmatched)"
            rec["topic"] = rec.get("topic") or "(unmatched)"
            rec["expected_source_types"] = []
            unmatched.append((rec.get("prompt") or "")[:60])
        else:
            matched += 1
            rec["intent"] = e["intent"]
            rec["topic"] = e["topic"]
            rec["prompt_id"] = e["prompt_id"]
            rec["country"] = rec.get("country") or e["country"]
            rec["prompt_language"] = e["prompt_language"]
            rec["expected_source_types"] = e["expected_source_types"]
        for s in rec.get("sources", []):
            s["intent"] = rec.get("intent")
            s["topic"] = rec.get("topic")
            s["prompt_id"] = rec.get("prompt_id")

    run["has_intent"] = True
    run["manifest"] = {"applied": True, "matched": matched, "unmatched": len(unmatched),
                       "total": len(run.get("records", [])), "has_expected": manifest.get("has_expected", False),
                       "source_file_name": manifest.get("source_file_name")}
    return {"matched": matched, "unmatched": len(unmatched),
            "unmatched_prompts": unmatched[:10], "total": len(run.get("records", []))}
