from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src import brightdata
from src.chatgpt_pipeline import flatten_sources
from src.url_utils import domain as url_domain
from src.url_utils import normalize_url

MISSING_INTENT_LABEL = "missing_intent"


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold()).strip(" ?.!\"'")


def stable_hash(*parts: Any, n: int = 16) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part if part is not None else "").encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:n]


def _raw_from_any(ai_data: Any) -> str:
    if isinstance(ai_data, (str, bytes)):
        return ai_data.decode("utf-8", "replace") if isinstance(ai_data, bytes) else ai_data
    return json.dumps(ai_data, ensure_ascii=False)


def _manifest_from_any(manifest: Any) -> dict:
    if manifest is None:
        return {"entries": [], "warnings": [], "n": 0}
    if isinstance(manifest, dict) and "entries" in manifest:
        return manifest
    if isinstance(manifest, pd.DataFrame):
        if manifest.empty:
            return {"entries": [], "warnings": [], "n": 0}
        return brightdata.parse_manifest(manifest.to_csv(index=False), "manifest.csv")
    if isinstance(manifest, (str, bytes)):
        return brightdata.parse_manifest(manifest, "manifest.csv")
    return {"entries": [], "warnings": [f"Unsupported manifest object: {type(manifest).__name__}"], "n": 0}


def read_manifest(path: str | Path) -> dict:
    p = Path(path)
    return brightdata.parse_manifest(p.read_text("utf-8"), p.name)


def _record_meta(run: dict) -> dict[str, dict[str, Any]]:
    return {str(r.get("record_id")): r for r in run.get("records", [])}


def can_construct_cited(ai_data: Any) -> bool:
    run = brightdata.parse_run(_raw_from_any(ai_data), "input.json")
    return bool(run.get("n_sources", 0)) and (run.get("n_cited", 0) + run.get("n_more_only", 0) > 0)


def audit_ai_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    run = brightdata.parse_run(p.read_text("utf-8"), p.name)
    flat = flatten_sources(run)
    out.update(
        top_level_type="brightdata_parse_run",
        top_level_keys=None,
        n_records=int(run.get("n_records", 0)),
        raw_records=int(run.get("n_records", 0)),
        surfaced_sources_exist=bool(flat),
        n_surfaced_sources=len(flat),
        surfaced_sources_found=len(flat),
        cited_source_indicators_exist=bool(run.get("n_cited", 0) or run.get("n_more_only", 0)),
        cited_sources_found=int(run.get("n_cited", 0)),
        more_only_sources_found=int(run.get("n_more_only", 0)),
        source_urls_exist=any(s.get("url") for s in flat),
        source_titles_exist=any(s.get("title") for s in flat),
        source_descriptions_or_snippets_exist=any(s.get("description") for s in flat),
        cited_can_be_constructed=bool(flat and (run.get("n_cited", 0) + run.get("n_more_only", 0) > 0)),
        warnings=run.get("warnings", []),
    )
    return out


def audit_manifest(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    out: dict[str, Any] = {"path": str(p), "exists": p.exists()}
    if not p.exists():
        return out
    manifest = brightdata.parse_manifest(p.read_text("utf-8"), p.name)
    cols = manifest.get("columns", [])
    out.update(
        shape=[int(manifest.get("n", 0)), len(cols)],
        columns=cols,
        prompt_id_available="prompt_id" in cols,
        record_id_available="record_id" in cols,
        prompt_text_available="prompt" in cols,
        intent_available="intent" in cols,
        topic_available="topic" in cols,
        language_available=any(c in cols for c in ("language", "prompt_language")),
        country_available="country" in cols,
        candidate_join_keys=[c for c in ("prompt_hash", "prompt") if c in cols],
        warnings=manifest.get("warnings", []),
    )
    return out


def build_source_rows(ai_data: Any, manifest: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build one row per surfaced source using the original CiteScope parser.

    Reuses:
    - ``brightdata.parse_run`` / ``extract_sources`` for cited vs more-only
    - ``brightdata.parse_manifest`` / ``apply_manifest`` for prompt metadata
    - ``chatgpt_pipeline.flatten_sources`` for record context
    """
    run = brightdata.parse_run(_raw_from_any(ai_data), "ai_search.json")
    manifest_obj = _manifest_from_any(manifest)
    manifest_stats = None
    if manifest_obj.get("entries"):
        manifest_stats = brightdata.apply_manifest(run, manifest_obj)
    flat = flatten_sources(run)
    if not flat:
        raise ValueError("Cannot construct cited outcome. Input must include surfaced sources and cited-source indicators.")

    rec_by_id = _record_meta(run)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = list(run.get("warnings", [])) + list(manifest_obj.get("warnings", []))
    for i, s in enumerate(flat, start=1):
        rec = rec_by_id.get(str(s.get("record_id")), {})
        cited = int(s.get("cited_label", 0))
        nurl = s.get("normalized_url") or normalize_url(s.get("url", ""))
        source_row_id = str(s.get("source_id") or f"{s.get('record_id')}:{stable_hash(nurl, i)}")
        prompt_text = s.get("prompt") or rec.get("prompt") or ""
        row_hash = stable_hash(run.get("run_id"), s.get("record_id"), source_row_id, nurl, cited, n=32)
        expected = s.get("expected_source_types") or rec.get("expected_source_types") or []
        row = {
            "run_id": run.get("run_id"),
            "answer_id": s.get("record_id"),
            "record_id": s.get("record_id"),
            "prompt_id": s.get("prompt_id") or rec.get("prompt_id"),
            "prompt_text": prompt_text,
            "intent": s.get("intent") or rec.get("intent") or "",
            "intent_plot_label": (s.get("intent") or rec.get("intent") or "").strip() or MISSING_INTENT_LABEL,
            "topic": s.get("topic") or rec.get("topic") or "",
            "language": rec.get("prompt_language") or "",
            "prompt_language": rec.get("prompt_language") or "",
            "country": rec.get("country") or "",
            "expected_source_types": ";".join(expected) if isinstance(expected, list) else (expected or ""),
            "answer_text": s.get("answer_text") or rec.get("answer_text") or "",
            "source_id": s.get("source_id"),
            "source_row_id": source_row_id,
            "source_position": s.get("source_position"),
            "observed_rank": s.get("observed_rank"),
            "source_title": s.get("title") or "",
            "source_url": s.get("url") or "",
            "normalized_url": nurl,
            "canonical_url": s.get("canonical_url"),
            "final_url": s.get("final_url"),
            "source_domain": s.get("domain") or url_domain(nurl),
            "source_description": s.get("description") or "",
            "source_snippet": s.get("snippet") or "",
            "cited": cited,
            "cited_label": cited,
            "is_more_only": int(cited == 0),
            "source_group": s.get("source_group") or ("cited" if cited else "more_only"),
            "source_origin": s.get("source_origin") or "",
            "row_hash": row_hash,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    cited = pd.to_numeric(df["cited"], errors="coerce")
    if not set(cited.dropna().unique()).issubset({0, 1}):
        raise ValueError("cited must be binary 0/1")
    if int((cited == 1).sum()) == 0:
        warnings.append("outcome_has_no_cited_sources")
    if int((cited == 0).sum()) == 0:
        warnings.append("outcome_has_single_class_or_no_more_only")
    if df["row_hash"].duplicated().any():
        warnings.append("duplicate row_hash detected")
    if df["source_row_id"].duplicated().any():
        warnings.append("duplicate source_row_id detected")

    summary = {
        "raw_records": int(run.get("n_records", 0)),
        "surfaced_sources_found": int(len(df)),
        "cited_sources_found": int((cited == 1).sum()),
        "more_only_sources_found": int((cited == 0).sum()),
        "source_rows_exported": int(len(df)),
        "rows": int(len(df)),
        "unique_prompts": int(df["prompt_id"].replace("", pd.NA).nunique(dropna=True) or df["prompt_text"].nunique(dropna=True)),
        "unique_records": int(df["record_id"].nunique(dropna=True)),
        "unique_answers": int(df["answer_id"].nunique(dropna=True)),
        "unique_urls": int(df["normalized_url"].replace("", pd.NA).nunique(dropna=True)),
        "cited_count": int((cited == 1).sum()),
        "more_only_count": int((cited == 0).sum()),
        "cited_rate": float(cited.mean()) if len(cited) else 0.0,
        "rows_with_intent": int((df["intent_plot_label"] != MISSING_INTENT_LABEL).sum()),
        "rows_missing_intent": int((df["intent_plot_label"] == MISSING_INTENT_LABEL).sum()),
        "intent_missing_rate": float((df["intent_plot_label"] == MISSING_INTENT_LABEL).mean()) if len(df) else 0.0,
        "manifest": manifest_stats or {},
        "warnings": warnings,
    }
    return df, summary


def build_source_rows_from_files(ai_json: str | Path, manifest_path: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    ai_path = Path(ai_json)
    manifest_obj = None
    if manifest_path:
        mp = Path(manifest_path)
        if mp.exists():
            manifest_obj = brightdata.parse_manifest(mp.read_text("utf-8"), mp.name)
    df, summary = build_source_rows(ai_path.read_text("utf-8"), manifest_obj)
    summary["input_ai_json"] = str(ai_path)
    summary["manifest_path"] = str(manifest_path) if manifest_path else None
    return df, summary
