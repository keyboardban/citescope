"""Gemini API wrapper: grounded generation + trace extraction + embeddings.

Uses the stable `google-genai` generate_content path with the Google Search tool
and reads the grounding metadata it exposes. Everything is defensive: missing
fields degrade gracefully and the raw response is always preserved. We never
fabricate search queries or citations.
"""

from __future__ import annotations

import json
from typing import Any, Sequence


def _lazy_genai():
    from google import genai
    from google.genai import types
    return genai, types


def build_client(api_key: str):
    genai, _ = _lazy_genai()
    return genai.Client(api_key=api_key)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _first(obj: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _response_to_dict(resp: Any) -> dict:
    for attr in ("model_dump", "to_json_dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()  # type: ignore[misc]
            except Exception:
                pass
    fn = getattr(resp, "model_dump_json", None)
    if callable(fn):
        try:
            return json.loads(fn())
        except Exception:
            pass
    return {"repr": str(resp)[:8000]}


def _extract_text(resp: Any) -> str:
    text = _first(resp, "text")
    if text:
        return text
    parts_text: list[str] = []
    for cand in (_first(resp, "candidates", default=[]) or []):
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            t = getattr(part, "text", None)
            if t:
                parts_text.append(t)
    return "\n".join(parts_text).strip()


def _extract_grounding(resp: Any) -> dict:
    """Pull web_search_queries / grounding_chunks / grounding_supports."""
    out = {
        "search_queries": [],
        "citations": [],
        "supports": [],
        "search_entry_point_html": None,
    }
    cands = _first(resp, "candidates", default=[]) or []
    if not cands:
        return out
    gm = getattr(cands[0], "grounding_metadata", None)
    if gm is None:
        # Fallback: newer annotation-style citations (url_citation).
        out["citations"] = _extract_annotations(cands[0])
        return out

    out["search_queries"] = list(_first(gm, "web_search_queries", default=[]) or [])

    chunks = _first(gm, "grounding_chunks", default=[]) or []
    for i, ch in enumerate(chunks):
        web = getattr(ch, "web", None)
        if web is None:
            continue
        uri = getattr(web, "uri", None)
        if not uri:
            continue
        out["citations"].append(
            {"index": i, "raw_uri": uri, "title": getattr(web, "title", "") or ""}
        )

    supports = _first(gm, "grounding_supports", default=[]) or []
    for sup in supports:
        seg = getattr(sup, "segment", None)
        out["supports"].append(
            {
                "text": getattr(seg, "text", "") or "",
                "chunk_indices": list(_first(sup, "grounding_chunk_indices", default=[]) or []),
                "confidence": list(_first(sup, "confidence_scores", default=[]) or []),
            }
        )

    sep = getattr(gm, "search_entry_point", None)
    if sep is not None:
        out["search_entry_point_html"] = getattr(sep, "rendered_content", None)
    return out


def _extract_diagnostics(resp: Any) -> dict:
    """Capture why a run produced no text (finish reason / prompt blocks)."""
    diag = {"finish_reason": None, "prompt_feedback": None}
    cands = _first(resp, "candidates", default=[]) or []
    if cands:
        fr = getattr(cands[0], "finish_reason", None)
        diag["finish_reason"] = str(fr) if fr is not None else None
    pf = _first(resp, "prompt_feedback", default=None)
    if pf is not None:
        diag["prompt_feedback"] = str(pf)
    return diag


def _extract_annotations(candidate: Any) -> list[dict]:
    """Support newer url_citation annotations if grounding_metadata is absent."""
    cites: list[dict] = []
    content = getattr(candidate, "content", None)
    seen = set()
    for part in (getattr(content, "parts", None) or []):
        for ann in (getattr(part, "annotations", None) or []):
            uri = getattr(ann, "url", None) or getattr(ann, "uri", None)
            if uri and uri not in seen:
                seen.add(uri)
                cites.append(
                    {"index": len(cites), "raw_uri": uri, "title": getattr(ann, "title", "") or ""}
                )
    return cites


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def run_grounded(
    client,
    prompt: str,
    model: str,
    temperature: float = 0.2,
    grounding: bool = True,
    system_prompt: str | None = None,
) -> dict:
    """Run one grounded generation and return the extracted trace.

    Returns a dict with: output_text, search_queries, citations (raw_uri+title),
    supports, search_entry_point_html, raw, and any error string.
    """
    _, types = _lazy_genai()

    cfg_kwargs: dict[str, Any] = {"temperature": float(temperature)}
    if grounding:
        cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if system_prompt:
        cfg_kwargs["system_instruction"] = system_prompt
    config = types.GenerateContentConfig(**cfg_kwargs)

    error = None
    try:
        resp = client.models.generate_content(model=model, contents=prompt, config=config)
    except Exception as exc:  # surface API errors to the UI rather than crashing
        return {
            "output_text": "",
            "search_queries": [],
            "citations": [],
            "supports": [],
            "search_entry_point_html": None,
            "finish_reason": None,
            "prompt_feedback": None,
            "raw": None,
            "error": f"{type(exc).__name__}: {exc}",
            "model": model,
            "grounding": grounding,
        }

    grounding_data = _extract_grounding(resp)
    output_text = _extract_text(resp)
    diag = _extract_diagnostics(resp)
    if not output_text and not error:
        if diag["prompt_feedback"]:
            error = f"Prompt blocked (prompt_feedback={diag['prompt_feedback']})."
        elif diag["finish_reason"] and "STOP" not in diag["finish_reason"]:
            error = f"No answer text (finish_reason={diag['finish_reason']})."
        else:
            error = ("Model returned no answer text — verify the model name supports Google Search "
                     "grounding for your account/region.")
    return {
        "output_text": output_text,
        "search_queries": grounding_data["search_queries"],
        "citations": grounding_data["citations"],
        "supports": grounding_data["supports"],
        "search_entry_point_html": grounding_data["search_entry_point_html"],
        "finish_reason": diag["finish_reason"],
        "prompt_feedback": diag["prompt_feedback"],
        "raw": _response_to_dict(resp),
        "error": error,
        "model": model,
        "grounding": grounding,
    }


def embed_texts(client, texts: Sequence[str], model: str = "text-embedding-004") -> list[list[float]]:
    """Embed a batch of texts. Returns one vector per input (zeros on failure)."""
    texts = list(texts)
    if not texts:
        return []
    try:
        resp = client.models.embed_content(model=model, contents=texts)
        embs = _first(resp, "embeddings", default=None)
        if embs is not None:
            return [list(_first(e, "values", default=[]) or []) for e in embs]
        # single-embedding shape
        single = _first(resp, "embedding", default=None)
        if single is not None:
            return [list(getattr(single, "values", single))]
    except Exception:
        pass
    # Fall back to per-text calls; never raise into the pipeline.
    out: list[list[float]] = []
    for t in texts:
        try:
            r = client.models.embed_content(model=model, contents=t)
            embs = _first(r, "embeddings", default=None)
            if embs:
                out.append(list(_first(embs[0], "values", default=[]) or []))
            else:
                out.append([])
        except Exception:
            out.append([])
    return out
