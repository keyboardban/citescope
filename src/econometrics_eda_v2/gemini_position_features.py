"""Gemini-assisted semantic block classification for position-feature QA.

The LLM never calculates positions and never receives citation outcomes. HTML is
cleaned and converted into deterministic, token-positioned blocks first. Gemini
returns evidence block IDs; this module maps those IDs back to positions.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field

from src import config
from src.econometrics_eda_v2.paths import topic_output_dir
from src.econometrics_eda_v2.position_feature_eda import (
    COMPARISON_RE,
    DEFINITION_RE,
    DIRECT_ANSWER_RE,
    EVIDENCE_NUMBER_RE,
    QUESTION_RE,
    STEPS_RE,
    _node_start_positions,
    _remove_noise,
    _select_main_node,
    _tokens,
    _valid_list,
    _visible_text,
)


GEMINI_POSITION_VERSION = "gemini_position_semantic_prompt_v2_20260803"
DEFAULT_MODEL = os.getenv("CITESCOPE_POSITION_GEMINI_MODEL", "gemini-3.1-flash-lite")
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs/position_feature_eda_final_20260731/llm_semantic_smoke"
)
FRONTEND_DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "outputs/position_feature_eda_final_20260731/frontend/data"
)

SEMANTIC_FEATURES = (
    "direct_answer",
    "definition",
    "comparison",
    "steps",
    "numeric_evidence",
    "question_heading",
)
SEMANTIC_FEATURE_LABELS = {
    "direct_answer": "Direct answer",
    "definition": "Definition",
    "comparison": "Comparison",
    "steps": "Steps / procedure",
    "numeric_evidence": "Numeric evidence",
    "question_heading": "Question heading",
}
FEATURE_TO_RULE_COLUMN = {
    "direct_answer": "has_direct_answer",
    "definition": "has_definition_block",
    "comparison": "has_comparison",
    "steps": "has_steps",
    "numeric_evidence": "has_numeric_evidence",
    "question_heading": "has_question_heading",
}
BLOCK_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "dl", "ul", "ol", "table")
STRUCTURAL_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "dl", "ul", "ol", "table"}
PROMPT_BLOCK_FIELDS = {"block_id", "tag", "text"}


FeatureName = Literal[
    "direct_answer",
    "definition",
    "comparison",
    "steps",
    "numeric_evidence",
    "question_heading",
]
Confidence = Literal["high", "medium", "low"]


class SemanticDetection(BaseModel):
    block_id: str = Field(description="Exact block_id from the supplied block list.")
    feature: FeatureName
    confidence: Confidence
    rationale: str = Field(description="Short explanation grounded only in this block and nearby supplied context.")


class SemanticDetectionResponse(BaseModel):
    detections: list[SemanticDetection] = Field(default_factory=list)


@dataclass(frozen=True)
class ContentBlock:
    block_id: str
    tag: str
    start_token: int
    position_ratio: float
    text: str
    rule_candidates: tuple[str, ...]


def _candidate_features(tag: str, text: str) -> tuple[str, ...]:
    found: list[str] = []
    if tag.startswith("h") and QUESTION_RE.search(text):
        found.append("question_heading")
    if DIRECT_ANSWER_RE.search(text):
        found.append("direct_answer")
    if tag == "dl" or DEFINITION_RE.search(text):
        found.append("definition")
    if COMPARISON_RE.search(text):
        found.append("comparison")
    if tag == "ol" or STEPS_RE.search(text):
        found.append("steps")
    if EVIDENCE_NUMBER_RE.search(text):
        found.append("numeric_evidence")
    return tuple(found)


def _nested_in_structural_parent(tag: Tag) -> bool:
    parent = tag.parent
    while isinstance(parent, Tag):
        if parent.name in {"table", "ul", "ol", "dl"}:
            return True
        parent = parent.parent
    return False


def extract_semantic_blocks(
    raw_html: str,
    *,
    max_blocks: int = 80,
    max_chars: int = 18_000,
) -> tuple[list[ContentBlock], dict[str, Any]]:
    """Create deterministic main-content blocks with token positions."""
    if not str(raw_html or "").strip():
        return [], {
            "block_extraction_status": "no_html",
            "block_extraction_method": "none",
            "total_main_content_tokens": 0,
            "total_blocks_before_cap": 0,
            "selected_blocks": 0,
            "blocks_truncated": False,
        }

    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_noise(soup)
    main, method = _select_main_node(soup)
    starts, total_tokens = _node_start_positions(main)
    if total_tokens <= 0:
        return [], {
            "block_extraction_status": "main_content_parse_failed",
            "block_extraction_method": method,
            "total_main_content_tokens": 0,
            "total_blocks_before_cap": 0,
            "selected_blocks": 0,
            "blocks_truncated": False,
        }

    candidates: list[tuple[int, int, str, str, tuple[str, ...]]] = []
    for order, tag in enumerate(main.find_all(BLOCK_TAGS)):
        tag_name = str(tag.name).casefold()
        if tag_name in {"p", "blockquote"} and _nested_in_structural_parent(tag):
            continue
        if tag_name in {"ul", "ol"} and not _valid_list(tag)[0]:
            continue
        text = _visible_text(tag)
        if not text:
            continue
        start = starts.get(id(tag))
        if start is None:
            continue
        rule_candidates = _candidate_features(tag_name, text)
        priority = 3 if tag_name in STRUCTURAL_TAGS else (2 if rule_candidates else 1)
        candidates.append((order, priority, tag_name, text[:1600], rule_candidates))

    total_before_cap = len(candidates)
    if len(candidates) > max_blocks or sum(len(item[3]) for item in candidates) > max_chars:
        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        selected: list[tuple[int, int, str, str, tuple[str, ...]]] = []
        chars = 0
        for item in ranked:
            if len(selected) >= max_blocks:
                break
            if selected and chars + len(item[3]) > max_chars:
                continue
            selected.append(item)
            chars += len(item[3])
        candidates = sorted(selected, key=lambda item: item[0])

    blocks: list[ContentBlock] = []
    # Re-resolve positions by order because candidate tuples deliberately avoid retaining DOM objects.
    candidate_lookup = {(order, tag_name, text): rule for order, _, tag_name, text, rule in candidates}
    selected_orders = {item[0] for item in candidates}
    emitted = 0
    for order, tag in enumerate(main.find_all(BLOCK_TAGS)):
        if order not in selected_orders:
            continue
        tag_name = str(tag.name).casefold()
        text = _visible_text(tag)[:1600]
        key = (order, tag_name, text)
        if key not in candidate_lookup:
            continue
        start = starts.get(id(tag))
        if start is None:
            continue
        emitted += 1
        blocks.append(
            ContentBlock(
                block_id=f"B{emitted:04d}",
                tag=tag_name,
                start_token=int(start),
                position_ratio=float(np.clip(start / total_tokens, 0, 1)),
                text=text,
                rule_candidates=candidate_lookup[key],
            )
        )

    return blocks, {
        "block_extraction_status": "measured" if blocks else "no_eligible_blocks",
        "block_extraction_method": f"filtered_main_content_html:{method}",
        "total_main_content_tokens": int(total_tokens),
        "total_blocks_before_cap": int(total_before_cap),
        "selected_blocks": len(blocks),
        "blocks_truncated": len(blocks) < total_before_cap,
    }


def chunk_blocks(
    blocks: list[ContentBlock],
    *,
    max_blocks: int = 24,
    max_chars: int = 7_500,
) -> list[list[ContentBlock]]:
    """Pack contiguous blocks into bounded requests without splitting a block."""
    chunks: list[list[ContentBlock]] = []
    current: list[ContentBlock] = []
    chars = 0
    for block in blocks:
        size = len(block.text)
        if current and (len(current) >= max_blocks or chars + size > max_chars):
            chunks.append(current)
            current = []
            chars = 0
        current.append(block)
        chars += size
    if current:
        chunks.append(current)
    return chunks


def _system_instruction() -> str:
    return """You classify the semantic function of pre-extracted webpage content blocks.
Use only supplied block text and structural tags. Never infer citation outcomes, popularity, SEO value, or page rank.
Return a detection only when the semantic function is genuinely present:
- question_heading: an h1-h6 block that syntactically asks a meaningful user-facing question. It needs a question mark or a clear interrogative construction such as what, why, how, which, who, where, when, whether, can, should, อะไร, ทำไม, อย่างไร, ไหน, ใคร, ที่ไหน, เมื่อไหร่, หรือไม่, or ไหม. Do not label section titles such as FAQ, things to know, guide, overview, or checklist merely because they introduce answers. Paragraphs cannot receive this label.
- comparison: explicitly contrasts at least two options, entities, states, or attributes; a passing mention of 'different' is insufficient.
- definition: explicitly defines or explains the meaning of a concept; ordinary copular sentences are insufficient.
- steps: an actionable sequence that tells the reader what to do in an order, using instructions, imperatives, or a procedural checklist. Do not label calculations, benefits, rankings, timelines, directories, criteria lists, or ordinary numbered lists as steps.
- direct_answer: a concise answer that directly resolves a supplied question. The question must occur in the same block or an immediately preceding supplied block. Do not label summaries, key facts, introductory claims, or data lists when no question is present.
- numeric_evidence: a factual quantity with meaningful context; dates, navigation numbers, IDs, and decorative counters alone are insufficient.
The same block may receive multiple features. Omit blocks with no supported feature. Use exact supplied block_id values."""


def build_chunk_prompt(blocks: list[ContentBlock], *, page_title: str = "") -> str:
    payload = [
        {
            "block_id": block.block_id,
            "tag": block.tag,
            "text": block.text,
        }
        for block in blocks
    ]
    prompt = (
        "Classify the semantic functions in these webpage blocks. "
        "The title is context only and must not itself be labeled unless supplied as a block.\n"
        f"PAGE_TITLE: {str(page_title or '')[:500]}\n"
        f"BLOCKS_JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    if any(set(item) != PROMPT_BLOCK_FIELDS for item in payload):
        raise ValueError("Gemini position prompt contains an unexpected block field")
    return prompt


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        try:
            return response.model_dump(mode="json")
        except TypeError:
            return response.model_dump()
    return {"repr": str(response)[:8000]}


def _usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def call_gemini_classifier(
    client: Any,
    blocks: list[ContentBlock],
    *,
    page_title: str,
    model: str = DEFAULT_MODEL,
) -> tuple[SemanticDetectionResponse, dict[str, Any], dict[str, int | None]]:
    """Classify one bounded block chunk with structured JSON output."""
    from google.genai import types

    from src.retry import with_retry

    response = with_retry(
        lambda: client.models.generate_content(
            model=model,
            contents=build_chunk_prompt(blocks, page_title=page_title),
            config=types.GenerateContentConfig(
                system_instruction=_system_instruction(),
                temperature=1.0,
                max_output_tokens=1800,
                response_mime_type="application/json",
                response_schema=SemanticDetectionResponse,
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,
                    thinking_level=types.ThinkingLevel.MINIMAL,
                ),
            ),
        )
    )
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, SemanticDetectionResponse):
        result = parsed
    elif isinstance(parsed, dict):
        result = SemanticDetectionResponse.model_validate(parsed)
    else:
        result = SemanticDetectionResponse.model_validate_json(str(getattr(response, "text", "") or ""))
    return result, _response_dict(response), _usage_dict(response)


def _cache_key(model: str, page_id: str, chunk_index: int, prompt: str) -> str:
    value = f"{GEMINI_POSITION_VERSION}\n{model}\n{page_id}\n{chunk_index}\n{prompt}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_detections(
    response: SemanticDetectionResponse,
    blocks: list[ContentBlock],
) -> list[SemanticDetection]:
    known = {block.block_id: block for block in blocks}
    valid: list[SemanticDetection] = []
    seen: set[tuple[str, str]] = set()
    for detection in response.detections:
        block = known.get(detection.block_id)
        key = (detection.block_id, detection.feature)
        if block is None or key in seen:
            continue
        if detection.feature == "question_heading" and not block.tag.startswith("h"):
            continue
        seen.add(key)
        valid.append(detection)
    return valid


def classify_page(
    *,
    client: Any | None,
    normalized_url: str,
    page_title: str,
    blocks: list[ContentBlock],
    cache_dir: Path,
    model: str,
    execute_live: bool,
    force: bool,
    request_fn: Callable[..., tuple[SemanticDetectionResponse, dict[str, Any], dict[str, int | None]]] | None = None,
) -> tuple[list[SemanticDetection], dict[str, Any]]:
    """Classify all chunks for one page with per-chunk durable cache."""
    page_id = hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    detections: list[SemanticDetection] = []
    errors: list[str] = []
    input_tokens = output_tokens = thinking_tokens = total_tokens = 0
    cached_chunks = live_chunks = 0
    chunks = chunk_blocks(blocks)
    caller = request_fn or call_gemini_classifier

    for chunk_index, chunk in enumerate(chunks):
        prompt = build_chunk_prompt(chunk, page_title=page_title)
        key = _cache_key(model, page_id, chunk_index, prompt)
        path = cache_dir / f"{page_id}_{chunk_index:03d}_{key[:12]}.json"
        payload: dict[str, Any] | None = None
        if path.exists() and not force:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cached_chunks += 1
            except (OSError, json.JSONDecodeError):
                payload = None

        if payload is None and not execute_live:
            continue
        if payload is None:
            try:
                parsed, raw, usage = caller(
                    client,
                    chunk,
                    page_title=page_title,
                    model=model,
                )
                valid = _validate_detections(parsed, chunk)
                payload = {
                    "version": GEMINI_POSITION_VERSION,
                    "model": model,
                    "normalized_url": normalized_url,
                    "chunk_index": chunk_index,
                    "input_block_ids": [block.block_id for block in chunk],
                    "detections": [item.model_dump() for item in valid],
                    "usage": usage,
                    "raw_response": raw,
                    "error": "",
                }
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                temp.replace(path)
                live_chunks += 1
            except Exception as exc:
                errors.append(f"chunk_{chunk_index}:{type(exc).__name__}:{exc}")
                continue

        try:
            parsed_cached = SemanticDetectionResponse.model_validate(
                {"detections": payload.get("detections", [])}
            )
            detections.extend(_validate_detections(parsed_cached, chunk))
            usage = payload.get("usage") or {}
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
            thinking_tokens += int(usage.get("thinking_tokens") or 0)
            total_tokens += int(usage.get("total_tokens") or 0)
            if payload.get("error"):
                errors.append(str(payload["error"]))
        except Exception as exc:
            errors.append(f"chunk_{chunk_index}:cache_parse:{type(exc).__name__}:{exc}")

    deduped = {
        (item.block_id, item.feature): item
        for item in detections
    }
    if not chunks:
        status = "unmeasured_no_blocks"
    elif not execute_live and not cached_chunks:
        status = "dry_run"
    else:
        status = (
            "success"
            if not errors and (live_chunks + cached_chunks == len(chunks))
            else "partial_failure"
        )
    return list(deduped.values()), {
        "gemini_status": status,
        "gemini_error": " | ".join(errors),
        "chunks_total": len(chunks),
        "chunks_live": live_chunks,
        "chunks_cached": cached_chunks,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
    }


def aggregate_page_detections(
    blocks: list[ContentBlock],
    detections: list[SemanticDetection],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    block_map = {block.block_id: block for block in blocks}
    output: dict[str, Any] = {}
    evidence_rows: list[dict[str, Any]] = []
    for feature in SEMANTIC_FEATURES:
        selected = [item for item in detections if item.feature == feature and item.block_id in block_map]
        selected.sort(key=lambda item: block_map[item.block_id].start_token)
        output[f"has_{feature}_gemini_v1"] = int(bool(selected))
        output[f"{feature}_count_gemini_v1"] = len(selected)
        if selected:
            first = block_map[selected[0].block_id]
            output[f"first_{feature}_position_ratio_gemini_v1"] = first.position_ratio
            output[f"first_{feature}_block_id_gemini_v1"] = first.block_id
        else:
            output[f"first_{feature}_position_ratio_gemini_v1"] = np.nan
            output[f"first_{feature}_block_id_gemini_v1"] = ""
        for item in selected:
            block = block_map[item.block_id]
            evidence_rows.append(
                {
                    "feature": feature,
                    "block_id": block.block_id,
                    "tag": block.tag,
                    "start_token": block.start_token,
                    "position_ratio": block.position_ratio,
                    "confidence": item.confidence,
                    "rationale": item.rationale,
                    "evidence_text": block.text,
                }
            )
    return output, evidence_rows


def build_page_feature_scorecard(
    page: pd.Series | dict[str, Any],
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Build a six-feature manual-review scorecard for one classified page."""
    status = str(page.get("gemini_status", ""))
    measured = status == "success"
    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    rows: list[dict[str, Any]] = []

    for feature in SEMANTIC_FEATURES:
        rule_column = FEATURE_TO_RULE_COLUMN[feature]
        gemini_column = f"has_{feature}_gemini_v1"
        count_column = f"{feature}_count_gemini_v1"
        position_column = f"first_{feature}_position_ratio_gemini_v1"
        feature_evidence = (
            evidence[evidence["feature"].eq(feature)].copy()
            if not evidence.empty and "feature" in evidence
            else pd.DataFrame()
        )

        rule_value = pd.to_numeric(pd.Series([page.get(rule_column)]), errors="coerce").iloc[0]
        gemini_value = pd.to_numeric(pd.Series([page.get(gemini_column)]), errors="coerce").iloc[0]
        count_value = pd.to_numeric(pd.Series([page.get(count_column)]), errors="coerce").iloc[0]
        position_value = pd.to_numeric(
            pd.Series([page.get(position_column)]), errors="coerce"
        ).iloc[0]

        if not measured:
            gemini_score: int | str = "NA"
            detection_count: int | str = "NA"
            confidence = "unmeasured"
            first_position = "NA"
            agreement = "NA"
        else:
            gemini_score = int(gemini_value) if pd.notna(gemini_value) else 0
            detection_count = int(count_value) if pd.notna(count_value) else 0
            confidences = [
                str(value)
                for value in feature_evidence.get("confidence", pd.Series(dtype=str)).dropna()
                if str(value) in confidence_rank
            ]
            confidence = (
                max(confidences, key=confidence_rank.get)
                if confidences
                else "not detected"
            )
            first_position = f"{position_value:.1%}" if pd.notna(position_value) else "NA"
            agreement = (
                "Yes"
                if pd.notna(rule_value) and int(rule_value) == gemini_score
                else "No"
                if pd.notna(rule_value)
                else "NA"
            )

        rows.append(
            {
                "feature": SEMANTIC_FEATURE_LABELS[feature],
                "gemini_score": gemini_score,
                "confidence": confidence,
                "detections": detection_count,
                "first_position": first_position,
                "rule_score": int(rule_value) if pd.notna(rule_value) else "NA",
                "agreement": agreement,
                "measurement_status": status,
            }
        )
    return pd.DataFrame(rows)


def select_smoke_urls(
    urls: pd.DataFrame,
    max_urls: int | None,
    seed: int = 20260803,
) -> pd.DataFrame:
    """Select rule-positive coverage across features without using citation outcomes."""
    frame = urls.copy()
    measured = pd.to_numeric(frame.get("position_features_available"), errors="coerce").eq(1)
    frame = frame[measured].copy()
    if max_urls is None:
        return frame.sort_values("normalized_url", kind="stable").copy()
    selected_indices: list[int] = []
    per_feature = max(1, max_urls // max(len(SEMANTIC_FEATURES), 1))
    for offset, feature in enumerate(SEMANTIC_FEATURES):
        column = FEATURE_TO_RULE_COLUMN[feature]
        candidates = frame[pd.to_numeric(frame[column], errors="coerce").eq(1)]
        if candidates.empty:
            continue
        picks = candidates.sample(
            n=min(per_feature, len(candidates)),
            random_state=seed + offset,
        ).index.tolist()
        selected_indices.extend(index for index in picks if index not in selected_indices)
    remaining_n = max_urls - len(selected_indices)
    if remaining_n > 0:
        remaining = frame.loc[~frame.index.isin(selected_indices)]
        if not remaining.empty:
            picks = remaining.sample(
                n=min(remaining_n, len(remaining)),
                random_state=seed + 100,
            ).index.tolist()
            selected_indices.extend(picks)
    return frame.loc[selected_indices[:max_urls]].copy()


def _agreement_summary(pages: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    successful = pages[pages["gemini_status"].eq("success")]
    for feature in SEMANTIC_FEATURES:
        rule_col = FEATURE_TO_RULE_COLUMN[feature]
        llm_col = f"has_{feature}_gemini_v1"
        comparable = successful.dropna(subset=[rule_col, llm_col]).copy()
        rule = pd.to_numeric(comparable[rule_col], errors="coerce").eq(1)
        llm = pd.to_numeric(comparable[llm_col], errors="coerce").eq(1)
        records.append(
            {
                "feature": feature,
                "comparable_pages": len(comparable),
                "rule_positive_pages": int(rule.sum()),
                "gemini_positive_pages": int(llm.sum()),
                "both_positive": int((rule & llm).sum()),
                "rule_only": int((rule & ~llm).sum()),
                "gemini_only": int((~rule & llm).sum()),
                "both_negative": int((~rule & ~llm).sum()),
                "agreement_rate": float((rule == llm).mean()) if len(comparable) else np.nan,
                "quality_interpretation": "agreement_only_not_accuracy",
            }
        )
    return pd.DataFrame(records)


def _review_sample(pages: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evidence_lookup = {
        (url, feature): group.to_dict("records")
        for (url, feature), group in evidence.groupby(["normalized_url", "feature"], observed=True)
    } if not evidence.empty else {}
    for page in pages.itertuples(index=False):
        if getattr(page, "gemini_status") != "success":
            continue
        for feature in SEMANTIC_FEATURES:
            rule_raw = pd.to_numeric(
                pd.Series([getattr(page, FEATURE_TO_RULE_COLUMN[feature])]), errors="coerce"
            ).iloc[0]
            llm_raw = pd.to_numeric(
                pd.Series([getattr(page, f"has_{feature}_gemini_v1")]), errors="coerce"
            ).iloc[0]
            if pd.isna(rule_raw) or pd.isna(llm_raw):
                continue
            rule_value = int(rule_raw)
            llm_value = int(llm_raw)
            disagreement = rule_value != llm_value
            items = evidence_lookup.get((page.normalized_url, feature), [])
            rows.append(
                {
                    "review_priority": "disagreement" if disagreement else "agreement_sample",
                    "feature": feature,
                    "normalized_url": page.normalized_url,
                    "source_url": page.source_url,
                    "domain": page.source_root_domain,
                    "page_title": page.page_title,
                    "rule_value": rule_value,
                    "gemini_value": llm_value,
                    "agreement": not disagreement,
                    "gemini_evidence_block_ids": json.dumps([item["block_id"] for item in items]),
                    "gemini_confidence": ", ".join(sorted({str(item["confidence"]) for item in items})),
                    "gemini_rationale": " | ".join(str(item["rationale"]) for item in items)[:1200],
                    "gemini_evidence_text": " | ".join(str(item["evidence_text"]) for item in items)[:2400],
                    "human_review_result": "",
                    "human_review_note": "",
                }
            )
    review = pd.DataFrame(rows)
    if review.empty:
        return review
    return review.sort_values(["review_priority", "feature", "domain"], ascending=[True, True, True])


def run_gemini_position_smoke(
    *,
    position_urls_path: Path,
    document_features_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    frontend_data_dir: Path = FRONTEND_DATA_DIR,
    max_urls: int | None = 12,
    model: str = DEFAULT_MODEL,
    execute_live: bool = False,
    force: bool = False,
    request_fn: Callable[..., tuple[SemanticDetectionResponse, dict[str, Any], dict[str, int | None]]] | None = None,
) -> dict[str, Any]:
    """Run a bounded, checkpointed rule-versus-Gemini semantic detector smoke test."""
    output_dir = Path(output_dir).resolve()
    frontend_data_dir = Path(frontend_data_dir).resolve()
    table_dir = output_dir / "tables"
    cache_dir = output_dir / "raw_cache"
    prompt_dir = output_dir / "dry_run_prompts"
    for directory in (output_dir, table_dir, cache_dir, prompt_dir, frontend_data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    urls = pd.read_csv(position_urls_path, low_memory=False)
    documents = pd.read_csv(document_features_path, low_memory=False).drop_duplicates("normalized_url")
    smoke = select_smoke_urls(urls, max_urls=max_urls)
    smoke = smoke.merge(
        documents[["normalized_url", "snapshot_path", "html_available"]],
        on="normalized_url",
        how="left",
        validate="one_to_one",
    )
    full_run = max_urls is None

    progress_paths = (
        output_dir / "full_run_progress.json",
        frontend_data_dir / "gemini_position_full_run_progress.json",
    )

    def write_progress(payload: dict[str, Any]) -> None:
        for progress_path in progress_paths:
            temp_path = progress_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temp_path.replace(progress_path)

    write_progress(
        {
            "status": "running" if execute_live else "dry_run",
            "run_scope": "full" if full_run else "sample",
            "model": model,
            "total_urls": len(smoke),
            "processed_urls": 0,
            "successful_urls": 0,
            "failed_urls": 0,
            "chunks_total_processed": 0,
            "chunks_cached": 0,
            "chunks_live": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )

    api_key = config.get_secret("GEMINI_API_KEY")
    if execute_live and not api_key and request_fn is None:
        raise RuntimeError(
            "Live Gemini execution requires GEMINI_API_KEY in the project .env. "
            "The key value is never logged or written to outputs."
        )
    client = None
    if execute_live and request_fn is None:
        from src.gemini_client import build_client

        client = build_client(str(api_key))

    page_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for index, row in enumerate(smoke.itertuples(index=False), start=1):
        snapshot: dict[str, Any] = {}
        snapshot_path = Path(str(getattr(row, "snapshot_path", "") or ""))
        if snapshot_path.exists():
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                snapshot = {}
        blocks, block_meta = extract_semantic_blocks(str(snapshot.get("html") or ""))
        for block in blocks:
            block_rows.append(
                {
                    "normalized_url": row.normalized_url,
                    "source_url": row.source_url,
                    **asdict(block),
                    "rule_candidates": json.dumps(block.rule_candidates),
                }
            )
        chunks = chunk_blocks(blocks)
        for chunk_index, chunk in enumerate(chunks):
            prompt = build_chunk_prompt(chunk, page_title=str(getattr(row, "page_title", "") or ""))
            (prompt_dir / f"{index:03d}_{chunk_index:03d}.txt").write_text(prompt, encoding="utf-8")

        detections, api_meta = classify_page(
            client=client,
            normalized_url=row.normalized_url,
            page_title=str(getattr(row, "page_title", "") or ""),
            blocks=blocks,
            cache_dir=cache_dir,
            model=model,
            execute_live=execute_live,
            force=force,
            request_fn=request_fn,
        )
        aggregate, page_evidence = aggregate_page_detections(blocks, detections)
        if api_meta["gemini_status"] != "success":
            for feature in SEMANTIC_FEATURES:
                aggregate[f"has_{feature}_gemini_v1"] = np.nan
                aggregate[f"{feature}_count_gemini_v1"] = np.nan
                aggregate[f"first_{feature}_position_ratio_gemini_v1"] = np.nan
                aggregate[f"first_{feature}_block_id_gemini_v1"] = ""
        base = {
            "normalized_url": row.normalized_url,
            "source_url": row.source_url,
            "source_root_domain": row.source_root_domain,
            "page_title": getattr(row, "page_title", ""),
            "gemini_position_version": GEMINI_POSITION_VERSION,
            "gemini_model": model,
            **{column: getattr(row, column) for column in FEATURE_TO_RULE_COLUMN.values()},
            **block_meta,
            **api_meta,
            **aggregate,
        }
        page_rows.append(base)
        for evidence_item in page_evidence:
            evidence_rows.append(
                {
                    "normalized_url": row.normalized_url,
                    "source_url": row.source_url,
                    "domain": row.source_root_domain,
                    "page_title": getattr(row, "page_title", ""),
                    **evidence_item,
                }
            )
        print(
            f"[{index}/{len(smoke)}] {api_meta['gemini_status']} "
            f"blocks={len(blocks)} chunks={api_meta['chunks_total']} {row.normalized_url}",
            flush=True,
        )
        write_progress(
            {
                "status": "running" if execute_live else "dry_run",
                "run_scope": "full" if full_run else "sample",
                "model": model,
                "total_urls": len(smoke),
                "processed_urls": index,
                "successful_urls": sum(
                    item.get("gemini_status") == "success" for item in page_rows
                ),
                "failed_urls": sum(
                    item.get("gemini_status") in {"partial_failure", "unmeasured_no_blocks"}
                    for item in page_rows
                ),
                "chunks_total_processed": sum(int(item.get("chunks_total") or 0) for item in page_rows),
                "chunks_cached": sum(int(item.get("chunks_cached") or 0) for item in page_rows),
                "chunks_live": sum(int(item.get("chunks_live") or 0) for item in page_rows),
                "input_tokens": sum(int(item.get("input_tokens") or 0) for item in page_rows),
                "output_tokens": sum(int(item.get("output_tokens") or 0) for item in page_rows),
            }
        )

    pages = pd.DataFrame(page_rows)
    evidence_columns = [
        "normalized_url", "source_url", "domain", "page_title", "feature", "block_id",
        "tag", "start_token", "position_ratio", "confidence", "rationale", "evidence_text",
    ]
    block_columns = [
        "normalized_url", "source_url", "block_id", "tag", "start_token",
        "position_ratio", "text", "rule_candidates",
    ]
    review_columns = [
        "review_priority", "feature", "normalized_url", "source_url", "domain", "page_title",
        "rule_value", "gemini_value", "agreement", "gemini_evidence_block_ids",
        "gemini_confidence", "gemini_rationale", "gemini_evidence_text",
        "human_review_result", "human_review_note",
    ]
    evidence = pd.DataFrame(evidence_rows, columns=evidence_columns)
    blocks_frame = pd.DataFrame(block_rows, columns=block_columns)
    agreement = _agreement_summary(pages)
    review = _review_sample(pages, evidence)
    if review.empty:
        review = pd.DataFrame(columns=review_columns)

    outputs = {
        "gemini_position_smoke_pages": pages,
        "gemini_position_feature_agreement": agreement,
        "gemini_position_detection_evidence": evidence,
        "gemini_position_manual_review": review,
        "gemini_position_input_blocks": blocks_frame,
    }
    for name, frame in outputs.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
        frame.to_csv(frontend_data_dir / f"{name}.csv", index=False)

    successful = int(pages["gemini_status"].eq("success").sum()) if not pages.empty else 0
    manifest = {
        "status": (
            "live_full_complete"
            if execute_live and full_run
            else "live_smoke_complete"
            if execute_live
            else "dry_run_full_ready"
            if full_run
            else "dry_run_ready"
        ),
        "run_scope": "full" if full_run else "sample",
        "version": GEMINI_POSITION_VERSION,
        "model": model,
        "execute_live": execute_live,
        "sample_urls": len(pages),
        "successful_urls": successful,
        "failed_or_dry_urls": len(pages) - successful,
        "input_tokens": int(pd.to_numeric(pages.get("input_tokens"), errors="coerce").fillna(0).sum()),
        "output_tokens": int(pd.to_numeric(pages.get("output_tokens"), errors="coerce").fillna(0).sum()),
        "thinking_tokens": int(pd.to_numeric(pages.get("thinking_tokens"), errors="coerce").fillna(0).sum()),
        "leakage_guard": "prompt excludes citation outcomes, answers, ranks, positions, and domain citation statistics",
        "promotion_status": "qa_only_not_model_ready",
        "output_dir": str(output_dir),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (frontend_data_dir / "gemini_position_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_progress(
        {
            "status": "complete",
            "run_scope": manifest["run_scope"],
            "model": model,
            "total_urls": len(pages),
            "processed_urls": len(pages),
            "successful_urls": successful,
            "failed_urls": len(pages) - successful,
            "chunks_total_processed": int(pd.to_numeric(pages["chunks_total"], errors="coerce").fillna(0).sum()),
            "chunks_cached": int(pd.to_numeric(pages["chunks_cached"], errors="coerce").fillna(0).sum()),
            "chunks_live": int(pd.to_numeric(pages["chunks_live"], errors="coerce").fillna(0).sum()),
            "input_tokens": manifest["input_tokens"],
            "output_tokens": manifest["output_tokens"],
        }
    )
    return manifest


def default_document_features_path() -> Path:
    return (
        topic_output_dir()
        / "content_econometrics_ai_package/tables/12_document_structure_features/url_document_structure_features.csv"
    )
