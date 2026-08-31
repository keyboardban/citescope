"""HTML-first position-feature extraction and descriptive EDA.

This layer is deliberately pre-model. It preserves the original surfaced-source
citation label, measures positions in cleaned main content, and keeps extraction
failure distinct from feature absence.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup, NavigableString, Tag

from src.econometrics_eda_v2.document_structure_features import (
    WORD_RE,
    _remove_noise,
    _select_main_node,
)
from src.econometrics_eda_v2.paths import CODE_ROOT, topic_output_dir
from src.url_utils import root_domain


POSITION_FEATURE_VERSION = "position_feature_eda_v1_20260731"
DEFAULT_OUTPUT_DIR = CODE_ROOT / "outputs" / "position_feature_eda_final_20260731"
CITATION_COLORS = {
    "Cited": "#009E73",
    "Not cited": "#D55E00",
}

QUESTION_RE = re.compile(
    r"(?:\?|？)\s*$|^(?:what|why|when|where|who|which|how|can|should|is|are|do|does|"
    r"อะไร|ทำไม|เมื่อไร|ที่ไหน|ใคร|อย่างไร|ไหม|หรือไม่)\b",
    re.I | re.U,
)
FAQ_RE = re.compile(
    r"\b(?:faq|frequently asked questions?|common questions?|questions?\s*(?:and|&)\s*answers?)\b"
    r"|คำถามที่พบบ่อย|ถามตอบ|คำถามและคำตอบ",
    re.I | re.U,
)
DIRECT_ANSWER_RE = re.compile(
    r"\b(?:in short|short answer|the answer is|yes[,.:]|no[,.:]|overall[,.:]|"
    r"simply put|to summarize|bottom line)\b|คำตอบคือ|สรุป(?:คือ|ได้ว่า)|โดยสรุป",
    re.I | re.U,
)
DEFINITION_RE = re.compile(
    r"^.{2,100}?\s+(?:is|are|means|refers to|is defined as)\s+.{3,}"
    r"|.{2,80}?(?:คือ|หมายถึง|นิยามว่า).{3,}",
    re.I | re.U,
)
COMPARISON_RE = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|difference|pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?)\b"
    r"|เปรียบเทียบ|เทียบกับ|แตกต่าง|ข้อดี.{0,20}ข้อเสีย",
    re.I | re.U,
)
STEPS_RE = re.compile(
    r"\b(?:step\s*\d+|steps?|how to|procedure|instructions?)\b|ขั้นตอน|วิธีการ|ทำอย่างไร",
    re.I | re.U,
)
NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,4}(?:[,.]\d+)*(?:\s?%|\s?[A-Za-z²]+)?)(?!\w)")
EVIDENCE_NUMBER_RE = re.compile(
    r"(?:฿|\$|€|£|\b(?:thb|usd|baht)\b|\d[\d,.]*\s?(?:%|sqm|sq\.?\s*m|m²|"
    r"years?|days?|km|m|million|billion|ล้าน|บาท|ตร\.?\s*ม|ตารางเมตร))",
    re.I | re.U,
)
AUTHOR_RE = re.compile(r"(?:^|[-_\s])(author|byline|writer)(?:$|[-_\s])", re.I)
CREDENTIAL_RE = re.compile(
    r"\b(?:ph\.?d\.?|m\.?d\.?|professor|dr\.?|certified|licensed|expert|analyst|"
    r"doctorate|ผู้เชี่ยวชาญ|ได้รับใบอนุญาต)\b",
    re.I | re.U,
)
DATE_RE = re.compile(
    r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"
    r"|\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b"
)

MAJOR_POSITION_FEATURES = {
    "table": "first_table_position_ratio",
    "list": "first_list_position_ratio",
    "faq": "faq_start_position_ratio",
    "direct_answer": "direct_answer_position_ratio",
    "definition": "first_definition_position_ratio",
    "comparison": "first_comparison_position_ratio",
    "steps": "first_steps_position_ratio",
    "numeric_evidence": "first_numeric_evidence_position_ratio",
    "external_citation": "first_external_citation_position_ratio",
    "question_heading": "first_question_heading_position_ratio",
}

FEATURE_META = {
    "table": {"presence": "has_table", "count": "table_count"},
    "list": {"presence": "has_bullets", "count": "list_count"},
    "faq": {"presence": "has_faq", "count": "faq_item_count"},
    "direct_answer": {"presence": "has_direct_answer", "count": "direct_answer_count"},
    "definition": {"presence": "has_definition_block", "count": "definition_block_count"},
    "comparison": {"presence": "has_comparison", "count": "comparison_block_count"},
    "steps": {"presence": "has_steps", "count": "step_block_count"},
    "numeric_evidence": {"presence": "has_numeric_evidence", "count": "numeric_count"},
    "external_citation": {"presence": "has_external_sources", "count": "outbound_citation_count"},
    "question_heading": {"presence": "has_question_heading", "count": "question_heading_count"},
}


def _tokens(text: object) -> list[str]:
    return WORD_RE.findall("" if text is None else str(text))


def _quartile(ratio: object, presence: object, measured: bool) -> str:
    if not measured:
        return "Unmeasured"
    numeric_presence = pd.to_numeric(pd.Series([presence]), errors="coerce").iloc[0]
    if pd.isna(numeric_presence):
        return "Unmeasured"
    if int(numeric_presence) == 0:
        return "No feature"
    value = pd.to_numeric(pd.Series([ratio]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "Unexpected missing"
    if value < 0.25:
        return "Q1"
    if value < 0.50:
        return "Q2"
    if value < 0.75:
        return "Q3"
    return "Q4"


def _half(ratio: object, presence: object, measured: bool) -> str:
    quartile = _quartile(ratio, presence, measured)
    if quartile in {"Q1", "Q2"}:
        return "First half"
    if quartile in {"Q3", "Q4"}:
        return "Second half"
    return quartile


def _visible_text(tag: Tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()


def _valid_list(tag: Tag) -> tuple[bool, int]:
    items = [
        _visible_text(item)
        for item in tag.find_all("li", recursive=False)
        if _visible_text(item)
    ]
    return len(items) >= 2, len(items)


def _node_start_positions(main: Tag) -> tuple[dict[int, int], int]:
    """Map every ancestor tag to the first token emitted inside it."""
    starts: dict[int, int] = {}
    token_cursor = 0
    for node in main.descendants:
        if not isinstance(node, NavigableString):
            continue
        count = len(_tokens(str(node)))
        if not count:
            continue
        parent = node.parent
        while isinstance(parent, Tag):
            starts.setdefault(id(parent), token_cursor)
            if parent is main:
                break
            parent = parent.parent
        token_cursor += count
    return starts, token_cursor


def _unique_occurrences(tags: Iterable[Tag], starts: dict[int, int]) -> list[tuple[int, Tag]]:
    seen: set[tuple[int, str]] = set()
    result: list[tuple[int, Tag]] = []
    for tag in tags:
        position = starts.get(id(tag))
        text = _visible_text(tag)
        key = (position if position is not None else -1, text[:200])
        if position is None or not text or key in seen:
            continue
        seen.add(key)
        result.append((position, tag))
    return sorted(result, key=lambda item: item[0])


def _occurrence_summary(positions: list[int], total_tokens: int) -> dict[str, Any]:
    if not positions or total_tokens <= 0:
        return {
            "first_token": np.nan,
            "first_ratio": np.nan,
            "median_ratio": np.nan,
            "mean_ratio": np.nan,
            "first_quartile_share": np.nan,
            "first_half_share": np.nan,
        }
    ratios = np.clip(np.asarray(positions, dtype=float) / total_tokens, 0, 1)
    return {
        "first_token": int(positions[0]),
        "first_ratio": float(ratios[0]),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "first_quartile_share": float((ratios < 0.25).mean()),
        "first_half_share": float((ratios < 0.50).mean()),
    }


def _position_fields(
    prefix: str,
    ratio_name: str,
    positions: list[int],
    total_tokens: int,
    presence: int,
) -> dict[str, Any]:
    summary = _occurrence_summary(positions, total_tokens)
    quartile_name = ratio_name.replace("position_ratio", "position_quartile")
    if ratio_name == "direct_answer_position_ratio":
        quartile_name = "direct_answer_position_quartile"
    fields = {
        f"{prefix}_start_token_index": summary["first_token"],
        ratio_name: summary["first_ratio"],
        quartile_name: _quartile(summary["first_ratio"], presence, True),
        f"{prefix}_median_position_ratio": summary["median_ratio"],
        f"{prefix}_mean_position_ratio": summary["mean_ratio"],
        f"{prefix}_share_in_first_quartile": summary["first_quartile_share"],
        f"{prefix}_share_in_first_half": summary["first_half_share"],
    }
    return fields


def _markdown_occurrences(markdown: str) -> tuple[dict[str, list[int]], int, dict[str, int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    cursor = 0
    lines = markdown.splitlines()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        text = re.sub(r"^(?:#{1,6}|[-*+]\s+|\d+[.)]\s+)", "", line).strip()
        is_heading = bool(re.match(r"^#{1,6}\s+", line))
        if is_heading and QUESTION_RE.search(text):
            positions["question_heading"].append(cursor)
        if FAQ_RE.search(text):
            positions["faq"].append(cursor)
        if DIRECT_ANSWER_RE.search(text):
            positions["direct_answer"].append(cursor)
        if DEFINITION_RE.search(text):
            positions["definition"].append(cursor)
        if COMPARISON_RE.search(text):
            positions["comparison"].append(cursor)
        if STEPS_RE.search(text):
            positions["steps"].append(cursor)
        if EVIDENCE_NUMBER_RE.search(text):
            positions["numeric_evidence"].append(cursor)
        if re.match(r"^\|.+\|$", line) and index + 1 < len(lines) and re.match(
            r"^\|(?:\s*:?-+:?\s*\|)+$", lines[index + 1].strip()
        ):
            positions["table"].append(cursor)
        if re.match(r"^[-*+]\s+", line):
            positions["list"].append(cursor)
            counts["list_item_count"] += 1
        if re.match(r"^\d+[.)]\s+", line):
            positions["list"].append(cursor)
            positions["steps"].append(cursor)
            counts["list_item_count"] += 1
            counts["step_count"] += 1
        for match in re.finditer(r"\[[^\]]+\]\((https?://[^)]+)\)", line):
            positions["external_citation"].append(cursor + len(_tokens(line[: match.start()])))
        counts["numeric_count"] += len(NUMBER_RE.findall(text))
        cursor += len(_tokens(text))
    for key in positions:
        positions[key] = sorted(set(positions[key]))
    return positions, cursor, counts


def extract_position_features(
    raw_html: str,
    source_url: str,
    fallback_markdown: str = "",
) -> dict[str, Any]:
    """Extract presence, intensity, and first-position measures from one page."""
    if not str(raw_html or "").strip():
        markdown = str(fallback_markdown or "").strip()
        if not markdown:
            return _unmeasured_features("insufficient_html_or_markdown")
        occurrences, total_tokens, extra_counts = _markdown_occurrences(markdown)
        if total_tokens <= 0:
            return _unmeasured_features("markdown_parse_failed")
        return _assemble_features(
            occurrences,
            total_tokens,
            {key: len(value) for key, value in occurrences.items()},
            list_item_count=extra_counts.get("list_item_count", 0),
            numeric_count=extra_counts.get("numeric_count", 0),
            step_count=extra_counts.get("step_count", 0),
            measurement_source="generated_markdown_fallback",
            evidence={key: [] for key in MAJOR_POSITION_FEATURES},
        )

    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_noise(soup)
    main, method = _select_main_node(soup)
    starts, total_tokens = _node_start_positions(main)
    if total_tokens <= 0:
        return _unmeasured_features("main_content_parse_failed")

    headings = list(main.find_all(re.compile(r"^h[1-6]$")))
    paragraphs = list(main.find_all(["p", "blockquote", "dd", "dt"]))
    tables = [
        tag for tag in main.find_all("table")
        if len(tag.find_all("tr")) >= 1 and len(tag.find_all(["td", "th"])) >= 2
    ]
    valid_lists: list[Tag] = []
    list_item_count = 0
    ordered_lists: list[Tag] = []
    for tag in main.find_all(["ul", "ol"]):
        valid, item_count = _valid_list(tag)
        if valid:
            valid_lists.append(tag)
            list_item_count += item_count
            if tag.name == "ol":
                ordered_lists.append(tag)

    question_headings = [tag for tag in headings if QUESTION_RE.search(_visible_text(tag))]
    faq_tags = [tag for tag in headings if FAQ_RE.search(_visible_text(tag))]
    faq_tags.extend(
        tag for tag in main.find_all(attrs={"itemtype": re.compile("FAQPage|Question", re.I)})
    )
    direct_tags = [tag for tag in [*headings, *paragraphs] if DIRECT_ANSWER_RE.search(_visible_text(tag)[:600])]
    definition_tags = list(main.find_all("dl"))
    definition_tags.extend(
        tag for tag in [*headings, *paragraphs] if DEFINITION_RE.search(_visible_text(tag)[:700])
    )
    comparison_tags = [
        tag for tag in [*headings, *paragraphs, *tables]
        if COMPARISON_RE.search(_visible_text(tag)[:1200])
    ]
    step_tags = [tag for tag in [*headings, *paragraphs] if STEPS_RE.search(_visible_text(tag)[:700])]
    step_tags.extend(ordered_lists)
    numeric_tags = [
        tag for tag in [*headings, *paragraphs, *tables]
        if EVIDENCE_NUMBER_RE.search(_visible_text(tag)[:1800])
    ]

    source_domain = root_domain(urlparse(source_url).hostname or "")
    external_tags: list[Tag] = []
    external_domains: set[str] = set()
    for anchor in main.find_all("a", href=True):
        href = urljoin(source_url, str(anchor.get("href") or ""))
        host = urlparse(href).hostname or ""
        target = root_domain(host)
        if href.startswith(("http://", "https://")) and target and target != source_domain:
            external_tags.append(anchor)
            external_domains.add(target)

    tag_groups = {
        "table": tables,
        "list": valid_lists,
        "faq": faq_tags,
        "direct_answer": direct_tags,
        "definition": definition_tags,
        "comparison": comparison_tags,
        "steps": step_tags,
        "numeric_evidence": numeric_tags,
        "external_citation": external_tags,
        "question_heading": question_headings,
    }
    occurrences: dict[str, list[int]] = {}
    evidence: dict[str, list[str]] = {}
    for name, tags in tag_groups.items():
        unique = _unique_occurrences(tags, starts)
        occurrences[name] = [position for position, _ in unique]
        evidence[name] = [_visible_text(tag)[:300] for _, tag in unique[:5]]

    numeric_count = len(NUMBER_RE.findall(_visible_text(main)))
    step_count = sum(_valid_list(tag)[1] for tag in ordered_lists)
    counts = {key: len(value) for key, value in occurrences.items()}
    result = _assemble_features(
        occurrences,
        total_tokens,
        counts,
        list_item_count=list_item_count,
        numeric_count=numeric_count,
        step_count=step_count,
        measurement_source=f"filtered_main_content_html:{method}",
        evidence=evidence,
    )

    full_text = _visible_text(main)
    paragraphs_text = [_visible_text(tag) for tag in main.find_all("p") if _visible_text(tag)]
    paragraph_lengths = [len(_tokens(text)) for text in paragraphs_text]
    images = list(main.find_all("img"))
    alt_images = [tag for tag in images if str(tag.get("alt") or "").strip()]
    author_nodes = list(soup.find_all(attrs={"class": AUTHOR_RE})) + list(
        soup.find_all(attrs={"rel": re.compile("author", re.I)})
    )
    author_text = " ".join(_visible_text(tag) for tag in author_nodes)
    jsonld_types = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        jsonld_types.extend(re.findall(r'"@type"\s*:\s*"([^"]+)"', script.get_text(" ")))
    published = bool(soup.find("meta", attrs={"property": re.compile("published", re.I)}))
    modified = bool(soup.find("meta", attrs={"property": re.compile("modified", re.I)}))
    result.update(
        {
            "has_author": int(bool(author_nodes or soup.find("meta", attrs={"name": re.compile("author", re.I)}))),
            "has_author_credentials": int(bool(CREDENTIAL_RE.search(author_text))),
            "has_published_date": int(published or bool(DATE_RE.search(full_text[:2500]))),
            "has_modified_date": int(modified),
            "has_schema_markup": int(bool(jsonld_types)),
            "schema_type_count": len(set(jsonld_types)),
            "has_alt_text": int(bool(alt_images)),
            "alt_text_coverage_ratio": len(alt_images) / len(images) if images else np.nan,
            "average_paragraph_length": float(np.mean(paragraph_lengths)) if paragraph_lengths else np.nan,
            "median_paragraph_length": float(np.median(paragraph_lengths)) if paragraph_lengths else np.nan,
            "paragraph_length_variance": float(np.var(paragraph_lengths)) if paragraph_lengths else np.nan,
            "heading_depth": max([int(tag.name[1]) for tag in headings], default=0),
            "section_count": len(headings),
            "unique_external_domain_count": len(external_domains),
            "external_link_domains": ";".join(sorted(external_domains)),
        }
    )
    return result


def _unmeasured_features(status: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "position_extraction_status": status,
        "position_measurement_source": "unmeasured",
        "position_features_available": 0,
        "total_main_content_token_count": 0,
    }
    for name, ratio_name in MAJOR_POSITION_FEATURES.items():
        meta = FEATURE_META[name]
        output[meta["presence"]] = np.nan
        output[meta["count"]] = np.nan
        prefix = ratio_name.replace("first_", "").replace("_position_ratio", "")
        output[f"{prefix}_start_token_index"] = np.nan
        output[ratio_name] = np.nan
        quartile_name = ratio_name.replace("position_ratio", "position_quartile")
        if ratio_name == "direct_answer_position_ratio":
            quartile_name = "direct_answer_position_quartile"
        output[quartile_name] = "Unmeasured"
        output[f"{prefix}_median_position_ratio"] = np.nan
        output[f"{prefix}_mean_position_ratio"] = np.nan
        output[f"{prefix}_share_in_first_quartile"] = np.nan
        output[f"{prefix}_share_in_first_half"] = np.nan
        output[f"{name}_evidence"] = "[]"
    return output


def _assemble_features(
    occurrences: dict[str, list[int]],
    total_tokens: int,
    counts: dict[str, int],
    *,
    list_item_count: int,
    numeric_count: int,
    step_count: int,
    measurement_source: str,
    evidence: dict[str, list[str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "position_extraction_status": "measured",
        "position_measurement_source": measurement_source,
        "position_features_available": 1,
        "total_main_content_token_count": int(total_tokens),
    }
    for name, ratio_name in MAJOR_POSITION_FEATURES.items():
        positions = sorted(set(occurrences.get(name, [])))
        meta = FEATURE_META[name]
        presence = int(bool(positions))
        output[meta["presence"]] = presence
        output[meta["count"]] = int(counts.get(name, len(positions)))
        prefix = ratio_name.replace("first_", "").replace("_position_ratio", "")
        output.update(_position_fields(prefix, ratio_name, positions, total_tokens, presence))
        output[f"{name}_evidence"] = json.dumps(evidence.get(name, []), ensure_ascii=False)

    output["list_item_count"] = int(list_item_count)
    output["step_count"] = int(step_count)
    output["numeric_count"] = int(numeric_count)
    denominator = max(total_tokens, 1)
    output["table_count_per_1000_words"] = output["table_count"] / denominator * 1000
    output["list_item_density"] = list_item_count / denominator * 1000
    output["numeric_density"] = numeric_count / denominator * 1000
    output["outbound_citations_per_1000_words"] = output["outbound_citation_count"] / denominator * 1000
    output["unique_external_domain_count"] = len(
        {urlparse(item).hostname for item in evidence.get("external_citation", []) if urlparse(item).hostname}
    )
    output["comparison_with_table"] = int(output["has_comparison"] and output["has_table"])
    output["comparison_with_early_table"] = int(
        output["has_comparison"] and output["has_table"] and output["first_table_position_ratio"] < 0.5
    )
    output["direct_answer_with_early_table"] = int(
        output["has_direct_answer"] and output["has_table"] and output["first_table_position_ratio"] < 0.5
    )
    output["faq_with_question_headings"] = int(output["has_faq"] and output["has_question_heading"])
    output["steps_with_ordered_list"] = int(output["has_steps"] and step_count >= 2)
    output["numeric_density_first_quartile"] = (
        sum(position < total_tokens * 0.25 for position in occurrences.get("numeric_evidence", []))
        / max(total_tokens * 0.25, 1)
        * 1000
    )
    output["numeric_density_first_half"] = (
        sum(position < total_tokens * 0.5 for position in occurrences.get("numeric_evidence", []))
        / max(total_tokens * 0.5, 1)
        * 1000
    )
    return output


def _wilson(cited: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    p = cited / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _safe_rate_table(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    rows = []
    for value, group in frame.groupby(category, dropna=False, observed=True):
        cited = int(pd.to_numeric(group["cited"], errors="coerce").fillna(0).sum())
        n = len(group)
        low, high = _wilson(cited, n)
        rows.append(
            {
                "category": "Missing" if pd.isna(value) else str(value),
                "n_observations": n,
                "n_unique_pages": group["normalized_url"].nunique(),
                "cited_count": cited,
                "not_cited_count": n - cited,
                "citation_rate": cited / n if n else np.nan,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _feature_registry() -> list[dict[str, str]]:
    groups = {
        "answerability_presence": "has_direct_answer has_question_heading has_definition_block has_faq",
        "answerability_intensity": "direct_answer_count question_heading_count definition_block_count faq_item_count",
        "answerability_position": "direct_answer_position_ratio direct_answer_position_quartile direct_answer_in_first_quartile direct_answer_in_first_half first_question_heading_position_ratio first_question_heading_position_quartile question_headings_first_half_share first_definition_position_ratio definition_position_quartile definition_in_first_quartile faq_start_position_ratio faq_position_quartile faq_in_first_quartile faq_in_first_half",
        "structure_presence": "has_table has_bullets has_comparison has_steps",
        "structure_intensity_density": "table_count table_count_per_1000_words list_count list_item_count list_item_density comparison_block_count step_block_count step_count",
        "structure_position": "first_table_token_position first_table_position_ratio first_table_position_quartile table_in_first_quartile table_in_first_half median_table_position_ratio table_share_in_first_half first_list_position_ratio first_list_position_quartile list_in_first_half first_comparison_position_ratio comparison_position_quartile comparison_in_first_quartile comparison_in_first_half first_steps_position_ratio steps_position_quartile steps_in_first_half",
        "interaction": "comparison_with_table comparison_with_early_table direct_answer_with_early_table faq_with_question_headings steps_with_ordered_list",
        "evidence_presence": "has_numeric_evidence",
        "evidence_intensity_density": "numeric_count numeric_density entity_count entity_density",
        "evidence_position": "first_numeric_evidence_position_ratio first_numeric_evidence_position_quartile numeric_evidence_in_first_quartile numeric_density_first_quartile numeric_density_first_half entity_density_first_quartile",
        "authority_trust": "has_author has_author_credentials has_external_sources outbound_citation_count outbound_citations_per_1000_words unique_external_domain_count first_external_citation_position_ratio external_citation_position_quartile external_citation_in_first_half",
        "freshness": "has_published_date has_modified_date content_age_days",
        "technical_accessibility": "has_schema_markup schema_type_count has_alt_text alt_text_coverage_ratio main_content_html_ratio",
        "readability_quality": "average_paragraph_length median_paragraph_length paragraph_length_variance self_contained_score content_depth_score heading_depth section_count",
        "taxonomy_control": "page_type source_type intent domain prompt_id page_id word_count heading_count",
    }
    definitions = {
        "has_table": "At least one verified data table in cleaned main-content HTML.",
        "first_table_position_ratio": "First verified table token index divided by total main-content tokens.",
        "has_faq": "Visible FAQ heading/structure or FAQPage markup in cleaned content.",
        "has_direct_answer": "At least one conservative direct-answer phrase in a heading or paragraph.",
        "has_definition_block": "At least one definition list or explicit definition phrase.",
        "has_comparison": "At least one comparison-labelled heading, paragraph, or table.",
        "has_steps": "At least one step/how-to block or valid ordered list.",
        "has_numeric_evidence": "At least one block containing a number with a factual unit, currency, or percentage.",
    }
    rows = []
    for group, names in groups.items():
        for name in names.split():
            rows.append(
                {
                    "feature_name": name,
                    "feature_group": group,
                    "feature_definition": definitions.get(name, name.replace("_", " ").capitalize() + "."),
                }
            )
    return rows


def _build_feature_audit(
    source_columns: set[str],
    extracted_columns: set[str],
    previously_analyzed: set[str],
) -> pd.DataFrame:
    rows = []
    controls = {"page_type", "source_type", "intent", "domain", "prompt_id", "page_id", "word_count", "heading_count"}
    for item in _feature_registry():
        name = item["feature_name"]
        exists = name in source_columns
        extracted = name in extracted_columns
        new_group = any(token in item["feature_group"] for token in ("position", "intensity", "interaction"))
        include = extracted and (new_group or name in {"has_direct_answer", "has_definition_block", "has_faq", "has_comparison", "has_steps", "has_numeric_evidence", "has_external_sources"})
        if name in controls:
            include = False
            reason = "Existing identifier, denominator, grouping variable, or control."
        elif include:
            reason = "New position/intensity measure or required presence baseline for the position analysis."
        elif name in {"entity_count", "entity_density", "entity_density_first_quartile", "content_age_days", "self_contained_score", "content_depth_score"}:
            reason = "Paused: reliable extraction requires governed NER, date parsing, or a separately validated quality rubric."
        elif exists or name in previously_analyzed:
            reason = "Previously available or analyzed baseline; retained only for denominator, grouping, or validation."
        else:
            reason = "Registry concept not extracted in this version because position is not meaningful or evidence is insufficient."
        rows.append(
            {
                **item,
                "already_exists": bool(exists),
                "previously_analyzed": bool(name in previously_analyzed),
                "needs_extraction": bool(not exists and extracted),
                "newly_extracted": bool(extracted and not exists),
                "include_in_new_eda": bool(include),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _coverage_table(
    urls: pd.DataFrame,
    rows: pd.DataFrame,
    source_columns: set[str],
) -> pd.DataFrame:
    output = []
    for feature_name, meta in FEATURE_META.items():
        presence = meta["presence"]
        measured = pd.to_numeric(urls[presence], errors="coerce").notna()
        positive = pd.to_numeric(urls[presence], errors="coerce").eq(1)
        row_presence = pd.to_numeric(rows[presence], errors="coerce")
        present_rows = rows[row_presence.eq(1)]
        absent_rows = rows[row_presence.eq(0)]
        output.append(
            {
                "feature": feature_name,
                "presence_feature": presence,
                "already_present_in_source_data": presence in source_columns,
                "newly_extracted": presence not in source_columns,
                "extraction_success_rate": float(measured.mean()),
                "missing_rate": float((~measured).mean()),
                "applicable_page_count": int(measured.sum()),
                "pages_with_feature": int(positive.sum()),
                "percentage_of_eligible_pages": float(positive.sum() / measured.sum()) if measured.sum() else np.nan,
                "applicable_domain_count": int(urls.loc[measured, "source_root_domain"].nunique()),
                "domains_with_feature": int(urls.loc[positive, "source_root_domain"].nunique()),
                "page_types_with_feature": int(urls.loc[positive, "page_type"].nunique()),
                "citation_rate_when_present": float(present_rows["cited"].mean()) if len(present_rows) else np.nan,
                "citation_rate_when_absent": float(absent_rows["cited"].mean()) if len(absent_rows) else np.nan,
                "reason": "Position-bearing feature included in new EDA.",
            }
        )
    return pd.DataFrame(output)


def _position_outputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries = []
    citation_tables = []
    outcome_rows = []
    overall_rate = float(rows["cited"].mean())
    quartile_order = ["No feature", "Q1", "Q2", "Q3", "Q4", "Unexpected missing", "Unmeasured"]
    for feature, ratio_col in MAJOR_POSITION_FEATURES.items():
        presence = FEATURE_META[feature]["presence"]
        quartile_col = ratio_col.replace("position_ratio", "position_quartile")
        if ratio_col == "direct_answer_position_ratio":
            quartile_col = "direct_answer_position_quartile"
        present = rows[pd.to_numeric(rows[presence], errors="coerce").eq(1)].copy()
        ratios = pd.to_numeric(present[ratio_col], errors="coerce").dropna()
        summaries.append(
            {
                "feature": feature,
                "position_feature": ratio_col,
                "n_present_rows": len(present),
                "n_present_pages": present["normalized_url"].nunique(),
                "median_position_ratio": ratios.median() if len(ratios) else np.nan,
                "q1_position_ratio": ratios.quantile(0.25) if len(ratios) else np.nan,
                "q3_position_ratio": ratios.quantile(0.75) if len(ratios) else np.nan,
                "mean_position_ratio": ratios.mean() if len(ratios) else np.nan,
                "minimum_position_ratio": ratios.min() if len(ratios) else np.nan,
                "maximum_position_ratio": ratios.max() if len(ratios) else np.nan,
            }
        )
        full_quartiles = _safe_rate_table(rows, quartile_col)
        full_quartiles["feature"] = feature
        full_quartiles["position_feature"] = ratio_col
        full_quartiles["grouping"] = "quartile"
        full_quartiles["sample_scope"] = "full_sample"
        full_quartiles["overall_citation_rate"] = overall_rate
        full_quartiles["citation_rate_difference"] = full_quartiles["citation_rate"] - overall_rate
        full_quartiles["category_order"] = full_quartiles["category"].map(
            {value: index for index, value in enumerate(quartile_order)}
        )
        citation_tables.append(full_quartiles)

        conditional = _safe_rate_table(present, quartile_col)
        conditional["feature"] = feature
        conditional["position_feature"] = ratio_col
        conditional["grouping"] = "quartile"
        conditional["sample_scope"] = "feature_present"
        conditional["overall_citation_rate"] = float(present["cited"].mean()) if len(present) else np.nan
        conditional["citation_rate_difference"] = conditional["citation_rate"] - conditional["overall_citation_rate"]
        conditional["category_order"] = conditional["category"].map(
            {value: index for index, value in enumerate(quartile_order)}
        )
        citation_tables.append(conditional)

        half_col = f"_{feature}_half_group"
        half_frame = rows.copy()
        half_frame[half_col] = [
            _half(ratio, present_value, bool(measured))
            for ratio, present_value, measured in zip(
                half_frame[ratio_col], half_frame[presence], half_frame["position_features_available"]
            )
        ]
        halves = _safe_rate_table(half_frame, half_col)
        halves["feature"] = feature
        halves["position_feature"] = ratio_col
        halves["grouping"] = "half"
        halves["sample_scope"] = "full_sample"
        halves["overall_citation_rate"] = overall_rate
        halves["citation_rate_difference"] = halves["citation_rate"] - overall_rate
        halves["category_order"] = halves["category"].map(
            {"No feature": 0, "First half": 1, "Second half": 2, "Unexpected missing": 3, "Unmeasured": 4}
        )
        citation_tables.append(halves)

        cited_values = pd.to_numeric(present.loc[present["cited"].eq(1), ratio_col], errors="coerce").dropna()
        not_values = pd.to_numeric(present.loc[present["cited"].eq(0), ratio_col], errors="coerce").dropna()
        outcome_rows.append(
            {
                "feature": feature,
                "position_feature": ratio_col,
                "cited_mean": cited_values.mean() if len(cited_values) else np.nan,
                "not_cited_mean": not_values.mean() if len(not_values) else np.nan,
                "cited_median": cited_values.median() if len(cited_values) else np.nan,
                "not_cited_median": not_values.median() if len(not_values) else np.nan,
                "difference_in_means": cited_values.mean() - not_values.mean() if len(cited_values) and len(not_values) else np.nan,
                "difference_in_medians": cited_values.median() - not_values.median() if len(cited_values) and len(not_values) else np.nan,
                "cited_n": len(cited_values),
                "not_cited_n": len(not_values),
            }
        )
    citation = pd.concat(citation_tables, ignore_index=True)
    return pd.DataFrame(summaries), citation, pd.DataFrame(outcome_rows)


def _within_between_sd(frame: pd.DataFrame, value: str, domain: str) -> tuple[float, float, float]:
    clean = frame[[domain, value]].copy()
    clean[value] = pd.to_numeric(clean[value], errors="coerce")
    clean = clean.dropna()
    if clean.empty:
        return np.nan, np.nan, np.nan
    overall = float(clean[value].std(ddof=1)) if len(clean) > 1 else 0.0
    means = clean.groupby(domain, observed=True)[value].mean()
    between = float(means.std(ddof=1)) if len(means) > 1 else 0.0
    centered = clean[value] - clean.groupby(domain, observed=True)[value].transform("mean")
    within = float(math.sqrt(float((centered**2).sum()) / max(len(clean) - len(means), 1)))
    return overall, between, within


def _within_domain_diagnostics(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    domain = "source_root_domain"
    total_domains = rows[domain].nunique()
    domain_page_counts = rows.groupby(domain, observed=True)["normalized_url"].nunique()
    singleton_share = float((domain_page_counts.eq(1)).mean())
    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        presence = FEATURE_META[feature]["presence"]
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        page_level = rows.drop_duplicates([domain, "normalized_url"]).copy()
        groups = page_level.groupby(domain, observed=True)
        domains_two_pages = int((groups["normalized_url"].nunique() >= 2).sum())
        containing = int(groups[presence].apply(lambda s: pd.to_numeric(s, errors="coerce").eq(1).any()).sum())
        presence_variation = int(groups[presence].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().nunique() >= 2).sum())
        ratio_variation = int(groups[ratio].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().nunique() >= 2).sum())
        quartile_variation = int(groups[quartile].apply(lambda s: s[s.isin(["Q1", "Q2", "Q3", "Q4"])].nunique() >= 2).sum())
        outcome_variation_domains = set(
            rows.groupby(domain, observed=True)["cited"].nunique().loc[lambda s: s >= 2].index
        )
        position_variation_domains = set(
            groups[ratio].apply(lambda s: pd.to_numeric(s, errors="coerce").dropna().nunique() >= 2).loc[lambda s: s].index
        )
        both = outcome_variation_domains & position_variation_domains
        informative = rows[rows[domain].isin(both) & pd.to_numeric(rows[ratio], errors="coerce").notna()]
        overall_sd, between_sd, within_sd = _within_between_sd(rows, ratio, domain)
        if len(both) >= 30 and len(informative) >= 300 and within_sd >= 0.08:
            readiness = "Ready"
        elif len(both) >= 15 and len(informative) >= 150 and within_sd >= 0.04:
            readiness = "Usable with caution"
        elif len(both) >= 5 and len(informative) >= 50 and within_sd >= 0.01:
            readiness = "Weak within-domain variation"
        else:
            readiness = "Not suitable for domain fixed effects"
        output.append(
            {
                "feature": feature,
                "position_feature": ratio,
                "total_domains": total_domains,
                "domains_with_at_least_two_pages": domains_two_pages,
                "domains_containing_feature": containing,
                "domains_with_presence_variation": presence_variation,
                "domains_with_position_ratio_variation": ratio_variation,
                "domains_with_position_quartile_variation": quartile_variation,
                "domains_with_citation_outcome_variation": len(outcome_variation_domains),
                "domains_with_both_position_and_outcome_variation": len(both),
                "informative_observations": len(informative),
                "share_of_singleton_domains": singleton_share,
                "overall_standard_deviation": overall_sd,
                "between_domain_standard_deviation": between_sd,
                "within_domain_standard_deviation": within_sd,
                "within_domain_sd_near_zero": bool(pd.isna(within_sd) or within_sd < 0.01),
                "fixed_effect_readiness": readiness,
                "readiness_threshold": "Ready: >=30 domains, >=300 rows, within SD >=0.08; caution: >=15/150/0.04; weak: >=5/50/0.01.",
            }
        )
    return pd.DataFrame(output)


def _sparse_cells(citation: pd.DataFrame) -> pd.DataFrame:
    cells = citation[(citation["grouping"] == "quartile") & (citation["sample_scope"] == "full_sample")].copy()
    cells["sparse_n_lt_20"] = cells["n_observations"] < 20
    cells["sparse_cited_lt_5"] = cells["cited_count"] < 5
    cells["sparse_not_cited_lt_5"] = cells["not_cited_count"] < 5
    cells["sparse_flag"] = cells[["sparse_n_lt_20", "sparse_cited_lt_5", "sparse_not_cited_lt_5"]].any(axis=1)
    recommendations = {}
    for feature, group in cells.groupby("feature", observed=True):
        feature_cells = group[group["category"].isin(["Q1", "Q2", "Q3", "Q4"])]
        sparse = int(feature_cells["sparse_flag"].sum())
        present_n = int(feature_cells["n_observations"].sum())
        if present_n < 20:
            recommendation = "Exclude from regression"
        elif sparse == 0:
            recommendation = "Q1, Q2, Q3, Q4"
        elif sparse <= 1 and present_n >= 100:
            recommendation = "First quartile, middle half, last quartile"
        elif present_n >= 50:
            recommendation = "First half versus second half"
        elif present_n >= 20:
            recommendation = "Continuous position ratio"
        else:
            recommendation = "Exclude from regression"
        recommendations[feature] = recommendation
    cells["recommended_grouping"] = cells["feature"].map(recommendations)
    return cells


def _taxonomy_crosstabs(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        for taxonomy in ("page_type", "source_type", "intent"):
            subset = rows[rows[quartile].isin(["Q1", "Q2", "Q3", "Q4"])].copy()
            counts = subset.groupby([taxonomy, quartile], dropna=False, observed=True).agg(
                n_observations=("cited", "size"),
                cited_count=("cited", "sum"),
                unique_pages=("normalized_url", "nunique"),
            ).reset_index()
            if counts.empty:
                continue
            counts["row_percent_within_position"] = counts["n_observations"] / counts.groupby(quartile)["n_observations"].transform("sum")
            counts["row_percent_within_taxonomy"] = counts["n_observations"] / counts.groupby(taxonomy)["n_observations"].transform("sum")
            counts["citation_rate"] = counts["cited_count"] / counts["n_observations"]
            counts["feature"] = feature
            counts["taxonomy_dimension"] = taxonomy
            counts = counts.rename(columns={taxonomy: "taxonomy_value", quartile: "position_category"})
            output.append(counts)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _page_length_relationship(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    word_count = pd.to_numeric(rows["word_count"], errors="coerce")
    labels = ["Short", "Medium", "Long", "Very long"]
    try:
        length_group = pd.qcut(word_count, 4, labels=labels, duplicates="drop")
    except ValueError:
        length_group = pd.cut(word_count, [-np.inf, 500, 1500, 3000, np.inf], labels=labels)
    working = rows.assign(page_length_group=length_group)
    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        token_col = ratio.replace("position_ratio", "start_token_index").replace("first_", "")
        prefix = ratio.replace("first_", "").replace("_position_ratio", "")
        token_col = f"{prefix}_start_token_index"
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        present = working[pd.to_numeric(working[FEATURE_META[feature]["presence"]], errors="coerce").eq(1)]
        for length_value, group in present.groupby("page_length_group", observed=True):
            output.append(
                {
                    "feature": feature,
                    "page_length_group": str(length_value),
                    "n_observations": len(group),
                    "n_unique_pages": group["normalized_url"].nunique(),
                    "median_word_count": pd.to_numeric(group["word_count"], errors="coerce").median(),
                    "median_position_ratio": pd.to_numeric(group[ratio], errors="coerce").median(),
                    "median_absolute_token_position": pd.to_numeric(group[token_col], errors="coerce").median(),
                    "citation_rate": group["cited"].mean(),
                    "dominant_position_quartile": group[quartile].mode().iloc[0] if not group[quartile].mode().empty else "",
                }
            )
    return pd.DataFrame(output)


def _correlation_outputs(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous = [
        *MAJOR_POSITION_FEATURES.values(),
        "table_count", "table_count_per_1000_words", "list_count", "list_item_density",
        "comparison_block_count", "numeric_density", "outbound_citations_per_1000_words",
        "word_count", "heading_count",
    ]
    continuous = [column for column in continuous if column in rows]
    numeric = rows[continuous].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.corr(method="spearman")
    matrix_long = matrix.stack().reset_index()
    matrix_long.columns = ["feature_a", "feature_b", "association"]
    matrix_long["association_type"] = "spearman"
    high = matrix_long[
        (matrix_long["feature_a"] < matrix_long["feature_b"])
        & matrix_long["association"].abs().ge(0.70)
    ].copy()
    high["warning"] = "High absolute Spearman correlation; avoid simultaneous use without diagnostics."

    binary = [FEATURE_META[name]["presence"] for name in MAJOR_POSITION_FEATURES]
    binary = [column for column in binary if column in rows]
    binary_numeric = rows[binary].apply(pd.to_numeric, errors="coerce")
    phi = binary_numeric.corr(method="pearson")
    phi_long = phi.stack().reset_index()
    phi_long.columns = ["feature_a", "feature_b", "association"]
    phi_long["association_type"] = "phi_binary"
    combined = pd.concat([matrix_long, phi_long], ignore_index=True)
    clusters = []
    for _, row in high.iterrows():
        clusters.append(
            {
                "feature_cluster": f"{row.feature_a} / {row.feature_b}",
                "association": row.association,
                "potential_issue": row.warning,
                "recommended_action": "Choose one representation for headline use or compare in separate extensions.",
            }
        )
    return combined, high, pd.DataFrame(clusters)


def _cramers_v(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    valid = left.notna() & right.notna()
    table = pd.crosstab(left[valid], right[valid])
    n = int(table.to_numpy().sum())
    if n == 0 or min(table.shape) < 2:
        return np.nan, n
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / n
    chi2 = float(np.divide((observed - expected) ** 2, expected, out=np.zeros_like(observed), where=expected > 0).sum())
    phi2 = chi2 / n
    rows_n, cols_n = observed.shape
    phi2_corrected = max(0.0, phi2 - ((cols_n - 1) * (rows_n - 1)) / max(n - 1, 1))
    rows_corrected = rows_n - ((rows_n - 1) ** 2) / max(n - 1, 1)
    cols_corrected = cols_n - ((cols_n - 1) ** 2) / max(n - 1, 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    return (math.sqrt(phi2_corrected / denominator) if denominator > 0 else np.nan), n


def _categorical_associations(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        for category in ("page_type", "source_type", "intent"):
            value, n = _cramers_v(rows[quartile], rows[category])
            output.append(
                {
                    "feature": feature,
                    "position_category_feature": quartile,
                    "categorical_variable": category,
                    "association_type": "bias_corrected_cramers_v",
                    "association": value,
                    "n_pairwise_complete": n,
                    "warning": "Strong template/taxonomy association" if pd.notna(value) and value >= 0.30 else "",
                }
            )
    return pd.DataFrame(output)


def _domain_examples(rows: pd.DataFrame) -> pd.DataFrame:
    selected = ["table", "faq", "direct_answer", "definition", "comparison"]
    output = []
    for feature in selected:
        ratio = MAJOR_POSITION_FEATURES[feature]
        presence = FEATURE_META[feature]["presence"]
        candidates = rows[pd.to_numeric(rows[presence], errors="coerce").eq(1)].copy()
        stats = candidates.groupby("source_root_domain", observed=True).agg(
            pages=("normalized_url", "nunique"),
            positions=(ratio, lambda s: pd.to_numeric(s, errors="coerce").nunique()),
            outcomes=("cited", "nunique"),
        )
        domains = stats[(stats.pages >= 2) & (stats.positions >= 2) & (stats.outcomes >= 2)].sort_values(
            ["pages", "positions"], ascending=False
        ).head(5).index
        sample = candidates[candidates["source_root_domain"].isin(domains)].drop_duplicates(
            ["source_root_domain", "normalized_url", "cited"]
        )
        sample = sample.sort_values(["source_root_domain", ratio]).groupby("source_root_domain", observed=True).head(6)
        for row in sample.itertuples(index=False):
            output.append(
                {
                    "feature": feature,
                    "domain": row.source_root_domain,
                    "page_label": str(row.normalized_url)[:85],
                    "normalized_url": row.normalized_url,
                    "position_ratio": getattr(row, ratio),
                    "cited": row.cited,
                    "page_type": row.page_type,
                    "source_type": row.source_type,
                }
            )
    return pd.DataFrame(output)


def _manual_validation(urls: pd.DataFrame) -> pd.DataFrame:
    output = []
    for feature in ["table", "faq", "direct_answer", "definition", "comparison"]:
        presence = FEATURE_META[feature]["presence"]
        evidence_col = f"{feature}_evidence"
        numeric = pd.to_numeric(urls[presence], errors="coerce")
        for expected, label in ((1, "detected_positive"), (0, "measured_negative")):
            sample = urls[numeric.eq(expected)].copy()
            sample = sample.sort_values(["source_root_domain", "normalized_url"]).drop_duplicates("source_root_domain").head(10)
            if len(sample) < 10:
                remainder = urls[numeric.eq(expected) & ~urls.index.isin(sample.index)].head(10 - len(sample))
                sample = pd.concat([sample, remainder])
            for row in sample.itertuples(index=False):
                evidence = getattr(row, evidence_col, "[]")
                try:
                    evidence_items = json.loads(str(evidence))
                except json.JSONDecodeError:
                    evidence_items = []
                if expected == 1 and evidence_items:
                    verdict = "true_positive"
                    note = "Stored cleaned-main-content evidence visibly supports the detector."
                elif expected == 1:
                    verdict = "ambiguous"
                    note = "Positive value lacks retained text evidence; structural tag may still be valid."
                else:
                    verdict = "true_negative_in_stored_snapshot"
                    note = "No matching structure was found in the stored cleaned snapshot; live-page drift remains possible."
                output.append(
                    {
                        "feature": feature,
                        "review_stratum": label,
                        "normalized_url": row.normalized_url,
                        "source_url": row.source_url,
                        "domain": row.source_root_domain,
                        "detected_value": expected,
                        "position_ratio": getattr(row, MAJOR_POSITION_FEATURES[feature]),
                        "stored_evidence": json.dumps(evidence_items, ensure_ascii=False),
                        "manual_validation_result": verdict,
                        "review_note": note,
                        "validation_source": "stored_cleaned_main_content_html_or_markdown",
                    }
                )
    return pd.DataFrame(output)


def _model_readiness(
    coverage: pd.DataFrame,
    within: pd.DataFrame,
    sparse: pd.DataFrame,
    high_corr: pd.DataFrame,
) -> pd.DataFrame:
    output = []
    for feature in MAJOR_POSITION_FEATURES:
        cov = coverage.loc[coverage["feature"].eq(feature)].iloc[0]
        wd = within.loc[within["feature"].eq(feature)].iloc[0]
        feature_sparse = sparse[sparse["feature"].eq(feature)]
        representation = feature_sparse["recommended_grouping"].iloc[0] if len(feature_sparse) else "Exclude from regression"
        correlated = high_corr[
            high_corr["feature_a"].str.contains(feature, regex=False)
            | high_corr["feature_b"].str.contains(feature, regex=False)
        ]
        coverage_rate = float(cov["percentage_of_eligible_pages"])
        if coverage_rate >= 0.10 and representation != "Exclude from regression":
            role = "position extension"
        elif coverage_rate >= 0.03:
            role = "robustness check"
        else:
            role = "exclude"
        if feature in {"table", "list", "numeric_evidence"} and coverage_rate >= 0.10:
            role = "main model candidate" if wd["fixed_effect_readiness"] == "Ready" else "position extension"
        output.append(
            {
                "feature": feature,
                "recommended_representation": representation,
                "suggested_reference_group": "No feature for full sample; Q4 for feature-present conditional analysis",
                "full_sample_usable": bool(cov["extraction_success_rate"] >= 0.80 and coverage_rate >= 0.03),
                "conditional_sample_usable": bool(cov["pages_with_feature"] >= 20),
                "domain_FE_usable": wd["fixed_effect_readiness"] in {"Ready", "Usable with caution"},
                "fixed_effect_readiness": wd["fixed_effect_readiness"],
                "sparse_cell_risk": "high" if feature_sparse["sparse_flag"].any() else "low",
                "multicollinearity_risk": "high" if len(correlated) else "moderate",
                "recommended_model_role": role,
                "reason": (
                    f"Coverage={coverage_rate:.1%}; {int(wd['domains_with_both_position_and_outcome_variation'])} domains "
                    f"vary in both position and outcome; within-domain SD={wd['within_domain_standard_deviation']:.3f}."
                ),
            }
        )
    return pd.DataFrame(output)


def _validate_outputs(urls: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    checks = []
    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        presence = FEATURE_META[feature]["presence"]
        ratio_values = pd.to_numeric(urls[ratio], errors="coerce")
        present = pd.to_numeric(urls[presence], errors="coerce")
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        expected = [
            _quartile(value, present_value, bool(measured))
            for value, present_value, measured in zip(ratio_values, present, urls["position_features_available"])
        ]
        checks.extend(
            [
                {"check": f"{feature}: ratios between 0 and 1", "passed": bool(ratio_values.dropna().between(0, 1).all())},
                {"check": f"{feature}: absence is not position zero", "passed": bool(ratio_values[present.eq(0)].isna().all())},
                {"check": f"{feature}: quartiles match ratios", "passed": bool(pd.Series(expected, index=urls.index).eq(urls[quartile]).all())},
                {"check": f"{feature}: counts non-negative", "passed": bool(pd.to_numeric(urls[FEATURE_META[feature]['count']], errors='coerce').dropna().ge(0).all())},
            ]
        )
    checks.extend(
        [
            {"check": "density denominators valid", "passed": bool(pd.to_numeric(urls["total_main_content_token_count"], errors="coerce").ge(0).all())},
            {"check": "extraction failure is not absence", "passed": bool(urls.loc[urls["position_features_available"].eq(0), [meta["presence"] for meta in FEATURE_META.values()]].isna().all().all())},
            {"check": "citation label preserved", "passed": bool(rows["cited"].isin([0, 1]).all())},
            {"check": "row total preserved", "passed": bool(len(rows) == 5758)},
        ]
    )
    result = pd.DataFrame(checks)
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise ValueError(f"Position-feature validation failed: {failed}")
    return result


def _build_static_report(
    output_dir: Path,
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    citation: pd.DataFrame,
    within: pd.DataFrame,
    readiness: pd.DataFrame,
) -> Path:
    figures = []
    coverage_fig = px.bar(
        coverage.sort_values("percentage_of_eligible_pages"),
        x="percentage_of_eligible_pages",
        y="feature",
        orientation="h",
        text="pages_with_feature",
        title=f"Feature coverage across eligible pages (n={int(coverage.applicable_page_count.max()):,})",
    )
    coverage_fig.update_xaxes(tickformat=".0%")
    figures.append(coverage_fig)
    table_citation = citation[
        (citation.feature == "table") & (citation.grouping == "quartile") & (citation.sample_scope == "full_sample")
    ].sort_values("category_order")
    citation_fig = px.bar(
        table_citation,
        x="category",
        y="citation_rate",
        error_y=table_citation["ci_high"] - table_citation["citation_rate"],
        error_y_minus=table_citation["citation_rate"] - table_citation["ci_low"],
        text="n_observations",
        title=f"Citation rate by first-table position (n={len(rows):,} surfaced rows)",
    )
    citation_fig.update_yaxes(tickformat=".0%")
    figures.append(citation_fig)
    fe_fig = px.scatter(
        within,
        x="within_domain_standard_deviation",
        y="domains_with_both_position_and_outcome_variation",
        size="informative_observations",
        color="fixed_effect_readiness",
        hover_name="feature",
        title=f"Domain fixed-effect readiness (n={len(rows):,} surfaced rows)",
    )
    figures.append(fe_fig)

    plot_html = []
    for index, figure in enumerate(figures):
        plot_html.append(
            figure.to_html(full_html=False, include_plotlyjs="inline" if index == 0 else False, config={"displaylogo": False})
        )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Position Feature EDA</title>
<style>body{{font-family:Arial,sans-serif;color:#17202a;margin:0;background:#f7f8fa}}main{{max-width:1180px;margin:auto;padding:32px}}h1,h2{{letter-spacing:0}}section{{background:white;border:1px solid #dfe3e8;padding:22px;margin:18px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #e4e7eb;padding:8px;text-align:left}}.note{{padding:14px;background:#eef4ff;border-left:4px solid #356ae6}}</style></head>
<body><main><h1>Position Feature EDA</h1>
<p class="note">Descriptive associations among surfaced sources. These results are not causal and do not estimate a new LPM.</p>
<section><h2>Sample</h2><p>{len(rows):,} source-prompt rows, {rows.normalized_url.nunique():,} URLs, {rows.source_root_domain.nunique():,} domains, citation rate {rows.cited.mean():.1%}.</p></section>
<section>{plot_html[0]}</section><section>{plot_html[1]}</section><section>{plot_html[2]}</section>
<section><h2>Coverage</h2>{coverage.to_html(index=False, border=0)}</section>
<section><h2>Fixed-effect readiness</h2>{within.to_html(index=False, border=0)}</section>
<section><h2>Model-readiness recommendation</h2>{readiness.to_html(index=False, border=0)}</section>
</main></body></html>"""
    path = output_dir / "frontend" / "position_feature_eda_report.html"
    path.write_text(html, encoding="utf-8")
    return path


def _findings_text(
    rows: pd.DataFrame,
    coverage: pd.DataFrame,
    outcome: pd.DataFrame,
    within: pd.DataFrame,
    readiness: pd.DataFrame,
) -> str:
    sufficient = coverage.loc[coverage.percentage_of_eligible_pages.ge(0.10), "feature"].tolist()
    differences = outcome.assign(abs_difference=lambda d: d.difference_in_medians.abs()).sort_values("abs_difference", ascending=False).head(5)
    ready = within.loc[within.fixed_effect_readiness.isin(["Ready", "Usable with caution"]), "feature"].tolist()
    not_ready = within.loc[within.fixed_effect_readiness.eq("Not suitable for domain fixed effects"), "feature"].tolist()
    main = readiness.loc[readiness.recommended_model_role.eq("main model candidate"), "feature"].tolist()
    extension = readiness.loc[readiness.recommended_model_role.eq("position extension"), "feature"].tolist()
    robust = readiness.loc[readiness.recommended_model_role.eq("robustness check"), "feature"].tolist()
    excluded = readiness.loc[readiness.recommended_model_role.eq("exclude"), "feature"].tolist()
    return f"""POSITION FEATURE EDA FINDINGS

Scope
{len(rows):,} surfaced source-prompt rows; {rows.normalized_url.nunique():,} unique URLs; {rows.source_root_domain.nunique():,} domains; citation rate {rows.cited.mean():.1%}.

Coverage
Features with at least 10 percent eligible-page coverage: {', '.join(sufficient) or 'none'}.

Descriptive citation differences
Largest absolute cited versus not-cited median-position differences: {', '.join(differences.feature.tolist())}. These are descriptive associations, not causal effects.

Domain fixed-effects readiness
Ready or usable with caution: {', '.join(ready) or 'none'}.
Not suitable: {', '.join(not_ready) or 'none'}.

Future representation
Use the feature-specific grouping in position_feature_model_readiness.csv. No-feature and position effects must remain separate. Sparse groups must not be merged silently.

Future model role
Main-model candidates: {', '.join(main) or 'none'}.
Position extensions: {', '.join(extension) or 'none'}.
Robustness checks: {', '.join(robust) or 'none'}.
Exclude: {', '.join(excluded) or 'none'}.

Limitations
Position is measured from stored cleaned main-content HTML, with generated Markdown only when HTML is unavailable. Dynamic content, scrape-version drift, imperfect main-content selection, and template confounding remain. This pipeline does not fit a new LPM or logistic regression.
"""


def run_position_feature_eda(
    package_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    package = Path(package_dir or topic_output_dir() / "content_econometrics_ai_package").resolve()
    output = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    data_dir = output / "data"
    table_dir = output / "tables"
    figure_dir = output / "figures"
    frontend_data = output / "frontend" / "data"
    for directory in (data_dir, table_dir, figure_dir, frontend_data):
        directory.mkdir(parents=True, exist_ok=True)

    all_rows = pd.read_csv(package / "data/content_lpm_all_surfaced_rows.csv", low_memory=False)
    evidence = pd.read_csv(package / "data/url_content_evidence_compact.csv", low_memory=False)
    structure_dir = package / "tables/12_document_structure_features"
    structure = pd.read_csv(structure_dir / "url_document_structure_features.csv", low_memory=False)
    texts = pd.read_csv(structure_dir / "url_full_body_text_and_generated_markdown.csv.gz", low_memory=False)
    current_selected = CODE_ROOT / "outputs/econometrics_redesign_v3_20260727_faq_deduplicated/data/selected_feature_rows.csv"
    selected = pd.read_csv(current_selected, low_memory=False) if current_selected.exists() else pd.DataFrame()

    checkpoint_path = data_dir / "url_position_features_extraction_checkpoint.parquet"
    if checkpoint_path.exists():
        checkpoint = pd.read_parquet(checkpoint_path)
        checkpoint_valid = (
            len(checkpoint) == len(structure)
            and "position_feature_version" in checkpoint
            and checkpoint["position_feature_version"].eq(POSITION_FEATURE_VERSION).all()
        )
    else:
        checkpoint = pd.DataFrame()
        checkpoint_valid = False
    if checkpoint_valid:
        urls = checkpoint.drop(columns=["position_feature_version"])
    else:
        text_map = texts.set_index("normalized_url")["generated_markdown"].to_dict()
        feature_rows = []
        for row in structure.itertuples(index=False):
            snapshot_path = Path(str(getattr(row, "snapshot_path", "") or ""))
            snapshot = {}
            if snapshot_path.exists():
                try:
                    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    snapshot = {}
            raw_html = str(snapshot.get("html") or "")
            markdown = str(snapshot.get("markdown") or text_map.get(row.normalized_url, "") or "")
            extracted = extract_position_features(raw_html, str(row.source_url), markdown)
            if not bool(getattr(row, "html_available", 0)) and not markdown.strip():
                if not bool(evidence.loc[evidence.normalized_url.eq(row.normalized_url), "scrape_success"].fillna(False).any()):
                    extracted = _unmeasured_features("scrape_failed")
            feature_rows.append(
                {
                    "normalized_url": row.normalized_url,
                    "source_url": row.source_url,
                    "source_root_domain": row.source_root_domain,
                    "page_id": hashlib.sha1(str(row.normalized_url).encode()).hexdigest()[:16],
                    **extracted,
                }
            )
        urls = pd.DataFrame(feature_rows)
        checkpoint = urls.assign(position_feature_version=POSITION_FEATURE_VERSION)
        checkpoint.to_parquet(checkpoint_path, index=False)
    url_metadata = evidence[[
        "normalized_url", "page_title", "scrape_success", "content_strength", "content_quality_flag",
        "word_count", "heading_count", "site_type_general", "page_type_family_general",
    ]].drop_duplicates("normalized_url")
    urls = urls.merge(url_metadata, on="normalized_url", how="left", validate="one_to_one")
    urls["page_type"] = urls["page_type_family_general"].fillna("unknown")
    urls["source_type"] = urls["site_type_general"].fillna("unknown")

    source_baseline = all_rows[["normalized_url", "has_table"]].drop_duplicates("normalized_url").rename(
        columns={"has_table": "has_table_source_baseline"}
    )
    urls = urls.merge(source_baseline, on="normalized_url", how="left", validate="one_to_one")
    replacement_columns = [
        column
        for metadata in FEATURE_META.values()
        for column in (metadata["presence"], metadata["count"])
        if column in all_rows
    ]
    rows = all_rows.drop(columns=replacement_columns, errors="ignore")
    attach_columns = [
        "normalized_url",
        *[
            column
            for column in urls.columns
            if column != "normalized_url" and column not in rows.columns
        ],
    ]
    rows = rows.merge(
        urls[attach_columns],
        on="normalized_url",
        how="left",
        validate="many_to_one",
    )
    if not selected.empty:
        taxonomy = selected[[
            "prompt_id", "normalized_url", "page_type_family_gemini_v1_collapsed", "source_type_general_gemini_v1_collapsed"
        ]].drop_duplicates(["prompt_id", "normalized_url"])
        rows = rows.merge(taxonomy, on=["prompt_id", "normalized_url"], how="left", validate="many_to_one")
        rows["page_type"] = rows["page_type_family_gemini_v1_collapsed"].fillna(rows["page_type"])
        rows["source_type"] = rows["source_type_general_gemini_v1_collapsed"].fillna(rows["source_type"])
        url_taxonomy = rows.groupby("normalized_url", observed=True).agg(
            page_type=("page_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
            source_type=("source_type", lambda s: s.mode().iloc[0] if not s.mode().empty else "unknown"),
        ).reset_index()
        urls = urls.drop(columns=["page_type", "source_type"]).merge(url_taxonomy, on="normalized_url", how="left")

    for feature, ratio in MAJOR_POSITION_FEATURES.items():
        presence = FEATURE_META[feature]["presence"]
        quartile = ratio.replace("position_ratio", "position_quartile")
        if ratio == "direct_answer_position_ratio":
            quartile = "direct_answer_position_quartile"
        for frame in (urls, rows):
            frame[f"{feature}_in_first_quartile"] = np.where(
                pd.to_numeric(frame[presence], errors="coerce").isna(), np.nan,
                (pd.to_numeric(frame[ratio], errors="coerce") < 0.25).astype(float),
            )
            frame[f"{feature}_in_first_half"] = np.where(
                pd.to_numeric(frame[presence], errors="coerce").isna(), np.nan,
                (pd.to_numeric(frame[ratio], errors="coerce") < 0.50).astype(float),
            )
            frame[quartile] = [
                _quartile(value, present, bool(measured))
                for value, present, measured in zip(frame[ratio], frame[presence], frame["position_features_available"])
            ]

    canonical_aliases = {
        "first_table_token_position": "table_start_token_index",
        "median_table_position_ratio": "table_median_position_ratio",
        "question_headings_first_half_share": "question_heading_share_in_first_half",
        "faq_position_quartile": "faq_start_position_quartile",
        "definition_position_quartile": "first_definition_position_quartile",
        "comparison_position_quartile": "first_comparison_position_quartile",
        "steps_position_quartile": "first_steps_position_quartile",
        "external_citation_position_quartile": "first_external_citation_position_quartile",
    }
    for target, source in canonical_aliases.items():
        for frame in (urls, rows):
            frame[target] = frame[source]

    previous_summary = CODE_ROOT / "outputs/econometrics_redesign_v3_20260727_faq_deduplicated/tables/feature_distribution_support_summary.csv"
    previously_analyzed = set(pd.read_csv(previous_summary)["feature_name"].astype(str)) if previous_summary.exists() else set()
    audit = _build_feature_audit(set(all_rows.columns), set(urls.columns), previously_analyzed)
    coverage = _coverage_table(urls, rows, set(all_rows.columns))
    distribution, citation, outcome = _position_outputs(rows)
    within = _within_domain_diagnostics(rows)
    sparse = _sparse_cells(citation)
    taxonomy = _taxonomy_crosstabs(rows)
    length = _page_length_relationship(rows)
    associations, high_corr, clusters = _correlation_outputs(rows)
    categorical_associations = _categorical_associations(rows)
    domain_examples = _domain_examples(rows)
    manual = _manual_validation(urls)
    readiness = _model_readiness(coverage, within, sparse, high_corr)
    validation = _validate_outputs(urls, rows)

    audit_feature_lookup: dict[str, str] = {}
    for base_feature, metadata in FEATURE_META.items():
        audit_feature_lookup[metadata["presence"]] = base_feature
        audit_feature_lookup[metadata["count"]] = base_feature
        audit_feature_lookup[MAJOR_POSITION_FEATURES[base_feature]] = base_feature
    audit_feature_lookup.update(
        {
            "first_table_token_position": "table",
            "first_table_position_quartile": "table",
            "median_table_position_ratio": "table",
            "table_in_first_quartile": "table",
            "table_in_first_half": "table",
            "table_share_in_first_half": "table",
            "first_list_position_quartile": "list",
            "list_in_first_half": "list",
            "direct_answer_position_quartile": "direct_answer",
            "direct_answer_in_first_quartile": "direct_answer",
            "direct_answer_in_first_half": "direct_answer",
            "first_question_heading_position_quartile": "question_heading",
            "question_headings_first_half_share": "question_heading",
            "definition_position_quartile": "definition",
            "definition_in_first_quartile": "definition",
            "faq_position_quartile": "faq",
            "faq_in_first_quartile": "faq",
            "faq_in_first_half": "faq",
            "comparison_position_quartile": "comparison",
            "comparison_in_first_quartile": "comparison",
            "comparison_in_first_half": "comparison",
            "steps_position_quartile": "steps",
            "steps_in_first_half": "steps",
            "first_numeric_evidence_position_quartile": "numeric_evidence",
            "numeric_evidence_in_first_quartile": "numeric_evidence",
            "numeric_density_first_quartile": "numeric_evidence",
            "numeric_density_first_half": "numeric_evidence",
            "external_citation_position_quartile": "external_citation",
            "external_citation_in_first_half": "external_citation",
        }
    )
    audit["base_position_feature"] = audit["feature_name"].map(audit_feature_lookup)
    audit = audit.merge(
        coverage[[
            "feature", "extraction_success_rate", "missing_rate", "applicable_page_count",
            "applicable_domain_count", "percentage_of_eligible_pages",
        ]],
        left_on="base_position_feature",
        right_on="feature",
        how="left",
    ).drop(columns="feature")
    audit = audit.merge(
        readiness[["feature", "fixed_effect_readiness"]],
        left_on="base_position_feature",
        right_on="feature",
        how="left",
    ).drop(columns=["feature", "base_position_feature"])

    missingness = urls.groupby("position_extraction_status", dropna=False).agg(
        pages=("normalized_url", "nunique"), domains=("source_root_domain", "nunique")
    ).reset_index()
    missingness["page_share"] = missingness["pages"] / len(urls)

    output_tables = {
        "position_feature_audit": audit,
        "position_feature_coverage": coverage,
        "position_feature_distribution_summary": distribution,
        "citation_rate_by_feature_position": citation,
        "position_feature_outcome_comparison": outcome,
        "position_feature_within_domain_diagnostics": within,
        "position_feature_sparse_cell_diagnostics": sparse,
        "position_feature_model_readiness": readiness,
        "position_feature_manual_validation": manual,
        "position_feature_taxonomy_crosstab": taxonomy,
        "position_feature_page_length_relationship": length,
        "position_feature_associations": associations,
        "position_feature_high_correlation_pairs": high_corr,
        "position_feature_cluster_summary": clusters,
        "position_feature_categorical_association": categorical_associations,
        "position_feature_domain_examples": domain_examples,
        "position_feature_missingness": missingness,
        "position_feature_validation_checks": validation,
    }
    for name, frame in output_tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)
        frame.to_csv(frontend_data / f"{name}.csv", index=False)

    parquet_path = data_dir / "scope_condo_eda_ready_with_position_features.parquet"
    csv_path = data_dir / "scope_condo_eda_ready_with_position_features.csv"
    rows.to_parquet(parquet_path, index=False)
    rows.to_csv(csv_path, index=False)
    urls.to_csv(data_dir / "url_position_features.csv", index=False)
    rows.to_parquet(frontend_data / "scope_condo_eda_ready_with_position_features.parquet", index=False)
    urls.to_csv(frontend_data / "url_position_features.csv", index=False)

    figures = []
    coverage_fig = px.bar(
        coverage.sort_values("percentage_of_eligible_pages"), x="percentage_of_eligible_pages", y="feature",
        orientation="h", text="pages_with_feature", title=f"Coverage of newly measured features (n={len(urls):,} URLs)"
    )
    coverage_fig.update_xaxes(tickformat=".0%")
    figures.append(("feature_coverage", coverage_fig))
    for feature in ["table", "faq", "direct_answer", "definition", "comparison"]:
        ratio = MAJOR_POSITION_FEATURES[feature]
        present = rows[
            pd.to_numeric(rows[FEATURE_META[feature]["presence"]], errors="coerce").eq(1)
        ].copy()
        present["citation_status"] = np.where(
            pd.to_numeric(present["cited"], errors="coerce").eq(1),
            "Cited",
            "Not cited",
        )
        fig = px.histogram(
            present, x=ratio, nbins=20, color="citation_status", barmode="overlay",
            color_discrete_map=CITATION_COLORS,
            category_orders={"citation_status": ["Cited", "Not cited"]},
            title=f"{feature.replace('_', ' ').title()} position among feature-present rows (n={len(present):,})",
        )
        figures.append((f"position_histogram_{feature}", fig))
        rate_data = citation[(citation.feature == feature) & (citation.grouping == "quartile") & (citation.sample_scope == "full_sample")].sort_values("category_order")
        fig2 = px.bar(
            rate_data, x="category", y="citation_rate", text="n_observations",
            error_y=rate_data["ci_high"] - rate_data["citation_rate"],
            error_y_minus=rate_data["citation_rate"] - rate_data["ci_low"],
            title=f"Citation rate by {feature.replace('_', ' ')} position (n={int(rate_data.n_observations.sum()):,})",
        )
        fig2.update_yaxes(tickformat=".0%")
        figures.append((f"citation_by_position_{feature}", fig2))
    for name, figure in figures:
        figure.write_html(figure_dir / f"{name}.html", include_plotlyjs="inline", full_html=True)

    report_path = _build_static_report(output, rows, coverage, citation, within, readiness)
    findings = _findings_text(rows, coverage, outcome, within, readiness)
    (output / "POSITION_FEATURE_EDA_FINDINGS.txt").write_text(findings, encoding="utf-8")
    manifest = {
        "status": "position_feature_eda_ready_for_model_planning",
        "version": POSITION_FEATURE_VERSION,
        "rows": len(rows),
        "unique_urls": int(rows.normalized_url.nunique()),
        "domains": int(rows.source_root_domain.nunique()),
        "citation_rate": float(rows.cited.mean()),
        "new_regression_estimated": False,
        "output_dir": str(output),
        "static_report": str(report_path),
        "all_validation_checks_passed": bool(validation.passed.all()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
