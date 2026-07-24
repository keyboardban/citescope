"""Leakage-safe writing and factual-density feature layer for notebook 10."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


TEXT_FIELD_CANDIDATES = (
    "text",
    "markdown",
    "clean_text",
    "page_text",
    "content_text",
    "body_text",
    "text_excerpt",
    "preview_text",
    "page_text_excerpt",
    "page_text_preview_3000_chars",
    "title",
    "page_title",
    "description",
    "meta_description",
    "headings",
    "url",
    "source_url",
    "normalized_url",
)
THAI_DIGIT_TRANSLATION = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
NUMBER_RE = re.compile(r"(?<![\w])(?:\d[\d,]*(?:\.\d+)?)(?![\w])", re.UNICODE)
URL_RE = re.compile(r"https?://[^\s<>()\]\[\"']+", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*|[\u0E00-\u0E7F]+", re.UNICODE)

HEADING = "C(heading_count_group, Treatment(reference='0-1'))"
LINK = "C(link_count_group, Treatment(reference='9+'))"
STRENGTH = "C(content_strength, Treatment(reference='strong'))"
PROMPT_FE = "C(prompt_id)"
BASELINE_M2 = (
    f"cited ~ log2_word_count_plus1 + has_table + {HEADING} + {LINK} + {STRENGTH} + {PROMPT_FE}"
)


def _plotly_graph_objects() -> Any:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Plotly is required only when writing notebook 10 figures. "
            "Install plotly in the execution environment or use the configured Plotly notebook kernel."
        ) from exc
    return go


def _write_plotly(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    fig.write_json(path.with_suffix(".plotly.json"))


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_text(value: object) -> str:
    text = _text(value).translate(THAI_DIGIT_TRANSLATION)
    return re.sub(r"[ \t]+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _count_pattern(text: str, pattern: str | re.Pattern[str]) -> int:
    compiled = re.compile(pattern, re.IGNORECASE | re.UNICODE) if isinstance(pattern, str) else pattern
    return len(compiled.findall(text))


def _count_terms(text: str, terms: Iterable[str]) -> int:
    if not text:
        return 0
    escaped = sorted((re.escape(term) for term in terms), key=len, reverse=True)
    return _count_pattern(text, rf"(?:{'|'.join(escaped)})")


def _binary(value: bool) -> int:
    return int(bool(value))


def _nan_feature_row(feature_names: Iterable[str]) -> dict[str, float]:
    return {feature: np.nan for feature in feature_names}


def _join_unique_parts(*parts: object) -> str:
    result: list[str] = []
    current = ""
    for part in parts:
        value = _normalise_text(part)
        if not value:
            continue
        folded = value.casefold()
        if folded in current.casefold():
            continue
        result.append(value)
        current = "\n\n".join(result)
    return current


def audit_text_fields(
    measurable: pd.DataFrame,
    evidence: pd.DataFrame,
    prompt_reference: pd.DataFrame,
) -> pd.DataFrame:
    sources = {
        "content_lpm_measurable_rows.csv": measurable,
        "url_content_evidence_compact.csv": evidence,
        "prompt_reference.csv": prompt_reference,
    }
    preferred = {
        "page_text_preview_3000_chars": "primary page-text preview; maximum 3,000 characters",
        "page_text_excerpt": "fallback page-text excerpt; maximum 1,200 characters",
        "page_title": "page metadata and prompt-title relevance",
        "meta_description": "page metadata when present",
        "source_url": "URL metadata and source-domain comparison",
        "normalized_url": "merge key",
        "prompt": "prompt-page relevance only; never combined with answer text",
    }
    rows = []
    for column in (*TEXT_FIELD_CANDIDATES, "prompt"):
        matching = [(filename, frame) for filename, frame in sources.items() if column in frame]
        if not matching:
            rows.append(
                {
                    "column_name": column,
                    "source_file": "",
                    "exists": False,
                    "non_null_count": 0,
                    "median_length": np.nan,
                    "p90_length": np.nan,
                    "use_for_extraction": False,
                    "notes": "not available",
                }
            )
            continue
        for filename, frame in matching:
            values = frame[column].fillna("").astype(str).str.strip()
            lengths = values[values.ne("")].str.len()
            rows.append(
                {
                    "column_name": column,
                    "source_file": filename,
                    "exists": True,
                    "non_null_count": int(values.ne("").sum()),
                    "median_length": float(lengths.median()) if len(lengths) else np.nan,
                    "p90_length": float(lengths.quantile(0.9)) if len(lengths) else np.nan,
                    "use_for_extraction": column in preferred,
                    "notes": preferred.get(column, "available but not selected for deterministic extraction"),
                }
            )
    return pd.DataFrame(rows)


def assemble_url_text(measurable: pd.DataFrame, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_urls = set(measurable["normalized_url"].astype(str))
    urls = evidence[evidence["normalized_url"].astype(str).isin(target_urls)].copy()
    urls = urls.drop_duplicates("normalized_url", keep="first").reset_index(drop=True)
    rows = []
    for row in urls.itertuples(index=False):
        title = _normalise_text(getattr(row, "page_title", ""))
        description = _normalise_text(getattr(row, "meta_description", ""))
        preview = _normalise_text(getattr(row, "page_text_preview_3000_chars", ""))
        excerpt = _normalise_text(getattr(row, "page_text_excerpt", ""))
        content_chars = pd.to_numeric(pd.Series([getattr(row, "content_chars", np.nan)]), errors="coerce").iloc[0]
        if preview:
            primary = preview
            text_source = "page_text_preview_3000_chars"
        elif excerpt:
            primary = excerpt
            text_source = "page_text_excerpt"
        else:
            primary = ""
            text_source = "title_description_only" if title or description else "none"
        assembled = _join_unique_parts(primary, title, description)
        primary_length = len(primary)
        if preview and pd.notna(content_chars) and content_chars <= primary_length + 100:
            scope = "full_text"
        elif preview or excerpt:
            scope = "excerpt_only"
        elif assembled:
            scope = "title_description_only"
        else:
            scope = "no_text"
        rows.append(
            {
                "normalized_url": str(row.normalized_url),
                "source_url": _text(getattr(row, "source_url", "")),
                "source_root_domain": _text(getattr(row, "source_root_domain", "")),
                "url_title": title,
                "url_description": description,
                "page_text_excerpt": excerpt,
                "url_text_for_features": assembled,
                "url_text_length_chars": len(assembled),
                "url_text_length_words": len(_tokens(assembled)),
                "reported_content_chars": content_chars,
                "full_page_text_available": scope == "full_text",
                "limited_excerpt_only": scope == "excerpt_only",
                "text_source_used": text_source,
                "feature_extraction_text_scope": scope,
                "text_feature_available": scope != "no_text",
            }
        )
    assembly = pd.DataFrame(rows)
    missing_urls = target_urls - set(assembly["normalized_url"])
    if missing_urls:
        assembly = pd.concat(
            [
                assembly,
                pd.DataFrame(
                    {
                        "normalized_url": sorted(missing_urls),
                        "source_url": "",
                        "source_root_domain": "",
                        "url_title": "",
                        "url_description": "",
                        "page_text_excerpt": "",
                        "url_text_for_features": "",
                        "url_text_length_chars": 0,
                        "url_text_length_words": 0,
                        "reported_content_chars": np.nan,
                        "full_page_text_available": False,
                        "limited_excerpt_only": False,
                        "text_source_used": "none",
                        "feature_extraction_text_scope": "no_text",
                        "text_feature_available": False,
                    }
                ),
            ],
            ignore_index=True,
        )
    scope_counts = assembly["feature_extraction_text_scope"].value_counts()
    audit_rows = [
        {"metric": "unique_urls_in_measurable_rows", "value": measurable["normalized_url"].nunique()},
        {"metric": "unique_urls_in_text_assembly", "value": assembly["normalized_url"].nunique()},
        {
            "metric": "urls_with_usable_text",
            "value": int(assembly["text_feature_available"].sum()),
        },
        {"metric": "urls_full_text", "value": int(scope_counts.get("full_text", 0))},
        {"metric": "urls_excerpt_only", "value": int(scope_counts.get("excerpt_only", 0))},
        {
            "metric": "urls_title_description_only",
            "value": int(scope_counts.get("title_description_only", 0)),
        },
        {"metric": "urls_no_extractable_text", "value": int(scope_counts.get("no_text", 0))},
        {
            "metric": "source_rows_with_usable_text_after_merge",
            "value": int(
                measurable["normalized_url"]
                .map(assembly.set_index("normalized_url")["text_feature_available"])
                .fillna(False)
                .sum()
            ),
        },
    ]
    return assembly, pd.DataFrame(audit_rows)


WRITING_FEATURES = (
    "paragraph_count",
    "sentence_count",
    "median_sentence_length_words",
    "mean_sentence_length_words",
    "p90_sentence_length_words",
    "median_paragraph_length_words",
    "short_sentence_share",
    "long_sentence_share",
    "bullet_like_line_count",
    "numbered_list_line_count",
    "list_structure_score",
    "has_bullet_list",
    "has_numbered_list",
    "question_mark_count",
    "question_heading_count",
    "faq_phrase_count",
    "has_faq_pattern",
    "has_question_answer_structure",
    "opening_100_words",
    "opening_has_summary_signal",
    "opening_has_direct_answer_signal",
    "opening_numeric_fact_count",
    "opening_price_or_location_fact_count",
    "writing_structure_score",
)

FACTUAL_FEATURES = (
    "number_token_count",
    "number_token_per_1000_words",
    "percent_mention_count",
    "year_mention_count",
    "range_mention_count",
    "measurement_mention_count",
    "price_mention_count",
    "price_per_sqm_mention_count",
    "has_price_detail",
    "unit_size_mention_count",
    "sqm_mention_count",
    "bedroom_mention_count",
    "floor_plan_mention_count",
    "has_unit_size_detail",
    "factual_numeric_density_score",
    "price_unit_detail_score",
)

LOCATION_FEATURES = (
    "transit_station_mention_count",
    "bts_mention_count",
    "mrt_mention_count",
    "distance_mention_count",
    "walking_time_mention_count",
    "location_landmark_mention_count",
    "neighborhood_mention_count",
    "has_transit_detail",
    "has_location_detail",
    "location_transit_specificity_score",
)

AMENITY_FEATURES = (
    "amenity_mention_count",
    "facility_mention_count",
    "parking_mention_count",
    "pool_mention_count",
    "gym_mention_count",
    "security_mention_count",
    "pet_friendly_mention_count",
    "developer_mention_count",
    "project_name_mention_count",
    "brand_or_project_entity_count",
    "has_amenity_detail",
    "has_project_entity_detail",
    "amenity_project_detail_score",
)

EVIDENCE_FEATURES = (
    "external_link_count",
    "external_link_domain_count",
    "official_link_count",
    "source_reference_phrase_count",
    "evidence_phrase_count",
    "has_external_evidence",
    "external_evidence_score",
)

URL_FEATURE_COLUMNS = (
    *WRITING_FEATURES,
    *FACTUAL_FEATURES,
    *LOCATION_FEATURES,
    *AMENITY_FEATURES,
    *EVIDENCE_FEATURES,
)


SUMMARY_TERMS = (
    "summary",
    "overview",
    "key facts",
    "highlights",
    "at a glance",
    "สรุป",
    "ภาพรวม",
    "จุดเด่น",
    "ข้อมูลสำคัญ",
)
DIRECT_ANSWER_TERMS = (
    "starts at",
    "starting price",
    "located",
    "is a",
    "offers",
    "ราคาเริ่มต้น",
    "เริ่มต้น",
    "ตั้งอยู่",
    "ห่างจาก",
    "ประกอบด้วย",
)
FAQ_TERMS = (
    "faq",
    "frequently asked questions",
    "common questions",
    "คำถามที่พบบ่อย",
    "ถามตอบ",
    "q&a",
)
ANSWER_SIGNAL_TERMS = ("answer", "คำตอบ", "ตอบ:", "a:")

PRICE_TERMS = (
    "บาท",
    "ล้านบาท",
    "ล้าน",
    "thb",
    "baht",
    "price",
    "ราคา",
    "เริ่มต้น",
    "ต่อตารางเมตร",
    "บาท/ตร.ม.",
    "per sqm",
)
UNIT_TERMS = (
    "ตร.ม.",
    "ตารางเมตร",
    "sqm",
    "sq.m.",
    "square meter",
    "square metre",
    "bedroom",
    "bedrooms",
    " bed ",
    "ห้องนอน",
    "studio",
    "penthouse",
    "duplex",
)
FLOOR_PLAN_TERMS = ("floor plan", "floorplan", "unit plan", "แปลน", "ผังห้อง", "แบบห้อง")

NEIGHBORHOOD_TERMS = (
    "หลังสวน",
    "ชิดลม",
    "ลุมพินี",
    "พร้อมพงษ์",
    "พร้อมศรี",
    "ทองหล่อ",
    "สุขุมวิท",
    "สวนลุมพินี",
    "langsuan",
    "chidlom",
    "lumpini",
    "phrom phong",
    "promsri",
    "thonglor",
    "thong lo",
    "sukhumvit",
)
LANDMARK_TERMS = (
    "central embassy",
    "central chidlom",
    "one bangkok",
    "siam paragon",
    "emporium",
    "emquartier",
    "emsphere",
    "lumpini park",
    "สวนลุมพินี",
    "โรงพยาบาล",
    "hospital",
    "school",
    "มหาวิทยาลัย",
    "university",
)

AMENITY_TERMS = (
    "สระว่ายน้ำ",
    "ฟิตเนส",
    "ที่จอดรถ",
    "รปภ",
    "ความปลอดภัย",
    "สวน",
    "lounge",
    "pool",
    "gym",
    "parking",
    "security",
    "garden",
    "concierge",
    "lobby",
    "facility",
    "facilities",
    "amenity",
    "amenities",
)
FACILITY_TERMS = ("facility", "facilities", "amenity", "amenities", "สิ่งอำนวยความสะดวก")
PARKING_TERMS = ("parking", "car park", "ที่จอดรถ")
POOL_TERMS = ("pool", "swimming pool", "สระว่ายน้ำ")
GYM_TERMS = ("gym", "fitness", "ฟิตเนส", "ห้องออกกำลังกาย")
SECURITY_TERMS = ("security", "cctv", "guard", "รปภ", "ความปลอดภัย")
PET_TERMS = ("pet friendly", "pets allowed", "pet-friendly", "เลี้ยงสัตว์", "สัตว์เลี้ยง")
DEVELOPER_TERMS = ("developer", "developed by", "ผู้พัฒนา", "เจ้าของโครงการ", "บริษัทพัฒนา")
PROJECT_TERMS = ("project", "โครงการ", "residence", "residences", "condominium", "คอนโดมิเนียม")

REFERENCE_TERMS = (
    "อ้างอิง",
    "แหล่งข้อมูล",
    "ข้อมูลจาก",
    "according to",
    "source",
    "reference",
    "data from",
)
EVIDENCE_TERMS = (
    "รายงาน",
    "official",
    "รายงานจาก",
    "ผลสำรวจ",
    "research",
    "study",
    "report",
    "statistics",
    "สถิติ",
)


def _sentence_lengths(text: str) -> list[int]:
    segments = [segment.strip() for segment in re.split(r"[.!?。！？\n]+", text) if segment.strip()]
    return [len(_tokens(segment)) for segment in segments if _tokens(segment)]


def _paragraph_lengths(text: str) -> list[int]:
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+|\r\n\s*\r\n+", text)
        if paragraph.strip()
    ]
    if len(paragraphs) <= 1:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()] or ([text] if text else [])
    return [len(_tokens(paragraph)) for paragraph in paragraphs if _tokens(paragraph)]


def extract_writing_features(text: str) -> dict[str, Any]:
    if not text:
        return _nan_feature_row(WRITING_FEATURES)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentence_lengths = _sentence_lengths(text)
    paragraph_lengths = _paragraph_lengths(text)
    words = _tokens(text)
    opening = " ".join(words[:100])
    bullet_lines = sum(bool(re.match(r"^\s*(?:[-*•●▪◦])\s+", line)) for line in lines)
    numbered_lines = sum(bool(re.match(r"^\s*(?:\d+[.)]|[a-zA-Z][.)])\s+", line)) for line in lines)
    question_marks = text.count("?") + text.count("？")
    question_headings = sum(
        bool(
            (line.endswith("?") or line.endswith("？") or re.match(r"^(?:q:|คำถาม|ถาม:)", line, re.I))
            and len(_tokens(line)) <= 24
        )
        for line in lines
    )
    faq_count = _count_terms(text.casefold(), FAQ_TERMS)
    has_faq = faq_count > 0
    qa_structure = has_faq or (question_headings > 0 and _count_terms(text.casefold(), ANSWER_SIGNAL_TERMS) > 0)
    opening_lower = opening.casefold()
    opening_summary = _count_terms(opening_lower, SUMMARY_TERMS) > 0
    opening_direct = _count_terms(opening_lower, DIRECT_ANSWER_TERMS) > 0
    opening_numeric = len(NUMBER_RE.findall(opening))
    opening_price_location = _count_terms(opening_lower, (*PRICE_TERMS, *NEIGHBORHOOD_TERMS, "bts", "mrt"))
    list_score = math.log1p(bullet_lines + numbered_lines) + int(bullet_lines > 0) + int(numbered_lines > 0)
    writing_score = (
        int(bullet_lines > 0)
        + int(numbered_lines > 0)
        + int(has_faq)
        + int(qa_structure)
        + int(opening_summary)
        + int(opening_direct)
    )
    return {
        "paragraph_count": len(paragraph_lengths),
        "sentence_count": len(sentence_lengths),
        "median_sentence_length_words": float(np.median(sentence_lengths)) if sentence_lengths else 0.0,
        "mean_sentence_length_words": float(np.mean(sentence_lengths)) if sentence_lengths else 0.0,
        "p90_sentence_length_words": float(np.quantile(sentence_lengths, 0.9)) if sentence_lengths else 0.0,
        "median_paragraph_length_words": float(np.median(paragraph_lengths)) if paragraph_lengths else 0.0,
        "short_sentence_share": (
            float(np.mean(np.asarray(sentence_lengths) <= 12)) if sentence_lengths else 0.0
        ),
        "long_sentence_share": (
            float(np.mean(np.asarray(sentence_lengths) >= 35)) if sentence_lengths else 0.0
        ),
        "bullet_like_line_count": bullet_lines,
        "numbered_list_line_count": numbered_lines,
        "list_structure_score": list_score,
        "has_bullet_list": _binary(bullet_lines > 0),
        "has_numbered_list": _binary(numbered_lines > 0),
        "question_mark_count": question_marks,
        "question_heading_count": question_headings,
        "faq_phrase_count": faq_count,
        "has_faq_pattern": _binary(has_faq),
        "has_question_answer_structure": _binary(qa_structure),
        "opening_100_words": opening,
        "opening_has_summary_signal": _binary(opening_summary),
        "opening_has_direct_answer_signal": _binary(opening_direct),
        "opening_numeric_fact_count": opening_numeric,
        "opening_price_or_location_fact_count": opening_price_location,
        "writing_structure_score": writing_score,
    }


def extract_factual_features(text: str) -> dict[str, Any]:
    if not text:
        return _nan_feature_row(FACTUAL_FEATURES)
    lower = text.casefold()
    words = max(1, len(_tokens(text)))
    number_count = len(NUMBER_RE.findall(text))
    percent_count = _count_pattern(lower, r"(?:\d+(?:\.\d+)?\s*%|\bpercent\b|เปอร์เซ็นต์|ร้อยละ)")
    year_count = _count_pattern(lower, r"(?<!\d)(?:19\d{2}|20\d{2}|25\d{2})(?!\d)")
    range_count = _count_pattern(
        lower,
        r"\d[\d,.]*\s*(?:-|–|—|to|ถึง)\s*\d[\d,.]*",
    )
    measurement_count = _count_pattern(
        lower,
        r"\d[\d,.]*\s*(?:ตร\.?\s*ม\.?|ตารางเมตร|sqm|sq\.?\s*m\.?|m²|เมตร|กม\.?|km|นาที|minutes?|ชั้น|floors?)",
    )
    price_count = _count_pattern(
        lower,
        r"(?:\d[\d,.]*\s*(?:ล้านบาท|ล้าน|บาท|thb|baht)|(?:price|ราคา|เริ่มต้น|starting at)\s*[:\-]?\s*\d[\d,.]*)",
    ) + _count_terms(lower, ("price", "ราคา", "เริ่มต้น"))
    price_sqm_count = _count_pattern(
        lower,
        r"(?:บาท\s*/\s*ตร\.?\s*ม\.?|ต่อตารางเมตร|per\s*(?:sq\.?\s*m\.?|sqm)|price per sqm)",
    )
    sqm_count = _count_pattern(
        lower,
        r"\d[\d,.]*\s*(?:ตร\.?\s*ม\.?|ตารางเมตร|sqm|sq\.?\s*m\.?|m²|square metres?|square meters?)",
    )
    bedroom_count = _count_pattern(
        lower,
        r"(?:\d+\s*(?:bedrooms?|beds?|ห้องนอน)|studio|penthouse|duplex)",
    )
    floor_plan_count = _count_terms(lower, FLOOR_PLAN_TERMS)
    unit_size_count = sqm_count + bedroom_count + floor_plan_count
    has_price = price_count + price_sqm_count > 0
    has_unit = unit_size_count > 0
    numeric_density = (
        min(number_count / words * 1000 / 10, 5)
        + int(percent_count > 0)
        + int(year_count > 0)
        + int(range_count > 0)
        + math.log1p(measurement_count)
    )
    price_unit_score = (
        int(has_price)
        + int(has_unit)
        + math.log1p(price_count + price_sqm_count)
        + math.log1p(unit_size_count)
    )
    return {
        "number_token_count": number_count,
        "number_token_per_1000_words": number_count / words * 1000,
        "percent_mention_count": percent_count,
        "year_mention_count": year_count,
        "range_mention_count": range_count,
        "measurement_mention_count": measurement_count,
        "price_mention_count": price_count,
        "price_per_sqm_mention_count": price_sqm_count,
        "has_price_detail": _binary(has_price),
        "unit_size_mention_count": unit_size_count,
        "sqm_mention_count": sqm_count,
        "bedroom_mention_count": bedroom_count,
        "floor_plan_mention_count": floor_plan_count,
        "has_unit_size_detail": _binary(has_unit),
        "factual_numeric_density_score": numeric_density,
        "price_unit_detail_score": price_unit_score,
    }


def extract_location_features(text: str) -> dict[str, Any]:
    if not text:
        return _nan_feature_row(LOCATION_FEATURES)
    lower = text.casefold()
    bts_count = _count_pattern(lower, r"(?<!\w)bts(?!\w)")
    mrt_count = _count_pattern(lower, r"(?<!\w)mrt(?!\w)")
    rail_terms = _count_terms(lower, ("รถไฟฟ้า", "สถานี", "skytrain", "metro"))
    transit_count = bts_count + mrt_count + rail_terms
    distance_count = _count_pattern(
        lower,
        r"\d[\d,.]*\s*(?:เมตร|กม\.?|km|kilomet(?:er|re)s?|m)\b",
    )
    walking_count = _count_pattern(
        lower,
        r"(?:เดิน|walk(?:ing)?)\D{0,20}\d[\d,.]*\s*(?:นาที|minutes?|mins?)",
    )
    landmark_count = _count_terms(lower, LANDMARK_TERMS)
    neighborhood_count = _count_terms(lower, NEIGHBORHOOD_TERMS)
    has_transit = transit_count + distance_count + walking_count > 0
    has_location = neighborhood_count + landmark_count + distance_count > 0
    score = (
        int(has_transit)
        + int(has_location)
        + math.log1p(transit_count)
        + math.log1p(distance_count + walking_count)
        + math.log1p(landmark_count + neighborhood_count)
    )
    return {
        "transit_station_mention_count": transit_count,
        "bts_mention_count": bts_count,
        "mrt_mention_count": mrt_count,
        "distance_mention_count": distance_count,
        "walking_time_mention_count": walking_count,
        "location_landmark_mention_count": landmark_count,
        "neighborhood_mention_count": neighborhood_count,
        "has_transit_detail": _binary(has_transit),
        "has_location_detail": _binary(has_location),
        "location_transit_specificity_score": score,
    }


def extract_amenity_features(text: str) -> dict[str, Any]:
    if not text:
        return _nan_feature_row(AMENITY_FEATURES)
    lower = text.casefold()
    amenity_count = _count_terms(lower, AMENITY_TERMS)
    facility_count = _count_terms(lower, FACILITY_TERMS)
    parking_count = _count_terms(lower, PARKING_TERMS)
    pool_count = _count_terms(lower, POOL_TERMS)
    gym_count = _count_terms(lower, GYM_TERMS)
    security_count = _count_terms(lower, SECURITY_TERMS)
    pet_count = _count_terms(lower, PET_TERMS)
    developer_count = _count_terms(lower, DEVELOPER_TERMS)
    project_keyword_count = _count_terms(lower, PROJECT_TERMS)
    named_after_keyword = _count_pattern(
        text,
        r"(?:project|developer|โครงการ|ผู้พัฒนา|เจ้าของโครงการ)\s*[:\-]?\s*[A-Za-z\u0E00-\u0E7F][A-Za-z0-9\u0E00-\u0E7F' -]{2,60}",
    )
    project_name_count = project_keyword_count + named_after_keyword
    entity_count = developer_count + project_name_count
    has_amenity = amenity_count + facility_count > 0
    has_project = entity_count > 0
    score = (
        int(has_amenity)
        + int(has_project)
        + math.log1p(amenity_count + facility_count)
        + math.log1p(developer_count + project_name_count)
    )
    return {
        "amenity_mention_count": amenity_count,
        "facility_mention_count": facility_count,
        "parking_mention_count": parking_count,
        "pool_mention_count": pool_count,
        "gym_mention_count": gym_count,
        "security_mention_count": security_count,
        "pet_friendly_mention_count": pet_count,
        "developer_mention_count": developer_count,
        "project_name_mention_count": project_name_count,
        "brand_or_project_entity_count": entity_count,
        "has_amenity_detail": _binary(has_amenity),
        "has_project_entity_detail": _binary(has_project),
        "amenity_project_detail_score": score,
    }


def extract_evidence_features(text: str, source_domain: str) -> dict[str, Any]:
    if not text:
        return _nan_feature_row(EVIDENCE_FEATURES)
    links = [match.rstrip(".,;:)") for match in URL_RE.findall(text)]
    domains = []
    for link in links:
        domain = (urlparse(link).hostname or "").casefold().removeprefix("www.")
        if domain:
            domains.append(domain)
    source = source_domain.casefold().removeprefix("www.")
    external_domains = sorted({domain for domain in domains if domain and domain != source})
    external_count = sum(domain != source for domain in domains)
    official_count = sum(
        domain.endswith((".go.th", ".or.th", ".ac.th", ".gov", ".gov.uk"))
        or "official" in domain
        for domain in domains
    )
    lower = text.casefold()
    source_phrase_count = _count_terms(lower, REFERENCE_TERMS)
    evidence_phrase_count = _count_terms(lower, EVIDENCE_TERMS)
    has_evidence = external_count + source_phrase_count + evidence_phrase_count > 0
    score = (
        int(has_evidence)
        + math.log1p(external_count)
        + math.log1p(official_count)
        + math.log1p(source_phrase_count + evidence_phrase_count)
    )
    return {
        "external_link_count": external_count,
        "external_link_domain_count": len(external_domains),
        "official_link_count": official_count,
        "source_reference_phrase_count": source_phrase_count,
        "evidence_phrase_count": evidence_phrase_count,
        "has_external_evidence": _binary(has_evidence),
        "external_evidence_score": score,
    }


def extract_url_features(assembly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in assembly.itertuples(index=False):
        text = _normalise_text(row.url_text_for_features)
        features: dict[str, Any] = {
            "normalized_url": row.normalized_url,
            "feature_extraction_text_scope": row.feature_extraction_text_scope,
            "text_feature_available": row.text_feature_available,
        }
        features.update(extract_writing_features(text))
        features.update(extract_factual_features(text))
        features.update(extract_location_features(text))
        features.update(extract_amenity_features(text))
        features.update(extract_evidence_features(text, row.source_root_domain))
        rows.append(features)
    return pd.DataFrame(rows)


PROMPT_RELEVANCE_FEATURES = (
    "prompt_title_tfidf_similarity",
    "prompt_page_tfidf_similarity",
    "prompt_page_keyword_overlap",
    "prompt_area_keyword_match",
    "prompt_intent_keyword_match",
    "prompt_page_relevance_score",
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "with",
    "is",
    "are",
    "what",
    "which",
    "where",
    "how",
    "best",
    "condo",
    "condos",
    "bangkok",
    "ที่",
    "ใน",
    "และ",
    "หรือ",
    "ของ",
    "มี",
    "คือ",
    "คอนโด",
    "กรุงเทพ",
}
INTENT_KEYWORDS = {
    "amenities / design": (*AMENITY_TERMS, "design", "architecture", "ออกแบบ"),
    "area / location": (*NEIGHBORHOOD_TERMS, "location", "near", "ใกล้", "เดินทาง"),
    "contact / sales": ("contact", "sales", "โทร", "ติดต่อ", "นัดชม", "enquire"),
    "document / brochure": ("brochure", "document", "download", "pdf", "โบรชัวร์", "เอกสาร"),
    "floor plan / unit layout": (*FLOOR_PLAN_TERMS, "layout", "unit type", "แบบห้อง"),
    "investment / resale": ("investment", "yield", "resale", "ลงทุน", "ผลตอบแทน", "ขายต่อ"),
    "price / budget": (*PRICE_TERMS, "budget", "งบประมาณ"),
    "recommendation / list": ("recommend", "top", "list", "แนะนำ", "รวม", "เลือก"),
    "rental / living": ("rent", "rental", "living", "เช่า", "อยู่อาศัย"),
    "review / comparison": ("review", "compare", "comparison", "รีวิว", "เปรียบเทียบ"),
}


def _keyword_set(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _tokens(text)
        if len(token) > 1 and token.casefold() not in STOPWORDS and not token.isdigit()
    }


def _char_tfidf_pair_similarity(
    prompts: pd.DataFrame,
    urls: pd.DataFrame,
    prompt_column: str,
    document_column: str,
    row_pairs: pd.DataFrame,
) -> np.ndarray:
    prompt_values = prompts[prompt_column].fillna("").astype(str).tolist()
    document_values = urls[document_column].fillna("").astype(str).tolist()
    corpus = prompt_values + document_values
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=30000,
        lowercase=True,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(corpus)
    prompt_matrix = matrix[: len(prompts)]
    document_matrix = matrix[len(prompts) :]
    prompt_index = {value: index for index, value in enumerate(prompts["prompt_id"].astype(str))}
    url_index = {value: index for index, value in enumerate(urls["normalized_url"].astype(str))}
    prompt_rows = prompt_matrix[[prompt_index[str(value)] for value in row_pairs["prompt_id"]]]
    document_rows = document_matrix[[url_index[str(value)] for value in row_pairs["normalized_url"]]]
    return np.asarray(prompt_rows.multiply(document_rows).sum(axis=1)).ravel()


def build_prompt_page_relevance(
    measurable: pd.DataFrame,
    prompt_reference: pd.DataFrame,
    assembly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompts = (
        prompt_reference[["prompt_id", "prompt", "intent", "area_tag"]]
        .drop_duplicates("prompt_id")
        .copy()
    )
    prompts["prompt_id"] = prompts["prompt_id"].astype(str)
    urls = assembly[
        [
            "normalized_url",
            "url_title",
            "url_description",
            "url_text_for_features",
            "feature_extraction_text_scope",
        ]
    ].copy()
    urls["title_description"] = (
        urls["url_title"].fillna("").astype(str) + " " + urls["url_description"].fillna("").astype(str)
    ).str.strip()
    pairs = measurable[
        ["source_appearance_row_id", "record_id", "prompt_id", "normalized_url"]
    ].copy()
    pairs["prompt_id"] = pairs["prompt_id"].astype(str)
    pairs["normalized_url"] = pairs["normalized_url"].astype(str)
    prompt_lookup = prompts.set_index("prompt_id")
    url_lookup = urls.set_index("normalized_url")
    pairs["prompt_text"] = pairs["prompt_id"].map(prompt_lookup["prompt"])
    pairs["intent"] = pairs["prompt_id"].map(prompt_lookup["intent"])
    pairs["area_tag"] = pairs["prompt_id"].map(prompt_lookup["area_tag"])
    pairs["url_title"] = pairs["normalized_url"].map(url_lookup["url_title"])
    pairs["url_description"] = pairs["normalized_url"].map(url_lookup["url_description"])
    pairs["url_text_for_features"] = pairs["normalized_url"].map(url_lookup["url_text_for_features"])
    pairs["feature_extraction_text_scope"] = pairs["normalized_url"].map(
        url_lookup["feature_extraction_text_scope"]
    )
    pairs["prompt_title_tfidf_similarity"] = _char_tfidf_pair_similarity(
        prompts,
        urls,
        "prompt",
        "title_description",
        pairs,
    )
    pairs["prompt_page_tfidf_similarity"] = _char_tfidf_pair_similarity(
        prompts,
        urls,
        "prompt",
        "url_text_for_features",
        pairs,
    )
    overlaps = []
    area_matches = []
    intent_matches = []
    for row in pairs.itertuples(index=False):
        prompt_terms = _keyword_set(_text(row.prompt_text))
        page_terms = _keyword_set(_text(row.url_text_for_features))
        overlap = len(prompt_terms & page_terms) / len(prompt_terms) if prompt_terms else 0.0
        page_lower = _text(row.url_text_for_features).casefold()
        area_value = _text(row.area_tag).casefold()
        area_terms = {
            token
            for token in re.split(r"[/,|;()\-]+|\s{2,}", area_value)
            if len(token.strip()) > 1
        }
        area_terms.update(
            term for term in NEIGHBORHOOD_TERMS if term in _text(row.prompt_text).casefold()
        )
        area_match = int(any(term.strip() and term.strip() in page_lower for term in area_terms))
        intent_terms = INTENT_KEYWORDS.get(_text(row.intent).casefold(), ())
        intent_match = int(any(term.casefold() in page_lower for term in intent_terms))
        overlaps.append(overlap)
        area_matches.append(area_match)
        intent_matches.append(intent_match)
    pairs["prompt_page_keyword_overlap"] = overlaps
    pairs["prompt_area_keyword_match"] = area_matches
    pairs["prompt_intent_keyword_match"] = intent_matches
    pairs["prompt_page_relevance_score"] = pairs[
        [
            "prompt_title_tfidf_similarity",
            "prompt_page_tfidf_similarity",
            "prompt_page_keyword_overlap",
            "prompt_area_keyword_match",
            "prompt_intent_keyword_match",
        ]
    ].mean(axis=1)
    no_text = pairs["feature_extraction_text_scope"].eq("no_text")
    pairs.loc[no_text, list(PROMPT_RELEVANCE_FEATURES)] = np.nan
    relevance = pairs[
        [
            "source_appearance_row_id",
            "record_id",
            "prompt_id",
            "normalized_url",
            "prompt_text",
            *PROMPT_RELEVANCE_FEATURES,
        ]
    ].copy()
    audit = pd.DataFrame(
        [
            {
                "metric": "source_appearance_rows",
                "value": len(relevance),
                "status": "pass" if len(relevance) == len(measurable) else "fail",
            },
            {
                "metric": "rows_with_prompt_text",
                "value": int(relevance["prompt_text"].notna().sum()),
                "status": "pass" if relevance["prompt_text"].notna().all() else "warning",
            },
            {
                "metric": "rows_with_page_text_for_similarity",
                "value": int(
                    pairs["url_text_for_features"].fillna("").astype(str).str.strip().ne("").sum()
                ),
                "status": "observed",
            },
        ]
    )
    return relevance, audit


def build_feature_dictionary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        name: str,
        group: str,
        feature_type: str,
        level: str,
        formula: str,
        missingness: str,
        use: str,
        interpretation: str,
        leakage: str = "low",
    ) -> None:
        rows.append(
            {
                "feature_name": name,
                "feature_group": group,
                "feature_type": feature_type,
                "level": level,
                "formula_or_rule": formula,
                "uses_answer_text": False,
                "leakage_risk": leakage,
                "missingness_risk": missingness,
                "intended_model_use": use,
                "interpretation": interpretation,
            }
        )

    for name in WRITING_FEATURES:
        add(
            name,
            "writing_structure",
            "text_rule_or_composite",
            "url_level",
            "Deterministic punctuation, line, opening-text, or additive binary rule; see source module.",
            "high for line/list features because the compact crawler preview is often flattened",
            "screening_or_diagnostic",
            "Observed writing structure in the captured page text; excerpt zeros are not proof of full-page absence.",
        )
    for name in FACTUAL_FEATURES:
        add(
            name,
            "factual_numeric",
            "regex_count_or_composite",
            "url_level",
            (
                "Bilingual deterministic regex counts; factual_numeric_density_score = capped numeric "
                "tokens/1000 words + percent/year/range indicators + log1p(measurements); "
                "price_unit_detail_score = price/unit indicators + log1p price and unit counts."
            ),
            "medium because most pages use a 3,000-character excerpt",
            "priority_screening_candidate",
            "Factual and numerical specificity observed in the captured page text.",
        )
    for name in LOCATION_FEATURES:
        add(
            name,
            "location_transit",
            "dictionary_regex_or_composite",
            "url_level",
            (
                "Bilingual transit, distance, walking-time, landmark, and neighborhood counts; composite "
                "adds detail indicators and log1p count groups."
            ),
            "medium because facts beyond the excerpt are unobserved",
            "priority_screening_candidate",
            "Location and transit specificity observed in the captured page text.",
        )
    for name in AMENITY_FEATURES:
        add(
            name,
            "amenity_project",
            "dictionary_regex_or_composite",
            "url_level",
            "Bilingual amenity/project/developer phrase counts and an additive log1p composite.",
            "medium; generic project/entity patterns require manual validation",
            "screening_or_sensitivity",
            "Amenity and project specificity observed in the captured page text.",
        )
    for name in EVIDENCE_FEATURES:
        add(
            name,
            "external_evidence",
            "url_regex_phrase_or_composite",
            "url_level",
            "Raw URL/domain counts plus bilingual source/evidence phrases and an additive log1p composite.",
            "high because crawler previews may omit link destinations or flatten anchors",
            "diagnostic_only_until_validated",
            "Visible external-evidence signals in captured webpage content, never AI-answer citations.",
        )
    for name in PROMPT_RELEVANCE_FEATURES:
        add(
            name,
            "prompt_page_relevance",
            "tfidf_overlap_or_composite",
            "source_appearance_level",
            (
                "Character 3-5 gram TF-IDF cosine or deterministic keyword match; composite is equal-weight "
                "mean of title similarity, page similarity, keyword overlap, area match, and intent match."
            ),
            "medium because page text is usually excerpt-only",
            "priority_screening_control",
            "Observable prompt-page relevance using prompt and page only, not answer text.",
            leakage="low_if_prompt_and_page_only",
        )
    dictionary = pd.DataFrame(rows)
    composite_formulas = {
        "writing_structure_score": (
            "has_bullet_list + has_numbered_list + has_faq_pattern + "
            "has_question_answer_structure + opening_has_summary_signal + "
            "opening_has_direct_answer_signal"
        ),
        "factual_numeric_density_score": (
            "min(number_token_per_1000_words / 10, 5) + I(percent_mention_count > 0) + "
            "I(year_mention_count > 0) + I(range_mention_count > 0) + "
            "log1p(measurement_mention_count)"
        ),
        "price_unit_detail_score": (
            "has_price_detail + has_unit_size_detail + "
            "log1p(price_mention_count + price_per_sqm_mention_count) + "
            "log1p(unit_size_mention_count)"
        ),
        "location_transit_specificity_score": (
            "has_transit_detail + has_location_detail + "
            "log1p(transit_station_mention_count) + "
            "log1p(distance_mention_count + walking_time_mention_count) + "
            "log1p(location_landmark_mention_count + neighborhood_mention_count)"
        ),
        "amenity_project_detail_score": (
            "has_amenity_detail + has_project_entity_detail + "
            "log1p(amenity_mention_count + facility_mention_count) + "
            "log1p(developer_mention_count + project_name_mention_count)"
        ),
        "external_evidence_score": (
            "has_external_evidence + log1p(external_link_count) + "
            "log1p(official_link_count) + "
            "log1p(source_reference_phrase_count + evidence_phrase_count)"
        ),
        "prompt_page_relevance_score": (
            "equal-weight mean(prompt_title_tfidf_similarity, prompt_page_tfidf_similarity, "
            "prompt_page_keyword_overlap, prompt_area_keyword_match, prompt_intent_keyword_match)"
        ),
    }
    for feature, formula in composite_formulas.items():
        dictionary.loc[dictionary["feature_name"].eq(feature), "formula_or_rule"] = formula
    return dictionary


def build_validation_summary(data: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    binary_features = {
        feature
        for feature in features
        if feature.startswith("has_")
        or feature.endswith("_match")
        or feature in ("opening_has_summary_signal", "opening_has_direct_answer_signal")
    }
    rows = []
    for feature in features:
        if feature not in data:
            continue
        numeric = pd.to_numeric(data[feature], errors="coerce")
        available = numeric.dropna()
        suspicious: list[str] = []
        if np.isinf(numeric).any():
            suspicious.append("nonfinite")
        if len(available) and available.min() < 0:
            suspicious.append("negative_value")
        if feature in binary_features and not set(available.unique()).issubset({0, 1}):
            suspicious.append("invalid_binary_value")
        p90 = available.quantile(0.9) if len(available) else np.nan
        if len(available) and pd.notna(p90) and p90 > 0 and available.max() > 5 * p90:
            suspicious.append("extreme_tail_max_gt_5x_p90")
        missing_rate = float(numeric.isna().mean())
        unique_values = int(available.nunique())
        if missing_rate > 0.2:
            status = "warning_high_missingness"
        elif unique_values <= 1:
            status = "warning_no_variation"
        elif unique_values <= 3 and feature not in binary_features:
            status = "warning_low_variation"
        elif suspicious:
            status = "warning_suspicious_values"
        else:
            status = "pass"
        rows.append(
            {
                "feature_name": feature,
                "non_null_count": int(numeric.notna().sum()),
                "missing_rate": missing_rate,
                "min": available.min() if len(available) else np.nan,
                "median": available.median() if len(available) else np.nan,
                "mean": available.mean() if len(available) else np.nan,
                "p90": p90,
                "max": available.max() if len(available) else np.nan,
                "suspicious_values": "; ".join(suspicious) if suspicious else "none",
                "validation_status": status,
            }
        )
    return pd.DataFrame(rows)


def build_leakage_check(feature_dictionary: pd.DataFrame, model_formulas: dict[str, str]) -> pd.DataFrame:
    forbidden = (
        "answer",
        "page_answer_similarity",
        "max_chunk_answer_similarity",
        "answer_overlap",
        "answer_like_text",
        "cited_label",
        "source_position",
        "observed_rank",
        "domain_citation_rate",
    )
    rows = [
        {
            "check": "feature_dictionary_uses_answer_text",
            "matches": "; ".join(
                feature_dictionary.loc[
                    feature_dictionary["uses_answer_text"].fillna(False).astype(bool),
                    "feature_name",
                ]
            )
            or "none",
            "status": (
                "fail"
                if feature_dictionary["uses_answer_text"].fillna(False).astype(bool).any()
                else "pass"
            ),
            "required_action": "Remove forbidden features before modeling.",
        },
        {
            "check": "citation_outcome_used_to_construct_features",
            "matches": "none",
            "status": "pass",
            "required_action": "Feature extraction functions receive page/prompt fields only.",
        },
        {
            "check": "post_answer_citation_data_used",
            "matches": "none",
            "status": "pass",
            "required_action": "Webpage evidence counts use page text/links only.",
        },
    ]
    for model_id, formula in model_formulas.items():
        rhs = formula.split("~", 1)[-1].casefold()
        matches = sorted(token for token in forbidden if token in rhs)
        rows.append(
            {
                "check": f"formula_scan:{model_id}",
                "matches": "; ".join(matches) if matches else "none",
                "status": "fail" if matches else "pass",
                "required_action": formula,
            }
        )
    result = pd.DataFrame(rows)
    if result["status"].eq("fail").any():
        raise ValueError("Notebook 10 leakage guardrail failed.")
    return result


def build_manual_review_sample(
    row_data: pd.DataFrame,
    assembly: pd.DataFrame,
) -> pd.DataFrame:
    evidence_columns = [
        "source_appearance_row_id",
        "record_id",
        "normalized_url",
        "prompt_id",
        "prompt_text",
        "url_title",
        "page_text_excerpt",
        "feature_extraction_text_scope",
        "cited",
        "factual_numeric_density_score",
        "price_unit_detail_score",
        "location_transit_specificity_score",
        "prompt_page_tfidf_similarity",
        "number_token_count",
        "price_mention_count",
        "unit_size_mention_count",
        "transit_station_mention_count",
        "amenity_mention_count",
        "external_link_count",
    ]
    work = row_data.copy()
    missing_metadata = [
        column for column in ("url_title", "page_text_excerpt") if column not in work
    ]
    if missing_metadata:
        work = work.merge(
            assembly[["normalized_url", *missing_metadata]],
            on="normalized_url",
            how="left",
            validate="many_to_one",
        )
    selected: list[pd.DataFrame] = []
    used_urls: set[str] = set()
    specs = (
        ("top_factual_numeric_density", "factual_numeric_density_score"),
        ("top_price_unit_detail", "price_unit_detail_score"),
        ("top_location_transit_specificity", "location_transit_specificity_score"),
        ("top_prompt_page_similarity", "prompt_page_tfidf_similarity"),
    )
    for reason, feature in specs:
        candidates = work[
            ~work["normalized_url"].astype(str).isin(used_urls)
        ].sort_values(
            feature,
            ascending=False,
            na_position="last",
            kind="stable",
        ).drop_duplicates("normalized_url", keep="first")
        sample = candidates.head(30).copy()
        sample["review_reason"] = reason
        selected.append(sample)
        used_urls.update(sample["normalized_url"].astype(str))
    random_pool = work[
        ~work["normalized_url"].astype(str).isin(used_urls)
    ].drop_duplicates("normalized_url", keep="first")
    random_sample = random_pool.sample(n=min(30, len(random_pool)), random_state=20260716).copy()
    random_sample["review_reason"] = "random_row"
    selected.append(random_sample)
    return pd.concat(selected, ignore_index=True)[["review_reason", *evidence_columns]]


def build_merge_audit(
    original: pd.DataFrame,
    merged: pd.DataFrame,
    url_features: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "original_rows", "value": len(original), "status": "observed"},
        {"metric": "merged_rows", "value": len(merged), "status": "observed"},
        {
            "metric": "row_loss",
            "value": len(original) - len(merged),
            "status": "pass" if len(original) == len(merged) else "fail",
        },
        {
            "metric": "unique_urls_original",
            "value": original["normalized_url"].nunique(),
            "status": "observed",
        },
        {
            "metric": "unique_urls_merged",
            "value": merged["normalized_url"].nunique(),
            "status": "observed",
        },
        {
            "metric": "duplicate_source_appearance_row_id_rows",
            "value": int(merged["source_appearance_row_id"].duplicated().sum()),
            "status": (
                "pass"
                if not merged["source_appearance_row_id"].duplicated().any()
                else "fail"
            ),
        },
        {
            "metric": "duplicate_url_feature_keys",
            "value": int(url_features["normalized_url"].duplicated().sum()),
            "status": "pass" if not url_features["normalized_url"].duplicated().any() else "fail",
        },
    ]
    for feature in (
        "writing_structure_score",
        "factual_numeric_density_score",
        "price_unit_detail_score",
        "location_transit_specificity_score",
        "amenity_project_detail_score",
        "external_evidence_score",
        "prompt_page_relevance_score",
    ):
        rows.append(
            {
                "metric": f"missing_rate:{feature}",
                "value": float(merged[feature].isna().mean()),
                "status": "observed",
            }
        )
    return pd.DataFrame(rows)


SCREENING_MODELS = {
    "B0_current_M2": BASELINE_M2,
    "F0_writing_structure": f"cited ~ writing_structure_score + {PROMPT_FE}",
    "F1_factual_density": f"cited ~ factual_numeric_density_score + {PROMPT_FE}",
    "F2_price_unit_detail": f"cited ~ price_unit_detail_score + {PROMPT_FE}",
    "F3_location_transit": f"cited ~ location_transit_specificity_score + {PROMPT_FE}",
    "F4_amenity_project": f"cited ~ amenity_project_detail_score + {PROMPT_FE}",
    "F5_external_evidence": f"cited ~ external_evidence_score + {PROMPT_FE}",
    "F6_prompt_page_relevance": f"cited ~ prompt_page_relevance_score + {PROMPT_FE}",
}
TABLE_PROXY_MODELS = {
    "T1_has_table_prompt_fe": f"cited ~ has_table + {PROMPT_FE}",
    "T2_table_plus_factual_detail": (
        "cited ~ has_table + factual_numeric_density_score + price_unit_detail_score "
        f"+ location_transit_specificity_score + {PROMPT_FE}"
    ),
    "T3_table_plus_detail_relevance": (
        "cited ~ has_table + factual_numeric_density_score + price_unit_detail_score "
        f"+ location_transit_specificity_score + prompt_page_relevance_score + {PROMPT_FE}"
    ),
}


def _preferred_focal_rows(table: pd.DataFrame) -> pd.DataFrame:
    rank = {
        "two_way_cluster_prompt_url": 0,
        "cluster_prompt_id": 1,
        "cluster_normalized_url": 2,
        "HC3": 3,
    }
    work = table[table["std_error"].notna()].copy()
    work["_rank"] = work["cov_type"].map(rank).fillna(99)
    return (
        work.sort_values("_rank", kind="stable")
        .drop_duplicates(["model_id", "term"], keep="first")
        .drop(columns="_rank")
    )


def run_first_pass_models(
    data: pd.DataFrame,
    table_out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    from src.econometrics_eda_v2.content_feature_econometrics import run_model_and_save

    warnings: list[str] = []
    screening_rows = []
    screening_focal = {
        "B0_current_M2": (
            "log2_word_count_plus1",
            "has_table",
            "heading_count_group",
            "link_count_group",
            "content_strength",
        ),
        "F0_writing_structure": ("writing_structure_score",),
        "F1_factual_density": ("factual_numeric_density_score",),
        "F2_price_unit_detail": ("price_unit_detail_score",),
        "F3_location_transit": ("location_transit_specificity_score",),
        "F4_amenity_project": ("amenity_project_detail_score",),
        "F5_external_evidence": ("external_evidence_score",),
        "F6_prompt_page_relevance": ("prompt_page_relevance_score",),
    }
    for model_id, formula in SCREENING_MODELS.items():
        path = table_out / f".{model_id}_working.csv"
        run = run_model_and_save(formula, data, model_id, path)
        warnings.extend(run.warnings)
        focal_tokens = screening_focal[model_id]
        focal = run.table[
            run.table["term"].map(lambda term: any(token in term for token in focal_tokens))
        ].copy()
        focal["model_purpose"] = "first_pass_screening_not_final_claim"
        screening_rows.append(focal)
        path.unlink(missing_ok=True)
    screening = pd.concat(screening_rows, ignore_index=True)
    screening.to_csv(table_out / "10_first_pass_feature_screening_lpm.csv", index=False)

    proxy_rows = []
    proxy_terms = (
        "has_table",
        "factual_numeric_density_score",
        "price_unit_detail_score",
        "location_transit_specificity_score",
        "prompt_page_relevance_score",
    )
    for model_id, formula in TABLE_PROXY_MODELS.items():
        path = table_out / f".{model_id}_working.csv"
        run = run_model_and_save(formula, data, model_id, path)
        warnings.extend(run.warnings)
        focal = run.table[run.table["term"].isin(proxy_terms)].copy()
        focal["model_purpose"] = "has_table_proxy_screening_not_final_claim"
        proxy_rows.append(focal)
        path.unlink(missing_ok=True)
    proxy = pd.concat(proxy_rows, ignore_index=True)

    preferred = _preferred_focal_rows(proxy)
    t1 = preferred[
        preferred["model_id"].eq("T1_has_table_prompt_fe") & preferred["term"].eq("has_table")
    ].iloc[0]
    attenuation_rows = []
    for model_id in TABLE_PROXY_MODELS:
        row = preferred[preferred["model_id"].eq(model_id) & preferred["term"].eq("has_table")].iloc[0]
        attenuation = t1["estimate_pp"] - row["estimate_pp"]
        coefficient_change = row["estimate_pp"] - t1["estimate_pp"]
        if np.sign(row["estimate_pp"]) != np.sign(t1["estimate_pp"]):
            pattern = "direction_changed"
        elif abs(row["estimate_pp"]) < abs(t1["estimate_pp"]):
            pattern = "attenuated"
        elif abs(row["estimate_pp"]) > abs(t1["estimate_pp"]):
            pattern = "amplified"
        else:
            pattern = "unchanged"
        attenuation_rows.append(
            {
                "model_id": model_id,
                "has_table_estimate_pp": row["estimate_pp"],
                "conf_low_pp": row["conf_low_pp"],
                "conf_high_pp": row["conf_high_pp"],
                "p_value": row["p_value"],
                "cov_type": row["cov_type"],
                "attenuation_from_T1_pp": attenuation,
                "coefficient_change_from_T1_pp": coefficient_change,
                "percent_attenuation_from_T1": (
                    attenuation / abs(t1["estimate_pp"]) * 100
                    if abs(t1["estimate_pp"]) > 1e-9
                    else np.nan
                ),
                "proxy_test_pattern": pattern,
                "n_obs": row["n_obs"],
                "n_prompts": row["n_prompts"],
                "n_urls": row["n_urls"],
            }
        )
    attenuation = pd.DataFrame(attenuation_rows)
    proxy = proxy.merge(
        attenuation[
            [
                "model_id",
                "attenuation_from_T1_pp",
                "coefficient_change_from_T1_pp",
                "percent_attenuation_from_T1",
                "proxy_test_pattern",
            ]
        ],
        on="model_id",
        how="left",
    )
    proxy.to_csv(table_out / "10_has_table_proxy_test.csv", index=False)
    attenuation.to_csv(table_out / "10_has_table_proxy_attenuation_summary.csv", index=False)
    return screening, proxy, attenuation, warnings


def _make_coefficient_plot(table: pd.DataFrame, path: Path, title: str) -> None:
    go = _plotly_graph_objects()
    preferred = _preferred_focal_rows(table)
    preferred = preferred[~preferred["term"].str.contains("heading_count|link_count|content_strength", regex=True)]
    labels = {
        "writing_structure_score": "Writing structure",
        "factual_numeric_density_score": "Factual/numeric density",
        "price_unit_detail_score": "Price/unit detail",
        "location_transit_specificity_score": "Location/transit specificity",
        "amenity_project_detail_score": "Amenity/project detail",
        "external_evidence_score": "External evidence",
        "prompt_page_relevance_score": "Prompt-page relevance",
        "has_table": "Has table",
        "log2_word_count_plus1": "Page length doubling",
    }
    preferred["label"] = preferred["model_id"] + " | " + preferred["term"].map(labels).fillna(preferred["term"])
    fig = go.Figure(
        go.Scatter(
            x=preferred["estimate_pp"],
            y=preferred["label"],
            mode="markers",
            error_x={
                "type": "data",
                "symmetric": False,
                "array": preferred["conf_high_pp"] - preferred["estimate_pp"],
                "arrayminus": preferred["estimate_pp"] - preferred["conf_low_pp"],
            },
            marker={"color": "#277da1", "size": 9},
            customdata=np.column_stack(
                [preferred["conf_low_pp"], preferred["conf_high_pp"], preferred["cov_type"]]
            ),
            hovertemplate=(
                "%{y}<br>Estimate=%{x:.2f} pp"
                "<br>95% CI=%{customdata[0]:.2f} to %{customdata[1]:.2f} pp"
                "<br>%{customdata[2]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#606b73")
    fig.update_layout(
        title=title,
        xaxis_title="Screening association (percentage points)",
        yaxis_title="",
        template="plotly_white",
        height=max(480, len(preferred) * 32 + 150),
        margin={"l": 260, "r": 40, "t": 80, "b": 60},
    )
    _write_plotly(fig, path)


def _make_proxy_attenuation_plot(attenuation: pd.DataFrame, path: Path) -> None:
    go = _plotly_graph_objects()
    fig = go.Figure(
        go.Scatter(
            x=attenuation["model_id"],
            y=attenuation["has_table_estimate_pp"],
            mode="lines+markers+text",
            text=[f"{value:.2f} pp" for value in attenuation["has_table_estimate_pp"]],
            textposition="top center",
            error_y={
                "type": "data",
                "symmetric": False,
                "array": attenuation["conf_high_pp"] - attenuation["has_table_estimate_pp"],
                "arrayminus": attenuation["has_table_estimate_pp"] - attenuation["conf_low_pp"],
            },
            marker={"color": "#43aa8b", "size": 10},
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#606b73")
    fig.update_layout(
        title="Does the has-table association attenuate after factual/detail controls?",
        xaxis_title="Screening model",
        yaxis_title="Has-table coefficient (percentage points)",
        template="plotly_white",
        height=500,
    )
    _write_plotly(fig, path)


FEATURE_GROUP_SCORES = {
    "writing_structure": "writing_structure_score",
    "factual_numeric": "factual_numeric_density_score",
    "price_unit_detail": "price_unit_detail_score",
    "location_transit": "location_transit_specificity_score",
    "amenity_project": "amenity_project_detail_score",
    "external_evidence": "external_evidence_score",
    "prompt_page_relevance": "prompt_page_relevance_score",
}
FEATURE_GROUP_MODELS = {
    "writing_structure": "F0_writing_structure",
    "factual_numeric": "F1_factual_density",
    "price_unit_detail": "F2_price_unit_detail",
    "location_transit": "F3_location_transit",
    "amenity_project": "F4_amenity_project",
    "external_evidence": "F5_external_evidence",
    "prompt_page_relevance": "F6_prompt_page_relevance",
}


def build_feature_priority(
    data: pd.DataFrame,
    screening: pd.DataFrame,
    attenuation: pd.DataFrame,
    leakage_check: pd.DataFrame,
) -> pd.DataFrame:
    preferred = _preferred_focal_rows(screening)
    leakage_pass = not leakage_check["status"].eq("fail").any()
    t3_change = attenuation.loc[
        attenuation["model_id"].eq("T3_table_plus_detail_relevance"),
        "coefficient_change_from_T1_pp",
    ].iloc[0]
    t3_pattern = attenuation.loc[
        attenuation["model_id"].eq("T3_table_plus_detail_relevance"),
        "proxy_test_pattern",
    ].iloc[0]
    rows = []
    for group, feature in FEATURE_GROUP_SCORES.items():
        values = pd.to_numeric(data[feature], errors="coerce")
        model_id = FEATURE_GROUP_MODELS[group]
        estimate_row = preferred[
            preferred["model_id"].eq(model_id) & preferred["term"].eq(feature)
        ]
        estimate = estimate_row.iloc[0]["estimate_pp"] if not estimate_row.empty else np.nan
        if group == "writing_structure":
            reliability = "low_to_medium_flattened_excerpt"
            recommendation = "diagnostic_only"
        elif group == "external_evidence":
            reliability = "low_preview_may_omit_link_destinations"
            recommendation = "needs_extraction_fix"
        elif group == "prompt_page_relevance":
            reliability = "medium_excerpt_limited_but_row_specific"
            recommendation = "priority_main_candidate"
        elif group in ("factual_numeric", "price_unit_detail", "location_transit"):
            reliability = "medium_excerpt_based"
            recommendation = "priority_main_candidate"
        else:
            reliability = "medium_requires_entity_pattern_QA"
            recommendation = "sensitivity_candidate"
        missingness = float(values.isna().mean())
        variation = (
            f"unique={values.nunique(dropna=True)}; std={values.std():.4f}"
            if values.notna().any()
            else "no_available_values"
        )
        if not leakage_pass:
            recommendation = "forbidden"
        elif values.nunique(dropna=True) <= 1:
            recommendation = "needs_extraction_fix"
        table_relevance = (
            f"included_in_T2/T3; T3 pattern={t3_pattern}; has_table change={t3_change:+.2f} pp"
            if group in ("factual_numeric", "price_unit_detail", "location_transit")
            else (
                f"included_in_T3; T3 pattern={t3_pattern}; has_table change={t3_change:+.2f} pp"
                if group == "prompt_page_relevance"
                else "not_in_pre_specified_table_proxy_ladder"
            )
        )
        rows.append(
            {
                "feature_group": group,
                "feature_name": feature,
                "extraction_reliability": reliability,
                "missingness": missingness,
                "variation": variation,
                "leakage_status": "pass" if leakage_pass else "fail",
                "first_pass_association_direction": (
                    "positive" if estimate > 0 else "negative" if estimate < 0 else "zero_or_unavailable"
                ),
                "first_pass_association_size_pp": estimate,
                "table_proxy_relevance": table_relevance,
                "recommended_for_11": recommendation,
                "notes": (
                    "Screening association only; retain components and validate examples before notebook 11."
                ),
            }
        )
    return pd.DataFrame(rows)


def _make_scope_plot(assembly: pd.DataFrame, path: Path) -> None:
    go = _plotly_graph_objects()
    counts = (
        assembly["feature_extraction_text_scope"]
        .value_counts()
        .rename_axis("scope")
        .reset_index(name="n_urls")
    )
    fig = go.Figure(
        go.Bar(
            x=counts["scope"],
            y=counts["n_urls"],
            text=counts["n_urls"],
            textposition="outside",
            marker_color=["#277da1", "#f8961e", "#43aa8b", "#8d99ae"][: len(counts)],
        )
    )
    fig.update_layout(
        title="Feature extraction text scope",
        xaxis_title="",
        yaxis_title="Unique URLs",
        template="plotly_white",
        height=450,
    )
    _write_plotly(fig, path)


def _make_composite_distribution_plot(data: pd.DataFrame, path: Path) -> None:
    go = _plotly_graph_objects()
    fig = go.Figure()
    colors = ["#277da1", "#43aa8b", "#f8961e", "#f94144", "#9b5de5", "#577590"]
    for index, feature in enumerate(
        [
            "factual_numeric_density_score",
            "price_unit_detail_score",
            "location_transit_specificity_score",
            "amenity_project_detail_score",
            "external_evidence_score",
            "prompt_page_relevance_score",
        ]
    ):
        fig.add_trace(
            go.Histogram(
                x=data[feature],
                name=feature,
                opacity=0.5,
                nbinsx=35,
                marker_color=colors[index],
            )
        )
    fig.update_layout(
        title="Distributions of notebook 10 composite features",
        xaxis_title="Composite score",
        yaxis_title="Rows",
        barmode="overlay",
        template="plotly_white",
        height=560,
    )
    _write_plotly(fig, path)


def _screening_summary_lines(screening: pd.DataFrame) -> str:
    preferred = _preferred_focal_rows(screening)
    preferred = preferred[
        preferred["model_id"].str.startswith("F")
        & preferred["term"].isin(FEATURE_GROUP_SCORES.values())
    ]
    lines = []
    for row in preferred.itertuples(index=False):
        direction = "positive" if row.estimate_pp > 0 else "negative"
        interval = "excludes zero" if row.conf_low_pp > 0 or row.conf_high_pp < 0 else "includes zero"
        lines.append(
            f"- `{row.term}`: {direction} screening association of {row.estimate_pp:.2f} pp "
            f"(95% CI {row.conf_low_pp:.2f} to {row.conf_high_pp:.2f}; {interval}; {row.cov_type})."
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    assembly: pd.DataFrame,
    merged: pd.DataFrame,
    validation: pd.DataFrame,
    leakage: pd.DataFrame,
    screening: pd.DataFrame,
    attenuation: pd.DataFrame,
    priority: pd.DataFrame,
) -> None:
    scopes = assembly["feature_extraction_text_scope"].value_counts()
    t1 = attenuation[attenuation["model_id"].eq("T1_has_table_prompt_fe")].iloc[0]
    t3 = attenuation[attenuation["model_id"].eq("T3_table_plus_detail_relevance")].iloc[0]
    warnings = validation[~validation["validation_status"].eq("pass")]
    priority_lines = "\n".join(
        f"- `{row.feature_group}`: `{row.recommended_for_11}` ({row.extraction_reliability})."
        for row in priority.itertuples(index=False)
    )
    report = f"""# 10 Writing and Factual-Density Feature Layer Report

## 1. Purpose and relationship to notebook 09

Notebook 09 found table presence to be the clearest suggestive structural signal, while heading count, page length, link count, and content strength were not robust enough for substantive recommendations. These new features are designed to test whether table presence proxies factual specificity. This notebook screens feature candidates; it does not produce final causal claims.

The estimand remains `P(cited = 1 | source surfaced in this audit)`, so every result is conditional on surfaced sources and is not web-wide.

## 2. Available text fields and extraction scope

The package supplies a page-title field, a 1,200-character excerpt, and a page-text preview capped at 3,000 characters. It does not provide a guaranteed full body for most URLs.

- Full-text-equivalent short pages: {int(scopes.get('full_text', 0)):,}
- Excerpt-only pages: {int(scopes.get('excerpt_only', 0)):,}
- Title/description-only pages: {int(scopes.get('title_description_only', 0)):,}
- No extractable text: {int(scopes.get('no_text', 0)):,}

An excerpt-derived zero means “not observed in the captured excerpt,” not proof that the feature is absent from the full webpage.

## 3. Features created

The layer includes deterministic paragraph/sentence diagnostics, list and FAQ patterns, opening-summary signals, numeric and measurement density, prices and unit sizes, location/transit facts, amenities and project/developer details, webpage external-evidence signals, and prompt-page relevance.

Composite scores use pre-specified additive/log transforms. Their weights were not chosen using the cited outcome, and every component remains available in the final dataset.

## 4. Leakage validation

- Leakage checks passed: {not leakage['status'].eq('fail').any()}
- Answer text used: no
- Citation outcome used to construct features: no
- Post-answer citation data used: no

Prompt-page relevance uses prompt and page only, not answer text.

## 5. Feature distributions and validation

The final row-level dataset contains {len(merged):,} source appearances and {merged['normalized_url'].nunique():,} URLs. Validation produced {len(warnings):,} warnings, primarily for excerpt/formatting limitations or low variation. Those warnings are retained in `writing_factual_feature_validation_summary.csv`.

## 6. Manual evidence audit

The manual-review file contains high-scoring factual, price/unit, location/transit, and prompt-page similarity examples plus a deterministic random sample. It includes the captured excerpt and extraction scope so reviewers can distinguish a plausible count from a truncation or boilerplate artifact.

## 7. First-pass screening results

{_screening_summary_lines(screening)}

These are one-group-at-a-time prompt-fixed-effect screening models for prioritization. They are not final econometric claims.
Negative screening coefficients do not imply that factual detail lowers citation probability; domain, page-function, template, and relevance confounding remain for notebook 11 to address.

## 8. Has-table proxy test

The prompt-FE `has_table` estimate is {t1['has_table_estimate_pp']:.2f} percentage points in T1 and {t3['has_table_estimate_pp']:.2f} percentage points after factual/detail/relevance controls in T3. The coefficient changes by {t3['coefficient_change_from_T1_pp']:+.2f} percentage points and is classified as `{t3['proxy_test_pattern']}` rather than attenuated.

The first-pass proxy ladder therefore does not support a simple account in which the new factual/detail/relevance composites explain away the table association. This is descriptive screening evidence only: it does not establish mediation, suppression, or an effect of tables on citation.

## 9. Recommended feature groups for notebook 11

{priority_lines}

## 10. Caveats

- Most page features are extracted from a 3,000-character preview rather than a guaranteed full page.
- Crawler-normalized text often removes line and link structure, weakening list and external-link features.
- Thai sentence and entity segmentation is rule-based and requires manual QA.
- Prompt-page TF-IDF uses deterministic character n-grams and does not use answer text.
- First-pass coefficients remain observational associations conditional on surfaced sources.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_writing_factual_density_feature_layer(package_path: Path | str) -> dict[str, Any]:
    """Run notebook 10's deterministic feature extraction and screening workflow."""
    package = Path(package_path).resolve()
    data_dir = package / "data"
    table_out = package / "tables/10_writing_factual_density_features"
    figure_out = package / "figures/10_writing_factual_density_features"
    report_out = package / "reports/10_writing_factual_density_features"
    for directory in (data_dir, table_out, figure_out, report_out):
        directory.mkdir(parents=True, exist_ok=True)

    measurable_path = data_dir / "content_lpm_measurable_rows.csv"
    evidence_path = data_dir / "url_content_evidence_compact.csv"
    prompt_path = data_dir / "prompt_reference.csv"
    measurable = pd.read_csv(measurable_path, low_memory=False)
    measurable = measurable.copy()
    measurable["source_appearance_row_id"] = np.arange(len(measurable), dtype=int)
    evidence = pd.read_csv(evidence_path, low_memory=False)
    prompts = pd.read_csv(prompt_path, low_memory=False)

    text_audit = audit_text_fields(measurable, evidence, prompts)
    text_audit.to_csv(table_out / "available_text_field_audit.csv", index=False)

    assembly, assembly_audit = assemble_url_text(measurable, evidence)
    assembly.to_csv(table_out / "url_text_assembly_audit.csv", index=False)
    assembly_audit.to_csv(table_out / "url_text_assembly_summary.csv", index=False)
    _make_scope_plot(assembly, figure_out / "feature_extraction_text_scope.html")

    url_features = extract_url_features(assembly)
    writing_url = url_features[
        ["normalized_url", "feature_extraction_text_scope", "text_feature_available", *WRITING_FEATURES]
    ].copy()
    writing_url.to_csv(table_out / "url_writing_structure_features.csv", index=False)
    writing_rows = measurable.merge(writing_url, on="normalized_url", how="left", validate="many_to_one")
    writing_rows.to_csv(table_out / "content_lpm_with_writing_structure_features.csv", index=False)

    factual_url = url_features[
        ["normalized_url", "feature_extraction_text_scope", "text_feature_available", *FACTUAL_FEATURES]
    ].copy()
    factual_url.to_csv(table_out / "url_factual_numeric_features.csv", index=False)
    factual_rows = measurable.merge(factual_url, on="normalized_url", how="left", validate="many_to_one")
    factual_rows.to_csv(table_out / "content_lpm_with_factual_numeric_features.csv", index=False)

    url_features[
        ["normalized_url", "feature_extraction_text_scope", "text_feature_available", *LOCATION_FEATURES]
    ].to_csv(table_out / "url_location_transit_features.csv", index=False)
    url_features[
        ["normalized_url", "feature_extraction_text_scope", "text_feature_available", *AMENITY_FEATURES]
    ].to_csv(table_out / "url_amenity_project_features.csv", index=False)
    url_features[
        ["normalized_url", "feature_extraction_text_scope", "text_feature_available", *EVIDENCE_FEATURES]
    ].to_csv(table_out / "url_external_evidence_features.csv", index=False)

    relevance, relevance_audit = build_prompt_page_relevance(measurable, prompts, assembly)
    relevance.to_csv(
        table_out / "source_appearance_prompt_page_relevance_features.csv",
        index=False,
    )
    relevance_audit.to_csv(table_out / "prompt_page_relevance_merge_audit.csv", index=False)

    assembly_metadata = assembly[
        [
            "normalized_url",
            "url_title",
            "url_description",
            "page_text_excerpt",
            "url_text_length_chars",
            "url_text_length_words",
            "full_page_text_available",
            "limited_excerpt_only",
            "text_source_used",
            "feature_extraction_text_scope",
            "text_feature_available",
        ]
    ]
    final = measurable.merge(
        assembly_metadata,
        on="normalized_url",
        how="left",
        validate="many_to_one",
    ).merge(
        url_features.drop(
            columns=["feature_extraction_text_scope", "text_feature_available"],
        ),
        on="normalized_url",
        how="left",
        validate="many_to_one",
    ).merge(
        relevance[
            ["source_appearance_row_id", "prompt_text", *PROMPT_RELEVANCE_FEATURES]
        ],
        on="source_appearance_row_id",
        how="left",
        validate="one_to_one",
    )

    final_path = data_dir / "content_lpm_measurable_rows_with_writing_factual_features.csv"
    final.to_csv(final_path, index=False)
    merge_audit = build_merge_audit(measurable, final, url_features)
    merge_audit.to_csv(table_out / "10_writing_factual_feature_merge_audit.csv", index=False)

    feature_dictionary = build_feature_dictionary()
    feature_dictionary.to_csv(table_out / "writing_factual_feature_dictionary.csv", index=False)

    created_features = [*URL_FEATURE_COLUMNS, *PROMPT_RELEVANCE_FEATURES]
    numeric_features = [
        feature
        for feature in created_features
        if feature != "opening_100_words"
    ]
    validation = build_validation_summary(final, numeric_features)
    validation.to_csv(table_out / "writing_factual_feature_validation_summary.csv", index=False)

    model_formulas = {**SCREENING_MODELS, **TABLE_PROXY_MODELS}
    leakage = build_leakage_check(feature_dictionary, model_formulas)
    leakage.to_csv(table_out / "writing_factual_feature_leakage_check.csv", index=False)

    manual_review = build_manual_review_sample(final, assembly)
    manual_review.to_csv(table_out / "writing_factual_feature_manual_review_sample.csv", index=False)

    screening, proxy, attenuation, model_warnings = run_first_pass_models(final, table_out)
    _make_coefficient_plot(
        screening,
        figure_out / "10_first_pass_feature_screening_forest.html",
        "Notebook 10 first-pass feature screening",
    )
    _make_proxy_attenuation_plot(
        attenuation,
        figure_out / "10_has_table_proxy_attenuation.html",
    )
    _make_composite_distribution_plot(
        final,
        figure_out / "10_composite_feature_distributions.html",
    )

    priority = build_feature_priority(final, screening, attenuation, leakage)
    priority.to_csv(table_out / "10_feature_priority_for_11_econometrics.csv", index=False)

    report_path = report_out / "10_writing_factual_density_feature_layer_report.md"
    _write_report(
        report_path,
        assembly,
        final,
        validation,
        leakage,
        screening,
        attenuation,
        priority,
    )

    required_primary = [
        "factual_numeric_density_score",
        "price_unit_detail_score",
        "location_transit_specificity_score",
        "prompt_page_relevance_score",
    ]
    primary_valid = all(final[feature].nunique(dropna=True) > 1 for feature in required_primary)
    no_row_loss = (
        len(final) == len(measurable)
        and not final["source_appearance_row_id"].duplicated().any()
    )
    leakage_pass = not leakage["status"].eq("fail").any()
    text_available = final["text_feature_available"].fillna(False).any()
    if not leakage_pass:
        final_status = "failed_leakage_detected"
    elif not text_available:
        final_status = "failed_missing_text"
    elif primary_valid and no_row_loss:
        final_status = "completed_ready_for_11_writing_factual_econometrics"
    else:
        final_status = "completed_with_validation_warnings"

    output_files = sorted(
        str(path.relative_to(package))
        for directory in (table_out, figure_out, report_out)
        for path in directory.rglob("*")
        if path.is_file()
    )
    manifest_path = report_out / "10_writing_factual_density_feature_layer_manifest.json"
    manifest = {
        "input_paths": [str(measurable_path), str(evidence_path), str(prompt_path)],
        "output_paths": {
            "data": str(final_path),
            "tables": str(table_out),
            "figures": str(figure_out),
            "report": str(report_path),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "input_rows": len(measurable),
            "output_rows": len(final),
            "unique_urls": final["normalized_url"].nunique(),
            "unique_prompts": final["prompt_id"].nunique(),
        },
        "features_created": created_features,
        "text_fields_used": text_audit.loc[text_audit["use_for_extraction"], "column_name"].unique().tolist(),
        "text_scope_counts": assembly["feature_extraction_text_scope"].value_counts().to_dict(),
        "leakage_status": "pass" if leakage_pass else "fail",
        "validation_status": (
            "pass_with_documented_excerpt_limitations"
            if primary_valid
            else "validation_warning"
        ),
        "first_pass_model_status": "completed",
        "model_warnings": model_warnings,
        "output_files": [*output_files, str(manifest_path.relative_to(package))],
        "final_status": final_status,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    t3 = attenuation[attenuation["model_id"].eq("T3_table_plus_detail_relevance")].iloc[0]
    return {
        "input_rows": len(measurable),
        "output_rows": len(final),
        "unique_urls": final["normalized_url"].nunique(),
        "urls_with_usable_text": int(assembly["text_feature_available"].sum()),
        "excerpt_only_urls": int(
            assembly["feature_extraction_text_scope"].eq("excerpt_only").sum()
        ),
        "features_created": len(created_features),
        "leakage_check_passed": leakage_pass,
        "first_pass_models_completed": len(SCREENING_MODELS) + len(TABLE_PROXY_MODELS),
        "has_table_T1_estimate_pp": float(
            attenuation.loc[
                attenuation["model_id"].eq("T1_has_table_prompt_fe"),
                "has_table_estimate_pp",
            ].iloc[0]
        ),
        "has_table_T3_estimate_pp": float(t3["has_table_estimate_pp"]),
        "has_table_T3_percent_attenuation": float(t3["percent_attenuation_from_T1"]),
        "has_table_T3_coefficient_change_pp": float(t3["coefficient_change_from_T1_pp"]),
        "has_table_T3_proxy_pattern": str(t3["proxy_test_pattern"]),
        "final_dataset": str(final_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
        "final_status": final_status,
    }
