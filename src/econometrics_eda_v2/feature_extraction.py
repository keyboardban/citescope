from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from src import brand_visibility
from src.chunking import chunk_text
from src.econometrics_eda_v2.page_type_classifier import classify_page_type_details
from src.similarity import SimilarityEngine, summarize_scores
from src.source_type import classify

REQUIRED_BOOL_FEATURES = [
    "has_faq", "has_price_or_package", "has_contact_info", "has_table", "has_bullets",
    "has_author", "has_reviewer", "has_schema", "has_phone_number", "has_email",
    "has_address", "has_opening_hours", "has_booking_or_appointment", "has_step_by_step",
    "has_medical_disclaimer", "has_references", "has_updated_date",
]


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w\u0E00-\u0E7F]+", text or "", flags=re.U))


def _extra_flag(name: str, text: str) -> bool:
    patterns = {
        "has_medical_disclaimer": r"medical advice|consult your doctor|disclaimer|ปรึกษาแพทย์",
        "has_references": r"\breferences\b|\bsources\b|doi\.org|pubmed|อ้างอิง",
    }
    return bool(re.search(patterns.get(name, r"$^"), text or "", flags=re.I | re.U))


def _page_from_parse(row: pd.Series) -> dict[str, Any]:
    def clean(v) -> str:
        return "" if pd.isna(v) else str(v)
    text = clean(row.get("page_text"))
    return {
        "url": clean(row.get("final_url")) or clean(row.get("requested_url")),
        "final_url": clean(row.get("final_url")),
        "title": clean(row.get("page_title")),
        "description": clean(row.get("meta_description")),
        "text": text,
        "markdown": text,
        "status": "success" if bool(row.get("scrape_success")) and bool(row.get("scraped_body_available")) else "failed",
    }


def classify_page_type(row: pd.Series | dict[str, Any]) -> tuple[str, str, str]:
    url = str(row.get("final_url") or row.get("requested_url") or row.get("normalized_url") or "")
    stype, _ = classify(url)
    result = classify_page_type_details(row, stype)
    return result.page_type, result.confidence, result.evidence


def extract_page_features(parse_df: pd.DataFrame, source_rows: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    sim = SimilarityEngine("lexical")
    source_by_url: dict[str, dict[str, Any]] = {}
    if source_rows is not None and not source_rows.empty:
        for _, s in source_rows.iterrows():
            nurl = str(s.get("normalized_url") or "")
            if nurl and nurl not in source_by_url:
                source_by_url[nurl] = s.to_dict()
    rows = []
    for _, row in parse_df.iterrows():
        nurl = str(row.get("requested_normalized_url") or row.get("normalized_url") or row.get("final_normalized_url") or "")
        src = source_by_url.get(nurl) or source_by_url.get(str(row.get("normalized_url") or "")) or {}
        prompt = src.get("prompt_text") or src.get("prompt") or ""
        intent = src.get("intent") or ""
        url = row.get("final_url") or row.get("requested_url") or nurl
        stype, _ = classify(url)
        page = _page_from_parse(row)
        available = bool(row.get("scraped_body_available")) and bool(str(row.get("page_text") or "").strip())
        out: dict[str, Any] = {
            "scrape_id": row.get("scrape_id", ""),
            "normalized_url": nurl,
            "requested_normalized_url": row.get("requested_normalized_url", ""),
            "final_normalized_url": row.get("final_normalized_url", ""),
            "domain": row.get("domain", ""),
            "page_type_scraped_enriched": np.nan,
            "page_type_scraped_confidence": np.nan,
            "page_type_evidence": np.nan,
            "page_type_score_map": np.nan,
            "page_type_unknown_reason": np.nan,
            "page_type_family_scraped": np.nan,
            "h1_or_top_heading": np.nan,
            "currency_count": np.nan,
            "price_keyword_count": np.nan,
            "content_feature_available": available,
            "content_feature_missing_reason": "" if available else (row.get("parse_error") or "no_scraped_body"),
            "word_count": row.get("word_count"),
            "heading_count": row.get("heading_count"),
            "table_count": row.get("table_count"),
            "link_count": row.get("link_count"),
            "page_prompt_similarity": np.nan,
            "max_chunk_prompt_similarity": np.nan,
        }
        for feat in REQUIRED_BOOL_FEATURES:
            out[feat] = np.nan
        if available:
            content = brand_visibility.extract_content_features(page, prompt, intent, stype, sim)
            text = str(row.get("page_text") or "")
            for feat in brand_visibility.CONTENT_BOOL_FEATURES:
                if feat in out:
                    out[feat] = int(bool(content.get(feat)))
            out["has_address"] = int(bool(content.get("has_location_info")))
            out["has_medical_disclaimer"] = int(_extra_flag("has_medical_disclaimer", text))
            out["has_references"] = int(_extra_flag("has_references", text))
            pt = classify_page_type_details(
                {
                    "final_url": row.get("final_url"),
                    "requested_url": row.get("requested_url"),
                    "normalized_url": nurl,
                    "page_title": row.get("page_title"),
                    "meta_description": row.get("meta_description"),
                    "page_text": text,
                    "table_count": row.get("table_count"),
                    "source_type_url": stype,
                },
                stype,
            )
            out["page_type_scraped_enriched"] = pt.page_type
            out["page_type_scraped_confidence"] = pt.confidence
            out["page_type_evidence"] = pt.evidence
            out["page_type_score_map"] = pt.score_map_json()
            out["page_type_unknown_reason"] = pt.unknown_reason
            out["page_type_family_scraped"] = pt.family
            out["h1_or_top_heading"] = pt.h1_or_top_heading
            out["currency_count"] = pt.currency_count
            out["price_keyword_count"] = pt.price_keyword_count
            if prompt and text:
                out["page_prompt_similarity"] = sim.score(text[:8000], prompt)
                chunks = chunk_text(text)
                if chunks:
                    scores = sim.score_many(prompt, [c["text"] for c in chunks])
                    out["max_chunk_prompt_similarity"] = summarize_scores(scores)["max"]
            out["word_count"] = row.get("word_count") or _word_count(text)
        rows.append(out)
    df = pd.DataFrame(rows)
    summary = {
        "rows": int(len(df)),
        "content_feature_available": int(df["content_feature_available"].sum()) if len(df) else 0,
        "content_feature_coverage": float(df["content_feature_available"].mean()) if len(df) else 0.0,
        "page_type_scraped_enriched_distribution": df["page_type_scraped_enriched"].value_counts(dropna=False).to_dict() if len(df) else {},
    }
    return df, summary
