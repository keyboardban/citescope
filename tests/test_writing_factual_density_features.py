import math

import pandas as pd

from src.econometrics_eda_v2.writing_factual_density_features import (
    assemble_url_text,
    build_feature_dictionary,
    build_leakage_check,
    extract_amenity_features,
    extract_factual_features,
    extract_location_features,
    extract_writing_features,
)


def test_bilingual_factual_location_and_amenity_patterns():
    text = (
        "ราคาเริ่มต้น 12.5 ล้านบาท พื้นที่ 85 ตร.ม. 2 ห้องนอน "
        "ใกล้ BTS ชิดลม เดิน 5 นาที มีสระว่ายน้ำ ฟิตเนส และที่จอดรถ"
    )
    factual = extract_factual_features(text)
    location = extract_location_features(text)
    amenity = extract_amenity_features(text)
    assert factual["has_price_detail"] == 1
    assert factual["has_unit_size_detail"] == 1
    assert factual["sqm_mention_count"] >= 1
    assert location["has_transit_detail"] == 1
    assert location["neighborhood_mention_count"] >= 1
    assert amenity["pool_mention_count"] >= 1
    assert amenity["gym_mention_count"] >= 1
    assert amenity["parking_mention_count"] >= 1


def test_missing_text_is_not_encoded_as_feature_absence():
    features = extract_writing_features("")
    assert math.isnan(features["writing_structure_score"])
    assert math.isnan(features["has_faq_pattern"])


def test_text_assembly_distinguishes_full_excerpt_and_missing():
    measurable = pd.DataFrame({"normalized_url": ["u1", "u2", "u3"]})
    evidence = pd.DataFrame(
        {
            "normalized_url": ["u1", "u2"],
            "source_url": ["u1", "u2"],
            "source_root_domain": ["a.test", "b.test"],
            "page_title": ["Short", "Long"],
            "meta_description": ["", ""],
            "page_text_preview_3000_chars": ["short complete text", "x" * 3000],
            "page_text_excerpt": ["short", "x" * 1200],
            "content_chars": [18, 10000],
        }
    )
    assembly, audit = assemble_url_text(measurable, evidence)
    scopes = assembly.set_index("normalized_url")["feature_extraction_text_scope"].to_dict()
    assert scopes == {"u1": "full_text", "u2": "excerpt_only", "u3": "no_text"}
    assert audit.loc[audit.metric.eq("unique_urls_in_text_assembly"), "value"].iloc[0] == 3


def test_feature_dictionary_and_formulas_pass_answer_leakage_guard():
    dictionary = build_feature_dictionary()
    checks = build_leakage_check(
        dictionary,
        {"F1": "cited ~ factual_numeric_density_score + C(prompt_id)"},
    )
    assert checks.status.eq("pass").all()
    assert not dictionary.uses_answer_text.astype(bool).any()
