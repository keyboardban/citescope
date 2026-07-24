from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlparse

import pandas as pd

PAGE_TYPE_FAMILY = {
    "homepage": "institutional_info",
    "official_info_page": "institutional_info",
    "article_health_info": "information_content",
    "disease_condition_page": "information_content",
    "service_or_treatment_page": "hospital_service_info",
    "department_or_center_page": "hospital_service_info",
    "doctor_profile": "hospital_service_info",
    "appointment_page": "access_contact",
    "contact_page": "access_contact",
    "location_page": "access_contact",
    "price_package_page": "commercial_price_package",
    "product_marketplace_page": "commercial_price_package",
    "news_announcement_page": "news_or_update",
    "directory_listing_page": "directory_or_platform",
    "third_party_platform_page": "third_party_platform",
    "pdf_document": "document",
    "forum_review_page": "user_generated",
    "faq_page": "information_content",
    "unknown": "unknown",
}

TYPE_PATTERNS = {
    "homepage": {
        "url": r"^/?$|/home/?$|/หน้าแรก/?$",
        "title": r"\bhome\b|หน้าแรก",
        "body": r"",
    },
    "official_info_page": {
        "url": r"/about|/mission|/vision|/policy|/organization|/history|เกี่ยวกับ|นโยบาย|วิสัยทัศน์|พันธกิจ|ประวัติ|องค์กร",
        "title": r"about|mission|vision|policy|organization|history|เกี่ยวกับ|นโยบาย|วิสัยทัศน์|พันธกิจ|ประวัติ|องค์กร",
        "body": r"about|mission|vision|policy|organization|เกี่ยวกับ|นโยบาย|วิสัยทัศน์|พันธกิจ|องค์กร",
    },
    "article_health_info": {
        "url": r"/article|/blog|/guide|/expert-advice|/health[-_/ ]?information|บทความ|ความรู้",
        "title": r"article|health information|knowledge|blog|guide|advice|บทความ|ความรู้สุขภาพ|เกร็ดความรู้|สุขภาพ|คำแนะนำ",
        "body": r"article|health information|knowledge|guide|advice|บทความ|ความรู้สุขภาพ|เกร็ดความรู้|สุขภาพ|คำแนะนำ",
    },
    "disease_condition_page": {
        "url": r"/disease|/condition|/symptoms|/diagnosis|โรค|อาการ|ภาวะ",
        "title": r"disease|condition|symptoms|diagnosis|โรค|อาการ|ภาวะ|สาเหตุ|การวินิจฉัย|วิธีรักษา",
        "body": r"disease|condition|symptoms|diagnosis|โรค|อาการ|ภาวะ|สาเหตุ|การวินิจฉัย|วิธีรักษา",
    },
    "service_or_treatment_page": {
        "url": r"/service|/services|/treatment|/procedure|/program|/checkup|บริการ|การรักษา|หัตถการ|ตรวจ|โปรแกรมตรวจ|ตรวจสุขภาพ",
        "title": r"service|treatment|procedure|program|checkup|บริการ|การรักษา|หัตถการ|ตรวจ|โปรแกรมตรวจ|ตรวจสุขภาพ",
        "body": r"service|treatment|procedure|program|checkup|บริการ|การรักษา|หัตถการ|ตรวจ|โปรแกรมตรวจ|ตรวจสุขภาพ",
    },
    "department_or_center_page": {
        "url": r"/department|/dept|/center|/centre|/clinic|/specialty|แผนก|ศูนย์|คลินิก|หน่วย|สาขา|ภาควิชา",
        "title": r"department|center|centre|clinic|specialty|แผนก|ศูนย์|คลินิก|หน่วย|สาขา|ภาควิชา",
        "body": r"department|center|centre|clinic|specialty|แผนก|ศูนย์|คลินิก|หน่วย|สาขา|ภาควิชา",
    },
    "doctor_profile": {
        "url": r"/doctor|/physician|/specialist|/profile|/provider|แพทย์|คุณหมอ|ตารางแพทย์|ประวัติแพทย์",
        "title": r"doctor|physician|specialist|profile|แพทย์|คุณหมอ|อาจารย์แพทย์|ตารางแพทย์|ประวัติแพทย์",
        "body": r"doctor|physician|specialist|profile|แพทย์|คุณหมอ|อาจารย์แพทย์|ตารางแพทย์|ประวัติแพทย์",
    },
    "appointment_page": {
        "url": r"/appointment|/booking|/book|/schedule|/reserve|นัดหมาย|จองคิว|จองตรวจ|ตารางออกตรวจ",
        "title": r"appointment|booking|schedule|reserve|นัดหมาย|จองคิว|จองตรวจ|ตารางออกตรวจ",
        "body": r"appointment|booking|schedule|reserve|นัดหมาย|จองคิว|จองตรวจ|ตารางออกตรวจ",
    },
    "contact_page": {
        "url": r"/contact|/inquiry|ติดต่อ|สอบถาม",
        "title": r"contact|contact us|phone|email|inquiry|ติดต่อ|ติดต่อเรา|สอบถาม|เบอร์โทร|โทรศัพท์|อีเมล",
        "body": r"contact|phone|email|inquiry|ติดต่อ|ติดต่อเรา|สอบถาม|เบอร์โทร|โทรศัพท์|อีเมล",
    },
    "location_page": {
        "url": r"/location|/map|/maps|/direction|/directions|/parking|แผนที่|การเดินทาง|ที่จอดรถ|อาคาร|สถานที่",
        "title": r"location|map|direction|parking|building|แผนที่|การเดินทาง|ที่จอดรถ|อาคาร|สถานที่",
        "body": r"location|map|direction|parking|building|แผนที่|การเดินทาง|ที่จอดรถ|อาคาร|สถานที่",
    },
    "faq_page": {
        "url": r"/faq|/questions|คำถามที่พบบ่อย|ถามตอบ",
        "title": r"faq|frequently asked questions|questions|คำถามที่พบบ่อย|ถามตอบ",
        "body": r"faq|frequently asked questions|questions|คำถามที่พบบ่อย|ถามตอบ",
    },
    "news_announcement_page": {
        "url": r"/news|/announcement|/press|/event|ข่าว|ประกาศ|กิจกรรม",
        "title": r"news|announcement|press|event|press release|ข่าว|ประกาศ|กิจกรรม",
        "body": r"news|announcement|press|event|press release|ข่าว|ประกาศ|กิจกรรม",
    },
    "directory_listing_page": {
        "url": r"/directory|/find-a|/listing|/search|รายชื่อ|ค้นหา",
        "title": r"directory|find a|listing|search|รายชื่อ|ค้นหา",
        "body": r"directory|listing|รายชื่อ|ค้นหา",
    },
    "forum_review_page": {
        "url": r"pantip|reddit|/forum|/thread|/community|/review|/reviews|กระทู้|รีวิว",
        "title": r"pantip|reddit|review|reviews|forum|community|discussion|thread|รีวิว|กระทู้|ความคิดเห็น",
        "body": r"pantip|reddit|review|reviews|forum|community|discussion|thread|รีวิว|กระทู้|ความคิดเห็น",
    },
    "product_marketplace_page": {
        "url": r"lazada|shopee|amazon|/product|/products|/shop|/store|/item|/pid|/collections?|cart|checkout|marketplace",
        "title": r"product detail|add to cart|buy|shop|store|marketplace|ซื้อสินค้า|สินค้า",
        "body": r"add to cart|buy now|checkout|marketplace|ซื้อสินค้า|สินค้า",
    },
    "third_party_platform_page": {
        "url": r"apps\.apple\.com|play\.google\.com|facebook\.com|youtube\.com|line\.me|instagram\.com|tiktok\.com|linkedin\.com",
        "title": r"app store|play store|facebook|youtube|line|instagram|tiktok|linkedin",
        "body": r"",
    },
}

PRICE_STRONG = r"price|pricing|package|packages|cost|fee|ราคา|ค่าใช้จ่าย|แพ็กเกจ|แพคเกจ|ค่ารักษา|โปรโมชั่น"
PRICE_BODY = r"price|cost|fee|ราคา|ค่าใช้จ่าย|แพ็กเกจ|แพคเกจ|package|packages|ค่ารักษา|โปรโมชั่น"
CURRENCY = r"(?:฿|บาท|THB|\$|USD|£|€)\s?\d|\d[\d,]*(?:\.\d+)?\s?(?:บาท|THB|USD|dollars?)"
SERVICE_HEADING = r"service|package|checkup|treatment|procedure|program|บริการ|โปรแกรม|ตรวจสุขภาพ|การรักษา|หัตถการ"


@dataclass
class PageTypeResult:
    page_type: str
    family: str
    confidence: str
    evidence: str
    score_map: dict[str, float]
    unknown_reason: str
    currency_count: int
    price_keyword_count: int
    h1_or_top_heading: str
    final_source: str

    def score_map_json(self) -> str:
        return json.dumps(self.score_map, ensure_ascii=False, sort_keys=True)


def _clean(v) -> str:
    return "" if pd.isna(v) else str(v)


def _count(pattern: str, text: str) -> int:
    if not pattern:
        return 0
    return len(re.findall(pattern, text or "", flags=re.I | re.U))


def _hit(pattern: str, text: str) -> bool:
    return bool(pattern and re.search(pattern, text or "", flags=re.I | re.U))


def _parse_url(url: str):
    if not str(url or "").strip():
        return urlparse("")
    return urlparse(url if re.match(r"^[a-z][a-z0-9+.-]*://", url, flags=re.I) else "https://" + url)


def top_heading(text: str, title: str = "") -> str:
    title_clean = title.strip()
    for line in re.split(r"[\r\n]+", text or ""):
        line = line.strip(" #\t")
        if not line:
            continue
        if title_clean and line.casefold() == title_clean.casefold():
            continue
        if len(line) <= 180:
            return line
    return title_clean[:180]


def _url_signal(url: str) -> str:
    p = _parse_url(url)
    path = p.path.replace("-", " ").replace("_", " ")
    return " ".join([p.netloc, path, PurePosixPath(p.path).suffix.lower()])


def _field_scores(patterns: dict[str, str], url_sig: str, title: str, meta: str, heading: str, body: str) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    if _hit(patterns["url"], url_sig):
        score += 4
        evidence.append("url_path_signal")
    if _hit(patterns["title"], title):
        score += 4
        evidence.append("title_signal")
    if _hit(patterns["title"], heading):
        score += 4
        evidence.append("h1_or_top_heading_signal")
    if _hit(patterns["title"], meta):
        score += 2
        evidence.append("meta_description_signal")
    body_hits = _count(patterns["body"], body[:12000])
    if body_hits:
        score += 1
        evidence.append("body_keyword_signal")
    if body_hits >= 4:
        score += min(2, body_hits // 4)
        evidence.append("repeated_body_keyword_signal")
    return score, evidence


def _price_score(url_sig: str, title: str, meta: str, heading: str, body: str, table_count: int) -> tuple[float, list[str], int, int]:
    currency_count = _count(CURRENCY, body)
    price_keyword_count = _count(PRICE_BODY, body)
    strong = " ".join([url_sig, title, heading])
    all_meta = " ".join([strong, meta])
    score = 0.0
    evidence: list[str] = []
    if _hit(PRICE_STRONG, strong):
        score += 8
        evidence.append("strong_price_signal_in_url_title_heading")
    elif _hit(PRICE_STRONG, all_meta):
        score += 5
        evidence.append("price_signal_in_meta")
    if table_count > 0 and currency_count >= 2 and _hit(PRICE_BODY, body):
        score += 5
        evidence.append("structured_pricing_table_currency_terms")
    if price_keyword_count >= 3 and currency_count >= 1 and _hit(SERVICE_HEADING, " ".join([title, heading])):
        score += 4
        evidence.append("multiple_independent_price_signals")
    return score, evidence, currency_count, price_keyword_count


def score_page_type_candidates(
    *,
    url: str = "",
    source_type_url: str = "",
    title: str = "",
    meta_description: str = "",
    h1_or_top_heading: str = "",
    headings: str = "",
    page_text: str = "",
    table_count: int = 0,
) -> tuple[dict[str, float], dict[str, list[str]], int, int]:
    url_sig = _url_signal(url)
    title_sig = title or ""
    heading_sig = " ".join([h1_or_top_heading or "", headings or ""])
    body = page_text or ""
    score_map: dict[str, float] = {}
    evidence_map: dict[str, list[str]] = {}

    p = _parse_url(url)
    suffix = PurePosixPath(p.path).suffix.lower()
    if suffix == ".pdf" or _hit(r"\bpdf\b|download|document|เอกสาร", " ".join([title_sig, heading_sig])):
        score_map["pdf_document"] = 8
        evidence_map["pdf_document"] = ["file_extension_or_document_signal"]

    price_score, price_ev, currency_count, price_keyword_count = _price_score(
        url_sig, title_sig, meta_description, heading_sig, body, table_count
    )
    score_map["price_package_page"] = price_score
    evidence_map["price_package_page"] = price_ev

    for page_type, pats in TYPE_PATTERNS.items():
        score, evidence = _field_scores(pats, url_sig, title_sig, meta_description, heading_sig, body)
        if page_type == "third_party_platform_page" and source_type_url in {"social", "video"}:
            score += 2
            evidence.append(f"source_type_url={source_type_url}")
        if page_type == "product_marketplace_page" and source_type_url == "ecommerce":
            score += 2
            evidence.append("source_type_url=ecommerce")
        if page_type == "forum_review_page" and source_type_url in {"forum", "review"}:
            score += 2
            evidence.append(f"source_type_url={source_type_url}")
        if page_type == "news_announcement_page" and source_type_url == "news":
            score += 2
            evidence.append("source_type_url=news")
        score_map[page_type] = score
        evidence_map[page_type] = evidence
    return score_map, evidence_map, currency_count, price_keyword_count


def classify_page_type_family(page_type: str) -> str:
    return PAGE_TYPE_FAMILY.get(str(page_type or "unknown"), "unknown")


def page_type_family(page_type: str) -> str:
    return classify_page_type_family(page_type)


def extract_page_type_evidence(result: PageTypeResult) -> str:
    return result.evidence


def _unknown_reason(url: str, title: str, heading: str, body: str, source_type_url: str, ranked: list[tuple[str, float]], confidence: str) -> str:
    if ranked and ranked[0][1] >= 3 and confidence == "low":
        return "conflicting_evidence_low_confidence"
    if not url:
        return "missing_no_url"
    if not body:
        return "no_scraped_body"
    if len(re.findall(r"[\w\u0E00-\u0E7F]+", body, flags=re.U)) < 40:
        return "body_too_short"
    if _hit(r"login|sign in|verify|please wait|enable javascript|captcha|404|not found", " ".join([title, heading, body[:300]])):
        return "dynamic_or_login_page"
    if not _hit(r"[a-zA-Z\u0E00-\u0E7F]", title + heading):
        return "no_title_heading_signal"
    p = _parse_url(url)
    if p.path in {"", "/"}:
        return "generic_homepage"
    if source_type_url in {"unknown", ""}:
        return "classifier_rules_missing_medical_terms"
    return "no_url_path_signal"


def classify_page_type_v2(
    *,
    url: str = "",
    source_type_url: str = "",
    title: str = "",
    meta_description: str = "",
    h1_or_top_heading: str = "",
    headings: str = "",
    page_text: str = "",
    table_count: int = 0,
    final_source: str = "metadata_fallback",
) -> PageTypeResult:
    heading = h1_or_top_heading or top_heading(page_text, title)
    score_map, evidence_map, currency_count, price_keyword_count = score_page_type_candidates(
        url=url,
        source_type_url=source_type_url,
        title=title,
        meta_description=meta_description,
        h1_or_top_heading=heading,
        headings=headings,
        page_text=page_text,
        table_count=table_count,
    )
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    best_type, best_score = ranked[0] if ranked else ("unknown", 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = best_score - second_score

    if best_type == "price_package_page":
        has_strong_price = any(
            ev in evidence_map.get(best_type, [])
            for ev in (
                "strong_price_signal_in_url_title_heading",
                "structured_pricing_table_currency_terms",
                "multiple_independent_price_signals",
            )
        )
        if not has_strong_price:
            best_score = 0.0

    if best_score >= 7 and gap >= 2:
        confidence = "high"
    elif best_score >= 5 and gap >= 1:
        confidence = "medium"
    elif best_score >= 3:
        confidence = "low"
    else:
        confidence = "unknown"

    if confidence in {"unknown", "low"}:
        unknown_reason = _unknown_reason(url, title, heading, page_text, source_type_url, ranked, confidence)
        if confidence == "unknown":
            best_type = "unknown"
    else:
        unknown_reason = ""

    if best_type == "unknown":
        family = "unknown"
        evidence = f"unknown_reason={unknown_reason}; top_scores={dict(ranked[:3])}"
    else:
        family = classify_page_type_family(best_type)
        evidence = "; ".join(evidence_map.get(best_type, [])) or "score_threshold_met"

    return PageTypeResult(
        page_type=best_type,
        family=family,
        confidence=confidence,
        evidence=evidence,
        score_map=score_map,
        unknown_reason=unknown_reason,
        currency_count=currency_count,
        price_keyword_count=price_keyword_count,
        h1_or_top_heading=heading,
        final_source=final_source,
    )


def classify_page_type_details(row: pd.Series | dict, source_type_url: str = "") -> PageTypeResult:
    get = row.get if isinstance(row, dict) else row.get
    url = _clean(get("final_url") or get("requested_url") or get("source_url") or get("normalized_url"))
    title = _clean(get("page_title") or get("title") or get("source_title"))
    meta = _clean(get("meta_description") or get("description") or get("source_description"))
    body = _clean(get("page_text") or get("text") or get("markdown"))
    heading = _clean(get("h1_or_top_heading")) or top_heading(body, title)
    headings = _clean(get("headings"))
    table_count = int(pd.to_numeric(pd.Series([get("table_count")]), errors="coerce").fillna(0).iloc[0])
    source_type_url = source_type_url or _clean(get("source_type_url"))
    final_source = "scraped_content" if body else ("url_seed" if url else "missing_no_url")
    return classify_page_type_v2(
        url=url,
        source_type_url=source_type_url,
        title=title,
        meta_description=meta,
        h1_or_top_heading=heading,
        headings=headings,
        page_text=body,
        table_count=table_count,
        final_source=final_source,
    )
