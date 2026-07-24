"""Evidence-scored, cross-domain website page-function taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, unquote, urlparse

import pandas as pd

from src.url_utils import root_domain


SITE_TYPES = (
    "official_company_or_brand", "official_organization", "government", "education", "research_or_academic",
    "news_media", "blog_or_content_site", "ecommerce_store", "marketplace_or_platform",
    "directory_or_listing_platform", "review_platform", "social_or_forum", "video_platform",
    "documentation_or_developer_site", "map_or_location_platform", "file_or_document_host", "unknown",
)
GENERAL_TAXONOMY_VERSION = "general_page_taxonomy_v2"
PAGE_FAMILIES = (
    "landing_or_brand_page", "informational_content", "news_or_press", "commercial_product_or_service",
    "pricing_or_package", "directory_or_listing", "comparison_or_review", "support_or_help",
    "contact_or_location", "trust_about_or_legal", "transactional_or_account", "document_or_media",
    "social_or_user_generated", "search_or_results", "unknown",
)
PAGE_TYPES = (
    "homepage", "landing_page", "about_page", "brand_page", "campaign_page", "blog_article", "guide_article",
    "educational_article", "evergreen_content", "glossary_or_definition_page", "news_article", "press_release",
    "announcement_page", "product_page", "service_page", "solution_page", "feature_page", "product_category_page",
    "collection_page", "promotion_page", "pricing_page", "package_page", "plans_page", "fees_page", "listing_page",
    "search_results_page", "category_listing_page", "directory_page", "marketplace_listing_page", "profile_page",
    "review_page", "comparison_page", "best_of_listicle", "ranking_page", "testimonial_page", "case_study",
    "faq_page", "help_center_article", "support_page", "troubleshooting_page", "documentation_page", "api_docs_page",
    "contact_page", "location_page", "branch_page", "map_or_directions_page", "appointment_or_booking_page",
    "privacy_policy", "terms_page", "legal_page", "compliance_page", "security_page", "login_page", "signup_page",
    "checkout_page", "cart_page", "payment_page", "account_page", "pdf_document", "report_document",
    "brochure_document", "video_page", "image_gallery_page", "downloadable_resource", "forum_thread", "social_post",
    "comment_thread", "community_page", "unknown",
)
DETAIL_TO_FAMILY = {
    **{name: "landing_or_brand_page" for name in ("homepage", "landing_page", "about_page", "brand_page", "campaign_page")},
    **{name: "informational_content" for name in ("blog_article", "guide_article", "educational_article", "evergreen_content", "glossary_or_definition_page")},
    **{name: "news_or_press" for name in ("news_article", "press_release", "announcement_page")},
    **{name: "commercial_product_or_service" for name in ("product_page", "service_page", "solution_page", "feature_page", "product_category_page", "collection_page", "promotion_page")},
    **{name: "pricing_or_package" for name in ("pricing_page", "package_page", "plans_page", "fees_page")},
    **{name: "directory_or_listing" for name in ("listing_page", "category_listing_page", "directory_page", "marketplace_listing_page", "profile_page")},
    "search_results_page": "search_or_results",
    **{name: "comparison_or_review" for name in ("review_page", "comparison_page", "best_of_listicle", "ranking_page", "testimonial_page", "case_study")},
    **{name: "support_or_help" for name in ("faq_page", "help_center_article", "support_page", "troubleshooting_page", "documentation_page", "api_docs_page")},
    **{name: "contact_or_location" for name in ("contact_page", "location_page", "branch_page", "map_or_directions_page", "appointment_or_booking_page")},
    **{name: "trust_about_or_legal" for name in ("privacy_policy", "terms_page", "legal_page", "compliance_page", "security_page")},
    **{name: "transactional_or_account" for name in ("login_page", "signup_page", "checkout_page", "cart_page", "payment_page", "account_page")},
    **{name: "document_or_media" for name in ("pdf_document", "report_document", "brochure_document", "video_page", "image_gallery_page", "downloadable_resource")},
    **{name: "social_or_user_generated" for name in ("forum_thread", "social_post", "comment_thread", "community_page")},
    "unknown": "unknown",
}

SOCIAL = ("reddit", "pantip", "facebook", "instagram", "threads", "x.com", "quora", "stackexchange")
VIDEO = ("youtube", "youtu.be", "tiktok", "vimeo", "dailymotion")
MAPS = ("google.com/maps", "maps.google", "openstreetmap", "waze")
MARKETPLACES = ("amazon", "ebay", "etsy", "shopee", "lazada", "airbnb", "booking", "agoda", "ddproperty", "propertyhub")
REVIEWS = ("trustpilot", "tripadvisor", "yelp", "glassdoor")
DOCS = ("readthedocs", "developer.mozilla", "docs.", "swagger", "postman")
FILE_HOSTS = ("drive.google", "dropbox", "onedrive", "docs.google", "scribd")


@dataclass(frozen=True)
class GeneralPageResult:
    detail: str
    family: str
    score: float
    confidence: str
    evidence_url: str
    evidence_title: str
    evidence_domain: str
    evidence_headings: str
    evidence_content: str
    reason: str


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _url_parts(url: str) -> tuple[str, str, str]:
    raw = _clean(url)
    if raw and not re.match(r"^[a-z][a-z0-9+.-]*://", raw, flags=re.I):
        raw = "https://" + raw
    parsed = urlparse(raw)
    return (parsed.hostname or "").casefold(), unquote(parsed.path or "").casefold(), unquote(parsed.query or "").casefold()


def _match(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, text or "", flags=re.I | re.U))


def _route_text(path: str, query: str) -> str:
    """Return page-route evidence without hostname or tracking parameters."""
    functional_query = []
    for key, value in parse_qsl(query, keep_blank_values=True):
        key = key.casefold()
        if key in {"q", "query", "search", "keyword", "filter", "category"}:
            functional_query.extend((key, value.casefold()))
    route_path = path if not path or path.endswith("/") else path + "/"
    readable_path = re.sub(r"[/_-]+", " ", path).strip()
    return " ".join(part for part in (route_path, readable_path, " ".join(functional_query)) if part)


def _site_type(url: str, domain: str, title: str, meta: str) -> str:
    host, path, _ = _url_parts(url)
    root = (domain or root_domain(url) or host).casefold()
    text = " ".join([root, path, title, meta]).casefold()
    if path.endswith(".pdf") or any(token in root for token in FILE_HOSTS): return "file_or_document_host"
    if any(token in root for token in VIDEO): return "video_platform"
    if any(token in root for token in SOCIAL): return "social_or_forum"
    if any(token in text for token in MAPS): return "map_or_location_platform"
    if root.endswith(".gov") or ".gov." in root or root.endswith(".go.th") or ".go." in root: return "government"
    if root.endswith(".edu") or root.endswith(".ac.th") or ".edu." in root: return "education"
    if any(token in root for token in ("arxiv", "pubmed", "researchgate", "jstor", "doi.org")): return "research_or_academic"
    if any(token in root for token in DOCS) or _match(r"\b(api|documentation|developer docs)\b", text): return "documentation_or_developer_site"
    if any(token in root for token in REVIEWS): return "review_platform"
    if any(token in root for token in MARKETPLACES): return "marketplace_or_platform"
    if _match(r"\b(news|press|journal|times|post|reuters|media)\b", root): return "news_media"
    if _match(r"\b(shop|store)\b", root): return "ecommerce_store"
    if _match(r"\b(directory|listing|marketplace|classifieds)\b", root): return "directory_or_listing_platform"
    if _match(r"\b(blog|insights|stories|articles)\b", root): return "blog_or_content_site"
    if _match(r"\bofficial\b", text): return "official_company_or_brand"
    return "unknown"


def _add(scores: dict[str, float], evidence: dict[str, list[str]], label: str, weight: float, field: str, signal: str) -> None:
    scores[label] = scores.get(label, 0) + weight
    evidence.setdefault(label, []).append(f"{field}:{signal}")


def _rule(scores: dict[str, float], evidence: dict[str, list[str]], label: str, pattern: str, text: str, field: str, weight: float, signal: str) -> None:
    if _match(pattern, text): _add(scores, evidence, label, weight, field, signal)


def _classify(row: pd.Series | dict[str, Any], include_content: bool) -> GeneralPageResult:
    get = row.get
    url = _clean(get("source_url") or get("final_url") or get("normalized_url"))
    domain = _clean(get("source_root_domain")) or root_domain(url)
    host, path, query = _url_parts(url)
    title = _clean(get("page_title") or get("source_title") or get("title"))
    meta = _clean(get("meta_description") or get("source_description"))
    headings = _clean(get("headings") or get("h1_or_top_heading"))
    content = _clean(get("page_text_excerpt") or get("page_text") or get("text")) if include_content else ""
    schema = _clean(get("structured_data_types")).casefold() if include_content else ""
    url_text = _route_text(path, query)
    scores: dict[str, float] = {"unknown": 0.0}; evidence: dict[str, list[str]] = {"unknown": []}
    site = _site_type(url, domain, title, meta)
    if path.endswith(".pdf"): _add(scores, evidence, "pdf_document", 20, "url", "pdf_suffix")
    if site == "video_platform": _add(scores, evidence, "video_page", 16, "domain", "video_platform")
    if site == "social_or_forum": _add(scores, evidence, "forum_thread", 14, "domain", "social_forum_platform")
    if site == "documentation_or_developer_site": _add(scores, evidence, "documentation_page", 12, "domain", "documentation_site")
    if site == "map_or_location_platform": _add(scores, evidence, "map_or_directions_page", 14, "domain", "map_platform")
    if schema:
        # Schema often describes a component embedded in a page (especially FAQPage),
        # so it supports a label but cannot define the primary page function alone.
        if "faqpage" in schema: _add(scores, evidence, "faq_page", 4, "structured_data", "faqpage")
        if "newsarticle" in schema: _add(scores, evidence, "news_article", 6, "structured_data", "newsarticle")
        elif any(token in schema for token in ("article", "blogposting")): _add(scores, evidence, "blog_article", 4, "structured_data", "article")
        if any(token in schema for token in ("product", "offer")): _add(scores, evidence, "product_page", 5, "structured_data", "product_or_offer")
        if "itemlist" in schema: _add(scores, evidence, "category_listing_page", 5, "structured_data", "itemlist")
        if "videoobject" in schema: _add(scores, evidence, "video_page", 8, "structured_data", "videoobject")
        if "contactpage" in schema: _add(scores, evidence, "contact_page", 6, "structured_data", "contactpage")
    # URL patterns are route-specific. Title/content patterns may be more natural
    # language, but generic words such as "plan" or a hostname are not page labels.
    rules = [
        ("privacy_policy", r"(?:^|/)(?:privacy(?:-policy)?|data-policy)(?:/|$)", r"\bprivacy policy\b|นโยบายความเป็นส่วนตัว", "privacy"),
        ("terms_page", r"(?:^|/)(?:terms|terms-and-conditions|conditions)(?:/|$)", r"\bterms (?:and conditions|of (?:use|service))\b|ข้อกำหนด|เงื่อนไข", "terms"),
        ("login_page", r"(?:^|/)(?:login|sign-in|auth)(?:/|$)", r"\b(?:log|sign) in\b", "login"),
        ("signup_page", r"(?:^|/)(?:sign-up|register|create-account)(?:/|$)", r"\b(?:sign up|register|create account)\b", "signup"),
        ("checkout_page", r"(?:^|/)(?:checkout)(?:/|$)", r"\bcheckout\b", "checkout"),
        ("cart_page", r"(?:^|/)(?:cart|basket)(?:/|$)", r"\b(?:shopping cart|basket)\b", "cart"),
        ("payment_page", r"(?:^|/)(?:payment|pay)(?:/|$)", r"\bpayment\b", "payment"),
        ("account_page", r"(?:^|/)(?:account|my-account)(?:/|$)", r"\bmy account\b", "account"),
        ("contact_page", r"(?:^|/)(?:contact|contact-us|enquiry|inquiry)(?:/|$)", r"\b(?:contact us|get in touch|sales enquiry)\b|ติดต่อ|สอบถาม", "contact"),
        ("location_page", r"(?:^|/)(?:locations?|branches?|directions?)(?:/|$)", r"\b(?:our locations?|branch locations?|directions|address)\b|สาขา|ที่ตั้ง|แผนที่", "location"),
        ("faq_page", r"(?:^|/)(?:faq|faqs|frequently-asked-questions)(?:/|$)", r"\b(?:faq|frequently asked questions)\b|คำถามที่พบบ่อย", "faq"),
        ("documentation_page", r"(?:^|/)(?:docs?|documentation|reference|developer)(?:/|$)", r"\b(?:documentation|developer reference|api reference)\b", "documentation"),
        ("help_center_article", r"(?:^|/)(?:help|support|help-centre|help-center|knowledge-base)(?:/|$)", r"\b(?:help cent(?:re|er)|support article|knowledge base|troubleshoot)\b|ศูนย์ช่วยเหลือ", "help"),
        ("pricing_page", r"(?:^|/)(?:pricing|price-list|fees|rates)(?:/|$)", r"\b(?:pricing|price list|fee schedule|subscription pricing|service rates)\b|ราคา|ค่าบริการ", "pricing"),
        ("package_page", r"(?:^|/)(?:packages?|subscriptions?)(?:/|$)", r"\b(?:service package|subscription package|pricing package|bundle)\b|แพ็กเกจ", "package"),
        ("search_results_page", r"(?:^|/)(?:search|search-results|results)(?:/|$)|\b(?:q|query|search|keyword)\b", r"\bsearch results\b|ผลการค้นหา", "search_results"),
        ("category_listing_page", r"(?:^|/)(?:category|categories|collections?|new-developments|our-properties)(?:/|$)", r"\b(?:browse all|all categories|all properties|new developments)\b", "category_listing"),
        ("listing_page", r"(?:^|/)(?:listings?|properties|property|condo-project|buildings?)(?:/|$)|(?:^|/)(?:condos?|apartments?)-(?:for-sale|for-rent)(?:/|$)", r"\b(?:property listing|properties for (?:sale|rent)|condos? for (?:sale|rent)|for sale|for rent|rentals)\b|ประกาศ|ขาย|เช่า", "listing"),
        ("directory_page", r"(?:^|/)(?:directory|providers|doctors|restaurants)(?:/|$)", r"\b(?:business directory|provider directory|doctor directory|restaurant directory)\b", "directory"),
        ("review_page", r"(?:^|/)(?:reviews?|testimonials?)(?:/|$)", r"\b(?:review|reviews|ratings|testimonial|testimonials)\b|รีวิว", "review"),
        ("comparison_page", r"(?:^|/)(?:compare|comparison|alternatives|best-of|ranking)(?:/|$)", r"\b(?:compare|comparison|alternatives|best|top \d+|ranking|versus|vs\.?)\b|เปรียบเทียบ|อันดับ", "comparison"),
        ("case_study", r"(?:^|/)(?:case-study|case-studies|customer-stories)(?:/|$)", r"\b(?:case study|customer story|success story)\b", "case_study"),
        ("press_release", r"(?:^|/)(?:press-release|press-room)(?:/|$)", r"\bpress release\b|ข่าวประชาสัมพันธ์", "press"),
        ("news_article", r"(?:^|/)(?:news|latest|media)(?:/|$)", r"\b(?:breaking news|news report)\b|ข่าว", "news"),
        ("guide_article", r"(?:^|/)(?:guides?|help-and-guides|how-to|tips|checklists?|neighbou?rhoods?|areas|location-guides)(?:/|$)", r"\b(?:guide|how to|tips|checklist|neighbou?rhood guide|area guide)\b|คู่มือ|วิธี|แนะนำ", "guide"),
        ("blog_article", r"(?:^|/)(?:blogs?|posts?|read|content|articles?|articledetail|insights?|property-insights?|hometips|journal|knowledge|stories|origin-blog|livingdetail(?:_en)?|news-and-articles|tcc_media)(?:/|$)", r"\b(?:blog|article|insight)\b|บทความ", "blog"),
        ("report_document", r"(?:^|/)(?:research|reports?|market-reports?|whitepapers?|outlook)(?:/|$)|(?:^|/)abs(?:/|$)", r"\b(?:research report|market report|industry outlook|white ?paper)\b", "report"),
        ("product_page", r"(?:^|/)(?:products?|items?|sku|courses?|apps?)(?:/|$)", r"\bproduct details\b", "product"),
        ("service_page", r"(?:^|/)(?:services?|treatments?|consulting|repairs?)(?:/|$)", r"\b(?:our services|service details|treatment|consulting service|repair service)\b|บริการ", "service"),
        ("solution_page", r"(?:^|/)(?:solutions?|features?)(?:/|$)", r"\b(?:solution|product feature)\b", "solution"),
        ("about_page", r"(?:^|/)(?:about|about-us|company|our-story)(?:/|$)", r"\b(?:about us|our company|our story)\b|เกี่ยวกับ", "about"),
        ("landing_page", r"(?:^|/)(?:landing|campaign|lp|get-started)(?:/|$)", r"\b(?:campaign|landing page|get started)\b", "landing"),
        ("appointment_or_booking_page", r"(?:^|/)(?:appointment|booking|ibooking|reserve|schedule|form-appointment)(?:/|$)", r"\b(?:book an appointment|make a booking|reserve|schedule)\b|จอง|นัดหมาย", "booking"),
        ("downloadable_resource", r"(?:^|/)(?:download|downloads|brochure|factsheet|resources?)(?:/|$)", r"\b(?:download|brochure|fact sheet|resource)\b", "download"),
    ]
    editorial_route = _match(
        r"(?:^|/)(?:blogs?|posts?|read|content|articles?|articledetail|insights?|property-insights?|"
        r"hometips|journal|knowledge|stories|origin-blog|livingdetail(?:_en)?|news|news-and-articles|"
        r"guides?|help-and-guides|tcc_media)(?:/|$)",
        path,
    )
    for detail, url_pattern, text_pattern, signal in rules:
        if not (editorial_route and detail in {"listing_page", "product_page", "service_page"}):
            _rule(scores, evidence, detail, url_pattern, url_text, "url", 8, signal)
            _rule(scores, evidence, detail, text_pattern, title, "title", 6, signal)
            _rule(scores, evidence, detail, text_pattern, meta, "title", 3, f"meta_{signal}")
        if include_content:
            _rule(scores, evidence, detail, text_pattern, headings, "headings", 3, signal)
            _rule(scores, evidence, detail, text_pattern, content, "content", 1.5, signal)

    # Route shape plus an independently assigned site role can safely distinguish
    # a marketplace listing from an official brand's project/product page. This
    # avoids teaching the cross-domain regexes real-estate brand names.
    legacy_site = _clean(get("source_type_real_estate")).casefold()
    segments = [segment for segment in path.split("/") if segment]
    while segments and _match(r"^(?:en|th|zh|cn|eng|th-th|th-en|en-th)$", segments[0]):
        segments.pop(0)
    primary_segment = segments[0] if segments else ""
    project_detail_route = primary_segment in {
        "project", "projects", "project-detail", "condo", "condominium", "property",
        "properties", "room", "unit", "detail", "โครงการ", "โครงการคอนโด", "คอนโด",
    } and len(segments) >= 2
    if project_detail_route:
        if legacy_site in {"property_portal", "listing_marketplace", "broker_agency"}:
            _add(scores, evidence, "listing_page", 10, "domain", "site_role_with_detail_route")
        elif legacy_site in {"developer_official", "project_official"}:
            _add(scores, evidence, "product_page", 10, "domain", "official_site_with_project_route")
    if path in ("", "/", "/home", "/th", "/en"):
        _add(scores, evidence, "homepage", 10, "url", "root_path")
    if include_content and _match(r"\?[^\n]{0,60}\?[^\n]{0,60}\?", content): _add(scores, evidence, "faq_page", 2, "content", "repeated_question_structure")
    if bool(get("has_price_or_package")) and _match(r"pricing|price|plan|package|ราคา|แพ็กเกจ", " ".join([title, headings, url_text])): _add(scores, evidence, "pricing_page", 4, "headings", "pricing_flag_with_metadata")
    if bool(get("has_contact_info")) and _match(r"contact|ติดต่อ|location|สาขา", " ".join([title, headings, url_text])): _add(scores, evidence, "contact_page", 4, "headings", "contact_flag_with_metadata")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True); detail, score = ranked[0]; second = ranked[1][1] if len(ranked) > 1 else 0
    fields = {item.split(":", 1)[0] for item in evidence.get(detail, [])}; gap = score - second
    tied = score > 0 and sum(candidate_score == score for _, candidate_score in ranked) > 1
    if detail == "unknown" or score < 4:
        confidence = "unknown"; detail = "unknown"
    elif tied or gap < 1:
        confidence = "low"; detail = "unknown"
    elif score >= 12 and gap >= 3 and len(fields & {"url", "title", "domain"}) >= 1:
        confidence = "high"
    elif score >= 6 and gap >= 1 and len(fields & {"url", "title", "domain", "structured_data"}) >= 1:
        confidence = "medium"
    else:
        confidence = "low"; detail = "unknown"
    reason = "; ".join(evidence.get(detail, [])) if detail != "unknown" else ("conflicting_top_scoring_rules" if tied else "insufficient_or_conflicting_general_taxonomy_evidence")
    def ev(field: str) -> str: return "; ".join(item for item in evidence.get(detail, []) if item.startswith(field + ":"))
    return GeneralPageResult(detail, DETAIL_TO_FAMILY[detail], float(score), confidence, ev("url"), ev("title"), ev("domain"), ev("headings"), ev("content"), reason)


def classify_general_page_type(row: pd.Series | dict[str, Any], include_content: bool = False) -> GeneralPageResult:
    return _classify(row, include_content)


def finalise_general_page_type(seed: GeneralPageResult, scraped: GeneralPageResult, content_quality_flag: str, content_strength: str = "") -> tuple[GeneralPageResult, str]:
    quality = _clean(content_quality_flag).casefold(); strength = _clean(content_strength).casefold()
    if seed.detail == "pdf_document": return seed, "pdf_rule"
    usable = quality == "ok" or strength in {"strong", "medium"}
    if usable and scraped.detail != "unknown" and scraped.confidence in {"high", "medium"} and (seed.detail == "unknown" or seed.confidence == "low" or scraped.score >= seed.score):
        return scraped, "scraped_content"
    if seed.detail != "unknown": return seed, "domain_rule" if seed.evidence_domain else "url_seed"
    return seed, "fallback_unknown"


def classify_general_site_type(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    url = _clean(get("source_url") or get("final_url") or get("normalized_url"))
    classified = _site_type(url, _clean(get("source_root_domain")), _clean(get("page_title") or get("source_title")), _clean(get("meta_description") or get("source_description")))
    if classified != "unknown":
        return classified
    # A pre-existing vertical source label is optional observable domain metadata,
    # not the main taxonomy. Map only its broad, unambiguous site role when it is
    # present; future topics can operate without this fallback.
    legacy = _clean(get("source_type_real_estate")).casefold()
    legacy_map = {
        "developer_official": "official_company_or_brand", "project_official": "official_company_or_brand",
        "broker_agency": "official_company_or_brand", "property_portal": "directory_or_listing_platform",
        "listing_marketplace": "marketplace_or_platform", "real_estate_media": "blog_or_content_site",
        "condo_review_site": "review_platform", "investment_content": "blog_or_content_site",
        "neighborhood_guide_site": "blog_or_content_site", "social_forum": "social_or_forum",
        "video_platform": "video_platform", "news_media": "news_media", "government_or_regulatory": "government",
        "pdf_document": "file_or_document_host", "map_or_transport_reference": "map_or_location_platform",
    }
    return legacy_map.get(legacy, "unknown")
