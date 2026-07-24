"""Real-estate taxonomy for the SCOPE condo topic.

This module is intentionally separate from the original medical-oriented page
type classifier.  It relies only on observable URL, domain, page metadata, and
scraped-page evidence; it never inspects citation outcomes, ranking, prompts,
or answer-derived fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import pandas as pd

from src.url_utils import root_domain


SOURCE_TYPE_REAL_ESTATE = (
    "developer_official",
    "project_official",
    "property_portal",
    "listing_marketplace",
    "broker_agency",
    "real_estate_media",
    "condo_review_site",
    "investment_content",
    "neighborhood_guide_site",
    "map_or_transport_reference",
    "social_forum",
    "video_platform",
    "news_media",
    "government_or_regulatory",
    "pdf_document",
    "unknown",
)

PAGE_TYPE_FAMILY_REAL_ESTATE = (
    "project_or_developer_page",
    "listing_or_marketplace_page",
    "article_or_review_page",
    "neighborhood_or_location_guide",
    "price_or_investment_page",
    "contact_or_sales_page",
    "social_or_forum_page",
    "document_or_pdf",
    "news_or_press_page",
    "map_or_transport_reference",
    "unknown",
)

PAGE_TYPE_DETAIL_REAL_ESTATE = (
    "condo_project_page",
    "developer_brand_page",
    "project_listing_page",
    "resale_listing_page",
    "rental_listing_page",
    "broker_property_page",
    "condo_review_page",
    "comparison_article",
    "buying_guide",
    "investment_guide",
    "price_market_report",
    "neighborhood_guide",
    "location_transport_page",
    "amenities_or_facilities_page",
    "floor_plan_page",
    "contact_sales_page",
    "promotion_page",
    "news_press_release",
    "forum_discussion",
    "video_page",
    "pdf_brochure",
    "unknown",
)

DETAIL_TO_FAMILY = {
    "condo_project_page": "project_or_developer_page",
    "developer_brand_page": "project_or_developer_page",
    "amenities_or_facilities_page": "project_or_developer_page",
    "floor_plan_page": "project_or_developer_page",
    "project_listing_page": "listing_or_marketplace_page",
    "resale_listing_page": "listing_or_marketplace_page",
    "rental_listing_page": "listing_or_marketplace_page",
    "broker_property_page": "listing_or_marketplace_page",
    "condo_review_page": "article_or_review_page",
    "comparison_article": "article_or_review_page",
    "buying_guide": "article_or_review_page",
    "investment_guide": "price_or_investment_page",
    "price_market_report": "price_or_investment_page",
    "neighborhood_guide": "neighborhood_or_location_guide",
    "location_transport_page": "neighborhood_or_location_guide",
    "contact_sales_page": "contact_or_sales_page",
    "promotion_page": "contact_or_sales_page",
    "forum_discussion": "social_or_forum_page",
    "video_page": "social_or_forum_page",
    "pdf_brochure": "document_or_pdf",
    "news_press_release": "news_or_press_page",
    "unknown": "unknown",
}

MEDICAL_STYLE_PAGE_TYPES = frozenset(
    {
        "article_health_info",
        "service_or_treatment_page",
        "department_or_center_page",
        "disease_condition_page",
        "doctor_profile",
        "treatment_page",
        "appointment_page",
    }
)

# Exact registrable-domain fragments are used before weaker text cues.  The
# list is deliberately transparent so additions can be reviewed alongside the
# audit rather than silently changing the taxonomy.
DEVELOPER_DOMAINS = (
    "sansiri",
    "apthai",
    "ap.co.th",
    "origin",
    "ananda",
    "scasset",
    "noble",
    "landandhouses",
    "lh.co.th",
    "qhouse",
    "pruksa",
    "assetwise",
    "plus.co.th",
    "plus",
    "major.co.th",
    "chewathai",
    "grandunity",
    "proudrealestate",
    "richmonts",
    "shangproperties",
    "frasersproperty",
    "sena",
)
PROJECT_DOMAINS = (
    "scope",
    "scopethonglor",
    "scopecollection",
    "langsuan",
    "muniq",
)
PROPERTY_PORTAL_DOMAINS = (
    "ddproperty",
    "fazwaz",
    "dotproperty",
    "propertyhub",
    "hipflat",
    "thailand-property",
    "zmyhome",
    "condo.com",
)
LISTING_MARKETPLACE_DOMAINS = (
    "kaidee",
    "livinginsider",
    "renthub",
    "bkkcondos",
    "thaiproperty",
    "thailandcondoshop",
    "condodee",
    "nestopa",
    "lazudi",
    "myproperty",
    "propertyscout",
)
BROKER_DOMAINS = (
    "superagent",
    "connex",
    "primarealty",
    "bangkokresidential",
    "amazingproperties",
    "9asset",
    "thaiprops",
    "thebkkresidence",
    "sellingbangkok",
    "investbangkokproperty",
    "panjapolproperty",
    "ownluxuryhomes",
    "passionaryestate",
    "lfsproperty",
    "embarkestate",
)
REVIEW_DOMAINS = (
    "thinkofliving",
    "condonewb",
    "condoreviewsthailand",
    "home.co.th",
    "terrabkk",
    "propholic",
    "baania",
    "estopolis",
    "condocontrol",
)
INVESTMENT_DOMAINS = (
    "jll",
    "cbre",
    "colliers",
    "savills",
    "knightfrank",
    "research.jll",
)
NEIGHBORHOOD_DOMAINS = (
    "bkkoracle",
    "whybangkok",
    "thailandstarterkit",
    "bangkokstarterkit",
    "siamconnect",
    "inyourpocket",
    "expatlife",
)
SOCIAL_DOMAINS = ("reddit", "pantip", "facebook", "lemon8", "instagram", "threads", "x.com")
VIDEO_DOMAINS = ("youtube", "youtu.be", "tiktok", "vimeo")
NEWS_DOMAINS = (
    "bangkokpost",
    "nationthailand",
    "reuters",
    "thetimes",
    "prnasia",
    "thaipr",
)
MAP_DOMAINS = ("google.com/maps", "maps.google", "bts.co.th", "mrta.co.th", "bangkokmetro")

# Exact root-domain overrides vetted from the recurring SCOPE condo sources.
# These are deliberately small: a generic-looking, rare, or mixed-purpose domain
# remains unknown rather than receiving a content-based guess.
SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES = {
    "108siam.com": ("listing_marketplace", "recurring condominium listings marketplace"),
    "bestbrandedresidences.com": ("real_estate_media", "recurring branded-residences market guide"),
    "bkkscene.com": ("neighborhood_guide_site", "recurring Bangkok neighborhood guide"),
    "brandedliving.co": ("real_estate_media", "recurring branded-residences editorial site"),
    "brightwillluxury.com": ("real_estate_media", "recurring luxury-residences editorial site"),
    "checkraka.com": ("property_portal", "recurring condominium information portal"),
    "hawook.com": ("neighborhood_guide_site", "recurring Bangkok property guide"),
    "herorealtor.com": ("broker_agency", "recurring realtor-branded domain"),
    "homefinderbangkok.com": ("broker_agency", "recurring Bangkok home-finder brokerage domain"),
    "housing.com": ("property_portal", "established property portal"),
    "officebangkok.com": ("broker_agency", "recurring Bangkok office-property brokerage domain"),
    "o-waw.com": ("neighborhood_guide_site", "recurring Bangkok property guide"),
    "reic.or.th": ("government_or_regulatory", "Real Estate Information Center official domain"),
    "theagent.co.th": ("broker_agency", "recurring property-agent domain"),
    "thefinestthai.com": ("real_estate_media", "recurring Thai luxury-property editorial site"),
    "thethaiger.com": ("news_media", "established Thai news publisher"),
    "varsoviaestate.com": ("broker_agency", "recurring estate-branded domain"),
}


@dataclass(frozen=True)
class RealEstatePageTypeResult:
    detail: str
    family: str
    score: float
    confidence: str
    evidence_url: str
    evidence_title: str
    evidence_domain: str
    evidence_content: str
    reason: str


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _normalise_url(url: str) -> str:
    raw = _clean(url)
    if raw and not re.match(r"^[a-z][a-z0-9+.-]*://", raw, flags=re.I):
        raw = "https://" + raw
    return raw


def _url_parts(url: str) -> tuple[str, str, str]:
    parsed = urlparse(_normalise_url(url))
    host = (parsed.hostname or "").casefold()
    path = unquote(parsed.path or "").casefold().replace("_", " ").replace("-", " ")
    query = unquote(parsed.query or "").casefold().replace("_", " ").replace("-", " ")
    return host, path, query


def _contains_domain(domain: str, candidates: tuple[str, ...]) -> bool:
    d = domain.casefold()
    return any(token in d for token in candidates)


def _matches(pattern: str, text: str) -> bool:
    return bool(pattern and re.search(pattern, text or "", flags=re.I | re.U))


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text or "", flags=re.I | re.U)) if pattern else 0


def is_real_estate_looking_url(url: str, domain: str = "") -> bool:
    """Conservative flag for diagnosing legacy medical labels on SCOPE URLs."""
    host, path, query = _url_parts(url)
    d = (domain or root_domain(url) or host).casefold()
    if classify_source_type_real_estate(url, d) != "unknown":
        return True
    text = " ".join([path, query])
    return _matches(
        r"condo|condominium|residence|property|real estate|apartment|villa|housing|"
        r"langsuan|chidlom|lumpini|sukhumvit|thon?glor|bts|mrt|คอนโด|โครงการ|อสังหา|"
        r"ขาย|เช่า|ทำเล|รถไฟฟ้า",
        text,
    )


def classify_source_type_real_estate(url: str, domain: str = "") -> str:
    """Classify a source into the SCOPE real-estate source taxonomy."""
    host, path, _query = _url_parts(url)
    root = (domain or root_domain(url) or host).casefold()
    suffix = PurePosixPath(urlparse(_normalise_url(url)).path).suffix.casefold()
    if suffix == ".pdf" or _matches(r"(?:^|[/?=&])pdf(?:$|[/?=&])", path):
        return "pdf_document"
    if root in SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES:
        return SAFE_SOURCE_TYPE_DOMAIN_OVERRIDES[root][0]
    if _contains_domain(root, VIDEO_DOMAINS):
        return "video_platform"
    if _contains_domain(root, SOCIAL_DOMAINS):
        return "social_forum"
    if root.endswith(".go.th") or ".go." in root or root.endswith(".gov") or ".gov." in root:
        return "government_or_regulatory"
    if _contains_domain(root, MAP_DOMAINS):
        return "map_or_transport_reference"
    if _contains_domain(root, NEWS_DOMAINS):
        return "news_media"
    if _contains_domain(root, REVIEW_DOMAINS):
        return "condo_review_site"
    if _contains_domain(root, INVESTMENT_DOMAINS):
        return "investment_content"
    if _contains_domain(root, NEIGHBORHOOD_DOMAINS):
        return "neighborhood_guide_site"
    if _contains_domain(root, BROKER_DOMAINS):
        return "broker_agency"
    if _contains_domain(root, PROPERTY_PORTAL_DOMAINS):
        return "property_portal"
    if _contains_domain(root, LISTING_MARKETPLACE_DOMAINS):
        return "listing_marketplace"
    if _contains_domain(root, PROJECT_DOMAINS):
        return "project_official"
    if _contains_domain(root, DEVELOPER_DOMAINS):
        return "developer_official"
    if _matches(r"(?:^|\.)news(?:\.|$)|press|journal|times", root):
        return "news_media"
    if _matches(r"(?:^|\.)forum(?:\.|$)|community", root):
        return "social_forum"
    if _matches(r"property|realty|realestate|condo", root) and not _matches(r"hotel|resort", root):
        return "broker_agency"
    return "unknown"


def _add_signal(
    scores: dict[str, float],
    evidence: dict[str, list[str]],
    detail: str,
    score: float,
    field: str,
    signal: str,
) -> None:
    scores[detail] = scores.get(detail, 0.0) + score
    evidence.setdefault(detail, []).append(f"{field}:{signal}")


def _add_pattern_signal(
    scores: dict[str, float],
    evidence: dict[str, list[str]],
    detail: str,
    pattern: str,
    text: str,
    field: str,
    weight: float,
    label: str,
) -> None:
    if _matches(pattern, text):
        _add_signal(scores, evidence, detail, weight, field, label)


def _evidence_for(evidence: dict[str, list[str]], detail: str, field: str) -> str:
    return "; ".join(item for item in evidence.get(detail, []) if item.startswith(field + ":"))


def _source_default_signals(
    source_type: str,
    scores: dict[str, float],
    evidence: dict[str, list[str]],
) -> None:
    if source_type == "video_platform":
        _add_signal(scores, evidence, "video_page", 12, "domain", "known_video_platform")
    elif source_type == "social_forum":
        _add_signal(scores, evidence, "forum_discussion", 12, "domain", "known_social_forum")
    elif source_type == "pdf_document":
        _add_signal(scores, evidence, "pdf_brochure", 12, "domain", "pdf_domain_or_extension")
    elif source_type == "news_media":
        _add_signal(scores, evidence, "news_press_release", 7, "domain", "known_news_domain")
    elif source_type == "map_or_transport_reference":
        _add_signal(scores, evidence, "location_transport_page", 10, "domain", "map_or_transport_domain")
    elif source_type == "condo_review_site":
        _add_signal(scores, evidence, "condo_review_page", 8, "domain", "known_condo_review_domain")
    elif source_type == "investment_content":
        _add_signal(scores, evidence, "price_market_report", 7, "domain", "known_market_research_domain")
    elif source_type == "neighborhood_guide_site":
        _add_signal(scores, evidence, "neighborhood_guide", 8, "domain", "known_neighborhood_guide_domain")
    elif source_type == "project_official":
        _add_signal(scores, evidence, "condo_project_page", 7, "domain", "known_project_official_domain")
    elif source_type == "developer_official":
        _add_signal(scores, evidence, "developer_brand_page", 7, "domain", "known_developer_domain")
    elif source_type in {"property_portal", "listing_marketplace"}:
        _add_signal(scores, evidence, "project_listing_page", 7, "domain", "known_listing_platform")
    elif source_type == "broker_agency":
        _add_signal(scores, evidence, "broker_property_page", 7, "domain", "known_broker_agency")


def _apply_common_rules(
    scores: dict[str, float],
    evidence: dict[str, list[str]],
    url_text: str,
    title: str,
    content: str,
    source_type: str,
    include_content: bool,
) -> None:
    fields = [("url", url_text, 5.0), ("title", title, 4.0)]
    if include_content:
        fields.append(("content", content[:12000], 1.0))
    editorial_route = _matches(r"/(?:blog|articles?|guides?|insights?|stories|content)(?:/|$)", url_text)

    def add(detail: str, pattern: str, label: str) -> None:
        for field, text, weight in fields:
            _add_pattern_signal(scores, evidence, detail, pattern, text, field, weight, label)

    add("pdf_brochure", r"\.pdf$|\bpdf\b|brochure|factsheet|e brochure|เอกสาร|โบรชัวร์", "pdf_or_brochure")
    add("contact_sales_page", r"contact|contact us|sales gallery|enquir|inquir|appointment|register|"
        r"book(?:ing)?|ติดต่อ|สอบถาม|นัดชม|ลงทะเบียน", "contact_or_sales")
    add("promotion_page", r"promotion|campaign|promo|discount|offer|launch offer|โปรโมชัน|โปรโมชั่น|ข้อเสนอ", "promotion")
    add("floor_plan_page", r"floor plan|unit plan|layout|master plan|แปลน|ผังห้อง|ผังโครงการ", "floor_plan")
    add("amenities_or_facilities_page", r"amenit(?:y|ies)|facilit(?:y|ies)|clubhouse|swimming pool|fitness|"
        r"ส่วนกลาง|สระว่ายน้ำ|ฟิตเนส", "amenities_or_facilities")
    add("location_transport_page", r"\bbts\b|\bmrt\b|airport rail link|station|transport|map|direction|"
        r"chidlom|chit lom|langsuan|lumpini|lumphini|sukhumvit|thon?glor|อโศก|ชิดลม|หลังสวน|ลุมพินี|สุขุมวิท|ทำเล|รถไฟฟ้า", "location_or_transport")
    add("neighborhood_guide", r"neighbou?rhood|area guide|where to live|district guide|living guide|"
        r"best areas|ย่าน|ทำเล|คู่มือ(?:การ)?อยู่อาศัย", "neighborhood_guide")
    add("price_market_report", r"market report|market outlook|price per sqm|price per sq\.?(?:m|meter)|"
        r"price benchmark|pricing|market price|valuation|transaction|ราคา(?:ขาย)?|ตารางราคา|ราคาต่อตร\.ม", "price_or_market_report")
    add("investment_guide", r"investment|investor|rental yield|yield|roi|capital gain|return on investment|"
        r"ลงทุน|ผลตอบแทน|อัตราผลตอบแทน|กำไร", "investment")
    add("comparison_article", r"\bvs\.?\b|compare|comparison|top \d+|best condo|best residences|"
        r"recommend(?:ation)?|เปรียบเทียบ|แนะนำ|อันดับ", "comparison_or_recommendation")
    add("buying_guide", r"buying guide|how to buy|first time buyer|buy condo|what to check|checklist|"
        r"purchase guide|วิธี(?:เลือก|ซื้อ)|ควรซื้อ|ซื้อคอนโด|เช็กลิสต์", "buying_guide")
    add("condo_review_page", r"condo review|project review|review(?:s)?|รีวิว(?:คอนโด|โครงการ)?", "condo_review")
    add("news_press_release", r"press release|press room|news|announcement|launches|ข่าว|ประกาศ", "news_or_press")

    # Listing specifics outweigh broad portal defaults.  The route/title text is
    # strongest because real-estate sites can host both listings and editorial pages.
    transaction_route = r"/(?:condo|property|apartment|residence)[ /]*(?:for )?"
    if not editorial_route:
        add("rental_listing_page", transaction_route + r"rent\b|/(?:rent|rental)\b|เช่า", "rental_listing")
        add("resale_listing_page", transaction_route + r"sale\b|/(?:sale|resale)\b|ขาย", "sale_listing")
    add("project_listing_page", r"new developments?|projects?|condos? for sale|search results?|"
        r"all condos?|listings?|โครงการใหม่|รวมคอนโด", "project_listing")
    add("broker_property_page", r"property details?|new developments?/condo|listing(?:s)?/|property/"
        r"(?:building|project)|อสังหาริมทรัพย์", "broker_or_property_listing")
    add("condo_project_page", r"/(?:project|condo|residence)(?:/|$)|/projects?/|"
        r"condo project|residence project|คอนโดมิเนียม|โครงการ", "condo_project")
    add("developer_brand_page", r"/about|/company|/our story|/developer|/brand|เกี่ยวกับ", "developer_brand")

    # An editorial route plus a real-estate cue is strong enough to classify as
    # article/review content even if the domain is not in the curated lists.
    real_estate_context = _matches(
        r"condo|condominium|residence|property|real estate|apartment|housing|"
        r"langsuan|chidlom|lumpini|sukhumvit|thon?glor|คอนโด|โครงการ|อสังหา|ขาย|เช่า",
        " ".join([url_text, title]),
    )
    if editorial_route and real_estate_context:
        _add_signal(scores, evidence, "condo_review_page", 8, "url", "editorial_real_estate_route")

    if _matches(r"price|pricing|market report|market outlook|ราคา|ตารางราคา", title):
        _add_signal(scores, evidence, "price_market_report", 6, "title", "explicit_price_or_market_context")

    if source_type in {"property_portal", "listing_marketplace"}:
        # A portal page with a project name but no sale/rent cue is a project page,
        # not an editorial review merely because its title contains "review".
        for field, text, weight in fields:
            _add_pattern_signal(
                scores,
                evidence,
                "condo_project_page",
                r"(?:/projects?/|/property/(?:project|building)/|condo for (?:sale|rent) at |information/review)",
                text,
                field,
                weight + 1,
                "portal_project_detail",
            )
        # Sale/rent signals describe the transaction itself and should win over
        # a location name such as Langsuan or Chidlom in the same listing title.
        listing_text = " ".join([url_text, title])
        if _matches(r"for rent|condo for rent|rental|\brent\b|เช่า", listing_text):
            _add_signal(scores, evidence, "rental_listing_page", 8, "domain", "portal_listing_context")
        if _matches(r"for sale|condo for sale|resale|\bsale\b|ขาย", listing_text):
            _add_signal(scores, evidence, "resale_listing_page", 8, "domain", "portal_listing_context")
    if source_type == "broker_agency":
        for field, text, weight in fields:
            _add_pattern_signal(
                scores,
                evidence,
                "broker_property_page",
                r"/(?:properties?|property|listing|condo|residence)(?:/|$)|new developments?|"
                r"\b(?:bedroom|sqm|sq\.m|unit)\b|ตร\.ม|ห้องนอน",
                text,
                field,
                weight,
                "broker_property_detail_context",
            )


def _confidence(best_score: float, second_score: float, evidence: dict[str, list[str]], detail: str) -> str:
    gap = best_score - second_score
    fields = {item.split(":", 1)[0] for item in evidence.get(detail, [])}
    strong_fields = len(fields & {"url", "title", "domain"})
    if best_score >= 10 and gap >= 2 and strong_fields >= 1:
        return "high"
    if best_score >= 7 and gap >= 1 and strong_fields >= 1:
        return "medium"
    return "low" if best_score >= 4 else "unknown"


def classify_real_estate_page_type(
    row: pd.Series | dict[str, Any],
    *,
    source_type: str = "",
    include_content: bool = False,
) -> RealEstatePageTypeResult:
    """Classify one page using the real-estate taxonomy.

    ``include_content=False`` creates the metadata/URL seed.  Content is only
    considered by the caller when its scrape-quality flag permits it.
    """
    get = row.get
    url = _clean(get("source_url") or get("final_url") or get("requested_url") or get("normalized_url"))
    domain = _clean(get("source_root_domain")) or root_domain(url)
    source_type = source_type or classify_source_type_real_estate(url, domain)
    host, path, query = _url_parts(url)
    url_text = " ".join([host, path, query])
    title = _clean(get("page_title") or get("source_title") or get("title"))
    meta = _clean(get("meta_description") or get("source_description") or get("description"))
    heading = _clean(get("h1_or_top_heading") or get("headings"))
    body = _clean(get("page_text") or get("text") or get("markdown"))
    content = " ".join(part for part in [meta, heading, body] if part)
    schema = _clean(get("structured_data_types")).casefold() if include_content else ""

    scores: dict[str, float] = {"unknown": 0.0}
    evidence: dict[str, list[str]] = {"unknown": []}
    _source_default_signals(source_type, scores, evidence)
    _apply_common_rules(scores, evidence, url_text, title, content, source_type, include_content)
    if schema:
        if "videoobject" in schema:
            _add_signal(scores, evidence, "video_page", 10, "structured_data", "videoobject")
        if any(token in schema for token in ("realestatelisting", "apartment", "accommodation", "product", "offer")):
            detail = "broker_property_page" if source_type == "broker_agency" else "project_listing_page"
            _add_signal(scores, evidence, detail, 7, "structured_data", "property_or_offer")
        if any(token in schema for token in ("article", "blogposting", "newsarticle")) and is_real_estate_looking_url(url, domain):
            detail = "news_press_release" if "newsarticle" in schema or source_type == "news_media" else "condo_review_page"
            _add_signal(scores, evidence, detail, 7, "structured_data", "real_estate_article")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_detail, best_score = ranked[0] if ranked else ("unknown", 0.0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = _confidence(best_score, second_score, evidence, best_detail)
    if best_detail == "unknown" or confidence in {"unknown", "low"}:
        if best_score <= 0:
            reason = "insufficient_real_estate_url_domain_or_metadata_evidence"
        elif confidence == "low":
            reason = "weak_or_conflicting_evidence; retained_as_unknown"
        else:
            reason = "insufficient_confident_evidence"
        best_detail = "unknown"
    else:
        reason = "; ".join(evidence.get(best_detail, [])) or "score_threshold_met"

    return RealEstatePageTypeResult(
        detail=best_detail,
        family=DETAIL_TO_FAMILY[best_detail],
        score=float(best_score),
        confidence=confidence,
        evidence_url=_evidence_for(evidence, best_detail, "url"),
        evidence_title=_evidence_for(evidence, best_detail, "title"),
        evidence_domain=_evidence_for(evidence, best_detail, "domain"),
        evidence_content=_evidence_for(evidence, best_detail, "content"),
        reason=reason,
    )


def finalise_real_estate_page_type(
    seed: RealEstatePageTypeResult,
    scraped: RealEstatePageTypeResult,
    content_quality_flag: str,
) -> tuple[RealEstatePageTypeResult, str]:
    """Select a safe final label without letting bad content erase a URL seed."""
    quality = _clean(content_quality_flag).casefold()
    if seed.detail == "pdf_brochure":
        return seed, "pdf_rule"
    if quality == "ok" and scraped.detail != "unknown" and scraped.confidence in {"high", "medium"}:
        return scraped, "scraped_content"
    if seed.detail != "unknown":
        source = "domain_rule" if seed.evidence_domain else "url_seed"
        return seed, source
    return seed, "fallback_unknown"
