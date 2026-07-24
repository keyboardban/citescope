from src.econometrics_eda_v2.general_page_taxonomy import (
    GENERAL_TAXONOMY_VERSION,
    classify_general_page_type,
    finalise_general_page_type,
)


def test_taxonomy_v2_does_not_use_hostname_as_page_route_evidence():
    result = classify_general_page_type(
        {
            "source_url": "https://amazingproperties.org/blogs/bangkok-condo-market-outlook",
            "page_title": "Bangkok Condo Market Outlook",
        }
    )
    assert GENERAL_TAXONOMY_VERSION == "general_page_taxonomy_v2"
    assert result.detail == "blog_article"


def test_floor_plan_article_is_not_pricing_page():
    result = classify_general_page_type(
        {
            "source_url": "https://example.com/blog/how-to-read-a-floor-plan",
            "page_title": "A Guide to Reading Floor Plans",
        }
    )
    assert result.detail in {"blog_article", "guide_article"}
    assert result.detail != "pricing_page"


def test_project_detail_route_reduces_unknowns_without_domain_override():
    result = classify_general_page_type(
        {
            "source_url": "https://www.9asset.com/en/condo-project/nimit-langsuan-1489",
            "page_title": "Nimit Langsuan",
        }
    )
    assert result.detail == "listing_page"
    assert result.confidence in {"high", "medium"}


def test_search_results_use_the_search_family():
    result = classify_general_page_type(
        {"source_url": "https://example.com/search?q=langsuan", "page_title": "Search results"}
    )
    assert result.detail == "search_results_page"
    assert result.family == "search_or_results"


def test_pagination_query_does_not_turn_article_into_search_results():
    result = classify_general_page_type(
        {
            "source_url": "https://example.com/blog/condo-guide?page=2&utm_source=chatgpt.com",
            "page_title": "Condo Guide",
        }
    )
    assert result.detail != "search_results_page"


def test_embedded_faq_schema_does_not_override_primary_project_route():
    row = {
        "source_url": "https://example.com/property/scope-langsuan",
        "page_title": "SCOPE Langsuan",
        "structured_data_types": "Product; FAQPage",
        "page_text_excerpt": "Project details and frequently asked questions.",
    }
    seed = classify_general_page_type(row, include_content=False)
    enriched = classify_general_page_type(row, include_content=True)
    final, _ = finalise_general_page_type(seed, enriched, "ok", "strong")
    assert final.detail in {"listing_page", "product_page"}
    assert final.detail != "faq_page"


def test_equal_strength_conflict_abstains_instead_of_alphabetical_tie_break():
    result = classify_general_page_type(
        {"source_url": "https://example.com/contact/pricing", "page_title": ""}
    )
    assert result.detail == "unknown"
    assert result.confidence == "low"
    assert result.reason == "conflicting_top_scoring_rules"


def test_explicit_faq_page_still_classifies_as_faq():
    result = classify_general_page_type(
        {
            "source_url": "https://example.com/faq",
            "page_title": "Frequently Asked Questions",
            "structured_data_types": "FAQPage",
        },
        include_content=True,
    )
    assert result.detail == "faq_page"
    assert result.confidence in {"high", "medium"}


def test_known_site_role_distinguishes_listing_from_official_project_page():
    portal = classify_general_page_type(
        {
            "source_url": "https://portal.example/en/project/scope-langsuan",
            "source_type_real_estate": "property_portal",
        }
    )
    official = classify_general_page_type(
        {
            "source_url": "https://developer.example/en/project/scope-langsuan",
            "source_type_real_estate": "developer_official",
        }
    )
    assert portal.detail == "listing_page"
    assert official.detail == "product_page"


def test_editorial_and_report_routes_have_explicit_page_functions():
    article = classify_general_page_type(
        {"source_url": "https://example.com/read/how-to-buy-condo"}
    )
    report = classify_general_page_type(
        {"source_url": "https://example.com/research/bangkok-market-report"}
    )
    assert article.detail == "blog_article"
    assert report.detail == "report_document"
