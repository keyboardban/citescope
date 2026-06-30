"""Structured **confounder framework** for the position-adjusted citation model.

The regression reports *controlled observational associations* between observable
source/page features and citation probability. This module reduces some
omitted-variable risk by:

1. **A registry** (`CONFOUNDER_REGISTRY`) cataloguing 32 confounders — their bias
   mechanism, whether CiteScope can observe them directly / only via a proxy / only
   with external data, and how each should be used (main control / sensitivity /
   diagnostic / caveat-only).
2. **Proxy derivation** (`derive_proxy_features`) — computes observable proxies from
   data already on hand (no extra scraping): CiteScope-observed domain/url visibility
   counts, URL semantics, prompt wording, language / local-relevance heuristics, and
   grouped content/trust scores. Every proxy is labelled a **proxy**, never the true
   construct.
3. **A confounder/proxy audit** (`confounder_audit`) — availability, proxy-quality,
   cited-vs-more-only balance, a proxy correlation matrix, proxy VIF, and an
   **unmeasured-confounders** note.

Hard rules (observational): proxies *reduce*, never *remove*, confounding — nothing
here proves causation. `more-only` = surfaced but not cited (never "rejected").
`source_position` = observable source panel position (never an internal AI / retrieval
/ Google rank). Answer-derived / post-output features (e.g. brand-in-answer) are
diagnostics only and never enter the effect-style model.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config

# --------------------------------------------------------------------------- #
# availability + role vocabularies
# --------------------------------------------------------------------------- #
DIRECT, HEURISTIC, PROXY, EXTERNAL, POST, NA = (
    "available_directly", "available_heuristic", "proxy_only",
    "external_required", "post_output_only", "not_currently_available")
MAIN, SENS, DIAG, POSTDIAG, CAVEAT = (
    "main_control", "sensitivity_control", "diagnostic_only",
    "post_output_diagnostic_only", "caveat_only")
_AVAILABLE_NOW = {DIRECT, HEURISTIC}

# --------------------------------------------------------------------------- #
# keyword lexicons (bilingual EN / TH where it matters for the Thai audits)
# --------------------------------------------------------------------------- #
_KW = {
    "price": ["price", "cost", "fee", "cheap", "package", "promotion", "discount", "ราคา", "ค่า", "บาท", "แพ็กเกจ", "โปรโมชั่น"],
    "official": ["official", "authority", "government", "gov", "ministry", "ทางการ", "ราชการ", "กระทรวง"],
    "safety": ["safe", "safety", "risk", "warning", "danger", "side effect", "ปลอดภัย", "อันตราย", "ความเสี่ยง", "ผลข้างเคียง", "เตือน"],
    "local": ["near me", "nearby", "bangkok", "thailand", "province", "district", "location", "ใกล้ฉัน", "กรุงเทพ", "ไทย", "จังหวัด", "อำเภอ", "ที่ตั้ง"],
    "medical": ["doctor", "hospital", "clinic", "treatment", "symptom", "disease", "medicine", "drug", "หมอ", "แพทย์", "โรงพยาบาล", "คลินิก", "รักษา", "อาการ", "โรค", "ยา"],
    "recommendation": ["best", "recommend", "top", "review", "should i", "worth", "ดีที่สุด", "แนะนำ", "รีวิว", "ควร"],
    "comparison": ["vs", "versus", "compare", "comparison", "better", "difference", "เปรียบเทียบ", "ดีกว่า", "ต่างกัน"],
}
_URL_PRODUCT = ["product", "/p/", "/dp/", "item", "sku", "shop", "store", "buy", "cart", "/pd/", "สินค้า"]
_URL_SERVICE = ["service", "booking", "appointment", "treatment", "clinic", "บริการ", "นัด", "รักษา"]
_MARKETPLACES = {"lazada": "lazada", "shopee": "shopee", "amazon": "amazon", "aliexpress": "aliexpress",
                 "jd.co.th": "jd_central", "central.co.th": "central", "konvy": "konvy", "ebay": "ebay"}
_THAI_RE = re.compile(r"[฀-๿]")
_THAILAND_TERMS = ["thai", "thailand", "ไทย", "ประเทศไทย"]


def _text(s) -> str:
    return "" if s is None else str(s)


def _has_any(text: str, words: list[str]) -> int:
    t = text.lower()
    return int(any(w in t for w in words))


def _thai_ratio(text: str) -> float:
    t = _text(text)
    if not t:
        return 0.0
    th = len(_THAI_RE.findall(t))
    letters = sum(1 for ch in t if ch.isalpha())
    return round(th / letters, 3) if letters else 0.0


# --------------------------------------------------------------------------- #
# 1) the registry (32 confounders)
# --------------------------------------------------------------------------- #
def _c(name, category, bias, status, columns, model_role, diag_role, *, proxy=True, external=False, caveat=""):
    return {"confounder": name, "category": category, "bias_mechanism": bias,
            "availability_status": status, "available_now": status in _AVAILABLE_NOW,
            "proxy_possible": bool(proxy), "requires_external_data": bool(external),
            "recommended_columns": list(columns), "model_role": model_role,
            "diagnostic_role": diag_role, "caveat": caveat}


_PROXY_LABEL = "Observable proxy only — labelled a proxy, not the true construct."
CONFOUNDER_REGISTRY: list[dict] = [
    _c("writing_quality", "content", "Better-written pages may be both more answer-ready and more cited.",
       PROXY, ["readability_score", "heading_density", "has_clear_headings", "answer_directness_score",
               "content_completeness_score", "answer_ready_score"], SENS, DIAG,
       caveat="True writing quality is unmeasured; scores are noisy proxies. " + _PROXY_LABEL),
    _c("domain_authority", "authority", "Authoritative domains may be surfaced higher and cited more.",
       PROXY, ["domain_seen_count", "domain_cited_count", "domain_citation_rate", "domain_avg_source_position",
               "institutional_official", "brand_official_candidate", "source_type"], SENS, DIAG, external=True,
       caveat="CiteScope-observed domain visibility, NOT external SEO authority (backlinks/DA)."),
    _c("brand_authority", "authority", "Strong brands may be surfaced and cited more, regardless of page content.",
       PROXY, ["brand_appeared_in_sources", "brand_cited_count", "brand_citation_rate", "brand_source_diversity",
               "brand_domain_count"], SENS, DIAG, external=True,
       caveat="Observed brand visibility, not true brand search demand/reputation."),
    _c("source_panel_placement", "placement", "Higher panel placement strongly tracks citation; may be a mediator.",
       DIRECT, ["source_position", "log1p_source_position", "position_band"], MAIN, DIAG, proxy=False,
       caveat="Observable source panel position — NOT an internal AI / retrieval / Google rank. "
              "May be downstream of relevance / source-panel construction; run with and without it."),
    _c("scrape_success", "selection", "Scrape failure may correlate with both page type and citation (selection).",
       DIRECT, ["scrape_success", "scrape_status", "content_length_scraped"], SENS, DIAG, proxy=False,
       caveat="A selection indicator, not a simple quality feature. Use mainly as a diagnostic/sensitivity control."),
    _c("page_type", "page_role", "Page type (article/contact/product) tracks both content features and citation.",
       HEURISTIC, ["page_type"], MAIN, DIAG, proxy=False,
       caveat="Heuristic classification; interpret coefficients relative to the omitted reference page type."),
    _c("index_history", "history", "Longer-indexed pages may accrue authority and be cited more.",
       PROXY, ["citescope_visibility_history_score", "url_seen_count", "domain_seen_count",
               "days_since_url_first_seen", "page_age_days"], SENS, DIAG, external=True,
       caveat="CiteScope-observed visibility history, NOT true search-engine index history "
              "(use citescope_visibility_history_score, never index_history_score)."),
    _c("prompt_intent", "prompt", "Different intents surface and cite different source/page types.",
       DIRECT, ["intent"], MAIN, DIAG, proxy=False,
       caveat="From the manifest; interpret intent dummies relative to the reference intent."),
    _c("prompt_wording", "prompt", "Wording (price/official/safety terms) steers which sources are surfaced/cited.",
       HEURISTIC, ["prompt_length", "prompt_has_price_terms", "prompt_has_official_terms", "prompt_has_safety_terms",
                   "prompt_has_local_terms", "prompt_has_medical_terms", "prompt_has_recommendation_terms",
                   "prompt_has_comparison_terms", "prompt_has_brand_terms"], SENS, DIAG,
       caveat="Keyword heuristics over the prompt text."),
    _c("language_match", "language", "Language alignment between prompt and page may drive citation.",
       HEURISTIC, ["prompt_language", "page_language", "language_match", "thai_content_ratio", "english_content_ratio"],
       SENS, DIAG, caveat="Simple character-ratio language heuristics."),
    _c("local_country_relevance", "local", "Local relevance (Thai domain/address) matters for local prompts.",
       HEURISTIC, ["is_thai_domain", "has_thai_address", "has_thai_phone_number", "country_relevance_score",
                   "tld_country_hint", "contains_thailand_terms"], SENS, DIAG,
       caveat="Heuristic local-relevance proxy; important for Thai local/healthcare audits."),
    _c("source_subtype", "source_role", "Granular source subtype (public vs private hospital) confounds source_type.",
       HEURISTIC, ["source_subtype", "government_health_authority", "public_hospital", "private_hospital",
                   "marketplace", "review_blog", "news_media", "directory", "medical_reference"], SENS, DIAG,
       caveat="Supplements (does not replace) source_type; heuristic from domain/url/title."),
    _c("page_purpose", "page_role", "Page purpose (answer vs navigation vs policy) tracks citation readiness.",
       HEURISTIC, ["is_answer_page", "is_transactional_page", "is_navigation_page", "is_policy_page",
                   "is_directory_page", "is_thin_contact_page", "is_product_page", "is_article_page"], SENS, DIAG,
       caveat="Heuristic page-purpose proxies from URL/title/headings."),
    _c("content_completeness", "content", "Complete pages may carry FAQ/price/contact AND be cited more.",
       HEURISTIC, ["num_distinct_sections", "content_completeness_score", "answer_ready_score",
                   "has_usage_info", "has_safety_warning", "has_where_to_buy"], SENS, DIAG,
       caveat="Reduces bias where single content flags are really proxies for overall completeness."),
    _c("content_freshness", "freshness", "Freshness/update quality may proxy authority or evergreen value.",
       DIRECT, ["freshness_days", "page_age_days", "days_since_updated", "has_published_date",
                "has_updated_date", "published_updated_gap_days"], SENS, DIAG, proxy=False,
       caveat="If older pages look positive, read as authority/index-history/evergreen — not 'make pages old'."),
    _c("structured_data_schema", "metadata", "Schema markup may aid surfacing and correlate with professional sites.",
       NA, ["has_schema", "has_product_schema", "has_faq_schema", "has_article_schema", "schema_type_count"],
       DIAG, DIAG, external=True, caveat="Needs raw HTML; skipped gracefully when unavailable."),
    _c("metadata_snippet_quality", "metadata", "Title/description quality affects surfacing and click/cite behaviour.",
       DIRECT, ["has_title", "title_length", "has_meta_description", "meta_description_length",
                "title_prompt_similarity", "description_prompt_similarity", "title_contains_intent_terms"],
       MAIN, DIAG, proxy=False, caveat="Use PROMPT-based metadata only; never answer-derived overlap, in the main model."),
    _c("page_accessibility", "access", "Unloadable/JS-heavy/noindex pages may be scraped + cited differently.",
       PROXY, ["http_status", "redirect_count", "js_heavy_page", "has_noindex", "robots_blocked",
               "content_length_scraped", "loadability_warning"], SENS, DIAG, external=True,
       caveat="Partly needs HTTP/HTML signals; otherwise proxied by scrape_status/length."),
    _c("url_stability_canonical", "url", "Tracking params / http-https / trailing-slash dupes distort dedup + counts.",
       DIRECT, ["normalized_url", "canonical_url", "has_tracking_params", "http_https_duplicate",
               "trailing_slash_duplicate", "canonical_duplicate_count", "duplicate_url_count"], DIAG, DIAG, proxy=False,
       caveat="Used in dedup diagnostics; canonicalize before interpreting visibility counts."),
    _c("commerciality", "commercial", "Transactional pages may be surfaced/cited differently by intent.",
       HEURISTIC, ["commercial_page_score", "has_add_to_cart", "has_shop_button", "has_delivery_info",
                   "has_stock_status", "has_discount_terms", "has_price_or_package"], SENS, DIAG,
       caveat="Use with intent interactions; price being positive ≠ price causes citation."),
    _c("medical_safety_sensitivity", "safety", "Safety-sensitive prompts may favour official/medical sources.",
       HEURISTIC, ["prompt_safety_sensitivity_score", "page_has_medical_disclaimer", "has_doctor_review",
                   "has_warning_section", "has_regulatory_reference"], SENS, DIAG,
       caveat="Especially relevant for Safety/Risk intents in healthcare audits."),
    _c("brand_mention_in_answer", "post_output", "Appearing in the answer is downstream of citation — circular.",
       POST, ["brand_appeared_in_answer"], POSTDIAG, POSTDIAG, proxy=False,
       caveat="POST-OUTPUT. Never in the effect-style model; post-output diagnostic only."),
    _c("search_query_reconstruction", "query", "The reconstructed web_search_query shapes which sources surface.",
       HEURISTIC, ["web_search_query", "query_prompt_similarity", "query_has_brand_terms", "query_has_local_terms",
                   "query_has_official_terms", "query_has_price_terms", "query_language"], SENS, DIAG,
       caveat="Do not assume it is the full internal query if not clearly available."),
    _c("source_availability_selection", "selection", "We observe only surfaced sources, not the full candidate set.",
       PROXY, ["url_seen_count", "domain_seen_count", "prompt_seen_frequency", "source_position"], CAVEAT, DIAG,
       external=True, caveat="The model estimates citation WITHIN surfaced sources, not retrieval from the whole web."),
    _c("competitor_ecosystem", "competition", "Competitor presence/diversity shifts a brand's citation odds.",
       HEURISTIC, ["competitor_domain_count", "competitor_source_count", "competitor_cited_count",
                   "brand_source_diversity", "competitor_content_type_diversity"], DIAG, DIAG,
       caveat="Computed in the brand-visibility layer; diagnostic for brand audits."),
    _c("internal_linking_architecture", "site", "Site architecture/internal links affect crawl + surfacing.",
       EXTERNAL, ["internal_link_count", "breadcrumb_present", "url_path_depth", "navigation_depth_proxy"],
       DIAG, DIAG, external=True, caveat="Needs page HTML; url_path_depth is the only no-HTML proxy."),
    _c("url_semantic_clarity", "url", "Readable, keyworded URLs may correlate with quality and surfacing.",
       HEURISTIC, ["url_path_depth", "url_contains_product_terms", "url_contains_service_terms",
                   "url_contains_thai_slug", "url_is_numeric_id", "url_readability_score"], SENS, DIAG,
       caveat="Derived from the URL string only."),
    _c("trust_signal_bundle", "authority", "Author/reviewer/policy signals bundle into perceived trust.",
       HEURISTIC, ["has_author", "has_reviewer", "has_about_us_link", "has_privacy_policy",
                   "has_credentials", "has_medical_reviewer", "trust_signal_score"], SENS, DIAG,
       caveat="Prefer the grouped trust_signal_score + rare-feature diagnostics when signals are sparse."),
    _c("marketplace_platform_authority", "authority", "Marketplace listings inherit platform authority.",
       HEURISTIC, ["is_marketplace", "platform_name", "is_official_store_on_marketplace", "platform_authority_proxy"],
       SENS, DIAG, caveat="Detected from domain; control when analysing product/commerce topics."),
    _c("content_duplication", "content", "Duplicated content may be cited as the canonical vs ignored copies.",
       PROXY, ["content_hash", "near_duplicate_cluster_id", "duplicate_content_count", "is_original_source_proxy"],
       DIAG, DIAG, external=True, caveat="Needs page text; diagnostic unless confident."),
    _c("run_temporal_volatility", "temporal", "Results shift across runs/time; pooling mixes regimes.",
       DIRECT, ["run_date", "run_hour", "day_of_week", "batch_id", "run_id"], SENS, DIAG, proxy=False,
       caveat="Use as a control or cluster variable when multiple runs exist."),
    _c("model_backend_version", "system", "Mixing model/backend versions mixes different selection processes.",
       HEURISTIC, ["model_name", "model_version", "search_mode", "brightdata_job_time"], CAVEAT, DIAG,
       caveat="Control/caveat when multiple models or backend modes are mixed."),
]
_REGISTRY_BY_NAME = {c["confounder"]: c for c in CONFOUNDER_REGISTRY}


def registry_rows() -> list[dict]:
    """CSV-ready master table (`econometrics_confounder_registry.csv`)."""
    return [{
        "confounder": c["confounder"], "category": c["category"], "bias_mechanism": c["bias_mechanism"],
        "availability_status": c["availability_status"], "available_now": c["available_now"],
        "proxy_possible": c["proxy_possible"], "requires_external_data": c["requires_external_data"],
        "recommended_columns": "; ".join(c["recommended_columns"]),
        "model_role": c["model_role"], "diagnostic_role": c["diagnostic_role"], "caveat": c["caveat"],
    } for c in CONFOUNDER_REGISTRY]


# --------------------------------------------------------------------------- #
# 2) proxy feature derivation (no extra scraping; defensive)
# --------------------------------------------------------------------------- #
# Tiers used by the confounder-aware sensitivity models E–H (controls, not focal).
PROMPT_WORDING_FEATURES = ["prompt_length", "prompt_has_price_terms", "prompt_has_official_terms",
                           "prompt_has_safety_terms", "prompt_has_local_terms", "prompt_has_medical_terms",
                           "prompt_has_recommendation_terms", "prompt_has_comparison_terms"]
LANG_LOCAL_FEATURES = ["is_thai_domain", "contains_thailand_terms", "country_relevance_score",
                       "language_match", "thai_content_ratio"]
COMPLETENESS_FEATURES = ["content_completeness_score", "answer_ready_score", "trust_signal_score"]
VISIBILITY_HISTORY_FEATURES = ["domain_seen_count", "domain_citation_rate", "domain_avg_source_position",
                               "url_seen_count", "citescope_visibility_history_score"]
META_ACCESS_FEATURES = ["title_length", "content_length_scraped", "has_schema", "url_path_depth", "is_marketplace"]

# Proxy features used by the audit (balance / correlation / VIF). Numeric/boolean only.
_PROXY_NUMERIC = ["domain_seen_count", "domain_cited_count", "domain_citation_rate", "domain_avg_source_position",
                  "domain_seen_prompt_count", "domain_seen_run_count", "url_seen_count", "url_seen_prompt_count",
                  "url_path_depth", "citescope_visibility_history_score", "prompt_length", "country_relevance_score",
                  "thai_content_ratio", "trust_signal_score", "content_completeness_score", "answer_ready_score",
                  "commercial_page_score", "title_length"]
_PROXY_BOOL = ["has_tracking_params", "url_is_numeric_id", "url_contains_product_terms", "url_contains_service_terms",
               "url_contains_thai_slug", "is_thai_domain", "contains_thailand_terms", "language_match",
               "is_marketplace"] + PROMPT_WORDING_FEATURES[1:]
DERIVED_PROXY_FEATURES = _PROXY_NUMERIC + _PROXY_BOOL


def _norm(s: pd.Series) -> pd.Series:
    """Min–max normalise to 0..1 (constant column → 0)."""
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def derive_proxy_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Add observable confounder PROXY columns where the source data allows. Returns
    (augmented_df, notes). Each note records a derivation step + whether it ran or was
    skipped (missing columns) — nothing raises."""
    out = df.copy()
    notes: list[dict] = []
    derived: list[str] = []

    def note(step, ok, detail=""):
        notes.append({"step": step, "status": "derived" if ok else "skipped", "detail": detail})

    def has(*cols):
        return all(c in out.columns for c in cols)

    # --- E. domain authority / index-history proxies (CiteScope-observed visibility) ---
    if has("domain"):
        dom = out["domain"].astype("string").fillna("∅")
        cited = pd.to_numeric(out.get("cited"), errors="coerce") if "cited" in out.columns else None
        out["domain_seen_count"] = dom.map(dom.value_counts()).astype(float)
        if cited is not None:
            out["domain_cited_count"] = cited.groupby(dom).transform("sum")
            out["domain_citation_rate"] = cited.groupby(dom).transform("mean")
        if "source_position" in out.columns:
            out["domain_avg_source_position"] = (pd.to_numeric(out["source_position"], errors="coerce")
                                                 .groupby(dom).transform("mean"))
        if "record_id" in out.columns:
            out["domain_seen_prompt_count"] = out.groupby(dom)["record_id"].transform("nunique").astype(float)
        if "run_id" in out.columns:
            out["domain_seen_run_count"] = out.groupby(dom)["run_id"].transform("nunique").astype(float)
        derived += [c for c in ("domain_seen_count", "domain_cited_count", "domain_citation_rate",
                                "domain_avg_source_position", "domain_seen_prompt_count", "domain_seen_run_count")
                    if c in out.columns]
        note("domain_visibility_proxies", True, "CiteScope-observed domain visibility (NOT external SEO authority).")
    else:
        note("domain_visibility_proxies", False, "no `domain` column.")

    # --- G. url visibility (repeat-measurement / index-history proxy) ---
    url_key = "normalized_url" if "normalized_url" in out.columns else ("url" if "url" in out.columns else None)
    if url_key:
        u = out[url_key].astype("string").fillna("∅")
        out["url_seen_count"] = u.map(u.value_counts()).astype(float)
        if "record_id" in out.columns:
            out["url_seen_prompt_count"] = out.groupby(u)["record_id"].transform("nunique").astype(float)
        if "run_id" in out.columns:
            out["url_seen_run_count"] = out.groupby(u)["run_id"].transform("nunique").astype(float)
        derived += [c for c in ("url_seen_count", "url_seen_prompt_count", "url_seen_run_count") if c in out.columns]
        if "domain_seen_count" in out.columns:
            out["citescope_visibility_history_score"] = (
                0.5 * _norm(np.log1p(out["domain_seen_count"])) + 0.5 * _norm(np.log1p(out["url_seen_count"]))).round(4)
            derived.append("citescope_visibility_history_score")
        note("url_visibility_proxies", True, "CiteScope-observed visibility history (NOT true index history).")
    else:
        note("url_visibility_proxies", False, "no url/normalized_url column.")

    # --- Q / X. URL stability + semantic clarity (from the URL string) ---
    if "url" in out.columns:
        url = out["url"].astype("string").fillna("")
        low = url.str.lower()
        path = low.str.replace(r"^https?://[^/]+", "", regex=True).str.replace(r"\?.*$", "", regex=True)
        out["url_path_depth"] = path.str.count("/").astype(float)
        out["has_tracking_params"] = low.str.contains(r"utm_|gclid|fbclid", regex=True, na=False).astype(float)
        out["url_is_numeric_id"] = path.str.contains(r"/\d{4,}(?:/|$)", regex=True, na=False).astype(float)
        out["url_contains_product_terms"] = low.apply(lambda t: _has_any(t, _URL_PRODUCT)).astype(float)
        out["url_contains_service_terms"] = low.apply(lambda t: _has_any(t, _URL_SERVICE)).astype(float)
        out["url_contains_thai_slug"] = path.apply(lambda t: int(bool(_THAI_RE.search(t)))).astype(float)
        out["is_marketplace"] = low.apply(lambda t: int(any(k in t for k in _MARKETPLACES))).astype(float)
        out["platform_name"] = low.apply(lambda t: next((v for k, v in _MARKETPLACES.items() if k in t), ""))
        derived += ["url_path_depth", "has_tracking_params", "url_is_numeric_id", "url_contains_product_terms",
                    "url_contains_service_terms", "url_contains_thai_slug", "is_marketplace"]
        note("url_semantic_proxies", True)
    else:
        note("url_semantic_proxies", False, "no `url` column.")

    # --- H. prompt wording flags ---
    if "prompt" in out.columns:
        pr = out["prompt"].astype("string").fillna("")
        out["prompt_length"] = pr.str.len().astype(float)
        for key in ("price", "official", "safety", "local", "medical", "recommendation", "comparison"):
            out[f"prompt_has_{key}_terms"] = pr.apply(lambda t, k=key: _has_any(t, _KW[k])).astype(float)
        out["prompt_safety_sensitivity_score"] = (
            out["prompt_has_safety_terms"] + out["prompt_has_medical_terms"]).clip(0, 2) / 2.0
        derived += ["prompt_length"] + [f"prompt_has_{k}_terms" for k in
                                        ("price", "official", "safety", "local", "medical", "recommendation", "comparison")]
        note("prompt_wording_proxies", True)
    else:
        note("prompt_wording_proxies", False, "no `prompt` column.")

    # --- I / J. language + local-country relevance ---
    if "title" in out.columns or "prompt" in out.columns:
        title = out["title"].astype("string").fillna("") if "title" in out.columns else pd.Series("", index=out.index)
        pr = out["prompt"].astype("string").fillna("") if "prompt" in out.columns else pd.Series("", index=out.index)
        out["thai_content_ratio"] = title.apply(_thai_ratio)
        out["english_content_ratio"] = (1.0 - out["thai_content_ratio"]).round(3)
        out["page_language"] = np.where(out["thai_content_ratio"] >= 0.3, "th", "en")
        out["prompt_language"] = np.where(pr.apply(_thai_ratio) >= 0.3, "th", "en")
        out["language_match"] = (out["page_language"] == out["prompt_language"]).astype(float)
        out["contains_thailand_terms"] = (pr + " " + title).str.lower().apply(
            lambda t: _has_any(t, _THAILAND_TERMS)).astype(float)
        derived += ["thai_content_ratio", "language_match", "contains_thailand_terms"]
        note("language_proxies", True, "character-ratio language heuristic.")
    else:
        note("language_proxies", False, "no title/prompt column.")

    if "domain" in out.columns:
        dl = out["domain"].astype("string").fillna("").str.lower()
        out["is_thai_domain"] = (dl.str.endswith(".th") | dl.apply(lambda t: int(bool(_THAI_RE.search(t))).__bool__())).astype(float)
        out["tld_country_hint"] = dl.apply(lambda t: "TH" if t.endswith(".th") else ("" if "." not in t else t.rsplit(".", 1)[-1].upper()))
        local_bits = [out.get("is_thai_domain"), out.get("contains_thailand_terms"), out.get("url_contains_thai_slug")]
        local_bits = [b for b in local_bits if b is not None]
        if local_bits:
            out["country_relevance_score"] = (sum(local_bits) / len(local_bits)).round(3)
            derived.append("country_relevance_score")
        derived.append("is_thai_domain")
        note("local_relevance_proxies", True)
    else:
        note("local_relevance_proxies", False, "no `domain` column.")

    # --- D / L / Y. grouped content / trust scores from existing content booleans ---
    def _score(name, cols, label):
        present = [c for c in cols if c in out.columns]
        if len(present) >= 2:
            out[name] = out[present].apply(lambda c: pd.to_numeric(c, errors="coerce")).mean(axis=1).round(3)
            derived.append(name)
            note(label, True, f"mean of {len(present)} signals: {', '.join(present)}.")
        else:
            note(label, False, f"need ≥2 of {cols}.")

    _score("content_completeness_score",
           ["has_faq", "has_table", "has_bullets", "has_many_headings", "has_step_by_step",
            "has_price_or_package", "has_contact_info", "has_location_info"], "content_completeness_score")
    _score("answer_ready_score",
           ["has_faq", "has_step_by_step", "has_many_headings", "has_table", "heading_prompt_match",
            "has_price_or_package", "has_contact_info"], "answer_ready_score")
    _score("trust_signal_score",
           ["has_author", "has_reviewer", "has_schema", "has_published_date", "has_updated_date"], "trust_signal_score")
    _score("commercial_page_score",
           ["has_price_or_package", "product_page", "url_contains_product_terms", "is_marketplace"], "commercial_page_score")

    # --- O. metadata length proxies ---
    if "title" in out.columns:
        out["title_length"] = out["title"].astype("string").fillna("").str.len().astype(float)
        derived.append("title_length")
    if "description" in out.columns:
        out["meta_description_length"] = out["description"].astype("string").fillna("").str.len().astype(float)
        derived.append("meta_description_length")
    if "char_count" in out.columns:
        out["content_length_scraped"] = pd.to_numeric(out["char_count"], errors="coerce")
        derived.append("content_length_scraped")

    return out, [{**n} for n in notes] + [{"step": "_derived_columns", "status": "info",
                                           "detail": "; ".join(sorted(set(derived)))}]


# --------------------------------------------------------------------------- #
# 3) confounder / proxy audit (the 7 diagnostic outputs)
# --------------------------------------------------------------------------- #
_PROXY_QUALITY = {
    "domain_authority": "moderate_proxy", "index_history": "moderate_proxy", "writing_quality": "weak_proxy",
    "brand_authority": "weak_proxy", "content_completeness": "moderate_proxy", "trust_signal_bundle": "moderate_proxy",
    "url_semantic_clarity": "weak_proxy", "local_country_relevance": "moderate_proxy", "language_match": "moderate_proxy",
    "prompt_wording": "strong_proxy", "commerciality": "moderate_proxy", "marketplace_platform_authority": "moderate_proxy",
    "structured_data_schema": "external_required", "page_accessibility": "external_required",
    "internal_linking_architecture": "external_required", "content_duplication": "external_required",
    "brand_mention_in_answer": "post_output_only", "source_availability_selection": "weak_proxy",
}


def _is_binary(s: pd.Series) -> bool:
    vals = set(np.unique(pd.to_numeric(s, errors="coerce").dropna().values).tolist())
    return bool(vals) and vals.issubset({0.0, 1.0})


def _feature_availability_rows(df: pd.DataFrame, derived_set: set) -> list[dict]:
    rows = []
    n = len(df)
    for c in CONFOUNDER_REGISTRY:
        for feat in c["recommended_columns"]:
            present = feat in df.columns
            miss = round(float(pd.to_numeric(df[feat], errors="coerce").isna().mean()), 3) if present else 1.0
            rows.append({"feature": feat, "confounder": c["confounder"], "available": present,
                         "source_column": feat if present else "", "derived": feat in derived_set,
                         "missing_rate": miss,
                         "notes": ("post-output — diagnostic only" if c["availability_status"] == POST else
                                   ("needs external/HTML data" if c["requires_external_data"] and not present else ""))})
    return rows


def _proxy_summary_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for c in CONFOUNDER_REGISTRY:
        avail = [f for f in c["recommended_columns"] if f in df.columns]
        if c["availability_status"] in (DIRECT,) and not c["proxy_possible"]:
            continue  # directly observed, not a proxy
        quality = _PROXY_QUALITY.get(c["confounder"],
                                     "external_required" if c["requires_external_data"] else "moderate_proxy")
        if not avail and quality not in ("external_required", "post_output_only"):
            quality = "external_required" if c["requires_external_data"] else "weak_proxy"
        rows.append({"confounder": c["confounder"], "proxy_features": "; ".join(avail) or "(none available)",
                     "proxy_quality": quality, "requires_external_data": c["requires_external_data"],
                     "recommended_interpretation": c["caveat"]})
    return rows


def _balance_rows(df: pd.DataFrame, labels: dict | None = None) -> list[dict]:
    labels = labels or {}
    if "cited" not in df.columns:
        return []
    y = pd.to_numeric(df["cited"], errors="coerce")
    feats = [f for f in DERIVED_PROXY_FEATURES if f in df.columns]
    rows = []
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        if s.notna().sum() < 5:
            continue
        c1, c0 = s[y == 1], s[y == 0]
        if not len(c1) or not len(c0):
            continue
        m1, m0 = float(c1.mean()), float(c0.mean())
        miss = round(float(s.isna().mean()), 3)
        spread = float(s.std(ddof=0)) or 1.0
        warn = ""
        if abs(m1 - m0) / spread >= 0.5:
            warn = "Large cited vs more-only imbalance — a candidate confounder; include as a sensitivity control."
        if miss >= 0.4:
            warn = (warn + " " if warn else "") + "High missingness — interpret balance cautiously."
        rows.append({"confounder_proxy": labels.get(f, f), "feature": f,
                     "kind": "proportion" if _is_binary(s) else "mean",
                     "cited_value": round(m1, 4), "more_only_value": round(m0, 4),
                     "difference": round(m1 - m0, 4), "missing_rate": miss, "warning": warn})
    return rows


def _correlation_matrix_rows(df: pd.DataFrame) -> list[dict]:
    feats = [f for f in _PROXY_NUMERIC if f in df.columns
             and pd.to_numeric(df[f], errors="coerce").notna().sum() > 3
             and pd.to_numeric(df[f], errors="coerce").dropna().nunique() > 1]
    if len(feats) < 2:
        return []
    corr = df[feats].apply(lambda c: pd.to_numeric(c, errors="coerce")).corr().round(3)
    return [{"feature": f, **{g: (None if pd.isna(corr.loc[f, g]) else float(corr.loc[f, g])) for g in feats}}
            for f in feats]


def _confounder_vif_rows(df: pd.DataFrame) -> list[dict]:
    """VIF over the numeric confounder proxy controls (so the model is not overloaded)."""
    from . import econometrics as E
    if not E.HAVE_STATSMODELS:
        return []
    feats = [f for f in _PROXY_NUMERIC if f in df.columns]
    cols = {}
    for f in feats:
        s = pd.to_numeric(df[f], errors="coerce")
        if s.notna().mean() < 0.5 or s.dropna().nunique() < 2:
            continue
        cols[f] = s.fillna(s.median())
    if len(cols) < 2:
        return []
    import statsmodels.api as sm
    X = pd.DataFrame(cols)
    X = X[X.notna().all(axis=1)]
    X, _ = E._drop_collinear(X, protect=set())
    if X.shape[1] < 2:
        return []
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    vmap = E._vif_map(Xc)
    rows = []
    for f in X.columns:
        v = vmap.get(f)
        lvl, interp = E.vif_level(v)
        rows.append({"confounder_proxy": f, "vif": v, "vif_level": lvl, "interpretation": interp})
    return rows


def _unmeasured_rows() -> list[dict]:
    rows = []
    for c in CONFOUNDER_REGISTRY:
        if c["availability_status"] in (EXTERNAL, NA) or (c["requires_external_data"] and c["availability_status"] == PROXY):
            rows.append({"confounder": c["confounder"], "category": c["category"],
                         "why_unmeasured": c["caveat"],
                         "external_data_needed": _EXTERNAL_DATA.get(c["confounder"], "external data source")})
    return rows


_EXTERNAL_DATA = {
    "domain_authority": "SEO authority (backlinks / Domain Authority / referring domains)",
    "brand_authority": "brand search volume / brand tracking",
    "index_history": "Google Search Console first-indexed date / crawl logs",
    "structured_data_schema": "raw page HTML",
    "page_accessibility": "HTTP status / robots / rendered HTML",
    "internal_linking_architecture": "full-site crawl / internal link graph",
    "content_duplication": "full page text corpus",
    "source_availability_selection": "the AI's full retrieval candidate set",
}


def confounder_audit(df: pd.DataFrame, *, labels: dict | None = None,
                     derivation_notes: list[dict] | None = None) -> dict:
    """Assemble the full confounder/proxy audit from a (proxy-derived) feature frame.
    Defensive: every table degrades to [] when its inputs are absent."""
    derived_set = set()
    for n in (derivation_notes or []):
        if n.get("step") == "_derived_columns":
            derived_set = set(s.strip() for s in (n.get("detail") or "").split(";") if s.strip())
    warnings: list[str] = []
    balance = _balance_rows(df, labels)
    flagged = [r["confounder_proxy"] for r in balance if r["warning"]]
    if flagged:
        warnings.append("Imbalanced confounder proxies between cited and more-only (consider as sensitivity "
                        "controls): " + ", ".join(sorted(set(flagged))[:8]) + ".")
    if "scrape_success" in df.columns and "cited" in df.columns:
        ss = df["scrape_success"].astype(float)
        cited = pd.to_numeric(df["cited"], errors="coerce")
        if (cited == 1).any() and (cited == 0).any():
            d = abs(float(ss[cited == 1].mean()) - float(ss[cited == 0].mean()))
            if d >= 0.15:
                warnings.append(f"Scrape success differs by {d:.0%} between cited and more-only — possible "
                                "selection bias; treat scrape_success as a diagnostic/sensitivity control.")
    return {
        "available": True,
        "registry": registry_rows(),
        "feature_availability": _feature_availability_rows(df, derived_set),
        "proxy_summary": _proxy_summary_rows(df),
        "balance_by_cited": balance,
        "correlation_matrix": _correlation_matrix_rows(df),
        "confounder_vif": _confounder_vif_rows(df),
        "unmeasured_confounders": _unmeasured_rows(),
        "derivation_notes": derivation_notes or [],
        "warnings": warnings,
        "report_caveats": [config.CAVEAT_CONFOUNDER_PROXY, config.CAVEAT_CONTACT_LOCATION,
                           config.CAVEAT_AGE_FRESHNESS, config.CAVEAT_PRICE_ASSOCIATION,
                           config.CAVEAT_POSITION_PANEL, config.CAVEAT_SOURCE_AVAILABILITY],
    }
