"""Central configuration: paths, secrets, defaults, and black-box framing text.

This module is deliberately free of Streamlit imports so the engine can be used
and tested headlessly. Secrets are read from the environment only (never written
to disk or logged).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, from the project root, without overriding real env vars.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # raw API responses (audit trail)
RUNS_DIR = DATA_DIR / "runs"          # one JSON snapshot per completed run
EXPORTS_DIR = DATA_DIR / "exports"    # generated CSV/JSON/Markdown reports
BATCHES_DIR = DATA_DIR / "batches"    # batch-run summaries (multi-prompt)
CHATGPT_DIR = DATA_DIR / "chatgpt"    # ChatGPT Bright Data audit snapshots
DB_PATH = DATA_DIR / "audit.db"       # SQLite: run index + API/embedding cache


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, RUNS_DIR, EXPORTS_DIR, BATCHES_DIR, CHATGPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Secrets (environment only)
# --------------------------------------------------------------------------- #
def get_secret(name: str) -> str | None:
    """Read a secret from the environment. Handles the APIFY token alias."""
    if name == "APIFY_TOKEN":
        return os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN") or None
    return os.getenv(name) or None


def secret_present(name: str) -> bool:
    return bool(get_secret(name))


REQUIRED_SECRETS = ("GEMINI_API_KEY", "APIFY_TOKEN")


# --------------------------------------------------------------------------- #
# Defaults (overridable via env or the UI)
# --------------------------------------------------------------------------- #
APIFY_SERP_ACTOR = os.getenv("APIFY_SERP_ACTOR", "apify/google-search-scraper")
APIFY_SCRAPER_ACTOR = os.getenv("APIFY_SCRAPER_ACTOR", "apify/website-content-crawler")

# Model picker offers these; the proven grounding path uses generate_content.
# gemini-2.5-flash is the reliable default for Google Search Grounding.
# (Names verified against the models available to this account; newer previews
# may require specific account access.)
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.0-flash",
]
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", GEMINI_MODELS[0])
DEFAULT_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")

DEFAULT_TEMPERATURE = 0.2
DEFAULT_SERP_TOP_K = 20
DEFAULT_COUNTRY = "us"
DEFAULT_LANGUAGE = "en"
DEFAULT_SCRAPE_TOP_K = 12
RECALL_KS = (5, 10, 20, 50)

# Crawler types exposed by apify/website-content-crawler.
CRAWLER_TYPES = ["cheerio", "playwright:adaptive", "playwright:firefox"]
DEFAULT_CRAWLER_TYPE = "cheerio"  # fastest/cheapest; good for static content

# Citation match tiers, strongest -> weakest. "domain_only" is the only weak one.
MATCH_TIERS = [
    "exact",
    "normalized",
    "final_redirect",
    "canonical",
    "amp_canonical",
    "domain_only",
    "no_match",
]
# Tier groupings used by the three recall variants:
#   strict_recall          -> direct URL-identity matches only
#   canonical_recall       -> identity + canonical/amp equivalence (these are "strong")
#   domain_inclusive_recall-> strong + weak domain-only (exploratory)
STRICT_TIERS = {"exact", "normalized", "final_redirect"}
STRONG_TIERS = STRICT_TIERS | {"canonical", "amp_canonical"}  # set cited_label = 1
WEAK_TIERS = {"domain_only"}                                   # never cited by default
RECALL_MODES = ("strict", "canonical", "domain_inclusive")

SIMILARITY_METHODS = ["lexical (offline)", "gemini embeddings"]

# --------------------------------------------------------------------------- #
# Robustness / cost controls
# --------------------------------------------------------------------------- #
RETRY_COUNT = int(os.getenv("RETRY_COUNT", "3"))
RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("RETRY_MAX_DELAY", "20.0"))

MAX_SIM_CHARS = 8000          # page text cap before similarity scoring (reported)
REDIRECT_TIMEOUT = 4.0        # per-redirect resolution timeout (seconds)
REDIRECT_MAX_WORKERS = 8      # concurrency for redirect resolution

# --------------------------------------------------------------------------- #
# Caveat text (kept honest, reused across UI + report)
# --------------------------------------------------------------------------- #
CAVEAT_POST_OUTPUT = (
    "Page–answer and chunk–answer similarity are **post-output** overlap metrics. "
    "They may be partly **circular** because the AI answer may have been generated "
    "from cited sources. Treat them as semantic-overlap visualizers, not independent "
    "evidence of source selection."
)
CAVEAT_LENGTH = (
    "Lexical page–answer similarity can correlate with page length — longer pages "
    "share more vocabulary with a long answer. Prefer chunk-level similarity for "
    "headline comparisons."
)
CAVEAT_RECALL = (
    "Recall@K measures how many AI citation URLs were recovered in the reconstructed "
    "SERP within top-K ranks. Unmatched citations are not evidence the model did not "
    "use them; they only mean the reconstructed SERP did not recover them."
)
CAVEAT_BATCH = (
    "Batch results are observable associations across runs, not causal evidence about "
    "how the AI selects or cites sources."
)

# --------------------------------------------------------------------------- #
# ChatGPT Bright Data Source Audit framing
# --------------------------------------------------------------------------- #
CHATGPT_INTRO = (
    "**ChatGPT Bright Data Audit** analyzes sources surfaced by ChatGPT through Bright "
    "Data. It compares sources marked as **cited** with additional sources **shown but "
    "not cited** (more-only). This does **not** reveal ChatGPT's full internal retrieval "
    "set; it only studies observable **source placement**."
)
CAVEAT_MORE_ONLY = (
    "**More-only sources** are surfaced in the Bright Data / ChatGPT output but not marked "
    "as cited. This does **not** mean ChatGPT rejected or ignored them."
)
CAVEAT_ANSWER_CG = (
    "Page–answer and chunk–answer similarity are **post-output** semantic-overlap metrics. "
    "They may be partly **circular** because the answer may have been generated from cited "
    "sources. Treat them as overlap visualizers, not proof of source selection."
)

# --------------------------------------------------------------------------- #
# Non-branded Brand Visibility Audit framing
# --------------------------------------------------------------------------- #
BRAND_VISIBILITY_INTRO = (
    "**Non-branded Brand Visibility Audit** studies non-branded prompts that do **not** "
    "directly mention the client brand. For each prompt it measures whether the client or a "
    "competitor appears in the **observable** ChatGPT / Bright Data answer or source panel — "
    "and, among surfaced pages, which content features are associated with being **cited** "
    "rather than only **shown but not cited** (more-only). This does **not** reveal ChatGPT's "
    "internal retrieval process; it only studies observable brand visibility and citation "
    "behavior."
)
CAVEAT_BRAND_VISIBILITY = (
    "This is an **observable brand visibility** audit, not internal retrieval analysis. "
    "**More-only** = *shown but not cited* — it does **not** mean the source was rejected or "
    "ignored. Content features are heuristics associated with citation; they are not proof of "
    "why a source was cited."
)

# Optional fallback brand terms (semicolon/comma separated). The manifest is the
# source of truth; these are used only when a record carries no terms. Kept EMPTY
# by design so nothing brand-specific is hardcoded into the engine.
DEFAULT_CLIENT_BRAND_TERMS: list[str] = []
DEFAULT_COMPETITOR_BRAND_TERMS: list[str] = []

# Source-position bands for the position-controlled content comparison.
POSITION_BANDS = ("1-3", "4-6", "7-10", "11+", "unknown")

# --------------------------------------------------------------------------- #
# Econometrics — the position-adjusted "citation model" layer
# --------------------------------------------------------------------------- #
# Cluster-robust SEs are only trustworthy with many clusters (textbook rule of
# thumb); below this, fall back to the wild cluster bootstrap.
MIN_CLUSTERS = 40
VIF_WATCH = 5.0          # multicollinearity: watch above this
VIF_PROBLEM = 10.0       # multicollinearity: problem above this
ECON_SE_DEFAULT = "HC3"  # heteroskedasticity-robust (0/1 outcome is always heteroskedastic)
ECON_BOOTSTRAP_ITERS = 1999
ECON_RNG_SEED = 12345    # match batch.py for determinism
ECON_MIN_ROWS = 20       # refuse to fit below this many usable rows

# This layer is a SCOPED exception to the app's "observable patterns only" rule:
# it reports position-adjusted regression coefficients that may be read as cautious
# EFFECT ESTIMATES, but ONLY under explicitly stated assumptions + a signed
# omitted-variable caveat. The rest of CiteScope stays strictly observational.
CAVEAT_REGRESSION = (
    "**Position-adjusted citation model.** Each coefficient is the association between a "
    "feature and being cited, **holding the other listed features (including position) fixed** "
    "(a linear probability model, so coefficients are in **probability points**). It may be read "
    "as a *cautious effect estimate* — but only under the assumptions below, none of which the "
    "data can verify. Robust/cluster-robust error bars are honest about noise; they say nothing "
    "about whether an unobserved confounder is biasing the estimate."
)
CAVEAT_ASSUMPTIONS = (
    "Reading a coefficient as an effect assumes: (1) **exogeneity** — no important unobserved "
    "confounder correlated with this feature; (2) **positivity/overlap** — both feature-present "
    "and feature-absent sources exist across the position range; (3) **functional form** — the "
    "shape (including how position enters) is approximately right. Observational data cannot "
    "confirm any of these."
)
# Signed omitted-variable templates (named confounder + likely direction of bias).
CAVEAT_OVB_GEMINI = (
    "Most likely unmeasured confounder: a domain's **authority / popularity prior**, which "
    "plausibly raises both its reconstructed-SERP rank and its chance of being cited. Rank is "
    "adjusted for, but residual authority that still drives citation and also tracks a content/"
    "similarity feature would bias that feature **away from zero** — the true association is "
    "likely **smaller** than shown. The sign reverses for any feature more common on lower-"
    "authority pages."
)
CAVEAT_OVB_CHATGPT = (
    "Most likely unmeasured confounder: **brand familiarity / editorial trust** in a source, "
    "plausibly raising both how prominently it is surfaced (position) and whether it is cited. "
    "Position is adjusted for, but residual familiarity that also tracks a content feature "
    "(e.g. structured, contact-rich pages tend to be established brands) biases that feature "
    "**away from zero** — the true association is likely **smaller**. The sign reverses for a "
    "feature more common on less-established pages."
)
CAVEAT_OVB_BRAND = (
    "Most likely unmeasured confounder: a brand's **offline reputation / search demand**, which "
    "can raise both a page's placement and its citation. Position is adjusted for; residual "
    "reputation that also tracks a content feature biases that feature **away from zero** "
    "(true association likely **smaller**), and reverses sign for features common on smaller brands."
)
CAVEAT_FEW_CLUSTERS = (
    "Few clusters (prompts): cluster-robust error bars are unreliable below ~40 clusters and run "
    "too narrow. Treat significance cautiously; the wild cluster bootstrap is the honest fallback."
)
CAVEAT_SEPARATION = (
    "Perfect/quasi-separation: a feature predicted citation flawlessly, so the logit did not "
    "converge to a finite estimate. Reporting the linear-probability-model estimate instead."
)
# Business-facing caveats for the sensitivity report (verbatim-safe observational wording).
CAVEAT_MODEL_OBSERVATIONAL = (
    "This is an observational association analysis. It estimates which observable source/page "
    "features are associated with citation probability. It does not prove that adding a feature "
    "will cause citation probability to increase."
)
CAVEAT_POSITION_DOWNSTREAM = (
    "Source position is a strong observable placement feature. Because it may sit downstream of "
    "relevance or source selection (a mediator / post-treatment variable), results should be "
    "compared with and without source_position. It is observable placement/ranking, not the AI's "
    "internal ranking."
)
CAVEAT_CONTACT_LOCATION = (
    "Negative coefficients for contact/location features should not be interpreted as evidence that "
    "contact information is bad. They may indicate that thin contact/location pages are less often "
    "cited than answer-ready article or product pages."
)
CAVEAT_SIMILARITY_FEATURES = (
    "Prompt similarity features are highly overlapping. Individual coefficients should not be "
    "overinterpreted when VIF is high; use a reduced relevance score or one preferred similarity feature."
)
CAVEAT_AGE_FRESHNESS = (
    "If older pages are positively associated with citation, this may reflect authority, index "
    "history, or evergreen content rather than age itself."
)
# Few-cluster warning (exact business-facing wording).
CAVEAT_CLUSTER_FEW = (
    "Cluster-robust SE may be unstable with few clusters; consider wild cluster bootstrap or "
    "interpret cautiously."
)
# Source-position language + the required with/without-position caveat.
CAVEAT_POSITION_PANEL = (
    "`source_position` is the **observable source panel position** (observed source order / source "
    "panel placement) — not an internal AI rank, retrieval rank, or Google rank. It is a strong "
    "observable placement feature, and because it may be downstream of relevance, source selection, "
    "or source panel construction, content-feature estimates should be compared with and without "
    "`source_position` (Models A/B vs C)."
)
# LPM reading template (controlled association, never bare causation).
CAVEAT_LPM_INTERPRET = (
    "Using a Linear Probability Model, a feature is **associated with a ±X percentage point "
    "difference** in citation probability, **controlling for the included variables** — a controlled "
    "association, not proof that adding the feature will cause more citations (unless the data came "
    "from a randomized experiment, which this is not)."
)
# Missing-data caveat.
CAVEAT_MISSINGNESS = (
    "Missingness may be informative, especially when scrape failure or missing metadata differs by "
    "source type, page type, or citation status. Numeric gaps are median-filled with a `*_missing` "
    "indicator; categorical gaps enter as an explicit `unknown` level."
)
# Reference-category caveat for dummy variables.
CAVEAT_REFERENCE_CATEGORY = (
    "Dummy coefficients are interpreted **relative to the omitted reference category** (e.g. a "
    "`page_type=article` coefficient of +0.30 means article pages are associated with +30 percentage "
    "points higher citation probability **compared with the reference page type**, holding other "
    "variables fixed)."
)
# Signed OVB example + the named-confounder list (cautious, hedged wording).
CAVEAT_OVB_SIGNED_EXAMPLE = (
    "Signed example: there may be **possible upward bias** if unobserved writing quality is positively "
    "correlated with `has_faq` and positively affects citation probability — the estimate would then "
    "overstate the FAQ association. The sign flips for a confounder negatively related to the feature."
)
CAVEAT_OVB_CONFOUNDERS = (
    "Unobserved **writing quality**, **domain authority**, **source-panel placement**, **scrape "
    "success**, and **page type** may confound content-feature associations. These are not measured "
    "here, so coefficients are associations under the stated assumptions, not clean effects."
)
# Business-safe recommendation wording (do not strip access features; embed them in answer-ready pages).
CAVEAT_BUSINESS_REC = (
    "Do not interpret negative contact/location coefficients as evidence that contact information "
    "should be removed. The result more likely indicates that thin access pages are less "
    "citation-ready than answer-style article/product pages. A better recommendation is to embed "
    "contact, price, booking, and location information **inside answer-ready product/service pages**."
)
# Rare-feature caveat.
CAVEAT_RARE_FEATURES = (
    "Rare features (prevalence <5% or >95%) can produce unstable coefficients and wide confidence "
    "intervals — read them with caution or report them jointly."
)
# Confounder / proxy framework caveats.
CAVEAT_CONFOUNDER_PROXY = (
    "Confounder proxies reduce some omitted-variable risk but do not prove causality. The estimates "
    "remain **controlled observational associations**. Proxies (e.g. CiteScope-observed domain "
    "visibility, visibility history) are labelled as proxies — they are not the true construct "
    "(external SEO authority, true index history)."
)
CAVEAT_PRICE_ASSOCIATION = (
    "If a price/package feature is positive, do not claim price *causes* citations — price/package "
    "pages are **associated with** citation within the observed sources."
)
CAVEAT_SOURCE_AVAILABILITY = (
    "The model estimates citation **within surfaced sources**, not the probability of being retrieved "
    "from the entire web. The AI's full retrieval candidate set is unobserved; visibility-count proxies "
    "only describe what CiteScope has seen surfaced."
)
# Outcome-definition block (exported verbatim as econometrics_outcome_definition.txt).
OUTCOME_DEFINITION_TEXT = (
    "Outcome definition — citation model\n"
    "===================================\n"
    "cited = 1  : the source was explicitly cited.\n"
    "cited = 0  : the source was surfaced / shown as more-only but not cited.\n\n"
    "This is NOT 'cited vs not retrieved'. more-only means surfaced but not cited.\n"
    "Do not interpret more-only as rejected or ignored by the AI. source_position is the\n"
    "observable source panel position (observed source order), not an internal AI rank,\n"
    "retrieval rank, or Google rank.\n"
)


# --------------------------------------------------------------------------- #
# Black-box framing — reused across the UI to keep terminology honest.
# --------------------------------------------------------------------------- #
DISCLAIMER_SHORT = (
    "Black-box observational audit. We only observe what the Gemini API exposes "
    "(output text, search queries, citations, grounding metadata). The Apify SERP "
    "is a *reconstructed* candidate set — not the exact internal results the AI used."
)

DISCLAIMER_LONG = (
    "**What this is.** This tool compares **cited websites** (URLs surfaced in "
    "Gemini's grounding metadata) against **non-cited reconstructed SERP candidates** "
    "(results we independently fetch from Apify for the same search queries).\n\n"
    "**What we can claim.** Observable patterns — e.g. *cited websites tended to rank "
    "higher in the reconstructed SERP*, or *had higher semantic overlap (a proxy) with "
    "the answer*.\n\n"
    "**What we cannot claim.** We do **not** know the AI's true internal retrieval set "
    "or why any page was or wasn't cited. A non-cited candidate was **not** \"rejected\"; "
    "it simply did not appear in the observed citations. Chunk similarity is a "
    "*semantic overlap proxy*, not proof the model read that chunk."
)

GLOSSARY = {
    "reconstructed SERP": "Search results we independently fetch via Apify for an "
    "observed query. A parallel candidate set, not the AI's internal results.",
    "candidate websites": "All results in the reconstructed SERP.",
    "cited websites": "URLs present in Gemini's grounding metadata (after resolving "
    "redirect links).",
    "non-cited SERP candidate": "A reconstructed candidate that was not matched to any "
    "observed citation. Not evidence of rejection.",
    "citation matching": "Linking a cited URL to a SERP candidate via tiered URL rules "
    "(exact → normalized → redirect → canonical → amp → domain-only).",
    "citation recall@K": "Share of citations whose matched candidate appears within the "
    "top-K reconstructed SERP ranks.",
    "semantic overlap proxy": "A similarity score between two texts. A proxy for "
    "relatedness — not proof of causal use by the model.",
    "chunk-level similarity": "Similarity between a page passage (chunk) and the AI "
    "answer or query.",
}
