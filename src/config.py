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
DB_PATH = DATA_DIR / "audit.db"       # SQLite: run index + API result cache


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, RUNS_DIR, EXPORTS_DIR):
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
WEAK_TIERS = {"domain_only"}

SIMILARITY_METHODS = ["lexical (offline)", "gemini embeddings"]


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
