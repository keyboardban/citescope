"""Rule-based source-type classification (a deterministic heuristic).

Returns (source_type, is_official). Conservative by design: 'official' is
reserved for government / education / military / inter-governmental domains.
This is an observable proxy for the kind of source, not a quality judgement.
"""

from __future__ import annotations

from .url_utils import domain, root_domain

# Canonical source-type labels used by the dashboard.
SOURCE_TYPES = [
    "government", "education", "news", "documentation", "forum", "reference",
    "video", "social", "ecommerce", "review", "blog", "unknown",
]

_KNOWN: dict[str, str] = {}


def _add(label: str, *domains: str) -> None:
    for d in domains:
        _KNOWN[d] = label


_add("news", "nytimes.com", "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com",
     "theguardian.com", "apnews.com", "bloomberg.com", "forbes.com", "wsj.com",
     "washingtonpost.com", "aljazeera.com", "ft.com", "nbcnews.com", "cnbc.com",
     "businessinsider.com", "techcrunch.com", "theverge.com", "wired.com",
     "bangkokpost.com", "nationthailand.com")
_add("forum", "reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com",
     "ycombinator.com", "discourse.org")
_add("reference", "wikipedia.org", "wiktionary.org", "britannica.com",
     "fandom.com", "investopedia.com")
_add("video", "youtube.com", "youtu.be", "vimeo.com")
_add("social", "twitter.com", "x.com", "facebook.com", "instagram.com",
     "linkedin.com", "tiktok.com", "threads.net", "pinterest.com", "medium.com")
_add("ecommerce", "amazon.com", "ebay.com", "etsy.com", "aliexpress.com",
     "alibaba.com", "walmart.com", "bestbuy.com", "lazada.com", "shopee.com",
     "shopify.com")
_add("review", "yelp.com", "tripadvisor.com", "trustpilot.com", "g2.com",
     "capterra.com", "glassdoor.com")
_add("blog", "substack.com", "wordpress.com", "blogspot.com", "blogger.com",
     "tumblr.com", "dev.to", "ghost.io", "hashnode.dev")
_add("documentation", "readthedocs.io", "readthedocs.org", "developer.mozilla.org",
     "github.io")

_TLD_OFFICIAL = {
    ".gov": "government", ".mil": "government", ".int": "government",
    ".edu": "education", ".ac": "education",
}


def classify(url: str) -> tuple[str, bool]:
    host = domain(url)
    root = root_domain(url)
    if not host:
        return "unknown", False

    # 1) Official institutional TLDs (also catches .gov.uk, .ac.th, .edu.au …).
    for tld, label in _TLD_OFFICIAL.items():
        if host.endswith(tld) or f"{tld}." in host:
            return label, True

    # 2) Known-domain map (exact registrable domain or any host suffix).
    if root in _KNOWN:
        return _KNOWN[root], False
    for known, label in _KNOWN.items():
        if host == known or host.endswith("." + known):
            return label, False

    # 3) Subdomain / keyword heuristics.
    low = host
    if low.startswith("docs.") or low.startswith("developer.") or low.startswith("dev."):
        return "documentation", False
    if low.startswith("blog.") or ".blog" in low:
        return "blog", False
    if low.startswith("forum.") or low.startswith("community.") or "forum" in low:
        return "forum", False
    if low.startswith("shop.") or low.startswith("store.") or "shop" in low:
        return "ecommerce", False
    if "news" in low:
        return "news", False
    if "wiki" in low:
        return "reference", False

    return "unknown", False
