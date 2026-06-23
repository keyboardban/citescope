"""URL normalisation, domain extraction, and redirect resolution.

Citation matching lives or dies on consistent URL handling, so all comparison
forms are produced here and used everywhere (matching, features, classification).
"""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

import requests
import tldextract

# Use the bundled public-suffix snapshot (no network at runtime, deterministic).
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

# Query params that never change page identity — dropped during normalisation.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "gclid", "gclsrc", "dclid", "fbclid", "msclkid", "mc_cid",
    "mc_eid", "igshid", "ref", "ref_src", "spm", "_hsenc", "_hsmi",
    "yclid", "wt_mc", "vero_id", "oly_enc_id", "oly_anon_id",
}

_DEFAULT_PORTS = {"http": "80", "https": "443"}

# Hosts that wrap a real destination URL behind a redirect.
_REDIRECT_HOSTS = ("vertexaisearch.cloud.google.com", "www.google.com", "google.com")
_REDIRECT_MARKERS = ("grounding-api-redirect", "/url?", "/aclk?")

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def _ensure_scheme(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        url = "https://" + url
    return url


def normalize_url(url: str) -> str:
    """Canonical comparison form.

    Lower-cases scheme/host, drops www and default ports, removes the fragment
    and tracking params, sorts the remaining query, and trims a trailing slash.
    """
    url = _ensure_scheme(url)
    if not url:
        return ""
    try:
        p = urlparse(url)
    except ValueError:
        return url

    # Force https in the comparison form so http/https variants of the same page
    # match during citation matching.
    scheme = "https"
    host = (p.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]

    netloc = host
    if p.port and _DEFAULT_PORTS.get(scheme) != str(p.port):
        netloc = f"{host}:{p.port}"

    path = unquote(p.path or "")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    kept = [
        (k, v)
        for k, v in parse_qsl(p.query, keep_blank_values=False)
        if k.lower() not in _TRACKING_PARAMS
    ]
    query = urlencode(sorted(kept))

    return urlunparse((scheme, netloc, path, "", query, ""))


def domain(url: str) -> str:
    """Host without a leading www (subdomains preserved)."""
    p = urlparse(_ensure_scheme(url))
    host = (p.hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def root_domain(url: str) -> str:
    """Registrable domain, public-suffix aware (sub.a.co.uk -> a.co.uk)."""
    ext = _EXTRACT(_ensure_scheme(url))
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return domain(url)


def strip_amp(url: str) -> str:
    """Best-effort AMP -> canonical variant for weak matching."""
    n = normalize_url(url)
    p = urlparse(n)
    host = p.netloc
    if host.startswith("amp."):
        host = host[4:]
    path = p.path
    path = re.sub(r"/amp/?$", "", path)
    path = re.sub(r"\.amp(\.html?)?$", r"\1", path)
    if path.startswith("/amp/"):
        path = path[4:]
    return urlunparse((p.scheme, host, path, "", p.query, ""))


def is_redirect_wrapper(url: str) -> bool:
    """True if the URL looks like a Google/Vertex redirect wrapper."""
    low = (url or "").lower()
    host_hit = any(h in low for h in _REDIRECT_HOSTS)
    marker_hit = any(m in low for m in _REDIRECT_MARKERS)
    return host_hit and marker_hit


def resolve_redirect(url: str, timeout: float = 4.0) -> str | None:
    """Follow redirects to the final destination URL.

    Grounding-chunk URIs are Vertex redirect links; resolving them reveals the
    real publisher URL so it can be matched against SERP candidates. Returns the
    final URL, or None on failure (caller falls back to the raw URI).
    """
    if not url:
        return None
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        if r.url and r.url != url:
            return r.url
        # Some redirectors don't honour HEAD — retry with a lightweight GET.
        r = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
        r.close()
        return r.url or None
    except requests.RequestException:
        try:
            r = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
            r.close()
            return r.url or None
        except requests.RequestException:
            return None


def pretty_url(url: str, max_len: int = 64) -> str:
    """Compact display form (host + truncated path)."""
    p = urlparse(_ensure_scheme(url))
    text = f"{domain(url)}{p.path}"
    return text if len(text) <= max_len else text[: max_len - 1] + "…"
