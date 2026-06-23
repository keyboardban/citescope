"""Reusable presentational components (cards, badges, headers, callouts)."""

from __future__ import annotations

import html

import streamlit as st

from src import config
from src.url_utils import pretty_url

from .theme import COLORS


def _esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def num(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}" if abs(x) < 1000 else f"{x:,.0f}"
    return f"{x:,}" if isinstance(x, int) else str(x)


# --------------------------------------------------------------------------- #
# headers
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="cs-hero"><h1>{_esc(title)}</h1><p>{_esc(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, desc: str | None = None, icon: str = "") -> None:
    st.markdown(
        f'<div class="cs-section"><span class="ico">{icon}</span><h2>{_esc(title)}</h2></div>',
        unsafe_allow_html=True,
    )
    if desc:
        st.markdown(f'<div class="cs-section-desc">{_esc(desc)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metric_cards(items: list[dict]) -> None:
    """items: [{'value':..., 'label':..., 'sub':optional}]."""
    cards = "".join(
        f'<div class="cs-metric"><div class="v">{_esc(it["value"])}</div>'
        f'<div class="l">{_esc(it["label"])}</div>'
        + (f'<div class="s">{_esc(it["sub"])}</div>' if it.get("sub") else "")
        + "</div>"
        for it in items
    )
    st.markdown(f'<div class="cs-metrics">{cards}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# badges
# --------------------------------------------------------------------------- #
def badge(text: str, kind: str = "src") -> str:
    cls = {"cited": "b-cited", "noncited": "b-noncited", "weak": "b-weak",
           "src": "b-src", "rank": "b-rank"}.get(kind, "b-src")
    return f'<span class="cs-badge {cls}">{_esc(text)}</span>'


def cited_badge(is_cited: bool) -> str:
    return badge("● cited", "cited") if is_cited else badge("non-cited", "noncited")


def match_badge(tier: str) -> str:
    if tier == "no_match":
        return badge("no match", "noncited")
    if tier == "domain_only":
        return badge("domain-only (weak)", "weak")
    return badge(tier.replace("_", " "), "cited")


# --------------------------------------------------------------------------- #
# callouts
# --------------------------------------------------------------------------- #
def limitation_box(long: bool = False) -> None:
    st.info(config.DISCLAIMER_LONG if long else config.DISCLAIMER_SHORT, icon="🔬")


def proxy_note(text: str) -> None:
    st.markdown(f'<div class="cs-note">{_esc(text)}</div>', unsafe_allow_html=True)


def empty_state(message: str, icon: str = "🧭") -> None:
    st.markdown(
        f'<div class="cs-card" style="text-align:center;color:{COLORS["muted"]};padding:30px">'
        f'<div style="font-size:2rem">{icon}</div>'
        f'<div style="margin-top:8px">{_esc(message)}</div></div>',
        unsafe_allow_html=True,
    )


def glossary_expander() -> None:
    with st.expander("📖 Terminology (careful, black-box wording)"):
        for term, desc in config.GLOSSARY.items():
            st.markdown(f"**{term}** — {desc}")


# --------------------------------------------------------------------------- #
# pipeline diagram
# --------------------------------------------------------------------------- #
def pipeline_diagram(counts: dict) -> None:
    steps = [
        ("Prompt", counts.get("prompt", 1)),
        ("Queries", counts.get("queries", 0)),
        ("Citations", counts.get("citations", 0)),
        ("SERP cand.", counts.get("candidates", 0)),
        ("Scraped", counts.get("scraped", 0)),
        ("Matched", counts.get("matched", 0)),
        ("Features", counts.get("features", 0)),
    ]
    html_parts = ['<div class="pipe">']
    for i, (label, value) in enumerate(steps):
        html_parts.append(
            f'<div class="pipe-step"><div class="pv">{_esc(value)}</div>'
            f'<div class="pl">{_esc(label)}</div></div>'
        )
        if i < len(steps) - 1:
            html_parts.append('<div class="pipe-arrow">▸</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# website cards
# --------------------------------------------------------------------------- #
def _bar(label: str, value: float | None) -> str:
    v = 0.0 if value is None else max(0.0, min(1.0, float(value)))
    shown = "—" if value is None else f"{v:.2f}"
    return (
        f'<div class="bar-lab"><span>{_esc(label)}</span><span>{shown}</span></div>'
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{v*100:.0f}%"></div></div>'
    )


def site_card(row: dict) -> None:
    cited = bool(row.get("cited"))
    klass = "cited" if cited else "noncited"
    badges = cited_badge(cited) + " " + badge(row.get("source_type", "unknown"), "src")
    if row.get("official_source"):
        badges += " " + badge("official", "src")
    if cited:
        badges += " " + match_badge(row.get("match_type", "no_match"))
    status = "✓ scraped" if row.get("scrape_success") else "✗ not scraped"

    st.markdown(
        f'<div class="site-card {klass}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'{badge("#" + str(row.get("serp_rank", "?")), "rank")}'
        f'<span style="font-size:.72rem;color:{COLORS["muted"]}">{_esc(status)}</span></div>'
        f'<p class="t" style="margin-top:6px">{_esc(row.get("title") or row.get("domain"))}</p>'
        f'<p class="u">{_esc(pretty_url(row.get("url", "")))}</p>'
        f'<div style="margin-bottom:8px">{badges}</div>'
        f'{_bar("page–answer similarity", row.get("page_output_sim"))}'
        f'{_bar("best chunk–answer", row.get("max_chunk_output_sim"))}'
        "</div>",
        unsafe_allow_html=True,
    )
