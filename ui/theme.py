"""Visual theme: colour palette + global CSS injected once per session."""

from __future__ import annotations

import streamlit as st

COLORS = {
    "primary": "#4f46e5",
    "primary_soft": "#eef2ff",
    "cited": "#16a34a",
    "cited_soft": "#dcfce7",
    "noncited": "#94a3b8",
    "noncited_soft": "#f1f5f9",
    "weak": "#f59e0b",
    "weak_soft": "#fef3c7",
    "danger": "#ef4444",
    "text": "#1e2330",
    "muted": "#6b7280",
    "bg": "#f7f8fc",
    "card": "#ffffff",
    "border": "#e8eaf2",
}

# Colour per source type (stable across all charts).
SOURCE_COLORS = {
    "government": "#1d4ed8", "education": "#0891b2", "news": "#dc2626",
    "documentation": "#7c3aed", "forum": "#ea580c", "reference": "#0d9488",
    "video": "#db2777", "social": "#9333ea", "ecommerce": "#ca8a04",
    "review": "#16a34a", "blog": "#2563eb", "unknown": "#94a3b8",
}

# Colour per citation match tier.
TIER_COLORS = {
    "exact": "#15803d", "normalized": "#16a34a", "final_redirect": "#0d9488",
    "canonical": "#0891b2", "amp_canonical": "#2563eb", "domain_only": "#f59e0b",
    "no_match": "#cbd5e1",
}

CITED_SEQ = {"cited": COLORS["cited"], "non-cited": COLORS["noncited"]}


def inject_css() -> None:
    if st.session_state.get("_css_done"):
        return
    st.session_state["_css_done"] = True
    c = COLORS
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px; }}

        /* hero header */
        .cs-hero {{
            background: linear-gradient(120deg, {c['primary']} 0%, #6366f1 55%, #8b5cf6 100%);
            color: #fff; border-radius: 18px; padding: 22px 28px; margin-bottom: 18px;
            box-shadow: 0 8px 24px rgba(79,70,229,.22);
        }}
        .cs-hero h1 {{ color:#fff; font-size: 1.55rem; margin: 0 0 4px 0; font-weight: 700; }}
        .cs-hero p {{ color: #e9e9ff; margin: 0; font-size: .92rem; }}

        /* section header */
        .cs-section {{ display:flex; align-items:center; gap:10px; margin: 6px 0 2px 0; }}
        .cs-section .ico {{ font-size: 1.25rem; }}
        .cs-section h2 {{ font-size: 1.18rem; margin: 0; font-weight: 700; color: {c['text']}; }}
        .cs-section-desc {{ color: {c['muted']}; font-size: .86rem; margin: 0 0 12px 0; }}

        /* metric cards */
        .cs-metrics {{ display:flex; gap:12px; flex-wrap:wrap; margin: 6px 0 16px 0; }}
        .cs-metric {{
            background:{c['card']}; border:1px solid {c['border']}; border-radius:14px;
            padding:14px 16px; min-width:135px; flex:1; box-shadow:0 1px 3px rgba(16,24,40,.04);
        }}
        .cs-metric .v {{ font-size:1.7rem; font-weight:750; color:{c['primary']}; line-height:1.1; }}
        .cs-metric .l {{ font-size:.72rem; color:{c['muted']}; text-transform:uppercase;
            letter-spacing:.04em; margin-top:4px; font-weight:600; }}
        .cs-metric .s {{ font-size:.72rem; color:{c['muted']}; margin-top:2px; }}

        /* generic card / panel */
        .cs-card {{
            background:{c['card']}; border:1px solid {c['border']}; border-radius:14px;
            padding:16px 18px; box-shadow:0 1px 3px rgba(16,24,40,.04); margin-bottom:12px;
        }}

        /* badges */
        .cs-badge {{ display:inline-block; padding:2px 10px; border-radius:999px;
            font-size:.72rem; font-weight:700; line-height:1.5; }}
        .b-cited {{ background:{c['cited_soft']}; color:{c['cited']}; }}
        .b-noncited {{ background:{c['noncited_soft']}; color:#475569; }}
        .b-weak {{ background:{c['weak_soft']}; color:#b45309; }}
        .b-src {{ background:{c['primary_soft']}; color:{c['primary']}; }}
        .b-rank {{ background:#eef2ff; color:#3730a3; }}
        .b-brand {{ background:#ecfeff; color:#0e7490; }}

        /* website card */
        .site-card {{ background:{c['card']}; border:1px solid {c['border']}; border-radius:14px;
            padding:14px 16px; margin-bottom:10px; box-shadow:0 1px 3px rgba(16,24,40,.04); }}
        .site-card.cited {{ border-left:4px solid {c['cited']}; }}
        .site-card.noncited {{ border-left:4px solid {c['noncited']}; }}
        .site-card .t {{ font-weight:700; font-size:.98rem; color:{c['text']}; margin:0; }}
        .site-card .u {{ font-size:.78rem; color:{c['muted']}; margin:2px 0 8px 0;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .bar-wrap {{ background:#eef0f6; border-radius:6px; height:8px; width:100%; margin-top:3px; }}
        .bar-fill {{ background:{c['primary']}; height:8px; border-radius:6px; }}
        .bar-lab {{ font-size:.7rem; color:{c['muted']}; display:flex; justify-content:space-between; }}

        /* pipeline diagram */
        .pipe {{ display:flex; align-items:stretch; gap:6px; flex-wrap:wrap; margin:8px 0 4px 0; }}
        .pipe-step {{ background:{c['card']}; border:1px solid {c['border']}; border-radius:12px;
            padding:10px 12px; text-align:center; min-width:96px; flex:1; }}
        .pipe-step .pv {{ font-size:1.25rem; font-weight:750; color:{c['primary']}; }}
        .pipe-step .pl {{ font-size:.68rem; color:{c['muted']}; text-transform:uppercase; font-weight:600; }}
        .pipe-arrow {{ display:flex; align-items:center; color:{c['noncited']}; font-size:1.1rem; }}

        /* note / callout */
        .cs-note {{ background:{c['primary_soft']}; border-left:4px solid {c['primary']};
            padding:10px 14px; border-radius:10px; font-size:.82rem; color:#3730a3; margin:6px 0; }}
        .cs-caveat {{ background:{c['weak_soft']}; border-left:4px solid {c['weak']};
            padding:10px 14px; border-radius:10px; font-size:.82rem; color:#92400e; margin:6px 0; }}

        /* buttons */
        .stButton>button, .stDownloadButton>button {{ border-radius:10px; font-weight:600; }}
        .stButton>button[kind="primary"] {{ background:{c['primary']}; border:none; }}

        /* sidebar */
        section[data-testid="stSidebar"] {{ background:#ffffff; border-right:1px solid {c['border']}; }}
        [data-testid="stMetricValue"] {{ color:{c['primary']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
