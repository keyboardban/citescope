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
           "src": "b-src", "rank": "b-rank", "brand": "b-brand"}.get(kind, "b-src")
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


def caveat_box(text: str) -> None:
    """Loud amber caveat (renders markdown via st.warning)."""
    st.warning(text, icon="⚠️")


def regression_block(fits, focal_only: bool = True) -> None:
    """Render the position-adjusted citation model: forest plot + coefficient table +
    diagnostics + the signed omitted-variable caveat + assumptions. Accepts a list of
    fit_results or a {group: fit} dict (degrades gracefully if absent/unfitted)."""
    import pandas as pd

    from . import charts

    if isinstance(fits, dict):
        fits = list(fits.values())
    fits = [f for f in (fits or []) if f]
    if not fits:
        st.info("No position-adjusted model yet (parse a run, and pool prompts / apply a manifest for clustering).")
        return
    st.warning(config.CAVEAT_REGRESSION, icon="📐")
    for f in fits:
        if not f.get("available", True):
            st.info((f.get("warnings") or ["Install `statsmodels` to enable the citation model."])[0])
            continue
        if not f.get("fitted"):
            st.caption("ℹ️ " + (f.get("warnings") or ["Not fitted."])[0])
            continue
        st.plotly_chart(charts.coefficient_forest(f, focal_only=focal_only), width="stretch")
        rows = [{"feature": c["label"], "Δ prob": c["estimate"], "se": c["se"],
                 "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p": c["p"],
                 "q(BH)": c.get("p_adj"), "VIF": c.get("vif"),
                 "focal": "✓" if c.get("is_focal") else ""} for c in f.get("coefficients", [])]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        meta = f"n={f.get('n')}"
        meta += (f" · {f['n_clusters']} clusters · {f['se_type']} SE"
                 if f.get("n_clusters") else f" · {f.get('se_type')} SE")
        if f.get("r2") is not None:
            meta += f" · R²={f['r2']}"
        st.caption(meta + " · coefficients are Δ probability of citation, holding other features (incl. position) fixed.")
        if f.get("ame"):
            with st.expander("🔁 Logit AME cross-check (Δ probability — should track the LPM coefficients)"):
                st.dataframe(pd.DataFrame([
                    {"feature": r["label"], "AME": r["ame"], "se": r["se"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "p": r["p"]} for r in f["ame"]]),
                    width="stretch", hide_index=True)


def sensitivity_block(mc) -> None:
    """Render the A/B/C/D model comparison + VIF/anomaly/group diagnostics + forest plots."""
    import pandas as pd

    from src import report

    if not mc or not mc.get("available"):
        st.info("Sensitivity analysis needs `statsmodels`.")
        return
    if not mc.get("fitted"):
        st.info("Not enough usable rows to fit the model comparison — scrape pages and apply a manifest, "
                "and ensure enough sources. (The CSV exports still generate.)")
        return

    st.warning(config.CAVEAT_MODEL_OBSERVATIONAL, icon="📐")
    for s in mc.get("executive_summary") or []:
        st.markdown(f"- {s}")
    with st.expander("⚠️ Interpretation caveats"):
        for cv in (config.CAVEAT_POSITION_DOWNSTREAM, config.CAVEAT_SIMILARITY_FEATURES,
                   config.CAVEAT_CONTACT_LOCATION, config.CAVEAT_AGE_FRESHNESS):
            st.markdown(f"- {cv}")

    comp = pd.DataFrame(mc.get("comparison_rows") or [])
    if not comp.empty:
        st.markdown(f"**Model comparison — Δ probability across A→D** "
                    f"(clustered by `{mc.get('cluster_variable')}`, {mc.get('cluster_count')} clusters)")
        piv = comp.pivot_table(index="feature", columns="model_name", values="delta_prob", aggfunc="first").round(4)
        st.dataframe(piv, width="stretch")
        st.caption("A coefficient stable across A→D is more trustworthy; large swings indicate confounding or "
                   "collinearity. A = content · B = +source/authority · C = +position · D = reduced similarity.")

    cmod = next((m["fit"] for m in mc.get("models", []) if m["model_name"].startswith("C") and m["fit"].get("fitted")),
                mc.get("diagnostic_model") or {})
    p1 = report.forest_png(cmod, title="Position-adjusted (Model C)")
    p2 = report.forest_png(cmod, exclude_groups=("authority", "page_type", "intent"), title="Content features only")
    if p1 or p2:
        fc = st.columns(2)
        if p1:
            fc[0].image(p1, caption="Focal features (Δ probability ± 95% CI)")
        if p2:
            fc[1].image(p2, caption="Content features only (excl. source position)")

    vif = pd.DataFrame(mc.get("vif_rows") or [])
    if not vif.empty:
        with st.expander("VIF diagnostics (multicollinearity)"):
            st.dataframe(vif[["feature", "vif", "vif_level", "interpretation"]], width="stretch", hide_index=True)

    st.markdown("**Anomaly diagnostics**")
    anom = pd.DataFrame(mc.get("anomaly_rows") or [])
    if anom.empty:
        st.caption("No anomalies flagged.")
    else:
        st.dataframe(anom[["check", "feature", "estimate", "p", "severity", "message"]],
                     width="stretch", hide_index=True)

    grp = pd.DataFrame(mc.get("group_rows") or [])
    if not grp.empty:
        with st.expander("Feature group summary"):
            st.dataframe(grp, width="stretch", hide_index=True)
        if f.get("ovb_caveat"):
            caveat_box("**Omitted-variable note (signed).** " + f["ovb_caveat"])
        for w in f.get("warnings", []):
            st.caption("⚠️ " + w)
        for asm in f.get("assumptions", []):
            st.caption(asm)


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
    if row.get("institutional_official") or row.get("official_source"):
        badges += " " + badge("official", "src")
    if row.get("brand_official_candidate"):
        badges += " " + badge("official?", "brand")
    if cited:
        badges += " " + match_badge(row.get("match_type", "no_match"))
    elif row.get("weak_domain_match"):
        badges += " " + badge("weak domain", "weak")
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
