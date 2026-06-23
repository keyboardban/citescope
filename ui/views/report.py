"""Report / Export: download datasets and generate Markdown/HTML reports."""

from __future__ import annotations

import streamlit as st

from src import report

from .. import components as C
from ..state import get_run


def render() -> None:
    run = get_run()
    C.section("Report / Export",
              "Download the labelled dataset and a black-box-framed audit report.", "📤")
    if not run:
        C.empty_state("No run to export yet — run an audit or load the demo first.", "📤")
        return

    rid = run.get("run_id", "run")
    md = report.markdown_report(run)

    C.section("Datasets (CSV / JSON)", icon="📦")
    c = st.columns(3)
    c[0].download_button("⬇️ Feature table (CSV)", report.features_csv(run),
                         f"{rid}_features.csv", "text/csv", width="stretch")
    c[1].download_button("⬇️ SERP candidates (CSV)", report.serp_csv(run),
                         f"{rid}_serp.csv", "text/csv", width="stretch")
    c[2].download_button("⬇️ Citation matches (CSV)", report.matches_csv(run),
                         f"{rid}_matches.csv", "text/csv", width="stretch")

    C.section("Reports", icon="📝")
    c2 = st.columns(3)
    c2[0].download_button("⬇️ Report (Markdown)", md, f"{rid}_report.md",
                          "text/markdown", width="stretch")
    c2[1].download_button("⬇️ Report (HTML)", report.html_report(run), f"{rid}_report.html",
                          "text/html", width="stretch")
    c2[2].download_button("⬇️ Full run (JSON)", report.run_json(run), f"{rid}_run.json",
                          "application/json", width="stretch")

    if st.button("💾 Write everything to data/exports/"):
        paths = report.write_all(run)
        st.success("Exported:")
        for name, path in paths.items():
            st.code(f"{name}: {path}")

    C.section("Report preview", icon="📄")
    with st.container(border=True):
        st.markdown(md)
