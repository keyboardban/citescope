#!/usr/bin/env python3
"""Create compact, no-scrape notebook tables from a large Bright Data export."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import brightdata
from src.chatgpt_pipeline import flatten_sources
from src.source_type import classify
from src.url_utils import root_domain


def _save_bar(frame: pd.DataFrame, label: str, path: Path, limit: int = 14) -> None:
    plot = frame.head(limit).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4.5, len(plot) * 0.45)))
    ax.barh(plot[label], plot["cited_rate"] * 100, color="#187b8d")
    ax.axvline(0, color="#667085", linewidth=1)
    ax.set_xlabel("Cited rate (%)")
    ax.set_title(f"Cited rate by {label.replace('_', ' ')}")
    for y, (_, row) in enumerate(plot.iterrows()):
        ax.text(row["cited_rate"] * 100 + 0.6, y, f"n={int(row['source_rows']):,}", va="center", fontsize=9)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def prepare(output_json: Path, manifest_csv: Path, out: Path, figures: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    run = brightdata.parse_run(output_json.read_text("utf-8"), output_json.name)
    manifest = brightdata.parse_manifest(manifest_csv.read_text("utf-8"), manifest_csv.name)
    match = brightdata.apply_manifest(run, manifest)
    raw_sources = flatten_sources(run)
    rows = []
    for source in raw_sources:
        url = source.get("url") or ""
        source_type, institutional = classify(url)
        rows.append({
            "record_id": source.get("record_id"),
            "prompt_id": source.get("prompt_id"),
            "intent": source.get("intent") or "(unmatched)",
            "topic": source.get("topic") or "(unmatched)",
            "source_url": url,
            "normalized_url": source.get("normalized_url") or "",
            "source_root_domain": root_domain(url),
            "source_title": source.get("title") or "",
            "source_description": source.get("description") or "",
            "cited": int(source.get("cited_label") or 0),
            "source_group": source.get("source_group") or "more_only",
            "source_origin": source.get("source_origin") or "",
            "source_position": source.get("source_position"),
            "observed_rank": source.get("observed_rank"),
            "source_type": source_type,
            "institutional_official": institutional,
        })
    sources = pd.DataFrame(rows)
    sources.to_csv(out / "area_condo_brightdata_500_normalized_sources.csv", index=False)

    records = pd.DataFrame([{
        "record_id": rec.get("record_id"), "prompt_id": rec.get("prompt_id"),
        "intent": rec.get("intent") or "(unmatched)", "topic": rec.get("topic") or "(unmatched)",
        "has_sources": bool(rec.get("sources")), "source_rows": len(rec.get("sources") or []),
        "cited_sources": sum(int(s.get("cited_label") or 0) for s in rec.get("sources") or []),
    } for rec in run["records"]])
    records.to_csv(out / "area_condo_brightdata_500_records.csv", index=False)

    def summary(group: str) -> pd.DataFrame:
        frame = sources.groupby(group, dropna=False).agg(
            source_rows=("cited", "size"), cited_sources=("cited", "sum"),
            unique_urls=("normalized_url", "nunique"), unique_prompts=("prompt_id", "nunique"),
        ).reset_index()
        frame["more_only_sources"] = frame["source_rows"] - frame["cited_sources"]
        frame["cited_rate"] = frame["cited_sources"] / frame["source_rows"]
        return frame.sort_values(["source_rows", group], ascending=[False, True], kind="stable")

    for column, filename in [
        ("intent", "intent_summary.csv"),
        ("source_origin", "source_origin_summary.csv"),
        ("source_type", "source_type_summary.csv"),
        ("source_root_domain", "top_domain_summary.csv"),
    ]:
        table = summary(column)
        if column == "source_root_domain":
            table = table.head(50)
        table.to_csv(out / filename, index=False)

    intent = pd.read_csv(out / "intent_summary.csv")
    source_type = pd.read_csv(out / "source_type_summary.csv")
    domains = pd.read_csv(out / "top_domain_summary.csv")
    _save_bar(intent.sort_values("source_rows", ascending=False), "intent", figures / "cited_rate_by_intent.png")
    _save_bar(source_type.sort_values("source_rows", ascending=False), "source_type", figures / "cited_rate_by_source_type.png")
    _save_bar(domains.sort_values("source_rows", ascending=False), "source_root_domain", figures / "cited_rate_by_top_domain.png", limit=20)

    manifest_frame = pd.read_csv(manifest_csv, low_memory=False)
    manifest_audit = pd.DataFrame([{
        "manifest_rows": len(manifest_frame), "manifest_prompt_ids": manifest_frame["prompt_id"].nunique(),
        "manifest_matched": match["matched"], "manifest_unmatched": match["unmatched"],
        "all_prompts_nonbranded": bool(manifest_frame["prompt_is_nonbranded"].astype(str).str.casefold().isin({"true", "1", "yes"}).all()),
        "intent_count": manifest_frame["intent"].nunique(),
    }])
    manifest_audit.to_csv(out / "manifest_coverage_audit.csv", index=False)

    result = {
        "output_json": str(output_json), "manifest_csv": str(manifest_csv),
        "records": int(run["n_records"]), "records_with_sources": int(records["has_sources"].sum()),
        "records_without_sources": int((~records["has_sources"]).sum()),
        "source_rows": int(len(sources)), "unique_urls": int(sources["normalized_url"].nunique()),
        "cited_sources": int(sources["cited"].sum()), "more_only_sources": int((sources["cited"] == 0).sum()),
        "cited_rate": float(sources["cited"].mean()), "manifest_matched": int(match["matched"]),
        "manifest_unmatched": int(match["unmatched"]), "parser_warnings": run["warnings"],
        "scraping_performed": False,
    }
    (out / "run_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brightdata-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.brightdata_json, args.manifest, args.output_dir, args.figure_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
