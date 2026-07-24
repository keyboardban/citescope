#!/usr/bin/env python3
"""Compare deterministic general-taxonomy rule v2 with historical labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import econometrics_qa
from src.econometrics_eda_v2.paths import topic_output_dir


def _rate(series: pd.Series, label: str = "unknown") -> float:
    return float(series.fillna(label).astype(str).eq(label).mean())


def run(input_path: Path, output_dir: Path) -> dict[str, object]:
    historical = pd.read_csv(input_path, low_memory=False)
    audit = econometrics_qa.add_general_taxonomy_v2(historical)
    output_dir.mkdir(parents=True, exist_ok=True)

    old_final = audit["page_type_general"].fillna("unknown").astype(str)
    new_final = audit["page_type_general_rule_v2"].fillna("unknown").astype(str)
    audit["rule_v2_label_changed"] = old_final.ne(new_final)
    audit["rule_v2_change_type"] = "known_to_different_known"
    audit.loc[old_final.eq(new_final), "rule_v2_change_type"] = "unchanged"
    audit.loc[old_final.eq("unknown") & new_final.ne("unknown"), "rule_v2_change_type"] = "unknown_to_known"
    audit.loc[old_final.ne("unknown") & new_final.eq("unknown"), "rule_v2_change_type"] = "known_to_unknown"

    preferred = [
        "normalized_url", "source_url", "source_root_domain", "page_title", "meta_description",
        "content_strength", "content_quality_flag", "word_count", "structured_data_types",
        "site_type_general", "site_type_general_rule_v2", "page_type_url_seed_general",
        "page_type_url_seed_general_rule_v2", "page_type_general", "page_type_general_rule_v2",
        "page_type_family_general", "page_type_family_general_rule_v2", "page_type_general_confidence",
        "page_type_general_confidence_rule_v2", "page_type_general_source_rule_v2",
        "page_type_general_reason", "page_type_general_reason_rule_v2", "rule_v2_label_changed",
        "rule_v2_change_type", "source_appearances", "cited_appearances", "cited_rate",
    ]
    audit.reindex(columns=[column for column in preferred if column in audit]).to_csv(
        output_dir / "general_page_taxonomy_rule_v2_url_audit.csv", index=False
    )

    metrics = [
        ("unique_urls", len(audit), len(audit), "count"),
        ("url_seed_unknown_rate", _rate(audit["page_type_url_seed_general"]), _rate(audit["page_type_url_seed_general_rule_v2"]), "rate"),
        ("final_unknown_rate", _rate(audit["page_type_general"]), _rate(audit["page_type_general_rule_v2"]), "rate"),
        (
            "high_or_medium_confidence_rate",
            float(audit["page_type_general_confidence"].isin(["high", "medium"]).mean()),
            float(audit["page_type_general_confidence_rule_v2"].isin(["high", "medium"]).mean()),
            "rate",
        ),
        ("changed_final_labels", 0, int(audit["rule_v2_label_changed"].sum()), "count"),
        ("unknown_to_known", 0, int(audit["rule_v2_change_type"].eq("unknown_to_known").sum()), "count"),
        ("known_to_unknown", 0, int(audit["rule_v2_change_type"].eq("known_to_unknown").sum()), "count"),
    ]
    summary = pd.DataFrame(metrics, columns=["metric", "historical_value", "rule_v2_value", "value_type"])
    summary["difference"] = summary["rule_v2_value"] - summary["historical_value"]
    summary.to_csv(output_dir / "general_page_taxonomy_rule_v2_summary.csv", index=False)

    changed = audit[audit["rule_v2_label_changed"]].copy()
    review_parts = []
    for change_type in ("unknown_to_known", "known_to_unknown", "known_to_different_known"):
        group = changed[changed["rule_v2_change_type"].eq(change_type)].sort_values(
            ["source_appearances", "cited_appearances"], ascending=False
        )
        review_parts.append(group.head(100))
    review = pd.concat(review_parts, ignore_index=True)
    review.reindex(columns=[column for column in preferred if column in review]).to_csv(
        output_dir / "general_page_taxonomy_rule_v2_changed_review_sample.csv", index=False
    )

    result = {
        "status": "general_taxonomy_rule_v2_audit_complete",
        "input": str(input_path),
        "output_dir": str(output_dir),
        "unique_urls": len(audit),
        "historical_final_unknown_rate": _rate(audit["page_type_general"]),
        "rule_v2_final_unknown_rate": _rate(audit["page_type_general_rule_v2"]),
        "changed_final_labels": int(audit["rule_v2_label_changed"].sum()),
    }
    (output_dir / "general_page_taxonomy_rule_v2_run.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> int:
    package = econometrics_qa.default_package_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=package / "data/url_content_evidence_compact.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=topic_output_dir() / "tables/general_page_taxonomy_rule_v2",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
