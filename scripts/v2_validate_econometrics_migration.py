#!/usr/bin/env python3
"""Validate the migrated econometrics package against its canonical contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.econometrics_eda_v2.paths import topic_output_dir


def collect_metrics(frame: pd.DataFrame) -> dict[str, int]:
    metrics = {"rows": len(frame), "columns": len(frame.columns)}
    if "normalized_url" in frame:
        metrics["unique_urls"] = int(frame["normalized_url"].nunique())
    if "prompt_id" in frame:
        metrics["unique_prompts"] = int(frame["prompt_id"].nunique())
    if "source_root_domain" in frame:
        metrics["unique_domains"] = int(frame["source_root_domain"].nunique())
    if "cited" in frame:
        cited = pd.to_numeric(frame["cited"], errors="coerce").fillna(0)
        metrics["cited_rows"] = int(cited.sum())
    return metrics


def validate_package(package_dir: Path, manifest: dict) -> dict:
    checks: list[dict] = []
    for contract in manifest["data_contract"]:
        path = package_dir / contract["path"]
        check = {"path": contract["path"], "exists": path.exists()}
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            actual = collect_metrics(frame)
            expected = contract["expected"]
            missing_columns = sorted(set(contract["required_columns"]) - set(frame.columns))
            mismatches = {
                key: {"expected": value, "actual": actual.get(key)}
                for key, value in expected.items()
                if actual.get(key) != value
            }
            check.update(
                actual=actual,
                missing_columns=missing_columns,
                mismatches=mismatches,
                passed=not missing_columns and not mismatches,
            )
        else:
            check["passed"] = False
        checks.append(check)

    artifact_checks = [
        {"path": path, "exists": (package_dir / path).exists()}
        for path in manifest["required_model_artifacts"]
    ]
    passed = all(check["passed"] for check in checks) and all(
        check["exists"] for check in artifact_checks
    )
    return {
        "status": "migration_parity_passed" if passed else "migration_parity_failed",
        "study_name": manifest["study_name"],
        "full_audit_prompts": manifest["full_audit_prompt_count"],
        "measurable_content_prompts": manifest["measurable_content_prompt_count"],
        "data_checks": checks,
        "artifact_checks": artifact_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "config/econometrics_pipeline_manifest.json",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=topic_output_dir() / "content_econometrics_ai_package",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text("utf-8"))
    result = validate_package(args.package_dir, manifest)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", "utf-8")
    print(rendered)
    return 0 if result["status"] == "migration_parity_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
