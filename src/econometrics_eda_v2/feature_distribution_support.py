"""Governed distribution and prompt-variation artifacts for feature QA."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.manual_feature_validation import COMPONENTS


ARTIFACT_VERSION = "feature_distribution_support_v1"
FOCAL_FEATURES = (
    "log2_word_count_plus1",
    "has_verified_html_table",
    "factual_numeric_density_score",
    "writing_structure_score",
)
BINARY_FEATURES = ("has_verified_html_table", *COMPONENTS)
CONTINUOUS_FEATURES = ("log2_word_count_plus1", "factual_numeric_density_score")
SCORE_FEATURES = ("writing_structure_score",)
CATEGORICAL_FEATURES = ("content_strength",)
OPTIONAL_DIAGNOSTICS = ("heading_count_group",)
DASHBOARD_FEATURES = (*FOCAL_FEATURES, "content_strength", *COMPONENTS, *OPTIONAL_DIAGNOSTICS)
FEATURE_LABELS = {
    "log2_word_count_plus1": "Measured Content Length (log2)",
    "has_verified_html_table": "Verified HTML Table Presence",
    "factual_numeric_density_score": "Factual and Numeric Specificity Score",
    "writing_structure_score": "Answer-Oriented Writing Structure Score",
    "content_strength": "Extraction Strength",
    "has_bullet_list": "Bullet List Detected",
    "has_numbered_list": "Numbered List Detected",
    "has_faq_pattern": "FAQ Pattern Detected",
    "has_question_answer_structure": "Question-Answer Structure Detected",
    "opening_has_summary_signal": "Opening Summary Signal",
    "opening_has_direct_answer_signal": "Opening Direct-Answer Signal",
    "heading_count_group": "Heading Count Group (D0/QA only)",
}
SUPPORT_FILES = (
    "feature_distribution_support_summary.csv",
    "feature_distribution_bins.csv",
    "feature_within_prompt_variation.csv",
    "feature_cited_rate_by_bin.csv",
    "feature_distribution_by_content_strength.csv",
    "manual_qa_review_rows.csv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_type(feature: str) -> str:
    if feature in BINARY_FEATURES:
        return "binary"
    if feature in CONTINUOUS_FEATURES:
        return "continuous"
    if feature in SCORE_FEATURES:
        return "score"
    if feature in OPTIONAL_DIAGNOSTICS:
        return "diagnostic_categorical"
    return "categorical"


def _numeric(frame: pd.DataFrame, feature: str) -> pd.Series:
    return pd.to_numeric(frame[feature], errors="coerce")


def feature_bins(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Return stable per-row bin metadata without converting missing values to zero."""
    result = pd.DataFrame(index=frame.index)
    kind = feature_type(feature)
    if kind == "binary":
        numeric = _numeric(frame, feature)
        result["bin_key"] = np.where(
            numeric.isna(), "unmeasured", np.where(numeric.eq(1), "detected", "not_detected")
        )
        result["bin_label"] = result["bin_key"].map(
            {"not_detected": "Not detected", "detected": "Detected", "unmeasured": "Unmeasured"}
        )
        result["bin_order"] = result["bin_key"].map(
            {"not_detected": 0, "detected": 1, "unmeasured": 2}
        )
        result["bin_low"] = numeric
        result["bin_high"] = numeric
        return result
    if kind == "score":
        numeric = _numeric(frame, feature)
        result["bin_key"] = numeric.map(lambda value: f"score_{int(value)}" if pd.notna(value) else "unmeasured")
        result["bin_label"] = numeric.map(lambda value: str(int(value)) if pd.notna(value) else "NA")
        result["bin_order"] = numeric.fillna(7)
        result["bin_low"] = numeric
        result["bin_high"] = numeric
        return result
    if kind == "continuous":
        numeric = _numeric(frame, feature)
        measured = numeric.dropna()
        result["bin_key"] = "unmeasured"
        result["bin_label"] = "NA"
        result["bin_order"] = 99
        result["bin_low"] = np.nan
        result["bin_high"] = np.nan
        if measured.empty:
            return result
        quantiles = min(5, measured.nunique())
        if quantiles < 2:
            result.loc[measured.index, ["bin_key", "bin_label", "bin_order"]] = [
                "bin_1", f"{measured.iloc[0]:.3g}", 0
            ]
            result.loc[measured.index, "bin_low"] = measured.iloc[0]
            result.loc[measured.index, "bin_high"] = measured.iloc[0]
            return result
        assigned = pd.qcut(measured, q=quantiles, duplicates="drop")
        intervals = sorted(assigned.cat.categories, key=lambda interval: interval.left)
        for order, interval in enumerate(intervals):
            member_index = assigned[assigned.eq(interval)].index
            key = f"q{order + 1}"
            observed_low = float(measured.loc[member_index].min())
            observed_high = float(measured.loc[member_index].max())
            label = f"Q{order + 1}: {observed_low:.3g} to {observed_high:.3g}"
            result.loc[member_index, "bin_key"] = key
            result.loc[member_index, "bin_label"] = label
            result.loc[member_index, "bin_order"] = order
            result.loc[member_index, "bin_low"] = observed_low
            result.loc[member_index, "bin_high"] = observed_high
        return result

    values = frame[feature].astype("string")
    if feature == "content_strength":
        normalized = values.str.casefold().where(
            values.str.casefold().isin(["strong", "medium", "weak"]), "failed_or_unknown"
        )
        order = {"strong": 0, "medium": 1, "weak": 2, "failed_or_unknown": 3}
    elif feature == "heading_count_group":
        normalized = values.fillna("NA")
        order = {"0-1": 0, "2-6": 1, "7-12": 2, "13+": 3, "NA": 4}
    else:
        normalized = values.fillna("NA")
        order = {value: index for index, value in enumerate(sorted(normalized.unique()))}
    result["bin_key"] = normalized.astype(str)
    result["bin_label"] = normalized.astype(str)
    result["bin_order"] = normalized.map(order).fillna(99)
    result["bin_low"] = np.nan
    result["bin_high"] = np.nan
    return result


def prompt_variation(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    numeric_kind = feature_type(feature) in {"binary", "continuous", "score"}
    values = _numeric(frame, feature) if numeric_kind else frame[feature].astype("string")
    working = pd.DataFrame({"prompt_id": frame["prompt_id"], "value": values})
    total_prompts = int(frame["prompt_id"].nunique())
    unique_by_prompt = working.groupby("prompt_id")["value"].nunique(dropna=True)
    varying_prompts = set(unique_by_prompt[unique_by_prompt.gt(1)].index.astype(str))
    prompt_ids = frame["prompt_id"].astype(str)
    rows_in_varying = int(prompt_ids.isin(varying_prompts).sum())
    measured_by_prompt = working.dropna(subset=["value"]).groupby("prompt_id")["value"]
    if numeric_kind:
        within_sd = measured_by_prompt.std(ddof=0)
        varying_mask = unique_by_prompt.gt(1)
        median_within_sd = (
            float(within_sd.reindex(unique_by_prompt.index)[varying_mask].median())
            if varying_prompts
            else np.nan
        )
    else:
        median_within_sd = np.nan
    both_binary = 0
    if feature_type(feature) == "binary":
        both_binary = int(
            measured_by_prompt.apply(lambda series: set(pd.to_numeric(series, errors="coerce").dropna()) >= {0, 1}).sum()
        )
    return {
        "feature_name": feature,
        "feature_type": feature_type(feature),
        "total_prompts": total_prompts,
        "prompts_with_no_variation": total_prompts - len(varying_prompts),
        "prompts_with_usable_variation": len(varying_prompts),
        "prompts_with_variation_pct": len(varying_prompts) / total_prompts if total_prompts else np.nan,
        "rows_in_prompts_with_usable_variation": rows_in_varying,
        "rows_in_prompts_with_usable_variation_pct": rows_in_varying / len(frame) if len(frame) else np.nan,
        "median_within_prompt_std": median_within_sd,
        "prompts_containing_both_0_and_1": both_binary if feature_type(feature) == "binary" else np.nan,
        "_varying_prompts": varying_prompts,
    }


def wilson_interval(cited_rows: int, n_rows: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n_rows <= 0:
        return np.nan, np.nan
    rate = cited_rows / n_rows
    denominator = 1 + z * z / n_rows
    center = (rate + z * z / (2 * n_rows)) / denominator
    spread = z * math.sqrt(rate * (1 - rate) / n_rows + z * z / (4 * n_rows * n_rows)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def distribution_table(frame: pd.DataFrame, feature: str) -> pd.DataFrame:
    bins = feature_bins(frame, feature)
    working = pd.concat(
        [
            frame[["prompt_id", "cited", "content_strength"]].reset_index(drop=True),
            bins.reset_index(drop=True),
        ],
        axis=1,
    )
    variation = prompt_variation(frame, feature)
    working["prompt_varies"] = working["prompt_id"].astype(str).isin(variation["_varying_prompts"])
    measured_n = int(frame[feature].notna().sum())
    rows: list[dict[str, Any]] = []
    for (key, label, order, low, high), group in working.groupby(
        ["bin_key", "bin_label", "bin_order", "bin_low", "bin_high"], dropna=False, sort=False
    ):
        n_rows = len(group)
        cited_rows = int(pd.to_numeric(group["cited"], errors="coerce").fillna(0).eq(1).sum())
        lower, upper = wilson_interval(cited_rows, n_rows)
        unmeasured = key == "unmeasured" or label in {"NA", "Unmeasured"}
        rows.append(
            {
                "feature_name": feature,
                "feature_type": feature_type(feature),
                "bin_key": key,
                "bin_label": label,
                "bin_order": int(order),
                "bin_low": low,
                "bin_high": high,
                "n_rows": n_rows,
                "percentage_all_rows": n_rows / len(frame) if len(frame) else np.nan,
                "percentage_measured_rows": np.nan if unmeasured or not measured_n else n_rows / measured_n,
                "cited_rows": cited_rows,
                "more_only_rows": n_rows - cited_rows,
                "cited_rate": cited_rows / n_rows if n_rows else np.nan,
                "ci_low": lower,
                "ci_high": upper,
                "unique_prompts": int(group["prompt_id"].nunique()),
                "prompts_with_usable_variation": int(
                    group.loc[group["prompt_varies"], "prompt_id"].nunique()
                ),
            }
        )
    output = pd.DataFrame(rows)
    if feature_type(feature) == "binary":
        binary_levels = [
            ("not_detected", "Not detected", 0),
            ("detected", "Detected", 1),
            ("unmeasured", "Unmeasured", 2),
        ]
        existing = set(output["bin_key"]) if not output.empty else set()
        empty_rows = []
        for key, label, order in binary_levels:
            if key in existing:
                continue
            empty_rows.append(
                {
                    "feature_name": feature,
                    "feature_type": "binary",
                    "bin_key": key,
                    "bin_label": label,
                    "bin_order": order,
                    "bin_low": np.nan if key == "unmeasured" else order,
                    "bin_high": np.nan if key == "unmeasured" else order,
                    "n_rows": 0,
                    "percentage_all_rows": 0.0,
                    "percentage_measured_rows": np.nan if key == "unmeasured" else 0.0,
                    "cited_rows": 0,
                    "more_only_rows": 0,
                    "cited_rate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "unique_prompts": 0,
                    "prompts_with_usable_variation": 0,
                }
            )
        output = pd.concat([output, pd.DataFrame(empty_rows)], ignore_index=True)
    if feature == "writing_structure_score":
        existing = set(output["bin_key"]) if not output.empty else set()
        empty_rows = []
        for score in range(7):
            key = f"score_{score}"
            if key not in existing:
                empty_rows.append(
                    {
                        "feature_name": feature,
                        "feature_type": "score",
                        "bin_key": key,
                        "bin_label": str(score),
                        "bin_order": score,
                        "n_rows": 0,
                        "percentage_all_rows": 0.0,
                        "percentage_measured_rows": 0.0,
                        "cited_rows": 0,
                        "more_only_rows": 0,
                        "cited_rate": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "unique_prompts": 0,
                        "prompts_with_usable_variation": 0,
                    }
                )
        if "unmeasured" not in existing:
            empty_rows.append(
                {
                    "feature_name": feature,
                    "feature_type": "score",
                    "bin_key": "unmeasured",
                    "bin_label": "NA",
                    "bin_order": 7,
                    "n_rows": 0,
                    "percentage_all_rows": 0.0,
                    "percentage_measured_rows": np.nan,
                    "cited_rows": 0,
                    "more_only_rows": 0,
                    "cited_rate": np.nan,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "unique_prompts": 0,
                    "prompts_with_usable_variation": 0,
                }
            )
        output = pd.concat([output, pd.DataFrame(empty_rows)], ignore_index=True)
    return output.sort_values(["bin_order", "bin_label"]).reset_index(drop=True)


def _extraction_dependence(frame: pd.DataFrame, feature: str) -> tuple[bool, str]:
    if feature == "content_strength":
        return False, "Measurement control itself"
    kind = feature_type(feature)
    if kind not in {"binary", "continuous", "score"}:
        return False, ""
    numeric = _numeric(frame, feature)
    grouped = pd.DataFrame({"value": numeric, "strength": frame["content_strength"]}).dropna()
    means = grouped.groupby("strength")["value"].mean()
    if len(means) < 2:
        return False, ""
    if kind == "binary":
        flagged = float(means.max() - means.min()) >= 0.20
        return flagged, "Detected prevalence differs by at least 20 pp across extraction-strength groups" if flagged else ""
    overall_sd = float(numeric.std(ddof=1))
    flagged = overall_sd > 0 and float(means.max() - means.min()) / overall_sd >= 0.5
    return flagged, "Strength-group means differ by at least 0.5 overall SD" if flagged else ""


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    total_n = len(frame)
    total_prompts = int(frame["prompt_id"].nunique())
    rows: list[dict[str, Any]] = []
    for feature in DASHBOARD_FEATURES:
        kind = feature_type(feature)
        series = frame[feature]
        numeric = _numeric(frame, feature) if kind in {"binary", "continuous", "score"} else None
        measured = numeric.dropna() if numeric is not None else series.dropna()
        measured_n = len(measured)
        missing_n = total_n - measured_n
        variation = prompt_variation(frame, feature)
        prevalence_or_mean = (
            float(numeric.mean()) if numeric is not None and measured_n else np.nan
        )
        detected_n = int(numeric.eq(1).sum()) if kind == "binary" else np.nan
        low_support: list[str] = []
        imbalance: list[str] = []
        if measured_n < 100:
            low_support.append("fewer than 100 measured rows")
        if kind == "binary" and measured_n:
            prevalence = float(numeric.mean())
            if detected_n < 20:
                low_support.append("fewer than 20 detected rows")
            if prevalence < 0.05:
                imbalance.append("detected prevalence below 5%")
            if prevalence > 0.95:
                imbalance.append("detected prevalence above 95%")
        observed_bins = distribution_table(frame, feature)
        if kind in {"score", "categorical", "diagnostic_categorical"}:
            sparse_levels = observed_bins[
                observed_bins["n_rows"].gt(0) & observed_bins["n_rows"].lt(20)
            ]
            if not sparse_levels.empty:
                low_support.append("one or more observed levels have fewer than 20 rows")
        missing_pct = missing_n / total_n if total_n else np.nan
        variation_pct = variation["prompts_with_variation_pct"]
        if missing_pct > 0.20:
            imbalance.append("missingness above 20%")
        if pd.notna(variation_pct) and variation_pct < 0.10:
            imbalance.append("fewer than 10% of prompts vary")
        extraction_flag, extraction_note = _extraction_dependence(frame, feature)
        quantiles = numeric.quantile([0.05, 0.25, 0.5, 0.75, 0.95]) if numeric is not None else {}
        imbalance_score = max(
            missing_pct if pd.notna(missing_pct) else 0,
            abs(prevalence_or_mean - 0.5) * 2 if kind == "binary" and pd.notna(prevalence_or_mean) else 0,
            1 - variation_pct if pd.notna(variation_pct) else 0,
        )
        rows.append(
            {
                "feature_name": feature,
                "feature_label": FEATURE_LABELS[feature],
                "feature_type": kind,
                "model_role": (
                    "focal_predictor" if feature in FOCAL_FEATURES
                    else "measurement_control" if feature == "content_strength"
                    else "writing_component" if feature in COMPONENTS
                    else "D0_QA_diagnostic_only"
                ),
                "total_observations": total_n,
                "measured_n": measured_n,
                "missing_n": missing_n,
                "missing_pct": missing_pct,
                "unique_values": int(series.nunique(dropna=True)),
                "mean": float(numeric.mean()) if numeric is not None and measured_n else np.nan,
                "standard_deviation": float(numeric.std(ddof=1)) if numeric is not None and measured_n > 1 else np.nan,
                "minimum": float(numeric.min()) if numeric is not None and measured_n else np.nan,
                "p05": quantiles.get(0.05, np.nan),
                "p25": quantiles.get(0.25, np.nan),
                "median": quantiles.get(0.5, np.nan),
                "p75": quantiles.get(0.75, np.nan),
                "p95": quantiles.get(0.95, np.nan),
                "maximum": float(numeric.max()) if numeric is not None and measured_n else np.nan,
                "number_of_prompts": total_prompts,
                "prompts_with_variation": variation["prompts_with_usable_variation"],
                "prompts_with_variation_pct": variation_pct,
                "rows_within_identifying_prompts": variation["rows_in_prompts_with_usable_variation"],
                "rows_within_identifying_prompts_pct": variation[
                    "rows_in_prompts_with_usable_variation_pct"
                ],
                "prevalence_or_mean": prevalence_or_mean,
                "zero_pct": float(numeric.eq(0).sum() / measured_n) if numeric is not None and measured_n else np.nan,
                "low_support_warning": "; ".join(low_support),
                "imbalance_warning": "; ".join(imbalance),
                "imbalance_score": imbalance_score,
                "extraction_dependence_warning": extraction_note if extraction_flag else "",
            }
        )
    return pd.DataFrame(rows)


def build_support_artifacts(
    review_rows: pd.DataFrame,
    tables_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    missing = [feature for feature in DASHBOARD_FEATURES if feature not in review_rows.columns]
    if missing:
        raise ValueError(f"Distribution dashboard fields missing from review rows: {missing}")
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = feature_summary(review_rows)
    distributions = pd.concat(
        [distribution_table(review_rows, feature) for feature in DASHBOARD_FEATURES],
        ignore_index=True,
    )
    within_rows = []
    for feature in FOCAL_FEATURES:
        row = prompt_variation(review_rows, feature)
        row.pop("_varying_prompts")
        within_rows.append(row)
    within = pd.DataFrame(within_rows)
    cited = distributions[
        [
            "feature_name", "feature_type", "bin_key", "bin_label", "bin_order",
            "bin_low", "bin_high", "n_rows", "cited_rows", "more_only_rows",
            "cited_rate", "ci_low", "ci_high", "unique_prompts",
        ]
    ].copy()
    strength_rows = []
    for feature in DASHBOARD_FEATURES:
        bins = feature_bins(review_rows, feature)
        working = pd.concat(
            [
                review_rows[["content_strength", "cited"]].reset_index(drop=True),
                bins.reset_index(drop=True),
            ],
            axis=1,
        )
        for (strength, key, label, order), group in working.groupby(
            ["content_strength", "bin_key", "bin_label", "bin_order"], dropna=False
        ):
            cited_rows = int(pd.to_numeric(group["cited"], errors="coerce").fillna(0).eq(1).sum())
            strength_rows.append(
                {
                    "feature_name": feature,
                    "content_strength": strength,
                    "bin_key": key,
                    "bin_label": label,
                    "bin_order": order,
                    "n_rows": len(group),
                    "percentage_within_strength": len(group)
                    / int(review_rows["content_strength"].eq(strength).sum()),
                    "cited_rows": cited_rows,
                    "cited_rate": cited_rows / len(group),
                }
            )
    strength_distribution = pd.DataFrame(strength_rows)
    review_export = review_rows.copy()
    output_frames = {
        "feature_distribution_support_summary.csv": summary,
        "feature_distribution_bins.csv": distributions,
        "feature_within_prompt_variation.csv": within,
        "feature_cited_rate_by_bin.csv": cited,
        "feature_distribution_by_content_strength.csv": strength_distribution,
        "manual_qa_review_rows.csv": review_export,
    }
    files = {}
    for filename, frame in output_frames.items():
        path = tables_dir / filename
        frame.to_csv(path, index=False)
        files[filename] = {"path": str(path), "sha256": _sha256(path), "rows": len(frame)}
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "validated": True,
        "model_ready_rows": len(review_rows),
        "unique_prompts": int(review_rows["prompt_id"].nunique()),
        "features": list(DASHBOARD_FEATURES),
        "focal_features": list(FOCAL_FEATURES),
        "blocked_features": ["external_evidence_structure_score"],
        "diagnostic_only_features": list(OPTIONAL_DIAGNOSTICS),
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_support_artifacts(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("validated"):
        raise ValueError("Feature distribution artifact is not validated")
    frames: dict[str, pd.DataFrame] = {}
    for filename, metadata in manifest["files"].items():
        path = Path(metadata["path"])
        if not path.exists() or _sha256(path) != metadata["sha256"]:
            raise ValueError(f"Feature distribution artifact hash mismatch: {filename}")
        frames[filename] = pd.read_csv(path, low_memory=False)
        if "bin_label" in frames[filename].columns:
            frames[filename]["bin_label"] = frames[filename]["bin_label"].fillna("NA")
    return frames, manifest


def apply_review_filter(
    frame: pd.DataFrame,
    feature: str,
    *,
    bin_key: str | None = None,
    content_strength: str | None = None,
    variation_mode: str = "all",
    variation_reference: pd.DataFrame | None = None,
    bin_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if bin_key:
        reference = bin_reference if bin_reference is not None else frame
        governed_bins = feature_bins(reference, feature)["bin_key"].reindex(frame.index)
        mask &= governed_bins.eq(bin_key)
    if content_strength:
        mask &= frame["content_strength"].astype(str).eq(str(content_strength))
    if variation_mode != "all":
        variation = prompt_variation(
            variation_reference if variation_reference is not None else frame,
            feature,
        )
        prompt_varies = frame["prompt_id"].astype(str).isin(variation["_varying_prompts"])
        mask &= prompt_varies if variation_mode == "with_variation" else ~prompt_varies
    return frame.loc[mask].copy()
