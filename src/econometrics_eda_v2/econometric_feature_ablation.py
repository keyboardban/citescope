"""Offline with-versus-without feature comparisons for the QA frontend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .content_feature_econometrics import (
    GEMINI_PAGE_FAMILY_TERM,
    GEMINI_SOURCE_TYPE_TERM,
    GEMINI_TAXONOMY_TERMS,
    HEADING,
    LINK,
    M2_FORMULA,
    STRENGTH,
    forbidden_formula_matches,
)
from .gemini_taxonomy_features import (
    GEMINI_PAGE_FAMILY_COLLAPSED,
    GEMINI_SOURCE_TYPE_COLLAPSED,
    attach_gemini_taxonomy,
)
from .writing_factual_density_econometrics import MAIN_FEATURES, W1_FORMULA


OUTPUT_RELATIVE_PATH = Path("tables/frontend/econometric_feature_ablation.csv")


@dataclass(frozen=True)
class AblationSpec:
    model_family: str
    model_label: str
    full_formula: str
    features: tuple[tuple[str, str, str], ...]


SPECS = (
    AblationSpec(
        model_family="notebook_09_m2_structural",
        model_label="Notebook 09 M2 structural content",
        full_formula=M2_FORMULA,
        features=(
            ("log2_word_count_plus1", "Word count (log2)", "log2_word_count_plus1"),
            ("has_table", "HTML table presence", "has_table"),
            ("heading_count_group", "Heading count group", HEADING),
            ("link_count_group", "Link count group", LINK),
            ("content_strength", "Extraction strength", STRENGTH),
        ),
    ),
    AblationSpec(
        model_family="notebook_11_w1_writing",
        model_label="Notebook 11 W1 writing and factual density",
        full_formula=W1_FORMULA,
        features=tuple(
            (feature, feature.replace("_", " ").title(), feature) for feature in MAIN_FEATURES
        ),
    ),
    AblationSpec(
        model_family="notebook_09_m4_gemini_taxonomy",
        model_label="Notebook 09 M4 structural content + Gemini taxonomy",
        full_formula=f"{M2_FORMULA} + {GEMINI_TAXONOMY_TERMS}",
        features=(
            ("log2_word_count_plus1", "Word count (log2)", "log2_word_count_plus1"),
            ("has_table", "HTML table presence", "has_table"),
            ("heading_count_group", "Heading count group", HEADING),
            ("link_count_group", "Link count group", LINK),
            ("content_strength", "Extraction strength", STRENGTH),
            (
                GEMINI_PAGE_FAMILY_COLLAPSED,
                "Gemini page-function family",
                GEMINI_PAGE_FAMILY_TERM,
            ),
            (
                GEMINI_SOURCE_TYPE_COLLAPSED,
                "Gemini source/site type",
                GEMINI_SOURCE_TYPE_TERM,
            ),
        ),
    ),
)


def _prepare(frame: pd.DataFrame, spec: AblationSpec) -> pd.DataFrame:
    required = {"cited", "prompt_id"}
    for feature_name, _, _ in spec.features:
        required.add(feature_name)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing ablation columns: {', '.join(missing)}")

    data = frame.copy().reset_index(drop=True)
    data["cited"] = pd.to_numeric(data["cited"], errors="coerce")
    categorical = {
        "prompt_id",
        "heading_count_group",
        "link_count_group",
        "content_strength",
        GEMINI_PAGE_FAMILY_COLLAPSED,
        GEMINI_SOURCE_TYPE_COLLAPSED,
    }
    for column in required:
        if column in categorical:
            data[column] = data[column].fillna("unknown").astype(str)
        elif column != "cited":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.dropna(subset=sorted(required)).reset_index(drop=True)


def _metrics(result, outcome: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(result.fittedvalues, dtype=float)
    clipped = np.clip(prediction, 0, 1)
    residual = outcome - prediction
    return {
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "brier": float(np.mean(np.square(outcome - clipped))),
        "mae": float(np.mean(np.abs(residual))),
    }


def compute_feature_ablation(
    frame: pd.DataFrame,
    specs: tuple[AblationSpec, ...] = SPECS,
) -> pd.DataFrame:
    """Compare each full model with a nested model that removes one feature group."""
    rows: list[dict] = []
    for spec in specs:
        if forbidden_formula_matches(spec.full_formula):
            raise ValueError(f"Leakage guardrail failed for {spec.model_family}")
        data = _prepare(frame, spec)
        full = smf.ols(spec.full_formula, data=data).fit()
        outcome = data["cited"].to_numpy(dtype=float)
        full_metrics = _metrics(full, outcome)
        for feature_name, feature_label, formula_term in spec.features:
            remaining = [term for name, _, term in spec.features if name != feature_name]
            reduced_formula = "cited ~ " + " + ".join([*remaining, "C(prompt_id)"])
            if forbidden_formula_matches(reduced_formula):
                raise ValueError(f"Leakage guardrail failed after removing {feature_name}")
            reduced = smf.ols(reduced_formula, data=data).fit()
            reduced_metrics = _metrics(reduced, outcome)
            f_stat, f_p_value, df_difference = full.compare_f_test(reduced)
            partial_r_squared = (
                (reduced.ssr - full.ssr) / reduced.ssr if reduced.ssr > 0 else np.nan
            )
            rows.append(
                {
                    "model_family": spec.model_family,
                    "model_label": spec.model_label,
                    "feature": feature_name,
                    "feature_label": feature_label,
                    "formula_term": formula_term,
                    "with_feature_formula": spec.full_formula,
                    "without_feature_formula": reduced_formula,
                    "n_obs": int(full.nobs),
                    "n_prompts": int(data["prompt_id"].nunique()),
                    "feature_df": int(round(df_difference)),
                    "with_r_squared": full_metrics["r_squared"],
                    "without_r_squared": reduced_metrics["r_squared"],
                    "r_squared_gain": full_metrics["r_squared"] - reduced_metrics["r_squared"],
                    "partial_r_squared": partial_r_squared,
                    "with_adjusted_r_squared": full_metrics["adjusted_r_squared"],
                    "without_adjusted_r_squared": reduced_metrics["adjusted_r_squared"],
                    "adjusted_r_squared_gain": (
                        full_metrics["adjusted_r_squared"] - reduced_metrics["adjusted_r_squared"]
                    ),
                    "with_rmse": full_metrics["rmse"],
                    "without_rmse": reduced_metrics["rmse"],
                    "rmse_reduction": reduced_metrics["rmse"] - full_metrics["rmse"],
                    "with_brier": full_metrics["brier"],
                    "without_brier": reduced_metrics["brier"],
                    "brier_reduction": reduced_metrics["brier"] - full_metrics["brier"],
                    "with_mae": full_metrics["mae"],
                    "without_mae": reduced_metrics["mae"],
                    "mae_reduction": reduced_metrics["mae"] - full_metrics["mae"],
                    "nested_f_statistic": float(f_stat),
                    "nested_f_p_value": float(f_p_value),
                    "leakage_guardrail_passed": True,
                    "interpretation": (
                        "In-sample nested-model contribution among surfaced sources; not causal and "
                        "not an out-of-sample performance claim."
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_feature_ablation(package_dir: str | Path) -> Path:
    package = Path(package_dir)
    data_path = package / "data/content_lpm_measurable_rows_with_writing_factual_features.csv"
    output_path = package / OUTPUT_RELATIVE_PATH
    data, _, audit = attach_gemini_taxonomy(
        pd.read_csv(data_path, low_memory=False),
        package,
    )
    result = compute_feature_ablation(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    pd.DataFrame([audit]).to_csv(
        output_path.with_name("econometric_feature_ablation_gemini_join_audit.csv"),
        index=False,
    )
    return output_path
