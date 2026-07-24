from __future__ import annotations

import numpy as np
import pandas as pd

from src.econometrics_eda_v2.leakage import DIAGNOSTIC_ONLY, LEAKAGE_EXCLUSIONS

CONTENT_BINARY_FEATURES = [
    "has_faq", "has_price_or_package", "has_contact_info", "has_table", "has_bullets",
    "has_author", "has_reviewer", "has_schema", "has_phone_number", "has_email",
    "has_address", "has_opening_hours", "has_booking_or_appointment", "has_step_by_step",
    "has_medical_disclaimer", "has_references", "has_updated_date",
]
BINARY_FEATURES = CONTENT_BINARY_FEATURES + ["https_flag", "url_has_query_params"]
NUMERIC_FEATURES = [
    "word_count", "heading_count", "table_count", "link_count", "title_prompt_similarity",
    "description_prompt_similarity", "page_prompt_similarity", "max_chunk_prompt_similarity",
    "relevance_score_prompt_only", "domain_seen_count", "domain_seen_count_loo",
    "url_length", "url_path_depth",
]
CATEGORICAL_FEATURES = [
    "intent", "topic", "language", "country", "source_type_url", "page_type_final",
    "page_type_family", "page_type_final_source", "domain_plot_label",
]

REPRESENTATIVE_CORRELATED_FEATURES = {"domain_seen_count", "title_prompt_similarity"}
REDUNDANT_CORRELATED_FEATURES = {"domain_seen_count_loo", "relevance_score_prompt_only"}


def build_lpm_readiness(df: pd.DataFrame, vif: pd.DataFrame | None = None) -> pd.DataFrame:
    features = [c for c in BINARY_FEATURES + NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    rows = []
    vif_flags = set()
    if vif is not None and not vif.empty and {"feature", "vif"}.issubset(vif.columns):
        vif_flags = set(vif[pd.to_numeric(vif["vif"], errors="coerce") >= 10]["feature"].astype(str))
    for f in features:
        s = df[f]
        role = "binary" if f in BINARY_FEATURES else ("numeric" if f in NUMERIC_FEATURES else "control")
        coverage = float(s.notna().mean()) if len(s) else 0.0
        missing_rate = 1 - coverage
        n_unique = int(s.dropna().nunique()) if len(s) else 0
        gap = np.nan
        sparse = False
        shape = ""
        if role == "binary":
            x = pd.to_numeric(s, errors="coerce")
            n0 = int((x == 0).sum())
            n1 = int((x == 1).sum())
            sparse = min(n0, n1) < 20
            if n0 and n1:
                gap = float(df.loc[x == 1, "cited"].mean() - df.loc[x == 0, "cited"].mean())
        elif role == "numeric":
            vals = pd.to_numeric(s, errors="coerce")
            shape = "consider log1p or bins" if f in {"word_count", "domain_seen_count", "domain_seen_count_loo"} else "linear form plausible; inspect binned plot"
            sparse = int(vals.notna().sum()) < 20
        else:
            sparse = int(s.nunique(dropna=True)) > max(1, len(df) // 5) or int(s.value_counts(dropna=True).min() if s.notna().any() else 0) < 5
        leakage = f in LEAKAGE_EXCLUSIONS or f in DIAGNOSTIC_ONLY
        diagnostic_only = f in DIAGNOSTIC_ONLY
        answer_leakage = f in LEAKAGE_EXCLUSIONS
        low_coverage = coverage < 0.6
        constant = n_unique < 2
        redundant_family = f in REDUNDANT_CORRELATED_FEATURES
        collinear = f in vif_flags or redundant_family
        single_class = "cited" in df.columns and pd.to_numeric(df["cited"], errors="coerce").nunique(dropna=True) < 2
        reason = []
        if single_class:
            reason.append("outcome has one class; no cited-rate association estimable")
        if constant:
            reason.append("constant in this run")
        if low_coverage:
            if f in CONTENT_BINARY_FEATURES:
                reason.append("low coverage; use scraped-subset sensitivity only")
            else:
                reason.append("low coverage / sensitivity only")
        if sparse:
            reason.append("sparse")
        if answer_leakage:
            reason.append("leakage")
        if diagnostic_only:
            reason.append("diagnostic-only")
        if collinear:
            if f in REPRESENTATIVE_CORRELATED_FEATURES:
                reason.append("high VIF; selected as representative of correlated family")
            else:
                reason.append("high VIF or redundant correlated family")
        if not reason:
            reason.append("usable descriptive candidate")
        recommended = bool(
            not single_class
            and not constant
            and not low_coverage
            and not sparse
            and not answer_leakage
            and not diagnostic_only
            and (not collinear or f in REPRESENTATIVE_CORRELATED_FEATURES)
        )
        rows.append(
            {
                "feature": f,
                "role": role,
                "coverage": coverage,
                "missing_rate": missing_rate,
                "n_unique": n_unique,
                "cited_rate_gap_if_binary": gap,
                "numeric_shape_recommendation": shape,
                "sparse_flag": sparse,
                "diagnostic_only_flag": diagnostic_only,
                "leakage_flag": answer_leakage,
                "collinearity_flag": collinear,
                "recommended_for_lpm": recommended,
                "recommendation_reason": "; ".join(reason),
            }
        )
    return pd.DataFrame(rows)
