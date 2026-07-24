from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_outcome_balance(df: pd.DataFrame, outdir: Path) -> list[str]:
    labels = df["cited"].map({1: "cited", 0: "more-only"}).fillna("unknown")
    counts = labels.value_counts().reindex(["cited", "more-only"], fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3))
    counts.plot(kind="bar", ax=ax, color=["#2c7fb8", "#f28e2b"])
    ax.set_ylabel("Rows")
    ax.set_title(f"Outcome Balance (cited rate={df['cited'].mean():.2%})")
    total = max(int(counts.sum()), 1)
    for i, v in enumerate(counts):
        ax.text(i, v, f"{int(v)}\n{v/total:.1%}", ha="center", va="bottom", fontsize=9)
    if (counts > 0).sum() < 2:
        ax.text(0.5, 0.85, "WARNING: outcome has one class.\nAssociation diagnostics skipped.", transform=ax.transAxes, ha="center", va="top", bbox={"facecolor": "#fff3cd", "edgecolor": "#b58900"})
    return [_save(fig, outdir / "01_outcome_balance.png")]


def plot_feature_availability(availability: pd.DataFrame, outdir: Path) -> list[str]:
    paths = []
    availability = availability[availability.get("show_in_main_coverage", True).astype(bool)].copy() if not availability.empty else availability
    if availability.empty:
        return paths
    grouped = availability.groupby("feature_group", as_index=False).agg(mean_coverage=("coverage", "mean"), features=("feature", "count"))
    grouped = grouped.sort_values("mean_coverage")
    fig, ax = plt.subplots(figsize=(7, max(3, len(grouped) * 0.45)))
    ax.barh(grouped["feature_group"], grouped["mean_coverage"], color="#59a14f")
    for _, r in grouped.iterrows():
        ax.text(r["mean_coverage"], r["feature_group"], f" {r['mean_coverage']:.0%} ({int(r['features'])} features)", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean non-null coverage")
    ax.set_title("Feature Coverage by Group")
    paths.append(_save(fig, outdir / "02_feature_coverage_by_group.png"))

    top = availability.sort_values("missing_rate", ascending=False).head(35)
    fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.24)))
    ax.barh(top["feature"], top["missing_rate"], color="#e15759")
    for _, r in top.iterrows():
        ax.text(r["missing_rate"], r["feature"], f" {int(r['non_null_count'])} non-null / {r['coverage']:.0%}", va="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Missing rate")
    ax.set_title("Top Missing Features")
    paths.append(_save(fig, outdir / "02_top_missing_features.png"))
    return paths


def plot_page_type_funnel(funnel: pd.DataFrame, outdir: Path) -> list[str]:
    if funnel.empty or not {"stage", "rows", "coverage"}.issubset(funnel.columns):
        return []
    fig, ax = plt.subplots(figsize=(7, max(3, len(funnel) * 0.45)))
    ax.barh(funnel["stage"], funnel["coverage"], color="#76b7b2")
    for _, r in funnel.iterrows():
        ax.text(r["coverage"], r["stage"], f" {int(r['rows'])} / {r['coverage']:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of all rows")
    ax.set_title("Page Type Coverage Funnel")
    return [_save(fig, outdir / "03_page_type_coverage_funnel.png")]


def plot_binary_rates(binary: pd.DataFrame, outdir: Path) -> list[str]:
    eligible = binary[binary.get("plot_eligible", False).astype(bool)].copy() if not binary.empty else binary
    if "scraped_subset_only" in eligible.columns:
        eligible = eligible[eligible["scraped_subset_only"].astype(bool)].copy()
    if eligible.empty:
        return []
    eligible = eligible.sort_values("diff_pp", key=lambda s: s.abs())
    fig, ax = plt.subplots(figsize=(8, max(4, len(eligible) * 0.32)))
    colors = np.where(eligible["sparse_flag"], "#f28e2b", "#4e79a7")
    ax.errorbar(eligible["diff_pp"], eligible["feature"], xerr=[eligible["diff_pp"] - eligible["ci_low"], eligible["ci_high"] - eligible["diff_pp"]], fmt="none", ecolor="#777", alpha=0.8)
    ax.scatter(eligible["diff_pp"], eligible["feature"], c=colors, zorder=3)
    for _, r in eligible.iterrows():
        ax.text(r["diff_pp"], r["feature"], f" n0={int(r['n0'])}, n1={int(r['n1'])}", va="center", ha="left" if r["diff_pp"] >= 0 else "right", fontsize=7)
    ax.axvline(0, color="#333", linewidth=1)
    ax.set_xlabel("Cited-rate difference, percentage points (feature=1 minus feature=0)")
    ax.set_title("Scraped-content binary feature cited-rate gaps among rows with content features")
    return [_save(fig, outdir / "04_binary_feature_forest_diff_pp.png")]


def plot_numeric_binned(numeric: pd.DataFrame, outdir: Path) -> list[str]:
    paths = []
    if numeric.empty:
        return paths
    for feature, g in numeric.groupby("feature"):
        g = g.sort_values("x_mean")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(g["x_mean"], g["cited_rate"], color="#4e79a7")
        ax.scatter(g["x_mean"], g["cited_rate"], s=np.maximum(g["n"], 1) * 8, color="#f28e2b", alpha=0.75)
        ax.set_ylabel("Cited rate")
        ax.set_xlabel(feature)
        ax.set_ylim(0, 1)
        ax.set_title(f"Numeric Binned Scatter: {feature}")
        paths.append(_save(fig, outdir / f"05_numeric_binned_{feature}.png"))
    return paths


def plot_categorical_rates(categorical: pd.DataFrame, outdir: Path) -> list[str]:
    paths = []
    if categorical.empty:
        return paths
    for feature, g in categorical.groupby("feature"):
        g = g.sort_values("n").tail(20)
        fig, ax = plt.subplots(figsize=(6, max(3, len(g) * 0.25)))
        xerr = [g["cited_rate"] - g["ci_low"], g["ci_high"] - g["cited_rate"]] if {"ci_low", "ci_high"}.issubset(g.columns) else None
        ax.errorbar(g["cited_rate"], g["category"], xerr=xerr, fmt="none", ecolor="#777", alpha=0.7)
        ax.scatter(g["cited_rate"], g["category"], s=np.maximum(g["n"], 1) * 8, color=np.where(g.get("sparse_flag", False), "#f28e2b", "#4e79a7"))
        for _, r in g.iterrows():
            ax.annotate(f"n={int(r['n'])}", (r["cited_rate"], r["category"]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cited rate")
        ax.set_title(f"Categorical Cited Rate: {feature}")
        paths.append(_save(fig, outdir / f"06_categorical_{feature}_cited_rate.png"))
    return paths


def plot_intent_page_type(df: pd.DataFrame, outdir: Path) -> list[str]:
    page_type_col = "page_type_final"
    if "intent_plot_label" not in df.columns or page_type_col not in df.columns:
        return []
    paths = []
    usable = df[df[page_type_col].notna() & (df[page_type_col].astype(str).str.strip() != "")].copy()
    if usable.empty:
        return []
    pt = pd.crosstab(usable["intent_plot_label"].fillna("missing_intent"), usable[page_type_col].astype(str))
    def _heat(mat, name, title, cmap="Blues", annotate_rate=False):
        if mat.empty:
            return None
        fig, ax = plt.subplots(figsize=(max(5, mat.shape[1] * 0.75), max(4, mat.shape[0] * 0.35)))
        arr = mat.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(arr)
        im = ax.imshow(masked, aspect="auto", cmap=cmap)
        ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=45, ha="right")
        ax.set_yticks(range(mat.shape[0]), mat.index)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.0%}" if annotate_rate else f"{int(val)}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax)
        return _save(fig, outdir / name)
    p = _heat(pt, "07_intent_page_type_cell_n.png", "Intent x Page Type Sample Size", "Blues")
    if p: paths.append(p)
    rate = usable.pivot_table(index=usable["intent_plot_label"].fillna("missing_intent"), columns=usable[page_type_col].astype(str), values="cited", aggfunc="mean")
    counts = pt.reindex_like(rate)
    rate = rate.where(counts >= 20)
    p = _heat(rate, "07_intent_page_type_cited_rate_eligible_only.png", "Intent x Page Type Cited Rate (n >= 20 only)", "Greens", annotate_rate=True)
    if p: paths.append(p)
    filt = usable[(usable["intent_plot_label"].fillna("missing_intent") != "missing_intent") & (usable[page_type_col].fillna("unknown") != "unknown")]
    pt2 = pd.crosstab(filt["intent_plot_label"], filt[page_type_col])
    p = _heat(pt2, "07_intent_page_type_filtered_cell_n.png", "Intent x Page Type Sample Size (filtered)", "Blues")
    if p: paths.append(p)
    return paths


def plot_correlation(corr: pd.DataFrame, outdir: Path) -> list[str]:
    if corr.empty:
        return []
    fig, ax = plt.subplots(figsize=(max(5, len(corr) * 0.4), max(4, len(corr) * 0.35)))
    im = ax.imshow(corr.fillna(0).to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Numeric Correlation Matrix")
    fig.colorbar(im, ax=ax)
    return [_save(fig, outdir / "correlation_matrix.png")]


def plot_lightgbm_discovery(df: pd.DataFrame, outdir: Path) -> list[str]:
    from lightgbm import LGBMClassifier
    from src.econometrics_eda_v2.leakage import DIAGNOSTIC_ONLY, LEAKAGE_EXCLUSIONS

    y = pd.to_numeric(df["cited"], errors="coerce").fillna(0).astype(int)
    X = df.drop(columns=[c for c in ["cited", "answer_text"] if c in df.columns]).copy()
    blocked = LEAKAGE_EXCLUSIONS | {"cited_label", "is_more_only", "source_group", "source_origin", "answer_text"}
    safe_blocked = blocked | DIAGNOSTIC_ONLY | {"domain_seen_count_loo", "relevance_score_prompt_only"}
    diagnostic_position_features = {"source_position", "observed_rank", "log1p_source_position"}
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c].fillna("(missing)").astype(str))[0]
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-1)
    if len(X) < 10 or y.nunique() < 2 or X.shape[1] < 1:
        return []
    paths = []
    model_sets = {
        "safe_discovery_model": [c for c in X.columns if c not in safe_blocked],
        "diagnostic_position_model": [c for c in X.columns if c not in blocked and (c not in DIAGNOSTIC_ONLY or c in diagnostic_position_features)],
    }
    for label, cols in model_sets.items():
        cols = [c for c in cols if X[c].nunique(dropna=True) > 1]
        if not cols:
            continue
        model = LGBMClassifier(n_estimators=40, min_child_samples=2, verbose=-1, random_state=7)
        model.fit(X[cols], y)
        imp = pd.Series(model.feature_importances_, index=cols).sort_values().tail(20)
        fig, ax = plt.subplots(figsize=(6, max(3, len(imp) * 0.25)))
        ax.barh(imp.index, imp.values, color="#59a14f")
        ax.set_title(f"LightGBM importance ({label})\nDiscovery-only, not causal importance.")
        ax.set_xlabel("Importance")
        paths.append(_save(fig, outdir / f"lightgbm_{label}.png"))
    return paths
