"""Position-adjusted **citation model** — the multivariate, error-bar-bearing
upgrade of the univariate `analysis.correlation_with_citation`.

The question: *what features make a source more likely to be **cited** (1) vs
**more-only / non-cited** (0), holding the other features — especially
**position** — fixed, and how sure are we?* We fit a **Linear Probability Model**
(OLS on the 0/1 `cited` outcome, so coefficients are in probability points) with
**heteroskedasticity-robust (HC3)** standard errors by default, **cluster-robust**
SEs when sources nest within prompts (cluster on `record_id` / `run_id`), plus
**VIF** multicollinearity diagnostics and a **Benjamini–Hochberg** FDR correction
over the focal feature family.

Framing (a scoped exception to the app's observational rule): coefficients may be
read as *cautious effect estimates*, but ONLY under stated assumptions
(exogeneity, positivity, functional form) and a **signed omitted-variable caveat**.
Robust error bars are honest about noise — never about an unobserved confounder.

statsmodels is required; it is imported behind a guard so the rest of the app keeps
working (with a clear message) if it is absent. Logit/AME (cross-check) and the wild
cluster bootstrap slot into the same result schema in later slices.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config

try:  # graceful: the engine degrades to a clear message, never a crash.
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    HAVE_STATSMODELS = True
    _IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001
    HAVE_STATSMODELS = False
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

_MIN_COVERAGE = 0.5      # a candidate regressor must be non-null on ≥ this share of rows
_MIN_SUPPORT = 3         # a dummy/boolean must have ≥ this many 1s (and 0s) to enter
_RARE_LEVEL = 3          # categorical levels with < this many rows collapse to "other"
_LEVERAGE_FLOOR = 1e-6   # guard 1/(1-h) blow-ups (not used for HC3 via statsmodels, kept for safety)

_OVB_BY_CONTEXT = {
    "gemini": "CAVEAT_OVB_GEMINI",
    "chatgpt": "CAVEAT_OVB_CHATGPT",
    "brand": "CAVEAT_OVB_BRAND",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _f(x):
    """JSON-safe float: None for NaN/inf, else rounded float."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, 6)


def available() -> bool:
    return HAVE_STATSMODELS


def unavailable_result(context: str = "", note: str = "") -> dict:
    return {
        "available": False,
        "context": context,
        "warnings": [note or f"statsmodels not installed ({_IMPORT_ERROR}). "
                             "Run `pip install statsmodels scipy` to enable the citation model."],
        "coefficients": [], "diagnostics": {}, "assumptions": [], "ovb_caveat": "",
    }


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
def build_spec(focal: list[str], position_col: str, *, controls: list[str] | None = None,
               position_fallbacks: list[str] | None = None, position_spec: str = "log1p",
               categoricals: list[str] | None = None, cluster_key: str | None = None,
               phase_map: dict | None = None, labels: dict | None = None,
               outcome: str = "cited", model: str = "lpm", context: str = "",
               phase_filter: str | None = None, title: str = "",
               crosscheck_logit: bool = True, wild_bootstrap: bool = True) -> dict:
    """Describe one regression. `focal` features get the BH correction + are read as
    the headline; `controls` (incl. position) are adjusted-for nuisances."""
    return {
        "focal": list(focal),
        "controls": list(controls or []),
        "position_col": position_col,
        "position_fallbacks": list(position_fallbacks or []),
        "position_spec": position_spec,
        "categoricals": list(categoricals or []),
        "cluster_key": cluster_key,
        "phase_map": phase_map or {},
        "labels": labels or {},
        "outcome": outcome,
        "model": model,
        "context": context,
        "phase_filter": phase_filter,   # e.g. only "pre_answer" focal features (non-circular)
        "title": title,
        "crosscheck_logit": crosscheck_logit,
        "wild_bootstrap": wild_bootstrap,
    }


# --------------------------------------------------------------------------- #
# design matrix
# --------------------------------------------------------------------------- #
def _coerce_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _pos_band(p) -> str:
    try:
        v = int(p)
    except (TypeError, ValueError):
        return "unknown"
    if v <= 0:
        return "unknown"
    if v <= 3:
        return "1-3"
    if v <= 6:
        return "4-6"
    if v <= 10:
        return "7-10"
    return "11+"


def design_matrix(df: pd.DataFrame, spec: dict) -> dict:
    """Build (X, y) for one spec. Returns a dict with X (incl. const), y, focal/control
    column lists, the cluster groups aligned to X's rows, dropped columns, and diagnostics."""
    dropped: list[dict] = []
    diagnostics: dict = {}
    outcome = spec["outcome"]
    if outcome not in df.columns:
        return {"error": f"outcome '{outcome}' not in data"}

    y = pd.to_numeric(df[outcome], errors="coerce")
    base = df[y.notna()].copy()
    y = y[y.notna()]
    if base.empty:
        return {"error": "no rows with a valid outcome"}

    cols: dict[str, pd.Series] = {}
    focal_cols: list[str] = []
    control_cols: list[str] = []

    def _consider(name: str, series: pd.Series, *, focal: bool, support_check: bool) -> None:
        s = _coerce_numeric(series)
        cov = s.notna().mean()
        if cov < _MIN_COVERAGE:
            dropped.append({"name": name, "reason": "low_coverage"})
            return
        nun = s.dropna().nunique()
        if nun < 2:
            dropped.append({"name": name, "reason": "zero_variance"})
            return
        if support_check:  # 0/1 column needs enough of each class
            ones = float((s.dropna() == 1).sum())
            if ones < _MIN_SUPPORT or (len(s.dropna()) - ones) < _MIN_SUPPORT:
                dropped.append({"name": name, "reason": "low_support"})
                return
        cols[name] = s
        (focal_cols if focal else control_cols).append(name)

    # 1) position control (median-fill + missing indicator; transform)
    pos_name = None
    for cand in [spec["position_col"], *spec["position_fallbacks"]]:
        if cand and cand in base.columns and _coerce_numeric(base[cand]).notna().any():
            pos_name = cand
            break
    if pos_name:
        pos = _coerce_numeric(base[pos_name])
        miss = pos.isna()
        pos = pos.fillna(pos.median())
        if spec["position_spec"] == "log1p":
            cols[f"log1p_{pos_name}"] = np.log1p(pos.clip(lower=0))
            control_cols.append(f"log1p_{pos_name}")
        elif spec["position_spec"] == "bins":
            band = pos.apply(_pos_band)                      # flexible shape, reference = "1-3"
            for b in ("4-6", "7-10", "11+"):
                ind = (band == b).astype(float)
                if ind.sum() >= _MIN_SUPPORT:
                    cols[f"{pos_name}_band={b}"] = ind
                    control_cols.append(f"{pos_name}_band={b}")
        else:
            cols[pos_name] = pos
            control_cols.append(pos_name)
        if miss.any():
            cols[f"{pos_name}_missing"] = miss.astype(float)
            control_cols.append(f"{pos_name}_missing")
        diagnostics["position_col"] = pos_name

    # 2) focal numeric/boolean features
    for name in spec["focal"]:
        if name in base.columns and name not in cols:
            uniq = pd.to_numeric(base[name], errors="coerce").dropna().unique()
            is_binary = set(np.unique(uniq)).issubset({0.0, 1.0}) and len(uniq) > 0
            _consider(name, base[name], focal=True, support_check=is_binary)

    # 3) explicit numeric controls
    for name in spec["controls"]:
        if name in base.columns and name not in cols:
            _consider(name, base[name], focal=False, support_check=False)

    # 4) categoricals -> one-hot (collapse rare, drop reference level)
    reference_levels: dict[str, str] = {}
    for cat in spec["categoricals"]:
        if cat not in base.columns:
            continue
        raw = base[cat].astype("string").fillna("unknown")
        counts = raw.value_counts()
        rare = set(counts[counts < _RARE_LEVEL].index)
        lvls = raw.where(~raw.isin(rare), other="other")
        uniq = list(lvls.value_counts().index)
        if len(uniq) < 2:
            dropped.append({"name": cat, "reason": "single_level"})
            continue
        ref = uniq[0]  # most common = reference
        reference_levels[cat] = ref
        for lvl in uniq[1:]:
            col = f"{cat}={lvl}"
            ind = (lvls == lvl).astype(float)
            if ind.sum() < _MIN_SUPPORT:
                dropped.append({"name": col, "reason": "low_support"})
                continue
            cols[col] = ind
            focal_cols.append(col)

    if not focal_cols:
        return {"error": "no usable focal features", "dropped": dropped}

    # 5) assemble, listwise-drop remaining NaN rows, add const
    X = pd.DataFrame(cols, index=base.index)
    keep = X.notna().all(axis=1)
    X, y2 = X[keep], y[keep]
    n_dropped_rows = int((~keep).sum())
    if n_dropped_rows:
        diagnostics["rows_dropped_missing"] = n_dropped_rows

    # drop any column that lost its variance after row filtering
    for c in list(X.columns):
        if X[c].nunique() < 2:
            X = X.drop(columns=c)
            dropped.append({"name": c, "reason": "zero_variance_after_filter"})
            if c in focal_cols:
                focal_cols.remove(c)
            if c in control_cols:
                control_cols.remove(c)

    # drop perfectly-collinear (aliased) columns so X'X is identified; keep controls/
    # position over focal features when forced to choose.
    X, aliased = _drop_collinear(X, protect=set(control_cols))
    for c in aliased:
        dropped.append({"name": c, "reason": "collinear"})
        if c in focal_cols:
            focal_cols.remove(c)
        if c in control_cols:
            control_cols.remove(c)
    if not focal_cols or X.empty:
        return {"error": "no usable focal features after filtering", "dropped": dropped}

    Xc = sm.add_constant(X.astype(float), has_constant="add")
    groups = None
    if spec["cluster_key"] and spec["cluster_key"] in base.columns:
        groups = base.loc[X.index, spec["cluster_key"]].astype("string").fillna("∅")

    try:
        diagnostics["condition_number"] = _f(np.linalg.cond(Xc.values))
    except np.linalg.LinAlgError:
        diagnostics["condition_number"] = None
    diagnostics["reference_levels"] = reference_levels

    return {"X": Xc, "y": y2, "focal_cols": focal_cols, "control_cols": control_cols,
            "groups": groups, "dropped": dropped, "diagnostics": diagnostics}


# --------------------------------------------------------------------------- #
# VIF
# --------------------------------------------------------------------------- #
def _drop_collinear(X: pd.DataFrame, protect: set[str]) -> tuple[pd.DataFrame, list[str]]:
    """Greedily keep columns that raise the matrix rank; drop the rest (aliased).
    Protected columns (controls/position) are considered first so a redundant *focal*
    column is dropped rather than a control."""
    cols = list(X.columns)
    ordered = [c for c in cols if c in protect] + [c for c in cols if c not in protect]
    kept: list[str] = []
    dropped: list[str] = []
    mat = np.empty((len(X), 0))
    rank = 0
    for c in ordered:
        trial = np.hstack([mat, X[[c]].values]) if mat.size else X[[c]].values
        r = int(np.linalg.matrix_rank(trial, tol=1e-10))
        if r > rank:
            mat, rank = trial, r
            kept.append(c)
        else:
            dropped.append(c)
    return X[[c for c in cols if c in kept]], dropped


def _vif_map(X: pd.DataFrame) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    cols = list(X.columns)
    arr = X.values
    with np.errstate(divide="ignore", invalid="ignore"):  # perfect collinearity -> inf -> None
        for i, name in enumerate(cols):
            if name == "const":
                continue
            try:
                out[name] = _f(variance_inflation_factor(arr, i))
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                out[name] = None
    return out


def _two_sided_p(z: float) -> float:
    """Two-sided normal-tail p-value from a z/t statistic (scipy-free, matches batch.py)."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _wild_cluster_bootstrap_se(Xv: np.ndarray, yv: np.ndarray, beta: np.ndarray,
                               gi: np.ndarray, n_groups: int, iters: int, seed: int) -> np.ndarray:
    """Unrestricted wild cluster bootstrap SEs for every coefficient. Fast: the OLS
    projection P = (X'X)⁻¹X' is precomputed once, then each iteration is one matrix–vector
    product. Rademacher (±1) signs are drawn **per cluster** so within-cluster correlation
    is preserved. Deterministic given `seed`."""
    yhat = Xv @ beta
    resid = yv - yhat
    rng = np.random.default_rng(seed)
    P = np.linalg.pinv(Xv.T @ Xv) @ Xv.T          # (k, n)
    boot = np.empty((iters, Xv.shape[1]))
    for b in range(iters):
        w = rng.choice((-1.0, 1.0), size=n_groups)[gi]
        boot[b] = P @ (yhat + w * resid)
    return boot.std(axis=0, ddof=1)


# --------------------------------------------------------------------------- #
# logit cross-check: Average Marginal Effects (probability points)
# --------------------------------------------------------------------------- #
def _logit_ame(X: pd.DataFrame, y: pd.Series, groups, focal_names: set, labels: dict):
    """Fit a logit and return its Average Marginal Effects in probability points (with
    SE/CI) for the focal features — the textbook cross-check that should land near the
    LPM coefficients. Returns (ame_rows, separation, warning). Never raises."""
    import warnings as _w

    try:
        from statsmodels.tools.sm_exceptions import PerfectSeparationError
    except Exception:  # noqa: BLE001
        PerfectSeparationError = Exception  # type: ignore[assignment]

    def _is_sep(msgs):
        return any(("separation" in m.lower()) or ("perfect" in m.lower()) for m in msgs)

    # Capture (don't print) warnings — statsmodels emits PerfectSeparationWarning + a
    # flood of overflow RuntimeWarnings on separation, then may raise LinAlgError.
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        try:
            logit = sm.Logit(y, X)
            kw = ({"cov_type": "cluster", "cov_kwds": {"groups": groups.values}}
                  if groups is not None else {"cov_type": "HC0"})
            lres = logit.fit(disp=0, maxiter=200, **kw)
        except PerfectSeparationError:
            return [], True, config.CAVEAT_SEPARATION
        except Exception as exc:  # noqa: BLE001
            if _is_sep([str(w.message) for w in caught]):
                return [], True, config.CAVEAT_SEPARATION
            return [], False, f"Logit cross-check skipped ({type(exc).__name__})."
        msgs = [str(w.message) for w in caught]

    if (_is_sep(msgs) or not bool(lres.mle_retvals.get("converged", True))
            or float(np.max(np.abs(np.asarray(lres.params)))) > 25):
        return [], True, config.CAVEAT_SEPARATION

    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            sf = lres.get_margeff(at="overall").summary_frame()
    except Exception as exc:  # noqa: BLE001
        return [], False, f"AME computation skipped ({type(exc).__name__})."

    out = []
    for name in sf.index:
        if name not in focal_names:
            continue
        row = sf.loc[name]
        base = name.split("=", 1)[0]
        lbl = labels.get(base, base.replace("_", " ")) + (f" = {name.split('=', 1)[1]}" if "=" in name else "")
        out.append({
            "name": name, "label": lbl,
            "ame": _f(row.iloc[0]), "se": _f(row.iloc[1]),
            "p": _f(row.iloc[3]) if len(row) > 3 else None,
            "ci_low": _f(row.iloc[4]) if len(row) > 4 else None,
            "ci_high": _f(row.iloc[5]) if len(row) > 5 else None,
            "method": "logit AME (overall)",
        })
    return out, False, ""


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def fit_citation_model(df: pd.DataFrame, spec: dict) -> dict:
    """Fit one position-adjusted LPM and return the result schema (see module docstring)."""
    context = spec.get("context", "")
    if not HAVE_STATSMODELS:
        return unavailable_result(context)

    dm = design_matrix(df, spec)
    if "error" in dm:
        return {"available": True, "context": context, "model": spec["model"], "fitted": False,
                "warnings": [f"Could not fit: {dm['error']}."],
                "coefficients": [], "diagnostics": {"dropped_columns": dm.get("dropped", [])},
                "assumptions": [], "ovb_caveat": "", "title": spec.get("title", "")}

    X, y = dm["X"], dm["y"]
    n, k = len(X), X.shape[1]
    warnings: list[str] = []
    if n < max(config.ECON_MIN_ROWS, k + 5):
        return {"available": True, "context": context, "model": spec["model"], "fitted": False,
                "n": n, "warnings": [f"Insufficient data: {n} usable rows for {k} parameters "
                                     "— regression skipped (single runs are exploratory at best)."],
                "coefficients": [], "diagnostics": dm["diagnostics"], "assumptions": [],
                "ovb_caveat": "", "title": spec.get("title", "")}

    # ---- standard-error type ----
    groups = dm["groups"]
    n_clusters = int(groups.nunique()) if groups is not None else None
    few_clusters = bool(n_clusters is not None and n_clusters < config.MIN_CLUSTERS)
    use_cluster = n_clusters is not None and n_clusters >= 2
    se_type = "cluster" if use_cluster else config.ECON_SE_DEFAULT

    try:
        model = sm.OLS(y.values, X.values, hasconst=True)
        if use_cluster:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": groups.values})
        else:
            res = model.fit(cov_type="HC3")
    except (np.linalg.LinAlgError, ValueError) as exc:
        return {"available": True, "context": context, "model": spec["model"], "fitted": False,
                "n": n, "warnings": [f"Fit failed: {type(exc).__name__}: {exc}"],
                "coefficients": [], "diagnostics": dm["diagnostics"], "assumptions": [],
                "ovb_caveat": "", "title": spec.get("title", "")}

    names = list(X.columns)
    params = np.asarray(res.params, dtype=float)
    bse = np.asarray(res.bse, dtype=float)
    pvals = np.asarray(res.pvalues, dtype=float)
    tvals = np.asarray(res.tvalues, dtype=float)
    ci = np.asarray(res.conf_int(), dtype=float)  # (k, 2)
    focal_set = set(dm["focal_cols"])

    # Few clusters → analytic cluster SE is anti-conservative. Replace the FOCAL
    # coefficients' SE / CI / p with a wild cluster bootstrap.
    wild_applied = False
    if few_clusters and use_cluster and spec.get("wild_bootstrap", True):
        gi = pd.factorize(groups.values)[0]
        b_se = _wild_cluster_bootstrap_se(X.values, y.values, params, gi, int(gi.max()) + 1,
                                          config.ECON_BOOTSTRAP_ITERS, config.ECON_RNG_SEED)
        for i, nm in enumerate(names):
            if nm in focal_set and np.isfinite(b_se[i]) and b_se[i] > 0:
                bse[i] = b_se[i]
                tvals[i] = params[i] / b_se[i]
                pvals[i] = _two_sided_p(tvals[i])
                ci[i, 0], ci[i, 1] = params[i] - 1.96 * b_se[i], params[i] + 1.96 * b_se[i]
        wild_applied = True
        se_type = "cluster + wild bootstrap (focal)"

    vif = _vif_map(X)

    # BH over the focal family only (on the final, possibly bootstrapped, p-values)
    focal_idx = [i for i, nm in enumerate(names) if nm in focal_set]
    p_adj_by_name: dict[str, float | None] = {}
    if focal_idx:
        try:
            _, q, _, _ = multipletests([pvals[i] for i in focal_idx], alpha=0.05, method="fdr_bh")
            for j, i in enumerate(focal_idx):
                p_adj_by_name[names[i]] = _f(q[j])
        except Exception:  # noqa: BLE001
            pass

    labels = spec.get("labels", {})
    phase_map = spec.get("phase_map", {})

    def _base_name(nm: str) -> str:  # strip one-hot suffix for label/phase lookup
        return nm.split("=", 1)[0]

    coefficients = []
    high_vif = []
    for i, nm in enumerate(names):
        if nm == "const":
            continue
        is_focal = nm in focal_set
        base = _base_name(nm)
        v = vif.get(nm)
        if v is not None and v >= config.VIF_WATCH:
            high_vif.append({"name": nm, "vif": v})
        support = None
        col = X[nm]
        if set(np.unique(col.values)).issubset({0.0, 1.0}):
            support = int(col.sum())
        coefficients.append({
            "name": nm,
            "label": labels.get(base, base.replace("_", " ")) + (f" = {nm.split('=', 1)[1]}" if "=" in nm else ""),
            "phase": phase_map.get(base, "pre_answer"),
            "estimate": _f(params[i]),
            "se": _f(bse[i]),
            "ci_low": _f(ci[i][0]),
            "ci_high": _f(ci[i][1]),
            "t": _f(tvals[i]),
            "p": _f(pvals[i]),
            "p_adj": p_adj_by_name.get(nm),
            "vif": v,
            "support": support,
            "is_focal": is_focal,
            "is_control": not is_focal,
            "is_intercept": False,
        })

    max_vif = max([c["vif"] for c in coefficients if c["vif"] is not None] or [None]) if coefficients else None
    if few_clusters and not wild_applied:
        warnings.append(config.CAVEAT_FEW_CLUSTERS)
    elif wild_applied:
        warnings.append("Few clusters (prompts): focal p-values & CIs use the **wild cluster bootstrap** "
                        "(more honest than the analytic cluster SE here).")
    if max_vif is not None and max_vif >= config.VIF_PROBLEM:
        warnings.append(f"High multicollinearity (max VIF={max_vif:.1f}); entangled features have "
                        "wide error bars — not bias. Consider reporting them jointly.")

    diagnostics = {
        **dm["diagnostics"],
        "max_vif": max_vif,
        "high_vif": high_vif,
        "dropped_columns": dm["dropped"],
        "separation": False,
        "few_clusters": few_clusters,
    }

    # logit cross-check: AME in probability points (should land near the LPM coefficients)
    ame_rows: list[dict] = []
    if spec.get("crosscheck_logit", True):
        ame_rows, separation, ame_warn = _logit_ame(
            X, y, groups if use_cluster else None, focal_set, labels)
        if separation:
            diagnostics["separation"] = True
            if config.CAVEAT_SEPARATION not in warnings:
                warnings.append(config.CAVEAT_SEPARATION)
        elif ame_warn:
            warnings.append(ame_warn)

    ovb = getattr(config, _OVB_BY_CONTEXT.get(context, ""), "")
    return {
        "available": True,
        "fitted": True,
        "model": "lpm",
        "context": context,
        "title": spec.get("title", ""),
        "outcome": spec["outcome"],
        "n": n,
        "n_clusters": n_clusters,
        "cluster_key": spec["cluster_key"] if use_cluster else None,
        "se_type": se_type,
        "r2": _f(res.rsquared),
        "adj_r2": _f(res.rsquared_adj),
        "position_spec": (f"{spec['position_spec']}({diagnostics.get('position_col')})"
                          if diagnostics.get("position_col") else None),
        "coefficients": coefficients,
        "ame": ame_rows,  # logit AME cross-check (probability points)
        "diagnostics": diagnostics,
        "warnings": warnings,
        "assumptions": [config.CAVEAT_ASSUMPTIONS],
        "ovb_caveat": ovb,
        "spec": {"focal": dm["focal_cols"], "controls": dm["control_cols"],
                 "phase_filter": spec.get("phase_filter")},
    }


# --------------------------------------------------------------------------- #
# Feature grouping (for grouped interpretation + anomaly checks + forest plots)
# --------------------------------------------------------------------------- #
FEATURE_GROUPS: dict[str, list[str]] = {
    "position": ["source_position", "observed_rank", "serp_rank"],
    "relevance": ["title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity",
                  "max_chunk_prompt_similarity", "best_chunk_prompt_similarity", "relevance_score",
                  "title_query_sim", "snippet_query_sim", "page_query_sim", "max_chunk_query_sim",
                  "heading_prompt_match"],
    "structure": ["has_faq", "has_table", "has_bullets", "heading_count", "has_many_headings", "has_step_by_step"],
    "commercial": ["has_price_or_package", "price_package_page", "product_page"],
    "access": ["has_contact_info", "has_location_info", "has_booking_or_appointment",
               "has_phone_number", "has_email", "has_opening_hours"],
    "authority": ["source_type", "institutional_official", "brand_official_candidate",
                  "has_author", "has_reviewer", "has_schema", "title_contains_intent_terms",
                  "answer_like_text_in_first_500_chars"],
    "freshness": ["age_days", "freshness_days", "has_updated_date", "has_published_date"],
    "page_type": ["page_type"],
    "intent": ["intent"],
}
_GROUP_OF = {feat: grp for grp, feats in FEATURE_GROUPS.items() for feat in feats}

_GROUP_INTERP = {
    "position": "Observable placement/ranking (not the AI's internal ranking); may be a mediator — compare models with and without it.",
    "relevance": "Prompt–text similarity proxies; highly overlapping — prefer one relevance feature or the combined score.",
    "structure": "Answer-ready structure (FAQ / tables / bullets / headings).",
    "commercial": "Commercial/transactional signals (price / package / product).",
    "access": "Contact / booking / location signals — thin access pages may be surfaced but not cited; read by page_type.",
    "authority": "Source-type / official / authorship signals (observational proxies for authority).",
    "freshness": "Recency/age signals — older may proxy authority or evergreen content, not age itself.",
    "page_type": "Heuristic page-type dummies; interpret relative to the omitted reference category.",
    "intent": "Prompt-intent dummies (from the manifest).",
    "other": "Uncategorized features.",
}

_ANOMALY_MESSAGES = {
    "position_dominates": ("Source position is a strong observable placement feature and may dominate content "
                           "features. It is observable placement/ranking — not the AI's internal ranking — and may be "
                           "a mediator / post-treatment variable. Interpret content effects both with and without source position."),
    "similarity_collinear": ("Similarity / relevance features overlap heavily (high VIF); do not interpret individual "
                             "similarity coefficients separately. Use one preferred relevance feature or the combined relevance score."),
    "access_negative": ("A negative coefficient may reflect that thin contact/location pages are surfaced but not cited "
                        "(more-only), not that contact information is bad. Analyze by page_type."),
    "authorship_negative": ("May be confounded with page type, article format, source position, or scraped-template "
                            "artifacts rather than authorship itself."),
    "age_positive": ("Older pages may proxy authority, index history, or stable evergreen content; this is not a "
                     "recommendation to make pages old."),
    "pagetype_large": "Large page-type coefficient — check the omitted/reference page_type category before interpreting.",
}

_DEFAULT_LABELS = {
    "has_faq": "Has FAQ", "has_step_by_step": "Has steps", "has_contact_info": "Has contact info",
    "has_location_info": "Has location", "has_price_or_package": "Has price/package", "has_opening_hours": "Has hours",
    "has_booking_or_appointment": "Has booking", "has_phone_number": "Has phone", "has_email": "Has email",
    "has_author": "Has author", "has_reviewer": "Has reviewer", "has_published_date": "Has published date",
    "has_updated_date": "Has updated date", "has_schema": "Has schema.org", "has_table": "Has table",
    "has_bullets": "Has bullets", "has_many_headings": "Has many headings", "heading_prompt_match": "Heading–prompt match",
    "title_contains_intent_terms": "Title has intent terms", "answer_like_text_in_first_500_chars": "Answer-like intro",
    "relevance_score": "Relevance score (combined)", "word_count": "Word count", "char_count": "Char count",
    "heading_count": "Heading count", "freshness_days": "Age (days)", "institutional_official": "Institutional/official",
    "brand_official_candidate": "Brand-official (heuristic)", "source_position": "Source position",
    "observed_rank": "Observed rank", "serp_rank": "SERP rank", "page_type": "Page type", "source_type": "Source type",
    "intent": "Intent", "title_prompt_similarity": "Title–prompt sim", "description_prompt_similarity": "Desc–prompt sim",
    "page_prompt_similarity": "Page–prompt sim", "max_chunk_prompt_similarity": "Best chunk–prompt sim",
}


def _base_feature(name: str) -> str:
    base = name[6:] if name.startswith("log1p_") else name
    return base.split("=", 1)[0].split("_band", 1)[0]


def _feature_group(name: str) -> str:
    return _GROUP_OF.get(_base_feature(name), "other")


def vif_level(v) -> tuple[str, str]:
    if v is None:
        return ("unknown", "not estimable")
    if v < 2:
        return ("low", "low overlap")
    if v < 5:
        return ("moderate", "moderate overlap")
    if v < config.VIF_PROBLEM:
        return ("high", "high overlap — interpret jointly")
    return ("severe", "severe overlap — do not interpret separately")


# Feature pools for the A/B/C/D model specifications.
_CONTENT_FEATURES = [
    "has_faq", "has_step_by_step", "has_contact_info", "has_location_info", "has_price_or_package",
    "has_opening_hours", "has_booking_or_appointment", "has_phone_number", "has_email", "has_author",
    "has_reviewer", "has_published_date", "has_updated_date", "has_schema", "has_table", "has_bullets",
    "has_many_headings", "heading_prompt_match", "title_contains_intent_terms",
    "answer_like_text_in_first_500_chars", "word_count", "heading_count", "freshness_days",
]
_SOURCE_BOOL = ["institutional_official", "brand_official_candidate"]
_SOURCE_CATS = ["source_type", "page_type", "intent"]
_SIM_CONTINUOUS = [
    "title_prompt_similarity", "description_prompt_similarity", "page_prompt_similarity",
    "max_chunk_prompt_similarity", "best_chunk_prompt_similarity",
    "title_query_sim", "snippet_query_sim", "page_query_sim", "max_chunk_query_sim",
]


def _coef_by_base(coefs, *bases):
    return [c for c in coefs if _base_feature(c["name"]) in bases]


def _vif_rows(fit: dict) -> list[dict]:
    rows = []
    for c in fit.get("coefficients", []):
        lvl, interp = vif_level(c.get("vif"))
        rows.append({"feature": c["label"], "vif": c.get("vif"), "vif_level": lvl,
                     "interpretation": interp, "feature_group": _feature_group(c["name"])})
    return rows


def _anomaly_rows(fit: dict) -> list[dict]:
    coefs = [c for c in fit.get("coefficients", []) if c.get("estimate") is not None]
    rows: list[dict] = []

    def add(check, feats, est, p, vif, severity, msg):
        rows.append({"check": check, "feature": feats, "estimate": est, "p": p,
                     "vif": vif, "severity": severity, "message": msg})

    # 1) position dominates
    pos = [c for c in coefs if _feature_group(c["name"]) == "position"]
    if pos and coefs:
        pc = max(pos, key=lambda c: abs(c["estimate"]))
        max_abs = max(abs(c["estimate"]) for c in coefs)
        if pc.get("p") is not None and pc["p"] < 0.01 and abs(pc["estimate"]) >= 0.999 * max_abs:
            add("position_dominates", pc["label"], pc["estimate"], pc["p"], pc.get("vif"), "high",
                _ANOMALY_MESSAGES["position_dominates"])
    # 2) similarity severe VIF
    sim_sev = [c for c in coefs if _feature_group(c["name"]) == "relevance"
               and c.get("vif") is not None and c["vif"] > config.VIF_PROBLEM]
    if len(sim_sev) >= 2:
        add("similarity_collinear", "; ".join(c["label"] for c in sim_sev), None, None,
            max(c["vif"] for c in sim_sev), "high", _ANOMALY_MESSAGES["similarity_collinear"])
    # 3) contact / location / phone negative
    for c in _coef_by_base(coefs, "has_contact_info", "has_location_info", "has_phone_number"):
        if c["estimate"] < 0:
            add("access_negative", c["label"], c["estimate"], c.get("p"), c.get("vif"), "medium",
                _ANOMALY_MESSAGES["access_negative"])
    # 4) reviewer / author negative & suggestive
    for c in _coef_by_base(coefs, "has_reviewer", "has_author"):
        if c["estimate"] < 0 and c.get("p") is not None and c["p"] < 0.1:
            add("authorship_negative", c["label"], c["estimate"], c["p"], c.get("vif"), "medium",
                _ANOMALY_MESSAGES["authorship_negative"])
    # 5) age / freshness positive & significant (freshness_days = age in days)
    for c in _coef_by_base(coefs, "freshness_days", "age_days"):
        if c["estimate"] > 0 and c.get("p") is not None and c["p"] < 0.05:
            add("age_positive", c["label"], c["estimate"], c["p"], c.get("vif"), "medium",
                _ANOMALY_MESSAGES["age_positive"])
    # 6) page-type coefficients large
    ref = (fit.get("diagnostics", {}).get("reference_levels", {}) or {}).get("page_type")
    for c in coefs:
        if _feature_group(c["name"]) == "page_type" and abs(c["estimate"]) > 0.20:
            msg = _ANOMALY_MESSAGES["pagetype_large"] + (f" Reference category = '{ref}'." if ref else "")
            add("pagetype_large", c["label"], c["estimate"], c.get("p"), c.get("vif"), "low", msg)
    return rows


def _group_rows(fit: dict) -> list[dict]:
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for c in fit.get("coefficients", []):
        groups[_feature_group(c["name"])].append(c)
    rows = []
    for grp in sorted(groups):
        cs = groups[grp]
        est = [c for c in cs if c.get("estimate") is not None]
        pos = sorted((c for c in est if c["estimate"] > 0), key=lambda c: -c["estimate"])[:3]
        neg = sorted((c for c in est if c["estimate"] < 0), key=lambda c: c["estimate"])[:3]
        rows.append({
            "feature_group": grp, "num_features": len(cs),
            "top_positive_features": ", ".join(c["label"] for c in pos),
            "top_negative_features": ", ".join(c["label"] for c in neg),
            "num_significant_p_05": sum(1 for c in cs if c.get("p") is not None and c["p"] < 0.05),
            "num_significant_q_10": sum(1 for c in cs if c.get("p_adj") is not None and c["p_adj"] < 0.10),
            "interpretation": _GROUP_INTERP.get(grp, ""),
        })
    return rows


def _exec_summary(diag: dict, models: list[dict], cluster_var, cluster_count) -> list[str]:
    out: list[str] = []
    coefs = [c for c in diag.get("coefficients", []) if c.get("estimate") is not None]
    # exclude severe-collinearity coefs from the "strongest" picks — their magnitude is unreliable
    reliable = [c for c in coefs if not (c.get("vif") is not None and c["vif"] >= config.VIF_PROBLEM)]
    sig = [c for c in reliable if c.get("p") is not None and c["p"] < 0.05]
    if sig:
        top = max(sig, key=lambda c: abs(c["estimate"]))
        grp = _feature_group(top["name"])
        out.append(f"Strongest observable predictor: **{top['label']}** ({top['estimate']:+.3f} prob.) "
                   f"— group *{grp}*" + (" (position is observable placement, not internal AI ranking)."
                                         if grp == "position" else "."))
    content = [c for c in sig if _feature_group(c["name"]) in ("structure", "commercial", "access", "page_type")]
    if content:
        names = ", ".join(f"{c['label']} ({c['estimate']:+.3f})"
                          for c in sorted(content, key=lambda c: -abs(c["estimate"]))[:4])
        out.append(f"Strongest content / page-type signals: {names}.")
    uncertain = [c for c in coefs if c.get("p") is not None and c["p"] >= 0.05
                 and c.get("ci_low") is not None and (c["ci_high"] - c["ci_low"]) > 0.10]
    if uncertain:
        out.append("Uncertain estimates (wide CI, not distinguishable from zero): "
                   + ", ".join(c["label"] for c in uncertain[:6]) + ".")
    high_vif = [c["label"] for c in coefs if c.get("vif") is not None and c["vif"] >= config.VIF_PROBLEM]
    if high_vif:
        out.append("Caution — severe multicollinearity (VIF ≥ 10), do not read individually: "
                   + ", ".join(high_vif[:6]) + ".")
    if cluster_count is not None:
        note = f"Standard errors clustered by **{cluster_var}** ({cluster_count} clusters)."
        if cluster_count < config.MIN_CLUSTERS:
            note += " Few clusters — focal CIs use the wild cluster bootstrap; treat significance cautiously."
        out.append(note)
    return out


# --------------------------------------------------------------------------- #
# model comparison / sensitivity analysis (A / B / C / D + a FULL diagnostic fit)
# --------------------------------------------------------------------------- #
def model_comparison(df: pd.DataFrame, *, context: str = "", position_col: str = "source_position",
                     position_fallbacks=("observed_rank",), cluster_candidates=("domain", "record_id", "run_id"),
                     labels: dict | None = None, phase_map: dict | None = None) -> dict:
    """Fit content-only (A), +source/authority (B), +position (C), and reduced-similarity (D)
    specifications, plus a FULL diagnostic fit, and return the comparison table + VIF /
    anomaly / grouped-feature diagnostics + an executive summary. Sensitivity = whether each
    feature's coefficient is stable across specifications."""
    empty = {"available": HAVE_STATSMODELS, "fitted": False, "context": context, "models": [],
             "comparison_rows": [], "vif_rows": [], "anomaly_rows": [], "group_rows": [],
             "executive_summary": [], "cluster_variable": None, "cluster_count": None, "warnings": []}
    if not HAVE_STATSMODELS:
        empty["warnings"] = [f"statsmodels not installed ({_IMPORT_ERROR})."]
        return empty
    if df is None or df.empty or "cited" not in df.columns:
        return empty

    labels = {**_DEFAULT_LABELS, **(labels or {})}
    work = df.copy()

    # combined relevance score (standardized mean of available similarity features) for Model D
    sims = [c for c in _SIM_CONTINUOUS if c in work.columns
            and pd.to_numeric(work[c], errors="coerce").notna().sum() > 3]
    if sims:
        Z = work[sims].apply(pd.to_numeric, errors="coerce")
        work["relevance_score"] = ((Z - Z.mean()) / Z.std(ddof=0).replace(0, 1)).mean(axis=1)

    # cluster: prefer domain, then record_id / run_id
    cluster_key, cluster_count = None, None
    for cand in cluster_candidates:
        if cand in work.columns:
            nun = int(work[cand].astype("string").nunique(dropna=True))
            if nun >= 2:
                cluster_key, cluster_count = cand, nun
                break

    content = [c for c in _CONTENT_FEATURES if c in work.columns]
    src_bool = [c for c in _SOURCE_BOOL if c in work.columns]
    cats = [c for c in _SOURCE_CATS if c in work.columns]
    has_pos = position_col in work.columns or any(f in work.columns for f in position_fallbacks)

    def _spec(title, focal, with_cats, with_pos, notes):
        return (title, notes, build_spec(
            focal=focal, position_col=(position_col if with_pos else None),
            position_fallbacks=(list(position_fallbacks) if with_pos else []),
            categoricals=(cats if with_cats else []), cluster_key=cluster_key, phase_map=phase_map or {},
            labels=labels, context=context, title=title, crosscheck_logit=False, wild_bootstrap=True))

    rel = ["relevance_score"] if "relevance_score" in work.columns else []
    specs = [
        _spec("A · content only", content, False, False, "content/page features only"),
        _spec("B · + source/authority", content + src_bool, True, False,
              "A + source_type / official / brand / page_type / intent"),
        _spec("C · + source position", content + src_bool, True, True, "B + log1p(source_position)"),
        _spec("D · reduced similarity", content + src_bool + rel, True, True,
              "C + a single combined relevance_score (not all similarity features)"),
    ]
    models = []
    for title, notes, spec in specs:
        models.append({"model_name": title, "spec_notes": notes, "fit": fit_citation_model(work, spec)})

    # FULL diagnostic fit: everything incl. all (collinear) similarity features → surfaces VIF
    full_focal = content + src_bool + [c for c in _SIM_CONTINUOUS if c in work.columns]
    _, _, full_spec = _spec("FULL · all features (diagnostic)", full_focal, True, has_pos, "all features incl. raw similarities")
    full = fit_citation_model(work, full_spec)
    diag = full if full.get("fitted") else next((m["fit"] for m in reversed(models) if m["fit"].get("fitted")), full)

    comparison_rows = []
    for m in models:
        f = m["fit"]
        if not f.get("fitted"):
            continue
        for c in f["coefficients"]:
            if not c.get("is_focal"):
                continue
            comparison_rows.append({
                "feature": c["label"], "model_name": m["model_name"], "delta_prob": c["estimate"],
                "se": c["se"], "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p": c["p"],
                "q_bh": c.get("p_adj"), "vif": c.get("vif"), "n": f["n"],
                "cluster_variable": f.get("cluster_key"), "cluster_count": f.get("n_clusters"),
                "spec_notes": m["spec_notes"],
            })

    warnings = list(diag.get("warnings", []))
    sim_sev = [c for c in diag.get("coefficients", []) if _feature_group(c["name"]) == "relevance"
               and c.get("vif") is not None and c["vif"] > config.VIF_PROBLEM]
    if len(sim_sev) >= 2:
        warnings.append("Similarity/relevance features have severe VIF (>10) as a group — reduce them to a "
                        "single relevance score (Model D) rather than interpreting them separately.")

    return {
        "available": True, "fitted": bool(diag.get("fitted")), "context": context,
        "cluster_variable": cluster_key, "cluster_count": cluster_count,
        "models": models, "full_model": full, "diagnostic_model": diag,
        "comparison_rows": comparison_rows, "vif_rows": _vif_rows(diag),
        "anomaly_rows": _anomaly_rows(diag), "group_rows": _group_rows(diag),
        "executive_summary": _exec_summary(diag, models, cluster_key, cluster_count),
        "warnings": warnings,
    }
