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
