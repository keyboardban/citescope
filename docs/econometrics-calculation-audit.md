# CiteScope Econometrics — Calculation Audit Reference

> A **white-box walkthrough** of every feature that is computed and every econometric
> calculation, with the exact formula and the code location (`file:line`) for each — so you
> can check the numbers yourself and locate anything that looks wrong.
>
> Companion to `src/econometrics.py`, `src/confounders.py`, `src/chatgpt_pipeline.py`.
> _Written 2026-06-30 for a self-audit of a "suspicious" outcome._

---

## 0. First, orient yourself (30 seconds)

Before suspecting a bug, confirm **what you're looking at**:

| Question | Why it matters |
|---|---|
| **Demo or real data?** | The offline demo is a *synthetic* data-generating process (fabricated coefficients). Its numbers are not a real audit — a "weird" demo coefficient is expected. Load a real `data/chatgpt/*.json` audit to judge real results. |
| **Which model?** | The headline is **Model C** (content + source/authority + position). Models A/B/D/E–H are *sensitivity* fits — their coefficients are *supposed* to differ from C. |
| **Which number & tab?** | A coefficient, a p-value, a similarity score, and a proxy count are computed completely differently. Pin the exact cell. |
| **Scale?** | LPM coefficients are in **probability points**: `0.12` = **+12 percentage points**, *not* 12%. This is the #1 misread. |

**Unit of analysis:** one row = **one surfaced source**. **Outcome:** `cited = 1` (explicitly cited)
vs `cited = 0` (surfaced / more-only, not cited) — `src/chatgpt_pipeline.py:116`. The model estimates
citation **within surfaced sources**, not "retrieved from the whole web."

---

## 1. Input features — what each is and how it's computed

### 1a. Similarity features (⚠️ a proxy, not ground truth)

All `*_similarity` values come from `SimilarityEngine.score(a, b)`:

- **Default = `lexical`**: offline **bag-of-words cosine with sublinear TF weighting**
  (`src/similarity.py:4`, `_cosine_sparse` at `:44`). It is a **semantic-overlap proxy**, not a
  meaning model — two texts about the same topic with different words score *low*. If similarity
  looks "too low / doesn't separate cited vs more-only," that is a **known, expected** property of the
  lexical proxy (and a documented finding), not necessarily a bug.
- **`embedding`**: cosine over Gemini embeddings (`:56`), only if an `embed_fn` is injected; otherwise
  it **falls back to lexical** (`:73`).

| Feature | Computed as | Code | Admissible in main model? |
|---|---|---|---|
| `title_prompt_similarity` | `score(title, prompt)` | `chatgpt_pipeline.py:121` | ✅ prompt-based |
| `description_prompt_similarity` | `score(description, prompt)` | `:122` | ✅ prompt-based |
| `page_prompt_similarity` | `score(page_text[:MAX], prompt)` | `:143` | ✅ prompt-based |
| `max_chunk_prompt_similarity` | `max over chunks of score(chunk, prompt)` | `:156` | ✅ prompt-based |
| `page_answer_similarity` | `score(page_text, answer)` | `:144` | ❌ **answer-derived / circular** |
| `max_chunk_answer_similarity` | `max score(chunk, answer)` | `:157` | ❌ **answer-derived / circular** |

**Audit hook:** answer-derived similarity is **excluded** from the effect model (it's downstream of the
citation). If you see `page_answer_similarity` inside a fitted model's coefficients, *that* is a bug.

### 1b. Content / page features (heuristic booleans, need a scraped page)

`has_faq`, `has_table`, `has_bullets`, `has_contact_info`, `has_location_info`,
`has_price_or_package`, `has_phone_number`, `has_author`, `has_reviewer`, `has_schema`,
`has_many_headings`, `heading_prompt_match`, `title_contains_intent_terms`, `page_type`, … are
regex/keyword **heuristics** over the scraped text (`brand_visibility.extract_content_features`).
They are `None` when the page was **not scraped** → those rows drop out via the coverage filter (below).

⚠️ **`freshness_days` is AGE in days** (bigger = older), despite the "freshness" name (label = "Age
(days)"). A *positive* coefficient means *older* pages are associated with more citation — surprising
but a known pattern (authority / evergreen), **not** a sign bug.

### 1c. Source / position / metadata

`source_position` (observable panel position, 1 = top), `observed_rank`, `source_type` (forum/news/…),
`institutional_official`, `brand_official_candidate`, `word_count`, `char_count`, `heading_count`.

### 1d. Derived confounder proxies (`src/confounders.py:derive_proxy_features`)

~40 no-scrape proxies, each a **labelled proxy** (never the true construct):

- **Visibility counts** (from `domain`/`url`): `domain_seen_count` = rows per domain,
  `domain_cited_count` = cited rows per domain, `domain_citation_rate` = mean `cited` per domain,
  `citescope_visibility_history_score` = `0.5·norm(log1p(domain_seen)) + 0.5·norm(log1p(url_seen))`.
- **URL semantics** (`url_path_depth`, `has_tracking_params`, `url_is_numeric_id`, …),
  **prompt-wording flags** (`prompt_has_price_terms`, …), **language/local** (`thai_content_ratio`,
  `is_thai_domain`, …), **grouped scores** (`content_completeness_score`, `answer_ready_score`,
  `trust_signal_score`).

⚠️ `domain_citation_rate` is a **leaky-looking** aggregate: it's the mean of `cited` **within the same
domain including the current row**. It is used only as a *confounder-proxy control* in Models G/H (never
focal) — but if a G/H coefficient looks too strong, this partial self-prediction is the reason.

### 1e. Combined relevance score (Model D)

```
relevance_score = mean over available prompt-sims of  (x - mean(x)) / std_pop(x)      # z-score, then average
```
`src/econometrics.py:1168`. Built from **prompt-based sims only** (`_PROMPT_SIM`); std uses `ddof=0`
(population); a zero-variance column is replaced by 1 to avoid divide-by-zero.

---

## 2. The econometric pipeline — step by step

Everything below is `fit_citation_model(df, spec)` (`src/econometrics.py:422`) and
`model_comparison` (`:1143`).

### Step 1 — Outcome & row selection
`y = to_numeric(df["cited"])`; rows with a non-null `y` are kept (`design_matrix`, `:149-153`).

### Step 2 — Build the design matrix `X` (`design_matrix`, `:140`)
In order:
1. **Coerce** each candidate to numeric; `bool → float` (`:118`).
2. **Filters** (`_consider`, `:159`): drop if coverage `< 0.5` (`_MIN_COVERAGE`), if `< 2` distinct
   values (zero variance), or (for 0/1 columns) if either class has `< 3` rows (`_MIN_SUPPORT`).
3. **Missing data**: median-fill each numeric feature and add a `{feat}_missing` 0/1 indicator when
   `≥ 3` values are missing (`:174-180`). → changes which rows/columns enter; a shifted `n` is usually this.
4. **Position transform** (`:193-199`): `log1p_source_position = log(1 + position)` (default), or bins
   `1-3 / 4-6 / 7-10 / 11+` (reference = `1-3`), or raw linear. Plus a `source_position_missing` flag.
5. **One-hot categoricals** (`:224-238`): rare levels (`< 3`) → `"other"`; **most-common level dropped
   as the reference**; only `k−1` dummies enter → **no dummy-variable trap**.
6. **Collinearity drop** (`_drop_collinear`, `:298`): greedily keep columns that raise
   `matrix_rank`; drop exact aliases (logged `reason: collinear`).
7. **Add intercept** `sm.add_constant` (`:280`); align **cluster groups**; record **condition number**
   `cond(X)` (`:281`).

### Step 3 — Fit the Linear Probability Model
```
β = (XᵀX)⁻¹ Xᵀ y            # OLS on the 0/1 outcome
```
`sm.OLS(y, X).fit(...)` (`:453`). Coefficients are **Δ probability of citation per unit of the
feature**, holding the others fixed.

### Step 4 — Standard errors (this is where "suspicious" p-values usually live)
- **Default = HC3** heteroskedasticity-robust (`cov_type="HC3"`, `:457`). Correct choice because a 0/1
  outcome is inherently heteroskedastic.
- **Cluster-robust** when a cluster key with `≥ 2` groups exists (`cov_type="cluster"`, `:455`).
- **Few clusters** (`< MIN_CLUSTERS = 40`): the **focal** coefficients' SE/CI/p are replaced by a
  **wild cluster bootstrap** (`:474-486`).

  ⚠️ **Known soft spot (audit this first if p-values look too small / CIs too narrow).** The wild
  bootstrap here is the *unrestricted* variant; it mathematically reproduces the **CR0** cluster
  variance (no finite-sample correction), so it comes out **~10% smaller** than the analytic cluster
  SE it replaces, and it builds the CI as `β ± 1.96·se` (`:484`) using the **normal** 1.96 rather than
  `t(G−1)`. Net effect with few clusters (e.g. clustering on `record_id` with ~36 prompts): focal
  significance is **mildly overstated**. This is documented and is the single most likely reason a
  small-upload result looks "too significant." (Mechanically correct; methodologically anti-conservative.)

### Step 5 — t, p, CI
```
t = β / se
p = 2·(1 − Φ(|t|))          # two-sided NORMAL tail, _two_sided_p, :334
CI = β ± 1.96·se            # normal critical value (see the few-cluster caveat above)
```
Non-bootstrapped rows use statsmodels' own `res.pvalues` / `res.conf_int()` (`:467-469`).

### Step 6 — VIF & condition number
`VIF_j = 1 / (1 − R²_j)` where `R²_j` regresses column `j` on all other columns **including the
constant** (`variance_inflation_factor`, `_vif_map` `:318`). Reported two ways: **focal** (content
features only) and **full** design matrix. High VIF = wide error bars, **not** bias.

### Step 7 — Multiple testing (Benjamini–Hochberg)
BH is applied **within each model × feature family** (`:497-510`), not across all features. A **family
with one test gets `q = p`** (no correction). So if you expected heavy correction and see `q ≈ p`, it's
because that feature was alone in its family — expected, not a bug.

### Step 8 — Logit + AME cross-check
A logistic regression is fit and converted to **Average Marginal Effects** (`get_margeff(at="overall")`)
in probability points (`_logit_ame`, `:358`). These should land **near** the LPM coefficients. If a
feature **perfectly predicts** `cited`, separation is detected (`:384`) → AME suppressed, `logit_status
= failed_perfect_separation`, and the **LPM is kept** as the headline. Big LPM-vs-AME gaps are a
legitimate red flag worth reporting.

### Step 9 — Fit statistics
`r2 = res.rsquared`, `adj_r2 = res.rsquared_adj` (`:593-594`). LPM R² is typically **low** (0/1 outcome)
— a low R² is normal, not a bug.

### Step 10 — The model ladder (`model_comparison`, :1143)
Fits **A** (content) → **B** (+source/authority) → **C** (+position, headline) → **D** (+relevance_score),
plus **FULL** (all raw sims → surfaces VIF) and **E–H** (Model-D focal + confounder-proxy *controls*).
`comparison_rows` pivots focal Δprob across models. Coefficients **shifting across models is the
intended output** (sensitivity), not an error.

### Step 11 — Confounder audit (`confounder_audit`, src/confounders.py)
- **balance_by_cited**: for each proxy, `mean(cited=1) − mean(cited=0)`, missing rate, imbalance flag.
- **correlation_matrix**: Pearson corr among numeric proxies.
- **confounder_vif**: VIF over the proxy controls.
None of these feed the headline; they're diagnostics.

---

## 3. Spot-check recipes (verify a number by hand)

Paste into `python` (venv active). Replace the run path / feature as needed.

**Recompute one coefficient independently (the definitive check):**
```python
import json, glob, os, numpy as np, pandas as pd, statsmodels.api as sm
from src import chatgpt_pipeline as cgp, econometrics as E
from src.similarity import SimilarityEngine
run = json.load(open(sorted(glob.glob("data/chatgpt/*.json"), key=os.path.getsize)[-1]))
feats = cgp.build_features(run, {}, SimilarityEngine("lexical"))["features"]
df = pd.DataFrame(feats)
spec = E.build_spec(focal=["has_faq"], position_col="source_position", context="chatgpt",
                    categoricals=["source_type"], cluster_key="domain", crosscheck_logit=False)
dm = E.design_matrix(df, spec)
manual = sm.OLS(dm["y"].values, dm["X"].values).fit(cov_type="HC3")
j = list(dm["X"].columns).index("has_faq")
print("manual  β=%.5f  se=%.5f" % (manual.params[j], manual.bse[j]))
fit = E.fit_citation_model(df, spec)
c = next(c for c in fit["coefficients"] if c["name"] == "has_faq")
print("engine  β=%.5f  se=%.5f" % (c["estimate"], c["se"]))   # must match to 6 dp
```

**Recompute a similarity score:**
```python
se = SimilarityEngine("lexical")
print(se.score("your title here", "your prompt here"))   # compare to the feature table cell
```

**Recompute a proportion / count:**
```python
print("cited-rate when has_faq=1:", df.loc[df.has_faq==1, "cited"].mean())
print("domain_seen_count for d:", (df.domain=="example.com").sum())
```

---

## 4. "Suspicious, or expected?" — reading the outcome

| What you see | Verdict | Why |
|---|---|---|
| Coefficient `0.12` interpreted as 12% | **misread** | It's **12 percentage points** (probability). |
| A content effect **shrinks/flips** when position is added (B→C) | **expected** | Position is a **mediator**; that's the whole point of A/B vs C. |
| A `page_type` dummy is huge (e.g. `+0.30`) | **expected** | It's **relative to the omitted reference** page type; check `reference_categories`. |
| `log1p_source_position` coefficient looks small | **expected** | It's per **unit of log(1+position)**, not per rank. |
| Content effects ≈ 0 with **wide CIs** | **expected** | Underpowered / collinear on a single small run — not "no effect proven." |
| Similarity barely separates cited vs more-only | **expected** | Documented finding; the lexical proxy + source-type dominance. |
| **p-values tiny / CIs narrow on a small upload clustered on `record_id`** | ⚠️ **prime suspect** | The few-cluster wild bootstrap is anti-conservative (Step 4). |
| Predicted probability outside [0,1] | **expected (LPM limitation)** | Use the logit/AME cross-check to sanity-check. |
| Older pages (`freshness_days↑`) associated with **more** citation | **expected** | `freshness_days` is **age**; authority/evergreen proxy. |
| `n` changed vs the raw source count | **expected** | Coverage filter + median-fill + listwise drop of remaining NaN. |
| A **G/H** confounder model coefficient is very strong | **check** | `domain_citation_rate` partially self-predicts `cited` (§1d). |
| An **answer-derived** feature appears in a fitted model | **BUG** | Report it — it must be excluded. |
| Engine β ≠ manual OLS β (§3 recipe) by more than 1e-5 | **BUG** | Report it with the run + feature. |

---

## 5. What I already verified (so you can focus)

In a prior independent pass I recomputed the core math from first principles and got **exact** matches
(≤ 1e-9): OLS β vs normal equations, **HC3** vs the manual sandwich, **cluster SE** vs the manual CR
sandwich, **VIF** vs `1/(1−R²)`, **Benjamini–Hochberg** vs the textbook step-up, the two-sided p vs
`scipy`, and the `log1p`/median-fill/one-hot construction. **The arithmetic is correct.**

The **one** substantive concern is the **few-cluster wild bootstrap** (Step 4): it's *computed*
correctly but is *methodologically* anti-conservative, so it makes few-cluster p-values/CIs look a bit
too strong. If your "suspicious" result is **significant content effects on a single small upload**,
that path is where I'd look first.

---

## 6. Fastest way to resolve this

Tell me the **exact number and where you saw it** — e.g. *"Model C, `has_contact_info` = −0.28, p = 0.003,
on the Real-Estate audit"* — and I'll (a) recompute it independently with the §3 recipe, (b) check it
against the logit/AME cross-check, and (c) tell you whether it's a genuine bug or one of the expected
behaviours in §4. If you point me at the run file, I can run the check in one step.
