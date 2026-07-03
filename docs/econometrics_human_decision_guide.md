# Econometrics — Human Decision Guide

> A guide for deciding **which choices need a human econometric judgment** versus **which
> implementation tasks can be safely vibe-coded** once the specification is fixed.
> **This document does not change code.** It is a decision aid for future development of the
> CiteScope citation model.
>
> Scope: CiteScope analyzes AI citation behavior. Unit of analysis = **one surfaced source
> appearance** (one source/page observed under one AI prompt). Outcome: `cited = 1` if the
> source was **explicitly cited**, `cited = 0` if it was **surfaced / more-only** but not cited.
> The headline model is a **Linear Probability Model (LPM)**; coefficients are read as
> **probability-point associations**, not causal effects.

---

## 1. Purpose

A regression pipeline is easy to build and easy to point in the wrong direction. Code can fit a model,
compute robust standard errors, correct for multiple testing, and draw a forest plot — all correctly —
and still answer the **wrong question**: wrong unit of analysis, a leaked outcome, a mediator treated as
a control, or a proxy read as a true measurement.

> **Vibe coding can build the econometrics pipeline, but econometric knowledge is needed to decide
> whether the pipeline is asking the right question.**

This guide draws the line. Sections 2–4 are **human decisions** (fix these first, on purpose). Sections 5
are the **vibe-codeable** mechanics that follow once the spec is set. Section 6 lists changes that must
never be made without review. Sections 7–8 keep the reporting language honest and give acceptance checks.

Everything here stays **observational**: we describe features **associated with** citation **among
surfaced sources**. We do not claim to know the AI's internal retrieval set or why any page was or was
not cited.

---

## 2. Human Decisions Required Before Coding

Fix each of these **before** writing or changing pipeline code. The "Recommended Default" is a sensible
starting point, not a substitute for judgment on a specific audit.

| # | Decision Area | Question Humans Must Answer | Why It Matters | Recommended Default | Risk If Wrong |
|---|---|---|---|---|---|
| 1 | **Unit of analysis** | Is one row a surfaced source appearance, a unique page, a domain, or a prompt? | Changes what "an observation" is, the standard-error structure, and what a coefficient means. | **One surfaced source appearance** = one page/source under one prompt. | Duplicated pages inflate n; domain-level questions answered with page-level rows (or vice versa); wrong inference. |
| 2 | **Outcome definition** | Is `cited = 1` only explicit citations? Are more-only / surfaced-but-not-cited rows `cited = 0`? | Defines the whole estimand. | `cited = 1` explicit citation; `cited = 0` surfaced/more-only. This is **citation within surfaced sources**, not retrieval from the whole web. | Treating more-only as "rejected" or as "not retrieved"; conflating surfacing with citation. |
| 3 | **Feature timing** | Did the feature exist **before** the AI answer, or is it **derived from** the answer? | Post-answer features leak the outcome. | Pre-output features may enter models; post-output/answer-derived features may **not**. | Circular models that "predict" citation from the answer itself. |
| 4 | **Leakage prevention** | Which features are circular or downstream of citation? | A leaked feature manufactures significance. | Post-output diagnostics only for: `page_answer_similarity`, `max_chunk_answer_similarity`, answer overlap, `brand_appeared_in_answer`. | Inflated, meaningless "effects"; false confidence. |
| 5 | **Focal vs controls vs diagnostics** | Which features are the business/econometric variables of interest, which are controls, which are diagnostics only? | Determines what gets the headline, the BH family, and the interpretation. | Content/page features = focal; source_type/page_type/position = controls; scrape/selection features = diagnostics. | Interpreting a nuisance control as a finding; correcting the wrong family. |
| 6 | **Mediators / bad controls** | Is `source_position` a confounder, a mediator, or a post-treatment variable? | Controlling for a mediator changes the estimand by blocking part of the pathway from content features to citation. | Show models **with and without** `source_position`; interpret content effects both ways. | "Controlling away" the effect; or over-crediting position. |
| 7 | **Proxy labeling** | Which features are proxies rather than true measurements? | A proxy read literally overclaims. | Label proxies explicitly. `domain_seen_count` = CiteScope-**observed visibility** proxy, not true domain authority; `page_age_days`/visibility-history = proxy, **not** true index history. | Claiming "domain authority" / "index history" the pipeline cannot measure. |
| 8 | **Cluster choice** | Cluster SEs by domain, prompt_id, canonical_url, or two-way domain × prompt? | Wrong clustering gives wrong uncertainty. | **domain** and **prompt_id** as primary inference/sensitivity dimensions; `page_type` is a **control, not a cluster**. | Understated SEs; false significance. |
| 9 | **Few-cluster inference** | What happens when the cluster count is small? | Cluster SEs and unrestricted bootstraps mislead with few clusters. | **Analytic cluster SE as headline** + a few-cluster warning; keep the current (unrestricted) wild cluster bootstrap as **sensitivity only** unless upgraded to a **restricted wild cluster bootstrap-t**. Do **not** let it overwrite the headline SE by default. | Overstated significance on small runs (see §4 note). |
| 10 | **Missing data & scrape selection** | Are missing values random, or tied to scrape success / source type? | Non-random missingness is informative and biasing. | Median-fill + missing indicators **plus** scrape-success diagnostics by cited status / source_type / page_type; report dropped rows. | Silent selection bias mistaken for a content effect. |
| 11 | **Multicollinearity** | Which features overlap conceptually? | Overlap widens error bars and destabilizes signs. | Use VIF + model-stability as evidence, but a human decides whether to **combine, drop, or relabel** (e.g. collapse similarity features into one relevance score). | Interpreting entangled coefficients individually. |
| 12 | **Multiple testing** | Which feature families get BH-corrected together? | Family definition changes which results "survive." | Define families **before** coding (content_structure, access, authority, relevance, commercial, page_type, source_type, freshness). | Cherry-picked families; over- or under-correction. |
| 13 | **Interpretation wording** | Report as causal effects or as associations? | Language is where overclaiming happens. | **Associations only**, unless there is randomization or a valid causal design. | Business/legal overclaim; loss of credibility. |
| 14 | **External data requirements** | Which constructs cannot be measured from the current pipeline? | Prevents dressing up proxies as truth. | Name them as out-of-scope: true domain authority, true brand authority, true search-engine index history, the full candidate retrieval set, internal AI ranking. | Confident claims about unmeasured constructs. |
| 15 | **Page-type independent / stratified analysis** | Should the model be run separately within each `page_type`, or should `page_type` only enter as a control in the pooled model? | This changes the estimand and reduces sample size, but can reveal heterogeneity by page role. | Use the **pooled model with `page_type` controls as the headline**. Add page-type independent / stratified analysis as **sensitivity / heterogeneity analysis** when each subgroup has enough rows and outcome variation. | Small subgroup regressions can create unstable coefficients, separation, weak cluster inference, and false discoveries. |

---

## 3. Feature Decision Registry

A template to classify every feature **before** it enters (or is barred from) the model. Fill or audit
the "Human Decision Needed" column per audit.

- **Timing** ∈ {pre-output, post-output, aggregate, scrape-diagnostic}
- **Role** ∈ {focal, control, mediator-sensitive control, proxy control, diagnostic-only, excluded}
- **Include in Main Model?** ∈ {yes, no, sensitivity only}

| Feature | Intended Construct | Timing | Role | Main Model? | Diagnostic? | Proxy Caveat | Human Decision Needed |
|---|---|---|---|---|---|---|---|
| `has_faq` | Answer-ready Q&A structure | pre-output | focal | yes | yes | Heuristic detection, not verified content quality | Proxy for broader completeness? |
| `has_price_or_package` | Commercial/transactional signal | pre-output | focal | yes | yes | Heuristic; meaning depends on intent | Interact with intent? |
| `has_contact_info` | Access/contact signal | pre-output | focal | yes | yes | Thin contact pages may be surfaced-not-cited | Read within `page_type`? |
| `has_bullets` | Scannable structure | pre-output | focal | yes | yes | Heuristic | Combine with other structure flags? |
| `heading_count` | Document structure | pre-output | control | yes (control) | yes | Scrape-dependent; count noise | Focal or control here? |
| `word_count` | Content length | pre-output | control | yes (control) | yes | Scrape-dependent; length ≠ quality | Length confounds structure flags? |
| `page_type` | Page role (article/product/contact/…) | pre-output (heuristic) | control (categorical) | yes (control) | yes | Heuristic class; coefficients relative to reference level | **Control, not a cluster** (see §4) |
| `source_type` | Source role (forum/news/official/…) | pre-output | control (categorical) | yes (control) | yes | Taxonomy heuristic | Reference category choice |
| `source_position` | **Observable source panel position** | **observed-output metadata / source-panel placement** | **mediator-sensitive control** | **sensitivity only** | yes | Panel placement — **not** an internal AI/retrieval/Google rank; may mediate relevance | Confounder vs mediator; with/without models |
| `log1p_source_position` | Log transform of panel position | **observed-output metadata / source-panel placement** | mediator-sensitive control | sensitivity only | yes | Same as `source_position`; per unit of `log(1+position)` | Functional form (log vs bins) |

> **`source_position` timing note.** `source_position` is an **observable source panel placement**
> captured from the AI output. It is **not** an internal retrieval rank, **not** a Google rank, and it
> **may mediate** part of the relationship between relevance/content and citation. It is therefore
> classified as **observed-output metadata**, used as a mediator-sensitive control (always shown with and
> without), never as a plain pre-output feature.
| `title_prompt_similarity` | Title↔prompt relevance | pre-output | focal/control | yes | yes | Lexical/embedding **proxy** for relevance | Prefer one relevance feature? |
| `page_prompt_similarity` | Page↔prompt relevance | pre-output | focal/control | yes | yes | Semantic-overlap **proxy** | Collinear with other sims |
| `max_chunk_prompt_similarity` | Best passage↔prompt relevance | pre-output | focal/control | yes | yes | **Proxy**; highly collinear with sibling sims | Collapse into `relevance_score`? |
| `page_answer_similarity` | Page↔**answer** overlap | **post-output** | **excluded** | **no** | yes (post-output) | **Circular** — derived from the answer | Never in main model |
| `max_chunk_answer_similarity` | Best chunk↔**answer** overlap | **post-output** | **excluded** | **no** | yes (post-output) | **Circular** — downstream of citation | Never in main model |
| `domain_seen_count` | CiteScope-observed domain visibility | aggregate | proxy control | sensitivity only | yes | **Proxy**, not true domain authority | De-leak / cap? |
| `domain_citation_rate` | Domain-level cite rate | aggregate | proxy control (**leaky-looking**) | sensitivity only (careful) | yes | Partially **self-predicts** `cited` (same-domain rows) | **Not focal**; leave-one-out to de-leak? |
| `citescope_visibility_history_score` | Observed visibility history | aggregate | proxy control | sensitivity only | yes | **Proxy**, not true search-engine index history | Keep out of headline |
| `scraped_ok` / `scrape_success` | Scrape/selection indicator | scrape-diagnostic | diagnostic-only (selection check) | **no** | yes | A **selection indicator**, not a content-quality variable | Selection-bias check by cited status |
| `freshness_days` | **Age in days (larger = older)** | pre-output (metadata) | control | sensitivity only | yes | Older may **proxy** authority/evergreen; not a recency virtue | Rename? Interpret sign carefully |
| `brand_appeared_in_answer` | Brand mention in the answer | **post-output** | **excluded** | **no** | yes (post-output) | **Circular** — downstream of citation | Never in main model |

**Standing rules encoded above:**
- **Prompt-based** similarity may be used in main models; **answer-derived** similarity is **diagnostic-only**.
- `source_position` is a **sensitivity/control with a mediator caveat** — always shown with and without.
- `domain_citation_rate` is **not focal** because it is leaky-looking (self-prediction within a domain).
  **If `domain_citation_rate` is used at all, prefer a leave-one-out version that excludes the current
  row.** Never treat `domain_citation_rate` as a focal feature because it partially self-predicts `cited`
  when computed using same-domain rows.
- `heading_count` and `word_count` are **controls by default**. They should become **focal only if the
  audit question is specifically about document structure or length** — otherwise they adjust for
  length/structure without being interpreted as findings.
- `scraped_ok` is a **selection/diagnostic** check, not a normal content-quality variable.
- `freshness_days` means **age in days**; larger = older.

---

## 4. Cluster and Inference Decision Matrix

Standard errors depend entirely on the clustering choice. Decide this by hand — it is not a mechanical detail.

| Cluster Candidate | What It Captures | Pros | Cons | Recommended Use |
|---|---|---|---|---|
| **domain** | Website/source-side correlation (many pages per site behave alike) | Usually many clusters; matches the "same site" dependence | Misses prompt-side dependence | **Primary** cluster / sensitivity dimension |
| **prompt_id** | Prompt/query-side correlation (many sources per prompt) | Captures shared-prompt dependence | Often **few** clusters (tens of prompts) → few-cluster problems | **Primary** cluster / sensitivity dimension |
| **domain × prompt_id** (two-way) | Both source-side and prompt-side dependence at once | Most honest when both matter | More complex; needs enough clusters on both margins | Use **when implementation and cluster counts allow** |
| **canonical_url / page_id** | Correlation from the **same page** scored repeatedly | Right level when pages recur across prompts | Useless if pages rarely repeat (degenerate) | **Sensitivity only**, and only when enough repeated pages exist |
| **run_id** | Between-run/temporal correlation | Useful with several runs | Degenerate with a single run (one cluster) | Use **only if multiple runs** exist |
| **record_id** | A single audit record / prompt-source unit | Fine when it maps to a repeated unit | **Degenerate if unique per row** (every cluster size 1) | Use **only if it maps meaningfully** to a repeated prompt/audit unit; avoid if unique |
| **page_type** | Page-role grouping | — | It is a **feature of interest / confounder**, not an error-correlation unit | **Do not cluster on it** — use it as a **control / dummy variable** |

**Why `page_type` is a control, not a cluster:** clustering assumes the grouping variable indexes
*correlated errors you are not modeling*. `page_type` is a **substantive feature** whose association with
citation you want to estimate — so it belongs in `X` as a dummy (interpreted against a reference level),
not in the SE machinery. Clustering on it would both hide its effect and mis-state uncertainty.

**Recommended headline inference rule:**
- Use the **analytic cluster SE as the default headline**.
- Prefer **domain**, or **two-way domain × prompt_id** when valid and cluster counts allow.
- If the cluster count is **fewer than ~40**, add a **few-cluster warning**.
- Keep the **unrestricted wild cluster bootstrap as sensitivity only** — do **not** let it overwrite
  the headline SE — **unless** it is upgraded to a **restricted wild cluster bootstrap-t** (impose the
  null, bootstrap the t-statistic, invert), which is the honest few-cluster procedure.

**Two-way cluster fallback rule:** use two-way `domain × prompt_id` clustered SEs **only when both
margins have enough non-degenerate clusters**. If one margin has too few clusters, **report domain-only
and prompt-only sensitivity separately, each with a warning**, rather than forcing a fragile two-way
estimate.

> **Current-state note.** The pipeline **now implements this rule**: the headline SE is the analytic
> HC3 / cluster SE and is **never overwritten**; with fewer than 40 clusters it adds a few-cluster
> warning and reports the unrestricted wild cluster bootstrap as a **sensitivity value only**
> (`se_wild_bootstrap` on each coefficient, `se_wild_bootstrap_sensitivity` in the inference table). An
> `inference_sensitivity` table reports the SE under HC3, cluster(domain), cluster(prompt_id), and
> two-way `domain × prompt_id` with a `recommended_inference`. A **restricted** wild cluster bootstrap-t
> remains the proper future upgrade for few-cluster settings and is a **human-reviewed decision (§2.9 / §6)**.

---

## 4A. Page-Type Independent / Stratified Analysis

**Page-type independent analysis means fitting separate citation models within each major `page_type`.
The row remains a surfaced source appearance, but the dataset is restricted to one `page_type` at a time.
This helps identify whether feature associations differ by page role. These subgroup results should be
treated as sensitivity or heterogeneity analysis, not as causal effects.**

Key points:
- It means running **one separate regression per `page_type`** (article, product, contact/location,
  directory/listing, unknown/other) — one `page_type` → one independent / stratified analysis.
- It is used to see whether **feature associations differ by page role** ("within this page type, which
  features are **associated with** citation probability among surfaced sources?").
- It is **not** the same as clustering by `page_type`. `page_type` is a **control or a stratification
  dimension**, never an error-correlation (cluster) unit.
- It should be treated as **heterogeneity / sensitivity analysis**, not as the headline and not as causal
  effects.
- It **reduces sample size** and can create **unstable estimates**, separation, and weak cluster inference.
- It should only run when each `page_type` subgroup has **enough rows and enough cited / more-only
  variation**. Suggested defaults: `min_n = 50`, `min_cited = 10`, `min_more_only = 10`. Subgroups that
  fail are **skipped with a diagnostic row** explaining why.

**Within each subgroup:** keep one row = one surfaced source appearance; do **not** include `page_type`
dummies (it is constant within the subgroup); keep the same outcome (`cited = 1` explicit citation,
`cited = 0` surfaced / more-only); use the same focal/content features where variation exists; drop
zero-variance / low-support features; prefer clustering by **domain and/or prompt_id**, falling back to
**HC3 with a warning** when no valid cluster exists inside the subgroup; and report subgroup `n`,
`cited_n`, `more_only_n`, `cited_rate`, and per-margin cluster counts.

The **pooled model with `page_type` controls remains the headline** (Decision §2.15); page-type
stratified models are the heterogeneity read alongside it.

---

## 5. What Can Be Vibe-Coded

Once the human decisions in §2–§4 are fixed, these implementation tasks are mechanical and safe to build
or regenerate freely (with tests):

- Build the design matrix from the agreed feature list.
- Coerce booleans/numerics to a numeric matrix.
- Median-fill numeric missing values.
- Add `{feature}_missing` indicators.
- One-hot encode categorical variables.
- Track and report reference categories.
- Drop zero-variance and perfectly-collinear (aliased) columns.
- Fit the LPM (OLS on the 0/1 outcome).
- Compute HC3 heteroskedasticity-robust SEs.
- Compute analytic cluster-robust SEs.
- Compute the inference-sensitivity table (SE type by clustering choice).
- Compute VIF (focal and full) and the condition number.
- Compute Benjamini–Hochberg q-values **by model × feature family** (families defined by a human).
- Fit a logit and compute the AME cross-check (probability-point scale).
- Detect perfect/quasi-separation.
- Generate forest plots (with and without position).
- Export CSV/Markdown diagnostics.
- Add warnings for **few clusters, high VIF, rare features, scrape failures, and low coverage**.
- Write tests that assert **answer-derived features are excluded from the main models**.

These are "how to compute it" tasks. They assume the "what to compute and how to read it" was decided by
a human.

---

## 6. What Should Not Be Vibe-Coded Without Human Review

These change **what question the model answers** or **what a result means**. Require an explicit human
econometric review before changing:

- Changing the **unit of analysis**.
- Changing the **outcome definition** (what counts as `cited`).
- **Adding answer-derived / post-output features** to the main effect models.
- Treating **proxies as true measurements** (relabeling `domain_seen_count` as "domain authority", etc.).
- **Choosing or changing cluster variables**.
- **Switching the headline inference method** (e.g. letting a bootstrap replace the analytic cluster SE).
- Interpreting **`source_position` as a normal control** without the mediator/post-treatment caveat.
- Making **causal claims** from observational associations.
- **Removing features only because coefficient signs look surprising** (surprise is not evidence of a bug).
- **Reporting significant p-values from few-cluster settings without a warning**.

---

## 7. Safe Reporting Language

Wording is where overclaiming happens. Prefer the left column.

**Safe (associational, honest about proxies and scope):**
- "Pages with FAQ sections are **associated with** higher citation probability **among surfaced sources**."
- "This is an **observational association**, not a causal estimate."
- "`domain_seen_count` is a **CiteScope-observed visibility proxy**, not true domain authority."
- "`source_position` is an **observable source panel placement** variable and **may mediate** part of the relationship."
- "`more-only` means **surfaced but not cited** — not rejected and not un-retrieved."
- "This coefficient is a **probability-point association**, holding the listed features fixed."

**Unsafe (avoid entirely):**
- "Adding FAQs **causes** the AI to cite the page."
- "The AI **rejected** these sources."
- "This **proves the reason** the AI selected the source."
- "`domain_seen_count` **measures domain authority**."
- "`source_position` is the model's **internal retrieval rank**."
- "This reflects the AI's **internal retrieval set**."
- "This is the page's **true index history**."

---

## 8. Acceptance Criteria

This document is complete when it:

1. **Separates** human econometric decisions (§2–§4) from vibe-codeable implementation (§5), with §6 as the guardrail. ✅
2. Defines the **unit of analysis** as one **surfaced source appearance**. ✅ (Scope + §2.1)
3. Explains the **`cited` vs more-only** outcome and that it is citation **within surfaced sources**. ✅ (§2.2)
4. Marks **answer-derived features** as **diagnostic-only**. ✅ (§2.3–§2.4, §3)
5. Explains **`source_position` as mediator-sensitive** (with/without models). ✅ (§2.6, §3, §4)
6. **Labels proxies** carefully (`domain_seen_count`, visibility history, `freshness_days`). ✅ (§2.7, §3)
7. Recommends **analytic cluster SE as headline** and the **wild bootstrap as sensitivity only** unless upgraded to a restricted bootstrap-t. ✅ (§2.9, §4)
8. Explains why **`page_type` is a control, not a cluster**. ✅ (§4)
9. **Avoids causal language** and gives safe/unsafe examples. ✅ (§7)
10. Includes **decision tables** future developers can fill in or audit. ✅ (§2, §3, §4)

---

_Companion docs: `docs/econometrics-process.md` (per-iteration change log), `docs/econometrics-calculation-audit.md` (white-box formula/line reference), and the engine `src/econometrics.py` / `src/confounders.py`._
