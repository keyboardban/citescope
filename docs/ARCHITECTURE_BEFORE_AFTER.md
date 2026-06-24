# Architecture — Before vs After the Last Edit

> **"Last edit" = commit `dcce37b`** ("Improve citation audit validity, matching, and batch analysis", 2026-06-24).
> **Baseline = commit `4a61da4`** (initial build, 2026-06-23).
> The edit touched **34 files (+1,550 / −279)**. This document shows the architecture **before**, the architecture **after**, and exactly **what changed**.

---

## 0. At a glance

| Aspect | Before (`4a61da4`) | After (`dcce37b`) |
|---|---|---|
| Engine modules (`src/`) | 15 | **17** (+ `retry.py`, `batch.py`) |
| Dashboard views | 8 | **9** (+ Batch Mode) |
| Recall metric | 1 flat `recall@K` | **3 variants**: strict / canonical / domain-inclusive |
| Cited label | strong **or** weak (via `include_weak` toggle) | **strong only**; weak tracked separately |
| Domain-only match target | best-ranked page on the domain | **closest-path** page on the domain |
| Feature framing | one undifferentiated set | **pre-answer vs post-output** split + caveats |
| Text → similarity | silent `text[:8000]` | capped **with** `truncated` + char-count metadata |
| Official signal | `official_source` (gov/edu/mil/int) | `institutional_official` **+** `brand_official_candidate` |
| API calls | direct | wrapped in **retry/backoff** |
| Failed Gemini run | continues → spends Apify credits | **aborts before Apify** |
| Redirect resolution | serial, 8 s each | **concurrent** (8 workers), 4 s each |
| Embeddings | in-memory per engine | **persistent** SQLite cache |
| Multi-prompt analysis | none | **Batch mode** + Mann-Whitney U + bootstrap CIs |
| SQLite tables | `cache`, `runs` | + `embeddings`, `batches` |
| Tests / CI | none | `tests/` (5 files) + GitHub Actions |

---

## 1. Module map (NEW marked)

```
src/                          ui/
  config.py    (changed)        theme.py        (changed)
  ids.py                        state.py        (changed)
  url_utils.py (changed)        components.py   (changed)
  storage.py   (changed)        charts.py       (changed)
  chunking.py                   views/
  similarity.py                   overview.py            (changed)
  source_type.py (changed)        run_search.py
  gemini_client.py (changed)      serp.py                (changed)
  apify_runner.py (changed)       scraping.py
  matching.py  (changed)          matching.py            (changed)
  features.py  (changed)          content_visualizer.py  (changed)
  analysis.py  (changed)          feature_analysis.py    (changed)
  pipeline.py  (changed)          batch.py        ★NEW
  report.py    (changed)          report.py
  demo.py      (changed)
  retry.py     ★NEW            tests/            ★NEW (conftest + 5 test files)
  batch.py     ★NEW            .github/workflows/ci.yml  ★NEW
                               requirements-dev.txt      ★NEW
```

---

## 2. Architecture BEFORE (`4a61da4`)

**Pipeline:** linear, no early-abort, serial redirect resolution, no retry layer.

```
Prompt → stage_gemini → stage_serp → select_scrape_urls → stage_scrape
        → stage_match(…, include_weak) → stage_features → stage_analyze → run
```

**Matching (`matching.py`):**
- `match_all(citations, cands, pages, include_weak=False)`
- One flat recall dict: `recall = {"5":…, "10":…, "20":…, "50":…}`.
- `_match_one` returns a single best tier; `domain_only` resolved to the **best-ranked** candidate on the domain.
- `cited_candidate_ids` could include weak matches when `include_weak=True`.

**Features (`features.py`):** one undifferentiated set — `cited, match_type, strong_match, serp_rank, *_query_sim, page_output_sim, max_chunk_*_sim, source_type, official_source, word_count, heading_count, freshness_days, scrape_success`. Page text fed to similarity as `text[:8000]` with **no record** that truncation happened.

**Analysis (`analysis.py`):** `group_compare` = **means only** (no median, no phase); `official_compare` = official vs non-official; `summary_metrics` = flat `recall_5/10/20/50`. No length-bias check, no recall normaliser.

**Storage (`storage.py`):** tables `cache`, `runs`. No embedding cache, no batch storage.

**UI:** 8 views. Citation Matching had an **`include_weak` toggle**; recall shown as a single `recall_bar`; Feature Analysis was one section; Overview used flat `recall_10/20`. No Batch view, no `caveat_box`, no brand badge.

---

## 3. Architecture AFTER (`dcce37b`)

**Pipeline:** guarded + concurrent + retried.

```
Prompt → stage_gemini ─(retry)─► extract ─► resolve redirects (ThreadPool, 4s)
   │  └─ if NOT gemini_is_usable() → raise PipelineError (no Apify spend)
   ▼
stage_serp ─(retry)─► stage_scrape ─(retry)─►
stage_match(gemini, serp, scrape)            # no include_weak
stage_features  (embedding cache via storage) →  stage_analyze → run
```

**Matching:** strong vs weak separation, closest-path weak match, three recall variants.
- `match_all(citations, cands, pages)` — toggle removed.
- `recall = {strict:{…}, canonical:{…}, domain_inclusive:{…}}`.
- New helpers `_closest_path_candidate`, `_recalled(m, k, mode)`.
- `cited_candidate_ids` = **strong only**; new `weak_candidate_ids`.

**Features:** phase-split + truncation metadata + official split (see §5).

**Analysis:** `group_compare` adds **median + phase**; `summary_metrics` exposes nested `recall` + `recall_strict_*` + `recall_domain_10` + `n_weak_candidates`; new `length_sim_correlation` and `_normalize_recall`; `official_compare` reports institutional / brand-candidate / other.

**Storage:** `cache`, `runs`, **`embeddings`**, **`batches`** + `embedding_get/set`, `save_batch/load_batch/list_batches`.

**UI:** 9 views (+ **Batch Mode**). Recall shown via `recall_grouped` (3 variants); Feature Analysis split into **pre-answer** and **post-output** sections with amber caveats; new charts (`length_vs_sim_scatter`, `official_bar`); brand/weak badges; `caveat_box`.

---

## 4. What changed — by area (before → after)

### New files
| File | Purpose |
|---|---|
| `src/retry.py` | `with_retry` / `is_retryable` — backoff for 429/5xx/timeouts, never 4xx |
| `src/batch.py` | `run_batch`, `aggregate`, `mann_whitney_u`, `bootstrap_median_diff` |
| `ui/views/batch.py` | Batch Mode dashboard page |
| `tests/` (conftest + 5) | matching/recall, truncation, retry, abort, embedding cache |
| `.github/workflows/ci.yml` | compile + pytest on push |
| `requirements-dev.txt` | pytest |

### `config.py`
- **Added:** `STRICT_TIERS`, `STRONG_TIERS`, `RECALL_MODES`; `RETRY_COUNT/BASE_DELAY/MAX_DELAY`; `MAX_SIM_CHARS`, `REDIRECT_TIMEOUT`, `REDIRECT_MAX_WORKERS`; `CAVEAT_POST_OUTPUT/LENGTH/RECALL/BATCH`; `BATCHES_DIR` (added to `ensure_dirs`).
- `WEAK_TIERS` kept; `include_weak` concept retired.

### Matching & recall (`matching.py`)
```
- match_all(citations, cands, pages, include_weak=False)
+ match_all(citations, cands, pages)
- recall = {"5":…,"10":…,"20":…,"50":…}
+ recall = {"strict":{…},"canonical":{…},"domain_inclusive":{…}}
+ weak_candidate_ids = [...]          # cited_candidate_ids is now strong-only
+ matches[i] += weak_domain_match, strong_rank, weak_rank
+ domain_only → _closest_path_candidate(...)   # was: best rank on domain
```

### Features & validity (`features.py`)
- **Added fields:** `weak_domain_match`, `char_count`, `original_char_count`, `used_char_count`, `truncated`, `institutional_official`, `brand_official_candidate` (`official_source` kept as alias).
- Page similarity now scored on the capped slice **and the cap is reported** (`MAX_SIM_CHARS`).

### Official detection (`source_type.py`)
```
- classify(url) -> (source_type, is_official)
+ classify(url) -> (source_type, institutional_official)
+ brand_official_candidate(url, title, query, answer) -> bool   # NEW heuristic
```

### Analysis (`analysis.py`)
- `group_compare`: + `cited_median`, `noncited_median`, `phase`.
- `summary_metrics`: flat `recall_*` → nested `recall` + `recall_strict_*` + `recall_domain_10` + `n_weak_candidates`.
- **New:** `PRE_ANSWER_FEATURES`, `POST_OUTPUT_FEATURES`, `FEATURE_PHASE`, `_normalize_recall`, `length_sim_correlation`.
- `official_compare`: institutional / brand-candidate / other.

### Pipeline (`pipeline.py`)
```
+ class PipelineError(RuntimeError)
+ def gemini_is_usable(gemini) -> bool      # abort before Apify if false
- def stage_match(gemini, serp, scrape, analysis_inputs)
+ def stage_match(gemini, serp, scrape)
+ _resolve_citations  → ThreadPoolExecutor (REDIRECT_MAX_WORKERS, REDIRECT_TIMEOUT)
+ make_sim_engine     → persistent embedding cache via storage.embedding_get/set
  (Gemini/Apify calls now wrapped in retry.with_retry inside their clients)
```

### Storage (`storage.py`)
- **+ tables** `embeddings`, `batches`; **+ functions** `embedding_get/set/count`, `save_batch/load_batch/list_batches`.
- `save_run` recall extraction now handles nested recall (`recall.strict.10`).

### Report (`report.py`)
- Single flat recall table → **three-variant** recall table.
- One comparison table → **pre-answer + post-output** sections with caveats and length correlation.
- Official split table; **new** `batch_markdown_report`, `batch_features_csv`.

### UI (`charts.py`, `components.py`, `state.py`, `theme.py`, views)
- **charts:** + `recall_grouped`, `length_vs_sim_scatter`, `official_bar`.
- **components:** + `caveat_box`, brand badge; `site_card` shows institutional/brand/weak badges.
- **state:** `default_inputs` drops `include_weak`; `recompute_downstream` calls `stage_match(...)` without it.
- **views:** `matching` (toggle removed → 3 recall variants + counts), `feature_analysis` (pre/post sections + length scatter + official bar + insights), `content_visualizer` (truncation metadata + post-output caveat + brand badge), `overview` (recall_grouped + strict recall), `serp` (fallback-query warning when Gemini unusable), **new** `batch`. `app.py` adds the Batch nav entry.

### `url_utils.py`
- `resolve_redirect(timeout=8.0)` → `resolve_redirect(timeout=4.0)`.

---

## 5. Data-model diff (field level)

| Container | Before | After |
|---|---|---|
| `matching.recall` | flat `{"10":…}` | nested `{strict,canonical,domain_inclusive}` |
| `matching` | `cited_candidate_ids` (could be weak) | `cited_candidate_ids` (strong) **+ `weak_candidate_ids`** |
| `matching.matches[i]` | `match_type, strong, matched_rank, …` | **+ `weak_domain_match, strong_rank, weak_rank`** |
| feature row | `official_source` | **+ `institutional_official, brand_official_candidate`** (alias kept) |
| feature row | — | **+ `weak_domain_match, char_count, original_char_count, used_char_count, truncated`** |
| `analysis.group_compare[i]` | mean/delta | **+ `cited_median, noncited_median, phase`** |
| `analysis.summary` | `recall_5/10/20/50` | `recall` (nested) + `recall_strict_*` + `recall_domain_10` + `n_weak_candidates` |
| `analysis` | — | **+ `length_sim_corr`** |
| `inputs.analysis` | `include_weak` | *(removed)* |
| (new) `batch` | — | `{batch_id, prompts, per_prompt, features, aggregate{sample_sizes, group_stats(MWU+CI), recall, source_breakdown}}` |

---

## 6. Behavior-change summary (why each matters)

| Risk addressed | Before | After |
|---|---|---|
| Weak match inflating recall / mislabeling | domain-only could count as cited and use a homepage rank | strong-only `cited`; weak excluded from strict/canonical recall; closest-path candidate |
| Circular "page–answer similarity" conclusion | shown as a normal feature | isolated **post-output** section with a loud caveat; insights prioritize pre-answer |
| Hidden length bias / silent truncation | `text[:8000]` silently | `truncated` + char counts exposed; length↔similarity correlation surfaced |
| Wasted Apify spend on a failed Gemini run | pipeline continued | aborts with `PipelineError` before any Apify call |
| Transient 429/5xx failing a whole run | hard failure | retry with backoff (never retries 4xx) |
| Re-embedding on every recompute | in-memory only | persistent SQLite embedding cache |
| Single-run anecdotes | only mode | Batch mode with sample sizes, MWU p-values, bootstrap CIs (still observational) |
| Narrow "official" detection | gov/edu/mil/int only | + brand-official-candidate (entity's own site) |

---

## 7. Later additions (after `dcce37b`)

This document compares the `4a61da4 → dcce37b` edit. Two further iterations followed — see the full detail in **`DEVELOPMENT.md` §4 (Iterations G & H)**:

- **Topic Studies** (commit `9569012`) — `src/question_sets.py` (3 topic packs + paste-many, no ID/intent needed) and `ui/views/topics.py`; `batch.py` gained `by_topic` / `by_intent` / `patterns`. Engine modules 17→18, views 9→10.
- **ChatGPT Bright Data Source Audit** (working tree) — a **second audit mode** behind a sidebar switch: `src/brightdata.py` (parser), `src/chatgpt_pipeline.py`, `ui/views/chatgpt.py`, an `app.py` mode selector, and parameterized analysis helpers so both modes share them. Compares **cited** vs **more-only** sources; no recall@K. Engine modules 18→20, views 10→11, SQLite/JSON gains `data/chatgpt/`. Includes an input-vs-output file guard (`looks_like_input`). Tests 12→23.
