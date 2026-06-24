# CiteScope — Development Process, Architecture & Change Log

**Project:** AI Search Citation Audit (`citescope`)
**Repo:** https://github.com/keyboardban/citescope
**Status:** 2 commits — `4a61da4` (initial build, 2026-06-23) · `dcce37b` (validity/matching/batch upgrade, 2026-06-24)
**Size:** ~5,200 LOC across `src/` (engine), `ui/` (dashboard), `tests/`.

This document explains *why* the system is built the way it is, *how* it is structured, and *what changed at each step* — in detail.

---

## 1. What the system is (and the rule it never breaks)

CiteScope is a **black-box observational audit** of how an AI Search system (Gemini + Google Search Grounding) cites websites. It compares:

- **cited websites** — URLs that appear in Gemini's grounding metadata, and
- **non-cited reconstructed SERP candidates** — results we *independently* fetch from Apify for the same observed search queries.

**The governing rule:** we only describe **observable patterns**. We never claim to know the AI's internal retrieval set or why any page was/wasn't cited. This rule shaped the data model, the metric design, and the UI copy.

**Terminology contract** (used consistently in code + UI + report):

> reconstructed SERP · candidate websites · cited websites · non-cited SERP candidates · citation recall@K · weak domain-only match · semantic overlap proxy · observable patterns

Banned phrasing: *"AI rejected this site"*, *"exact internal SERP"*, *"proof of citation reason"*, *"AI definitely saw this page"*.

---

## 2. Development process & principles

The project was built MVP-first and verified continuously. The working principles, in priority order:

1. **Engine / UI separation.** Everything in `src/` is free of Streamlit imports, so the pipeline is testable headlessly (pytest, smoke scripts, `streamlit.testing.v1.AppTest`). `ui/` is the only Streamlit layer.
2. **Honest framing by construction.** Caveats and terminology live in `src/config.py` (`DISCLAIMER_*`, `GLOSSARY`, `CAVEAT_*`) and are reused everywhere, so the wording can't drift.
3. **Verify before claiming.** External APIs were confirmed against the *installed* SDKs (not memory): `google-genai` grounding fields, `apify-client` actor I/O, and — when a bug appeared — the live response shape. Each feature was exercised before being called "done."
4. **Defensive integration.** SDK calls use `getattr`/fallbacks, lazy imports, broad try/except that surfaces (not swallows) errors, and the raw API response is always preserved for the audit trail.
5. **Never cache failures.** Transient/empty results are not cached, so a blip can't "stick."
6. **Secrets stay in the environment.** Read from `.env` only; never written to disk, logged, or committed (`.gitignore` covers `.env`, `.streamlit/secrets.toml`, `.claude/settings.local.json`).
7. **Offline-explorable.** A synthetic `src/demo.py` run flows through the *real* matching/feature/analysis code, so the whole dashboard works with zero keys and zero spend — and doubles as an end-to-end smoke test.

### Verification toolchain
| Tool | Purpose |
|------|---------|
| `python -m compileall` | catch syntax/import errors fast |
| engine smoke scripts | exercise matching → features → analysis → report on the demo run |
| `streamlit.testing.v1.AppTest` | run `app.py` headlessly and render **every view** with demo data (catches runtime errors, no browser) |
| `pytest` (`tests/`) | unit tests for matching/recall, truncation, retry, abort, embedding cache |
| GitHub Actions (`.github/workflows/ci.yml`) | compile + pytest on every push |

---

## 3. Architecture

### 3.1 Directory layout

```
citescope/
├── app.py                     # Streamlit entry: page config, theme, sidebar nav, routing
├── requirements.txt           # runtime deps
├── requirements-dev.txt       # + pytest
├── .streamlit/config.toml     # theme (indigo, light)
├── .github/workflows/ci.yml   # CI: compile + pytest
├── src/                       # ENGINE (no Streamlit imports)
│   ├── config.py              # paths, secrets, defaults, tier groups, caveat text
│   ├── ids.py                 # run ids, stable hashes
│   ├── url_utils.py           # normalize / domain / redirect resolution
│   ├── storage.py             # SQLite: cache + run index + embedding cache + batches
│   ├── retry.py               # exponential backoff for transient API errors
│   ├── gemini_client.py       # grounded generation + trace extraction + embeddings
│   ├── apify_runner.py        # SERP actor + content-crawler actor + normalizers
│   ├── chunking.py            # heading-aware text chunking
│   ├── similarity.py          # lexical OR embedding cosine ("semantic overlap proxy")
│   ├── source_type.py         # source classification + institutional/brand-official
│   ├── matching.py            # tiered citation↔candidate matching + recall variants
│   ├── features.py            # one feature row per candidate (+ chunk scores)
│   ├── analysis.py            # cited vs non-cited comparison, recall, correlations
│   ├── batch.py               # multi-prompt runs + Mann-Whitney U + bootstrap CIs
│   ├── pipeline.py            # stage orchestration + run_full + abort/retry/cache
│   ├── report.py              # CSV/JSON/Markdown/HTML exports (single run + batch)
│   └── demo.py                # synthetic run for offline exploration + smoke test
├── ui/
│   ├── theme.py               # palette + injected CSS
│   ├── state.py               # session state + cached clients + recompute_downstream
│   ├── components.py          # cards, badges, callouts, pipeline diagram
│   ├── charts.py              # all Plotly visualizations
│   └── views/                 # one module per dashboard section (9 views)
├── tests/                     # pytest suite + conftest (isolated temp storage)
└── data/                      # runtime artifacts (gitignored): audit.db, runs/, raw/, exports/, batches/
```

### 3.2 Layered design

```
            ┌─────────────────────────── ui/ (Streamlit) ───────────────────────────┐
            │  app.py ─ sidebar/nav ─ views/* ─ components ─ charts ─ theme ─ state    │
            └───────────────▲───────────────────────────────────▲─────────────────────┘
                            │ reads run dict / calls stages       │ cached clients
            ┌───────────────┴───────────────────────────────────┴─────────────────────┐
            │                              src/ (engine)                                │
            │  pipeline (orchestration) ── matching ── features ── analysis ── batch     │
            │  gemini_client   apify_runner   similarity   chunking   source_type        │
            │  url_utils   retry   storage   ids   config   report   demo                │
            └───────────────────────────────────────────────────────────────────────────┘
                       │ google-genai            │ apify-client          │ SQLite + JSON
                  Gemini API (grounding)    Apify actors            data/
```

The UI never talks to external APIs directly; it calls engine *stages* and reads the **run dict**.

### 3.3 The data model (the "run dict")

A single dict is the contract between engine and UI; `pipeline.assemble_run()` builds it and `storage.save_run()` persists it (JSON snapshot + SQLite index).

```
run = {
  run_id, created_at, is_demo, used_fallback_query,
  inputs: { prompt, gemini{model,temperature,grounding,system_prompt},
            serp{top_k,country,language,selected_queries},
            scrape{scope,top_k,selected_urls,use_cache,crawler_type},
            analysis{similarity_method,embedding_model} },
  gemini: { output_text, search_queries[{query,is_fallback}], citations[{index,raw_uri,resolved_url,title,domain}],
            supports[], search_entry_point_html, finish_reason, prompt_feedback, raw, error, model, grounding },
  serp:   { candidates[{query,rank,url,title,snippet,displayed_url,result_type}], items, run_id, dataset_id, status, error },
  scrape: { pages{ normalized_url: {url,final_url,canonical_url,title,headings,text,markdown,metadata,status,published_date,...} }, apify },
  matching: { matches[], unmatched[], recall{strict,canonical,domain_inclusive}, rate_counts, cited_candidate_ids, weak_candidate_ids, n_citations, unique_candidates },
  features: [ one row per unique candidate ],   # see §3.5
  chunks:   { candidate_id: [{index,heading,text,n_words,output_sim,query_sim}] },
  analysis: { summary, group_compare, source_breakdown, official, correlation, length_sim_corr },
}
```

### 3.4 Pipeline & data flow

```
Prompt
  └► stage_gemini ─ grounded generate_content ─ extract trace ─ resolve redirect URLs (concurrent)
        │  (abort here if no output/citations/queries → save metadata, do NOT spend Apify credits)
        ▼
     stage_serp ─ Apify google-search-scraper ─ normalize → ranked candidate rows
        ▼
     select_scrape_urls (scope) ─ stage_scrape ─ Apify website-content-crawler ─ normalize pages
        ▼
     stage_match ─ unique_candidates → tiered match → cited/weak labels + 3 recall variants
        ▼
     stage_features ─ per-candidate features + chunk scores (pre-answer vs post-output)
        ▼
     stage_analyze ─ cited vs non-cited, recall, source/official, correlations
        ▼
     assemble_run → save_run → dashboard / export
```

Two entry modes share the same stages:
- **Interactive** (one button per view) — each view calls a single stage and stores the result; `state.recompute_downstream()` re-derives matching→features→analysis cheaply.
- **One-click** `run_full()` — chains all stages with a progress callback.
- **Batch** `run_batch()` — loops `run_full()` over many prompts, then aggregates.

### 3.5 Feature row (one per unique SERP candidate)

| group | fields |
|-------|--------|
| identity | `candidate_id, url, domain, root_domain, title` |
| label | `cited` (strong only), `weak_domain_match`, `match_type`, `strong_match` |
| **pre-answer** (non-circular) | `serp_rank, title_query_sim, snippet_query_sim, page_query_sim, max_chunk_query_sim, word_count, char_count, heading_count, freshness_days` |
| **post-output** (may be circular) | `page_output_sim, max_chunk_output_sim` |
| truncation | `original_char_count, used_char_count, truncated` |
| source | `source_type, institutional_official, brand_official_candidate` |
| status | `scrape_success` |

### 3.6 Caching & storage strategy

- **SQLite** (`data/audit.db`): `cache` (API results), `runs` (index), `embeddings` (persistent vectors), `batches`.
- **JSON on disk**: full run snapshots (`data/runs/`), raw API payloads (`data/raw/`), exports (`data/exports/`), batch summaries (`data/batches/`).
- Cache keys are `stable_hash` of stage inputs. **Failures are never cached** (Gemini errors, empty SERPs, failed scrapes). Embeddings are keyed by `sha1(text)+model+provider` so toggling settings never re-embeds.

### 3.7 Key design decisions & rationale

| Decision | Why |
|----------|-----|
| Stable `generate_content` + `grounding_metadata` path | Proven, defensive; raw response preserved. (Newer "Interactions API" / `url_citation` annotations also handled as a fallback.) |
| Resolve Vertex redirect URLs before matching | Grounding chunk URIs are `vertexaisearch…/grounding-api-redirect/…` wrappers; the real publisher URL is needed to match SERP candidates. |
| `normalize_url` forces `https` | http/https variants of the same page must match during citation matching. |
| Lexical similarity as default proxy | Offline, deterministic, free, no model download. Gemini embeddings optional. |
| Only **strong** matches set `cited=1` | A weak domain-only match must not mislabel an arbitrary same-domain page as cited. |
| Three recall variants | Separates URL-identity recall from canonical-equivalence and from exploratory domain-level recall. |
| Pre-answer vs post-output feature split | Page–answer similarity is partly *circular* (the answer is generated from cited sources); the split keeps conclusions honest. |

---

## 4. Development timeline & change log (detailed)

Two commits exist; iterations **A–E happened during the initial uncommitted development session and were folded into commit `4a61da4`**, while iteration **F is commit `dcce37b`**. Each entry lists *what, why, files, verification*.

### Iteration A — From-scratch build (→ `4a61da4`)
**What:** Designed and implemented the full system: engine (config, ids, url_utils, storage, chunking, similarity, source_type, gemini_client, apify_runner, matching, features, analysis, pipeline, report, demo) + Streamlit dashboard (theme, state, components, charts, 8 views) + README, `.env.example`, `.streamlit/config.toml`.
**Why:** Deliver an end-to-end black-box citation audit with a research-grade dashboard.
**Notes:** Reused proven conventions discovered in sibling projects (env var names `GEMINI_API_KEY`/`APIFY_TOKEN`, actors `apify/google-search-scraper` + `apify/website-content-crawler`, tiered URL matching, lexical similarity) but as a clean new implementation.
**Verification:** venv install on Python 3.14; engine smoke test on the demo run; `AppTest` rendered all 8 views green.

### Iteration B — Streamlit `width` API migration (→ `4a61da4`)
**What:** Replaced deprecated `use_container_width=True` with `width="stretch"` across all UI files.
**Why:** The flag was past its removal date (warnings on every render).
**Files:** `app.py`, `ui/views/*`.
**Verification:** AppTest re-run, warnings gone, all green.

### Iteration C — Fix `apify-client` v3 `Run` object (→ `4a61da4`)
**Symptom:** `Run full audit` → `AttributeError: 'Run' object has no attribute 'get'`.
**Cause:** apify-client **3.x** returns a pydantic `Run` object from `actor().call()`; the code read it as a dict (`run.get("defaultDatasetId")`).
**Fix:** Added `_run_field()` in `src/apify_runner.py` that reads `id` / `status` / `default_dataset_id` by attribute (and still supports legacy dict/camelCase). Applied in `run_serp` + `run_scrape`.
**Verification:** Reproduced with a real `Run.model_construct(...)` object + a fake Apify client; `run_full` (caching off) completed end-to-end → recall@10 = 1.0.

### Iteration D — Empty Gemini answer + heatmap crash (→ `4a61da4`)
**Symptom 1:** Observable trace showed `0 / 0 / 0` with grounding "on" and no error.
**Cause:** A failed/empty Gemini result was being **cached** and the full-audit path **swallowed** the error. (Confirmed via a live call that the integration itself works: 2,684-char answer, 9 citations.)
**Fix:** `gemini_client` now captures `finish_reason` / `prompt_feedback` and sets a helpful `error` on empty output; `run_search` view surfaces it; `pipeline` **stops caching failed/empty** Gemini runs (and empty SERPs / failed scrapes).
**Symptom 2:** `ValueError: Invalid value … 'colorscale' … 'Indigo'`.
**Cause:** `'Indigo'` is not a valid Plotly heatmap colorscale.
**Fix:** Custom indigo scale `[[0,'#f7f8fc'],[0.5,'#a5b4fc'],[1,'#4f46e5']]` in `ui/charts.feature_heatmap`.
**Files:** `src/gemini_client.py`, `src/pipeline.py`, `ui/views/run_search.py`, `ui/charts.py`.
**Verification:** Unit-tested the success/empty/exception branches of `run_grounded`; AppTest green incl. Feature Analysis.

### Iteration E — Add `gemini-3.5-flash` (→ `4a61da4`)
**What:** Added `gemini-3.5-flash` to the model selector and replaced a non-existent `gemini-3-flash` placeholder with the real `gemini-3-flash-preview` / `gemini-3-pro-preview`.
**Why:** User request; verified against the models the account can actually access.
**Note:** A live test of `gemini-3.5-flash` returned `429 RESOURCE_EXHAUSTED` (account quota), now surfaced clearly instead of silent zeros.
**Files:** `src/config.py`.

> **Then:** `git init` + GitHub repo `citescope` created (public), committed as `4a61da4` (37 files, +4,266). Secrets verified absent from the remote tree.

### Iteration F — Validity, matching & batch upgrade (commit `dcce37b`, 34 files, +1,550/−279)

A pipeline review identified that some conclusions could be misleading. This iteration addressed P0 (metric validity), P1 (robustness/cost), and P2 (research value).

**P0.1 — Pre-answer vs post-output feature split.**
*Why:* page/chunk–answer similarity is partly **circular** (the answer can be generated from cited sources). *Change:* `analysis.py` defines `PRE_ANSWER_FEATURES` / `POST_OUTPUT_FEATURES` + `FEATURE_PHASE`; `group_compare` tags each row with `phase`; Feature Analysis + report split into two sections; the post-output section carries a loud caveat (`config.CAVEAT_POST_OUTPUT`); insights prioritize pre-answer signals.

**P0.2 — Length-bias handling + truncation transparency.**
*Why:* lexical page–answer similarity correlates with page length. *Change:* `features.py` records `char_count`, `original_char_count`, `used_char_count`, `truncated` (cap = `config.MAX_SIM_CHARS = 8000`, no silent slicing); `analysis.length_sim_correlation()` + a length-vs-similarity scatter (`charts.length_vs_sim_scatter`) + `config.CAVEAT_LENGTH`; chunk-level similarity is led in headline comparisons.

**P0.3 — Strong vs weak (domain-only) matching.**
*Why:* a citation to `example.com/deep` must not be credited to `example.com/` (homepage) and must not inflate recall or mislabel pages. *Change:* `matching.py` separates **strong** tiers (exact/normalized/final_redirect/canonical/amp) from the **weak** `domain_only` tier; weak matches use a **closest-path** candidate; only strong matches set `cited=1`; weak candidates are tracked in `weak_candidate_ids` + `weak_domain_match`.

**P0.4 — Three recall variants.**
*Change:* `strict_recall` (URL identity), `canonical_recall` (+ canonical/amp), `domain_inclusive_recall` (+ weak domain) at @5/10/20/50, plus per-tier counts; `config.STRICT_TIERS`/`STRONG_TIERS`/`WEAK_TIERS`; UI cards + `charts.recall_grouped`; `config.CAVEAT_RECALL`. The `include_weak` toggle was removed (replaced by always-visible variants).

**P1.1 — Retry/backoff.** New `src/retry.py` (`with_retry`, `is_retryable`): retries 429/5xx/timeout hints, never 400/401/403/404; configurable via `RETRY_*` env. Wrapped Gemini `generate_content`/`embed_content` and Apify `actor().call()`.

**P1.2 — Abort before Apify on unusable Gemini.** `pipeline.run_full` raises `PipelineError` (after saving the failed trace) if Gemini produced no output/citations/queries — so it never spends Apify credits on an invalid run. The SERP view offers manual fallback queries instead.

**P1.3 — Concurrent redirect resolution.** `pipeline._resolve_citations` resolves Vertex redirect wrappers in a `ThreadPoolExecutor` (`REDIRECT_MAX_WORKERS=8`, timeout `4s`), cached per URL.

**P1.4 — Persistent embedding cache.** `storage.embedding_get/set` (SQLite `embeddings` table); `pipeline.make_sim_engine` checks it before embedding — so recompute/toggles don't re-bill embeddings.

**P2.1 — Batch mode.** New `src/batch.py`: `run_batch` over many prompts (failures isolated), pooled feature rows, sample sizes, recall@K averaged across runs, and cited-vs-non-cited **Mann-Whitney U** (scipy-free, tie-corrected) + **bootstrap median-difference CIs**. New `ui/views/batch.py` + nav entry + `report.batch_markdown_report` / `batch_features_csv`.

**P2.2 — Official detection split.** `source_type.classify` returns `(source_type, institutional_official)` (gov/edu/mil/int); new `brand_official_candidate()` flags an entity's *own* site (heuristic, lower confidence). Both surfaced as badges + an official-signals chart.

**P2.3 — Dashboard/report polish.** Amber `caveat_box`, brand badge, recall-variant cards/charts, pre/post sections, batch summary page; `report.py` rewritten for nested recall + sections + caveats + batch report.

**Tests + CI.** New `tests/` (matching/recall, truncation, retry, Gemini-abort, embedding cache) with isolated temp storage; `.github/workflows/ci.yml`; `requirements-dev.txt`.

**Verification:** `pytest -q` → 12 passed; compile clean; `AppTest` rendered all **9 views** green; engine smoke confirmed recall variants, weak-not-cited, brand detection, length correlation, retry classification, and batch stats.

---

## 5. Testing & verification (current)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m compileall -q src ui app.py tests   # syntax/imports
pytest -q                                      # 12 unit tests
streamlit run app.py                           # manual; or AppTest headless
```

`tests/` coverage: domain-only-not-cited + recall variants + feature labels (`test_matching.py`), truncation metadata (`test_features.py`), retry classification + backoff (`test_retry.py`), Gemini-abort short-circuit / Apify-not-called (`test_pipeline.py`), embedding-cache no-recompute (`test_embedding.py`). CI runs compile + pytest on every push.

---

## 6. Known limitations

- Reconstructed SERP ≠ the AI's internal results (time/region/personalization/ranking drift).
- Post-output similarity may be circular; pre-answer signals + rank are the cleaner evidence.
- `brand_official_candidate` is a domain-token heuristic (NER would be stronger).
- Single-run findings are anecdotal; use Batch mode for aggregated associations — and even those are observational, not causal.
- Freshness depends on a parseable page date; embeddings/batches consume API quota.

## 7. Future work

- Logistic regression / feature importances on the pooled batch dataset.
- Cross-engine/model comparison for the same prompt.
- SERP feature parity (PAA, knowledge panels, dates) + position-vs-fold modeling.
- Pluggable embedding providers + embedding-cache TTL; NER-based entity detection.

---

## Appendix

**Environment (`.env`):** `GEMINI_API_KEY`, `APIFY_TOKEN` (alias `APIFY_API_TOKEN`). Optional: `GEMINI_DEFAULT_MODEL`, `GEMINI_EMBED_MODEL`, `APIFY_SERP_ACTOR`, `APIFY_SCRAPER_ACTOR`, `RETRY_COUNT`/`RETRY_BASE_DELAY`/`RETRY_MAX_DELAY`.

**Match tiers (strong → weak):** `exact → normalized → final_redirect → canonical → amp_canonical → domain_only → no_match`. Strong = first five (sets `cited=1`); `domain_only` = weak (never cited by default); `no_match` = not in the reconstructed top-K.

**Recall variants:** `strict` = identity tiers · `canonical` = + canonical/amp · `domain_inclusive` = + weak domain (exploratory).
