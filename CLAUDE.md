# CLAUDE.md — CiteScope session recap & guide

> Auto-loaded each session. Read this first to recap where we are. _Last updated: 2026-06-29._

## What this project is
**CiteScope** = an AI-search **citation audit** (black-box, observational). Streamlit app.
Repo: https://github.com/keyboardban/citescope . Python **3.14** venv at `.venv`.

**Golden rule (never break the framing):** we only report **observable patterns**. Never claim we know the AI's
internal retrieval set or why a page was/wasn't cited. "more-only" / "non-cited" ≠ "rejected".
Similarity = a *semantic overlap proxy*, not proof of use.

## Two modes (sidebar switch in `app.py`)
1. **Gemini SERP Reconstruction Audit** — prompt → Gemini grounding → reconstructed SERP (Apify) → scrape →
   compare **cited** vs **non-cited SERP candidates** (citation recall@K). Views: Overview, Run AI Search, SERP,
   Web Scraping, Citation Matching, Content Visualizer, Feature Analysis, **Topic Studies**, **Batch Mode**, Report.
2. **ChatGPT Bright Data Source Audit** — upload a Bright Data export of ChatGPT runs → compare **cited sources**
   vs **more-only** (shown-but-not-cited). No SERP reconstruction, **no recall@K**. One tabbed page
   (Upload · Records · Source Table · Scrape · Feature Analysis · Questions · Intent · **Brand Visibility** · Content · Report).
   The **🏷️ Brand Visibility** tab is the **Non-branded Brand Visibility Audit** layer (client vs competitor).

## Run / test
```bash
source .venv/bin/activate
streamlit run app.py                 # launch UI (no keys? click "Load demo run" / "Load sample")
pytest -q                            # 78 tests
python -m compileall -q src ui app.py tests
```
Keys live in `.env`: `GEMINI_API_KEY` (Gemini mode + embeddings), `APIFY_TOKEN` (scraping in BOTH modes).
Headless check: `streamlit.testing.v1.AppTest` over `app.py` (renders every view with demo data).

## Repo map (engine = `src/`, no Streamlit imports; UI = `ui/`)
- Gemini engine: `pipeline.py` (orchestration/run_full), `gemini_client.py`, `apify_runner.py`, `matching.py`
  (tiered match + 3 recall variants), `features.py`, `analysis.py`, `report.py`.
- Shared: `url_utils.py`, `chunking.py`, `similarity.py` (lexical default / Gemini embeddings),
  `source_type.py`, `retry.py`, `storage.py` (SQLite cache+runs+embeddings+batches), `config.py`, `demo.py`.
- Topic Studies / Batch: `batch.py` (+ `question_sets.py` = 3 packs + paste parser).
- **ChatGPT mode:** `brightdata.py` (parser + **Prompt Manifest** match, incl. brand-term fields), `chatgpt_pipeline.py` (features + **intent→source-type** analysis), `ui/views/chatgpt.py` (Upload/Records/Sources/Scrape/Feature/Questions/**Intent**/**Brand Visibility**/Content/Report).
- **Per-question / clustering:** `cluster.py` (question×domain matrix + Jaccard agglomerative clustering) → ChatGPT "🧩 Questions" tab + Topic Studies "Question clusters".
- **Non-branded Brand Visibility:** `src/brand_visibility.py` (engine: term detection + record/intent/source/content/position tables) → ChatGPT "🏷️ Brand Visibility" tab + `report.py` brand exports/section + `demo.make_demo_brand_run()`.
- **Econometrics (citation model):** `src/econometrics.py` (statsmodels, guarded) — position-adjusted **LPM** (`cited`∼features) with **HC3 / cluster-robust** SEs, **wild cluster bootstrap** (<40 clusters), **VIF**, **Benjamini–Hochberg**, **logit+AME** cross-check. Wired via `analysis.econometric_analysis` into `stage_analyze` / `chatgpt_pipeline.analyze` (cluster `record_id`) / `batch.aggregate` (cluster `run_id`) / `brand_visibility.position_adjusted_regression`; rendered by `charts.coefficient_forest` + `components.regression_block` + report/JSON. **Cautious effect estimates under stated assumptions + a signed OVB caveat — a scoped exception to the observational rule.** Deps: `statsmodels`/`scipy` (wheel-verified on 3.14).
  - **Sensitivity & diagnostics:** `econometrics.model_comparison(df)` fits **A** content / **B** +source-authority / **C** +position / **D** reduced-similarity (+ FULL diagnostic), clustered by **domain** → model-comparison table (Δprob stability across A–D) + **VIF**, **anomaly** (6 auto-flags), **grouped-feature**, and **executive-summary** tables. Content features (`has_faq`, `page_type`, …) now computed for **every** ChatGPT source in `chatgpt_pipeline.build_features`. `report.forest_png` (matplotlib, guarded) + 4 CSV exporters (`econometrics_model_comparison/vif_diagnostics/anomaly_diagnostics/feature_group_summary`) + `report._sensitivity_section` + `components.sensitivity_block` (ChatGPT analysis tab "🧪 Sensitivity & diagnostics" + report-tab CSV/PNG downloads). Business caveats in `config.py` (observational / position-mediator / contact-location / similarity / age). Dep: `matplotlib` (guarded).
- Docs: `docs/DEVELOPMENT.md` (full architecture + change log A–I), `docs/ARCHITECTURE_BEFORE_AFTER.md`, `docs/24_06_2026.docx`.
- Data (gitignored): `data/{runs,chatgpt,batches,raw,exports}/`, `data/audit.db`.

## Session history (what we built)
1. Built the whole Gemini system from scratch → first commit `4a61da4`; created the GitHub repo `citescope`.
2. `dcce37b` — validity/matching/batch upgrade: strong-vs-weak matching, 3 recall variants
   (strict/canonical/domain_inclusive), pre-answer vs post-output feature split + caveats, retry/backoff,
   abort-before-Apify on failed Gemini, concurrent redirect resolution, persistent embedding cache,
   institutional vs brand-official, batch mode (Mann-Whitney U + bootstrap CIs), tests + CI.
3. `9569012` — **Topic Studies** mode (3 packs: Healthcare/Skincare, Automotive, Real Estate + paste-many) + docs.
4. `e3f9fce` — **ChatGPT Bright Data Source Audit** mode + input-vs-output guard + `CLAUDE.md` + doc updates.
5. **Per-question separation + question clustering** (`src/cluster.py`) — ChatGPT "🧩 Questions" tab + Topic Studies "Question clusters". (commit `9e31082`)
6. **Prompt Manifest + Intent → Source Type analysis** (commit `4254949`) — manifest (`prompt_id,topic,intent,prompt[,country,prompt_language,expected_source_types]`) matched to records by prompt text/hash → attaches intent/topic to every record/source/feature. "🎯 Intent" tab: intent×source-type counts+%, cited-by-intent, more-only-by-intent, cited-vs-more comparison, expected-vs-actual.
7. **Upload limit → 500 MB** (`.streamlit/config.toml` `maxUploadSize/maxMessageSize`; needs server restart) + **AI-ready reports** — both reports now embed a feature dictionary, a feature↔citation correlation table, intent breakdowns (ChatGPT), an "how to analyze (for an AI)" guide, and the **raw per-source/candidate CSV**; ChatGPT adds an **Analysis bundle (JSON)** + per-source dataset CSV downloads. (commit `8ccc94d`)
8. **Non-branded Brand Visibility Audit** (Iteration I, commit `4a667bf` on `main`) — `src/brand_visibility.py` layer + "🏷️ Brand Visibility" tab. Manifest gains `client_brand_terms_to_detect_in_output` / `competitor_terms_to_detect_in_output` / `prompt_is_nonbranded` / `visibility_goal`. For **non-branded** prompts: detects client/competitor in prompt/answer/sources/scraped pages → record table (all prompts kept = denominator), intent rollup (denominator = non-branded prompts; client-vs-competitor cited delta + examples), source/page table (brand-matched only), bilingual heuristic **content features** + `page_type`, **cited-vs-more-only** comparison, and **position-controlled** (1-3/4-6/7-10/11+) comparison. 6 CSV exports + report section + JSON block; offline brand demo (Thai hospital + auto).
9. **Econometrics layer — position-adjusted citation model** (Iteration J, branch `econometrics-layer`) — `src/econometrics.py` (statsmodels): LPM of `cited` on features adjusting for position, HC3 / cluster-robust SEs (cluster `record_id`/`run_id`), wild cluster bootstrap (<40 clusters), VIF, Benjamini–Hochberg, logit+AME cross-check. Wired into all 3 modes + report + JSON + forest-plot UI. Reports **cautious effect estimates under stated assumptions + signed OVB** (scoped exception to the observational rule); keeps the old correlation table relabeled "unadjusted." Added `statsmodels`/`scipy` deps.
10. **Econometrics careful-reporting upgrade** (Iteration K, branch `econometrics-layer`) — **safer clustering** (`econometrics.choose_cluster`: prefer `domain`→`prompt_id`→`record_id`→repeated page key; skip degenerate unique-id / single-run; `cluster_variable`/`cluster_count`/`cluster_warning`). **BH within each model × feature family** (`fit_citation_model.bh_families` + `econometrics_multiple_testing_summary.csv`). **Prompt-vs-answer similarity separation** (answer-derived features barred from the main model; `relevance_score` built from prompt sims only + missingness count). **Missing-indicator** for every numeric feature in `design_matrix`. **VIF focal + full + condition number**; feature groups split (`content_structure`, `source_type`, `missingness`) with `num_p_lt_05`/`num_q_lt_10`/`warnings`. **Logit-AME cross-check table** (LPM headline) + **per-feature separation diagnostics**. New diagnostics: **dedup / scrape-success / overlap-positivity / rare-feature / reference-categories / outcome-definition**. Report `_sensitivity_section` gains exec-summary + interpretation-safety + model-spec + diagnostics + business-safe-recommendation sections; **16 econometrics exports** (CSV/TXT/2 forest PNGs: focal + no-position) wired in the ChatGPT Report tab; `components.sensitivity_block` renders the new tables. New `config.py` caveats (cluster-few / position-panel / LPM / missingness / reference / signed-OVB / business-rec / outcome-def / rare). 6 new tests. _(latest)_

## Repo state
`main` = `4a667bf` (Non-branded Brand Visibility Audit, pushed). **Current branch = `econometrics-layer`** (off `4a667bf`),
holding the **Econometrics citation-model layer** (NEW `src/econometrics.py`; `analysis.econometric_analysis`; wiring in
`pipeline`/`chatgpt_pipeline`/`batch`/`brand_visibility`; `report.py` regression section + JSON; `ui/charts.coefficient_forest` +
`components.regression_block` + view sections; `config.py` econ caveats/thresholds; `requirements.txt` statsmodels+scipy;
`tests/test_econometrics.py`; demo similarity de-correlated; `docs/DEVELOPMENT.md` Iteration J) **plus a
sensitivity/diagnostics extension** (`econometrics.model_comparison` + A/B/C/D specs + VIF/anomaly/group/
exec-summary tables; content features on every ChatGPT source; `report.forest_png` + 4 CSV exporters +
`_sensitivity_section`; `components.sensitivity_block` + ChatGPT analysis/report UI; `config.py` business
caveats; `requirements.txt` matplotlib) **plus the Iteration K careful-reporting upgrade** (safer
`choose_cluster`; BH within model×family; prompt/answer similarity separation; numeric missing-indicators;
VIF focal+full+condition number; logit-AME-check + separation/dedup/scrape/overlap/rare/reference-category/
outcome-definition diagnostics; 16 econometrics exports incl. focal + no-position forest PNGs; expanded
`_sensitivity_section`; new `config.py` caveats).
**78 pytest tests pass; AppTest renders both modes incl. the regression + sensitivity sections; verified end-to-end on a real 1,270-source ChatGPT audit.** Untracked reference files
(textbook PDFs, `docs/demo/` HTML, `scripts/`) deliberately left uncommitted (public repo — copyright/size). If you change
code: run `pytest -q` + AppTest, then commit/push when the user asks.

## Key gotchas (these bit us — remember them)
- **Bright Data INPUT vs OUTPUT files.** The `*_prompts.csv` (cols `url,prompt,country,…`) are *input* prompt lists
  (0 sources → empty audit). The real *output* is the large `sd_*.json` (10–100 MB, has `citations`/`search_sources`).
  Parser now flags inputs via `looks_like_input`/`n_sources` and the Upload tab errors clearly.
- **Where scores/graphs live:** the **Feature Analysis** page/tab (+ Citation Matching, Topic Studies/Batch for Gemini).
  **Report = export only** (CSV/JSON/MD). To get me data: I can read `data/chatgpt/*.json` & `data/runs/*.json` directly.
- **Content scores (page–answer / chunk similarity, Content-tab graphs) need scraping** (Apify token, costs credits).
- Scraper = Apify `website-content-crawler`; `crawler_type` ∈ {`cheerio` (default), `playwright:adaptive`, `playwright:firefox`} — `cheerio` for static, `playwright:*` for JS sites. No non-Apify scraper yet.
- apify-client is **v3** → `.call()` returns a pydantic `Run` (read attrs, not `.get`). google-genai uses
  `generate_content` + grounding_metadata. Never cache failed/empty results.

## User's data on disk right now
3 real ChatGPT audits already parsed & saved in `data/chatgpt/` (re-uploaded after the input-file mixup):
- `sd_mqrviyf…` = **Real Estate** (218 sources, 116 cited), `sd_mqrvj5…` = **Automotive** (210/95),
  `sd_mqrvjd7…` = **Healthcare/Cream** (178/107). A 98 MB **all-3-topics** file (36 records) is at `~/Downloads/sd_mqrrwo0aolwt05d78.json`.
- Stale empty snapshots (the `*_prompts.csv` uploads, 0 sources) still clutter the sidebar — offer to delete.

## Observable findings so far (from saved data, no scraping)
- **Reddit / forums are the most-cited source in ALL 3 topics** (forum cite-rate ≈100%; Reddit = #1 cited domain each).
- "Official/institutional cited more" is **not** clearly supported (institutional cite-rate 43–67%, often ≤ "other").
- Title–prompt similarity barely separates cited vs more-only → source *type/domain* matters more than title wording.

## Likely next steps (ask the user which)
- Commit + push the ChatGPT mode + doc updates to `citescope`.
- Deeper cross-topic analysis report (no cost), or **scrape** cited+more-only pages for content/answer-similarity scores (Apify).
- ✅ "What predicts citation" regression — done (Iteration J, `econometrics-layer`; merge when ready). Optional next: Gemini-vs-ChatGPT cross-mode comparison; clear stale empty snapshots; run the citation model on the 3 real ChatGPT audits (178–218 sources → enough rows to fit).
