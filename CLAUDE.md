# CLAUDE.md — CiteScope session recap & guide

> Auto-loaded each session. Read this first to recap where we are. _Last updated: 2026-06-25._

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
   (Upload · Records · Source Table · Scrape · Feature Analysis · Content · Report).

## Run / test
```bash
source .venv/bin/activate
streamlit run app.py                 # launch UI (no keys? click "Load demo run" / "Load sample")
pytest -q                            # 27 tests
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
- **ChatGPT mode:** `brightdata.py` (parser), `chatgpt_pipeline.py` (features+analysis, reuses engine), `ui/views/chatgpt.py`.
- **Per-question / clustering:** `cluster.py` (question×domain matrix + Jaccard agglomerative clustering) → ChatGPT "🧩 Questions" tab + Topic Studies "Question clusters".
- Docs: `docs/DEVELOPMENT.md` (full architecture + change log A–H), `docs/ARCHITECTURE_BEFORE_AFTER.md`, `docs/24_06_2026.docx`.
- Data (gitignored): `data/{runs,chatgpt,batches,raw,exports}/`, `data/audit.db`.

## Session history (what we built)
1. Built the whole Gemini system from scratch → first commit `4a61da4`; created the GitHub repo `citescope`.
2. `dcce37b` — validity/matching/batch upgrade: strong-vs-weak matching, 3 recall variants
   (strict/canonical/domain_inclusive), pre-answer vs post-output feature split + caveats, retry/backoff,
   abort-before-Apify on failed Gemini, concurrent redirect resolution, persistent embedding cache,
   institutional vs brand-official, batch mode (Mann-Whitney U + bootstrap CIs), tests + CI.
3. `9569012` — **Topic Studies** mode (3 packs: Healthcare/Skincare, Automotive, Real Estate + paste-many) + docs.
4. `e3f9fce` — **ChatGPT Bright Data Source Audit** mode + input-vs-output guard + `CLAUDE.md` + doc updates.
5. **Per-question separation + question clustering** (`src/cluster.py`) — ChatGPT "🧩 Questions" tab (per-question drilldown, question×domain heatmap, clusters) + Topic Studies "Question clusters". _(latest)_

## Repo state
All session work is **committed & pushed to `main`** (latest commit covers per-question clustering); working tree clean.
**27 pytest tests pass; AppTest renders both modes.** If you change code: run `pytest -q` + AppTest, then commit/push when asked.

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
- Optional: logistic-regression "what predicts citation" on pooled data; Gemini-vs-ChatGPT cross-mode comparison; clear stale empty snapshots.
