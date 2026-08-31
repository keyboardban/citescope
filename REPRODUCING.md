# Reproducing this analysis

The code is on GitHub. **The data is not** — it is large, and some of it is
client material. This lists exactly which files someone needs from you, in the
smallest set that still reproduces a given result.

Sizes and paths below were measured on the source machine, not estimated.

---

## The short version

| To do this | Send them | Size |
|---|---|---:|
| Run the app, read every model result | the econometrics package | **145 MB** |
| Also re-extract features from HTML | + the scrape cache | **+82 MB** |
| Everything, including provenance | the full research archive | 16 GB |

**227 MB covers almost every use.** The 16 GB figure is misleading: 5.8 GB of it
is duplicate pre-import backups, and the code only ever reads
`data/econometrics_v2/` inside it.

Nobody needs API keys to *analyse*. Keys are only for collecting new data.

---

## Level 1 — run the app and read the models (145 MB)

One directory. This is the validated econometrics package; features are already
extracted from HTML into tables, so no page bodies are required.

```
econometrics_redesign_v4_20260803_gemini_semantic_features/
├── model_run_manifest.json          the run this package came from
├── data/                     8.9 MB model-ready rows (model_ready_rows.csv, 3.3 MB)
├── tables/                    85 MB feature and result tables
├── frontend/                  47 MB precomputed views + manifests
├── feature_dictionary/       2.5 MB column definitions
├── diagnostics/              1.4 MB post-estimation diagnostics
└── reports/                         written interpretations
```

Point the app at it:

```bash
export CITESCOPE_ECONOMETRICS_DIR=/path/to/econometrics_redesign_v4_20260803_gemini_semantic_features
```

Covers: the QA frontend, D0–FE4 governed content models, the M0–M6 position
model views, feature distributions, and model comparison.

Does **not** cover: recomputing any feature from raw HTML.

---

## Level 2 — re-extract features from HTML (+82 MB)

Needed only to recompute rather than read. The HTML-parsing layers are
`document_structure_features`, `position_feature_eda`, `gemini_position_features`,
`verified_table_diagnostics` and `manual_feature_validation` — all of them parse
page bodies with BeautifulSoup.

```
CompareSearch-v2-clean/
└── data/econometrics_v2/
    ├── scrape_cache/          82 MB, 1,252 files
    │   ├── raw/                      Bright Data responses
    │   └── parsed/                   page_parse_rows.csv
    ├── exports/              9.6 MB  source_rows_raw, page_features,
    │                                 econometrics_row_level_sources
    └── scrape_queue/         428 KB
```

```bash
export CITESCOPE_RESEARCH_DATA_DIR=/path/to/CompareSearch-v2-clean
```

The variable points at the archive root, but only `data/econometrics_v2/` is
read. You can send that subtree alone and keep the directory name.

---

## Level 3 — the full archive (16 GB)

Only for provenance or re-running collection from scratch.

```
4.2 GB  outputs/
3.8 GB  data/                                   ← 92 MB of this is actually read
2.9 GB  data_backup_before_old_workspace_import/  ← backup, not needed
2.9 GB  data_backup_before_import/                ← backup, not needed
```

Do not ship this by default.

---

## Environment variables

| Variable | Needed for | Level |
|---|---|---|
| `CITESCOPE_ECONOMETRICS_DIR` | the model package | 1 |
| `CITESCOPE_RESEARCH_DATA_DIR` | HTML re-extraction | 2 |
| `CITESCOPE_ECONOMETRICS_DATA_DIR` | only if inputs and outputs live apart | optional |
| `CITESCOPE_ECONOMETRICS_OUTPUT_DIR` | as above | optional |
| `CITESCOPE_GEMINI_TAXONOMY_PATH` | custom taxonomy file | optional |
| `CITESCOPE_AREA_CONDO_*` | overriding the bundled 500-prompt inputs | optional |

---

## Credentials — do not send

Recipients create their own `.env` from `.env.example`. **Sending your keys means
paying for their usage**, and rotating afterwards is messy.

| Key | Needed for |
|---|---|
| `BRIGHTDATA_API_KEY`, `BRIGHTDATA_CRAWLER_DATASET_ID` | live page scraping |
| `GEMINI_API_KEY` | taxonomy and position block classification |
| `APIFY_TOKEN` | fallback scraper |

Every analysis path runs offline from cached data.

---

## Verification, in order

Each step is meant to fail loudly rather than produce a wrong number quietly.

```bash
# 1. code only — needs no data and no keys
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m compileall -q src ui scripts tests app.py
.venv/bin/pytest -q                       # expect 248 passed
```

```bash
# 2. with the package in place — checks the data contract before anything reads it
.venv/bin/python scripts/v2_validate_econometrics_migration.py
```

Expected output:

```text
migration_parity_passed
```

```bash
# 3. the app
.venv/bin/streamlit run app.py            # http://localhost:8501
```

```bash
# 4. the separate position model (Level 2 data required)
.venv/bin/python scripts/v2_run_position_model.py
# writes only to outputs/position_model_v1/, then:
#   Streamlit sidebar -> "Position Model — New"
```

If step 1 passes and step 2 fails, the problem is the data or the environment
variable, not the code.

---

## What will not be identical

- **Re-collecting gives different numbers.** The Bright Data exports are a
  snapshot of what ChatGPT served on a given date. Refetching produces different
  sources and different estimates. Reproducing *these* results requires *these*
  files.
- **Anything under `outputs/` regenerates.** Do not send it; a stale copy is
  worse than none, because it looks authoritative.
- **Superseded artifacts stay superseded.** `writing_structure_score_v2` and the
  v1 model specification are kept so old estimates remain reproducible, not
  because they are current. See `docs/CHANGELOG.md`.

---

## Before you send anything

Both datasets carry a real client's brand list and a competitor watchlist.
Decide who should receive that. This repository is currently **public** — the
code and documentation are clean of credentials and no data is committed, but
the analysis subject is identifiable from them.

Start recipients on Level 1. They can run the full test suite before you send a
single byte of data, which separates "the code works" from "my data is missing".
