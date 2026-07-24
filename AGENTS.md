# AGENTS.md

## Project focus

CiteScope is currently a ChatGPT Bright Data source audit and content-econometrics
QA application. The product compares cited with more-only sources surfaced in an
observable Bright Data export. Gemini code remains in repository history but is
not part of the current application navigation.

Never describe more-only as rejected or ignored. Never claim to observe
ChatGPT's complete internal retrieval set. Econometric results are conditional
associations among surfaced sources, not causal effects or web-wide citation
probabilities.

## Run and test

```bash
source .venv/bin/activate
streamlit run app.py
python scripts/v2_validate_econometrics_migration.py
python -m compileall -q src ui scripts tests app.py
pytest -q
```

The app is normally available at `http://localhost:8501`.

## Data boundary

Code and clean notebooks are versioned. Raw crawler responses, generated tables,
figures, reports, backups, and review packets are not committed.

`CITESCOPE_RESEARCH_DATA_DIR` points to the external research archive. More
specific input/output overrides are available in `.env.example`. Never copy API
keys from the archive and never commit `.env`.

## Canonical econometric sequence

1. Notebook 07: scrape and content QA.
2. Notebook 08: taxonomy, missingness, and final pre-LPM diagnostics.
3. Notebook 09: content-feature LPMs.
4. Notebook 09 interpretation patch: canonical robustness interpretation.
5. Notebook 10: writing and factual-density features.
6. Notebook 11: writing/factual-density econometrics.

Full audit = 500 prompts. Measurable-content LPM sample = 498 prompts.
Migration counts and required artifacts are defined in
`config/econometrics_pipeline_manifest.json`.

Main reporting rules:

- Start with M1 and M2.
- Do not use final enriched page type as a main control.
- Use `page_type_url_seed_general_collapsed` only as a sensitivity.
- Report M5 strong-content and M10 outlier sensitivity before interpretation.
- Exclude answer-derived similarity, source position, observed rank, source
  origin, and citation-rate proxies from the main content model.

## Architecture

- `app.py`: ChatGPT audit and Content Econometrics QA routing.
- `src/`: headless engine; no Streamlit imports.
- `src/econometrics_eda_v2/`: scrape, taxonomy, diagnostics, feature, and model layers.
- `src/econometrics_qa.py`: read-only frontend data adapter and snapshot lookup.
- `src/storage.py`: local cache plus separate manual review persistence.
- `ui/views/chatgpt.py`: Bright Data upload/source audit.
- `ui/views/econometrics_qa.py`: overview, page comparison, prompts, taxonomy,
  model tables, and review export.
- `notebooks/`: clean source notebooks with outputs stripped.

Manual reviews must remain separate from model datasets. The QA frontend reads
validated outputs; it must not execute notebooks or refit models on Streamlit
reruns.

## Page comparison

The left panel displays historical normalized crawler content. The right panel
attempts a live iframe only after a user action. Many sites block framing through
`X-Frame-Options` or CSP. Always retain an open-original-page fallback. Do not
strip frame protections or proxy third-party pages to bypass them.
