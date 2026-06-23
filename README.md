# 🔎 AI Search Citation Audit

A Streamlit research dashboard that audits **how AI Search systems cite websites**.
It sends a prompt to **Gemini with Google Search Grounding**, captures the
observable trace (answer, search queries, citations, grounding metadata),
independently **reconstructs the SERP** for those queries via **Apify**, scrapes
the candidate pages, matches citations to candidates, and compares
**cited websites** against **non-cited reconstructed SERP candidates**.

> **This is a black-box observational audit.** We only observe what the Gemini API
> exposes. The Apify SERP is a *reconstructed candidate set*, **not** the exact
> internal results the AI used. We measure *observable patterns* associated with
> citations — we do **not** claim to reveal the AI's true retrieval or citation
> mechanism. A non-cited candidate was **not** "rejected"; similarity is a
> *semantic overlap proxy*, not proof of use.

---

## 1. What was built

A clean, modular system (engine + dashboard), ~3,900 LOC:

```
app.py                  Streamlit entry point (sidebar nav + routing)
src/                    Engine (no Streamlit imports — testable headless)
  config.py             paths, secrets, defaults, black-box framing text
  url_utils.py          URL normalize / domain / redirect resolution
  storage.py            SQLite cache + run index + JSON snapshots
  gemini_client.py      grounded generation + trace extraction + embeddings
  apify_runner.py       SERP actor + content-crawler actor (+ normalizers)
  chunking.py           heading-aware text chunking
  similarity.py         lexical (offline) OR Gemini-embedding cosine
  source_type.py        rule-based source classification + official flag
  matching.py           tiered citation↔candidate matching + recall@K
  features.py           one feature row per candidate (+ chunk scores)
  analysis.py           cited vs non-cited comparison, correlations
  pipeline.py           cache-aware stages + one-click run_full()
  report.py             CSV / JSON / Markdown / HTML exports
  demo.py               synthetic run for offline exploration
ui/                     theme, components, Plotly charts, 8 dashboard views
```

**Storage:** SQLite (`data/audit.db`) for an API-result cache (so Gemini/Apify
calls are never repeated by accident) and a run index; full run snapshots as JSON
in `data/runs/`; raw API payloads in `data/raw/`; exports in `data/exports/`.

---

## 2. How the pipeline works

```
Prompt
  → Gemini (Google Search Grounding)        gemini_client.run_grounded
      → answer · search queries · citation URLs · grounding metadata
      → resolve Vertex redirect URLs to real publisher URLs
  → Reconstructed SERP (Apify)              apify_runner.run_serp
      → ranked candidate websites
  → Scrape candidates (Apify)               apify_runner.run_scrape
      → title · headings · text · markdown · metadata
  → Citation matching                        matching.match_all
      → exact → normalized → final_redirect → canonical → amp → domain-only
      → label: cited=1 / non-cited candidate=0 · recall@5/10/20/50
  → Feature extraction                       features.build_features
      → rank, similarities (proxy), source type, freshness, word/heading counts
  → Compare / analyze                        analysis.*
  → Dashboard + Export
```

Each stage is cached on a hash of its inputs. Run stages one-by-one (each section
has its own button) or click **⚡ Run full audit** to chain them end-to-end with a
progress bar.

---

## 3. Set up API keys

Two keys are required for live runs (the **demo run needs none**):

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=...      # https://aistudio.google.com/apikey
APIFY_TOKEN=...         # https://console.apify.com/account/integrations
```

Keys are read from the environment only (never written to disk or logged). The
sidebar shows ✅/⛔ for each key. Optional overrides (actors, default model,
embedding model) are listed in `.env.example`.

---

## 4. Run the app

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints. **No keys yet?** Click
**🧪 Load demo run** in the sidebar to explore the entire dashboard with realistic
synthetic data and zero API spend.

**Tests:** `pip install -r requirements-dev.txt && pytest -q` covers matching/recall
variants, truncation metadata, retry/backoff, the Gemini-failure short-circuit, and the
embedding cache. CI runs them on every push (`.github/workflows/ci.yml`).

> **Model note:** the default is `gemini-2.5-flash` (a reliable grounding model).
> Model availability depends on your account — pick a current model from the
> selector if needed. The integration uses the stable `generate_content` +
> `grounding_metadata` path and preserves the raw response for every run.

---

## 5. Inputs the app expects

- **Prompt** — your question (e.g. *"What are the best tailors in Bangkok for custom suits?"*).
- **Gemini settings** — model, temperature, grounding on/off, optional system prompt.
- **SERP settings** — top-K (10/20/30/50), country code, language code, which
  queries to reconstruct (observed queries, manual queries, or a prompt fallback).
- **Scraping settings** — scope (top-K / only cited / all / selected URLs),
  crawler type (cheerio / playwright), use-cache toggle.
- **Analysis settings** — similarity method (lexical offline / Gemini embeddings).
- **Batch mode** — a newline-separated list of prompts to run and aggregate.

---

## 6. Outputs produced

- **Gemini trace** — answer, observed search queries (fallbacks clearly marked),
  citation URLs (raw + resolved), grounding supports, raw response.
- **Reconstructed SERP table** — query, rank, title, URL, snippet, domain, type,
  with cited rows flagged.
- **Scraped page dataset** — url, final/canonical url, title, headings, text,
  markdown, metadata, status.
- **Citation matching** — per-citation match type (strong vs **weak domain-only**),
  matched rank, unmatched list, three recall variants
  `strict_recall` / `canonical_recall` / `domain_inclusive_recall @5/10/20/50`,
  and per-tier match counts. Only strong matches set `cited = 1`.
- **Feature table** — one row per candidate: cited label (strong only),
  `weak_domain_match`, rank, **pre-answer** + **post-output** similarity proxies,
  source type, `institutional_official` + `brand_official_candidate`, word/char
  counts + truncation metadata, heading count, freshness, scrape status, match type.
- **Exports** — feature/SERP/matches CSV, full-run JSON, Markdown + HTML report.

---

## 7. Visualizations included

- **Pipeline diagram** with live stage counts (Overview).
- **Citation recall@K** grouped bar — strict / canonical / domain-inclusive.
- **SERP rank box/strip** — where cited sites sit in the reconstruction.
- **Cited vs non-cited** grouped means + per-feature distribution boxes.
- **Source-type** stacked bars + per-type cite-rate.
- **Match-type distribution** bar.
- **Chunk-relevance** chart (best chunk highlighted) + expandable chunk text.
- **Similarity radar** per candidate (vs all-candidate average).
- **Feature heatmap** (cited rows marked, min-max normalized).
- **Query → candidate → citation Sankey** flow.
- **Website cards** with inline similarity bars and badges.
- **Length-vs-similarity** scatter (length-bias check) + **official-signals** bar.
- **Batch summary** — pooled cited-vs-non-cited medians with Mann-Whitney U
  p-values and bootstrap CIs, plus recall@K averaged across runs.

The dashboard separates **pre-answer signals** (rank, query similarity — the cleaner,
non-circular signals) from **post-output overlap** (page/chunk–answer similarity), which
carries a loud caveat because the answer may be generated from cited sources.

---

## 8. Limitations

- We observe only what the Gemini API exposes; the **true internal retrieval set
  is unknown**.
- The **reconstructed SERP can differ** from what the AI saw (time, region,
  personalization, ranking churn, logged-in state).
- **Similarity is a proxy** for relatedness, not evidence the model read a page or
  chunk. Lexical similarity is shallow; embeddings are better but still a proxy.
- **Source-type / official** flags are heuristics.
- **Freshness** depends on the page exposing a parseable date.
- Citation **redirect resolution** and scraping can fail for some pages (handled
  gracefully, surfaced as failures).
- Findings are **per-run and correlational** — not statistically powered claims.

---

## 9. What to improve next

Implemented in this iteration: **batch multi-prompt aggregation** with Mann-Whitney U +
bootstrap CIs, **three recall variants**, **weak/strong match separation**, **pre-answer vs
post-output** feature split with circularity/length caveats, **brand-official candidate**
detection, **retry/backoff**, **Gemini-failure short-circuit**, **concurrent redirect
resolution**, and a **persistent embedding cache**.

Still worth doing:
- **Logistic regression / feature importances** on the pooled batch dataset.
- **Compare across AI engines/models** for the same prompt.
- **SERP feature parity** (people-also-ask, knowledge panels, dates) and
  position-vs-page-fold modeling.
- **Pluggable embedding providers** + embedding-cache TTL.
- **NER-based** entity/brand detection to replace the domain-token heuristic.

---

### Terminology (used carefully throughout)

reconstructed SERP · candidate websites · cited websites · non-cited SERP
candidates · citation matching · citation recall@K · observable patterns ·
semantic overlap proxy · chunk-level similarity · black-box analysis.

We avoid: "AI rejected this website", "AI definitely saw this", "exact internal
search result", "proof of citation reason", "causal explanation".
