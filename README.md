# Supplier-Discovery Agent — interview demo

A small, self-contained prototype of an AI procurement **supplier-discovery agent**. It turns a
free-text request ("find copper cable suppliers in the EU, ISO 9001 certified…") into a ranked,
**verified** shortlist, and — crucially — exposes the whole pipeline as an observable trace: how
many candidates entered and left each stage, and the reason every supplier was dropped or demoted.

The point of the demo is the **architecture**: a clean `find → rank → verify` pipeline, designed so
the data topology can change (add a graph engine, swap in real vector search or live verification
APIs) **without redesigning the flow**. Each module maps 1:1 to a stage of that pipeline.

```
Understand → Find (structured + semantic) → Fuse (RRF) → Filter (hard gate)
           → Score (weighted MCDA) → Verify (cited) → Top 5
```

## What is real vs mocked

Everything *interesting* runs for real; everything *expensive* is a local mock.

| Real (genuinely runs)                                              | Mocked (local lookup)                                        |
| ------------------------------------------------------------------ | ----------------------------------------------------------- |
| Entity resolution (rapidfuzz strong-key + fuzzy name merge)        | The ~40-supplier dataset (`data/suppliers.json`)            |
| Structured retrieval + TF-IDF semantic retrieval (scikit-learn)    | Certificate registry (IAF CertSearch → `verify.py`)         |
| Reciprocal Rank Fusion                                             | Sanctions screening (OFAC → `verify.py`)                    |
| Hard-filter gate + weighted MCDA scoring                          | Customer-reference corroboration (Fortune 500 claims)       |
| Per-claim verification logic, flags, faithful rationales          | The LLM query parse is optional; a deterministic parser is default |

No database, no auth, no Docker, no cloud, no message queue, and **no network calls at runtime**.

## Setup & run

```bash
python -m venv venv && source venv/bin/activate     # Python 3.11+
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

Runs **fully offline**. Query parsing uses a deterministic rule-based parser by default. If you
export `ANTHROPIC_API_KEY`, `understand.py` will instead call Claude with a tool schema to extract
the query spec — but the demo never depends on it (any failure falls back automatically).

Run the eval harness:

```bash
python -m app.eval        # prints precision@5 + a faithfulness check
```

## Demo script

Type these into the page and watch the **Pipeline trace** panel.

**1. The full pipeline** — paste:

> `find copper cable suppliers in the EU, ISO 9001 certified, that supply Fortune 500 companies, with no compliance issues`

What it proves, visible in the trace and cards:
- **Entity resolution** collapses a duplicate — *Nordic Cable Co* + *Nordic Cable Company Ltd*
  (same domain) merge into one golden record (shown with a "merged sup_004" badge).
- **Sanctions drop** — *Adria Copper Trading Ltd* (CY) scores well on paper but hits the mock OFAC
  list, so it is **dropped at the gate** with a visible reason.
- **Expired-cert demotion** — *Rheinland Kupferkabel GmbH* has the *best* product-match, track
  record and proximity of the field, yet its ISO 9001 comes back **expired** from the registry,
  collapsing its compliance score and **demoting it from #1 to #4** (it carries an `✕ expired` badge
  and a low compliance bar). This is the find-vs-verify tension made concrete.
- **Unverified Fortune 500 claims** — corroborated relationships (e.g. Batavia → Shell) show a
  `✓ verified` badge; uncorroborated self-reported claims (e.g. Airbus, Siemens) are shown but
  flagged `⚠ unverified` — never asserted as fact.
- Counts shrink through the trace: **30 retrieved → 30 fused → 22 after the gate → top 5**.

**2. A looser query** — paste:

> `find copper cable suppliers in the EU`

The gate behaves differently: with no required certification, *Hellas Copper SA* (which lacks ISO
9001) now **survives** where it was dropped before — while the sanctioned entity stays dropped (a
sanctioned supplier is never recommended, regardless of the query). The trace makes the contrast obvious.

Use the **MCDA weight sliders** to re-rank live (each change re-calls `/search`).

## Module map

| File              | Pipeline stage                                                             |
| ----------------- | -------------------------------------------------------------------------- |
| `app/understand.py` | Query understanding — NL → `QuerySpec` (LLM optional, deterministic fallback) |
| `app/store.py`      | Golden-record store + entity resolution (the data layer)                 |
| `app/retrieve.py`   | "Find": structured retriever + TF-IDF semantic retriever                  |
| `app/fuse.py`       | Reciprocal Rank Fusion (recall)                                           |
| `app/rank.py`       | "Rank": hard-filter gate + weighted MCDA + reranker stub                  |
| `app/verify.py`     | "Verify": mock cert / sanctions / identity checks with citations         |
| `app/pipeline.py`   | Orchestrates the stages and assembles the `Trace`                        |
| `app/main.py`       | FastAPI `/search` endpoint + serves the single-page UI                    |
| `app/eval.py`       | Gold-set precision@5 + faithfulness check                                 |
| `static/index.html` | Single-page UI (no build step)                                           |

## Production extensions (deliberately out of scope here)

Named so a reader sees where each plugs in without a redesign:
- **Graph engine** for relationship queries (ownership, shared-director, tier-N supply chains) — joins the retrieval layer alongside structured + vector search.
- **Real vector embeddings** replacing TF-IDF in `retrieve.py` (same ranked-list interface, so fusion is unchanged).
- **Live verification APIs** — IAF CertSearch, GLEIF, OFAC — dropped behind the same `verify.py` function signatures.
- **A real reranker** (cross-encoder or LLM) in the labeled no-op stub in `rank.py`.
- **Streaming responses** so the trace renders stage-by-stage as it runs.
- **Access control** on `/search` and per-source verification credentials.
