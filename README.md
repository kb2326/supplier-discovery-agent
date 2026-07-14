# Supplier-Discovery Agent 

A self-contained procurement AI prototype. It turns a free-text buyer request into a **ranked, verified top-5 shortlist of suppliers** — and makes the whole pipeline visible: how many candidates entered and left each stage, why each supplier was dropped or demoted, and a per-criterion score breakdown for every result.

The focus is **architecture and reasoning**, not volume of code.

---

## Why this architecture — the core design decision

The first question for a supplier-discovery system is: *what retrieval engine do you use?*

We deliberately did **not** commit to a single pattern (Graph RAG, pure vector search, or text-to-SQL alone). The reason: each pattern fits a different kind of query clause, and the right engine depends on the data topology — which is unknown at MVP time:

| Query clause | Best engine | Why |
|---|---|---|
| "copper cable suppliers" | Vector / semantic search | Fuzzy — "stranded copper conductor" must match "copper cable" |
| "in the EU, ISO 9001 certified" | Structured filter / SQL | Hard binary constraint — exact, not fuzzy |
| "supplies Fortune 500 companies" | Graph traversal or attribute join | A relationship question |
| "no compliance issues" | External API (OFAC, sanctions) | Requires an authoritative live source |

Locking into Graph RAG prematurely would be wrong if the data is mostly tabular. Locking into pure vector search would miss the relationship question. **The answer is a router-based hybrid**: route each clause to the engine that fits, fuse the ranked candidate lists, then apply hard filters, weighted scoring, and verification.

This is the same pattern used by Scoutbee (acquired by Coupa, Oct 2025), Veridion, and Azure AI Search's "agentic retrieval."

The prototype implements this hybrid for the MVP stage. The graph engine and paid external APIs are in the architecture diagram but intentionally deferred — they slot into the same retrieval interface when needed, with no redesign.

---

## The pipeline

```
Understand → Find (structured + semantic) → Fuse (RRF) → Filter (gate) → Score (MCDA) → Verify → Top 5
```

### Stage 1 — Query understanding (`app/understand.py`)

**What it does:** Converts free text into a structured `QuerySpec` — product name, synonym list, location (ISO-2 country codes), required certifications, compliance flags, and MCDA weights.

**Why LLM function-calling:** The buyer's intent is unstructured and varies in phrasing. An LLM with a JSON tool schema (Claude Haiku) extracts it reliably: "no sanctions issues" → `compliance_must_pass: true`; "Europe" → all 27 EU member-state codes. This is standard structured-output extraction — cheap and fast.

**Demo default:** A deterministic keyword parser runs fully offline with no API key. Set `ANTHROPIC_API_KEY` to switch to real Claude Haiku — already wired in, any failure falls back automatically.

---

### Stage 2 — Entity resolution (`app/store.py`)

**What it does:** Loads the supplier dataset and collapses duplicates into golden records. Uses rapidfuzz (Jaro-Winkler fuzzy name match) anchored by strong identifiers — shared domain, shared tax ID.

**Why this matters first:** Duplicate vendor records are the silent killer in procurement data. One Fortune 500 audit found 47 different records for the same supplier across ERP, procurement, and finance systems — costing $12M+ annually in duplicate payments and reporting errors. Get entity resolution wrong and every downstream score and dedup is meaningless.

**Demo narrative:** `Nordic Cable Co` and `Nordic Cable Company Ltd` share a domain (`nordiccable.example`) and the same tax ID. The store collapses them into one golden record. The pipeline trace shows the merge.

**What the real API would be:**
- **GLEIF LEI** (free, CC0) — ISO 17442 legal entity IDs. "Who is who / who owns whom." Adds a third strong key beyond domain and tax ID.
- **D&B Direct+** match API — returns a `confidenceCode` (1–10) and a `matchGrade` (A/B/F/Z per field) for probabilistic entity matching at scale.

---

### Stage 3 — Retrieval / Find (`app/retrieve.py`)

**What it does:** Two retrieval arms run in parallel and each return a ranked candidate list:
1. **Structured retriever** — filters by country set, product list, certifications. Exact match.
2. **Semantic retriever** — TF-IDF over product descriptions. Fuzzy, synonym-aware.

**Why two arms:** The structured arm handles hard constraints exactly. The semantic arm handles product-capability fuzzy matching — "stranded copper conductor" scores high for a "copper cable" query, even if those exact words don't appear together. Neither arm alone is sufficient.

**Why TF-IDF and not real embeddings:** TF-IDF is zero-dependency and runs fully offline. In production it is a drop-in replacement — the `retrieve.py` function returns a ranked list either way; only the internals change.

**What the real APIs would be:**
- **Veridion Search API** — 134M+ companies across 250 countries, 320 attributes per profile, refreshed weekly. Accepts NL search with per-clause `strictness` controls. Returns `search_details` with a `headline`, `source_url`, and `snippets` as grounding evidence. Replaces the structured retriever arm.
- **sentence-transformers + vector DB** (Qdrant / Weaviate / pgvector) — real dense embeddings replace TF-IDF in the semantic arm. Same ranked-list interface.
- **Graph engine** (Phase 3, optional) — for relationship queries ("suppliers that supply to Fortune 500 companies" via graph traversal). Deliberately not built yet: if relationship queries aren't frequent in the real product, the graph engine never needs to exist.

---

### Stage 4 — Fusion (`app/fuse.py`)

**What it does:** Reciprocal Rank Fusion merges the two ranked candidate lists into one unified pool.

**Formula:** `score = Σ 1 / (k + rank)` where `k = 60`

**Why RRF and why k=60:** RRF is score-agnostic — it doesn't matter that the structured retriever and TF-IDF produce scores on different scales. No calibration needed. The k=60 constant and superiority over individual ranking methods come from Cormack, Clarke & Büttcher, *"Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods,"* SIGIR 2009. It is the default in Elasticsearch, OpenSearch, Weaviate, Qdrant, and Azure AI Search hybrid search.

**What production changes:** Nothing. Adding a third retriever arm (graph, or Veridion) is one line — pass its ranked list into `reciprocal_rank_fusion()`.

---

### Stage 5 — Hard-filter gate (`app/rank.py`)

**What it does:** Drops any candidate that fails a must-have. Three checks in order:
1. Location — country must be in the requested scope
2. Required certifications — must be present (claimed; validity checked later at verify)
3. Sanctions — a positive hit is always disqualifying

Every removal records a human-readable `drop_reason` so the trace can show *why*.

**Why binary before scoring:** Scoring a sanctioned supplier would be misleading — a high composite score implies recommendation. Hard fails must be excluded first. The reason must be recorded for the buyer's audit trail.

**Demo narrative:** `Adria Copper Trading Ltd` (Cyprus) has the highest `financial_health` score in the dataset. It still gets dropped with reason "OFAC sanctions hit" — because a sanctions hit is never overrideable by a good score.

**What the real API would be:**
- **OFAC SDN + Consolidated Sanctions** (free) — bulk files from treasury.gov + Sanctions List Search API with fuzzy name matching. Also covers EU, UN, and UK consolidated lists via third-party wrappers.

---

### Stage 6 — Weighted MCDA scoring (`app/rank.py`)

**What it does:** Multi-Criteria Decision Analysis. Each of six criteria is normalized to 0–1, multiplied by a configurable weight, and summed into a composite score. Per-criterion contributions are stored so the UI can show exactly why a supplier ranked where it did.

**Criteria:** product match strength (from RRF), track record (Fortune 500 count + years active), geographic proximity, financial health, ESG score, compliance (from the verify layer).

**Why MCDA and not a learned model:** Explainability is a hard requirement in procurement. A buyer needs to justify a shortlist to their manager, auditor, or legal team. MCDA gives a per-criterion breakdown for every result, in plain numbers. A learned reranker can be added later (it goes into the labeled no-op stub) once feedback data exists to train it.

**The weight sliders in the UI** let you drag "Compliance" high and watch the expired-cert supplier drop further in real time — each change re-calls `/search`.

---

### Stage 7 — Verify & cite (`app/verify.py`)

**What it does:** For each of the top-5 survivors, validates claims against mock registries:
- ISO certification status — active / expired / stale (stale = not checked recently)
- Sanctions screening — OFAC mock
- Fortune 500 customer corroboration — tiny corroborated allow-list vs self-reported claims

Returns a `Verification` object per claim: `status`, `source`, `source_url`, `snippet`, `confidence`, `last_verified`, `stale`. The compliance MCDA score is derived from these verifications — so a bad cert *demotes* the supplier even if it passed the gate.

**Why verification is non-negotiable:** Supplier claims are often self-reported marketing. An ungrounded recommendation is a liability — the buyer may act on a false claim. Every fact shown to the buyer must have a citation and a confidence score. Anything that cannot be corroborated is flagged `⚠ unverified`, not asserted as fact.

**Demo narrative:** `Rheinland Kupferkabel GmbH` has the best product match and track record in the field. The mock ISO registry returns its cert as **expired** → compliance score collapses → demoted from #1 to #4. The card shows `✕ expired ISO 9001` badge and a low compliance bar. This is the find-vs-verify tension made concrete.

**What the real APIs would be:**
- **IAF CertSearch / Global Accreditation Cooperation** (free) — 400K+ valid ISO certifications across 150+ economies. **Note:** IAF merged with ILAC into Global Accreditation Cooperation Incorporated on January 1, 2026 — confirm the live endpoint before relying on it.
- **GLEIF LEI REST API** (free, CC0) — legal entity identity and corporate hierarchy ("who owns whom").
- **OFAC Sanctions List Search API** (free) — per-supplier compliance confirmation at query time. Sanctions data changes daily; this must be live, not cached.

---

### Stage 8 — Reranker (`app/rank.py: rerank()`)

**What it does:** Currently a deliberately labeled no-op. Returns candidates in MCDA order unchanged.

**Why a stub:** MCDA ranking is fully explainable today. A cross-encoder or LLM reranker adds precision at the top of the list but reduces explainability and requires training signal. It plugs in here in Phase 2 once buyer-feedback data exists to train or prompt it.

---

## Demo script

Paste these into the search bar and watch the **Pipeline trace** panel.

**Query 1 — full pipeline:**
> `find copper cable suppliers in the EU, ISO 9001 certified, that supply Fortune 500 companies, with no compliance issues`

What to point at:
- **Entity resolution merge** — `Nordic Cable Co` + `Nordic Cable Company Ltd` collapsed into one golden record (visible in trace as a green merge card)
- **Sanctions drop** — `Adria Copper Trading Ltd` (CY) dropped at the gate; reason shown explicitly
- **Expired-cert demotion** — `Rheinland Kupferkabel GmbH` would be #1 on product match and track record alone; the expired ISO 9001 collapses its compliance score to near-zero → demoted to #4 (`✕ expired` badge, low compliance bar)
- **Unverified F500 claims** — `✓ verified` badge (e.g. Batavia → Shell, corroborated) vs `⚠ unverified` (self-reported Airbus, Siemens claims — shown but not asserted as fact)
- Stage counts: **40 total → 30 retrieved → 30 fused → 22 after gate → top 5**

**Query 2 — looser query:**
> `find copper cable suppliers in the EU`

With no cert requirement, `Hellas Copper SA` (has ISO 14001 but not ISO 9001) **survives** the gate. The sanctioned supplier stays dropped regardless — sanctions is never query-dependent.

**Use the MCDA weight sliders** to re-rank live. Drag "Compliance" to max — `Rheinland Kupferkabel` drops further. Drag "ESG" up — Dutch and Nordic suppliers rise.

---

## Setup & run

```bash
python -m venv venv && source venv/bin/activate   # Python 3.11+
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

Fully offline. No API keys required. Set `ANTHROPIC_API_KEY` to enable real Claude Haiku query parsing (falls back automatically on any failure).

```bash
python -m app.eval   # precision@5 + faithfulness check
```

---

## Module map

| File | Pipeline stage |
|---|---|
| `app/understand.py` | Query understanding — NL → `QuerySpec` (LLM + deterministic fallback) |
| `app/store.py` | Golden-record store + entity resolution |
| `app/retrieve.py` | Find — structured retriever + TF-IDF semantic retriever |
| `app/fuse.py` | Reciprocal Rank Fusion (k=60) |
| `app/rank.py` | Filter gate + weighted MCDA + reranker stub |
| `app/verify.py` | Verify & cite — mock cert/sanctions/identity checks |
| `app/pipeline.py` | Orchestration + observable Trace |
| `app/main.py` | FastAPI `/search` endpoint + static UI |
| `app/eval.py` | Gold-set precision@5 + faithfulness check |
| `static/index.html` | Single-page UI with pipeline trace and weight sliders |

---

## What's real vs mocked

Everything *interesting* runs for real; everything *expensive* is a local mock designed to be swapped at the same function interface.

| Stage | Runs for real today | Mocked today | Real API it would become |
|---|---|---|---|
| Entity resolution | rapidfuzz fuzzy match + domain/tax-id strong keys | — | GLEIF LEI + D&B Direct+ match API |
| Structured retrieval | Country/product/cert filter logic | 40-record JSON dataset | Veridion Search API (134M+ companies) |
| Semantic retrieval | TF-IDF over descriptions | — | sentence-transformers + pgvector |
| RRF fusion (k=60) | Full algorithm | — | Same — no change needed |
| Hard-filter gate | Location, cert, sanctions logic | — | OFAC SDN API (free) |
| MCDA scoring | Weighted composite with contributions | — | Same — weights tunable |
| Cert validation | Validation logic | Mock registry lookup | IAF / Global Accreditation Cooperation API (free) |
| Sanctions screening | Screening logic | 3-entry hardcoded list | OFAC Sanctions List Search + EU/UN/UK lists (free) |
| F500 corroboration | Corroboration logic | Tiny allow-list | No clean API — confidence flag is the honest answer |
| Query parsing | Deterministic keyword parser | — | Claude Haiku (set `ANTHROPIC_API_KEY`) |
| Reranker | Labeled no-op stub | — | Cross-encoder or LLM reranker (Phase 2) |

---

## Phased roadmap

- **Phase 0 (done):** Lock query schema, scoring criteria, mock dataset with deliberate narrative seeds (duplicate, expired cert, sanctions hit, unverified F500 claim).
- **Phase 1 / MVP (this prototype):** End-to-end pipeline on mock data. Optional real Claude Haiku for query parsing. Eval harness (precision@5, faithfulness).
- **Phase 2:** Wire in GLEIF (free) + OFAC (free) as first real verifications. License one firmographics provider (Veridion or D&B). Real embeddings for semantic search. Cross-encoder reranker.
- **Phase 3:** Knowledge graph if and only if relationship queries prove valuable in the real product. CDC for data freshness. RBAC, human-in-the-loop review, learning-to-rank from buyer feedback.

**Decision gates:** If precision@5 < 0.7 on the gold set → improve retrieval or entity resolution before adding features. If relationship queries are rare → defer the graph indefinitely.

<img width="498" height="278" alt="image" src="https://github.com/user-attachments/assets/462ced10-8b57-475f-9031-a14f00e53e96" />

