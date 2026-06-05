"""Stage 2 — the "find" layer: two complementary retrievers.

Maps to the two retrieval boxes in the diagram:

  * structured_retrieve  -> candidate generation by attribute match.
        Stands in for SQL/structured filters over a supplier DB.
  * semantic_retrieve    -> TF-IDF cosine similarity over products + description.
        Stands in for vector / embedding search.

In production these would be SQL filters + a vector index, and a graph engine and
live supplier APIs would join them. Those are named in the README as the production
extension; here two local retrievers are enough to demonstrate the fusion pattern.

Each retriever returns a ranked list of (supplier_id, score) descending by score.
Ranks (not raw scores) are what fusion consumes, which is exactly why RRF is robust
to the two retrievers using completely different score scales.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import QuerySpec, Supplier


def structured_retrieve(suppliers: list[Supplier], spec: QuerySpec) -> list[tuple[str, float]]:
    """Attribute-match candidate generation.

    Score = product/synonym token overlap, with a small bonus when the supplier's
    country is in the requested scope. This is deliberately simple and explainable —
    it is the "structured filter" arm, not the smart-ranking arm.
    """
    want_terms = _term_set([spec.product, *spec.synonyms])
    scope = set(spec.location)

    scored: list[tuple[str, float]] = []
    for s in suppliers:
        have_terms = _term_set(s.products)
        overlap = len(want_terms & have_terms)
        if overlap == 0:
            continue  # structured arm only surfaces attribute matches
        # Normalize overlap by the query breadth, then add a location affinity bonus.
        score = overlap / max(1, len(want_terms))
        if scope and s.country in scope:
            score += 0.25
        scored.append((s.id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def semantic_retrieve(suppliers: list[Supplier], spec: QuerySpec) -> list[tuple[str, float]]:
    """TF-IDF semantic-ish retrieval over products + description.

    Builds a TF-IDF matrix over every supplier's "products + description" text and
    ranks them by cosine similarity to the query "product + synonyms". No model
    download — scikit-learn computes everything locally. This surfaces near-misses
    (e.g. a solar supplier mentioning "power") that the structured arm rejects, which
    is precisely the recall the gate later has to discipline.
    """
    docs = [_supplier_text(s) for s in suppliers]
    query = " ".join([spec.product, *spec.synonyms])

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(docs + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()

    scored = [(suppliers[i].id, float(sims[i])) for i in range(len(suppliers)) if sims[i] > 0.0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _term_set(items: list[str]) -> set[str]:
    """Normalize a list of product phrases into a lowercase phrase set."""
    return {it.strip().lower() for it in items if it.strip()}


def _supplier_text(s: Supplier) -> str:
    return " ".join(s.products) + " " + s.description
