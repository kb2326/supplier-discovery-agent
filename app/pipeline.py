"""Orchestration — run the stages in order and build the observable Trace.

This is the spine of the demo. It wires the modules together exactly in the order of
the architecture diagram and records, at every step, how many candidates went in and
came out, who was dropped and why, and which duplicates entity resolution collapsed.

    understand -> retrieve (x2) -> fuse -> gate -> score (MCDA + verify) -> rerank -> top-5

The Trace it returns is the centerpiece the frontend renders.
"""
from __future__ import annotations

from . import fuse, rank, retrieve, understand
from .schemas import (
    Candidate,
    DroppedSupplier,
    QuerySpec,
    Result,
    StageCount,
    Trace,
)
from .store import SupplierStore

TOP_N = 5


def run_search(
    query_text: str,
    store: SupplierStore,
    weights: dict[str, float] | None = None,
) -> Result:
    """Execute the full find -> rank -> verify pipeline for one query."""
    suppliers = store.all()

    # --- Stage 1: understand ---------------------------------------------------
    spec: QuerySpec = understand.parse_query(query_text)
    if weights:  # UI slider overrides
        spec.weights = {**spec.weights, **weights}

    trace = Trace(query_text=query_text, spec=spec)
    # Entity-resolution report comes from the store (data layer), shown in the trace.
    trace.merges = store.merges

    # --- Stage 2: retrieve (the "find" layer, two arms) ------------------------
    structured = retrieve.structured_retrieve(suppliers, spec)
    semantic = retrieve.semantic_retrieve(suppliers, spec)
    trace.stages.append(StageCount(
        stage="retrieve",
        count_in=len(suppliers),
        count_out=len({sid for sid, _ in structured} | {sid for sid, _ in semantic}),
        note=f"structured: {len(structured)} | semantic: {len(semantic)}",
    ))

    # --- Stage 3: fuse (recall) ------------------------------------------------
    candidates: list[Candidate] = fuse.reciprocal_rank_fusion(suppliers, structured, semantic)
    trace.after_retrieval = trace.stages[-1].count_out
    trace.after_fusion = len(candidates)
    trace.stages.append(StageCount(
        stage="fuse",
        count_in=trace.after_retrieval,
        count_out=len(candidates),
        note="reciprocal rank fusion (k=60)",
    ))

    # --- Stage 4a: hard-filter gate -------------------------------------------
    survivors, dropped = rank.hard_filter_gate(candidates, spec)
    trace.after_gate = len(survivors)
    trace.dropped = [
        DroppedSupplier(id=c.supplier.id, name=c.supplier.name, reason=c.drop_reason or "")
        for c in dropped
    ]
    trace.stages.append(StageCount(
        stage="filter",
        count_in=len(candidates),
        count_out=len(survivors),
        note=f"dropped {len(dropped)} on must-have filters",
    ))

    # --- Stage 4b: score (MCDA) + Stage 5: verify ------------------------------
    scored = rank.score_mcda(survivors, spec)
    trace.stages.append(StageCount(
        stage="score+verify",
        count_in=len(survivors),
        count_out=len(scored),
        note="weighted MCDA; per-claim verification attached",
    ))

    # --- Reranker (no-op stub) + top-N ----------------------------------------
    ranked = rank.rerank(scored)
    top = ranked[:TOP_N]
    for c in top:
        c.rationale = _rationale(c)
    trace.final_ids = [c.supplier.id for c in top]
    trace.stages.append(StageCount(
        stage="rerank+top",
        count_in=len(scored),
        count_out=len(top),
        note="reranker is a no-op stub; taking top 5",
    ))

    return Result(top=top, trace=trace)


def _rationale(c: Candidate) -> str:
    """One-sentence, plain-English rationale built ONLY from real record fields.

    No LLM and — importantly for the faithfulness eval — no invented attributes.
    Every clause maps to a field or a verification the card also displays.
    """
    s = c.supplier
    parts: list[str] = []

    # Product + location.
    primary_product = s.products[0] if s.products else "the requested product"
    parts.append(f"{s.name} supplies {primary_product} from {s.country}")

    # Track record (only assert verified F500 as fact; flag the rest).
    verified_customers = [
        v.claim.split(": ", 1)[1]
        for v in c.verifications
        if v.claim.startswith("Supplies Fortune 500") and v.status == "verified"
    ]
    if verified_customers:
        parts.append(f"with a verified Fortune 500 relationship ({', '.join(verified_customers)})")
    parts.append(f"and {s.years_active} years active")

    # Compliance posture, drawn straight from the verification flags.
    if any(f.startswith("expired") for f in c.flags):
        parts.append("but its certification is flagged expired")
    elif any("stale" in f for f in c.flags):
        parts.append("though its certification check is stale")

    return ", ".join(parts) + "."
