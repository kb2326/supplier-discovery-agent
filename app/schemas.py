"""Pydantic models shared across the pipeline.

These are the data contracts that flow between stages. Keeping them in one place
makes the find -> rank -> verify pipeline easy to read: every stage takes and
returns one of these shapes.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# Default MCDA criterion weights. The UI sliders override these per request.
# They do not need to sum to 1 — rank.py normalizes the composite by the weight total.
DEFAULT_WEIGHTS: dict[str, float] = {
    "product_match": 0.35,   # how well the supplier matches the requested product (from fusion)
    "track_record": 0.20,    # Fortune 500 customer count + years_active
    "proximity": 0.10,       # mock geographic proximity by country
    "financial_health": 0.15,
    "esg_score": 0.10,
    "compliance": 0.10,      # verified certs / no flags (filled from verify.py signals)
}


# --------------------------------------------------------------------------- #
# Stage 1 output: the structured intent extracted from the free-text query.
# --------------------------------------------------------------------------- #
class QuerySpec(BaseModel):
    product: str = Field(..., description="Primary product the buyer is looking for.")
    synonyms: list[str] = Field(default_factory=list)
    location: list[str] = Field(
        default_factory=list,
        description="ISO-2 country codes in scope (EU expands to member states).",
    )
    required_certs: list[str] = Field(default_factory=list)
    track_record_required: bool = False
    compliance_must_pass: bool = False
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    parsed_by: str = "fallback"  # "llm" or "fallback" — surfaced in the UI for transparency.


# --------------------------------------------------------------------------- #
# Data layer: a resolved ("golden") supplier record.
# --------------------------------------------------------------------------- #
class Certification(BaseModel):
    type: str
    number: str
    claimed_status: str = "active"


class Supplier(BaseModel):
    id: str
    name: str
    domain: Optional[str] = None
    tax_id: Optional[str] = None
    country: str
    products: list[str] = Field(default_factory=list)
    description: str = ""
    certifications: list[Certification] = Field(default_factory=list)
    fortune500_customers: list[str] = Field(default_factory=list)
    financial_health: float = 0.0
    esg_score: float = 0.0
    years_active: int = 0
    last_updated: Optional[str] = None
    # Entity-resolution provenance: ids of duplicate records merged into this one.
    merged_from: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Verify layer: one verification object per claim, imitating an authoritative source.
# --------------------------------------------------------------------------- #
class Verification(BaseModel):
    claim: str
    status: str                       # "verified" | "unverified" | "expired"
    source: str                       # e.g. "IAF CertSearch (mock)"
    source_url: str
    snippet: str
    confidence: float
    last_verified: Optional[str] = None
    stale: bool = False               # True when last_verified is older than the freshness threshold.


# --------------------------------------------------------------------------- #
# A candidate flowing through retrieval -> fusion -> ranking.
# --------------------------------------------------------------------------- #
class Candidate(BaseModel):
    supplier: Supplier
    # Per-retriever ranks (1-based); None if a retriever did not surface this supplier.
    structured_rank: Optional[int] = None
    semantic_rank: Optional[int] = None
    fused_score: float = 0.0          # reciprocal-rank-fusion score
    product_match: float = 0.0        # normalized 0-1 product-match strength (from fusion)
    # MCDA output, filled in rank.py:
    criteria: dict[str, float] = Field(default_factory=dict)        # per-criterion normalized value
    contributions: dict[str, float] = Field(default_factory=dict)   # weight * value per criterion
    composite: float = 0.0
    # Verify output (only populated for survivors / top results):
    verifications: list[Verification] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    # Set when the candidate was removed at the hard-filter gate:
    drop_reason: Optional[str] = None


# --------------------------------------------------------------------------- #
# Trace + Result: the observable record of the whole run (the demo centerpiece).
# --------------------------------------------------------------------------- #
class DroppedSupplier(BaseModel):
    id: str
    name: str
    reason: str


class MergeReport(BaseModel):
    golden_id: str
    golden_name: str
    merged_ids: list[str]
    reason: str


class StageCount(BaseModel):
    stage: str
    count_in: int
    count_out: int
    note: str = ""


class Trace(BaseModel):
    query_text: str
    spec: QuerySpec
    stages: list[StageCount] = Field(default_factory=list)
    after_retrieval: int = 0
    after_fusion: int = 0
    after_gate: int = 0
    dropped: list[DroppedSupplier] = Field(default_factory=list)
    merges: list[MergeReport] = Field(default_factory=list)
    final_ids: list[str] = Field(default_factory=list)


class Result(BaseModel):
    top: list[Candidate] = Field(default_factory=list)
    trace: Trace
