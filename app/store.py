"""Data layer: golden-record store + entity resolution.

Maps to the "golden record store / entity resolution" box in the architecture
diagram. On load we:

  1. Normalize supplier names (strip legal suffixes, lowercase, collapse whitespace).
  2. Resolve duplicates by anchoring on STRONG keys first (domain, tax_id), then
     falling back to fuzzy name+country similarity via rapidfuzz.
  3. Merge matched records into a single golden record, keeping a `merged_from`
     list and producing a small merge report the demo can show.

In production this is where a master-data-management / entity-resolution service
would live; the rest of the pipeline only ever sees the resolved golden store.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rapidfuzz import fuzz

from .schemas import Certification, MergeReport, Supplier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "suppliers.json"

# Legal-form suffixes stripped during name normalization so that
# "Nordic Cable Co" and "Nordic Cable Company Ltd" collapse to the same key-ish form.
_LEGAL_SUFFIXES = [
    "incorporated", "inc", "ltd", "limited", "llc", "co", "company", "corp",
    "corporation", "ag", "gmbh", "sas", "sarl", "sa", "spa", "srl", "bv", "nv",
    "oy", "ab", "as", "zrt", "lda", "uab", "sp z o o", "s r o", "plc", "kg",
]

# Fuzzy name-similarity threshold (0-100). Above this, with same country, we treat
# two records as the same entity even without a shared strong key.
NAME_MATCH_THRESHOLD = 88


def normalize_name(name: str) -> str:
    """Lowercase, drop punctuation + legal suffixes, collapse whitespace."""
    n = name.lower()
    n = re.sub(r"[.,&/]", " ", n)          # punctuation -> space
    n = re.sub(r"\s+", " ", n).strip()
    tokens = [t for t in n.split(" ") if t and t not in _LEGAL_SUFFIXES]
    return " ".join(tokens).strip()


def _load_raw() -> list[dict]:
    with DATA_PATH.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["suppliers"]


def _to_supplier(rec: dict) -> Supplier:
    certs = [Certification(**c) for c in rec.get("certifications", [])]
    return Supplier(
        id=rec["id"],
        name=rec["name"],
        domain=rec.get("domain"),
        tax_id=rec.get("tax_id"),
        country=rec["country"],
        products=rec.get("products", []),
        description=rec.get("description", ""),
        certifications=certs,
        fortune500_customers=rec.get("fortune500_customers", []),
        financial_health=rec.get("financial_health", 0.0),
        esg_score=rec.get("esg_score", 0.0),
        years_active=rec.get("years_active", 0),
        last_updated=rec.get("last_updated"),
    )


def _merge_into(golden: Supplier, dup: Supplier) -> None:
    """Fold a duplicate record into the golden record (union the claim lists)."""
    golden.merged_from.append(dup.id)
    # Union products / fortune500 / certifications, preserving order without dupes.
    golden.products = _dedupe(golden.products + dup.products)
    golden.fortune500_customers = _dedupe(golden.fortune500_customers + dup.fortune500_customers)
    seen = {(c.type, c.number) for c in golden.certifications}
    for c in dup.certifications:
        if (c.type, c.number) not in seen:
            golden.certifications.append(c)
            seen.add((c.type, c.number))
    # Keep the strongest mock scores / most recent freshness across the pair.
    golden.financial_health = max(golden.financial_health, dup.financial_health)
    golden.esg_score = max(golden.esg_score, dup.esg_score)
    golden.years_active = max(golden.years_active, dup.years_active)
    if dup.last_updated and (not golden.last_updated or dup.last_updated > golden.last_updated):
        golden.last_updated = dup.last_updated


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out


class SupplierStore:
    """Holds the resolved golden suppliers plus the entity-resolution merge report."""

    def __init__(self, suppliers: list[Supplier], merges: list[MergeReport]):
        self.suppliers = suppliers
        self.merges = merges
        self._by_id = {s.id: s for s in suppliers}

    def all(self) -> list[Supplier]:
        return self.suppliers

    def get(self, supplier_id: str) -> Supplier | None:
        return self._by_id.get(supplier_id)


def resolve(records: list[Supplier]) -> SupplierStore:
    """Run entity resolution over raw records and return a golden store.

    Strategy (cheap and explainable, which is the point for the demo):
      * Strong keys first: identical non-empty `domain` or `tax_id` is a hard match.
      * Fuzzy fallback: same country AND normalized-name token_sort_ratio >= threshold.
    The first record in a cluster becomes the golden record; the rest fold in.
    """
    golden: list[Supplier] = []
    merges: list[MergeReport] = []

    for rec in records:
        match = None
        match_reason = ""
        norm = normalize_name(rec.name)
        for g in golden:
            # 1) Strong-key anchor.
            if rec.domain and g.domain and rec.domain.lower() == g.domain.lower():
                match, match_reason = g, f"shared domain ({rec.domain})"
                break
            if rec.tax_id and g.tax_id and rec.tax_id.upper() == g.tax_id.upper():
                match, match_reason = g, f"shared tax_id ({rec.tax_id})"
                break
            # 2) Fuzzy name + country fallback.
            if rec.country == g.country:
                score = fuzz.token_sort_ratio(norm, normalize_name(g.name))
                if score >= NAME_MATCH_THRESHOLD:
                    match = g
                    match_reason = f"fuzzy name match {score:.0f}% + same country ({rec.country})"
                    break

        if match is None:
            golden.append(rec)
        else:
            _merge_into(match, rec)
            merges.append(
                MergeReport(
                    golden_id=match.id,
                    golden_name=match.name,
                    merged_ids=[rec.id],
                    reason=match_reason,
                )
            )

    return SupplierStore(golden, merges)


def load_store() -> SupplierStore:
    """Load the JSON dataset and return the resolved golden store."""
    raw = _load_raw()
    suppliers = [_to_supplier(r) for r in raw]
    return resolve(suppliers)


if __name__ == "__main__":  # quick manual check: python -m app.store
    store = load_store()
    print(f"{len(store.all())} golden suppliers after resolution")
    for m in store.merges:
        print(f"  merged {m.merged_ids} -> {m.golden_id} ({m.reason})")
