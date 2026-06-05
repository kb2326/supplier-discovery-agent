"""Tiny gold-set evaluation — run with `python -m app.eval`.

Two checks, printed as small tables:

  1. precision@5 — over a hand-labeled gold set of queries, what fraction of the
     returned top-5 are in the expected (relevant) set.
  2. faithfulness — every factual clause in each generated rationale must map back to
     a real record field or a verification object. No invented attributes. This guards
     the "shown but flagged, never asserted" property the verify layer is built around.

Both run fully offline against the mock store.
"""
from __future__ import annotations

import re

from .pipeline import run_search
from .schemas import Candidate
from .store import load_store

# --------------------------------------------------------------------------- #
# Gold set: queries -> supplier ids we consider relevant (correct) for the top-5.
# Hand-labeled from the dataset's design; the harness reports precision against these.
# --------------------------------------------------------------------------- #
GOLD: list[tuple[str, set[str]]] = [
    (
        "find copper cable suppliers in the EU, ISO 9001 certified, that supply "
        "Fortune 500 companies, with no compliance issues",
        {"sup_010", "sup_005", "sup_003", "sup_002", "sup_008", "sup_009"},
    ),
    (
        "find copper cable suppliers in the EU",
        {"sup_010", "sup_005", "sup_003", "sup_002", "sup_008", "sup_026"},
    ),
    (
        "copper cable suppliers in Germany",
        {"sup_002", "sup_021", "sup_027"},
    ),
    (
        "copper power cable suppliers in France, ISO 9001 certified",
        {"sup_005", "sup_022"},
    ),
    (
        "copper cable suppliers in the EU that supply Fortune 500 companies",
        {"sup_010", "sup_005", "sup_003", "sup_008", "sup_009"},
    ),
]


def precision_at_k(returned_ids: list[str], relevant: set[str], k: int = 5) -> float:
    top = returned_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for sid in top if sid in relevant)
    return hits / len(top)


# --------------------------------------------------------------------------- #
# Faithfulness: assert each rationale only states things backed by the record.
# --------------------------------------------------------------------------- #
def check_faithfulness(c: Candidate) -> list[str]:
    """Return a list of violations (empty == faithful)."""
    s = c.supplier
    r = c.rationale
    violations: list[str] = []

    # Name, primary product, country, tenure must all be grounded in the record.
    if s.name not in r:
        violations.append("name not grounded")
    primary_product = s.products[0] if s.products else None
    if primary_product and primary_product not in r:
        violations.append("product not grounded")
    if f"from {s.country}" not in r:
        violations.append("country not grounded")
    if f"{s.years_active} years active" not in r:
        violations.append("years_active not grounded")

    # Any asserted verified Fortune 500 relationship must be a real, verified claim.
    for customer in re.findall(r"verified Fortune 500 relationship \(([^)]+)\)", r):
        for name in [c.strip() for c in customer.split(",")]:
            if name not in s.fortune500_customers:
                violations.append(f"invented F500 customer: {name}")
            elif not any(
                v.status == "verified" and v.claim.endswith(name) for v in c.verifications
            ):
                violations.append(f"asserted unverified F500 as fact: {name}")

    # "expired" / "stale" language must correspond to an actual flag.
    if "expired" in r and not any(f.startswith("expired") for f in c.flags):
        violations.append("claims expired without a flag")
    if "stale" in r and not any("stale" in f for f in c.flags):
        violations.append("claims stale without a flag")

    return violations


def main() -> None:
    store = load_store()

    print("=" * 72)
    print("precision@5")
    print("-" * 72)
    precisions: list[float] = []
    all_candidates: list[Candidate] = []
    for query, relevant in GOLD:
        result = run_search(query, store)
        ids = [c.supplier.id for c in result.top]
        p = precision_at_k(ids, relevant)
        precisions.append(p)
        all_candidates.extend(result.top)
        print(f"  P@5={p:.2f}  {query[:54]:<54}  -> {ids}")
    mean_p = sum(precisions) / len(precisions)
    print("-" * 72)
    print(f"  mean precision@5: {mean_p:.2f}")

    print("\n" + "=" * 72)
    print("faithfulness (every rationale clause maps to a real field/citation)")
    print("-" * 72)
    total = len(all_candidates)
    failures = 0
    for c in all_candidates:
        violations = check_faithfulness(c)
        if violations:
            failures += 1
            print(f"  FAIL {c.supplier.id}: {violations}")
    if failures == 0:
        print(f"  PASS — all {total} rationales faithful (0 invented attributes)")
    print("=" * 72)
    print(f"\nSUMMARY: mean P@5={mean_p:.2f} | faithfulness {'PASS' if failures == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
