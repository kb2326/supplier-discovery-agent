"""Stage 4 — the "rank" layer: hard-filter gate + weighted MCDA + reranker stub.

Maps to the "filter + score + rerank" boxes in the diagram.

  1. hard_filter_gate  — drops any candidate failing a MUST-HAVE (location out of
     scope, a required certification not present, or a sanctions hit). Every removal
     records a human-readable drop_reason so the trace can show *why*.
  2. score_mcda        — weighted Multi-Criteria Decision Analysis. Each criterion is
     normalized to 0-1, multiplied by a configurable weight, and summed into a
     composite. The per-criterion contributions are kept so the UI can explain the
     ranking. This stage also pulls the verification signal from verify.py, which is
     how the seeded expired-cert supplier gets DEMOTED (its compliance score drops).
  3. rerank            — a clearly-labeled NO-OP. A cross-encoder or LLM reranker
     plugs in here in production; we keep MCDA order for the demo.
"""
from __future__ import annotations

from . import verify
from .schemas import Candidate, QuerySpec, Supplier

# Mock geographic proximity (0-1) relative to an assumed EU buyer. Stands in for a real
# distance / logistics-cost model. EU is closest; the further out, the lower the score.
_PROXIMITY = {
    # EU member states
    "AT": 0.95, "BE": 0.95, "CZ": 0.93, "DE": 0.97, "ES": 0.88, "FI": 0.85,
    "FR": 0.95, "GR": 0.82, "HU": 0.9, "IT": 0.9, "LT": 0.86, "NL": 0.96,
    "PL": 0.92, "PT": 0.84, "RO": 0.85, "SE": 0.88, "SK": 0.9,
    # near-EU / rest of world
    "CH": 0.78, "GB": 0.7, "TR": 0.6, "US": 0.4, "CN": 0.3, "IN": 0.3,
}


def hard_filter_gate(
    candidates: list[Candidate], spec: QuerySpec
) -> tuple[list[Candidate], list[Candidate]]:
    """Apply must-have filters. Returns (survivors, dropped).

    Dropped candidates carry a populated `drop_reason`. Order of checks is
    deliberate: location, then required certs, then sanctions — each is reported with
    the specific failing value so the demo can show the gate's reasoning.
    """
    survivors: list[Candidate] = []
    dropped: list[Candidate] = []
    scope = set(spec.location)

    for c in candidates:
        s = c.supplier
        reason = None

        # 1) Location must be in the requested scope.
        if scope and s.country not in scope:
            reason = f"location '{s.country}' not in requested scope"

        # 2) Every required certification must be present (claimed). Validity is
        #    checked later at verify; absence is a hard fail here.
        if reason is None and spec.required_certs:
            have = {cert.type.lower() for cert in s.certifications}
            missing = [rc for rc in spec.required_certs if rc.lower() not in have]
            if missing:
                reason = f"missing required certification: {', '.join(missing)}"

        # 3) Sanctions screening — a positive hit is always disqualifying.
        if reason is None and verify.sanctions_hit(s):
            reason = verify.sanctions_reason(s)

        if reason is None:
            survivors.append(c)
        else:
            c.drop_reason = reason
            dropped.append(c)

    return survivors, dropped


def score_mcda(candidates: list[Candidate], spec: QuerySpec) -> list[Candidate]:
    """Weighted MCDA scoring. Mutates and returns the candidates, sorted by composite.

    Also attaches each supplier's verification objects + flags (the verify stage),
    because the compliance criterion is derived from them — this is what demotes the
    expired-cert supplier even though it passed the gate.
    """
    weights = spec.weights or {}
    weight_total = sum(weights.values()) or 1.0

    for c in candidates:
        s = c.supplier

        # Verify the supplier's claims (verify-layer call). Yields compliance signal,
        # the per-claim Verification objects, and human-readable flags.
        verifications, compliance, flags = verify.verify_supplier(s)
        c.verifications = verifications
        c.flags = flags

        criteria = {
            "product_match": round(c.product_match, 4),
            "track_record": _track_record(s),
            "proximity": _PROXIMITY.get(s.country, 0.5),
            "financial_health": s.financial_health,
            "esg_score": s.esg_score,
            "compliance": round(compliance, 4),
        }
        c.criteria = criteria

        # Composite = weighted sum, normalized by the total weight so it stays in 0-1.
        contributions = {k: round(weights.get(k, 0.0) * v, 4) for k, v in criteria.items()}
        c.contributions = contributions
        c.composite = round(sum(contributions.values()) / weight_total, 4)

    candidates.sort(key=lambda x: x.composite, reverse=True)
    return candidates


def rerank(candidates: list[Candidate]) -> list[Candidate]:
    """Reranker — INTENTIONAL NO-OP for the demo.

    A production system plugs a cross-encoder or LLM reranker in here to reorder the
    top of the MCDA list using richer signals. We deliberately keep MCDA order so the
    ranking stays fully explainable from the per-criterion breakdown.
    """
    return candidates


# --------------------------------------------------------------------------- #
# Criterion helpers.
# --------------------------------------------------------------------------- #
def _track_record(s: Supplier) -> float:
    """Normalize track record from Fortune 500 customer count + years active (0-1)."""
    f500 = 0.25 * len(s.fortune500_customers)
    tenure = min(1.0, s.years_active / 40.0)
    return round(min(1.0, 0.6 * f500 + 0.4 * tenure), 4)
