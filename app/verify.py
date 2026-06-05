"""Stage 5 — MOCK verification & citation (the "verify" layer).

Maps to the "verification & citation" box in the diagram. In production each of these
would be a live call to an authoritative registry (IAF CertSearch for certificates,
OFAC for sanctions, GLEIF for legal-entity identity, plus customer-reference checks).
Here they are LOCAL LOOKUP TABLES that imitate those sources so the demo runs offline
and deterministically.

Every claim a supplier makes is turned into a Verification object carrying a status,
a (mock) source + URL, a snippet, a confidence and a last_verified date. The ranking
stage consumes a compliance signal from here; the UI renders the badges and citations.

Two functions are exposed:
  * sanctions_hit(supplier)  -> used by the hard-filter gate to DROP sanctioned entities.
  * verify_supplier(...)     -> full per-claim verification + a compliance score + flags.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

from .schemas import Supplier, Verification

# --------------------------------------------------------------------------- #
# Mock authoritative-source tables (the seeded narrative cases live here).
# --------------------------------------------------------------------------- #

# OFAC (mock): supplier ids that hit the sanctions list. Seeded case (3).
_SANCTIONS_LIST = {"sup_007"}

# IAF CertSearch (mock): (supplier_id, cert_type) whose certificate is actually EXPIRED,
# even though the supplier claims it active. Seeded case (2).
_EXPIRED_CERTS = {("sup_002", "ISO 9001")}

# IAF CertSearch (mock): when each certificate was last verified at the registry.
# Anything older than FRESHNESS_DAYS is flagged stale. sup_009 is seeded stale.
_CERT_LAST_VERIFIED = {
    "sup_009": "2024-11-20",  # > 1 year before the demo's "today" -> stale badge
}
_DEFAULT_CERT_LAST_VERIFIED = "2026-03-15"

# Customer-reference corroboration (mock): self-reported Fortune 500 relationships we
# could actually corroborate. Everything NOT in here comes back "unverified". Seeded
# case (4): most F500 claims (e.g. sup_017 -> Apple) are shown but flagged unverified.
_CORROBORATED_F500 = {
    ("sup_010", "Shell"),
    ("sup_003", "Volvo"),
}

# Freshness threshold: verifications older than this are marked stale (⌛).
FRESHNESS_DAYS = 365

# The demo's reference "today" (the brief's current date) keeps stale logic deterministic.
TODAY = date(2026, 6, 4)


# --------------------------------------------------------------------------- #
# Gate helper: sanctions screening.
# --------------------------------------------------------------------------- #
def sanctions_hit(supplier: Supplier) -> bool:
    """True if the supplier matches the mock OFAC sanctions list."""
    return supplier.id in _SANCTIONS_LIST


def sanctions_reason(supplier: Supplier) -> str:
    return "OFAC sanctions match (mock) — flagged on the consolidated screening list"


# --------------------------------------------------------------------------- #
# Full per-claim verification.
# --------------------------------------------------------------------------- #
def verify_supplier(supplier: Supplier) -> tuple[list[Verification], float, list[str]]:
    """Verify a supplier's claims.

    Returns (verifications, compliance_score, flags):
      * verifications  — one object per claim, for the UI's badges + citations.
      * compliance_score — 0-1 signal fed into the MCDA "compliance" criterion. An
        expired cert sharply demotes a supplier here (this is *the* demotion in the demo);
        stale verification nudges it down a little.
      * flags — short human-readable strings shown on the supplier card.
    """
    verifications: list[Verification] = []
    flags: list[str] = []
    compliance = 1.0

    # 1) Sanctions screening (always clean for anything that survived the gate).
    if sanctions_hit(supplier):
        compliance = 0.0
        flags.append("sanctions match")
        verifications.append(Verification(
            claim="OFAC sanctions screening",
            status="expired",  # treat a positive hit as a hard compliance failure
            source="OFAC (mock)",
            source_url="https://sanctionssearch.ofac.treas.gov/",
            snippet=f"Match found for {supplier.name} on the consolidated sanctions list.",
            confidence=0.97,
            last_verified=TODAY.isoformat(),
        ))
    else:
        verifications.append(Verification(
            claim="OFAC sanctions screening",
            status="verified",
            source="OFAC (mock)",
            source_url="https://sanctionssearch.ofac.treas.gov/",
            snippet=f"No matches found for {supplier.name} on the consolidated sanctions list.",
            confidence=0.95,
            last_verified=TODAY.isoformat(),
        ))

    # 2) Certificates (IAF CertSearch mock). Seeded expired cert demotes the supplier.
    last_verified = _CERT_LAST_VERIFIED.get(supplier.id, _DEFAULT_CERT_LAST_VERIFIED)
    stale = _is_stale(last_verified)
    for cert in supplier.certifications:
        expired = (supplier.id, cert.type) in _EXPIRED_CERTS
        if expired:
            compliance = min(compliance, 0.25)
            flags.append(f"expired {cert.type}")
            verifications.append(Verification(
                claim=f"{cert.type} certification ({cert.number})",
                status="expired",
                source="IAF CertSearch (mock)",
                source_url="https://www.iafcertsearch.org/",
                snippet=f"Certificate {cert.number} for {cert.type} lapsed; registry status: EXPIRED.",
                confidence=0.92,
                last_verified=last_verified,
                stale=stale,
            ))
        else:
            if stale:
                compliance *= 0.9
                flags.append(f"stale {cert.type} check")
            verifications.append(Verification(
                claim=f"{cert.type} certification ({cert.number})",
                status="verified",
                source="IAF CertSearch (mock)",
                source_url="https://www.iafcertsearch.org/",
                snippet=f"Certificate {cert.number} for {cert.type} is active in the registry.",
                confidence=0.9,
                last_verified=last_verified,
                stale=stale,
            ))

    # 3) Fortune 500 customer claims (corroboration mock). Unverified unless allow-listed.
    for customer in supplier.fortune500_customers:
        corroborated = (supplier.id, customer) in _CORROBORATED_F500
        if corroborated:
            verifications.append(Verification(
                claim=f"Supplies Fortune 500 customer: {customer}",
                status="verified",
                source="Customer reference (mock)",
                source_url="https://example.com/refs/" + supplier.id,
                snippet=f"Corroborated supply relationship between {supplier.name} and {customer}.",
                confidence=0.8,
                last_verified="2026-02-10",
            ))
        else:
            flags.append(f"unverified F500 claim: {customer}")
            verifications.append(Verification(
                claim=f"Supplies Fortune 500 customer: {customer}",
                status="unverified",
                source="Customer reference (mock)",
                source_url="https://example.com/refs/" + supplier.id,
                snippet=f"Self-reported relationship with {customer}; no corroborating source found.",
                confidence=0.3,
                last_verified=None,
            ))

    return verifications, compliance, flags


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _is_stale(last_verified: str | None) -> bool:
    if not last_verified:
        return False
    try:
        d = datetime.strptime(last_verified, "%Y-%m-%d").date()
    except ValueError:
        return False
    return (TODAY - d) > timedelta(days=FRESHNESS_DAYS)
