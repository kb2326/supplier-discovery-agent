"""Stage 1 — query understanding: free text -> QuerySpec.

Maps to the "query understanding" box in the diagram. Two paths, same output shape:

  * LLM path: if ANTHROPIC_API_KEY is set, call Claude with a tool/function schema
    to extract the spec. This is the only place the prototype touches a network.
  * Deterministic fallback: a simple, well-commented keyword parser that correctly
    handles the demo queries with NO API key and NO network. This is the DEFAULT
    path and guarantees the demo runs fully offline.

Any failure in the LLM path (no key, no network, bad response) silently falls back.
"""
from __future__ import annotations

import os

from .schemas import DEFAULT_WEIGHTS, QuerySpec

# EU member-state ISO-2 codes. "EU" / "europe" in a query expands to this set so the
# structured retriever and the location gate can reason over concrete countries.
EU_COUNTRIES = [
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK",
    "SI", "ES", "SE",
]

# Map free-text country mentions to ISO-2 codes for the fallback parser.
_COUNTRY_ALIASES = {
    "germany": "DE", "german": "DE",
    "france": "FR", "french": "FR",
    "spain": "ES", "spanish": "ES",
    "italy": "IT", "italian": "IT",
    "poland": "PL", "polish": "PL",
    "netherlands": "NL", "dutch": "NL",
    "sweden": "SE", "swedish": "SE",
    "switzerland": "CH", "swiss": "CH",
    "china": "CN", "chinese": "CN",
    "usa": "US", "united states": "US", "america": "US",
    "uk": "GB", "united kingdom": "GB", "britain": "GB",
}

# Product synonyms used by both retrievers. Keeps "copper cable" recall high without
# needing a real embedding model.
_PRODUCT_SYNONYMS = {
    "copper cable": [
        "stranded copper conductor", "power cable", "copper wire",
        "control cable", "bare copper conductor", "medium voltage cable",
        "copper conductor",
    ],
}


def parse_query(text: str) -> QuerySpec:
    """Public entry point. Try the LLM, fall back to the deterministic parser."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        spec = _parse_with_llm(text)
        if spec is not None:
            return spec
    return _parse_fallback(text)


# --------------------------------------------------------------------------- #
# Deterministic keyword parser (offline default).
# --------------------------------------------------------------------------- #
def _parse_fallback(text: str) -> QuerySpec:
    low = text.lower()

    # --- product + synonyms ---
    product = "copper cable"  # demo domain default
    synonyms: list[str] = []
    for key, syns in _PRODUCT_SYNONYMS.items():
        if key in low or all(w in low for w in key.split()):
            product = key
            synonyms = list(syns)
            break

    # --- location scope ---
    location: list[str] = []
    if "eu" in _tokens(low) or "europe" in low or "european union" in low:
        location = list(EU_COUNTRIES)
    for alias, code in _COUNTRY_ALIASES.items():
        if alias in low and code not in location:
            location.append(code)

    # --- required certifications ---
    required_certs: list[str] = []
    if "iso 9001" in low or "iso9001" in low:
        required_certs.append("ISO 9001")
    if "iso 14001" in low or "iso14001" in low:
        required_certs.append("ISO 14001")

    # --- track record (Fortune 500) ---
    track_record_required = "fortune 500" in low or "fortune500" in low

    # --- compliance gate ---
    compliance_must_pass = (
        "no compliance issues" in low
        or "compliant" in low
        or "no sanctions" in low
        or "sanction" in low
    )

    return QuerySpec(
        product=product,
        synonyms=synonyms,
        location=location,
        required_certs=required_certs,
        track_record_required=track_record_required,
        compliance_must_pass=compliance_must_pass,
        weights=dict(DEFAULT_WEIGHTS),
        parsed_by="fallback",
    )


def _tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", text))


# --------------------------------------------------------------------------- #
# LLM parser (optional; never required for the demo).
# --------------------------------------------------------------------------- #
# Tool schema handed to Claude so it returns a structured QuerySpec, not prose.
_QUERY_TOOL = {
    "name": "build_query_spec",
    "description": "Extract a structured supplier-search spec from the buyer's request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "product": {"type": "string", "description": "Primary product sought."},
            "synonyms": {"type": "array", "items": {"type": "string"}},
            "location": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ISO-2 country codes. Expand 'EU'/'Europe' to all member states.",
            },
            "required_certs": {"type": "array", "items": {"type": "string"}},
            "track_record_required": {"type": "boolean"},
            "compliance_must_pass": {"type": "boolean"},
        },
        "required": ["product"],
    },
}


def _parse_with_llm(text: str) -> QuerySpec | None:
    """Call Claude to fill the tool schema. Returns None on any failure -> fallback."""
    try:
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",  # cheap + fast; parsing is a light task
            max_tokens=512,
            tools=[_QUERY_TOOL],
            tool_choice={"type": "tool", "name": "build_query_spec"},
            messages=[{
                "role": "user",
                "content": (
                    "Extract a supplier-search spec from this procurement request. "
                    "Expand 'EU' to ISO-2 member-state codes.\n\nRequest: " + text
                ),
            }],
        )
        data = None
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                data = block.input
                break
        if not data:
            return None

        # Normalize: ensure EU expansion + synonyms even if the model was terse.
        location = [c.upper() for c in data.get("location", [])]
        if any(c in ("EU", "EUROPE") for c in location):
            location = [c for c in location if c not in ("EU", "EUROPE")] + EU_COUNTRIES
        synonyms = data.get("synonyms", [])
        product = data.get("product", "copper cable")
        if not synonyms and product in _PRODUCT_SYNONYMS:
            synonyms = list(_PRODUCT_SYNONYMS[product])

        return QuerySpec(
            product=product,
            synonyms=synonyms,
            location=location,
            required_certs=data.get("required_certs", []),
            track_record_required=bool(data.get("track_record_required", False)),
            compliance_must_pass=bool(data.get("compliance_must_pass", False)),
            weights=dict(DEFAULT_WEIGHTS),
            parsed_by="llm",
        )
    except Exception:
        # No key, no network, SDK/response issue — fall back deterministically.
        return None
