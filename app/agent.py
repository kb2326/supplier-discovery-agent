"""
LLM orchestrator — Claude with tool use routes queries across 4 data stores.
Falls back to a deterministic keyword router when ANTHROPIC_API_KEY is absent.
"""
from __future__ import annotations
import json, os
from .tools import query_sql, search_vector, traverse_graph, call_external_api

# ── Tool schemas handed to Claude ─────────────────────────────────────────
TOOLS = [
    {
        "name": "query_sql",
        "description": (
            "Query the INTERNAL STRUCTURED DATABASE using exact filters. "
            "Use for: country, ISO 9001/14001 certification, employee count, "
            "ESG score, financial health, revenue. Best when the user specifies "
            "hard criteria like 'suppliers in Germany with ISO 9001' or "
            "'more than 500 employees'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "ISO-2 codes comma-separated e.g. 'DE,FR'"},
                "iso_9001":  {"type": "boolean"},
                "iso_14001": {"type": "boolean"},
                "min_employees": {"type": "integer"},
                "max_employees": {"type": "integer"},
                "min_esg":  {"type": "number", "description": "0-1 scale"},
                "min_financial_health": {"type": "number", "description": "0-1 scale"},
                "limit": {"type": "integer", "default": 8},
            },
        },
    },
    {
        "name": "search_vector",
        "description": (
            "Semantic search over CAPABILITY DESCRIPTIONS. "
            "Use for fuzzy product/capability matching where exact keywords vary — "
            "e.g. 'offshore wind cable systems', 'high-voltage rail traction', "
            "'renewable energy copper conductors', 'automotive shielded harness'. "
            "Good when the user describes a capability rather than a hard filter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Capability description in natural language"},
                "top_k": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "traverse_graph",
        "description": (
            "Traverse the SUPPLIER KNOWLEDGE GRAPH for relationship questions. "
            "Use for: 'which suppliers already work with Siemens / BASF / Shell', "
            "'who owns this supplier', 'are any suppliers connected to sanctioned entities', "
            "'show ownership chains', 'find risk connections'. "
            "query_type must be one of: suppliers_to | customers_of | ownership | risk_connections"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["suppliers_to", "customers_of", "ownership", "risk_connections"],
                    "description": "suppliers_to=find suppliers that supply a named company; customers_of=find customers of a named supplier; ownership=show parent company chains; risk_connections=find suppliers linked to sanctioned entities",
                },
                "entity_name": {
                    "type": "string",
                    "description": "Name of the company to traverse from (e.g. 'Siemens', 'BASF', 'GlobalCopper Holdings')",
                },
            },
            "required": ["query_type"],
        },
    },
    {
        "name": "call_external_api",
        "description": (
            "Call EXTERNAL LIVE SUPPLIER DATA API (simulates Veridion / D&B). "
            "Returns suppliers NOT in the internal database — fresh market data. "
            "Use for: finding new/emerging suppliers, broadening discovery beyond the "
            "internal database, or when the user explicitly asks for external sources."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query for the external API"},
                "country_filter": {"type": "string", "description": "ISO-2 codes comma-separated"},
                "iso_9001": {"type": "boolean"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
]

SYSTEM = """You are a procurement AI assistant. You help buyers find and evaluate suppliers.

You have four data sources — choose based on what the query needs:

• query_sql       → hard structured filters (country, ISO certs, employee count, ESG, financials)
• search_vector   → fuzzy capability/product matching (semantic, handles synonym variation)
• traverse_graph  → relationship questions (who supplies whom, ownership, sanctions risk)
• call_external_api → external live data for new/emerging suppliers not in internal DB

Rules:
- Use multiple tools when the query has multiple dimensions (e.g. country filter + capability match).
- Always state which source each result came from.
- Flag any risk signals clearly (sanctions connections, low financials, etc.).
- Combine and deduplicate results across sources when calling multiple tools.
- Be concise. Format results as a numbered list with key attributes.
- End with a one-sentence summary of what sources were used and why.
"""


# ── Main entry ────────────────────────────────────────────────────────────
def run_agent(message: str, history: list[dict]) -> dict:
    """
    Route the message through the appropriate data sources.
    Returns {response, sources_used, tool_calls}.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return _llm_agent(message, history, key)
    return _fallback_router(message)


# ── LLM path (Claude with tool use) ──────────────────────────────────────
def _llm_agent(message: str, history: list[dict], key: str) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    messages = list(history) + [{"role": "user", "content": message}]
    sources, calls = [], []

    _TOOL_FNS = {
        "query_sql": query_sql,
        "search_vector": search_vector,
        "traverse_graph": traverse_graph,
        "call_external_api": call_external_api,
    }
    _SOURCE_LABELS = {
        "query_sql": "SQL", "search_vector": "Vector",
        "traverse_graph": "Graph", "call_external_api": "External API",
    }

    while True:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )

        if resp.stop_reason == "end_turn":
            text = next((b.text for b in resp.content if hasattr(b, "text")), "")
            return {"response": text, "sources_used": list(dict.fromkeys(sources)), "tool_calls": calls}

        # Execute tool calls
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            fn = _TOOL_FNS.get(block.name)
            try:
                result = fn(**block.input) if fn else {"error": "unknown tool"}
            except Exception as e:
                result = {"error": str(e)}

            label = _SOURCE_LABELS.get(block.name, block.name)
            sources.append(label)
            calls.append({
                "tool": block.name,
                "label": label,
                "input_summary": _summarise(block.input),
                "result_count": len(result) if isinstance(result, list) else 1,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, default=str),
            })

        messages += [
            {"role": "assistant", "content": resp.content},
            {"role": "user", "content": tool_results},
        ]


def _summarise(inp: dict) -> str:
    parts = []
    for k, v in inp.items():
        if k == "limit": continue
        parts.append(f"{k}={v}")
    return ", ".join(parts) if parts else "—"


# ── Fallback router (no API key) ──────────────────────────────────────────
_CUSTOMER_NAMES = ["Siemens","BASF","Shell","Airbus","Volvo","BP","Ford","Apple",
                   "Ericsson","Stellantis","Telefonica","JPMorgan","ABB"]
_EU_MAP = {
    "germany":"DE","france":"FR","netherlands":"NL","switzerland":"CH",
    "sweden":"SE","italy":"IT","spain":"ES","poland":"PL","austria":"AT",
    "finland":"FI","belgium":"BE","czech":"CZ","denmark":"DK","portugal":"PT",
    "romania":"RO","hungary":"HU","slovakia":"SK","greece":"GR","lithuania":"LT",
    "uk":"GB","britain":"GB",
}

def _fallback_router(message: str) -> dict:
    low = message.lower()
    sources, calls, all_results = [], [], []
    seen_ids: set = set()

    def _add(items: list[dict], label: str) -> None:
        for item in items:
            iid = item.get("id") or item.get("supplier_id") or item.get("name","?")
            if iid not in seen_ids:
                seen_ids.add(iid)
                all_results.append(item)
        sources.append(label)
        calls.append({"tool": label.lower().replace(" ","_"), "label": label,
                      "input_summary": "keyword-based", "result_count": len(items)})

    # 1. Graph — relationship / risk signals
    cust_hit = next((c for c in _CUSTOMER_NAMES if c.lower() in low), None)
    if cust_hit:
        _add(traverse_graph("suppliers_to", entity_name=cust_hit), "Graph")
    if any(w in low for w in ["sanction","risk","owned","ownership","parent","connected"]):
        qt = "risk_connections" if any(w in low for w in ["sanction","risk"]) else "ownership"
        _add(traverse_graph(qt), "Graph")

    # 2. SQL — structured filters
    countries = [code for alias, code in _EU_MAP.items() if alias in low]
    if "eu" in low.split() or "europe" in low:
        countries = list(_EU_MAP.values())
    need_sql = countries or any(w in low for w in [
        "iso","certif","employees","revenue","esg","financial","health"])
    if need_sql:
        _add(query_sql(
            country=",".join(dict.fromkeys(countries)) or None,
            iso_9001=True if "iso 9001" in low or "iso9001" in low else None,
            iso_14001=True if "iso 14001" in low else None,
            min_employees=500 if "500 employee" in low or "> 500" in low else None,
            min_esg=0.75 if "esg" in low and ("high" in low or "good" in low) else None,
            limit=8,
        ), "SQL")

    # 3. Vector — capability / product description
    if any(w in low for w in ["cable","conductor","copper","insulation","voltage","rail",
                               "renewable","offshore","wind","solar","automotive",
                               "aerospace","capability","speciali"]):
        _add(search_vector(message, top_k=6), "Vector")

    # 4. External API — new/emerging or fallback
    if any(w in low for w in ["new","emerging","external","discover","startup","additional"]) \
            or not sources:
        _add(call_external_api(message, limit=5), "External API")

    # Build plain-text response
    if not all_results:
        response = (
            "No suppliers matched that query. Try:\n"
            "• 'ISO 9001 certified copper cable suppliers in Germany'\n"
            "• 'suppliers that work with Siemens'\n"
            "• 'find emerging copper cable suppliers in Eastern Europe'\n"
            "• 'any suppliers with sanctions risk connections?'"
        )
    else:
        lines = [f"Found **{len(all_results)}** suppliers via: **{', '.join(sources)}**\n"]
        for i, s in enumerate(all_results[:10], 1):
            name = s.get("name","?")
            country = s.get("country","")
            rel  = s.get("relationship") or s.get("risk_flag","")
            src  = s.get("source","")
            extra = f" · {rel}" if rel else ""
            lines.append(f"{i}. **{name}** ({country}){extra}")
            lines.append(f"   *{src}*")
        lines.append(f"\n*No ANTHROPIC_API_KEY set — using keyword router. "
                     f"Set the key for full LLM-orchestrated routing.*")
        response = "\n".join(lines)

    return {"response": response, "sources_used": list(dict.fromkeys(sources)), "tool_calls": calls}
