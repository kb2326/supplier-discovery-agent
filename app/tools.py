"""
Four data-source tools — each returns a list of dicts with a 'source' field.
SQL → hard filters | Vector → semantic capability | Graph → relationships | API → live external
"""
from __future__ import annotations
import json, pickle, sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).parent.parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — SQL  (structured filters on the internal supplier table)
# ─────────────────────────────────────────────────────────────────────────────
def query_sql(
    country: str | None = None,
    iso_9001: bool | None = None,
    iso_14001: bool | None = None,
    min_employees: int | None = None,
    max_employees: int | None = None,
    min_esg: float | None = None,
    min_financial_health: float | None = None,
    limit: int = 8,
) -> list[dict]:
    """Query the internal structured supplier table with exact filters."""
    conn = sqlite3.connect(DATA_DIR / "suppliers.db")
    conn.row_factory = sqlite3.Row
    conds, params = [], []

    if country:
        codes = [c.strip().upper() for c in country.replace(";", ",").split(",")]
        conds.append(f"country IN ({','.join('?'*len(codes))})")
        params.extend(codes)
    if iso_9001 is not None:
        conds.append("iso_9001 = ?"); params.append(int(iso_9001))
    if iso_14001 is not None:
        conds.append("iso_14001 = ?"); params.append(int(iso_14001))
    if min_employees is not None:
        conds.append("employees >= ?"); params.append(min_employees)
    if max_employees is not None:
        conds.append("employees <= ?"); params.append(max_employees)
    if min_esg is not None:
        conds.append("esg_score >= ?"); params.append(min_esg)
    if min_financial_health is not None:
        conds.append("financial_health >= ?"); params.append(min_financial_health)

    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = conn.execute(
        f"SELECT * FROM suppliers {where} ORDER BY financial_health DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()

    return [
        {
            "id": r["id"], "name": r["name"],
            "country": r["country"], "city": r["city"],
            "employees": r["employees"],
            "annual_revenue_usd": r["annual_revenue_usd"],
            "iso_9001": bool(r["iso_9001"]),
            "iso_14001": bool(r["iso_14001"]),
            "financial_health": round(r["financial_health"], 2),
            "esg_score": round(r["esg_score"], 2),
            "years_active": r["years_active"],
            "source": "SQL — internal structured table",
            "source_color": "blue",
        }
        for r in rows
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — Vector  (TF-IDF semantic search over capability descriptions)
# ─────────────────────────────────────────────────────────────────────────────
_VEC: dict | None = None

def _vec():
    global _VEC
    if _VEC is None:
        with open(DATA_DIR / "vector_store.pkl", "rb") as f:
            _VEC = pickle.load(f)
    return _VEC


def search_vector(query: str, top_k: int = 6) -> list[dict]:
    """Semantic search over supplier capability descriptions (TF-IDF)."""
    v = _vec()
    q_vec = v["vectorizer"].transform([query])
    scores = cosine_similarity(q_vec, v["matrix"]).flatten()
    top = scores.argsort()[::-1][:top_k]
    results = []
    for idx in top:
        if scores[idx] < 0.02:
            continue
        s = v["suppliers"][idx]
        results.append({
            **s,
            "semantic_score": round(float(scores[idx]), 4),
            "source": "Vector — semantic capability search",
            "source_color": "purple",
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — Graph  (NetworkX knowledge graph traversal)
# ─────────────────────────────────────────────────────────────────────────────
_G = None

def _graph():
    global _G
    if _G is None:
        with open(DATA_DIR / "supplier_graph.pkl", "rb") as f:
            _G = pickle.load(f)
    return _G


def traverse_graph(
    query_type: str,
    entity_name: str | None = None,
) -> list[dict]:
    """
    Traverse the supplier knowledge graph.
    query_type: 'suppliers_to' | 'customers_of' | 'ownership' | 'risk_connections'
    entity_name: customer/parent company name for targeted traversal
    """
    G = _graph()
    results = []

    if query_type == "suppliers_to":
        # Which suppliers → supply to a given customer?
        target = (entity_name or "").lower()
        for node, data in G.nodes(data=True):
            if data.get("type") == "company" and target in node.lower():
                for sup_node in G.predecessors(node):
                    sd = G.nodes[sup_node]
                    if sd.get("type") == "supplier":
                        results.append({
                            "id": sd.get("id", sup_node),
                            "name": sd.get("name", sup_node),
                            "country": sd.get("country", ""),
                            "relationship": f"Supplies to {node}",
                            "source": "Graph — relationship traversal",
                            "source_color": "green",
                        })

    elif query_type == "customers_of":
        # Which customers does a supplier serve?
        target = (entity_name or "").lower()
        for node, data in G.nodes(data=True):
            if data.get("type") == "supplier" and target in data.get("name", "").lower():
                for cust in G.successors(node):
                    cd = G.nodes[cust]
                    if cd.get("type") == "company":
                        results.append({
                            "supplier_id": data.get("id"),
                            "supplier_name": data.get("name"),
                            "customer": cust,
                            "customer_type": cd.get("company_type", ""),
                            "source": "Graph — customer relationship",
                            "source_color": "green",
                        })

    elif query_type == "ownership":
        # Who owns which suppliers?
        target = (entity_name or "").lower()
        for node, data in G.nodes(data=True):
            if data.get("type") == "parent_company":
                if target and target not in node.lower():
                    continue
                for sup_node in G.successors(node):
                    sd = G.nodes[sup_node]
                    if sd.get("type") == "supplier":
                        results.append({
                            "id": sd.get("id", sup_node),
                            "name": sd.get("name", sup_node),
                            "country": sd.get("country", ""),
                            "parent_company": node,
                            "parent_sanctioned": G.nodes[node].get("sanctions_flag", False),
                            "relationship": f"Owned by {node}",
                            "source": "Graph — ownership traversal",
                            "source_color": "green",
                        })

    elif query_type == "risk_connections":
        # Suppliers connected to sanctioned entities
        sanctioned = [n for n, d in G.nodes(data=True) if d.get("sanctions_flag")]
        seen = set()
        for s_node in sanctioned:
            for neighbor in list(G.predecessors(s_node)) + list(G.successors(s_node)):
                nd = G.nodes[neighbor]
                if nd.get("type") == "supplier" and neighbor not in seen:
                    seen.add(neighbor)
                    results.append({
                        "id": nd.get("id", neighbor),
                        "name": nd.get("name", neighbor),
                        "country": nd.get("country", ""),
                        "risk_entity": s_node,
                        "risk_flag": "⚠ Connected to sanctioned entity",
                        "source": "Graph — risk traversal",
                        "source_color": "red",
                    })

    return results[:12]


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — External API  (mock live supplier discovery API, e.g. Veridion)
# ─────────────────────────────────────────────────────────────────────────────
def call_external_api(
    query: str,
    country_filter: str | None = None,
    iso_9001: bool | None = None,
    limit: int = 6,
) -> list[dict]:
    """
    Simulate a live external supplier API (Veridion/D&B style).
    Returns suppliers NOT in the internal database — fresh/external data.
    """
    with open(DATA_DIR / "external_suppliers.json") as f:
        external = json.load(f)["suppliers"]

    q_lower = query.lower()
    scored = []
    for s in external:
        text = (s.get("description", "") + " " + " ".join(s.get("products", []))).lower()
        score = sum(w in text for w in q_lower.split() if len(w) > 3)
        if country_filter:
            codes = [c.strip().upper() for c in country_filter.split(",")]
            if s.get("country") in codes:
                score += 2
        if iso_9001 is not None and s.get("iso_9001") == iso_9001:
            score += 1
        scored.append((score, s))

    scored.sort(reverse=True, key=lambda x: x[0])
    results = []
    for score, s in scored[:limit]:
        if score == 0:
            continue
        results.append({
            "id": s["id"], "name": s["name"],
            "country": s["country"], "city": s.get("city", ""),
            "iso_9001": s.get("iso_9001", False),
            "esg_score": s.get("esg_score", 0.5),
            "financial_health": s.get("financial_health", 0.5),
            "years_active": s.get("years_active", 0),
            "snippet": s.get("description", "")[:180] + "…",
            "confidence": round(min(0.94, 0.55 + score * 0.08), 2),
            "source_url": f"https://{s.get('domain','example.com')}",
            "source": "External API — live lookup (Veridion-style)",
            "source_color": "orange",
        })
    return results
