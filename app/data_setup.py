"""
One-time setup — builds all 4 synthetic data stores.
Run: python -m app.data_setup
"""
from __future__ import annotations
import json, pickle, sqlite3
from pathlib import Path
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).parent.parent / "data"

# ── seed: reuse existing suppliers.json ──────────────────────────────────────
def _load_seed() -> list[dict]:
    with open(DATA_DIR / "suppliers.json") as f:
        return json.load(f)["suppliers"]

# Extra attributes not in the original JSON (employees, revenue, city, parent)
_EXTRA = {
    "sup_001": {"city":"Zurich",      "employees":450,  "revenue":85_000_000,  "parent":"Alpen Industrial Group AG"},
    "sup_002": {"city":"Cologne",     "employees":820,  "revenue":210_000_000, "parent":None},
    "sup_003": {"city":"Stockholm",   "employees":310,  "revenue":62_000_000,  "parent":"Nordic Industries AB"},
    "sup_004": {"city":"Stockholm",   "employees":310,  "revenue":62_000_000,  "parent":"Nordic Industries AB"},
    "sup_005": {"city":"Paris",       "employees":680,  "revenue":175_000_000, "parent":None},
    "sup_006": {"city":"Warsaw",      "employees":210,  "revenue":38_000_000,  "parent":None},
    "sup_007": {"city":"Nicosia",     "employees":90,   "revenue":29_000_000,  "parent":"GlobalCopper Holdings BV"},
    "sup_008": {"city":"Madrid",      "employees":520,  "revenue":120_000_000, "parent":None},
    "sup_009": {"city":"Milan",       "employees":430,  "revenue":95_000_000,  "parent":None},
    "sup_010": {"city":"Amsterdam",   "employees":760,  "revenue":230_000_000, "parent":None},
    "sup_011": {"city":"Prague",      "employees":190,  "revenue":28_000_000,  "parent":None},
    "sup_012": {"city":"Vienna",      "employees":280,  "revenue":51_000_000,  "parent":"Alpen Industrial Group AG"},
    "sup_013": {"city":"Helsinki",    "employees":240,  "revenue":44_000_000,  "parent":"Nordic Industries AB"},
    "sup_014": {"city":"Athens",      "employees":160,  "revenue":22_000_000,  "parent":None},
    "sup_015": {"city":"Brussels",    "employees":195,  "revenue":33_000_000,  "parent":None},
    "sup_016": {"city":"Istanbul",    "employees":390,  "revenue":71_000_000,  "parent":None},
    "sup_017": {"city":"Shenzhen",    "employees":1200, "revenue":310_000_000, "parent":"GlobalCopper Holdings BV"},
    "sup_018": {"city":"Detroit",     "employees":540,  "revenue":140_000_000, "parent":None},
    "sup_019": {"city":"Birmingham",  "employees":410,  "revenue":88_000_000,  "parent":None},
    "sup_020": {"city":"Mumbai",      "employees":680,  "revenue":55_000_000,  "parent":None},
    "sup_021": {"city":"Frankfurt",   "employees":260,  "revenue":47_000_000,  "parent":None},
    "sup_022": {"city":"Marseille",   "employees":170,  "revenue":26_000_000,  "parent":None},
    "sup_023": {"city":"Vilnius",     "employees":140,  "revenue":18_000_000,  "parent":None},
    "sup_024": {"city":"Bucharest",   "employees":185,  "revenue":24_000_000,  "parent":None},
    "sup_025": {"city":"Lisbon",      "employees":200,  "revenue":31_000_000,  "parent":None},
    "sup_026": {"city":"Rotterdam",   "employees":220,  "revenue":42_000_000,  "parent":None},
    "sup_027": {"city":"Dresden",     "employees":310,  "revenue":58_000_000,  "parent":None},
    "sup_028": {"city":"Barcelona",   "employees":165,  "revenue":27_000_000,  "parent":None},
    "sup_029": {"city":"Bratislava",  "employees":120,  "revenue":16_000_000,  "parent":None},
    "sup_030": {"city":"Budapest",    "employees":230,  "revenue":39_000_000,  "parent":None},
    "sup_031": {"city":"Seville",     "employees":95,   "revenue":12_000_000,  "parent":None},
    "sup_032": {"city":"Gothenburg",  "employees":320,  "revenue":65_000_000,  "parent":None},
    "sup_033": {"city":"Munich",      "employees":180,  "revenue":55_000_000,  "parent":None},
    "sup_034": {"city":"Nice",        "employees":80,   "revenue":9_000_000,   "parent":None},
    "sup_035": {"city":"Florence",    "employees":140,  "revenue":20_000_000,  "parent":None},
    "sup_036": {"city":"Krakow",      "employees":210,  "revenue":31_000_000,  "parent":None},
    "sup_037": {"city":"Geneva",      "employees":260,  "revenue":78_000_000,  "parent":"Alpen Industrial Group AG"},
    "sup_038": {"city":"Amsterdam",   "employees":110,  "revenue":18_000_000,  "parent":None},
    "sup_039": {"city":"Athens",      "employees":90,   "revenue":11_000_000,  "parent":None},
    "sup_040": {"city":"London",      "employees":340,  "revenue":92_000_000,  "parent":None},
}

_DESCRIPTIONS = {
    "sup_001": "Specializes in precision copper power cables and stranded conductors for industrial automation and rail traction systems. Certified for demanding environments including high-voltage insulation and fire-resistant cable assemblies.",
    "sup_002": "High-volume manufacturer of copper power and medium-voltage cables for European utility companies and industrial plants. Strong track record supplying energy infrastructure projects across Germany and neighbouring markets.",
    "sup_003": "Swedish supplier of copper power cables engineered for rail traction, light rail, and heavy industrial applications. Products certified for cold-climate and marine environments.",
    "sup_004": "Nordic Cable — copper conductors and control cables for industrial automation and building infrastructure across Scandinavia.",
    "sup_005": "French aerospace-grade copper cables and control harnesses certified for Airbus supply chains. Specializes in lightweight shielded cables and EMC-compliant assemblies for avionic systems.",
    "sup_006": "Polish bare copper conductors and power cables for grid infrastructure and industrial distribution. Cost-competitive supplier for Eastern European energy projects.",
    "sup_007": "Cyprus-based copper cable trader serving European industrial buyers. Broad product range including medium-voltage and stranded conductors sourced from Asian and Eastern European mills.",
    "sup_008": "Spanish copper conductor specialist supplying telecom infrastructure, 5G base station cabling, and utility distribution networks across Iberia and Latin America.",
    "sup_009": "Italian automotive-grade copper cables and control harnesses for Stellantis, Iveco, and tier-1 automotive suppliers. Specializes in high-temperature resistant insulation.",
    "sup_010": "Dutch manufacturer of medium-voltage copper power cables and offshore subsea cable systems for North Sea oil and gas and offshore wind energy projects.",
    "sup_011": "Czech copper wire and cable manufacturer focusing on industrial wiring, panel building, and switchgear applications.",
    "sup_012": "Austrian copper power and control cables for rail infrastructure and public transport electrification projects. Supplies ÖBB and regional transit authorities.",
    "sup_013": "Finnish copper cable supplier certified for arctic environments. Products used in energy sector projects in Finland, Sweden, and Norway including offshore applications.",
    "sup_014": "Greek copper power cable manufacturer serving construction, utilities, and port infrastructure projects in Southern Europe and the Eastern Mediterranean.",
    "sup_015": "Belgian copper control cable specialist for industrial automation, robotics, and smart factory applications across Western Europe.",
    "sup_016": "Turkish copper cable manufacturer with export focus. Products cover power, control, and instrumentation cables for infrastructure and industrial projects.",
    "sup_017": "Shenzhen-based high-volume copper conductor and cable manufacturer exporting globally. Produces stranded conductors, copper wire, and power cables at competitive cost.",
    "sup_018": "US manufacturer of copper medium-voltage power cable for utility grid modernisation and renewable energy interconnections. Serves Ford and utility sector customers.",
    "sup_019": "UK copper cable manufacturer specialising in rail traction cables, signalling cables, and control cables for Network Rail and London Underground projects.",
    "sup_020": "Indian copper wire and bare conductor producer for power distribution networks across South Asia. Large-volume, cost-efficient supplier.",
    "sup_021": "German copper power cable supplier for industrial plants, chemical processing facilities, and energy infrastructure.",
    "sup_022": "French stranded copper conductors and cables for solar farm grid connections, offshore wind onshore infrastructure, and grid storage projects.",
    "sup_023": "Lithuanian bare copper conductors and cables for Baltic power distribution networks and cross-border interconnection projects.",
    "sup_024": "Romanian copper power and control cables for Eastern European industrial and infrastructure markets.",
    "sup_025": "Portuguese copper cable supplier for telecom, energy, and marine applications. Exports to Portuguese-speaking markets in Africa and Brazil.",
    "sup_026": "Dutch copper power cables and conductors for renewable energy projects including offshore wind, solar, and green hydrogen infrastructure.",
    "sup_027": "German copper wire, cable, and control cable manufacturer for industrial automation, robotics, and machine building applications.",
    "sup_028": "Spanish copper power cable manufacturer for construction, building infrastructure, and photovoltaic solar installations.",
    "sup_029": "Slovak copper wire and bare conductor producer for Central European industrial wiring and distribution applications.",
    "sup_030": "Hungarian copper power cable and stranded conductor manufacturer for regional infrastructure and industrial projects.",
    "sup_031": "Spanish olive oil producer — not a relevant supplier for cable procurement.",
    "sup_032": "Swedish sawn timber and wood pulp supplier for construction — not relevant for copper cable procurement.",
    "sup_033": "Bavarian brewery — not relevant for procurement of industrial cables or conductors.",
    "sup_034": "French cotton linens and textiles manufacturer — not relevant for cable procurement.",
    "sup_035": "Italian marble quarrier — not relevant for copper or cable procurement.",
    "sup_036": "Polish PVC pipe and plastic fittings manufacturer — tangentially relevant for cable conduit applications.",
    "sup_037": "Swiss precision watch movements and gear manufacturer — not relevant for copper cable procurement.",
    "sup_038": "Dutch tulip bulb exporter — not relevant.",
    "sup_039": "Greek solar panel and power inverter manufacturer. May be a downstream customer for copper cable rather than a supplier.",
    "sup_040": "UK payments software and fraud analytics provider — not a supplier of physical goods.",
}

# ── STORE 1: SQLite ────────────────────────────────────────────────────────
def create_sql_db(suppliers: list[dict]) -> None:
    db_path = DATA_DIR / "suppliers.db"
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS suppliers")
    conn.execute("""
        CREATE TABLE suppliers (
            id TEXT PRIMARY KEY,
            name TEXT,
            country TEXT,
            city TEXT,
            employees INTEGER,
            annual_revenue_usd INTEGER,
            year_founded INTEGER,
            iso_9001 INTEGER,
            iso_14001 INTEGER,
            iso_27001 INTEGER,
            financial_health REAL,
            esg_score REAL,
            years_active INTEGER,
            sanctions_flag INTEGER,
            products TEXT
        )
    """)
    for s in suppliers:
        ex = _EXTRA.get(s["id"], {})
        has_9001  = any(c["type"] == "ISO 9001"  for c in s.get("certifications", []))
        has_14001 = any(c["type"] == "ISO 14001" for c in s.get("certifications", []))
        has_27001 = any(c["type"] == "ISO 27001" for c in s.get("certifications", []))
        conn.execute("""
            INSERT INTO suppliers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            s["id"], s["name"], s["country"],
            ex.get("city", ""),
            ex.get("employees", 0),
            ex.get("revenue", 0),
            2024 - s.get("years_active", 10),
            int(has_9001), int(has_14001), int(has_27001),
            s.get("financial_health", 0.5),
            s.get("esg_score", 0.5),
            s.get("years_active", 0),
            0,   # sanctions — handled in verify layer
            ",".join(s.get("products", [])),
        ))
    conn.commit()
    conn.close()
    print(f"  SQL: {len(suppliers)} rows → {db_path}")


# ── STORE 2: Vector (TF-IDF over capability descriptions) ──────────────────
def create_vector_store(suppliers: list[dict]) -> None:
    docs, meta = [], []
    for s in suppliers:
        base = _DESCRIPTIONS.get(s["id"], s.get("description", ""))
        products = " ".join(s.get("products", []))
        text = f"{s['name']} {products} {base}"
        docs.append(text)
        meta.append({"id": s["id"], "name": s["name"], "country": s["country"],
                      "description": base[:200]})

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(docs)

    out = DATA_DIR / "vector_store.pkl"
    with open(out, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "suppliers": meta}, f)
    print(f"  Vector: {len(docs)} documents → {out}")


# ── STORE 3: Knowledge Graph (NetworkX) ───────────────────────────────────
def create_graph(suppliers: list[dict]) -> None:
    G = nx.DiGraph()

    # Supplier nodes
    for s in suppliers:
        ex = _EXTRA.get(s["id"], {})
        G.add_node(s["id"], type="supplier", name=s["name"],
                   country=s["country"], id=s["id"],
                   city=ex.get("city", ""),
                   financial_health=s.get("financial_health", 0.5))

    # Customer (Fortune 500) nodes
    customers = set()
    for s in suppliers:
        for c in s.get("fortune500_customers", []):
            customers.add(c)
    for c in customers:
        G.add_node(c, type="company", company_type="fortune500")

    # SUPPLIES_TO edges
    for s in suppliers:
        for c in s.get("fortune500_customers", []):
            G.add_edge(s["id"], c, relationship="SUPPLIES_TO")

    # Parent company nodes + OWNED_BY edges
    parents = {
        "Alpen Industrial Group AG":   {"country":"CH", "sanctions_flag": False},
        "Nordic Industries AB":         {"country":"SE", "sanctions_flag": False},
        "GlobalCopper Holdings BV":     {"country":"NL", "sanctions_flag": True},  # sanctioned
    }
    for p_name, p_attrs in parents.items():
        G.add_node(p_name, type="parent_company", **p_attrs)

    for sid, ex in _EXTRA.items():
        if ex.get("parent"):
            G.add_edge(ex["parent"], sid, relationship="OWNS")

    # Certification body nodes
    cert_bodies = {
        "TÜV Rheinland": {"country": "DE"},
        "SGS":            {"country": "CH"},
        "SQS":            {"country": "CH"},
        "Bureau Veritas": {"country": "FR"},
        "Lloyd's Register":{"country":"GB"},
    }
    _cert_map = {
        "DE": "TÜV Rheinland", "AT": "TÜV Rheinland", "CH": "SQS",
        "FR": "Bureau Veritas", "GB": "Lloyd's Register",
    }
    for cb, cb_attrs in cert_bodies.items():
        G.add_node(cb, type="cert_body", **cb_attrs)

    for s in suppliers:
        has_cert = any(c["type"] in ("ISO 9001","ISO 14001") for c in s.get("certifications", []))
        if has_cert:
            cb = _cert_map.get(s["country"], "SGS")
            G.add_edge(s["id"], cb, relationship="CERTIFIED_BY")

    out = DATA_DIR / "supplier_graph.pkl"
    with open(out, "wb") as f:
        pickle.dump(G, f)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges → {out}")


# ── STORE 4: External API data (suppliers NOT in internal DB) ──────────────
EXTERNAL_SUPPLIERS = [
    {"id":"ext_001","name":"Elbe Copper GmbH","domain":"elbecopper.example","country":"DE","city":"Hamburg",
     "products":["copper cable","medium voltage cable","offshore cable"],
     "description":"Hamburg-based specialist in offshore wind cable systems and medium-voltage copper cables for North Sea projects. Emerging player with strong ESG credentials.",
     "iso_9001":True,"iso_14001":True,"financial_health":0.78,"esg_score":0.88,"years_active":6},
    {"id":"ext_002","name":"Vltava Conductors s.r.o.","domain":"vltava-conductors.example","country":"CZ","city":"Brno",
     "products":["stranded copper conductor","copper wire","bare copper conductor"],
     "description":"Czech manufacturer of stranded copper conductors and bare wire for European panel builders and OEMs. Competitive pricing with ISO 9001 certification.",
     "iso_9001":True,"iso_14001":False,"financial_health":0.71,"esg_score":0.65,"years_active":8},
    {"id":"ext_003","name":"Adriatic Cable doo","domain":"adriaticcable.example","country":"HR","city":"Split",
     "products":["copper cable","power cable","control cable"],
     "description":"Croatian copper cable manufacturer serving Adriatic infrastructure projects. Certified for marine and port environments.",
     "iso_9001":True,"iso_14001":False,"financial_health":0.67,"esg_score":0.72,"years_active":5},
    {"id":"ext_004","name":"Silesia Wire SA","domain":"silesiawire.example","country":"PL","city":"Katowice",
     "products":["copper wire","stranded copper conductor","bare copper conductor"],
     "description":"Polish copper wire specialist with large-volume production capacity. Supplies Eastern European automotive and energy sector OEMs.",
     "iso_9001":True,"iso_14001":True,"financial_health":0.74,"esg_score":0.61,"years_active":9},
    {"id":"ext_005","name":"Transdanube Kabel Kft","domain":"transdanube.example","country":"HU","city":"Pécs",
     "products":["copper cable","control cable"],
     "description":"Hungarian startup producing copper control cables for smart grid and industrial automation. Founded 2019, growing rapidly.",
     "iso_9001":True,"iso_14001":False,"financial_health":0.62,"esg_score":0.70,"years_active":4},
    {"id":"ext_006","name":"Éire Copper Systems Ltd","domain":"eirecopper.example","country":"IE","city":"Cork",
     "products":["copper cable","power cable","stranded copper conductor"],
     "description":"Irish copper cable manufacturer focused on renewable energy connections and offshore wind onshore infrastructure.",
     "iso_9001":True,"iso_14001":True,"financial_health":0.76,"esg_score":0.85,"years_active":7},
    {"id":"ext_007","name":"Kaliningrad Copper OOO","domain":"kalcopper.example","country":"RU","city":"Kaliningrad",
     "products":["copper cable","stranded copper conductor"],
     "description":"Russian copper cable producer — subject to EU trade restrictions. Do not source from this entity.",
     "iso_9001":False,"iso_14001":False,"financial_health":0.55,"esg_score":0.30,"years_active":18},
    {"id":"ext_008","name":"Aegean Copper SA","domain":"aegeancopper.example","country":"GR","city":"Thessaloniki",
     "products":["copper cable","power cable","bare copper conductor"],
     "description":"Greek copper cable and conductor manufacturer serving Balkan energy and construction markets.",
     "iso_9001":True,"iso_14001":False,"financial_health":0.65,"esg_score":0.69,"years_active":10},
    {"id":"ext_009","name":"Galicia Cable SL","domain":"galiciacable.example","country":"ES","city":"Vigo",
     "products":["copper cable","medium voltage cable","power cable"],
     "description":"Spanish medium-voltage copper cable manufacturer with offshore and renewable energy focus. New market entrant 2021.",
     "iso_9001":True,"iso_14001":True,"financial_health":0.70,"esg_score":0.81,"years_active":3},
    {"id":"ext_010","name":"Lodz Copper Works SA","domain":"lodzcopperwks.example","country":"PL","city":"Łódź",
     "products":["copper wire","bare copper conductor","stranded copper conductor"],
     "description":"Polish copper wire and conductor producer, recently ISO 9001 certified. Cost-competitive for Eastern European and export markets.",
     "iso_9001":True,"iso_14001":False,"financial_health":0.69,"esg_score":0.63,"years_active":5},
]

def create_external_api_data() -> None:
    out = DATA_DIR / "external_suppliers.json"
    with open(out, "w") as f:
        json.dump({"suppliers": EXTERNAL_SUPPLIERS}, f, indent=2)
    print(f"  External API: {len(EXTERNAL_SUPPLIERS)} records → {out}")


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building 4 data stores…")
    suppliers = _load_seed()
    create_sql_db(suppliers)
    create_vector_store(suppliers)
    create_graph(suppliers)
    create_external_api_data()
    print("Done — all 4 data stores ready.")
