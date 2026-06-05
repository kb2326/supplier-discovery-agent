"""FastAPI app — the /search endpoint + the single-page frontend.

Thin transport layer over pipeline.run_search. The store (with entity resolution
already applied) is loaded ONCE at startup and reused across requests.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .pipeline import run_search
from .schemas import Result
from .store import load_store

app = FastAPI(title="Supplier-Discovery Agent (demo)")

# Resolve the golden store once. Cheap, in-memory, no DB.
STORE = load_store()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class SearchRequest(BaseModel):
    query: str
    # Optional per-criterion weight overrides from the UI sliders. Omitted -> defaults.
    weights: dict[str, float] | None = None


@app.post("/search", response_model=Result)
def search(req: SearchRequest) -> Result:
    """Run the full find -> rank -> verify pipeline and return the result + trace."""
    return run_search(req.query, STORE, weights=req.weights)


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(STATIC_DIR / "index.html")
