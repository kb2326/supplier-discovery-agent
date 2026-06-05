from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATIC   = Path(__file__).resolve().parent.parent / "static"


def _setup_if_needed() -> None:
    """Build data stores on first boot (Railway / any fresh environment)."""
    if not (DATA_DIR / "suppliers.db").exists():
        print("Data stores not found — building now…")
        from .data_setup import (
            _load_seed, create_sql_db, create_vector_store,
            create_graph, create_external_api_data,
        )
        suppliers = _load_seed()
        create_sql_db(suppliers)
        create_vector_store(suppliers)
        create_graph(suppliers)
        create_external_api_data()
        print("Data stores ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_if_needed()
    yield


from .agent import run_agent   # noqa: E402 — import after setup guard

app = FastAPI(title="Supplier Discovery Agent", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return run_agent(req.message, req.history)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "chat.html")
