"""Minimal HTTP API (FastAPI). Run with `sungrid serve` and open http://127.0.0.1:8000/docs."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .factory import Components, build

_state: dict[str, Components] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    _state["c"] = build()
    yield
    _state["c"].store.close()


app = FastAPI(title="SunGrid Copilot", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, examples=["Am I eligible for the rooftop rebate?"])
    session_id: str | None = Field(None, description="Reuse to continue a multi-turn conversation.")
    include_trace: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    c = _state["c"]
    return {
        "status": "ok",
        "chunks": c.store.count(),
        "embedder": c.embedder.name,
        "llm": c.copilot.llm.name if c.copilot.llm else None,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    resp = _state["c"].copilot.ask(req.query, req.session_id).to_dict()
    if not req.include_trace:
        resp.pop("trace", None)
    return resp
