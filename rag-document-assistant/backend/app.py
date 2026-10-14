"""FastAPI backend.  Run:  uvicorn app:app --reload --port 8000   (from the backend/ folder)"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import rag

app = FastAPI(title="RAG Document Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Question(BaseModel):
    question: str
    top_k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(rag.store.chunks)}


@app.post("/ask")
def ask(body: Question):
    return rag.ask(body.question, body.top_k)


@app.get("/")
def index():
    """Serve the React frontend from the same server."""
    from pathlib import Path
    return FileResponse(Path(__file__).parent.parent / "frontend" / "index.html")
