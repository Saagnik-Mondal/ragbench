"""FastAPI service: /search, /ask, /strategies, plus a small single-page UI at /."""

from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .data import Corpus
from .generate import Answerer, default_answerer, top_docs
from .retrievers import Retriever

STATIC = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=50)
    strategy: str = "hybrid"


class PassageOut(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    strategy: str
    latency_ms: float
    results: list[PassageOut]


class AskResponse(SearchResponse):
    answer: str
    citations: list[str]
    backend: str


def create_app(
    corpus: Corpus, retrievers: dict[str, Retriever], answerer: Answerer | None = None
) -> FastAPI:
    app = FastAPI(
        title="ragbench",
        version="0.1.0",
        description="Retrieval-augmented QA with swappable retrieval strategies.",
    )
    answerer = answerer or default_answerer()

    def run(req: SearchRequest) -> SearchResponse:
        retriever = retrievers.get(req.strategy)
        if retriever is None:
            raise HTTPException(404, f"Unknown strategy '{req.strategy}'. Try: {list(retrievers)}")
        start = time.perf_counter()
        hits = retriever.search(req.query, req.k)
        latency = (time.perf_counter() - start) * 1000
        results = [
            PassageOut(
                doc_id=h.doc_id,
                title=corpus.get(h.doc_id).title,
                text=corpus.get(h.doc_id).text,
                score=h.score,
            )
            for h in hits
        ]
        return SearchResponse(strategy=req.strategy, latency_ms=round(latency, 2), results=results)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "documents": len(corpus)}

    @app.get("/strategies")
    def strategies() -> list[str]:
        return list(retrievers)

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        return run(req)

    @app.post("/ask", response_model=AskResponse)
    def ask(req: SearchRequest) -> AskResponse:
        res = run(req)
        docs = top_docs(corpus, [r.doc_id for r in res.results])
        ans = answerer.answer(req.query, docs)
        return AskResponse(
            **res.model_dump(), answer=ans.text, citations=ans.citations, backend=ans.backend
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    return app


def build_default_app() -> FastAPI:
    """Production entrypoint: SciFact corpus + all strategies."""
    from .data import load_scifact
    from .retrievers import FastEmbedder, build_retrievers, cross_encoder_scorer

    data_dir = Path(os.environ.get("RAGBENCH_DATA_DIR", "data"))
    corpus = load_scifact(data_dir)
    scorer = None if os.environ.get("RAGBENCH_NO_RERANK") else cross_encoder_scorer()
    retrievers = build_retrievers(corpus, FastEmbedder(), scorer, cache_dir=data_dir / "cache")
    return create_app(corpus, retrievers)


def serve() -> None:
    import uvicorn

    uvicorn.run(build_default_app(), host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))  # noqa: S104


if __name__ == "__main__":
    serve()
