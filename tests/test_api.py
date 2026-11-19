import pytest
from fastapi.testclient import TestClient

from ragbench.api import create_app
from ragbench.generate import ExtractiveAnswerer
from ragbench.retrievers import HashingEmbedder, build_retrievers


@pytest.fixture
def client(corpus):
    retrievers = build_retrievers(corpus, HashingEmbedder())
    return TestClient(create_app(corpus, retrievers, ExtractiveAnswerer()))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "documents": 5}


def test_strategies_listed(client):
    assert client.get("/strategies").json() == ["bm25", "dense", "hybrid"]


def test_search_returns_ranked_passages(client):
    r = client.post(
        "/search", json={"query": "aspirin colorectal cancer", "strategy": "bm25", "k": 2}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["results"][0]["doc_id"] == "d2"
    assert len(body["results"]) == 2


def test_ask_returns_cited_answer(client):
    r = client.post(
        "/ask", json={"query": "does vitamin D reduce fracture risk", "strategy": "hybrid", "k": 3}
    )
    body = r.json()
    assert "d1" in body["citations"]
    assert "[" in body["answer"]
    assert body["backend"] == "extractive"


def test_unknown_strategy_is_404(client):
    r = client.post("/search", json={"query": "x", "strategy": "nope"})
    assert r.status_code == 404


def test_validation_rejects_empty_query(client):
    assert client.post("/search", json={"query": ""}).status_code == 422


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "ragbench" in r.text
