"""Retrieval strategies: BM25, dense, hybrid (reciprocal rank fusion) and cross-encoder rerank."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .data import Corpus

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "with which who what when where how not no we our their these those than then there".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Hit:
    doc_id: str
    score: float


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int = 10) -> list[Hit]: ...


# --------------------------------------------------------------------------- BM25
class BM25Retriever:
    """Sparse lexical retrieval (Okapi BM25)."""

    name = "bm25"

    def __init__(self, corpus: Corpus) -> None:
        self._ids = [d.id for d in corpus.documents]
        self._bm25 = BM25Okapi([tokenize(d.full_text) for d in corpus.documents])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [Hit(self._ids[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------- Dense
class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedder:
    """ONNX sentence embeddings via fastembed (no PyTorch required)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        vectors = []
        for i, vec in enumerate(self._model.embed(texts, batch_size=64), start=1):
            vectors.append(vec)
            if i % 500 == 0 or i == len(texts):
                print(f"  embedded {i}/{len(texts)} documents", flush=True)
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._model.query_embed(text))), dtype=np.float32)


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedder. Zero downloads: used for tests and demos."""

    def __init__(self, dim: int = 512) -> None:
        self.model_name = f"hashing-{dim}"
        self._dim = dim

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # noqa: S324 (not security)
            vec[h % self._dim] += 1.0 if (h >> 64) % 2 == 0 else -1.0
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._embed(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class DenseRetriever:
    """Semantic retrieval: cosine similarity between query and document embeddings."""

    name = "dense"

    def __init__(self, corpus: Corpus, embedder: Embedder, cache_dir: Path | None = None) -> None:
        self._ids = [d.id for d in corpus.documents]
        self._embedder = embedder
        self._matrix = _normalize(self._load_or_embed(corpus, cache_dir))

    def _load_or_embed(self, corpus: Corpus, cache_dir: Path | None) -> np.ndarray:
        cache_file = None
        if cache_dir is not None:
            fingerprint = hashlib.md5(  # noqa: S324 (cache key only)
                "".join(self._ids).encode()
            ).hexdigest()[:10]
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", self._embedder.model_name)
            cache_file = cache_dir / f"emb_{safe}_{len(self._ids)}_{fingerprint}.npy"
            if cache_file.exists():
                return np.load(cache_file)
        matrix = self._embedder.embed_documents([d.full_text for d in corpus.documents])
        if cache_file is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
            np.save(cache_file, matrix)
        return matrix

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = _normalize(self._embedder.embed_query(query))
        scores = self._matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self._ids[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------- Hybrid
def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[Hit]:
    """Fuse several ranked lists. score(d) = sum_r 1 / (k + rank_r(d)). Robust, parameter-light."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return [Hit(d, s) for d, s in sorted(scores.items(), key=lambda kv: -kv[1])]


class HybridRetriever:
    """BM25 + dense fused with reciprocal rank fusion."""

    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 100, rrf_k: int = 60) -> None:
        self._retrievers = list(retrievers)
        self._depth = depth
        self._rrf_k = rrf_k

    def search(self, query: str, k: int = 10) -> list[Hit]:
        rankings = [[h.doc_id for h in r.search(query, self._depth)] for r in self._retrievers]
        return reciprocal_rank_fusion(rankings, self._rrf_k)[:k]


# --------------------------------------------------------------------------- Rerank
Scorer = Callable[[str, list[str]], list[float]]


def cross_encoder_scorer(model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> Scorer:
    """Build a (query, passages) -> scores function backed by a fastembed cross-encoder."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model = TextCrossEncoder(model_name)

    def score(query: str, passages: list[str]) -> list[float]:
        return [float(s) for s in model.rerank(query, passages)]

    return score


class RerankRetriever:
    """Two-stage retrieval: a cheap first stage proposes candidates, a cross-encoder re-scores them."""

    def __init__(
        self,
        base: Retriever,
        corpus: Corpus,
        scorer: Scorer,
        candidates: int = 30,
        name: str | None = None,
    ) -> None:
        self._base = base
        self._corpus = corpus
        self._scorer = scorer
        self._candidates = candidates
        self.name = name or f"{base.name}+rerank"

    def search(self, query: str, k: int = 10) -> list[Hit]:
        cands = self._base.search(query, self._candidates)
        passages = [self._corpus.get(h.doc_id).full_text for h in cands]
        scores = self._scorer(query, passages)
        order = np.argsort(-np.asarray(scores))[:k]
        return [Hit(cands[i].doc_id, float(scores[i])) for i in order]


# --------------------------------------------------------------------------- Registry
def build_retrievers(
    corpus: Corpus,
    embedder: Embedder,
    scorer: Scorer | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Retriever]:
    """Build every strategy. Pass scorer=None to skip the reranked variants."""
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus, embedder, cache_dir)
    hybrid = HybridRetriever([bm25, dense])
    retrievers: dict[str, Retriever] = {r.name: r for r in (bm25, dense, hybrid)}
    if scorer is not None:
        rerank = RerankRetriever(hybrid, corpus, scorer, name="hybrid+rerank")
        retrievers[rerank.name] = rerank
    return retrievers
