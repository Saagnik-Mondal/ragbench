from ragbench.evaluate import evaluate_retriever
from ragbench.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HashingEmbedder,
    RerankRetriever,
    build_retrievers,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Risk of Fractures, in the elderly!") == ["risk", "fractures", "elderly"]


def test_bm25_finds_lexical_match(corpus):
    hits = BM25Retriever(corpus).search("aspirin colorectal", k=3)
    assert hits[0].doc_id == "d2"


def test_dense_finds_match_and_caches_embeddings(corpus, tmp_path):
    emb = HashingEmbedder()
    first = DenseRetriever(corpus, emb, cache_dir=tmp_path)
    assert first.search("vitamin D fractures", k=1)[0].doc_id == "d1"
    assert len(list(tmp_path.glob("*.npy"))) == 1

    class Exploding(HashingEmbedder):
        def embed_documents(self, texts):  # must not be called: cache hit
            raise AssertionError("cache was not used")

    DenseRetriever(corpus, Exploding(), cache_dir=tmp_path)


def test_rrf_rewards_agreement_between_rankers():
    # "a" is 1st in one list and 2nd in the other; "b" and "c" are each 1st in only one list.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "a"]], k=60)
    assert fused[0].doc_id == "a"
    assert fused[0].score == 1 / 61 + 1 / 62
    assert {h.doc_id for h in fused} == {"a", "b", "c"}


def test_rrf_includes_docs_seen_by_only_one_ranker():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert {h.doc_id for h in fused} == {"a", "b"}


def test_rerank_reorders_candidates(corpus):
    base = BM25Retriever(corpus)
    prefer_d5 = lambda q, passages: [1.0 if "BRCA1" in p else 0.0 for p in passages]  # noqa: E731
    rr = RerankRetriever(base, corpus, prefer_d5, candidates=5)
    assert rr.search("cancer", k=1)[0].doc_id == "d5"
    assert rr.name == "bm25+rerank"


def test_build_retrievers_registry_and_evaluation(corpus):
    retrievers = build_retrievers(corpus, HashingEmbedder(), scorer=lambda q, p: [0.0] * len(p))
    assert set(retrievers) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    result = evaluate_retriever(retrievers["bm25"], corpus, k=3)
    assert result["Recall@3"] == 1.0
    assert 0 < result["nDCG@3"] <= 1.0
