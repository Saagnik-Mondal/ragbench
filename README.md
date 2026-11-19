# ragbench

[![CI](https://github.com/Saagnik-Mondal/ragbench/actions/workflows/ci.yml/badge.svg)](https://github.com/Saagnik-Mondal/ragbench/actions/workflows/ci.yml)

**A retrieval-augmented QA system that measures its own retrieval quality.** It compares four retrieval strategies (BM25, dense embeddings, hybrid, and hybrid with a cross-encoder reranker) on a public benchmark, and serves the best ones through an API with cited answers.

Most RAG demos never check whether retrieval actually works. This project treats retrieval as something to measure, and the measurements changed what I would ship.

## Results

Benchmark: [BEIR SciFact](https://github.com/beir-cellar/beir) (5,183 scientific abstracts, 300 test claims with expert relevance labels). Embeddings: `BAAI/bge-small-en-v1.5` (quantized ONNX). Reranker: `ms-marco-MiniLM-L-6-v2` cross-encoder over the top 30 hybrid candidates.

| Strategy | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | Mean latency |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.669 | 0.789 | 0.886 | 0.638 | 7.9 ms |
| Dense (bge-small) | 0.721 | 0.840 | 0.955 | 0.689 | 30 ms |
| **Hybrid (BM25 + dense, RRF)** | **0.726** | **0.842** | **0.965** | **0.695** | 59 ms |
| Hybrid + cross-encoder rerank | 0.696 | 0.838 | 0.921 | 0.659 | 3,145 ms |

![Retrieval quality by strategy](results/benchmark.png)

Latency is mean time per query on a standard GitHub Actions runner (CPU only, one query at a time). Raw numbers are in [`results/benchmark.json`](results/benchmark.json); the run that produced them is reproducible from the Actions tab (see below).

### What the numbers say

1. **Hybrid retrieval is the best default.** Fusing BM25 and dense rankings with reciprocal rank fusion lifts nDCG@10 from 0.669 (BM25) to 0.726, about 8.5% relative, for roughly 51 ms extra per query. It also has the best recall at 100, which matters when a downstream step needs a good candidate pool.
2. **The reranker did not help here.** Adding a general-purpose cross-encoder lowered nDCG@10 from 0.726 to 0.696 and made queries about 50x slower. The reranker was trained on web search passages, while SciFact queries are scientific claims, so the mismatch is plausible. I did not test a domain-tuned reranker, so this is a result for this model on this dataset, not a claim that reranking never helps.
3. **The harness checks out.** BM25 scores 0.669 nDCG@10, close to the roughly 0.665 reported for BM25 on SciFact in the BEIR paper, which is a useful sanity check that the metrics are implemented correctly.

## What is in the box

- **Retrieval strategies** behind one interface: BM25, dense, hybrid (reciprocal rank fusion), and two-stage rerank. See [`retrievers.py`](src/ragbench/retrievers.py).
- **Metrics written from scratch** (nDCG, Recall@k, MRR) with graded relevance, and unit-tested against hand computations. See [`metrics.py`](src/ragbench/metrics.py).
- **Evaluation harness** that runs every strategy over a labelled dataset and writes a table, JSON and chart. See [`evaluate.py`](src/ragbench/evaluate.py).
- **FastAPI service**: `/search`, `/ask` (answer with citations), `/strategies`, `/health`, and a small web page at `/` that compares strategies side by side.
- **Two answer backends**: an offline extractive backend that always works, and a Claude backend (set `ANTHROPIC_API_KEY`) that must cite passage numbers.
- **Tests and CI**: 20 tests, lint and format checks on every push (GitHub Actions), plus a one-click benchmark workflow.

## Architecture

```
                         question
                            |
          +-----------------+-----------------+
          |                                   |
      BM25 (lexical)                  Dense (bge-small)
          |                                   |
          +---------- reciprocal rank --------+
                         fusion
                            |
                  (optional) cross-encoder
                       rerank top 30
                            |
                     top-k passages
                            |
              +-------------+-------------+
              |                           |
        extractive answer          Claude answer
        (offline, cited)        (needs API key, cited)
```

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Saagnik-Mondal/ragbench.git
cd ragbench
uv sync --extra dev --extra bench
uv run pytest                 # 20 tests, no downloads needed
```

Run the API and UI:

```bash
uv run ragbench-serve         # http://localhost:8000
```

The first start downloads SciFact and the two models, then embeds all 5,183 documents. That took about 17 minutes on a CPU runner; the vectors are cached in `data/cache`, so later starts are fast. To skip the reranker (and its download), set `RAGBENCH_NO_RERANK=1`.

Try it:

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"query": "Does vitamin D reduce fracture risk?", "strategy": "hybrid", "k": 5}'
```

To use Claude for answers, install the extra and set a key:

```bash
uv sync --extra llm
export ANTHROPIC_API_KEY=...   # optional; without it the offline backend is used
```

A `Dockerfile` is included that bakes the dataset and models into the image.

## Reproduce the benchmark

On GitHub (no local compute): open the **Actions** tab, choose **Benchmark**, and click **Run workflow**. It runs the full evaluation and saves `results/` as a downloadable artifact.

Locally:

```bash
uv run python -m ragbench.evaluate               # full run, all four strategies
uv run python -m ragbench.evaluate --no-rerank   # skip the slow reranker
uv run python -m ragbench.evaluate --replot      # redraw the chart from saved results
```

## Design decisions

- **Reciprocal rank fusion over score blending.** BM25 and cosine scores live on different scales, so blending them needs tuning per dataset. RRF only uses ranks and has one well-behaved parameter.
- **ONNX embeddings through fastembed, not PyTorch.** The install stays small and runs on CPU-only machines and CI.
- **Retrievers share one `search(query, k)` interface** and the embedder and reranker are injected, so tests run in under a second with a deterministic hashing embedder and no model downloads.
- **An offline answer backend ships by default**, so the demo works without an API key and the citation behaviour is testable.

## Limitations

- One dataset (SciFact, about 5k documents). Rankings of strategies can change on other corpora, so treat the numbers as evidence for this setting.
- Only retrieval is measured. Answer faithfulness and citation accuracy are not evaluated yet.
- One reranker model and a fixed candidate depth of 30 were tested; I did not tune either or try a domain-specific reranker.
- Latency is single-query on a CI runner, not a load test.

## Project layout

```
src/ragbench/
  data.py          dataset download and loading
  retrievers.py    BM25, dense, hybrid (RRF), rerank
  metrics.py       nDCG, Recall@k, MRR
  evaluate.py      benchmark harness and chart
  generate.py      extractive and Claude answer backends
  api.py           FastAPI app
  static/          web UI
tests/             20 unit and API tests
results/           benchmark.json / .md / .png
.github/workflows/ ci.yml and benchmark.yml
```

## License

MIT. See [LICENSE](LICENSE).
