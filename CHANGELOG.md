# Changelog

## 0.3.0 - 2026-11-19

### Added
- The benchmark runs on any BEIR dataset (`--dataset scifact|nfcorpus`), and the benchmark workflow lets you pick one.
- Results on NFCorpus: hybrid still beats BM25 (nDCG@10 0.346 vs 0.311), but the reranker, which hurt on SciFact, helps here (0.362) at about 133x the latency.
- Loader tests.

### Changed
- README notes that the reranked row's Recall@100 is not comparable, because the reranker only reorders its top 30 candidates.

## 0.2.0 - 2026-11-19

### Added
- Calibrated retrieval confidence: raw cosine scores are badly overconfident (ECE 0.193); calibration brings it to about 0.06, and combining signals lets the system answer 54% of questions at about 84% precision (16% for the raw score). Own implementations of ECE, Brier score, histogram binning and the risk-coverage and threshold logic, with tests.
- A cloud workflow to run the calibration experiment.

## 0.1.0 - 2026-11-19

First release: BM25, dense, hybrid (reciprocal rank fusion) and reranked retrieval; nDCG, Recall and MRR written from scratch; a benchmark harness on BEIR SciFact; a FastAPI service with cited answers and a strategy-comparison page; Docker, tests and CI.
