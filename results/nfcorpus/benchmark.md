Dataset: BEIR NFCorpus, 3,633 documents, 323 test queries.

| Strategy | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | latency_ms_mean | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.311 | 0.153 | 0.235 | 0.522 | 1.8 | 3.9 |
| dense | 0.343 | 0.163 | 0.306 | 0.533 | 15.2 | 25.1 |
| hybrid | 0.346 | 0.165 | **0.310** | 0.558 | 22.3 | 38.0 |
| hybrid+rerank | **0.362** | **0.170** | 0.228 | **0.579** | 2971.2 | 3297.7 |
