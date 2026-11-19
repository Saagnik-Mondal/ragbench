Dataset: BEIR SciFact, 5,183 documents, 300 test queries.

| Strategy | nDCG@10 | Recall@10 | Recall@100 | MRR@10 | latency_ms_mean | latency_ms_p95 |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.669 | 0.789 | 0.886 | 0.638 | 7.9 | 13.7 |
| dense | 0.721 | 0.840 | 0.955 | 0.689 | 30.4 | 56.0 |
| hybrid | **0.726** | **0.842** | **0.965** | **0.695** | 59.0 | 101.3 |
| hybrid+rerank | 0.696 | 0.838 | 0.921 | 0.659 | 3145.4 | 3748.7 |
