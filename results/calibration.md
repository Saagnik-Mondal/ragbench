SciFact test claims: 300. Answering every claim, the hybrid retriever's top result is correct 62.3% of the time. Calibrators were fit on 566 train claims; thresholds were chosen on 243 other train claims.

### How well calibrated is the confidence? (lower ECE and Brier are better)

| Method | ECE | 95% CI | Brier | AUROC |
|---|---:|---:|---:|---:|
| raw cosine (naive) | 0.193 | 0.139 to 0.250 | 0.260 | 0.667 |
| Platt scaling (dense score) | 0.066 | 0.055 to 0.133 | 0.220 | 0.667 |
| histogram binning (dense score) | 0.051 | 0.035 to 0.117 | 0.222 | 0.652 |
| isotonic regression (dense score) | 0.059 | 0.043 to 0.123 | 0.222 | 0.655 |
| logistic regression (all signals) | 0.060 | 0.046 to 0.115 | 0.162 | 0.837 |

### Can it abstain safely? Choose a confidence threshold on held-out data, then test it

| Target precision | Method | Test coverage | Test precision |
|---:|---|---:|---:|
| 70% | raw cosine (naive) | 65.3% | 70.4% |
| 70% | isotonic regression (dense score) | 70.7% | 68.9% |
| 70% | logistic regression (all signals) | 88.3% | 67.9% |
| 80% | raw cosine (naive) | 41.0% | 75.6% |
| 80% | isotonic regression (dense score) | 55.3% | 72.9% |
| 80% | logistic regression (all signals) | 65.7% | 77.7% |
| 90% | raw cosine (naive) | 15.7% | 85.1% |
| 90% | isotonic regression (dense score) | 25.3% | 82.9% |
| 90% | logistic regression (all signals) | 54.0% | 84.0% |
