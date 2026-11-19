"""Confidence calibration for retrieval.

Question: when the retriever's top result looks confident, is it actually right?
A raw similarity score is not a probability, so "answer only if the score is high" has no
guaranteed meaning. Here we turn retrieval signals into calibrated probabilities that the
top result is correct, measure how well calibrated they are, and use them to abstain.

Everything the evaluation needs (ECE, Brier score, histogram binning, risk-coverage and
threshold selection) is implemented here from scratch and unit-tested. scikit-learn is used
only for isotonic regression, logistic regression and AUROC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Corpus
from .retrievers import Retriever, tokenize

# --------------------------------------------------------------------------- metrics


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and the 0/1 outcome (lower is better)."""
    return float(np.mean((probs - labels) ** 2))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins.

    Weighted average, over confidence bins, of |average confidence - observed accuracy|.
    0 means that of the answers given with ~70% confidence, about 70% really are right.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Put p == 1.0 into the last bin.
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


def reliability_curve(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """(mean confidence, observed accuracy, count) per non-empty bin, for a reliability diagram."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out.append((float(probs[mask].mean()), float(labels[mask].mean()), int(mask.sum())))
    return out


def risk_coverage(conf: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Answer the most confident queries first. Returns (coverage, accuracy among answered)."""
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    hits = np.asarray(labels, dtype=float)[order]
    n = np.arange(1, len(hits) + 1)
    return n / len(hits), np.cumsum(hits) / n


def pick_threshold(
    conf: np.ndarray, labels: np.ndarray, target_precision: float, min_answered: int = 20
) -> float | None:
    """Lowest confidence threshold whose answered set still reaches `target_precision`.

    "Lowest" = answers as many questions as possible. Returns None if the target is not
    reachable with at least `min_answered` answers (so we never trust a tiny lucky prefix).
    """
    conf = np.asarray(conf, dtype=float)
    order = np.argsort(-conf, kind="stable")
    hits = np.asarray(labels, dtype=float)[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    ok = np.where((precision >= target_precision) & (np.arange(1, len(hits) + 1) >= min_answered))[
        0
    ]
    if len(ok) == 0:
        return None
    return float(conf[order][ok[-1]])


def evaluate_threshold(conf: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    """Coverage and precision when answering only if confidence >= threshold."""
    conf = np.asarray(conf, dtype=float)
    labels = np.asarray(labels, dtype=float)
    answered = conf >= threshold
    n = int(answered.sum())
    return {
        "coverage": float(answered.mean()),
        "answered": n,
        "precision": float(labels[answered].mean()) if n else float("nan"),
    }


# --------------------------------------------------------------------------- calibrators


class HistogramBinning:
    """Calibrate by equal-frequency binning of the score; each bin predicts its observed accuracy."""

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins
        self._edges: np.ndarray | None = None
        self._rates: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> HistogramBinning:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        idx = np.searchsorted(edges, scores, side="right")
        prior = labels.mean()
        rates = np.empty(len(edges) + 1)
        for b in range(len(edges) + 1):
            m = idx == b
            # One pseudo-observation at the global rate keeps empty/tiny bins sane.
            rates[b] = (labels[m].sum() + prior) / (m.sum() + 1)
        self._edges, self._rates = edges, rates
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        assert self._edges is not None and self._rates is not None, "call fit() first"
        return self._rates[np.searchsorted(self._edges, np.asarray(scores, dtype=float), "right")]


# --------------------------------------------------------------------------- features


@dataclass
class QueryExample:
    query_id: str
    features: dict[str, float]
    correct: int  # 1 if the fused top-1 document is judged relevant


FEATURE_NAMES = [
    "dense_top1",
    "dense_margin",
    "bm25_top1",
    "bm25_margin_ratio",
    "retrievers_agree",
    "fused_top1",
    "fused_margin",
    "query_len",
]


def extract_features(
    query: str, bm25: Retriever, dense: Retriever, hybrid: Retriever
) -> tuple[dict[str, float], str]:
    """Signals available at answer time (no labels). Also returns the fused top-1 doc id."""
    d = dense.search(query, 5)
    b = bm25.search(query, 5)
    h = hybrid.search(query, 5)
    b1 = b[0].score
    feats = {
        "dense_top1": d[0].score,
        "dense_margin": d[0].score - d[1].score,
        "bm25_top1": b1,
        "bm25_margin_ratio": (b1 - b[1].score) / b1 if b1 > 0 else 0.0,
        "retrievers_agree": float(d[0].doc_id == b[0].doc_id),
        "fused_top1": h[0].score,
        "fused_margin": h[0].score - h[1].score,
        "query_len": float(len(tokenize(query))),
    }
    return feats, h[0].doc_id


def build_examples(
    corpus: Corpus, bm25: Retriever, dense: Retriever, hybrid: Retriever
) -> list[QueryExample]:
    examples = []
    for qid, query in corpus.queries.items():
        feats, top1 = extract_features(query, bm25, dense, hybrid)
        examples.append(QueryExample(qid, feats, int(top1 in corpus.qrels[qid])))
    return examples


def to_matrix(examples: list[QueryExample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[e.features[n] for n in FEATURE_NAMES] for e in examples], dtype=float)
    y = np.array([e.correct for e in examples], dtype=float)
    return x, y


def naive_confidence(cosine: np.ndarray) -> np.ndarray:
    """What many demos do: treat the raw cosine similarity as the probability of being right."""
    return np.clip(np.asarray(cosine, dtype=float), 0.0, 1.0)
