import numpy as np
import pytest

from ragbench.calibrate_eval import fit_histogram, fit_isotonic, fit_logistic_all, fit_naive
from ragbench.calibration import (
    FEATURE_NAMES,
    HistogramBinning,
    brier_score,
    build_examples,
    evaluate_threshold,
    expected_calibration_error,
    pick_threshold,
    reliability_curve,
    risk_coverage,
    to_matrix,
)
from ragbench.retrievers import HashingEmbedder, build_retrievers


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.array([0.8] * 10)
    labels = np.array([1] * 8 + [0] * 2)
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_ece_of_overconfident_predictions():
    probs = np.array([0.9] * 10)
    labels = np.array([1] * 5 + [0] * 5)  # 90% sure, 50% right
    assert expected_calibration_error(probs, labels) == pytest.approx(0.4)


def test_ece_puts_probability_one_in_the_last_bin():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0


def test_brier_score():
    assert brier_score(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    assert brier_score(np.array([0.5, 0.5]), np.array([1, 0])) == pytest.approx(0.25)


def test_reliability_curve_counts_add_up():
    probs = np.array([0.1, 0.15, 0.9, 0.95])
    curve = reliability_curve(probs, np.array([0, 0, 1, 1]))
    assert sum(c for _, _, c in curve) == 4
    assert curve[0][1] == 0.0 and curve[-1][1] == 1.0


def test_histogram_binning_learns_a_monotone_trend_and_stays_in_range():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 2000)
    labels = (rng.uniform(0, 1, 2000) < scores).astype(float)  # P(correct) = score
    hb = HistogramBinning(n_bins=10).fit(scores, labels)
    pred = hb.predict(np.array([0.05, 0.5, 0.95]))
    assert pred[0] < pred[1] < pred[2]
    assert ((pred >= 0) & (pred <= 1)).all()


def test_risk_coverage_orders_by_confidence():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    labels = np.array([1, 1, 0, 1])
    coverage, accuracy = risk_coverage(conf, labels)
    assert list(coverage) == [0.25, 0.5, 0.75, 1.0]
    assert list(accuracy[:3]) == [1.0, 1.0, pytest.approx(2 / 3)]


def test_pick_threshold_finds_the_most_answers_that_still_meet_the_target():
    conf = np.linspace(1, 0, 100)
    labels = np.array([1] * 60 + [0] * 40)  # the top 60 are all correct
    tau = pick_threshold(conf, labels, target_precision=0.95, min_answered=20)
    result = evaluate_threshold(conf, labels, tau)
    assert result["precision"] >= 0.95
    assert result["coverage"] > 0.6  # keeps answering a little past the clean prefix


def test_pick_threshold_returns_none_when_target_unreachable():
    conf = np.linspace(1, 0, 50)
    labels = np.zeros(50)
    assert pick_threshold(conf, labels, 0.9) is None


def test_calibrators_return_probabilities_and_improve_on_naive():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=(n, len(FEATURE_NAMES)))
    dense = FEATURE_NAMES.index("dense_top1")
    x[:, dense] = rng.uniform(0.5, 0.9, n)  # cosines live in a narrow high band
    y = (rng.uniform(0, 1, n) < (x[:, dense] - 0.5) / 0.4).astype(float)
    fit_x, fit_y, test_x, test_y = x[:400], y[:400], x[400:], y[400:]

    ece = {}
    for name, fit in [
        ("naive", fit_naive),
        ("hist", fit_histogram),
        ("iso", fit_isotonic),
        ("lr", fit_logistic_all),
    ]:
        p = fit(fit_x, fit_y)(test_x)
        assert ((p >= 0) & (p <= 1)).all()
        ece[name] = expected_calibration_error(p, test_y)
    assert ece["iso"] < ece["naive"]
    assert ece["hist"] < ece["naive"]


def test_build_examples_labels_top1_correctness(corpus):
    r = build_retrievers(corpus, HashingEmbedder())
    examples = build_examples(corpus, r["bm25"], r["dense"], r["hybrid"])
    assert len(examples) == len(corpus.queries)
    assert all(set(e.features) == set(FEATURE_NAMES) for e in examples)
    x, y = to_matrix(examples)
    assert x.shape == (2, len(FEATURE_NAMES)) and set(y) <= {0.0, 1.0}
    assert y.sum() >= 1  # the toy queries have obvious answers
