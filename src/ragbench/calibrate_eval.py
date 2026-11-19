"""Calibrated retrieval experiment on SciFact.

For each claim we take the hybrid retriever's top result and ask: is it a correct (expert-marked
relevant) paper? Then we compare ways of turning retrieval signals into a probability of being
right, and test whether that probability lets the system safely say "I'm not sure".

Data split (no leakage): SciFact train claims are split into a calibrator-fit part and a
threshold-selection part; the official test claims are used only for the final numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .calibration import (
    FEATURE_NAMES,
    HistogramBinning,
    brier_score,
    build_examples,
    evaluate_threshold,
    expected_calibration_error,
    naive_confidence,
    pick_threshold,
    reliability_curve,
    risk_coverage,
    to_matrix,
)
from .data import load_scifact
from .retrievers import FastEmbedder, HashingEmbedder, build_retrievers

DENSE = FEATURE_NAMES.index("dense_top1")
TARGETS = (0.7, 0.8, 0.9)

Predictor = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- methods
def fit_naive(x: np.ndarray, y: np.ndarray) -> Predictor:
    return lambda z: naive_confidence(z[:, DENSE])


def fit_platt(x: np.ndarray, y: np.ndarray) -> Predictor:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e4).fit(x[:, [DENSE]], y)
    return lambda z: model.predict_proba(z[:, [DENSE]])[:, 1]


def fit_histogram(x: np.ndarray, y: np.ndarray) -> Predictor:
    hb = HistogramBinning(n_bins=10).fit(x[:, DENSE], y)
    return lambda z: hb.predict(z[:, DENSE])


def fit_isotonic(x: np.ndarray, y: np.ndarray) -> Predictor:
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x[:, DENSE], y)
    return lambda z: iso.predict(z[:, DENSE])


def fit_logistic_all(x: np.ndarray, y: np.ndarray) -> Predictor:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(x), y)
    return lambda z: model.predict_proba(scaler.transform(z))[:, 1]


METHODS: dict[str, Callable[[np.ndarray, np.ndarray], Predictor]] = {
    "raw cosine (naive)": fit_naive,
    "Platt scaling (dense score)": fit_platt,
    "histogram binning (dense score)": fit_histogram,
    "isotonic regression (dense score)": fit_isotonic,
    "logistic regression (all signals)": fit_logistic_all,
}
THRESHOLD_METHODS = [
    "raw cosine (naive)",
    "isotonic regression (dense score)",
    "logistic regression (all signals)",
]


def bootstrap_ci(
    fn: Callable[[np.ndarray, np.ndarray], float],
    probs: np.ndarray,
    labels: np.ndarray,
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():  # AUROC undefined
            continue
        vals.append(fn(probs[idx], labels[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run(
    seed: int = 0,
    data_dir: Path = Path("data"),
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    smoke: bool = False,
) -> dict:
    from sklearn.metrics import roc_auc_score

    train = load_scifact(data_dir, "train")
    test = load_scifact(data_dir, "test")
    print(
        f"{len(train.documents):,} documents | {len(train.queries)} train claims | {len(test.queries)} test claims"
    )

    # --smoke swaps in a cheap hashing embedder so the pipeline can be checked in seconds.
    # Its numbers are NOT meaningful results.
    embedder = HashingEmbedder() if smoke else FastEmbedder(embedding_model)
    cache = None if smoke else data_dir / "cache"
    retrievers = build_retrievers(train, embedder, None, cache_dir=cache)
    bm25, dense, hybrid = retrievers["bm25"], retrievers["dense"], retrievers["hybrid"]
    x_train, y_train = to_matrix(build_examples(train, bm25, dense, hybrid))
    x_test, y_test = to_matrix(build_examples(test, bm25, dense, hybrid))

    perm = np.random.default_rng(seed).permutation(len(y_train))
    k = int(0.7 * len(perm))
    fit_idx, dev_idx = perm[:k], perm[k:]
    print(f"fit {len(fit_idx)} | threshold-selection {len(dev_idx)} | test {len(y_test)}")
    print(f"Top-1 accuracy of the hybrid retriever on test: {y_test.mean():.3f}")

    out: dict = {
        "split": {
            "fit": int(len(fit_idx)),
            "threshold_selection": int(len(dev_idx)),
            "test": int(len(y_test)),
        },
        "top1_accuracy_test": float(y_test.mean()),
        "methods": {},
        "_curves": {},
    }
    for name, fit in METHODS.items():
        predict = fit(x_train[fit_idx], y_train[fit_idx])
        p_dev, p_test = predict(x_train[dev_idx]), predict(x_test)
        ece_ci = bootstrap_ci(expected_calibration_error, p_test, y_test)
        auc_ci = bootstrap_ci(lambda p, lab: roc_auc_score(lab, p), p_test, y_test)
        entry = {
            "ECE": expected_calibration_error(p_test, y_test),
            "ECE_ci95": ece_ci,
            "Brier": brier_score(p_test, y_test),
            "AUROC": float(roc_auc_score(y_test, p_test)),
            "AUROC_ci95": auc_ci,
            "thresholds": {},
        }
        for target in TARGETS:
            tau = pick_threshold(p_dev, y_train[dev_idx], target)
            entry["thresholds"][str(target)] = (
                None
                if tau is None
                else {"threshold": tau, **evaluate_threshold(p_test, y_test, tau)}
            )
        out["methods"][name] = entry
        out["_curves"][name] = (p_test, reliability_curve(p_test, y_test))
        print(
            f"{name:36s} ECE {entry['ECE']:.3f}  Brier {entry['Brier']:.3f}  AUROC {entry['AUROC']:.3f}"
        )
    out["_labels"] = y_test
    return out


# --------------------------------------------------------------------------- outputs
def to_markdown(res: dict) -> str:
    n = res["split"]["test"]
    lines = [
        f"SciFact test claims: {n}. Answering every claim, the hybrid retriever's top result is correct "
        f"{res['top1_accuracy_test']:.1%} of the time. Calibrators were fit on {res['split']['fit']} "
        f"train claims; thresholds were chosen on {res['split']['threshold_selection']} other train claims.\n",
        "### How well calibrated is the confidence? (lower ECE and Brier are better)\n",
        "| Method | ECE | 95% CI | Brier | AUROC |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in res["methods"].items():
        lines.append(
            f"| {name} | {m['ECE']:.3f} | {m['ECE_ci95'][0]:.3f} to {m['ECE_ci95'][1]:.3f} "
            f"| {m['Brier']:.3f} | {m['AUROC']:.3f} |"
        )
    lines += [
        "\n### Can it abstain safely? Choose a confidence threshold on held-out data, then test it\n",
        "| Target precision | Method | Test coverage | Test precision |",
        "|---:|---|---:|---:|",
    ]
    for target in TARGETS:
        for name in THRESHOLD_METHODS:
            r = res["methods"][name]["thresholds"][str(target)]
            if r is None:
                lines.append(f"| {target:.0%} | {name} | target not reachable | n/a |")
            else:
                lines.append(
                    f"| {target:.0%} | {name} | {r['coverage']:.1%} | {r['precision']:.1%} |"
                )
    return "\n".join(lines) + "\n"


def plots(res: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = [
        "raw cosine (naive)",
        "isotonic regression (dense score)",
        "logistic regression (all signals)",
    ]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
    for name, c in zip(show, colors, strict=True):
        pts = res["_curves"][name][1]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=c, label=name, ms=4)
    ax.set_xlabel("Stated confidence")
    ax.set_ylabel("How often the top result is actually right")
    ax.set_title("Reliability: does confidence mean what it says?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=160)
    plt.close(fig)

    y = res["_labels"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(y.mean(), ls="--", color="gray", label="answer everything")
    for name, c in zip(show[1:], colors[1:], strict=True):
        cov, acc = risk_coverage(res["_curves"][name][0], y)
        ax.plot(cov, acc, color=c, label=name)
    ax.set_xlabel("Coverage (share of questions answered)")
    ax.set_ylabel("Accuracy of the answers given")
    ax.set_title("Answering fewer questions makes answers more reliable")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrated retrieval experiment on SciFact.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="Fast pipeline check; results meaningless.")
    args = p.parse_args()

    res = run(args.seed, args.data_dir, args.embedding_model, args.smoke)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = to_markdown(res)
    (args.out_dir / "calibration.md").write_text(md)
    serializable = {k: v for k, v in res.items() if not k.startswith("_")}
    (args.out_dir / "calibration.json").write_text(json.dumps(serializable, indent=2))
    plots(res, args.out_dir)
    print("\n" + md)


if __name__ == "__main__":
    main()
