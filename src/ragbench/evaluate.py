"""Benchmark harness: run every retrieval strategy over a labelled dataset and report metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .data import Corpus, load_scifact
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k
from .retrievers import (
    FastEmbedder,
    Retriever,
    build_retrievers,
    cross_encoder_scorer,
)


def evaluate_retriever(retriever: Retriever, corpus: Corpus, k: int = 10) -> dict[str, float]:
    ndcg, recall, recall100, mrr, latencies = [], [], [], [], []
    for qid, query in corpus.queries.items():
        relevant = corpus.qrels[qid]
        start = time.perf_counter()
        ranked = [h.doc_id for h in retriever.search(query, max(k, 100))]
        latencies.append((time.perf_counter() - start) * 1000)
        ndcg.append(ndcg_at_k(ranked, relevant, k))
        recall.append(recall_at_k(ranked, relevant, k))
        recall100.append(recall_at_k(ranked, relevant, 100))
        mrr.append(mrr_at_k(ranked, relevant, k))
    return {
        f"nDCG@{k}": statistics.fmean(ndcg),
        f"Recall@{k}": statistics.fmean(recall),
        "Recall@100": statistics.fmean(recall100),
        f"MRR@{k}": statistics.fmean(mrr),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p95": sorted(latencies)[int(0.95 * len(latencies)) - 1],
    }


def to_markdown(results: dict[str, dict[str, float]], n_docs: int, n_queries: int) -> str:
    cols = list(next(iter(results.values())).keys())
    best = {c: max(r[c] for r in results.values()) for c in cols if "latency" not in c}
    lines = [
        f"Dataset: BEIR SciFact, {n_docs:,} documents, {n_queries} test queries.\n",
        "| Strategy | " + " | ".join(cols) + " |",
        "|---|" + "---:|" * len(cols),
    ]
    for name, row in results.items():
        cells = []
        for c in cols:
            v = row[c]
            if "latency" in c:
                cells.append(f"{v:.1f}")
            else:
                cell = f"{v:.3f}"
                cells.append(f"**{cell}**" if v == best[c] else cell)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plot(results: dict[str, dict[str, float]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    metrics = ["nDCG@10", "Recall@10", "Recall@100"]
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for j, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        pos = [i + j * width for i in range(len(names))]
        bars = ax.bar(pos, vals, width, label=m)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks([i + width for i in range(len(names))], names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Retrieval quality on SciFact by strategy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark retrieval strategies on SciFact.")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--no-rerank", action="store_true", help="Skip the cross-encoder variants.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N queries.")
    args = p.parse_args()

    corpus = load_scifact(args.data_dir)
    if args.limit:
        keep = list(corpus.queries)[: args.limit]
        corpus.queries = {q: corpus.queries[q] for q in keep}
    print(f"{len(corpus):,} documents, {len(corpus.queries)} queries")

    scorer = None if args.no_rerank else cross_encoder_scorer()
    retrievers = build_retrievers(
        corpus, FastEmbedder(args.embedding_model), scorer, cache_dir=args.data_dir / "cache"
    )

    results = {}
    for name, retriever in retrievers.items():
        print(f"Evaluating {name} ...")
        results[name] = evaluate_retriever(retriever, corpus)
        print("  ", {k: round(v, 3) for k, v in results[name].items()})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark.json").write_text(json.dumps(results, indent=2))
    md = to_markdown(results, len(corpus), len(corpus.queries))
    (args.out_dir / "benchmark.md").write_text(md)
    try:
        plot(results, args.out_dir / "benchmark.png")
    except ImportError:
        print("matplotlib not installed; skipping chart (pip install 'ragbench[bench]').")
    print("\n" + md)


if __name__ == "__main__":
    main()
