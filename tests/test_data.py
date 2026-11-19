import json

from ragbench.data import load_beir
from ragbench.evaluate import to_markdown


def make_beir_folder(root):
    root.mkdir()
    (root / "qrels").mkdir()
    docs = [
        {"_id": "d1", "title": "Aspirin", "text": "Aspirin lowers cancer risk."},
        {"_id": "d2", "title": "", "text": "Vitamin D and bones."},
        {"_id": "d3", "title": "Sleep", "text": "Sleep loss harms attention."},
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    queries = [{"_id": "q1", "text": "does aspirin help"}, {"_id": "q2", "text": "bone health"}]
    (root / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries))
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td3\t0\nq2\td2\t1\n"
    )
    return root


def test_load_beir_reads_graded_relevance_and_ignores_zero_scores(tmp_path):
    corpus = load_beir(make_beir_folder(tmp_path / "ds"), "test")
    assert len(corpus) == 3
    assert corpus.qrels == {"q1": {"d1": 2}, "q2": {"d2": 1}}  # the score-0 pair is dropped
    assert corpus.queries["q2"] == "bone health"
    assert corpus.get("d2").full_text == "Vitamin D and bones."  # no title, no leading ". "


def test_report_names_the_dataset():
    results = {"bm25": {"nDCG@10": 0.5, "latency_ms_mean": 1.0}}
    assert "BEIR NFCorpus, 3,633 documents, 323" in to_markdown(results, 3633, 323, "NFCorpus")
    assert "BEIR SciFact" in to_markdown(results, 10, 5)
